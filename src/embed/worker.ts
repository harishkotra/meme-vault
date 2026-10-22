/**
 * Embedding worker. Owns the model, the prompt prefixes, and batching.
 * The page only ever sends blobs and strings; it never touches tensors.
 */
import { AutoConfig, AutoModel, AutoProcessor, RawImage, env } from '@huggingface/transformers';
import { BATCH_BUDGET, DIM, DTYPE, IMAGE_TOKENS, MODEL_ID, queryPrefix, type Device } from './config';

const ctx = self as unknown as {
  postMessage: (message: unknown, transfer?: Transferable[]) => void;
  onmessage: ((event: MessageEvent) => void) | null;
};

type State = {
  model: any;
  processor: any;
  device: Device;
};

let state: State | null = null;
let queue: Promise<unknown> = Promise.resolve();

// The wasm files come from the CDN unless we point at our own copy. Keeping the
// default means one less build step; the browser cache makes it a one-time cost.
env.allowLocalModels = false;

const emit = (event: Record<string, unknown>) => ctx.postMessage({ type: 'event', event });

async function pickDevice(): Promise<{ device: Device; reason: string | null }> {
  const gpu = (navigator as any).gpu;
  if (!gpu) return { device: 'wasm', reason: 'This browser has no WebGPU, so the model runs on WASM (879 MB download).' };
  try {
    const adapter = await gpu.requestAdapter();
    if (!adapter) return { device: 'wasm', reason: 'No WebGPU adapter found, falling back to WASM (879 MB download).' };
    return { device: 'webgpu', reason: null };
  } catch (error: any) {
    return { device: 'wasm', reason: `WebGPU adapter request failed (${error?.message ?? error}), using WASM.` };
  }
}

async function loadOn(device: Device) {
  const config: any = await AutoConfig.from_pretrained(MODEL_ID);
  // Audio is not part of this app, so its encoder never downloads.
  config.audio_config = null;

  const progress_callback = (info: any) => {
    if (info.status === 'progress_total') {
      emit({
        kind: 'download',
        progress: info.progress ?? 0,
        loaded: info.loaded ?? 0,
        total: info.total ?? 0,
      });
    } else if (info.status === 'initiate') {
      emit({ kind: 'file', file: info.file });
    }
  };

  const [processor, model] = await Promise.all([
    AutoProcessor.from_pretrained(MODEL_ID, { progress_callback }),
    AutoModel.from_pretrained(MODEL_ID, {
      config,
      device,
      dtype: DTYPE[device] as any,
      progress_callback,
    }),
  ]);

  return { processor, model, device } as State;
}

async function load(): Promise<{ device: Device; reason: string | null }> {
  if (state) return { device: state.device, reason: null };

  const { device, reason } = await pickDevice();
  emit({ kind: 'status', message: `Loading EmbeddingGemma 2 on ${device === 'webgpu' ? 'WebGPU' : 'WASM'}…` });

  try {
    state = await loadOn(device);
  } catch (error: any) {
    if (device === 'wasm') throw error;
    emit({ kind: 'status', message: 'WebGPU failed to start, retrying on WASM…' });
    state = await loadOn('wasm');
  }

  emit({ kind: 'status', message: 'Warming up…' });
  await embedQuery('warm up');
  emit({ kind: 'status', message: 'Ready' });
  return { device: state!.device, reason: state!.device === 'wasm' ? reason : null };
}

function requireModel() {
  if (!state) throw new Error('Model is not loaded yet.');
  return state;
}

/** One forward pass. Returns one 768-d vector per input. */
async function run(...inputs: any[]): Promise<Float32Array[]> {
  const model = requireModel();
  const processed = await model.processor(...inputs);
  const output = await model.model(processed);
  const embedding = output?.sentence_embedding;
  if (!embedding) throw new Error('The model returned no sentence_embedding.');
  const flat = new Float32Array(embedding.data);
  for (const tensor of [...Object.values(output), ...Object.values(processed)]) {
    (tensor as any)?.dispose?.();
  }
  const count = flat.length / DIM;
  return Array.from({ length: count }, (_, i) => flat.slice(i * DIM, (i + 1) * DIM));
}

export async function embedQuery(text: string): Promise<Float32Array> {
  requireModel();
  const [vector] = await run([queryPrefix(text)]);
  return vector;
}

/** Batches blobs so that count × 282 tokens stays under the device budget. */
function pack(blobs: Blob[], device: Device): Blob[][] {
  const perBatch = Math.max(1, Math.floor(BATCH_BUDGET[device] / IMAGE_TOKENS));
  const batches: Blob[][] = [];
  for (let i = 0; i < blobs.length; i += perBatch) batches.push(blobs.slice(i, i + perBatch));
  return batches;
}

export async function embedImages(blobs: Blob[], onProgress?: (done: number, total: number) => void): Promise<Float32Array[]> {
  const model = requireModel();
  const vectors: Float32Array[] = [];
  const batches = pack(blobs, model.device);

  for (const batch of batches) {
    const images = await Promise.all(
      batch.map(async (blob) => {
        const image = await RawImage.fromBlob(blob);
        // One list per image, otherwise the processor merges the batch into a single input.
        return [image];
      }),
    );
    const out = await run(null, images);
    vectors.push(...out);
    onProgress?.(vectors.length, blobs.length);
    emit({ kind: 'embed', done: vectors.length, total: blobs.length });
  }

  return vectors;
}

type Request = { id: number; method: 'load' | 'embedQuery' | 'embedImages'; args: any[] };

ctx.onmessage = ({ data }: MessageEvent<Request>) => {
  // One request at a time: the GPU is the bottleneck and concurrent sessions contend.
  queue = queue.then(async () => {
    try {
      let result: unknown;
      if (data.method === 'load') result = await load();
      else if (data.method === 'embedQuery') result = await embedQuery(data.args[0]);
      else result = await embedImages(data.args[0]);

      const transfer: Transferable[] = [];
      if (result instanceof Float32Array) transfer.push(result.buffer);
      else if (Array.isArray(result)) for (const v of result) if (v instanceof Float32Array) transfer.push(v.buffer);

      ctx.postMessage({ id: data.id, result }, transfer);
    } catch (error: any) {
      ctx.postMessage({ id: data.id, error: error?.message ?? String(error) });
    }
  });
};
