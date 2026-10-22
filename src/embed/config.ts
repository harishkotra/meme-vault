export const MODEL_ID = 'onnx-community/embeddinggemma-2-ONNX';
export const DIM = 768;

export type Device = 'webgpu' | 'wasm';

/**
 * Quantisation per device, and the reason it is not one setting:
 * the q4/q8/q4f16 text and vision graphs use GatherBlockQuantized, which the
 * ONNX Runtime WASM build has no kernel for, so a WASM tab must load fp16.
 */
export const DTYPE: Record<Device, { model: string; vision_encoder: string }> = {
  webgpu: { model: 'q4', vision_encoder: 'q4' },
  wasm: { model: 'fp16', vision_encoder: 'fp16' },
};

/** One-time download, in MB, for the graphs actually loaded (text + vision). */
export const DOWNLOAD_MB: Record<Device, number> = { webgpu: 284, wasm: 879 };

/** An image costs up to 282 tokens: 280 soft tokens plus begin/end markers. */
export const IMAGE_TOKENS = 282;

/** WebGPU kernels hit a dispatch limit around 2,700 tokens; WASM runs one image per batch. */
export const BATCH_BUDGET: Record<Device, number> = { webgpu: 2600, wasm: 300 };

export const queryPrefix = (text: string) => `task: search result | query: ${text.trim()}`;
