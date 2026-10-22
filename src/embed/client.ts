import type { Device } from './config';

export type WorkerEvent =
  | { kind: 'download'; progress: number; loaded: number; total: number }
  | { kind: 'status'; message: string }
  | { kind: 'file'; file: string }
  | { kind: 'embed'; done: number; total: number };

type Pending = { resolve: (value: any) => void; reject: (error: Error) => void };

/** Thin RPC over the embedding worker. One model per page. */
export class Embedder {
  private worker: Worker;
  private nextId = 1;
  private pending = new Map<number, Pending>();
  private listeners = new Set<(event: WorkerEvent) => void>();

  constructor() {
    this.worker = new Worker(new URL('./worker.ts', import.meta.url), { type: 'module' });
    this.worker.onmessage = ({ data }) => {
      if (data.type === 'event') {
        for (const listener of this.listeners) listener(data.event);
        return;
      }
      const entry = this.pending.get(data.id);
      if (!entry) return;
      this.pending.delete(data.id);
      if (data.error) entry.reject(new Error(data.error));
      else entry.resolve(data.result);
    };
    this.worker.onerror = (event) => {
      const error = new Error(event.message || 'The embedding worker crashed.');
      for (const entry of this.pending.values()) entry.reject(error);
      this.pending.clear();
    };
  }

  on(listener: (event: WorkerEvent) => void) {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  }

  private call<T>(method: string, args: unknown[] = []): Promise<T> {
    const id = this.nextId++;
    return new Promise<T>((resolve, reject) => {
      this.pending.set(id, { resolve, reject });
      this.worker.postMessage({ id, method, args });
    });
  }

  load() {
    return this.call<{ device: Device; reason: string | null }>('load');
  }

  embedQuery(text: string) {
    return this.call<Float32Array>('embedQuery', [text]);
  }

  embedImages(blobs: Blob[]) {
    return this.call<Float32Array[]>('embedImages', [blobs]);
  }

  dispose() {
    this.worker.terminate();
  }
}
