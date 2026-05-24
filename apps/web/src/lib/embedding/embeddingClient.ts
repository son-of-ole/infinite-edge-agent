export interface EmbeddingClientOptions {
  modelId: string;
  preferWebGPU: boolean;
}

interface PendingRequest<T> {
  resolve: (value: T) => void;
  reject: (reason?: unknown) => void;
  timeoutId: ReturnType<typeof setTimeout>;
}

interface InitResult {
  device: "webgpu" | "wasm";
  modelId: string;
}

export class EmbeddingClient {
  private worker: Worker;
  private nextId = 0;
  private pending = new Map<string, PendingRequest<unknown>>();
  private initialized = false;
  private readonly requestTimeoutMs: number;

  constructor(options: { requestTimeoutMs?: number } = {}) {
    this.requestTimeoutMs = options.requestTimeoutMs ?? 120_000;
    this.worker = new Worker(new URL("../../workers/embedding.worker.ts", import.meta.url), {
      type: "module"
    });
    this.worker.onmessage = (event: MessageEvent) => {
      const { id, type, result, vectors, error } = event.data ?? {};
      const pending = this.pending.get(id);
      if (!pending) return;
      this.pending.delete(id);
      clearTimeout(pending.timeoutId);
      if (type === "error") {
        pending.reject(new Error(error ?? "Unknown embedding worker error"));
      } else {
        pending.resolve(result ?? vectors);
      }
    };
  }

  async init(options: EmbeddingClientOptions): Promise<InitResult> {
    const result = await this.request<InitResult>({
      type: "init",
      modelId: options.modelId,
      preferWebGPU: options.preferWebGPU
    });
    this.initialized = true;
    return result;
  }

  async embed(text: string): Promise<number[]> {
    const [vector] = await this.embedBatch([text]);
    return vector ?? [];
  }

  async embedBatch(texts: string[]): Promise<number[][]> {
    if (!this.initialized) throw new Error("Embedding client is not initialized.");
    return this.request<number[][]>({ type: "embed", texts });
  }

  private request<T>(payload: Record<string, unknown>): Promise<T> {
    const id = `emb_${this.nextId++}`;
    return new Promise<T>((resolve, reject) => {
      const timeoutId = setTimeout(() => {
        this.pending.delete(id);
        reject(new Error(`Embedding worker ${String(payload.type ?? "request")} timed out after ${this.requestTimeoutMs}ms.`));
      }, this.requestTimeoutMs);
      this.pending.set(id, { resolve: resolve as (value: unknown) => void, reject, timeoutId });
      this.worker.postMessage({ id, ...payload });
    });
  }
}
