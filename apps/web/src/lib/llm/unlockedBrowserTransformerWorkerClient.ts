import type { ChatClient, ChatClientMessage, ChatStreamOptions } from "./types";
import type {
  UnlockedBrowserDecodeProof,
  UnlockedBrowserGenerationStopReason,
  UnlockedBrowserTransformerClientDisposeOptions,
  UnlockedBrowserTransformerClientOptions,
  UnlockedBrowserWarmupMode,
} from "./unlockedBrowserTransformerClient";
import type { KVSwapPersistenceHealth } from "../runtime/kvSwapPersistence";

type WorkerRuntimeState = {
  promptTokenIds: number[];
  generatedTokenIds: number[];
  generatedTokenTexts: string[];
  generationStopReason: UnlockedBrowserGenerationStopReason | null;
};

type WorkerWarmupState = {
  warmupMs: number | null;
  warmupMode: UnlockedBrowserWarmupMode | null;
  warmupUploadedEntries: number | null;
  warmupCacheHits: number | null;
  residentReadbackCount: number | null;
  warmupProof: UnlockedBrowserDecodeProof | null;
};

type InitRequest = {
  id: string;
  type: "init";
  options: UnlockedBrowserTransformerClientOptions;
};

type StreamRequest = {
  id: string;
  type: "stream";
  messages: ChatClientMessage[];
  options: ChatStreamOptions;
};

type ClearKvPersistenceRequest = {
  id: string;
  type: "clearKvPersistence";
};

type FlushKvPersistenceRequest = {
  id: string;
  type: "flushKvPersistence";
};

type WorkerResponse =
  | { id: string; type: "progress"; progress: string }
  | { id: string; type: "ready"; modelId: string; backendId: string; kvPersistence?: KVSwapPersistenceHealth; warmup?: WorkerWarmupState }
  | { id: string; type: "chunk"; chunk: string; proof: UnlockedBrowserDecodeProof | null; state?: WorkerRuntimeState }
  | { id: string; type: "done"; result: string; proof: UnlockedBrowserDecodeProof | null; state?: WorkerRuntimeState }
  | { id: string; type: "kvPersistenceCleared"; kvPersistence: KVSwapPersistenceHealth }
  | { id: string; type: "kvPersistenceFlushed"; kvPersistence: KVSwapPersistenceHealth; proof: UnlockedBrowserDecodeProof | null; state?: WorkerRuntimeState }
  | { id: string; type: "error"; error: string };

type PendingInit = {
  kind: "init";
  resolve: (value: { modelId: string; backendId: string }) => void;
  reject: (reason?: unknown) => void;
};

type PendingClear = {
  kind: "clear";
  resolve: () => void;
  reject: (reason?: unknown) => void;
};

type PendingFlush = {
  kind: "flush";
  resolve: () => void;
  reject: (reason?: unknown) => void;
};

type PendingStream = {
  kind: "stream";
  chunks: string[];
  done: boolean;
  result: string;
  error: Error | null;
  notify: (() => void) | null;
};

export class UnlockedBrowserTransformerWorkerClient implements ChatClient {
  readonly backendId = "unlocked-browser-transformer";
  readonly modelId: string;
  lastDecodeProof: UnlockedBrowserDecodeProof | null = null;
  lastPromptTokenIds: number[] = [];
  lastGeneratedTokenIds: number[] = [];
  lastGeneratedTokenTexts: string[] = [];
  lastGenerationStopReason: UnlockedBrowserGenerationStopReason | null = null;
  lastWarmupMs: number | null = null;
  lastWarmupProof: UnlockedBrowserDecodeProof | null = null;
  lastWarmupMode: UnlockedBrowserWarmupMode | null = null;
  lastWarmupUploadedEntries: number | null = null;
  lastWarmupCacheHits: number | null = null;
  lastResidentReadbackCount: number | null = null;
  kvPersistenceHealth: KVSwapPersistenceHealth = { enabled: false, mode: "disabled", namespace: "default", decodeReuse: false };

  private readonly worker: Worker;
  private readonly options: UnlockedBrowserTransformerClientOptions;
  private nextId = 0;
  private initialized = false;
  private terminated = false;
  private activeRequest: Promise<void> = Promise.resolve();
  private pending = new Map<string, PendingInit | PendingClear | PendingFlush | PendingStream>();
  private readonly onProgress: ((progress: string) => void) | undefined;

  constructor(options: UnlockedBrowserTransformerClientOptions, callbacks: { onProgress?: (progress: string) => void } = {}) {
    this.modelId = options.modelId;
    this.options = options;
    this.onProgress = callbacks.onProgress;
    this.worker = new Worker(new URL("../../workers/unlockedTransformer.worker.ts", import.meta.url), {
      type: "module",
    });
    this.worker.onmessage = (event: MessageEvent<WorkerResponse>) => this.handleMessage(event.data);
  }

  async init(): Promise<void> {
    const release = await this.beginRequest();
    try {
      const id = this.nextRequestId();
      const result = await new Promise<{ modelId: string; backendId: string }>((resolve, reject) => {
        this.pending.set(id, { kind: "init", resolve, reject });
        this.worker.postMessage({ id, type: "init", options: this.options } satisfies InitRequest);
      });
      if (result.modelId !== this.modelId) {
        throw new Error(`Unlocked browser transformer worker modelId mismatch: expected ${this.modelId}, got ${result.modelId}.`);
      }
      this.initialized = true;
    } finally {
      release();
    }
  }

  async *streamChat(messages: ChatClientMessage[], options: ChatStreamOptions = {}): AsyncGenerator<string, string, void> {
    if (!this.initialized) throw new Error("UnlockedBrowserTransformerWorkerClient.init() must complete before streamChat().");
    const release = await this.beginRequest();
    this.lastDecodeProof = null;
    this.lastPromptTokenIds = [];
    this.lastGeneratedTokenIds = [];
    this.lastGeneratedTokenTexts = [];
    this.lastGenerationStopReason = null;
    const id = this.nextRequestId();
    const stream: PendingStream = {
      kind: "stream",
      chunks: [],
      done: false,
      result: "",
      error: null,
      notify: null,
    };
    try {
      this.pending.set(id, stream);
      this.worker.postMessage({ id, type: "stream", messages, options } satisfies StreamRequest);

      while (!stream.done || stream.chunks.length > 0) {
        const chunk = stream.chunks.shift();
        if (chunk !== undefined) {
          yield chunk;
          continue;
        }
        if (stream.error) throw stream.error;
        await new Promise<void>((resolve) => {
          stream.notify = resolve;
        });
        stream.notify = null;
        if (stream.error) throw stream.error;
      }
      return stream.result;
    } finally {
      this.pending.delete(id);
      release();
    }
  }

  terminate(): void {
    if (this.terminated) return;
    this.terminated = true;
    const error = new Error("Unlocked browser transformer worker was terminated.");
    for (const [id, pending] of this.pending) {
      this.pending.delete(id);
      if (pending.kind === "stream") {
        pending.error = error;
        pending.done = true;
        pending.notify?.();
      } else {
        pending.reject(error);
      }
    }
    this.worker.terminate();
  }

  async dispose(_options: UnlockedBrowserTransformerClientDisposeOptions = {}): Promise<void> {
    this.terminate();
  }

  async clearKvPersistence(): Promise<void> {
    const release = await this.beginRequest();
    try {
      const id = this.nextRequestId();
      await new Promise<void>((resolve, reject) => {
        this.pending.set(id, { kind: "clear", resolve, reject });
        this.worker.postMessage({ id, type: "clearKvPersistence" } satisfies ClearKvPersistenceRequest);
      });
    } finally {
      release();
    }
  }

  async flushKvPersistence(): Promise<void> {
    const release = await this.beginRequest();
    try {
      const id = this.nextRequestId();
      await new Promise<void>((resolve, reject) => {
        this.pending.set(id, { kind: "flush", resolve, reject });
        this.worker.postMessage({ id, type: "flushKvPersistence" } satisfies FlushKvPersistenceRequest);
      });
    } finally {
      release();
    }
  }

  private handleMessage(message: WorkerResponse): void {
    if (message.type === "progress") {
      this.onProgress?.(message.progress);
      return;
    }
    const pending = this.pending.get(message.id);
    if (!pending) return;
    if (message.type === "error") {
      const error = new Error(message.error);
      if (pending.kind === "init") {
        this.pending.delete(message.id);
        pending.reject(error);
      } else if (pending.kind === "clear" || pending.kind === "flush") {
        this.pending.delete(message.id);
        pending.reject(error);
      } else {
        pending.error = error;
        pending.done = true;
        pending.notify?.();
      }
      return;
    }

    if (pending.kind === "clear") {
      if (message.type === "kvPersistenceCleared") {
        this.pending.delete(message.id);
        this.kvPersistenceHealth = message.kvPersistence;
        pending.resolve();
      }
      return;
    }

    if (pending.kind === "flush") {
      if (message.type === "kvPersistenceFlushed") {
        this.pending.delete(message.id);
        this.kvPersistenceHealth = message.kvPersistence;
        this.lastDecodeProof = message.proof;
        if (message.state) this.updateRuntimeState(message.state);
        if (message.proof?.kvPersistence) this.kvPersistenceHealth = message.proof.kvPersistence;
        pending.resolve();
      }
      return;
    }

    if (pending.kind === "init") {
      if (message.type !== "ready") return;
      this.pending.delete(message.id);
      if (message.kvPersistence) this.kvPersistenceHealth = message.kvPersistence;
      if (message.warmup) this.updateWarmupState(message.warmup);
      pending.resolve({ modelId: message.modelId, backendId: message.backendId });
      return;
    }

    if (message.type === "chunk") {
      pending.chunks.push(message.chunk);
      this.lastDecodeProof = message.proof;
      if (message.state) this.updateRuntimeState(message.state);
      if (message.proof?.kvPersistence) this.kvPersistenceHealth = message.proof.kvPersistence;
      pending.notify?.();
      return;
    }

    if (message.type === "done") {
      pending.result = message.result;
      this.lastDecodeProof = message.proof;
      if (message.state) this.updateRuntimeState(message.state);
      if (message.proof?.kvPersistence) this.kvPersistenceHealth = message.proof.kvPersistence;
      pending.done = true;
      pending.notify?.();
    }
  }

  private updateRuntimeState(state: WorkerRuntimeState): void {
    this.lastPromptTokenIds = [...state.promptTokenIds];
    this.lastGeneratedTokenIds = [...state.generatedTokenIds];
    this.lastGeneratedTokenTexts = [...state.generatedTokenTexts];
    this.lastGenerationStopReason = state.generationStopReason;
  }

  private updateWarmupState(state: WorkerWarmupState): void {
    this.lastWarmupMs = state.warmupMs;
    this.lastWarmupMode = state.warmupMode;
    this.lastWarmupUploadedEntries = state.warmupUploadedEntries;
    this.lastWarmupCacheHits = state.warmupCacheHits;
    this.lastResidentReadbackCount = state.residentReadbackCount;
    this.lastWarmupProof = state.warmupProof;
  }

  private nextRequestId(): string {
    this.assertActive();
    const id = `unlocked_worker_${this.nextId}`;
    this.nextId += 1;
    return id;
  }

  private assertActive(): void {
    if (this.terminated) throw new Error("Unlocked browser transformer worker was terminated.");
  }

  private async beginRequest(): Promise<() => void> {
    this.assertActive();
    const previous = this.activeRequest.catch(() => undefined);
    let release!: () => void;
    this.activeRequest = previous.then(() => new Promise<void>((resolve) => {
      release = resolve;
    }));
    await previous;
    try {
      this.assertActive();
    } catch (error) {
      release();
      throw error;
    }
    let released = false;
    return () => {
      if (released) return;
      released = true;
      release();
    };
  }
}
