import { afterEach, describe, expect, it, vi } from "vitest";
import { UnlockedBrowserTransformerWorkerClient } from "./unlockedBrowserTransformerWorkerClient";

type WorkerMessage = { id: string; type: string; [key: string]: unknown };

class FakeWorker {
  static instances: FakeWorker[] = [];
  onmessage: ((event: MessageEvent<WorkerMessage>) => void) | null = null;
  messages: WorkerMessage[] = [];
  terminated = false;

  constructor() {
    FakeWorker.instances.push(this);
  }

  postMessage(message: WorkerMessage): void {
    this.messages.push(message);
  }

  terminate(): void {
    this.terminated = true;
  }

  emit(message: WorkerMessage): void {
    this.onmessage?.({ data: message } as MessageEvent<WorkerMessage>);
  }
}

describe("UnlockedBrowserTransformerWorkerClient", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
    FakeWorker.instances = [];
  });

  it("rejects pending requests when the worker is terminated", async () => {
    vi.stubGlobal("Worker", FakeWorker);
    const client = new UnlockedBrowserTransformerWorkerClient({
      modelId: "Qwen/Qwen3-0.6B",
      allowFixtureWeights: true,
      backendPreference: "cpu",
    });
    const initPromise = client.init();
    await flushMicrotasks();

    client.terminate();

    await expect(initPromise).rejects.toThrow(/terminated/i);
    expect(FakeWorker.instances[0]?.terminated).toBe(true);
  });

  it("serializes flush requests behind an active stream", async () => {
    vi.stubGlobal("Worker", FakeWorker);
    const client = new UnlockedBrowserTransformerWorkerClient({
      modelId: "Qwen/Qwen3-0.6B",
      allowFixtureWeights: true,
      backendPreference: "cpu",
    });
    const worker = FakeWorker.instances[0]!;

    const initPromise = client.init();
    await flushMicrotasks();
    const initMessage = worker.messages[0]!;
    worker.emit({
      id: initMessage.id,
      type: "ready",
      modelId: "Qwen/Qwen3-0.6B",
      backendId: "unlocked-browser-transformer",
      kvPersistence: { enabled: true, mode: "memory", namespace: "test", decodeReuse: false },
    });
    await initPromise;

    const iterator = client.streamChat([{ role: "user", content: "hello" }], { maxTokens: 1 })[Symbol.asyncIterator]();
    const firstChunk = iterator.next();
    await flushMicrotasks();
    const streamMessage = worker.messages.find((message) => message.type === "stream")!;
    worker.emit({ id: streamMessage.id, type: "chunk", chunk: "ok", proof: null });
    await expect(firstChunk).resolves.toMatchObject({ done: false, value: "ok" });

    const flushPromise = client.flushKvPersistence();
    await flushMicrotasks();
    expect(worker.messages.some((message) => message.type === "flushKvPersistence")).toBe(false);

    const done = iterator.next();
    worker.emit({ id: streamMessage.id, type: "done", result: "ok", proof: null });
    await expect(done).resolves.toMatchObject({ done: true });
    await flushMicrotasks();

    const flushMessage = worker.messages.find((message) => message.type === "flushKvPersistence")!;
    expect(flushMessage).toBeDefined();
    worker.emit({
      id: flushMessage.id,
      type: "kvPersistenceFlushed",
      kvPersistence: { enabled: true, mode: "memory", namespace: "test", decodeReuse: false },
      proof: {
        tensorControl: true,
        kvPersistence: {
          enabled: true,
          mode: "memory",
          namespace: "test",
          decodeReuse: false,
          kvPersistDeferred: true,
          kvPersistFlushMs: 4,
          events: [],
        },
      },
      state: {
        promptTokenIds: [1, 2],
        generatedTokenIds: [3],
        generatedTokenTexts: ["ok"],
        generationStopReason: "max_tokens",
      },
    });
    await expect(flushPromise).resolves.toBeUndefined();
    expect(client.lastDecodeProof?.kvPersistence?.kvPersistFlushMs).toBe(4);
    expect(client.lastPromptTokenIds).toEqual([1, 2]);
  });
});

async function flushMicrotasks(): Promise<void> {
  await Promise.resolve();
  await Promise.resolve();
}
