import type { ChatClientMessage, ChatStreamOptions } from "../lib/llm/types";
import {
  UnlockedBrowserTransformerClient,
  type UnlockedBrowserTransformerClientOptions,
} from "../lib/llm/unlockedBrowserTransformerClient";

type InitMessage = {
  id: string;
  type: "init";
  options: UnlockedBrowserTransformerClientOptions;
};

type StreamMessage = {
  id: string;
  type: "stream";
  messages: ChatClientMessage[];
  options: ChatStreamOptions;
};

type ClearKvPersistenceMessage = {
  id: string;
  type: "clearKvPersistence";
};

type FlushKvPersistenceMessage = {
  id: string;
  type: "flushKvPersistence";
};

type IncomingMessage = InitMessage | StreamMessage | ClearKvPersistenceMessage | FlushKvPersistenceMessage;

let client: UnlockedBrowserTransformerClient | null = null;
let operationQueue: Promise<void> = Promise.resolve();

self.onmessage = (event: MessageEvent<IncomingMessage>) => {
  const message = event.data;
  operationQueue = operationQueue
    .catch(() => undefined)
    .then(() => handleWorkerMessage(message));
};

async function handleWorkerMessage(message: IncomingMessage): Promise<void> {
  try {
    if (message.type === "init") {
      postProgress(message.id, message.options.manifestPath
        ? "Loading unlocked Qwen manifest and weight shards"
        : "Loading unlocked tensor-control fixture");
      client = new UnlockedBrowserTransformerClient(message.options);
      await client.init();
      postProgress(message.id, "Unlocked transformer runtime ready");
      self.postMessage({
        id: message.id,
        type: "ready",
        modelId: client.modelId,
        backendId: client.backendId,
        kvPersistence: client.getKvPersistenceHealth(),
        warmup: readWarmupState(client),
      });
      return;
    }

    if (!client) throw new Error("Unlocked browser transformer worker has not been initialized.");
    if (message.type === "clearKvPersistence") {
      const event = await client.clearKvPersistence();
      self.postMessage({
        id: message.id,
        type: "kvPersistenceCleared",
        kvPersistence: client.getKvPersistenceHealth(),
        event,
      });
      return;
    }
    if (message.type === "flushKvPersistence") {
      await client.flushKvPersistence();
      self.postMessage({
        id: message.id,
        type: "kvPersistenceFlushed",
        kvPersistence: client.getKvPersistenceHealth(),
        proof: client.lastDecodeProof,
        state: readRuntimeState(client),
      });
      return;
    }
    let result = "";
    postProgress(message.id, "Running Qwen prefill and SSA/KV/TSP decode");
    for await (const chunk of client.streamChat(message.messages, message.options)) {
      result += chunk;
      self.postMessage({
        id: message.id,
        type: "chunk",
        chunk,
        proof: client.lastDecodeProof,
        state: readRuntimeState(client),
      });
    }
    self.postMessage({
      id: message.id,
      type: "done",
      result,
      proof: client.lastDecodeProof,
      state: readRuntimeState(client),
    });
  } catch (error) {
    self.postMessage({
      id: message.id,
      type: "error",
      error: error instanceof Error ? error.message : String(error),
    });
  }
}

function postProgress(id: string, progress: string): void {
  self.postMessage({
    id,
    type: "progress",
    progress,
  });
}

function readRuntimeState(client: UnlockedBrowserTransformerClient) {
  return {
    promptTokenIds: [...client.lastPromptTokenIds],
    generatedTokenIds: [...client.lastGeneratedTokenIds],
    generatedTokenTexts: [...client.lastGeneratedTokenTexts],
    generationStopReason: client.lastGenerationStopReason,
  };
}

function readWarmupState(client: UnlockedBrowserTransformerClient) {
  return {
    warmupMs: client.lastWarmupMs,
    warmupMode: client.lastWarmupMode,
    warmupUploadedEntries: client.lastWarmupUploadedEntries,
    warmupCacheHits: client.lastWarmupCacheHits,
    residentReadbackCount: client.lastResidentReadbackCount,
    warmupProof: client.lastWarmupProof,
  };
}
