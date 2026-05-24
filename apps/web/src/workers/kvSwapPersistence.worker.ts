import {
  createBrowserKVSwapPersistence,
  type BrowserKVSwapPersistenceOptions,
  type KVSwapPersistenceHealth,
  type KVSwapPersistenceListResult,
  type KVSwapPersistenceStore,
  type KVSwapPersistenceTraceEvent,
  type SerializedKVSwapBlock,
  normalizeKVSwapNamespace,
} from "../lib/runtime/kvSwapPersistence";

export type KVSwapPersistenceWorkerMessage =
  | { id: string; type: "init"; options: BrowserKVSwapPersistenceOptions }
  | { id: string; type: "save"; block: SerializedKVSwapBlock }
  | { id: string; type: "load"; namespace: string; blockId: string }
  | { id: string; type: "delete"; namespace: string; blockId: string }
  | { id: string; type: "list"; namespace: string }
  | { id: string; type: "persist"; blocks: SerializedKVSwapBlock[] }
  | { id: string; type: "hydrate"; namespace: string }
  | { id: string; type: "evict"; namespace: string; blockIds: string[] }
  | { id: string; type: "clear"; namespace: string }
  | { id: string; type: "health"; namespace?: string };

export type KVSwapPersistenceWorkerResponse =
  | { id: string; type: "ready"; health: KVSwapPersistenceHealth }
  | { id: string; type: "event"; event: KVSwapPersistenceTraceEvent; health: KVSwapPersistenceHealth }
  | { id: string; type: "loadResult"; block: SerializedKVSwapBlock | null; event: KVSwapPersistenceTraceEvent; health: KVSwapPersistenceHealth }
  | { id: string; type: "listResult"; result: KVSwapPersistenceListResult; health: KVSwapPersistenceHealth }
  | { id: string; type: "hydrateResult"; blocks: SerializedKVSwapBlock[]; event: KVSwapPersistenceTraceEvent; health: KVSwapPersistenceHealth }
  | { id: string; type: "health"; health: KVSwapPersistenceHealth }
  | { id: string; type: "error"; error: string };

type MessagePortLike = {
  postMessage(message: KVSwapPersistenceWorkerResponse): void;
  start?: () => void;
  onmessage: ((event: { data: KVSwapPersistenceWorkerMessage }) => void) | null;
};

type WorkerScopeLike = {
  navigator?: Pick<Navigator, "storage"> & { locks?: import("../lib/runtime/kvSwapPersistence").KVSwapLockManager };
  indexedDB?: IDBFactory;
  postMessage?: (message: KVSwapPersistenceWorkerResponse) => void;
  onmessage?: ((event: { data: KVSwapPersistenceWorkerMessage }) => void) | null;
  onconnect?: ((event: { ports: MessagePortLike[] }) => void) | null;
};

const stores = new Map<string, KVSwapPersistenceStore>();
let activeNamespace = "";
let operationQueue: Promise<void> = Promise.resolve();

export async function handleKVSwapPersistenceWorkerMessage(
  message: KVSwapPersistenceWorkerMessage,
  env: Pick<WorkerScopeLike, "navigator" | "indexedDB"> = workerScope,
): Promise<KVSwapPersistenceWorkerResponse> {
  return await enqueueWorkerOperation(() => handleKVSwapPersistenceWorkerMessageUnqueued(message, env));
}

async function handleKVSwapPersistenceWorkerMessageUnqueued(
  message: KVSwapPersistenceWorkerMessage,
  env: Pick<WorkerScopeLike, "navigator" | "indexedDB">,
): Promise<KVSwapPersistenceWorkerResponse> {
  try {
    if (message.type === "init") {
      const namespace = normalizeKVSwapNamespace(message.options.namespace);
      const store = await createBrowserKVSwapPersistence({ ...message.options, namespace }, {
        ...(env.navigator ? { navigator: env.navigator } : {}),
        ...(env.indexedDB ? { indexedDB: env.indexedDB } : {}),
        singleWorkerRoute: true,
      });
      stores.set(namespace, store);
      activeNamespace = namespace;
      return { id: message.id, type: "ready", health: store.health() };
    }

    if (message.type === "health") {
      const activeStore = requireStore(message.namespace ?? activeNamespace);
      return { id: message.id, type: "health", health: activeStore.health() };
    }
    if (message.type === "save") {
      const activeStore = requireStore(message.block.namespace);
      const event = await activeStore.save(message.block);
      return { id: message.id, type: "event", event, health: activeStore.health() };
    }
    if (message.type === "load") {
      const activeStore = requireStore(message.namespace);
      const loaded = await activeStore.load(message.namespace, message.blockId);
      return { id: message.id, type: "loadResult", ...loaded, health: activeStore.health() };
    }
    if (message.type === "delete") {
      const activeStore = requireStore(message.namespace);
      const event = await activeStore.delete(message.namespace, message.blockId);
      return { id: message.id, type: "event", event, health: activeStore.health() };
    }
    if (message.type === "list") {
      const activeStore = requireStore(message.namespace);
      const result = await activeStore.list(message.namespace);
      return { id: message.id, type: "listResult", result, health: activeStore.health() };
    }
    if (message.type === "persist") {
      const activeStore = requireStore(readPersistNamespace(message.blocks));
      const event = await activeStore.persist(message.blocks);
      return { id: message.id, type: "event", event, health: activeStore.health() };
    }
    if (message.type === "hydrate") {
      const activeStore = requireStore(message.namespace);
      const hydrated = await activeStore.hydrate(message.namespace);
      return { id: message.id, type: "hydrateResult", ...hydrated, health: activeStore.health() };
    }
    if (message.type === "evict") {
      const activeStore = requireStore(message.namespace);
      const event = await activeStore.evict(message.namespace, message.blockIds);
      return { id: message.id, type: "event", event, health: activeStore.health() };
    }
    const activeStore = requireStore(message.namespace);
    const event = await activeStore.clear(message.namespace);
    return { id: message.id, type: "event", event, health: activeStore.health() };
  } catch (error) {
    return {
      id: message.id,
      type: "error",
      error: error instanceof Error ? error.message : String(error),
    };
  }
}

function enqueueWorkerOperation<T>(operation: () => Promise<T>): Promise<T> {
  const queued = operationQueue.then(operation, operation);
  operationQueue = queued.then(
    () => undefined,
    () => undefined,
  );
  return queued;
}

function requireStore(namespace: string): KVSwapPersistenceStore {
  const normalized = normalizeKVSwapNamespace(namespace);
  const store = stores.get(normalized);
  if (!store) throw new Error(`KVSwap persistence worker has not been initialized for namespace: ${normalized}`);
  return store;
}

function readPersistNamespace(blocks: SerializedKVSwapBlock[]): string {
  const namespace = blocks[0]?.namespace ?? activeNamespace;
  if (!namespace) throw new Error("KVSwap persistence worker persist requires at least one block or an active namespace.");
  return namespace;
}

function attachPort(port: MessagePortLike): void {
  port.onmessage = (event) => {
    void handleKVSwapPersistenceWorkerMessage(event.data).then((response) => {
      port.postMessage(response);
    });
  };
  port.start?.();
}

const workerScope = globalThis as unknown as WorkerScopeLike;

workerScope.onmessage = (event) => {
  void handleKVSwapPersistenceWorkerMessage(event.data).then((response) => {
    workerScope.postMessage?.(response);
  });
};

workerScope.onconnect = (event) => {
  const port = event.ports[0];
  if (port) attachPort(port);
};
