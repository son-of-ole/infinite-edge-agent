import type {
  ContextPackTraceRecord,
  ContextPackTraceStore,
  MemoryChunk,
  MemoryProviderMode,
  MemorySnapshotStore,
  MemoryStore,
  RuntimeTrace,
  RuntimeTraceSnapshotStore,
  RuntimeTraceStore
} from "@infinite-edge-agent/core";

export interface MemoryExportBundle {
  version: 1;
  exportedAt: string;
  providerMode?: MemoryProviderMode;
  chunks: MemoryChunk[];
  runtimeTraces: RuntimeTrace[];
  contextPackTraces?: ContextPackTraceRecord[];
}

export async function exportMemoryBundle(
  store: MemoryStore,
  options: {
    providerMode?: MemoryProviderMode;
    sessionId?: string;
    tenantId?: string;
    cellId?: string;
    limit?: number;
  } = {}
): Promise<MemoryExportBundle> {
  if (!hasMemoryChunkList(store)) {
    throw new Error("The active memory store does not support export.");
  }
  const listOptions = {
    ...(options.sessionId ? { sessionId: options.sessionId } : {}),
    ...(options.tenantId ? { tenantId: options.tenantId } : {}),
    ...(options.cellId ? { cellId: options.cellId } : {}),
    ...(options.limit !== undefined ? { limit: options.limit } : {})
  };
  return {
    version: 1,
    exportedAt: new Date().toISOString(),
    ...(options.providerMode ? { providerMode: options.providerMode } : {}),
    chunks: await store.listMemoryChunks(listOptions),
    runtimeTraces: hasRuntimeTraceStore(store) ? await store.listRuntimeTraces(listOptions) : [],
    contextPackTraces: hasContextPackTraceStore(store) ? await store.listContextPackTraces(listOptions) : []
  };
}

export async function importMemoryBundle(store: MemoryStore, bundle: MemoryExportBundle): Promise<void> {
  if (bundle.version !== 1) {
    throw new Error(`Unsupported memory bundle version: ${bundle.version}`);
  }
  if (hasMemoryChunkImporter(store)) {
    await store.importMemoryChunks(bundle.chunks);
  } else {
    await store.upsert(bundle.chunks);
  }
  if (hasRuntimeTraceSnapshotStore(store) && bundle.runtimeTraces.length > 0) {
    await store.importRuntimeTraces(bundle.runtimeTraces);
  }
  if (hasContextPackTraceStore(store) && bundle.contextPackTraces && bundle.contextPackTraces.length > 0) {
    await store.writeContextPackTraces(bundle.contextPackTraces);
  }
}

export function hasSnapshotStore(store: MemoryStore): store is MemoryStore & MemorySnapshotStore {
  return "listMemoryChunks" in store
    && typeof store.listMemoryChunks === "function"
    && "importMemoryChunks" in store
    && typeof store.importMemoryChunks === "function";
}

function hasRuntimeTraceStore(store: MemoryStore): store is MemoryStore & RuntimeTraceStore {
  return "listRuntimeTraces" in store && typeof store.listRuntimeTraces === "function";
}

function hasRuntimeTraceSnapshotStore(store: MemoryStore): store is MemoryStore & RuntimeTraceSnapshotStore {
  return "importRuntimeTraces" in store && typeof store.importRuntimeTraces === "function";
}

function hasContextPackTraceStore(store: MemoryStore): store is MemoryStore & ContextPackTraceStore {
  return "listContextPackTraces" in store
    && typeof store.listContextPackTraces === "function"
    && "writeContextPackTraces" in store
    && typeof store.writeContextPackTraces === "function";
}

function hasMemoryChunkList(store: MemoryStore): store is MemoryStore & Pick<MemorySnapshotStore, "listMemoryChunks"> {
  return "listMemoryChunks" in store && typeof store.listMemoryChunks === "function";
}

function hasMemoryChunkImporter(store: MemoryStore): store is MemoryStore & Pick<MemorySnapshotStore, "importMemoryChunks"> {
  return "importMemoryChunks" in store && typeof store.importMemoryChunks === "function";
}
