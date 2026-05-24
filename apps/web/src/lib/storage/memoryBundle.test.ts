import type { ContextPackTraceStore, MemoryChunk, MemorySnapshotStore, MemoryStore, RuntimeTraceStore } from "@infinite-edge-agent/core";
import { describe, expect, it, vi } from "vitest";
import { exportMemoryBundle, importMemoryBundle } from "./memoryBundle";

const chunk: MemoryChunk = {
  id: "chunk_1",
  text: "Exportable memory",
  embedding: [0.1],
  sessionId: "session_1",
  source: "chat",
  role: "user",
  createdAt: "2026-05-12T00:00:00.000Z",
  updatedAt: "2026-05-12T00:00:00.000Z",
  tags: ["user"],
  metadata: {},
  tokenCount: 2,
};

describe("memoryBundle", () => {
  it("exports chunks and runtime traces from capable stores", async () => {
    const store = {
      listMemoryChunks: vi.fn(async () => [chunk]),
      listRuntimeTraces: vi.fn(async () => [{ traceId: "trace_1", requestId: "req_1", sessionId: "session_1", modelId: "m", backend: "b", createdAt: "2026-05-12T00:00:00.000Z", runtime: {} }]),
      listContextPackTraces: vi.fn(async () => [contextPackTrace]),
      writeContextPackTraces: vi.fn(async () => ({ ok: true, count: 1, traceId: "trace_1" })),
    } as unknown as MemoryStore & MemorySnapshotStore & RuntimeTraceStore & ContextPackTraceStore;

    const bundle = await exportMemoryBundle(store, {
      providerMode: "browser-vector",
      tenantId: "tenant_1",
      cellId: "cell_1",
      limit: 20
    });

    expect(bundle.version).toBe(1);
    expect(bundle.providerMode).toBe("browser-vector");
    expect(bundle.chunks).toEqual([chunk]);
    expect(bundle.runtimeTraces).toHaveLength(1);
    expect(bundle.contextPackTraces).toEqual([contextPackTrace]);
    expect(store.listMemoryChunks).toHaveBeenCalledWith({ tenantId: "tenant_1", cellId: "cell_1", limit: 20 });
    expect(store.listRuntimeTraces).toHaveBeenCalledWith({ tenantId: "tenant_1", cellId: "cell_1", limit: 20 });
    expect(store.listContextPackTraces).toHaveBeenCalledWith({ tenantId: "tenant_1", cellId: "cell_1", limit: 20 });
  });

  it("imports chunks through the snapshot API when available", async () => {
    const store = {
      importMemoryChunks: vi.fn(async () => undefined),
      upsert: vi.fn(async () => undefined),
    } as unknown as MemoryStore & MemorySnapshotStore;

    await importMemoryBundle(store, { version: 1, exportedAt: "2026-05-12T00:00:00.000Z", chunks: [chunk], runtimeTraces: [] });

    expect(store.importMemoryChunks).toHaveBeenCalledWith([chunk]);
    expect(store.upsert).not.toHaveBeenCalled();
  });

  it("imports context-pack traces when the active store supports trace persistence", async () => {
    const store = {
      importMemoryChunks: vi.fn(async () => undefined),
      upsert: vi.fn(async () => undefined),
      writeContextPackTraces: vi.fn(async () => ({ ok: true, count: 1, traceId: "trace_1" })),
      listContextPackTraces: vi.fn(async () => []),
    } as unknown as MemoryStore & MemorySnapshotStore & ContextPackTraceStore;

    await importMemoryBundle(store, {
      version: 1,
      exportedAt: "2026-05-12T00:00:00.000Z",
      chunks: [chunk],
      runtimeTraces: [],
      contextPackTraces: [contextPackTrace],
    });

    expect(store.writeContextPackTraces).toHaveBeenCalledWith([contextPackTrace]);
  });
});

const contextPackTrace = {
  id: "ctx_1",
  traceId: "trace_1",
  tenantId: "tenant_1",
  cellId: "cell_1",
  sessionId: "session_1",
  queryId: "req_1",
  contextPackId: "pack_1",
  rawMemoryIds: ["raw_1"],
  representativeIds: [],
  identityPinIds: [],
  tokenBudget: 100,
  estimatedTokens: 10,
  packingStrategy: "test",
  includedMemoryIds: ["raw_1"],
  createdAt: "2026-05-12T00:00:00.000Z",
};
