import type { ContextPackTraceRecord, MemoryChunk, RuntimeTrace, StoredMemoryChunk } from "@infinite-edge-agent/core";
import { beforeEach, describe, expect, it, vi } from "vitest";

const records = new Map<string, StoredMemoryChunk>();
const runtimeTraces = new Map<string, RuntimeTrace>();
const contextPackTraces = new Map<string, ContextPackTraceRecord>();
const deletedIds: string[] = [];

vi.mock("idb", () => ({
  openDB: vi.fn(async () => ({
    transaction: vi.fn((storeName: string | string[]) => ({
      store: makeObjectStore(Array.isArray(storeName) ? storeName[0] ?? "chunks" : storeName),
      objectStore: vi.fn((name: string) => makeObjectStore(name)),
      done: Promise.resolve()
    })),
    put: vi.fn(async (storeName: string, value: StoredMemoryChunk | RuntimeTrace | ContextPackTraceRecord) => {
      putRecord(storeName, value);
    }),
    getAll: vi.fn(async (storeName: string) => getRecords(storeName)),
    getAllFromIndex: vi.fn(async (storeName: string, indexName: string, value: string) =>
      getRecords(storeName).filter((record) => matchesIndex(record, indexName, value))
    )
  }))
}));

describe("IndexedDbMemoryStore.deleteMemory", () => {
  beforeEach(() => {
    records.clear();
    runtimeTraces.clear();
    contextPackTraces.clear();
    deletedIds.length = 0;
  });

  it("rejects empty delete scope", async () => {
    const { IndexedDbMemoryStore } = await import("./indexedDbMemoryStore");
    const store = new IndexedDbMemoryStore("test-db");

    await expect(store.deleteMemory({})).rejects.toThrow("sessionId or at least one tag");
  });

  it("deletes only chunks matching the session and tag intersection", async () => {
    const { IndexedDbMemoryStore } = await import("./indexedDbMemoryStore");
    const store = new IndexedDbMemoryStore("test-db");
    await store.upsert([
      makeChunk("match", "session_1", ["project:edge-ai", "user"]),
      makeChunk("wrong_session", "session_2", ["project:edge-ai", "user"]),
      makeChunk("wrong_tag", "session_1", ["assistant"])
    ]);

    const count = await store.deleteMemory({ sessionId: "session_1", tags: ["project:edge-ai"] });

    expect(count).toBe(1);
    expect(deletedIds).toEqual(["match"]);
    await expect(store.listMemoryChunks()).resolves.toEqual([
      expect.objectContaining({ id: "wrong_session" }),
      expect.objectContaining({ id: "wrong_tag" })
    ]);
  });

  it("filters search, list, and delete by tenant and cell metadata", async () => {
    const { IndexedDbMemoryStore } = await import("./indexedDbMemoryStore");
    const store = new IndexedDbMemoryStore("test-db");
    await store.upsert([
      makeChunk("tenant_a", "session_1", ["project:edge-ai"], "tenant_a", "cell_a"),
      makeChunk("tenant_b", "session_1", ["project:edge-ai"], "tenant_b", "cell_b")
    ]);

    await expect(store.search([0.1, 0.2], {
      tenantId: "tenant_a",
      cellId: "cell_a",
      limit: 10
    })).resolves.toEqual([
      expect.objectContaining({ id: "tenant_a" })
    ]);
    await expect(store.listMemoryChunks({
      tenantId: "tenant_b",
      cellId: "cell_b"
    })).resolves.toEqual([
      expect.objectContaining({ id: "tenant_b" })
    ]);

    await expect(store.deleteMemory({
      tenantId: "tenant_a",
      cellId: "cell_a",
      tags: ["project:edge-ai"]
    })).resolves.toBe(1);
    await expect(store.listMemoryChunks({ sessionId: "session_1" })).resolves.toEqual([
      expect.objectContaining({ id: "tenant_b" })
    ]);
  });

  it("performs deterministic vector search with metadata filters", async () => {
    const { IndexedDbMemoryStore } = await import("./indexedDbMemoryStore");
    const store = new IndexedDbMemoryStore("test-db");
    await store.upsert([
      makeChunk("b_older", "session_1", ["project:edge-ai"], "tenant_a", "cell_a", { kind: "note" }, "2026-05-11T00:00:00.000Z"),
      makeChunk("a_newer", "session_1", ["project:edge-ai"], "tenant_a", "cell_a", { kind: "note" }, "2026-05-12T00:00:00.000Z"),
      makeChunk("c_other_kind", "session_1", ["project:edge-ai"], "tenant_a", "cell_a", { kind: "trace" }, "2026-05-13T00:00:00.000Z")
    ]);

    await expect(store.search([0.1, 0.2], {
      metadata: { kind: "note" },
      limit: 10
    })).resolves.toEqual([
      expect.objectContaining({ id: "a_newer", score: expect.any(Number) }),
      expect.objectContaining({ id: "b_older", score: expect.any(Number) })
    ]);
  });

  it("persists and lists context-pack traces for browser-local production memory", async () => {
    const { IndexedDbMemoryStore } = await import("./indexedDbMemoryStore");
    const store = new IndexedDbMemoryStore("test-db");
    const trace = makeContextPackTrace("ctx_1", "session_1", "tenant_a", "cell_a");

    await expect(store.writeContextPackTraces([trace])).resolves.toMatchObject({ ok: true, count: 1, traceId: "trace_1" });
    await expect(store.listContextPackTraces({ tenantId: "tenant_a", cellId: "cell_a" })).resolves.toEqual([trace]);
    await expect(store.listContextPackTraces({ tenantId: "tenant_b" })).resolves.toEqual([]);
  });
});

function makeObjectStore(storeName: string) {
  return {
    put: vi.fn(async (value: StoredMemoryChunk | RuntimeTrace | ContextPackTraceRecord) => {
      putRecord(storeName, value);
    }),
    getAll: vi.fn(async () => getRecords(storeName)),
    delete: vi.fn(async (id: string) => {
      if (storeName === "chunks") {
        deletedIds.push(id);
        records.delete(id);
      }
    }),
    clear: vi.fn(async () => {
      if (storeName === "chunks") records.clear();
      if (storeName === "runtimeTraces") runtimeTraces.clear();
      if (storeName === "contextPackTraces") contextPackTraces.clear();
    }),
  };
}

function putRecord(storeName: string, value: StoredMemoryChunk | RuntimeTrace | ContextPackTraceRecord): void {
  if (storeName === "chunks") records.set((value as StoredMemoryChunk).id, value as StoredMemoryChunk);
  if (storeName === "runtimeTraces") runtimeTraces.set((value as RuntimeTrace).traceId, value as RuntimeTrace);
  if (storeName === "contextPackTraces") contextPackTraces.set((value as ContextPackTraceRecord).id, value as ContextPackTraceRecord);
}

function getRecords(storeName: string): Array<StoredMemoryChunk | RuntimeTrace | ContextPackTraceRecord> {
  if (storeName === "runtimeTraces") return [...runtimeTraces.values()];
  if (storeName === "contextPackTraces") return [...contextPackTraces.values()];
  return [...records.values()];
}

function matchesIndex(record: StoredMemoryChunk | RuntimeTrace | ContextPackTraceRecord, indexName: string, value: string): boolean {
  if (indexName === "by_session") return record.sessionId === value;
  if (indexName === "by_context_pack") return "contextPackId" in record && record.contextPackId === value;
  return false;
}

function makeChunk(
  id: string,
  sessionId: string,
  tags: string[],
  tenantId = "tenant_default",
  cellId = "cell_default",
  metadata: Record<string, unknown> = {},
  createdAt = "2026-05-11T00:00:00.000Z"
): MemoryChunk {
  return {
    id,
    text: id,
    embedding: [0.1, 0.2],
    sessionId,
    source: "chat",
    role: "user",
    createdAt,
    updatedAt: createdAt,
    tags,
    metadata: {
      ...metadata,
      edgeTenantId: tenantId,
      edgeCellId: cellId
    },
    tokenCount: 1
  };
}

function makeContextPackTrace(id: string, sessionId: string, tenantId: string, cellId: string): ContextPackTraceRecord {
  return {
    id,
    traceId: "trace_1",
    tenantId,
    cellId,
    sessionId,
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
}
