import {
  cosineSimilarity,
  fromStoredMemoryChunk,
  toStoredMemoryChunk,
  type ContextPackTraceRecord,
  type GacListOptions,
  type GacWriteResult,
  type MemoryChunk,
  type MemoryDeleteOptions,
  type MemorySearchHit,
  type MemorySearchOptions,
  type MemorySnapshotStore,
  type MemoryStore,
  type RuntimeTrace,
  type RuntimeTraceListOptions,
  type RuntimeTraceSnapshotStore,
  type RuntimeTraceStore,
  type StoredMemoryChunk
} from "@infinite-edge-agent/core";
import { openDB, type DBSchema, type IDBPDatabase } from "idb";

interface AgentDbSchema extends DBSchema {
  chunks: {
    key: string;
    value: StoredMemoryChunk;
    indexes: {
      by_session: string;
      by_created: string;
    };
  };
  runtimeTraces: {
    key: string;
    value: RuntimeTrace;
    indexes: {
      by_session: string;
      by_created: string;
    };
  };
  contextPackTraces: {
    key: string;
    value: ContextPackTraceRecord;
    indexes: {
      by_session: string;
      by_context_pack: string;
      by_created: string;
    };
  };
}

export class IndexedDbMemoryStore implements MemoryStore, MemorySnapshotStore, RuntimeTraceStore, RuntimeTraceSnapshotStore {
  private dbPromise: Promise<IDBPDatabase<AgentDbSchema>>;

  constructor(dbName = "infinite-edge-agent") {
    this.dbPromise = openDB<AgentDbSchema>(dbName, 3, {
      upgrade(db, oldVersion) {
        if (oldVersion < 1) {
          const chunks = db.createObjectStore("chunks", { keyPath: "id" });
          chunks.createIndex("by_session", "sessionId");
          chunks.createIndex("by_created", "createdAt");
        }
        if (oldVersion < 2) {
          const traces = db.createObjectStore("runtimeTraces", { keyPath: "traceId" });
          traces.createIndex("by_session", "sessionId");
          traces.createIndex("by_created", "createdAt");
        }
        if (oldVersion < 3) {
          const contextPackTraces = db.createObjectStore("contextPackTraces", { keyPath: "id" });
          contextPackTraces.createIndex("by_session", "sessionId");
          contextPackTraces.createIndex("by_context_pack", "contextPackId");
          contextPackTraces.createIndex("by_created", "createdAt");
        }
      }
    });
  }

  async upsert(chunks: MemoryChunk[]): Promise<void> {
    const db = await this.dbPromise;
    const tx = db.transaction("chunks", "readwrite");
    await Promise.all(chunks.map((chunk) => tx.store.put(toStoredMemoryChunk(chunk))));
    await tx.done;
  }

  async search(queryEmbedding: number[], options: MemorySearchOptions = {}): Promise<MemorySearchHit[]> {
    const db = await this.dbPromise;
    const all = await db.getAll("chunks");
    const now = Date.now();
    const minScore = options.minScore ?? -1;
    const tags = options.tags ?? [];

    return all
      .map(fromStoredMemoryChunk)
      .filter((chunk) => !options.sessionId || chunk.sessionId === options.sessionId)
      .filter((chunk) => matchesScope(chunk, options))
      .filter((chunk) => matchesMetadataFilters(chunk, options.metadata))
      .filter((chunk) => tags.length === 0 || tags.every((tag) => chunk.tags.includes(tag)))
      .filter((chunk) => {
        if (!options.maxAgeMs) return true;
        return now - new Date(chunk.createdAt).getTime() <= options.maxAgeMs;
      })
      .map((chunk) => ({ ...chunk, score: cosineSimilarity(queryEmbedding, chunk.embedding) }))
      .filter((hit) => hit.score >= minScore)
      .sort((a, b) => b.score - a.score || b.createdAt.localeCompare(a.createdAt) || a.id.localeCompare(b.id))
      .slice(0, options.limit ?? 8);
  }

  async clear(): Promise<void> {
    const db = await this.dbPromise;
    const tx = db.transaction(["chunks", "runtimeTraces", "contextPackTraces"], "readwrite");
    await Promise.all([
      tx.objectStore("chunks").clear(),
      tx.objectStore("runtimeTraces").clear(),
      tx.objectStore("contextPackTraces").clear()
    ]);
    await tx.done;
  }

  async deleteMemory(options: MemoryDeleteOptions): Promise<number> {
    assertTargetedDelete(options);
    const db = await this.dbPromise;
    const tx = db.transaction("chunks", "readwrite");
    const chunks = await tx.store.getAll();
    const ids = chunks
      .map(fromStoredMemoryChunk)
      .filter((chunk) => matchesDeleteOptions(chunk, options))
      .map((chunk) => chunk.id);
    await Promise.all(ids.map((id) => tx.store.delete(id)));
    await tx.done;
    return ids.length;
  }

  async listMemoryChunks(options: { sessionId?: string; limit?: number; tenantId?: string; cellId?: string } = {}): Promise<MemoryChunk[]> {
    const db = await this.dbPromise;
    const chunks = options.sessionId
      ? await db.getAllFromIndex("chunks", "by_session", options.sessionId)
      : await db.getAll("chunks");
    return chunks
      .map(fromStoredMemoryChunk)
      .filter((chunk) => matchesScope(chunk, options))
      .sort((a, b) => b.createdAt.localeCompare(a.createdAt))
      .slice(0, options.limit ?? chunks.length);
  }

  async importMemoryChunks(chunks: MemoryChunk[]): Promise<void> {
    await this.upsert(chunks);
  }

  async writeRuntimeTrace(trace: RuntimeTrace): Promise<void> {
    const db = await this.dbPromise;
    await db.put("runtimeTraces", trace);
  }

  async listRuntimeTraces(options: RuntimeTraceListOptions = {}): Promise<RuntimeTrace[]> {
    const db = await this.dbPromise;
    const traces = options.sessionId
      ? await db.getAllFromIndex("runtimeTraces", "by_session", options.sessionId)
      : await db.getAll("runtimeTraces");
    return traces
      .filter((trace) => !options.tenantId || trace.tenantId === options.tenantId)
      .filter((trace) => !options.cellId || trace.cellId === options.cellId)
      .sort((a, b) => b.createdAt.localeCompare(a.createdAt))
      .slice(0, options.limit ?? 20);
  }

  async importRuntimeTraces(traces: RuntimeTrace[]): Promise<void> {
    const db = await this.dbPromise;
    const tx = db.transaction("runtimeTraces", "readwrite");
    await Promise.all(traces.map((trace) => tx.store.put(trace)));
    await tx.done;
  }

  async writeContextPackTraces(records: ContextPackTraceRecord[]): Promise<GacWriteResult> {
    const traceId = records[0]?.traceId ?? makeGacTraceId();
    if (records.length === 0) return { ok: true, count: 0, traceId };
    const db = await this.dbPromise;
    const tx = db.transaction("contextPackTraces", "readwrite");
    await Promise.all(records.map((record) => tx.store.put(record)));
    await tx.done;
    return { ok: true, count: records.length, traceId };
  }

  async listContextPackTraces(options: GacListOptions = {}): Promise<ContextPackTraceRecord[]> {
    const db = await this.dbPromise;
    const traces = options.sessionId
      ? await db.getAllFromIndex("contextPackTraces", "by_session", options.sessionId)
      : options.contextPackId
        ? await db.getAllFromIndex("contextPackTraces", "by_context_pack", options.contextPackId)
        : await db.getAll("contextPackTraces");
    return traces
      .filter((trace) => !options.tenantId || trace.tenantId === options.tenantId)
      .filter((trace) => !options.cellId || trace.cellId === options.cellId)
      .filter((trace) => !options.contextPackId || trace.contextPackId === options.contextPackId)
      .sort((a, b) => b.createdAt.localeCompare(a.createdAt))
      .slice(0, options.limit ?? traces.length);
  }
}

function assertTargetedDelete(options: MemoryDeleteOptions): void {
  if (!options.sessionId && (!options.tags || options.tags.length === 0)) {
    throw new Error("deleteMemory requires a sessionId or at least one tag.");
  }
}

function matchesDeleteOptions(chunk: MemoryChunk, options: MemoryDeleteOptions): boolean {
  const tags = options.tags ?? [];
  return (!options.sessionId || chunk.sessionId === options.sessionId)
    && matchesScope(chunk, options)
    && (tags.length === 0 || tags.every((tag) => chunk.tags.includes(tag)));
}

function matchesScope(
  chunk: MemoryChunk,
  options: Pick<MemorySearchOptions, "tenantId" | "cellId">
): boolean {
  return (!options.tenantId || chunk.metadata.edgeTenantId === options.tenantId)
    && (!options.cellId || chunk.metadata.edgeCellId === options.cellId);
}

function matchesMetadataFilters(chunk: MemoryChunk, filters: Record<string, unknown> | undefined): boolean {
  if (!filters) return true;
  return Object.entries(filters).every(([key, expected]) => {
    const actual = chunk.metadata[key];
    if (Array.isArray(expected)) return expected.includes(actual);
    return actual === expected;
  });
}

function makeGacTraceId(): string {
  if (typeof crypto !== "undefined" && "randomUUID" in crypto) {
    return `gac_context_pack_trace_${crypto.randomUUID()}`;
  }
  return `gac_context_pack_trace_${Date.now()}`;
}
