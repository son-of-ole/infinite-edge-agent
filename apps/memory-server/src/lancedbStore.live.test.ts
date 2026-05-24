import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import * as lancedb from "@lancedb/lancedb";
import { describe, expect, it } from "vitest";
import type { ContextPackTraceRecord, IdentityPinRecord, MemoryChunk, MemoryLineageRecord, MemoryRepresentativeRecord, RawMemoryRecord, RetrievalAuditRecord, RuntimeTrace } from "@infinite-edge-agent/core";
import { LanceMemoryStore } from "./lancedbStore";

describe("LanceMemoryStore live GAC schema", () => {
  it("creates first GAC tables when optional filter fields are absent", async () => {
    const dbUri = await mkdtemp(join(tmpdir(), "edge-ai-lancedb-gac-"));
    try {
      const store = new LanceMemoryStore(dbUri, "memory_chunks");

      await store.writeContextPackTraces([
        {
          id: "ctx_1",
          traceId: "trace_1",
          tenantId: "tenant_1",
          cellId: "cell_1",
          sessionId: "session_1",
          queryId: "query_1",
          contextPackId: "pack_1",
          rawMemoryIds: [],
          representativeIds: [],
          identityPinIds: [],
          tokenBudget: 100,
          packingStrategy: "test",
          includedMemoryIds: [],
          createdAt: "2026-05-11T00:00:00.000Z"
        }
      ]);
      await store.writeMemoryRepresentatives([
        {
          id: "rep_no_optional_fields",
          tenantId: "tenant_1",
          cellId: "cell_1",
          clusterId: "cluster_1",
          clusterVersion: 1,
          type: "centroid",
          embedding: [0.1, 0.2],
          riskScore: 0.1,
          coverageScore: 0.9,
          createdByRunId: "run_1",
          createdAt: "2026-05-11T00:00:00.000Z"
        }
      ]);

      await expect(store.listContextPackTraces({ tenantId: "tenant_1", cellId: "cell_1" })).resolves.toHaveLength(1);
      await expect(store.listMemoryRepresentatives({ rawMemoryId: "raw_missing" })).resolves.toEqual([]);
    } finally {
      await rm(dbUri, { recursive: true, force: true });
    }
  });

  it("migrates an existing memory_chunks table that lacks trusted scope columns", async () => {
    const dbUri = await mkdtemp(join(tmpdir(), "edge-ai-lancedb-memory-migration-"));
    try {
      const db = await lancedb.connect(dbUri);
      await db.createTable("memory_chunks", [
        {
          id: "legacy_chunk",
          text: "legacy memory",
          vector: [0.1, 0.2],
          sessionId: "legacy_session",
          source: "chat",
          role: "user",
          createdAt: "2026-05-11T00:00:00.000Z",
          updatedAt: "2026-05-11T00:00:00.000Z",
          tagsJson: JSON.stringify(["legacy"]),
          metadataJson: JSON.stringify({}),
          tokenCount: 2
        }
      ]);

      const store = new LanceMemoryStore(dbUri, "memory_chunks");
      await store.upsert([
        {
          id: "scoped_chunk",
          text: "scoped memory",
          embedding: [0.1, 0.2],
          sessionId: "session_1",
          source: "chat",
          role: "user",
          createdAt: "2026-05-11T00:00:01.000Z",
          updatedAt: "2026-05-11T00:00:01.000Z",
          tags: ["user"],
          metadata: {
            edgeTenantId: "tenant_1",
            edgeCellId: "cell_1"
          },
          tokenCount: 2
        }
      ]);

      await expect(store.listMemoryChunks({ tenantId: "tenant_1", cellId: "cell_1" })).resolves.toEqual([
        expect.objectContaining({ id: "scoped_chunk" })
      ]);
      await expect(store.listMemoryChunks({ tenantId: "tenant_2", cellId: "cell_2" })).resolves.toEqual([]);
    } finally {
      await rm(dbUri, { recursive: true, force: true });
    }
  });

  it("recreates an incompatible memory_chunks table when embedding dimension changes", async () => {
    const dbUri = await mkdtemp(join(tmpdir(), "edge-ai-lancedb-memory-dimension-"));
    try {
      const db = await lancedb.connect(dbUri);
      await db.createTable("memory_chunks", [
        {
          id: "legacy_dim_chunk",
          text: "legacy memory",
          vector: [0.1, 0.2],
          sessionId: "legacy_session",
          source: "chat",
          role: "user",
          createdAt: "2026-05-11T00:00:00.000Z",
          updatedAt: "2026-05-11T00:00:00.000Z",
          tagsJson: JSON.stringify(["legacy"]),
          metadataJson: JSON.stringify({}),
          tokenCount: 2
        }
      ]);

      const store = new LanceMemoryStore(dbUri, "memory_chunks");
      await store.upsert([
        {
          id: "current_dim_chunk",
          text: "current memory",
          embedding: [0.1, 0.2, 0.3],
          sessionId: "session_1",
          source: "chat",
          role: "user",
          createdAt: "2026-05-11T00:00:01.000Z",
          updatedAt: "2026-05-11T00:00:01.000Z",
          tags: ["user"],
          metadata: {},
          tokenCount: 2
        }
      ]);

      await expect(store.search([0.1, 0.2, 0.3], { limit: 5 })).resolves.toEqual([
        expect.objectContaining({ id: "current_dim_chunk" })
      ]);
      await expect(store.listMemoryChunks({ sessionId: "legacy_session" })).resolves.toEqual([]);

      const tableNames = await db.tableNames();
      const backupName = tableNames.find((name) => name.startsWith("memory_chunks_incompatible_"));
      expect(backupName).toBeDefined();
      const backupTable = await db.openTable(backupName ?? "");
      await expect(backupTable.query().toArray()).resolves.toEqual([
        expect.objectContaining({ id: "legacy_dim_chunk" })
      ]);

      const status = await store.getDatabaseStatus();
      expect(status.memoryTable).toMatchObject({
        name: "memory_chunks",
        exists: true,
        vectorDimension: 3,
        requiredColumns: { tenantId: true, cellId: true, role: true }
      });
      expect(status.backupTables).toEqual([
        expect.objectContaining({ name: backupName, exists: true, rowCount: 1 })
      ]);
    } finally {
      await rm(dbUri, { recursive: true, force: true });
    }
  });

  it("persists runtime traces in LanceDB across store instances", async () => {
    const dbUri = await mkdtemp(join(tmpdir(), "edge-ai-lancedb-runtime-traces-"));
    try {
      const store = new LanceMemoryStore(dbUri, "memory_chunks");
      await store.writeRuntimeTrace(makeTrace("trace_1", "session_1", "2026-05-11T00:00:00.000Z", { pass: 1 }));
      await store.writeRuntimeTrace(makeTrace("trace_1", "session_1", "2026-05-11T00:00:01.000Z", { pass: 2 }));

      const reopened = new LanceMemoryStore(dbUri, "memory_chunks");

      await expect(reopened.listRuntimeTraces({ sessionId: "session_1" })).resolves.toEqual([
        expect.objectContaining({
          traceId: "trace_1",
          runtime: { pass: 2 }
        })
      ]);
      await expect(reopened.getDatabaseStatus()).resolves.toMatchObject({
        runtimeTraceTable: {
          name: "runtime_traces",
          exists: true,
          rowCount: 1
        }
      });
    } finally {
      await rm(dbUri, { recursive: true, force: true });
    }
  });

  it("replaces duplicate scoped memory ids on upsert and import without crossing tenant scope", async () => {
    const dbUri = await mkdtemp(join(tmpdir(), "edge-ai-lancedb-memory-dedupe-"));
    try {
      const store = new LanceMemoryStore(dbUri, "memory_chunks");

      await store.upsert([
        makeChunk("chunk_1", "session_1", "old tenant text", "tenant_1", "cell_1"),
        makeChunk("chunk_1", "session_2", "other tenant text", "tenant_2", "cell_2")
      ]);
      await store.importMemoryChunks([
        makeChunk("chunk_1", "session_1", "new tenant text", "tenant_1", "cell_1")
      ]);

      await expect(store.listMemoryChunks({ tenantId: "tenant_1", cellId: "cell_1" })).resolves.toEqual([
        expect.objectContaining({ id: "chunk_1", text: "new tenant text", sessionId: "session_1" })
      ]);
      await expect(store.listMemoryChunks({ tenantId: "tenant_2", cellId: "cell_2" })).resolves.toEqual([
        expect.objectContaining({ id: "chunk_1", text: "other tenant text", sessionId: "session_2" })
      ]);
    } finally {
      await rm(dbUri, { recursive: true, force: true });
    }
  });

  it("applies list/export limits in LanceDB-scoped reads", async () => {
    const dbUri = await mkdtemp(join(tmpdir(), "edge-ai-lancedb-memory-limit-"));
    try {
      const store = new LanceMemoryStore(dbUri, "memory_chunks");
      await store.upsert([
        makeChunk("chunk_1", "session_1", "one", "tenant_1", "cell_1"),
        makeChunk("chunk_2", "session_1", "two", "tenant_1", "cell_1"),
        makeChunk("chunk_3", "session_1", "three", "tenant_1", "cell_1")
      ]);

      await expect(store.listMemoryChunks({ tenantId: "tenant_1", cellId: "cell_1", limit: 2 })).resolves.toHaveLength(2);
    } finally {
      await rm(dbUri, { recursive: true, force: true });
    }
  });

  it("migrates a table created from role-less memory rows before browser role upserts", async () => {
    const dbUri = await mkdtemp(join(tmpdir(), "edge-ai-lancedb-memory-role-migration-"));
    try {
      const db = await lancedb.connect(dbUri);
      await db.createTable("memory_chunks", [
        {
          id: "system_chunk",
          tenantId: "",
          cellId: "",
          text: "system memory",
          vector: [0.1, 0.2],
          sessionId: "session_1",
          source: "system",
          createdAt: "2026-05-11T00:00:00.000Z",
          updatedAt: "2026-05-11T00:00:00.000Z",
          tagsJson: JSON.stringify(["system"]),
          metadataJson: JSON.stringify({}),
          tokenCount: 2
        }
      ]);

      const store = new LanceMemoryStore(dbUri, "memory_chunks");
      await store.upsert([
        {
          id: "user_chunk",
          text: "browser user memory",
          embedding: [0.1, 0.2],
          sessionId: "session_1",
          source: "chat",
          role: "user",
          createdAt: "2026-05-11T00:00:01.000Z",
          updatedAt: "2026-05-11T00:00:01.000Z",
          tags: ["user"],
          metadata: {},
          tokenCount: 2
        }
      ]);

      await expect(store.listMemoryChunks({ sessionId: "session_1" })).resolves.toEqual([
        expect.objectContaining({ id: "system_chunk" }),
        expect.objectContaining({ id: "user_chunk", role: "user" })
      ]);
      await expect(store.getDatabaseStatus()).resolves.toMatchObject({
        memoryTable: {
          requiredColumns: { tenantId: true, cellId: true, role: true }
        }
      });
    } finally {
      await rm(dbUri, { recursive: true, force: true });
    }
  });

  it("reports and repairs legacy memory table scope columns without deleting rows", async () => {
    const dbUri = await mkdtemp(join(tmpdir(), "edge-ai-lancedb-memory-repair-"));
    try {
      const db = await lancedb.connect(dbUri);
      await db.createTable("memory_chunks", [
        {
          id: "legacy_chunk",
          text: "legacy memory",
          vector: [0.1, 0.2],
          sessionId: "legacy_session",
          source: "chat",
          role: "user",
          createdAt: "2026-05-11T00:00:00.000Z",
          updatedAt: "2026-05-11T00:00:00.000Z",
          tagsJson: JSON.stringify(["legacy"]),
          metadataJson: JSON.stringify({}),
          tokenCount: 2
        }
      ]);

      const store = new LanceMemoryStore(dbUri, "memory_chunks");
      await expect(store.getDatabaseStatus()).resolves.toMatchObject({
        repaired: false,
        memoryTable: {
          exists: true,
          rowCount: 1,
          vectorDimension: 2,
          requiredColumns: { tenantId: false, cellId: false, role: true }
        }
      });
      const repaired = await store.getDatabaseStatus({ repair: true, expectedVectorDimension: 384 });

      expect(repaired).toMatchObject({
        repaired: true,
        repairs: expect.arrayContaining(["memory_chunks:vector-dimension"]),
        memoryTable: {
          exists: false,
          rowCount: null,
          vectorDimension: null,
          requiredColumns: { tenantId: false, cellId: false, role: false }
        }
      });
      expect(repaired.backupTables).toEqual([
        expect.objectContaining({ exists: true, rowCount: 1 })
      ]);
      await expect(store.search(Array.from({ length: 384 }, () => 0.1), { limit: 1 })).resolves.toEqual([]);
    } finally {
      await rm(dbUri, { recursive: true, force: true });
    }
  });

  it("repairs legacy runtime trace and GAC index columns without deleting rows", async () => {
    const dbUri = await mkdtemp(join(tmpdir(), "edge-ai-lancedb-gac-repair-"));
    try {
      const db = await lancedb.connect(dbUri);
      await db.createTable("runtime_traces", [
        {
          traceId: "legacy_trace",
          requestId: "request_legacy_trace",
          sessionId: "session_1",
          modelId: "model_1",
          backend: "backend_1",
          createdAt: "2026-05-11T00:00:00.000Z",
          runtimeJson: JSON.stringify({ legacy: true })
        }
      ]);
      await db.createTable("raw_memory", [
        {
          id: "raw_legacy",
          tenantId: "tenant_1",
          cellId: "cell_1",
          sessionId: "session_1",
          rawMemoryId: "raw_legacy",
          createdAt: "2026-05-11T00:00:00.000Z",
          recordJson: JSON.stringify(makeRawMemory("raw_legacy", "tenant_1", "cell_1", "session_1", "legacy memory"))
        }
      ]);

      const store = new LanceMemoryStore(dbUri, "memory_chunks");
      const before = await store.getDatabaseStatus();
      expect(before.runtimeTraceTable.schema.map((field) => field.name)).not.toContain("tenantId");
      expect(before.gacTables.find((table) => table.name === "raw_memory")?.schema.map((field) => field.name)).not.toContain("gacTable");

      const repaired = await store.getDatabaseStatus({ repair: true });

      expect(repaired).toMatchObject({
        repaired: true,
        repairs: expect.arrayContaining(["runtime_traces:columns", "raw_memory:columns"])
      });
      expect(repaired.runtimeTraceTable.rowCount).toBe(1);
      expect(repaired.gacTables.find((table) => table.name === "raw_memory")).toMatchObject({ rowCount: 1 });
      await expect(store.searchRawMemory({ tenantId: "tenant_1", cellId: "cell_1", queryText: "legacy" })).resolves.toEqual([
        expect.objectContaining({ id: "raw_legacy" })
      ]);
    } finally {
      await rm(dbUri, { recursive: true, force: true });
    }
  });

  it("propagates scoped memory deletion to raw and derived GAC tables", async () => {
    const dbUri = await mkdtemp(join(tmpdir(), "edge-ai-lancedb-gac-delete-"));
    try {
      const store = new LanceMemoryStore(dbUri, "memory_chunks");
      await store.upsert([
        makeChunk("raw_delete", "session_1", "delete this memory", "tenant_1", "cell_1"),
        makeChunk("raw_keep", "session_1", "keep this memory", "tenant_2", "cell_2")
      ]);
      await store.writeRawMemory([
        makeRawMemory("raw_delete", "tenant_1", "cell_1", "session_1", "delete this raw memory"),
        makeRawMemory("raw_keep", "tenant_2", "cell_2", "session_1", "keep this raw memory")
      ]);
      await store.writeIdentityPins([makeIdentityPin("pin_delete", "raw_delete", "tenant_1", "cell_1", "session_1")]);
      await store.writeMemoryRepresentatives([
        makeRepresentative("rep_delete", "raw_delete", "tenant_1", "cell_1", "session_1")
      ], [
        makeLineage("rep_delete", "raw_delete", "tenant_1", "cell_1", "session_1")
      ]);
      await store.writeRetrievalAudits([makeRetrievalAudit("audit_delete", "raw_delete", "rep_delete", "tenant_1", "cell_1", "session_1")]);
      await store.writeContextPackTraces([makeContextPackTrace("ctx_delete", "raw_delete", "rep_delete", "pin_delete", "tenant_1", "cell_1", "session_1")]);

      await expect(store.deleteMemory({ tenantId: "tenant_1", cellId: "cell_1" })).resolves.toBe(1);

      await expect(store.listRawMemory({ tenantId: "tenant_1", cellId: "cell_1" })).resolves.toEqual([]);
      await expect(store.listIdentityPins({ tenantId: "tenant_1", cellId: "cell_1" })).resolves.toEqual([]);
      await expect(store.listMemoryRepresentatives({ tenantId: "tenant_1", cellId: "cell_1" })).resolves.toEqual([]);
      await expect(store.listMemoryLineage({ tenantId: "tenant_1", cellId: "cell_1" })).resolves.toEqual([]);
      await expect(store.listRetrievalAudits({ tenantId: "tenant_1", cellId: "cell_1" })).resolves.toEqual([]);
      await expect(store.listContextPackTraces({ tenantId: "tenant_1", cellId: "cell_1" })).resolves.toEqual([]);
      await expect(store.listRawMemory({ tenantId: "tenant_2", cellId: "cell_2" })).resolves.toEqual([
        expect.objectContaining({ id: "raw_keep" })
      ]);
    } finally {
      await rm(dbUri, { recursive: true, force: true });
    }
  });

  it("searches raw memory by scope as a fallback when representatives are unavailable", async () => {
    const dbUri = await mkdtemp(join(tmpdir(), "edge-ai-lancedb-raw-fallback-"));
    try {
      const store = new LanceMemoryStore(dbUri, "memory_chunks");
      await store.writeRawMemory([
        makeRawMemory("raw_important", "tenant_1", "cell_1", "session_1", "Remember the Newton architecture decision", 0.9),
        makeRawMemory("raw_low", "tenant_1", "cell_1", "session_1", "Remember a background note", 0.1),
        makeRawMemory("raw_other_scope", "tenant_2", "cell_2", "session_1", "Remember the Newton architecture decision", 1),
        {
          ...makeRawMemory("raw_deleted", "tenant_1", "cell_1", "session_1", "Remember the deleted decision", 1),
          deletedAt: "2026-05-11T00:00:01.000Z",
          retentionClass: "user_deleted"
        }
      ]);

      await expect(store.listMemoryRepresentatives({ tenantId: "tenant_1", cellId: "cell_1" })).resolves.toEqual([]);
      await expect(store.searchRawMemory({
        tenantId: "tenant_1",
        cellId: "cell_1",
        queryText: "architecture",
        limit: 2
      })).resolves.toEqual([
        expect.objectContaining({ id: "raw_important" })
      ]);
    } finally {
      await rm(dbUri, { recursive: true, force: true });
    }
  });
});

function makeChunk(id: string, sessionId: string, text: string, tenantId: string, cellId: string): MemoryChunk {
  return {
    id,
    text,
    embedding: [0.1, 0.2],
    sessionId,
    source: "chat",
    role: "user",
    createdAt: "2026-05-11T00:00:00.000Z",
    updatedAt: "2026-05-11T00:00:00.000Z",
    tags: ["test"],
    metadata: {
      edgeTenantId: tenantId,
      edgeCellId: cellId
    },
    tokenCount: 2
  };
}

function makeTrace(traceId: string, sessionId: string, createdAt: string, runtime: Record<string, unknown>): RuntimeTrace {
  return {
    traceId,
    requestId: `request_${traceId}`,
    sessionId,
    modelId: "test-model",
    backend: "test-backend",
    createdAt,
    runtime
  };
}

function makeRawMemory(
  id: string,
  tenantId: string,
  cellId: string,
  sessionId: string,
  text: string,
  importance = 0.5
): RawMemoryRecord {
  return {
    id,
    tenantId,
    cellId,
    sessionId,
    sourceType: "chat",
    text,
    memoryKind: "decision",
    importance,
    identityRiskSeed: 0.2,
    createdAt: "2026-05-11T00:00:00.000Z",
    updatedAt: "2026-05-11T00:00:00.000Z",
    retentionClass: "normal",
    hash: `hash_${id}`
  };
}

function makeIdentityPin(id: string, rawMemoryId: string, tenantId: string, cellId: string, sessionId: string): IdentityPinRecord {
  return {
    id,
    tenantId,
    cellId,
    sessionId,
    rawMemoryId,
    pinReason: "architecture_decision",
    pinStrength: 0.9,
    createdBy: "policy",
    createdAt: "2026-05-11T00:00:00.000Z"
  };
}

function makeRepresentative(id: string, rawMemoryId: string, tenantId: string, cellId: string, sessionId: string): MemoryRepresentativeRecord {
  return {
    id,
    tenantId,
    cellId,
    sessionId,
    clusterId: "cluster_1",
    clusterVersion: 1,
    type: "summary",
    embedding: [0.1, 0.2],
    text: "decision representative",
    sourceRawMemoryId: rawMemoryId,
    riskScore: 0.2,
    coverageScore: 0.8,
    createdByRunId: "run_1",
    createdAt: "2026-05-11T00:00:00.000Z",
    modelVisible: true,
    factual: true
  };
}

function makeLineage(representativeId: string, rawMemoryId: string, tenantId: string, cellId: string, sessionId: string): MemoryLineageRecord {
  return {
    representativeId,
    rawMemoryId,
    tenantId,
    cellId,
    sessionId,
    membershipWeight: 1,
    distanceToRep: 0.1,
    isPrimary: true,
    createdAt: "2026-05-11T00:00:00.000Z"
  };
}

function makeRetrievalAudit(
  id: string,
  rawMemoryId: string,
  representativeId: string,
  tenantId: string,
  cellId: string,
  sessionId: string
): RetrievalAuditRecord {
  return {
    id,
    tenantId,
    cellId,
    sessionId,
    queryText: "decision",
    expectedRawMemoryId: rawMemoryId,
    retrievedRawMemoryIds: [rawMemoryId],
    retrievedRepresentativeIds: [representativeId],
    hitAtK: 1,
    identityPreserved: true,
    createdAt: "2026-05-11T00:00:00.000Z"
  };
}

function makeContextPackTrace(
  id: string,
  rawMemoryId: string,
  representativeId: string,
  identityPinId: string,
  tenantId: string,
  cellId: string,
  sessionId: string
): ContextPackTraceRecord {
  return {
    id,
    traceId: `trace_${id}`,
    tenantId,
    cellId,
    sessionId,
    queryId: "query_1",
    contextPackId: "pack_1",
    rawMemoryIds: [rawMemoryId],
    representativeIds: [representativeId],
    identityPinIds: [identityPinId],
    tokenBudget: 100,
    packingStrategy: "test",
    includedMemoryIds: [rawMemoryId, representativeId],
    createdAt: "2026-05-11T00:00:00.000Z"
  };
}
