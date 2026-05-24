import { beforeEach, describe, expect, it, vi } from "vitest";
import { deleteMemoryRequestSchema } from "./types";

const rows: LanceTestRow[] = [];
const deletedPredicates: string[] = [];
const createdTables = new Set<string>();

vi.mock("@lancedb/lancedb", () => ({
  connect: vi.fn(async () => ({
    openTable: vi.fn(async (name: string) => {
      if (!createdTables.has(name)) throw new Error(`Table not found: ${name}`);
      return table;
    }),
    createTable: vi.fn(async (name: string, initialRows: LanceTestRow[]) => {
      createdTables.add(name);
      rows.push(...initialRows);
      return table;
    }),
    dropTable: vi.fn(async (name: string) => {
      createdTables.delete(name);
    })
  }))
}));

const table = {
  add: vi.fn(async (newRows: LanceTestRow[]) => {
    rows.push(...newRows);
  }),
  vectorSearch: vi.fn(() => makeQuery(rows)),
  query: vi.fn(() => makeQuery(rows)),
  countRows: vi.fn(async (predicate: string) => rows.filter((row) => matchesPredicate(row, predicate)).length),
  delete: vi.fn(async (predicate: string) => {
    deletedPredicates.push(predicate);
    const remaining = rows.filter((row) => !matchesPredicate(row, predicate));
    rows.length = 0;
    rows.push(...remaining);
    return { numDeletedRows: rows.length - remaining.length, version: 1 };
  })
};

describe("memory-server targeted deletion", () => {
  beforeEach(() => {
    rows.length = 0;
    deletedPredicates.length = 0;
    createdTables.clear();
    vi.clearAllMocks();
  });

  it("rejects route requests without a delete scope", () => {
    expect(deleteMemoryRequestSchema.safeParse({ options: {} }).success).toBe(false);
  });

  it("deletes only rows matching the session and tag intersection", async () => {
    const { LanceMemoryStore } = await import("./lancedbStore");
    const store = new LanceMemoryStore("memory://test", "memory_chunks");
    await store.upsert([
      makeChunk("match", "session_1", ["project:edge-ai", "user"]),
      makeChunk("wrong_session", "session_2", ["project:edge-ai", "user"]),
      makeChunk("wrong_tag", "session_1", ["assistant"])
    ]);

    const count = await store.deleteMemory({ sessionId: "session_1", tags: ["project:edge-ai"] });

    expect(count).toBe(1);
    expect(deletedPredicates).toEqual(["sessionId = 'session_1' AND id IN ('match')"]);
    await expect(store.listMemoryChunks()).resolves.toEqual([
      expect.objectContaining({ id: "wrong_session" }),
      expect.objectContaining({ id: "wrong_tag" })
    ]);
  });
});

describe("LanceMemoryStore GAC persistence", () => {
  beforeEach(() => {
    rows.length = 0;
    deletedPredicates.length = 0;
    createdTables.clear();
    vi.clearAllMocks();
  });

  it("rejects model-visible representatives when no lineage is provided", async () => {
    const { LanceMemoryStore } = await import("./lancedbStore");
    const store = new LanceMemoryStore("memory://test", "memory_chunks");

    await expect(store.writeMemoryRepresentatives([
      {
        id: "rep_1",
        tenantId: "tenant_1",
        cellId: "cell_1",
        clusterId: "cluster_1",
        clusterVersion: 1,
        type: "summary",
        embedding: [0.2, 0.3],
        text: "Architecture summary",
        riskScore: 0.2,
        coverageScore: 0.9,
        createdByRunId: "run_1",
        createdAt: "2026-05-11T00:00:00.000Z",
        modelVisible: true,
        factual: true
      }
    ])).rejects.toThrow("MISSING_LINEAGE");
  });

  it("writes representatives and lineage together so model-visible records remain traceable", async () => {
    const { LanceMemoryStore } = await import("./lancedbStore");
    const store = new LanceMemoryStore("memory://test", "memory_chunks");

    const result = await store.writeMemoryRepresentatives([
      {
        id: "rep_1",
        tenantId: "tenant_1",
        cellId: "cell_1",
        clusterId: "cluster_1",
        clusterVersion: 1,
        type: "summary",
        embedding: [0.2, 0.3],
        text: "Architecture summary",
        riskScore: 0.2,
        coverageScore: 0.9,
        createdByRunId: "run_1",
        createdAt: "2026-05-11T00:00:00.000Z",
        modelVisible: true,
        factual: true
      }
    ], [
      {
        representativeId: "rep_1",
        rawMemoryId: "raw_1",
        tenantId: "tenant_1",
        cellId: "cell_1",
        membershipWeight: 1,
        distanceToRep: 0.1,
        isPrimary: true,
        createdAt: "2026-05-11T00:00:00.000Z"
      }
    ]);

    expect(result).toMatchObject({ ok: true, count: 1 });
    await expect(store.listMemoryRepresentatives({ tenantId: "tenant_1", cellId: "cell_1" })).resolves.toEqual([
      expect.objectContaining({ id: "rep_1", clusterId: "cluster_1" })
    ]);
    await expect(store.listMemoryLineage({ representativeId: "rep_1" })).resolves.toEqual([
      expect.objectContaining({ representativeId: "rep_1", rawMemoryId: "raw_1" })
    ]);
  });

  it("does not duplicate initial GAC rows when creating a table", async () => {
    const { LanceMemoryStore } = await import("./lancedbStore");
    const store = new LanceMemoryStore("memory://test", "memory_chunks");

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

    await expect(store.listContextPackTraces({ tenantId: "tenant_1", cellId: "cell_1" })).resolves.toHaveLength(1);
    expect(table.add).not.toHaveBeenCalled();
  });

  it("creates stable GAC rows with string filter columns even when record fields are absent", async () => {
    const { LanceMemoryStore } = await import("./lancedbStore");
    const store = new LanceMemoryStore("memory://test", "memory_chunks");

    await store.writeMemoryRepresentatives([
      {
        id: "rep_no_source",
        tenantId: "tenant_1",
        cellId: "cell_1",
        clusterId: "cluster_1",
        clusterVersion: 1,
        type: "centroid",
        embedding: [0.2, 0.3],
        riskScore: 0.1,
        coverageScore: 0.9,
        createdByRunId: "run_1",
        createdAt: "2026-05-11T00:00:00.000Z"
      }
    ]);

    expect(rows[0]).toMatchObject({
      tenantId: "tenant_1",
      cellId: "cell_1",
      sessionId: "",
      rawMemoryId: "",
      representativeId: "rep_no_source",
      contextPackId: ""
    });
    await expect(store.listMemoryRepresentatives({ rawMemoryId: "raw_missing" })).resolves.toEqual([]);
  });

  it("persists cluster, metric, consolidation, model action, contradiction, source document, and training rows", async () => {
    const { LanceMemoryStore } = await import("./lancedbStore");
    const store = new LanceMemoryStore("memory://test", "memory_chunks");

    await store.writeMemoryClusters([{
      id: "cluster_1",
      tenantId: "tenant_1",
      cellId: "cell_1",
      sessionId: "session_1",
      clusterVersion: 1,
      algorithm: "local_radius",
      memberCount: 2,
      status: "open",
      rawMemoryIds: ["raw_1", "raw_2"],
      createdAt: "2026-05-11T00:00:00.000Z",
      updatedAt: "2026-05-11T00:00:00.000Z",
    }]);
    await store.writeClusterMetrics([{
      id: "metric_1",
      tenantId: "tenant_1",
      cellId: "cell_1",
      sessionId: "session_1",
      clusterId: "cluster_1",
      clusterVersion: 1,
      meanDistance: 0.2,
      maxDistance: 0.4,
      medianDistance: 0.2,
      effectiveDimension: 1.2,
      theta: 0.85,
      thetaPrime: 0.55,
      contradictionScore: 0.5,
      computedAt: "2026-05-11T00:00:00.000Z",
    }]);
    await store.writeConsolidationRuns([{
      id: "run_1",
      tenantId: "tenant_1",
      cellId: "cell_1",
      sessionId: "session_1",
      mode: "immediate",
      inputCount: 2,
      clusterCount: 1,
      representativeCount: 1,
      pinCount: 1,
      status: "complete",
      startedAt: "2026-05-11T00:00:00.000Z",
      completedAt: "2026-05-11T00:00:00.000Z",
      configHash: "config_1",
    }]);
    await store.writeModelMemoryActions([{
      id: "action_1",
      tenantId: "tenant_1",
      cellId: "cell_1",
      sessionId: "session_1",
      modelId: "Qwen/Qwen3-0.6B",
      actionType: "pin_memory",
      targetIds: ["raw_1"],
      argumentsJson: { pinReason: "user_instruction" },
      confidence: 0.91,
      approvedByPolicy: true,
      createdAt: "2026-05-11T00:00:00.000Z",
    }]);
    await store.writeMemoryContradictions([{
      id: "contradiction_1",
      tenantId: "tenant_1",
      cellId: "cell_1",
      sessionId: "session_1",
      rawMemoryIds: ["raw_1", "raw_2"],
      contradictionType: "negation",
      confidence: 0.8,
      status: "open",
      createdAt: "2026-05-11T00:00:00.000Z",
    }]);
    await store.writeSourceDocuments([{
      id: "source_1",
      tenantId: "tenant_1",
      cellId: "cell_1",
      sessionId: "session_1",
      sourceUri: "https://example.test/doc",
      sourceType: "external",
      trustLevel: "untrusted",
      memoryWritePolicy: "quarantine",
      createdAt: "2026-05-11T00:00:00.000Z",
    }]);
    await store.writeTrainingExamples([{
      id: "training_1",
      tenantId: "tenant_1",
      cellId: "cell_1",
      sessionId: "session_1",
      datasetType: "raw_memory_event",
      sourceRawMemoryIds: ["raw_1"],
      inputJson: { text: "Remember this" },
      labelsJson: { pin: true },
      privacyClass: "private",
      exportAllowed: false,
      createdAt: "2026-05-11T00:00:00.000Z",
    }]);

    await expect(store.listMemoryClusters({ tenantId: "tenant_1", cellId: "cell_1" })).resolves.toEqual([
      expect.objectContaining({ id: "cluster_1", rawMemoryIds: ["raw_1", "raw_2"] }),
    ]);
    await expect(store.listClusterMetrics({ tenantId: "tenant_1", cellId: "cell_1" })).resolves.toEqual([
      expect.objectContaining({ id: "metric_1", clusterId: "cluster_1" }),
    ]);
    await expect(store.listConsolidationRuns({ tenantId: "tenant_1", cellId: "cell_1" })).resolves.toEqual([
      expect.objectContaining({ id: "run_1", mode: "immediate" }),
    ]);
    await expect(store.listModelMemoryActions({ tenantId: "tenant_1", cellId: "cell_1" })).resolves.toEqual([
      expect.objectContaining({ id: "action_1", actionType: "pin_memory" }),
    ]);
    await expect(store.listMemoryContradictions({ tenantId: "tenant_1", cellId: "cell_1" })).resolves.toEqual([
      expect.objectContaining({ id: "contradiction_1", rawMemoryIds: ["raw_1", "raw_2"] }),
    ]);
    await expect(store.listSourceDocuments({ tenantId: "tenant_1", cellId: "cell_1" })).resolves.toEqual([
      expect.objectContaining({ id: "source_1", memoryWritePolicy: "quarantine" }),
    ]);
    await expect(store.listTrainingExamples({ tenantId: "tenant_1", cellId: "cell_1" })).resolves.toEqual([
      expect.objectContaining({ id: "training_1", exportAllowed: false }),
    ]);
  });

  it("encrypts memory and GAC payload fields at rest when a string codec is configured", async () => {
    const { LanceMemoryStore } = await import("./lancedbStore");
    const codec = {
      encodeString: (value: string) => `enc(${Buffer.from(value, "utf8").toString("base64")})`,
      decodeString: (value: string) => value.startsWith("enc(") && value.endsWith(")")
        ? Buffer.from(value.slice(4, -1), "base64").toString("utf8")
        : value,
    };
    const store = new LanceMemoryStore("memory://test", "memory_chunks", { stringCodec: codec });

    await store.upsert([makeChunk("secret_chunk", "session_1", ["user"])]);
    expect(rows.find((row) => row.id === "secret_chunk")?.text).toContain("enc(");
    expect(rows.find((row) => row.id === "secret_chunk")?.text).not.toContain("secret_chunk");
    await expect(store.listMemoryChunks({ sessionId: "session_1" })).resolves.toEqual([
      expect.objectContaining({ id: "secret_chunk", text: "secret_chunk" }),
    ]);

    await store.writeRawMemory([{
      id: "raw_secret",
      tenantId: "tenant_1",
      cellId: "cell_1",
      sessionId: "session_1",
      sourceType: "chat",
      text: "private launch detail",
      memoryKind: "fact",
      importance: 0.8,
      identityRiskSeed: 0.8,
      createdAt: "2026-05-11T00:00:00.000Z",
      updatedAt: "2026-05-11T00:00:00.000Z",
      retentionClass: "normal",
      hash: "hash_secret",
    }]);

    expect(rows.find((row) => row.id === "raw_secret")?.recordJson).toContain("enc(");
    expect(rows.find((row) => row.id === "raw_secret")?.recordJson).not.toContain("private launch detail");
    await expect(store.listRawMemory({ tenantId: "tenant_1", cellId: "cell_1" })).resolves.toEqual([
      expect.objectContaining({ id: "raw_secret", text: "private launch detail" }),
    ]);
  });
});

describe("LanceMemoryStore trusted remote memory scope", () => {
  beforeEach(() => {
    rows.length = 0;
    deletedPredicates.length = 0;
    createdTables.clear();
    vi.clearAllMocks();
  });

  it("does not search or delete by forged reserved scope tags from another tenant", async () => {
    const { LanceMemoryStore } = await import("./lancedbStore");
    const store = new LanceMemoryStore("memory://test", "memory_chunks");
    await store.upsert([
      {
        ...makeChunk("forged", "session_1", ["user", "edge-tenant:tenant_b", "edge-cell:cell_b"]),
        metadata: {
          edgeTenantId: "tenant_a",
          edgeCellId: "cell_a"
        }
      }
    ]);

    await expect(store.search([0.1, 0.2], {
      tenantId: "tenant_b",
      cellId: "cell_b",
      tags: ["edge-tenant:tenant_b", "edge-cell:cell_b"]
    })).resolves.toEqual([]);
    await expect(store.deleteMemory({
      tenantId: "tenant_b",
      cellId: "cell_b",
      tags: ["edge-tenant:tenant_b", "edge-cell:cell_b"]
    })).resolves.toBe(0);
    await expect(store.listMemoryChunks({ tenantId: "tenant_a", cellId: "cell_a" })).resolves.toEqual([
      expect.objectContaining({ id: "forged" })
    ]);
  });
});

describe("LanceMemoryStore runtime trace scope", () => {
  beforeEach(() => {
    rows.length = 0;
    deletedPredicates.length = 0;
    createdTables.clear();
    vi.clearAllMocks();
  });

  it("lists runtime traces only for the requested tenant, cell, and session", async () => {
    const { LanceMemoryStore } = await import("./lancedbStore");
    const store = new LanceMemoryStore("memory://test", "memory_chunks");

    await store.writeRuntimeTrace(makeTrace("trace_a", "tenant_a", "cell_a", "session_shared", "2026-05-11T00:00:00.000Z"));
    await store.writeRuntimeTrace(makeTrace("trace_b", "tenant_b", "cell_a", "session_shared", "2026-05-11T00:00:01.000Z"));
    await store.writeRuntimeTrace(makeTrace("trace_c", "tenant_a", "cell_b", "session_shared", "2026-05-11T00:00:02.000Z"));
    await store.writeRuntimeTrace(makeTrace("trace_d", "tenant_a", "cell_a", "session_other", "2026-05-11T00:00:03.000Z"));

    await expect(store.listRuntimeTraces({
      tenantId: "tenant_a",
      cellId: "cell_a",
      sessionId: "session_shared"
    })).resolves.toEqual([
      expect.objectContaining({
        traceId: "trace_a",
        tenantId: "tenant_a",
        cellId: "cell_a",
        sessionId: "session_shared"
      })
    ]);
  });
});

interface LanceTestRow {
  id: string;
  traceId?: string;
  gacTable?: string;
  text?: string;
  vector?: number[];
  requestId?: string;
  sessionId: string;
  tenantId?: string;
  cellId?: string;
  rawMemoryId?: string;
  representativeId?: string;
  contextPackId?: string;
  source?: string;
  role?: string;
  createdAt: string;
  updatedAt?: string;
  tagsJson?: string;
  metadataJson?: string;
  tokenCount?: number;
  modelId?: string;
  backend?: string;
  runtimeJson?: string;
  recordJson?: string;
}

function makeQuery(sourceRows: LanceTestRow[], predicate?: string) {
  return {
    where(nextPredicate: string) {
      return makeQuery(sourceRows, nextPredicate);
    },
    limit() {
      return this;
    },
    async toArray() {
      return predicate ? sourceRows.filter((row) => matchesPredicate(row, predicate)) : [...sourceRows];
    }
  };
}

function matchesPredicate(row: LanceTestRow, predicate: string): boolean {
  const gacTableMatch = predicate.match(/gacTable = '([^']+)'/);
  if (gacTableMatch && row.gacTable !== gacTableMatch[1]) return false;

  const traceMatch = predicate.match(/traceId = '([^']+)'/);
  if (traceMatch && row.traceId !== traceMatch[1]) return false;

  const tenantMatch = predicate.match(/tenantId = '([^']+)'/);
  if (tenantMatch && row.tenantId !== tenantMatch[1]) return false;

  const cellMatch = predicate.match(/cellId = '([^']+)'/);
  if (cellMatch && row.cellId !== cellMatch[1]) return false;

  const sessionMatch = predicate.match(/sessionId = '([^']+)'/);
  if (sessionMatch && row.sessionId !== sessionMatch[1]) return false;

  const representativeMatch = predicate.match(/representativeId = '([^']+)'/);
  if (representativeMatch && row.representativeId !== representativeMatch[1]) return false;

  const rawMemoryMatch = predicate.match(/rawMemoryId = '([^']+)'/);
  if (rawMemoryMatch && row.rawMemoryId !== rawMemoryMatch[1]) return false;

  const idsMatch = predicate.match(/id IN \((.*)\)/);
  if (idsMatch?.[1]) {
    const ids = idsMatch[1].split(",").map((id) => id.trim().replace(/^'|'$/g, ""));
    return ids.includes(row.id);
  }

  return true;
}

function makeChunk(id: string, sessionId: string, tags: string[]) {
  return {
    id,
    text: id,
    embedding: [0.1, 0.2],
    sessionId,
    source: "chat" as const,
    role: "user" as const,
    createdAt: "2026-05-11T00:00:00.000Z",
    updatedAt: "2026-05-11T00:00:00.000Z",
    tags,
    metadata: {},
    tokenCount: 1
  };
}

function makeTrace(traceId: string, tenantId: string, cellId: string, sessionId: string, createdAt: string) {
  return {
    traceId,
    requestId: `request_${traceId}`,
    sessionId,
    tenantId,
    cellId,
    modelId: "model_1",
    backend: "backend_1",
    createdAt,
    runtime: { traceId }
  };
}
