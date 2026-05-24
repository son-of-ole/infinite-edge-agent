import type {
  ClusterMetricRecord,
  ConsolidationRunRecord,
  ContextPackTraceRecord,
  GacWriteResult,
  IdentityPinRecord,
  MemoryChunk,
  MemoryClusterRecord,
  MemoryContradictionRecord,
  MemoryDeleteOptions,
  MemoryLineageRecord,
  MemoryRepresentativeRecord,
  MemorySearchHit,
  MemoryStore,
  RawMemoryRecord,
  RetrievalAuditRecord,
  TrainingExampleRecord,
} from "@infinite-edge-agent/core";
import { describe, expect, it, vi } from "vitest";
import type { EmbeddingClient } from "../embedding/embeddingClient";
import type { ChatClient } from "../llm/types";
import { LocalAgent } from "./localAgent";

describe("LocalAgent scoped transcript deletion", () => {
  it("drops private transcript messages for the current session", () => {
    const agent = makeAgent("session_1");

    setTranscript(agent, [
      makeMessage("msg_1", "session_1", "delete me"),
      makeMessage("msg_2", "session_2", "keep me")
    ]);

    const deleted = agent.deleteTranscript({ sessionId: "session_1" });

    expect(deleted).toBe(1);
    expect(agent.transcript.map((message) => message.content)).toEqual(["keep me"]);
  });

  it("does not drop transcript messages for tag-only memory deletion", () => {
    const agent = makeAgent("session_1");
    setTranscript(agent, [makeMessage("msg_1", "session_1", "visible chat")]);

    const deleted = agent.deleteTranscript({ tags: ["project:edge-ai"] });

    expect(deleted).toBe(0);
    expect(agent.transcript.map((message) => message.content)).toEqual(["visible chat"]);
  });

  it("blocks generation when the configured memory store cannot persist context-pack traces", async () => {
    const memory = makeMemoryStore();
    const llm = { backendId: "test", modelId: "test-model", streamChat: vi.fn() } as unknown as ChatClient;
    const agent = makeAgent("session_1", { memory, llm });

    await expect(agent.submitUserMessage("hello")).rejects.toThrow("context-pack trace");
    expect(llm.streamChat).not.toHaveBeenCalled();
  });

  it("persists a context-pack trace before streaming model output", async () => {
    const writes: ContextPackTraceRecord[][] = [];
    const memory = {
      ...makeMemoryStore(),
      writeContextPackTraces: vi.fn(async (records: ContextPackTraceRecord[]): Promise<GacWriteResult> => {
        writes.push(records);
        return { ok: true, count: records.length, traceId: records[0]?.traceId ?? "trace_missing" };
      }),
      listContextPackTraces: vi.fn(async () => [] as ContextPackTraceRecord[]),
    };
    const llm = {
      backendId: "test",
      modelId: "test-model",
      streamChat: vi.fn(async function* () {
        yield "ok";
      }),
    } as unknown as ChatClient;
    const agent = makeAgent("session_1", { memory, llm });

    const message = await agent.submitUserMessage("hello");

    expect(message.content).toBe("ok");
    expect(writes).toHaveLength(1);
    expect(llm.streamChat).toHaveBeenCalled();
  });

  it("passes the configured request generation budget to the chat client", async () => {
    const llm = {
      backendId: "unlocked-browser-transformer",
      modelId: "test-model",
      streamChat: vi.fn(async function* () {
        yield "ok";
      }),
    } as unknown as ChatClient;
    const agent = makeAgent("session_1", { llm, memory: makeContextTraceMemoryStore() });

    await agent.submitUserMessage("hello");

    expect(llm.streamChat).toHaveBeenCalledWith(expect.any(Array), { maxTokens: 128 });
  });

  it("starts deferred KV persistence after streamed generation completes", async () => {
    const flushKvPersistence = vi.fn(async () => undefined);
    const llm = {
      backendId: "unlocked-browser-transformer",
      modelId: "test-model",
      flushKvPersistence,
      streamChat: vi.fn(async function* () {
        yield "ok";
      }),
    } as unknown as ChatClient;
    const agent = makeAgent("session_1", { llm, memory: makeContextTraceMemoryStore() });

    await agent.submitUserMessage("hello");

    expect(flushKvPersistence).toHaveBeenCalledOnce();
  });

  it("clears KV persistence when clearing private session memory", async () => {
    const clearKvPersistence = vi.fn(async () => undefined);
    const memory = makeMemoryStore();
    const agent = makeAgent("session_1", {
      memory,
      llm: {
        backendId: "unlocked-browser-transformer",
        modelId: "test-model",
        clearKvPersistence,
        streamChat: vi.fn(),
      } as unknown as ChatClient,
    });

    await agent.clearMemory();

    expect(memory.clear).toHaveBeenCalled();
    expect(clearKvPersistence).toHaveBeenCalledOnce();
  });

  it("persists context-pack traces under the configured tenant and cell", async () => {
    const writes: ContextPackTraceRecord[][] = [];
    const memory = {
      ...makeMemoryStore(),
      writeContextPackTraces: vi.fn(async (records: ContextPackTraceRecord[]): Promise<GacWriteResult> => {
        writes.push(records);
        return { ok: true, count: records.length, traceId: records[0]?.traceId ?? "trace_missing" };
      }),
      listContextPackTraces: vi.fn(async () => [] as ContextPackTraceRecord[]),
    };
    const llm = {
      backendId: "test",
      modelId: "test-model",
      streamChat: vi.fn(async function* () {
        yield "ok";
      }),
    } as unknown as ChatClient;
    const agent = makeAgent("session_1", {
      memory,
      llm,
      tenantId: "tenant_configured",
      cellId: "cell_configured",
    });

    await agent.submitUserMessage("hello");

    expect(writes[0]?.[0]).toMatchObject({
      tenantId: "tenant_configured",
      cellId: "cell_configured",
      sessionId: "session_1",
    });
  });

  it("attaches generation decode proof to assistant metadata and persisted runtime trace after streaming", async () => {
    const runtimeWrites: unknown[] = [];
    const decodeProof = {
      mtp: {
        mode: "draft_verify",
        draftModelId: "browser/ngram-drafter",
        draftSource: "local_tokenizer_ngram",
        acceptedTokens: 1,
        rejectedTokens: 0,
        correctedTokens: 0,
        latencyDisablePolicy: "unsupported_without_target_baseline",
      },
    };
    const memory = {
      ...makeContextTraceMemoryStore(),
      writeRuntimeTrace: vi.fn(async (trace) => {
        runtimeWrites.push(trace);
      }),
    };
    const llm = {
      backendId: "test",
      modelId: "test-model",
      lastDecodeProof: decodeProof,
      streamChat: vi.fn(async function* () {
        yield "ok";
      }),
    } as unknown as ChatClient;
    const agent = makeAgent("session_1", { memory, llm });

    const message = await agent.submitUserMessage("hello");

    expect(message.metadata).toMatchObject({
      runtime: {
        generation: {
          decodeProof,
          mtp: decodeProof.mtp,
        },
      },
    });
    expect(runtimeWrites).toEqual([
      expect.objectContaining({
        runtime: expect.objectContaining({
          generation: {
            decodeProof,
            mtp: decodeProof.mtp,
          },
        }),
      }),
    ]);
  });

  it("does not publish the post-generation decode proof callback before runtime trace persistence succeeds", async () => {
    const traces: unknown[] = [];
    const decodeProof = {
      mtp: {
        mode: "draft_verify",
        draftModelId: "browser/ngram-drafter",
      },
    };
    const memory = {
      ...makeContextTraceMemoryStore(),
      writeRuntimeTrace: vi.fn(async () => {
        throw new Error("runtime trace unavailable");
      }),
    };
    const llm = {
      backendId: "test",
      modelId: "test-model",
      lastDecodeProof: decodeProof,
      streamChat: vi.fn(async function* () {
        yield "ok";
      }),
    } as unknown as ChatClient;
    const agent = makeAgent("session_1", { memory, llm });

    await expect(agent.submitUserMessage("hello", {
      onRuntimeTrace: (trace) => traces.push(trace),
    })).rejects.toThrow("runtime trace unavailable");

    expect(traces).toHaveLength(1);
    expect(traces[0]).toMatchObject({
      runtime: expect.not.objectContaining({
        generation: expect.anything(),
      }),
    });
  });

  it("reports retrieval details and timing metrics for operator inspection", async () => {
    const retrieved: unknown[] = [];
    const metrics: string[] = [];
    const memory = {
      ...makeMemoryStore(),
      search: vi.fn(async () => [{
        id: "chunk_1",
        text: "Remember this production detail",
        embedding: [0.1, 0.2],
        sessionId: "session_1",
        source: "chat" as const,
        role: "user" as const,
        createdAt: "2026-05-12T00:00:00.000Z",
        updatedAt: "2026-05-12T00:00:00.000Z",
        tags: ["project:edge-ai"],
        metadata: { gac: { rawMemoryIds: ["raw_1"], representativeId: "rep_1" } },
        tokenCount: 4,
        score: 0.92,
      }]),
      writeContextPackTraces: vi.fn(async (records: ContextPackTraceRecord[]): Promise<GacWriteResult> => ({
        ok: true,
        count: records.length,
        traceId: records[0]?.traceId ?? "trace_missing",
      })),
      listContextPackTraces: vi.fn(async () => [] as ContextPackTraceRecord[]),
    };
    const llm = {
      backendId: "test",
      modelId: "test-model",
      streamChat: vi.fn(async function* () {
        yield "ok";
      }),
    } as unknown as ChatClient;
    const agent = makeAgent("session_1", { memory, llm });

    await agent.submitUserMessage("hello", {
      onRetrievedMemory: (details) => retrieved.push(...details),
      onMetric: (name) => metrics.push(name),
    });

    expect(retrieved).toMatchObject([{
      id: "chunk_1",
      score: 0.92,
      rawMemoryIds: ["raw_1"],
      representativeId: "rep_1",
    }]);
    expect(metrics).toContain("embedding_query_ms");
    expect(metrics).toContain("memory_search_ms");
    expect(metrics).toContain("generation_ms");
  });

  it("queues memory writes in the background while generation continues", async () => {
    let releaseUpsert: (() => void) | undefined;
    const upsertStarted = vi.fn();
    let upsertCalls = 0;
    const memory = {
      ...makeContextTraceMemoryStore(),
      upsert: vi.fn(async () => {
        upsertCalls += 1;
        upsertStarted();
        if (upsertCalls > 1) return;
        await new Promise<void>((resolve) => {
          releaseUpsert = resolve;
        });
      }),
    };
    const llm = {
      backendId: "test",
      modelId: "test-model",
      streamChat: vi.fn(async function* () {
        yield "ok";
      }),
    } as unknown as ChatClient;
    const agent = makeAgent("session_1", { memory, llm });

    const message = await agent.submitUserMessage("remember this detail");

    expect(message.content).toBe("ok");
    expect(upsertStarted).toHaveBeenCalled();
    releaseUpsert?.();
    await expect(agent.flushMemoryIngestion({ throwOnError: true })).resolves.toMatchObject({
      failed: 0,
    });
  });

  it("writes GAC raw memory, identity pins, representatives, lineage, and cluster metrics during background ingestion", async () => {
    const writes = {
      raw: [] as RawMemoryRecord[][],
      pins: [] as IdentityPinRecord[][],
      reps: [] as MemoryRepresentativeRecord[][],
      lineage: [] as MemoryLineageRecord[][],
      clusters: [] as MemoryClusterRecord[][],
      metrics: [] as ClusterMetricRecord[][],
      runs: [] as ConsolidationRunRecord[][],
      contradictions: [] as MemoryContradictionRecord[][],
      training: [] as TrainingExampleRecord[][],
    };
    const memory = {
      ...makeContextTraceMemoryStore(),
      writeRawMemory: vi.fn(async (records: RawMemoryRecord[]): Promise<GacWriteResult> => {
        writes.raw.push(records);
        return { ok: true, count: records.length, traceId: "raw_trace" };
      }),
      writeIdentityPins: vi.fn(async (records: IdentityPinRecord[]): Promise<GacWriteResult> => {
        writes.pins.push(records);
        return { ok: true, count: records.length, traceId: "pin_trace" };
      }),
      writeMemoryRepresentatives: vi.fn(async (records: MemoryRepresentativeRecord[], lineage?: MemoryLineageRecord[]): Promise<GacWriteResult> => {
        writes.reps.push(records);
        writes.lineage.push(lineage ?? []);
        return { ok: true, count: records.length, traceId: "rep_trace" };
      }),
      writeMemoryClusters: vi.fn(async (records: MemoryClusterRecord[]): Promise<GacWriteResult> => {
        writes.clusters.push(records);
        return { ok: true, count: records.length, traceId: "cluster_trace" };
      }),
      writeClusterMetrics: vi.fn(async (records: ClusterMetricRecord[]): Promise<GacWriteResult> => {
        writes.metrics.push(records);
        return { ok: true, count: records.length, traceId: "metric_trace" };
      }),
      writeConsolidationRuns: vi.fn(async (records: ConsolidationRunRecord[]): Promise<GacWriteResult> => {
        writes.runs.push(records);
        return { ok: true, count: records.length, traceId: "run_trace" };
      }),
      writeMemoryContradictions: vi.fn(async (records: MemoryContradictionRecord[]): Promise<GacWriteResult> => {
        writes.contradictions.push(records);
        return { ok: true, count: records.length, traceId: "contradiction_trace" };
      }),
      writeTrainingExamples: vi.fn(async (records: TrainingExampleRecord[]): Promise<GacWriteResult> => {
        writes.training.push(records);
        return { ok: true, count: records.length, traceId: "training_trace" };
      }),
    };
    const llm = {
      backendId: "test",
      modelId: "test-model",
      streamChat: vi.fn(async function* () {
        yield "ok";
      }),
    } as unknown as ChatClient;
    const agent = makeAgent("session_1", {
      memory,
      llm,
      tenantId: "tenant_configured",
      cellId: "cell_configured",
    });

    await agent.submitUserMessage("Remember: do not use Sandbox in production.");
    await agent.flushMemoryIngestion({ throwOnError: true });

    expect(writes.raw.flat()).toEqual(expect.arrayContaining([
      expect.objectContaining({
        tenantId: "tenant_configured",
        cellId: "cell_configured",
        memoryKind: "instruction",
      }),
    ]));
    expect(writes.pins.flat()).toEqual(expect.arrayContaining([
      expect.objectContaining({ pinReason: "user_instruction" }),
    ]));
    expect(writes.reps.flat()).toEqual(expect.arrayContaining([
      expect.objectContaining({ type: "pin_shadow", modelVisible: true }),
    ]));
    expect(writes.lineage.flat()).toEqual(expect.arrayContaining([
      expect.objectContaining({ isPrimary: true }),
    ]));
    expect(writes.clusters.flat()).toHaveLength(2);
    expect(writes.metrics.flat()).toHaveLength(2);
    expect(writes.runs.flat()).toHaveLength(2);
    expect(writes.training.flat()).toEqual(expect.arrayContaining([
      expect.objectContaining({ exportAllowed: false, privacyClass: "private" }),
    ]));
  });

  it("schedules adaptive consolidation jobs from persisted GAC state after ingestion", async () => {
    const consolidationRuns: ConsolidationRunRecord[][] = [];
    const now = "2026-05-11T00:00:00.000Z";
    const memory = {
      ...makeContextTraceMemoryStore(),
      writeConsolidationRuns: vi.fn(async (records: ConsolidationRunRecord[]): Promise<GacWriteResult> => {
        consolidationRuns.push(records);
        return { ok: true, count: records.length, traceId: records[0]?.id ?? "run_trace" };
      }),
      listRawMemory: vi.fn(async (): Promise<RawMemoryRecord[]> => [
        {
          id: "raw_pin",
          tenantId: "tenant_configured",
          cellId: "cell_configured",
          sessionId: "session_1",
          sourceType: "chat",
          text: "Pinned instruction must remain exact.",
          memoryKind: "instruction",
          importance: 1,
          identityRiskSeed: 1,
          createdAt: now,
          updatedAt: now,
          retentionClass: "pinned",
          hash: "hash_raw_pin",
        },
        {
          id: "raw_failed",
          tenantId: "tenant_configured",
          cellId: "cell_configured",
          sessionId: "session_1",
          sourceType: "chat",
          text: "Failed retrieval raw memory must not be compressed yet.",
          memoryKind: "fact",
          importance: 0.8,
          identityRiskSeed: 0.8,
          createdAt: now,
          updatedAt: now,
          retentionClass: "normal",
          hash: "hash_raw_failed",
        },
        {
          id: "raw_candidate_1",
          tenantId: "tenant_configured",
          cellId: "cell_configured",
          sessionId: "session_1",
          sourceType: "chat",
          text: "Candidate one.",
          memoryKind: "fact",
          importance: 0.3,
          identityRiskSeed: 0.2,
          createdAt: now,
          updatedAt: now,
          retentionClass: "normal",
          hash: "hash_candidate_1",
        },
        {
          id: "raw_candidate_2",
          tenantId: "tenant_configured",
          cellId: "cell_configured",
          sessionId: "session_1",
          sourceType: "chat",
          text: "Candidate two.",
          memoryKind: "fact",
          importance: 0.3,
          identityRiskSeed: 0.2,
          createdAt: now,
          updatedAt: now,
          retentionClass: "normal",
          hash: "hash_candidate_2",
        },
      ]),
      listIdentityPins: vi.fn(async (): Promise<IdentityPinRecord[]> => [{
        id: "pin_1",
        tenantId: "tenant_configured",
        cellId: "cell_configured",
        sessionId: "session_1",
        rawMemoryId: "raw_pin",
        pinReason: "user_instruction",
        pinStrength: 1,
        createdBy: "policy",
        createdAt: now,
      }]),
      listRetrievalAudits: vi.fn(async (): Promise<RetrievalAuditRecord[]> => [{
        id: "audit_1",
        tenantId: "tenant_configured",
        cellId: "cell_configured",
        sessionId: "session_1",
        queryText: "Retrieve failed raw memory.",
        expectedRawMemoryId: "raw_failed",
        retrievedRawMemoryIds: ["raw_candidate_1"],
        retrievedRepresentativeIds: [],
        identityPreserved: false,
        failureMode: "over_pruned",
        createdAt: now,
      }]),
    };
    const llm = {
      backendId: "test",
      modelId: "test-model",
      streamChat: vi.fn(async function* () {
        yield "ok";
      }),
    } as unknown as ChatClient;
    const agent = makeAgent("session_1", {
      memory,
      llm,
      tenantId: "tenant_configured",
      cellId: "cell_configured",
    });

    await agent.submitUserMessage("hello");
    await agent.flushMemoryIngestion({ throwOnError: true });

    expect(consolidationRuns.flat()).toEqual(expect.arrayContaining([
      expect.objectContaining({
        mode: "sleep",
        status: "running",
        inputCount: 2,
        pinCount: 1,
      }),
    ]));
    expect(memory.listRawMemory).toHaveBeenCalledWith(expect.objectContaining({
      tenantId: "tenant_configured",
      cellId: "cell_configured",
      sessionId: "session_1",
    }));
  });

  it("scopes memory writes and searches to the configured tenant and cell", async () => {
    const upserted: MemoryChunk[][] = [];
    const memory = {
      ...makeMemoryStore(),
      upsert: vi.fn(async (chunks: MemoryChunk[]) => {
        upserted.push(chunks);
      }),
      search: vi.fn(async () => [] as MemorySearchHit[]),
      writeContextPackTraces: vi.fn(async (records: ContextPackTraceRecord[]): Promise<GacWriteResult> => ({
        ok: true,
        count: records.length,
        traceId: records[0]?.traceId ?? "trace_missing",
      })),
      listContextPackTraces: vi.fn(async () => [] as ContextPackTraceRecord[]),
    };
    const llm = {
      backendId: "test",
      modelId: "test-model",
      streamChat: vi.fn(async function* () {
        yield "ok";
      }),
    } as unknown as ChatClient;
    const agent = makeAgent("session_1", {
      memory,
      llm,
      tenantId: "tenant_configured",
      cellId: "cell_configured",
    });

    await agent.submitUserMessage("hello");
    await agent.flushMemoryIngestion({ throwOnError: true });

    expect(memory.search).toHaveBeenCalledWith([0.1, 0.2], expect.objectContaining({
      tenantId: "tenant_configured",
      cellId: "cell_configured",
    }));
    expect(upserted.flat()).toEqual(expect.arrayContaining([
      expect.objectContaining({
        metadata: expect.objectContaining({
          edgeTenantId: "tenant_configured",
          edgeCellId: "cell_configured",
        }),
      }),
    ]));
  });
});

function makeAgent(sessionId: string, overrides: Partial<LocalAgentDepsForTest> = {}): LocalAgent {
  return new LocalAgent({
    llm: overrides.llm ?? { backendId: "test", modelId: "test-model", streamChat: vi.fn() } as unknown as ChatClient,
    embeddings: overrides.embeddings ?? {
      init: vi.fn(),
      embed: vi.fn(async () => [0.1, 0.2]),
      embedBatch: vi.fn(async (texts: string[]) => texts.map(() => [0.1, 0.2])),
    } as unknown as EmbeddingClient,
    memory: overrides.memory ?? makeMemoryStore(),
    tenantId: overrides.tenantId ?? "tenant_test",
    cellId: overrides.cellId ?? "cell_test",
    sessionId,
    systemPrompt: "system",
    config: {
      modelId: "test-model",
      embeddingModelId: "test-embedding",
      memoryTopK: 4,
      maxRetrievedMemoryTokens: 200,
      maxRecentConversationTokens: 200,
      maxPromptTokens: 1024,
      maxGenerationTokens: 128
    },
    modelProfile: {
      modelId: "test-model",
      parameterBytes: 1,
      layers: 1,
      hiddenSize: 1,
      kvHeads: 1,
      contextWindowTokens: 1024
    },
    deviceProfile: {
      name: "test-device",
      vramBudgetBytes: 1024,
      ramBudgetBytes: 1024
    },
    backendProfile: {
      id: "test",
      label: "Test",
      mode: "custom"
    },
    memoryMode: "indexeddb"
  });
}

interface LocalAgentDepsForTest {
  llm: ChatClient;
  embeddings: EmbeddingClient;
  memory: MemoryStore;
  tenantId: string;
  cellId: string;
}

function setTranscript(agent: LocalAgent, messages: ReturnType<typeof makeMessage>[]): void {
  (agent as unknown as { messages: ReturnType<typeof makeMessage>[] }).messages = messages;
}

function makeMessage(id: string, sessionId: string, content: string) {
  return {
    id,
    sessionId,
    content,
    role: "user" as const,
    createdAt: "2026-05-11T00:00:00.000Z"
  };
}

function makeMemoryStore(): MemoryStore {
  return {
    upsert: vi.fn(async (_chunks: MemoryChunk[]) => undefined),
    search: vi.fn(async (_embedding: number[]) => [] as MemorySearchHit[]),
    deleteMemory: vi.fn(async (_options: MemoryDeleteOptions) => 0),
    clear: vi.fn(async () => undefined)
  };
}

function makeContextTraceMemoryStore(): MemoryStore & {
  writeContextPackTraces(records: ContextPackTraceRecord[]): Promise<GacWriteResult>;
  listContextPackTraces(): Promise<ContextPackTraceRecord[]>;
} {
  return {
    ...makeMemoryStore(),
    writeContextPackTraces: vi.fn(async (records: ContextPackTraceRecord[]): Promise<GacWriteResult> => ({
      ok: true,
      count: records.length,
      traceId: records[0]?.traceId ?? "trace_missing",
    })),
    listContextPackTraces: vi.fn(async () => [] as ContextPackTraceRecord[]),
  };
}
