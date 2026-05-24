import {
  buildAdvancedRuntimeGenerationPlan,
  buildAdaptiveConsolidationJobPlan,
  buildImmediateGacIngestionPlan,
  chunkText,
  makeMemoryChunks,
  redactSensitiveMemoryText,
  writeImmediateGacIngestionPlan,
  type AgentRuntimeConfig,
  type ChatMessage,
  type ContextPackTraceStore,
  type GacMemoryStore,
  type MemoryDeleteOptions,
  type MemorySearchHit,
  type MemoryStore,
  type RuntimeTraceStore,
  type RuntimeFeatureStatus,
  type RuntimeTrace,
  type AdvancedRuntimeModelProfile,
  type DeviceProfile,
  type InferenceBackendProfile
} from "@infinite-edge-agent/core";
import type { MemoryProviderMode } from "@infinite-edge-agent/core";
import type { EmbeddingClient } from "../embedding/embeddingClient";
import type { ChatClient, ChatStreamOptions } from "../llm/types";
import { timed, type BrowserMetricSink } from "../runtime/browserMetrics";
import { MemoryIngestionQueue, type MemoryIngestionFlushOptions, type MemoryIngestionQueueStats } from "./memoryIngestionQueue";
import { makeMessageId } from "./session";

export interface LocalAgentDeps {
  llm: ChatClient;
  embeddings: EmbeddingClient;
  memory: MemoryStore;
  tenantId: string;
  cellId: string;
  sessionId: string;
  systemPrompt: string;
  config: AgentRuntimeConfig;
  modelProfile: AdvancedRuntimeModelProfile;
  deviceProfile: DeviceProfile;
  backendProfile: InferenceBackendProfile;
  memoryMode: MemoryProviderMode;
}

export interface RunAgentCallbacks {
  onToken?: (token: string) => void;
  onStatus?: (status: string) => void;
  onMemoryIds?: (ids: string[]) => void;
  onRetrievedMemory?: (memory: RetrievedMemoryDetail[]) => void;
  onRuntimeTrace?: (trace: RuntimeTrace, features: RuntimeFeatureStatus[]) => void;
  onMetric?: (name: string, valueMs: number) => void;
}

export class LocalAgent {
  private messages: ChatMessage[] = [];
  private readonly memoryQueue = new MemoryIngestionQueue();

  constructor(private readonly deps: LocalAgentDeps) {}

  get transcript(): ChatMessage[] {
    return [...this.messages];
  }

  async submitUserMessage(content: string, callbacks: RunAgentCallbacks = {}): Promise<ChatMessage> {
    assertContextPackTraceStore(this.deps.memory);
    const memoryStore = this.deps.memory;
    const totalStarted = performance.now();
    const metricSink: BrowserMetricSink = {
      addMetric: (name, valueMs) => callbacks.onMetric?.(name, valueMs),
    };
    const userMessage: ChatMessage = {
      id: makeMessageId(),
      role: "user",
      content,
      createdAt: new Date().toISOString(),
      sessionId: this.deps.sessionId
    };
    this.messages.push(userMessage);

    callbacks.onStatus?.("Queueing user memory...");
    this.queueRememberMessage(userMessage, "memory_upsert_user_ms", callbacks, metricSink);

    const queryEmbedding = await timed("embedding_query_ms", metricSink, () => this.deps.embeddings.embed(content));
    callbacks.onStatus?.("Searching long-term memory...");
    const retrievedMemory = await timed("memory_search_ms", metricSink, () => this.deps.memory.search(queryEmbedding, {
      limit: this.deps.config.memoryTopK,
      minScore: 0.15,
      tenantId: this.deps.tenantId,
      cellId: this.deps.cellId,
      sessionId: this.deps.sessionId
    }));
    callbacks.onRetrievedMemory?.(retrievedMemory.map(toRetrievedMemoryDetail));

    const runtimePlan = await timed("runtime_plan_ms", metricSink, () => buildAdvancedRuntimeGenerationPlan({
      requestId: userMessage.id,
      tenantId: this.deps.tenantId,
      cellId: this.deps.cellId,
      sessionId: this.deps.sessionId,
      systemPrompt: this.deps.systemPrompt,
      userMessage: content,
      recentMessages: this.messages.slice(0, -1),
      retrievedMemory,
      config: this.deps.config,
      backend: this.deps.backendProfile,
      model: this.deps.modelProfile,
      device: this.deps.deviceProfile,
      memoryMode: this.deps.memoryMode,
      memoryStore
    }));
    callbacks.onMemoryIds?.(runtimePlan.packed.includedMemoryIds);
    callbacks.onRuntimeTrace?.(runtimePlan.trace, runtimePlan.features);

    callbacks.onStatus?.(`Generating with ~${runtimePlan.packed.estimatedTokens} prompt tokens...`);
    let assistantText = "";
    const generationStarted = performance.now();
    let firstTokenAt: number | null = null;
    for await (const token of this.deps.llm.streamChat(runtimePlan.packed.messages, makeGenerationStreamOptions(this.deps.config))) {
      if (firstTokenAt === null) {
        firstTokenAt = performance.now();
        callbacks.onMetric?.("generation_ttft_ms", firstTokenAt - generationStarted);
      }
      assistantText += token;
      callbacks.onToken?.(token);
    }
    callbacks.onMetric?.("generation_ms", performance.now() - generationStarted);
    void flushKvPersistenceIfAvailable(this.deps.llm).catch((error) => {
      callbacks.onStatus?.(`KV persistence sync failed: ${error instanceof Error ? error.message : String(error)}`);
    });
    const runtimeTrace = attachGenerationDecodeProof(runtimePlan.trace, this.deps.llm.lastDecodeProof);
    await timed("runtime_trace_persist_ms", metricSink, () => this.persistRuntimeTrace(runtimeTrace));
    callbacks.onRuntimeTrace?.(runtimeTrace, runtimePlan.features);

    const assistantMessage: ChatMessage = {
      id: makeMessageId(),
      role: "assistant",
      content: assistantText,
      createdAt: new Date().toISOString(),
      sessionId: this.deps.sessionId,
      metadata: {
        includedMemoryIds: runtimePlan.packed.includedMemoryIds,
        estimatedPromptTokens: runtimePlan.packed.estimatedTokens,
        runtimeTraceId: runtimeTrace.traceId,
        runtime: runtimeTrace.runtime
      }
    };
    this.messages.push(assistantMessage);

    callbacks.onStatus?.("Queueing assistant memory...");
    this.queueRememberMessage(assistantMessage, "memory_upsert_assistant_ms", callbacks, metricSink);
    callbacks.onMetric?.("agent_turn_total_ms", performance.now() - totalStarted);
    callbacks.onStatus?.(this.memoryQueue.stats.pending > 0 ? "Ready (memory syncing...)" : "Ready");
    return assistantMessage;
  }

  async clearMemory(): Promise<void> {
    await this.memoryQueue.flush();
    this.messages = [];
    await this.deps.memory.clear();
    await clearKvPersistenceIfAvailable(this.deps.llm);
  }

  flushMemoryIngestion(options: MemoryIngestionFlushOptions = {}): Promise<MemoryIngestionQueueStats> {
    return this.memoryQueue.flush(options);
  }

  deleteTranscript(options: MemoryDeleteOptions): number {
    if (!options.sessionId || options.tags?.length) return 0;
    const before = this.messages.length;
    this.messages = this.messages.filter((message) => message.sessionId !== options.sessionId);
    return before - this.messages.length;
  }

  private async rememberMessage(message: ChatMessage): Promise<void> {
    const redacted = redactSensitiveMemoryText(message.content);
    const chunks = chunkText(redacted.text, {
      chunkTokens: 220,
      overlapTokens: 40,
      minChunkTokens: 4
    });
    if (chunks.length === 0) return;
    const embeddings = await this.deps.embeddings.embedBatch(chunks.map((chunk) => chunk.text));
    const memoryChunks = makeMemoryChunks({
      text: redacted.text,
      embeddings,
      sessionId: message.sessionId,
      source: "chat",
      role: message.role,
      tags: redacted.findings.length > 0 ? [message.role, "sensitive_redacted"] : [message.role],
      metadata: {
        edgeTenantId: this.deps.tenantId,
        edgeCellId: this.deps.cellId,
        messageId: message.id,
        ...(redacted.findings.length > 0
          ? {
              sensitiveFindings: redacted.findings.map((finding) => finding.kind),
              redactedBeforeEmbedding: true
            }
          : {})
      },
      chunkOptions: {
        chunkTokens: 220,
        overlapTokens: 40,
        minChunkTokens: 4
      }
    });
    const ingestionPlan = buildImmediateGacIngestionPlan({
      tenantId: this.deps.tenantId,
      cellId: this.deps.cellId,
      sessionId: message.sessionId,
      sourceType: "chat",
      sourceUri: `chat://${message.sessionId}/${message.id}`,
      chunks: memoryChunks,
      now: new Date()
    });
    await this.deps.memory.upsert(ingestionPlan.chunks);
    await writeImmediateGacIngestionPlan(this.deps.memory as Partial<GacMemoryStore>, ingestionPlan);
    await this.scheduleAdaptiveConsolidationJob();
  }

  private async persistRuntimeTrace(trace: RuntimeTrace): Promise<void> {
    if (hasRuntimeTraceStore(this.deps.memory)) {
      await this.deps.memory.writeRuntimeTrace(trace);
    }
  }

  private queueRememberMessage(
    message: ChatMessage,
    metricName: string,
    callbacks: RunAgentCallbacks,
    metricSink: BrowserMetricSink,
  ): void {
    const job = this.memoryQueue.enqueue(`${message.role}_${message.id}`, () =>
      timed(metricName, metricSink, () => this.rememberMessage(message))
    );
    void job.settled.then((result) => {
      if (!result.ok) {
        callbacks.onStatus?.(`Memory ingestion failed: ${result.error ?? "unknown error"}`);
      } else if (this.memoryQueue.stats.pending === 0) {
        callbacks.onStatus?.("Ready");
      }
    });
  }

  private async scheduleAdaptiveConsolidationJob(): Promise<void> {
    const store = this.deps.memory as Partial<GacMemoryStore>;
    if (
      typeof store.listRawMemory !== "function"
      || typeof store.listIdentityPins !== "function"
      || typeof store.listRetrievalAudits !== "function"
      || typeof store.writeConsolidationRuns !== "function"
    ) {
      return;
    }
    const listOptions = {
      tenantId: this.deps.tenantId,
      cellId: this.deps.cellId,
      sessionId: this.deps.sessionId,
      limit: 128,
    };
    const [rawMemory, identityPins, retrievalAudits] = await Promise.all([
      store.listRawMemory(listOptions),
      store.listIdentityPins(listOptions),
      store.listRetrievalAudits(listOptions),
    ]);
    const plan = buildAdaptiveConsolidationJobPlan({
      tenantId: this.deps.tenantId,
      cellId: this.deps.cellId,
      sessionId: this.deps.sessionId,
      rawMemory,
      identityPins,
      retrievalAudits,
      minCandidateCount: 2,
      now: new Date(),
    });
    if (plan.consolidationRun) {
      await store.writeConsolidationRuns([plan.consolidationRun]);
    }
  }
}

function makeGenerationStreamOptions(config: AgentRuntimeConfig): ChatStreamOptions {
  return config.maxGenerationTokens === undefined
    ? {}
    : { maxTokens: config.maxGenerationTokens };
}

export interface RetrievedMemoryDetail {
  id: string;
  score: number;
  sessionId: string;
  source: string;
  role?: string;
  createdAt: string;
  tags: string[];
  tokenCount: number;
  textPreview: string;
  rawMemoryIds: string[];
  representativeId?: string;
  identityPinId?: string;
}

function toRetrievedMemoryDetail(hit: MemorySearchHit): RetrievedMemoryDetail {
  const metadata = getMergedMetadata(hit.metadata);
  const detail: RetrievedMemoryDetail = {
    id: hit.id,
    score: Number(hit.score.toFixed(4)),
    sessionId: hit.sessionId,
    source: hit.source,
    ...(hit.role ? { role: hit.role } : {}),
    createdAt: hit.createdAt,
    tags: hit.tags,
    tokenCount: hit.tokenCount,
    textPreview: hit.text.slice(0, 360),
    rawMemoryIds: readStringList(metadata, "rawMemoryIds", "rawMemoryId"),
  };
  const representativeId = readString(metadata, "representativeId");
  const identityPinId = readString(metadata, "identityPinId");
  if (representativeId) detail.representativeId = representativeId;
  if (identityPinId) detail.identityPinId = identityPinId;
  return detail;
}

function attachGenerationDecodeProof(trace: RuntimeTrace, decodeProof: unknown): RuntimeTrace {
  if (!decodeProof || typeof decodeProof !== "object" || Array.isArray(decodeProof)) return trace;
  const proofRecord = decodeProof as Record<string, unknown>;
  const mtp = proofRecord.mtp;
  return {
    ...trace,
    runtime: {
      ...trace.runtime,
      generation: {
        decodeProof,
        ...(mtp && typeof mtp === "object" && !Array.isArray(mtp) ? { mtp } : {}),
      },
    },
  };
}

function getMergedMetadata(metadata: Record<string, unknown>): Record<string, unknown> {
  const gac = metadata.gac;
  return typeof gac === "object" && gac !== null && !Array.isArray(gac) ? { ...metadata, ...gac } : metadata;
}

function readString(metadata: Record<string, unknown>, key: string): string | undefined {
  const value = metadata[key];
  return typeof value === "string" && value.length > 0 ? value : undefined;
}

function readStringList(metadata: Record<string, unknown>, arrayKey: string, singleKey: string): string[] {
  const arrayValue = metadata[arrayKey];
  if (Array.isArray(arrayValue)) {
    return arrayValue.filter((value): value is string => typeof value === "string" && value.length > 0);
  }
  const singleValue = readString(metadata, singleKey);
  return singleValue ? [singleValue] : [];
}

function hasRuntimeTraceStore(store: MemoryStore): store is MemoryStore & RuntimeTraceStore {
  return "writeRuntimeTrace" in store && typeof store.writeRuntimeTrace === "function";
}

async function clearKvPersistenceIfAvailable(llm: ChatClient): Promise<void> {
  const candidate = llm as ChatClient & { clearKvPersistence?: () => Promise<unknown> };
  if (typeof candidate.clearKvPersistence === "function") await candidate.clearKvPersistence();
}

async function flushKvPersistenceIfAvailable(llm: ChatClient): Promise<void> {
  if (typeof llm.flushKvPersistence === "function") await llm.flushKvPersistence();
}

function assertContextPackTraceStore(store: MemoryStore): asserts store is MemoryStore & ContextPackTraceStore {
  if (
    !("writeContextPackTraces" in store)
    || typeof store.writeContextPackTraces !== "function"
    || !("listContextPackTraces" in store)
    || typeof store.listContextPackTraces !== "function"
  ) {
    throw new Error("GAC context-pack trace persistence is required before generation.");
  }
}
