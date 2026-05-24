import {
  F16Matrix,
  F32Matrix,
  UnlockedBrowserTransformerBackend,
  WebGpuRuntimeBufferCache,
  compressKeyRowsToLowRankSummary,
  createSsaToyTensorHandle,
  DEFAULT_PREFILL_DISPATCH_BUDGET_MS,
  planKVSwap,
  planPrefillChunks,
  readUnlockedBrowserDecodeHandle,
  readUnlockedBrowserKvCacheHandle,
  verifySpeculativeBatch,
  type DecodePerfSummary,
  type SSALayerRoutingPolicy,
  type DraftToken,
  type RuntimeMatrix,
  type RuntimeVector,
  type SpeculativeBatchMetrics,
  type TargetVerificationToken,
  type UnlockedBrowserKvCacheHandle,
  type UnlockedBrowserPrefillBackendProof,
  type UnlockedBrowserTransformerWeights,
  type PrefillChunkPlan,
  type WebGpuSsaBackendOptions,
} from "@infinite-edge-agent/core";
import type { KVBlock, KVLowRankKeySummary, KVLowRankQuerySummary } from "@infinite-edge-agent/core";
import {
  createBrowserKVSwapPersistence,
  KV_SWAP_STORAGE_VERSION,
  normalizeKVSwapNamespace,
  type KVSwapPredictiveHotBlockTrace,
  type KVSwapPersistenceHealth,
  type KVSwapPersistenceStore,
  type KVSwapPersistenceTraceEvent,
  type KVSwapPrefetchStrategy,
  type KVSwapLowRankQuerySource,
  type SerializedKVLowRankKeySummary,
  type SerializedKVSwapBlock,
  type SerializedKVSwapKernelBackend,
  type SerializedKVSwapPrefillProof,
} from "../runtime/kvSwapPersistence";
import type { ChatClient, ChatClientMessage, ChatStreamOptions } from "./types";

export const LOCAL_BROWSER_NGRAM_MTP_DRAFT_MODEL_ID = "browser/ngram-drafter";
export const LOCAL_BROWSER_QWEN_PREFIX_MTP_DRAFT_MODEL_ID = "browser/qwen-prefix-drafter";
export const LOCAL_BROWSER_MTP_DRAFT_MODEL_ID = LOCAL_BROWSER_QWEN_PREFIX_MTP_DRAFT_MODEL_ID;
export const LOCAL_BROWSER_MTP_DEFAULT_SPECULATIVE_TOKENS = 2;
export const LOCAL_BROWSER_MTP_MAX_SPECULATIVE_TOKENS = 3;
const UNLOCKED_PROOF_MARKER = "[unlocked:ssa-kv-tsp]";
const THINK_START_MARKER = "<think>";
const THINK_END_MARKER = "</think>";
const ASSISTANT_CONTROL_MARKERS = ["<|im_start|>"];
const ASSISTANT_STOP_MARKERS = ["<|im_end|>", "<|endoftext|>", "</s>", "<eos>"];
const OUTPUT_FILTER_MARKERS = [UNLOCKED_PROOF_MARKER, THINK_START_MARKER, THINK_END_MARKER, ...ASSISTANT_CONTROL_MARKERS, ...ASSISTANT_STOP_MARKERS];
const CLIENT_ATTENTION_BLOCK_SIZE = 16;
const CLIENT_FULL_CONTEXT_DECODE_TOKEN_LIMIT = 40_960;
const CLIENT_MAX_SPARSE_DECODE_BLOCKS = 256;
const CLIENT_SPARSE_ANCHOR_BLOCKS = 8;
const CLIENT_MAX_SPECULATIVE_CONTINUATION_TOKENS = 8;
const CLIENT_TRAILING_QUERY_BLOCKS = Math.max(2, Math.ceil((CLIENT_MAX_SPECULATIVE_CONTINUATION_TOKENS + 1) / CLIENT_ATTENTION_BLOCK_SIZE));
const CLIENT_LOW_RANK_PREFETCH_RANK = 4;
const CLIENT_LOW_RANK_PROJECTION_ID = "unlocked-browser:key-low-rank:v1";
const CLIENT_REPEAT_SUPPRESSION_WINDOW = 2;
const SHARED_MODEL_BUFFER_CACHES = new Map<string, WebGpuRuntimeBufferCache>();

export type UnlockedBrowserWarmupMode = "pipeline_preload" | "target_probe";
export type UnlockedBrowserGenerationStopReason = "stop_marker" | "stop_after_sequence" | "max_tokens";

function getSharedModelBufferCache(input: {
  modelId: string;
  manifestPath: string;
  manifestSha256: string;
}): WebGpuRuntimeBufferCache {
  const key = [
    input.modelId.trim() || "unknown-model",
    input.manifestSha256.trim() || "no-manifest-sha",
    input.manifestPath.trim() || "no-manifest-path",
  ].join("|");
  let cache = SHARED_MODEL_BUFFER_CACHES.get(key);
  if (!cache) {
    cache = new WebGpuRuntimeBufferCache();
    SHARED_MODEL_BUFFER_CACHES.set(key, cache);
  }
  return cache;
}

export type QwenThinkingMode = "disabled" | "enabled";

export interface UnlockedBrowserTransformerClientOptions {
  modelId: string;
  manifestPath?: string;
  manifestSha256?: string;
  allowFixtureWeights?: boolean;
  backendPreference?: WebGpuSsaBackendOptions["backendPreference"];
  requireWebGpu?: boolean;
  maxRuntimePromptTokens?: number | null;
  maxRuntimeLayers?: number | null;
  logitCandidateLimit?: number | null;
  logitTopK?: number | null;
  logitTileRows?: number | null;
  maxGenerationTokens?: number | null;
  qwenThinkingMode?: QwenThinkingMode;
  strictChunkedPrefill?: boolean;
  warmModelResidency?: boolean;
  warmModelResidencyMode?: UnlockedBrowserWarmupMode;
  mtp?: UnlockedBrowserMtpOptions;
  kvPersistence?: UnlockedBrowserKvPersistenceOptions;
}

export interface UnlockedBrowserMtpOptions {
  enabled?: boolean;
  draftModelId?: string | null;
  numSpeculativeTokens?: number;
  minAcceptanceRate?: number;
  disableWhenLatencyWorse?: boolean;
  draftLayerCount?: number | null;
}

export interface UnlockedBrowserKvPersistenceOptions {
  enabled?: boolean;
  namespace?: string;
  preferOpfs?: boolean;
  maxBlocks?: number;
  maxBytes?: number;
  clearOnInit?: boolean;
}

export interface UnlockedBrowserTransformerClientDisposeOptions {
  clearSharedBuffers?: boolean;
}

export interface UnlockedBrowserPrefillChunkPlan {
  prefillChunkCount: number;
  prefillChunkSize: number;
  shapeBucket: string;
  pipelineCacheKey: string;
  prefillDispatchTargetMs: number;
  maxDispatchEstimatedMs: number;
}

export interface UnlockedBrowserDecodeProof {
  tensorControl: true;
  warmupMode?: UnlockedBrowserWarmupMode;
  warmupBlockingMs?: number;
  warmupUploadedEntries?: number;
  warmupCacheHits?: number;
  residentUploadCount?: number;
  residentCacheHitCount?: number;
  residentReadbackCount?: number;
  tspSteps: string[];
  selectedBlockIds: string[];
  kvPagingEvents: number;
  tokenId: number;
  expectedLayerCount?: number;
  executedLayerCount?: number;
  mlpKernelBackends?: Array<{
    layerIndex: number;
    backend: "webgpu" | "cpu_reference";
    activationKind: "silu_gated" | "gelu";
  }>;
  prefillMlpKernelBackends?: Array<{
    layerIndex: number;
    backend: "webgpu" | "cpu_reference" | "mixed";
    activationKind: "silu_gated" | "gelu";
    rowCount: number;
  }>;
  logitProjectionBackend?: "webgpu" | "cpu_reference";
  logitProjectionSelectedRows?: number;
  logitProjectionFullRows?: number;
  logitProjectionPurpose?: "candidate_logit_projection" | "full_vocab_logit_projection" | "full_vocab_topk_logit_projection" | "greedy_argmax_logit_projection" | "compact_topk_logit_projection";
  logitProjectionReadbackStrategy?: "full_logits" | "gpu_top1_candidates" | "gpu_argmax_token_id" | "gpu_compact_topk";
  logitProjectionGpuReducedRows?: number;
  logitProjectionReadbackRows?: number;
  logitProjectionReadbackBytes?: number;
  logitProjectionDispatchCount?: number;
  logitProjectionTiles?: number;
  logitProjectionTileRows?: number;
  logitProjectionCandidateTokenIds?: number[];
  logitProjectionCandidateScores?: number[];
  compactLogitTopK?: number;
  samplingTemperature?: number;
  samplingTopP?: number;
  repetitionPenalty?: number;
  greedyDecodeUsed?: boolean;
  sampledTokenRank?: number;
  prefillChunkCount?: number;
  prefillChunkSize?: number;
  shapeBucket?: string;
  pipelineCacheKey?: string;
  prefillDispatchTargetMs?: number;
  maxDispatchEstimatedMs?: number;
  prefillChunkDispatch?: "single_dispatch" | "chunked_dispatch";
  prefillProjectionBackends?: Array<{
    layerIndex: number;
    qProjection: "webgpu" | "cpu_reference";
    kProjection: "webgpu" | "cpu_reference";
    vProjection: "webgpu" | "cpu_reference";
    oProjection: "webgpu" | "cpu_reference";
  }>;
  decodeProjectionBackends?: Array<{
    layerIndex: number;
    qProjection?: "webgpu" | "cpu_reference";
    kProjection?: "webgpu" | "cpu_reference";
    vProjection?: "webgpu" | "cpu_reference";
    oProjection?: "webgpu" | "cpu_reference";
    projectionKind: "matvec" | "matmul";
    tokens?: number;
    selectedRows?: number;
  }>;
  prefillAttentionBackends?: Array<{
    layerIndex: number;
    attentionBackend: "webgpu" | "cpu_reference" | "mixed";
    packedHeadBackends: Array<"webgpu" | "cpu_reference">;
    packedHeadCount: number;
    keyValueHeadCount?: number;
    keyValueCompressionRatio?: number;
  }>;
  decodeAttentionBackends?: Array<{
    layerIndex: number;
    attentionBackend: "webgpu" | "cpu_reference" | "mixed" | "unknown";
    packedHeadBackends: Array<"webgpu" | "cpu_reference">;
    packedHeadCount?: number;
    keyValueHeadCount?: number;
    keyValueCompressionRatio?: number;
  }>;
  mtp?: UnlockedBrowserMtpProof;
  kvPersistence?: KVSwapPersistenceHealth & {
    events: KVSwapPersistenceTraceEvent[];
  };
  decodePerf?: DecodePerfSummary;
}

interface BrowserKvPrefetchTrace {
  prefetchStrategy: KVSwapPrefetchStrategy;
  lowRankSummaryRank?: number;
  lowRankQuerySource?: KVSwapLowRankQuerySource;
  predictedHotBlocks: KVSwapPredictiveHotBlockTrace[];
  prefetchedBlocks: string[];
  prefetchHitRate: number;
  prefetchBytes: number;
  prefetchLatencyMs: number;
  attentionStallMs: number;
}

type BrowserKvLowRankQuerySummary = KVLowRankQuerySummary & {
  source: KVSwapLowRankQuerySource;
};

export interface UnlockedBrowserMtpProof {
  mode: "target_only" | "draft_verify";
  draftModelId?: string | null;
  draftSource?: "local_tokenizer_ngram" | "qwen_prefix_draft";
  verifierStrategy?: "batched_continuation";
  draftTokenIds: number[];
  draftTokens: string[];
  acceptedTokens: number;
  rejectedTokens: number;
  correctedTokens: number;
  acceptanceRate: number;
  verifierBackend?: "unlocked-browser-transformer";
  numSpeculativeTokens: number;
  verifiedTokenCount?: number;
  targetDecodeCalls?: number;
  committedInputTokens?: number;
  latencyDisablePolicy?: "unsupported_without_target_baseline" | "paired_benchmark_required";
  disabledReason?: string;
  metrics?: SpeculativeBatchMetrics;
}

function accumulateDecodePerf(
  previous: DecodePerfSummary | null,
  next: DecodePerfSummary | undefined,
): DecodePerfSummary | null {
  if (!next) return previous;
  if (!previous) return { ...next };
  const generatedTokenCount = previous.generatedTokenCount + next.generatedTokenCount;
  const prefillExecutionsDuringDecode = previous.prefillExecutionsDuringDecode + next.prefillExecutionsDuringDecode;
  const residentDecodeLayerCount = previous.residentDecodeLayerCount + next.residentDecodeLayerCount;
  const totalDecodeLayerCount = previous.totalDecodeLayerCount + next.totalDecodeLayerCount;
  const decodeSubmitCount = previous.decodeSubmitCount + next.decodeSubmitCount;
  const dispatchCount = previous.dispatchCount + next.dispatchCount;
  const fusedPackedQkvLayerCount = previous.fusedPackedQkvLayerCount + next.fusedPackedQkvLayerCount;
  const fusedQkvNormRopeKvAppendLayerCount = previous.fusedQkvNormRopeKvAppendLayerCount + next.fusedQkvNormRopeKvAppendLayerCount;
  const fusedOneTokenAttentionLayerCount = previous.fusedOneTokenAttentionLayerCount + next.fusedOneTokenAttentionLayerCount;
  const fusedResidualRmsNormLayerCount = previous.fusedResidualRmsNormLayerCount + next.fusedResidualRmsNormLayerCount;
  const fusedMlpLayerCount = previous.fusedMlpLayerCount + next.fusedMlpLayerCount;
  const fusedFullLayerCount = previous.fusedFullLayerCount + next.fusedFullLayerCount;
  const fusedLayerStageHits = fusedPackedQkvLayerCount
    + fusedQkvNormRopeKvAppendLayerCount
    + fusedOneTokenAttentionLayerCount
    + fusedResidualRmsNormLayerCount
    + fusedMlpLayerCount
    + fusedFullLayerCount;
  return {
    ...(previous.requestId || next.requestId ? { requestId: previous.requestId ?? next.requestId } : {}),
    generatedTokenCount,
    decodeCallCount: previous.decodeCallCount + next.decodeCallCount,
    decodeSubmitCount,
    dispatchCount,
    decodeSubmitCountPerToken: generatedTokenCount > 0
      ? decodeSubmitCount / generatedTokenCount
      : decodeSubmitCount,
    decodeDispatchCountPerToken: generatedTokenCount > 0
      ? dispatchCount / generatedTokenCount
      : dispatchCount,
    decodeDispatchCountPerLayerPerToken: totalDecodeLayerCount > 0
      ? dispatchCount / totalDecodeLayerCount
      : dispatchCount,
    readbackCount: previous.readbackCount + next.readbackCount,
    totalReadbackRows: previous.totalReadbackRows + next.totalReadbackRows,
    totalReadbackBytes: previous.totalReadbackBytes + next.totalReadbackBytes,
    fullLogitsReadbackCount: previous.fullLogitsReadbackCount + next.fullLogitsReadbackCount,
    compactLogitReadbackCount: previous.compactLogitReadbackCount + next.compactLogitReadbackCount,
    weightUploadBytesDuringDecode: previous.weightUploadBytesDuringDecode + next.weightUploadBytesDuringDecode,
    weightUploadCountDuringDecode: previous.weightUploadCountDuringDecode + next.weightUploadCountDuringDecode,
    activationUploadBytesDuringDecode: previous.activationUploadBytesDuringDecode + next.activationUploadBytesDuringDecode,
    activationUploadCountDuringDecode: previous.activationUploadCountDuringDecode + next.activationUploadCountDuringDecode,
    hiddenReadbackCountDuringDecode: previous.hiddenReadbackCountDuringDecode + next.hiddenReadbackCountDuringDecode,
    f32ExpansionCountDuringDecode: previous.f32ExpansionCountDuringDecode + next.f32ExpansionCountDuringDecode,
    f32ExpansionBytesDuringDecode: previous.f32ExpansionBytesDuringDecode + next.f32ExpansionBytesDuringDecode,
    cpuFallbackUsed: previous.cpuFallbackUsed || next.cpuFallbackUsed,
    cpuValidationUsed: previous.cpuValidationUsed || next.cpuValidationUsed,
    prefillExecutionsDuringDecode,
    prefillCountPerGeneratedToken: generatedTokenCount > 0
      ? prefillExecutionsDuringDecode / generatedTokenCount
      : prefillExecutionsDuringDecode,
    residentDecodeLayerCount,
    totalDecodeLayerCount,
    residentDecodeLayerCoverage: totalDecodeLayerCount > 0
      ? residentDecodeLayerCount / totalDecodeLayerCount
      : 0,
    residentFinalHiddenUsedForLogits: previous.residentFinalHiddenUsedForLogits && next.residentFinalHiddenUsedForLogits,
    kvDecodeReused: previous.kvDecodeReused && next.kvDecodeReused,
    fusedPackedQkvLayerCount,
    fusedQkvNormRopeKvAppendLayerCount,
    fusedOneTokenAttentionLayerCount,
    fusedResidualRmsNormLayerCount,
    fusedMlpLayerCount,
    fusedFullLayerCount,
    fusedLayerCoverage: totalDecodeLayerCount > 0
      ? fusedLayerStageHits / Math.max(1, totalDecodeLayerCount * 6)
      : 0,
    tokensPerSecond: next.tokensPerSecond ?? previous.tokensPerSecond,
  };
}

export class UnlockedBrowserTransformerClient implements ChatClient {
  readonly backendId = "unlocked-browser-transformer";
  readonly modelId: string;
  lastDecodeProof: UnlockedBrowserDecodeProof | null = null;
  lastPromptTokenIds: number[] = [];
  lastGeneratedTokenIds: number[] = [];
  lastGeneratedTokenTexts: string[] = [];
  lastGenerationStopReason: UnlockedBrowserGenerationStopReason | null = null;
  lastPrefillChunkPlan: UnlockedBrowserPrefillChunkPlan | null = null;
  lastWarmupMs: number | null = null;
  lastWarmupProof: UnlockedBrowserDecodeProof | null = null;
  lastWarmupMode: UnlockedBrowserWarmupMode | null = null;
  lastWarmupUploadedEntries: number | null = null;
  lastWarmupCacheHits: number | null = null;
  lastResidentReadbackCount: number | null = null;
  lastKvPersistDeferred = false;
  lastKvPersistCriticalPathMs = 0;
  lastKvPersistFlushMs: number | null = null;
  lastKvPersistPendingBlockCount = 0;

  private readonly manifestPath: string;
  private readonly manifestSha256: string;
  private readonly allowFixtureWeights: boolean;
  private readonly backendPreference: WebGpuSsaBackendOptions["backendPreference"];
  private readonly requireWebGpu: boolean;
  private readonly maxRuntimePromptTokens: number | null;
  private readonly maxRuntimeLayers: number | null;
  private readonly logitCandidateLimit: number | null;
  private readonly logitTopK: number | null;
  private readonly logitTileRows: number | null;
  private readonly maxGenerationTokens: number | null;
  private readonly qwenThinkingMode: QwenThinkingMode;
  private readonly strictChunkedPrefill: boolean;
  private readonly warmModelResidency: boolean;
  private readonly warmModelResidencyMode: UnlockedBrowserWarmupMode;
  private readonly mtp: NormalizedUnlockedBrowserMtpOptions;
  private readonly kvPersistenceOptions: NormalizedUnlockedBrowserKvPersistenceOptions;
  private mtpAutoDisabledReason: string | null = null;
  private backend: UnlockedBrowserTransformerBackend | null = null;
  private draftBackend: UnlockedBrowserTransformerBackend | null = null;
  private modelBufferCache: WebGpuRuntimeBufferCache | null = null;
  private draftModelBufferCache: WebGpuRuntimeBufferCache | null = null;
  private kvPersistence: KVSwapPersistenceStore | null = null;
  private kvPersistenceEvents: KVSwapPersistenceTraceEvent[] = [];
  private hydratedKvBlocks: SerializedKVSwapBlock[] = [];
  private kvDecodeReuseUsed = false;
  private pendingKvPersistFlush: Promise<void> | null = null;
  private pendingKvPersistStarter: (() => Promise<void> | null) | null = null;
  private lastKvPrefetchTrace: BrowserKvPrefetchTrace = emptyKvPrefetchTrace();
  private tokenizer: UnlockedBrowserTokenizer | null = null;
  private chatFormatter: ChatFormatter = formatQwenChatMessages;
  private vocabSize = 0;
  private layerCount = 0;

  constructor(options: UnlockedBrowserTransformerClientOptions) {
    this.modelId = options.modelId;
    this.manifestPath = options.manifestPath?.trim() ?? "";
    this.manifestSha256 = options.manifestSha256?.trim() ?? "";
    this.allowFixtureWeights = options.allowFixtureWeights ?? false;
    this.backendPreference = options.backendPreference;
    this.requireWebGpu = options.requireWebGpu === true;
    this.maxRuntimePromptTokens = normalizePositiveInteger(options.maxRuntimePromptTokens);
    this.maxRuntimeLayers = normalizePositiveInteger(options.maxRuntimeLayers);
    this.logitCandidateLimit = normalizePositiveInteger(options.logitCandidateLimit);
    this.logitTopK = normalizePositiveInteger(options.logitTopK);
    this.logitTileRows = normalizePositiveInteger(options.logitTileRows);
    this.maxGenerationTokens = normalizePositiveInteger(options.maxGenerationTokens);
    this.qwenThinkingMode = normalizeQwenThinkingMode(options.qwenThinkingMode);
    this.strictChunkedPrefill = options.strictChunkedPrefill === true;
    this.warmModelResidency = options.warmModelResidency === true;
    this.warmModelResidencyMode = normalizeWarmupMode(options.warmModelResidencyMode);
    this.mtp = normalizeMtpOptions(options.mtp);
    this.kvPersistenceOptions = normalizeKvPersistenceOptions(options.kvPersistence);
  }

  async init(): Promise<void> {
    const { weights, tokenizer } = await this.loadModelAssets();
    const modelBufferCache = getSharedModelBufferCache({
      modelId: weights.modelId,
      manifestPath: this.manifestPath,
      manifestSha256: this.manifestSha256,
    });
    this.modelBufferCache = modelBufferCache;
    const backend = new UnlockedBrowserTransformerBackend({
      weights,
      bufferCache: modelBufferCache,
      ...(this.backendPreference ? { backendPreference: this.backendPreference } : {}),
      ...(this.requireWebGpu ? { requireWebGpu: true } : {}),
    });
    await backend.initializeModel(weights.modelId);
    this.backend = backend;
    this.vocabSize = weights.vocabSize;
    this.layerCount = weights.layers.length;
    if (this.warmModelResidency) {
      await this.warmTargetModelResidency(backend);
    }
    if (this.mtp.enabled && this.mtp.draftModelId === LOCAL_BROWSER_QWEN_PREFIX_MTP_DRAFT_MODEL_ID) {
      const draftWeights = makeQwenPrefixDraftWeights(weights, this.mtp.draftLayerCount);
      const draftModelBufferCache = getSharedModelBufferCache({
        modelId: draftWeights.modelId,
        manifestPath: this.manifestPath,
        manifestSha256: this.manifestSha256,
      });
      this.draftModelBufferCache = draftModelBufferCache;
      this.draftBackend = new UnlockedBrowserTransformerBackend({
        weights: draftWeights,
        bufferCache: draftModelBufferCache,
        ...(this.backendPreference ? { backendPreference: this.backendPreference } : {}),
        ...(this.requireWebGpu ? { requireWebGpu: true } : {}),
      });
      await this.draftBackend.initializeModel(draftWeights.modelId);
    }
    if (this.kvPersistenceOptions.enabled) {
      this.kvPersistence = await createBrowserKVSwapPersistence(this.kvPersistenceOptions);
      if (this.kvPersistenceOptions.clearOnInit) {
        this.recordKvPersistenceEvent(await this.kvPersistence.clear(this.kvPersistenceOptions.namespace));
      } else {
        const hydrated = await this.kvPersistence.hydrate(this.kvPersistenceOptions.namespace);
        this.hydratedKvBlocks = hydrated.blocks;
        this.recordKvPersistenceEvent(hydrated.event);
      }
    }
    this.tokenizer = tokenizer;
    this.chatFormatter = (messages) => tokenizer.formatMessages(messages, { qwenThinkingMode: this.qwenThinkingMode });
  }

  async dispose(options: UnlockedBrowserTransformerClientDisposeOptions = {}): Promise<void> {
    await this.flushKvPersistence().catch(() => undefined);

    const backend = this.backend;
    const draftBackend = this.draftBackend;
    this.backend = null;
    this.draftBackend = null;
    this.tokenizer = null;
    this.kvPersistence = null;
    this.hydratedKvBlocks = [];
    this.kvPersistenceEvents = [];
    this.lastKvPrefetchTrace = emptyKvPrefetchTrace();
    this.lastDecodeProof = null;
    this.lastPromptTokenIds = [];
    this.lastGeneratedTokenIds = [];
    this.lastGeneratedTokenTexts = [];
    this.lastPrefillChunkPlan = null;
    this.lastWarmupMs = null;
    this.lastWarmupProof = null;
    this.lastWarmupMode = null;
    this.lastWarmupUploadedEntries = null;
    this.lastWarmupCacheHits = null;
    this.lastResidentReadbackCount = null;
    this.lastKvPersistDeferred = false;
    this.lastKvPersistCriticalPathMs = 0;
    this.lastKvPersistFlushMs = null;
    this.lastKvPersistPendingBlockCount = 0;
    this.pendingKvPersistFlush = null;
    this.pendingKvPersistStarter = null;
    this.vocabSize = 0;
    this.layerCount = 0;

    await Promise.all([
      backend?.dispose(),
      draftBackend?.dispose(),
    ].filter((task): task is Promise<void> => Boolean(task)));

    if (options.clearSharedBuffers) {
      const caches = new Set<WebGpuRuntimeBufferCache>();
      if (this.modelBufferCache) caches.add(this.modelBufferCache);
      if (this.draftModelBufferCache) caches.add(this.draftModelBufferCache);
      for (const cache of caches) cache.clear();
    }
    this.modelBufferCache = null;
    this.draftModelBufferCache = null;
  }

  async *streamChat(messages: ChatClientMessage[], options: ChatStreamOptions = {}): AsyncGenerator<string, string, void> {
    const backend = this.requireBackend();
    const tokenizer = this.requireTokenizer();
    this.kvDecodeReuseUsed = false;
    this.lastKvPrefetchTrace = emptyKvPrefetchTrace();
    this.lastDecodeProof = null;
    this.kvPersistenceEvents = this.kvPersistenceEvents.filter((event) => event.operation === "hydrate");
    this.lastKvPersistDeferred = false;
    this.lastKvPersistCriticalPathMs = 0;
    this.lastKvPersistFlushMs = null;
    this.lastKvPersistPendingBlockCount = 0;
    const requestId = `unlocked_${Date.now().toString(36)}`;
    const formattedPrompt = this.chatFormatter(messages);
    const tokenIds = trimRuntimeTokenIds(
      tokenizer.encode(formattedPrompt, this.vocabSize),
      this.maxRuntimePromptTokens,
    );
    this.lastPromptTokenIds = [...tokenIds];
    this.lastGeneratedTokenIds = [];
    this.lastGeneratedTokenTexts = [];
    this.lastGenerationStopReason = null;
    const prefillTokenIds = tokenIds.length > 1 ? tokenIds.slice(0, -1) : tokenIds;
    let previousTokenId = tokenIds[tokenIds.length - 1] ?? 0;
    const layerCount = this.runtimeLayerCount();
    const prefillLayerPolicies = buildClientPolicies(prefillTokenIds.length, layerCount);
    const prefillChunkPlan = buildClientPrefillChunkPlan(prefillTokenIds.length, prefillLayerPolicies);
    this.lastPrefillChunkPlan = prefillChunkPlan;
    const reusedKvCacheHandle = this.tryReusePersistedKvCache({
      requestId,
      tokenIds: prefillTokenIds,
      layerCount,
    });
    let kvCacheHandle: unknown;
    let predictivePrefetch: Promise<BrowserKvPrefetchTrace> | null = null;
    const startPredictivePrefetch = (): Promise<BrowserKvPrefetchTrace> | null => {
      if (reusedKvCacheHandle || predictivePrefetch) return predictivePrefetch;
      predictivePrefetch = this.prefetchPredictiveKvBlocks({
        requestId,
        tokenIds: prefillTokenIds,
        layerCount,
      }).then((trace) => {
        this.lastKvPrefetchTrace = trace;
        this.refreshKvPersistenceProof();
        return trace;
      }).catch((error) => {
        const trace = {
          ...emptyKvPrefetchTrace("miss_stall"),
          attentionStallMs: 0,
        };
        this.recordKvPersistenceEvent({
          operation: "load",
          mode: this.kvPersistence?.health().mode ?? "disabled",
          ok: false,
          namespace: this.kvPersistenceOptions.namespace,
          blockIds: [],
          bytes: 0,
          reason: error instanceof Error ? error.message : String(error),
          ...trace,
          at: new Date().toISOString(),
        });
        this.lastKvPrefetchTrace = trace;
        this.refreshKvPersistenceProof();
        return trace;
      });
      return predictivePrefetch;
    };
    if (reusedKvCacheHandle) {
      kvCacheHandle = reusedKvCacheHandle;
    } else {
      const prefillResult = await backend.prefill(Int32Array.from(prefillTokenIds), {
        requestId,
        layerPolicies: prefillLayerPolicies,
      });
      kvCacheHandle = prefillResult.kvCacheHandle;
    }
    let qwenDraftState = await this.createQwenPrefixDraftState({
      requestId,
      prefillTokenIds,
    });
    let deferredPrefillPersist: Promise<void> | null = null;
    const startDeferredPrefillPersist = (): Promise<void> | null => {
      if (reusedKvCacheHandle || deferredPrefillPersist) return deferredPrefillPersist;
      if (this.pendingKvPersistStarter === startDeferredPrefillPersist) {
        this.pendingKvPersistStarter = null;
      }
      const persistTask = this.persistKvCache(kvCacheHandle, {
        phase: "prefill",
        tokenIds: prefillTokenIds,
        runtimeLayerCount: layerCount,
        policyHash: policyHashFor(prefillTokenIds.length, layerCount),
      }, { deferred: true });
      deferredPrefillPersist = persistTask.finally(() => {
        if (this.pendingKvPersistFlush === deferredPrefillPersist) {
          this.pendingKvPersistFlush = null;
        }
        this.refreshKvPersistenceProof();
      });
      this.pendingKvPersistFlush = deferredPrefillPersist;
      this.refreshKvPersistenceProof();
      return deferredPrefillPersist;
    };
    const scheduleDeferredPrefillPersist = (): void => {
      if (reusedKvCacheHandle || deferredPrefillPersist || this.pendingKvPersistStarter === startDeferredPrefillPersist) return;
      this.lastKvPersistDeferred = true;
      this.lastKvPersistCriticalPathMs = 0;
      this.pendingKvPersistStarter = startDeferredPrefillPersist;
      this.refreshKvPersistenceProof();
    };

    const maxTokens = resolveGenerationTokenLimit(options.maxTokens, this.maxGenerationTokens);
    const chunks: string[] = [];
    const outputFilter = new AssistantOutputFilter();
    const stopAfter = new InclusiveStopSequenceFilter(options.stopAfterSequences);
    const tokenStreamDecoder = tokenizer.createStreamDecoder?.();
    const logitCandidateTokenIds = buildLogitCandidateTokenIds(tokenIds, tokenizer, this.vocabSize, this.logitCandidateLimit);
    const logitTopK = logitCandidateTokenIds ? null : (this.logitTopK ?? 64);
    const baseSuppressedTokenIds = buildQwenThinkingSuppressedTokenIds(tokenizer, this.vocabSize, this.qwenThinkingMode);
    if (options.includeProofMarker) {
      chunks.push(UNLOCKED_PROOF_MARKER);
      yield UNLOCKED_PROOF_MARKER;
    }

    let generatedTokens = 0;
    const generatedTokenIds: number[] = [];
    const generatedTokenTexts: string[] = [];
    let accumulatedDecodePerf: DecodePerfSummary | null = null;
    while (generatedTokens < maxTokens) {
      const remainingTokens = maxTokens - generatedTokens;
      const suppressedTokenIds = buildDecodeSuppressedTokenIds(
        baseSuppressedTokenIds,
        generatedTokenIds,
        generatedTokenTexts,
        this.vocabSize,
      );
      if (this.shouldUseMtp(remainingTokens)) {
        const step = await this.decodeSpeculativeStep({
          backend,
          tokenizer,
          requestId,
          previousTokenId,
          kvCacheHandle,
          tokenPosition: prefillTokenIds.length + generatedTokens + 1,
          layerCount,
          logitCandidateTokenIds,
          suppressedTokenIds,
          logitTopK,
          logitTileRows: this.logitTileRows,
          promptTokenIds: tokenIds,
          generatedTokenIds,
          remainingTokens,
          qwenDraftState,
          tokenStreamDecoder,
        });
        previousTokenId = step.previousTokenId;
        generatedTokens += step.tokenIds.length;
        generatedTokenIds.push(...step.tokenIds);
        generatedTokenTexts.push(...step.chunks);
        this.lastGeneratedTokenIds = [...generatedTokenIds];
        this.lastGeneratedTokenTexts = [...generatedTokenTexts];
        if (step.refreshQwenDraftState) {
          qwenDraftState = await this.createQwenPrefixDraftState({
            requestId,
            prefillTokenIds: [...tokenIds, ...generatedTokenIds].slice(0, -1),
          });
        }
        for (const chunk of step.chunks) {
          const filtered = outputFilter.push(chunk);
          if (filtered) {
            const stoppedFiltered = stopAfter.push(filtered);
            if (stoppedFiltered) {
              chunks.push(stoppedFiltered);
              scheduleDeferredPrefillPersist();
              yield stoppedFiltered;
            }
          }
          if (stopAfter.stopped) break;
        }
        if (stopAfter.stopped) {
          this.lastGenerationStopReason = "stop_after_sequence";
          break;
        }
        if (outputFilter.stopped) {
          this.lastGenerationStopReason = "stop_marker";
          break;
        }
        continue;
      }

      const decodeSampling = logitTopK && logitTopK > 1
        ? buildCompactDecodeSamplingOptions({
            options,
            requestId,
            step: generatedTokens,
            recentTokenIds: generatedTokenIds,
          })
        : null;
      const decode = await backend.decode({
        requestId,
        inputTokenId: previousTokenId,
        kvCacheHandle,
        policy: buildClientPolicies(prefillTokenIds.length + generatedTokens + 1, layerCount),
        ...(logitCandidateTokenIds ? { logitCandidateTokenIds } : {}),
        ...(suppressedTokenIds.length > 0 ? { suppressedTokenIds } : {}),
        ...(logitTopK ? { logitTopK } : {}),
        ...(this.logitTileRows ? { logitTileRows: this.logitTileRows } : {}),
        ...(decodeSampling ? decodeSampling : {}),
      });
      previousTokenId = decode.tokenId;
      const tokenDecodeProof = this.withKvPersistenceProof(toDecodeProof(decode, this.targetOnlyMtpProof(), kvCacheHandle, layerCount));
      accumulatedDecodePerf = accumulateDecodePerf(accumulatedDecodePerf, tokenDecodeProof.decodePerf);
      this.lastDecodeProof = this.withKvPersistenceProof({
        ...tokenDecodeProof,
        ...(accumulatedDecodePerf ? { decodePerf: accumulatedDecodePerf } : {}),
      });
      const chunk = decodeStreamToken(tokenizer, tokenStreamDecoder, decode.tokenId);
      const filtered = outputFilter.push(chunk);
      if (filtered) {
        const stoppedFiltered = stopAfter.push(filtered);
        if (stoppedFiltered) {
          chunks.push(stoppedFiltered);
          scheduleDeferredPrefillPersist();
          yield stoppedFiltered;
        }
      }
      generatedTokenIds.push(decode.tokenId);
      generatedTokenTexts.push(chunk);
      this.lastGeneratedTokenIds = [...generatedTokenIds];
      this.lastGeneratedTokenTexts = [...generatedTokenTexts];
      generatedTokens += 1;
      if (stopAfter.stopped) {
        this.lastGenerationStopReason = "stop_after_sequence";
        break;
      }
      if (outputFilter.stopped) {
        this.lastGenerationStopReason = "stop_marker";
        break;
      }
    }

    const decoderFlushed = tokenStreamDecoder?.flush() ?? "";
    if (decoderFlushed) {
      const filtered = outputFilter.push(decoderFlushed);
      if (filtered) {
        const stoppedFiltered = stopAfter.push(filtered);
        if (stoppedFiltered) chunks.push(stoppedFiltered);
        scheduleDeferredPrefillPersist();
        if (stoppedFiltered) yield stoppedFiltered;
      }
    }
    const flushed = outputFilter.flush();
    if (flushed) {
      const stoppedFiltered = stopAfter.push(flushed);
      if (stoppedFiltered) chunks.push(stoppedFiltered);
      scheduleDeferredPrefillPersist();
      if (stoppedFiltered) yield stoppedFiltered;
    }
    if (this.lastGenerationStopReason === null) {
      this.lastGenerationStopReason = stopAfter.stopped
        ? "stop_after_sequence"
        : (outputFilter.stopped ? "stop_marker" : "max_tokens");
    }

    if (options.awaitKvPredictivePrefetchProof && predictivePrefetch) {
      await predictivePrefetch;
      this.refreshKvPersistenceProof();
    } else if (options.awaitKvPredictivePrefetchProof) {
      const proofPrefetch = startPredictivePrefetch();
      if (proofPrefetch) {
        await proofPrefetch;
        this.refreshKvPersistenceProof();
      }
    }

    if (!reusedKvCacheHandle && !deferredPrefillPersist) {
      scheduleDeferredPrefillPersist();
    }
    void deferredPrefillPersist;

    return chunks.join("");
  }

  private async loadModelAssets(): Promise<LoadedUnlockedBrowserManifest> {
    if (this.manifestPath) {
      const response = await fetch(this.manifestPath);
      if (!response.ok) {
        throw new Error(`Unlocked browser transformer manifest failed to load: ${response.status}`);
      }
      const rawManifest = await response.text();
      if (this.manifestSha256) await verifyTextSha256(rawManifest, this.manifestSha256, "VITE_UNLOCKED_MODEL_MANIFEST_SHA256");
      return normalizeManifest(JSON.parse(rawManifest) as unknown, this.modelId, this.manifestPath);
    }

    if (this.allowFixtureWeights) {
      return {
        weights: makeFixtureWeights(this.modelId),
        tokenizer: new FixtureTokenizer(),
      };
    }

    throw new Error("Unlocked browser transformer requires VITE_UNLOCKED_MODEL_MANIFEST_PATH or VITE_UNLOCKED_ALLOW_FIXTURE=true.");
  }

  private requireBackend(): UnlockedBrowserTransformerBackend {
    if (!this.backend) throw new Error("UnlockedBrowserTransformerClient.init() must complete before streamChat().");
    return this.backend;
  }

  private requireTokenizer(): UnlockedBrowserTokenizer {
    if (!this.tokenizer) throw new Error("UnlockedBrowserTransformerClient.init() must load a tokenizer before streamChat().");
    return this.tokenizer;
  }

  private requireLayerCount(): number {
    if (this.layerCount <= 0) throw new Error("UnlockedBrowserTransformerClient.init() must load at least one transformer layer before streamChat().");
    return this.layerCount;
  }

  private async warmTargetModelResidency(backend: UnlockedBrowserTransformerBackend): Promise<void> {
    const startedAt = performance.now();
    const layerCount = this.runtimeLayerCount();
    const mode = this.warmModelResidencyMode;
    if (this.backendPreference !== "cpu") {
      const warmup = await backend.warmModelResidency({
        layerCount,
        logitTopK: this.logitTopK ?? 1,
        ...(this.logitTileRows ? { logitTileRows: this.logitTileRows } : {}),
      });
      if (mode === "pipeline_preload") {
        this.lastWarmupMs = Math.max(0, performance.now() - startedAt);
        this.lastWarmupMode = "pipeline_preload";
        this.lastWarmupUploadedEntries = warmup.uploadedEntries;
        this.lastWarmupCacheHits = warmup.cacheHits;
        this.lastResidentReadbackCount = 0;
        this.lastWarmupProof = makePipelinePreloadWarmupProof({
          layerCount,
          warmupMs: this.lastWarmupMs,
          uploadedEntries: warmup.uploadedEntries,
          cacheHits: warmup.cacheHits,
          mtp: this.targetOnlyMtpProof(),
        });
        return;
      }
      const targetPass = await this.runTargetWarmupPass(backend, layerCount);
      this.lastWarmupMs = Math.max(0, performance.now() - startedAt);
      this.lastWarmupMode = "target_probe";
      this.lastWarmupUploadedEntries = warmup.uploadedEntries;
      this.lastWarmupCacheHits = warmup.cacheHits;
      this.lastResidentReadbackCount = 1;
      this.lastWarmupProof = {
        ...toDecodeProof(targetPass.decode, this.targetOnlyMtpProof(), targetPass.kvCacheHandle, layerCount),
        warmupMode: "target_probe",
        warmupBlockingMs: this.lastWarmupMs,
        warmupUploadedEntries: warmup.uploadedEntries,
        warmupCacheHits: warmup.cacheHits,
        residentUploadCount: warmup.uploadedEntries,
        residentCacheHitCount: warmup.cacheHits,
        residentReadbackCount: 1,
      };
      return;
    }
    if (mode === "pipeline_preload") {
      this.lastWarmupMs = Math.max(0, performance.now() - startedAt);
      this.lastWarmupMode = "pipeline_preload";
      this.lastWarmupUploadedEntries = 0;
      this.lastWarmupCacheHits = 0;
      this.lastResidentReadbackCount = 0;
      this.lastWarmupProof = makePipelinePreloadWarmupProof({
        layerCount,
        warmupMs: this.lastWarmupMs,
        uploadedEntries: 0,
        cacheHits: 0,
        mtp: this.targetOnlyMtpProof(),
      });
      return;
    }
    const targetPass = await this.runTargetWarmupPass(backend, layerCount);
    this.lastWarmupMs = Math.max(0, performance.now() - startedAt);
    this.lastWarmupMode = "target_probe";
    this.lastWarmupUploadedEntries = null;
    this.lastWarmupCacheHits = null;
    this.lastResidentReadbackCount = 1;
    this.lastWarmupProof = {
      ...toDecodeProof(targetPass.decode, this.targetOnlyMtpProof(), targetPass.kvCacheHandle, layerCount),
      warmupMode: "target_probe",
      warmupBlockingMs: this.lastWarmupMs,
      residentReadbackCount: 1,
    };
  }

  private async runTargetWarmupPass(
    backend: UnlockedBrowserTransformerBackend,
    layerCount: number,
  ): Promise<{
    decode: Awaited<ReturnType<UnlockedBrowserTransformerBackend["decode"]>>;
    kvCacheHandle: unknown;
  }> {
    const requestId = `unlocked_warm_${Date.now().toString(36)}`;
    const prefill = await backend.prefill(Int32Array.from([0]), {
      requestId,
      layerPolicies: buildClientPolicies(1, layerCount),
    });
    const decode = await backend.decode({
      requestId,
      inputTokenId: 1,
      kvCacheHandle: prefill.kvCacheHandle,
      policy: buildClientPolicies(2, layerCount),
      logitTopK: this.logitTopK ?? 1,
      ...(this.logitTileRows ? { logitTileRows: this.logitTileRows } : {}),
    });
    return {
      decode,
      kvCacheHandle: prefill.kvCacheHandle,
    };
  }

  private runtimeLayerCount(): number {
    const layerCount = this.requireLayerCount();
    return this.maxRuntimeLayers ? Math.min(layerCount, this.maxRuntimeLayers) : layerCount;
  }

  private shouldUseMtp(remainingTokens: number): boolean {
    return this.mtp.enabled
      && !this.mtpAutoDisabledReason
      && Boolean(this.mtp.draftModelId)
      && this.mtp.numSpeculativeTokens > 0
      && remainingTokens > 0;
  }

  private targetOnlyMtpProof(): UnlockedBrowserMtpProof {
    return {
      mode: "target_only",
      draftModelId: this.mtp.draftModelId,
      draftTokenIds: [],
      draftTokens: [],
      acceptedTokens: 0,
      rejectedTokens: 0,
      correctedTokens: 0,
      acceptanceRate: 0,
      numSpeculativeTokens: 0,
      ...(this.mtp.latencyDisableUnsupported ? { latencyDisablePolicy: "unsupported_without_target_baseline" } : {}),
      ...(this.mtp.draftModelId === LOCAL_BROWSER_QWEN_PREFIX_MTP_DRAFT_MODEL_ID && this.mtp.disableWhenLatencyWorse
        ? { latencyDisablePolicy: "paired_benchmark_required" as const }
        : {}),
      disabledReason: this.mtp.disabledReason ?? this.mtpAutoDisabledReason ?? (this.mtp.enabled ? "mtp_unavailable" : "mtp_disabled"),
    };
  }

  private async createQwenPrefixDraftState(input: {
    requestId: string;
    prefillTokenIds: number[];
  }): Promise<QwenPrefixDraftState | null> {
    if (!this.mtp.enabled) return null;
    if (this.mtp.draftModelId !== LOCAL_BROWSER_QWEN_PREFIX_MTP_DRAFT_MODEL_ID || !this.draftBackend) return null;
    if (input.prefillTokenIds.length === 0) return null;
    const layerCount = this.draftRuntimeLayerCount();
    const requestId = `${input.requestId}:draft`;
    const prefill = await this.draftBackend.prefill(Int32Array.from(input.prefillTokenIds), {
      requestId,
      layerPolicies: buildClientPolicies(input.prefillTokenIds.length, layerCount),
    });
    return {
      backend: this.draftBackend,
      requestId,
      kvCacheHandle: prefill.kvCacheHandle,
      layerCount,
    };
  }

  private draftRuntimeLayerCount(): number {
    return Math.max(1, Math.min(this.runtimeLayerCount(), this.mtp.draftLayerCount));
  }

  private async decodeSpeculativeStep(input: {
    backend: UnlockedBrowserTransformerBackend;
    tokenizer: UnlockedBrowserTokenizer;
    requestId: string;
    previousTokenId: number;
    kvCacheHandle: unknown;
    tokenPosition: number;
    layerCount: number;
    logitCandidateTokenIds: number[] | null;
    suppressedTokenIds: number[];
    logitTopK: number | null;
    logitTileRows: number | null;
    promptTokenIds: number[];
    generatedTokenIds: number[];
    remainingTokens: number;
    qwenDraftState: QwenPrefixDraftState | null;
    tokenStreamDecoder: UnlockedBrowserTokenStreamDecoder | undefined;
  }): Promise<{ previousTokenId: number; tokenIds: number[]; chunks: string[]; refreshQwenDraftState: boolean }> {
    const draft = await this.draftSpeculativeTokenIds({
      ...input,
      maxDraftTokens: Math.min(this.mtp.numSpeculativeTokens, input.remainingTokens),
    });
    const draftTokenIds = draft.tokenIds;
    const draftTokens: DraftToken[] = draftTokenIds.map((tokenId) => ({ token: input.tokenizer.decode(tokenId) }));
    const verificationBatch = await input.backend.verifySpeculativeDraft({
      requestId: input.requestId,
      previousTokenId: input.previousTokenId,
      draftTokenIds,
      kvCacheHandle: input.kvCacheHandle,
      policy: buildClientPolicies(input.tokenPosition + draftTokenIds.length - 1, input.layerCount),
      ...(input.logitCandidateTokenIds ? { logitCandidateTokenIds: input.logitCandidateTokenIds } : {}),
      ...(input.suppressedTokenIds.length > 0 ? { suppressedTokenIds: input.suppressedTokenIds } : {}),
      ...(input.logitTopK ? { logitTopK: input.logitTopK } : {}),
      ...(input.logitTileRows ? { logitTileRows: input.logitTileRows } : {}),
    });
    const targetTokenIds = verificationBatch.targetTokenIds;
    const verification: TargetVerificationToken[] = draftTokenIds.map((draftTokenId, index) => {
      const targetTokenId = targetTokenIds[index];
      const accepted = targetTokenId === draftTokenId;
      return {
        token: input.tokenizer.decode(draftTokenId),
        accepted,
        ...(!accepted && targetTokenId !== undefined ? { replacement: input.tokenizer.decode(targetTokenId) } : {}),
      };
    });
    const result = await verifySpeculativeBatch({
      requestId: input.requestId,
      modelPair: {
        draftModelId: this.mtp.draftModelId,
        targetModelId: this.modelId,
      },
      taskType: "browser_chat_decode",
      branches: [{ branchId: "browser-local-draft", draft: draftTokens }],
      draftLatencyMs: draft.latencyMs,
      targetOnlyLatencyMs: Math.max(
        verificationBatch.verifyLatencyMs * Math.max(1, verificationBatch.verifiedTokenCount),
        verificationBatch.verifiedTokenCount,
      ),
      minAcceptanceRate: this.mtp.minAcceptanceRate,
      disableWhenLatencyWorse: this.mtp.disableWhenLatencyWorse,
    }, () => ({
      requestId: input.requestId,
      verifyLatencyMs: verificationBatch.verifyLatencyMs,
      branches: [{ branchId: "browser-local-draft", verification }],
    }));
    const branch = result.branches[0];
    const streamedTokenIds = toStreamedTokenIds(draftTokenIds, targetTokenIds, branch?.acceptedTokens ?? 0);
    const streamedChunks = streamedTokenIds.map((tokenId) => decodeStreamToken(input.tokenizer, input.tokenStreamDecoder, tokenId));
    const correctedTokens = branch?.correctedToken ? 1 : 0;
    const mtpProof: UnlockedBrowserMtpProof = {
      mode: "draft_verify",
      draftModelId: this.mtp.draftModelId,
      draftSource: draft.source,
      verifierStrategy: "batched_continuation",
      draftTokenIds,
      draftTokens: draftTokenIds.map((tokenId) => input.tokenizer.decode(tokenId)),
      acceptedTokens: branch?.acceptedTokens ?? 0,
      rejectedTokens: branch?.rejectedTokens ?? 0,
      correctedTokens,
      acceptanceRate: result.metrics.acceptanceRate,
      verifierBackend: "unlocked-browser-transformer",
      numSpeculativeTokens: draftTokenIds.length,
      verifiedTokenCount: verificationBatch.verifiedTokenCount,
      targetDecodeCalls: verificationBatch.targetDecodeCalls,
      committedInputTokens: verificationBatch.committedInputTokenIds.length,
      ...(this.mtp.latencyDisableUnsupported ? { latencyDisablePolicy: "unsupported_without_target_baseline" } : {}),
      ...(draft.source === "qwen_prefix_draft" && this.mtp.disableWhenLatencyWorse ? { latencyDisablePolicy: "paired_benchmark_required" as const } : {}),
      ...(result.metrics.disabledReason ? { disabledReason: result.metrics.disabledReason } : {}),
      metrics: result.metrics,
    };
    if (result.metrics.disabledReason) this.mtpAutoDisabledReason = result.metrics.disabledReason;
    this.lastDecodeProof = this.withKvPersistenceProof(toDecodeProof(verificationBatch.decodeOutput, mtpProof, input.kvCacheHandle, input.layerCount));
    return {
      previousTokenId: streamedTokenIds.at(-1) ?? input.previousTokenId,
      tokenIds: streamedTokenIds,
      chunks: streamedChunks,
      refreshQwenDraftState: draft.source === "qwen_prefix_draft" && (branch?.acceptedTokens ?? 0) < draftTokenIds.length,
    };
  }

  private async draftSpeculativeTokenIds(input: {
    tokenizer: UnlockedBrowserTokenizer;
    previousTokenId: number;
    logitCandidateTokenIds: number[] | null;
    suppressedTokenIds: number[];
    logitTopK: number | null;
    logitTileRows: number | null;
    promptTokenIds: number[];
    generatedTokenIds: number[];
    vocabSize?: number;
    maxDraftTokens: number;
    qwenDraftState: QwenPrefixDraftState | null;
  }): Promise<{ tokenIds: number[]; source: NonNullable<UnlockedBrowserMtpProof["draftSource"]>; latencyMs: number }> {
    if (this.mtp.draftModelId === LOCAL_BROWSER_QWEN_PREFIX_MTP_DRAFT_MODEL_ID && input.qwenDraftState) {
      return draftQwenPrefixTokenIds({
        state: input.qwenDraftState,
        previousTokenId: input.previousTokenId,
        maxDraftTokens: input.maxDraftTokens,
        logitCandidateTokenIds: input.logitCandidateTokenIds,
        suppressedTokenIds: input.suppressedTokenIds,
        logitTopK: input.logitTopK,
        logitTileRows: input.logitTileRows,
      });
    }
    return {
      tokenIds: draftLocalTokenizerTokenIds({
        promptTokenIds: input.promptTokenIds,
        generatedTokenIds: input.generatedTokenIds,
        vocabSize: this.vocabSize,
        maxDraftTokens: input.maxDraftTokens,
      }),
      source: "local_tokenizer_ngram",
      latencyMs: 0,
    };
  }

  async clearKvPersistence(): Promise<KVSwapPersistenceTraceEvent | null> {
    if (!this.kvPersistence) return null;
    await this.flushKvPersistence();
    const event = await this.kvPersistence.clear(this.kvPersistenceOptions.namespace);
    this.hydratedKvBlocks = [];
    this.kvDecodeReuseUsed = false;
    this.recordKvPersistenceEvent(event);
    return event;
  }

  async flushKvPersistence(): Promise<void> {
    const starter = this.pendingKvPersistStarter;
    if (starter) {
      this.pendingKvPersistStarter = null;
      starter();
    }
    const pending = this.pendingKvPersistFlush;
    if (!pending) {
      this.refreshKvPersistenceProof();
      return;
    }
    try {
      await pending;
    } finally {
      if (this.pendingKvPersistFlush === pending) {
        this.pendingKvPersistFlush = null;
      }
      this.refreshKvPersistenceProof();
    }
  }

  getKvPersistenceHealth(): KVSwapPersistenceHealth {
    if (!this.kvPersistence) {
      return {
        enabled: false,
        mode: "disabled",
        namespace: this.kvPersistenceOptions.namespace,
        decodeReuse: false,
        kvPersistDeferred: this.lastKvPersistDeferred,
        kvPersistCriticalPathMs: this.lastKvPersistCriticalPathMs,
        ...(this.lastKvPersistFlushMs !== null ? { kvPersistFlushMs: this.lastKvPersistFlushMs } : {}),
        kvPersistPendingBlockCount: this.lastKvPersistPendingBlockCount,
      };
    }
    const health = this.kvPersistence.health();
    return {
      ...health,
      decodeReuse: health.decodeReuse || this.kvDecodeReuseUsed,
      kvPersistDeferred: this.lastKvPersistDeferred,
      kvPersistCriticalPathMs: this.lastKvPersistCriticalPathMs,
      ...(this.lastKvPersistFlushMs !== null ? { kvPersistFlushMs: this.lastKvPersistFlushMs } : {}),
      kvPersistPendingBlockCount: this.lastKvPersistPendingBlockCount,
    };
  }

  private tryReusePersistedKvCache(input: {
    requestId: string;
    tokenIds: number[];
    layerCount: number;
  }): UnlockedBrowserKvCacheHandle | null {
    if (!this.kvPersistence || this.hydratedKvBlocks.length === 0 || input.tokenIds.length === 0) return null;
    const reused = buildReusedKvCacheHandle({
      blocks: this.hydratedKvBlocks,
      namespace: this.kvPersistenceOptions.namespace,
      modelId: this.modelId,
      modelFingerprint: this.modelFingerprint(),
      requestId: input.requestId,
      tokenIds: input.tokenIds,
      layerCount: input.layerCount,
    });
    if (!reused) return null;
    this.kvDecodeReuseUsed = true;
    const health = this.kvPersistence.health();
    this.lastKvPrefetchTrace = {
      prefetchStrategy: "exact_reuse",
      predictedHotBlocks: [],
      prefetchedBlocks: reused.kvBlocks.map((block) => block.id),
      prefetchHitRate: 1,
      prefetchBytes: reused.kvBlocks.reduce((sum, block) => sum + Math.max(0, block.estimatedBytes), 0),
      prefetchLatencyMs: 0,
      attentionStallMs: 0,
    };
    this.recordKvPersistenceEvent({
      operation: "reuse",
      mode: health.mode,
      ok: true,
      namespace: this.kvPersistenceOptions.namespace,
      blockIds: reused.kvBlocks.map((block) => block.id),
      bytes: reused.kvBlocks.reduce((sum, block) => sum + Math.max(0, block.estimatedBytes), 0),
      ...(health.quotaBytes !== undefined ? { quotaBytes: health.quotaBytes } : {}),
      ...(health.usageBytes !== undefined ? { usageBytes: health.usageBytes } : {}),
      reason: "decode_reuse_prefill_skipped",
      ...this.lastKvPrefetchTrace,
      at: new Date().toISOString(),
    });
    return reused;
  }

  private async prefetchPredictiveKvBlocks(input: {
    requestId: string;
    tokenIds: number[];
    layerCount: number;
  }): Promise<BrowserKvPrefetchTrace> {
    if (!this.kvPersistence || this.hydratedKvBlocks.length === 0 || input.tokenIds.length === 0) {
      return emptyKvPrefetchTrace();
    }
    const candidateBlocks = this.hydratedKvBlocks
      .map(toPredictiveKvBlock)
      .filter((block): block is KVBlock => Boolean(block));
    if (candidateBlocks.length === 0) return emptyKvPrefetchTrace("miss_stall");

    const querySummary = buildClientLowRankQuerySummary(input.tokenIds, input.layerCount, this.hydratedKvBlocks);
    const decision = planKVSwap(candidateBlocks, {
      mode: "predictive",
      now: Date.now(),
      vramPressureThreshold: 0.82,
      ramPressureThreshold: 0.85,
    }, 0, [], [], {
      querySummary,
      maxBlocks: Math.min(4, candidateBlocks.length),
      minScore: 0,
    });
    const decisionPredictedHotBlocks = decision.predictedHotBlocks ?? [];
    const decisionPrefetchedBlocks = decision.prefetchedBlocks ?? [];
    const predictedHotBlocks = decisionPredictedHotBlocks.map((block) => {
      const persisted = this.hydratedKvBlocks.find((candidate) => candidate.id === block.blockId);
      return {
        ...block,
        ...(persisted?.runtimeBlockId ? { runtimeBlockId: persisted.runtimeBlockId } : {}),
      };
    });
    if (decisionPrefetchedBlocks.length === 0) {
      return {
        ...emptyKvPrefetchTrace(decision.prefetchStrategy ?? "miss_stall"),
        ...(decision.lowRankSummaryRank !== undefined ? { lowRankSummaryRank: decision.lowRankSummaryRank } : {}),
        lowRankQuerySource: querySummary.source,
        predictedHotBlocks,
        prefetchHitRate: decision.prefetchHitRate ?? 0,
        attentionStallMs: decision.attentionStallMs ?? 0,
      };
    }

    const startedAt = performance.now();
    const loaded: SerializedKVSwapBlock[] = [];
    const loadResults = await Promise.all(decisionPrefetchedBlocks.map((blockId) => (
      this.kvPersistence?.load(this.kvPersistenceOptions.namespace, blockId)
    )));
    const latencyMs = Math.max(0, performance.now() - startedAt);
    for (const result of loadResults) {
      if (!result) continue;
      if (result.block) loaded.push(result.block);
    }
    const prefetchedBlocks = loaded.map((block) => block.id);
    const prefetchBytes = loaded.reduce((sum, block) => sum + Math.max(0, block.byteLength || block.estimatedBytes), 0);
    const trace: BrowserKvPrefetchTrace = {
      prefetchStrategy: prefetchedBlocks.length > 0 ? "predictive_prefetch" : "miss_stall",
      ...(decision.lowRankSummaryRank !== undefined ? { lowRankSummaryRank: decision.lowRankSummaryRank } : {}),
      lowRankQuerySource: querySummary.source,
      predictedHotBlocks,
      prefetchedBlocks,
      prefetchHitRate: decision.prefetchHitRate ?? 0,
      prefetchBytes,
      prefetchLatencyMs: latencyMs,
      attentionStallMs: prefetchedBlocks.length > 0 ? 0 : Math.max(decision.attentionStallMs ?? 0, latencyMs),
    };
    for (const result of loadResults) {
      if (!result) continue;
      this.recordKvPersistenceEvent({
        ...result.event,
        reason: "predictive_prefetch",
        blockIds: result.block ? [result.block.id] : result.event.blockIds,
        bytes: prefetchBytes,
        ...trace,
      });
    }
    if (loaded.length > 0) {
      this.hydratedKvBlocks = mergeSerializedKvBlocks(this.hydratedKvBlocks, loaded);
    }
    return trace;
  }

  private async persistKvCache(
    kvCacheHandle: unknown,
    metadata: KVCachePersistenceMetadata = { phase: "decode" },
    options: { deferred?: boolean } = {},
  ): Promise<void> {
    const persistence = this.kvPersistence;
    if (!persistence) return;
    const startedAt = performance.now();
    const deferred = options.deferred === true;
    if (deferred) {
      this.lastKvPersistDeferred = true;
      this.lastKvPersistCriticalPathMs = 0;
    }
    try {
      const cache = readUnlockedBrowserKvCacheHandle(kvCacheHandle);
      const tokenIds = metadata.tokenIds ?? cache.tokenIds;
      const blocks = serializeKvCacheBlocks(cache, this.kvPersistenceOptions.namespace, {
        ...metadata,
        tokenIds,
        modelFingerprint: this.modelFingerprint(),
      });
      if (blocks.length === 0) return;
      this.lastKvPersistPendingBlockCount = blocks.length;
      const event = await persistence.persist(blocks);
      const flushMs = Math.max(0, performance.now() - startedAt);
      this.lastKvPersistFlushMs = flushMs;
      this.recordKvPersistenceEvent({
        ...event,
        kvPersistDeferred: deferred,
        kvPersistCriticalPathMs: deferred ? 0 : flushMs,
        kvPersistFlushMs: flushMs,
        kvPersistPendingBlockCount: blocks.length,
      });
      this.hydratedKvBlocks = mergeSerializedKvBlocks(this.hydratedKvBlocks, blocks);
    } catch (error) {
      const flushMs = Math.max(0, performance.now() - startedAt);
      this.lastKvPersistFlushMs = flushMs;
      this.recordKvPersistenceEvent({
        operation: "persist",
        mode: persistence.health().mode,
        ok: false,
        namespace: this.kvPersistenceOptions.namespace,
        blockIds: [],
        bytes: 0,
        reason: error instanceof Error ? error.message : String(error),
        kvPersistDeferred: deferred,
        kvPersistCriticalPathMs: deferred ? 0 : flushMs,
        kvPersistFlushMs: flushMs,
        kvPersistPendingBlockCount: this.lastKvPersistPendingBlockCount,
        at: new Date().toISOString(),
      });
    }
  }

  private recordKvPersistenceEvent(event: KVSwapPersistenceTraceEvent): void {
    this.kvPersistenceEvents = [event, ...this.kvPersistenceEvents].slice(0, 12);
  }

  private withKvPersistenceProof(proof: UnlockedBrowserDecodeProof): UnlockedBrowserDecodeProof {
    const health = this.getKvPersistenceHealth();
    const prefetchTrace = mergePrefetchTraceWithAttention(this.lastKvPrefetchTrace, proof.selectedBlockIds);
    return {
      ...proof,
      kvPersistence: {
        ...health,
        ...prefetchTrace,
        kvPersistDeferred: this.lastKvPersistDeferred,
        kvPersistCriticalPathMs: this.lastKvPersistCriticalPathMs,
        ...(this.lastKvPersistFlushMs !== null ? { kvPersistFlushMs: this.lastKvPersistFlushMs } : {}),
        kvPersistPendingBlockCount: this.lastKvPersistPendingBlockCount,
        events: [...this.kvPersistenceEvents],
      },
    };
  }

  private refreshKvPersistenceProof(): void {
    if (!this.lastDecodeProof) return;
    this.lastDecodeProof = this.withKvPersistenceProof(this.lastDecodeProof);
  }

  private modelFingerprint(): string {
    return this.manifestSha256 || this.manifestPath || this.modelId;
  }
}

interface LoadedUnlockedBrowserManifest {
  weights: UnlockedBrowserTransformerWeights;
  tokenizer: UnlockedBrowserTokenizer;
}

interface NormalizedUnlockedBrowserMtpOptions {
  enabled: boolean;
  draftModelId: string | null;
  numSpeculativeTokens: number;
  minAcceptanceRate: number;
  disableWhenLatencyWorse: boolean;
  draftLayerCount: number;
  latencyDisableUnsupported: boolean;
  disabledReason?: string;
}

interface QwenPrefixDraftState {
  backend: UnlockedBrowserTransformerBackend;
  requestId: string;
  kvCacheHandle: unknown;
  layerCount: number;
}

interface NormalizedUnlockedBrowserKvPersistenceOptions {
  enabled: boolean;
  namespace: string;
  preferOpfs: boolean;
  maxBlocks: number;
  maxBytes: number;
  clearOnInit: boolean;
}

interface KVCachePersistenceMetadata {
  phase: "prefill" | "decode";
  tokenIds?: number[];
  runtimeLayerCount?: number;
  policyHash?: string;
  modelFingerprint?: string;
}

function makePipelinePreloadWarmupProof(input: {
  layerCount: number;
  warmupMs: number;
  uploadedEntries: number;
  cacheHits: number;
  mtp: UnlockedBrowserMtpProof;
}): UnlockedBrowserDecodeProof {
  return {
    tensorControl: true,
    warmupMode: "pipeline_preload",
    warmupBlockingMs: input.warmupMs,
    warmupUploadedEntries: input.uploadedEntries,
    warmupCacheHits: input.cacheHits,
    residentUploadCount: input.uploadedEntries,
    residentCacheHitCount: input.cacheHits,
    residentReadbackCount: 0,
    tspSteps: [],
    selectedBlockIds: [],
    kvPagingEvents: 0,
    tokenId: 0,
    expectedLayerCount: input.layerCount,
    executedLayerCount: 0,
    mtp: input.mtp,
  };
}

function normalizeWarmupMode(mode: UnlockedBrowserWarmupMode | undefined): UnlockedBrowserWarmupMode {
  return mode === "target_probe" ? "target_probe" : "pipeline_preload";
}

function normalizeMtpOptions(options: UnlockedBrowserMtpOptions | undefined): NormalizedUnlockedBrowserMtpOptions {
  const draftModelId = options?.draftModelId?.trim() || null;
  const unsupportedDraft = Boolean(options?.enabled && draftModelId && !isSupportedBrowserMtpDrafter(draftModelId));
  const latencyDisableUnsupported = options?.enabled === true
    && options?.disableWhenLatencyWorse !== false
    && draftModelId !== LOCAL_BROWSER_QWEN_PREFIX_MTP_DRAFT_MODEL_ID;
  const numSpeculativeTokens = Math.min(
    normalizePositiveInteger(options?.numSpeculativeTokens) ?? 0,
    LOCAL_BROWSER_MTP_MAX_SPECULATIVE_TOKENS,
  );
  const qwenPrefixWindowCannotAccelerate = options?.enabled === true
    && draftModelId === LOCAL_BROWSER_QWEN_PREFIX_MTP_DRAFT_MODEL_ID
    && options?.disableWhenLatencyWorse !== false
    && numSpeculativeTokens < 2;
  const disabledReason = unsupportedDraft
    ? "unsupported_draft_model_id"
    : qwenPrefixWindowCannotAccelerate
      ? "speculation_slower_than_target_only"
      : undefined;
  return {
    enabled: options?.enabled === true && !unsupportedDraft && !qwenPrefixWindowCannotAccelerate,
    draftModelId,
    numSpeculativeTokens,
    minAcceptanceRate: clampRatio(options?.minAcceptanceRate ?? 0),
    disableWhenLatencyWorse: options?.disableWhenLatencyWorse ?? true,
    draftLayerCount: normalizePositiveInteger(options?.draftLayerCount) ?? 4,
    latencyDisableUnsupported,
    ...(disabledReason ? { disabledReason } : {}),
  };
}

function isSupportedBrowserMtpDrafter(draftModelId: string): boolean {
  return draftModelId === LOCAL_BROWSER_NGRAM_MTP_DRAFT_MODEL_ID
    || draftModelId === LOCAL_BROWSER_QWEN_PREFIX_MTP_DRAFT_MODEL_ID;
}

function normalizeKvPersistenceOptions(options: UnlockedBrowserKvPersistenceOptions | undefined): NormalizedUnlockedBrowserKvPersistenceOptions {
  return {
    enabled: options?.enabled === true,
    namespace: normalizeKVSwapNamespace(options?.namespace?.trim() || "default"),
    preferOpfs: options?.preferOpfs !== false,
    maxBlocks: normalizePositiveInteger(options?.maxBlocks) ?? 512,
    maxBytes: normalizePositiveInteger(options?.maxBytes) ?? 256 * 1024 * 1024,
    clearOnInit: options?.clearOnInit === true,
  };
}

function emptyKvPrefetchTrace(strategy: KVSwapPrefetchStrategy = "none"): BrowserKvPrefetchTrace {
  return {
    prefetchStrategy: strategy,
    predictedHotBlocks: [],
    prefetchedBlocks: [],
    prefetchHitRate: 0,
    prefetchBytes: 0,
    prefetchLatencyMs: 0,
    attentionStallMs: 0,
  };
}

function buildClientLowRankQuerySummary(
  tokenIds: number[],
  layerCount: number,
  persistedBlocks: readonly SerializedKVSwapBlock[] = [],
): BrowserKvLowRankQuerySummary {
  const projectedQueryRows = collectPersistedQueryRowsForPrompt(tokenIds, persistedBlocks);
  if (projectedQueryRows.length > 0) {
    const summary = compressKeyRowsToLowRankSummary({
      blockId: "current-query",
      projectionId: CLIENT_LOW_RANK_PROJECTION_ID,
      layer: 0,
      headGroupId: "all_heads",
      rows: projectedQueryRows,
      rank: CLIENT_LOW_RANK_PREFETCH_RANK,
      qualityScore: 1,
    });
    return {
      rank: summary.rank,
      projectionId: summary.projectionId,
      layer: summary.layer,
      headGroupId: summary.headGroupId,
      values: Array.from(summary.values),
      source: "persisted_q_rows",
    };
  }
  return {
    rank: CLIENT_LOW_RANK_PREFETCH_RANK,
    projectionId: CLIENT_LOW_RANK_PROJECTION_ID,
    layer: Math.max(0, Math.min(layerCount - 1, 0)),
    headGroupId: "all_heads",
    values: buildTokenLowRankValues(tokenIds, CLIENT_LOW_RANK_PREFETCH_RANK),
    source: "token_id_fallback",
  };
}

function collectPersistedQueryRowsForPrompt(
  tokenIds: readonly number[],
  blocks: readonly SerializedKVSwapBlock[],
): number[][] {
  if (tokenIds.length === 0 || blocks.length === 0) return [];
  const wanted = new Set(tokenIds.slice(-64));
  const matchingLayerZeroRows = collectMatchingPersistedQueryRows(wanted, blocks.filter((block) => block.layer === 0));
  if (matchingLayerZeroRows.length > 0) return matchingLayerZeroRows;
  return collectMatchingPersistedQueryRows(wanted, blocks);
}

function collectMatchingPersistedQueryRows(
  wantedTokenIds: ReadonlySet<number>,
  blocks: readonly SerializedKVSwapBlock[],
): number[][] {
  const rows: number[][] = [];
  for (const block of blocks) {
    if (!block.queryRows || !block.tokenIds) continue;
    const limit = Math.min(block.queryRows.length, block.tokenIds.length);
    for (let index = 0; index < limit; index += 1) {
      const tokenId = block.tokenIds[index];
      const row = block.queryRows[index];
      if (tokenId === undefined || !row || !wantedTokenIds.has(tokenId)) continue;
      rows.push([...row]);
      if (rows.length >= 64) return rows;
    }
  }
  return rows;
}

function buildTokenLowRankValues(tokenIds: number[], rank: number): number[] {
  const values = new Array(rank).fill(0);
  const recent = tokenIds.slice(-64);
  if (recent.length === 0) return values;
  for (let index = 0; index < recent.length; index += 1) {
    const tokenId = recent[index] ?? 0;
    const slot = positiveModulo(tokenId, rank);
    const normalizedToken = (positiveModulo(tokenId, 997) / 997) * 2 - 1;
    const recencyWeight = (index + 1) / recent.length;
    values[slot] += normalizedToken * recencyWeight;
  }
  const norm = Math.sqrt(values.reduce((sum, value) => sum + value * value, 0)) || 1;
  return values.map((value) => value / norm);
}

function serializeLowRankKeySummary(summary: KVLowRankKeySummary): SerializedKVLowRankKeySummary {
  return {
    blockId: summary.blockId,
    rank: summary.rank,
    projectionId: summary.projectionId,
    layer: summary.layer,
    headGroupId: summary.headGroupId,
    checksum: summary.checksum,
    qualityScore: summary.qualityScore,
    values: Array.from(summary.values),
  };
}

function deserializeLowRankKeySummary(summary: SerializedKVLowRankKeySummary): KVLowRankKeySummary {
  return {
    ...summary,
    values: new Float32Array(summary.values),
  };
}

function toPredictiveKvBlock(block: SerializedKVSwapBlock): KVBlock | null {
  const lowRankKeySummary = lowRankSummaryFromSerializedBlock(block);
  if (!lowRankKeySummary) return null;
  const runtimeBlockId = block.runtimeBlockId ?? block.id;
  return {
    id: block.id,
    layer: block.layer,
    startToken: block.startToken,
    endToken: block.endToken,
    tier: "ram",
    pinned: block.pinned,
    importance: block.importance,
    lastAccessAt: block.lastAccessAt,
    sourceBlockId: runtimeBlockId,
    estimatedBytes: block.estimatedBytes,
    ...(block.checksum ? { checksum: block.checksum } : {}),
    summaryRank: lowRankKeySummary.rank,
    compressedKeySummary: new Float32Array(lowRankKeySummary.values),
    lowRankKeySummary,
  };
}

function lowRankSummaryFromSerializedBlock(block: SerializedKVSwapBlock): KVLowRankKeySummary | null {
  if (block.lowRankKeySummary) {
    return deserializeLowRankKeySummary({
      ...block.lowRankKeySummary,
      blockId: block.id,
    });
  }
  if (!Array.isArray(block.compressedKeySummary) || !block.summaryRank) return null;
  return {
    blockId: block.id,
    rank: block.summaryRank,
    projectionId: CLIENT_LOW_RANK_PROJECTION_ID,
    layer: block.layer,
    headGroupId: "all_heads",
    checksum: block.checksum ?? `persisted:${block.id}`,
    qualityScore: clampRatio(block.importance),
    values: new Float32Array(block.compressedKeySummary.slice(0, block.summaryRank)),
  };
}

function mergePrefetchTraceWithAttention(trace: BrowserKvPrefetchTrace, selectedBlockIds: string[]): BrowserKvPrefetchTrace {
  if (trace.prefetchStrategy === "exact_reuse" || selectedBlockIds.length === 0 || trace.predictedHotBlocks.length === 0) return trace;
  const prefetched = new Set(trace.prefetchedBlocks);
  const prefetchedRuntimeAliases = new Set<string>();
  for (const block of trace.predictedHotBlocks) {
    if (!prefetched.has(block.blockId)) continue;
    for (const alias of runtimeBlockAliases(block)) prefetchedRuntimeAliases.add(alias);
  }
  const hits = selectedBlockIds.filter((blockId) => prefetchedRuntimeAliases.has(blockId)).length;
  const hitRate = selectedBlockIds.length > 0 ? hits / selectedBlockIds.length : trace.prefetchHitRate;
  const misses = selectedBlockIds.length - hits;
  return {
    ...trace,
    prefetchHitRate: round6(hitRate),
    attentionStallMs: misses > 0 ? Math.max(trace.attentionStallMs, trace.prefetchLatencyMs) : 0,
    prefetchStrategy: hits > 0 ? trace.prefetchStrategy : "miss_stall",
  };
}

function runtimeBlockAliases(block: KVSwapPredictiveHotBlockTrace): string[] {
  const aliases = [block.blockId];
  if (block.runtimeBlockId) {
    aliases.push(block.runtimeBlockId);
    const suffix = block.runtimeBlockId.split(":").at(-1);
    if (suffix) aliases.push(suffix);
  }
  return aliases;
}

function round6(value: number): number {
  if (!Number.isFinite(value)) return 0;
  return Math.round(value * 1_000_000) / 1_000_000;
}

function normalizePositiveInteger(value: number | null | undefined): number | null {
  if (value === undefined || value === null || !Number.isFinite(value)) return null;
  const normalized = Math.floor(value);
  return normalized > 0 ? normalized : null;
}

function normalizeQwenThinkingMode(value: QwenThinkingMode | undefined): QwenThinkingMode {
  return value === "enabled" ? "enabled" : "disabled";
}

function resolveGenerationTokenLimit(streamMaxTokens: number | undefined, configuredMaxGenerationTokens: number | null): number {
  const resolved = normalizePositiveInteger(streamMaxTokens) ?? configuredMaxGenerationTokens;
  if (resolved === null) {
    throw new Error("Unlocked browser transformer requires an explicit maxTokens or maxGenerationTokens budget.");
  }
  return resolved;
}

function clampRatio(value: number): number {
  if (!Number.isFinite(value)) return 0;
  return Math.max(0, Math.min(1, value));
}

function trimRuntimeTokenIds(tokenIds: number[], maxRuntimePromptTokens: number | null): number[] {
  if (!maxRuntimePromptTokens || tokenIds.length <= maxRuntimePromptTokens) return tokenIds;
  return tokenIds.slice(-maxRuntimePromptTokens);
}

function buildLogitCandidateTokenIds(
  promptTokenIds: number[],
  tokenizer: UnlockedBrowserTokenizer,
  vocabSize: number,
  candidateLimit: number | null,
): number[] | null {
  if (!candidateLimit) return null;
  const candidates = new Set<number>();
  const addTokenId = (tokenId: number) => candidates.add(positiveModulo(tokenId, vocabSize));
  for (const tokenId of promptTokenIds.slice(-candidateLimit)) addTokenId(tokenId);
  for (const tokenId of tokenizer.encode(" ready ok done yes no . ,\n", vocabSize)) addTokenId(tokenId);
  for (let tokenId = 0; candidates.size < candidateLimit && tokenId < vocabSize; tokenId += 1) addTokenId(tokenId);
  const candidateTokenIds = [...candidates].slice(0, candidateLimit);
  return candidateTokenIds.length > 0 ? candidateTokenIds : null;
}

function buildQwenThinkingSuppressedTokenIds(
  tokenizer: UnlockedBrowserTokenizer,
  vocabSize: number,
  qwenThinkingMode: QwenThinkingMode,
): number[] {
  const ids = new Set<number>();
  const markers = [
    ...ASSISTANT_CONTROL_MARKERS,
    ...ASSISTANT_STOP_MARKERS,
    ...(qwenThinkingMode === "disabled" ? [THINK_START_MARKER, THINK_END_MARKER] : []),
  ];
  for (const marker of markers) {
    const tokenId = specialTokenIdForMarker(tokenizer, marker, vocabSize);
    if (tokenId !== null) ids.add(tokenId);
  }
  return [...ids];
}

function buildDecodeSuppressedTokenIds(
  baseSuppressedTokenIds: number[],
  generatedTokenIds: readonly number[],
  generatedTokenTexts: readonly string[],
  vocabSize: number,
): number[] {
  const ids = new Set(baseSuppressedTokenIds);
  const repeatedTokenId = repeatedVisibleTokenId(generatedTokenIds, generatedTokenTexts, vocabSize);
  if (repeatedTokenId !== null) ids.add(repeatedTokenId);
  return [...ids];
}

function repeatedVisibleTokenId(
  generatedTokenIds: readonly number[],
  generatedTokenTexts: readonly string[],
  vocabSize: number,
): number | null {
  if (generatedTokenIds.length < CLIENT_REPEAT_SUPPRESSION_WINDOW) return null;
  const recentIds = generatedTokenIds.slice(-CLIENT_REPEAT_SUPPRESSION_WINDOW);
  const recentTexts = generatedTokenTexts
    .slice(-CLIENT_REPEAT_SUPPRESSION_WINDOW)
    .map(normalizeGeneratedTokenForRepetition);
  const firstText = recentTexts[0] ?? "";
  if (!firstText || recentTexts.some((text) => text !== firstText)) return null;
  const lastTokenId = recentIds.at(-1);
  return Number.isInteger(lastTokenId) ? positiveModulo(lastTokenId ?? 0, vocabSize) : null;
}

function normalizeGeneratedTokenForRepetition(text: string): string {
  const visible = text
    .replaceAll(UNLOCKED_PROOF_MARKER, "")
    .replaceAll(THINK_START_MARKER, "")
    .replaceAll(THINK_END_MARKER, "")
    .trim()
    .toLowerCase();
  const words = visible.match(/[a-z0-9]+/g) ?? [];
  return words.length === 1 ? words[0] ?? "" : "";
}

function specialTokenIdForMarker(tokenizer: UnlockedBrowserTokenizer, marker: string, vocabSize: number): number | null {
  const tokenIds = tokenizer.encode(marker, vocabSize);
  if (tokenIds.length !== 1) return null;
  const tokenId = positiveModulo(tokenIds[0] ?? 0, vocabSize);
  return tokenizer.decode(tokenId) === marker ? tokenId : null;
}

function toDecodeProof(
  decode: Awaited<ReturnType<UnlockedBrowserTransformerBackend["decode"]>>,
  mtp: UnlockedBrowserMtpProof,
  kvCacheHandle: unknown,
  expectedLayerCount: number,
): UnlockedBrowserDecodeProof {
  const proof = readUnlockedBrowserDecodeHandle(decode.logitsHandle);
  const prefillProof = readUnlockedBrowserKvCacheHandle(kvCacheHandle).prefillProof;
  const executedLayerCount = countProofLayerIndexes([
    ...(prefillProof?.layers ?? []),
    ...(proof.backendProof?.mlpLayers ?? []),
    ...(proof.backendProof?.projectionLayers ?? []),
    ...(proof.backendProof?.oProjectionLayers ?? []),
    ...decode.traces,
  ]);
  const decodeProjectionBackends = proof.backendProof?.projectionLayers?.length
    ? proof.backendProof.projectionLayers.map((layer) => ({
        layerIndex: layer.layerIndex,
        qProjection: layer.qProjection.backend,
        kProjection: layer.kProjection.backend,
        vProjection: layer.vProjection.backend,
        oProjection: layer.oProjection.backend,
        projectionKind: isDenseMatMulTrace(layer.oProjection.trace) ? "matmul" as const : "matvec" as const,
        ...(isDenseMatMulTrace(layer.oProjection.trace) ? { tokens: layer.oProjection.trace.tokens } : {}),
        ...(isDenseMatVecTrace(layer.oProjection.trace) ? { selectedRows: layer.oProjection.trace.selectedRows } : {}),
      }))
    : proof.backendProof?.oProjectionLayers?.map((layer) => ({
        layerIndex: layer.layerIndex,
        oProjection: layer.backend,
        projectionKind: isDenseMatMulTrace(layer.trace) ? "matmul" as const : "matvec" as const,
        ...(isDenseMatMulTrace(layer.trace) ? { tokens: layer.trace.tokens } : {}),
        ...(isDenseMatVecTrace(layer.trace) ? { selectedRows: layer.trace.selectedRows } : {}),
      }));
  return {
    tensorControl: true,
    tspSteps: proof.tspTrace.map((step) => step.kind),
    selectedBlockIds: [...new Set(decode.traces.flatMap((trace) => trace.selectedBlockIds))],
    kvPagingEvents: proof.kvPagingEvents.length,
    tokenId: decode.tokenId,
    expectedLayerCount,
    executedLayerCount,
    ...(proof.backendProof?.mlpLayers?.length
      ? {
          mlpKernelBackends: proof.backendProof.mlpLayers.map((mlpProof) => ({
            layerIndex: mlpProof.layerIndex,
            backend: mlpProof.backend,
            activationKind: mlpProof.trace.activationKind,
          })),
        }
      : {}),
    ...(proof.backendProof?.logitProjection
      ? {
          logitProjectionBackend: proof.backendProof.logitProjection.backend,
          logitProjectionSelectedRows: proof.backendProof.logitProjection.trace.selectedRows,
          logitProjectionFullRows: proof.backendProof.logitProjection.fullRowCount,
          logitProjectionPurpose: proof.backendProof.logitProjection.purpose,
          ...(proof.backendProof.logitProjection.trace.readbackStrategy ? { logitProjectionReadbackStrategy: proof.backendProof.logitProjection.trace.readbackStrategy } : {}),
          ...(proof.backendProof.logitProjection.trace.gpuReducedRows !== undefined ? { logitProjectionGpuReducedRows: proof.backendProof.logitProjection.trace.gpuReducedRows } : {}),
          ...(proof.backendProof.logitProjection.trace.readbackRows !== undefined ? { logitProjectionReadbackRows: proof.backendProof.logitProjection.trace.readbackRows } : {}),
          ...(proof.backendProof.logitProjection.trace.readbackBytes !== undefined ? { logitProjectionReadbackBytes: proof.backendProof.logitProjection.trace.readbackBytes } : {}),
          ...(proof.backendProof.logitProjection.trace.dispatchCount !== undefined ? { logitProjectionDispatchCount: proof.backendProof.logitProjection.trace.dispatchCount } : {}),
          ...(proof.backendProof.logitProjection.trace.tiles !== undefined ? { logitProjectionTiles: proof.backendProof.logitProjection.trace.tiles } : {}),
          ...(proof.backendProof.logitProjection.trace.tileRows !== undefined ? { logitProjectionTileRows: proof.backendProof.logitProjection.trace.tileRows } : {}),
          ...(proof.backendProof.logitProjection.selectedRowIds
            ? { logitProjectionCandidateTokenIds: proof.backendProof.logitProjection.selectedRowIds.slice(0, 64) }
            : {}),
          ...(proof.logits ? { logitProjectionCandidateScores: proof.logits.slice(0, 64) } : {}),
          ...(proof.backendProof.logitProjection.trace.compactTopK !== undefined ? { compactLogitTopK: proof.backendProof.logitProjection.trace.compactTopK } : {}),
          ...(proof.backendProof.sampling
            ? {
                compactLogitTopK: proof.backendProof.sampling.compactLogitTopK,
                samplingTemperature: proof.backendProof.sampling.temperature,
                samplingTopP: proof.backendProof.sampling.topP,
                repetitionPenalty: proof.backendProof.sampling.repetitionPenalty,
                greedyDecodeUsed: proof.backendProof.sampling.greedyDecodeUsed,
                sampledTokenRank: proof.backendProof.sampling.selectedRank,
              }
            : {}),
        }
      : {}),
    ...(prefillProof?.prefillChunkCount !== undefined ? { prefillChunkCount: prefillProof.prefillChunkCount } : {}),
    ...(prefillProof?.prefillChunkSize !== undefined ? { prefillChunkSize: prefillProof.prefillChunkSize } : {}),
    ...(prefillProof?.shapeBucket !== undefined ? { shapeBucket: prefillProof.shapeBucket } : {}),
    ...(prefillProof?.pipelineCacheKey !== undefined ? { pipelineCacheKey: prefillProof.pipelineCacheKey } : {}),
    ...(prefillProof?.prefillDispatchTargetMs !== undefined ? { prefillDispatchTargetMs: prefillProof.prefillDispatchTargetMs } : {}),
    ...(prefillProof?.maxDispatchEstimatedMs !== undefined ? { maxDispatchEstimatedMs: prefillProof.maxDispatchEstimatedMs } : {}),
    ...(prefillProof?.prefillChunkDispatch !== undefined ? { prefillChunkDispatch: prefillProof.prefillChunkDispatch } : {}),
    ...(decodeProjectionBackends?.length
      ? {
          decodeProjectionBackends,
        }
      : {}),
    ...(prefillProof?.layers.length
      ? {
          prefillMlpKernelBackends: prefillProof.layers
            .filter((layer) => layer.mlp)
            .map((layer) => ({
              layerIndex: layer.layerIndex,
              backend: layer.mlp?.backend ?? "cpu_reference",
              activationKind: layer.mlp?.lastTrace.activationKind ?? "gelu",
              rowCount: layer.mlp?.rowCount ?? 0,
            })),
          prefillProjectionBackends: prefillProof.layers.map((layer) => ({
            layerIndex: layer.layerIndex,
            qProjection: layer.qProjection.backend,
            kProjection: layer.kProjection.backend,
            vProjection: layer.vProjection.backend,
            oProjection: layer.oProjection.backend,
          })),
          prefillAttentionBackends: prefillProof.layers.map((layer) => ({
            layerIndex: layer.layerIndex,
            attentionBackend: layer.attentionBackend,
            packedHeadBackends: layer.packedHeadBackends,
            packedHeadCount: layer.packedHeadCount,
            ...(layer.keyValueHeadCount !== undefined ? { keyValueHeadCount: layer.keyValueHeadCount } : {}),
            ...(layer.keyValueCompressionRatio !== undefined ? { keyValueCompressionRatio: layer.keyValueCompressionRatio } : {}),
          })),
        }
      : {}),
    ...(decode.traces.length
      ? {
          decodeAttentionBackends: decode.traces.map((trace) => {
            const backendTrace = trace as typeof trace & {
              attentionBackend?: "webgpu" | "cpu_reference" | "mixed";
              packedHeadBackends?: Array<"webgpu" | "cpu_reference">;
              packedHeadCount?: number;
              keyValueHeadCount?: number;
              keyValueCompressionRatio?: number;
            };
            return {
              layerIndex: trace.layerIndex,
              attentionBackend: backendTrace.attentionBackend ?? "unknown",
              packedHeadBackends: backendTrace.packedHeadBackends ?? [],
              ...(backendTrace.packedHeadCount !== undefined ? { packedHeadCount: backendTrace.packedHeadCount } : {}),
              ...(backendTrace.keyValueHeadCount !== undefined ? { keyValueHeadCount: backendTrace.keyValueHeadCount } : {}),
              ...(backendTrace.keyValueCompressionRatio !== undefined ? { keyValueCompressionRatio: backendTrace.keyValueCompressionRatio } : {}),
            };
          }),
        }
      : {}),
    mtp,
    ...(proof.backendProof?.decodePerf ? { decodePerf: proof.backendProof.decodePerf } : {}),
  };
}

function countProofLayerIndexes(layers: Array<{ layerIndex: number }>): number {
  return new Set(layers.filter((layer) => Number.isInteger(layer.layerIndex) && layer.layerIndex >= 0).map((layer) => layer.layerIndex)).size;
}

function isDenseMatMulTrace(trace: unknown): trace is { tokens: number } {
  return typeof trace === "object" && trace !== null && "tokens" in trace;
}

function isDenseMatVecTrace(trace: unknown): trace is { selectedRows: number } {
  return typeof trace === "object" && trace !== null && "selectedRows" in trace;
}

function serializeKvCacheBlocks(
  cache: ReturnType<typeof readUnlockedBrowserKvCacheHandle>,
  namespace: string,
  metadata: KVCachePersistenceMetadata,
): SerializedKVSwapBlock[] {
  const now = new Date().toISOString();
  const tokenLimit = metadata.phase === "prefill" ? metadata.tokenIds?.length : undefined;
  return cache.kvBlocks
    .filter((block) => tokenLimit === undefined || block.startToken < tokenLimit)
    .map((block) => serializeKvBlock(
      tokenLimit === undefined || block.endToken <= tokenLimit ? block : { ...block, endToken: tokenLimit },
      cache,
      namespace,
      now,
      metadata,
    ));
}

function serializeKvBlock(
  block: KVBlock,
  cache: ReturnType<typeof readUnlockedBrowserKvCacheHandle>,
  namespace: string,
  now: string,
  metadata: KVCachePersistenceMetadata,
): SerializedKVSwapBlock {
  const state = cache.layerStates[block.layer];
  const tokenIds = metadata.tokenIds?.slice(block.startToken, block.endToken);
  const queryRows = state?.q.slice(block.startToken, block.endToken).map((row) => [...row]) ?? [];
  const keyRows = state?.k.slice(block.startToken, block.endToken).map((row) => [...row]) ?? [];
  const valueRows = state?.v.slice(block.startToken, block.endToken).map((row) => [...row]) ?? [];
  const compactKeyRows = state?.compactK?.slice(block.startToken, block.endToken).map((row) => [...row]) ?? [];
  const compactValueRows = state?.compactV?.slice(block.startToken, block.endToken).map((row) => [...row]) ?? [];
  const hiddenRows = state?.hidden.slice(block.startToken, block.endToken).map((row) => [...row]) ?? [];
  const prefillProof = metadata.phase === "prefill"
    ? serializePrefillProofForPersistence(cache.prefillProof)
    : undefined;
  const lowRankKeySummary = serializeLowRankKeySummary(compressKeyRowsToLowRankSummary({
    blockId: block.id,
    projectionId: CLIENT_LOW_RANK_PROJECTION_ID,
    layer: block.layer,
    headGroupId: "all_heads",
    rows: keyRows,
    rank: CLIENT_LOW_RANK_PREFETCH_RANK,
    ...(block.checksum !== undefined ? { checksum: block.checksum } : {}),
    qualityScore: block.importance,
  }));
  const compressedKeySummary = lowRankKeySummary.values;
  const promptTokenHash = metadata.tokenIds ? tokenHash(metadata.tokenIds) : undefined;
  const byteLength = estimateSerializedBlockBytes(
    keyRows,
    valueRows,
    compressedKeySummary,
    queryRows,
    hiddenRows,
    tokenIds,
    prefillProof,
    compactKeyRows,
    compactValueRows,
  );
  return {
    version: KV_SWAP_STORAGE_VERSION,
    namespace,
    id: persistedBlockId(cache.modelId, block.id, metadata.phase, promptTokenHash),
    modelId: cache.modelId,
    requestId: cache.requestId,
    runtimeBlockId: block.id,
    phase: metadata.phase,
    ...(metadata.modelFingerprint ? { modelFingerprint: metadata.modelFingerprint } : {}),
    ...(promptTokenHash ? { promptTokenHash } : {}),
    ...(metadata.tokenIds ? { promptTokenIds: metadata.tokenIds } : {}),
    ...(metadata.tokenIds ? { prefillTokenCount: metadata.tokenIds.length } : {}),
    ...(metadata.runtimeLayerCount !== undefined ? { runtimeLayerCount: metadata.runtimeLayerCount } : {}),
    ...(metadata.policyHash ? { policyHash: metadata.policyHash } : {}),
    layer: block.layer,
    startToken: block.startToken,
    endToken: block.endToken,
    pinned: block.pinned,
    importance: block.importance,
    estimatedBytes: block.estimatedBytes,
    checksum: block.checksum ?? lowRankKeySummary.checksum,
    summaryRank: lowRankKeySummary.rank,
    compressedKeySummary,
    lowRankKeySummary,
    ...(prefillProof ? { prefillProof } : {}),
    ...(tokenIds && tokenIds.length > 0 ? { tokenIds } : {}),
    ...(queryRows.length > 0 ? { queryRows } : {}),
    keyRows,
    valueRows,
    ...(compactKeyRows.length > 0 ? { compactKeyRows } : {}),
    ...(compactValueRows.length > 0 ? { compactValueRows } : {}),
    ...(hiddenRows.length > 0 ? { hiddenRows } : {}),
    createdAt: now,
    updatedAt: now,
    lastAccessAt: block.lastAccessAt,
    byteLength,
  };
}

function estimateSerializedBlockBytes(
  keyRows: number[][],
  valueRows: number[][],
  compressedKeySummary?: number[] | string,
  queryRows: number[][] = [],
  hiddenRows: number[][] = [],
  tokenIds: number[] = [],
  prefillProof?: SerializedKVSwapPrefillProof,
  compactKeyRows: number[][] = [],
  compactValueRows: number[][] = [],
): number {
  const tensorBytes = (
    keyRows.flat().length
    + valueRows.flat().length
    + compactKeyRows.flat().length
    + compactValueRows.flat().length
    + queryRows.flat().length
    + hiddenRows.flat().length
  ) * Float32Array.BYTES_PER_ELEMENT;
  const tokenBytes = tokenIds.length * Uint32Array.BYTES_PER_ELEMENT;
  const summaryBytes = Array.isArray(compressedKeySummary)
    ? compressedKeySummary.length * Float32Array.BYTES_PER_ELEMENT
    : (compressedKeySummary?.length ?? 0);
  const proofBytes = prefillProof ? JSON.stringify(prefillProof).length : 0;
  return tensorBytes + tokenBytes + summaryBytes + proofBytes + 512;
}

function serializePrefillProofForPersistence(
  proof: UnlockedBrowserPrefillBackendProof | undefined,
): SerializedKVSwapPrefillProof | undefined {
  if (!proof?.layers.length) return undefined;
  return {
    layers: proof.layers.map((layer) => ({
      layerIndex: layer.layerIndex,
      qProjection: serializeKernelBackend(layer.qProjection.backend),
      kProjection: serializeKernelBackend(layer.kProjection.backend),
      vProjection: serializeKernelBackend(layer.vProjection.backend),
      oProjection: serializeKernelBackend(layer.oProjection.backend),
      ...(layer.mlp
        ? {
            mlpBackend: serializeKernelBackend(layer.mlp.backend),
            mlpActivationKind: layer.mlp.lastTrace.activationKind,
            mlpRowCount: layer.mlp.rowCount,
          }
        : {}),
      attentionBackend: serializeKernelBackend(layer.attentionBackend),
      packedHeadBackends: layer.packedHeadBackends.map(serializeKernelBackend),
      packedHeadCount: layer.packedHeadCount,
      selectedKeyRows: layer.selectedKeyRows,
      ...(layer.prefillChunkDispatch ? { prefillChunkDispatch: layer.prefillChunkDispatch } : {}),
      ...(layer.attentionDispatchCount !== undefined ? { attentionDispatchCount: layer.attentionDispatchCount } : {}),
      ...(layer.awaitedDispatchBreaks !== undefined ? { awaitedDispatchBreaks: layer.awaitedDispatchBreaks } : {}),
    })),
    ...(proof.prefillChunkCount !== undefined ? { prefillChunkCount: proof.prefillChunkCount } : {}),
    ...(proof.prefillChunkSize !== undefined ? { prefillChunkSize: proof.prefillChunkSize } : {}),
    ...(proof.shapeBucket !== undefined ? { shapeBucket: proof.shapeBucket } : {}),
    ...(proof.pipelineCacheKey !== undefined ? { pipelineCacheKey: proof.pipelineCacheKey } : {}),
    ...(proof.prefillDispatchTargetMs !== undefined ? { prefillDispatchTargetMs: proof.prefillDispatchTargetMs } : {}),
    ...(proof.maxDispatchEstimatedMs !== undefined ? { maxDispatchEstimatedMs: proof.maxDispatchEstimatedMs } : {}),
    ...(proof.prefillChunkDispatch !== undefined ? { prefillChunkDispatch: proof.prefillChunkDispatch } : {}),
    ...(proof.attentionDispatchCount !== undefined ? { attentionDispatchCount: proof.attentionDispatchCount } : {}),
    ...(proof.awaitedDispatchBreaks !== undefined ? { awaitedDispatchBreaks: proof.awaitedDispatchBreaks } : {}),
  };
}

function serializeKernelBackend(value: string | undefined): SerializedKVSwapKernelBackend {
  return value === "webgpu" || value === "mixed" ? value : "cpu_reference";
}

function persistedBlockId(modelId: string, runtimeBlockId: string, phase: "prefill" | "decode", promptTokenHash?: string): string {
  return `${modelId}:${phase}:${promptTokenHash ?? "live"}:${runtimeBlockId}`;
}

function mergeSerializedKvBlocks(existing: SerializedKVSwapBlock[], next: SerializedKVSwapBlock[]): SerializedKVSwapBlock[] {
  const merged = new Map(existing.map((block) => [block.id, block]));
  for (const block of next) merged.set(block.id, block);
  return [...merged.values()];
}

function buildReusedKvCacheHandle(input: {
  blocks: SerializedKVSwapBlock[];
  namespace: string;
  modelId: string;
  modelFingerprint: string;
  requestId: string;
  tokenIds: number[];
  layerCount: number;
}): UnlockedBrowserKvCacheHandle | null {
  const promptTokenHash = tokenHash(input.tokenIds);
  const policyHash = policyHashFor(input.tokenIds.length, input.layerCount);
  const candidates = input.blocks.filter((block) => (
    block.namespace === input.namespace
    && block.modelId === input.modelId
    && block.phase === "prefill"
    && block.modelFingerprint === input.modelFingerprint
    && block.promptTokenHash === promptTokenHash
    && block.prefillTokenCount === input.tokenIds.length
    && block.runtimeLayerCount === input.layerCount
    && block.policyHash === policyHash
    && arraysEqual(block.promptTokenIds ?? [], input.tokenIds)
  ));
  if (candidates.length === 0) return null;

  const layers: UnlockedBrowserKvCacheHandle["layers"] = {};
  const layerStates: UnlockedBrowserKvCacheHandle["layerStates"] = {};
  const kvBlocks: KVBlock[] = [];
  let blockTokenRanges: UnlockedBrowserKvCacheHandle["blockTokenRanges"] | null = null;
  const prefillProof = restorePrefillProofFromPersistedBlocks(candidates, input.layerCount);

  for (let layerIndex = 0; layerIndex < input.layerCount; layerIndex += 1) {
    const layerBlocks = candidates
      .filter((block) => block.layer === layerIndex)
      .sort((a, b) => a.startToken - b.startToken || a.endToken - b.endToken);
    const merged = mergeLayerRowsForReuse(layerBlocks, input.tokenIds);
    if (!merged) return null;
    const ranges = Object.fromEntries(layerBlocks.map((block) => [
      runtimeRangeKey(block),
      { tokenStart: block.startToken, tokenEnd: block.endToken },
    ]));
    blockTokenRanges ??= ranges;
    layerStates[layerIndex] = {
      hidden: merged.hidden,
      q: merged.q,
      k: merged.k,
      v: merged.v,
      ...(merged.compactK ? { compactK: merged.compactK } : {}),
      ...(merged.compactV ? { compactV: merged.compactV } : {}),
      projectedTokenCount: input.tokenIds.length,
    };
    layers[layerIndex] = {
      qHandle: createSsaToyTensorHandle({
        id: `unlocked:${input.requestId}:layer${layerIndex}:q:reused`,
        matrix: merged.q,
        blockTokenRanges: ranges,
      }),
      kHandle: createSsaToyTensorHandle({
        id: `unlocked:${input.requestId}:layer${layerIndex}:k:reused`,
        matrix: merged.k,
        blockTokenRanges: ranges,
      }),
      vHandle: createSsaToyTensorHandle({
        id: `unlocked:${input.requestId}:layer${layerIndex}:v:reused`,
        matrix: merged.v,
        blockTokenRanges: ranges,
      }),
    };
    kvBlocks.push(...layerBlocks.map((block) => toReusedKvBlock(block)));
  }

  if (!blockTokenRanges) return null;
  return {
    kind: "unlocked_browser_transformer_kv_cache",
    id: `unlocked:${input.requestId}:kv:reused`,
    modelId: input.modelId,
    requestId: input.requestId,
    tokenIds: [...input.tokenIds],
    blockTokenRanges,
    kvBlocks,
    layers,
    layerStates,
    ...(prefillProof ? { prefillProof } : {}),
  };
}

function restorePrefillProofFromPersistedBlocks(
  blocks: SerializedKVSwapBlock[],
  layerCount: number,
): UnlockedBrowserPrefillBackendProof | undefined {
  const serialized = blocks.find((block) => block.prefillProof)?.prefillProof;
  if (!serialized?.layers.length) return undefined;
  const seenLayers = new Set<number>();
  const layers = serialized.layers
    .filter((layer) => (
      Number.isInteger(layer.layerIndex)
      && layer.layerIndex >= 0
      && layer.layerIndex < layerCount
    ))
    .filter((layer) => {
      if (seenLayers.has(layer.layerIndex)) return false;
      seenLayers.add(layer.layerIndex);
      return true;
    })
    .map((layer) => {
      return {
        layerIndex: layer.layerIndex,
        qProjection: restoredPrefillProjectionProof(layer.qProjection),
        kProjection: restoredPrefillProjectionProof(layer.kProjection),
        vProjection: restoredPrefillProjectionProof(layer.vProjection),
        oProjection: restoredPrefillProjectionProof(layer.oProjection),
        ...(layer.mlpBackend
          ? {
              mlp: {
                backend: layer.mlpBackend,
                rowCount: layer.mlpRowCount ?? 0,
                lastTrace: {
                  backend: restoreKernelTraceBackend(layer.mlpBackend),
                  tokens: layer.mlpRowCount ?? 0,
                  inputSize: 0,
                  intermediateSize: 0,
                  outputSize: 0,
                  activationKind: restoreMlpActivationKind(layer.mlpActivationKind),
                  computeMs: 0,
                  metadata: {
                    source: "persisted_prefill_proof",
                  },
                },
              },
            }
          : {}),
        attentionBackend: layer.attentionBackend,
        packedHeadBackends: layer.packedHeadBackends,
        packedHeadCount: layer.packedHeadCount,
        selectedKeyRows: layer.selectedKeyRows ?? 0,
        ...(layer.prefillChunkDispatch ? { prefillChunkDispatch: layer.prefillChunkDispatch } : {}),
        ...(layer.attentionDispatchCount !== undefined ? { attentionDispatchCount: layer.attentionDispatchCount } : {}),
        ...(layer.awaitedDispatchBreaks !== undefined ? { awaitedDispatchBreaks: layer.awaitedDispatchBreaks } : {}),
      };
    });
  if (layers.length === 0) return undefined;
  return {
    layers,
    ...(serialized.prefillChunkCount !== undefined ? { prefillChunkCount: serialized.prefillChunkCount } : {}),
    ...(serialized.prefillChunkSize !== undefined ? { prefillChunkSize: serialized.prefillChunkSize } : {}),
    ...(serialized.shapeBucket !== undefined ? { shapeBucket: serialized.shapeBucket } : {}),
    ...(serialized.pipelineCacheKey !== undefined ? { pipelineCacheKey: serialized.pipelineCacheKey } : {}),
    ...(serialized.prefillDispatchTargetMs !== undefined ? { prefillDispatchTargetMs: serialized.prefillDispatchTargetMs } : {}),
    ...(serialized.maxDispatchEstimatedMs !== undefined ? { maxDispatchEstimatedMs: serialized.maxDispatchEstimatedMs } : {}),
    ...(serialized.prefillChunkDispatch !== undefined ? { prefillChunkDispatch: serialized.prefillChunkDispatch } : {}),
    ...(serialized.attentionDispatchCount !== undefined ? { attentionDispatchCount: serialized.attentionDispatchCount } : {}),
    ...(serialized.awaitedDispatchBreaks !== undefined ? { awaitedDispatchBreaks: serialized.awaitedDispatchBreaks } : {}),
  } as unknown as UnlockedBrowserPrefillBackendProof;
}

function restoredPrefillProjectionProof(backend: SerializedKVSwapKernelBackend) {
  const projectionBackend = backend === "webgpu" ? "webgpu" : "cpu_reference";
  return {
    backend: projectionBackend,
    trace: {
      backend: projectionBackend,
      tokens: 0,
      hidden: 0,
      outputSize: 0,
      computeMs: 0,
      metadata: {
        source: "persisted_prefill_proof",
      },
    },
  };
}

function restoreMlpActivationKind(value: string | undefined): "silu_gated" | "gelu" {
  return value === "silu_gated" ? "silu_gated" : "gelu";
}

function restoreKernelTraceBackend(value: SerializedKVSwapKernelBackend): "webgpu" | "cpu_reference" {
  return value === "webgpu" ? "webgpu" : "cpu_reference";
}

function mergeLayerRowsForReuse(
  blocks: SerializedKVSwapBlock[],
  expectedTokenIds: number[],
): { q: number[][]; k: number[][]; v: number[][]; compactK?: number[][]; compactV?: number[][]; hidden: number[][] } | null {
  if (blocks.length === 0) return null;
  const q: number[][] = [];
  const k: number[][] = [];
  const v: number[][] = [];
  const compactK: number[][] = [];
  const compactV: number[][] = [];
  const hidden: number[][] = [];
  let hasCompactRows = false;
  let cursor = 0;
  for (const block of blocks) {
    if (block.startToken !== cursor) return null;
    if (block.endToken > expectedTokenIds.length) return null;
    const tokenSlice = expectedTokenIds.slice(block.startToken, block.endToken);
    if (!arraysEqual(block.tokenIds ?? [], tokenSlice)) return null;
    if (!block.queryRows || !block.hiddenRows) return null;
    if (
      block.queryRows.length !== tokenSlice.length
      || block.keyRows.length !== tokenSlice.length
      || block.valueRows.length !== tokenSlice.length
      || block.hiddenRows.length !== tokenSlice.length
    ) return null;
    if (block.compactKeyRows || block.compactValueRows) {
      if (
        !block.compactKeyRows
        || !block.compactValueRows
        || block.compactKeyRows.length !== tokenSlice.length
        || block.compactValueRows.length !== tokenSlice.length
      ) return null;
      hasCompactRows = true;
      compactK.push(...block.compactKeyRows.map((row) => [...row]));
      compactV.push(...block.compactValueRows.map((row) => [...row]));
    }
    q.push(...block.queryRows.map((row) => [...row]));
    k.push(...block.keyRows.map((row) => [...row]));
    v.push(...block.valueRows.map((row) => [...row]));
    hidden.push(...block.hiddenRows.map((row) => [...row]));
    cursor = block.endToken;
  }
  if (cursor !== expectedTokenIds.length) return null;
  return {
    q,
    k,
    v,
    ...(hasCompactRows ? { compactK, compactV } : {}),
    hidden,
  };
}

function toReusedKvBlock(block: SerializedKVSwapBlock): KVBlock {
  const runtimeBlockId = block.runtimeBlockId ?? block.id;
  return {
    id: runtimeBlockId,
    layer: block.layer,
    startToken: block.startToken,
    endToken: block.endToken,
    tier: block.pinned ? "vram" : "ram",
    pinned: block.pinned,
    importance: block.importance,
    lastAccessAt: Date.now(),
    sourceBlockId: block.id,
    estimatedBytes: block.estimatedBytes,
    ...(block.checksum ? { checksum: block.checksum } : {}),
    ...(block.summaryRank !== undefined ? { summaryRank: block.summaryRank } : {}),
    ...(block.compressedKeySummary !== undefined
      ? { compressedKeySummary: Array.isArray(block.compressedKeySummary) ? new Float32Array(block.compressedKeySummary) : block.compressedKeySummary }
      : {}),
    ...(block.lowRankKeySummary ? { lowRankKeySummary: deserializeLowRankKeySummary(block.lowRankKeySummary) } : {}),
    tensorHandles: {
      key: { backend: "unlocked-browser-transformer", id: `${runtimeBlockId}:key`, dtype: "f32" },
      value: { backend: "unlocked-browser-transformer", id: `${runtimeBlockId}:value`, dtype: "f32" },
    },
  };
}

function runtimeRangeKey(block: SerializedKVSwapBlock): string {
  const runtimeBlockId = block.runtimeBlockId ?? block.id;
  const match = /^layer\d+:(.+)$/.exec(runtimeBlockId);
  return match?.[1] ?? runtimeBlockId;
}

function policyHashFor(tokenCount: number, layerCount: number): string {
  return `client-v4:block${CLIENT_ATTENTION_BLOCK_SIZE}:full${CLIENT_FULL_CONTEXT_DECODE_TOKEN_LIMIT}:sparse${CLIENT_MAX_SPARSE_DECODE_BLOCKS}:anchors${CLIENT_SPARSE_ANCHOR_BLOCKS}:trail${CLIENT_TRAILING_QUERY_BLOCKS}:tokens${tokenCount}:layers${layerCount}`;
}

function tokenHash(tokenIds: number[]): string {
  let hash = 2166136261;
  for (const tokenId of tokenIds) {
    hash ^= positiveModulo(tokenId, 0x100000000);
    hash = Math.imul(hash, 16777619) >>> 0;
  }
  return hash.toString(16).padStart(8, "0");
}

function buildCompactDecodeSamplingOptions(input: {
  options: ChatStreamOptions;
  requestId: string;
  step: number;
  recentTokenIds: number[];
}): {
  samplingTemperature: number;
  samplingTopP: number;
  repetitionPenalty: number;
  recentTokenIds: number[];
  samplingSeed: number;
} {
  return {
    samplingTemperature: normalizeSamplingNumber(input.options.temperature, 0.7, { min: 0 }),
    samplingTopP: normalizeSamplingNumber(input.options.topP, 0.9, { min: Number.EPSILON, max: 1 }),
    repetitionPenalty: normalizeSamplingNumber(input.options.repetitionPenalty, 1.05, { min: Number.EPSILON }),
    recentTokenIds: input.recentTokenIds.slice(-256),
    samplingSeed: Number.isInteger(input.options.samplingSeed)
      ? input.options.samplingSeed as number
      : hashRequestStep(input.requestId, input.step),
  };
}

function normalizeSamplingNumber(
  value: number | undefined,
  fallback: number,
  bounds: { min?: number; max?: number } = {},
): number {
  if (typeof value !== "number" || !Number.isFinite(value)) return fallback;
  const min = bounds.min ?? Number.NEGATIVE_INFINITY;
  const max = bounds.max ?? Number.POSITIVE_INFINITY;
  return Math.min(max, Math.max(min, value));
}

function hashRequestStep(requestId: string, step: number): number {
  let hash = 2166136261;
  for (let index = 0; index < requestId.length; index += 1) {
    hash ^= requestId.charCodeAt(index);
    hash = Math.imul(hash, 16777619) >>> 0;
  }
  hash ^= positiveModulo(step, 0x100000000);
  return Math.imul(hash, 16777619) >>> 0;
}

function arraysEqual(left: readonly number[], right: readonly number[]): boolean {
  return left.length === right.length && left.every((value, index) => value === right[index]);
}

function draftLocalTokenizerTokenIds(input: {
  promptTokenIds: number[];
  generatedTokenIds: number[];
  vocabSize: number;
  maxDraftTokens: number;
}): number[] {
  const recent = [...input.generatedTokenIds, ...input.promptTokenIds].slice(-64).reverse();
  const candidates: number[] = [];
  const seen = new Set<number>();
  for (const tokenId of recent) {
    const normalized = positiveModulo(tokenId, input.vocabSize);
    if (normalized === 0 || seen.has(normalized)) continue;
    seen.add(normalized);
    candidates.push(normalized);
  }
  if (candidates.length === 0) candidates.push(0);
  return Array.from(
    { length: input.maxDraftTokens },
    (_value, index) => candidates[index % candidates.length] as number,
  );
}

async function draftQwenPrefixTokenIds(input: {
  state: QwenPrefixDraftState;
  previousTokenId: number;
  maxDraftTokens: number;
  logitCandidateTokenIds: number[] | null;
  suppressedTokenIds: number[];
  logitTopK: number | null;
  logitTileRows: number | null;
}): Promise<{ tokenIds: number[]; source: "qwen_prefix_draft"; latencyMs: number }> {
  const started = performance.now();
  const tokenIds: number[] = [];
  let previousTokenId = input.previousTokenId;
  for (let index = 0; index < input.maxDraftTokens; index += 1) {
    const tokenCount = readUnlockedBrowserKvCacheHandle(input.state.kvCacheHandle).tokenIds.length + 1;
    const decode = await input.state.backend.decode({
      requestId: input.state.requestId,
      inputTokenId: previousTokenId,
      kvCacheHandle: input.state.kvCacheHandle,
      policy: buildClientPolicies(tokenCount, input.state.layerCount),
      ...(input.logitCandidateTokenIds ? { logitCandidateTokenIds: input.logitCandidateTokenIds } : {}),
      ...(input.suppressedTokenIds.length > 0 ? { suppressedTokenIds: input.suppressedTokenIds } : {}),
      ...(input.logitTopK ? { logitTopK: input.logitTopK } : {}),
      ...(input.logitTileRows ? { logitTileRows: input.logitTileRows } : {}),
    });
    tokenIds.push(decode.tokenId);
    previousTokenId = decode.tokenId;
  }
  return {
    tokenIds,
    source: "qwen_prefix_draft",
    latencyMs: Math.max(0, performance.now() - started),
  };
}

function toStreamedTokenIds(draftTokenIds: number[], targetTokenIds: number[], acceptedTokens: number): number[] {
  const accepted = draftTokenIds.slice(0, acceptedTokens);
  const correction = targetTokenIds[acceptedTokens];
  return correction === undefined ? accepted : [...accepted, correction];
}

interface UnlockedBrowserTokenizer {
  encode(text: string, vocabSize: number): number[];
  decode(tokenId: number): string;
  decodeStreamToken(tokenId: number): string;
  createStreamDecoder?(): UnlockedBrowserTokenStreamDecoder;
  formatMessages(messages: ChatClientMessage[], options?: QwenFormatOptions): string;
}

interface UnlockedBrowserTokenStreamDecoder {
  decodeToken(tokenId: number): string;
  flush(): string;
}

interface QwenFormatOptions {
  qwenThinkingMode?: QwenThinkingMode;
}

function decodeStreamToken(
  tokenizer: UnlockedBrowserTokenizer,
  streamDecoder: UnlockedBrowserTokenStreamDecoder | undefined,
  tokenId: number,
): string {
  return streamDecoder ? streamDecoder.decodeToken(tokenId) : tokenizer.decodeStreamToken(tokenId);
}

type ChatFormatter = (messages: ChatClientMessage[]) => string;

async function normalizeManifest(
  value: unknown,
  fallbackModelId: string,
  manifestPath: string,
): Promise<LoadedUnlockedBrowserManifest> {
  if (!isRecord(value)) throw new Error("Unlocked browser transformer manifest must be a JSON object.");
  if (value.schemaVersion !== 1) {
    throw new Error("Unlocked browser transformer manifest requires schemaVersion: 1.");
  }
  const weightsSource = isRecord(value.weights) ? value.weights : value;
  const weights = weightsSource as Partial<UnlockedBrowserTransformerWeights>;
  const manifestModelId = typeof weights.modelId === "string" && weights.modelId.trim() ? weights.modelId : "";
  if (manifestModelId && manifestModelId !== fallbackModelId) {
    throw new Error(`Unlocked browser transformer manifest modelId mismatch: expected ${fallbackModelId}, received ${manifestModelId}.`);
  }
  const shardCache = new Map<string, Promise<ArrayBuffer>>();
  const shardDigestCache = new Map<string, Promise<string>>();
  const normalizedWeights: UnlockedBrowserTransformerWeights = {
    modelId: manifestModelId || fallbackModelId,
    architecture: weights.architecture ?? "qwen3_decoder_control",
    vocabSize: requireNumber(weights.vocabSize, "vocabSize"),
    hiddenSize: requireNumber(weights.hiddenSize, "hiddenSize"),
    headDim: requireNumber(weights.headDim, "headDim"),
    ...(readOptionalPositiveInteger(weights.numAttentionHeads, "numAttentionHeads") !== undefined
      ? { numAttentionHeads: readOptionalPositiveInteger(weights.numAttentionHeads, "numAttentionHeads") as number }
      : {}),
    ...(readOptionalPositiveInteger(weights.numKeyValueHeads, "numKeyValueHeads") !== undefined
      ? { numKeyValueHeads: readOptionalPositiveInteger(weights.numKeyValueHeads, "numKeyValueHeads") as number }
      : {}),
    ...(readOptionalPositiveInteger(weights.maxPositionEmbeddings, "maxPositionEmbeddings") !== undefined
      ? { maxPositionEmbeddings: readOptionalPositiveInteger(weights.maxPositionEmbeddings, "maxPositionEmbeddings") as number }
      : {}),
    ...(readOptionalPositiveNumber(weights.ropeTheta, "ropeTheta") !== undefined
      ? { ropeTheta: readOptionalPositiveNumber(weights.ropeTheta, "ropeTheta") as number }
      : {}),
    ...(typeof weights.tieWordEmbeddings === "boolean" ? { tieWordEmbeddings: weights.tieWordEmbeddings } : {}),
    ...(typeof weights.rmsNormEps === "number" ? { rmsNormEps: weights.rmsNormEps } : {}),
    tokenEmbedding: await loadMatrix(weights.tokenEmbedding, "tokenEmbedding", manifestPath, shardCache, shardDigestCache),
    outputProjection: await loadMatrix(weights.outputProjection, "outputProjection", manifestPath, shardCache, shardDigestCache),
    ...(weights.finalNorm ? { finalNorm: await loadVector(weights.finalNorm, "finalNorm", manifestPath, shardCache, shardDigestCache) } : {}),
    layers: Array.isArray(weights.layers)
      ? await Promise.all(weights.layers.map((layer, index) => normalizeLayer(layer, index, manifestPath, shardCache, shardDigestCache)))
      : [],
  };
  return {
    weights: normalizedWeights,
    tokenizer: normalizeTokenizer(value.tokenizer, normalizedWeights.vocabSize),
  };
}

function normalizeTokenizer(value: unknown, vocabSize: number): UnlockedBrowserTokenizer {
  if (!isRecord(value) || (value.kind !== "vocab" && value.kind !== "qwen-bpe") || !Array.isArray(value.tokens)) {
    throw new Error("Unlocked browser transformer manifest requires tokenizer.tokens for non-fixture model decoding.");
  }
  const tokens = value.tokens.map((token) => {
    if (typeof token !== "string") throw new Error("Unlocked browser transformer tokenizer.tokens must be strings.");
    return token;
  });
  if (tokens.length < vocabSize) {
    throw new Error("Unlocked browser transformer tokenizer.tokens must cover the configured vocabSize.");
  }
  const unknownTokenId = typeof value.unknownTokenId === "number" && Number.isInteger(value.unknownTokenId)
    ? positiveModulo(value.unknownTokenId, vocabSize)
    : 0;
  if (value.kind === "qwen-bpe") {
    if (!Array.isArray(value.merges)) {
      throw new Error("Unlocked browser transformer qwen-bpe tokenizer requires tokenizer.merges.");
    }
    const merges = value.merges.map((merge, index): [string, string] => {
      if (Array.isArray(merge) && merge.length === 2 && typeof merge[0] === "string" && typeof merge[1] === "string") {
        return [merge[0], merge[1]];
      }
      if (typeof merge === "string") {
        const parts = merge.split(" ");
        if (parts.length >= 2) return [parts[0] as string, parts.slice(1).join(" ")];
      }
      throw new Error(`Unlocked browser transformer qwen-bpe tokenizer.merges[${index}] is invalid.`);
    });
    const specialTokens = Array.isArray(value.specialTokens)
      ? value.specialTokens.map((token) => {
          if (typeof token !== "string") throw new Error("Unlocked browser transformer tokenizer.specialTokens must be strings.");
          return token;
        })
      : tokens.filter((token) => token.startsWith("<|") && token.endsWith("|>"));
    const chatTemplate = typeof value.chatTemplate === "string" ? value.chatTemplate : undefined;
    return new QwenByteLevelBpeTokenizer(tokens, merges, unknownTokenId, specialTokens, chatTemplate);
  }
  return new VocabTokenizer(tokens, unknownTokenId);
}

function makeQwenPrefixDraftWeights(
  weights: UnlockedBrowserTransformerWeights,
  draftLayerCount: number,
): UnlockedBrowserTransformerWeights {
  return {
    ...weights,
    layers: weights.layers.slice(0, Math.max(1, Math.min(weights.layers.length, draftLayerCount))),
  };
}

async function normalizeLayer(
  value: unknown,
  index: number,
  manifestPath: string,
  shardCache: Map<string, Promise<ArrayBuffer>>,
  shardDigestCache: Map<string, Promise<string>>,
): Promise<UnlockedBrowserTransformerWeights["layers"][number]> {
  if (!isRecord(value)) throw new Error(`layers[${index}] must be a JSON object.`);
  return {
    ...(value.inputLayerNorm ? { inputLayerNorm: await loadVector(value.inputLayerNorm, `layers[${index}].inputLayerNorm`, manifestPath, shardCache, shardDigestCache) } : {}),
    qProj: await loadMatrix(value.qProj, `layers[${index}].qProj`, manifestPath, shardCache, shardDigestCache),
    kProj: await loadMatrix(value.kProj, `layers[${index}].kProj`, manifestPath, shardCache, shardDigestCache),
    vProj: await loadMatrix(value.vProj, `layers[${index}].vProj`, manifestPath, shardCache, shardDigestCache),
    oProj: await loadMatrix(value.oProj, `layers[${index}].oProj`, manifestPath, shardCache, shardDigestCache),
    ...(value.qNorm ? { qNorm: await loadVector(value.qNorm, `layers[${index}].qNorm`, manifestPath, shardCache, shardDigestCache) } : {}),
    ...(value.kNorm ? { kNorm: await loadVector(value.kNorm, `layers[${index}].kNorm`, manifestPath, shardCache, shardDigestCache) } : {}),
    ...(value.postAttentionLayerNorm ? { postAttentionLayerNorm: await loadVector(value.postAttentionLayerNorm, `layers[${index}].postAttentionLayerNorm`, manifestPath, shardCache, shardDigestCache) } : {}),
    ...(value.mlpGateProj ? { mlpGateProj: await loadMatrix(value.mlpGateProj, `layers[${index}].mlpGateProj`, manifestPath, shardCache, shardDigestCache) } : {}),
    ...(value.mlpUpProj ? { mlpUpProj: await loadMatrix(value.mlpUpProj, `layers[${index}].mlpUpProj`, manifestPath, shardCache, shardDigestCache) } : {}),
    ...(value.mlpDownProj ? { mlpDownProj: await loadMatrix(value.mlpDownProj, `layers[${index}].mlpDownProj`, manifestPath, shardCache, shardDigestCache) } : {}),
  };
}

function requireNumber(value: unknown, name: string): number {
  if (typeof value !== "number" || !Number.isInteger(value) || value <= 0) {
    throw new Error(`Unlocked browser transformer manifest requires positive integer ${name}.`);
  }
  return value;
}

function readOptionalPositiveInteger(value: unknown, name: string): number | undefined {
  if (value === undefined) return undefined;
  if (typeof value !== "number" || !Number.isInteger(value) || value <= 0) {
    throw new Error(`Unlocked browser transformer manifest ${name} must be a positive integer when provided.`);
  }
  return value;
}

function readOptionalPositiveNumber(value: unknown, name: string): number | undefined {
  if (value === undefined) return undefined;
  if (typeof value !== "number" || !Number.isFinite(value) || value <= 0) {
    throw new Error(`Unlocked browser transformer manifest ${name} must be a positive number when provided.`);
  }
  return value;
}

function requireMatrix(value: unknown, name: string): number[][] {
  if (!Array.isArray(value) || value.some((row) => !Array.isArray(row) || row.some((item) => typeof item !== "number"))) {
    throw new Error(`Unlocked browser transformer manifest requires numeric matrix ${name}.`);
  }
  return value as number[][];
}

function requireVector(value: unknown, name: string): number[] {
  if (!Array.isArray(value) || value.some((item) => typeof item !== "number")) {
    throw new Error(`Unlocked browser transformer manifest requires numeric vector ${name}.`);
  }
  return value as number[];
}

async function loadMatrix(
  value: unknown,
  name: string,
  manifestPath: string,
  shardCache: Map<string, Promise<ArrayBuffer>>,
  shardDigestCache: Map<string, Promise<string>>,
): Promise<RuntimeMatrix> {
  if (isRecord(value) && isPackedShardKind(value.kind)) {
    return loadPackedShardMatrix(requirePackedShardDescriptor(value, name), name, manifestPath, shardCache, shardDigestCache);
  }
  return requireMatrix(value, name);
}

async function loadVector(
  value: unknown,
  name: string,
  manifestPath: string,
  shardCache: Map<string, Promise<ArrayBuffer>>,
  shardDigestCache: Map<string, Promise<string>>,
): Promise<RuntimeVector> {
  if (isRecord(value) && isPackedShardKind(value.kind)) {
    return loadPackedShardVector(requirePackedShardDescriptor(value, name), name, manifestPath, shardCache, shardDigestCache);
  }
  return requireVector(value, name);
}

type PackedShardKind = "f32-shard" | "f16-shard";
type PackedShardDtype = "f32" | "f16";

interface PackedShardDescriptor {
  kind: PackedShardKind;
  uri: string;
  shape: [number] | [number, number];
  byteOffset?: number;
  sha256?: string;
  dtype?: PackedShardDtype;
}

function isPackedShardKind(value: unknown): value is PackedShardKind {
  return value === "f32-shard" || value === "f16-shard";
}

function isPackedShardDescriptor(value: unknown): value is PackedShardDescriptor {
  if (!isRecord(value)) return false;
  const byteOffset = value.byteOffset;
  const expectedDtype = value.kind === "f16-shard" ? "f16" : "f32";
  return isRecord(value)
    && isPackedShardKind(value.kind)
    && typeof value.uri === "string"
    && Array.isArray(value.shape)
    && (value.shape.length === 1 || value.shape.length === 2)
    && value.shape.every((item) => Number.isInteger(item) && item > 0)
    && (byteOffset === undefined || (typeof byteOffset === "number" && Number.isInteger(byteOffset) && byteOffset >= 0))
    && (value.sha256 === undefined || typeof value.sha256 === "string")
    && (value.dtype === undefined || value.dtype === expectedDtype);
}

function requirePackedShardDescriptor(value: unknown, name: string): PackedShardDescriptor {
  if (!isPackedShardDescriptor(value)) {
    throw new Error(`Unlocked browser transformer packed shard descriptor ${name} is invalid.`);
  }
  if (!value.sha256?.trim()) {
    throw new Error(`Unlocked browser transformer ${value.kind} descriptor ${name} requires sha256.`);
  }
  if (!/^[a-fA-F0-9]{64}$/.test(value.sha256)) {
    throw new Error(`Unlocked browser transformer ${value.kind} descriptor ${name} sha256 must be a 64-character hexadecimal SHA-256 digest.`);
  }
  return value;
}

function requireMatrixShape(descriptor: PackedShardDescriptor, name: string): [number, number] {
  if (descriptor.shape.length !== 2) {
    throw new Error(`Unlocked browser transformer ${descriptor.kind} descriptor ${name} requires a 2D matrix shape.`);
  }
  return descriptor.shape;
}

function requireVectorShape(descriptor: PackedShardDescriptor, name: string): [number] {
  if (descriptor.shape.length !== 1) {
    throw new Error(`Unlocked browser transformer ${descriptor.kind} descriptor ${name} requires a 1D vector shape.`);
  }
  return descriptor.shape;
}

async function loadPackedShardMatrix(
  descriptor: PackedShardDescriptor,
  name: string,
  manifestPath: string,
  shardCache: Map<string, Promise<ArrayBuffer>>,
  shardDigestCache: Map<string, Promise<string>>,
): Promise<RuntimeMatrix> {
  const shape = requireMatrixShape(descriptor, name);
  const [rowCount, colCount] = shape;
  const { shard, byteOffset } = await loadPackedShardBytes(descriptor, name, manifestPath, shardCache, shardDigestCache, rowCount * colCount);
  return descriptor.kind === "f16-shard"
    ? new F16Matrix(shard, byteOffset, rowCount, colCount)
    : new F32Matrix(shard, byteOffset, rowCount, colCount);
}

async function loadPackedShardVector(
  descriptor: PackedShardDescriptor,
  name: string,
  manifestPath: string,
  shardCache: Map<string, Promise<ArrayBuffer>>,
  shardDigestCache: Map<string, Promise<string>>,
): Promise<number[]> {
  const shape = requireVectorShape(descriptor, name);
  const { shard, byteOffset } = await loadPackedShardBytes(descriptor, name, manifestPath, shardCache, shardDigestCache, shape[0]);
  if (descriptor.kind === "f32-shard") {
    return Array.from(new Float32Array(shard, byteOffset, shape[0]));
  }
  const f16 = new Uint16Array(shard, byteOffset, shape[0]);
  const values = new Array<number>(f16.length);
  for (let index = 0; index < f16.length; index += 1) {
    values[index] = float16BitsToFloat32(f16[index] ?? 0);
  }
  return values;
}

async function loadPackedShardBytes(
  descriptor: PackedShardDescriptor,
  name: string,
  manifestPath: string,
  shardCache: Map<string, Promise<ArrayBuffer>>,
  shardDigestCache: Map<string, Promise<string>>,
  valueCount: number,
): Promise<{ shard: ArrayBuffer; byteOffset: number }> {
  const shardUrl = resolveShardUrl(descriptor.uri, manifestPath);
  const byteOffset = descriptor.byteOffset ?? 0;
  const sourceByteWidth = descriptor.kind === "f16-shard" ? Uint16Array.BYTES_PER_ELEMENT : Float32Array.BYTES_PER_ELEMENT;
  if (byteOffset % sourceByteWidth !== 0) {
    throw new Error(`Unlocked browser transformer shard ${name} byteOffset must align to ${descriptor.kind === "f16-shard" ? "f16" : "f32"} values.`);
  }
  const requiredBytes = valueCount * sourceByteWidth;
  const shard = await getShard(shardUrl, shardCache);
  if (descriptor.sha256) await verifyShardSha256(shardUrl, shard, descriptor.sha256, `sha256 for ${name}`, shardDigestCache);
  if (byteOffset + requiredBytes > shard.byteLength) {
    throw new Error(`Unlocked browser transformer shard ${name} does not contain enough ${descriptor.kind === "f16-shard" ? "f16" : "f32"} values for shape ${descriptor.shape.join("x")}.`);
  }
  if (descriptor.kind === "f32-shard") {
    return { shard, byteOffset };
  }
  return { shard, byteOffset };
}

async function getShard(url: string, shardCache: Map<string, Promise<ArrayBuffer>>): Promise<ArrayBuffer> {
  const existing = shardCache.get(url);
  if (existing) return existing;
  const promise = fetch(url).then(async (response) => {
    if (!response.ok) throw new Error(`Unlocked browser transformer shard failed to load: ${response.status}`);
    return response.arrayBuffer();
  });
  shardCache.set(url, promise);
  return promise;
}

function resolveShardUrl(uri: string, manifestPath: string): string {
  const baseHref = typeof globalThis.location === "object" && globalThis.location?.href
    ? globalThis.location.href
    : "http://localhost/";
  const absoluteManifestUrl = new URL(manifestPath, baseHref);
  return new URL(uri, absoluteManifestUrl).toString();
}

function makeFixtureWeights(modelId: string): UnlockedBrowserTransformerWeights {
  const identity = [
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1],
  ];
  return {
    modelId,
    architecture: "qwen3_decoder_control",
    vocabSize: 16,
    hiddenSize: 4,
    headDim: 4,
    tokenEmbedding: [
      [1, 0, 0, 0],
      [0, 1, 0, 0],
      [0, 0, 1, 0],
      [0, 0, 0, 1],
      [0.8, 0.2, 0.1, 0],
      [0.1, 0.8, 0.2, 0],
      [0, 0.1, 0.8, 0.2],
      [0.2, 0, 0.1, 0.8],
      [0.7, 0.5, 0.1, 0],
      [0, 0.7, 0.5, 0.1],
      [0.1, 0, 0.7, 0.5],
      [0.5, 0.1, 0, 0.7],
      [0.3, 0.3, 0.3, 0.1],
      [0.1, 0.3, 0.3, 0.3],
      [0.3, 0.1, 0.3, 0.3],
      [0.3, 0.3, 0.1, 0.3],
    ],
    outputProjection: [
      [0.2, 0.1, 0.1, 0],
      [0.1, 0.2, 0.1, 0],
      [0.1, 0.1, 0.2, 0],
      [0, 0.1, 0.1, 0.2],
      [0.4, 0.2, 0.1, 0],
      [0.1, 0.4, 0.2, 0],
      [0, 0.1, 0.4, 0.2],
      [0.2, 0, 0.1, 0.4],
      [0.5, 0.3, 0.1, 0],
      [0, 0.5, 0.3, 0.1],
      [0.1, 0, 0.5, 0.3],
      [0.3, 0.1, 0, 0.5],
      [0.6, 0.4, 0.1, 0],
      [0, 0.6, 0.4, 0.1],
      [0.1, 0, 0.6, 0.4],
      [0.4, 0.1, 0, 0.6],
    ],
    layers: [
      {
        qProj: identity,
        kProj: identity,
        vProj: identity,
        oProj: identity,
        mlpUpProj: identity,
        mlpDownProj: identity,
      },
    ],
  };
}

function buildClientPolicies(tokenCount: number, layerCount: number): SSALayerRoutingPolicy[] {
  return Array.from({ length: layerCount }, (_, layerIndex) => buildClientPolicy(tokenCount, layerIndex));
}

function buildClientPrefillChunkPlan(
  tokenCount: number,
  policies: SSALayerRoutingPolicy[],
): UnlockedBrowserPrefillChunkPlan {
  const plan = planPrefillChunks(tokenCount, {
    operation: "prefill_chunk",
    selectedBlockCount: maxSelectedBlockCount(policies),
    blockSize: maxClientPolicyBlockSize(policies),
    maxDispatchEstimatedMs: DEFAULT_PREFILL_DISPATCH_BUDGET_MS,
  });
  return toClientPrefillChunkPlan(plan);
}

function toClientPrefillChunkPlan(plan: PrefillChunkPlan): UnlockedBrowserPrefillChunkPlan {
  return {
    prefillChunkCount: plan.prefillChunkCount,
    prefillChunkSize: plan.prefillChunkSize,
    shapeBucket: plan.shapeBucket,
    pipelineCacheKey: plan.pipelineCacheKey,
    prefillDispatchTargetMs: plan.dispatchBudgetMs,
    maxDispatchEstimatedMs: plan.maxDispatchEstimatedMs,
  };
}

function maxSelectedBlockCount(policies: SSALayerRoutingPolicy[]): number {
  return policies.reduce((max, policy) => Math.max(max, collectClientSelectedBlockIds(policy).length), 0);
}

function maxClientPolicyBlockSize(policies: SSALayerRoutingPolicy[]): number {
  return policies.reduce((max, policy) => Math.max(max, policy.blockSize), 1);
}

function collectClientSelectedBlockIds(policy: SSALayerRoutingPolicy): string[] {
  const ids: string[] = [];
  for (const id of policy.pinnedBlockIds) pushUniqueClientBlockId(ids, id);
  for (const selected of Object.values(policy.selectedBlockIdsByQueryBlock)) {
    for (const id of selected) pushUniqueClientBlockId(ids, id);
  }
  return ids;
}

function pushUniqueClientBlockId(values: string[], value: string): void {
  if (!values.includes(value)) values.push(value);
}

function buildClientPolicy(tokenCount: number, layerIndex: number): SSALayerRoutingPolicy {
  const blockCount = Math.max(1, Math.ceil(tokenCount / CLIENT_ATTENTION_BLOCK_SIZE));
  const latestQueryBlock = blockCount - 1;
  const firstExplicitQueryBlock = Math.max(0, latestQueryBlock - CLIENT_TRAILING_QUERY_BLOCKS + 1);
  const fullContext = tokenCount <= CLIENT_FULL_CONTEXT_DECODE_TOKEN_LIMIT;
  const maxSelectedBlocks = fullContext ? blockCount : Math.min(blockCount, CLIENT_MAX_SPARSE_DECODE_BLOCKS);
  const selectedBlockIdsByQueryBlock: Record<number, string[]> = {};

  for (let queryBlockIndex = firstExplicitQueryBlock; queryBlockIndex <= latestQueryBlock; queryBlockIndex += 1) {
    if (fullContext) {
      selectedBlockIdsByQueryBlock[queryBlockIndex] = Array.from(
        { length: queryBlockIndex + 1 },
        (_value, blockIndex) => `b${blockIndex}`,
      );
      continue;
    }

    selectedBlockIdsByQueryBlock[queryBlockIndex] = buildAdaptiveSparseBlockIds(queryBlockIndex, maxSelectedBlocks);
  }

  return {
    layerIndex,
    blockSize: CLIENT_ATTENTION_BLOCK_SIZE,
    topKBlocks: maxSelectedBlocks,
    localWindowBlocks: fullContext ? blockCount : Math.max(0, maxSelectedBlocks - CLIENT_SPARSE_ANCHOR_BLOCKS - 1),
    pinnedBlockIds: ["b0"],
    selectedBlockIdsByQueryBlock,
    denseFallback: true,
  };
}

function buildAdaptiveSparseBlockIds(queryBlockIndex: number, maxSelectedBlocks: number): string[] {
  const visibleBlockCount = queryBlockIndex + 1;
  if (visibleBlockCount <= maxSelectedBlocks) {
    return Array.from({ length: visibleBlockCount }, (_value, blockIndex) => `b${blockIndex}`);
  }
  const anchorBudget = Math.min(CLIENT_SPARSE_ANCHOR_BLOCKS, Math.max(0, maxSelectedBlocks - 2));
  const recentBudget = Math.max(1, maxSelectedBlocks - anchorBudget - 1);
  const recentStart = Math.max(1, queryBlockIndex - recentBudget + 1);
  const anchorEndExclusive = Math.max(1, recentStart);
  const anchorIds = chooseSparseAnchorBlockIndexes(anchorEndExclusive, anchorBudget);
  const recentIds = Array.from(
    { length: queryBlockIndex - recentStart + 1 },
    (_value, offset) => recentStart + offset,
  );
  return uniqueSortedBlockIds([0, ...anchorIds, ...recentIds]).slice(-maxSelectedBlocks);
}

function chooseSparseAnchorBlockIndexes(endExclusive: number, anchorBudget: number): number[] {
  if (anchorBudget <= 0 || endExclusive <= 1) return [];
  const span = endExclusive - 1;
  const anchors: number[] = [];
  for (let anchorIndex = 1; anchorIndex <= anchorBudget; anchorIndex += 1) {
    const blockIndex = Math.max(1, Math.min(endExclusive - 1, Math.floor((span * anchorIndex) / (anchorBudget + 1))));
    anchors.push(blockIndex);
  }
  return anchors;
}

function uniqueSortedBlockIds(blockIndexes: number[]): string[] {
  return [...new Set(blockIndexes)]
    .filter((blockIndex) => blockIndex >= 0)
    .sort((left, right) => left - right)
    .map((blockIndex) => `b${blockIndex}`);
}

function formatQwenChatMessages(
  messages: ChatClientMessage[],
  specialTokens: string[] = [],
  qwenThinkingMode: QwenThinkingMode = "enabled",
): string {
  const qwenMessages = withQwenThinkingDirective(messages, qwenThinkingMode);
  return appendQwenDisabledAssistantPrefill(
    `${qwenMessages.map((message) => `<|im_start|>${formatQwenRole(message.role)}\n${escapeQwenChatContent(message.content, specialTokens)}<|im_end|>\n`).join("")}<|im_start|>assistant\n`,
    qwenThinkingMode,
  );
}

function formatQwenRole(role: ChatClientMessage["role"]): string {
  return role === "system" || role === "assistant" || role === "user" ? role : "user";
}

function escapeQwenChatContent(content: string, specialTokens: string[] = []): string {
  const tokens = [...new Set(["<|im_start|>", "<|im_end|>", ...specialTokens])]
    .filter((token) => token.length > 0)
    .sort((left, right) => right.length - left.length);
  let escaped = content;
  for (const token of tokens) {
    escaped = escaped.replaceAll(token, neutralizeSpecialToken(token));
  }
  return escaped;
}

function neutralizeSpecialToken(token: string): string {
  if (token.length <= 1) return "";
  return `${token.slice(0, -1)} ${token.slice(-1)}`;
}

function withQwenThinkingDirective(
  messages: ChatClientMessage[],
  qwenThinkingMode: QwenThinkingMode,
): ChatClientMessage[] {
  if (qwenThinkingMode !== "disabled") return messages;
  const targetIndex = findLastUserMessageIndex(messages);
  if (targetIndex < 0) return messages;
  return messages.map((message, index) => (
    index === targetIndex
      ? { ...message, content: appendQwenNoThinkDirective(message.content) }
      : message
  ));
}

function findLastUserMessageIndex(messages: ChatClientMessage[]): number {
  for (let index = messages.length - 1; index >= 0; index -= 1) {
    if (messages[index]?.role === "user") return index;
  }
  return -1;
}

function appendQwenNoThinkDirective(content: string): string {
  if (/\/(?:no_)?think\b/i.test(content)) return content;
  return `${content.trimEnd()}\n/no_think`;
}

function appendQwenDisabledAssistantPrefill(prompt: string, qwenThinkingMode: QwenThinkingMode): string {
  if (qwenThinkingMode !== "disabled") return prompt;
  const prefill = `${THINK_START_MARKER}\n\n${THINK_END_MARKER}\n\n`;
  return prompt.endsWith(prefill) ? prompt : `${prompt}${prefill}`;
}

class InclusiveStopSequenceFilter {
  stopped = false;
  private readonly sequences: string[];
  private text = "";
  private emittedLength = 0;

  constructor(sequences: readonly string[] | undefined) {
    this.sequences = [...new Set((sequences ?? [])
      .map((sequence) => sequence.trim())
      .filter(Boolean)
      .map((sequence) => sequence.toLowerCase()))];
  }

  push(text: string): string {
    if (this.stopped || !text) return "";
    this.text += text;
    const stopEnd = this.findStopEnd();
    const emitEnd = stopEnd ?? this.text.length;
    const output = this.text.slice(this.emittedLength, emitEnd);
    this.emittedLength = emitEnd;
    if (stopEnd !== null) this.stopped = true;
    return output;
  }

  private findStopEnd(): number | null {
    if (this.sequences.length === 0) return null;
    const lower = this.text.toLowerCase();
    let best: number | null = null;
    for (const sequence of this.sequences) {
      const index = lower.indexOf(sequence);
      if (index === -1) continue;
      const end = index + sequence.length;
      if (best === null || end < best) best = end;
    }
    return best;
  }
}

class AssistantOutputFilter {
  stopped = false;
  private buffer = "";
  private insideThinking = false;

  push(text: string): string {
    if (this.stopped || !text) return "";
    this.buffer += text;
    return this.drain(false);
  }

  flush(): string {
    if (this.stopped) {
      this.buffer = "";
      return "";
    }
    return this.drain(true);
  }

  private drain(flush: boolean): string {
    let output = "";
    while (this.buffer.length > 0 && !this.stopped) {
      if (this.insideThinking) {
        const endIndex = this.buffer.indexOf(THINK_END_MARKER);
        if (endIndex === -1) {
          const holdback = partialMarkerHoldback(this.buffer, [THINK_END_MARKER]);
          this.buffer = holdback > 0 ? this.buffer.slice(-holdback) : "";
          if (flush) this.buffer = "";
          break;
        }
        this.buffer = this.buffer.slice(endIndex + THINK_END_MARKER.length).replace(/^\s+/, "");
        this.insideThinking = false;
        continue;
      }

      const next = findNextOutputMarker(this.buffer);
      if (!next) {
        const holdback = flush ? 0 : partialMarkerHoldback(this.buffer, OUTPUT_FILTER_MARKERS);
        output += this.buffer.slice(0, this.buffer.length - holdback);
        this.buffer = holdback > 0 ? this.buffer.slice(-holdback) : "";
        break;
      }

      output += this.buffer.slice(0, next.index);
      this.buffer = this.buffer.slice(next.index + next.marker.length);
      if (next.kind === "proof") continue;
      if (next.kind === "think_start") {
        this.insideThinking = true;
        continue;
      }
      if (next.kind === "think_end") {
        this.buffer = this.buffer.replace(/^\s+/, "");
        continue;
      }
      this.stopped = true;
      this.buffer = "";
      break;
    }
    return output;
  }
}

function findNextOutputMarker(text: string): { index: number; marker: string; kind: "proof" | "think_start" | "think_end" | "stop" } | null {
  const markers: Array<{ marker: string; kind: "proof" | "think_start" | "think_end" | "stop" }> = [
    { marker: UNLOCKED_PROOF_MARKER, kind: "proof" },
    { marker: THINK_START_MARKER, kind: "think_start" },
    { marker: THINK_END_MARKER, kind: "think_end" },
    ...ASSISTANT_CONTROL_MARKERS.map((marker) => ({ marker, kind: "proof" as const })),
    ...ASSISTANT_STOP_MARKERS.map((marker) => ({ marker, kind: "stop" as const })),
  ];
  let result: { index: number; marker: string; kind: "proof" | "think_start" | "think_end" | "stop" } | null = null;
  for (const candidate of markers) {
    const index = text.indexOf(candidate.marker);
    if (index === -1) continue;
    if (!result || index < result.index) result = { index, ...candidate };
  }
  return result;
}

function partialMarkerHoldback(text: string, markers: string[]): number {
  const max = Math.min(text.length, Math.max(...markers.map((marker) => marker.length)) - 1);
  for (let length = max; length > 0; length -= 1) {
    const suffix = text.slice(-length);
    if (markers.some((marker) => marker.startsWith(suffix))) return length;
  }
  return 0;
}

class FixtureTokenizer implements UnlockedBrowserTokenizer {
  private readonly words = [
    "control",
    "sparse",
    "memory",
    "schedule",
    "tensor",
    "kv",
    "routing",
    "verify",
    "browser",
    "qkv",
    "layer",
    "prefetch",
    "attention",
    "mlp",
    "proof",
    "ready",
  ];

  encode(text: string, vocabSize: number): number[] {
    const words = text.toLowerCase().match(/[a-z0-9_:-]+/g) ?? ["empty"];
    return words.slice(-16).map((word) => {
      let hash = 0;
      for (const char of word) hash = (hash * 31 + char.charCodeAt(0)) % vocabSize;
      return hash;
    });
  }

  decode(tokenId: number): string {
    return this.words[positiveModulo(tokenId, this.words.length)] ?? "token";
  }

  decodeStreamToken(tokenId: number): string {
    return ` ${this.decode(tokenId)}`;
  }

  formatMessages(messages: ChatClientMessage[], options: QwenFormatOptions = {}): string {
    return formatQwenChatMessages(messages, [], options.qwenThinkingMode);
  }
}

class VocabTokenizer implements UnlockedBrowserTokenizer {
  private readonly tokenToId: Map<string, number>;

  constructor(private readonly tokens: string[], private readonly unknownTokenId: number) {
    this.tokenToId = new Map(tokens.map((token, index) => [token.toLowerCase(), index]));
  }

  encode(text: string, vocabSize: number): number[] {
    const pieces = text.toLowerCase().match(/[a-z0-9_:-]+|\S/g) ?? [];
    const ids = pieces.map((piece) => this.tokenToId.get(piece) ?? this.unknownTokenId);
    return ids.length > 0 ? ids.slice(-32).map((id) => positiveModulo(id, vocabSize)) : [this.unknownTokenId];
  }

  decode(tokenId: number): string {
    return this.tokens[positiveModulo(tokenId, this.tokens.length)] ?? "";
  }

  decodeStreamToken(tokenId: number): string {
    const decoded = this.decode(tokenId);
    return decoded ? ` ${decoded}` : "";
  }

  formatMessages(messages: ChatClientMessage[], options: QwenFormatOptions = {}): string {
    return formatQwenChatMessages(messages, [], options.qwenThinkingMode);
  }
}

class QwenByteLevelBpeTokenizer implements UnlockedBrowserTokenizer {
  private readonly tokenToId: Map<string, number>;
  private readonly mergeRanks: Map<string, number>;
  private readonly specialTokens: string[];
  private readonly byteEncoder = makeByteEncoder();
  private readonly byteDecoder: Map<string, number>;

  constructor(
    private readonly tokens: string[],
    merges: Array<[string, string]>,
    private readonly unknownTokenId: number,
    specialTokens: string[],
    private readonly chatTemplate: string | undefined,
  ) {
    this.tokenToId = new Map(tokens.map((token, index) => [token, index]));
    this.mergeRanks = new Map(merges.map(([left, right], index) => [`${left}\u0000${right}`, index]));
    this.specialTokens = [...specialTokens].filter((token) => this.tokenToId.has(token)).sort((a, b) => b.length - a.length);
    this.byteDecoder = new Map([...this.byteEncoder.entries()].map(([byte, char]) => [char, byte]));
  }

  encode(text: string, vocabSize: number): number[] {
    const ids: number[] = [];
    for (const part of this.splitSpecialTokens(text.normalize("NFC"))) {
      if (part.special) {
        ids.push(this.tokenToId.get(part.text) ?? this.unknownTokenId);
        continue;
      }
      for (const piece of preTokenize(part.text)) {
        const bytePiece = this.byteEncode(piece);
        for (const token of this.applyBpe(bytePiece)) {
          ids.push(this.tokenToId.get(token) ?? this.unknownTokenId);
        }
      }
    }
    return ids.length > 0 ? ids.map((id) => positiveModulo(id, vocabSize)) : [this.unknownTokenId];
  }

  decode(tokenId: number): string {
    const token = this.tokens[positiveModulo(tokenId, this.tokens.length)] ?? "";
    if (this.specialTokens.includes(token)) return token;
    return new TextDecoder().decode(new Uint8Array(this.tokenToBytes(token)));
  }

  decodeStreamToken(tokenId: number): string {
    return this.decode(tokenId);
  }

  createStreamDecoder(): UnlockedBrowserTokenStreamDecoder {
    const decoder = new TextDecoder();
    return {
      decodeToken: (tokenId: number) => {
        const token = this.tokens[positiveModulo(tokenId, this.tokens.length)] ?? "";
        if (this.specialTokens.includes(token)) return `${decoder.decode()}${token}`;
        return decoder.decode(new Uint8Array(this.tokenToBytes(token)), { stream: true });
      },
      flush: () => decoder.decode(),
    };
  }

  formatMessages(messages: ChatClientMessage[], options: QwenFormatOptions = {}): string {
    return formatMessagesFromTemplate(messages, this.chatTemplate, this.specialTokens, options.qwenThinkingMode);
  }

  private tokenToBytes(token: string): number[] {
    const bytes: number[] = [];
    for (const char of token) {
      const byte = this.byteDecoder.get(char);
      if (byte !== undefined) bytes.push(byte);
    }
    return bytes;
  }

  private splitSpecialTokens(text: string): Array<{ text: string; special: boolean }> {
    if (this.specialTokens.length === 0) return [{ text, special: false }];
    const parts: Array<{ text: string; special: boolean }> = [];
    let index = 0;
    while (index < text.length) {
      const special = this.specialTokens.find((token) => text.startsWith(token, index));
      if (special) {
        parts.push({ text: special, special: true });
        index += special.length;
        continue;
      }
      let nextSpecialIndex = text.length;
      for (const token of this.specialTokens) {
        const found = text.indexOf(token, index + 1);
        if (found !== -1 && found < nextSpecialIndex) nextSpecialIndex = found;
      }
      parts.push({ text: text.slice(index, nextSpecialIndex), special: false });
      index = nextSpecialIndex;
    }
    return parts.filter((part) => part.text.length > 0);
  }

  private byteEncode(text: string): string {
    return [...new TextEncoder().encode(text)].map((byte) => this.byteEncoder.get(byte) ?? "").join("");
  }

  private applyBpe(bytePiece: string): string[] {
    if (bytePiece.length <= 1) return bytePiece ? [bytePiece] : [];
    let word = Array.from(bytePiece);
    while (word.length > 1) {
      let bestIndex = -1;
      let bestRank = Number.POSITIVE_INFINITY;
      for (let index = 0; index < word.length - 1; index += 1) {
        const rank = this.mergeRanks.get(`${word[index]}\u0000${word[index + 1]}`);
        if (rank !== undefined && rank < bestRank) {
          bestRank = rank;
          bestIndex = index;
        }
      }
      if (bestIndex === -1) break;
      word = [
        ...word.slice(0, bestIndex),
        `${word[bestIndex]}${word[bestIndex + 1]}`,
        ...word.slice(bestIndex + 2),
      ];
    }
    return word;
  }
}

function preTokenize(text: string): string[] {
  return text.match(/'s|'t|'re|'ve|'m|'ll|'d|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+/giu) ?? [];
}

function makeByteEncoder(): Map<number, string> {
  const bytes: number[] = [];
  for (let byte = 33; byte <= 126; byte += 1) bytes.push(byte);
  for (let byte = 161; byte <= 172; byte += 1) bytes.push(byte);
  for (let byte = 174; byte <= 255; byte += 1) bytes.push(byte);
  const chars = [...bytes];
  let next = 0;
  for (let byte = 0; byte <= 255; byte += 1) {
    if (!bytes.includes(byte)) {
      bytes.push(byte);
      chars.push(256 + next);
      next += 1;
    }
  }
  return new Map(bytes.map((byte, index) => [byte, String.fromCodePoint(chars[index] as number)]));
}

function formatMessagesFromTemplate(
  messages: ChatClientMessage[],
  chatTemplate: string | undefined,
  specialTokens: string[] = [],
  qwenThinkingMode: QwenThinkingMode = "enabled",
): string {
  const qwenMessages = withQwenThinkingDirective(messages, qwenThinkingMode);
  const template = chatTemplate;
  if (!template?.trim()) return appendQwenDisabledAssistantPrefill(formatQwenChatMessages(qwenMessages, specialTokens, "enabled"), qwenThinkingMode);
  const rendered = renderSimpleChatTemplate(qwenMessages, template, specialTokens);
  if (rendered !== undefined) return appendQwenDisabledAssistantPrefill(rendered, qwenThinkingMode);
  if (template.includes("<|im_start|>") && template.includes("<|im_end|>") && template.includes("messages")) {
    return appendQwenDisabledAssistantPrefill(formatQwenChatMessages(qwenMessages, specialTokens, "enabled"), qwenThinkingMode);
  }
  return appendQwenDisabledAssistantPrefill(formatQwenChatMessages(qwenMessages, specialTokens, "enabled"), qwenThinkingMode);
}

function renderSimpleChatTemplate(
  messages: ChatClientMessage[],
  template: string,
  specialTokens: string[],
): string | undefined {
  const startMarker = "{{#messages}}";
  const endMarker = "{{/messages}}";
  const start = template.indexOf(startMarker);
  const end = template.indexOf(endMarker);
  if (start === -1 || end === -1 || end < start) return undefined;
  const prefix = template.slice(0, start);
  const messageTemplate = template.slice(start + startMarker.length, end);
  const suffix = template.slice(end + endMarker.length);
  return `${prefix}${messages.map((message) => renderMessageTemplate(messageTemplate, message, specialTokens)).join("")}${suffix}`;
}

function renderMessageTemplate(template: string, message: ChatClientMessage, specialTokens: string[]): string {
  return template
    .replaceAll("{{role}}", formatQwenRole(message.role))
    .replaceAll("{{content}}", escapeQwenChatContent(message.content, specialTokens));
}

async function verifyTextSha256(value: string, expected: string, label: string): Promise<void> {
  await verifyArrayBufferSha256(new TextEncoder().encode(value).buffer, expected, label);
}

async function verifyArrayBufferSha256(value: ArrayBuffer, expected: string, label: string): Promise<void> {
  const actual = await computeArrayBufferSha256(value, label);
  verifySha256Digest(actual, expected, label);
}

async function verifyShardSha256(
  url: string,
  value: ArrayBuffer,
  expected: string,
  label: string,
  digestCache: Map<string, Promise<string>>,
): Promise<void> {
  const cached = digestCache.get(url);
  const digestPromise = cached ?? computeArrayBufferSha256(value, label);
  if (!cached) digestCache.set(url, digestPromise);
  const actual = await digestPromise;
  verifySha256Digest(actual, expected, label);
}

async function computeArrayBufferSha256(value: ArrayBuffer, label: string): Promise<string> {
  const digest = await globalThis.crypto.subtle.digest("SHA-256", value);
  return [...new Uint8Array(digest)].map((byte) => byte.toString(16).padStart(2, "0")).join("");
}

function verifySha256Digest(actual: string, expected: string, label: string): void {
  if (!/^[a-fA-F0-9]{64}$/.test(expected)) {
    throw new Error(`${label} must be a 64-character hexadecimal SHA-256 digest.`);
  }
  if (actual !== expected.toLowerCase()) {
    throw new Error(`Unlocked browser transformer ${label} does not match expected SHA-256.`);
  }
}

function positiveModulo(value: number, modulus: number): number {
  return ((value % modulus) + modulus) % modulus;
}

function float16BitsToFloat32(bits: number): number {
  const sign = bits & 0x8000 ? -1 : 1;
  const exponent = (bits >> 10) & 0x1f;
  const mantissa = bits & 0x03ff;
  if (exponent === 0) return sign * (mantissa === 0 ? 0 : (mantissa / 1024) * 2 ** -14);
  if (exponent === 0x1f) return mantissa === 0 ? sign * Infinity : Number.NaN;
  return sign * (1 + mantissa / 1024) * 2 ** (exponent - 15);
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

export const __unlockedBrowserTransformerClientTestHooks = {
  normalizeTokenizer,
  formatQwenChatMessages,
  preTokenize,
  buildQwenThinkingSuppressedTokenIds,
  buildDecodeSuppressedTokenIds,
  buildClientLowRankQuerySummary,
  buildClientPolicy,
  buildReusedKvCacheHandle,
  float16BitsToFloat32,
  createAssistantOutputFilter: () => new AssistantOutputFilter(),
};
