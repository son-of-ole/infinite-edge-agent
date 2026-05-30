import {
  getBrowserBackendRegistryEntry,
  type BrowserBackendSelection,
} from "../lib/runtime/backendBroker";

export interface BrowserPreviewBenchmarkMetrics {
  initLoadMs: number;
  warmupMs?: number;
  warmupMode?: "pipeline_preload" | "target_probe";
  warmupBlockingMs?: number;
  warmupUploadedEntries?: number;
  warmupCacheHits?: number;
  residentUploadCount?: number;
  residentCacheHitCount?: number;
  residentReadbackCount?: number;
  prefillMs: number;
  timeToFirstTokenMs: number;
  decodeLatencyMs: number;
  tokensPerSecond: number | null;
  generatedTokens: number;
}

export interface BrowserPrefillChunkMetadata {
  prefillChunkCount?: number;
  prefillChunkSize?: number;
  shapeBucket?: string;
  pipelineCacheKey?: string;
  prefillDispatchTargetMs?: number;
  maxDispatchEstimatedMs?: number;
  prefillChunkDispatch?: "single_dispatch" | "chunked_dispatch";
  prefillChunkReason?: string;
}

export interface BrowserKvPrefetchBenchmarkSummary {
  prefetchStrategy?: string;
  lowRankSummaryRank?: number;
  lowRankQuerySource?: string;
  exactReuseCount?: number;
  predictivePrefetchCount?: number;
  missStallCount?: number;
  noPrefetchCount?: number;
  predictedHotBlockCount?: number;
  prefetchedBlockCount?: number;
  prefetchHitRate?: number;
  prefetchBytes?: number;
  prefetchLatencyMs?: number;
  attentionStallMs?: number;
}

export interface BrowserMemoryRetrievalAuditProof {
  corpusCount: number;
  queryCount: number;
  top1CorrectCount: number;
  reciprocalRankSum: number;
  recallAt1: number;
  mrr: number;
  canonicalQueryCount?: number;
  canonicalTop1CorrectCount?: number;
  canonicalRecallAt1?: number;
  canonicalMrr?: number;
  aliasQueryCount?: number;
  aliasTop1CorrectCount?: number;
  aliasRecallAt1?: number;
  aliasMrr?: number;
  generatedParaphraseQueryCount?: number;
  generatedParaphraseTop1CorrectCount?: number;
  generatedParaphraseRecallAt1?: number;
  generatedParaphraseMrr?: number;
  queryClassBreakdown?: Array<{
    queryClass: "canonical" | "alias" | "generated_paraphrase";
    queryCount: number;
    top1CorrectCount: number;
    recallAt1: number;
    mrr: number;
  }>;
  minTopScoreMargin: number | null;
  meanExpectedHitRank: number | null;
  passed: boolean;
  elapsedMs: number;
  minRequiredRecallAt1: number;
}

export interface BrowserMemoryGroundingProof {
  mode: "seeded_browser_vector_context_rebuild";
  caseId: string;
  corpusCount: number;
  retrievedMemoryIds: string[];
  includedMemoryIds: string[];
  expectedMemoryIds: string[];
  expectedMemoryHitPassed: boolean;
  contextRebuildPassed: boolean;
  answerOnlyExpected: boolean;
  answerOnlyPassed?: boolean;
  contextEstimatedTokens: number;
  retrievalMs: number;
  contextRebuildMs: number;
  retrievalRank?: number | null;
  retrievalScore?: number | null;
  retrievalTopScoreMargin?: number | null;
  retrievalAudit?: BrowserMemoryRetrievalAuditProof;
}

export const MAX_LONG_PROMPT_TARGET_TOKENS = 8192;
export const MAX_LONG_PROMPT_REPEAT = 1024;
export const MAX_LONG_PROMPT_SEED_WORDS = 64;
export const MAX_LONG_PROMPT_SEED_CHARS = 8192;

export interface BrowserPreviewBenchmarkRun {
  promptId: string;
  prompt: string;
  response: string;
  coherent: boolean;
  expectedSubstrings: string[];
  expectedSubstringMatches: string[];
  expectedExact?: string[];
  expectedExactMatches?: Array<{ expected: string; matched: boolean }>;
  metrics: BrowserPreviewBenchmarkMetrics;
  tokenDiagnostics?: {
    promptTokenHeadIds: number[];
    promptTokenTailIds: number[];
    generatedTokenIds: number[];
    generatedTokenTexts: string[];
  };
  prefillChunkCount?: number;
  prefillChunkSize?: number;
  shapeBucket?: string;
  pipelineCacheKey?: string;
  maxDispatchEstimatedMs?: number;
  runtimeTrace: {
    backend: string;
    brokerSelection?: BrowserBackendSelection;
    tensorControl: boolean;
    tspSteps: string[];
    kvPagingEvents: number;
    selectedBlockIds: string[];
  };
  predictive: {
    promptTokenCount: number;
    generatedTokenCount: number;
    selectedBlockCount: number;
    kvPagingEventCount: number;
    tspStepCount: number;
  };
  webGpu: {
    available: boolean;
    requestedBackendPreference: string;
    logitProjectionBackend: string;
    logitProjectionReadbackStrategy?: string;
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
    decodeSubmitCount?: number;
    decodeSubmitCountPerToken?: number;
    decodeDispatchCount?: number;
    decodeDispatchCountPerToken?: number;
    decodeDispatchCountPerLayerPerToken?: number;
    decodeReadbackCount?: number;
    decodeReadbackBytes?: number;
    fullLogitsReadbackCount?: number;
    compactLogitReadbackCount?: number;
    weightUploadBytesDuringDecode?: number;
    weightUploadCountDuringDecode?: number;
    activationUploadBytesDuringDecode?: number;
    activationUploadCountDuringDecode?: number;
    hiddenReadbackCountDuringDecode?: number;
    residentDecodeLayerCount?: number;
    totalDecodeLayerCount?: number;
    residentDecodeLayerCoverage?: number;
    residentFinalHiddenUsedForLogits?: boolean;
    f32ExpansionCountDuringDecode?: number;
    f32ExpansionBytesDuringDecode?: number;
    cpuValidationUsed?: boolean;
    prefillExecutionsDuringDecode?: number;
    prefillCountPerGeneratedToken?: number;
    kvDecodeReused?: boolean;
    fusedPackedQkvLayerCount?: number;
    fusedQkvNormRopeKvAppendLayerCount?: number;
    fusedOneTokenAttentionLayerCount?: number;
    fusedResidualRmsNormLayerCount?: number;
    fusedMlpLayerCount?: number;
    fusedFullLayerCount?: number;
    fusedLayerCoverage?: number;
    cpuFallbackUsed: boolean;
    noCpuFallback: boolean;
    requestedGates: string[];
    passedGates: string[];
    failedGates: string[];
    positiveKernelProof: boolean;
  };
  mtp: {
    mode: "target_only" | "draft_verify" | "none";
    draftModelId?: string | null;
    draftSource?: string;
    latencyDisablePolicy?: string;
    acceptedTokens: number;
    rejectedTokens: number;
    acceptanceRate: number;
    numSpeculativeTokens: number;
    verifiedTokenCount: number;
    targetDecodeCalls: number;
    verifierStrategy: string;
  };
  kvPersistence: {
    enabled: boolean;
    mode: string;
    eventCount: number;
    persistEvents: number;
    hydrateEvents: number;
    reuseEvents: number;
    kvPersistDeferred?: boolean;
    kvPersistCriticalPathMs?: number;
    kvPersistFlushMs?: number;
    kvPersistPendingBlockCount?: number;
    prefetchStrategy?: string;
    lowRankSummaryRank?: number;
    lowRankQuerySource?: string;
    predictedHotBlocks: string[];
    prefetchedBlocks: string[];
    prefetchHitRate?: number;
    prefetchBytes?: number;
    prefetchLatencyMs?: number;
    attentionStallMs?: number;
  };
  memoryGrounding?: BrowserMemoryGroundingProof;
  expectedAnswerOnlyPassed?: boolean;
  generationStopReason?: string | null;
}

export interface BrowserPreviewBenchmarkPayload {
  name: "browser-preview-benchmark";
  schemaVersion: number;
  createdAt: string;
  passed: boolean;
  summary: Record<string, unknown>;
  runs: BrowserPreviewBenchmarkRun[];
}

export const BROWSER_PREVIEW_BENCHMARK_SCHEMA_VERSION = 2;
const PROOF_MARKER = "[unlocked:ssa-kv-tsp]";
const PRODUCTION_SPEED_FLOOR_TOKENS_PER_SECOND = 2;
const PRODUCTION_MIN_LAYER_VISITS_PER_TOKEN = 28;
const STOP_FRAGMENTS = [
  "<|im_end|>",
  "<|endoftext|>",
  "<|end|>",
  "</s>",
];
const LONG_PROMPT_WORD_BANK = [
  "alpha",
  "beta",
  "gamma",
  "delta",
  "runtime",
  "memory",
  "context",
  "tensor",
] as const;

export function buildDeterministicLongPrompt(input: {
  targetTokens?: number | null;
  repeat?: number | null;
  seedText?: string | null;
} = {}): string {
  const seedWords = tokenizePromptSeed(input.seedText);
  const words = seedWords.length > 0 ? seedWords : [...LONG_PROMPT_WORD_BANK];
  const targetTokens = normalizePositiveInteger(input.targetTokens, MAX_LONG_PROMPT_TARGET_TOKENS);
  const repeat = normalizePositiveInteger(input.repeat, MAX_LONG_PROMPT_REPEAT);
  const repeatedWordCount = repeat ? words.length * repeat : words.length;
  const targetWordCount = targetTokens ?? Math.min(repeatedWordCount, MAX_LONG_PROMPT_TARGET_TOKENS);
  return Array.from({ length: targetWordCount }, (_value, index) => words[index % words.length]).join(" ");
}

export function buildDeterministicLongPromptSeed(input: string | null | undefined): string {
  return tokenizePromptSeed(input).join(" ");
}

export function buildBrowserPreviewBenchmarkPayload(input: {
  createdAt: string;
  profile: string;
  runs: BrowserPreviewBenchmarkRun[];
  strictWebGpuRequested: boolean;
  requireKvReuse?: boolean;
  requireKvPredictivePrefetch?: boolean;
  minGeneratedTokens?: number;
  technicalProofOnly?: boolean;
  sourceGitSha?: string | null;
}): BrowserPreviewBenchmarkPayload {
  const coherentResponseCount = input.runs.filter((run) => run.coherent && hasVisibleResponseQuality(run)).length;
  const visibleResponseQualityPassed = input.runs.length > 0 && coherentResponseCount === input.runs.length;
  const visibleResponseQualityRequired = input.technicalProofOnly !== true;
  const runtimeTraceCount = input.runs.filter((run) => Boolean(run.runtimeTrace.backend)).length;
  const tensorControlTraceCount = input.runs.filter((run) => run.runtimeTrace.tensorControl).length;
  const expectedSubstringCheckCount = input.runs.filter((run) => run.expectedSubstrings.length > 0).length;
  const expectedSubstringPassCount = input.runs.filter((run) =>
    run.expectedSubstrings.length > 0
    && run.expectedSubstrings.every((expected) => visibleExpectedSubstringMatches(run.response, expected))
  ).length;
  const expectedSubstringsPassed = expectedSubstringPassCount === expectedSubstringCheckCount;
  const expectedExactMatches = input.runs.flatMap((run) => evaluateExpectedExactMatches(run.response, run.expectedExact ?? []));
  const expectedExactCheckCount = expectedExactMatches.length;
  const expectedExactPassCount = expectedExactMatches.filter((match) => match.matched).length;
  const expectedExactPassed = expectedExactPassCount === expectedExactCheckCount;
  const expectedExactRequired = expectedExactCheckCount > 0;
  const cpuFallbackUsed = input.runs.some((run) => run.webGpu.cpuFallbackUsed);
  const noCpuFallback = input.runs.every((run) => run.webGpu.noCpuFallback);
  const positiveWebGpuKernelProof = input.runs.every((run) => run.webGpu.positiveKernelProof);
  const webGpuFailedGateCount = sum(input.runs.map((run) => run.webGpu.failedGates.length));
  const decodeHotPathFailureCount = input.runs.filter((run) => (
    run.webGpu.cpuValidationUsed === true
    || (run.webGpu.weightUploadBytesDuringDecode ?? 0) > 0
    || (run.webGpu.activationUploadBytesDuringDecode ?? 0) > 0
    || (run.webGpu.hiddenReadbackCountDuringDecode ?? 0) > 0
    || (run.webGpu.f32ExpansionCountDuringDecode ?? 0) > 0
    || (run.webGpu.fullLogitsReadbackCount ?? 0) > 0
    || (run.webGpu.prefillCountPerGeneratedToken ?? 0) > 0
    || run.webGpu.residentFinalHiddenUsedForLogits === false
    || (run.webGpu.residentDecodeLayerCoverage ?? 1) < 1
  )).length;
  const decodeHotPathPassed = decodeHotPathFailureCount === 0;
  const logitProjectionReadbackStrategy = summarizeStringField(input.runs.map((run) => run.webGpu.logitProjectionReadbackStrategy));
  const kvReuseEventCount = sum(input.runs.map((run) => run.kvPersistence.reuseEvents));
  const kvPrefetchSummary = summarizeKvPrefetchMetadata(input.runs.map((run) => run.kvPersistence));
  const memoryGroundingRuns = input.runs
    .map((run) => run.memoryGrounding)
    .filter((proof): proof is BrowserMemoryGroundingProof => Boolean(proof));
  const memoryGroundingRequired = memoryGroundingRuns.length > 0;
  const memoryGroundingCoveragePassed = !memoryGroundingRequired || memoryGroundingRuns.length === input.runs.length;
  const memoryExpectedHitPassed = !memoryGroundingRequired || memoryGroundingRuns.every((proof) => proof.expectedMemoryHitPassed);
  const memoryContextRebuildPassed = !memoryGroundingRequired || memoryGroundingRuns.every((proof) => proof.contextRebuildPassed);
  const memoryAnswerOnlyPassed = !memoryGroundingRequired || input.runs.every((run) =>
    run.memoryGrounding?.answerOnlyExpected === true ? run.expectedAnswerOnlyPassed === true : Boolean(run.memoryGrounding)
  );
  const memoryRetrievalAuditRuns = memoryGroundingRuns
    .map((proof) => proof.retrievalAudit)
    .filter((audit): audit is BrowserMemoryRetrievalAuditProof => Boolean(audit));
  const memoryRetrievalAuditRequired = memoryRetrievalAuditRuns.length > 0;
  const memoryRetrievalAuditPassed = !memoryRetrievalAuditRequired || memoryRetrievalAuditRuns.every((audit) => audit.passed);
  const memoryGeneratedParaphraseQueryCount = maxOrNull(memoryRetrievalAuditRuns.map((audit) => audit.generatedParaphraseQueryCount));
  const memoryGeneratedParaphraseRecallAt1 = minOrNull(memoryRetrievalAuditRuns.map((audit) => audit.generatedParaphraseRecallAt1));
  const memoryGeneratedParaphraseMrr = minOrNull(memoryRetrievalAuditRuns.map((audit) => audit.generatedParaphraseMrr));
  const memoryGeneratedParaphraseRequired = memoryGroundingRuns.some((proof) => proof.caseId === "qa_corpus_v1");
  const memoryGeneratedParaphrasePassed = !memoryGeneratedParaphraseRequired
    || ((memoryGeneratedParaphraseQueryCount ?? 0) > 0 && (memoryGeneratedParaphraseRecallAt1 ?? 0) >= 1);
  const memoryRetrievalRanks = memoryGroundingRuns.map((proof) => proof.retrievalRank).filter(isFiniteNumber);
  const memoryRetrievalTopScoreMargins = memoryGroundingRuns.map((proof) => proof.retrievalTopScoreMargin).filter(isFiniteNumber);
  const memoryGroundingPassed = memoryGroundingCoveragePassed
    && memoryExpectedHitPassed
    && memoryContextRebuildPassed
    && memoryAnswerOnlyPassed
    && memoryRetrievalAuditPassed;
  const stopQualityFailureCount = input.runs.filter((run) => runHasStopQualityFailure(run)).length;
  const stopQualityPassed = stopQualityFailureCount === 0;
  const stopQualityRequired = visibleResponseQualityRequired || memoryGroundingRequired;
  const runawayRepetitionFailureCount = input.runs.filter((run) => runHasRunawayRepetition(run)).length;
  const runawayRepetitionPassed = runawayRepetitionFailureCount === 0;
  const markerOnlyResponseFailureCount = input.runs.filter((run) => runHasMarkerOnlyResponse(run)).length;
  const markerOnlyResponsePassed = markerOnlyResponseFailureCount === 0;
  const mtpSpeculativeTokenCounts = input.runs.map((run) => run.mtp.numSpeculativeTokens).filter(isFiniteNumber);
  const mtpVerifiedTokenCounts = input.runs.map((run) => run.mtp.verifiedTokenCount).filter(isFiniteNumber);
  const mtpTargetDecodeCalls = input.runs.map((run) => run.mtp.targetDecodeCalls).filter(isFiniteNumber);
  const mtpModes = [...new Set(input.runs.map((run) => run.mtp.mode))];
  const strictWebGpuPassed = !input.strictWebGpuRequested || (noCpuFallback && positiveWebGpuKernelProof && webGpuFailedGateCount === 0);
  const minGeneratedTokens = input.minGeneratedTokens ?? 1;
  const minGeneratedTokensPassed = input.runs.every((run) => run.metrics.generatedTokens >= minGeneratedTokens);
  const kvReusePassed = input.requireKvReuse !== true || kvReuseEventCount > 0;
  const prefillChunkMetadata = summarizePrefillChunkMetadata(input.runs);
  const kvPredictivePrefetchPassed = input.requireKvPredictivePrefetch !== true
    || (
      (kvPrefetchSummary.predictivePrefetchCount ?? 0) > 0
      && (kvPrefetchSummary.lowRankSummaryRank ?? 0) > 0
      && (kvPrefetchSummary.predictedHotBlockCount ?? 0) > 0
      && (kvPrefetchSummary.prefetchedBlockCount ?? 0) > 0
    );
  const meanTokensPerSecond = mean(input.runs.map((run) => run.metrics.tokensPerSecond));
  const warmResidentRuns = input.runs.filter(isWarmResidentBenchmarkRun);
  const meanWarmResidentTokensPerSecond = mean(warmResidentRuns.map((run) => run.metrics.tokensPerSecond));
  const productionSpeedTokensPerSecond = meanWarmResidentTokensPerSecond ?? meanTokensPerSecond;
  const productionSpeedMeasurement = meanWarmResidentTokensPerSecond !== null ? "warm_resident" : "all_runs";
  const productionSpeedFloorPassed = productionSpeedTokensPerSecond !== null
    && productionSpeedTokensPerSecond >= PRODUCTION_SPEED_FLOOR_TOKENS_PER_SECOND;
  const generatedTokenCount = sum(input.runs.map((run) => run.metrics.generatedTokens));
  const decodeSubmitCount = sum(input.runs.map((run) => run.webGpu.decodeSubmitCount));
  const decodeDispatchCount = sum(input.runs.map((run) => run.webGpu.decodeDispatchCount));
  const totalDecodeLayerCount = sum(input.runs.map((run) => run.webGpu.totalDecodeLayerCount));
  const fusedPackedQkvLayerCount = sum(input.runs.map((run) => run.webGpu.fusedPackedQkvLayerCount));
  const fusedQkvNormRopeKvAppendLayerCount = sum(input.runs.map((run) => run.webGpu.fusedQkvNormRopeKvAppendLayerCount));
  const fusedOneTokenAttentionLayerCount = sum(input.runs.map((run) => run.webGpu.fusedOneTokenAttentionLayerCount));
  const fusedResidualRmsNormLayerCount = sum(input.runs.map((run) => run.webGpu.fusedResidualRmsNormLayerCount));
  const fusedMlpLayerCount = sum(input.runs.map((run) => run.webGpu.fusedMlpLayerCount));
  const fusedFullLayerCount = sum(input.runs.map((run) => run.webGpu.fusedFullLayerCount));
  const fusedLayerStageHits = fusedPackedQkvLayerCount
    + fusedQkvNormRopeKvAppendLayerCount
    + fusedOneTokenAttentionLayerCount
    + fusedResidualRmsNormLayerCount
    + fusedMlpLayerCount
    + fusedFullLayerCount;
  const fusedLayerCoverage = totalDecodeLayerCount > 0
    ? round(fusedLayerStageHits / Math.max(1, totalDecodeLayerCount * 6))
    : null;
  const layerVisitsPerToken = generatedTokenCount > 0 && totalDecodeLayerCount > 0
    ? totalDecodeLayerCount / generatedTokenCount
    : null;
  const runtimeBackendId = summarizeStringField(input.runs.map((run) => run.runtimeTrace.backend)) ?? "unknown";
  const runtimeBackendEntry = getBrowserBackendRegistryEntry(runtimeBackendId);
  const runtimeBackendRole = runtimeBackendEntry?.productionRole ?? "unknown";
  const brokerSelections = input.runs
    .map((run) => run.runtimeTrace.brokerSelection)
    .filter((selection): selection is BrowserBackendSelection => Boolean(selection));
  const backendBrokerTraceCount = brokerSelections.length;
  const backendBrokerSelectedBackendId = summarizeStringField(brokerSelections.map((selection) => selection.backendId));
  const backendBrokerSelectedModelId = summarizeStringField(brokerSelections.map((selection) => selection.modelId));
  const backendBrokerProductionRole = summarizeStringField(brokerSelections.map((selection) => selection.productionRole));
  const backendBrokerDeployReadyCandidate = brokerSelections.length > 0
    ? brokerSelections.every((selection) => selection.deployReadyCandidate === true)
    : false;
  const backendBrokerReason = summarizeStringField(brokerSelections.map((selection) => selection.reason));
  const backendBrokerProofRequirements = uniqueSorted(brokerSelections.flatMap((selection) => selection.proofRequirements));
  const backendBrokerSelectionPassed = input.runs.length > 0
    && backendBrokerTraceCount === input.runs.length
    && input.runs.every((run) => brokerSelectionMatchesRuntime(run.runtimeTrace.brokerSelection, run.runtimeTrace.backend));
  const kernelLabBackend = runtimeBackendEntry?.productionRole === "research_kernel_lab";
  const productionCandidateBackend = runtimeBackendEntry?.productionRole === "production_candidate";
  const kernelLabCommandBatchingPassed = generatedTokenCount > 0
    && layerVisitsPerToken !== null
    && decodeSubmitCount / generatedTokenCount <= layerVisitsPerToken + 2;
  const v11CommandBatchingPassed = productionCandidateBackend || kernelLabCommandBatchingPassed;
  const directModelFactualProofUsed = !memoryGroundingRequired && expectedSubstringCheckCount > 0;
  const answerQualityEvidencePassed = expectedSubstringCheckCount > 0 || expectedExactRequired || memoryGroundingRequired;
  const productionLayerCoverageRequired = input.technicalProofOnly !== true
    && input.strictWebGpuRequested
    && answerQualityEvidencePassed;
  const productionLayerCoveragePassed = !productionLayerCoverageRequired
    || ((layerVisitsPerToken ?? 0) >= PRODUCTION_MIN_LAYER_VISITS_PER_TOKEN);
  const diagnosticCappedLayerRun = productionLayerCoverageRequired
    && !productionLayerCoveragePassed
    && layerVisitsPerToken !== null
    && layerVisitsPerToken < PRODUCTION_MIN_LAYER_VISITS_PER_TOKEN;
  const sharedProductionQualityPassed = input.technicalProofOnly !== true
    && input.runs.length > 0
    && answerQualityEvidencePassed
    && visibleResponseQualityPassed
    && expectedSubstringsPassed
    && expectedExactPassed
    && runtimeTraceCount === input.runs.length
    && minGeneratedTokensPassed
    && memoryGroundingPassed
    && runawayRepetitionPassed
    && markerOnlyResponsePassed
    && (!stopQualityRequired || stopQualityPassed);
  const kernelLabQualityPassed = sharedProductionQualityPassed
    && kernelLabBackend
    && input.strictWebGpuRequested
    && productionLayerCoveragePassed
    && strictWebGpuPassed
    && kvReusePassed
    && kvPredictivePrefetchPassed
    && decodeHotPathPassed;
  const compiledBackendReadyPassed = sharedProductionQualityPassed
    && productionCandidateBackend
    && backendBrokerSelectionPassed
    && productionSpeedFloorPassed
    && memoryGroundingRequired
    && memoryGroundingPassed
    && expectedExactRequired
    && expectedExactPassed
    && noCpuFallback
    && positiveWebGpuKernelProof
    && !directModelFactualProofUsed;
  const productionQualityPassed = kernelLabQualityPassed || compiledBackendReadyPassed;
  const customKernelLabReadyPassed = kernelLabQualityPassed
    && productionSpeedFloorPassed
    && memoryGroundingRequired
    && memoryGroundingPassed
    && memoryGeneratedParaphrasePassed
    && expectedExactRequired
    && expectedExactPassed
    && kvReusePassed
    && v11CommandBatchingPassed
    && !directModelFactualProofUsed;
  const groundedProductionReadyPassed = customKernelLabReadyPassed || compiledBackendReadyPassed;
  const productionDeployReadyPassed = compiledBackendReadyPassed;
  return {
    name: "browser-preview-benchmark",
    schemaVersion: BROWSER_PREVIEW_BENCHMARK_SCHEMA_VERSION,
    createdAt: input.createdAt,
    passed: input.runs.length > 0
      && (!visibleResponseQualityRequired || visibleResponseQualityPassed)
      && expectedSubstringsPassed
      && expectedExactPassed
      && runtimeTraceCount === input.runs.length
      && strictWebGpuPassed
      && minGeneratedTokensPassed
      && kvReusePassed
      && kvPredictivePrefetchPassed
      && decodeHotPathPassed
      && memoryGroundingPassed
      && runawayRepetitionPassed
      && markerOnlyResponsePassed
      && (!stopQualityRequired || stopQualityPassed),
    summary: {
      profile: input.profile,
      v12ProductionProofSchemaVersion: BROWSER_PREVIEW_BENCHMARK_SCHEMA_VERSION,
      ...(input.sourceGitSha?.trim() ? { v12ProductionProofSourceGitSha: input.sourceGitSha.trim() } : {}),
      memoryQueryMode: memoryGroundingRequired
        ? summarizeStringField(memoryGroundingRuns.map((proof) => proof.mode)) ?? "seeded_browser_vector_context_rebuild"
        : "direct_model_no_memory_retrieval",
      directModelFactualProofUsed,
      memoryGroundingRequired,
      memoryGroundingPassed,
      memoryGroundingCoveragePassed,
      memoryGroundedRunCount: memoryGroundingRuns.length,
      memoryExpectedHitPassed,
      memoryContextRebuildPassed,
      memoryAnswerOnlyPassed,
      memorySeededCorpusCount: maxOrNull(memoryGroundingRuns.map((proof) => proof.corpusCount)),
      memoryRetrievedCount: sum(memoryGroundingRuns.map((proof) => proof.retrievedMemoryIds.length)),
      memoryIncludedCount: sum(memoryGroundingRuns.map((proof) => proof.includedMemoryIds.length)),
      memoryContextEstimatedTokens: maxOrNull(memoryGroundingRuns.map((proof) => proof.contextEstimatedTokens)),
      memoryRetrievalMs: mean(memoryGroundingRuns.map((proof) => proof.retrievalMs)),
      memoryContextRebuildMs: mean(memoryGroundingRuns.map((proof) => proof.contextRebuildMs)),
      memoryRetrievalAuditRequired,
      memoryRetrievalAuditPassed,
      memoryGeneratedParaphraseRequired,
      memoryGeneratedParaphrasePassed,
      memoryRetrievalAuditQueryCount: maxOrNull(memoryRetrievalAuditRuns.map((audit) => audit.queryCount)),
      memoryRetrievalAuditTop1CorrectCount: maxOrNull(memoryRetrievalAuditRuns.map((audit) => audit.top1CorrectCount)),
      memoryRetrievalAuditElapsedMs: maxOrNull(memoryRetrievalAuditRuns.map((audit) => audit.elapsedMs)),
      memoryRecallAt1: minOrNull(memoryRetrievalAuditRuns.map((audit) => audit.recallAt1)),
      memoryMrr: minOrNull(memoryRetrievalAuditRuns.map((audit) => audit.mrr)),
      memoryCanonicalQueryCount: maxOrNull(memoryRetrievalAuditRuns.map((audit) => audit.canonicalQueryCount)),
      memoryCanonicalTop1CorrectCount: maxOrNull(memoryRetrievalAuditRuns.map((audit) => audit.canonicalTop1CorrectCount)),
      memoryCanonicalRecallAt1: minOrNull(memoryRetrievalAuditRuns.map((audit) => audit.canonicalRecallAt1)),
      memoryCanonicalMrr: minOrNull(memoryRetrievalAuditRuns.map((audit) => audit.canonicalMrr)),
      memoryAliasQueryCount: maxOrNull(memoryRetrievalAuditRuns.map((audit) => audit.aliasQueryCount)),
      memoryAliasTop1CorrectCount: maxOrNull(memoryRetrievalAuditRuns.map((audit) => audit.aliasTop1CorrectCount)),
      memoryAliasRecallAt1: minOrNull(memoryRetrievalAuditRuns.map((audit) => audit.aliasRecallAt1)),
      memoryAliasMrr: minOrNull(memoryRetrievalAuditRuns.map((audit) => audit.aliasMrr)),
      memoryGeneratedParaphraseQueryCount,
      memoryGeneratedParaphraseTop1CorrectCount: maxOrNull(memoryRetrievalAuditRuns.map((audit) => audit.generatedParaphraseTop1CorrectCount)),
      memoryGeneratedParaphraseRecallAt1,
      memoryGeneratedParaphraseMrr,
      memoryMinTopScoreMargin: minOrNull(memoryRetrievalAuditRuns.map((audit) => audit.minTopScoreMargin)),
      memoryMeanExpectedHitRank: mean(memoryRetrievalAuditRuns.map((audit) => audit.meanExpectedHitRank)),
      memoryExpectedHitMeanRank: mean(memoryRetrievalRanks),
      memoryExpectedHitMinTopScoreMargin: minOrNull(memoryRetrievalTopScoreMargins),
      promptCount: input.runs.length,
      technicalProofOnly: input.technicalProofOnly === true,
      coherentResponseCount,
      visibleResponseQualityPassed,
      stopQualityPassed,
      stopQualityFailureCount,
      runawayRepetitionPassed,
      runawayRepetitionFailureCount,
      markerOnlyResponsePassed,
      markerOnlyResponseFailureCount,
      generationStopReason: summarizeStringField(input.runs.map((run) => run.generationStopReason ?? undefined)) ?? "unknown",
      expectedSubstringCheckCount,
      expectedSubstringPassCount,
      expectedSubstringsPassed,
      expectedExactCheckCount,
      expectedExactPassCount,
      expectedExactPassed,
      minGeneratedTokens,
      minGeneratedTokensPassed,
      runtimeTraceCount,
      tensorControlTraceCount,
      meanInitLoadMs: mean(input.runs.map((run) => run.metrics.initLoadMs)),
      meanWarmupMs: mean(input.runs.map((run) => run.metrics.warmupMs)),
      meanWarmupBlockingMs: mean(input.runs.map((run) => run.metrics.warmupBlockingMs)),
      residentReadbackCount: sum(input.runs.map((run) => run.metrics.residentReadbackCount)),
      meanPrefillMs: mean(input.runs.map((run) => run.metrics.prefillMs)),
      meanTimeToFirstTokenMs: mean(input.runs.map((run) => run.metrics.timeToFirstTokenMs)),
      meanDecodeLatencyMs: mean(input.runs.map((run) => run.metrics.decodeLatencyMs)),
      meanTokensPerSecond,
      warmResidentRunCount: warmResidentRuns.length,
      meanWarmResidentTokensPerSecond,
      productionSpeedMeasurement,
      productionSpeedTokensPerSecond,
      productionSpeedFloorTokensPerSecond: PRODUCTION_SPEED_FLOOR_TOKENS_PER_SECOND,
      productionSpeedFloorPassed,
      productionLayerCoverageRequired,
      productionLayerCoveragePassed,
      productionLayerVisitsPerToken: layerVisitsPerToken !== null ? round(layerVisitsPerToken) : null,
      productionMinLayerVisitsPerToken: PRODUCTION_MIN_LAYER_VISITS_PER_TOKEN,
      diagnosticCappedLayerRun,
      productionQualityPassed,
      groundedProductionReadyPassed,
      compiledBackendReadyPassed,
      customKernelLabReadyPassed,
      runtimeBackendId,
      runtimeBackendRole,
      backendBrokerTraceCount,
      backendBrokerSelectionPassed,
      backendBrokerSelectedBackendId,
      backendBrokerSelectedModelId,
      backendBrokerProductionRole,
      backendBrokerDeployReadyCandidate,
      backendBrokerReason,
      backendBrokerProofRequirements,
      deployBackendId: productionDeployReadyPassed ? runtimeBackendId : null,
      researchBackendId: runtimeBackendEntry?.productionRole === "research_kernel_lab" ? runtimeBackendId : null,
      productionDeployReadyPassed,
      primarySpeedBottleneck: identifyPrimarySpeedBottleneck(input.runs),
      generatedTokenCount,
      predictiveSelectedBlockCount: sum(input.runs.map((run) => run.predictive.selectedBlockCount)),
      predictiveKvPagingEventCount: sum(input.runs.map((run) => run.predictive.kvPagingEventCount)),
      predictiveTspStepCount: sum(input.runs.map((run) => run.predictive.tspStepCount)),
      webGpuAvailable: input.runs.some((run) => run.webGpu.available),
      strictWebGpuRequested: input.strictWebGpuRequested,
      strictWebGpuPassed,
      cpuFallbackUsed,
      noCpuFallback,
      positiveWebGpuKernelProof,
      webGpuFailedGateCount,
      decodeHotPathPassed,
      decodeHotPathFailureCount,
      ...(logitProjectionReadbackStrategy ? { logitProjectionReadbackStrategy } : {}),
      ...sumWebGpuNumberField(input.runs, "logitProjectionGpuReducedRows"),
      ...sumWebGpuNumberField(input.runs, "logitProjectionReadbackRows"),
      ...sumWebGpuNumberField(input.runs, "logitProjectionReadbackBytes"),
      ...sumWebGpuNumberField(input.runs, "logitProjectionDispatchCount"),
      ...sumWebGpuNumberField(input.runs, "logitProjectionTiles"),
      ...maxWebGpuNumberField(input.runs, "logitProjectionTileRows"),
      decodeSubmitCount,
      decodeSubmitCountPerToken: generatedTokenCount > 0 ? round(decodeSubmitCount / generatedTokenCount) : null,
      v11CommandBatchingPassed,
      ...sumWebGpuNumberField(input.runs, "decodeDispatchCount"),
      decodeDispatchCountPerToken: generatedTokenCount > 0 ? round(decodeDispatchCount / generatedTokenCount) : null,
      decodeDispatchCountPerLayerPerToken: totalDecodeLayerCount > 0 ? round(decodeDispatchCount / totalDecodeLayerCount) : null,
      ...sumWebGpuNumberField(input.runs, "decodeReadbackCount"),
      ...sumWebGpuNumberField(input.runs, "decodeReadbackBytes"),
      ...sumWebGpuNumberField(input.runs, "fullLogitsReadbackCount"),
      ...sumWebGpuNumberField(input.runs, "compactLogitReadbackCount"),
      ...sumWebGpuNumberField(input.runs, "weightUploadBytesDuringDecode"),
      ...sumWebGpuNumberField(input.runs, "weightUploadCountDuringDecode"),
      ...sumWebGpuNumberField(input.runs, "activationUploadBytesDuringDecode"),
      ...sumWebGpuNumberField(input.runs, "activationUploadCountDuringDecode"),
      ...sumWebGpuNumberField(input.runs, "hiddenReadbackCountDuringDecode"),
      ...sumWebGpuNumberField(input.runs, "residentDecodeLayerCount"),
      ...sumWebGpuNumberField(input.runs, "totalDecodeLayerCount"),
      ...minWebGpuNumberField(input.runs, "residentDecodeLayerCoverage"),
      residentFinalHiddenUsedForLogits: input.runs.every((run) => run.webGpu.residentFinalHiddenUsedForLogits === true),
      ...sumWebGpuNumberField(input.runs, "f32ExpansionCountDuringDecode"),
      ...sumWebGpuNumberField(input.runs, "f32ExpansionBytesDuringDecode"),
      cpuValidationUsed: input.runs.some((run) => run.webGpu.cpuValidationUsed === true),
      ...sumWebGpuNumberField(input.runs, "prefillExecutionsDuringDecode"),
      ...maxWebGpuNumberField(input.runs, "prefillCountPerGeneratedToken"),
      kvDecodeReused: input.runs.every((run) => run.webGpu.kvDecodeReused === true),
      fusedPackedQkvLayerCount,
      fusedQkvNormRopeKvAppendLayerCount,
      fusedOneTokenAttentionLayerCount,
      fusedResidualRmsNormLayerCount,
      fusedMlpLayerCount,
      fusedFullLayerCount,
      fusedLayerCoverage,
      fusedStageCounts: {
        packed_qkv_projection: fusedPackedQkvLayerCount,
        qkv_norm_rope_kv_append: fusedQkvNormRopeKvAppendLayerCount,
        one_token_attention: fusedOneTokenAttentionLayerCount,
        residual_rmsnorm: fusedResidualRmsNormLayerCount,
        swiglu_mlp: fusedMlpLayerCount,
        full_layer: fusedFullLayerCount,
      },
      parityRecordCount: 0,
      parityPassedCount: 0,
      parityFailedCount: 0,
      requireKvReuse: input.requireKvReuse === true,
      kvReusePassed,
      requireKvPredictivePrefetch: input.requireKvPredictivePrefetch === true,
      kvPredictivePrefetchPassed,
      mtpMode: mtpModes.length === 1 ? mtpModes[0] ?? "none" : "mixed",
      mtpAcceptanceRate: mean(input.runs.map((run) => run.mtp.acceptanceRate)),
      mtpMaxSpeculativeTokens: mtpSpeculativeTokenCounts.length > 0 ? Math.max(...mtpSpeculativeTokenCounts) : 0,
      mtpMeanSpeculativeTokens: mean(mtpSpeculativeTokenCounts),
      mtpVerifiedTokenCount: sum(mtpVerifiedTokenCounts),
      mtpTargetDecodeCalls: sum(mtpTargetDecodeCalls),
      mtpVerifierStrategy: summarizeStringField(input.runs.map((run) => run.mtp.verifierStrategy)) ?? "none",
      kvPersistenceEventCount: sum(input.runs.map((run) => run.kvPersistence.eventCount)),
      kvPersistEventCount: sum(input.runs.map((run) => run.kvPersistence.persistEvents)),
      kvHydrateEventCount: sum(input.runs.map((run) => run.kvPersistence.hydrateEvents)),
      kvReuseEventCount,
      kvPersistDeferred: input.runs.some((run) => run.kvPersistence.kvPersistDeferred === true),
      kvPersistCriticalPathMs: maxOrNull(input.runs.map((run) => run.kvPersistence.kvPersistCriticalPathMs)),
      kvPersistFlushMs: maxOrNull(input.runs.map((run) => run.kvPersistence.kvPersistFlushMs)),
      kvPersistPendingBlockCount: sum(input.runs.map((run) => run.kvPersistence.kvPersistPendingBlockCount)),
      kvPrefetchStrategy: kvPrefetchSummary.prefetchStrategy ?? "none",
      kvExactReuseRunCount: kvPrefetchSummary.exactReuseCount ?? 0,
      kvPredictivePrefetchRunCount: kvPrefetchSummary.predictivePrefetchCount ?? 0,
      kvMissStallRunCount: kvPrefetchSummary.missStallCount ?? 0,
      kvNoPrefetchRunCount: kvPrefetchSummary.noPrefetchCount ?? 0,
      kvLowRankSummaryRank: kvPrefetchSummary.lowRankSummaryRank ?? null,
      kvLowRankQuerySource: kvPrefetchSummary.lowRankQuerySource ?? "none",
      kvPredictedHotBlockCount: kvPrefetchSummary.predictedHotBlockCount ?? 0,
      kvPrefetchedBlockCount: kvPrefetchSummary.prefetchedBlockCount ?? 0,
      kvPrefetchHitRate: kvPrefetchSummary.prefetchHitRate ?? null,
      kvPrefetchBytes: kvPrefetchSummary.prefetchBytes ?? 0,
      kvPrefetchLatencyMs: kvPrefetchSummary.prefetchLatencyMs ?? 0,
      kvAttentionStallMs: kvPrefetchSummary.attentionStallMs ?? 0,
      ...prefillChunkMetadata,
    },
    runs: input.runs.map((run) => ({
      ...run,
      ...(run.expectedExact?.length
        ? { expectedExactMatches: evaluateExpectedExactMatches(run.response, run.expectedExact) }
        : {}),
    })),
  };
}

export function summarizeKvPrefetchMetadata(
  items: Array<Partial<BrowserKvPrefetchBenchmarkSummary> & {
    predictedHotBlocks?: string[];
    prefetchedBlocks?: string[];
  }>,
): BrowserKvPrefetchBenchmarkSummary {
  const strategies = items
    .map((item) => item.prefetchStrategy?.trim())
    .filter((value): value is string => Boolean(value));
  const ranks = items.map((item) => item.lowRankSummaryRank).filter(isFiniteNumber);
  const querySource = summarizeStringField(items.map((item) => item.lowRankQuerySource));
  const predictedCounts = items.map((item) => item.predictedHotBlockCount ?? item.predictedHotBlocks?.length).filter(isFiniteNumber);
  const prefetchedCounts = items.map((item) => item.prefetchedBlockCount ?? item.prefetchedBlocks?.length).filter(isFiniteNumber);
  const hitRates = items.map((item) => item.prefetchHitRate).filter(isFiniteNumber);
  const bytes = items.map((item) => item.prefetchBytes).filter(isFiniteNumber);
  const latencies = items.map((item) => item.prefetchLatencyMs).filter(isFiniteNumber);
  const stalls = items.map((item) => item.attentionStallMs).filter(isFiniteNumber);
  const prefetchStrategy = summarizeStringField(strategies);
  const prefetchHitRate = mean(hitRates);
  const prefetchLatencyMs = mean(latencies);
  const attentionStallMs = mean(stalls);
  return {
    ...(prefetchStrategy ? { prefetchStrategy } : {}),
    exactReuseCount: strategies.filter((strategy) => strategy === "exact_reuse").length,
    predictivePrefetchCount: strategies.filter((strategy) => strategy === "predictive_prefetch").length,
    missStallCount: strategies.filter((strategy) => strategy === "miss_stall").length,
    noPrefetchCount: strategies.filter((strategy) => strategy === "none").length,
    ...(ranks.length > 0 ? { lowRankSummaryRank: Math.max(...ranks) } : {}),
    ...(querySource ? { lowRankQuerySource: querySource } : {}),
    ...(predictedCounts.length > 0 ? { predictedHotBlockCount: sum(predictedCounts) } : {}),
    ...(prefetchedCounts.length > 0 ? { prefetchedBlockCount: sum(prefetchedCounts) } : {}),
    ...(prefetchHitRate !== null ? { prefetchHitRate } : {}),
    ...(bytes.length > 0 ? { prefetchBytes: sum(bytes) } : {}),
    ...(prefetchLatencyMs !== null ? { prefetchLatencyMs } : {}),
    ...(attentionStallMs !== null ? { attentionStallMs } : {}),
  };
}

export function summarizePrefillChunkMetadata(
  items: BrowserPrefillChunkMetadata[],
): BrowserPrefillChunkMetadata {
  const chunkCounts = items.map((item) => item.prefillChunkCount).filter(isFiniteNumber);
  const chunkSizes = items.map((item) => item.prefillChunkSize).filter(isFiniteNumber);
  const dispatchEstimates = items.map((item) => item.maxDispatchEstimatedMs).filter(isFiniteNumber);
  const dispatchTargets = items.map((item) => item.prefillDispatchTargetMs).filter(isFiniteNumber);
  const shapeBucket = summarizeStringField(items.map((item) => item.shapeBucket));
  const pipelineCacheKey = summarizeStringField(items.map((item) => item.pipelineCacheKey));
  const prefillChunkDispatch = summarizeStringField(items.map((item) => item.prefillChunkDispatch));
  const prefillChunkReason = summarizeStringField(items.map((item) => item.prefillChunkReason));
  return {
    ...(chunkCounts.length ? { prefillChunkCount: Math.max(...chunkCounts) } : {}),
    ...(chunkSizes.length ? { prefillChunkSize: Math.max(...chunkSizes) } : {}),
    ...(shapeBucket ? { shapeBucket } : {}),
    ...(pipelineCacheKey ? { pipelineCacheKey } : {}),
    ...(dispatchTargets.length ? { prefillDispatchTargetMs: Math.max(...dispatchTargets) } : {}),
    ...(dispatchEstimates.length ? { maxDispatchEstimatedMs: Math.max(...dispatchEstimates) } : {}),
    ...(isPrefillChunkDispatch(prefillChunkDispatch) ? { prefillChunkDispatch } : {}),
    ...(prefillChunkReason ? { prefillChunkReason } : {}),
  };
}

function hasVisibleResponseQuality(run: BrowserPreviewBenchmarkRun): boolean {
  const visible = stripProofScaffolding(run.response);
  const words = visible.match(/[A-Za-z0-9]+/g) ?? [];
  const alnumChars = visible.match(/[A-Za-z0-9]/g)?.length ?? 0;
  const dominantWordCount = maxWordFrequency(words);
  const expectedMatched = run.expectedSubstrings.length > 0
    && run.expectedSubstrings.some((expected) => visibleExpectedSubstringMatches(run.response, expected));
  const exactMatched = (run.expectedExact ?? []).some((expected) => visibleExpectedExactMatches(run.response, expected));
  const basicVisibleQuality = run.metrics.generatedTokens > 0
    && visible.length > 0
    && alnumChars / Math.max(1, visible.length) >= 0.25
    && !hasConsecutiveWordRepetition(words, 3)
    && visible !== "**"
    && visible !== "The";
  if (expectedMatched || exactMatched) return basicVisibleQuality;
  return run.metrics.generatedTokens > 0
    && visible.length >= 12
    && words.length >= 3
    && alnumChars / Math.max(1, visible.length) >= 0.35
    && dominantWordCount / Math.max(1, words.length) <= 0.6;
}

function runHasStopQualityFailure(run: BrowserPreviewBenchmarkRun): boolean {
  const visible = stripProofScaffolding(run.response);
  const words = visible.match(/[A-Za-z0-9]+/g) ?? [];
  return runHasRunawayRepetition(run)
    || runHasMarkerOnlyResponse(run)
    || (run.memoryGrounding?.answerOnlyExpected === true && run.expectedAnswerOnlyPassed !== true);
}

function runHasMarkerOnlyResponse(run: BrowserPreviewBenchmarkRun): boolean {
  return run.response.includes(PROOF_MARKER) && stripProofScaffolding(run.response).length === 0;
}

function runHasRunawayRepetition(run: BrowserPreviewBenchmarkRun): boolean {
  const visible = stripProofScaffolding(run.response);
  const words = visible.match(/[A-Za-z0-9]+/g) ?? [];
  if (hasConsecutiveWordRepetition(words, 3)) return true;
  if (words.length >= 6 && maxWordFrequency(words) / Math.max(1, words.length) > 0.7) return true;
  return hasRepeatedSubstringRun(visible);
}

function stripProofScaffolding(value: string): string {
  let stripped = value
    .replace(/<think\b[^>]*>[\s\S]*?<\/think>/gi, "")
    .replaceAll(PROOF_MARKER, "")
    .replaceAll("<think>", "")
    .replaceAll("</think>", "");
  for (const fragment of STOP_FRAGMENTS) {
    stripped = stripped.replaceAll(fragment, "");
  }
  return stripped.trim();
}

function visibleExpectedSubstringMatches(response: string, expected: string): boolean {
  return stripProofScaffolding(response).toLowerCase().includes(expected.toLowerCase());
}

function evaluateExpectedExactMatches(
  response: string,
  expectedExact: string[],
): Array<{ expected: string; matched: boolean }> {
  return expectedExact
    .map((expected) => expected.trim())
    .filter(Boolean)
    .map((expected) => ({
      expected,
      matched: visibleExpectedExactMatches(response, expected),
    }));
}

function visibleExpectedExactMatches(response: string, expected: string): boolean {
  return stripProofScaffolding(response) === expected.trim();
}

function maxWordFrequency(words: string[]): number {
  const counts = new Map<string, number>();
  for (const word of words) {
    const key = word.toLowerCase();
    counts.set(key, (counts.get(key) ?? 0) + 1);
  }
  return Math.max(0, ...counts.values());
}

function hasConsecutiveWordRepetition(words: string[], limit: number): boolean {
  let previous = "";
  let count = 0;
  for (const word of words) {
    const normalized = word.toLowerCase();
    if (normalized && normalized === previous) {
      count += 1;
    } else {
      previous = normalized;
      count = normalized ? 1 : 0;
    }
    if (count >= limit) return true;
  }
  return false;
}

function hasRepeatedSubstringRun(value: string): boolean {
  const normalized = value.toLowerCase().replace(/[^a-z0-9]+/g, "");
  if (normalized.length < 6) return false;
  if (/([a-z0-9])\1{5,}/.test(normalized)) return true;
  const maxWindow = Math.min(24, Math.floor(normalized.length / 3));
  for (let windowSize = 2; windowSize <= maxWindow; windowSize += 1) {
    for (let start = 0; start + windowSize * 3 <= normalized.length; start += 1) {
      const chunk = normalized.slice(start, start + windowSize);
      if (new Set(chunk).size < Math.min(2, chunk.length)) continue;
      if (
        normalized.slice(start + windowSize, start + windowSize * 2) === chunk
        && normalized.slice(start + windowSize * 2, start + windowSize * 3) === chunk
      ) {
        return true;
      }
    }
  }
  return false;
}

function sumWebGpuNumberField(
  runs: BrowserPreviewBenchmarkRun[],
  key: keyof BrowserPreviewBenchmarkRun["webGpu"],
): Record<string, number> {
  const values = runs.map((run) => run.webGpu[key]).filter(isFiniteNumber);
  return values.length > 0 ? { [key]: sum(values) } : {};
}

function maxWebGpuNumberField(
  runs: BrowserPreviewBenchmarkRun[],
  key: keyof BrowserPreviewBenchmarkRun["webGpu"],
): Record<string, number> {
  const values = runs.map((run) => run.webGpu[key]).filter(isFiniteNumber);
  return values.length > 0 ? { [key]: Math.max(...values) } : {};
}

function minWebGpuNumberField(
  runs: BrowserPreviewBenchmarkRun[],
  key: keyof BrowserPreviewBenchmarkRun["webGpu"],
): Record<string, number> {
  const values = runs.map((run) => run.webGpu[key]).filter(isFiniteNumber);
  return values.length > 0 ? { [key]: Math.min(...values) } : {};
}

function mean(values: Array<number | null | undefined>): number | null {
  const finite = values.filter((value): value is number => typeof value === "number" && Number.isFinite(value));
  if (finite.length === 0) return null;
  return round(finite.reduce((total, value) => total + value, 0) / finite.length);
}

function isWarmResidentBenchmarkRun(run: BrowserPreviewBenchmarkRun): boolean {
  const warmupBlockingMs = run.metrics.warmupBlockingMs ?? run.metrics.warmupMs ?? 0;
  return run.metrics.initLoadMs <= 0 && warmupBlockingMs <= 0;
}

function sum(values: Array<number | null | undefined>): number {
  return round(values.filter(isFiniteNumber).reduce((total, value) => total + value, 0));
}

function maxOrNull(values: Array<number | null | undefined>): number | null {
  const finite = values.filter(isFiniteNumber);
  return finite.length > 0 ? round(Math.max(...finite)) : null;
}

function minOrNull(values: Array<number | null | undefined>): number | null {
  const finite = values.filter(isFiniteNumber);
  return finite.length > 0 ? round(Math.min(...finite)) : null;
}

function identifyPrimarySpeedBottleneck(runs: BrowserPreviewBenchmarkRun[]): string {
  if (runs.length === 0) return "no_runs";
  if (sum(runs.map((run) => run.webGpu.fullLogitsReadbackCount)) > 0) return "full_logits_readback";
  if (sum(runs.map((run) => run.webGpu.weightUploadBytesDuringDecode)) > 0) return "weight_upload_during_decode";
  if (sum(runs.map((run) => run.webGpu.activationUploadBytesDuringDecode)) > 0) return "activation_upload_during_decode";
  if (sum(runs.map((run) => run.webGpu.hiddenReadbackCountDuringDecode)) > 0) return "hidden_readback_during_decode";
  if (sum(runs.map((run) => run.webGpu.f32ExpansionCountDuringDecode)) > 0) return "f32_expansion_during_decode";
  if (runs.some((run) => (run.webGpu.prefillCountPerGeneratedToken ?? 0) > 0)) return "prefill_during_decode";
  if (runs.some((run) => run.webGpu.residentFinalHiddenUsedForLogits === false)) return "nonresident_final_hidden_for_logits";
  if (runs.some((run) => (run.webGpu.residentDecodeLayerCoverage ?? 1) < 1)) return "incomplete_resident_decode_coverage";
  const maxSubmitPerToken = maxOrNull(runs.map((run) => (
    isFiniteNumber(run.webGpu.decodeSubmitCountPerToken)
      ? run.webGpu.decodeSubmitCountPerToken
      : run.metrics.generatedTokens > 0
        ? (run.webGpu.decodeSubmitCount ?? 0) / run.metrics.generatedTokens
        : undefined
  )));
  const maxDispatchPerLayerPerToken = maxOrNull(runs.map((run) => (
    isFiniteNumber(run.webGpu.decodeDispatchCountPerLayerPerToken)
      ? run.webGpu.decodeDispatchCountPerLayerPerToken
      : (run.webGpu.totalDecodeLayerCount ?? 0) > 0
        ? (run.webGpu.decodeDispatchCount ?? 0) / (run.webGpu.totalDecodeLayerCount ?? 1)
        : undefined
  )));
  const maxLayerVisitsPerToken = maxOrNull(runs.map((run) => {
    const layerVisits = run.webGpu.totalDecodeLayerCount;
    const tokenCount = run.metrics.generatedTokens;
    return isFiniteNumber(layerVisits) && tokenCount > 0 ? layerVisits / tokenCount : undefined;
  }));
  if (
    maxSubmitPerToken !== null
    && maxLayerVisitsPerToken !== null
    && maxSubmitPerToken > maxLayerVisitsPerToken + 2
  ) {
    return "submit_fragmentation";
  }
  if (maxDispatchPerLayerPerToken !== null && maxDispatchPerLayerPerToken > 1) return "decode_dispatch_fragmentation";
  const meanTps = mean(runs.map((run) => run.metrics.tokensPerSecond));
  if (meanTps !== null && meanTps >= PRODUCTION_SPEED_FLOOR_TOKENS_PER_SECOND) return "passed_speed_floor";
  const fusedCoverages = runs.map((run) => run.webGpu.fusedLayerCoverage).filter(isFiniteNumber);
  if (fusedCoverages.length > 0 && Math.min(...fusedCoverages) < 1) return "unfused_decode_layer_path";
  const buckets = [
    ["init_load", mean(runs.map((run) => run.metrics.initLoadMs))],
    ["warmup_blocking", mean(runs.map((run) => run.metrics.warmupBlockingMs))],
    ["prefill", mean(runs.map((run) => run.metrics.prefillMs))],
    ["decode", mean(runs.map((run) => run.metrics.decodeLatencyMs))],
    ["time_to_first_token", mean(runs.map((run) => run.metrics.timeToFirstTokenMs))],
    ["kv_persist_flush", maxOrNull(runs.map((run) => run.kvPersistence.kvPersistFlushMs))],
  ] as const;
  const [name] = buckets.reduce((largest, candidate) => (
    (candidate[1] ?? -1) > (largest[1] ?? -1) ? candidate : largest
  ), buckets[0]);
  return name;
}

function round(value: number): number {
  return Math.round(value * 1000) / 1000;
}

function tokenizePromptSeed(value: string | null | undefined): string[] {
  const words: string[] = [];
  const source = (value ?? "").slice(0, MAX_LONG_PROMPT_SEED_CHARS).toLowerCase();
  const pattern = /[a-z0-9_:-]+/g;
  let match: RegExpExecArray | null;
  while ((match = pattern.exec(source)) !== null) {
    words.push(match[0]);
    if (words.length >= MAX_LONG_PROMPT_SEED_WORDS) break;
  }
  return words;
}

function normalizePositiveInteger(value: number | null | undefined, max: number): number | null {
  if (typeof value !== "number" || !Number.isFinite(value)) return null;
  const normalized = Math.floor(value);
  return normalized > 0 ? Math.min(normalized, max) : null;
}

function summarizeStringField(values: Array<string | undefined>): string | undefined {
  const unique = [...new Set(values.map((value) => value?.trim()).filter((value): value is string => Boolean(value)))];
  if (unique.length === 0) return undefined;
  if (unique.length === 1) return unique[0];
  return "mixed";
}

function uniqueSorted(values: string[]): string[] {
  return [...new Set(values.map((value) => value.trim()).filter(Boolean))].sort();
}

function brokerSelectionMatchesRuntime(
  selection: BrowserBackendSelection | undefined,
  runtimeBackendId: string,
): boolean {
  if (!selection) return false;
  const registryEntry = getBrowserBackendRegistryEntry(runtimeBackendId);
  if (!registryEntry) return false;
  return selection.backendId === runtimeBackendId
    && selection.productionRole === registryEntry?.productionRole
    && selection.deployReadyCandidate === (registryEntry.productionRole === "production_candidate")
    && selection.proofRequirements.includes("backend_trace");
}

function isFiniteNumber(value: unknown): value is number {
  return typeof value === "number" && Number.isFinite(value);
}

function isPrefillChunkDispatch(value: string | undefined): value is "single_dispatch" | "chunked_dispatch" {
  return value === "single_dispatch" || value === "chunked_dispatch";
}
