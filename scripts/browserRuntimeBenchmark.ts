import { createHash } from "node:crypto";
import { existsSync } from "node:fs";
import { mkdir, readFile, stat, writeFile } from "node:fs/promises";
import { basename, join, resolve } from "node:path";
import { pathToFileURL } from "node:url";
import {
  LOCAL_BROWSER_MTP_DEFAULT_SPECULATIVE_TOKENS,
  LOCAL_BROWSER_MTP_DRAFT_MODEL_ID,
  LOCAL_BROWSER_MTP_MAX_SPECULATIVE_TOKENS,
  UnlockedBrowserTransformerClient,
  type UnlockedBrowserDecodeProof,
  type UnlockedBrowserMtpProof,
  type QwenThinkingMode,
} from "../apps/web/src/lib/llm/unlockedBrowserTransformerClient";
import {
  buildDeterministicLongPrompt,
  buildDeterministicLongPromptSeed,
  MAX_LONG_PROMPT_REPEAT,
  MAX_LONG_PROMPT_SEED_CHARS,
  MAX_LONG_PROMPT_TARGET_TOKENS,
  summarizeKvPrefetchMetadata,
  summarizePrefillChunkMetadata,
  type BrowserPrefillChunkMetadata,
} from "../apps/web/src/bench/browserPreviewBenchmark";
import {
  assertUnlockedFullProfile,
  resolveUnlockedRuntimeProfile,
  type UnlockedRuntimeProfileResolution,
} from "../apps/web/src/lib/runtime/unlockedRuntimeProfile";
import {
  evaluateUnlockedWebGpuCoverageGates,
  readStrictUnlockedWebGpuGatesFromEnv,
  STRICT_UNLOCKED_WEBGPU_GATES,
  summarizeUnlockedWebGpuCoverage,
  type StrictUnlockedWebGpuGate,
  type UnlockedWebGpuCoverageSummary,
} from "./unlockedWebGpuCoverage";

type BackendPreference = "cpu" | "webgpu";
type BenchmarkMode = "generated-fixture" | "configured";

const DEFAULT_STRICT_BROWSER_BENCH_PROMPTS = [
  "What is the capital of Utah? Answer in one clear sentence.",
  "Write two clear sentences about Earth.",
].join("|");
const DEFAULT_STRICT_BROWSER_BENCH_EXPECTED_SUBSTRINGS = "Salt Lake|Earth";
const STRICT_BROWSER_PREVIEW_MIN_GENERATED_TOKENS = 8;
const PRODUCTION_SPEED_FLOOR_TOKENS_PER_SECOND = 2;

interface BrowserRuntimeBenchmarkPrompt {
  id: string;
  text: string;
  expectedSubstrings?: string[];
  expectedExact?: string[];
}

export interface BrowserRuntimeBenchmarkMetrics {
  initLoadMs: number;
  prefillMs: number;
  timeToFirstTokenMs: number;
  decodeLatencyMs: number;
  tokensPerSecond: number | null;
  generatedTokens: number;
}

export interface BrowserRuntimeBenchmarkRun {
  promptId: string;
  prompt: string;
  response: string;
  expectedSubstrings?: string[];
  expectedSubstringMatches?: Array<{ expected: string; matched: boolean }>;
  expectedExact?: string[];
  expectedExactMatches?: Array<{ expected: string; matched: boolean }>;
  metrics: BrowserRuntimeBenchmarkMetrics;
  prefillChunkCount?: number;
  prefillChunkSize?: number;
  shapeBucket?: string;
  pipelineCacheKey?: string;
  prefillDispatchTargetMs?: number;
  maxDispatchEstimatedMs?: number;
  prefillChunkDispatch?: "single_dispatch" | "chunked_dispatch";
  mtp: {
    mode: "target_only" | "draft_verify" | "none";
    draftModelId?: string | null;
    draftSource?: UnlockedBrowserMtpProof["draftSource"];
    latencyDisablePolicy?: UnlockedBrowserMtpProof["latencyDisablePolicy"];
    acceptedTokens: number;
    rejectedTokens: number;
    acceptanceRate: number;
    numSpeculativeTokens?: number;
    verifiedTokenCount?: number;
    targetDecodeCalls?: number;
    committedInputTokens?: number;
    verifierStrategy?: UnlockedBrowserMtpProof["verifierStrategy"];
    disabledReason?: string;
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
  decodePerf?: UnlockedBrowserDecodeProof["decodePerf"];
}

export type BrowserRuntimeBenchmarkMtpAcceleration =
  | {
      mode: "skipped";
      requested: false;
      passed: true;
      reason: "not_requested";
    }
  | {
      mode: "completed";
      requested: true;
      passed: boolean;
      minAcceptanceRate: number;
      minNetSpeedup: number;
      targetOnlyRuns: BrowserRuntimeBenchmarkRun[];
      targetOnlyMeanTokensPerSecond: number | null;
      draftVerifyMeanTokensPerSecond: number | null;
      acceptanceRate: number | null;
      netSpeedupRatio: number | null;
      failedReasons: string[];
    };

export interface BrowserRuntimeBenchmarkThreshold {
  name: "maxInitLoadMs" | "maxTimeToFirstTokenMs" | "minTokensPerSecond";
  threshold: number;
}

export interface BrowserRuntimeBenchmarkGate extends BrowserRuntimeBenchmarkThreshold {
  observed: number | null;
  passed: boolean;
  blocking: boolean;
}

export interface BrowserRuntimeBenchmarkWebGpuGate {
  required: boolean;
  gates: StrictUnlockedWebGpuGate[];
  passed: boolean;
  failedReasons: string[];
}

export interface BrowserRuntimeBenchmarkFatalError {
  message: string;
  stack?: string;
}

export type BrowserPreviewBenchmarkResult =
  | { mode: "skipped"; requested: false; reason: "not_requested" }
  | {
      mode: "completed";
      requested: true;
      url: string;
      passed: boolean;
      summary: Record<string, unknown>;
      runs: BrowserRuntimeBenchmarkRun[];
    }
  | {
      mode: "failed";
      requested: true;
      url: string;
      passed: false;
      reason: string;
      prefillChunkCount?: number;
      prefillChunkSize?: number;
      shapeBucket?: string;
      pipelineCacheKey?: string;
      maxDispatchEstimatedMs?: number;
      prefillChunkReason?: string;
    };

export interface BrowserPreviewPayloadShape {
  passed?: boolean;
  summary?: Record<string, number | string | boolean | null>;
  runs?: BrowserRuntimeBenchmarkRun[];
}

export interface BrowserPreviewPayloadReaderInput {
  url: string;
  timeoutMs: number;
}

export type BrowserPreviewPayloadReader = (
  input: BrowserPreviewPayloadReaderInput,
) => Promise<BrowserPreviewPayloadShape>;

interface BenchmarkArgs {
  manifestPath: string;
  manifestSha256: string;
  modelId: string;
  publicDir: string;
  artifactDir: string;
  browserPreviewUrl: string;
  requireBrowserPreview: boolean;
  browserPreviewTimeoutMs: number;
  browserPreviewMinGeneratedTokens: number;
  browserPreviewRequireKvReuse: boolean;
  browserPreviewRequireKvPredictivePrefetch: boolean;
  browserPreviewMemoryGroundingCase?: string;
  browserPreviewMemoryGroundingCorpusSize?: number;
  browserPreviewMemoryGroundingPromptLimit?: number;
  browserPreviewWebGpuGates: StrictUnlockedWebGpuGate[];
  backendPreference?: BackendPreference;
  qwenThinkingMode: QwenThinkingMode;
  runtimeProfile: UnlockedRuntimeProfileResolution;
  strictThresholds: boolean;
  requireConfigured: boolean;
  requireManifestSha256: boolean;
  requireFullProfile: boolean;
  thresholds: BrowserRuntimeBenchmarkThreshold[];
  webGpuGates: StrictUnlockedWebGpuGate[];
  prompts: BrowserRuntimeBenchmarkPrompt[];
  longPromptTargetTokens?: number;
  longPromptRepeat?: number;
  longPromptSeed?: string;
  strictLongPromptProof: boolean;
  generationTokenBudget?: number;
  mtpEnabled: boolean;
  mtpDraftModelId: string;
  mtpNumSpeculativeTokens: number;
  mtpMinAcceptanceRate: number;
  mtpDisableWhenLatencyWorse: boolean;
  mtpDraftLayerCount: number;
  requireMtpAcceleration: boolean;
  mtpAccelerationMinAcceptanceRate: number;
  mtpAccelerationMinNetSpeedup: number;
}

interface BenchmarkTarget {
  mode: BenchmarkMode;
  manifestUrl: string;
  modelId: string;
  manifestSha256: string;
}

export interface BrowserRuntimeBenchmarkArtifact {
  name: "browser-runtime-bench";
  createdAt: string;
  passed: boolean;
  mode: BenchmarkMode;
  manifestUrl: string;
  modelId: string;
  requestedBackendPreference: string;
  qwenThinkingMode: QwenThinkingMode;
  runtimeProfile: {
    profile: string;
    resolvedCaps: Record<string, number | null>;
    capsActive: Record<string, boolean>;
    runIsCapped: boolean;
  };
  memoryMode: string;
  backendProofs: {
    tensorControl: boolean;
    tspSteps: string[];
    kvPagingEvents: number;
  };
  webGpuCoverage: UnlockedWebGpuCoverageSummary;
  webGpuGate: BrowserRuntimeBenchmarkWebGpuGate;
  runs: BrowserRuntimeBenchmarkRun[];
  thresholds: BrowserRuntimeBenchmarkGate[];
  mtpAcceleration: BrowserRuntimeBenchmarkMtpAcceleration;
  browserPreview: BrowserPreviewBenchmarkResult;
  browserPreviewRequired: boolean;
  generationTokenBudgetUsed: number;
  fatalError?: BrowserRuntimeBenchmarkFatalError;
  summary: Record<string, unknown>;
}

const repoRoot = resolve(new URL("..", import.meta.url).pathname);

class BrowserPreviewNeedsBrowserExecutionError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "BrowserPreviewNeedsBrowserExecutionError";
  }
}

if (isMainModule()) {
  await runBrowserRuntimeBenchmarkCli();
}

async function runBrowserRuntimeBenchmarkCli(): Promise<void> {
  const args = readBrowserRuntimeBenchmarkArgs(process.argv.slice(2));
  let target: BenchmarkTarget | undefined;
  try {
    validateStrictBenchmarkArgs(args);
    await installFileFetch();
    target = args.manifestPath
      ? await resolveManifestTarget(args)
      : await createFixtureTarget(args.artifactDir);
    const artifact = await runBrowserRuntimeBenchmark(args, target);
    await writeBenchmarkArtifact(args, artifact);
    printBenchmarkArtifact(artifact);
    if (!artifact.passed) process.exitCode = 1;
  } catch (error) {
    const mode: BenchmarkMode = target?.mode ?? (args.manifestPath ? "configured" : "generated-fixture");
    const artifact = buildBrowserRuntimeBenchmarkFailureArtifact({
      createdAt: new Date().toISOString(),
      mode,
      manifestUrl: target?.manifestUrl ?? (args.manifestPath ? pathToFileURL(resolve(args.manifestPath)).toString() : ""),
      modelId: target?.modelId ?? args.modelId,
      requestedBackendPreference: args.backendPreference ?? "auto",
      qwenThinkingMode: args.qwenThinkingMode,
      runtimeProfile: args.runtimeProfile,
      memoryMode: readMemoryMode(mode),
      thresholds: args.thresholds,
      strictThresholds: args.strictThresholds,
      webGpuGates: args.webGpuGates,
      generationTokenBudgetUsed: args.generationTokenBudget ?? resolveBrowserRuntimeBenchmarkGenerationTokens({ mode }),
      browserPreviewRequired: args.requireBrowserPreview,
      browserPreviewUrl: args.browserPreviewUrl,
      error,
    });
    await writeBenchmarkArtifact(args, artifact);
    printBenchmarkArtifact(artifact);
    console.error(artifact.fatalError?.stack ?? artifact.fatalError?.message ?? "Browser runtime benchmark failed.");
    process.exitCode = 1;
  }
}

function printBenchmarkArtifact(artifact: BrowserRuntimeBenchmarkArtifact): void {
  console.log(`Browser runtime benchmark: ${artifact.passed ? "PASS" : "FAIL"}`);
  console.log(`Mode: ${artifact.mode}`);
  console.log(`Runtime profile: ${artifact.runtimeProfile.profile}`);
  console.log(`Qwen thinking: ${artifact.qwenThinkingMode}`);
  console.log(`Prompts: ${artifact.summary.promptCount}`);
  console.log(`Mean TTFT ms: ${artifact.summary.meanTimeToFirstTokenMs}`);
  console.log(`Mean tokens/sec: ${artifact.summary.meanTokensPerSecond}`);
  console.log(`MTP mode: ${artifact.summary.mtpMode}`);
  console.log(`Browser preview: ${artifact.summary.browserPreviewMode}`);
  console.log(`CPU fallback used: ${artifact.webGpuCoverage.cpuFallbackUsed}`);
  console.log(`Strict WebGPU: ${artifact.webGpuGate.required ? artifact.webGpuGate.passed ? "PASS" : "FAIL" : "not requested"}`);
  if (artifact.fatalError) console.log(`Fatal error: ${artifact.fatalError.message}`);
}

async function runBrowserRuntimeBenchmark(
  args: BenchmarkArgs,
  target: BenchmarkTarget,
): Promise<BrowserRuntimeBenchmarkArtifact> {
  const initStart = performance.now();
  const client = createBenchmarkClient(args, target, args.mtpEnabled);
  await client.init();
  const initLoadMs = elapsed(initStart);
  const generationTokenBudget = args.generationTokenBudget
    ?? resolveBrowserRuntimeBenchmarkGenerationTokens({ mode: target.mode });
  const runs = await runPromptSuite(client, args, initLoadMs, generationTokenBudget);

  if (!client.lastDecodeProof?.tensorControl) {
    throw new Error("Browser runtime benchmark did not produce an unlocked tensor-control proof.");
  }
  const mtpAcceleration = args.requireMtpAcceleration
    ? await measureMtpAcceleration({
        args,
        target,
        draftVerifyRuns: runs,
        generationTokenBudget,
      })
    : { mode: "skipped", requested: false, passed: true, reason: "not_requested" } satisfies BrowserRuntimeBenchmarkMtpAcceleration;

  return buildBrowserRuntimeBenchmarkArtifact({
    createdAt: new Date().toISOString(),
    mode: target.mode,
    manifestUrl: target.manifestUrl,
    modelId: target.modelId,
    requestedBackendPreference: args.backendPreference ?? "auto",
    qwenThinkingMode: args.qwenThinkingMode,
    runtimeProfile: args.runtimeProfile,
    memoryMode: readMemoryMode(target.mode),
    backendProofs: {
      tensorControl: client.lastDecodeProof.tensorControl,
      tspSteps: client.lastDecodeProof.tspSteps,
      kvPagingEvents: client.lastDecodeProof.kvPagingEvents,
    },
    webGpuCoverage: summarizeUnlockedWebGpuCoverage(client.lastDecodeProof),
    webGpuGates: args.webGpuGates,
    runs,
    generationTokenBudgetUsed: generationTokenBudget,
    thresholds: evaluateBrowserRuntimeBenchmarkThresholds(runs, args.thresholds, args.strictThresholds),
    mtpAcceleration,
    browserPreview: args.browserPreviewUrl
        ? await readBrowserPreviewBenchmark({
            url: args.browserPreviewUrl,
            runtimeProfile: args.runtimeProfile.profile,
            prompts: args.prompts,
            ...(args.backendPreference ? { backendPreference: args.backendPreference } : {}),
            webGpuGates: args.browserPreviewWebGpuGates,
            generationTokenBudget,
            timeoutMs: args.browserPreviewTimeoutMs,
            minGeneratedTokens: args.browserPreviewMinGeneratedTokens,
            requireKvReuse: args.browserPreviewRequireKvReuse,
            requireKvPredictivePrefetch: args.browserPreviewRequireKvPredictivePrefetch,
            ...(args.browserPreviewMemoryGroundingCase ? { memoryGroundingCase: args.browserPreviewMemoryGroundingCase } : {}),
            ...(args.browserPreviewMemoryGroundingCorpusSize ? { memoryGroundingCorpusSize: args.browserPreviewMemoryGroundingCorpusSize } : {}),
            ...(args.browserPreviewMemoryGroundingPromptLimit ? { memoryGroundingPromptLimit: args.browserPreviewMemoryGroundingPromptLimit } : {}),
            qwenThinkingMode: args.qwenThinkingMode,
            strictLongPromptProof: args.strictLongPromptProof,
            ...(args.longPromptTargetTokens ? { longPromptTargetTokens: args.longPromptTargetTokens } : {}),
            ...(args.longPromptRepeat ? { longPromptRepeat: args.longPromptRepeat } : {}),
            ...(args.longPromptSeed ? { longPromptSeed: args.longPromptSeed } : {}),
          })
      : args.requireBrowserPreview
        ? {
            mode: "failed",
            requested: true,
            url: "",
            passed: false,
            reason: "browser preview proof is required but no browser preview URL was provided.",
          }
        : { mode: "skipped", requested: false, reason: "not_requested" },
    browserPreviewRequired: args.requireBrowserPreview,
  });
}

function createBenchmarkClient(
  args: BenchmarkArgs,
  target: BenchmarkTarget,
  mtpEnabled: boolean,
): UnlockedBrowserTransformerClient {
  return new UnlockedBrowserTransformerClient({
    modelId: target.modelId,
    manifestPath: target.manifestUrl,
    manifestSha256: target.manifestSha256,
    allowFixtureWeights: false,
    ...(args.backendPreference ? { backendPreference: args.backendPreference } : {}),
    ...(shouldRequireWebGpuForBenchmark(args.webGpuGates) ? { requireWebGpu: true } : {}),
    maxRuntimePromptTokens: args.runtimeProfile.caps.maxRuntimePromptTokens,
    maxRuntimeLayers: args.runtimeProfile.caps.maxRuntimeLayers,
    logitCandidateLimit: args.runtimeProfile.caps.logitCandidateLimit,
    maxGenerationTokens: args.runtimeProfile.caps.maxGenerationTokens,
    qwenThinkingMode: args.qwenThinkingMode,
    strictChunkedPrefill: args.strictLongPromptProof,
    kvPersistence: {
      enabled: true,
      namespace: `browser-runtime-benchmark:${target.mode}`,
      preferOpfs: false,
      maxBlocks: 64,
      maxBytes: 64 * 1024 * 1024,
      clearOnInit: true,
    },
    mtp: {
      enabled: mtpEnabled,
      draftModelId: args.mtpDraftModelId,
      numSpeculativeTokens: args.mtpNumSpeculativeTokens,
      minAcceptanceRate: args.mtpMinAcceptanceRate,
      disableWhenLatencyWorse: args.mtpDisableWhenLatencyWorse,
      draftLayerCount: args.mtpDraftLayerCount,
    },
  });
}

async function runPromptSuite(
  client: UnlockedBrowserTransformerClient,
  args: BenchmarkArgs,
  initLoadMs: number,
  generationTokenBudget: number,
): Promise<BrowserRuntimeBenchmarkRun[]> {
  const runs: BrowserRuntimeBenchmarkRun[] = [];
  for (const prompt of args.prompts) {
    const streamStart = performance.now();
    let firstChunkAt: number | null = null;
    let firstGeneratedAt: number | null = null;
    let endAt = streamStart;
    const chunks: string[] = [];
    for await (const chunk of client.streamChat([{ role: "user", content: prompt.text }], {
      maxTokens: generationTokenBudget,
      ...(prompt.expectedExact?.length ? { stopAfterSequences: prompt.expectedExact } : {}),
    })) {
      const now = performance.now();
      firstChunkAt ??= now;
      if (chunk.length > 0 && firstGeneratedAt === null) firstGeneratedAt = now;
      endAt = now;
      chunks.push(chunk);
    }
    const generatedTokens = Math.max(0, client.lastGeneratedTokenIds.length);
    const timing = calculateBrowserRuntimeRunTiming({
      streamStart,
      firstChunkAt,
      firstGeneratedAt,
      endAt,
      generatedTokens,
    });
    const response = chunks.join("");
    const expectedSubstringMatches = evaluateExpectedSubstringMatches(response, prompt.expectedSubstrings ?? []);
    const expectedExactMatches = evaluateExpectedExactMatches(response, prompt.expectedExact ?? []);
    runs.push({
      promptId: prompt.id,
      prompt: prompt.text,
      response,
      ...(prompt.expectedSubstrings?.length ? { expectedSubstrings: prompt.expectedSubstrings } : {}),
      ...(expectedSubstringMatches.length ? { expectedSubstringMatches } : {}),
      ...(prompt.expectedExact?.length ? { expectedExact: prompt.expectedExact } : {}),
      ...(expectedExactMatches.length ? { expectedExactMatches } : {}),
      metrics: {
        initLoadMs,
        prefillMs: timing.prefillMs,
        timeToFirstTokenMs: timing.timeToFirstTokenMs,
        decodeLatencyMs: timing.decodeLatencyMs,
        generatedTokens,
        tokensPerSecond: timing.tokensPerSecond,
      },
      ...readPrefillChunkMetadata(client.lastDecodeProof),
      ...(client.lastDecodeProof?.decodePerf ? { decodePerf: client.lastDecodeProof.decodePerf } : {}),
      mtp: normalizeMtpProof(client.lastDecodeProof?.mtp),
      kvPersistence: summarizeRuntimeKvPersistence(client.lastDecodeProof?.kvPersistence),
    });
  }
  return runs;
}

export function calculateBrowserRuntimeRunTiming(input: {
  streamStart: number;
  firstChunkAt: number | null;
  firstGeneratedAt: number | null;
  endAt: number;
  generatedTokens: number;
}): Pick<BrowserRuntimeBenchmarkMetrics, "prefillMs" | "timeToFirstTokenMs" | "decodeLatencyMs" | "tokensPerSecond"> {
  const prefillRawMs = input.firstChunkAt === null
    ? Math.max(0, input.endAt - input.streamStart)
    : Math.max(0, input.firstChunkAt - input.streamStart);
  const timeToFirstRawMs = input.firstGeneratedAt === null
    ? prefillRawMs
    : Math.max(0, input.firstGeneratedAt - input.streamStart);
  const postFirstChunkDecodeRawMs = input.firstChunkAt === null
    ? 0
    : Math.max(0, input.endAt - input.firstChunkAt);
  const totalGenerationRawMs = Math.max(0, input.endAt - input.streamStart);
  const throughputWindowMs = Math.max(postFirstChunkDecodeRawMs, totalGenerationRawMs);
  return {
    prefillMs: roundMs(prefillRawMs),
    timeToFirstTokenMs: roundMs(timeToFirstRawMs),
    decodeLatencyMs: roundMs(throughputWindowMs),
    tokensPerSecond: calculateTokensPerSecond(input.generatedTokens, throughputWindowMs),
  };
}

async function measureMtpAcceleration(input: {
  args: BenchmarkArgs;
  target: BenchmarkTarget;
  draftVerifyRuns: BrowserRuntimeBenchmarkRun[];
  generationTokenBudget: number;
}): Promise<BrowserRuntimeBenchmarkMtpAcceleration> {
  const initStart = performance.now();
  const targetOnlyClient = createBenchmarkClient(input.args, input.target, false);
  await targetOnlyClient.init();
  const initLoadMs = elapsed(initStart);
  const targetOnlyRuns = await runPromptSuite(targetOnlyClient, input.args, initLoadMs, input.generationTokenBudget);
  const targetOnlyMeanTokensPerSecond = mean(targetOnlyRuns.map((run) => run.metrics.tokensPerSecond));
  const draftVerifyMeanTokensPerSecond = mean(input.draftVerifyRuns.map((run) => run.metrics.tokensPerSecond));
  const acceptanceRate = mean(input.draftVerifyRuns.map((run) => run.mtp.acceptanceRate));
  const netSpeedupRatio = targetOnlyMeanTokensPerSecond !== null
    && draftVerifyMeanTokensPerSecond !== null
    && targetOnlyMeanTokensPerSecond > 0
      ? roundMetric(draftVerifyMeanTokensPerSecond / targetOnlyMeanTokensPerSecond)
      : null;
  const failedReasons: string[] = [];
  if (summarizeMtpMode(input.draftVerifyRuns) !== "draft_verify") {
    failedReasons.push("primary benchmark did not run in draft_verify mode");
  }
  if (acceptanceRate === null || acceptanceRate < input.args.mtpAccelerationMinAcceptanceRate) {
    failedReasons.push(`MTP acceptance rate ${acceptanceRate} is below ${input.args.mtpAccelerationMinAcceptanceRate}`);
  }
  if (netSpeedupRatio === null || netSpeedupRatio < input.args.mtpAccelerationMinNetSpeedup) {
    failedReasons.push(`MTP net speedup ratio ${netSpeedupRatio} is below ${input.args.mtpAccelerationMinNetSpeedup}`);
  }
  return {
    mode: "completed",
    requested: true,
    passed: failedReasons.length === 0,
    minAcceptanceRate: input.args.mtpAccelerationMinAcceptanceRate,
    minNetSpeedup: input.args.mtpAccelerationMinNetSpeedup,
    targetOnlyRuns,
    targetOnlyMeanTokensPerSecond,
    draftVerifyMeanTokensPerSecond,
    acceptanceRate,
    netSpeedupRatio,
    failedReasons,
  };
}

export function buildBrowserRuntimeBenchmarkArtifact(input: {
  createdAt: string;
  mode: BenchmarkMode;
  manifestUrl: string;
  modelId: string;
  requestedBackendPreference: string;
  qwenThinkingMode?: QwenThinkingMode;
  runtimeProfile: UnlockedRuntimeProfileResolution;
  memoryMode: string;
  backendProofs: BrowserRuntimeBenchmarkArtifact["backendProofs"];
  webGpuCoverage: UnlockedWebGpuCoverageSummary;
  webGpuGates?: StrictUnlockedWebGpuGate[];
  runs: BrowserRuntimeBenchmarkRun[];
  thresholds: BrowserRuntimeBenchmarkGate[];
  mtpAcceleration: BrowserRuntimeBenchmarkMtpAcceleration;
  browserPreview: BrowserPreviewBenchmarkResult;
  browserPreviewRequired?: boolean;
  generationTokenBudgetUsed?: number;
  fatalError?: BrowserRuntimeBenchmarkFatalError;
  prefillChunkMetadata?: BrowserPrefillChunkMetadata;
}): BrowserRuntimeBenchmarkArtifact {
  const meanInitLoadMs = mean(input.runs.map((run) => run.metrics.initLoadMs));
  const meanPrefillMs = mean(input.runs.map((run) => run.metrics.prefillMs));
  const meanTimeToFirstTokenMs = mean(input.runs.map((run) => run.metrics.timeToFirstTokenMs));
  const meanDecodeLatencyMs = mean(input.runs.map((run) => run.metrics.decodeLatencyMs));
  const meanTokensPerSecond = mean(input.runs.map((run) => run.metrics.tokensPerSecond));
  const warmResidentRuns = input.runs.filter(isWarmResidentBenchmarkRun);
  const meanWarmResidentTokensPerSecond = mean(warmResidentRuns.map((run) => run.metrics.tokensPerSecond));
  const productionSpeedTokensPerSecond = meanWarmResidentTokensPerSecond ?? meanTokensPerSecond;
  const productionSpeedMeasurement = meanWarmResidentTokensPerSecond !== null ? "warm_resident" : "all_runs";
  const mtpAcceptanceRate = mean(input.runs.map((run) => run.mtp.acceptanceRate));
  const mtpSpeculativeTokenCounts = input.runs.map((run) => run.mtp.numSpeculativeTokens).filter(isFiniteNumber);
  const mtpVerifiedTokenCounts = input.runs.map((run) => run.mtp.verifiedTokenCount).filter(isFiniteNumber);
  const mtpTargetDecodeCalls = input.runs.map((run) => run.mtp.targetDecodeCalls).filter(isFiniteNumber);
  const kvPrefetchSummary = summarizeKvPrefetchMetadata(input.runs.map((run) => run.kvPersistence));
  const blockingFailures = input.thresholds.filter((gate) => gate.blocking && !gate.passed).length;
  const browserPreviewFailed = input.browserPreview.mode === "failed"
    || (input.browserPreview.mode === "completed" && input.browserPreview.passed === false);
  const browserPreviewRequiredFailed = input.browserPreviewRequired === true
    && input.browserPreview.mode !== "completed";
  const browserPreviewSummary = input.browserPreview.mode === "completed" ? input.browserPreview.summary : {};
  const browserPreviewProductionDeployReadyPassed = browserPreviewSummary.productionDeployReadyPassed === true;
  const browserPreviewDeployReadyFailed = input.browserPreviewRequired === true
    && input.browserPreview.mode === "completed"
    && browserPreviewSummary.technicalProofOnly !== true
    && !browserPreviewProductionDeployReadyPassed;
  const mtpAccelerationFailed = input.mtpAcceleration.requested && !input.mtpAcceleration.passed;
  const webGpuGate = evaluateBrowserRuntimeBenchmarkWebGpuGate(input.webGpuCoverage, input.webGpuGates ?? []);
  const webGpuGateFailed = webGpuGate.required && !webGpuGate.passed;
  const expectedSubstringMatches = input.runs.flatMap((run) => run.expectedSubstringMatches ?? []);
  const expectedSubstringCheckCount = expectedSubstringMatches.length;
  const expectedSubstringPassCount = expectedSubstringMatches.filter((match) => match.matched).length;
  const expectedSubstringFailed = expectedSubstringMatches.some((match) => !match.matched);
  const expectedExactMatches = input.runs.flatMap((run) => run.expectedExactMatches ?? []);
  const expectedExactCheckCount = expectedExactMatches.length;
  const expectedExactPassCount = expectedExactMatches.filter((match) => match.matched).length;
  const expectedExactFailed = expectedExactMatches.some((match) => !match.matched);
  const fatalErrorFailed = Boolean(input.fatalError);
  const prefillChunkMetadata = summarizePrefillChunkMetadata([
    ...input.runs,
    ...(input.prefillChunkMetadata ? [input.prefillChunkMetadata] : []),
  ]);
  const generatedTokenCount = sum(input.runs.map((run) => run.metrics.generatedTokens));
  const decodeSubmitCount = sum(input.runs.map((run) => run.decodePerf?.decodeSubmitCount));
  const decodeDispatchCount = sum(input.runs.map((run) => run.decodePerf?.dispatchCount));
  const totalDecodeLayerCount = sum(input.runs.map((run) => run.decodePerf?.totalDecodeLayerCount));
  const fusedPackedQkvLayerCount = sum(input.runs.map((run) => run.decodePerf?.fusedPackedQkvLayerCount));
  const fusedQkvNormRopeKvAppendLayerCount = sum(input.runs.map((run) => run.decodePerf?.fusedQkvNormRopeKvAppendLayerCount));
  const fusedOneTokenAttentionLayerCount = sum(input.runs.map((run) => run.decodePerf?.fusedOneTokenAttentionLayerCount));
  const fusedResidualRmsNormLayerCount = sum(input.runs.map((run) => run.decodePerf?.fusedResidualRmsNormLayerCount));
  const fusedMlpLayerCount = sum(input.runs.map((run) => run.decodePerf?.fusedMlpLayerCount));
  const fusedFullLayerCount = sum(input.runs.map((run) => run.decodePerf?.fusedFullLayerCount));
  const fusedLayerStageHits = fusedPackedQkvLayerCount
    + fusedQkvNormRopeKvAppendLayerCount
    + fusedOneTokenAttentionLayerCount
    + fusedResidualRmsNormLayerCount
    + fusedMlpLayerCount
    + fusedFullLayerCount;
  const fusedLayerCoverage = totalDecodeLayerCount > 0
    ? roundMetric(fusedLayerStageHits / Math.max(1, totalDecodeLayerCount * 6))
    : null;
  const layerVisitsPerToken = generatedTokenCount > 0 && totalDecodeLayerCount > 0
    ? totalDecodeLayerCount / generatedTokenCount
    : null;
  const v11CommandBatchingPassed = generatedTokenCount > 0
    && layerVisitsPerToken !== null
    && decodeSubmitCount / generatedTokenCount <= layerVisitsPerToken + 2;
  return {
    name: "browser-runtime-bench",
    createdAt: input.createdAt,
    passed: !fatalErrorFailed
      && blockingFailures === 0
      && !browserPreviewFailed
      && !browserPreviewRequiredFailed
      && !browserPreviewDeployReadyFailed
      && !mtpAccelerationFailed
      && !webGpuGateFailed
      && !expectedSubstringFailed
      && !expectedExactFailed,
    mode: input.mode,
    manifestUrl: input.manifestUrl,
    modelId: input.modelId,
    requestedBackendPreference: input.requestedBackendPreference,
    qwenThinkingMode: input.qwenThinkingMode ?? "disabled",
    runtimeProfile: {
      profile: input.runtimeProfile.profile,
      resolvedCaps: input.runtimeProfile.caps,
      capsActive: input.runtimeProfile.capStatus,
      runIsCapped: Object.values(input.runtimeProfile.capStatus).some(Boolean),
    },
    memoryMode: input.memoryMode,
    backendProofs: input.backendProofs,
    webGpuCoverage: input.webGpuCoverage,
    webGpuGate,
    runs: input.runs,
    thresholds: input.thresholds,
    mtpAcceleration: input.mtpAcceleration,
    browserPreview: input.browserPreview,
    browserPreviewRequired: input.browserPreviewRequired === true,
    generationTokenBudgetUsed: input.generationTokenBudgetUsed ?? maxGeneratedTokens(input.runs),
    ...(input.fatalError ? { fatalError: input.fatalError } : {}),
    summary: {
      profile: input.runtimeProfile.profile,
      memoryMode: input.memoryMode,
      memoryQueryMode: "direct_model_no_memory_retrieval",
      qwenThinkingMode: input.qwenThinkingMode ?? "disabled",
      promptCount: input.runs.length,
      meanInitLoadMs,
      meanPrefillMs,
      meanTimeToFirstTokenMs,
      meanDecodeLatencyMs,
      meanTokensPerSecond,
      warmResidentRunCount: warmResidentRuns.length,
      meanWarmResidentTokensPerSecond,
      productionSpeedMeasurement,
      productionSpeedTokensPerSecond,
      productionSpeedFloorTokensPerSecond: PRODUCTION_SPEED_FLOOR_TOKENS_PER_SECOND,
      productionSpeedFloorPassed: productionSpeedTokensPerSecond !== null && productionSpeedTokensPerSecond >= PRODUCTION_SPEED_FLOOR_TOKENS_PER_SECOND,
      primarySpeedBottleneck: identifyPrimarySpeedBottleneck(input.runs),
      decodeSubmitCount,
      decodeSubmitCountPerToken: generatedTokenCount > 0 ? roundMetric(decodeSubmitCount / generatedTokenCount) : null,
      v11CommandBatchingPassed,
      decodeDispatchCount,
      decodeDispatchCountPerToken: generatedTokenCount > 0 ? roundMetric(decodeDispatchCount / generatedTokenCount) : null,
      decodeDispatchCountPerLayerPerToken: totalDecodeLayerCount > 0 ? roundMetric(decodeDispatchCount / totalDecodeLayerCount) : null,
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
      generationTokenBudgetUsed: input.generationTokenBudgetUsed ?? maxGeneratedTokens(input.runs),
      browserPreviewMode: input.browserPreview.mode,
      browserPreviewRequested: input.browserPreview.requested,
      browserPreviewPassed: input.browserPreview.mode === "completed" ? input.browserPreview.passed : false,
      browserPreviewRequired: input.browserPreviewRequired === true,
      browserPreviewProductionDeployReadyPassed,
      browserPreviewGroundedProductionReadyPassed: browserPreviewSummary.groundedProductionReadyPassed === true,
      browserPreviewMemoryGroundingPassed: browserPreviewSummary.memoryGroundingPassed === true,
      browserPreviewMemoryGeneratedParaphrasePassed: browserPreviewSummary.memoryGeneratedParaphrasePassed === true,
      ...(input.browserPreview.mode === "skipped" || input.browserPreview.mode === "failed"
        ? { browserPreviewReason: input.browserPreview.reason }
        : {}),
      mtpMode: summarizeMtpMode(input.runs),
      mtpAcceptanceRate,
      mtpMaxSpeculativeTokens: mtpSpeculativeTokenCounts.length > 0 ? Math.max(...mtpSpeculativeTokenCounts) : 0,
      mtpMeanSpeculativeTokens: mean(mtpSpeculativeTokenCounts),
      mtpVerifiedTokenCount: sum(mtpVerifiedTokenCounts),
      mtpTargetDecodeCalls: sum(mtpTargetDecodeCalls),
      mtpVerifierStrategy: summarizeMtpVerifierStrategy(input.runs),
      mtpAccelerationMode: input.mtpAcceleration.mode,
      mtpAccelerationRequested: input.mtpAcceleration.requested,
      mtpAccelerationPassed: input.mtpAcceleration.passed,
      ...(input.mtpAcceleration.mode === "completed"
        ? {
            mtpNetSpeedupRatio: input.mtpAcceleration.netSpeedupRatio,
            mtpTargetOnlyMeanTokensPerSecond: input.mtpAcceleration.targetOnlyMeanTokensPerSecond,
            mtpDraftVerifyMeanTokensPerSecond: input.mtpAcceleration.draftVerifyMeanTokensPerSecond,
            mtpAccelerationMinAcceptanceRate: input.mtpAcceleration.minAcceptanceRate,
            mtpAccelerationMinNetSpeedup: input.mtpAcceleration.minNetSpeedup,
            mtpAccelerationFailureCount: input.mtpAcceleration.failedReasons.length,
          }
        : {}),
      kvPersistenceEventCount: sum(input.runs.map((run) => run.kvPersistence.eventCount)),
      kvPersistEventCount: sum(input.runs.map((run) => run.kvPersistence.persistEvents)),
      kvHydrateEventCount: sum(input.runs.map((run) => run.kvPersistence.hydrateEvents)),
      kvReuseEventCount: sum(input.runs.map((run) => run.kvPersistence.reuseEvents)),
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
      cpuFallbackUsed: input.webGpuCoverage.cpuFallbackUsed,
      logitProjectionBackend: input.webGpuCoverage.logitProjection.backend,
      strictWebGpuRequired: webGpuGate.required,
      strictWebGpuPassed: webGpuGate.passed,
      strictWebGpuGateCount: webGpuGate.gates.length,
      strictWebGpuFailureCount: webGpuGate.failedReasons.length,
      expectedSubstringCheckCount,
      expectedSubstringPassCount,
      expectedSubstringsPassed: !expectedSubstringFailed,
      expectedExactCheckCount,
      expectedExactPassCount,
      expectedExactPassed: !expectedExactFailed,
      fatalError: fatalErrorFailed,
      ...(input.fatalError ? { fatalErrorMessage: input.fatalError.message } : {}),
      ...prefillChunkMetadata,
      blockingThresholdFailures: blockingFailures
        + (browserPreviewFailed || browserPreviewRequiredFailed ? 1 : 0)
        + (browserPreviewDeployReadyFailed ? 1 : 0)
        + (mtpAccelerationFailed ? 1 : 0)
        + (webGpuGateFailed ? 1 : 0)
        + (expectedSubstringFailed ? 1 : 0)
        + (expectedExactFailed ? 1 : 0)
        + (fatalErrorFailed ? 1 : 0),
    },
  };
}

export function buildBrowserRuntimeBenchmarkFailureArtifact(input: {
  createdAt: string;
  mode: BenchmarkMode;
  manifestUrl: string;
  modelId: string;
  requestedBackendPreference: string;
  qwenThinkingMode?: QwenThinkingMode;
  runtimeProfile: UnlockedRuntimeProfileResolution;
  memoryMode: string;
  thresholds: BrowserRuntimeBenchmarkThreshold[];
  strictThresholds?: boolean;
  webGpuGates: StrictUnlockedWebGpuGate[];
  generationTokenBudgetUsed?: number;
  browserPreviewRequired?: boolean;
  browserPreviewUrl?: string;
  error: unknown;
}): BrowserRuntimeBenchmarkArtifact {
  return buildBrowserRuntimeBenchmarkArtifact({
    createdAt: input.createdAt,
    mode: input.mode,
    manifestUrl: input.manifestUrl,
    modelId: input.modelId,
    requestedBackendPreference: input.requestedBackendPreference,
    qwenThinkingMode: input.qwenThinkingMode,
    runtimeProfile: input.runtimeProfile,
    memoryMode: input.memoryMode,
    backendProofs: {
      tensorControl: false,
      tspSteps: [],
      kvPagingEvents: 0,
    },
    webGpuCoverage: makeNoExecutionWebGpuCoverage(),
    webGpuGates: input.webGpuGates,
    runs: [],
    thresholds: evaluateBrowserRuntimeBenchmarkThresholds([], input.thresholds, Boolean(input.strictThresholds)),
    mtpAcceleration: {
      mode: "skipped",
      requested: false,
      passed: true,
      reason: "not_requested",
    },
    browserPreview: input.browserPreviewRequired === true
      ? {
          mode: "failed",
          requested: true,
          url: input.browserPreviewUrl ?? "",
          passed: false,
          reason: input.browserPreviewUrl
            ? "browser preview proof is required but benchmark aborted before browser proof completed."
            : "browser preview proof is required but no browser preview URL was provided.",
          ...readPrefillChunkMetadata(input.error),
        }
      : {
          mode: "skipped",
          requested: false,
          reason: "not_requested",
        },
    browserPreviewRequired: input.browserPreviewRequired === true,
    generationTokenBudgetUsed: input.generationTokenBudgetUsed ?? resolveBrowserRuntimeBenchmarkGenerationTokens({ mode: input.mode }),
    fatalError: normalizeBenchmarkFatalError(input.error),
    prefillChunkMetadata: readPrefillChunkMetadata(input.error),
  });
}

function normalizeBenchmarkFatalError(error: unknown): BrowserRuntimeBenchmarkFatalError {
  if (error instanceof Error) {
    return {
      message: error.message || error.name,
      ...(error.stack ? { stack: error.stack } : {}),
    };
  }
  return {
    message: String(error),
  };
}

function makeNoExecutionWebGpuCoverage(): UnlockedWebGpuCoverageSummary {
  return {
    ...summarizeUnlockedWebGpuCoverage({}),
    cpuFallbackUsed: false,
  };
}

export function evaluateBrowserRuntimeBenchmarkWebGpuGate(
  coverage: UnlockedWebGpuCoverageSummary,
  gates: StrictUnlockedWebGpuGate[],
): BrowserRuntimeBenchmarkWebGpuGate {
  const result = evaluateUnlockedWebGpuCoverageGates(coverage, gates);
  return {
    required: result.gates.length > 0,
    gates: result.gates,
    passed: result.passed,
    failedReasons: result.failedReasons,
  };
}

export function shouldRequireWebGpuForBenchmark(gates: StrictUnlockedWebGpuGate[]): boolean {
  return gates.length > 0;
}

export function parseBrowserRuntimeBenchmarkThresholds(env: Partial<Record<string, string | undefined>>): BrowserRuntimeBenchmarkThreshold[] {
  return [
    readThreshold("maxInitLoadMs", "BROWSER_RUNTIME_BENCH_MAX_INIT_MS", env.BROWSER_RUNTIME_BENCH_MAX_INIT_MS),
    readThreshold("maxTimeToFirstTokenMs", "BROWSER_RUNTIME_BENCH_MAX_TTFT_MS", env.BROWSER_RUNTIME_BENCH_MAX_TTFT_MS),
    readThreshold("minTokensPerSecond", "BROWSER_RUNTIME_BENCH_MIN_TOKENS_PER_SEC", env.BROWSER_RUNTIME_BENCH_MIN_TOKENS_PER_SEC),
  ].filter((threshold): threshold is BrowserRuntimeBenchmarkThreshold => threshold !== null);
}

export function evaluateBrowserRuntimeBenchmarkThresholds(
  runs: BrowserRuntimeBenchmarkRun[],
  thresholds: BrowserRuntimeBenchmarkThreshold[],
  strict: boolean,
): BrowserRuntimeBenchmarkGate[] {
  const observed = {
    maxInitLoadMs: mean(runs.map((run) => run.metrics.initLoadMs)),
    maxTimeToFirstTokenMs: mean(runs.map((run) => run.metrics.timeToFirstTokenMs)),
    minTokensPerSecond: mean(runs.map((run) => run.metrics.tokensPerSecond)),
  };
  return thresholds.map((threshold) => {
    const value = observed[threshold.name];
    const meetsThreshold = value !== null && (threshold.name === "minTokensPerSecond"
      ? value >= threshold.threshold
      : value <= threshold.threshold);
    return {
      ...threshold,
      observed: value,
      passed: strict ? meetsThreshold : true,
      blocking: strict,
    };
  });
}

export async function readBrowserPreviewBenchmark(input: {
  url: string;
  runtimeProfile: string;
  prompts: BrowserRuntimeBenchmarkPrompt[];
  backendPreference?: string;
  webGpuGates?: StrictUnlockedWebGpuGate[];
  generationTokenBudget?: number;
  timeoutMs?: number;
  minGeneratedTokens?: number;
  requireKvReuse?: boolean;
  requireKvPredictivePrefetch?: boolean;
  memoryGroundingCase?: string;
  memoryGroundingCorpusSize?: number;
  memoryGroundingPromptLimit?: number;
  qwenThinkingMode?: QwenThinkingMode;
  strictLongPromptProof?: boolean;
  longPromptTargetTokens?: number;
  longPromptRepeat?: number;
  longPromptSeed?: string;
  browserPayloadReader?: BrowserPreviewPayloadReader;
}): Promise<BrowserPreviewBenchmarkResult> {
  try {
    const url = normalizeBrowserPreviewBenchmarkUrl(input.url);
    url.searchParams.set("profile", input.runtimeProfile);
    url.searchParams.delete("prompt");
    url.searchParams.delete("prompts");
    url.searchParams.delete("expected");
    url.searchParams.delete("expectedJson");
    url.searchParams.delete("expectedExact");
    url.searchParams.delete("expectedExactJson");
    url.searchParams.delete("expectedSubstrings");
    url.searchParams.delete("longPromptTargetTokens");
    url.searchParams.delete("longPromptRepeat");
    url.searchParams.delete("promptSeed");
    const usesCompactLongPrompt = Boolean(input.longPromptTargetTokens || input.longPromptRepeat);
    if (input.longPromptTargetTokens) url.searchParams.set("longPromptTargetTokens", String(input.longPromptTargetTokens));
    if (input.longPromptRepeat) url.searchParams.set("longPromptRepeat", String(input.longPromptRepeat));
    if (usesCompactLongPrompt) {
      if (input.longPromptSeed) url.searchParams.set("promptSeed", input.longPromptSeed);
    }
    const useBrowserMemoryGroundingDefaults = Boolean(input.memoryGroundingCase);
    if (!usesCompactLongPrompt && !useBrowserMemoryGroundingDefaults) {
      for (const prompt of input.prompts) {
        url.searchParams.append("prompt", prompt.text);
      }
    }
    const expectedSubstrings = input.prompts.map((prompt) => prompt.expectedSubstrings ?? []);
    if (!useBrowserMemoryGroundingDefaults && expectedSubstrings.some((expected) => expected.length > 0)) {
      for (const expected of expectedSubstrings) {
        url.searchParams.append("expectedJson", JSON.stringify(expected));
      }
    }
    const expectedExact = input.prompts.map((prompt) => prompt.expectedExact ?? []);
    if (!useBrowserMemoryGroundingDefaults && expectedExact.some((expected) => expected.length > 0)) {
      for (const expected of expectedExact) {
        url.searchParams.append("expectedExactJson", JSON.stringify(expected));
      }
    }
    if (input.backendPreference) url.searchParams.set("backendPreference", input.backendPreference);
    if (input.webGpuGates?.length) url.searchParams.set("webGpuGates", input.webGpuGates.join(","));
    if (input.generationTokenBudget) url.searchParams.set("generationTokens", String(input.generationTokenBudget));
    if (input.timeoutMs) url.searchParams.set("timeoutMs", String(input.timeoutMs));
    if (input.minGeneratedTokens) url.searchParams.set("minGeneratedTokens", String(input.minGeneratedTokens));
    if (input.requireKvReuse) url.searchParams.set("requireKvReuse", "true");
    if (input.requireKvPredictivePrefetch) url.searchParams.set("requireKvPredictivePrefetch", "true");
    if (input.memoryGroundingCase) url.searchParams.set("memoryGrounding", input.memoryGroundingCase);
    if (input.memoryGroundingCorpusSize) url.searchParams.set("memoryCorpusSize", String(input.memoryGroundingCorpusSize));
    if (input.memoryGroundingPromptLimit) url.searchParams.set("memoryPromptLimit", String(input.memoryGroundingPromptLimit));
    if (input.qwenThinkingMode) url.searchParams.set("qwenThinkingMode", input.qwenThinkingMode);
    if (input.strictLongPromptProof) url.searchParams.set("strictLongPrompt", "true");
    const response = await fetch(url);
    if (!response.ok) {
      return {
        mode: "failed",
        requested: true,
        url: input.url,
        passed: false,
        reason: `browser preview benchmark returned HTTP ${response.status}`,
      };
    }
    let parsed: BrowserPreviewPayloadShape;
    try {
      parsed = await readBrowserPreviewPayload(response);
    } catch (error) {
      if (!(error instanceof BrowserPreviewNeedsBrowserExecutionError)) throw error;
      const browserPayloadReader = input.browserPayloadReader ?? readBrowserPreviewPayloadWithBrowser;
      parsed = await browserPayloadReader({
        url: url.toString(),
        timeoutMs: (input.timeoutMs ?? 120_000) + 5_000,
      });
    }
    const invalidReason = validateBrowserPreviewPayload(parsed);
    if (invalidReason) {
      return {
        mode: "failed",
        requested: true,
        url: input.url,
        passed: false,
        reason: invalidReason,
      };
    }
    return {
      mode: "completed",
      requested: true,
      url: input.url,
      passed: parsed.passed,
      summary: parsed.summary,
      runs: parsed.runs,
    };
  } catch (error) {
    return {
      mode: "failed",
      requested: true,
      url: input.url,
      passed: false,
      reason: error instanceof Error ? error.message : String(error),
      ...readPrefillChunkMetadata(error),
    };
  }
}

export function normalizeBrowserPreviewBenchmarkUrl(rawUrl: string): URL {
  const url = new URL(rawUrl);
  const protocol = url.protocol.toLowerCase();
  if ((protocol === "http:" || protocol === "https:")
    && (url.pathname === "" || url.pathname === "/")) {
    url.pathname = "/__bench/browser-runtime";
  }
  return url;
}

async function readBrowserPreviewPayload(response: Response): Promise<BrowserPreviewPayloadShape> {
  const body = await response.text();
  const trimmed = body.trim();
  const contentType = response.headers.get("content-type") ?? "";
  if (contentType.includes("application/json") || trimmed.startsWith("{")) {
    return JSON.parse(trimmed) as BrowserPreviewPayloadShape;
  }
  const embeddedJson = extractBrowserPreviewPayloadJson(body);
  if (!embeddedJson) {
    throw new BrowserPreviewNeedsBrowserExecutionError("browser preview HTML did not contain script#browser-preview-benchmark-payload. Executing the preview route in a real browser is required.");
  }
  return JSON.parse(embeddedJson) as BrowserPreviewPayloadShape;
}

function extractBrowserPreviewPayloadJson(html: string): string | null {
  const scriptMatch = html.match(/<script\b[^>]*\bid=["']browser-preview-benchmark-payload["'][^>]*>([\s\S]*?)<\/script>/i);
  if (scriptMatch?.[1]) return scriptMatch[1].trim();
  return null;
}

async function readBrowserPreviewPayloadWithBrowser(
  input: BrowserPreviewPayloadReaderInput,
): Promise<BrowserPreviewPayloadShape> {
  let playwright: typeof import("playwright-core");
  try {
    playwright = await import("playwright-core");
  } catch (error) {
    throw new Error(`browser preview route returned an SPA shell and needs browser execution, but playwright-core is unavailable. Install dependencies or pass a captured JSON artifact. ${error instanceof Error ? error.message : String(error)}`);
  }

  const browser = await playwright.chromium.launch({
    headless: true,
    ...resolveBrowserLaunchTarget(),
    args: [
      "--enable-unsafe-webgpu",
      "--enable-features=WebGPUDeveloperFeatures",
      "--disable-gpu-sandbox",
      ...(process.env.BROWSER_RUNTIME_BENCH_BROWSER_ARGS ?? "")
        .split(/\s+/)
        .map((arg) => arg.trim())
        .filter(Boolean),
    ],
  });
  try {
    const page = await browser.newPage();
    await page.goto(input.url, { waitUntil: "domcontentloaded", timeout: input.timeoutMs });
    await page.waitForFunction(() => {
      const script = document.querySelector("script#browser-preview-benchmark-payload");
      if (script?.textContent?.trim()) return true;
      const pre = document.querySelector("pre[data-browser-preview-benchmark-json]");
      const text = pre?.textContent?.trim() ?? "";
      return Boolean(text && !/"status"\s*:\s*"running"/.test(text));
    }, undefined, { timeout: input.timeoutMs });
    const payloadText = await page.evaluate(() => {
      const script = document.querySelector("script#browser-preview-benchmark-payload");
      const scriptText = script?.textContent?.trim();
      if (scriptText) return scriptText;
      const pre = document.querySelector("pre[data-browser-preview-benchmark-json]");
      const preText = pre?.textContent?.trim();
      if (preText) return preText;
      throw new Error("browser preview route did not render benchmark JSON.");
    });
    return JSON.parse(payloadText) as BrowserPreviewPayloadShape;
  } finally {
    await browser.close();
  }
}

function resolveBrowserLaunchTarget(): { channel?: string; executablePath?: string } {
  const executablePath = process.env.BROWSER_RUNTIME_BENCH_BROWSER_EXECUTABLE?.trim();
  if (executablePath) return { executablePath };
  const channel = process.env.BROWSER_RUNTIME_BENCH_BROWSER_CHANNEL?.trim() || "chrome";
  return { channel };
}

export function readBrowserRuntimeBenchmarkArgs(
  argv: string[],
  env: Partial<Record<string, string | undefined>> = process.env,
): BenchmarkArgs {
  const parsed = new Map<string, string>();
  const flags = new Set<string>();
  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index];
    if (arg === "--") continue;
    if (!arg.startsWith("--")) continue;
    const value = argv[index + 1];
    if (!value || value.startsWith("--")) {
      flags.add(arg.slice(2));
      continue;
    }
    parsed.set(arg.slice(2), value);
    index += 1;
  }
  const manifestPath = parsed.get("manifest-path") ?? env.VITE_UNLOCKED_MODEL_MANIFEST_PATH ?? "";
  const requireBrowserPreview = flags.has("require-browser-preview") || env.BROWSER_RUNTIME_BENCH_REQUIRE_BROWSER_PREVIEW === "true";
  const runtimeProfile = resolveUnlockedRuntimeProfile({
    VITE_UNLOCKED_RUNTIME_PROFILE: parsed.get("runtime-profile")
      ?? env.BROWSER_RUNTIME_BENCH_PROFILE
      ?? env.VITE_UNLOCKED_RUNTIME_PROFILE
      ?? (env.CI ? "ci" : "full"),
    VITE_UNLOCKED_MAX_RUNTIME_PROMPT_TOKENS: parsed.get("max-runtime-prompt-tokens") ?? env.VITE_UNLOCKED_MAX_RUNTIME_PROMPT_TOKENS,
    VITE_UNLOCKED_MAX_RUNTIME_LAYERS: parsed.get("max-runtime-layers") ?? env.VITE_UNLOCKED_MAX_RUNTIME_LAYERS,
    VITE_UNLOCKED_LOGIT_CANDIDATE_LIMIT: parsed.get("logit-candidate-limit") ?? env.VITE_UNLOCKED_LOGIT_CANDIDATE_LIMIT,
    VITE_UNLOCKED_MAX_GENERATION_TOKENS: parsed.get("max-generation-tokens") ?? env.VITE_UNLOCKED_MAX_GENERATION_TOKENS,
  });
  const webGpuGates = readBenchmarkWebGpuGates(flags, env);
  const browserPreviewWebGpuGates = readBrowserPreviewWebGpuGates(flags, env, webGpuGates);
  const strictRealBrowserPreview = Boolean(manifestPath)
    && requireBrowserPreview
    && browserPreviewWebGpuGates.length > 0;
  const promptValue = parsed.get("prompts")
    ?? env.BROWSER_RUNTIME_BENCH_PROMPTS
    ?? (strictRealBrowserPreview ? DEFAULT_STRICT_BROWSER_BENCH_PROMPTS : undefined);
  const expectedValue = parsed.get("expected-substrings")
    ?? env.BROWSER_RUNTIME_BENCH_EXPECTED_SUBSTRINGS
    ?? (strictRealBrowserPreview ? DEFAULT_STRICT_BROWSER_BENCH_EXPECTED_SUBSTRINGS : undefined);
  const expectedExactValue = parsed.get("expected-exact")
    ?? env.BROWSER_RUNTIME_BENCH_EXPECTED_EXACT;
  const longPromptTargetTokens = parsePositiveInteger(
    parsed.get("long-prompt-target-tokens") ?? env.BROWSER_RUNTIME_BENCH_LONG_PROMPT_TARGET_TOKENS,
    0,
    MAX_LONG_PROMPT_TARGET_TOKENS,
  );
  const longPromptRepeat = parsePositiveInteger(
    parsed.get("long-prompt-repeat") ?? env.BROWSER_RUNTIME_BENCH_LONG_PROMPT_REPEAT,
    0,
    MAX_LONG_PROMPT_REPEAT,
  );
  const longPromptSeed = longPromptTargetTokens || longPromptRepeat
    ? readFirstPromptSeed(promptValue)
    : "";
  const generationTokenBudget = readOptionalGenerationTokenBudget(
    parsed.get("generation-max-tokens") ?? env.BROWSER_RUNTIME_BENCH_MAX_GENERATION_TOKENS,
  );
  return {
    manifestPath,
    manifestSha256: parsed.get("manifest-sha256") ?? env.VITE_UNLOCKED_MODEL_MANIFEST_SHA256 ?? "",
    modelId: parsed.get("model-id") ?? env.VITE_DEFAULT_MODEL ?? "Qwen/Qwen3-0.6B",
    publicDir: resolve(repoRoot, parsed.get("public-dir") ?? "apps/web/public"),
    artifactDir: parsed.get("artifact-dir") ?? env.EVAL_ARTIFACT_DIR ?? ".artifacts/evals",
    browserPreviewUrl: parsed.get("browser-preview-url") ?? env.BROWSER_RUNTIME_BENCH_PREVIEW_URL ?? "",
    requireBrowserPreview,
    browserPreviewTimeoutMs: parsePositiveInteger(
      parsed.get("browser-preview-timeout-ms") ?? env.BROWSER_RUNTIME_BENCH_PREVIEW_TIMEOUT_MS,
      120_000,
    ),
    browserPreviewMinGeneratedTokens: parsePositiveInteger(
      parsed.get("browser-preview-min-generated-tokens") ?? env.BROWSER_RUNTIME_BENCH_PREVIEW_MIN_GENERATED_TOKENS,
      strictRealBrowserPreview ? STRICT_BROWSER_PREVIEW_MIN_GENERATED_TOKENS : 1,
    ),
    browserPreviewRequireKvReuse: flags.has("browser-preview-require-kv-reuse")
      || env.BROWSER_RUNTIME_BENCH_PREVIEW_REQUIRE_KV_REUSE === "true"
      || (strictRealBrowserPreview && env.BROWSER_RUNTIME_BENCH_PREVIEW_REQUIRE_KV_REUSE !== "false"),
    browserPreviewRequireKvPredictivePrefetch: flags.has("browser-preview-require-kv-predictive-prefetch")
      || env.BROWSER_RUNTIME_BENCH_PREVIEW_REQUIRE_KV_PREDICTIVE_PREFETCH === "true",
    ...(parsed.get("browser-preview-memory-grounding") ?? env.BROWSER_RUNTIME_BENCH_PREVIEW_MEMORY_GROUNDING_CASE ?? env.BROWSER_RUNTIME_BENCH_MEMORY_GROUNDING_CASE
      ? { browserPreviewMemoryGroundingCase: parsed.get("browser-preview-memory-grounding") ?? env.BROWSER_RUNTIME_BENCH_PREVIEW_MEMORY_GROUNDING_CASE ?? env.BROWSER_RUNTIME_BENCH_MEMORY_GROUNDING_CASE }
      : {}),
    ...(parsePositiveInteger(
      parsed.get("browser-preview-memory-corpus-size") ?? env.BROWSER_RUNTIME_BENCH_PREVIEW_MEMORY_CORPUS_SIZE ?? env.BROWSER_RUNTIME_BENCH_MEMORY_GROUNDING_CORPUS_SIZE,
      0,
    ) > 0
      ? {
          browserPreviewMemoryGroundingCorpusSize: parsePositiveInteger(
            parsed.get("browser-preview-memory-corpus-size") ?? env.BROWSER_RUNTIME_BENCH_PREVIEW_MEMORY_CORPUS_SIZE ?? env.BROWSER_RUNTIME_BENCH_MEMORY_GROUNDING_CORPUS_SIZE,
            0,
          ),
        }
      : {}),
    ...(parsePositiveInteger(
      parsed.get("browser-preview-memory-prompt-limit")
        ?? env.BROWSER_RUNTIME_BENCH_PREVIEW_MEMORY_PROMPT_LIMIT
        ?? env.BROWSER_RUNTIME_BENCH_MEMORY_GROUNDING_PROMPT_LIMIT,
      0,
    ) > 0
      ? {
          browserPreviewMemoryGroundingPromptLimit: parsePositiveInteger(
            parsed.get("browser-preview-memory-prompt-limit")
              ?? env.BROWSER_RUNTIME_BENCH_PREVIEW_MEMORY_PROMPT_LIMIT
              ?? env.BROWSER_RUNTIME_BENCH_MEMORY_GROUNDING_PROMPT_LIMIT,
            0,
          ),
        }
      : {}),
    browserPreviewWebGpuGates,
    ...(parseBackendPreference(parsed.get("backend-preference") ?? env.VITE_UNLOCKED_BACKEND_PREFERENCE) !== undefined
      ? { backendPreference: parseBackendPreference(parsed.get("backend-preference") ?? env.VITE_UNLOCKED_BACKEND_PREFERENCE) }
      : {}),
    qwenThinkingMode: parseQwenThinkingMode(parsed.get("qwen-thinking-mode") ?? env.VITE_QWEN_THINKING_MODE),
    runtimeProfile,
    strictThresholds: flags.has("strict") || env.BROWSER_RUNTIME_BENCH_STRICT === "true",
    requireConfigured: flags.has("require-configured") || env.RELEASE_REQUIRE_UNLOCKED_MODEL === "true",
    requireManifestSha256: flags.has("require-manifest-sha256") || env.RELEASE_REQUIRE_UNLOCKED_MODEL === "true",
    requireFullProfile: flags.has("require-full-profile") || env.RELEASE_REQUIRE_UNLOCKED_FULL_PROFILE === "true",
    thresholds: parseBrowserRuntimeBenchmarkThresholds(env),
    webGpuGates,
    prompts: readPrompts(promptValue, expectedValue, {
      expectedExactValue,
      longPromptTargetTokens,
      longPromptRepeat,
      longPromptSeed,
    }),
    ...(longPromptTargetTokens ? { longPromptTargetTokens } : {}),
    ...(longPromptRepeat ? { longPromptRepeat } : {}),
    ...(longPromptSeed ? { longPromptSeed } : {}),
    strictLongPromptProof: flags.has("require-long-prompt-proof")
      || env.BROWSER_RUNTIME_BENCH_REQUIRE_LONG_PROMPT_PROOF === "true",
    ...(generationTokenBudget.generationTokenBudget !== undefined
      ? generationTokenBudget
      : strictRealBrowserPreview
        ? { generationTokenBudget: 16 }
        : {}),
    mtpEnabled: !flags.has("mtp-disabled")
      && (flags.has("mtp-enabled") || parsed.get("mtp-enabled") === "true" || env.VITE_MTP_ENABLED === "true"),
    mtpDraftModelId: parsed.get("mtp-draft-model-id") ?? env.VITE_MTP_DRAFT_MODEL_ID ?? LOCAL_BROWSER_MTP_DRAFT_MODEL_ID,
    mtpNumSpeculativeTokens: parseBrowserMtpSpeculativeTokens(parsed.get("mtp-num-speculative-tokens") ?? env.VITE_MTP_NUM_SPECULATIVE_TOKENS),
    mtpMinAcceptanceRate: parseRatio(parsed.get("mtp-min-acceptance-rate") ?? env.VITE_MTP_MIN_ACCEPTANCE_RATE, 0),
    mtpDisableWhenLatencyWorse: (parsed.get("mtp-disable-when-latency-worse") ?? env.VITE_MTP_DISABLE_WHEN_LATENCY_WORSE) !== "false",
    mtpDraftLayerCount: parsePositiveInteger(parsed.get("mtp-draft-layer-count") ?? env.VITE_MTP_DRAFT_LAYER_COUNT, 4),
    requireMtpAcceleration: flags.has("require-mtp-acceleration") || env.BROWSER_RUNTIME_BENCH_REQUIRE_MTP_ACCELERATION === "true",
    mtpAccelerationMinAcceptanceRate: parseRatio(
      parsed.get("mtp-acceleration-min-acceptance-rate") ?? env.BROWSER_RUNTIME_BENCH_MIN_MTP_ACCEPTANCE_RATE,
      0.25,
    ),
    mtpAccelerationMinNetSpeedup: parsePositiveNumber(
      parsed.get("mtp-acceleration-min-net-speedup") ?? env.BROWSER_RUNTIME_BENCH_MIN_MTP_NET_SPEEDUP,
      1.05,
    ),
  };
}

export function resolveBrowserRuntimeBenchmarkGenerationTokens(input: {
  mode: BenchmarkMode;
  explicitValue?: string | undefined;
}): number {
  const explicit = parsePositiveInteger(input.explicitValue, 0);
  if (explicit > 0) return explicit;
  return input.mode === "configured" ? 16 : 4;
}

function validateStrictBenchmarkArgs(args: BenchmarkArgs): void {
  if (args.requireConfigured && !args.manifestPath) {
    throw new Error("Browser runtime benchmark requires --manifest-path or VITE_UNLOCKED_MODEL_MANIFEST_PATH when strict unlocked model mode is enabled.");
  }
  if (args.requireManifestSha256 && !args.manifestSha256) {
    throw new Error("Browser runtime benchmark requires --manifest-sha256 or VITE_UNLOCKED_MODEL_MANIFEST_SHA256 when strict unlocked model mode is enabled.");
  }
  if (args.requireBrowserPreview && !args.browserPreviewUrl) {
    throw new Error("Browser runtime benchmark requires --browser-preview-url or BROWSER_RUNTIME_BENCH_PREVIEW_URL when browser preview proof is required.");
  }
  if (args.requireFullProfile) assertUnlockedFullProfile(args.runtimeProfile);
}

export function readBenchmarkWebGpuGates(
  flags: Set<string>,
  env: NodeJS.ProcessEnv,
): StrictUnlockedWebGpuGate[] {
  const gates = new Set<StrictUnlockedWebGpuGate>(readStrictUnlockedWebGpuGatesFromEnv(env));
  const requireAll = flags.has("require-strict-webgpu")
    || env.BROWSER_RUNTIME_BENCH_REQUIRE_STRICT_WEBGPU === "true"
    || env.RELEASE_REQUIRE_UNLOCKED_STRICT_WEBGPU === "true"
    || env.VITE_REQUIRE_WEBGPU_KERNELS === "true";
  if (requireAll) {
    for (const gate of STRICT_UNLOCKED_WEBGPU_GATES) gates.add(gate);
  }
  if (flags.has("require-webgpu-mlp") || env.BROWSER_RUNTIME_BENCH_REQUIRE_WEBGPU_MLP === "true") gates.add("mlp");
  if (flags.has("require-webgpu-logits") || env.BROWSER_RUNTIME_BENCH_REQUIRE_WEBGPU_LOGITS === "true") gates.add("logits");
  if (flags.has("require-webgpu-attention") || env.BROWSER_RUNTIME_BENCH_REQUIRE_WEBGPU_ATTENTION === "true") gates.add("attention");
  if (flags.has("require-webgpu-projection") || env.BROWSER_RUNTIME_BENCH_REQUIRE_WEBGPU_PROJECTION === "true") gates.add("projection");
  return STRICT_UNLOCKED_WEBGPU_GATES.filter((gate) => gates.has(gate));
}

export function readBrowserPreviewWebGpuGates(
  flags: Set<string>,
  env: NodeJS.ProcessEnv,
  benchmarkWebGpuGates: StrictUnlockedWebGpuGate[],
): StrictUnlockedWebGpuGate[] {
  if (flags.has("browser-preview-require-strict-webgpu")
    || env.BROWSER_RUNTIME_BENCH_PREVIEW_REQUIRE_STRICT_WEBGPU === "true"
    || env.VITE_REQUIRE_WEBGPU_KERNELS === "true") {
    return STRICT_UNLOCKED_WEBGPU_GATES;
  }
  return benchmarkWebGpuGates;
}

async function resolveManifestTarget(args: BenchmarkArgs): Promise<BenchmarkTarget> {
  const manifestUrl = toManifestUrl(args.manifestPath, args.publicDir);
  const manifestText = await readManifestText(manifestUrl);
  const manifest = JSON.parse(manifestText) as { modelId?: string };
  const modelId = manifest.modelId ?? args.modelId;
  return {
    mode: "configured",
    manifestUrl,
    modelId,
    manifestSha256: args.manifestSha256 || sha256Text(manifestText),
  };
}

async function createFixtureTarget(artifactDir: string): Promise<BenchmarkTarget> {
  const fixtureDir = resolve(repoRoot, artifactDir, "browser-runtime-bench-fixture");
  await mkdir(fixtureDir, { recursive: true });
  const weights = new Float32Array([
    1, 0,
    0, 1,
    1, 1,
    0.5, 0.5,
    0, 0,
    0, 0,
    10, 10,
    -1, -1,
    1, 0,
    0, 1,
  ]);
  const weightsPath = resolve(fixtureDir, "weights.bin");
  await writeFile(weightsPath, Buffer.from(weights.buffer));
  const weightSha256 = sha256Buffer(Buffer.from(weights.buffer));
  const manifest = {
    schemaVersion: 1,
    modelId: "Qwen/Qwen3-0.6B",
    architecture: "qwen3_decoder_control",
    vocabSize: 4,
    hiddenSize: 2,
    headDim: 2,
    tokenEmbedding: shardTensor(0, [4, 2], weightSha256),
    outputProjection: shardTensor(8, [4, 2], weightSha256),
    tokenizer: {
      kind: "vocab",
      tokens: ["alpha", "beta", "MANIFEST_TOKEN", "delta"],
      unknownTokenId: 0,
    },
    layers: [
      {
        qProj: shardTensor(16, [2, 2], weightSha256),
        kProj: shardTensor(16, [2, 2], weightSha256),
        vProj: shardTensor(16, [2, 2], weightSha256),
        oProj: shardTensor(16, [2, 2], weightSha256),
        mlpUpProj: shardTensor(16, [2, 2], weightSha256),
        mlpDownProj: shardTensor(16, [2, 2], weightSha256),
      },
    ],
  };
  const manifestPath = resolve(fixtureDir, "manifest.json");
  const manifestText = `${JSON.stringify(manifest, null, 2)}\n`;
  await writeFile(manifestPath, manifestText);
  return {
    mode: "generated-fixture",
    manifestUrl: pathToFileURL(manifestPath).toString(),
    modelId: manifest.modelId,
    manifestSha256: sha256Text(manifestText),
  };
}

async function writeBenchmarkArtifact(args: BenchmarkArgs, artifact: BrowserRuntimeBenchmarkArtifact): Promise<void> {
  const timestamp = artifact.createdAt.replace(/[:.]/g, "-");
  const outputDir = resolve(repoRoot, args.artifactDir, "browser-runtime-bench", timestamp);
  await mkdir(outputDir, { recursive: true });
  await writeFile(join(outputDir, "results.json"), `${JSON.stringify(artifact, null, 2)}\n`);
  await writeFile(join(outputDir, "summary.md"), buildMarkdownSummary(artifact));
  await mkdir(resolve(repoRoot, args.artifactDir), { recursive: true });
  await writeFile(resolve(repoRoot, args.artifactDir, "browser-runtime-bench-latest.json"), `${JSON.stringify(artifact, null, 2)}\n`);
}

export function buildMarkdownSummary(artifact: BrowserRuntimeBenchmarkArtifact): string {
  const runRows = artifact.runs.map((run) => (
    `| ${run.promptId} | ${run.metrics.initLoadMs} | ${run.metrics.prefillMs} | ${run.metrics.timeToFirstTokenMs} | ${run.metrics.decodeLatencyMs} | ${run.metrics.tokensPerSecond} | ${run.prefillChunkCount ?? "n/a"} | ${run.prefillChunkSize ?? "n/a"} | ${run.maxDispatchEstimatedMs ?? "n/a"} | ${formatExpectedSubstringStatus(run)} | ${run.mtp.mode} | ${run.mtp.acceptanceRate} |`
  )).join("\n");
  const gateRows = artifact.thresholds.length === 0
    ? "| none |  |  |  |  |"
    : artifact.thresholds.map((gate) => (
      `| ${gate.name} | ${gate.threshold} | ${gate.observed} | ${gate.blocking} | ${gate.passed} |`
    )).join("\n");
  return `# Browser Runtime Benchmark

- Created: ${artifact.createdAt}
- Passed: ${artifact.passed}
- Mode: ${artifact.mode}
- Model: ${artifact.modelId}
- Runtime profile: ${artifact.runtimeProfile.profile}
- Runtime caps: ${JSON.stringify(artifact.runtimeProfile.resolvedCaps)}
- Memory mode: ${artifact.memoryMode}
- Backend preference requested: ${artifact.requestedBackendPreference}
- Qwen thinking mode: ${artifact.qwenThinkingMode}
- CPU fallback used: ${artifact.webGpuCoverage.cpuFallbackUsed}
- Logit projection backend: ${artifact.webGpuCoverage.logitProjection.backend}
- Strict WebGPU required: ${artifact.webGpuGate.required}
- Strict WebGPU passed: ${artifact.webGpuGate.passed}
- Strict WebGPU gates: ${artifact.webGpuGate.gates.length > 0 ? artifact.webGpuGate.gates.join(", ") : "none"}
- Strict WebGPU failures: ${artifact.webGpuGate.failedReasons.length > 0 ? artifact.webGpuGate.failedReasons.join("; ") : "none"}
- Expected substring checks: ${artifact.summary.expectedSubstringPassCount}/${artifact.summary.expectedSubstringCheckCount} (passed=${artifact.summary.expectedSubstringsPassed})
- Expected exact checks: ${artifact.summary.expectedExactPassCount}/${artifact.summary.expectedExactCheckCount} (passed=${artifact.summary.expectedExactPassed})
- Production speed floor: ${artifact.summary.productionSpeedFloorTokensPerSecond} tok/s (passed=${artifact.summary.productionSpeedFloorPassed}, bottleneck=${artifact.summary.primarySpeedBottleneck})
- Decode fragmentation: submits/token=${artifact.summary.decodeSubmitCountPerToken ?? "n/a"}, dispatches/token=${artifact.summary.decodeDispatchCountPerToken ?? "n/a"}, dispatches/layer/token=${artifact.summary.decodeDispatchCountPerLayerPerToken ?? "n/a"}, fusedCoverage=${artifact.summary.fusedLayerCoverage ?? "n/a"}
- Prefill chunk count: ${artifact.summary.prefillChunkCount ?? "n/a"}
- Prefill chunk size: ${artifact.summary.prefillChunkSize ?? "n/a"}
- Prefill shape bucket: ${artifact.summary.shapeBucket ?? "n/a"}
- Prefill pipeline key: ${artifact.summary.pipelineCacheKey ?? "n/a"}
- Prefill dispatch target ms: ${artifact.summary.prefillDispatchTargetMs ?? "n/a"}
- Prefill dispatch estimate ms: ${artifact.summary.maxDispatchEstimatedMs ?? "n/a"}
- Fatal error: ${artifact.fatalError?.message ?? "none"}
- MTP mode: ${artifact.summary.mtpMode}
- MTP acceptance rate: ${artifact.summary.mtpAcceptanceRate}
- MTP max speculative tokens: ${artifact.summary.mtpMaxSpeculativeTokens ?? 0}
- MTP verifier strategy: ${artifact.summary.mtpVerifierStrategy ?? "none"}
- MTP acceleration: ${artifact.mtpAcceleration.mode}${artifact.mtpAcceleration.mode === "completed" ? ` (passed=${artifact.mtpAcceleration.passed}, netSpeedup=${artifact.mtpAcceleration.netSpeedupRatio})` : ""}
- KV prefetch strategy: ${artifact.summary.kvPrefetchStrategy ?? "none"}
- KV predictive prefetch runs: ${artifact.summary.kvPredictivePrefetchRunCount ?? 0}
- KV miss/stall runs: ${artifact.summary.kvMissStallRunCount ?? 0}
- KV prefetch hit rate: ${artifact.summary.kvPrefetchHitRate ?? "n/a"}
- KV attention stall ms: ${artifact.summary.kvAttentionStallMs ?? "n/a"}
- Browser preview: ${artifact.browserPreview.mode}${artifact.browserPreview.mode === "skipped" || artifact.browserPreview.mode === "failed" ? ` (${artifact.browserPreview.reason})` : ""}

## Runs

| Prompt | Init/load ms | Prefill ms | TTFT ms | Decode latency ms | Tokens/sec | Prefill chunks | Chunk size | Max dispatch ms | Expected text | MTP | MTP acceptance |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | ---: |
${runRows}

## Thresholds

| Gate | Threshold | Observed | Blocking | Passed |
| --- | ---: | ---: | --- | --- |
${gateRows}
`;
}

export function evaluateExpectedSubstringMatches(
  response: string,
  expectedSubstrings: string[],
): Array<{ expected: string; matched: boolean }> {
  const lowerResponse = stripProofScaffolding(response).toLowerCase();
  return expectedSubstrings
    .map((expected) => expected.trim())
    .filter(Boolean)
    .map((expected) => ({
      expected,
      matched: lowerResponse.includes(expected.toLowerCase()),
    }));
}

export function evaluateExpectedExactMatches(
  response: string,
  expectedExact: string[],
): Array<{ expected: string; matched: boolean }> {
  const visibleResponse = stripProofScaffolding(response);
  return expectedExact
    .map((expected) => expected.trim())
    .filter(Boolean)
    .map((expected) => ({
      expected,
      matched: visibleResponse === expected,
    }));
}

function stripProofScaffolding(value: string): string {
  return value
    .replace(/<think\b[^>]*>[\s\S]*?<\/think>/gi, "")
    .replaceAll("[unlocked:ssa-kv-tsp]", "")
    .replaceAll("<think>", "")
    .replaceAll("</think>", "")
    .replaceAll("<|im_end|>", "")
    .replaceAll("<|endoftext|>", "")
    .replaceAll("<|end|>", "")
    .replaceAll("</s>", "")
    .trim();
}

function formatExpectedSubstringStatus(run: BrowserRuntimeBenchmarkRun): string {
  const matches = run.expectedSubstringMatches ?? [];
  if (matches.length === 0) return "none";
  const passed = matches.filter((match) => match.matched).length;
  return `${passed}/${matches.length}`;
}

function normalizeMtpProof(proof: UnlockedBrowserMtpProof | undefined): BrowserRuntimeBenchmarkRun["mtp"] {
  if (!proof) return { mode: "none", acceptedTokens: 0, rejectedTokens: 0, acceptanceRate: 0 };
  return {
    mode: proof.mode,
    ...(proof.draftModelId !== undefined ? { draftModelId: proof.draftModelId } : {}),
    ...(proof.draftSource ? { draftSource: proof.draftSource } : {}),
    ...(proof.latencyDisablePolicy ? { latencyDisablePolicy: proof.latencyDisablePolicy } : {}),
    ...(proof.verifierStrategy ? { verifierStrategy: proof.verifierStrategy } : {}),
    ...(proof.disabledReason ? { disabledReason: proof.disabledReason } : {}),
    acceptedTokens: proof.acceptedTokens,
    rejectedTokens: proof.rejectedTokens,
    acceptanceRate: roundMetric(proof.acceptanceRate),
    ...(isFiniteNumber(proof.numSpeculativeTokens) ? { numSpeculativeTokens: proof.numSpeculativeTokens } : {}),
    ...(isFiniteNumber(proof.verifiedTokenCount) ? { verifiedTokenCount: proof.verifiedTokenCount } : {}),
    ...(isFiniteNumber(proof.targetDecodeCalls) ? { targetDecodeCalls: proof.targetDecodeCalls } : {}),
    ...(isFiniteNumber(proof.committedInputTokens) ? { committedInputTokens: proof.committedInputTokens } : {}),
  };
}

function summarizeRuntimeKvPersistence(
  health: UnlockedBrowserDecodeProof["kvPersistence"] | null | undefined,
): BrowserRuntimeBenchmarkRun["kvPersistence"] {
  const events = health?.events ?? [];
  return {
    enabled: health?.enabled ?? false,
    mode: health?.mode ?? "disabled",
    eventCount: events.length,
    persistEvents: countKvPersistenceEvents(events, "persist"),
    hydrateEvents: countKvPersistenceEvents(events, "hydrate"),
    reuseEvents: countKvPersistenceEvents(events, "reuse"),
    prefetchStrategy: health?.prefetchStrategy ?? "none",
    ...(isFiniteNumber(health?.lowRankSummaryRank) ? { lowRankSummaryRank: health.lowRankSummaryRank } : {}),
    ...(typeof health?.lowRankQuerySource === "string" ? { lowRankQuerySource: health.lowRankQuerySource } : {}),
    kvPersistDeferred: health?.kvPersistDeferred === true,
    kvPersistCriticalPathMs: isFiniteNumber(health?.kvPersistCriticalPathMs) ? roundMetric(health.kvPersistCriticalPathMs) : 0,
    ...(isFiniteNumber(health?.kvPersistFlushMs) ? { kvPersistFlushMs: roundMetric(health.kvPersistFlushMs) } : {}),
    kvPersistPendingBlockCount: isFiniteNumber(health?.kvPersistPendingBlockCount) ? health.kvPersistPendingBlockCount : 0,
    predictedHotBlocks: (health?.predictedHotBlocks ?? [])
      .map((block) => block.blockId)
      .filter((blockId): blockId is string => typeof blockId === "string" && blockId.length > 0),
    prefetchedBlocks: (health?.prefetchedBlocks ?? []).filter((blockId): blockId is string => typeof blockId === "string" && blockId.length > 0),
    prefetchHitRate: isFiniteNumber(health?.prefetchHitRate) ? roundMetric(health.prefetchHitRate) : 0,
    prefetchBytes: isFiniteNumber(health?.prefetchBytes) ? roundMetric(health.prefetchBytes) : 0,
    prefetchLatencyMs: isFiniteNumber(health?.prefetchLatencyMs) ? roundMetric(health.prefetchLatencyMs) : 0,
    attentionStallMs: isFiniteNumber(health?.attentionStallMs) ? roundMetric(health.attentionStallMs) : 0,
  };
}

function countKvPersistenceEvents(
  events: NonNullable<UnlockedBrowserDecodeProof["kvPersistence"]>["events"],
  operation: string,
): number {
  return events.filter((event) => event.operation === operation).length;
}

function readPrefillChunkMetadata(value: unknown): BrowserPrefillChunkMetadata {
  if (!isRecord(value)) return {};
  return {
    ...(isFiniteNumber(value.prefillChunkCount) ? { prefillChunkCount: value.prefillChunkCount } : {}),
    ...(isFiniteNumber(value.prefillChunkSize) ? { prefillChunkSize: value.prefillChunkSize } : {}),
    ...(typeof value.shapeBucket === "string" ? { shapeBucket: value.shapeBucket } : {}),
    ...(typeof value.pipelineCacheKey === "string" ? { pipelineCacheKey: value.pipelineCacheKey } : {}),
    ...(isFiniteNumber(value.prefillDispatchTargetMs) ? { prefillDispatchTargetMs: value.prefillDispatchTargetMs } : {}),
    ...(isFiniteNumber(value.maxDispatchEstimatedMs) ? { maxDispatchEstimatedMs: value.maxDispatchEstimatedMs } : {}),
    ...(value.prefillChunkDispatch === "single_dispatch" || value.prefillChunkDispatch === "chunked_dispatch"
      ? { prefillChunkDispatch: value.prefillChunkDispatch }
      : {}),
    ...(typeof value.prefillChunkReason === "string" ? { prefillChunkReason: value.prefillChunkReason } : {}),
  };
}

function summarizeMtpMode(runs: BrowserRuntimeBenchmarkRun[]): "target_only" | "draft_verify" | "mixed" | "none" {
  const modes = [...new Set(runs.map((run) => run.mtp.mode))];
  if (modes.length === 0) return "none";
  if (modes.length === 1) return modes[0] ?? "none";
  return "mixed";
}

function summarizeMtpVerifierStrategy(runs: BrowserRuntimeBenchmarkRun[]): string {
  const strategies = [...new Set(runs.map((run) => run.mtp.verifierStrategy).filter((strategy): strategy is string => Boolean(strategy)))];
  if (strategies.length === 0) return "none";
  if (strategies.length === 1) return strategies[0] ?? "none";
  return "mixed";
}

function identifyPrimarySpeedBottleneck(runs: BrowserRuntimeBenchmarkRun[]): string {
  if (runs.length === 0) return "no_runs";
  const maxSubmitPerToken = maxOrNull(runs.map((run) => (
    isFiniteNumber(run.decodePerf?.decodeSubmitCountPerToken)
      ? run.decodePerf?.decodeSubmitCountPerToken
      : run.metrics.generatedTokens > 0
        ? (run.decodePerf?.decodeSubmitCount ?? 0) / run.metrics.generatedTokens
        : undefined
  )));
  const maxDispatchPerLayerPerToken = maxOrNull(runs.map((run) => (
    isFiniteNumber(run.decodePerf?.decodeDispatchCountPerLayerPerToken)
      ? run.decodePerf?.decodeDispatchCountPerLayerPerToken
      : (run.decodePerf?.totalDecodeLayerCount ?? 0) > 0
        ? (run.decodePerf?.dispatchCount ?? 0) / (run.decodePerf?.totalDecodeLayerCount ?? 1)
        : undefined
  )));
  const maxLayerVisitsPerToken = maxOrNull(runs.map((run) => {
    const layerVisits = run.decodePerf?.totalDecodeLayerCount;
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
  const fusedCoverages = runs.map((run) => run.decodePerf?.fusedLayerCoverage).filter(isFiniteNumber);
  if (fusedCoverages.length > 0 && Math.min(...fusedCoverages) < 1) return "unfused_decode_layer_path";
  const buckets = [
    ["init_load", mean(runs.map((run) => run.metrics.initLoadMs))],
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

function readMemoryMode(mode: BenchmarkMode): string {
  return process.env.VITE_MEMORY_PROVIDER
    ?? process.env.MEMORY_PROVIDER
    ?? (mode === "generated-fixture" ? "browser-local-fixture" : "browser-local");
}

function readPrompts(
  value: string | undefined,
  expectedValue?: string,
  options: {
    expectedExactValue?: string;
    longPromptTargetTokens?: number;
    longPromptRepeat?: number;
    longPromptSeed?: string;
  } = {},
): BrowserRuntimeBenchmarkPrompt[] {
  const expectedByIndex = readExpectedSubstrings(expectedValue);
  const expectedExactByIndex = readExpectedSubstrings(options.expectedExactValue);
  if (options.longPromptTargetTokens || options.longPromptRepeat) {
    const seedText = options.longPromptSeed ?? readFirstPromptSeed(value);
    const expectedExact = expectedExactByIndex[0] ?? [];
    return [{
      id: "long-prompt-1",
      text: buildDeterministicLongPrompt({
        targetTokens: options.longPromptTargetTokens,
        repeat: options.longPromptRepeat,
        seedText,
      }),
      expectedSubstrings: expectedByIndex[0] ?? [],
      ...(expectedExact.length ? { expectedExact } : {}),
    }];
  }
  if (!value?.trim()) {
    return [
      { id: "short-recall", text: "alpha beta" },
      { id: "instruction", text: "alpha beta MANIFEST_TOKEN" },
    ].map((prompt, index) => {
      const expectedExact = expectedExactByIndex[index] ?? [];
      return {
        ...prompt,
        expectedSubstrings: expectedByIndex[index] ?? [],
        ...(expectedExact.length ? { expectedExact } : {}),
      };
    });
  }
  return value.split("|")
    .map((text, index) => {
      const expectedExact = expectedExactByIndex[index] ?? [];
      return {
        id: `prompt-${index + 1}`,
        text: text.trim(),
        expectedSubstrings: expectedByIndex[index] ?? [],
        ...(expectedExact.length ? { expectedExact } : {}),
      };
    })
    .filter((prompt) => prompt.text.length > 0);
}

function readFirstPromptSeed(value: string | undefined): string {
  if (!value) return "";
  const bounded = value.slice(0, MAX_LONG_PROMPT_SEED_CHARS);
  const separatorIndex = bounded.indexOf("|");
  const firstPromptSeed = separatorIndex >= 0 ? bounded.slice(0, separatorIndex) : bounded;
  return buildDeterministicLongPromptSeed(firstPromptSeed);
}

function readExpectedSubstrings(value: string | undefined): string[][] {
  if (!value?.trim()) return [];
  return value.split("|").map((group) => group
    .split(",")
    .map((expected) => expected.trim())
    .filter(Boolean));
}

function readOptionalGenerationTokenBudget(value: string | undefined): Pick<BenchmarkArgs, "generationTokenBudget"> {
  const parsed = parsePositiveInteger(value, 0);
  return parsed > 0 ? { generationTokenBudget: parsed } : {};
}

function readThreshold(
  name: BrowserRuntimeBenchmarkThreshold["name"],
  envName: string,
  value: string | undefined,
): BrowserRuntimeBenchmarkThreshold | null {
  if (!value?.trim()) return null;
  const parsed = Number(value);
  if (!Number.isFinite(parsed) || parsed <= 0) throw new Error(`${envName} must be a positive number, received "${value}".`);
  return { name, threshold: parsed };
}

export function calculateTokensPerSecond(generatedTokens: number, decodeLatencyMs: number): number | null {
  if (!Number.isFinite(generatedTokens) || generatedTokens <= 0) return null;
  if (!Number.isFinite(decodeLatencyMs) || decodeLatencyMs <= 0) return null;
  return roundMetric(generatedTokens / (decodeLatencyMs / 1000));
}

function validateBrowserPreviewPayload(value: {
  passed?: boolean;
  summary?: Record<string, number | string | boolean | null>;
  runs?: BrowserRuntimeBenchmarkRun[];
}): string | null {
  if (typeof value.passed !== "boolean") return "browser preview benchmark response must include boolean passed.";
  if (!isRecord(value.summary)) return "browser preview benchmark response must include summary object.";
  if (!Array.isArray(value.runs)) return "browser preview benchmark response must include runs array.";
  const requiredNumericSummaryFields = [
    "meanInitLoadMs",
    "meanPrefillMs",
    "meanTimeToFirstTokenMs",
    "meanDecodeLatencyMs",
  ];
  const allowNullFailureTiming = value.passed === false && value.runs.length === 0;
  for (const field of requiredNumericSummaryFields) {
    if (!(field in value.summary)) return `browser preview benchmark summary missing ${field}.`;
    if (allowNullFailureTiming && value.summary[field] === null) continue;
    if (!isFiniteNumber(value.summary[field])) return `browser preview benchmark summary ${field} must be a finite number.`;
  }
  if (!("meanTokensPerSecond" in value.summary)) return "browser preview benchmark summary missing meanTokensPerSecond.";
  if (value.summary.meanTokensPerSecond !== null && !isFiniteNumber(value.summary.meanTokensPerSecond)) {
    return "browser preview benchmark summary meanTokensPerSecond must be a finite number or null.";
  }
  if (typeof value.summary.mtpMode !== "string") {
    return "browser preview benchmark summary mtpMode must be a string.";
  }
  for (const field of ["technicalProofOnly", "productionQualityPassed", "productionDeployReadyPassed"]) {
    if (typeof value.summary[field] !== "boolean") {
      return `browser preview benchmark summary ${field} must be a boolean.`;
    }
  }
  const invalidGroundingSummary = validateBrowserPreviewGroundingSummary(value.summary);
  if (invalidGroundingSummary) return invalidGroundingSummary;
  const auditOnlyPreview = value.summary.memoryGroundingAuditOnly === true;
  if (value.passed === true && value.runs.length === 0 && !auditOnlyPreview) {
    return "browser preview benchmark passed snapshots must include at least one run.";
  }
  if (value.passed === true || value.runs.length > 0) {
    const invalidMtpSummary = validateBrowserPreviewMtpSummary(value.summary);
    if (invalidMtpSummary) return invalidMtpSummary;
    const invalidKvSummary = validateBrowserPreviewKvSummary(value.summary);
    if (invalidKvSummary) return invalidKvSummary;
  }
  for (let index = 0; index < value.runs.length; index += 1) {
    const invalidRun = validateBrowserPreviewRun(value.runs[index], index);
    if (invalidRun) return invalidRun;
  }
  return null;
}

function validateBrowserPreviewGroundingSummary(summary: Record<string, number | string | boolean | null>): string | null {
  if (!("memoryGroundingRequired" in summary)) return null;
  if (typeof summary.memoryGroundingRequired !== "boolean") {
    return "browser preview benchmark summary memoryGroundingRequired must be a boolean.";
  }
  if (summary.memoryGroundingRequired !== true) return null;
  for (const field of [
    "memoryGroundingPassed",
    "memoryGeneratedParaphraseRequired",
    "memoryGeneratedParaphrasePassed",
  ]) {
    if (typeof summary[field] !== "boolean") {
      return `browser preview benchmark summary ${field} must be a boolean.`;
    }
  }
  for (const field of [
    "memoryGeneratedParaphraseQueryCount",
    "memoryGeneratedParaphraseTop1CorrectCount",
  ]) {
    if (!isFiniteNumber(summary[field])) {
      return `browser preview benchmark summary ${field} must be a finite number.`;
    }
  }
  for (const field of [
    "memoryGeneratedParaphraseRecallAt1",
    "memoryGeneratedParaphraseMrr",
  ]) {
    if (summary[field] !== null && !isFiniteNumber(summary[field])) {
      return `browser preview benchmark summary ${field} must be a finite number or null.`;
    }
  }
  return null;
}

function validateBrowserPreviewMtpSummary(summary: Record<string, number | string | boolean | null>): string | null {
  if (typeof summary.mtpVerifierStrategy !== "string") {
    return "browser preview benchmark summary mtpVerifierStrategy must be a string.";
  }
  for (const field of ["mtpMaxSpeculativeTokens", "mtpVerifiedTokenCount", "mtpTargetDecodeCalls"]) {
    if (!isFiniteNumber(summary[field])) {
      return `browser preview benchmark summary ${field} must be a finite number.`;
    }
  }
  if (summary.mtpMeanSpeculativeTokens !== null && !isFiniteNumber(summary.mtpMeanSpeculativeTokens)) {
    return "browser preview benchmark summary mtpMeanSpeculativeTokens must be a finite number or null.";
  }
  return null;
}

function validateBrowserPreviewKvSummary(summary: Record<string, number | string | boolean | null>): string | null {
  if (typeof summary.kvPrefetchStrategy !== "string") {
    return "browser preview benchmark summary kvPrefetchStrategy must be a string.";
  }
  for (const field of [
    "kvPersistenceEventCount",
    "kvPersistEventCount",
    "kvHydrateEventCount",
    "kvReuseEventCount",
    "kvExactReuseRunCount",
    "kvPredictivePrefetchRunCount",
    "kvMissStallRunCount",
    "kvNoPrefetchRunCount",
    "kvPredictedHotBlockCount",
    "kvPrefetchedBlockCount",
    "kvPrefetchBytes",
    "kvPrefetchLatencyMs",
    "kvAttentionStallMs",
  ]) {
    if (!isFiniteNumber(summary[field])) {
      return `browser preview benchmark summary ${field} must be a finite number.`;
    }
  }
  for (const field of ["kvLowRankSummaryRank", "kvPrefetchHitRate"]) {
    if (summary[field] !== null && !isFiniteNumber(summary[field])) {
      return `browser preview benchmark summary ${field} must be a finite number or null.`;
    }
  }
  return null;
}

function validateBrowserPreviewRun(value: unknown, index: number): string | null {
  const prefix = `browser preview benchmark runs[${index}]`;
  if (!isRecord(value)) return `${prefix} must be an object.`;
  for (const field of ["promptId", "prompt", "response"]) {
    if (typeof value[field] !== "string") return `${prefix}.${field} must be a string.`;
  }
  if (!isRecord(value.metrics)) return `${prefix}.metrics must be an object.`;
  for (const field of [
    "initLoadMs",
    "prefillMs",
    "timeToFirstTokenMs",
    "decodeLatencyMs",
    "generatedTokens",
  ]) {
    if (!isFiniteNumber(value.metrics[field])) return `${prefix}.metrics.${field} must be a finite number.`;
  }
  if (!("tokensPerSecond" in value.metrics)) return `${prefix}.metrics.tokensPerSecond is required.`;
  if (value.metrics.tokensPerSecond !== null && !isFiniteNumber(value.metrics.tokensPerSecond)) {
    return `${prefix}.metrics.tokensPerSecond must be a finite number or null.`;
  }
  if (!isRecord(value.mtp)) return `${prefix}.mtp must be an object.`;
  if (value.mtp.mode !== "target_only" && value.mtp.mode !== "draft_verify" && value.mtp.mode !== "none") {
    return `${prefix}.mtp.mode must be target_only, draft_verify, or none.`;
  }
  if ("draftModelId" in value.mtp && value.mtp.draftModelId !== null && typeof value.mtp.draftModelId !== "string") {
    return `${prefix}.mtp.draftModelId must be a string or null when present.`;
  }
  if ("draftSource" in value.mtp && typeof value.mtp.draftSource !== "string") {
    return `${prefix}.mtp.draftSource must be a string when present.`;
  }
  if ("latencyDisablePolicy" in value.mtp && typeof value.mtp.latencyDisablePolicy !== "string") {
    return `${prefix}.mtp.latencyDisablePolicy must be a string when present.`;
  }
  for (const field of ["acceptedTokens", "rejectedTokens", "acceptanceRate"]) {
    if (!isFiniteNumber(value.mtp[field])) return `${prefix}.mtp.${field} must be a finite number.`;
  }
  for (const field of ["numSpeculativeTokens", "verifiedTokenCount", "targetDecodeCalls"]) {
    if (!isFiniteNumber(value.mtp[field])) return `${prefix}.mtp.${field} must be a finite number.`;
  }
  if (typeof value.mtp.verifierStrategy !== "string") {
    return `${prefix}.mtp.verifierStrategy must be a string.`;
  }
  const invalidKvPersistence = validateBrowserPreviewRunKvPersistence(value.kvPersistence, prefix);
  if (invalidKvPersistence) return invalidKvPersistence;
  return null;
}

function validateBrowserPreviewRunKvPersistence(value: unknown, prefix: string): string | null {
  if (!isRecord(value)) return `${prefix}.kvPersistence must be an object.`;
  if (typeof value.enabled !== "boolean") return `${prefix}.kvPersistence.enabled must be a boolean.`;
  if (typeof value.mode !== "string") return `${prefix}.kvPersistence.mode must be a string.`;
  for (const field of ["eventCount", "persistEvents", "hydrateEvents", "reuseEvents"]) {
    if (!isFiniteNumber(value[field])) return `${prefix}.kvPersistence.${field} must be a finite number.`;
  }
  if (typeof value.prefetchStrategy !== "string") {
    return `${prefix}.kvPersistence.prefetchStrategy must be a string.`;
  }
  if ("lowRankSummaryRank" in value && !isFiniteNumber(value.lowRankSummaryRank)) {
    return `${prefix}.kvPersistence.lowRankSummaryRank must be a finite number when present.`;
  }
  if ("lowRankQuerySource" in value && typeof value.lowRankQuerySource !== "string") {
    return `${prefix}.kvPersistence.lowRankQuerySource must be a string when present.`;
  }
  if ("kvPersistDeferred" in value && typeof value.kvPersistDeferred !== "boolean") {
    return `${prefix}.kvPersistence.kvPersistDeferred must be a boolean when present.`;
  }
  for (const field of ["kvPersistCriticalPathMs", "kvPersistFlushMs", "kvPersistPendingBlockCount"]) {
    if (field in value && !isFiniteNumber(value[field])) return `${prefix}.kvPersistence.${field} must be a finite number when present.`;
  }
  if (!Array.isArray(value.predictedHotBlocks) || value.predictedHotBlocks.some((blockId) => typeof blockId !== "string")) {
    return `${prefix}.kvPersistence.predictedHotBlocks must be an array of strings.`;
  }
  if (!Array.isArray(value.prefetchedBlocks) || value.prefetchedBlocks.some((blockId) => typeof blockId !== "string")) {
    return `${prefix}.kvPersistence.prefetchedBlocks must be an array of strings.`;
  }
  for (const field of ["prefetchHitRate", "prefetchBytes", "prefetchLatencyMs", "attentionStallMs"]) {
    if (!isFiniteNumber(value[field])) return `${prefix}.kvPersistence.${field} must be a finite number.`;
  }
  return null;
}

function shardTensor(floatOffset: number, shape: number[], sha256: string): Record<string, unknown> {
  return {
    kind: "f32-shard",
    uri: "weights.bin",
    byteOffset: floatOffset * Float32Array.BYTES_PER_ELEMENT,
    shape,
    sha256,
  };
}

function parseBackendPreference(value: string | undefined): BackendPreference | undefined {
  if (!value) return undefined;
  if (value === "cpu" || value === "webgpu") return value;
  throw new Error(`Browser runtime benchmark backend preference must be "cpu" or "webgpu", received "${value}".`);
}

function toManifestUrl(manifestPath: string, publicDir: string): string {
  if (manifestPath.startsWith("http://") || manifestPath.startsWith("https://") || manifestPath.startsWith("file://")) {
    return manifestPath;
  }
  if (manifestPath.startsWith("/") && existsSync(manifestPath)) return pathToFileURL(manifestPath).toString();
  if (manifestPath.startsWith("/")) return pathToFileURL(resolve(publicDir, manifestPath.slice(1))).toString();
  return pathToFileURL(resolve(manifestPath)).toString();
}

async function readManifestText(manifestUrl: string): Promise<string> {
  if (manifestUrl.startsWith("file://")) return readFile(new URL(manifestUrl), "utf8");
  const response = await fetch(manifestUrl);
  if (!response.ok) throw new Error(`Unlocked manifest failed to load: ${response.status}`);
  const contentType = response.headers.get("content-type") ?? "";
  if (contentType.toLowerCase().includes("text/html")) {
    throw new Error("Unlocked manifest request returned HTML; this usually means the app shell was served instead of manifest JSON.");
  }
  return response.text();
}

async function installFileFetch(): Promise<void> {
  const originalFetch = globalThis.fetch.bind(globalThis);
  globalThis.fetch = async (input: RequestInfo | URL, init?: RequestInit): Promise<Response> => {
    const url = String(input);
    if (!url.startsWith("file://")) return originalFetch(input, init);
    const path = new URL(url);
    const stats = await stat(path);
    if (!stats.isFile()) return new Response("not found", { status: 404 });
    const bytes = await readFile(path);
    const name = basename(path.pathname);
    return new Response(bytes, {
      status: 200,
      headers: {
        "content-type": name.endsWith(".json") ? "application/json" : "application/octet-stream",
        "content-length": String(bytes.byteLength),
      },
    });
  };
}

function parsePositiveInteger(value: string | undefined, fallback: number, max = Number.MAX_SAFE_INTEGER): number {
  if (!value?.trim()) return fallback;
  const parsed = Number(value);
  if (!Number.isFinite(parsed)) return fallback;
  const integer = Math.floor(parsed);
  return integer > 0 ? Math.min(integer, max) : fallback;
}

function parsePositiveNumber(value: string | undefined, fallback: number): number {
  if (!value?.trim()) return fallback;
  const parsed = Number(value);
  return Number.isFinite(parsed) && parsed > 0 ? parsed : fallback;
}

function parseBrowserMtpSpeculativeTokens(value: string | undefined): number {
  return Math.min(
    parsePositiveInteger(value, LOCAL_BROWSER_MTP_DEFAULT_SPECULATIVE_TOKENS),
    LOCAL_BROWSER_MTP_MAX_SPECULATIVE_TOKENS,
  );
}

function parseRatio(value: string | undefined, fallback: number): number {
  if (!value?.trim()) return fallback;
  const parsed = Number(value);
  if (!Number.isFinite(parsed)) return fallback;
  return Math.max(0, Math.min(1, parsed));
}

export function parseQwenThinkingMode(value: string | undefined): QwenThinkingMode {
  const normalized = value?.trim().toLowerCase();
  if (normalized === "enabled") return "enabled";
  return "disabled";
}

function mean(values: Array<number | null | undefined>): number | null {
  const finite = values.filter((value): value is number => typeof value === "number" && Number.isFinite(value));
  if (finite.length === 0) return null;
  return roundMetric(finite.reduce((sum, value) => sum + value, 0) / finite.length);
}

function isWarmResidentBenchmarkRun(run: BrowserRuntimeBenchmarkRun): boolean {
  return run.metrics.initLoadMs <= 0;
}

function sum(values: Array<number | null | undefined>): number {
  return roundMetric(values.filter(isFiniteNumber).reduce((total, value) => total + value, 0));
}

function maxOrNull(values: Array<number | null | undefined>): number | null {
  const finite = values.filter(isFiniteNumber);
  return finite.length > 0 ? roundMetric(Math.max(...finite)) : null;
}

function maxGeneratedTokens(runs: BrowserRuntimeBenchmarkRun[]): number {
  return Math.max(0, ...runs.map((run) => run.metrics.generatedTokens));
}

function elapsed(start: number): number {
  return roundMs(performance.now() - start);
}

function roundMs(value: number): number {
  return Math.round(value * 100) / 100;
}

function roundMetric(value: number): number {
  return Math.round(value * 1000) / 1000;
}

function sha256Text(value: string): string {
  return createHash("sha256").update(value).digest("hex");
}

function sha256Buffer(value: Buffer): string {
  return createHash("sha256").update(value).digest("hex");
}

function isMainModule(): boolean {
  return process.argv[1] ? import.meta.url === pathToFileURL(process.argv[1]).toString() : false;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function isFiniteNumber(value: unknown): value is number {
  return typeof value === "number" && Number.isFinite(value);
}
