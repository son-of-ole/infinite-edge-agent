import { useEffect, useMemo, useRef, useState } from "react";
import {
  buildContextRuntimePlan,
  evaluateUnlockedWebGpuCoverageGates,
  normalizeVector,
  summarizeUnlockedWebGpuCoverage,
  type ChatMessage,
  type MemoryChunk,
  type StrictUnlockedWebGpuGate,
  type UnlockedKernelBackend,
  type UnlockedWebGpuCoverageSummary,
} from "@infinite-edge-agent/core";
import {
  BENCHMARK_TELEMETRY_CONFIG,
  CHAT_MAX_GENERATION_TOKENS,
  COMPILED_WEBLLM_ENABLED,
  DEFAULT_LLM_BACKEND,
  DEFAULT_MODEL,
  KVSWAP_PERSISTENCE_CLEAR_ON_INIT,
  KVSWAP_PERSISTENCE_ENABLED,
  KVSWAP_PERSISTENCE_MAX_BLOCKS,
  KVSWAP_PERSISTENCE_MAX_BYTES,
  KVSWAP_PERSISTENCE_PREFER_OPFS,
  MEMORY_CELL_ID,
  MEMORY_TENANT_ID,
  MTP_DISABLE_WHEN_LATENCY_WORSE,
  MTP_DRAFT_LAYER_COUNT,
  MTP_DRAFT_MODEL_ID,
  MTP_ENABLED,
  MTP_MIN_ACCEPTANCE_RATE,
  MTP_NUM_SPECULATIVE_TOKENS,
  QWEN_THINKING_MODE,
  REQUIRE_WEBGPU_KERNELS,
  UNLOCKED_ALLOW_FIXTURE,
  UNLOCKED_BACKEND_PREFERENCE,
  UNLOCKED_MODEL_MANIFEST_PATH,
  UNLOCKED_MODEL_MANIFEST_SHA256,
} from "../config";
import {
  collectBenchmarkTelemetryBrowserContext,
  submitBenchmarkTelemetry,
  type BenchmarkTelemetryConfig,
} from "./benchmarkTelemetry";
import { CompiledWebLlmClient, type CompiledWebLlmProof } from "../lib/llm/compiledWebLlmClient";
import {
  UnlockedBrowserTransformerClient,
  type UnlockedBrowserDecodeProof,
  type UnlockedBrowserWarmupMode,
} from "../lib/llm/unlockedBrowserTransformerClient";
import { UnlockedBrowserTransformerWorkerClient } from "../lib/llm/unlockedBrowserTransformerWorkerClient";
import type { ChatClientMessage } from "../lib/llm/types";
import type { KVSwapPersistenceOperation } from "../lib/runtime/kvSwapPersistence";
import {
  resolveUnlockedRuntimeProfile,
  type UnlockedRuntimeProfileResolution,
} from "../lib/runtime/unlockedRuntimeProfile";
import {
  selectBrowserBackend,
  type BrowserBackendSelection,
  type BrowserBackendTask,
} from "../lib/runtime/backendBroker";
import { IndexedDbMemoryStore } from "../lib/storage/indexedDbMemoryStore";
import {
  buildDeterministicLongPrompt,
  buildBrowserPreviewBenchmarkPayload,
  BROWSER_PREVIEW_BENCHMARK_SCHEMA_VERSION,
  MAX_LONG_PROMPT_SEED_CHARS,
  MAX_LONG_PROMPT_REPEAT,
  MAX_LONG_PROMPT_TARGET_TOKENS,
  type BrowserMemoryGroundingProof,
  type BrowserMemoryRetrievalAuditProof,
  type BrowserPrefillChunkMetadata,
  type BrowserPreviewBenchmarkPayload,
  type BrowserPreviewBenchmarkRun,
} from "./browserPreviewBenchmark";

type BackendPreference = "cpu" | "webgpu";
type WebGpuGate = StrictUnlockedWebGpuGate;

export interface BrowserPreviewBenchmarkRequest {
  backendId: string;
  modelId: string;
  profile: UnlockedRuntimeProfileResolution;
  prompts: Array<{ id: string; text: string; expectedSubstrings: string[]; expectedExact?: string[] }>;
  backendPreference?: BackendPreference;
  strictWebGpuRequested: boolean;
  webGpuGates: WebGpuGate[];
  requireKvReuse: boolean;
  requireWarmResidentSpeedProof: boolean;
  requireKvPredictivePrefetch: boolean;
  kvNamespace: string;
  qwenThinkingMode: "disabled" | "enabled";
  mtpEnabled: boolean;
  logitTopK?: number;
  logitTileRows?: number;
  samplingTemperature?: number;
  samplingTopP?: number;
  repetitionPenalty?: number;
  samplingSeed?: number;
  warmModelResidency: boolean;
  warmModelResidencyMode: UnlockedBrowserWarmupMode;
  minGeneratedTokens: number;
  timeoutMs: number;
  longPromptTargetTokens?: number;
  longPromptRepeat?: number;
  strictLongPromptProof: boolean;
  strictExpectedLayerMode: "requested" | "full";
  memoryGroundingCase: BrowserMemoryGroundingCase | null;
  memoryGroundingCorpusSize: number;
  memoryGroundingPromptLimit?: number;
  memoryGroundingAuditOnly: boolean;
  benchmarkTelemetryRequested: boolean;
  benchmarkTelemetryConfig: BenchmarkTelemetryConfig;
}

const BENCHMARK_PATH = "/__bench/browser-runtime";
const PROOF_MARKER = "[unlocked:ssa-kv-tsp]";
const STOP_FRAGMENTS = [
  "<|im_end|>",
  "<|endoftext|>",
  "<|end|>",
  "</s>",
];
const ALL_WEBGPU_GATES: WebGpuGate[] = ["mlp", "logits", "attention", "projection"];
const BENCHMARK_GREEDY_LOGIT_TILE_ROWS = 32_768;
const STRICT_EXPECTED_SUBSTRING_MIN_RUNTIME_LAYERS = 28;
const BROWSER_BENCHMARK_LOCK_NAME = "edge-ai-browser-runtime-benchmark";
const MEMORY_GROUNDING_SESSION_ID = "browser-memory-grounding";
const MEMORY_GROUNDING_VECTOR_DIM = 512;
const DEFAULT_MEMORY_GROUNDING_CORPUS_SIZE = 256;
const MAX_MEMORY_GROUNDING_CORPUS_SIZE = 4096;
const MEMORY_GROUNDING_RETRIEVAL_LIMIT = 8;
const LARGE_SYNTHETIC_GROUNDING_FACT_COUNT = 64;
const LARGE_SYNTHETIC_MIN_CORPUS_SIZE = 1024;
const LARGE_SYNTHETIC_MIN_RECALL_AT_1 = 1;

type BrowserMemoryGroundingCase = "montana_capital" | "qa_corpus_v1" | "large_synthetic_v1";
type BrowserMemoryGroundingAuditQueryClass = "canonical" | "alias" | "generated_paraphrase";

interface BrowserMemoryGroundingAuditQuery {
  query: string;
  queryClass: BrowserMemoryGroundingAuditQueryClass;
}

interface BrowserMemoryGroundingAuditClassStats {
  queryCount: number;
  top1CorrectCount: number;
  reciprocalRankSum: number;
}

interface BrowserMemoryGroundingHarness {
  caseId: BrowserMemoryGroundingCase;
  store: IndexedDbMemoryStore;
  corpus: MemoryChunk[];
  retrievalAudit?: BrowserMemoryRetrievalAuditProof;
}

interface BrowserBenchmarkLockManager {
  request<T>(
    name: string,
    options: { mode: "exclusive"; ifAvailable: true },
    callback: (lock: unknown | null) => T | Promise<T>,
  ): Promise<T>;
}

type BrowserPreviewClient = UnlockedBrowserTransformerClient | UnlockedBrowserTransformerWorkerClient | CompiledWebLlmClient;

export function isBrowserPreviewBenchmarkPath(pathname: string = globalThis.location?.pathname ?? ""): boolean {
  return pathname === BENCHMARK_PATH;
}

export function BrowserPreviewBenchmarkRoute() {
  const request = useMemo(() => readBrowserPreviewBenchmarkRequest(new URL(globalThis.location.href)), []);
  const runRef = useRef<Promise<BrowserPreviewBenchmarkPayload> | null>(null);
  const [payload, setPayload] = useState<BrowserPreviewBenchmarkPayload | null>(null);

  useEffect(() => {
    runRef.current ??= runBrowserPreviewBenchmarkWithTimeout(request);
    let cancelled = false;
    void runRef.current
      .then((result) => {
        if (!cancelled) setPayload(result);
      })
      .catch((error) => {
        if (!cancelled) setPayload(buildBrowserPreviewBenchmarkFailurePayload(request, error));
      });
    return () => {
      cancelled = true;
    };
  }, [request]);

  const output = payload
    ? JSON.stringify(payload, null, 2)
    : JSON.stringify({ name: "browser-preview-benchmark", status: "running" }, null, 2);

  return (
    <main className="bench-json-page">
      <pre data-browser-preview-benchmark-json>{output}</pre>
      {payload ? (
        <script
          id="browser-preview-benchmark-payload"
          type="application/json"
          dangerouslySetInnerHTML={{ __html: escapeJsonForScript(output) }}
        />
      ) : null}
    </main>
  );
}

function runBrowserPreviewBenchmarkWithTimeout(
  request: BrowserPreviewBenchmarkRequest,
): Promise<BrowserPreviewBenchmarkPayload> {
  const controller = new AbortController();
  let timeoutId: ReturnType<typeof setTimeout> | undefined;
  let activeClient: BrowserPreviewClient | null = null;
  const timeoutError = new Error(`Browser preview benchmark timed out after ${request.timeoutMs}ms.`);
  const timeout = new Promise<never>((_, reject) => {
    timeoutId = setTimeout(() => {
      controller.abort(timeoutError);
      void activeClient?.dispose({ clearSharedBuffers: true });
      reject(timeoutError);
    }, request.timeoutMs);
  });
  const benchmark = runBrowserPreviewBenchmarkWithExclusiveLock(
    readBrowserBenchmarkLocks(),
    () => runBrowserPreviewBenchmark(request, controller.signal, (client) => {
      activeClient = client;
    }),
  );
  return Promise.race([benchmark, timeout])
    .then((payload) => attachBenchmarkTelemetryResult(payload, request))
    .finally(() => {
      if (timeoutId !== undefined) clearTimeout(timeoutId);
    });
}

async function attachBenchmarkTelemetryResult(
  payload: BrowserPreviewBenchmarkPayload,
  request: BrowserPreviewBenchmarkRequest,
): Promise<BrowserPreviewBenchmarkPayload> {
  const result = await submitBenchmarkTelemetry({
    requested: request.benchmarkTelemetryRequested,
    config: request.benchmarkTelemetryConfig,
    benchmarkPayload: payload,
  });
  return {
    ...payload,
    summary: {
      ...payload.summary,
      benchmarkTelemetryRequested: result.requested,
      benchmarkTelemetryConfigured: result.configured,
      benchmarkTelemetrySubmitted: result.submitted,
      ...(result.status !== undefined ? { benchmarkTelemetryStatus: result.status } : {}),
      ...(result.error ? { benchmarkTelemetryError: result.error } : {}),
    },
  };
}

export function runBrowserPreviewBenchmarkWithExclusiveLock<T>(
  locks: BrowserBenchmarkLockManager | undefined,
  task: () => Promise<T>,
): Promise<T> {
  if (!locks?.request) return task();
  return locks.request(BROWSER_BENCHMARK_LOCK_NAME, { mode: "exclusive", ifAvailable: true }, async (lock) => {
    if (!lock) {
      throw new Error("Another browser preview benchmark is already running; close the existing benchmark tab or wait for it to finish before starting another.");
    }
    return task();
  });
}

async function runBrowserPreviewBenchmark(
  request: BrowserPreviewBenchmarkRequest,
  signal?: AbortSignal,
  onClient?: (client: BrowserPreviewClient) => void,
): Promise<BrowserPreviewBenchmarkPayload> {
  if (request.memoryGroundingAuditOnly) {
    return runMemoryGroundingAuditOnlyBenchmark(request, signal);
  }
  assertBenchmarkNotAborted(signal);
  const { client, initLoadMs } = await createPreviewClient(request, signal, onClient);
  assertBenchmarkNotAborted(signal);
  const memoryGrounding = request.memoryGroundingCase
    ? await createMemoryGroundingHarness(request, signal)
    : null;
  assertBenchmarkNotAborted(signal);
  const warmupMs = roundMs(client.lastWarmupMs ?? 0);
  const warmup = {
    warmupMs,
    warmupMode: client.lastWarmupMode,
    warmupBlockingMs: client.lastWarmupProof?.warmupBlockingMs ?? warmupMs,
    warmupUploadedEntries: client.lastWarmupUploadedEntries,
    warmupCacheHits: client.lastWarmupCacheHits,
    residentReadbackCount: client.lastResidentReadbackCount,
  };
  try {
    const runs: BrowserPreviewBenchmarkRun[] = [];
    const runPrompts = buildBrowserPreviewRunPrompts(request);
    for (const prompt of runPrompts) {
      assertBenchmarkNotAborted(signal);
      runs.push(await runPrompt(client, prompt, request, initLoadMs, warmup, memoryGrounding, signal));
      if (request.requireKvPredictivePrefetch && prompt.id.endsWith("-kv-predictive-seed")) {
        await flushKvPersistenceForProof(client, signal);
      }
    }
    if (request.requireKvReuse && request.prompts[0]) {
      assertBenchmarkNotAborted(signal);
      await flushKvPersistenceForProof(client, signal);
      runs.push(await runPrompt(client, {
        id: `${request.prompts[0].id}-kv-reuse`,
        text: request.prompts[0].text,
        expectedSubstrings: request.prompts[0].expectedSubstrings,
        ...(request.prompts[0].expectedExact?.length ? { expectedExact: request.prompts[0].expectedExact } : {}),
      }, request, 0, {
        warmupMs: 0,
        warmupMode: null,
        warmupBlockingMs: 0,
        warmupUploadedEntries: null,
        warmupCacheHits: null,
        residentReadbackCount: null,
      }, memoryGrounding, signal));
    }
    if (request.requireWarmResidentSpeedProof && !request.requireKvReuse && request.prompts[0]) {
      assertBenchmarkNotAborted(signal);
      runs.push(await runPrompt(client, {
        id: `${request.prompts[0].id}-warm-resident-speed`,
        text: request.prompts[0].text,
        expectedSubstrings: request.prompts[0].expectedSubstrings,
        ...(request.prompts[0].expectedExact?.length ? { expectedExact: request.prompts[0].expectedExact } : {}),
      }, request, 0, {
        warmupMs: 0,
        warmupMode: null,
        warmupBlockingMs: 0,
        warmupUploadedEntries: null,
        warmupCacheHits: null,
        residentReadbackCount: null,
      }, memoryGrounding, signal));
    }
    assertBenchmarkNotAborted(signal);
    const benchmarkDeviceInfo = await collectBenchmarkTelemetryBrowserContext().catch(() => null);
    return buildBrowserPreviewBenchmarkPayload({
      createdAt: new Date().toISOString(),
      profile: request.profile.profile,
      runs,
      strictWebGpuRequested: request.strictWebGpuRequested,
      requireKvReuse: request.requireKvReuse,
      requireKvPredictivePrefetch: request.requireKvPredictivePrefetch,
      minGeneratedTokens: request.minGeneratedTokens,
      sourceGitSha: request.benchmarkTelemetryConfig.gitSha ?? null,
      benchmarkDeviceInfo,
      deployUrl: request.benchmarkTelemetryConfig.deployUrl ?? benchmarkDeviceInfo?.deployUrl ?? null,
      technicalProofOnly: (
        request.strictLongPromptProof
        || request.requireKvPredictivePrefetch
      ) && runPrompts.every((prompt) => prompt.expectedSubstrings.length === 0),
    });
  } finally {
    await client.dispose({ clearSharedBuffers: true });
  }
}

export async function runMemoryGroundingAuditOnlyBenchmark(
  request: BrowserPreviewBenchmarkRequest,
  signal?: AbortSignal,
): Promise<BrowserPreviewBenchmarkPayload> {
  assertBenchmarkNotAborted(signal);
  const startedAt = performance.now();
  const harness = await createMemoryGroundingHarness(request, signal);
  const grounded = await Promise.all(request.prompts.map((prompt) =>
    buildMemoryGroundedMessages({ harness, prompt: prompt.text })
  ));
  const proofs = grounded.map((result) => result.proof);
  const retrievalAudit = harness.retrievalAudit;
  const memoryExpectedHitPassed = proofs.every((proof) => proof.expectedMemoryHitPassed);
  const memoryContextRebuildPassed = proofs.every((proof) => proof.contextRebuildPassed);
  const memoryRetrievalAuditRequired = Boolean(retrievalAudit);
  const memoryRetrievalAuditPassed = !memoryRetrievalAuditRequired || retrievalAudit?.passed === true;
  const memoryGeneratedParaphraseRequired = request.memoryGroundingCase === "qa_corpus_v1";
  const memoryGeneratedParaphraseQueryCount = retrievalAudit?.generatedParaphraseQueryCount ?? 0;
  const memoryGeneratedParaphraseRecallAt1 = retrievalAudit?.generatedParaphraseRecallAt1 ?? null;
  const memoryGeneratedParaphrasePassed = !memoryGeneratedParaphraseRequired
    || (memoryGeneratedParaphraseQueryCount > 0 && (memoryGeneratedParaphraseRecallAt1 ?? 0) >= 1);
  const memoryGroundingPassed = memoryExpectedHitPassed
    && memoryContextRebuildPassed
    && memoryRetrievalAuditPassed
    && memoryGeneratedParaphrasePassed;
  const elapsedMs = roundMs(performance.now() - startedAt);
  return {
    name: "browser-preview-benchmark",
    schemaVersion: BROWSER_PREVIEW_BENCHMARK_SCHEMA_VERSION,
    createdAt: new Date().toISOString(),
    passed: memoryGroundingPassed,
    summary: {
      profile: request.profile.profile,
      v12ProductionProofSchemaVersion: BROWSER_PREVIEW_BENCHMARK_SCHEMA_VERSION,
      memoryGroundingAuditOnly: true,
      memoryQueryMode: "seeded_browser_vector_context_rebuild",
      memoryGroundingRequired: true,
      memoryGroundingPassed,
      memoryGroundingCoveragePassed: true,
      memoryGroundedRunCount: proofs.length,
      memoryExpectedHitPassed,
      memoryContextRebuildPassed,
      memoryAnswerOnlyPassed: false,
      memorySeededCorpusCount: harness.corpus.length,
      memoryGeneratedParaphraseRequired,
      memoryGeneratedParaphrasePassed,
      memoryGeneratedParaphraseQueryCount,
      memoryGeneratedParaphraseTop1CorrectCount: retrievalAudit?.generatedParaphraseTop1CorrectCount ?? 0,
      memoryGeneratedParaphraseRecallAt1,
      memoryGeneratedParaphraseMrr: retrievalAudit?.generatedParaphraseMrr ?? null,
      memoryRetrievedCount: proofs.reduce((total, proof) => total + proof.retrievedMemoryIds.length, 0),
      memoryIncludedCount: proofs.reduce((total, proof) => total + proof.includedMemoryIds.length, 0),
      memoryContextEstimatedTokens: Math.max(0, ...proofs.map((proof) => proof.contextEstimatedTokens)),
      memoryRetrievalMs: roundMs(proofs.reduce((total, proof) => total + proof.retrievalMs, 0) / Math.max(1, proofs.length)),
      memoryContextRebuildMs: roundMs(proofs.reduce((total, proof) => total + proof.contextRebuildMs, 0) / Math.max(1, proofs.length)),
      memoryRetrievalAuditRequired,
      memoryRetrievalAuditPassed,
      memoryRetrievalAuditQueryCount: retrievalAudit?.queryCount ?? 0,
      memoryRetrievalAuditTop1CorrectCount: retrievalAudit?.top1CorrectCount ?? 0,
      memoryRetrievalAuditElapsedMs: retrievalAudit?.elapsedMs ?? null,
      memoryRecallAt1: retrievalAudit?.recallAt1 ?? null,
      memoryMrr: retrievalAudit?.mrr ?? null,
      memoryMinTopScoreMargin: retrievalAudit?.minTopScoreMargin ?? null,
      memoryMeanExpectedHitRank: retrievalAudit?.meanExpectedHitRank ?? null,
      memoryExpectedHitMeanRank: roundMs(proofs.reduce((total, proof) => total + (proof.retrievalRank ?? MEMORY_GROUNDING_RETRIEVAL_LIMIT + 1), 0) / Math.max(1, proofs.length)),
      promptCount: request.prompts.length,
      technicalProofOnly: true,
      productionQualityPassed: false,
      productionDeployReadyPassed: false,
      meanInitLoadMs: 0,
      meanWarmupMs: 0,
      meanWarmupBlockingMs: 0,
      meanPrefillMs: 0,
      meanTimeToFirstTokenMs: elapsedMs,
      meanDecodeLatencyMs: 0,
      meanTokensPerSecond: null,
      strictWebGpuRequested: false,
      strictWebGpuPassed: false,
      cpuFallbackUsed: false,
      noCpuFallback: true,
      positiveWebGpuKernelProof: false,
      webGpuFailedGateCount: 0,
      mtpMode: "target_only",
      mtpAcceptanceRate: 0,
      mtpMaxSpeculativeTokens: 0,
      mtpMeanSpeculativeTokens: 0,
      mtpVerifiedTokenCount: 0,
      mtpTargetDecodeCalls: 0,
      mtpVerifierStrategy: "none",
      kvPersistenceEventCount: 0,
      kvPersistEventCount: 0,
      kvHydrateEventCount: 0,
      kvReuseEventCount: 0,
      kvPersistDeferred: false,
      kvPersistCriticalPathMs: 0,
      kvPersistFlushMs: 0,
      kvPersistPendingBlockCount: 0,
      kvPrefetchStrategy: "none",
      kvExactReuseRunCount: 0,
      kvPredictivePrefetchRunCount: 0,
      kvMissStallRunCount: 0,
      kvNoPrefetchRunCount: 0,
      kvLowRankSummaryRank: null,
      kvLowRankQuerySource: "none",
      kvPredictedHotBlockCount: 0,
      kvPrefetchedBlockCount: 0,
      kvPrefetchHitRate: null,
      kvPrefetchBytes: 0,
      kvPrefetchLatencyMs: 0,
      kvAttentionStallMs: 0,
      primarySpeedBottleneck: "memory_retrieval_audit_only",
    },
    runs: [],
  };
}

async function flushKvPersistenceForProof(
  client: BrowserPreviewClient,
  signal?: AbortSignal,
): Promise<void> {
  assertBenchmarkNotAborted(signal);
  await client.flushKvPersistence();
  assertBenchmarkNotAborted(signal);
}

export function buildBrowserPreviewRunPrompts(
  request: BrowserPreviewBenchmarkRequest,
): Array<{ id: string; text: string; expectedSubstrings: string[]; expectedExact?: string[] }> {
  const prompts = [...request.prompts];
  if (!request.requireKvPredictivePrefetch || !request.prompts[0]) return prompts;
  const seed = request.prompts[0];
  return [
    {
      id: `${seed.id}-kv-predictive-seed`,
      text: request.memoryGroundingCase
        ? buildMemoryGroundedPredictiveKvSeedPrompt(seed.text, request.memoryGroundingCase)
        : buildPredictiveKvSeedPrompt(seed.text),
      expectedSubstrings: request.memoryGroundingCase ? seed.expectedSubstrings : [],
      ...(request.memoryGroundingCase && seed.expectedExact?.length ? { expectedExact: seed.expectedExact } : {}),
    },
    ...prompts,
  ];
}

function buildPredictiveKvSeedPrompt(prompt: string): string {
  const trimmed = prompt.trim() || "alpha beta runtime cache";
  return `${trimmed} runtime cache warm seed for low rank predictive kv prefetch`;
}

function buildMemoryGroundedPredictiveKvSeedPrompt(
  prompt: string,
  caseId: BrowserMemoryGroundingCase,
): string {
  const fact = caseId === "montana_capital"
    ? QA_GROUNDING_FACTS[0]
    : groundingFactForPrompt(prompt, groundingFactsForCase(caseId));
  if (!fact) return prompt;
  return `Using retrieved memory only, ${fact.question} ${fact.answerInstruction} Use the persisted answer token only.`;
}

async function createPreviewClient(
  request: BrowserPreviewBenchmarkRequest,
  signal?: AbortSignal,
  onClient?: (client: BrowserPreviewClient) => void,
): Promise<{ client: BrowserPreviewClient; initLoadMs: number }> {
  assertBenchmarkNotAborted(signal);
  const initStart = performance.now();
  if (request.backendId === "compiled-browser-webllm") {
    if (!COMPILED_WEBLLM_ENABLED) {
      throw new Error("Compiled browser backend requested but VITE_COMPILED_WEBLLM_ENABLED is not true for this build.");
    }
    const client = new CompiledWebLlmClient({
      modelId: request.modelId,
      qwenThinkingMode: request.qwenThinkingMode,
    });
    onClient?.(client);
    try {
      await waitForBenchmarkTask(client.init(), signal);
      assertBenchmarkNotAborted(signal);
      return { client, initLoadMs: roundMs(performance.now() - initStart) };
    } catch (error) {
      await client.dispose();
      throw error;
    }
  }
  if (request.backendId !== "unlocked-browser-transformer") {
    throw new Error(`Browser preview benchmark backend "${request.backendId}" is not an answer-generation backend for this route.`);
  }
  const client = new UnlockedBrowserTransformerWorkerClient({
    modelId: request.modelId,
    manifestPath: UNLOCKED_MODEL_MANIFEST_PATH,
    manifestSha256: UNLOCKED_MODEL_MANIFEST_SHA256,
    allowFixtureWeights: UNLOCKED_ALLOW_FIXTURE,
    backendPreference: request.backendPreference ?? parseBackendPreference(UNLOCKED_BACKEND_PREFERENCE),
    ...(request.strictWebGpuRequested ? { requireWebGpu: true } : {}),
    maxRuntimePromptTokens: request.profile.caps.maxRuntimePromptTokens,
    maxRuntimeLayers: request.profile.caps.maxRuntimeLayers,
    logitCandidateLimit: request.profile.caps.logitCandidateLimit,
    ...(request.logitTopK ? { logitTopK: request.logitTopK } : {}),
    ...(request.logitTileRows ? { logitTileRows: request.logitTileRows } : {}),
    warmModelResidency: request.warmModelResidency,
    warmModelResidencyMode: request.warmModelResidencyMode,
    maxGenerationTokens: request.profile.caps.maxGenerationTokens ?? CHAT_MAX_GENERATION_TOKENS,
    qwenThinkingMode: request.qwenThinkingMode,
    strictChunkedPrefill: request.strictLongPromptProof,
    mtp: {
      enabled: request.mtpEnabled,
      draftModelId: MTP_DRAFT_MODEL_ID,
      numSpeculativeTokens: MTP_NUM_SPECULATIVE_TOKENS,
      minAcceptanceRate: MTP_MIN_ACCEPTANCE_RATE,
      disableWhenLatencyWorse: MTP_DISABLE_WHEN_LATENCY_WORSE,
      draftLayerCount: MTP_DRAFT_LAYER_COUNT,
    },
    kvPersistence: {
      enabled: KVSWAP_PERSISTENCE_ENABLED,
      namespace: request.kvNamespace,
      preferOpfs: KVSWAP_PERSISTENCE_PREFER_OPFS,
      maxBlocks: KVSWAP_PERSISTENCE_MAX_BLOCKS,
      maxBytes: KVSWAP_PERSISTENCE_MAX_BYTES,
      clearOnInit: request.requireKvReuse ? false : KVSWAP_PERSISTENCE_CLEAR_ON_INIT,
    },
  });
  onClient?.(client);
  try {
    await waitForBenchmarkTask(client.init(), signal);
    assertBenchmarkNotAborted(signal);
    return { client, initLoadMs: roundMs(performance.now() - initStart) };
  } catch (error) {
    await client.dispose({ clearSharedBuffers: true });
    throw error;
  }
}

export async function createMemoryGroundingHarness(
  request: BrowserPreviewBenchmarkRequest,
  signal?: AbortSignal,
): Promise<BrowserMemoryGroundingHarness> {
  assertBenchmarkNotAborted(signal);
  const caseId = request.memoryGroundingCase ?? "montana_capital";
  const store = new IndexedDbMemoryStore(`edge-ai-browser-memory-bench-${safeDbName(request.kvNamespace)}`);
  await store.clear();
  assertBenchmarkNotAborted(signal);
  const corpus = buildSeededMemoryCorpus(caseId, request.memoryGroundingCorpusSize);
  await store.upsert(corpus);
  assertBenchmarkNotAborted(signal);
  const retrievalAudit = caseId === "large_synthetic_v1" || caseId === "qa_corpus_v1"
    ? await runMemoryRetrievalAudit({
      caseId,
      corpusCount: corpus.length,
      store,
      ...(signal ? { signal } : {}),
    })
    : undefined;
  return {
    caseId,
    store,
    corpus,
    ...(retrievalAudit ? { retrievalAudit } : {}),
  };
}

async function runMemoryRetrievalAudit(input: {
  caseId: BrowserMemoryGroundingCase;
  corpusCount: number;
  store: IndexedDbMemoryStore;
  signal?: AbortSignal;
}): Promise<BrowserMemoryRetrievalAuditProof> {
  const startedAt = performance.now();
  let top1CorrectCount = 0;
  let reciprocalRankSum = 0;
  let totalExpectedRank = 0;
  const topScoreMargins: number[] = [];
  const facts = groundingFactsForCase(input.caseId);
  const classStats = new Map<BrowserMemoryGroundingAuditQueryClass, BrowserMemoryGroundingAuditClassStats>();
  for (const fact of facts) {
    for (const auditQuery of groundingAuditQueriesForFact(fact, input.caseId)) {
      assertBenchmarkNotAborted(input.signal);
      const resolvedQuery = memoryGroundingQueryText(auditQuery.query, input.caseId);
      const hits = await input.store.search(embedMemoryGroundingText(resolvedQuery), {
        limit: MEMORY_GROUNDING_RETRIEVAL_LIMIT,
        minScore: 0,
        tenantId: MEMORY_TENANT_ID,
        cellId: MEMORY_CELL_ID,
        sessionId: MEMORY_GROUNDING_SESSION_ID,
      });
      const rankIndex = hits.findIndex((hit) => hit.id === fact.id);
      const rank = rankIndex >= 0 ? rankIndex + 1 : 0;
      if (rank === 1) top1CorrectCount += 1;
      reciprocalRankSum += rank > 0 ? 1 / rank : 0;
      const stats = getGroundingAuditClassStats(classStats, auditQuery.queryClass);
      stats.queryCount += 1;
      if (rank === 1) stats.top1CorrectCount += 1;
      stats.reciprocalRankSum += rank > 0 ? 1 / rank : 0;
      totalExpectedRank += rank > 0 ? rank : MEMORY_GROUNDING_RETRIEVAL_LIMIT + 1;
      const topScore = hits[0]?.score;
      const secondScore = hits[1]?.score;
      if (isFiniteNumber(topScore) && isFiniteNumber(secondScore)) {
        topScoreMargins.push(topScore - secondScore);
      }
    }
  }
  const queryCount = [...classStats.values()].reduce((count, stats) => count + stats.queryCount, 0);
  const recallAt1 = queryCount > 0 ? top1CorrectCount / queryCount : 0;
  const mrr = queryCount > 0 ? reciprocalRankSum / queryCount : 0;
  const queryClassBreakdown = buildGroundingAuditClassBreakdown(classStats);
  const canonical = classStats.get("canonical");
  const alias = classStats.get("alias");
  const generatedParaphrase = classStats.get("generated_paraphrase");
  return {
    corpusCount: input.corpusCount,
    queryCount,
    top1CorrectCount,
    reciprocalRankSum: roundMetric(reciprocalRankSum),
    recallAt1: roundMetric(recallAt1),
    mrr: roundMetric(mrr),
    ...(canonical ? {
      canonicalQueryCount: canonical.queryCount,
      canonicalTop1CorrectCount: canonical.top1CorrectCount,
      canonicalRecallAt1: roundMetric(canonical.top1CorrectCount / canonical.queryCount),
      canonicalMrr: roundMetric(canonical.reciprocalRankSum / canonical.queryCount),
    } : {}),
    ...(alias ? {
      aliasQueryCount: alias.queryCount,
      aliasTop1CorrectCount: alias.top1CorrectCount,
      aliasRecallAt1: roundMetric(alias.top1CorrectCount / alias.queryCount),
      aliasMrr: roundMetric(alias.reciprocalRankSum / alias.queryCount),
    } : {}),
    ...(generatedParaphrase ? {
      generatedParaphraseQueryCount: generatedParaphrase.queryCount,
      generatedParaphraseTop1CorrectCount: generatedParaphrase.top1CorrectCount,
      generatedParaphraseRecallAt1: roundMetric(generatedParaphrase.top1CorrectCount / generatedParaphrase.queryCount),
      generatedParaphraseMrr: roundMetric(generatedParaphrase.reciprocalRankSum / generatedParaphrase.queryCount),
    } : {}),
    queryClassBreakdown,
    minTopScoreMargin: topScoreMargins.length > 0 ? roundMetric(Math.min(...topScoreMargins)) : null,
    meanExpectedHitRank: queryCount > 0 ? roundMetric(totalExpectedRank / queryCount) : null,
    passed: recallAt1 >= LARGE_SYNTHETIC_MIN_RECALL_AT_1,
    elapsedMs: roundMs(performance.now() - startedAt),
    minRequiredRecallAt1: LARGE_SYNTHETIC_MIN_RECALL_AT_1,
  };
}

function getGroundingAuditClassStats(
  classStats: Map<BrowserMemoryGroundingAuditQueryClass, BrowserMemoryGroundingAuditClassStats>,
  queryClass: BrowserMemoryGroundingAuditQueryClass,
): BrowserMemoryGroundingAuditClassStats {
  const existing = classStats.get(queryClass);
  if (existing) return existing;
  const next = { queryCount: 0, top1CorrectCount: 0, reciprocalRankSum: 0 };
  classStats.set(queryClass, next);
  return next;
}

function buildGroundingAuditClassBreakdown(
  classStats: Map<BrowserMemoryGroundingAuditQueryClass, BrowserMemoryGroundingAuditClassStats>,
): NonNullable<BrowserMemoryRetrievalAuditProof["queryClassBreakdown"]> {
  return (["canonical", "alias", "generated_paraphrase"] as const)
    .map((queryClass) => {
      const stats = classStats.get(queryClass);
      if (!stats || stats.queryCount <= 0) return null;
      return {
        queryClass,
        queryCount: stats.queryCount,
        top1CorrectCount: stats.top1CorrectCount,
        recallAt1: roundMetric(stats.top1CorrectCount / stats.queryCount),
        mrr: roundMetric(stats.reciprocalRankSum / stats.queryCount),
      };
    })
    .filter((item): item is NonNullable<BrowserMemoryRetrievalAuditProof["queryClassBreakdown"]>[number] => Boolean(item));
}

function groundingAuditQueriesForFact(
  fact: BrowserGroundingFact,
  caseId: BrowserMemoryGroundingCase,
): BrowserMemoryGroundingAuditQuery[] {
  const canonicalQuery = `${fact.question} ${fact.promptNeedles.join(" ")}`;
  const canonical = { query: canonicalQuery, queryClass: "canonical" as const };
  if (caseId !== "qa_corpus_v1") return [canonical];
  return [
    canonical,
    ...(fact.queryAliases ?? []).map((query) => ({ query, queryClass: "alias" as const })),
    ...generatedGroundingParaphrasesForFact(fact).map((query) => ({
      query,
      queryClass: "generated_paraphrase" as const,
    })),
  ];
}

function generatedGroundingParaphrasesForFact(fact: BrowserGroundingFact): string[] {
  const meaningfulTags = fact.tags
    .filter((tag) => !["bench", "fact", "pinned"].includes(tag))
    .slice(0, 3)
    .join(" ");
  const needleSummary = fact.promptNeedles.join(" ");
  return [
    `Using retrieved memory only, what answer token belongs to ${needleSummary}? Answer only.`,
    `Find the ${meaningfulTags || needleSummary} memory record and return its stored answer. Answer only.`,
  ];
}

export function readBrowserPreviewBenchmarkRequest(url: URL): BrowserPreviewBenchmarkRequest {
  const backendId = readFirstSearchParam(url, ["backend", "backendId", "llmBackend"]) ?? DEFAULT_LLM_BACKEND;
  const modelId = readFirstSearchParam(url, ["modelId", "model"]) ?? DEFAULT_MODEL;
  const resolvedProfile = resolveUnlockedRuntimeProfile({
    VITE_UNLOCKED_RUNTIME_PROFILE: url.searchParams.get("profile") ?? undefined,
    VITE_UNLOCKED_MAX_RUNTIME_PROMPT_TOKENS: readFirstSearchParam(url, [
      "maxRuntimePromptTokens",
      "promptTokens",
    ]),
    VITE_UNLOCKED_MAX_RUNTIME_LAYERS: readFirstSearchParam(url, [
      "maxRuntimeLayers",
      "layers",
    ]),
    VITE_UNLOCKED_LOGIT_CANDIDATE_LIMIT: readFirstSearchParam(url, [
      "logitCandidateLimit",
      "candidateLogits",
    ]),
    VITE_UNLOCKED_MAX_GENERATION_TOKENS: readFirstSearchParam(url, [
      "generationTokens",
      "maxGenerationTokens",
      "maxTokens",
    ]),
  });
  const gates = (url.searchParams.get("webGpuGates") ?? "")
    .split(",")
    .map((gate) => gate.trim())
    .filter(isWebGpuGate);
  const strictWebGpu = url.searchParams.get("strictWebGpu") === "true";
  const webGpuGates = resolveRequestedWebGpuGates({
    gates,
    strictWebGpu,
    requireWebGpuKernels: backendId === "unlocked-browser-transformer" && REQUIRE_WEBGPU_KERNELS,
  });
  const longPromptTargetTokens = readPositiveSearchParam(url, ["longPromptTargetTokens", "targetPromptTokens"], MAX_LONG_PROMPT_TARGET_TOKENS);
  const longPromptRepeat = readPositiveSearchParam(url, ["longPromptRepeat", "promptRepeat"], MAX_LONG_PROMPT_REPEAT);
  const memoryGroundingAuditOnly = parseBooleanSearchParam(url, [
    "memoryGroundingAuditOnly",
    "memoryAuditOnly",
    "databaseAuditOnly",
  ]);
  const parsedMemoryGroundingCase = parseMemoryGroundingCase(readFirstSearchParam(url, [
    "memoryGrounding",
    "groundedMemory",
    "memoryCase",
  ]));
  const memoryGroundingCase = parsedMemoryGroundingCase ?? (memoryGroundingAuditOnly ? "large_synthetic_v1" : null);
  const memoryGroundingCorpusSize = readPositiveSearchParam(url, [
    "memoryCorpusSize",
    "memoryGroundingCorpusSize",
    "groundedCorpusSize",
  ], MAX_MEMORY_GROUNDING_CORPUS_SIZE) ?? DEFAULT_MEMORY_GROUNDING_CORPUS_SIZE;
  const memoryGroundingPromptLimit = readPositiveSearchParam(url, [
    "memoryPromptLimit",
    "memoryGroundingPromptLimit",
    "groundedPromptLimit",
  ], 64);
  const logitProofProfile = webGpuGates.includes("logits")
    ? withFullVocabLogitProofCaps(resolvedProfile)
    : resolvedProfile;
  const prompts = limitMemoryGroundingPrompts(
    withMemoryGroundingPromptDefaults(readPrompts(url, {
      ...(longPromptTargetTokens ? { longPromptTargetTokens } : {}),
      ...(longPromptRepeat ? { longPromptRepeat } : {}),
    }), memoryGroundingCase, hasExplicitPrompt(url)),
    memoryGroundingCase,
    memoryGroundingPromptLimit,
  );
  const strictWebGpuRequested = webGpuGates.length > 0
    || strictWebGpu
    || (backendId === "unlocked-browser-transformer" && REQUIRE_WEBGPU_KERNELS);
  const strictExpectedLayerMode = parseStrictExpectedLayerMode(
    url.searchParams.get("strictExpectedLayers") ?? url.searchParams.get("expectedLayerMode"),
    prompts.some((prompt) => prompt.expectedSubstrings.length > 0),
  );
  const profile = strictWebGpuRequested
    && strictExpectedLayerMode === "full"
    && prompts.some((prompt) => prompt.expectedSubstrings.length > 0)
    ? withStrictExpectedSubstringLayerCaps(logitProofProfile)
    : logitProofProfile;
  const requireKvReuse = url.searchParams.get("requireKvReuse") === "true";
  const requireWarmResidentSpeedProof = parseBooleanSearchParam(url, [
    "requireWarmResidentSpeedProof",
    "warmResidentSpeedProof",
    "warmSpeedProof",
  ]) || url.searchParams.get("speedProof")?.trim().toLowerCase() === "warm_resident";
  const requireKvPredictivePrefetch = parseBooleanSearchParam(url, [
    "requireKvPredictivePrefetch",
    "kvPredictiveProof",
    "strictKvPredictiveProof",
  ]);
  const kvNamespace = url.searchParams.get("kvNamespace")?.trim()
    || `${MEMORY_TENANT_ID}:${MEMORY_CELL_ID}:browser-preview-bench:${Date.now().toString(36)}`;
  const minGeneratedTokens = readPositiveSearchParam(url, ["minGeneratedTokens", "minTokens"]) ?? 1;
  const timeoutMs = readPositiveSearchParam(url, ["timeoutMs", "timeout"]) ?? 120_000;
  const strictLongPromptProof = parseBooleanSearchParam(url, ["strictLongPrompt", "requireLongPromptProof"]);
  const backendPreference = parseBackendPreference(url.searchParams.get("backendPreference") ?? UNLOCKED_BACKEND_PREFERENCE);
  const qwenThinkingMode = parseQwenThinkingMode(url.searchParams.get("qwenThinkingMode") ?? QWEN_THINKING_MODE);
  const mtpEnabled = parseMtpEnabled(url.searchParams.get("mtp") ?? url.searchParams.get("mtpEnabled"));
  const samplingTemperature = readNumberSearchParam(url, ["temperature", "samplingTemperature"]);
  const samplingTopP = readNumberSearchParam(url, ["topP", "samplingTopP"]);
  const repetitionPenalty = readNumberSearchParam(url, ["repetitionPenalty", "repeatPenalty"]);
  const samplingSeed = readPositiveSearchParam(url, ["samplingSeed", "seed"]);
  const qualityDecode = parseBooleanSearchParam(url, ["qualityDecode", "qualitySampling"])
    || url.searchParams.get("decodeProfile")?.trim().toLowerCase() === "quality"
    || samplingTemperature !== undefined
    || samplingTopP !== undefined
    || repetitionPenalty !== undefined;
  const logitTopK = readPositiveSearchParam(url, ["logitTopK", "topK"])
    ?? (qualityDecode ? 40 : webGpuGates.includes("logits") ? 1 : undefined);
  const explicitLogitTileRows = readPositiveSearchParam(url, ["logitTileRows", "tileRows"]);
  const logitTileRows = explicitLogitTileRows ?? (logitTopK === 1 ? BENCHMARK_GREEDY_LOGIT_TILE_ROWS : null);
  const warmModelResidencyParamNames = [
    "warmModelResidency",
    "warmRuntimeResidency",
    "warmupModelResidency",
  ];
  const warmModelResidencyValue = readFirstSearchParam(url, warmModelResidencyParamNames);
  const warmModelResidency = warmModelResidencyValue
    ? parseBooleanSearchValue(warmModelResidencyValue)
    : strictWebGpuRequested;
  const warmModelResidencyMode = parseWarmupMode(url.searchParams.get("warmupMode") ?? url.searchParams.get("warmModelResidencyMode"));
  const benchmarkTelemetryRequested = parseBooleanSearchParam(url, [
    "submitTelemetry",
    "benchmarkTelemetry",
    "saveBenchmark",
  ]);
  return {
    backendId,
    modelId,
    profile,
    prompts,
    ...(backendPreference ? { backendPreference } : {}),
    strictWebGpuRequested,
    webGpuGates,
    requireKvReuse,
    requireWarmResidentSpeedProof,
    requireKvPredictivePrefetch,
    kvNamespace,
    qwenThinkingMode,
    mtpEnabled,
    ...(logitTopK ? { logitTopK } : {}),
    ...(logitTileRows ? { logitTileRows } : {}),
    ...(samplingTemperature !== undefined ? { samplingTemperature } : {}),
    ...(samplingTopP !== undefined ? { samplingTopP } : {}),
    ...(repetitionPenalty !== undefined ? { repetitionPenalty } : {}),
    ...(samplingSeed !== undefined ? { samplingSeed } : {}),
    warmModelResidency,
    warmModelResidencyMode,
    minGeneratedTokens,
    timeoutMs,
    ...(longPromptTargetTokens ? { longPromptTargetTokens } : {}),
    ...(longPromptRepeat ? { longPromptRepeat } : {}),
    strictLongPromptProof,
    strictExpectedLayerMode,
    memoryGroundingCase,
    memoryGroundingCorpusSize,
    ...(memoryGroundingPromptLimit ? { memoryGroundingPromptLimit } : {}),
    memoryGroundingAuditOnly,
    benchmarkTelemetryRequested,
    benchmarkTelemetryConfig: BENCHMARK_TELEMETRY_CONFIG,
  };
}

export async function buildMemoryGroundedMessages(input: {
  harness: BrowserMemoryGroundingHarness;
  prompt: string;
}): Promise<{ messages: ChatClientMessage[]; proof: BrowserMemoryGroundingProof }> {
  const queryEmbedding = embedMemoryGroundingText(memoryGroundingQueryText(input.prompt, input.harness.caseId));
  const retrievalStartedAt = performance.now();
  const retrievedMemory = await input.harness.store.search(queryEmbedding, {
    limit: input.harness.caseId === "large_synthetic_v1" ? MEMORY_GROUNDING_RETRIEVAL_LIMIT : 1,
    minScore: 0,
    tenantId: MEMORY_TENANT_ID,
    cellId: MEMORY_CELL_ID,
    sessionId: MEMORY_GROUNDING_SESSION_ID,
  });
  const retrievalMs = roundMs(performance.now() - retrievalStartedAt);
  const contextStartedAt = performance.now();
  const { packed } = buildContextRuntimePlan({
    requestId: `grounded_${Date.now().toString(36)}`,
    systemPrompt: "You are a local browser agent. Answer only from retrieved long-term memory. If the retrieved memory contains an Answer token, copy that exact answer and stop.",
    userMessage: input.prompt,
    recentMessages: [],
    retrievedMemory,
    maxRetrievedMemoryTokens: 600,
    maxRecentConversationTokens: 0,
    maxPromptTokens: 1400,
  });
  const contextRebuildMs = roundMs(performance.now() - contextStartedAt);
  const included = new Set(packed.includedMemoryIds);
  const retrieved = new Set(retrievedMemory.map((memory) => memory.id));
  const expectedMemoryIds = expectedMemoryIdsForPrompt(input.harness.caseId, input.prompt);
  const expectedMemoryAvailable = expectedMemoryIds.length > 0;
  const expectedRank = expectedMemoryAvailable
    ? retrievedMemory.findIndex((memory) => expectedMemoryIds.includes(memory.id)) + 1
    : null;
  const expectedHit = expectedRank && expectedRank > 0 ? retrievedMemory[expectedRank - 1] : undefined;
  const topScore = retrievedMemory[0]?.score;
  const secondScore = retrievedMemory[1]?.score;
  const topScoreMargin = isFiniteNumber(topScore) && isFiniteNumber(secondScore)
    ? roundMs(topScore - secondScore)
    : null;
  const expectedMemoryHitPassed = expectedMemoryAvailable && expectedMemoryIds.every((id) => retrieved.has(id));
  const contextRebuildPassed = expectedMemoryAvailable && expectedMemoryIds.every((id) => included.has(id));
  return {
    messages: packed.messages.map((message) => ({
      role: message.role as ChatMessage["role"],
      content: message.content,
    })),
    proof: {
      mode: "seeded_browser_vector_context_rebuild",
      caseId: input.harness.caseId,
      corpusCount: input.harness.corpus.length,
      retrievedMemoryIds: retrievedMemory.map((memory) => memory.id),
      includedMemoryIds: packed.includedMemoryIds,
      expectedMemoryIds,
      expectedMemoryHitPassed,
      contextRebuildPassed,
      answerOnlyExpected: true,
      contextEstimatedTokens: packed.estimatedTokens,
      retrievalMs,
      contextRebuildMs,
      retrievalRank: expectedRank && expectedRank > 0 ? expectedRank : null,
      retrievalScore: isFiniteNumber(expectedHit?.score) ? roundMs(expectedHit.score) : null,
      retrievalTopScoreMargin: topScoreMargin,
      ...(input.harness.retrievalAudit ? { retrievalAudit: input.harness.retrievalAudit } : {}),
    },
  };
}

interface BrowserGroundingFact {
  marker: string;
  id: string;
  question: string;
  answer: string;
  answerInstruction: string;
  document: string;
  promptNeedles: readonly string[];
  queryAliases?: readonly string[];
  uniqueTokens: readonly string[];
  tags: readonly string[];
}

const QA_GROUNDING_FACTS: readonly BrowserGroundingFact[] = [
  {
    marker: "MEMORY_FACT_MONTANA_CAPITAL",
    id: "bench_memory_montana_capital",
    question: "In the Cedar Ridge operations dossier, which city is listed as the Montana field office hub?",
    answer: "Helena",
    answerInstruction: "Answer with only the city.",
    document: "Cedar Ridge operations dossier section 14 says the Montana field office hub for agency routing is Helena.",
    promptNeedles: ["cedar ridge", "montana", "field office hub"],
    queryAliases: [
      "capital of montana",
      "montana capital",
      "capital city of montana",
      "state capital of montana",
      "what is the capital of montana",
    ],
    uniqueTokens: ["cedar", "ridge"],
    tags: ["bench", "fact", "pinned", "montana", "cedar-ridge", "operations-dossier"],
  },
  {
    marker: "MEMORY_FACT_EDGE_RUNTIME_SENTINEL",
    id: "bench_memory_edge_runtime_sentinel",
    question: "What exact browser runtime sentinel does the Edge Runtime deployment check require?",
    answer: "edge-runtime-ok",
    answerInstruction: "Return exactly the sentinel token.",
    document: "Edge Runtime deployment checklist says the exact browser production sentinel token is edge-runtime-ok.",
    promptNeedles: ["edge runtime", "deployment check", "sentinel"],
    uniqueTokens: ["sentinel"],
    tags: ["bench", "fact", "pinned", "edge-runtime", "production-sentinel"],
  },
  {
    marker: "MEMORY_FACT_DESERT_LANTERN_ARCHIVE_COLOR",
    id: "bench_memory_desert_lantern_archive_color",
    question: "In the Desert Lantern onboarding memo, what color is assigned to the secure archive?",
    answer: "amber",
    answerInstruction: "Answer with only the color.",
    document: "Desert Lantern onboarding memo alpha assigns amber as the secure archive color for local retrieval drills.",
    promptNeedles: ["desert lantern", "secure archive", "color"],
    uniqueTokens: ["desert", "lantern"],
    tags: ["bench", "fact", "pinned", "desert-lantern", "onboarding-memo"],
  },
  {
    marker: "MEMORY_FACT_ORBITAL_PIER_CALIBRATION_CODE",
    id: "bench_memory_orbital_pier_calibration_code",
    question: "Which calibration code does the Orbital Pier maintenance note require?",
    answer: "Vela-42",
    answerInstruction: "Answer with only the code.",
    document: "Orbital Pier maintenance note requires calibration code Vela-42 before the browser runtime proof is accepted.",
    promptNeedles: ["orbital pier", "calibration code", "maintenance note"],
    uniqueTokens: ["orbital", "pier"],
    tags: ["bench", "fact", "pinned", "orbital-pier", "maintenance-note"],
  },
  {
    marker: "MEMORY_FACT_MAPLE_LOCK_RECOVERY_CONTACT",
    id: "bench_memory_maple_lock_recovery_contact",
    question: "Who is the recovery contact named in the Maple Lock incident memo?",
    answer: "Nora Vale",
    answerInstruction: "Answer with only the name.",
    document: "Maple Lock incident memo names Nora Vale as the recovery contact for database handoff verification.",
    promptNeedles: ["maple lock", "recovery contact", "incident memo"],
    uniqueTokens: ["maple", "lock"],
    tags: ["bench", "fact", "pinned", "maple-lock", "incident-memo"],
  },
  {
    marker: "MEMORY_FACT_TIDAL_FORGE_RETENTION_WINDOW",
    id: "bench_memory_tidal_forge_retention_window",
    question: "What retention window does the Tidal Forge storage note specify?",
    answer: "37 days",
    answerInstruction: "Answer with only the duration.",
    document: "Tidal Forge storage note specifies a 37 days retention window for persisted browser memory rows.",
    promptNeedles: ["tidal forge", "retention window", "storage note"],
    uniqueTokens: ["tidal", "forge"],
    tags: ["bench", "fact", "pinned", "tidal-forge", "storage-note"],
  },
] as const;

const LARGE_SYNTHETIC_GROUNDING_FACTS: readonly BrowserGroundingFact[] = Array.from(
  { length: LARGE_SYNTHETIC_GROUNDING_FACT_COUNT },
  (_value, index): BrowserGroundingFact => {
    const ordinal = String(index + 1).padStart(4, "0");
    const shard = `synth${ordinal}`;
    const answer = `Aster-${ordinal}`;
    return {
      marker: `MEMORY_FACT_SYNTHETIC_${ordinal}`,
      id: `bench_memory_synthetic_${ordinal}`,
      question: `In the Helix Ledger synthetic corpus, what answer code is assigned to shard ${shard}?`,
      answer,
      answerInstruction: "Answer with only the code.",
      document: `Helix Ledger synthetic corpus entry ${shard} assigns answer code ${answer} for browser-local vector retrieval validation.`,
      promptNeedles: ["helix ledger", "synthetic corpus", shard, "answer code"],
      uniqueTokens: [shard],
      tags: ["bench", "fact", "pinned", "large-synthetic", shard],
    };
  },
);

const ALL_GROUNDING_FACTS: readonly BrowserGroundingFact[] = [
  ...QA_GROUNDING_FACTS,
  ...LARGE_SYNTHETIC_GROUNDING_FACTS,
];

export function buildSeededMemoryCorpus(
  caseId: BrowserMemoryGroundingCase,
  requestedCorpusSize = DEFAULT_MEMORY_GROUNDING_CORPUS_SIZE,
): MemoryChunk[] {
  const facts = groundingFactsForCase(caseId);
  const pinnedFacts = facts.map((fact) => makeGroundingChunk(
    fact.id,
    caseId === "montana_capital"
      ? `${fact.marker}: ${fact.document} Answer token: ${fact.answer}.`
      : `${fact.document} Answer token: ${fact.answer}.`,
    [...fact.tags],
    {
      rawMemoryId: `raw_${fact.id}`,
      identityPinId: `pin_${fact.id}`,
      memoryClass: "PINNED_EXACT",
      mustAttend: true,
      identityRisk: 1,
      pinStrength: 1,
      groundingCase: caseId,
      expectedAnswer: fact.answer,
    },
  ));
  const support = [
    makeGroundingChunk(
      "bench_memory_browser_runtime",
      "Browser WebGPU proof records strict kernels, KV reuse, and context rebuild traces separately from answer quality.",
      ["bench", "runtime"],
    ),
  ];
  const minimumCorpusSize = caseId === "montana_capital"
    ? 16
    : caseId === "large_synthetic_v1"
      ? Math.max(LARGE_SYNTHETIC_MIN_CORPUS_SIZE, requestedCorpusSize)
      : Math.max(64, requestedCorpusSize);
  const targetCorpusSize = Math.max(minimumCorpusSize, pinnedFacts.length + support.length);
  const distractorCount = targetCorpusSize - pinnedFacts.length - support.length;
  return [
    ...pinnedFacts,
    ...support,
    ...Array.from({ length: distractorCount }, (_value, index) => makeGroundingChunk(
      `bench_memory_distractor_${index + 1}`,
      `Distractor memory ${index + 1}: local browser storage record about runtime weather, cache policy, interface notes, document shards, and unrelated city notes.`,
      ["bench", "distractor"],
    )),
  ];
}

function expectedMemoryIdsForPrompt(caseId: BrowserMemoryGroundingCase, prompt: string): string[] {
  if (caseId === "montana_capital") return ["bench_memory_montana_capital"];
  const fact = groundingFactForPrompt(prompt, groundingFactsForCase(caseId));
  return fact ? [fact.id] : [];
}

function groundingFactsForCase(caseId: BrowserMemoryGroundingCase): readonly BrowserGroundingFact[] {
  if (caseId === "montana_capital") return QA_GROUNDING_FACTS.slice(0, 3);
  if (caseId === "large_synthetic_v1") return LARGE_SYNTHETIC_GROUNDING_FACTS;
  return QA_GROUNDING_FACTS;
}

function makeGroundingChunk(
  id: string,
  text: string,
  tags: string[],
  metadata: Record<string, unknown> = {},
): MemoryChunk {
  const now = "2026-05-23T00:00:00.000Z";
  return {
    id,
    text,
    embedding: embedMemoryGroundingText(text),
    sessionId: MEMORY_GROUNDING_SESSION_ID,
    source: "document",
    createdAt: now,
    updatedAt: now,
    tags,
    metadata: {
      edgeTenantId: MEMORY_TENANT_ID,
      edgeCellId: MEMORY_CELL_ID,
      ...metadata,
    },
    tokenCount: Math.max(1, Math.ceil(text.length / 4)),
  };
}

export function embedMemoryGroundingText(text: string): number[] {
  const vector = Array.from({ length: MEMORY_GROUNDING_VECTOR_DIM }, () => 0);
  const tokens = text.toLowerCase().match(/[a-z0-9_]+/g) ?? [];
  for (const token of tokens) {
    const index = memoryGroundingTokenIndex(token) ?? (hashMemoryToken(token) % vector.length);
    vector[index] = (vector[index] ?? 0) + tokenWeight(token);
  }
  return normalizeVector(vector);
}

function memoryGroundingTokenIndex(token: string): number | null {
  const facts = allGroundingFacts();
  const factIndex = facts.findIndex((fact) => fact.marker.toLowerCase() === token);
  if (factIndex >= 0) return factIndex;
  const uniqueFactIndex = facts.findIndex((fact) => fact.uniqueTokens.includes(token));
  return uniqueFactIndex >= 0 ? 32 + uniqueFactIndex : null;
}

function tokenWeight(token: string): number {
  const facts = allGroundingFacts();
  if (token.startsWith("memory_fact_")) return 12;
  if (facts.some((fact) => fact.answer.toLowerCase().split(/[^a-z0-9_]+/).includes(token))) return 8;
  if (facts.some((fact) => fact.uniqueTokens.includes(token))) return 40;
  if (["montana", "archive", "calibration", "recovery", "retention"].includes(token)) return 5;
  if (["helix", "ledger", "synthetic", "corpus", "shard"].includes(token)) return 5;
  if (["dossier", "memo", "brief", "archive", "note", "routing", "coordination", "records", "office", "hub", "contact", "window", "answer", "code"].includes(token)) return 4;
  return 1;
}

function allGroundingFacts(): readonly BrowserGroundingFact[] {
  return ALL_GROUNDING_FACTS;
}

function hashMemoryToken(token: string): number {
  let hash = 2166136261;
  for (let index = 0; index < token.length; index += 1) {
    hash ^= token.charCodeAt(index);
    hash = Math.imul(hash, 16777619);
  }
  return hash >>> 0;
}

function safeDbName(value: string): string {
  return value.replace(/[^a-z0-9_-]+/gi, "-").slice(0, 96) || Date.now().toString(36);
}

function withFullVocabLogitProofCaps(
  profile: UnlockedRuntimeProfileResolution,
): UnlockedRuntimeProfileResolution {
  if (profile.caps.logitCandidateLimit === null && !profile.capStatus.logits) return profile;
  return {
    ...profile,
    caps: {
      ...profile.caps,
      logitCandidateLimit: null,
    },
    capStatus: {
      ...profile.capStatus,
      logits: false,
    },
  };
}

function withStrictExpectedSubstringLayerCaps(
  profile: UnlockedRuntimeProfileResolution,
): UnlockedRuntimeProfileResolution {
  const cappedLayers = profile.caps.maxRuntimeLayers;
  if (cappedLayers === null || cappedLayers >= STRICT_EXPECTED_SUBSTRING_MIN_RUNTIME_LAYERS) return profile;
  return {
    ...profile,
    caps: {
      ...profile.caps,
      maxRuntimeLayers: null,
    },
    capStatus: {
      ...profile.capStatus,
      layers: false,
    },
  };
}

function readBrowserBenchmarkLocks(): BrowserBenchmarkLockManager | undefined {
  return (globalThis.navigator as (Navigator & { locks?: BrowserBenchmarkLockManager }) | undefined)?.locks;
}

function parseMtpEnabled(value: string | null | undefined): boolean {
  if (value === null || value === undefined || !value.trim()) return MTP_ENABLED;
  const normalized = value.trim().toLowerCase();
  if (normalized === "false" || normalized === "disabled" || normalized === "off" || normalized === "0") return false;
  if (normalized === "true" || normalized === "enabled" || normalized === "on" || normalized === "1") return true;
  return MTP_ENABLED;
}

function parseWarmupMode(value: string | null | undefined): UnlockedBrowserWarmupMode {
  return value?.trim() === "target_probe" ? "target_probe" : "pipeline_preload";
}

function resolveRequestedWebGpuGates(input: {
  gates: WebGpuGate[];
  strictWebGpu: boolean;
  requireWebGpuKernels: boolean;
}): WebGpuGate[] {
  if (input.requireWebGpuKernels) return ALL_WEBGPU_GATES;
  if (input.strictWebGpu && input.gates.length === 0) return ALL_WEBGPU_GATES;
  return ALL_WEBGPU_GATES.filter((gate) => input.gates.includes(gate));
}

function parseQwenThinkingMode(value: string | null | undefined): "disabled" | "enabled" {
  return value?.trim().toLowerCase() === "enabled" ? "enabled" : "disabled";
}

function parseStrictExpectedLayerMode(
  value: string | null | undefined,
  hasExpectedSubstrings = false,
): BrowserPreviewBenchmarkRequest["strictExpectedLayerMode"] {
  const normalized = value?.trim().toLowerCase();
  if (normalized === "full" || normalized === "production" || normalized === "quality") return "full";
  if (normalized === "requested" || normalized === "capped" || normalized === "proof") return "requested";
  if (hasExpectedSubstrings) return "full";
  return "requested";
}

function parseMemoryGroundingCase(value: string | null | undefined): BrowserMemoryGroundingCase | null {
  const normalized = value?.trim().toLowerCase();
  if (!normalized || normalized === "false" || normalized === "off" || normalized === "0" || normalized === "none") return null;
  if (normalized === "true" || normalized === "on" || normalized === "1" || normalized === "montana" || normalized === "montana_capital") {
    return "montana_capital";
  }
  if (normalized === "qa" || normalized === "qa_corpus" || normalized === "qa_corpus_v1" || normalized === "large_qa") {
    return "qa_corpus_v1";
  }
  if (
    normalized === "large_synthetic"
    || normalized === "large_synthetic_v1"
    || normalized === "synthetic_large"
    || normalized === "synthetic_corpus"
  ) {
    return "large_synthetic_v1";
  }
  return null;
}

function hasExplicitPrompt(url: URL): boolean {
  return url.searchParams.has("prompt")
    || url.searchParams.has("prompts")
    || url.searchParams.has("promptSeed");
}

function withMemoryGroundingPromptDefaults(
  prompts: Array<{ id: string; text: string; expectedSubstrings: string[]; expectedExact?: string[] }>,
  caseId: BrowserMemoryGroundingCase | null,
  explicitPrompt: boolean,
): Array<{ id: string; text: string; expectedSubstrings: string[]; expectedExact?: string[] }> {
  if (!caseId) return prompts;
  if (caseId === "montana_capital") {
    const base = explicitPrompt && prompts.length > 0
      ? prompts
        : [{
          id: "prompt-1",
          text: "Using retrieved memory only, in the Cedar Ridge operations dossier, which city is listed as the Montana field office hub? Answer with only the city.",
          expectedSubstrings: [],
        }];
    return base.map((prompt) => ({
      ...prompt,
      expectedSubstrings: prompt.expectedSubstrings.length > 0 ? prompt.expectedSubstrings : ["Helena"],
      expectedExact: prompt.expectedExact?.length
        ? prompt.expectedExact
        : (prompt.expectedSubstrings.length > 0 ? prompt.expectedSubstrings : ["Helena"]),
    }));
  }
  if (caseId === "qa_corpus_v1") {
    const base: Array<{ id: string; text: string; expectedSubstrings: string[]; expectedExact?: string[] }> = explicitPrompt && prompts.length > 0
      ? prompts
      : QA_GROUNDING_FACTS.map((fact, index) => ({
          id: `qa-${index + 1}`,
          text: `Using retrieved memory only, ${fact.question} ${fact.answerInstruction}`,
          expectedSubstrings: [fact.answer],
        }));
    return base.map((prompt) => ({
      ...prompt,
      expectedSubstrings: prompt.expectedSubstrings.length > 0
        ? prompt.expectedSubstrings
        : expectedAnswersForGroundingPrompt(prompt.text),
      expectedExact: prompt.expectedExact?.length
        ? prompt.expectedExact
        : (prompt.expectedSubstrings.length > 0
          ? prompt.expectedSubstrings
          : expectedAnswersForGroundingPrompt(prompt.text)),
    }));
  }
  if (caseId === "large_synthetic_v1") {
    const proofFacts = [
      LARGE_SYNTHETIC_GROUNDING_FACTS[0],
      LARGE_SYNTHETIC_GROUNDING_FACTS[15],
      LARGE_SYNTHETIC_GROUNDING_FACTS[31],
      LARGE_SYNTHETIC_GROUNDING_FACTS[47],
      LARGE_SYNTHETIC_GROUNDING_FACTS[63],
    ].filter((fact): fact is BrowserGroundingFact => Boolean(fact));
    const base: Array<{ id: string; text: string; expectedSubstrings: string[]; expectedExact?: string[] }> = explicitPrompt && prompts.length > 0
      ? prompts
      : proofFacts.map((fact, index) => ({
          id: `large-synthetic-${index + 1}`,
          text: `Using retrieved memory only, ${fact.question} ${fact.answerInstruction}`,
          expectedSubstrings: [fact.answer],
        }));
    return base.map((prompt) => ({
      ...prompt,
      expectedSubstrings: prompt.expectedSubstrings.length > 0
        ? prompt.expectedSubstrings
        : expectedAnswersForGroundingPrompt(prompt.text, LARGE_SYNTHETIC_GROUNDING_FACTS),
      expectedExact: prompt.expectedExact?.length
        ? prompt.expectedExact
        : (prompt.expectedSubstrings.length > 0
          ? prompt.expectedSubstrings
          : expectedAnswersForGroundingPrompt(prompt.text, LARGE_SYNTHETIC_GROUNDING_FACTS)),
    }));
  }
  return prompts;
}

function limitMemoryGroundingPrompts(
  prompts: Array<{ id: string; text: string; expectedSubstrings: string[]; expectedExact?: string[] }>,
  caseId: BrowserMemoryGroundingCase | null,
  limit: number | undefined,
): Array<{ id: string; text: string; expectedSubstrings: string[]; expectedExact?: string[] }> {
  if (!caseId || !limit || prompts.length <= limit) return prompts;
  return prompts.slice(0, limit);
}

function expectedAnswersForGroundingPrompt(
  prompt: string,
  facts: readonly BrowserGroundingFact[] = QA_GROUNDING_FACTS,
): string[] {
  const fact = groundingFactForPrompt(prompt, facts);
  return fact ? [fact.answer] : [];
}

function memoryGroundingQueryText(prompt: string, caseId: BrowserMemoryGroundingCase): string {
  const facts = groundingFactsForCase(caseId);
  const marker = prompt.toUpperCase().match(/MEMORY_FACT_[A-Z0-9_]+/)?.[0];
  const fact = marker
    ? facts.find((candidate) => candidate.marker === marker)
    : groundingFactForPrompt(prompt, facts);
  return fact ? `${fact.question} ${fact.promptNeedles.join(" ")}` : prompt;
}

function groundingFactForPrompt(
  prompt: string,
  facts: readonly BrowserGroundingFact[],
): BrowserGroundingFact | undefined {
  const normalized = prompt.toLowerCase();
  const marker = prompt.toUpperCase().match(/MEMORY_FACT_[A-Z0-9_]+/)?.[0];
  if (marker) return facts.find((candidate) => candidate.marker === marker);
  const exactNeedleMatch = facts.find((fact) =>
    fact.promptNeedles.every((needle) => normalized.includes(needle))
  );
  if (exactNeedleMatch) return exactNeedleMatch;
  const normalizedQueryTokens = tokenizeGroundingIntent(prompt);
  const normalizedQuery = normalizedQueryTokens.join(" ");
  const aliasMatch = facts.find((fact) =>
    (fact.queryAliases ?? []).some((alias) => groundingAliasMatches(normalizedQuery, normalizedQueryTokens, alias))
  );
  if (aliasMatch) return aliasMatch;
  return bestHybridGroundingFactForPrompt(normalizedQueryTokens, facts);
}

function bestHybridGroundingFactForPrompt(
  normalizedQueryTokens: string[],
  facts: readonly BrowserGroundingFact[],
): BrowserGroundingFact | undefined {
  if (normalizedQueryTokens.length < 2) return undefined;
  let best: { fact: BrowserGroundingFact; score: number } | undefined;
  for (const fact of facts) {
    const score = scoreGroundingFactTokenOverlap(normalizedQueryTokens, fact);
    if (!best || score > best.score) best = { fact, score };
  }
  return best && best.score >= 8 ? best.fact : undefined;
}

function scoreGroundingFactTokenOverlap(
  normalizedQueryTokens: string[],
  fact: BrowserGroundingFact,
): number {
  const weightedFactTokens = groundingFactWeightedTokens(fact);
  const seen = new Set<string>();
  let score = 0;
  for (const token of normalizedQueryTokens) {
    if (seen.has(token)) continue;
    seen.add(token);
    score += weightedFactTokens.get(token) ?? 0;
  }
  return score;
}

function groundingFactWeightedTokens(fact: BrowserGroundingFact): Map<string, number> {
  const tokens = new Map<string, number>();
  addWeightedGroundingTokens(tokens, fact.question, 2);
  addWeightedGroundingTokens(tokens, fact.document, 2);
  addWeightedGroundingTokens(tokens, fact.answerInstruction, 1);
  for (const needle of fact.promptNeedles) addWeightedGroundingTokens(tokens, needle, 3);
  for (const tag of fact.tags) addWeightedGroundingTokens(tokens, tag, 2);
  for (const token of fact.uniqueTokens) addWeightedGroundingTokens(tokens, token, 4);
  return tokens;
}

function addWeightedGroundingTokens(tokens: Map<string, number>, value: string, weight: number): void {
  for (const token of tokenizeGroundingIntent(value)) {
    tokens.set(token, Math.max(tokens.get(token) ?? 0, weight));
  }
}

function groundingAliasMatches(
  normalizedQuery: string,
  normalizedQueryTokens: string[],
  alias: string,
): boolean {
  const aliasTokens = tokenizeGroundingIntent(alias);
  if (aliasTokens.length < 2) return false;
  const normalizedAlias = aliasTokens.join(" ");
  if (normalizedQuery.includes(normalizedAlias)) return true;
  const queryTokenSet = new Set(normalizedQueryTokens);
  return aliasTokens.every((token) => queryTokenSet.has(token));
}

const GROUNDING_INTENT_STOPWORDS = new Set([
  "a",
  "an",
  "answer",
  "are",
  "city",
  "did",
  "do",
  "does",
  "for",
  "from",
  "in",
  "is",
  "memory",
  "of",
  "only",
  "on",
  "retrieved",
  "s",
  "the",
  "using",
  "was",
  "were",
  "what",
  "which",
  "who",
]);

function tokenizeGroundingIntent(value: string): string[] {
  return (value.toLowerCase().match(/[a-z0-9]+/g) ?? [])
    .filter((token) => !GROUNDING_INTENT_STOPWORDS.has(token));
}

function isWebGpuGate(value: string): value is WebGpuGate {
  return value === "mlp" || value === "logits" || value === "attention" || value === "projection";
}

function readPositiveSearchParam(url: URL, names: string[], max = Number.MAX_SAFE_INTEGER): number | undefined {
  const value = readFirstSearchParam(url, names);
  if (!value) return undefined;
  const parsed = Number(value);
  return Number.isInteger(parsed) && parsed > 0 ? Math.min(parsed, max) : undefined;
}

function readNumberSearchParam(url: URL, names: string[]): number | undefined {
  const value = readFirstSearchParam(url, names);
  if (!value) return undefined;
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : undefined;
}

function parseBooleanSearchParam(url: URL, names: string[]): boolean {
  const value = readFirstSearchParam(url, names);
  if (!value) return false;
  return parseBooleanSearchValue(value);
}

function parseBooleanSearchValue(value: string): boolean {
  const normalized = value.toLowerCase();
  return normalized === "true" || normalized === "1" || normalized === "yes" || normalized === "on";
}

function readFirstSearchParam(url: URL, names: string[]): string | undefined {
  for (const name of names) {
    const value = url.searchParams.get(name)?.trim();
    if (value) return value;
  }
  return undefined;
}

async function runPrompt(
  client: BrowserPreviewClient,
  prompt: { id: string; text: string; expectedSubstrings: string[]; expectedExact?: string[] },
  request: BrowserPreviewBenchmarkRequest,
  initLoadMs: number,
  warmup: {
    warmupMs: number;
    warmupMode: BrowserPreviewClient["lastWarmupMode"];
    warmupBlockingMs: number;
    warmupUploadedEntries: number | null;
    warmupCacheHits: number | null;
    residentReadbackCount: number | null;
  },
  memoryGrounding: BrowserMemoryGroundingHarness | null,
  signal?: AbortSignal,
): Promise<BrowserPreviewBenchmarkRun> {
  assertBenchmarkNotAborted(signal);
  const grounded = memoryGrounding
    ? await buildMemoryGroundedMessages({ harness: memoryGrounding, prompt: prompt.text })
    : null;
  assertBenchmarkNotAborted(signal);
  const messages = grounded?.messages ?? [{ role: "user" as const, content: prompt.text }];
  const stopAfterSequences = buildStopAfterSequences(client, prompt, grounded !== null);
  const start = performance.now();
  let firstChunkAt: number | null = null;
  let firstGeneratedAt: number | null = null;
  let endAt = start;
  const chunks: string[] = [];
  for await (const chunk of client.streamChat(messages, {
    includeProofMarker: true,
    maxTokens: request.profile.caps.maxGenerationTokens ?? CHAT_MAX_GENERATION_TOKENS,
    ...(request.samplingTemperature !== undefined ? { temperature: request.samplingTemperature } : {}),
    ...(request.samplingTopP !== undefined ? { topP: request.samplingTopP } : {}),
    ...(request.repetitionPenalty !== undefined ? { repetitionPenalty: request.repetitionPenalty } : {}),
    ...(request.samplingSeed !== undefined ? { samplingSeed: request.samplingSeed } : {}),
    ...(stopAfterSequences?.length ? { stopAfterSequences } : {}),
    ...(request.requireKvPredictivePrefetch ? { awaitKvPredictivePrefetchProof: true } : {}),
  })) {
    assertBenchmarkNotAborted(signal);
    const now = performance.now();
    const visibleChunk = stripProofScaffolding(chunk);
    if (visibleChunk.length > 0) {
      firstChunkAt ??= now;
      firstGeneratedAt ??= now;
    }
    endAt = now;
    chunks.push(chunk);
  }
  assertBenchmarkNotAborted(signal);
  if (request.requireKvReuse || request.requireKvPredictivePrefetch) {
    await flushKvPersistenceForProof(client, signal);
  }

  const proof = client.lastDecodeProof;
  const compiledProof = isCompiledWebLlmProof(proof) ? proof : null;
  const unlockedProof = compiledProof ? null : proof as UnlockedBrowserDecodeProof | null;
  const brokerSelection = resolveBenchmarkBrokerSelection(request, grounded !== null);
  const generatedTokens = readGeneratedTokenCount(client, compiledProof);
  const response = chunks.join("");
  const timing = calculateBrowserPreviewRunTiming({
    start,
    firstChunkAt,
    firstGeneratedAt,
    endAt,
    generatedTokens,
  });
  return {
    promptId: prompt.id,
    prompt: prompt.text,
    response,
    coherent: isCoherentBrowserPreviewResponse(response, generatedTokens, prompt.expectedSubstrings),
    expectedSubstrings: prompt.expectedSubstrings,
    expectedSubstringMatches: matchingExpectedSubstrings(response, prompt.expectedSubstrings),
    ...(prompt.expectedExact?.length ? {
      expectedExact: prompt.expectedExact,
      expectedExactMatches: matchingExpectedExact(response, prompt.expectedExact),
    } : {}),
    ...(grounded ? { expectedAnswerOnlyPassed: isExpectedAnswerOnlyResponse(response, prompt.expectedSubstrings) } : {}),
    generationStopReason: client.lastGenerationStopReason,
    metrics: {
      initLoadMs,
      warmupMs: warmup.warmupMs,
      ...(warmup.warmupMode ? { warmupMode: warmup.warmupMode } : {}),
      warmupBlockingMs: warmup.warmupBlockingMs,
      ...(warmup.warmupUploadedEntries !== null ? { warmupUploadedEntries: warmup.warmupUploadedEntries } : {}),
      ...(warmup.warmupCacheHits !== null ? { warmupCacheHits: warmup.warmupCacheHits } : {}),
      ...(warmup.warmupUploadedEntries !== null ? { residentUploadCount: warmup.warmupUploadedEntries } : {}),
      ...(warmup.warmupCacheHits !== null ? { residentCacheHitCount: warmup.warmupCacheHits } : {}),
      ...(warmup.residentReadbackCount !== null ? { residentReadbackCount: warmup.residentReadbackCount } : {}),
      prefillMs: timing.prefillMs,
      timeToFirstTokenMs: timing.timeToFirstTokenMs,
      decodeLatencyMs: timing.decodeLatencyMs,
      generatedTokens,
      tokensPerSecond: timing.tokensPerSecond,
    },
    tokenDiagnostics: {
      promptTokenHeadIds: client.lastPromptTokenIds.slice(0, 32),
      promptTokenTailIds: client.lastPromptTokenIds.slice(-32),
      generatedTokenIds: [...client.lastGeneratedTokenIds],
      generatedTokenTexts: [...client.lastGeneratedTokenTexts],
    },
    ...readPrefillChunkMetadata(unlockedProof),
    runtimeTrace: {
      backend: client.backendId,
      brokerSelection,
      tensorControl: unlockedProof?.tensorControl === true,
      tspSteps: unlockedProof?.tspSteps ?? [],
      kvPagingEvents: unlockedProof?.kvPagingEvents ?? 0,
      selectedBlockIds: unlockedProof?.selectedBlockIds ?? [],
    },
    predictive: {
      promptTokenCount: client.lastPromptTokenIds.length,
      generatedTokenCount: generatedTokens,
      selectedBlockCount: unlockedProof?.selectedBlockIds.length ?? 0,
      kvPagingEventCount: unlockedProof?.kvPagingEvents ?? 0,
      tspStepCount: unlockedProof?.tspSteps.length ?? 0,
    },
    webGpu: compiledProof
      ? buildCompiledBackendProof(compiledProof)
      : buildWebGpuProof(unlockedProof, request.backendPreference, request.webGpuGates),
    mtp: normalizeMtpProof(unlockedProof),
    kvPersistence: compiledProof ? summarizeCompiledKvPersistence() : summarizeKvPersistence(unlockedProof),
    ...(grounded ? { memoryGrounding: {
      ...grounded.proof,
      answerOnlyPassed: isExpectedAnswerOnlyResponse(response, prompt.expectedSubstrings),
    } } : {}),
  };
}

function resolveBenchmarkBrokerSelection(
  request: BrowserPreviewBenchmarkRequest,
  grounded: boolean,
): BrowserBackendSelection {
  return selectBrowserBackend({
    task: resolveBenchmarkBrokerTask(request, grounded),
    preferredBackendId: request.backendId,
    preferredModelId: request.modelId,
  });
}

function resolveBenchmarkBrokerTask(
  request: BrowserPreviewBenchmarkRequest,
  grounded: boolean,
): BrowserBackendTask {
  if (request.backendId === "unlocked-browser-transformer") {
    return request.strictWebGpuRequested ? "strict_custom_proof" : "kernel_research";
  }
  return grounded ? "grounded_answer" : "final_answer";
}

export function buildStopAfterSequences(
  client: BrowserPreviewClient,
  prompt: { expectedSubstrings: string[]; expectedExact?: string[] },
  grounded: boolean,
): string[] | undefined {
  if (client.backendId === "compiled-browser-webllm") {
    return [...exactOutputSuffixStopFragments(prompt.expectedExact), ...STOP_FRAGMENTS];
  }
  return prompt.expectedExact?.length
    ? prompt.expectedExact
    : grounded ? prompt.expectedSubstrings : undefined;
}

function exactOutputSuffixStopFragments(expectedExact: readonly string[] | undefined): string[] {
  if (!expectedExact?.length) return [];
  const exactValueContainsSentencePunctuation = expectedExact.some((expected) => /[.!?]/.test(expected));
  return exactValueContainsSentencePunctuation ? [] : [".", "!", "?"];
}

function readGeneratedTokenCount(
  client: BrowserPreviewClient,
  compiledProof: CompiledWebLlmProof | null,
): number {
  if (client.lastGeneratedTokenIds.length > 0) return client.lastGeneratedTokenIds.length;
  return compiledProof?.generatedTokenEstimate ?? 0;
}

function assertBenchmarkNotAborted(signal: AbortSignal | undefined): void {
  if (!signal?.aborted) return;
  if (signal.reason instanceof Error) throw signal.reason;
  throw new Error("Browser preview benchmark was cancelled.");
}

async function waitForBenchmarkTask<T>(task: Promise<T>, signal?: AbortSignal): Promise<T> {
  if (!signal) return task;
  assertBenchmarkNotAborted(signal);
  let abort: (() => void) | null = null;
  const aborted = new Promise<never>((_, reject) => {
    abort = () => {
      reject(signal.reason instanceof Error ? signal.reason : new Error("Browser preview benchmark was cancelled."));
    };
    signal.addEventListener("abort", abort, { once: true });
  });
  try {
    return await Promise.race([task, aborted]);
  } finally {
    if (abort) signal.removeEventListener("abort", abort);
  }
}

export function calculateBrowserPreviewRunTiming(input: {
  start: number;
  firstChunkAt: number | null;
  firstGeneratedAt: number | null;
  endAt: number;
  generatedTokens: number;
}): Pick<BrowserPreviewBenchmarkRun["metrics"], "prefillMs" | "timeToFirstTokenMs" | "decodeLatencyMs" | "tokensPerSecond"> {
  const fallbackEnd = Math.max(input.endAt, input.start);
  const prefillRawMs = input.firstChunkAt === null
    ? Math.max(0, fallbackEnd - input.start)
    : Math.max(0, input.firstChunkAt - input.start);
  const timeToFirstRawMs = input.firstGeneratedAt === null
    ? prefillRawMs
    : Math.max(0, input.firstGeneratedAt - input.start);
  const postFirstChunkDecodeRawMs = input.firstChunkAt === null
    ? 0
    : Math.max(0, fallbackEnd - input.firstChunkAt);
  const totalGenerationRawMs = Math.max(0, fallbackEnd - input.start);
  const throughputWindowMs = Math.max(postFirstChunkDecodeRawMs, totalGenerationRawMs);
  return {
    prefillMs: roundMs(prefillRawMs),
    timeToFirstTokenMs: roundMs(timeToFirstRawMs),
    decodeLatencyMs: roundMs(throughputWindowMs),
    tokensPerSecond: calculateTokensPerSecond(input.generatedTokens, throughputWindowMs),
  };
}

export function buildWebGpuProof(
  proof: UnlockedBrowserDecodeProof | null,
  backendPreference: BackendPreference | undefined,
  requestedGates: WebGpuGate[],
): BrowserPreviewBenchmarkRun["webGpu"] {
  const summary = summarizeUnlockedWebGpuCoverage(proof ?? {});
  const cpuFallbackUsed = summary.cpuFallbackUsed;
  const passedGates = requestedGates.filter((gate) => evaluateUnlockedWebGpuCoverageGates(summary, [gate]).passed);
  const failedGates = requestedGates.filter((gate) => !passedGates.includes(gate));
  return {
    available: Boolean((globalThis.navigator as Navigator & { gpu?: unknown }).gpu),
    requestedBackendPreference: backendPreference ?? "auto",
    logitProjectionBackend: summary.logitProjection.backend,
    ...(proof?.logitProjectionReadbackStrategy ? { logitProjectionReadbackStrategy: proof.logitProjectionReadbackStrategy } : {}),
    ...(isFiniteNumber(proof?.logitProjectionGpuReducedRows) ? { logitProjectionGpuReducedRows: proof.logitProjectionGpuReducedRows } : {}),
    ...(isFiniteNumber(proof?.logitProjectionReadbackRows) ? { logitProjectionReadbackRows: proof.logitProjectionReadbackRows } : {}),
    ...(isFiniteNumber(proof?.logitProjectionReadbackBytes) ? { logitProjectionReadbackBytes: proof.logitProjectionReadbackBytes } : {}),
    ...(isFiniteNumber(proof?.logitProjectionDispatchCount) ? { logitProjectionDispatchCount: proof.logitProjectionDispatchCount } : {}),
    ...(isFiniteNumber(proof?.logitProjectionTiles) ? { logitProjectionTiles: proof.logitProjectionTiles } : {}),
    ...(isFiniteNumber(proof?.logitProjectionTileRows) ? { logitProjectionTileRows: proof.logitProjectionTileRows } : {}),
    ...(proof?.logitProjectionCandidateTokenIds ? { logitProjectionCandidateTokenIds: proof.logitProjectionCandidateTokenIds } : {}),
    ...(proof?.logitProjectionCandidateScores ? { logitProjectionCandidateScores: proof.logitProjectionCandidateScores } : {}),
    ...(isFiniteNumber(proof?.compactLogitTopK) ? { compactLogitTopK: proof.compactLogitTopK } : {}),
    ...(isFiniteNumber(proof?.samplingTemperature) ? { samplingTemperature: proof.samplingTemperature } : {}),
    ...(isFiniteNumber(proof?.samplingTopP) ? { samplingTopP: proof.samplingTopP } : {}),
    ...(isFiniteNumber(proof?.repetitionPenalty) ? { repetitionPenalty: proof.repetitionPenalty } : {}),
    ...(proof?.greedyDecodeUsed !== undefined ? { greedyDecodeUsed: proof.greedyDecodeUsed } : {}),
    ...(isFiniteNumber(proof?.sampledTokenRank) ? { sampledTokenRank: proof.sampledTokenRank } : {}),
    ...(isFiniteNumber(proof?.decodePerf?.decodeSubmitCount) ? { decodeSubmitCount: proof.decodePerf.decodeSubmitCount } : {}),
    ...(isFiniteNumber(proof?.decodePerf?.decodeSubmitCountPerToken) ? { decodeSubmitCountPerToken: proof.decodePerf.decodeSubmitCountPerToken } : {}),
    ...(isFiniteNumber(proof?.decodePerf?.dispatchCount) ? { decodeDispatchCount: proof.decodePerf.dispatchCount } : {}),
    ...(isFiniteNumber(proof?.decodePerf?.decodeDispatchCountPerToken) ? { decodeDispatchCountPerToken: proof.decodePerf.decodeDispatchCountPerToken } : {}),
    ...(isFiniteNumber(proof?.decodePerf?.decodeDispatchCountPerLayerPerToken) ? { decodeDispatchCountPerLayerPerToken: proof.decodePerf.decodeDispatchCountPerLayerPerToken } : {}),
    ...(isFiniteNumber(proof?.decodePerf?.readbackCount) ? { decodeReadbackCount: proof.decodePerf.readbackCount } : {}),
    ...(isFiniteNumber(proof?.decodePerf?.totalReadbackBytes) ? { decodeReadbackBytes: proof.decodePerf.totalReadbackBytes } : {}),
    ...(isFiniteNumber(proof?.decodePerf?.fullLogitsReadbackCount) ? { fullLogitsReadbackCount: proof.decodePerf.fullLogitsReadbackCount } : {}),
    ...(isFiniteNumber(proof?.decodePerf?.compactLogitReadbackCount) ? { compactLogitReadbackCount: proof.decodePerf.compactLogitReadbackCount } : {}),
    ...(isFiniteNumber(proof?.decodePerf?.weightUploadBytesDuringDecode) ? { weightUploadBytesDuringDecode: proof.decodePerf.weightUploadBytesDuringDecode } : {}),
    ...(isFiniteNumber(proof?.decodePerf?.weightUploadCountDuringDecode) ? { weightUploadCountDuringDecode: proof.decodePerf.weightUploadCountDuringDecode } : {}),
    ...(isFiniteNumber(proof?.decodePerf?.activationUploadBytesDuringDecode) ? { activationUploadBytesDuringDecode: proof.decodePerf.activationUploadBytesDuringDecode } : {}),
    ...(isFiniteNumber(proof?.decodePerf?.activationUploadCountDuringDecode) ? { activationUploadCountDuringDecode: proof.decodePerf.activationUploadCountDuringDecode } : {}),
    ...(isFiniteNumber(proof?.decodePerf?.hiddenReadbackCountDuringDecode) ? { hiddenReadbackCountDuringDecode: proof.decodePerf.hiddenReadbackCountDuringDecode } : {}),
    ...(isFiniteNumber(proof?.decodePerf?.residentDecodeLayerCount) ? { residentDecodeLayerCount: proof.decodePerf.residentDecodeLayerCount } : {}),
    ...(isFiniteNumber(proof?.decodePerf?.totalDecodeLayerCount) ? { totalDecodeLayerCount: proof.decodePerf.totalDecodeLayerCount } : {}),
    ...(isFiniteNumber(proof?.decodePerf?.residentDecodeLayerCoverage) ? { residentDecodeLayerCoverage: proof.decodePerf.residentDecodeLayerCoverage } : {}),
    ...(proof?.decodePerf?.residentFinalHiddenUsedForLogits !== undefined
      ? { residentFinalHiddenUsedForLogits: proof.decodePerf.residentFinalHiddenUsedForLogits }
      : {}),
    ...(isFiniteNumber(proof?.decodePerf?.f32ExpansionCountDuringDecode) ? { f32ExpansionCountDuringDecode: proof.decodePerf.f32ExpansionCountDuringDecode } : {}),
    ...(isFiniteNumber(proof?.decodePerf?.f32ExpansionBytesDuringDecode) ? { f32ExpansionBytesDuringDecode: proof.decodePerf.f32ExpansionBytesDuringDecode } : {}),
    ...(proof?.decodePerf?.cpuValidationUsed !== undefined ? { cpuValidationUsed: proof.decodePerf.cpuValidationUsed } : {}),
    ...(isFiniteNumber(proof?.decodePerf?.prefillExecutionsDuringDecode) ? { prefillExecutionsDuringDecode: proof.decodePerf.prefillExecutionsDuringDecode } : {}),
    ...(isFiniteNumber(proof?.decodePerf?.prefillCountPerGeneratedToken) ? { prefillCountPerGeneratedToken: proof.decodePerf.prefillCountPerGeneratedToken } : {}),
    ...(proof?.decodePerf?.kvDecodeReused !== undefined ? { kvDecodeReused: proof.decodePerf.kvDecodeReused } : {}),
    ...(isFiniteNumber(proof?.decodePerf?.fusedPackedQkvLayerCount) ? { fusedPackedQkvLayerCount: proof.decodePerf.fusedPackedQkvLayerCount } : {}),
    ...(isFiniteNumber(proof?.decodePerf?.fusedQkvNormRopeKvAppendLayerCount) ? { fusedQkvNormRopeKvAppendLayerCount: proof.decodePerf.fusedQkvNormRopeKvAppendLayerCount } : {}),
    ...(isFiniteNumber(proof?.decodePerf?.fusedOneTokenAttentionLayerCount) ? { fusedOneTokenAttentionLayerCount: proof.decodePerf.fusedOneTokenAttentionLayerCount } : {}),
    ...(isFiniteNumber(proof?.decodePerf?.fusedResidualRmsNormLayerCount) ? { fusedResidualRmsNormLayerCount: proof.decodePerf.fusedResidualRmsNormLayerCount } : {}),
    ...(isFiniteNumber(proof?.decodePerf?.fusedMlpLayerCount) ? { fusedMlpLayerCount: proof.decodePerf.fusedMlpLayerCount } : {}),
    ...(isFiniteNumber(proof?.decodePerf?.fusedFullLayerCount) ? { fusedFullLayerCount: proof.decodePerf.fusedFullLayerCount } : {}),
    ...(isFiniteNumber(proof?.decodePerf?.fusedLayerCoverage) ? { fusedLayerCoverage: proof.decodePerf.fusedLayerCoverage } : {}),
    cpuFallbackUsed,
    noCpuFallback: !cpuFallbackUsed,
    requestedGates,
    passedGates,
    failedGates,
    positiveKernelProof: requestedGates.length > 0
      ? failedGates.length === 0
      : summaryHasAnyWebGpuKernel(summary),
  };
}

function buildCompiledBackendProof(proof: CompiledWebLlmProof): BrowserPreviewBenchmarkRun["webGpu"] {
  return {
    available: Boolean((globalThis.navigator as Navigator & { gpu?: unknown }).gpu),
    requestedBackendPreference: "compiled-browser",
    logitProjectionBackend: "backend_native",
    logitProjectionReadbackStrategy: "backend_native_compiled_graph",
    cpuFallbackUsed: false,
    noCpuFallback: true,
    requestedGates: [],
    passedGates: [],
    failedGates: [],
    positiveKernelProof: proof.adapterKind === "compiled-browser",
  };
}

function normalizeMtpProof(proof: UnlockedBrowserDecodeProof | null): BrowserPreviewBenchmarkRun["mtp"] {
  if (!proof?.mtp) {
    return {
      mode: "none",
      acceptedTokens: 0,
      rejectedTokens: 0,
      acceptanceRate: 0,
      numSpeculativeTokens: 0,
      verifiedTokenCount: 0,
      targetDecodeCalls: 0,
      verifierStrategy: "none",
    };
  }
  return {
    mode: proof.mtp.mode,
    ...(proof.mtp.draftModelId !== undefined ? { draftModelId: proof.mtp.draftModelId } : {}),
    ...(proof.mtp.draftSource ? { draftSource: proof.mtp.draftSource } : {}),
    ...(proof.mtp.latencyDisablePolicy ? { latencyDisablePolicy: proof.mtp.latencyDisablePolicy } : {}),
    acceptedTokens: proof.mtp.acceptedTokens,
    rejectedTokens: proof.mtp.rejectedTokens,
    acceptanceRate: roundMetric(proof.mtp.acceptanceRate),
    numSpeculativeTokens: proof.mtp.numSpeculativeTokens ?? 0,
    verifiedTokenCount: proof.mtp.verifiedTokenCount ?? 0,
    targetDecodeCalls: proof.mtp.targetDecodeCalls ?? 0,
    verifierStrategy: proof.mtp.verifierStrategy ?? "none",
  };
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

function summarizeKvPersistence(proof: UnlockedBrowserDecodeProof | null): BrowserPreviewBenchmarkRun["kvPersistence"] {
  const health = proof?.kvPersistence;
  const events = health?.events ?? [];
  return {
    enabled: health?.enabled ?? false,
    mode: health?.mode ?? "disabled",
    eventCount: events.length,
    persistEvents: countKvEvents(events, "persist"),
    hydrateEvents: countKvEvents(events, "hydrate"),
    reuseEvents: countKvEvents(events, "reuse"),
    prefetchStrategy: health?.prefetchStrategy ?? "none",
    ...(isFiniteNumber(health?.lowRankSummaryRank) ? { lowRankSummaryRank: health.lowRankSummaryRank } : {}),
    ...(typeof health?.lowRankQuerySource === "string" ? { lowRankQuerySource: health.lowRankQuerySource } : {}),
    kvPersistDeferred: health?.kvPersistDeferred === true,
    kvPersistCriticalPathMs: isFiniteNumber(health?.kvPersistCriticalPathMs) ? health.kvPersistCriticalPathMs : 0,
    ...(isFiniteNumber(health?.kvPersistFlushMs) ? { kvPersistFlushMs: health.kvPersistFlushMs } : {}),
    kvPersistPendingBlockCount: isFiniteNumber(health?.kvPersistPendingBlockCount) ? health.kvPersistPendingBlockCount : 0,
    predictedHotBlocks: (health?.predictedHotBlocks ?? []).map((block) => block.blockId),
    prefetchedBlocks: health?.prefetchedBlocks ?? [],
    prefetchHitRate: isFiniteNumber(health?.prefetchHitRate) ? health.prefetchHitRate : 0,
    prefetchBytes: isFiniteNumber(health?.prefetchBytes) ? health.prefetchBytes : 0,
    prefetchLatencyMs: isFiniteNumber(health?.prefetchLatencyMs) ? health.prefetchLatencyMs : 0,
    attentionStallMs: isFiniteNumber(health?.attentionStallMs) ? health.attentionStallMs : 0,
  };
}

function summarizeCompiledKvPersistence(): BrowserPreviewBenchmarkRun["kvPersistence"] {
  return {
    enabled: false,
    mode: "backend_native",
    eventCount: 0,
    persistEvents: 0,
    hydrateEvents: 0,
    reuseEvents: 0,
    prefetchStrategy: "none",
    predictedHotBlocks: [],
    prefetchedBlocks: [],
  };
}

function isCompiledWebLlmProof(value: unknown): value is CompiledWebLlmProof {
  return isRecord(value)
    && value.backendId === "compiled-browser-webllm"
    && value.adapterKind === "compiled-browser";
}

export function buildBrowserPreviewBenchmarkFailurePayload(
  request: BrowserPreviewBenchmarkRequest,
  error: unknown,
): BrowserPreviewBenchmarkPayload {
  const message = error instanceof Error ? error.message : String(error);
  const prefillChunkMetadata = readPrefillChunkMetadata(error);
  return {
    name: "browser-preview-benchmark",
    schemaVersion: BROWSER_PREVIEW_BENCHMARK_SCHEMA_VERSION,
    createdAt: new Date().toISOString(),
    passed: false,
    summary: {
      profile: request.profile.profile,
      v12ProductionProofSchemaVersion: BROWSER_PREVIEW_BENCHMARK_SCHEMA_VERSION,
      promptCount: request.prompts.length,
      coherentResponseCount: 0,
      expectedSubstringCheckCount: request.prompts.filter((prompt) => prompt.expectedSubstrings.length > 0).length,
      expectedSubstringPassCount: 0,
      expectedSubstringsPassed: false,
      expectedExactCheckCount: request.prompts.reduce((count, prompt) => count + (prompt.expectedExact?.length ?? 0), 0),
      expectedExactPassCount: 0,
      expectedExactPassed: false,
      runtimeTraceCount: 0,
      meanInitLoadMs: null,
      meanWarmupMs: null,
      meanPrefillMs: null,
      meanTimeToFirstTokenMs: null,
      meanDecodeLatencyMs: null,
      meanTokensPerSecond: null,
      generatedTokenCount: 0,
      predictiveSelectedBlockCount: 0,
      predictiveKvPagingEventCount: 0,
      predictiveTspStepCount: 0,
      webGpuAvailable: Boolean((globalThis.navigator as Navigator & { gpu?: unknown }).gpu),
      strictWebGpuRequested: request.strictWebGpuRequested,
      cpuFallbackUsed: false,
      noCpuFallback: false,
      decodeHotPathPassed: false,
      decodeHotPathFailureCount: request.prompts.length,
      decodeDispatchCount: 0,
      decodeReadbackCount: 0,
      decodeReadbackBytes: 0,
      fullLogitsReadbackCount: 0,
      compactLogitReadbackCount: 0,
      weightUploadBytesDuringDecode: 0,
      weightUploadCountDuringDecode: 0,
      f32ExpansionCountDuringDecode: 0,
      f32ExpansionBytesDuringDecode: 0,
      cpuValidationUsed: false,
      prefillExecutionsDuringDecode: 0,
      prefillCountPerGeneratedToken: 0,
      mtpMode: "none",
      mtpAcceptanceRate: null,
      mtpMaxSpeculativeTokens: 0,
      mtpMeanSpeculativeTokens: null,
      mtpVerifiedTokenCount: 0,
      mtpTargetDecodeCalls: 0,
      mtpVerifierStrategy: "none",
      kvPersistenceEventCount: 0,
      kvPersistEventCount: 0,
      kvHydrateEventCount: 0,
      kvReuseEventCount: 0,
      requireKvReuse: request.requireKvReuse,
      kvReusePassed: !request.requireKvReuse,
      requireKvPredictivePrefetch: request.requireKvPredictivePrefetch,
      kvPredictivePrefetchPassed: !request.requireKvPredictivePrefetch,
      kvPrefetchStrategy: "none",
      kvExactReuseRunCount: 0,
      kvPredictivePrefetchRunCount: 0,
      kvMissStallRunCount: 0,
      kvNoPrefetchRunCount: 0,
      kvLowRankSummaryRank: null,
      kvPredictedHotBlockCount: 0,
      kvPrefetchedBlockCount: 0,
      kvPrefetchHitRate: null,
      kvPrefetchBytes: 0,
      kvPrefetchLatencyMs: 0,
      kvAttentionStallMs: 0,
      directModelFactualProofUsed: request.memoryGroundingCase === null
        && request.prompts.some((prompt) => prompt.expectedSubstrings.length > 0),
      groundedProductionReadyPassed: false,
      error: message,
      ...prefillChunkMetadata,
    },
    runs: [],
  };
}

function readPrompts(
  url: URL,
  options: { longPromptTargetTokens?: number; longPromptRepeat?: number } = {},
): Array<{ id: string; text: string; expectedSubstrings: string[]; expectedExact?: string[] }> {
  const expectedByIndex = readExpectedSubstrings(url);
  const expectedExactByIndex = readExpectedExact(url);
  if (options.longPromptTargetTokens || options.longPromptRepeat) {
    const seedText = readLongPromptSeed(url);
    const expectedExact = expectedExactByIndex[0] ?? [];
    return [{
      id: "long-prompt-1",
      text: buildDeterministicLongPrompt({
        ...(options.longPromptTargetTokens ? { targetTokens: options.longPromptTargetTokens } : {}),
        ...(options.longPromptRepeat ? { repeat: options.longPromptRepeat } : {}),
        ...(seedText ? { seedText } : {}),
      }),
      expectedSubstrings: expectedByIndex[0] ?? [],
      ...(expectedExact.length ? { expectedExact } : {}),
    }];
  }
  const repeatedPrompts = url.searchParams.getAll("prompt").map((text) => text.trim()).filter(Boolean);
  const pipePrompts = (url.searchParams.get("prompts") ?? "")
    .split("|")
    .map((text) => text.trim())
    .filter(Boolean);
  const prompts = repeatedPrompts.length > 0 ? repeatedPrompts : pipePrompts;
  return (prompts.length > 0 ? prompts : ["alpha beta"])
    .map((text, index) => {
      const expectedExact = expectedExactByIndex[index] ?? [];
      return {
        id: `prompt-${index + 1}`,
        text,
        expectedSubstrings: expectedByIndex[index] ?? [],
        ...(expectedExact.length ? { expectedExact } : {}),
      };
    });
}

function readLongPromptSeed(url: URL): string | undefined {
  const promptSeed = url.searchParams.get("promptSeed");
  if (promptSeed) return promptSeed.slice(0, MAX_LONG_PROMPT_SEED_CHARS);
  const promptParam = url.searchParams.get("prompt");
  if (promptParam) return promptParam.slice(0, MAX_LONG_PROMPT_SEED_CHARS);
  const promptsParam = url.searchParams.get("prompts");
  if (!promptsParam) return undefined;
  const bounded = promptsParam.slice(0, MAX_LONG_PROMPT_SEED_CHARS);
  const separatorIndex = bounded.indexOf("|");
  return separatorIndex >= 0 ? bounded.slice(0, separatorIndex) : bounded;
}

function readExpectedSubstrings(url: URL): string[][] {
  const json = url.searchParams.getAll("expectedJson").map(parseExpectedJsonList);
  if (json.length > 0) return json;
  const repeated = url.searchParams.getAll("expected")
    .map((value) => value.trim())
    .map((value) => value ? [value] : []);
  if (repeated.length > 0) return repeated;
  const pipe = url.searchParams.get("expectedSubstrings") ?? "";
  if (!pipe.trim()) return [];
  return pipe.split("|").map(splitExpectedList);
}

function readExpectedExact(url: URL): string[][] {
  const json = url.searchParams.getAll("expectedExactJson").map(parseExpectedJsonList);
  if (json.length > 0) return json;
  const repeated = url.searchParams.getAll("expectedExact")
    .map((value) => value.trim())
    .map((value) => value ? [value] : []);
  if (repeated.length > 0) return repeated;
  return [];
}

function parseExpectedJsonList(value: string): string[] {
  try {
    const parsed = JSON.parse(value) as unknown;
    if (!Array.isArray(parsed)) return [];
    return parsed
      .filter((item): item is string => typeof item === "string")
      .map((item) => item.trim())
      .filter(Boolean);
  } catch {
    return [];
  }
}

function splitExpectedList(value: string): string[] {
  return value
    .split(",")
    .map((item) => item.trim())
    .filter(Boolean);
}

function parseBackendPreference(value: string | null | undefined): BackendPreference | undefined {
  const normalized = value?.trim().toLowerCase();
  if (normalized === "cpu" || normalized === "webgpu") return normalized;
  return undefined;
}

function summaryHasAnyWebGpuKernel(summary: UnlockedWebGpuCoverageSummary): boolean {
  return collectSummaryKernelBackends(summary).some((backend) => backend === "webgpu");
}

function collectSummaryKernelBackends(summary: UnlockedWebGpuCoverageSummary): UnlockedKernelBackend[] {
  return [
    summary.logitProjection.backend,
    ...summary.mlpLayers.map((layer) => layer.backend),
    ...summary.prefillProjectionBackends.layers.flatMap((layer) => [
      layer.qProjection,
      layer.kProjection,
      layer.vProjection,
      layer.oProjection,
    ]),
    ...summary.decodeProjectionBackends.layers.flatMap((layer) => [
      layer.qProjection,
      layer.kProjection,
      layer.vProjection,
      layer.oProjection,
    ]),
    ...summary.attentionBackends.prefillLayers.flatMap((layer) => [
      layer.attentionBackend,
      ...layer.packedHeadBackends,
    ]),
    ...summary.attentionBackends.decodeLayers.flatMap((layer) => [
      layer.attentionBackend,
      ...layer.packedHeadBackends,
    ]),
  ];
}

export function isCoherentBrowserPreviewResponse(
  response: string,
  generatedTokens: number,
  expectedSubstrings: string[] = [],
): boolean {
  const visible = stripProofScaffolding(response);
  const words = visible.match(/[A-Za-z0-9]+/g) ?? [];
  const alnumChars = visible.match(/[A-Za-z0-9]/g)?.length ?? 0;
  const dominantWordCount = maxWordFrequency(words);
  const expectedMatched = expectedSubstrings.length > 0
    && expectedSubstrings.some((expected) => visible.toLowerCase().includes(expected.toLowerCase()));
  const basicVisibleQuality = generatedTokens > 0
    && visible.length > 0
    && alnumChars / Math.max(1, visible.length) >= 0.25
    && !hasConsecutiveWordRepetition(words, 3)
    && visible !== "**"
    && visible !== "The";
  if (expectedMatched) return basicVisibleQuality;
  return generatedTokens > 0
    && visible.length >= 12
    && words.length >= 3
    && alnumChars / Math.max(1, visible.length) >= 0.35
    && dominantWordCount / Math.max(1, words.length) <= 0.6
    && !hasConsecutiveWordRepetition(words, 3)
    && visible !== "**"
    && visible !== "The";
}

function matchingExpectedSubstrings(response: string, expectedSubstrings: string[]): string[] {
  const visible = stripProofScaffolding(response).toLowerCase();
  return expectedSubstrings.filter((expected) => visible.includes(expected.toLowerCase()));
}

function matchingExpectedExact(
  response: string,
  expectedExact: string[],
): Array<{ expected: string; matched: boolean }> {
  const visible = stripProofScaffolding(response);
  return expectedExact
    .map((expected) => expected.trim())
    .filter(Boolean)
    .map((expected) => ({
      expected,
      matched: visible === expected,
    }));
}

function isExpectedAnswerOnlyResponse(response: string, expectedSubstrings: string[]): boolean {
  const visible = stripProofScaffolding(response).trim();
  if (!visible || expectedSubstrings.length === 0) return false;
  return expectedSubstrings.some((expected) => {
    const expectedTrimmed = expected.trim();
    if (!expectedTrimmed) return false;
    const escaped = escapeRegExp(expectedTrimmed);
    return new RegExp(`^${escaped}[\\s.!?]*$`, "i").test(visible);
  });
}

function escapeRegExp(value: string): string {
  return value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
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

function calculateTokensPerSecond(generatedTokens: number, decodeLatencyMs: number): number | null {
  if (!Number.isFinite(generatedTokens) || generatedTokens <= 0) return null;
  if (!Number.isFinite(decodeLatencyMs) || decodeLatencyMs <= 0) return null;
  return roundMetric(generatedTokens / (decodeLatencyMs / 1000));
}

function countKvEvents(
  events: Array<{ operation: KVSwapPersistenceOperation }>,
  operation: KVSwapPersistenceOperation,
): number {
  return events.filter((event) => event.operation === operation).length;
}

function roundMs(value: number): number {
  return Math.round(value * 100) / 100;
}

function roundMetric(value: number): number {
  return Math.round(value * 1000) / 1000;
}

function escapeJsonForScript(json: string): string {
  return json.replaceAll("<", "\\u003c");
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function isFiniteNumber(value: unknown): value is number {
  return typeof value === "number" && Number.isFinite(value);
}
