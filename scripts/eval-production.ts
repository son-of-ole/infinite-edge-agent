import {
  buildFallbackTSPPlan,
  buildImmediateGacIngestionPlan,
  buildWakeContext,
  deserializeKVTensorBlock,
  evaluateSpeculationAutoDisable,
  executeTSPSchedule,
  FallbackSSARuntime,
  getNativeEdgeLayerTensorHandles,
  KVTensorPagingRegistry,
  NativeEdgeReferenceBackend,
  packContext,
  planKVSwap,
  proposeModelMemoryAction,
  readSsaToyTensorHandle,
  rollbackSleepCycle,
  serializeKVTensorBlock,
  shouldDisableSpeculation,
  shouldRunSleepCycle,
  verifySpeculativeBatch,
  type ContextBlock,
  type ContextPackTraceRecord,
  type GacWriteResult,
  type KVBlock,
  type MemoryChunk,
  type MemorySearchHit,
} from "@infinite-edge-agent/core";
import { spawn, type ChildProcess } from "node:child_process";
import { mkdir, writeFile } from "node:fs/promises";
import { join } from "node:path";

interface GateResult {
  name: string;
  passed: boolean;
  actual: number | string | boolean;
  expected: number | string | boolean;
}

interface SuiteResult {
  name: string;
  passed: boolean;
  metrics: Record<string, number | string | boolean>;
  gates: GateResult[];
}

type SidecarMode = "auto" | "required" | "skip";

interface SidecarConfig {
  mode: SidecarMode;
  baseUrl: string;
  expectedDbUri: string;
  expectedTableName: string;
  startCommand: string;
  timeoutMs: number;
}

interface SidecarHealth {
  ok?: boolean;
  dbUri?: string;
  tableName?: string;
}

const createdAt = new Date().toISOString();
const timestamp = createdAt.replace(/[:.]/g, "-");
const artifactRoot = process.env.EVAL_ARTIFACT_DIR ?? ".artifacts/evals";
const suiteDir = join(artifactRoot, "production", timestamp);
const sidecarConfig = readSidecarConfig();
const UNLOCKED_TARGET_MODEL_ID = "Qwen/Qwen3-0.6B";
const BROWSER_DRAFT_MODEL_ID = "browser/qwen-prefix-drafter";
const UNLOCKED_BROWSER_BACKEND = "unlocked-browser-transformer";

let sidecarProcess: ChildProcess | undefined;
try {
  const sidecarSetup = await prepareSidecar(sidecarConfig);
  sidecarProcess = sidecarSetup.process;

  const suites: SuiteResult[] = [
    evalMemoryRecall(),
    evalGacIngestionAndContradictions(),
    evalContextRebuild(),
    await evalSsaRouting(),
    await evalNativeSsaBridge(),
    evalTspPlanning(),
    await evalTspScheduleExecution(),
    evalMtpFallback(),
    await evalMtpVerifierBatching(),
    evalKvSwapPolicy(),
    evalKvTensorPaging(),
    await evalSidecarBehavior(sidecarConfig, sidecarSetup.available, sidecarSetup.reason),
    await evalModelMemoryActions(),
    evalSleepWakeContinuity(),
  ];

  const artifact = {
    name: "production-readiness",
    createdAt,
    passed: suites.every((suite) => suite.passed),
    sidecar: {
      mode: sidecarConfig.mode,
      baseUrl: sidecarConfig.baseUrl,
      startedByEval: Boolean(sidecarSetup.process),
      available: sidecarSetup.available,
      reason: sidecarSetup.reason,
    },
    suites,
  };

  await mkdir(suiteDir, { recursive: true });
  await writeFile(join(suiteDir, "results.json"), `${JSON.stringify(artifact, null, 2)}\n`);
  await writeFile(join(suiteDir, "trace.jsonl"), `${suites.map((suite) => JSON.stringify({ event: "suite", createdAt, suite })).join("\n")}\n`);
  await writeFile(join(suiteDir, "summary.md"), buildSummary(artifact));
  await writeFile(join(artifactRoot, "production-latest.json"), `${JSON.stringify(artifact, null, 2)}\n`);

  console.log(`Production eval: ${artifact.passed ? "PASS" : "FAIL"}`);
  console.log(`Sidecar: ${artifact.sidecar.available ? "available" : artifact.sidecar.mode === "skip" ? "skipped" : "unavailable"} (${artifact.sidecar.reason})`);
  console.log(`Results: ${join(suiteDir, "results.json")}`);
  console.log(`Summary: ${join(suiteDir, "summary.md")}`);

  if (!artifact.passed) {
    throw new Error("Production readiness eval failed acceptance gates. See the summary artifact for the failing gate and remediation hint.");
  }
} finally {
  if (sidecarProcess) {
    sidecarProcess.kill("SIGTERM");
  }
}

function evalMemoryRecall(): SuiteResult {
  const corpus = buildMemoryCorpus();
  const queries = corpus.filter((memory) => memory.tags.includes("eval-query"));
  const recallHits = queries.filter((query) => topK(corpus, query.embedding, 10).some((hit) => hit.id === query.id)).length;
  const pinned = corpus.filter((memory) => memory.metadata.gac && (memory.metadata.gac as { memoryClass?: string }).memoryClass === "PINNED_EXACT");
  const pinnedHits = pinned.filter((query) => topK(corpus, query.embedding, 5).some((hit) => hit.id === query.id)).length;
  const provenanceComplete = corpus.every((memory) => Boolean(memory.id && memory.sessionId && memory.source && memory.createdAt));
  const recallAt10 = recallHits / queries.length;
  const pinnedRecallAt5 = pinnedHits / pinned.length;
  return suite("memory-recall", {
    recallAt10,
    pinnedRecallAt5,
    provenanceComplete,
    queryCount: queries.length,
  }, [
    gate("recall@10", recallAt10, ">= 0.90", recallAt10 >= 0.9),
    gate("pinned recall@5", pinnedRecallAt5, ">= 0.98", pinnedRecallAt5 >= 0.98),
    gate("provenance completeness", provenanceComplete, true, provenanceComplete),
  ]);
}

function evalGacIngestionAndContradictions(): SuiteResult {
  const plan = buildImmediateGacIngestionPlan({
    tenantId: "eval_tenant",
    cellId: "eval_cell",
    sessionId: "eval_session",
    sourceType: "chat",
    sourceUri: "chat://eval_session/gac",
    chunks: [
      makeEvalChunk("gac_allow", "Use Sandbox in production.", [1, 0, 0]),
      makeEvalChunk("gac_deny", "Remember: do not use Sandbox in production.", [0.95, 0.05, 0]),
    ],
    now: new Date(createdAt),
  });
  const modelVisibleRepresentatives = plan.representatives.filter((record) => record.modelVisible || record.factual);
  const representativeLineageCoverage = modelVisibleRepresentatives.length === 0
    ? 1
    : modelVisibleRepresentatives.filter((record) => plan.lineage.some((lineage) => lineage.representativeId === record.id)).length / modelVisibleRepresentatives.length;
  const privateTrainingBlocked = plan.trainingExamples.every((record) => record.privacyClass !== "private" || record.exportAllowed === false);
  const rawMemoryWritten = plan.rawMemory.length === 2;
  const identityPinned = plan.identityPins.some((pin) => pin.pinReason === "user_instruction");
  const clusterMetricComputed = plan.clusterMetrics.length === 1 && (plan.clusterMetrics[0]?.contradictionScore ?? 0) > 0;
  const contradictionDetected = plan.contradictions.length === 1;
  return suite("gac-ingestion", {
    rawMemoryCount: plan.rawMemory.length,
    identityPinCount: plan.identityPins.length,
    representativeCount: plan.representatives.length,
    representativeLineageCoverage,
    contradictionCount: plan.contradictions.length,
    privateTrainingBlocked,
  }, [
    gate("raw memory written", rawMemoryWritten, true, rawMemoryWritten),
    gate("identity pin policy", identityPinned, true, identityPinned),
    gate("cluster metric computed", clusterMetricComputed, true, clusterMetricComputed),
    gate("representative lineage coverage", representativeLineageCoverage, 1, representativeLineageCoverage === 1),
    gate("contradiction eval", contradictionDetected, true, contradictionDetected),
    gate("training export policy", privateTrainingBlocked, true, privateTrainingBlocked),
  ]);
}

function evalContextRebuild(): SuiteResult {
  const memory = buildMemoryCorpus().map((hit, index) => ({ ...hit, score: index === 0 ? 0.99 : hit.score }));
  const packed = packContext({
    systemPrompt: "You are an eval agent. Preserve required anchors and stay inside budget.",
    retrievedMemory: memory,
    recentMessages: [
      {
        id: "recent_1",
        role: "user",
        content: "Recent state: production eval is running.",
        createdAt,
        sessionId: "eval_session",
      },
    ],
    userMessage: "Use the pinned project constraint and report the eval gate status.",
    config: {
      maxPromptTokens: 900,
      maxRecentConversationTokens: 120,
      maxRetrievedMemoryTokens: 500,
    },
  });
  const requiredAnchorIncluded = packed.includedMemoryIds.includes("memory_pin_0");
  return suite("context-rebuild", {
    requiredAnchorIncluded,
    estimatedTokens: packed.estimatedTokens,
    includedMemoryCount: packed.includedMemoryIds.length,
  }, [
    gate("required anchor included", requiredAnchorIncluded, true, requiredAnchorIncluded),
    gate("token budget", packed.estimatedTokens, "<= 900", packed.estimatedTokens <= 900),
  ]);
}

function makeEvalChunk(id: string, text: string, embedding: number[]): MemoryChunk {
  return {
    id,
    text,
    embedding,
    sessionId: "eval_session",
    source: "chat",
    role: "user",
    createdAt,
    updatedAt: createdAt,
    tags: ["eval"],
    metadata: {},
    tokenCount: Math.max(1, Math.ceil(text.length / 4)),
  };
}

async function evalSsaRouting(): Promise<SuiteResult> {
  const blocks = buildSsaBlocks();
  const memoryHits = buildMemoryCorpus().slice(0, 4);
  const plan = await new FallbackSSARuntime().plan({
    requestId: "eval_ssa",
    activeBlocks: blocks,
    anchors: [{ blockId: "block_anchor", reason: "required_eval_anchor", score: 1 }],
    memoryHits,
    maxBlocks: 6,
    minAnchorScore: 0.5,
  });
  const relevantIds = ["block_anchor", "block_pin", "block_raw", "memory_pin_0"];
  const relevantIncluded = relevantIds.filter((id) => plan.selectedBlockIds.includes(id)).length / relevantIds.length;
  return suite("ssa-routing", {
    relevantIncluded,
    selectedBlocks: plan.selectedBlockIds.length,
    sparsityRatio: Number(plan.sparsityRatio.toFixed(4)),
  }, [
    gate("relevant block inclusion", relevantIncluded, ">= 0.95", relevantIncluded >= 0.95),
    gate("anchor preserved", plan.selectedBlockIds.includes("block_anchor"), true, plan.selectedBlockIds.includes("block_anchor")),
  ]);
}

async function evalNativeSsaBridge(): Promise<SuiteResult> {
  const backend = new NativeEdgeReferenceBackend({ backendPreference: "cpu", headDim: 2 });
  await backend.initializeModel("native-edge-reference:evaluation");

  const policy = {
    layerIndex: 0,
    blockSize: 2,
    topKBlocks: 2,
    localWindowBlocks: 0,
    pinnedBlockIds: ["b0"],
    selectedBlockIdsByQueryBlock: {
      0: ["b0", "b1"],
      1: ["b0", "b1"],
    },
    denseFallback: true,
  };
  const prefill = await backend.prefill(new Int32Array([1, 1, 2, 3]), {
    requestId: "eval_native_ssa",
    layerPolicies: [policy],
  });
  const handles = getNativeEdgeLayerTensorHandles(prefill.kvCacheHandle, 0);
  const output = await backend.executeSparseLayer({
    requestId: prefill.requestId,
    layerIndex: 0,
    qHandle: handles.qHandle,
    kHandle: handles.kHandle,
    vHandle: handles.vHandle,
    policy,
  });
  const firstDecode = await backend.decode({
    requestId: prefill.requestId,
    inputTokenId: 5,
    kvCacheHandle: prefill.kvCacheHandle,
    policy: [policy],
  });
  const secondDecode = await backend.decode({
    requestId: prefill.requestId,
    inputTokenId: 5,
    kvCacheHandle: prefill.kvCacheHandle,
    policy: [policy],
  });
  const outputMatrix = readSsaToyTensorHandle(output.outputHandle).matrix;
  await backend.dispose();

  const selectedAllBlocks = output.selectedBlockIds.join(",") === "b0,b1";
  const denseReferencePassed = output.denseReference?.passed === true;
  const deterministicDecode = firstDecode.tokenId === secondDecode.tokenId;
  return suite("native-ssa-bridge", {
    supportsQkvAccess: backend.supportsQkvAccess,
    supportsLayerSparseRouting: backend.supportsLayerSparseRouting,
    outputRows: outputMatrix.length,
    selectedBlockIds: output.selectedBlockIds.join(","),
    denseReferencePassed,
    deterministicDecode,
  }, [
    gate("backend QKV access contract", backend.supportsQkvAccess, true, backend.supportsQkvAccess),
    gate("layer sparse routing contract", backend.supportsLayerSparseRouting, true, backend.supportsLayerSparseRouting),
    gate("sparse output rows", outputMatrix.length, 4, outputMatrix.length === 4),
    gate("selected blocks bridged", selectedAllBlocks, true, selectedAllBlocks),
    gate("dense reference validation", denseReferencePassed, true, denseReferencePassed),
    gate("deterministic decode", deterministicDecode, true, deterministicDecode),
  ]);
}

function evalTspPlanning(): SuiteResult {
  const plan = buildFallbackTSPPlan({
    device: {
      name: "eval-edge-device",
      vramBudgetBytes: 4 * 1024 * 1024 * 1024,
      ramBudgetBytes: 8 * 1024 * 1024 * 1024,
    },
    model: {
      modelId: UNLOCKED_TARGET_MODEL_ID,
      parameterBytes: 600_000_000,
      layers: 28,
      hiddenSize: 1024,
      kvHeads: 8,
    },
    requestedContextTokens: 4096,
    batchSize: 1,
    kvPrecisionBytes: 2,
    activationPrecisionBytes: 2,
    safetyMarginRatio: 0.2,
  });
  const usableVram = Math.floor(4 * 1024 * 1024 * 1024 * 0.8);
  return suite("tsp-planning", {
    estimatedVramBytes: plan.estimatedVramBytes,
    usableVramBytes: usableVram,
    scheduleSteps: plan.schedule.length,
  }, [
    gate("vram budget", plan.estimatedVramBytes, `<= ${usableVram}`, plan.estimatedVramBytes <= usableVram),
    gate("schedule non-empty", plan.schedule.length > 0, true, plan.schedule.length > 0),
  ]);
}

async function evalTspScheduleExecution(): Promise<SuiteResult> {
  const schedule = [
    { id: "attn_s0_t0", kind: "attention" as const, sequenceShard: 0, tensorShard: 0, tokenStart: 32, tokenEnd: 48 },
    { id: "prefetch_s0_t0", kind: "kv_prefetch" as const, sequenceShard: 0, tensorShard: 0, tokenStart: 0, tokenEnd: 16 },
    { id: "mlp_s1_t0", kind: "mlp" as const, sequenceShard: 1, tensorShard: 0, tokenStart: 48, tokenEnd: 64 },
  ];
  const calls: string[] = [];
  const trace = await executeTSPSchedule(schedule, {
    attention: ({ step }) => calls.push(step.id),
    kv_prefetch: ({ step }) => calls.push(step.id),
    mlp: ({ step }) => calls.push(step.id),
  }, {
    metadata: { requestId: "eval_tsp_execution", nested: { immutable: true } },
  });

  const preflightCalls: string[] = [];
  let preflightBlockedMutation = false;
  try {
    await executeTSPSchedule(schedule, {
      attention: ({ step }) => preflightCalls.push(step.id),
    });
  } catch {
    preflightBlockedMutation = preflightCalls.length === 0;
  }

  const orderPreservedWithinShard = calls.join(",") === "attn_s0_t0,prefetch_s0_t0,mlp_s1_t0";
  const metadataFrozen = Object.isFrozen(trace[0]?.metadata) && Object.isFrozen(trace[0]?.metadata.nested);
  return suite("tsp-schedule-execution", {
    traceCount: trace.length,
    callOrder: calls.join(","),
    preflightBlockedMutation,
    metadataFrozen,
  }, [
    gate("trace count", trace.length, schedule.length, trace.length === schedule.length),
    gate("planner order preserved within shard", orderPreservedWithinShard, true, orderPreservedWithinShard),
    gate("missing callback preflight", preflightBlockedMutation, true, preflightBlockedMutation),
    gate("metadata snapshot frozen", metadataFrozen, true, metadataFrozen),
  ]);
}

function evalMtpFallback(): SuiteResult {
  const disabledReason = shouldDisableSpeculation({
    draftTokens: 8,
    acceptedTokens: 2,
    rejectedTokens: 6,
    acceptanceRate: 0.25,
    draftLatencyMs: 80,
    verifyLatencyMs: 120,
    netSpeedupRatio: 0.8,
  }, 0.45);
  return suite("mtp-speculation", {
    disabledReason: disabledReason ?? "",
  }, [
    gate("disables when worse", disabledReason === "acceptance_rate_below_threshold", "acceptance_rate_below_threshold", disabledReason === "acceptance_rate_below_threshold"),
  ]);
}

async function evalMtpVerifierBatching(): Promise<SuiteResult> {
  const result = await verifySpeculativeBatch({
    requestId: "eval_mtp_batch",
    modelPair: {
      draftModelId: BROWSER_DRAFT_MODEL_ID,
      targetModelId: UNLOCKED_TARGET_MODEL_ID,
    },
    taskType: "chat",
    draftLatencyMs: 18,
    targetOnlyLatencyMs: 120,
    minAcceptanceRate: 0.45,
    disableWhenLatencyWorse: true,
    branches: [
      { branchId: "branch_accept", draft: [{ token: "Ut" }, { token: "ah" }] },
      { branchId: "branch_correct", draft: [{ token: "is" }, { token: "Denver" }] },
    ],
  }, () => ({
    requestId: "eval_mtp_batch",
    verifyLatencyMs: 24,
    branches: [
      {
        branchId: "branch_accept",
        verification: [
          { token: "Ut", accepted: true },
          { token: "ah", accepted: true },
        ],
      },
      {
        branchId: "branch_correct",
        verification: [
          { token: "is", accepted: true },
          { token: "Denver", accepted: false, replacement: "Salt Lake City" },
        ],
      },
    ],
  }));
  const health = evaluateSpeculationAutoDisable([result.metrics], {
    minAcceptanceRate: 0.45,
    disableWhenLatencyWorse: true,
  });
  const correctionStreamed = result.branches.some((branch) => branch.correctedToken === "Salt Lake City");
  const metricsScoped = result.metrics.modelPair.targetModelId === UNLOCKED_TARGET_MODEL_ID && result.metrics.taskType === "chat";
  const acceptanceRate = Number(result.metrics.acceptanceRate.toFixed(4));
  return suite("mtp-verifier-batching", {
    branchCount: result.branches.length,
    traceCount: result.traces.length,
    acceptanceRate,
    correctionStreamed,
    metricsScoped,
    healthDisabled: health.disabled,
  }, [
    gate("branch count", result.branches.length, 2, result.branches.length === 2),
    gate("trace count", result.traces.length, 2, result.traces.length === 2),
    gate("correction streamed", correctionStreamed, true, correctionStreamed),
    gate("metrics scoped", metricsScoped, true, metricsScoped),
    gate("healthy speculation not disabled", health.disabled, false, !health.disabled),
  ]);
}

function evalKvSwapPolicy(): SuiteResult {
  const blocks = [
    { id: "kv_pin", layer: 0, startToken: 0, endToken: 64, tier: "vram" as const, pinned: true, importance: 1, lastAccessAt: 1000, estimatedBytes: 1024 },
    { id: "kv_task", layer: 0, startToken: 64, endToken: 128, tier: "vram" as const, pinned: false, importance: 0.8, lastAccessAt: 900, estimatedBytes: 1024 },
    { id: "kv_cold", layer: 0, startToken: 128, endToken: 192, tier: "vram" as const, pinned: false, importance: 0.1, lastAccessAt: 0, estimatedBytes: 2048 },
  ];
  const first = planKVSwap(blocks, { mode: "metadata_only", vramPressureThreshold: 0.82, ramPressureThreshold: 0.85, now: 2000 }, 1024, []);
  const second = planKVSwap(blocks, { mode: "metadata_only", vramPressureThreshold: 0.82, ramPressureThreshold: 0.85, now: 2000 }, 1024, []);
  const pinnedEvicted = first.evictBlockIds.includes("kv_pin");
  const deterministic = JSON.stringify(first) === JSON.stringify(second);
  return suite("kvswap-policy", {
    pinnedEvicted,
    deterministic,
    estimatedBytesFreed: first.estimatedBytesFreed,
  }, [
    gate("pinned eviction violations", pinnedEvicted ? 1 : 0, 0, !pinnedEvicted),
    gate("deterministic output", deterministic, true, deterministic),
  ]);
}

function evalKvTensorPaging(): SuiteResult {
  const registry = new KVTensorPagingRegistry({ now: 3000, defaultEvictionTier: "disk" });
  const pinned = makeKvBlock("kv_pin", "vram", true, 1, 1024);
  const warm = makeKvBlock("kv_warm_selected", "ram", false, 0.8, 1024);
  const cold = makeKvBlock("kv_cold_selected", "disk", false, 0.4, 1024);
  registry.registerBlock(pinned, makeTensorHandles("kv_pin"));
  registry.registerBlock(warm, makeTensorHandles("kv_warm_selected"));
  registry.registerBlock(cold, makeTensorHandles("kv_cold_selected"));

  const decision = planKVSwap(registry.listBlocks(), {
    mode: "predictive",
    vramPressureThreshold: 0.82,
    ramPressureThreshold: 0.85,
    now: 3000,
    vramBudgetBytes: 2048,
    ramBudgetBytes: 4096,
  }, 1024, ["kv_warm_selected", "kv_cold_selected"]);
  const applyEvents = registry.applyKVSwapDecision(decision);
  const readiness = registry.ensureBlocksAvailableForSparseAttention(["kv_pin", "kv_warm_selected", "kv_cold_selected"]);
  const serialized = serializeKVTensorBlock({
    ...cold,
    compressedKeySummary: new Float32Array([0.25, 0.5, 0.75]),
  });
  const restored = deserializeKVTensorBlock(serialized);
  const allSelectedHot = readiness.availableBlockIds.every((id) => registry.getResidencyTier(id) === "hot");
  const pinnedStillHot = registry.getBlock("kv_pin")?.tier === "vram";
  const restoredSummaryTyped = restored.compressedKeySummary instanceof Float32Array;
  const pagingEventCount = applyEvents.length + readiness.events.length;
  return suite("kv-tensor-paging", {
    availableBlockIds: readiness.availableBlockIds.join(","),
    pagingEvents: pagingEventCount,
    pinnedStillHot,
    restoredSummaryTyped,
  }, [
    gate("selected blocks vram ready", allSelectedHot, true, allSelectedHot),
    gate("pinned block retained", pinnedStillHot, true, pinnedStillHot),
    gate("disk summary restores typed array", restoredSummaryTyped, true, restoredSummaryTyped),
    gate("prefetch events emitted", pagingEventCount, ">= 1", pagingEventCount >= 1),
  ]);
}

async function evalSidecarBehavior(config: SidecarConfig, available: boolean, reason: string): Promise<SuiteResult> {
  const baseUrl = config.baseUrl;
  if (config.mode === "skip") {
    return suite("sidecar-behavior", {
      skipped: true,
      reason,
      baseUrl,
    }, [
      gate("sidecar skipped", true, "PRODUCTION_EVAL_SIDECAR_MODE=skip", true),
    ]);
  }

  if (!available) {
    const optional = config.mode === "auto";
    const hint = config.startCommand
      ? `start command failed or did not become healthy within ${config.timeoutMs} ms`
      : "start the memory sidecar or set PRODUCTION_EVAL_SIDECAR_START_COMMAND";
    return suite("sidecar-behavior", {
      skipped: optional,
      reason,
      baseUrl,
      hint,
    }, [
      gate("sidecar availability", reason, optional ? "optional sidecar unavailable; required mode not set" : "running sidecar", optional),
    ]);
  }

  const id = `eval_sidecar_${Date.now()}`;
  const embedding = makeVector(384, 7);
  const chunk: MemoryChunk = {
    id,
    text: "Production sidecar eval chunk for LanceDB search, status, repair, export, and trace persistence.",
    embedding,
    sessionId: "eval_sidecar_session",
    source: "system",
    createdAt,
    updatedAt: createdAt,
    tags: ["eval", "sidecar"],
    metadata: {},
    tokenCount: 14,
  };

  const health = await fetchJson<{ ok?: boolean }>(`${baseUrl}/health`);
  const status = await fetchJson<{ ok?: boolean; memoryTable?: { exists?: boolean; vectorDimension?: number | null } }>(`${baseUrl}/memory/status`);
  await postJson(`${baseUrl}/memory/upsert`, { chunks: [chunk] });
  const search = await postJson<{ hits?: MemorySearchHit[] }>(`${baseUrl}/memory/search`, {
    embedding,
    options: { sessionId: chunk.sessionId, tags: ["eval"], limit: 1 },
  });
  await postJson(`${baseUrl}/runtime/traces`, {
    trace: {
      traceId: id,
      requestId: id,
      sessionId: chunk.sessionId,
      modelId: UNLOCKED_TARGET_MODEL_ID,
      backend: UNLOCKED_BROWSER_BACKEND,
      createdAt,
      runtime: { suite: "sidecar-behavior" },
    },
  });
  const traces = await fetchJson<{ traces?: Array<{ traceId: string }> }>(`${baseUrl}/runtime/traces?sessionId=${encodeURIComponent(chunk.sessionId)}&limit=1`);
  const exported = await fetchJson<{ chunks?: MemoryChunk[] }>(`${baseUrl}/memory/export?sessionId=${encodeURIComponent(chunk.sessionId)}&limit=1`);
  await deleteJson(`${baseUrl}/memory/query`, { options: { sessionId: chunk.sessionId, tags: ["eval"] } });

  const searchHit = search.hits?.[0]?.id === id;
  const tracePersisted = traces.traces?.some((trace) => trace.traceId === id) ?? false;
  const exportLimited = exported.chunks?.length === 1;
  return suite("sidecar-behavior", {
    healthOk: health.ok === true,
    memoryTableExists: status.memoryTable?.exists === true,
    vectorDimension: status.memoryTable?.vectorDimension ?? 0,
    searchHit,
    tracePersisted,
    exportLimited,
  }, [
    gate("health", health.ok === true, true, health.ok === true),
    gate("memory status", Boolean(status.memoryTable), true, Boolean(status.memoryTable)),
    gate("search hit", searchHit, true, searchHit),
    gate("runtime trace persisted", tracePersisted, true, tracePersisted),
    gate("export limit respected", exportLimited, true, exportLimited),
  ]);
}

function readSidecarConfig(): SidecarConfig {
  const rawMode = process.env.PRODUCTION_EVAL_SIDECAR_MODE ?? process.env.EVAL_SIDECAR_MODE ?? "auto";
  const mode = parseSidecarMode(rawMode);
  return {
    mode,
    baseUrl: (process.env.MEMORY_SERVER_URL ?? process.env.VITE_MEMORY_SERVER_URL ?? "http://127.0.0.1:8787").replace(/\/$/, ""),
    expectedDbUri: process.env.PRODUCTION_EVAL_EXPECTED_DB_URI ?? process.env.MEMORY_DB_URI ?? ".data/lancedb",
    expectedTableName: process.env.PRODUCTION_EVAL_EXPECTED_TABLE_NAME ?? process.env.MEMORY_TABLE ?? "memory_chunks",
    startCommand: process.env.PRODUCTION_EVAL_SIDECAR_START_COMMAND ?? process.env.EVAL_SIDECAR_START_COMMAND ?? "",
    timeoutMs: readPositiveInt(process.env.PRODUCTION_EVAL_SIDECAR_TIMEOUT_MS ?? process.env.EVAL_SIDECAR_TIMEOUT_MS, 12_000),
  };
}

function parseSidecarMode(value: string): SidecarMode {
  if (value === "auto" || value === "required" || value === "skip") return value;
  throw new Error(`Invalid PRODUCTION_EVAL_SIDECAR_MODE=${value}. Expected auto, required, or skip.`);
}

function readPositiveInt(value: string | undefined, fallback: number): number {
  if (!value) return fallback;
  const parsed = Number(value);
  if (!Number.isFinite(parsed) || parsed <= 0) {
    throw new Error(`Invalid positive integer: ${value}`);
  }
  return Math.floor(parsed);
}

async function prepareSidecar(config: SidecarConfig): Promise<{ available: boolean; reason: string; process?: ChildProcess }> {
  if (config.mode === "skip") {
    return { available: false, reason: "disabled by PRODUCTION_EVAL_SIDECAR_MODE=skip" };
  }

  const existing = await checkSidecarHealth(config, Math.min(config.timeoutMs, 2_000));
  if (existing.ok) {
    return { available: true, reason: "existing sidecar health check passed" };
  }

  if (!config.startCommand) {
    return {
      available: false,
      reason: humanizeSidecarError(existing.error, config),
    };
  }

  const child = spawn(config.startCommand, {
    shell: true,
    stdio: "ignore",
    env: process.env,
    detached: false,
  });
  const started = await waitForSidecar(config, config.timeoutMs);
  if (started.ok) {
    return { available: true, reason: `started sidecar with PRODUCTION_EVAL_SIDECAR_START_COMMAND`, process: child };
  }

  child.kill("SIGTERM");
  return {
    available: false,
    reason: humanizeSidecarError(started.error, config),
  };
}

async function waitForSidecar(config: SidecarConfig, timeoutMs: number): Promise<{ ok: true } | { ok: false; error: unknown }> {
  const started = Date.now();
  let lastError: unknown;
  while (Date.now() - started < timeoutMs) {
    const health = await checkSidecarHealth(config, 1_500);
    if (health.ok) return health;
    lastError = health.error;
    await sleep(500);
  }
  return { ok: false, error: lastError ?? new Error(`Timed out after ${timeoutMs} ms`) };
}

async function checkSidecarHealth(config: SidecarConfig, timeoutMs: number): Promise<{ ok: true } | { ok: false; error: unknown }> {
  try {
    const health = await fetchJson<SidecarHealth>(`${config.baseUrl}/health`, timeoutMs);
    if (health.ok !== true) {
      return { ok: false, error: new Error(`${config.baseUrl}/health did not return ok: true`) };
    }
    if (health.dbUri !== config.expectedDbUri || health.tableName !== config.expectedTableName) {
      return {
        ok: false,
        error: new Error(
          `memory sidecar profile mismatch at ${config.baseUrl}: expected dbUri=${config.expectedDbUri} tableName=${config.expectedTableName}, received dbUri=${health.dbUri ?? "unknown"} tableName=${health.tableName ?? "unknown"}`,
        ),
      };
    }
    return { ok: true };
  } catch (error) {
    return { ok: false, error };
  }
}

function humanizeSidecarError(error: unknown, config: SidecarConfig): string {
  const message = error instanceof Error ? error.message : String(error);
  if (/profile mismatch/i.test(message)) return message;
  const connectionRefused = /ECONNREFUSED|fetch failed|Failed to fetch|connection refused/i.test(message);
  if (connectionRefused) {
    return `memory sidecar is not reachable at ${config.baseUrl}; start it with \`pnpm dev:memory\`, set PRODUCTION_EVAL_SIDECAR_START_COMMAND, or use PRODUCTION_EVAL_SIDECAR_MODE=skip for browser-only gates`;
  }
  return `memory sidecar check failed at ${config.baseUrl}: ${message}`;
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

async function evalModelMemoryActions(): Promise<SuiteResult> {
  let executed = false;
  const shadow = await proposeModelMemoryAction({
    tenantId: "tenant_eval",
    cellId: "cell_eval",
    sessionId: "eval_session",
    modelId: UNLOCKED_TARGET_MODEL_ID,
    actionType: "pin_memory",
    targetIds: ["raw_pin_0"],
    arguments: { pinReason: "architecture_decision", pinStrength: 0.95 },
    confidence: 0.92,
  }, {
    mode: "shadow",
    policy: { scope: { tenantId: "tenant_eval", cellId: "cell_eval" } },
    execute: () => {
      executed = true;
      return ["pin_shadow"];
    },
  });
  const enforced = await proposeModelMemoryAction({
    tenantId: "tenant_eval",
    cellId: "cell_eval",
    sessionId: "eval_session",
    modelId: UNLOCKED_TARGET_MODEL_ID,
    actionType: "request_retrieval_probe",
    targetIds: ["raw_pin_0"],
    confidence: 0.9,
  }, {
    mode: "enforced",
    policy: { scope: { tenantId: "tenant_eval", cellId: "cell_eval" } },
    execute: () => ["audit_1"],
  });
  const rejected = await proposeModelMemoryAction({
    tenantId: "tenant_eval",
    cellId: "other_cell",
    sessionId: "eval_session",
    modelId: UNLOCKED_TARGET_MODEL_ID,
    actionType: "forget_memory",
    targetIds: ["raw_pin_0"],
    confidence: 0.99,
  }, {
    mode: "enforced",
    policy: { scope: { tenantId: "tenant_eval", cellId: "cell_eval" } },
  });
  return suite("model-memory-actions", {
    shadowApproved: shadow.approved,
    shadowExecuted: shadow.executed,
    enforcedExecuted: enforced.executed,
    rejectedApproved: rejected.approved,
    traceHasViolations: rejected.trace.policyViolations.length > 0,
  }, [
    gate("shadow approved", shadow.approved, true, shadow.approved),
    gate("shadow does not execute", shadow.executed || executed, false, !shadow.executed && !executed),
    gate("enforced executes", enforced.executed, true, enforced.executed),
    gate("unsafe rejected", rejected.approved, false, !rejected.approved),
    gate("policy trace", rejected.trace.policyViolations.length > 0, true, rejected.trace.policyViolations.length > 0),
  ]);
}

function evalSleepWakeContinuity(): SuiteResult {
  const rawMemories = [
    {
      id: "raw_pin_0",
      tenantId: "tenant_eval",
      cellId: "cell_eval",
      sessionId: "eval_session",
      sourceType: "chat" as const,
      text: "The unlocked browser transformer with Qwen/Qwen3-0.6B is the required full-control model target.",
      canonicalText: "Use the unlocked browser transformer with Qwen/Qwen3-0.6B as the required SSA/KV/TSP target.",
      memoryKind: "instruction" as const,
      importance: 1,
      identityRiskSeed: 1,
      createdAt,
      updatedAt: createdAt,
      retentionClass: "pinned" as const,
      hash: "hash_raw_pin_0",
    },
    {
      id: "raw_decision_0",
      tenantId: "tenant_eval",
      cellId: "cell_eval",
      sessionId: "eval_session",
      sourceType: "system" as const,
      text: "Browser-vector IndexedDB memory is the zero-config local production default; LanceDB sidecar is an optional scale layer.",
      memoryKind: "decision" as const,
      importance: 0.9,
      identityRiskSeed: 0.8,
      createdAt,
      updatedAt: createdAt,
      retentionClass: "normal" as const,
      hash: "hash_raw_decision_0",
    },
  ];
  const result = buildWakeContext({
    tenantId: "tenant_eval",
    cellId: "cell_eval",
    sessionId: "eval_session",
    runId: "run_eval_sleep",
    currentGoal: "Finish production readiness.",
    rawMemories,
    identityPins: [
      {
        id: "pin_0",
        tenantId: "tenant_eval",
        cellId: "cell_eval",
        sessionId: "eval_session",
        rawMemoryId: "raw_pin_0",
        pinReason: "architecture_decision",
        pinStrength: 1,
        createdBy: "policy",
        createdAt,
      },
    ],
    contextPackTraces: [
      {
        id: "ctx_0",
        traceId: "trace_0",
        tenantId: "tenant_eval",
        cellId: "cell_eval",
        sessionId: "eval_session",
        queryId: "query_0",
        contextPackId: "pack_0",
        rawMemoryIds: ["raw_pin_0"],
        representativeIds: [],
        identityPinIds: ["pin_0"],
        tokenBudget: 4096,
        packingStrategy: "eval",
        includedMemoryIds: ["raw_pin_0"],
        createdAt,
      } satisfies ContextPackTraceRecord,
    ],
    runtimeTraces: [
      {
        traceId: "trace_0",
        requestId: "query_0",
        sessionId: "eval_session",
        modelId: UNLOCKED_TARGET_MODEL_ID,
        backend: UNLOCKED_BROWSER_BACKEND,
        createdAt,
        runtime: {},
      },
    ],
    openTasks: ["Run production evals"],
    risks: ["Missing user-provided unlocked Qwen manifest keeps releases in deterministic fixture mode."],
    now: new Date(createdAt),
  });
  const duplicate = buildWakeContext({
    tenantId: "tenant_eval",
    cellId: "cell_eval",
    sessionId: "eval_session",
    runId: "run_eval_sleep",
    currentGoal: "Finish production readiness.",
    rawMemories,
    identityPins: result.retrievalAudits.map((_audit) => ({
      id: "pin_0",
      tenantId: "tenant_eval",
      cellId: "cell_eval",
      sessionId: "eval_session",
      rawMemoryId: "raw_pin_0",
      pinReason: "architecture_decision" as const,
      pinStrength: 1,
      createdBy: "policy" as const,
      createdAt,
    })),
    contextPackTraces: [],
    runtimeTraces: [],
    now: new Date(createdAt),
  });
  const rollback = rollbackSleepCycle({ runRecord: result.runRecord, reason: "eval rollback", now: new Date(createdAt) });
  const disabled = shouldRunSleepCycle({ enabled: false, pendingRawMemoryCount: rawMemories.length });
  const hasWakeSections = [
    "Cell Identity",
    "Current Goal",
    "Pinned Constraints",
    "Open Tasks",
    "Memory Map",
  ].every((section) => result.wakeContext.markdown.includes(`## ${section}`));
  const pinLinked = result.wakeContext.markdown.includes("[raw:raw_pin_0]");
  return suite("sleep-wake-continuity", {
    hasWakeSections,
    pinLinked,
    deterministicWakeId: result.wakeContext.id === duplicate.wakeContext.id,
    rollbackStatus: rollback.runRecord.status,
    disabledReason: disabled.reason,
    retrievalAuditCount: result.retrievalAudits.length,
  }, [
    gate("wake sections", hasWakeSections, true, hasWakeSections),
    gate("pinned raw link", pinLinked, true, pinLinked),
    gate("deterministic run ids", result.wakeContext.id === duplicate.wakeContext.id, true, result.wakeContext.id === duplicate.wakeContext.id),
    gate("rollback status", rollback.runRecord.status, "rolled_back", rollback.runRecord.status === "rolled_back"),
    gate("disable policy", disabled.reason, "disabled_by_policy", disabled.reason === "disabled_by_policy"),
    gate("retrieval audits", result.retrievalAudits.length, ">= 1", result.retrievalAudits.length >= 1),
  ]);
}

function makeKvBlock(
  id: string,
  tier: KVBlock["tier"],
  pinned: boolean,
  importance: number,
  lastAccessAt: number,
): KVBlock {
  return {
    id,
    layer: 0,
    startToken: 0,
    endToken: 64,
    tier,
    pinned,
    importance,
    lastAccessAt,
    estimatedBytes: 1024,
    compressedKeySummary: new Float32Array([importance, lastAccessAt / 10_000]),
  };
}

function makeTensorHandles(id: string): NonNullable<KVBlock["tensorHandles"]> {
  return {
    key: { backend: "production-eval", id: `${id}:key`, dtype: "f32", shape: [64, 2], bytes: 512 },
    value: { backend: "production-eval", id: `${id}:value`, dtype: "f32", shape: [64, 2], bytes: 512 },
  };
}

function buildMemoryCorpus(): MemorySearchHit[] {
  return Array.from({ length: 12 }, (_, index) => {
    const pinned = index < 2;
    const id = pinned ? `memory_pin_${index}` : `memory_${index}`;
    return {
      id,
      text: pinned
        ? `Pinned project constraint ${index}: unlocked Qwen browser runtime must keep exact memory links.`
        : `General production memory ${index} for deterministic recall testing.`,
      embedding: makeVector(32, index),
      sessionId: "eval_session",
      source: "chat",
      role: index % 2 === 0 ? "user" : "assistant",
      createdAt,
      updatedAt: createdAt,
      tags: ["eval", index % 3 === 0 ? "eval-query" : "background", ...(pinned ? ["pinned"] : [])],
      metadata: pinned
        ? {
            gac: {
              memoryClass: "PINNED_EXACT",
              rawMemoryId: `raw_pin_${index}`,
              identityPinId: `pin_${index}`,
              pinStrength: 1,
              mustAttend: true,
            },
          }
        : { rawMemoryId: `raw_${index}` },
      tokenCount: pinned ? 18 : 12,
      score: 0,
    };
  });
}

function buildSsaBlocks(): ContextBlock[] {
  return [
    { id: "block_anchor", text: "current request", tokenStart: 0, tokenEnd: 32, priority: 1, source: "ledger", tags: ["current_user_request"] },
    {
      id: "block_pin",
      text: "pinned exact memory",
      tokenStart: 32,
      tokenEnd: 64,
      priority: 0.8,
      source: "memory",
      gac: { blockId: "block_pin", memoryClass: "PINNED_EXACT", rawMemoryId: "raw_pin_0", identityRisk: 1, pinStrength: 1, mustAttend: true },
    },
    {
      id: "block_raw",
      text: "high risk raw memory",
      tokenStart: 64,
      tokenEnd: 96,
      priority: 0.7,
      source: "memory",
      gac: { blockId: "block_raw", memoryClass: "HIGH_RISK_RAW", rawMemoryId: "raw_7", identityRisk: 0.9 },
    },
    ...buildMemoryCorpus().slice(0, 4).map((memory, index) => ({
      id: memory.id,
      text: memory.text,
      tokenStart: 96 + index * 32,
      tokenEnd: 128 + index * 32,
      priority: 0.5,
      source: "memory",
      tags: memory.tags,
      gac: memory.metadata.gac as ContextBlock["gac"],
    })),
    { id: "block_noise", text: "low priority noise", tokenStart: 256, tokenEnd: 288, priority: 0.01, source: "summary" },
  ];
}

function topK(corpus: MemorySearchHit[], queryEmbedding: number[], k: number): MemorySearchHit[] {
  return [...corpus]
    .map((memory) => ({ ...memory, score: cosine(queryEmbedding, memory.embedding) }))
    .sort((a, b) => b.score - a.score)
    .slice(0, k);
}

function makeVector(dimension: number, seed: number): number[] {
  return Array.from({ length: dimension }, (_, index) => {
    const signal = index === seed % dimension ? 1 : 0;
    return signal + Math.sin((seed + 1) * (index + 1)) * 0.01;
  });
}

function cosine(a: number[], b: number[]): number {
  let dot = 0;
  let aMagnitude = 0;
  let bMagnitude = 0;
  for (let i = 0; i < Math.min(a.length, b.length); i += 1) {
    const x = a[i] ?? 0;
    const y = b[i] ?? 0;
    dot += x * y;
    aMagnitude += x * x;
    bMagnitude += y * y;
  }
  return dot / (Math.sqrt(aMagnitude) * Math.sqrt(bMagnitude) || 1);
}

function suite(name: string, metrics: SuiteResult["metrics"], gates: GateResult[]): SuiteResult {
  return {
    name,
    metrics,
    gates,
    passed: gates.every((item) => item.passed),
  };
}

function gate(name: string, actual: GateResult["actual"], expected: GateResult["expected"], passed: boolean): GateResult {
  return { name, actual, expected, passed };
}

async function fetchJson<T>(url: string, timeoutMs = 10_000): Promise<T> {
  const response = await fetchWithTimeout(url, {}, timeoutMs);
  if (!response.ok) throw new Error(`${url} failed with ${response.status}`);
  return response.json() as Promise<T>;
}

async function postJson<T = GacWriteResult>(url: string, body: unknown): Promise<T> {
  const response = await fetchWithTimeout(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!response.ok) throw new Error(`${url} failed with ${response.status}`);
  return response.json() as Promise<T>;
}

async function deleteJson<T = { ok: boolean; count?: number }>(url: string, body: unknown): Promise<T> {
  const response = await fetchWithTimeout(url, {
    method: "DELETE",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!response.ok) throw new Error(`${url} failed with ${response.status}`);
  return response.json() as Promise<T>;
}

async function fetchWithTimeout(url: string, init: RequestInit, timeoutMs = 10_000): Promise<Response> {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), timeoutMs);
  try {
    return await fetch(url, { ...init, signal: controller.signal });
  } catch (error) {
    throw new Error(`${url} request failed: ${error instanceof Error ? error.message : String(error)}`);
  } finally {
    clearTimeout(timeout);
  }
}

function buildSummary(results: {
  name: string;
  createdAt: string;
  passed: boolean;
  sidecar: Record<string, number | string | boolean>;
  suites: SuiteResult[];
}): string {
  const rows = results.suites
    .flatMap((suiteResult) => suiteResult.gates.map((item) => `| ${suiteResult.name} | ${item.name} | ${item.passed ? "PASS" : "FAIL"} | ${item.actual} | ${item.expected} |`))
    .join("\n");

  return `# Production Readiness Eval

- Created: ${results.createdAt}
- Passed: ${results.passed}
- Sidecar: ${results.sidecar.available ? "available" : "not available"} (${results.sidecar.reason})
- Suites: ${results.suites.length}

| Suite | Gate | Status | Actual | Expected |
| --- | --- | --- | --- | --- |
${rows}
`;
}
