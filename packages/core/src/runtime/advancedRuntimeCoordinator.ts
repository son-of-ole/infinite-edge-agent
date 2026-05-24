import type {
  AgentRuntimeConfig,
  ChatMessage,
  ContextPackTraceStore,
  ContextPackTraceRecord,
  GacKvSwapPriorityMetadata,
  GacMemoryStore,
  GacRoutingBlockMetadata,
  IdentityPinRecord,
  MemoryProviderMode,
  MemorySearchHit,
  PackedContext,
  RawMemoryRecord,
  RetrievalAuditRecord,
  RuntimeTrace
} from "../types";
import { getMemoryProviderCapabilities } from "../memoryProviders";
import { buildContextRuntimePlan, type ContextRebuildPlan } from "./contextRuntime";
import { createDefaultRuntimeFeatureRegistry, type RuntimeFeatureStatus } from "./features";
import {
  createLowRankKeySummary,
  planKVSwap,
  type KVBlock,
  type KVLowRankQuerySummary,
  type KVSwapDecision,
} from "./kvswap";
import { buildPredictiveRuntimePlan, type PredictiveRuntimePlan } from "./predictiveRuntime";
import {
  resolveSpeculativeDecodingConfig,
  SpeculativeModelRegistry,
  type SpeculativeDecodingConfig,
  type SpeculativeModelProfile,
} from "./speculative";
import { FallbackSSARuntime, type ContextBlock, type SSAPlan } from "./ssa";
import { buildFallbackTSPPlan, type DeviceProfile, type ModelProfile, type TSPPlan } from "./tsp";

export interface InferenceBackendProfile {
  id: string;
  label: string;
  mode: "custom" | "opaque";
  capabilities?: InferenceBackendCapabilities;
}

export interface InferenceBackendCapabilities {
  qkvAccess?: boolean;
  layerSparseRouting?: boolean;
  pinnedKvBlocks?: boolean;
  kvTensorPaging?: boolean;
  tspScheduleExecution?: boolean;
  speculativeVerifierBatching?: boolean;
}

export interface AdvancedRuntimeModelProfile extends ModelProfile {
  contextWindowTokens: number;
  effectiveParameterCount?: number;
}

export interface AdvancedRuntimeCoordinatorInput {
  requestId: string;
  tenantId?: string;
  cellId?: string;
  sessionId: string;
  systemPrompt: string;
  userMessage: string;
  recentMessages: ChatMessage[];
  retrievedMemory: MemorySearchHit[];
  config: AgentRuntimeConfig;
  backend: InferenceBackendProfile;
  model: AdvancedRuntimeModelProfile;
  device: DeviceProfile;
  memoryMode: MemoryProviderMode;
  memoryStore?: Partial<GacMemoryStore> | ContextPackTraceStore;
  now?: Date;
}

export interface AdvancedRuntimeGenerationPlan {
  packed: PackedContext;
  trace: RuntimeTrace;
  contextPlan: ContextRebuildPlan;
  tspPlan: TSPPlan;
  ssaPlan: SSAPlan;
  kvswapDecision: KVSwapDecision;
  speculativeConfig: SpeculativeDecodingConfig;
  predictivePlan: PredictiveRuntimePlan;
  features: RuntimeFeatureStatus[];
}

export interface RuntimeFeatureCapabilitySnapshotInput {
  backend: InferenceBackendProfile;
  model: AdvancedRuntimeModelProfile;
  memoryMode: MemoryProviderMode;
  mtp?: NonNullable<AgentRuntimeConfig["mtp"]>;
}

export function buildRuntimeFeatureCapabilitySnapshot(
  input: RuntimeFeatureCapabilitySnapshotInput,
): RuntimeFeatureStatus[] {
  const speculativeConfig = resolveRuntimeSpeculativeConfig({
    config: {
      ...(input.mtp ? { mtp: input.mtp } : {}),
    },
    backend: input.backend,
    model: input.model,
  });
  return buildFeatureStatuses(
    input,
    makeStartupSsaPlan(),
    makeStartupTspPlan(input.model),
    speculativeConfig,
  );
}

export async function buildAdvancedRuntimeGenerationPlan(
  input: AdvancedRuntimeCoordinatorInput,
): Promise<AdvancedRuntimeGenerationPlan> {
  const now = input.now ?? new Date();
  const traceStore = getRequiredContextPackTraceStore(input.memoryStore);
  const contextRebuildSignals = await readContextRebuildSignals(input, traceStore);
  const recoveredMemory = augmentRetrievedMemoryWithRecoveredRaw(
    input.retrievedMemory,
    contextRebuildSignals,
    input.sessionId,
  );
  const memoryGate = filterMemoryForModelVisibleContext(recoveredMemory);
  const context = buildContextRuntimePlan({
    requestId: input.requestId,
    systemPrompt: input.systemPrompt,
    userMessage: input.userMessage,
    recentMessages: input.recentMessages,
    retrievedMemory: memoryGate.visibleMemory,
    identityPins: contextRebuildSignals.identityPins,
    contextPackTraces: contextRebuildSignals.contextPackTraces,
    retrievalAudits: contextRebuildSignals.retrievalAudits,
    maxRetrievedMemoryTokens: input.config.maxRetrievedMemoryTokens,
    maxRecentConversationTokens: input.config.maxRecentConversationTokens,
    maxPromptTokens: Math.min(input.config.maxPromptTokens, input.model.contextWindowTokens),
  });

  const tspPlan = buildFallbackTSPPlan({
    device: input.device,
    model: input.model,
    requestedContextTokens: context.packed.estimatedTokens,
    batchSize: 1,
    kvPrecisionBytes: 2,
    activationPrecisionBytes: 2,
    safetyMarginRatio: 0.2,
  });

  const activeBlocks = toContextBlocks(context.plan, input.userMessage, memoryGate.visibleMemory, context.packed.includedMemoryIds);
  const ssaPlan = await new FallbackSSARuntime().plan({
    requestId: input.requestId,
    activeBlocks,
    anchors: context.plan.pinnedAnchorIds.map((blockId) => ({
      blockId,
      reason: "current_user_request",
      score: 1,
    })),
    memoryHits: memoryGate.visibleMemory,
    maxBlocks: 64,
    minAnchorScore: 0.5,
  });

  const kvBlocks = toKvBlocks(activeBlocks, ssaPlan.pinnedBlockIds, now.getTime());
  const speculativeConfig = resolveRuntimeSpeculativeConfig(input);
  const predictivePlan = buildPredictiveRuntimePlan({
    requestId: input.requestId,
    userMessage: input.userMessage,
    activeBlocks,
    ssaPlan,
    kvBlocks,
    speculativeConfig,
    tokenBudget: input.config.maxPromptTokens,
    ...(input.device.vramBudgetBytes ? { vramBudgetBytes: input.device.vramBudgetBytes } : {}),
    ...(input.device.ramBudgetBytes ? { ramBudgetBytes: input.device.ramBudgetBytes } : {}),
  });
  const kvswapDecision = planKVSwap(kvBlocks, {
    mode: "metadata_only",
    vramPressureThreshold: 0.82,
    ramPressureThreshold: 0.85,
    now: now.getTime(),
  }, 0, predictivePlan.cacheBudget.prefetchBlockIds, predictivePlan.kvHotPages.map((page) => ({
    blockId: page.blockId,
    confidence: page.priority,
    reasons: [
      page.source,
      `tier:${page.tier}`,
      ...(page.source === "gac" ? ["gac"] : []),
    ],
  })), {
    querySummary: buildTextLowRankQuerySummary(input.userMessage),
    maxBlocks: Math.min(4, Math.max(1, kvBlocks.length)),
    minScore: 0.05,
  });

  const features = buildFeatureStatuses(input, ssaPlan, tspPlan, speculativeConfig);
  const traceId = makeTraceId(input.requestId);
  const ssaRoutingBlocks = activeBlocks.map(toSsaRoutingBlockMetadata).filter((metadata): metadata is GacRoutingBlockMetadata => Boolean(metadata));
  const kvSwapPriorities = kvBlocks.map((block) => block.gacPriority).filter((metadata): metadata is GacKvSwapPriorityMetadata => Boolean(metadata));
  const contextTraceResult = await writeContextPackTrace(
    input,
    traceStore,
    context.packed,
    context.plan,
    traceId,
    now,
    memoryGate,
    ssaRoutingBlocks,
    kvSwapPriorities,
    predictivePlan,
  );
  const trace: RuntimeTrace = {
    traceId,
    requestId: input.requestId,
    sessionId: input.sessionId,
    ...(input.tenantId ? { tenantId: input.tenantId } : {}),
    ...(input.cellId ? { cellId: input.cellId } : {}),
    modelId: input.model.modelId,
    backend: input.backend.id,
    createdAt: now.toISOString(),
    runtime: {
      backend: input.backend,
      features,
      context: {
        estimatedTokens: context.packed.estimatedTokens,
        includedMemoryIds: context.packed.includedMemoryIds,
        droppedFrameIds: [...context.plan.droppedFrameIds, ...Object.keys(memoryGate.dropReasons)],
        pinnedAnchorIds: context.plan.pinnedAnchorIds,
        memoryPriorityMap: context.plan.memoryPriorityMap,
        sourceLineageMap: context.plan.sourceLineageMap,
        contextRebuildLearning: context.plan.learningSignals,
        rawMemoryRecovery: contextRebuildSignals.rawMemoryRecovery,
        ...(Object.keys(memoryGate.dropReasons).length > 0 ? { gacDropReasons: memoryGate.dropReasons } : {}),
        ...contextTraceResult,
      },
      tsp: tspPlan,
      ssa: ssaPlan,
      kvswap: kvswapDecision,
      mtp: speculativeConfig,
      predictive: predictivePlan,
    },
  };

  return {
    packed: context.packed,
    trace,
    contextPlan: context.plan,
    tspPlan,
    ssaPlan,
    kvswapDecision,
    speculativeConfig,
    predictivePlan,
    features,
  };
}

interface ContextRebuildSignalSnapshot {
  identityPins: IdentityPinRecord[];
  contextPackTraces: ContextPackTraceRecord[];
  retrievalAudits: RetrievalAuditRecord[];
  rawMemory: RawMemoryRecord[];
  rawMemoryRecovery: RawMemoryRecoverySnapshot;
}

interface RawMemoryRecoverySnapshot {
  requestedRawMemoryIds: string[];
  recoveredRawMemoryIds: string[];
  missingRawMemoryIds: string[];
}

async function readContextRebuildSignals(
  input: AdvancedRuntimeCoordinatorInput,
  traceStore: ContextPackTraceStore,
): Promise<ContextRebuildSignalSnapshot> {
  const listOptions = {
    ...(input.tenantId ? { tenantId: input.tenantId } : {}),
    ...(input.cellId ? { cellId: input.cellId } : {}),
    sessionId: input.sessionId,
    limit: 32,
  };
  try {
    const store = input.memoryStore as Partial<GacMemoryStore> | undefined;
    const [contextPackTraces, identityPins, retrievalAudits] = await Promise.all([
      traceStore.listContextPackTraces(listOptions),
      typeof store?.listIdentityPins === "function" ? store.listIdentityPins(listOptions) : Promise.resolve([]),
      typeof store?.listRetrievalAudits === "function" ? store.listRetrievalAudits(listOptions) : Promise.resolve([]),
    ]);
    const requestedRawMemoryIds = uniqueStable([
      ...identityPins.map((pin) => pin.rawMemoryId),
      ...retrievalAudits
        .filter((audit) => !audit.identityPreserved)
        .map((audit) => audit.expectedRawMemoryId),
    ]);
    const rawMemory = requestedRawMemoryIds.length > 0 && typeof store?.listRawMemory === "function"
      ? dedupeRawMemory((await Promise.all(requestedRawMemoryIds.map((rawMemoryId) =>
          store.listRawMemory?.({
            ...listOptions,
            rawMemoryId,
            limit: 1,
          }) ?? Promise.resolve([]),
        ))).flat())
      : [];
    const recoveredRawMemoryIds = uniqueStable(rawMemory.map((record) => record.id));
    return {
      contextPackTraces,
      identityPins,
      retrievalAudits,
      rawMemory,
      rawMemoryRecovery: {
        requestedRawMemoryIds,
        recoveredRawMemoryIds,
        missingRawMemoryIds: requestedRawMemoryIds.filter((id) => !recoveredRawMemoryIds.includes(id)),
      },
    };
  } catch (error) {
    throw new Error(`GAC context rebuild signal read failed: ${error instanceof Error ? error.message : String(error)}`);
  }
}

function dedupeRawMemory(records: RawMemoryRecord[]): RawMemoryRecord[] {
  const byId = new Map<string, RawMemoryRecord>();
  for (const record of records) {
    if (!byId.has(record.id)) byId.set(record.id, record);
  }
  return [...byId.values()];
}

function augmentRetrievedMemoryWithRecoveredRaw(
  retrievedMemory: MemorySearchHit[],
  signals: ContextRebuildSignalSnapshot,
  sessionId: string,
): MemorySearchHit[] {
  if (signals.rawMemory.length === 0) return retrievedMemory;
  const representedRawIds = new Set<string>();
  const existingMemoryIds = new Set(retrievedMemory.map((hit) => hit.id));
  for (const hit of retrievedMemory) {
    representedRawIds.add(hit.id);
    for (const rawId of getRawMemoryIds(getGacMetadata(hit))) representedRawIds.add(rawId);
  }
  const recovered = signals.rawMemory
    .filter((record) => !existingMemoryIds.has(record.id) && !representedRawIds.has(record.id))
    .map((record) => toRecoveredRawMemoryHit(record, signals, sessionId));
  return [...retrievedMemory, ...recovered];
}

function toRecoveredRawMemoryHit(
  record: RawMemoryRecord,
  signals: ContextRebuildSignalSnapshot,
  sessionId: string,
): MemorySearchHit {
  const pin = signals.identityPins.find((item) => item.rawMemoryId === record.id);
  const failedAudit = signals.retrievalAudits.find((audit) => !audit.identityPreserved && audit.expectedRawMemoryId === record.id);
  const protectedPin = Boolean(pin) || record.retentionClass === "pinned";
  const memoryClass = protectedPin ? "PINNED_EXACT" : "HIGH_RISK_RAW";
  const score = protectedPin ? 1 : failedAudit ? 0.88 : Math.max(0.65, record.importance);
  return {
    id: record.id,
    text: record.canonicalText ?? record.text,
    embedding: [],
    sessionId: record.sessionId ?? sessionId,
    source: rawSourceToMemorySource(record.sourceType),
    createdAt: record.createdAt,
    updatedAt: record.updatedAt,
    tags: [
      "gac",
      "raw_recovered",
      ...(protectedPin ? ["identity_pin_recovered"] : []),
      ...(failedAudit ? ["retrieval_audit_repair"] : []),
    ],
    metadata: {
      rawMemoryId: record.id,
      rawMemoryIds: [record.id],
      recoveredBy: [
        ...(protectedPin ? ["identity_pin"] : []),
        ...(failedAudit ? ["retrieval_audit_failure"] : []),
      ],
      gac: {
        rawMemoryId: record.id,
        rawMemoryIds: [record.id],
        memoryClass,
        identityRisk: record.identityRiskSeed,
        sourceTrust: 1,
        ...(pin ? {
          identityPinId: pin.id,
          pinStrength: pin.pinStrength,
          mustAttend: true,
        } : {}),
      },
    },
    tokenCount: Math.max(1, Math.ceil((record.canonicalText ?? record.text).length / 4)),
    score,
  };
}

function rawSourceToMemorySource(sourceType: RawMemoryRecord["sourceType"]): MemorySearchHit["source"] {
  if (sourceType === "file" || sourceType === "code" || sourceType === "external") return "document";
  if (sourceType === "reflection") return "summary";
  if (sourceType === "tool") return "tool";
  if (sourceType === "system") return "system";
  return "chat";
}

interface RuntimeSpeculativeConfigInput {
  config: { mtp?: NonNullable<AgentRuntimeConfig["mtp"]> };
  backend: InferenceBackendProfile;
  model: Pick<AdvancedRuntimeModelProfile, "modelId">;
}

function resolveRuntimeSpeculativeConfig(input: RuntimeSpeculativeConfigInput): SpeculativeDecodingConfig {
  const mtp = input.config.mtp;
  const backendCanVerify = input.backend.capabilities?.speculativeVerifierBatching === true;
  const registry = new SpeculativeModelRegistry([
    {
      modelId: input.model.modelId,
      role: "target",
      ...(mtp?.targetTokenizerId ? { tokenizerId: mtp.targetTokenizerId } : {}),
    },
    ...((mtp?.draftModelProfiles ?? []) as SpeculativeModelProfile[]),
  ]);

  return resolveSpeculativeDecodingConfig({
    enabled: mtp?.enabled === true && backendCanVerify,
    targetModelId: input.model.modelId,
    draftModelId: mtp?.draftModelId ?? null,
    mode: mtp?.mode ?? "draft_verify",
    ...(mtp?.numSpeculativeTokens !== undefined ? { numSpeculativeTokens: mtp.numSpeculativeTokens } : {}),
    ...(mtp?.minAcceptanceRate !== undefined ? { minAcceptanceRate: mtp.minAcceptanceRate } : {}),
    ...(mtp?.disableWhenLatencyWorse !== undefined ? { disableWhenLatencyWorse: mtp.disableWhenLatencyWorse } : {}),
  }, registry);
}

async function writeContextPackTrace(
  input: AdvancedRuntimeCoordinatorInput,
  store: ContextPackTraceStore,
  packed: PackedContext,
  plan: ContextRebuildPlan,
  traceId: string,
  now: Date,
  memoryGate: MemoryVisibilityGateResult,
  ssaRoutingBlocks: GacRoutingBlockMetadata[],
  kvSwapPriorities: GacKvSwapPriorityMetadata[],
  predictivePlan: PredictiveRuntimePlan,
): Promise<Record<string, unknown>> {
  const included = new Set(packed.includedMemoryIds);
  const includedMemory = memoryGate.visibleMemory.filter((memory) => included.has(memory.id));
  const record: ContextPackTraceRecord = {
    id: `ctx_${traceId}`,
    traceId,
    tenantId: input.tenantId ?? "local",
    cellId: input.cellId ?? "default",
    sessionId: input.sessionId,
    queryId: input.requestId,
    contextPackId: `pack_${input.requestId}`,
    rawMemoryIds: includedMemory.flatMap((memory) => {
      const ids = getRawMemoryIds(getGacMetadata(memory));
      return ids.length > 0 ? ids : [memory.id];
    }),
    representativeIds: uniqueStable(includedMemory.flatMap((memory) => {
      const metadata = getGacMetadata(memory);
      return [
        ...getStringArray(metadata, "representativeIds"),
        ...(getString(metadata, "representativeId") ? [getString(metadata, "representativeId")!] : []),
      ];
    })),
    identityPinIds: uniqueStable(includedMemory.flatMap((memory) => {
      const metadata = getGacMetadata(memory);
      return [
        ...getStringArray(metadata, "identityPinIds"),
        ...(getString(metadata, "identityPinId") ? [getString(metadata, "identityPinId")!] : []),
      ];
    })),
    tokenBudget: input.config.maxPromptTokens,
    estimatedTokens: packed.estimatedTokens,
    packingStrategy: "advanced-runtime",
    includedMemoryIds: packed.includedMemoryIds,
    omittedMemoryIds: [...plan.droppedFrameIds, ...Object.keys(memoryGate.dropReasons)],
    ...(ssaRoutingBlocks.length > 0 ? { ssaRoutingBlocks } : {}),
    ...(kvSwapPriorities.length > 0 ? { kvSwapPriorities } : {}),
    predictivePlanId: predictivePlan.planId,
    predictedRetrievals: predictivePlan.predictedRetrievals,
    contextBranches: predictivePlan.contextBranches,
    kvHotPages: predictivePlan.kvHotPages,
    mtpBranches: predictivePlan.mtpBranches,
    createdAt: now.toISOString(),
  };

  try {
    await store.writeContextPackTraces([record]);
    return {
      contextPackTraceId: record.id,
      contextPackTraceWrite: "ok",
    };
  } catch (error) {
    throw new Error(`GAC context pack trace persistence failed: ${error instanceof Error ? error.message : String(error)}`);
  }
}

interface MemoryVisibilityGateResult {
  visibleMemory: MemorySearchHit[];
  dropReasons: Record<string, string[]>;
}

function getRequiredContextPackTraceStore(store: AdvancedRuntimeCoordinatorInput["memoryStore"]): ContextPackTraceStore {
  if (!store || typeof store.writeContextPackTraces !== "function" || typeof store.listContextPackTraces !== "function") {
    throw new Error("GAC context pack trace persistence is required for every model call.");
  }
  return store as ContextPackTraceStore;
}

function filterMemoryForModelVisibleContext(memory: MemorySearchHit[]): MemoryVisibilityGateResult {
  const visibleMemory: MemorySearchHit[] = [];
  const dropReasons: Record<string, string[]> = {};

  for (const hit of memory) {
    if (requiresRepresentativeLineage(hit) && !hasRepresentativeLineage(hit)) {
      dropReasons[hit.id] = ["representative_missing_lineage"];
      continue;
    }
    visibleMemory.push(hit);
  }

  return { visibleMemory, dropReasons };
}

function requiresRepresentativeLineage(hit: MemorySearchHit): boolean {
  const metadata = getGacMetadata(hit);
  const memoryClass = getString(metadata, "memoryClass");
  return Boolean(getString(metadata, "representativeId"))
    || metadata.modelVisible === true
    || metadata.factual === true
    || memoryClass === "LOW_RISK_REPRESENTATIVE"
    || memoryClass === "BACKGROUND_SUMMARY";
}

function hasRepresentativeLineage(hit: MemorySearchHit): boolean {
  const metadata = getGacMetadata(hit);
  return Boolean(getString(metadata, "rawMemoryId"))
    || getStringArray(metadata, "rawMemoryIds").length > 0
    || getStringArray(metadata, "lineageRawMemoryIds").length > 0;
}

interface RuntimeFeatureStatusBuildInput {
  backend: InferenceBackendProfile;
  model: AdvancedRuntimeModelProfile;
  memoryMode: MemoryProviderMode;
}

function buildFeatureStatuses(
  input: RuntimeFeatureStatusBuildInput,
  ssaPlan: SSAPlan,
  tspPlan: TSPPlan,
  speculativeConfig: SpeculativeDecodingConfig,
): RuntimeFeatureStatus[] {
  const registry = createDefaultRuntimeFeatureRegistry();
  registry.set({
    name: "memoryProvider",
    state: input.memoryMode === "remote-http" || input.memoryMode === "lancedb-sidecar" || input.memoryMode === "browser-vector" ? "enabled" : "fallback",
    mode: input.memoryMode,
    ...memoryProviderStatus(input.memoryMode),
    metrics: memoryProviderCapabilityMetrics(input.memoryMode),
  });
  registry.set({
    name: "inferenceBackend",
    state: "enabled",
    mode: input.backend.mode,
    ...backendStatus(input.backend),
    metrics: {
      modelId: input.model.modelId,
      contextWindowTokens: input.model.contextWindowTokens,
      effectiveParameterCount: input.model.effectiveParameterCount ?? input.model.parameterBytes / 2,
    },
  });
  const fullControl = input.backend.capabilities ?? {};
  registry.set({
    name: "ssa",
    state: fullControl.qkvAccess && fullControl.layerSparseRouting ? "enabled" : "fallback",
    mode: fullControl.qkvAccess && fullControl.layerSparseRouting ? "backend_native_sparse_execution" : ssaPlan.mode,
    reason: fullControl.qkvAccess && fullControl.layerSparseRouting
      ? "Backend exposes Q/K/V tensors for sparse layer execution."
      : "Current backend has not exposed native sparse Q/K/V layer execution.",
    metrics: {
      selectedBlocks: ssaPlan.selectedBlockIds.length,
      droppedBlocks: ssaPlan.droppedBlockIds.length,
      sparsityRatio: Number(ssaPlan.sparsityRatio.toFixed(4)),
    },
  });
  registry.set({
    name: "tsp",
    state: fullControl.tspScheduleExecution ? "enabled" : "fallback",
    mode: fullControl.tspScheduleExecution ? "backend_schedule_execution" : tspPlan.mode,
    reason: fullControl.tspScheduleExecution
      ? "Backend executes planner-emitted sequence/tensor schedule callbacks."
      : tspPlan.degradationReason ?? "Current backend has not exposed callback-backed tensor/sequence schedule execution.",
    metrics: {
      maxSafeContextTokens: tspPlan.maxSafeContextTokens,
      sequenceShards: tspPlan.sequenceShards,
      tensorShards: tspPlan.tensorShards,
    },
  });
  const mtpEnabled = Boolean(
    fullControl.speculativeVerifierBatching
    && speculativeConfig.mode !== "target_only"
    && speculativeConfig.numSpeculativeTokens > 0,
  );
  const mtpFallbackReason = speculativeConfig.draftModelId && !fullControl.speculativeVerifierBatching
    ? "Backend does not expose batched target verification for draft branches."
    : "No compatible draft model is configured yet.";
  registry.set({
    name: "mtp",
    state: mtpEnabled ? "enabled" : "fallback",
    mode: mtpEnabled ? "verifier_batching" : "target_only",
    reason: mtpEnabled
      ? "Backend supports batched target verification for draft branches."
      : mtpFallbackReason,
  });
  registry.set({
    name: "kvswap",
    state: fullControl.kvTensorPaging && fullControl.pinnedKvBlocks ? "enabled" : "fallback",
    mode: fullControl.kvTensorPaging && fullControl.pinnedKvBlocks ? "tensor_paging" : "metadata_only",
    reason: fullControl.kvTensorPaging && fullControl.pinnedKvBlocks
      ? "Backend owns KV tensor handles and can page selected blocks before sparse attention."
      : "Current inference backend does not expose real KV tensor handles.",
  });
  registry.assertTierZeroReady();
  return registry.list();
}

function makeStartupSsaPlan(): SSAPlan {
  return {
    mode: "fallback_sparse_planner",
    targetProfile: "subq_compatible_public_ssa_path",
    selectedBlockIds: [],
    pinnedBlockIds: [],
    droppedBlockIds: [],
    routingReasons: {},
    layerPolicies: [],
    estimatedDenseTokens: 0,
    estimatedSparseTokens: 0,
    sparsityRatio: 0,
    routingTrace: [],
    kernelTraces: [],
  };
}

function makeStartupTspPlan(model: AdvancedRuntimeModelProfile): TSPPlan {
  return {
    mode: "fallback_budget_planner",
    sequenceShards: 1,
    tensorShards: 1,
    activationWindowTokens: 0,
    maxSafeContextTokens: model.contextWindowTokens,
    estimatedVramBytes: 0,
    estimatedRamBytes: 0,
    schedule: [],
  };
}

function backendStatus(backend: InferenceBackendProfile): Pick<RuntimeFeatureStatus, "reason" | "impact"> {
  if (backend.capabilities?.qkvAccess && backend.capabilities.layerSparseRouting) {
    return {
      reason: "Full-control browser transformer backend owns model tensors and runtime scheduling.",
      impact: "SSA/KV/TSP can execute against backend-owned tensors instead of an opaque text-generation API.",
    };
  }
  if (backend.mode === "opaque") {
    return {
      reason: "An opaque browser inference API is active.",
      impact: "Opaque text-generation APIs do not expose Q/K/V tensors or KV-cache ownership to the unlocked runtime.",
    };
  }
  return {};
}

function memoryProviderStatus(memoryMode: MemoryProviderMode): Pick<RuntimeFeatureStatus, "reason" | "impact"> {
  if (memoryMode === "remote-http") {
    return {
      reason: "Remote HTTP memory provider is active.",
      impact: "Memory and runtime traces are written to the configured remote database API.",
    };
  }
  if (memoryMode === "lancedb-sidecar") {
    return {
      reason: "Local LanceDB sidecar is active.",
      impact: "Memory remains local to the desktop/edge sidecar.",
    };
  }
  if (memoryMode === "browser-vector") {
    return {
      reason: "Browser-native vector memory provider is active.",
      impact: "Memory, deterministic vector search, import/export, and context-pack traces persist locally in browser IndexedDB without a sidecar or remote API.",
    };
  }
  if (memoryMode === "indexeddb") {
    return {
      reason: "IndexedDB compatibility memory alias is active.",
      impact: "Memory remains local to browser IndexedDB; use browser-vector for the explicit production browser-only provider name.",
    };
  }
  return {
    reason: "No memory provider is active.",
    impact: "The agent cannot persist long-term memory.",
  };
}

function memoryProviderCapabilityMetrics(memoryMode: MemoryProviderMode): Record<string, string | number | boolean> {
  const capabilities = getMemoryProviderCapabilities(memoryMode);
  return {
    storage: capabilities.storage,
    localOnly: capabilities.localOnly,
    vectorSearch: capabilities.vectorSearch,
    deterministicSearch: capabilities.deterministicSearch,
    metadataFilters: capabilities.metadataFilters,
    persistent: capabilities.persistent,
    importExport: capabilities.importExport,
    contextPackTracePersistence: capabilities.contextPackTracePersistence,
    remoteSync: capabilities.remoteSync,
    ...(capabilities.vectorDimension ? { vectorDimension: capabilities.vectorDimension } : {}),
  };
}

function toContextBlocks(
  plan: ContextRebuildPlan,
  userMessage: string,
  memoryHits: MemorySearchHit[],
  includedMemoryIds: string[],
): ContextBlock[] {
  const memoryById = new Map(memoryHits.map((hit) => [hit.id, hit]));
  const included = new Set(includedMemoryIds);
  let cursor = 0;
  return plan.frames.filter((frame) => frame.kind !== "retrieved" || included.has(frame.id)).map((frame) => {
    const tokenCount = Math.max(1, frame.tokenCount);
    const gac = frame.kind === "retrieved" ? toSsaRoutingBlockMetadataFromHit(frame.id, memoryById.get(frame.id)) : undefined;
    const block: ContextBlock = {
      id: frame.id,
      text: frame.text || userMessage,
      tokenStart: cursor,
      tokenEnd: cursor + tokenCount,
      priority: frame.priority,
      source: frame.provenance[0]?.sourceType ?? frame.kind,
      tags: frame.kind === "anchor" ? ["current_user_request"] : [frame.kind],
      ...(gac ? { gac } : {}),
    };
    cursor += tokenCount;
    return block;
  });
}

function toKvBlocks(blocks: ContextBlock[], pinnedBlockIds: string[], now: number): KVBlock[] {
  const pinned = new Set(pinnedBlockIds);
  return blocks.map((block, index) => {
    const blockId = `kv_${block.id}`;
    const isPinned = pinned.has(block.id) || block.gac?.mustAttend === true || block.gac?.memoryClass === "PINNED_EXACT";
    const importance = Math.max(0, Math.min(1, block.priority + (block.gac?.pinStrength ?? 0) + (block.gac?.identityRisk ?? 0) * 0.25));
    const lowRankKeySummary = createLowRankKeySummary({
      blockId,
      projectionId: TEXT_LOW_RANK_PROJECTION_ID,
      layer: 0,
      headGroupId: "all_heads",
      values: buildTextLowRankValues(block.text, TEXT_LOW_RANK_SUMMARY_RANK),
      qualityScore: Math.max(0.1, importance),
    });
    return {
      id: blockId,
      layer: 0,
      startToken: block.tokenStart,
      endToken: block.tokenEnd,
      tier: isPinned ? "vram" : "ram",
      pinned: isPinned,
      importance,
      lastAccessAt: now - index,
      sourceBlockId: block.id,
      estimatedBytes: Math.max(1, block.tokenEnd - block.tokenStart) * 4096,
      summaryRank: lowRankKeySummary.rank,
      compressedKeySummary: new Float32Array(lowRankKeySummary.values),
      lowRankKeySummary,
      ...(toKvSwapPriorityMetadata(block) ? { gacPriority: toKvSwapPriorityMetadata(block) } : {}),
    };
  });
}

const TEXT_LOW_RANK_SUMMARY_RANK = 4;
const TEXT_LOW_RANK_PROJECTION_ID = "advanced-runtime:text-low-rank:v1";

function buildTextLowRankQuerySummary(text: string): KVLowRankQuerySummary {
  return {
    rank: TEXT_LOW_RANK_SUMMARY_RANK,
    projectionId: TEXT_LOW_RANK_PROJECTION_ID,
    layer: 0,
    headGroupId: "all_heads",
    values: buildTextLowRankValues(text, TEXT_LOW_RANK_SUMMARY_RANK),
  };
}

function buildTextLowRankValues(text: string, rank: number): number[] {
  const values = new Array(rank).fill(0);
  const tokens = text.toLowerCase().match(/[a-z0-9_:-]+/g) ?? [];
  if (tokens.length === 0) return values;
  for (const token of tokens) {
    const hash = stableTextHash(token);
    const index = hash % rank;
    const sign = (hash & 1) === 0 ? 1 : -1;
    values[index] += sign * Math.min(1, Math.max(0.1, token.length / 12));
  }
  const norm = Math.sqrt(values.reduce((sum, value) => sum + value * value, 0)) || 1;
  return values.map((value) => value / norm);
}

function stableTextHash(text: string): number {
  let hash = 2166136261;
  for (const char of text) {
    hash ^= char.charCodeAt(0);
    hash = Math.imul(hash, 16777619) >>> 0;
  }
  return hash >>> 0;
}

function toSsaRoutingBlockMetadata(block: ContextBlock): GacRoutingBlockMetadata | null {
  return block.gac ?? null;
}

function toSsaRoutingBlockMetadataFromHit(blockId: string, hit: MemorySearchHit | undefined): GacRoutingBlockMetadata | undefined {
  if (!hit) return undefined;
  const metadata = getGacMetadata(hit);
  const memoryClass = getMemoryClass(metadata);
  if (!memoryClass) return undefined;
  const rawMemoryIds = getRawMemoryIds(metadata);
  return {
    blockId,
    memoryClass,
    ...(rawMemoryIds[0] ? { rawMemoryId: rawMemoryIds[0] } : {}),
    ...(getString(metadata, "representativeId") ? { representativeId: getString(metadata, "representativeId") } : {}),
    ...(getNumber(metadata, "identityRisk") !== undefined ? { identityRisk: getNumber(metadata, "identityRisk") } : {}),
    ...(getNumber(metadata, "pinStrength") !== undefined ? { pinStrength: getNumber(metadata, "pinStrength") } : {}),
    ...(getNumber(metadata, "sourceTrust") !== undefined ? { sourceTrust: getNumber(metadata, "sourceTrust") } : {}),
    ...(metadata.mustAttend === true ? { mustAttend: true } : {}),
  };
}

function toKvSwapPriorityMetadata(block: ContextBlock): GacKvSwapPriorityMetadata | undefined {
  if (!block.gac) return undefined;
  const reasonCodes: string[] = [];
  if (block.gac.memoryClass === "PINNED_EXACT") reasonCodes.push("identity_pin");
  if (block.gac.memoryClass === "HIGH_RISK_RAW") reasonCodes.push("high_risk_raw");
  if (block.gac.memoryClass === "LOW_RISK_REPRESENTATIVE") reasonCodes.push("low_risk_representative");
  if (block.gac.mustAttend) reasonCodes.push("must_attend");
  if (typeof block.gac.pinStrength === "number" && block.gac.pinStrength > 0) reasonCodes.push("pin_strength");
  const priorityScore = Math.max(0, Math.min(1, Math.max(block.priority, block.gac.pinStrength ?? 0, block.gac.identityRisk ?? 0)));
  return {
    blockId: `kv_${block.id}`,
    tier: toKvSwapTier(block.gac),
    priorityScore,
    reasonCodes,
  };
}

function toKvSwapTier(gac: GacRoutingBlockMetadata): GacKvSwapPriorityMetadata["tier"] {
  if (gac.memoryClass === "PINNED_EXACT" || gac.mustAttend) return "PIN_HOT";
  if (gac.memoryClass === "HIGH_RISK_RAW") return "TASK_HOT";
  if (gac.memoryClass === "RECENT_SESSION" || gac.memoryClass === "TASK_STATE") return "SESSION_WARM";
  if (gac.memoryClass === "LOW_RISK_REPRESENTATIVE" || gac.memoryClass === "BACKGROUND_SUMMARY") return "BACKGROUND_WARM";
  return "COLD";
}

function getGacMetadata(hit: MemorySearchHit): Record<string, unknown> {
  const nested = hit.metadata.gac;
  if (isRecord(nested)) return { ...hit.metadata, ...nested };
  return hit.metadata;
}

function getMemoryClass(metadata: Record<string, unknown>): GacRoutingBlockMetadata["memoryClass"] | undefined {
  const value = getString(metadata, "memoryClass");
  if (
    value === "PINNED_EXACT"
    || value === "HIGH_RISK_RAW"
    || value === "LOW_RISK_REPRESENTATIVE"
    || value === "BACKGROUND_SUMMARY"
    || value === "SOURCE_EVIDENCE"
    || value === "RECENT_SESSION"
    || value === "TASK_STATE"
  ) {
    return value;
  }
  if (getString(metadata, "identityPinId")) return "PINNED_EXACT";
  if (getString(metadata, "representativeId")) return "LOW_RISK_REPRESENTATIVE";
  if (getString(metadata, "rawMemoryId") || getStringArray(metadata, "rawMemoryIds").length > 0) return "HIGH_RISK_RAW";
  return undefined;
}

function getRawMemoryIds(metadata: Record<string, unknown>): string[] {
  const single = getString(metadata, "rawMemoryId");
  return single ? [single] : getStringArray(metadata, "rawMemoryIds");
}

function getString(metadata: Record<string, unknown>, key: string): string | undefined {
  const value = metadata[key];
  return typeof value === "string" && value.length > 0 ? value : undefined;
}

function getNumber(metadata: Record<string, unknown>, key: string): number | undefined {
  const value = metadata[key];
  return typeof value === "number" && Number.isFinite(value) ? value : undefined;
}

function getStringArray(metadata: Record<string, unknown>, key: string): string[] {
  const value = metadata[key];
  return Array.isArray(value) ? value.filter((item): item is string => typeof item === "string" && item.length > 0) : [];
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function uniqueStable(values: string[]): string[] {
  return [...new Set(values)];
}

function makeTraceId(requestId: string): string {
  if (typeof crypto !== "undefined" && "randomUUID" in crypto) {
    return `trace_${crypto.randomUUID()}`;
  }
  return `trace_${requestId}_${Date.now()}`;
}
