export type TSPMode = "fallback_budget_planner" | "webgpu_custom" | "native_edge" | "server_edge";

export interface DeviceProfile {
  name: string;
  vramBudgetBytes: number;
  ramBudgetBytes: number;
  maxBufferSizeBytes?: number;
  backend?: "webgpu" | "wasm" | "native" | "server" | "unknown";
}

export interface ModelProfile {
  modelId: string;
  parameterBytes: number;
  layers: number;
  hiddenSize: number;
  kvHeads: number;
  contextWindowTokens?: number;
  family?: string;
}

export interface TSPPlanInput {
  device: DeviceProfile;
  model: ModelProfile;
  requestedContextTokens: number;
  batchSize: number;
  kvPrecisionBytes: number;
  activationPrecisionBytes: number;
  safetyMarginRatio: number;
}

export interface TSPScheduleStep {
  id: string;
  kind: "attention" | "mlp" | "kv_prefetch" | "activation_checkpoint";
  sequenceShard: number;
  tensorShard: number;
  tokenStart: number;
  tokenEnd: number;
}

export interface TSPPlan {
  mode: TSPMode;
  sequenceShards: number;
  tensorShards: number;
  activationWindowTokens: number;
  maxSafeContextTokens: number;
  estimatedVramBytes: number;
  estimatedRamBytes: number;
  schedule: TSPScheduleStep[];
  degradationReason?: string;
}

export interface RuntimeDeviceHints {
  name?: string;
  vramBudgetBytes?: number;
  ramBudgetBytes?: number;
  maxBufferSizeBytes?: number;
  hardwareConcurrency?: number;
  deviceMemoryGb?: number;
  backend?: DeviceProfile["backend"];
}

interface RuntimeNavigatorLike {
  hardwareConcurrency?: number | undefined;
  deviceMemory?: number | undefined;
}

export interface TSPMemoryEstimate {
  usableVramBytes: number;
  modelBytes: number;
  kvBytesPerToken: number;
  activationBytesPerToken: number;
  requestedKvBytes: number;
  requestedActivationBytes: number;
  requestedVramBytes: number;
  spillBytes: number;
  maxSafeContextTokens: number;
  pressureRatio: number;
  fitsSingleWindow: boolean;
}

export class ModelProfileRegistry {
  private readonly profiles = new Map<string, ModelProfile>();

  constructor(initial: ModelProfile[] = DEFAULT_MODEL_PROFILES) {
    for (const profile of initial) this.register(profile);
  }

  register(profile: ModelProfile): void {
    assertPositiveInteger(profile.layers, "model.layers");
    assertPositiveInteger(profile.hiddenSize, "model.hiddenSize");
    assertPositiveInteger(profile.kvHeads, "model.kvHeads");
    this.profiles.set(profile.modelId, profile);
  }

  get(modelId: string): ModelProfile | undefined {
    return this.profiles.get(modelId);
  }

  require(modelId: string): ModelProfile {
    const profile = this.get(modelId);
    if (!profile) throw new Error(`Unknown model profile: ${modelId}`);
    return profile;
  }

  list(): ModelProfile[] {
    return [...this.profiles.values()].sort((a, b) => a.modelId.localeCompare(b.modelId));
  }
}

export const DEFAULT_MODEL_PROFILES: ModelProfile[] = [
  {
    modelId: "Qwen/Qwen3-0.6B",
    parameterBytes: 1_200_000_000,
    layers: 28,
    hiddenSize: 1024,
    kvHeads: 8,
    contextWindowTokens: 40_960,
    family: "qwen3",
  },
  {
    modelId: "Qwen/Qwen3-0.6B-sharded",
    parameterBytes: 1_200_000_000,
    layers: 28,
    hiddenSize: 1024,
    kvHeads: 8,
    contextWindowTokens: 40_960,
    family: "qwen3",
  },
];

export function createModelProfileRegistry(initial?: ModelProfile[]): ModelProfileRegistry {
  return new ModelProfileRegistry(initial ?? DEFAULT_MODEL_PROFILES);
}

export function getDefaultModelProfile(modelId: string): ModelProfile | undefined {
  return createModelProfileRegistry().get(modelId);
}

export function detectDeviceProfile(hints: RuntimeDeviceHints = {}): DeviceProfile {
  const nav = getRuntimeNavigator();
  const deviceMemoryGb = hints.deviceMemoryGb ?? getNavigatorNumber(nav, "deviceMemory") ?? 8;
  const hardwareConcurrency = hints.hardwareConcurrency ?? nav?.hardwareConcurrency ?? 4;
  const ramBudgetBytes = hints.ramBudgetBytes ?? gbToBytes(Math.max(2, deviceMemoryGb));
  const inferredVramBytes = hints.vramBudgetBytes ?? Math.floor(ramBudgetBytes * (hardwareConcurrency >= 8 ? 0.5 : 0.33));

  return {
    name: hints.name ?? `detected-${hints.backend ?? "browser"}-${hardwareConcurrency}c`,
    vramBudgetBytes: Math.max(gbToBytes(1), inferredVramBytes),
    ramBudgetBytes,
    ...(hints.maxBufferSizeBytes ? { maxBufferSizeBytes: hints.maxBufferSizeBytes } : {}),
    backend: hints.backend ?? "unknown",
  };
}

export function estimateTSPMemory(input: TSPPlanInput): TSPMemoryEstimate {
  const usableVramBytes = Math.max(0, Math.floor(input.device.vramBudgetBytes * (1 - input.safetyMarginRatio)));
  const headDim = Math.max(1, Math.ceil(input.model.hiddenSize / Math.max(1, input.model.kvHeads)));
  const kvBytesPerToken = input.model.layers * input.model.kvHeads * headDim * input.kvPrecisionBytes * 2 * input.batchSize;
  const activationBytesPerToken = input.model.hiddenSize * input.activationPrecisionBytes * input.batchSize;
  const requestedKvBytes = input.requestedContextTokens * kvBytesPerToken;
  const requestedActivationBytes = input.requestedContextTokens * activationBytesPerToken;
  const requestedVramBytes = input.model.parameterBytes + requestedKvBytes + requestedActivationBytes;
  const roomForContext = Math.max(0, usableVramBytes - input.model.parameterBytes);
  const maxSafeContextTokens = Math.max(256, Math.floor(roomForContext / Math.max(1, kvBytesPerToken + activationBytesPerToken)));
  const spillBytes = Math.max(0, requestedVramBytes - usableVramBytes);
  return {
    usableVramBytes,
    modelBytes: input.model.parameterBytes,
    kvBytesPerToken,
    activationBytesPerToken,
    requestedKvBytes,
    requestedActivationBytes,
    requestedVramBytes,
    spillBytes,
    maxSafeContextTokens,
    pressureRatio: usableVramBytes === 0 ? 1 : requestedVramBytes / usableVramBytes,
    fitsSingleWindow: input.requestedContextTokens <= maxSafeContextTokens && requestedVramBytes <= usableVramBytes,
  };
}

export function buildFallbackTSPPlan(input: TSPPlanInput): TSPPlan {
  const estimate = estimateTSPMemory(input);
  const bytesPerToken = Math.max(1, estimate.kvBytesPerToken + estimate.activationBytesPerToken);
  const maxSafeContextTokens = estimate.maxSafeContextTokens;
  const requested = input.requestedContextTokens;
  const sequenceShards = Math.max(1, Math.ceil(requested / Math.max(1, maxSafeContextTokens)));
  const tensorShards = Math.max(1, Math.ceil(input.model.parameterBytes / Math.max(1, estimate.usableVramBytes * 0.6)));
  const activationWindowTokens = Math.min(requested, maxSafeContextTokens);

  const plan: TSPPlan = {
    mode: "fallback_budget_planner",
    sequenceShards,
    tensorShards,
    activationWindowTokens,
    maxSafeContextTokens,
    estimatedVramBytes: Math.min(estimate.usableVramBytes, input.model.parameterBytes + maxSafeContextTokens * bytesPerToken),
    estimatedRamBytes: Math.max(0, requested - maxSafeContextTokens) * bytesPerToken,
    schedule: buildTSPSchedule({ requestedContextTokens: requested, sequenceShards, tensorShards }),
  };

  if (requested > maxSafeContextTokens) {
    plan.degradationReason = "requested_context_exceeds_safe_single_window";
  } else if (!estimate.fitsSingleWindow) {
    plan.degradationReason = "estimated_vram_pressure_exceeds_budget";
  }

  return plan;
}

export function buildTSPSchedule(input: {
  requestedContextTokens: number;
  sequenceShards: number;
  tensorShards: number;
}): TSPScheduleStep[] {
  const schedule: TSPScheduleStep[] = [];
  const shardSize = Math.ceil(input.requestedContextTokens / input.sequenceShards);
  for (let s = 0; s < input.sequenceShards; s++) {
    const tokenStart = s * shardSize;
    const tokenEnd = Math.min(input.requestedContextTokens, tokenStart + shardSize);
    for (let t = 0; t < input.tensorShards; t++) {
      schedule.push({ id: `prefetch_s${s}_t${t}`, kind: "kv_prefetch", sequenceShard: s, tensorShard: t, tokenStart, tokenEnd });
      schedule.push({ id: `attn_s${s}_t${t}`, kind: "attention", sequenceShard: s, tensorShard: t, tokenStart, tokenEnd });
      schedule.push({ id: `checkpoint_s${s}_t${t}`, kind: "activation_checkpoint", sequenceShard: s, tensorShard: t, tokenStart, tokenEnd });
      schedule.push({ id: `mlp_s${s}_t${t}`, kind: "mlp", sequenceShard: s, tensorShard: t, tokenStart, tokenEnd });
    }
  }
  return schedule;
}

function getRuntimeNavigator(): RuntimeNavigatorLike | undefined {
  const root = globalThis as typeof globalThis & { navigator?: RuntimeNavigatorLike };
  return root.navigator;
}

function getNavigatorNumber(nav: RuntimeNavigatorLike | undefined, key: "deviceMemory"): number | undefined {
  const value = nav?.[key];
  return typeof value === "number" && Number.isFinite(value) ? value : undefined;
}

function gbToBytes(gb: number): number {
  return Math.floor(gb * 1024 * 1024 * 1024);
}

function assertPositiveInteger(value: number, name: string): void {
  if (!Number.isInteger(value) || value <= 0) throw new Error(`${name} must be a positive integer`);
}
