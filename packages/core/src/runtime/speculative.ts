export type SpeculativeMode = "target_only" | "draft_verify" | "tree_draft_verify" | "backend_native";

export interface SpeculativeDecodingConfig {
  enabled: boolean;
  mode: SpeculativeMode;
  draftModelId?: string | null;
  targetModelId: string;
  numSpeculativeTokens: number;
  minAcceptanceRate: number;
  disableWhenLatencyWorse: boolean;
}

export interface SpeculativeModelProfile {
  modelId: string;
  role: "draft" | "target" | "both";
  tokenizerId?: string;
  maxSpeculativeTokens?: number;
  targetModelIds?: string[];
}

export class SpeculativeModelRegistry {
  private readonly profiles = new Map<string, SpeculativeModelProfile>();

  constructor(initial: SpeculativeModelProfile[] = []) {
    for (const profile of initial) this.register(profile);
  }

  register(profile: SpeculativeModelProfile): void {
    this.profiles.set(profile.modelId, profile);
  }

  get(modelId: string): SpeculativeModelProfile | undefined {
    return this.profiles.get(modelId);
  }

  listByRole(role: "draft" | "target"): SpeculativeModelProfile[] {
    return [...this.profiles.values()]
      .filter((profile) => profile.role === role || profile.role === "both")
      .sort((a, b) => a.modelId.localeCompare(b.modelId));
  }

  isCompatiblePair(draftModelId: string, targetModelId: string): boolean {
    const draft = this.profiles.get(draftModelId);
    const target = this.profiles.get(targetModelId);
    if (!draft || !target) return false;
    if (draft.role !== "draft" && draft.role !== "both") return false;
    if (target.role !== "target" && target.role !== "both") return false;
    if (draft.tokenizerId && target.tokenizerId && draft.tokenizerId !== target.tokenizerId) return false;
    return !draft.targetModelIds || draft.targetModelIds.includes(targetModelId);
  }
}

export interface ResolveSpeculativeConfigInput {
  enabled: boolean;
  targetModelId: string;
  draftModelId?: string | null;
  mode?: Exclude<SpeculativeMode, "target_only">;
  numSpeculativeTokens?: number;
  minAcceptanceRate?: number;
  disableWhenLatencyWorse?: boolean;
}

export interface DraftToken {
  token: string;
  logprob?: number;
}

export interface TargetVerificationToken {
  token: string;
  accepted: boolean;
  replacement?: string;
}

export interface SpeculativeStepResult {
  streamedTokens: string[];
  acceptedTokens: number;
  rejectedTokens: number;
  correctedToken?: string;
}

export interface SpeculativeMetrics {
  draftTokens: number;
  acceptedTokens: number;
  rejectedTokens: number;
  acceptanceRate: number;
  draftLatencyMs: number;
  verifyLatencyMs: number;
  netSpeedupRatio: number;
  disabledReason?: string;
}

export interface SpeculativeModelPair {
  draftModelId?: string | null;
  targetModelId: string;
}

export interface SpeculativeVerificationBranch {
  branchId: string;
  draft: DraftToken[];
}

export interface SpeculativeVerificationBatch {
  requestId: string;
  modelPair: SpeculativeModelPair;
  taskType: string;
  branches: SpeculativeVerificationBranch[];
  draftLatencyMs: number;
  targetOnlyLatencyMs: number;
  minAcceptanceRate: number;
  disableWhenLatencyWorse: boolean;
}

export interface SpeculativeVerificationBranchResult {
  branchId: string;
  verification: TargetVerificationToken[];
}

export interface SpeculativeVerifierBatchResult {
  requestId?: string;
  verifyLatencyMs: number;
  branches: SpeculativeVerificationBranchResult[];
}

export type SpeculativeVerifierBackend = (
  batch: SpeculativeVerificationBatch,
) => Promise<SpeculativeVerifierBatchResult> | SpeculativeVerifierBatchResult;

export interface SpeculativeTraceRecord {
  requestId: string;
  branchId: string;
  modelPair: SpeculativeModelPair;
  taskType: string;
  draftTokens: string[];
  acceptedTokens: string[];
  rejectedTokens: string[];
  streamedTokens: string[];
  correctedToken?: string;
}

export interface SpeculativeBatchBranchResult extends SpeculativeStepResult {
  branchId: string;
}

export interface SpeculativeBatchMetrics extends SpeculativeMetrics {
  requestId?: string;
  modelPair: SpeculativeModelPair;
  taskType: string;
}

export interface SpeculativeVerificationBatchResult {
  requestId: string;
  modelPair: SpeculativeModelPair;
  taskType: string;
  branches: SpeculativeBatchBranchResult[];
  traces: SpeculativeTraceRecord[];
  metrics: SpeculativeBatchMetrics;
}

export interface SpeculationHealth {
  disabled: boolean;
  disabledReason?: string;
  rollingAcceptanceRate: number;
  consecutiveWorseRequests: number;
  sampleCount: number;
}

export function resolveSpeculativeDecodingConfig(
  input: ResolveSpeculativeConfigInput,
  registry: SpeculativeModelRegistry = new SpeculativeModelRegistry(),
): SpeculativeDecodingConfig {
  const base = {
    minAcceptanceRate: input.minAcceptanceRate ?? 0.45,
    disableWhenLatencyWorse: input.disableWhenLatencyWorse ?? true,
  };

  if (!input.enabled || !input.draftModelId || !registry.isCompatiblePair(input.draftModelId, input.targetModelId)) {
    return {
      enabled: input.enabled,
      mode: "target_only",
      draftModelId: input.draftModelId ?? null,
      targetModelId: input.targetModelId,
      numSpeculativeTokens: 0,
      ...base,
    };
  }

  const draft = registry.get(input.draftModelId);
  return {
    enabled: true,
    mode: input.mode ?? "draft_verify",
    draftModelId: input.draftModelId,
    targetModelId: input.targetModelId,
    numSpeculativeTokens: Math.min(input.numSpeculativeTokens ?? 4, draft?.maxSpeculativeTokens ?? 4),
    ...base,
  };
}

export function applySpeculativeVerification(
  draft: DraftToken[],
  verification: TargetVerificationToken[],
): SpeculativeStepResult {
  const streamedTokens: string[] = [];
  let acceptedTokens = 0;
  let rejectedTokens = 0;

  for (let i = 0; i < draft.length; i++) {
    const draftToken = draft[i];
    if (!draftToken) break;

    const verdict = verification[i];
    if (!verdict || !verdict.accepted) {
      rejectedTokens += draft.length - i;
      if (verdict?.replacement) {
        streamedTokens.push(verdict.replacement);
        return { streamedTokens, acceptedTokens, rejectedTokens, correctedToken: verdict.replacement };
      }
      return { streamedTokens, acceptedTokens, rejectedTokens };
    }
    streamedTokens.push(draftToken.token);
    acceptedTokens += 1;
  }

  return { streamedTokens, acceptedTokens, rejectedTokens };
}

export async function verifySpeculativeBatch(
  batch: SpeculativeVerificationBatch,
  backend: SpeculativeVerifierBackend,
): Promise<SpeculativeVerificationBatchResult> {
  assertUniqueBranchIds(batch.branches, batch.requestId, "input");

  let backendResult: SpeculativeVerifierBatchResult;
  try {
    backendResult = await backend(batch);
  } catch (error) {
    throw withSpeculativeBatchContext(error, batch);
  }

  if (backendResult.requestId && backendResult.requestId !== batch.requestId) {
    throw new Error(
      `Speculative verifier failed for request ${batch.requestId}: backend returned request ${backendResult.requestId}`,
    );
  }
  assertUniqueBranchIds(backendResult.branches, batch.requestId, "backend");

  const verificationByBranch = new Map<string, SpeculativeVerificationBranchResult>();
  for (const branchResult of backendResult.branches) {
    verificationByBranch.set(branchResult.branchId, branchResult);
  }

  const branches: SpeculativeBatchBranchResult[] = [];
  const traces: SpeculativeTraceRecord[] = [];
  for (const branch of batch.branches) {
    const branchResult = verificationByBranch.get(branch.branchId);
    if (!branchResult) {
      throw new Error(
        `Speculative verifier failed for request ${batch.requestId} branch ${branch.branchId}: missing verification result`,
      );
    }

    const step = applySpeculativeVerification(branch.draft, branchResult.verification);
    branches.push({ branchId: branch.branchId, ...step });
    traces.push(createSpeculativeTraceRecord(batch, branch, branchResult.verification, step));
  }

  return {
    requestId: batch.requestId,
    modelPair: batch.modelPair,
    taskType: batch.taskType,
    branches,
    traces,
    metrics: summarizeSpeculativeBatchMetrics(traces, {
      requestId: batch.requestId,
      modelPair: batch.modelPair,
      taskType: batch.taskType,
      draftLatencyMs: batch.draftLatencyMs,
      verifyLatencyMs: backendResult.verifyLatencyMs,
      targetOnlyLatencyMs: batch.targetOnlyLatencyMs,
      minAcceptanceRate: batch.minAcceptanceRate,
      disableWhenLatencyWorse: batch.disableWhenLatencyWorse,
    }),
  };
}

export function summarizeSpeculativeBatchMetrics(
  traces: SpeculativeTraceRecord[],
  input: {
    requestId?: string;
    modelPair?: SpeculativeModelPair;
    taskType?: string;
    draftLatencyMs: number;
    verifyLatencyMs: number;
    targetOnlyLatencyMs: number;
    minAcceptanceRate: number;
    disableWhenLatencyWorse: boolean;
  },
): SpeculativeBatchMetrics {
  const firstTrace = traces[0];
  const modelPair = input.modelPair ?? firstTrace?.modelPair;
  const taskType = input.taskType ?? firstTrace?.taskType;
  const requestId = input.requestId ?? firstTrace?.requestId;
  if (!modelPair || !taskType) {
    throw new Error("Speculative batch metrics require a model pair and task type");
  }

  const totals = traces.reduce(
    (acc, trace) => ({
      draftTokens: acc.draftTokens + trace.draftTokens.length,
      acceptedTokens: acc.acceptedTokens + trace.acceptedTokens.length,
      rejectedTokens: acc.rejectedTokens + trace.rejectedTokens.length,
    }),
    { draftTokens: 0, acceptedTokens: 0, rejectedTokens: 0 },
  );
  const metrics = measureSpeculativeMetrics({
    ...totals,
    draftLatencyMs: input.draftLatencyMs,
    verifyLatencyMs: input.verifyLatencyMs,
    targetOnlyLatencyMs: input.targetOnlyLatencyMs,
    minAcceptanceRate: input.minAcceptanceRate,
    disableWhenLatencyWorse: input.disableWhenLatencyWorse,
  });

  return {
    ...metrics,
    ...(requestId ? { requestId } : {}),
    modelPair,
    taskType,
  };
}

export function measureSpeculativeMetrics(input: {
  draftTokens: number;
  acceptedTokens: number;
  rejectedTokens: number;
  draftLatencyMs: number;
  verifyLatencyMs: number;
  targetOnlyLatencyMs: number;
  minAcceptanceRate: number;
  disableWhenLatencyWorse: boolean;
}): SpeculativeMetrics {
  const totalVerified = input.acceptedTokens + input.rejectedTokens;
  const acceptanceRate = totalVerified === 0 ? 0 : input.acceptedTokens / totalVerified;
  const speculativeLatency = input.draftLatencyMs + input.verifyLatencyMs;
  const netSpeedupRatio = speculativeLatency <= 0 ? 1 : input.targetOnlyLatencyMs / speculativeLatency;
  const metrics: SpeculativeMetrics = {
    draftTokens: input.draftTokens,
    acceptedTokens: input.acceptedTokens,
    rejectedTokens: input.rejectedTokens,
    acceptanceRate,
    draftLatencyMs: input.draftLatencyMs,
    verifyLatencyMs: input.verifyLatencyMs,
    netSpeedupRatio,
  };
  const disabledReason = shouldDisableSpeculation(metrics, input.minAcceptanceRate, input.disableWhenLatencyWorse);
  return disabledReason ? { ...metrics, disabledReason } : metrics;
}

export function evaluateSpeculationAutoDisable(
  samples: SpeculativeMetrics[],
  config: Pick<SpeculativeDecodingConfig, "minAcceptanceRate" | "disableWhenLatencyWorse">,
  worseRequestLimit = 3,
): SpeculationHealth {
  const totals = samples.reduce(
    (acc, metrics) => ({
      accepted: acc.accepted + metrics.acceptedTokens,
      rejected: acc.rejected + metrics.rejectedTokens,
    }),
    { accepted: 0, rejected: 0 },
  );
  const totalVerified = totals.accepted + totals.rejected;
  const rollingAcceptanceRate = totalVerified === 0 ? 0 : totals.accepted / totalVerified;
  let consecutiveWorseRequests = 0;
  for (let i = samples.length - 1; i >= 0; i--) {
    const metrics = samples[i];
    if (!metrics || metrics.netSpeedupRatio >= 1) break;
    consecutiveWorseRequests += 1;
  }

  const disabledReason = rollingAcceptanceRate < config.minAcceptanceRate
    ? "acceptance_rate_below_threshold"
    : config.disableWhenLatencyWorse && consecutiveWorseRequests >= worseRequestLimit
      ? "speculation_slower_than_target_only"
      : undefined;

  return {
    disabled: Boolean(disabledReason),
    ...(disabledReason ? { disabledReason } : {}),
    rollingAcceptanceRate,
    consecutiveWorseRequests,
    sampleCount: samples.length,
  };
}

export function shouldDisableSpeculation(
  metrics: SpeculativeMetrics,
  minAcceptanceRate: number,
  disableWhenLatencyWorse = true,
): string | undefined {
  if (metrics.acceptanceRate < minAcceptanceRate) return "acceptance_rate_below_threshold";
  if (disableWhenLatencyWorse && metrics.netSpeedupRatio < 1) return "speculation_slower_than_target_only";
  return undefined;
}

function createSpeculativeTraceRecord(
  batch: SpeculativeVerificationBatch,
  branch: SpeculativeVerificationBranch,
  verification: TargetVerificationToken[],
  step: SpeculativeStepResult,
): SpeculativeTraceRecord {
  const acceptedTokens: string[] = [];
  const rejectedTokens: string[] = [];

  for (let i = 0; i < branch.draft.length; i++) {
    const draftToken = branch.draft[i];
    if (!draftToken) break;

    const verdict = verification[i];
    if (!verdict || !verdict.accepted) {
      rejectedTokens.push(...branch.draft.slice(i).map((token) => token.token));
      break;
    }
    acceptedTokens.push(draftToken.token);
  }

  return {
    requestId: batch.requestId,
    branchId: branch.branchId,
    modelPair: batch.modelPair,
    taskType: batch.taskType,
    draftTokens: branch.draft.map((token) => token.token),
    acceptedTokens,
    rejectedTokens,
    streamedTokens: step.streamedTokens,
    ...(step.correctedToken ? { correctedToken: step.correctedToken } : {}),
  };
}

function withSpeculativeBatchContext(error: unknown, batch: SpeculativeVerificationBatch): Error {
  const branchId = typeof error === "object" && error !== null && "branchId" in error
    ? String((error as { branchId?: unknown }).branchId)
    : batch.branches.map((branch) => branch.branchId).join(",");
  const originalMessage = error instanceof Error ? error.message : String(error);
  const wrapped = new Error(
    `Speculative verifier failed for request ${batch.requestId} branch ${branchId}: ${originalMessage}`,
  );
  if (error instanceof Error) {
    if (wrapped.stack && error.stack) wrapped.stack = `${wrapped.stack}\nCaused by: ${error.stack}`;
    wrapped.cause = error;
  }
  return wrapped;
}

function assertUniqueBranchIds(
  branches: readonly { branchId: string }[],
  requestId: string,
  source: "input" | "backend",
): void {
  const seen = new Set<string>();
  for (const branch of branches) {
    if (seen.has(branch.branchId)) {
      throw new Error(
        `Speculative verifier failed for request ${requestId} branch ${branch.branchId}: duplicate ${source} branch id`,
      );
    }
    seen.add(branch.branchId);
  }
}
