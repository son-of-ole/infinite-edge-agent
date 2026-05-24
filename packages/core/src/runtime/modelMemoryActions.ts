import type {
  ModelMemoryActionExecutionResult,
  ModelMemoryActionMode,
  ModelMemoryActionRecord,
  ModelMemoryActionTrace,
  ModelMemoryPolicyDecision,
  ModelMemoryPolicyViolation,
  ModelMemoryPolicyViolationCode,
  ProposeModelMemoryActionOptions,
  ProposeModelMemoryActionRequest,
  ProposeModelMemoryActionResponse,
} from "../types";

const DEFAULT_MIN_CONFIDENCE = 0.75;
const ACTIONS_REQUIRING_TARGETS = new Set<ProposeModelMemoryActionRequest["actionType"]>([
  "pin_memory",
  "request_consolidation",
  "request_retrieval_probe",
  "forget_memory",
]);

export interface ModelMemoryToolInterface {
  toolName: "propose_model_memory_action";
  mode: ModelMemoryActionMode;
  systemInstruction: string;
  jsonSchema: {
    type: "object";
    required: string[];
    properties: Record<string, unknown>;
  };
}

export function buildModelMemoryToolInterface(input: {
  mode?: ModelMemoryActionMode;
  tenantId?: string;
  cellId?: string;
} = {}): ModelMemoryToolInterface {
  const mode = input.mode ?? "shadow";
  return {
    toolName: "propose_model_memory_action",
    mode,
    systemInstruction: [
      `Model memory actions are available in ${mode} mode.`,
      "Use the tool only for explicit memory writes, pins, retrieval probes, consolidation requests, or forget requests.",
      "Never execute destructive memory changes unless policy approves them.",
      input.tenantId || input.cellId
        ? `Requests must stay within tenant=${input.tenantId ?? "unspecified"} and cell=${input.cellId ?? "unspecified"}.`
        : "Requests must stay within the active tenant and cell scope.",
    ].join(" "),
    jsonSchema: {
      type: "object",
      required: ["sessionId", "modelId", "actionType", "targetIds", "confidence"],
      properties: {
        tenantId: { type: "string" },
        cellId: { type: "string" },
        sessionId: { type: "string" },
        modelId: { type: "string" },
        actionType: {
          type: "string",
          enum: ["create_raw_memory", "pin_memory", "request_consolidation", "request_retrieval_probe", "forget_memory"],
        },
        targetIds: { type: "array", items: { type: "string" } },
        arguments: { type: "object" },
        confidence: { type: "number", minimum: 0, maximum: 1 },
      },
    },
  };
}

export function evaluateModelMemoryActionPolicy(
  request: ProposeModelMemoryActionRequest,
  options: Pick<ProposeModelMemoryActionOptions, "mode" | "policy"> = {},
): ModelMemoryPolicyDecision {
  const mode = options.mode ?? "shadow";
  const policy = options.policy ?? {};
  const minConfidence = policy.minConfidence ?? DEFAULT_MIN_CONFIDENCE;
  const violations: ModelMemoryPolicyViolation[] = [];

  if (mode === "disabled") {
    violations.push(violation(
      "memory_action_disabled",
      "Model memory actions are disabled by policy.",
    ));
  }

  if (!Number.isFinite(request.confidence) || request.confidence < minConfidence) {
    violations.push(violation(
      "low_confidence",
      `Model confidence ${formatConfidence(request.confidence)} is below policy threshold ${formatConfidence(minConfidence)}.`,
    ));
  }

  if (ACTIONS_REQUIRING_TARGETS.has(request.actionType) && getValidTargetIds(request).length === 0) {
    violations.push(violation(
      "missing_target_ids",
      `Action ${request.actionType} requires at least one target id.`,
    ));
  }

  if (request.actionType === "forget_memory" && policy.allowDestructiveActions !== true) {
    violations.push(violation(
      "destructive_action_not_allowed",
      "Destructive model memory actions require explicit policy approval.",
    ));
  }

  if (request.actionType === "pin_memory" && !hasValidPinArguments(request.arguments)) {
    violations.push(violation(
      "invalid_pin_arguments",
      "pin_memory requires a non-empty pinReason and pinStrength between 0 and 1.",
    ));
  }

  if (
    request.tenantId !== undefined
    && policy.scope?.tenantId !== undefined
    && request.tenantId !== policy.scope.tenantId
  ) {
    violations.push(violation(
      "tenant_scope_mismatch",
      `Request tenant ${request.tenantId} is outside policy scope ${policy.scope.tenantId}.`,
    ));
  }

  if (
    request.cellId !== undefined
    && policy.scope?.cellId !== undefined
    && request.cellId !== policy.scope.cellId
  ) {
    violations.push(violation(
      "cell_scope_mismatch",
      `Request cell ${request.cellId} is outside policy scope ${policy.scope.cellId}.`,
    ));
  }

  return {
    approved: violations.length === 0,
    policyNotes: violations.length > 0
      ? violations.map((item) => item.message)
      : [`Approved by model memory action policy in ${mode} mode.`],
    violations,
  };
}

export async function proposeModelMemoryAction(
  request: ProposeModelMemoryActionRequest,
  options: ProposeModelMemoryActionOptions = {},
): Promise<ProposeModelMemoryActionResponse> {
  const mode: ModelMemoryActionMode = options.mode ?? "shadow";
  const now = options.now ?? new Date();
  const actionId = options.actionIdFactory?.(request) ?? makeActionId(request, now);
  const decision = evaluateModelMemoryActionPolicy(request, {
    mode,
    policy: options.policy,
  });
  const trace: ModelMemoryActionTrace = {
    actionId,
    request: cloneRequest(request),
    decision,
    mode,
    timestamp: now.toISOString(),
    policyViolations: decision.violations,
  };

  let executed = false;
  let resultIds: string[] = [];
  const policyNotes = [...decision.policyNotes];

  if (decision.approved && mode === "shadow") {
    policyNotes.push("Shadow mode recorded the approved action without executing it.");
  }

  if (decision.approved && mode === "enforced" && options.execute) {
    const result = await options.execute(request, trace);
    resultIds = normalizeExecutionResult(result);
    executed = true;
  }

  if (decision.approved && mode === "enforced" && !options.execute) {
    policyNotes.push("Enforced mode approved the action, but no execute callback was provided.");
  }

  if (options.onTrace) {
    await options.onTrace(trace);
  }

  return {
    actionId,
    approved: decision.approved,
    executed,
    resultIds,
    policyNotes,
    trace,
  };
}

export function modelMemoryActionTraceToRecord(
  trace: ModelMemoryActionTrace,
  scope: { tenantId: string; cellId: string },
): ModelMemoryActionRecord {
  return {
    id: trace.actionId,
    tenantId: trace.request.tenantId ?? scope.tenantId,
    cellId: trace.request.cellId ?? scope.cellId,
    sessionId: trace.request.sessionId,
    modelId: trace.request.modelId,
    actionType: trace.request.actionType,
    targetIds: [...trace.request.targetIds],
    argumentsJson: trace.request.arguments ? { ...trace.request.arguments } : {},
    confidence: trace.request.confidence,
    approvedByPolicy: trace.decision.approved,
    createdAt: trace.timestamp,
    mode: trace.mode,
    policyViolations: trace.policyViolations.map((item) => ({ ...item })),
  };
}

function hasValidPinArguments(args: Record<string, unknown> | undefined): boolean {
  if (!args) return false;
  return typeof args.pinReason === "string"
    && args.pinReason.trim().length > 0
    && typeof args.pinStrength === "number"
    && Number.isFinite(args.pinStrength)
    && args.pinStrength > 0
    && args.pinStrength <= 1;
}

function getValidTargetIds(request: ProposeModelMemoryActionRequest): string[] {
  return Array.isArray(request.targetIds)
    ? request.targetIds.filter((id) => typeof id === "string" && id.trim().length > 0)
    : [];
}

function normalizeExecutionResult(result: ModelMemoryActionExecutionResult | string[] | void): string[] {
  if (Array.isArray(result)) return result.filter((id): id is string => typeof id === "string" && id.length > 0);
  if (result && Array.isArray(result.resultIds)) {
    return result.resultIds.filter((id): id is string => typeof id === "string" && id.length > 0);
  }
  return [];
}

function cloneRequest(request: ProposeModelMemoryActionRequest): ProposeModelMemoryActionRequest {
  return {
    ...request,
    targetIds: Array.isArray(request.targetIds) ? [...request.targetIds] : [],
    arguments: request.arguments ? { ...request.arguments } : undefined,
  };
}

function violation(code: ModelMemoryPolicyViolationCode, message: string): ModelMemoryPolicyViolation {
  return { code, message };
}

function formatConfidence(value: number): string {
  return Number.isFinite(value) ? value.toFixed(2) : String(value);
}

function makeActionId(request: ProposeModelMemoryActionRequest, now: Date): string {
  if (typeof crypto !== "undefined" && "randomUUID" in crypto) {
    return `action_${crypto.randomUUID()}`;
  }
  const safeAction = request.actionType.replace(/[^a-z0-9_]/gi, "_");
  return `action_${safeAction}_${now.getTime()}`;
}
