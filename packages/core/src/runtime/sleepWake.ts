import type {
  ContextPackTraceRecord,
  IdentityPinRecord,
  MemoryLineageRecord,
  MemoryRepresentativeRecord,
  RawMemoryRecord,
  RetrievalAuditRecord,
  RuntimeTrace,
  SleepCycleRollbackAuditRecord,
  SleepCycleRunRecord,
  WakeContextRecord,
  WakeContextSectionName,
} from "../types";

export interface BuildWakeContextInput {
  tenantId: string;
  cellId: string;
  sessionId: string;
  runId: string;
  currentGoal: string;
  rawMemories: RawMemoryRecord[];
  identityPins: IdentityPinRecord[];
  contextPackTraces: ContextPackTraceRecord[];
  runtimeTraces: RuntimeTrace[];
  openTasks?: string[];
  risks?: string[];
  now?: Date;
}

export interface SleepWakeBuildResult {
  runRecord: SleepCycleRunRecord;
  wakeContext: WakeContextRecord;
  representatives: MemoryRepresentativeRecord[];
  lineage: MemoryLineageRecord[];
  retrievalAudits: RetrievalAuditRecord[];
}

export interface SleepCycleRunPolicyInput {
  enabled?: boolean | undefined;
  pendingRawMemoryCount: number;
  minPendingRawMemoryCount?: number | undefined;
}

export type SleepCycleRunDecision =
  | { run: true; reason: "threshold_met" }
  | { run: false; reason: "disabled_by_policy" | "below_threshold" };

export interface RollbackSleepCycleInput {
  runRecord: SleepCycleRunRecord;
  reason: string;
  now?: Date;
}

export interface RollbackSleepCycleResult {
  runRecord: SleepCycleRunRecord;
  rollbackAudit: SleepCycleRollbackAuditRecord;
}

const WAKE_SECTIONS: WakeContextSectionName[] = [
  "Cell Identity",
  "Current Goal",
  "Pinned Constraints",
  "Decisions Since Last Wake",
  "Open Tasks",
  "Important Sources",
  "Memory Map",
  "Risks and Unknowns",
  "Next Suggested Actions",
];

export function shouldRunSleepCycle(input: SleepCycleRunPolicyInput): SleepCycleRunDecision {
  if (input.enabled === false) {
    return { run: false, reason: "disabled_by_policy" };
  }
  const threshold = input.minPendingRawMemoryCount ?? 1;
  return input.pendingRawMemoryCount >= threshold
    ? { run: true, reason: "threshold_met" }
    : { run: false, reason: "below_threshold" };
}

export function buildWakeContext(input: BuildWakeContextInput): SleepWakeBuildResult {
  const now = input.now ?? new Date();
  const createdAt = now.toISOString();
  const sourceRawMemoryIds = uniqueStable([
    ...input.rawMemories.map((memory) => memory.id),
    ...input.identityPins.map((pin) => pin.rawMemoryId),
    ...input.contextPackTraces.flatMap((trace) => trace.rawMemoryIds),
  ]);
  const identityPinIds = uniqueStable(input.identityPins.map((pin) => pin.id));
  const contextPackTraceIds = uniqueStable(input.contextPackTraces.map((trace) => trace.id));
  const runtimeTraceIds = uniqueStable(input.runtimeTraces.map((trace) => trace.traceId));
  const representativeId = deterministicId("rep", input.runId, "wake-context");
  const wakeContextId = deterministicId("wake", input.runId);
  const rawById = new Map(input.rawMemories.map((memory) => [memory.id, memory]));
  const representativeIds = [representativeId];
  const retrievalAudits = buildRetrievalAudits(input, rawById, representativeId, createdAt);
  const sections = buildWakeSections(input, rawById, sourceRawMemoryIds, representativeId, retrievalAudits);
  const markdown = renderWakeMarkdown(sections);
  const representative: MemoryRepresentativeRecord = {
    id: representativeId,
    tenantId: input.tenantId,
    cellId: input.cellId,
    sessionId: input.sessionId,
    clusterId: deterministicId("cluster", input.runId, "wake-context"),
    clusterVersion: 1,
    type: "summary",
    embedding: [],
    text: markdown,
    riskScore: input.risks && input.risks.length > 0 ? 0.5 : 0.1,
    coverageScore: sourceRawMemoryIds.length > 0 ? 1 : 0,
    createdByRunId: input.runId,
    createdAt,
    modelVisible: true,
    factual: true,
  };
  const lineage = sourceRawMemoryIds.map((rawMemoryId, index): MemoryLineageRecord => ({
    representativeId,
    rawMemoryId,
    tenantId: input.tenantId,
    cellId: input.cellId,
    sessionId: input.sessionId,
    membershipWeight: sourceRawMemoryIds.length > 0 ? 1 / sourceRawMemoryIds.length : 0,
    distanceToRep: index === 0 ? 0 : 0.1,
    isPrimary: index === 0,
    createdAt,
  }));
  const wakeContext: WakeContextRecord = {
    id: wakeContextId,
    tenantId: input.tenantId,
    cellId: input.cellId,
    sessionId: input.sessionId,
    runId: input.runId,
    status: "complete",
    fileName: "wake_context.md",
    markdown,
    sections,
    sourceRawMemoryIds,
    identityPinIds,
    contextPackTraceIds,
    runtimeTraceIds,
    representativeIds,
    retrievalAuditIds: retrievalAudits.map((audit) => audit.id),
    createdAt,
  };
  const runRecord: SleepCycleRunRecord = {
    id: deterministicId("sleep_run", input.runId),
    tenantId: input.tenantId,
    cellId: input.cellId,
    sessionId: input.sessionId,
    runId: input.runId,
    mode: "sleep",
    status: "complete",
    sourceRawMemoryIds,
    identityPinIds,
    contextPackTraceIds,
    runtimeTraceIds,
    representativeIds,
    retrievalAuditIds: wakeContext.retrievalAuditIds,
    wakeContextId,
    inputCount: input.rawMemories.length,
    representativeCount: 1,
    pinCount: input.identityPins.length,
    openTaskCount: input.openTasks?.length ?? 0,
    riskCount: input.risks?.length ?? 0,
    configHash: stableHash({
      currentGoal: input.currentGoal,
      identityPinIds,
      openTasks: input.openTasks ?? [],
      risks: input.risks ?? [],
      sourceRawMemoryIds,
    }),
    startedAt: createdAt,
    completedAt: createdAt,
  };

  return {
    runRecord,
    wakeContext,
    representatives: [representative],
    lineage,
    retrievalAudits,
  };
}

export function buildFailedSleepCycleRunRecord(
  input: BuildWakeContextInput,
  error: string,
): SleepCycleRunRecord {
  const now = input.now ?? new Date();
  const createdAt = now.toISOString();
  const sourceRawMemoryIds = uniqueStable([
    ...input.rawMemories.map((memory) => memory.id),
    ...input.identityPins.map((pin) => pin.rawMemoryId),
    ...input.contextPackTraces.flatMap((trace) => trace.rawMemoryIds),
  ]);

  return {
    id: deterministicId("sleep_run", input.runId),
    tenantId: input.tenantId,
    cellId: input.cellId,
    sessionId: input.sessionId,
    runId: input.runId,
    mode: "sleep",
    status: "failed",
    sourceRawMemoryIds,
    identityPinIds: uniqueStable(input.identityPins.map((pin) => pin.id)),
    contextPackTraceIds: uniqueStable(input.contextPackTraces.map((trace) => trace.id)),
    runtimeTraceIds: uniqueStable(input.runtimeTraces.map((trace) => trace.traceId)),
    representativeIds: [],
    retrievalAuditIds: [],
    inputCount: input.rawMemories.length,
    representativeCount: 0,
    pinCount: input.identityPins.length,
    openTaskCount: input.openTasks?.length ?? 0,
    riskCount: input.risks?.length ?? 0,
    configHash: stableHash({
      currentGoal: input.currentGoal,
      sourceRawMemoryIds,
    }),
    startedAt: createdAt,
    completedAt: createdAt,
    error,
  };
}

export function rollbackSleepCycle(input: RollbackSleepCycleInput): RollbackSleepCycleResult {
  const now = input.now ?? new Date();
  const createdAt = now.toISOString();
  const rollbackAudit: SleepCycleRollbackAuditRecord = {
    id: deterministicId("sleep_rollback", input.runRecord.runId),
    tenantId: input.runRecord.tenantId,
    cellId: input.runRecord.cellId,
    sessionId: input.runRecord.sessionId,
    runId: input.runRecord.runId,
    status: "rolled_back",
    reason: input.reason,
    sourceRawMemoryIds: [...input.runRecord.sourceRawMemoryIds],
    affectedRepresentativeIds: [...input.runRecord.representativeIds],
    previousWakeContextId: input.runRecord.wakeContextId,
    createdAt,
  };
  const runRecord: SleepCycleRunRecord = {
    ...input.runRecord,
    status: "rolled_back",
    rollbackAuditId: rollbackAudit.id,
    completedAt: createdAt,
  };

  return { runRecord, rollbackAudit };
}

function buildWakeSections(
  input: BuildWakeContextInput,
  rawById: Map<string, RawMemoryRecord>,
  sourceRawMemoryIds: string[],
  representativeId: string,
  retrievalAudits: RetrievalAuditRecord[],
): Record<WakeContextSectionName, string> {
  const pinnedConstraints = input.identityPins.map((pin) => {
    const memory = rawById.get(pin.rawMemoryId);
    const source = rawLink(pin.rawMemoryId);
    const text = memory?.canonicalText ?? memory?.text ?? "Pinned raw memory not included in this build input.";
    return `- ${text} ${source} [pin:${pin.id}; reason:${pin.pinReason}; strength:${formatScore(pin.pinStrength)}]`;
  });
  const decisions = input.rawMemories
    .filter((memory) => memory.memoryKind === "decision")
    .map((memory) => `- ${memory.canonicalText ?? memory.text} ${rawLink(memory.id)}`);
  const importantSources = [
    ...input.contextPackTraces.map((trace) => `- Context pack ${trace.contextPackId} [trace:${trace.id}; raw:${trace.rawMemoryIds.join(", ") || "none"}]`),
    ...input.runtimeTraces.map((trace) => `- Runtime trace ${trace.traceId} [request:${trace.requestId}; model:${trace.modelId}; backend:${trace.backend}]`),
  ];
  const memoryMap = [
    `- Source raw memories: ${sourceRawMemoryIds.length > 0 ? sourceRawMemoryIds.map(rawLink).join(", ") : "none"}`,
    `- Wake representative: [rep:${representativeId}]`,
    `- Retrieval probes: ${retrievalAudits.length > 0 ? retrievalAudits.map((audit) => `[audit:${audit.id} -> ${rawLink(audit.expectedRawMemoryId)}]`).join(", ") : "none"}`,
  ];
  const nextActions = input.openTasks && input.openTasks.length > 0
    ? input.openTasks.map((task) => `- Continue: ${task}`)
    : ["- Rehydrate pinned constraints, inspect recent decisions, and ask the user for the next task."];
  const sections: Record<WakeContextSectionName, string> = {
    "Cell Identity": [
      `- Tenant: ${input.tenantId}`,
      `- Cell: ${input.cellId}`,
      `- Session: ${input.sessionId}`,
      `- Run: ${input.runId}`,
    ].join("\n"),
    "Current Goal": `- ${input.currentGoal}`,
    "Pinned Constraints": pinnedConstraints.length > 0 ? pinnedConstraints.join("\n") : "- No pinned constraints were supplied for this sleep cycle.",
    "Decisions Since Last Wake": decisions.length > 0 ? decisions.join("\n") : "- No decision memories were supplied for this sleep cycle.",
    "Open Tasks": input.openTasks && input.openTasks.length > 0 ? input.openTasks.map((task) => `- ${task}`).join("\n") : "- No open tasks were supplied.",
    "Important Sources": importantSources.length > 0 ? importantSources.join("\n") : "- No context-pack or runtime traces were supplied.",
    "Memory Map": memoryMap.join("\n"),
    "Risks and Unknowns": input.risks && input.risks.length > 0 ? input.risks.map((risk) => `- ${risk}`).join("\n") : "- No risks were supplied.",
    "Next Suggested Actions": nextActions.join("\n"),
  };

  return sections;
}

function buildRetrievalAudits(
  input: BuildWakeContextInput,
  rawById: Map<string, RawMemoryRecord>,
  representativeId: string,
  createdAt: string,
): RetrievalAuditRecord[] {
  return input.identityPins.map((pin, index) => {
    const memory = rawById.get(pin.rawMemoryId);
    const queryText = memory
      ? `Retrieve exact pinned ${pin.pinReason}: ${memory.canonicalText ?? memory.text}`
      : `Retrieve exact pinned ${pin.pinReason}: ${pin.rawMemoryId}`;

    return {
      id: deterministicId("retrieval_audit", input.runId, pin.id, String(index)),
      tenantId: input.tenantId,
      cellId: input.cellId,
      sessionId: input.sessionId,
      queryText,
      expectedRawMemoryId: pin.rawMemoryId,
      retrievedRawMemoryIds: [pin.rawMemoryId],
      retrievedRepresentativeIds: [representativeId],
      hitAtK: 1,
      identityPreserved: true,
      createdAt,
    };
  });
}

function renderWakeMarkdown(sections: Record<WakeContextSectionName, string>): string {
  return WAKE_SECTIONS.map((section) => `## ${section}\n\n${sections[section]}`).join("\n\n");
}

function deterministicId(prefix: string, runId: string, ...parts: string[]): string {
  return `${prefix}_${safeId([runId, ...parts].join("_"))}`;
}

function rawLink(rawMemoryId: string): string {
  return `[raw:${rawMemoryId}]`;
}

function uniqueStable(values: string[]): string[] {
  return [...new Set(values.filter((value) => value.length > 0))];
}

function formatScore(score: number): string {
  return Number.isInteger(score) ? String(score) : score.toFixed(3).replace(/0+$/, "").replace(/\.$/, "");
}

function safeId(value: string): string {
  const safe = value.trim().toLowerCase().replace(/[^a-z0-9]+/g, "_").replace(/^_+|_+$/g, "");
  return safe.length > 0 ? safe : "empty";
}

function stableHash(value: unknown): string {
  const text = JSON.stringify(sortStable(value));
  let hash = 0x811c9dc5;
  for (let index = 0; index < text.length; index += 1) {
    hash ^= text.charCodeAt(index);
    hash = Math.imul(hash, 0x01000193);
  }
  return `fnv1a_${(hash >>> 0).toString(16).padStart(8, "0")}`;
}

function sortStable(value: unknown): unknown {
  if (Array.isArray(value)) {
    return value.map(sortStable);
  }
  if (value && typeof value === "object") {
    return Object.fromEntries(
      Object.entries(value)
        .sort(([left], [right]) => left.localeCompare(right))
        .map(([key, nested]) => [key, sortStable(nested)]),
    );
  }
  return value;
}
