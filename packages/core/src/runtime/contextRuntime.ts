import type {
  ChatMessage,
  ContextPackTraceRecord,
  IdentityPinRecord,
  MemorySearchHit,
  PackedContext,
  RetrievalAuditRecord,
} from "../types";
import { packContext } from "../contextPacking";

export type ContextFrameKind = "recent" | "retrieved" | "summary" | "anchor" | "tool" | "document";

export interface ProvenanceRef {
  sourceType: "ledger" | "memory_chunk" | "document" | "tool" | "summary";
  sourceId: string;
  startOffset?: number;
  endOffset?: number;
  hash?: string;
}

export interface ContextFrame {
  id: string;
  kind: ContextFrameKind;
  text: string;
  tokenCount: number;
  priority: number;
  provenance: ProvenanceRef[];
}

export interface ContextRebuildPlan {
  requestId: string;
  frames: ContextFrame[];
  pinnedAnchorIds: string[];
  retrievedMemoryIds: string[];
  droppedFrameIds: string[];
  estimatedTokens: number;
  memoryPriorityMap: Record<string, ContextMemoryPriority>;
  sourceLineageMap: Record<string, ContextSourceLineage>;
  learningSignals: ContextRebuildLearningSignals;
}

export interface ContextSourceLineage {
  rawMemoryIds: string[];
  representativeIds: string[];
  identityPinIds: string[];
}

export interface ContextMemoryPriority extends ContextSourceLineage {
  baseScore: number;
  finalScore: number;
  protected: boolean;
  reasons: string[];
}

export interface ContextRebuildLearningSignals {
  contextTraceCount: number;
  retrievalAuditCount: number;
  boostedMemoryIds: string[];
  protectedMemoryIds: string[];
}

export interface BuildContextPlanInput {
  requestId: string;
  systemPrompt: string;
  userMessage: string;
  recentMessages: ChatMessage[];
  retrievedMemory: MemorySearchHit[];
  identityPins?: IdentityPinRecord[];
  contextPackTraces?: ContextPackTraceRecord[];
  retrievalAudits?: RetrievalAuditRecord[];
  maxRetrievedMemoryTokens: number;
  maxRecentConversationTokens: number;
  maxPromptTokens: number;
}

export function buildContextRuntimePlan(input: BuildContextPlanInput): { plan: ContextRebuildPlan; packed: PackedContext } {
  const adaptive = buildAdaptiveMemorySelection(input);
  const packed = packContext({
    systemPrompt: input.systemPrompt,
    retrievedMemory: adaptive.memory,
    recentMessages: input.recentMessages,
    userMessage: input.userMessage,
    config: {
      maxRetrievedMemoryTokens: input.maxRetrievedMemoryTokens,
      maxRecentConversationTokens: input.maxRecentConversationTokens,
      maxPromptTokens: input.maxPromptTokens,
    },
  });

  const frames: ContextFrame[] = [
    {
      id: `${input.requestId}:current_user`,
      kind: "anchor",
      text: input.userMessage,
      tokenCount: estimateTokens(input.userMessage),
      priority: 1,
      provenance: [{ sourceType: "ledger", sourceId: `${input.requestId}:user` }],
    },
    ...input.retrievedMemory.map((memory, index) => ({
      id: memory.id,
      kind: "retrieved" as const,
      text: memory.text,
      tokenCount: memory.tokenCount,
      priority: Math.max(0, Math.min(1, adaptive.priorityMap[memory.id]?.finalScore ?? memory.score ?? 0)) - index * 0.001,
      provenance: [{ sourceType: "memory_chunk" as const, sourceId: memory.id }],
    })),
  ];

  const included = new Set(packed.includedMemoryIds);
  const droppedFrameIds = frames.filter((frame) => frame.kind === "retrieved" && !included.has(frame.id)).map((frame) => frame.id);

  return {
    packed,
    plan: {
      requestId: input.requestId,
      frames,
      pinnedAnchorIds: [`${input.requestId}:current_user`],
      retrievedMemoryIds: input.retrievedMemory.map((memory) => memory.id),
      droppedFrameIds,
      estimatedTokens: packed.estimatedTokens,
      memoryPriorityMap: adaptive.priorityMap,
      sourceLineageMap: adaptive.sourceLineageMap,
      learningSignals: adaptive.learningSignals,
    },
  };
}

interface AdaptiveMemorySelection {
  memory: MemorySearchHit[];
  priorityMap: Record<string, ContextMemoryPriority>;
  sourceLineageMap: Record<string, ContextSourceLineage>;
  learningSignals: ContextRebuildLearningSignals;
}

function buildAdaptiveMemorySelection(input: BuildContextPlanInput): AdaptiveMemorySelection {
  const identityPins = input.identityPins ?? [];
  const contextPackTraces = input.contextPackTraces ?? [];
  const retrievalAudits = input.retrievalAudits ?? [];
  const pinByRawId = new Map(identityPins.map((pin) => [pin.rawMemoryId, pin]));
  const failedAuditRawIds = new Set(retrievalAudits
    .filter((audit) => !audit.identityPreserved)
    .map((audit) => audit.expectedRawMemoryId));
  const preservedAuditRawIds = new Set(retrievalAudits
    .filter((audit) => audit.identityPreserved)
    .map((audit) => audit.expectedRawMemoryId));
  const priorityMap: Record<string, ContextMemoryPriority> = {};
  const sourceLineageMap: Record<string, ContextSourceLineage> = {};

  for (const hit of input.retrievedMemory) {
    const metadata = getMergedMetadata(hit.metadata);
    const lineage = readSourceLineage(metadata);
    const reasons: string[] = [];
    let finalScore = clamp01(hit.score ?? 0);
    const explicitPin = readString(metadata, "identityPinId");
    const pinFromRaw = lineage.rawMemoryIds.map((rawId) => pinByRawId.get(rawId)).find(Boolean);
    const memoryClass = readString(metadata, "memoryClass");
    const mustAttend = metadata.mustAttend === true;
    const identityRisk = readNumber(metadata, "identityRisk") ?? 0;
    const pinStrength = Math.max(readNumber(metadata, "pinStrength") ?? 0, pinFromRaw?.pinStrength ?? 0);
    const protectedMemory = Boolean(explicitPin || pinFromRaw || mustAttend || memoryClass === "PINNED_EXACT");

    if (explicitPin || pinFromRaw) reasons.push("identity_pin");
    if (memoryClass === "PINNED_EXACT") reasons.push("pinned_exact");
    if (memoryClass === "HIGH_RISK_RAW") reasons.push("high_risk_raw");
    if (memoryClass === "LOW_RISK_REPRESENTATIVE" || lineage.representativeIds.length > 0) reasons.push("representative_lineage");
    if (mustAttend) reasons.push("must_attend");
    if (identityRisk > 0) {
      finalScore += memoryClass === "HIGH_RISK_RAW" ? identityRisk * 0.2 : identityRisk * 0.12;
      reasons.push("identity_risk");
    }
    if (pinStrength > 0) {
      finalScore += pinStrength * 0.35;
      reasons.push("pin_strength");
    }

    const priorIncludedCount = countPriorContextMatches(contextPackTraces, hit.id, lineage);
    if (priorIncludedCount > 0) {
      finalScore += Math.min(0.15, priorIncludedCount * 0.05);
      reasons.push("prior_context_inclusion");
    }

    if (lineage.rawMemoryIds.some((rawId) => failedAuditRawIds.has(rawId))) {
      finalScore += 0.35;
      reasons.push("retrieval_audit_failure_repair");
    } else if (lineage.rawMemoryIds.some((rawId) => preservedAuditRawIds.has(rawId))) {
      finalScore += 0.08;
      reasons.push("retrieval_audit_identity_preserved");
    }

    if (protectedMemory) finalScore = 1;
    finalScore = clamp01(finalScore);
    const identityPinIds = uniqueStable([
      ...lineage.identityPinIds,
      ...(explicitPin ? [explicitPin] : []),
      ...(pinFromRaw ? [pinFromRaw.id] : []),
    ]);
    const priority: ContextMemoryPriority = {
      ...lineage,
      identityPinIds,
      baseScore: roundScore(hit.score ?? 0),
      finalScore: roundScore(finalScore),
      protected: protectedMemory,
      reasons: uniqueStable(reasons),
    };
    priorityMap[hit.id] = priority;
    sourceLineageMap[hit.id] = {
      rawMemoryIds: priority.rawMemoryIds,
      representativeIds: priority.representativeIds,
      identityPinIds: priority.identityPinIds,
    };
  }

  const boostedMemoryIds = Object.entries(priorityMap)
    .filter(([, priority]) => priority.finalScore > priority.baseScore)
    .map(([id]) => id);
  const protectedMemoryIds = Object.entries(priorityMap)
    .filter(([, priority]) => priority.protected)
    .map(([id]) => id);

  return {
    memory: input.retrievedMemory.map((hit) => ({
      ...hit,
      score: priorityMap[hit.id]?.finalScore ?? hit.score,
    })),
    priorityMap,
    sourceLineageMap,
    learningSignals: {
      contextTraceCount: contextPackTraces.length,
      retrievalAuditCount: retrievalAudits.length,
      boostedMemoryIds,
      protectedMemoryIds,
    },
  };
}

function estimateTokens(text: string): number {
  return Math.ceil(text.length / 4);
}

function countPriorContextMatches(
  traces: ContextPackTraceRecord[],
  memoryId: string,
  lineage: ContextSourceLineage,
): number {
  let count = 0;
  for (const trace of traces) {
    const included = new Set(trace.includedMemoryIds);
    const rawIds = new Set(trace.rawMemoryIds);
    const representativeIds = new Set(trace.representativeIds);
    const identityPinIds = new Set(trace.identityPinIds);
    if (
      included.has(memoryId)
      || lineage.rawMemoryIds.some((id) => rawIds.has(id))
      || lineage.representativeIds.some((id) => representativeIds.has(id))
      || lineage.identityPinIds.some((id) => identityPinIds.has(id))
    ) count += 1;
  }
  return count;
}

function readSourceLineage(metadata: Record<string, unknown>): ContextSourceLineage {
  const rawMemoryIds = uniqueStable([
    ...readStringList(metadata, "rawMemoryIds"),
    ...readStringList(metadata, "lineageRawMemoryIds"),
    ...(readString(metadata, "rawMemoryId") ? [readString(metadata, "rawMemoryId")!] : []),
  ]);
  const representativeIds = uniqueStable([
    ...readStringList(metadata, "representativeIds"),
    ...(readString(metadata, "representativeId") ? [readString(metadata, "representativeId")!] : []),
  ]);
  const identityPinIds = uniqueStable([
    ...readStringList(metadata, "identityPinIds"),
    ...(readString(metadata, "identityPinId") ? [readString(metadata, "identityPinId")!] : []),
  ]);
  return { rawMemoryIds, representativeIds, identityPinIds };
}

function getMergedMetadata(metadata: Record<string, unknown>): Record<string, unknown> {
  const gac = metadata.gac;
  return typeof gac === "object" && gac !== null && !Array.isArray(gac) ? { ...metadata, ...gac } : metadata;
}

function readString(metadata: Record<string, unknown>, key: string): string | undefined {
  const value = metadata[key];
  return typeof value === "string" && value.length > 0 ? value : undefined;
}

function readNumber(metadata: Record<string, unknown>, key: string): number | undefined {
  const value = metadata[key];
  return typeof value === "number" && Number.isFinite(value) ? value : undefined;
}

function readStringList(metadata: Record<string, unknown>, key: string): string[] {
  const value = metadata[key];
  return Array.isArray(value) ? value.filter((item): item is string => typeof item === "string" && item.length > 0) : [];
}

function uniqueStable(values: string[]): string[] {
  return [...new Set(values)];
}

function clamp01(value: number): number {
  if (!Number.isFinite(value)) return 0;
  return Math.max(0, Math.min(1, value));
}

function roundScore(value: number): number {
  return Number(clamp01(value).toFixed(6));
}
