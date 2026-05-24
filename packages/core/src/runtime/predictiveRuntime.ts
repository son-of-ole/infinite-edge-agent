import type { GacRoutingBlockMetadata } from "../types";
import type { KVBlock } from "./kvswap";
import type { ContextBlock, SSAPlan } from "./ssa";
import type { SpeculativeDecodingConfig } from "./speculative";

export interface PredictiveRuntimePlanInput {
  requestId: string;
  userMessage: string;
  activeBlocks: ContextBlock[];
  ssaPlan: SSAPlan;
  kvBlocks: KVBlock[];
  speculativeConfig: SpeculativeDecodingConfig;
  tokenBudget: number;
  vramBudgetBytes?: number;
  ramBudgetBytes?: number;
}

export interface PredictedRetrieval {
  query: string;
  targetMemoryClass?: GacRoutingBlockMetadata["memoryClass"];
  expectedRawMemoryIds?: string[];
  expectedRepresentativeIds?: string[];
  priority: number;
}

export interface PredictedContextBranch {
  branchId: string;
  trigger: "tool_result" | "mtp_accept" | "mtp_reject" | "memory_miss" | "user_followup";
  seedBlockIds: string[];
  tokenBudget: number;
  priority: number;
}

export interface PredictedKVHotPage {
  blockId: string;
  tier: "PIN_HOT" | "TASK_HOT" | "SESSION_WARM" | "BACKGROUND_WARM" | "COLD";
  source: "gac" | "ssa" | "mtp" | "recent" | "wake";
  priority: number;
}

export interface PredictedSparseBlock {
  blockId: string;
  source: "ssa" | "gac";
  priority: number;
}

export interface PredictedMtpBranch {
  branchId: string;
  draftModelId: string | null;
  maxDraftTokens: number;
  dependsOnBlockIds: string[];
  expectedAcceptanceRate?: number;
}

export interface PredictiveCacheBudget {
  vramTargetBytes: number;
  ramTargetBytes: number;
  prefetchBlockIds: string[];
  evictableBlockIds: string[];
}

export interface PredictiveRuntimePlan {
  planId: string;
  requestId: string;
  predictedRetrievals: PredictedRetrieval[];
  contextBranches: PredictedContextBranch[];
  kvHotPages: PredictedKVHotPage[];
  sparseBlocks: PredictedSparseBlock[];
  mtpBranches: PredictedMtpBranch[];
  cacheBudget: PredictiveCacheBudget;
  confidence: number;
  reasons: string[];
}

export function buildPredictiveRuntimePlan(input: PredictiveRuntimePlanInput): PredictiveRuntimePlan {
  const blockById = new Map(input.activeBlocks.map((block) => [block.id, block]));
  const selected = new Set(input.ssaPlan.selectedBlockIds);
  const pinned = new Set(input.ssaPlan.pinnedBlockIds);
  const kvHotPages = dedupeByBlockId(input.kvBlocks
    .filter((block) => block.pinned || selected.has(block.sourceBlockId ?? "") || pinned.has(block.sourceBlockId ?? "") || Boolean(block.gacPriority))
    .map((block) => toPredictedHotPage(block, blockById.get(block.sourceBlockId ?? ""))));
  const sparseBlocks = input.ssaPlan.selectedBlockIds.map((blockId) => {
    const block = blockById.get(blockId);
    return {
      blockId,
      source: block?.gac ? "gac" as const : "ssa" as const,
      priority: clamp01(block?.priority ?? 0.5),
    };
  });
  const mtpBranches = input.speculativeConfig.enabled && input.speculativeConfig.mode !== "target_only" && input.speculativeConfig.numSpeculativeTokens > 0
    ? [{
        branchId: `mtp_${input.requestId}_accept`,
        draftModelId: input.speculativeConfig.draftModelId ?? null,
        maxDraftTokens: input.speculativeConfig.numSpeculativeTokens,
        dependsOnBlockIds: kvHotPages.map((page) => page.blockId),
        expectedAcceptanceRate: input.speculativeConfig.minAcceptanceRate,
      }]
    : [];

  return {
    planId: `pred_${input.requestId}`,
    requestId: input.requestId,
    predictedRetrievals: buildPredictedRetrievals(input.activeBlocks, input.userMessage),
    contextBranches: buildContextBranches(input, sparseBlocks.map((block) => block.blockId), mtpBranches.length > 0),
    kvHotPages,
    sparseBlocks,
    mtpBranches,
    cacheBudget: {
      vramTargetBytes: Math.floor((input.vramBudgetBytes ?? 0) * 0.82),
      ramTargetBytes: Math.floor((input.ramBudgetBytes ?? 0) * 0.85),
      prefetchBlockIds: kvHotPages.map((page) => page.blockId),
      evictableBlockIds: input.kvBlocks
        .filter((block) => !block.pinned && !kvHotPages.some((page) => page.blockId === block.id))
        .map((block) => block.id),
    },
    confidence: calculateConfidence(kvHotPages.length, sparseBlocks.length, mtpBranches.length),
    reasons: [
      "ssa_selected_blocks",
      ...(kvHotPages.some((page) => page.source === "gac") ? ["gac_priority"] : []),
      ...(mtpBranches.length > 0 ? ["mtp_branch_pressure"] : []),
    ],
  };
}

function buildPredictedRetrievals(blocks: ContextBlock[], userMessage: string): PredictedRetrieval[] {
  return blocks.flatMap((block) => {
    if (!block.gac) return [];
    const rawIds = block.gac.rawMemoryId ? [block.gac.rawMemoryId] : [];
    const representativeIds = block.gac.representativeId ? [block.gac.representativeId] : [];
    if (rawIds.length === 0 && representativeIds.length === 0 && !block.gac.mustAttend) return [];
    return [{
      query: [
        userMessage,
        block.gac.rawMemoryId ? `raw:${block.gac.rawMemoryId}` : "",
        block.gac.representativeId ? `representative:${block.gac.representativeId}` : "",
      ].filter(Boolean).join(" "),
      targetMemoryClass: block.gac.memoryClass,
      ...(rawIds.length > 0 ? { expectedRawMemoryIds: rawIds } : {}),
      ...(representativeIds.length > 0 ? { expectedRepresentativeIds: representativeIds } : {}),
      priority: clamp01(block.priority + (block.gac.pinStrength ?? 0) * 0.25 + (block.gac.identityRisk ?? 0) * 0.2),
    }];
  }).sort((left, right) => right.priority - left.priority);
}

function buildContextBranches(
  input: PredictiveRuntimePlanInput,
  seedBlockIds: string[],
  mtpEnabled: boolean,
): PredictedContextBranch[] {
  const budget = Math.max(1, Math.floor(input.tokenBudget * 0.25));
  return [
    {
      branchId: `branch_${input.requestId}_followup`,
      trigger: "user_followup",
      seedBlockIds,
      tokenBudget: budget,
      priority: 0.6,
    },
    ...(mtpEnabled
      ? [
          {
            branchId: `branch_${input.requestId}_mtp_reject`,
            trigger: "mtp_reject" as const,
            seedBlockIds,
            tokenBudget: budget,
            priority: 0.5,
          },
        ]
      : []),
  ];
}

function toPredictedHotPage(block: KVBlock, sourceBlock: ContextBlock | undefined): PredictedKVHotPage {
  const gacTier = block.gacPriority?.tier ?? (sourceBlock?.gac ? tierFromGac(sourceBlock.gac) : undefined);
  return {
    blockId: block.id,
    tier: gacTier ?? (block.pinned ? "PIN_HOT" : "SESSION_WARM"),
    source: sourceFor(block, sourceBlock),
    priority: clamp01(Math.max(block.importance, block.gacPriority?.priorityScore ?? 0, sourceBlock?.priority ?? 0)),
  };
}

function sourceFor(block: KVBlock, sourceBlock: ContextBlock | undefined): PredictedKVHotPage["source"] {
  if (block.gacPriority?.tier === "PIN_HOT" || sourceBlock?.gac?.mustAttend || sourceBlock?.gac?.memoryClass === "PINNED_EXACT") return "gac";
  if (block.sourceBlockId) return "ssa";
  return "recent";
}

function tierFromGac(gac: GacRoutingBlockMetadata): PredictedKVHotPage["tier"] {
  if (gac.memoryClass === "PINNED_EXACT" || gac.mustAttend) return "PIN_HOT";
  if (gac.memoryClass === "HIGH_RISK_RAW") return "TASK_HOT";
  if (gac.memoryClass === "LOW_RISK_REPRESENTATIVE" || gac.memoryClass === "BACKGROUND_SUMMARY") return "BACKGROUND_WARM";
  if (gac.memoryClass === "RECENT_SESSION" || gac.memoryClass === "TASK_STATE") return "SESSION_WARM";
  return "COLD";
}

function dedupeByBlockId(pages: PredictedKVHotPage[]): PredictedKVHotPage[] {
  const byId = new Map<string, PredictedKVHotPage>();
  for (const page of pages) {
    const existing = byId.get(page.blockId);
    if (!existing || page.priority > existing.priority) byId.set(page.blockId, page);
  }
  return [...byId.values()].sort((left, right) => right.priority - left.priority || left.blockId.localeCompare(right.blockId));
}

function calculateConfidence(kvHotPages: number, sparseBlocks: number, mtpBranches: number): number {
  return clamp01(0.35 + Math.min(0.3, kvHotPages * 0.05) + Math.min(0.25, sparseBlocks * 0.03) + (mtpBranches > 0 ? 0.1 : 0));
}

function clamp01(value: number): number {
  if (!Number.isFinite(value)) return 0;
  return Math.max(0, Math.min(1, value));
}
