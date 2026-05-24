import type { GacRoutingBlockMetadata, MemorySearchHit } from "../types";
import type { SSAKernelTrace } from "./ssa_webgpu/types";

export type SSAMode = "disabled" | "fallback_sparse_planner" | "webgpu_reference" | "backend_native" | "hybrid";

export type SSATargetProfile =
  | "subq_compatible_public_ssa_path"
  | "public_ssa_reference"
  | "backend_native_vendor_specific";

export interface ContextBlock {
  id: string;
  text: string;
  tokenStart: number;
  tokenEnd: number;
  priority: number;
  source: string;
  tags?: string[];
  gac?: GacRoutingBlockMetadata;
}

export interface ContextAnchor {
  blockId: string;
  reason: string;
  score: number;
}

export interface SSASelectedAnchor extends ContextAnchor {
  source: "explicit" | "system_tag" | "current_user_tag" | "constraint_tag" | "gac";
  required: boolean;
}

export interface SSARouteTraceEntry {
  blockId: string;
  decision: "selected" | "pinned" | "dropped";
  score: number;
  reasons: string[];
}

export interface SSALayerRoutingPolicy {
  layerIndex: number;
  blockSize: number;
  topKBlocks: number;
  localWindowBlocks: number;
  pinnedBlockIds: string[];
  selectedBlockIdsByQueryBlock: Record<number, string[]>;
  denseFallback: boolean;
}

export interface SSAPlanInput {
  requestId: string;
  activeBlocks: ContextBlock[];
  anchors: ContextAnchor[];
  memoryHits: MemorySearchHit[];
  maxBlocks: number;
  minAnchorScore: number;
  blockSize?: number;
  topKBlocks?: number;
  localWindowBlocks?: number;
  targetProfile?: SSATargetProfile;
}

export interface SSAPlan {
  mode: SSAMode;
  targetProfile: SSATargetProfile;
  selectedBlockIds: string[];
  pinnedBlockIds: string[];
  droppedBlockIds: string[];
  routingReasons: Record<string, string[]>;
  layerPolicies: SSALayerRoutingPolicy[];
  estimatedDenseTokens: number;
  estimatedSparseTokens: number;
  sparsityRatio: number;
  routingTrace: SSARouteTraceEntry[];
  kernelTraces: SSAKernelTrace[];
}

export interface SparseForwardInput {
  requestId: string;
  layerIndex: number;
  qHandle: unknown;
  kHandle: unknown;
  vHandle: unknown;
  routingPolicy: SSALayerRoutingPolicy;
}

export interface SparseForwardOutput {
  requestId: string;
  layerIndex: number;
  outputHandle: unknown;
  selectedBlockIds: string[];
  timingMs?: Record<string, number>;
  trace?: SSAKernelTrace;
}

export interface SSAKernelBackend {
  supportsNativeSSA(): boolean;
  planSparseAttention(input: SSAPlanInput): Promise<SSAPlan>;
  executeSparseForward(input: SparseForwardInput): Promise<SparseForwardOutput>;
}

export interface SSARuntime {
  plan(input: SSAPlanInput): Promise<SSAPlan>;
}

export interface NativeSSAPlanningBackendContract extends SSAKernelBackend {
  readonly backendName: string;
  readonly supportsQkvAccess: boolean;
  readonly supportsLayerSparseRouting: boolean;
  readonly supportsPinnedKvBlocks: boolean;
  readonly supportsDenseReferenceMode: boolean;
}

const DEFAULT_TARGET_PROFILE: SSATargetProfile = "subq_compatible_public_ssa_path";
const DEFAULT_BLOCK_SIZE = 16;
const DEFAULT_TOP_K_BLOCKS = 16;
const DEFAULT_LOCAL_WINDOW_BLOCKS = 2;
const DEFAULT_LAYER_COUNT = 1;

export class FallbackSSARuntime implements SSARuntime {
  async plan(input: SSAPlanInput): Promise<SSAPlan> {
    return buildSparseSSAPlan(input);
  }
}

export function selectSSAAnchors(input: SSAPlanInput): SSASelectedAnchor[] {
  const blockById = new Map(input.activeBlocks.map((block) => [block.id, block]));
  const selected = new Map<string, SSASelectedAnchor>();

  for (const anchor of input.anchors) {
    if (anchor.score < input.minAnchorScore || !blockById.has(anchor.blockId)) continue;
    selected.set(anchor.blockId, { ...anchor, source: "explicit", required: true });
  }

  for (const block of input.activeBlocks) {
    const tagAnchor = tagAnchorForBlock(block);
    if (tagAnchor) {
      selected.set(block.id, {
        blockId: block.id,
        reason: tagAnchor.reason,
        score: Math.max(block.priority, 1),
        source: tagAnchor.source,
        required: true,
      });
    }

    if (block.gac?.mustAttend || block.gac?.memoryClass === "PINNED_EXACT") {
      selected.set(block.id, {
        blockId: block.id,
        reason: block.gac.mustAttend ? "gac_must_attend" : "gac_pinned_exact",
        score: Math.max(block.priority, block.gac.pinStrength ?? 1, 1),
        source: "gac",
        required: true,
      });
    }
  }

  return [...selected.values()].sort((a, b) => b.score - a.score || a.blockId.localeCompare(b.blockId));
}

export function buildSparseSSAPlan(input: SSAPlanInput): SSAPlan {
  const blockSize = input.blockSize ?? DEFAULT_BLOCK_SIZE;
  const topKBlocks = input.topKBlocks ?? DEFAULT_TOP_K_BLOCKS;
  const localWindowBlocks = input.localWindowBlocks ?? DEFAULT_LOCAL_WINDOW_BLOCKS;
  const targetProfile = input.targetProfile ?? DEFAULT_TARGET_PROFILE;
  const blocks = [...input.activeBlocks].sort((a, b) => a.tokenStart - b.tokenStart || a.id.localeCompare(b.id));
  const blockById = new Map(blocks.map((block) => [block.id, block]));
  const memoryScoreById = new Map(input.memoryHits.map((hit) => [hit.id, hit.score]));
  const reasons: Record<string, string[]> = {};
  const pinned = new Set<string>();
  const selected = new Set<string>();

  for (const anchor of selectSSAAnchors(input)) {
    if (!blockById.has(anchor.blockId)) continue;
    pinned.add(anchor.blockId);
    selected.add(anchor.blockId);
    addReason(reasons, anchor.blockId, `anchor:${anchor.reason}`);
    addReason(reasons, anchor.blockId, `anchor_source:${anchor.source}`);
  }

  for (const block of blocks) {
    addMetadataReasons(reasons, block);
    if (memoryScoreById.has(block.id)) addReason(reasons, block.id, "lancedb_memory_hit");
  }

  const scored = blocks
    .map((block) => ({ block, score: blockScore(block, memoryScoreById, pinned) }))
    .sort((a, b) => b.score - a.score || a.block.tokenStart - b.block.tokenStart || a.block.id.localeCompare(b.block.id));

  for (const { block } of scored) {
    if (selected.size >= input.maxBlocks && !pinned.has(block.id)) break;
    selected.add(block.id);
    addReason(reasons, block.id, memoryScoreById.has(block.id) ? "priority_budget:lancedb" : "priority_budget");
  }

  const queryBlockCount = Math.max(1, Math.ceil(sumTokens(blocks.map(tokenLength)) / blockSize));
  const selectedBlockIdsByQueryBlock: Record<number, string[]> = {};
  for (let queryBlockIndex = 0; queryBlockIndex < queryBlockCount; queryBlockIndex++) {
    selectedBlockIdsByQueryBlock[queryBlockIndex] = selectBlocksForQueryBlock({
      queryBlockIndex,
      blocks,
      selected,
      pinned,
      topKBlocks,
      localWindowBlocks,
      blockSize,
      memoryScoreById,
    });
  }

  const selectedBlockIds = blocks.map((block) => block.id).filter((id) => selected.has(id));
  const droppedBlockIds = blocks.map((block) => block.id).filter((id) => !selected.has(id));
  for (const id of droppedBlockIds) addReason(reasons, id, "dropped:not_selected_within_sparse_budget");

  const denseTokens = sumTokens(blocks.map(tokenLength));
  const sparseTokens = sumTokens(blocks.filter((block) => selected.has(block.id)).map(tokenLength));
  const layerPolicy: SSALayerRoutingPolicy = {
    layerIndex: -1,
    blockSize,
    topKBlocks,
    localWindowBlocks,
    pinnedBlockIds: [...pinned].sort(),
    selectedBlockIdsByQueryBlock,
    denseFallback: true,
  };

  return {
    mode: "fallback_sparse_planner",
    targetProfile,
    selectedBlockIds,
    pinnedBlockIds: [...pinned].sort(),
    droppedBlockIds,
    routingReasons: reasons,
    layerPolicies: [layerPolicy],
    estimatedDenseTokens: denseTokens,
    estimatedSparseTokens: sparseTokens,
    sparsityRatio: denseTokens === 0 ? 1 : sparseTokens / denseTokens,
    routingTrace: buildRoutingTrace(blocks, selected, pinned, reasons, memoryScoreById),
    kernelTraces: buildKernelTrace(input.requestId, layerPolicy, denseTokens, sparseTokens, DEFAULT_LAYER_COUNT),
  };
}

export class NativeSSABackendTestDouble implements NativeSSAPlanningBackendContract {
  readonly backendName = "native-ssa-test-double";
  readonly supportsQkvAccess = true;
  readonly supportsLayerSparseRouting = true;
  readonly supportsPinnedKvBlocks = true;
  readonly supportsDenseReferenceMode = true;

  private readonly fallback = new FallbackSSARuntime();

  supportsNativeSSA(): boolean {
    return true;
  }

  async planSparseAttention(input: SSAPlanInput): Promise<SSAPlan> {
    const plan = await this.fallback.plan(input);
    return {
      ...plan,
      mode: "backend_native",
      layerPolicies: plan.layerPolicies.map((policy) => ({ ...policy, denseFallback: false })),
    };
  }

  async executeSparseForward(input: SparseForwardInput): Promise<SparseForwardOutput> {
    const selectedBlockIds = input.routingPolicy.selectedBlockIdsByQueryBlock[0] ?? [];
    const trace: SSAKernelTrace = {
      requestId: input.requestId,
      layerIndex: input.layerIndex,
      queryBlockIndex: 0,
      selectedBlockIds,
      pinnedBlockIds: input.routingPolicy.pinnedBlockIds,
      denseTokenCountEstimate: 0,
      sparseTokenCountEstimate: 0,
      routingMs: 0,
      gatherMs: 0,
      attentionMs: 0,
    };
    return {
      requestId: input.requestId,
      layerIndex: input.layerIndex,
      outputHandle: {
        backend: this.backendName,
        qHandle: input.qHandle,
        kHandle: input.kHandle,
        vHandle: input.vHandle,
      },
      selectedBlockIds,
      timingMs: { routingMs: 0, gatherMs: 0, attentionMs: 0 },
      trace,
    };
  }
}

interface QueryBlockSelectionInput {
  queryBlockIndex: number;
  blocks: ContextBlock[];
  selected: Set<string>;
  pinned: Set<string>;
  topKBlocks: number;
  localWindowBlocks: number;
  blockSize: number;
  memoryScoreById: Map<string, number>;
}

function selectBlocksForQueryBlock(input: QueryBlockSelectionInput): string[] {
  const routeable = new Set<string>();
  const selectedBlocks = input.blocks.filter((block) => input.selected.has(block.id));
  for (const id of input.pinned) routeable.add(id);

  const queryTokenStart = input.queryBlockIndex * input.blockSize;
  const queryTokenEnd = queryTokenStart + input.blockSize;
  const foundIndex = input.blocks.findIndex((block) => block.tokenEnd > queryTokenStart && block.tokenStart < queryTokenEnd);
  const querySourceIndex = Math.max(0, foundIndex);
  for (let offset = -input.localWindowBlocks; offset <= input.localWindowBlocks; offset++) {
    const local = input.blocks[querySourceIndex + offset];
    if (local && input.selected.has(local.id)) routeable.add(local.id);
  }

  const remainingBudget = Math.max(0, input.topKBlocks + input.pinned.size - routeable.size);
  const ranked = selectedBlocks
    .filter((block) => !routeable.has(block.id))
    .map((block) => ({ block, score: blockScore(block, input.memoryScoreById, input.pinned) }))
    .sort((a, b) => b.score - a.score || a.block.tokenStart - b.block.tokenStart || a.block.id.localeCompare(b.block.id))
    .slice(0, remainingBudget);

  for (const { block } of ranked) routeable.add(block.id);
  return input.blocks.map((block) => block.id).filter((id) => routeable.has(id));
}

function buildRoutingTrace(
  blocks: ContextBlock[],
  selected: Set<string>,
  pinned: Set<string>,
  reasons: Record<string, string[]>,
  memoryScoreById: Map<string, number>,
): SSARouteTraceEntry[] {
  return blocks.map((block) => ({
    blockId: block.id,
    decision: pinned.has(block.id) ? "pinned" : selected.has(block.id) ? "selected" : "dropped",
    score: Number(blockScore(block, memoryScoreById, pinned).toFixed(6)),
    reasons: reasons[block.id] ?? ["unclassified"],
  }));
}

function buildKernelTrace(
  requestId: string,
  policy: SSALayerRoutingPolicy,
  denseTokens: number,
  sparseTokens: number,
  layers: number,
): SSAKernelTrace[] {
  const traces: SSAKernelTrace[] = [];
  for (let layer = 0; layer < layers; layer++) {
    for (const [queryBlockIndex, selectedBlockIds] of Object.entries(policy.selectedBlockIdsByQueryBlock)) {
      traces.push({
        requestId,
        layerIndex: policy.layerIndex === -1 ? layer : policy.layerIndex,
        queryBlockIndex: Number(queryBlockIndex),
        selectedBlockIds,
        pinnedBlockIds: policy.pinnedBlockIds,
        denseTokenCountEstimate: denseTokens,
        sparseTokenCountEstimate: sparseTokens,
        routingMs: 0,
        gatherMs: 0,
        attentionMs: 0,
      });
    }
  }
  return traces;
}

function tagAnchorForBlock(block: ContextBlock): Pick<SSASelectedAnchor, "source" | "reason"> | undefined {
  if (block.tags?.includes("system")) return { source: "system_tag", reason: "system_prompt" };
  if (block.tags?.includes("safety")) return { source: "system_tag", reason: "safety_policy" };
  if (block.tags?.includes("current_user_request")) return { source: "current_user_tag", reason: "current_user_request" };
  if (block.tags?.includes("constraint")) return { source: "constraint_tag", reason: "explicit_constraint" };
  return undefined;
}

function addMetadataReasons(reasons: Record<string, string[]>, block: ContextBlock): void {
  if (block.gac?.mustAttend) addReason(reasons, block.id, "gac_must_attend");
  if (block.gac?.memoryClass) addReason(reasons, block.id, `gac_memory_class:${block.gac.memoryClass}`);
  if (typeof block.gac?.identityRisk === "number") addReason(reasons, block.id, "gac_identity_risk");
  if (typeof block.gac?.pinStrength === "number") addReason(reasons, block.id, "gac_pin_strength");
}

function blockScore(block: ContextBlock, memoryScoreById: Map<string, number>, pinned: Set<string>): number {
  let score = block.priority;
  score += memoryScoreById.get(block.id) ?? 0;
  if (pinned.has(block.id)) score += 10;
  if (block.gac?.mustAttend) score += 10;
  if (block.gac?.memoryClass === "PINNED_EXACT") score += 8;
  if (block.gac?.memoryClass === "HIGH_RISK_RAW") score += 3;
  if (block.gac?.memoryClass === "SOURCE_EVIDENCE") score += 2;
  if (typeof block.gac?.sourceTrust === "number") score += block.gac.sourceTrust;
  if (typeof block.gac?.identityRisk === "number") score += block.gac.identityRisk;
  if (typeof block.gac?.pinStrength === "number") score += block.gac.pinStrength;
  if (block.tags?.includes("system")) score += 5;
  if (block.tags?.includes("safety")) score += 5;
  if (block.tags?.includes("current_user_request")) score += 4;
  if (block.tags?.includes("constraint")) score += 4;
  return score;
}

function tokenLength(block: ContextBlock): number {
  return Math.max(0, block.tokenEnd - block.tokenStart);
}

function addReason(reasons: Record<string, string[]>, blockId: string, reason: string): void {
  const current = reasons[blockId] ?? [];
  if (!current.includes(reason)) current.push(reason);
  reasons[blockId] = current;
}

function sumTokens(values: number[]): number {
  return values.reduce((sum, value) => sum + Math.max(0, value), 0);
}

class LegacyFallbackSSARuntime {
  async plan(input: SSAPlanInput): Promise<SSAPlan> {
    const reasons: Record<string, string[]> = {};
    const selected = new Set<string>();
    const pinned = new Set<string>();
    const blockById = new Map(input.activeBlocks.map((block) => [block.id, block]));

    for (const anchor of input.anchors) {
      if (anchor.score >= input.minAnchorScore && blockById.has(anchor.blockId)) {
        selected.add(anchor.blockId);
        pinned.add(anchor.blockId);
        addReason(reasons, anchor.blockId, `anchor:${anchor.reason}`);
      }
    }

    for (const block of input.activeBlocks) {
      if (block.gac?.mustAttend || block.gac?.memoryClass === "PINNED_EXACT") {
        selected.add(block.id);
        pinned.add(block.id);
        addReason(reasons, block.id, block.gac.mustAttend ? "gac_must_attend" : "gac_pinned_exact");
      }
      if (block.gac?.memoryClass) {
        addReason(reasons, block.id, `gac_memory_class:${block.gac.memoryClass}`);
      }
    }

    const memoryIds = new Set(input.memoryHits.map((hit) => hit.id));
    const scored = [...input.activeBlocks].sort((a, b) => {
      const scoreA = legacyBlockScore(a, memoryIds, pinned);
      const scoreB = legacyBlockScore(b, memoryIds, pinned);
      return scoreB - scoreA;
    });

    for (const block of scored) {
      if (selected.size >= input.maxBlocks) break;
      selected.add(block.id);
      addReason(reasons, block.id, memoryIds.has(block.id) ? "lancedb_memory_hit" : "priority_budget");
    }

    const selectedBlockIds = [...selected];
    const droppedBlockIds = input.activeBlocks.map((b) => b.id).filter((id) => !selected.has(id));
    for (const id of droppedBlockIds) {
      addReason(reasons, id, "dropped:not_selected_within_sparse_budget");
    }

    const denseTokens = sumTokens(input.activeBlocks.map(tokenLength));
    const sparseTokens = sumTokens(input.activeBlocks.filter((b) => selected.has(b.id)).map(tokenLength));
    const blockSize = input.blockSize ?? DEFAULT_BLOCK_SIZE;
    const topKBlocks = input.topKBlocks ?? DEFAULT_TOP_K_BLOCKS;
    const localWindowBlocks = input.localWindowBlocks ?? DEFAULT_LOCAL_WINDOW_BLOCKS;

    return {
      mode: "fallback_sparse_planner",
      targetProfile: input.targetProfile ?? DEFAULT_TARGET_PROFILE,
      selectedBlockIds,
      pinnedBlockIds: [...pinned],
      droppedBlockIds,
      routingReasons: reasons,
      layerPolicies: [
        {
          layerIndex: -1,
          blockSize,
          topKBlocks,
          localWindowBlocks,
          pinnedBlockIds: [...pinned],
          selectedBlockIdsByQueryBlock: { 0: selectedBlockIds },
          denseFallback: true,
        },
      ],
      estimatedDenseTokens: denseTokens,
      estimatedSparseTokens: sparseTokens,
      sparsityRatio: denseTokens === 0 ? 1 : sparseTokens / denseTokens,
      routingTrace: buildRoutingTrace(input.activeBlocks, selected, pinned, reasons, new Map()),
      kernelTraces: [],
    };
  }
}

function legacyBlockScore(block: ContextBlock, memoryIds: Set<string>, pinned: Set<string>): number {
  let score = block.priority;
  if (memoryIds.has(block.id)) score += 1;
  if (pinned.has(block.id)) score += 10;
  if (block.gac?.mustAttend) score += 10;
  if (block.gac?.memoryClass === "PINNED_EXACT") score += 8;
  if (block.gac?.memoryClass === "HIGH_RISK_RAW") score += 3;
  if (typeof block.gac?.identityRisk === "number") score += block.gac.identityRisk;
  if (typeof block.gac?.pinStrength === "number") score += block.gac.pinStrength;
  if (block.tags?.includes("system")) score += 5;
  if (block.tags?.includes("current_user_request")) score += 4;
  return score;
}
