import type { GacKvSwapPriorityMetadata } from "../types";

export type KVTier = "vram" | "ram" | "disk";
export type KVSwapMode = "metadata_only" | "ram_tier" | "disk_tier" | "predictive";

export interface BackendTensorHandle {
  backend: string;
  id: string;
  dtype?: "f16" | "f32" | "q8" | "unknown";
  shape?: readonly number[];
  bytes?: number;
}

export interface KVBackendTensorHandles {
  key?: BackendTensorHandle;
  value?: BackendTensorHandle;
}

export interface KVLowRankKeySummary {
  blockId: string;
  rank: number;
  projectionId: string;
  layer: number;
  headGroupId: string;
  checksum: string;
  qualityScore: number;
  values: Float32Array;
}

export interface KVLowRankKeySummaryInput {
  blockId: string;
  projectionId: string;
  layer: number;
  headGroupId: string;
  values: readonly number[] | Float32Array;
  rank?: number;
  checksum?: string;
  qualityScore?: number;
}

export interface KVLowRankQuerySummary {
  rank: number;
  projectionId: string;
  values: readonly number[] | Float32Array;
  layer?: number;
  headGroupId?: string;
}

export interface KVLowRankAttentionScore {
  blockId: string;
  score: number;
  dot: number;
  normProduct: number;
  rank: number;
  projectionId: string;
  layer: number;
  headGroupId: string;
  checksum: string;
  qualityScore: number;
}

export interface KVLowRankKeySummaryRowsInput {
  blockId: string;
  projectionId: string;
  layer: number;
  headGroupId: string;
  rows: readonly (readonly number[])[];
  rank?: number;
  checksum?: string;
  qualityScore?: number;
}

export type KVSwapPrefetchStrategy = "none" | "exact_reuse" | "predictive_prefetch" | "miss_stall";

export interface KVSwapPredictedHotBlock {
  blockId: string;
  score: number;
  rank: number;
  projectionId: string;
  layer: number;
  headGroupId: string;
  checksum: string;
  qualityScore: number;
  tier: KVTier;
  estimatedBytes: number;
  source: "low_rank_attention";
}

export interface KVSwapPredictivePrefetchOptions {
  querySummary?: KVLowRankQuerySummary;
  maxBlocks?: number;
  minScore?: number;
  actualAttentionBlockIds?: readonly string[];
  estimatedLatencyMsByTier?: Partial<Record<KVTier, number>>;
}

export interface KVSwapPredictivePrefetchPlan {
  lowRankSummaryRank?: number;
  predictedHotBlocks: KVSwapPredictedHotBlock[];
  prefetchedBlocks: string[];
  prefetchHitRate: number;
  prefetchBytes: number;
  prefetchLatencyMs: number;
  attentionStallMs: number;
  prefetchStrategy: KVSwapPrefetchStrategy;
}

export interface KVBlock {
  id: string;
  layer: number;
  startToken: number;
  endToken: number;
  tier: KVTier;
  pinned: boolean;
  importance: number;
  lastAccessAt: number;
  sourceBlockId?: string;
  estimatedBytes: number;
  checksum?: string;
  compressedKeySummary?: Float32Array | string;
  summaryRank?: number;
  lowRankKeySummary?: KVLowRankKeySummary;
  tensorHandles?: KVBackendTensorHandles;
  gacPriority?: GacKvSwapPriorityMetadata | undefined;
}

export interface KVSwapPolicy {
  mode: KVSwapMode;
  vramPressureThreshold: number;
  ramPressureThreshold: number;
  now: number;
  vramBudgetBytes?: number;
  ramBudgetBytes?: number;
}

export interface KVSwapPressureTelemetry {
  mode: KVSwapMode;
  vramBytes: number;
  ramBytes: number;
  diskBytes: number;
  totalBytes: number;
  vramPressureRatio: number;
  ramPressureRatio: number;
  targetFreeBytes: number;
  projectedVramBytesAfterEviction: number;
  backendTensorHandleCount: number;
}

export interface KVSwapDecision {
  pinBlockIds: string[];
  evictBlockIds: string[];
  prefetchBlockIds: string[];
  predictivePrefetchBlockIds: string[];
  predictionReasons: Record<string, string[]>;
  prefetchConfidenceByBlockId: Record<string, number>;
  lowRankSummaryRank?: number;
  predictedHotBlocks?: KVSwapPredictedHotBlock[];
  prefetchedBlocks?: string[];
  prefetchHitRate?: number;
  prefetchBytes?: number;
  prefetchLatencyMs?: number;
  attentionStallMs?: number;
  prefetchStrategy?: KVSwapPrefetchStrategy;
  reasons: Record<string, string[]>;
  estimatedBytesFreed: number;
  pressureTelemetry: KVSwapPressureTelemetry;
}

export interface KVSwapPredictionHint {
  blockId: string;
  confidence: number;
  reasons: string[];
}

const MAX_LOW_RANK_DIMENSIONS = 64;
const DEFAULT_LOW_RANK_PREFETCH_BLOCKS = 4;
const DEFAULT_LOW_RANK_MIN_SCORE = 0.35;
const DEFAULT_PREFETCH_LATENCY_MS_BY_TIER: Record<KVTier, number> = {
  vram: 0,
  ram: 2,
  disk: 12,
};

export function createLowRankKeySummary(input: KVLowRankKeySummaryInput): KVLowRankKeySummary {
  const rank = normalizeRank(input.rank ?? input.values.length);
  const values = normalizeSummaryVector(input.values, rank);
  return {
    blockId: input.blockId,
    rank: values.length,
    projectionId: input.projectionId,
    layer: Math.max(0, Math.floor(input.layer)),
    headGroupId: input.headGroupId.trim() || "all_heads",
    checksum: input.checksum?.trim() || checksumLowRankSummary(input.blockId, input.projectionId, values),
    qualityScore: clamp01(input.qualityScore ?? 1),
    values,
  };
}

export function compressKeyRowsToLowRankSummary(input: KVLowRankKeySummaryRowsInput): KVLowRankKeySummary {
  const widestRow = input.rows.reduce((width, row) => Math.max(width, row.length), 0);
  const mean = new Array(widestRow).fill(0);
  const counts = new Array(widestRow).fill(0);
  for (const row of input.rows) {
    for (let index = 0; index < row.length; index += 1) {
      const value = boundedFinite(row[index] ?? 0);
      mean[index] += value;
      counts[index] += 1;
    }
  }
  for (let index = 0; index < mean.length; index += 1) {
    mean[index] = counts[index] > 0 ? mean[index] / counts[index] : 0;
  }
  const projected = projectVectorToRank(mean, normalizeRank(input.rank ?? Math.min(widestRow, 8)));
  const finiteCoverage = widestRow > 0 && input.rows.length > 0
    ? counts.filter((count) => count > 0).length / widestRow
    : 0;
  return createLowRankKeySummary({
    blockId: input.blockId,
    projectionId: input.projectionId,
    layer: input.layer,
    headGroupId: input.headGroupId,
    values: projected,
    ...(input.checksum !== undefined ? { checksum: input.checksum } : {}),
    qualityScore: input.qualityScore ?? finiteCoverage,
  });
}

export function scoreLowRankAttention(
  summary: KVLowRankKeySummary,
  query: KVLowRankQuerySummary,
): KVLowRankAttentionScore {
  const rank = Math.min(
    MAX_LOW_RANK_DIMENSIONS,
    normalizeRank(summary.rank),
    normalizeRank(query.rank),
    summary.values.length,
    query.values.length,
  );
  const keyValues = normalizeSummaryVector(summary.values, rank);
  const queryValues = normalizeSummaryVector(query.values, rank);
  let dot = 0;
  let keyNormSq = 0;
  let queryNormSq = 0;
  for (let index = 0; index < rank; index += 1) {
    const key = keyValues[index] ?? 0;
    const q = queryValues[index] ?? 0;
    dot += key * q;
    keyNormSq += key * key;
    queryNormSq += q * q;
  }
  const normProduct = Math.sqrt(keyNormSq) * Math.sqrt(queryNormSq);
  const cosine = normProduct > 0 ? dot / normProduct : -1;
  const projectionWeight = summary.projectionId === query.projectionId ? 1 : 0.75;
  const layerWeight = query.layer === undefined || query.layer === summary.layer ? 1 : 0.85;
  const headWeight = query.headGroupId === undefined || query.headGroupId === summary.headGroupId ? 1 : 0.9;
  const coverage = rank / Math.max(1, Math.max(summary.rank, query.rank));
  const score = clamp01(((cosine + 1) / 2) * summary.qualityScore * projectionWeight * layerWeight * headWeight * coverage);
  return {
    blockId: summary.blockId,
    score: round6(score),
    dot: round6(dot),
    normProduct: round6(normProduct),
    rank,
    projectionId: summary.projectionId,
    layer: summary.layer,
    headGroupId: summary.headGroupId,
    checksum: summary.checksum,
    qualityScore: summary.qualityScore,
  };
}

export function planPredictiveKVSwapPrefetch(
  blocks: KVBlock[],
  options: KVSwapPredictivePrefetchOptions = {},
): KVSwapPredictivePrefetchPlan {
  const querySummary = options.querySummary;
  if (!querySummary) {
    return {
      predictedHotBlocks: [],
      prefetchedBlocks: [],
      prefetchHitRate: 0,
      prefetchBytes: 0,
      prefetchLatencyMs: 0,
      attentionStallMs: 0,
      prefetchStrategy: "none",
    };
  }

  const maxBlocks = Math.max(1, Math.min(blocks.length || 1, Math.floor(options.maxBlocks ?? DEFAULT_LOW_RANK_PREFETCH_BLOCKS)));
  const minScore = clamp01(options.minScore ?? DEFAULT_LOW_RANK_MIN_SCORE);
  const scored = blocks.flatMap((block): KVSwapPredictedHotBlock[] => {
    const summary = getBlockLowRankSummary(block);
    if (!summary) return [];
    const score = scoreLowRankAttention(summary, querySummary);
    if (score.score < minScore) return [];
    return [{
      blockId: block.id,
      score: score.score,
      rank: score.rank,
      projectionId: score.projectionId,
      layer: score.layer,
      headGroupId: score.headGroupId,
      checksum: score.checksum,
      qualityScore: score.qualityScore,
      tier: block.tier,
      estimatedBytes: Math.max(0, block.estimatedBytes),
      source: "low_rank_attention",
    }];
  }).sort((left, right) => (
    right.score - left.score
    || right.qualityScore - left.qualityScore
    || left.blockId.localeCompare(right.blockId)
  ));
  const predictedHotBlocks = scored.slice(0, maxBlocks);
  const predictedHotIds = new Set(predictedHotBlocks.map((block) => block.blockId));
  const prefetchedBlocks = predictedHotBlocks
    .filter((block) => block.tier !== "vram")
    .map((block) => block.blockId);
  const prefetched = new Set(prefetchedBlocks);
  const prefetchBytes = predictedHotBlocks
    .filter((block) => prefetched.has(block.blockId))
    .reduce((sum, block) => sum + block.estimatedBytes, 0);
  const latencyByTier = {
    ...DEFAULT_PREFETCH_LATENCY_MS_BY_TIER,
    ...(options.estimatedLatencyMsByTier ?? {}),
  };
  const prefetchLatencyMs = predictedHotBlocks
    .filter((block) => prefetched.has(block.blockId))
    .reduce((latency, block) => Math.max(latency, Math.max(0, latencyByTier[block.tier] ?? 0)), 0);
  const actualAttentionBlockIds = options.actualAttentionBlockIds ?? [];
  const prefetchHitRate = actualAttentionBlockIds.length > 0
    ? actualAttentionBlockIds.filter((id) => prefetched.has(id) || predictedHotIds.has(id)).length / actualAttentionBlockIds.length
    : (predictedHotBlocks.length > 0 ? prefetchedBlocks.length / predictedHotBlocks.length : 0);
  const attentionStallMs = actualAttentionBlockIds.length > 0
    ? estimateAttentionStallMs(blocks, actualAttentionBlockIds, prefetched, latencyByTier)
    : 0;

  return {
    lowRankSummaryRank: normalizeRank(querySummary.rank),
    predictedHotBlocks,
    prefetchedBlocks,
    prefetchHitRate: round6(prefetchHitRate),
    prefetchBytes,
    prefetchLatencyMs: round6(prefetchLatencyMs),
    attentionStallMs: round6(attentionStallMs),
    prefetchStrategy: predictedHotBlocks.length === 0
      ? "miss_stall"
      : (prefetchedBlocks.length > 0 ? "predictive_prefetch" : "none"),
  };
}

export function buildKVSwapPressureTelemetry(
  blocks: KVBlock[],
  policy: KVSwapPolicy,
  targetFreeBytes: number,
  estimatedBytesFreed = 0,
): KVSwapPressureTelemetry {
  const vramBytes = sumBytes(blocks.filter((block) => block.tier === "vram"));
  const ramBytes = sumBytes(blocks.filter((block) => block.tier === "ram"));
  const diskBytes = sumBytes(blocks.filter((block) => block.tier === "disk"));
  const totalBytes = vramBytes + ramBytes + diskBytes;
  return {
    mode: policy.mode,
    vramBytes,
    ramBytes,
    diskBytes,
    totalBytes,
    vramPressureRatio: policy.vramBudgetBytes ? vramBytes / policy.vramBudgetBytes : 0,
    ramPressureRatio: policy.ramBudgetBytes ? ramBytes / policy.ramBudgetBytes : 0,
    targetFreeBytes,
    projectedVramBytesAfterEviction: Math.max(0, vramBytes - estimatedBytesFreed),
    backendTensorHandleCount: blocks.filter((block) => Boolean(block.tensorHandles?.key || block.tensorHandles?.value)).length,
  };
}

export function planKVSwap(
  blocks: KVBlock[],
  policy: KVSwapPolicy,
  targetFreeBytes: number,
  predictedNeededBlockIds: string[] = [],
  predictionHints: KVSwapPredictionHint[] = [],
  predictiveOptions: KVSwapPredictivePrefetchOptions = {},
): KVSwapDecision {
  const needed = new Set(predictedNeededBlockIds);
  const hintsByBlockId = new Map(predictionHints.map((hint) => [hint.blockId, hint]));
  const reasons: Record<string, string[]> = {};
  const pinBlockIds = blocks.filter(shouldPinBlock).map((block) => block.id);
  const lowRankPrefetch = planPredictiveKVSwapPrefetch(blocks, predictiveOptions);
  const lowRankPrefetchIds = new Set(lowRankPrefetch.prefetchedBlocks);
  const prefetchBlockIds = uniqueStrings([
    ...blocks.filter((block) => needed.has(block.id) && block.tier !== "vram").map((block) => block.id),
    ...lowRankPrefetch.prefetchedBlocks,
  ]);
  const predictionReasons: Record<string, string[]> = {};
  const prefetchConfidenceByBlockId: Record<string, number> = {};

  const candidates = blocks
    .filter((block) => !shouldPinBlock(block) && block.tier === "vram" && !needed.has(block.id) && !lowRankPrefetchIds.has(block.id))
    .sort((a, b) => evictionScore(a, policy.now) - evictionScore(b, policy.now));

  const evictBlockIds: string[] = [];
  let estimatedBytesFreed = 0;

  for (const candidate of candidates) {
    if (estimatedBytesFreed >= targetFreeBytes) break;
    evictBlockIds.push(candidate.id);
    estimatedBytesFreed += candidate.estimatedBytes;
    reasons[candidate.id] = ["not_pinned", "low_importance_or_old", `tier:${candidate.tier}`];
  }

  for (const id of pinBlockIds) reasons[id] = [...(reasons[id] ?? []), "pinned"];
  for (const id of predictedNeededBlockIds) {
    const hint = hintsByBlockId.get(id);
    const hintReasons = hint?.reasons.length ? hint.reasons : ["ssa_selected"];
    predictionReasons[id] = hintReasons;
    prefetchConfidenceByBlockId[id] = clamp01(hint?.confidence ?? 1);
    reasons[id] = [
      ...(reasons[id] ?? []),
      "predicted_needed",
      ...hintReasons.map((reason) => `prediction:${reason}`),
    ];
  }
  for (const hotBlock of lowRankPrefetch.predictedHotBlocks) {
    predictionReasons[hotBlock.blockId] = [
      ...(predictionReasons[hotBlock.blockId] ?? []),
      "low_rank_attention",
      "approx_attention_score",
    ];
    prefetchConfidenceByBlockId[hotBlock.blockId] = Math.max(
      prefetchConfidenceByBlockId[hotBlock.blockId] ?? 0,
      hotBlock.score,
    );
    reasons[hotBlock.blockId] = [
      ...(reasons[hotBlock.blockId] ?? []),
      "predicted_needed",
      "prediction:low_rank_attention",
      "prediction:approx_attention_score",
    ];
  }
  for (const block of blocks) {
    if (!block.gacPriority) continue;
    reasons[block.id] = [
      ...(reasons[block.id] ?? []),
      `gac_tier:${block.gacPriority.tier}`,
      ...block.gacPriority.reasonCodes.map((reason) => `gac:${reason}`)
    ];
  }
  for (const block of blocks) {
    if (block.tensorHandles?.key) reasons[block.id] = [...(reasons[block.id] ?? []), `tensor_key:${block.tensorHandles.key.backend}`];
    if (block.tensorHandles?.value) reasons[block.id] = [...(reasons[block.id] ?? []), `tensor_value:${block.tensorHandles.value.backend}`];
  }
  const explicitPrefetchBytes = blocks
    .filter((block) => prefetchBlockIds.includes(block.id) && !lowRankPrefetchIds.has(block.id))
    .reduce((sum, block) => sum + Math.max(0, block.estimatedBytes), 0);
  const explicitPrefetchHitRate = predictedNeededBlockIds.length > 0
    ? prefetchBlockIds.filter((id) => needed.has(id)).length / predictedNeededBlockIds.length
    : 0;
  const prefetchStrategy = lowRankPrefetch.prefetchStrategy !== "none"
    ? lowRankPrefetch.prefetchStrategy
    : (prefetchBlockIds.length > 0 ? "predictive_prefetch" : "none");

  return {
    pinBlockIds,
    evictBlockIds,
    prefetchBlockIds,
    predictivePrefetchBlockIds: uniqueStrings([...predictedNeededBlockIds, ...lowRankPrefetch.prefetchedBlocks]),
    predictionReasons,
    prefetchConfidenceByBlockId,
    ...(lowRankPrefetch.lowRankSummaryRank !== undefined ? { lowRankSummaryRank: lowRankPrefetch.lowRankSummaryRank } : {}),
    predictedHotBlocks: lowRankPrefetch.predictedHotBlocks,
    prefetchedBlocks: prefetchBlockIds,
    prefetchHitRate: lowRankPrefetch.predictedHotBlocks.length > 0
      ? lowRankPrefetch.prefetchHitRate
      : round6(explicitPrefetchHitRate),
    prefetchBytes: lowRankPrefetch.prefetchBytes + explicitPrefetchBytes,
    prefetchLatencyMs: lowRankPrefetch.prefetchLatencyMs,
    attentionStallMs: lowRankPrefetch.attentionStallMs,
    prefetchStrategy,
    reasons,
    estimatedBytesFreed,
    pressureTelemetry: buildKVSwapPressureTelemetry(blocks, policy, targetFreeBytes, estimatedBytesFreed),
  };
}

function getBlockLowRankSummary(block: KVBlock): KVLowRankKeySummary | null {
  if (block.lowRankKeySummary) return block.lowRankKeySummary;
  if (!(block.compressedKeySummary instanceof Float32Array) || !block.summaryRank) return null;
  return createLowRankKeySummary({
    blockId: block.id,
    projectionId: "legacy-compressed-key-summary",
    layer: block.layer,
    headGroupId: "all_heads",
    values: block.compressedKeySummary,
    rank: block.summaryRank,
    ...(block.checksum !== undefined ? { checksum: block.checksum } : {}),
    qualityScore: block.importance,
  });
}

function clamp01(value: number): number {
  if (!Number.isFinite(value)) return 0;
  return Math.max(0, Math.min(1, value));
}

function normalizeRank(value: number): number {
  if (!Number.isFinite(value)) return 0;
  return Math.max(0, Math.min(MAX_LOW_RANK_DIMENSIONS, Math.floor(value)));
}

function normalizeSummaryVector(values: readonly number[] | Float32Array, rank: number): Float32Array {
  const normalizedRank = normalizeRank(rank);
  const result = new Float32Array(normalizedRank);
  for (let index = 0; index < normalizedRank; index += 1) {
    result[index] = boundedFinite(values[index] ?? 0);
  }
  return result;
}

function projectVectorToRank(values: readonly number[], rank: number): number[] {
  const normalizedRank = normalizeRank(rank);
  if (normalizedRank === 0 || values.length === 0) return [];
  if (values.length <= normalizedRank) return values.slice(0, normalizedRank).map(boundedFinite);
  return Array.from({ length: normalizedRank }, (_value, rankIndex) => {
    const start = Math.floor((rankIndex * values.length) / normalizedRank);
    const end = Math.max(start + 1, Math.floor(((rankIndex + 1) * values.length) / normalizedRank));
    let sum = 0;
    let count = 0;
    for (let index = start; index < Math.min(values.length, end); index += 1) {
      sum += boundedFinite(values[index] ?? 0);
      count += 1;
    }
    return count > 0 ? sum / count : 0;
  });
}

function boundedFinite(value: number): number {
  if (!Number.isFinite(value)) return 0;
  return Math.max(-1_000, Math.min(1_000, value));
}

function round6(value: number): number {
  if (!Number.isFinite(value)) return 0;
  return Math.round(value * 1_000_000) / 1_000_000;
}

function uniqueStrings(values: readonly string[]): string[] {
  return [...new Set(values)];
}

function checksumLowRankSummary(blockId: string, projectionId: string, values: Float32Array): string {
  let hash = 2166136261;
  for (const char of `${blockId}:${projectionId}`) {
    hash ^= char.charCodeAt(0);
    hash = Math.imul(hash, 16777619) >>> 0;
  }
  for (const value of values) {
    hash ^= Math.round(boundedFinite(value) * 1_000_000);
    hash = Math.imul(hash, 16777619) >>> 0;
  }
  return `fnv32:${hash.toString(16).padStart(8, "0")}`;
}

function estimateAttentionStallMs(
  blocks: KVBlock[],
  actualAttentionBlockIds: readonly string[],
  prefetched: Set<string>,
  latencyByTier: Record<KVTier, number>,
): number {
  const byId = new Map(blocks.map((block) => [block.id, block]));
  let stallMs = 0;
  for (const blockId of actualAttentionBlockIds) {
    if (prefetched.has(blockId)) continue;
    const block = byId.get(blockId);
    if (!block || block.tier === "vram") continue;
    stallMs += Math.max(0, latencyByTier[block.tier] ?? 0);
  }
  return stallMs;
}

function evictionScore(block: KVBlock, now: number): number {
  const ageMs = Math.max(0, now - block.lastAccessAt);
  const ageScore = Math.min(1, ageMs / 86_400_000);
  const gacPriority = block.gacPriority?.priorityScore ?? 0;
  return block.importance * 0.7 + gacPriority * 0.4 - ageScore * 0.3;
}

function shouldPinBlock(block: KVBlock): boolean {
  return block.pinned || block.gacPriority?.tier === "PIN_HOT";
}

function sumBytes(blocks: KVBlock[]): number {
  return blocks.reduce((sum, block) => sum + Math.max(0, block.estimatedBytes), 0);
}
