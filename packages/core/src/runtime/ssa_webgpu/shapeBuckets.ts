import { DEFAULT_SSA_WEBGPU_CONFIG, type SSAWebGpuConfig } from "./types";

export const PROMPT_LENGTH_BUCKETS = [
  1,
  16,
  32,
  64,
  128,
  256,
  512,
  1024,
  2048,
  4096,
  8192,
  16_384,
  32_768,
  40_960,
  65_536,
] as const;

export const SELECTED_BLOCK_COUNT_BUCKETS = [
  1,
  2,
  4,
  8,
  16,
  32,
  64,
  128,
  256,
  512,
] as const;

export const HEAD_DIM_BUCKETS = [
  1,
  16,
  32,
  64,
  80,
  96,
  128,
  160,
  192,
  256,
  320,
] as const;

export const TILE_ROW_BUCKETS = [
  1,
  256,
  512,
  1024,
  2048,
  4096,
  8192,
  16_384,
  32_768,
  65_536,
] as const;

export const DEFAULT_PREFILL_DISPATCH_BUDGET_MS = 32;
export const DEFAULT_PREFILL_DISPATCH_COST_PER_TOKEN_SELECTED_DIM_MS = 0.000003;

export interface StaticShapeBucketInput {
  operation?: string;
  promptLengthTokens?: number | null;
  selectedBlockCount?: number | null;
  headDim?: number | null;
  tileRows?: number | null;
  precision?: SSAWebGpuConfig["precision"];
}

export interface StaticShapeBucket {
  operation: string;
  promptLengthBucket: number;
  selectedBlockCountBucket: number;
  headDimBucket: number;
  tileRowsBucket: number;
  precision: SSAWebGpuConfig["precision"];
  shapeBucket: string;
  pipelineCacheKey: string;
}

export interface PrefillChunkPlanningOptions extends StaticShapeBucketInput {
  maxChunkTokens?: number | null;
  maxDispatchEstimatedMs?: number | null;
  blockSize?: number | null;
  dispatchCostPerTokenSelectedDimMs?: number | null;
}

export interface PrefillChunkPlanChunk {
  index: number;
  tokenStart: number;
  tokenEnd: number;
  tokenCount: number;
  estimatedDispatchMs: number;
  shapeBucket: string;
  pipelineCacheKey: string;
}

export interface PrefillChunkPlan {
  tokenCount: number;
  promptLengthBucket: number;
  selectedBlockCountBucket: number;
  headDimBucket: number;
  tileRowsBucket: number;
  precision: SSAWebGpuConfig["precision"];
  maxChunkTokens: number;
  prefillChunkCount: number;
  prefillChunkSize: number;
  dispatchBudgetMs: number;
  maxDispatchEstimatedMs: number;
  shapeBucket: string;
  pipelineCacheKey: string;
  chunks: PrefillChunkPlanChunk[];
}

export function bucketPromptLength(value: number | null | undefined): number {
  return bucketValue(normalizePositiveInteger(value, 1), PROMPT_LENGTH_BUCKETS);
}

export function bucketSelectedBlockCount(value: number | null | undefined): number {
  return bucketValue(normalizePositiveInteger(value, 1), SELECTED_BLOCK_COUNT_BUCKETS);
}

export function bucketHeadDim(value: number | null | undefined): number {
  return bucketValue(normalizePositiveInteger(value, 1), HEAD_DIM_BUCKETS);
}

export function bucketTileRows(value: number | null | undefined): number {
  return bucketValue(normalizePositiveInteger(value, 1), TILE_ROW_BUCKETS);
}

export function resolveStaticShapeBucket(input: StaticShapeBucketInput = {}): StaticShapeBucket {
  const operation = normalizeOperation(input.operation);
  const promptLengthBucket = bucketPromptLength(input.promptLengthTokens);
  const selectedBlockCountBucket = bucketSelectedBlockCount(input.selectedBlockCount);
  const headDimBucket = bucketHeadDim(input.headDim);
  const tileRowsBucket = bucketTileRows(input.tileRows);
  const precision = input.precision ?? DEFAULT_SSA_WEBGPU_CONFIG.precision;
  const shapeBucket = [
    `prompt<=${promptLengthBucket}`,
    `selected<=${selectedBlockCountBucket}`,
    `headDim<=${headDimBucket}`,
    `tileRows<=${tileRowsBucket}`,
    `precision=${precision}`,
  ].join(":");
  return {
    operation,
    promptLengthBucket,
    selectedBlockCountBucket,
    headDimBucket,
    tileRowsBucket,
    precision,
    shapeBucket,
    pipelineCacheKey: `${operation}:${shapeBucket}`,
  };
}

export function createStaticPipelineCacheKey(input: StaticShapeBucketInput = {}): string {
  return resolveStaticShapeBucket(input).pipelineCacheKey;
}

export function planPrefillChunks(tokenCountInput: number | null | undefined, options: PrefillChunkPlanningOptions = {}): PrefillChunkPlan {
  const tokenCount = normalizeNonNegativeInteger(tokenCountInput);
  const promptLengthBucket = bucketPromptLength(tokenCount);
  const selectedBlockCount = options.selectedBlockCount ?? defaultSelectedBlockCount();
  const selectedBlockCountBucket = bucketSelectedBlockCount(selectedBlockCount);
  const headDimBucket = bucketHeadDim(options.headDim ?? DEFAULT_SSA_WEBGPU_CONFIG.headDim);
  const tileRowsBucket = bucketTileRows(options.tileRows ?? DEFAULT_SSA_WEBGPU_CONFIG.maxContextBlocksPerDispatch);
  const precision = options.precision ?? DEFAULT_SSA_WEBGPU_CONFIG.precision;
  const dispatchBudgetMs = normalizePositiveNumber(options.maxDispatchEstimatedMs, DEFAULT_PREFILL_DISPATCH_BUDGET_MS);
  const blockSize = normalizePositiveInteger(options.blockSize, DEFAULT_SSA_WEBGPU_CONFIG.blockSize);
  const requestedMaxChunkTokens = normalizePositiveInteger(
    options.maxChunkTokens,
    DEFAULT_SSA_WEBGPU_CONFIG.maxQueryBlocksPerDispatch * DEFAULT_SSA_WEBGPU_CONFIG.blockSize,
  );
  const cost = normalizePositiveNumber(
    options.dispatchCostPerTokenSelectedDimMs,
    DEFAULT_PREFILL_DISPATCH_COST_PER_TOKEN_SELECTED_DIM_MS,
  );

  if (tokenCount === 0) {
    const emptyShape = resolveStaticShapeBucket({
      operation: "prefill_chunk",
      promptLengthTokens: 0,
      selectedBlockCount: selectedBlockCountBucket,
      headDim: headDimBucket,
      tileRows: tileRowsBucket,
      precision,
    });
    return {
      tokenCount,
      promptLengthBucket,
      selectedBlockCountBucket,
      headDimBucket,
      tileRowsBucket,
      precision,
      maxChunkTokens: requestedMaxChunkTokens,
      prefillChunkCount: 0,
      prefillChunkSize: 0,
      dispatchBudgetMs,
      maxDispatchEstimatedMs: 0,
      shapeBucket: emptyShape.shapeBucket,
      pipelineCacheKey: emptyShape.pipelineCacheKey,
      chunks: [],
    };
  }

  const selectedTokenEstimate = selectedBlockCountBucket * blockSize;
  const perTokenEstimateMs = selectedTokenEstimate * headDimBucket * cost;
  const budgetBoundChunkTokens = Math.max(1, Math.floor(dispatchBudgetMs / perTokenEstimateMs));
  const boundedChunkTokens = Math.min(requestedMaxChunkTokens, budgetBoundChunkTokens);
  const prefillChunkSize = alignChunkSizeToBlock(boundedChunkTokens, blockSize);
  const dispatchShape = resolveStaticShapeBucket({
    operation: options.operation ?? "prefill_chunk",
    promptLengthTokens: prefillChunkSize,
    selectedBlockCount: selectedBlockCountBucket,
    headDim: headDimBucket,
    tileRows: tileRowsBucket,
    precision,
  });
  const chunks: PrefillChunkPlanChunk[] = [];
  for (let tokenStart = 0; tokenStart < tokenCount; tokenStart += prefillChunkSize) {
    const tokenEnd = Math.min(tokenCount, tokenStart + prefillChunkSize);
    const chunkTokenCount = tokenEnd - tokenStart;
    chunks.push({
      index: chunks.length,
      tokenStart,
      tokenEnd,
      tokenCount: chunkTokenCount,
      estimatedDispatchMs: estimatePrefillDispatchMs({
        tokenCount: chunkTokenCount,
        selectedBlockCount: selectedBlockCountBucket,
        headDim: headDimBucket,
        blockSize,
        dispatchCostPerTokenSelectedDimMs: cost,
      }),
      shapeBucket: dispatchShape.shapeBucket,
      pipelineCacheKey: dispatchShape.pipelineCacheKey,
    });
  }

  return {
    tokenCount,
    promptLengthBucket,
    selectedBlockCountBucket,
    headDimBucket,
    tileRowsBucket,
    precision,
    maxChunkTokens: requestedMaxChunkTokens,
    prefillChunkCount: chunks.length,
    prefillChunkSize,
    dispatchBudgetMs,
    maxDispatchEstimatedMs: chunks.reduce((max, chunk) => Math.max(max, chunk.estimatedDispatchMs), 0),
    shapeBucket: dispatchShape.shapeBucket,
    pipelineCacheKey: dispatchShape.pipelineCacheKey,
    chunks,
  };
}

export function estimatePrefillDispatchMs(input: {
  tokenCount: number | null | undefined;
  selectedBlockCount: number | null | undefined;
  headDim: number | null | undefined;
  blockSize?: number | null;
  dispatchCostPerTokenSelectedDimMs?: number | null;
}): number {
  const tokenCount = normalizeNonNegativeInteger(input.tokenCount);
  if (tokenCount === 0) return 0;
  const selectedBlockCount = bucketSelectedBlockCount(input.selectedBlockCount);
  const headDim = bucketHeadDim(input.headDim);
  const blockSize = normalizePositiveInteger(input.blockSize, DEFAULT_SSA_WEBGPU_CONFIG.blockSize);
  const cost = normalizePositiveNumber(
    input.dispatchCostPerTokenSelectedDimMs,
    DEFAULT_PREFILL_DISPATCH_COST_PER_TOKEN_SELECTED_DIM_MS,
  );
  return roundMs(tokenCount * selectedBlockCount * blockSize * headDim * cost);
}

function bucketValue(value: number, buckets: readonly number[]): number {
  for (const bucket of buckets) {
    if (value <= bucket) return bucket;
  }
  return nextPowerOfTwo(value);
}

function normalizePositiveInteger(value: number | null | undefined, fallback: number): number {
  if (typeof value !== "number" || !Number.isFinite(value)) return fallback;
  const normalized = Math.floor(value);
  return normalized > 0 ? normalized : fallback;
}

function normalizeNonNegativeInteger(value: number | null | undefined): number {
  if (typeof value !== "number" || !Number.isFinite(value)) return 0;
  return Math.max(0, Math.floor(value));
}

function normalizePositiveNumber(value: number | null | undefined, fallback: number): number {
  if (typeof value !== "number" || !Number.isFinite(value) || value <= 0) return fallback;
  return value;
}

function normalizeOperation(value: string | undefined): string {
  const trimmed = value?.trim();
  if (!trimmed) return "ssa_webgpu";
  return trimmed.replace(/[^a-zA-Z0-9:._-]/g, "_");
}

function nextPowerOfTwo(value: number): number {
  if (value <= 1) return 1;
  return 2 ** Math.ceil(Math.log2(value));
}

function defaultSelectedBlockCount(): number {
  return DEFAULT_SSA_WEBGPU_CONFIG.topKBlocks
    + DEFAULT_SSA_WEBGPU_CONFIG.localWindowBlocks
    + DEFAULT_SSA_WEBGPU_CONFIG.pinnedAnchorBudget;
}

function alignChunkSizeToBlock(chunkSize: number, blockSize: number): number {
  if (chunkSize < blockSize) return Math.max(1, chunkSize);
  return Math.max(blockSize, Math.floor(chunkSize / blockSize) * blockSize);
}

function roundMs(value: number): number {
  return Number(value.toFixed(6));
}
