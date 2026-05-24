import type { SSALayerRoutingPolicy } from "../ssa";

export interface SSAWebGpuConfig {
  blockSize: number;
  topKBlocks: number;
  localWindowBlocks: number;
  pinnedAnchorBudget: number;
  headDim: number;
  maxQueryBlocksPerDispatch: number;
  maxContextBlocksPerDispatch: number;
  precision: "f32" | "f16";
}

export const DEFAULT_SSA_WEBGPU_CONFIG: SSAWebGpuConfig = {
  blockSize: 16,
  topKBlocks: 16,
  localWindowBlocks: 2,
  pinnedAnchorBudget: 8,
  headDim: 128,
  maxQueryBlocksPerDispatch: 64,
  maxContextBlocksPerDispatch: 8192,
  precision: "f32",
};

export interface SSARoutingBlock {
  id: string;
  blockIndex: number;
  tokenStart: number;
  tokenEnd: number;
  score?: number;
  pinned?: boolean;
}

export interface SSAKernelTrace {
  requestId: string;
  layerIndex: number;
  queryBlockIndex: number;
  selectedBlockIds: string[];
  pinnedBlockIds: string[];
  denseTokenCountEstimate: number;
  sparseTokenCountEstimate: number;
  routingMs: number;
  gatherMs: number;
  attentionMs: number;
  prefillChunkCount?: number;
  prefillChunkSize?: number;
  shapeBucket?: string;
  pipelineCacheKey?: string;
  prefillDispatchTargetMs?: number;
  maxDispatchEstimatedMs?: number;
  prefillChunkDispatch?: "single_dispatch" | "chunked_dispatch";
  attentionDispatchCount?: number;
  awaitedDispatchBreaks?: number;
}

export interface NativeSSABackendContract {
  readonly backendName: string;
  readonly supportsQkvAccess: true;
  readonly supportsLayerSparseRouting: true;
  readonly supportsPinnedKvBlocks: boolean;
  readonly supportsDenseReferenceMode: boolean;

  initializeModel(modelId: string): Promise<void>;
  prefill(inputTokenIds: Int32Array, policy: SSAPrefillPolicy): Promise<SSAPrefillHandle>;
  executeSparseLayer(input: SparseLayerForwardInput): Promise<SparseLayerForwardOutput>;
  decode(input: SSADecodeInput): Promise<SSADecodeOutput>;
  dispose(): Promise<void>;
}

export interface SSAPrefillPolicy {
  requestId: string;
  layerPolicies: SSALayerRoutingPolicy[];
}

export interface SSAPrefillHandle {
  requestId: string;
  tokenCount: number;
  kvCacheHandle: unknown;
  traces: SSAKernelTrace[];
  prefillChunkCount?: number;
  prefillChunkSize?: number;
  shapeBucket?: string;
  pipelineCacheKey?: string;
  prefillDispatchTargetMs?: number;
  maxDispatchEstimatedMs?: number;
  prefillChunkDispatch?: "single_dispatch" | "chunked_dispatch";
  attentionDispatchCount?: number;
  awaitedDispatchBreaks?: number;
}

export interface SparseLayerForwardInput {
  requestId: string;
  layerIndex: number;
  qHandle: unknown;
  kHandle: unknown;
  vHandle: unknown;
  residentQ?: unknown;
  residentK?: unknown;
  residentV?: unknown;
  policy: SSALayerRoutingPolicy;
  queryTokenIndexes?: number[];
  preferResidentOutput?: boolean;
  commandBatch?: unknown;
}

export interface SparseLayerForwardOutput {
  requestId: string;
  layerIndex: number;
  outputHandle: unknown;
  trace: SSAKernelTrace;
}

export interface SSADecodeInput {
  requestId: string;
  inputTokenId: number;
  kvCacheHandle: unknown;
  policy: SSALayerRoutingPolicy[];
  logitCandidateTokenIds?: number[];
  logitTopK?: number;
  logitTileRows?: number;
  suppressedTokenIds?: number[];
  samplingTemperature?: number;
  samplingTopP?: number;
  repetitionPenalty?: number;
  recentTokenIds?: number[];
  samplingSeed?: number;
}

export interface SSADecodeOutput {
  requestId: string;
  tokenId: number;
  logitsHandle?: unknown;
  traces: SSAKernelTrace[];
}
