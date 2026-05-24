import type { SparseForwardInput, SparseForwardOutput, SSAKernelBackend, SSAPlan, SSAPlanInput } from "../ssa";
import { FallbackSSARuntime } from "../ssa";
import type { Matrix } from "./denseReference";
import { sparseReferenceAttention } from "./sparseReference";
import type { SSAWebGpuConfig } from "./types";
import { DEFAULT_SSA_WEBGPU_CONFIG } from "./types";
import type { SSAKernelTrace } from "./types";
import { packedSparseAttentionDecodeWgsl, packedSparseAttentionWgsl, sparseAttentionWgsl } from "./wgsl/sparseAttention.wgsl";
import { PACKED_QKV_PROJECTION_WGSL } from "./fusedDecodeLayer/wgsl/packedQkvProjection.wgsl";
import { QWEN_QKV_NORM_ROPE_PAIR_WGSL } from "./fusedDecodeLayer/wgsl/qwenQkvNormRopeAppendKv.wgsl";
import { QWEN_ONE_TOKEN_ATTENTION_WGSL } from "./fusedDecodeLayer/wgsl/qwenOneTokenAttention.wgsl";
import { RESIDUAL_RMSNORM_ONE_TOKEN_WGSL } from "./fusedDecodeLayer/wgsl/residualRmsNorm.wgsl";

export interface WebGpuSsaBackendOptions {
  backendPreference?: "webgpu" | "cpu";
  device?: unknown;
  gpu?: unknown;
  requireWebGpu?: boolean;
  config?: Partial<SSAWebGpuConfig>;
}

export interface WebGpuDecodeCommandBatchLike {
  recordComputePass(input: {
    label: string;
    dispatches?: number;
    record: (encoder: MinimalGpuCommandEncoder) => void;
  }): void;
  recordCopy?(input: {
    label: string;
    record: (encoder: MinimalGpuCommandEncoder) => void;
  }): void;
}

export interface SsaToyTensorRange {
  tokenStart: number;
  tokenEnd: number;
}

export interface SsaToyTensorHandle {
  kind: "ssa_toy_tensor";
  id: string;
  matrix: Matrix;
  blockTokenRanges: Record<string, SsaToyTensorRange>;
}

export interface WebGpuSparseAttentionInput {
  q: Matrix;
  k: Matrix;
  v: Matrix;
  selectedKeyIndexesByQuery: number[][];
  causal?: boolean;
  scale?: number;
  backendPreference?: "webgpu" | "cpu";
  gpu?: unknown;
  device?: unknown;
  requireWebGpu?: boolean;
  bufferCache?: WebGpuRuntimeBufferCache;
  commandBatch?: WebGpuDecodeCommandBatchLike;
}

export interface WebGpuSparseAttentionTrace {
  backend: "webgpu" | "cpu_reference";
  queryTokens: number;
  keyTokens: number;
  headDim: number;
  selectedIndexSlots: number;
  maxSelectedPerQuery: number;
  computeMs: number;
  pipelineCacheKey?: string;
  pipelineCacheHit?: boolean;
  metadata?: Record<string, unknown>;
}

export interface WebGpuSparseAttentionResult {
  output: Matrix;
  backend: "webgpu" | "cpu_reference";
  trace: WebGpuSparseAttentionTrace;
}

export interface WebGpuPackedSparseAttentionInput extends Omit<WebGpuSparseAttentionInput, "q" | "k" | "v"> {
  q: Matrix;
  k: Matrix;
  v: Matrix;
  headCount: number;
  keyValueHeadCount?: number;
  headDim: number;
}

export interface WebGpuPackedSparseAttentionTrace extends WebGpuSparseAttentionTrace {
  packedHeads: true;
  headCount: number;
  keyValueHeadCount: number;
  outputSize: number;
  dispatchCount: number;
}

export interface WebGpuPackedSparseAttentionResult {
  output: Matrix;
  backend: "webgpu" | "cpu_reference";
  trace: WebGpuPackedSparseAttentionTrace;
}

export interface WebGpuPackedSparseAttentionResidentInput extends Omit<WebGpuPackedSparseAttentionInput, "q" | "k" | "v" | "backendPreference"> {
  q: Matrix | WebGpuResidentTensor;
  k: Matrix | WebGpuResidentTensor;
  v: Matrix | WebGpuResidentTensor;
  traceMetadata?: Record<string, unknown>;
}

export interface WebGpuPackedSparseAttentionResidentTrace extends WebGpuPackedSparseAttentionTrace {
  outputResident: true;
  readback: false;
  inputResident: {
    q: boolean;
    k: boolean;
    v: boolean;
  };
  metadata?: Record<string, unknown>;
}

export interface WebGpuPackedSparseAttentionResidentResult {
  tensor: WebGpuResidentTensor;
  backend: "webgpu";
  trace: WebGpuPackedSparseAttentionResidentTrace;
}

export type DenseMatVecMatrix = Matrix | {
  readonly rowCount: number;
  readonly colCount: number;
  row(index: number): ArrayLike<number> | undefined;
  toFloat32Array?(rowIds?: number[]): Float32Array;
};

export interface WebGpuDenseMatVecInput {
  vector: ArrayLike<number>;
  matrix: DenseMatVecMatrix;
  selectedRowIds?: number[];
  backendPreference?: "webgpu" | "cpu";
  gpu?: unknown;
  device?: unknown;
  requireWebGpu?: boolean;
  bufferCache?: WebGpuRuntimeBufferCache;
  projectionCacheKey?: string;
  projectionCachePolicy?: "stable";
  traceMetadata?: Record<string, unknown>;
  suppressedRowIds?: number[];
}

export interface WebGpuDenseMatVecTrace {
  backend: "webgpu" | "cpu_reference";
  rows: number;
  cols: number;
  selectedRows: number;
  computeMs: number;
  projectionCacheKey?: string;
  projectionCacheHit?: boolean;
  pipelineCacheKey?: string;
  pipelineCacheHit?: boolean;
  metadata?: Record<string, unknown>;
}

export interface WebGpuDenseMatVecResult {
  values: number[];
  selectedRowIds?: number[];
  backend: "webgpu" | "cpu_reference";
  trace: WebGpuDenseMatVecTrace;
}

export interface WebGpuDenseMatVecTopKInput extends WebGpuDenseMatVecInput {
  topK: number;
  tileRows?: number;
}

export interface WebGpuDenseMatVecTopKResult {
  values: number[];
  selectedRowIds: number[];
  backend: "webgpu" | "cpu_reference";
  trace: WebGpuDenseMatVecTrace & {
    topK: number;
    tileRows: number;
    tiles: number;
    scannedRows: number;
    materializedRows: number;
    readbackStrategy?: "full_logits" | "gpu_top1_candidates" | "gpu_argmax_token_id" | "gpu_compact_topk";
    gpuReducedRows?: number;
    readbackRows?: number;
    readbackBytes?: number;
    dispatchCount?: number;
    suppressedRowCount?: number;
  };
}

const DEFAULT_DENSE_MATVEC_TOPK_TILE_ROWS = 8192;
const DEFAULT_TOKEN_EMBEDDING_TILE_ROWS = 4096;

export interface WebGpuStableMatrixPreloadDescriptor {
  key: string;
  matrix: DenseMatVecMatrix;
  rows?: number;
  cols?: number;
  rowIds?: number[];
}

export interface WebGpuStableStaticPreloadDescriptor {
  key: string;
  data: Float32Array | Int32Array | Uint32Array;
  usageKind?: "storage" | "uniform";
  usage?: number;
}

export type WebGpuRuntimePipelinePreloadKind =
  | "sparse-attention"
  | "packed-sparse-attention"
  | "token-embedding-lookup-resident"
  | "rmsnorm-resident"
  | "residual-rmsnorm-pair-resident"
  | "residual-add-resident"
  | "qwen-qkv-post-projection-resident"
  | "qwen-qkv-norm-rope-pair-resident"
  | "packed-qkv-projection-resident"
  | "dense-matvec"
  | "dense-matvec-top1-candidates"
  | "dense-matvec-top1-reduce"
  | "dense-matvec-compact-topk-scores"
  | "dense-matvec-compact-topk-reduce"
  | "dense-matmul"
  | "mlp-batch-intermediate"
  | "mlp-batch-output";

export interface WebGpuStableRuntimePreloadInput {
  bufferCache: WebGpuRuntimeBufferCache;
  matrices?: WebGpuStableMatrixPreloadDescriptor[];
  staticBuffers?: WebGpuStableStaticPreloadDescriptor[];
  pipelines?: WebGpuRuntimePipelinePreloadKind[];
  gpu?: unknown;
  device?: unknown;
  requireWebGpu?: boolean;
}

export interface WebGpuStableRuntimePreloadEntry {
  kind: "matrix" | "static" | "pipeline";
  key: string;
  rows?: number;
  cols?: number;
  byteLength?: number;
  cacheHit: boolean;
}

export interface WebGpuStableRuntimePreloadResult {
  backend: "webgpu";
  computeMs: number;
  entries: WebGpuStableRuntimePreloadEntry[];
  uploadedEntries: number;
  cacheHits: number;
}

export interface WebGpuDenseMatMulInput {
  activations: Matrix;
  projection: DenseMatVecMatrix;
  backendPreference?: "webgpu" | "cpu";
  gpu?: unknown;
  device?: unknown;
  requireWebGpu?: boolean;
  bufferCache?: WebGpuRuntimeBufferCache;
  projectionCacheKey?: string;
  projectionCachePolicy?: "stable";
  traceMetadata?: Record<string, unknown>;
  commandBatch?: WebGpuDecodeCommandBatchLike;
}

export interface WebGpuDenseMatMulTrace {
  backend: "webgpu" | "cpu_reference";
  tokens: number;
  hidden: number;
  outputSize: number;
  computeMs: number;
  projectionCacheKey?: string;
  projectionCacheHit?: boolean;
  pipelineCacheKey?: string;
  pipelineCacheHit?: boolean;
  metadata?: Record<string, unknown>;
}

export interface WebGpuDenseMatMulResult {
  output: Matrix;
  backend: "webgpu" | "cpu_reference";
  trace: WebGpuDenseMatMulTrace;
}

export interface WebGpuResidentTensor {
  kind: "webgpu_resident_tensor";
  id: string;
  rows: number;
  cols: number;
  byteLength: number;
  buffer: unknown;
  device: unknown;
  retainedBuffers?: unknown[];
}

export interface WebGpuResidentRowCache {
  kind: "webgpu_resident_row_cache";
  tensor: WebGpuResidentTensor;
  rows: number;
  cols: number;
  capacityRows: number;
}

export interface WebGpuDenseMatMulResidentInput extends Omit<WebGpuDenseMatMulInput, "activations"> {
  activations: Matrix | WebGpuResidentTensor;
}

export interface WebGpuDenseMatMulResidentTrace extends WebGpuDenseMatMulTrace {
  outputResident: true;
  readback: false;
  inputResident: boolean;
}

export interface WebGpuDenseMatMulResidentResult {
  tensor: WebGpuResidentTensor;
  backend: "webgpu";
  trace: WebGpuDenseMatMulResidentTrace;
}

export interface WebGpuPackedQkvProjectionResidentInput {
  hidden: Matrix | WebGpuResidentTensor;
  qProjection: DenseMatVecMatrix;
  kProjection: DenseMatVecMatrix;
  vProjection: DenseMatVecMatrix;
  gpu?: unknown;
  device?: unknown;
  requireWebGpu?: boolean;
  bufferCache?: WebGpuRuntimeBufferCache;
  projectionCacheKey?: string;
  qProjectionCacheKey?: string;
  kProjectionCacheKey?: string;
  vProjectionCacheKey?: string;
  projectionCachePolicy?: "stable";
  traceMetadata?: Record<string, unknown>;
  commandBatch?: WebGpuDecodeCommandBatchLike;
}

export interface WebGpuPackedQkvProjectionResidentTrace {
  backend: "webgpu";
  tokens: number;
  hidden: number;
  qOutputSize: number;
  kOutputSize: number;
  vOutputSize: number;
  outputSize: number;
  computeMs: number;
  outputResident: true;
  readback: false;
  inputResident: boolean;
  dispatchCount: number;
  projectionCacheHits?: {
    qProjection: boolean;
    kProjection: boolean;
    vProjection: boolean;
  };
  pipelineCacheKey?: string;
  pipelineCacheHit?: boolean;
  metadata?: Record<string, unknown>;
}

export interface WebGpuPackedQkvProjectionResidentResult {
  q: WebGpuResidentTensor;
  k: WebGpuResidentTensor;
  v: WebGpuResidentTensor;
  backend: "webgpu";
  trace: WebGpuPackedQkvProjectionResidentTrace;
}

export interface WebGpuResidentTensorUploadInput {
  matrix: Matrix;
  gpu?: unknown;
  device?: unknown;
  requireWebGpu?: boolean;
  traceMetadata?: Record<string, unknown>;
}

export interface WebGpuResidentTensorTrace {
  backend: "webgpu";
  tokens: number;
  hidden: number;
  computeMs: number;
  outputResident: true;
  readback: false;
  inputResident?: boolean;
  pipelineCacheKey?: string;
  pipelineCacheHit?: boolean;
  metadata?: Record<string, unknown>;
}

export interface WebGpuResidentTensorResult {
  tensor: WebGpuResidentTensor;
  backend: "webgpu";
  trace: WebGpuResidentTensorTrace;
}

export interface WebGpuTokenEmbeddingLookupResidentInput {
  tokenId: number;
  tokenEmbedding: DenseMatVecMatrix;
  embeddingTileRows?: number;
  gpu?: unknown;
  device?: unknown;
  requireWebGpu?: boolean;
  bufferCache?: WebGpuRuntimeBufferCache;
  projectionCacheKey?: string;
  projectionCachePolicy?: "stable";
  traceMetadata?: Record<string, unknown>;
  commandBatch?: WebGpuDecodeCommandBatchLike;
}

export interface WebGpuTokenEmbeddingLookupResidentTrace extends WebGpuResidentTensorTrace {
  tokenId: number;
  vocabSize: number;
  embeddingTileStart?: number;
  embeddingTileRows?: number;
  embeddingCacheKey?: string;
  embeddingCacheHit?: boolean;
}

export interface WebGpuTokenEmbeddingLookupResidentResult {
  tensor: WebGpuResidentTensor;
  backend: "webgpu";
  trace: WebGpuTokenEmbeddingLookupResidentTrace;
}

export interface WebGpuQkvPostProjectionResidentInput {
  projected: Matrix | WebGpuResidentTensor;
  headCount: number;
  headDim: number;
  positions: number[];
  normWeight?: ArrayLike<number>;
  eps?: number;
  ropeTheta?: number;
  gpu?: unknown;
  device?: unknown;
  requireWebGpu?: boolean;
  bufferCache?: WebGpuRuntimeBufferCache;
  traceMetadata?: Record<string, unknown>;
  commandBatch?: WebGpuDecodeCommandBatchLike;
}

export interface WebGpuQkvNormRopePairResidentInput {
  qProjected: Matrix | WebGpuResidentTensor;
  kProjected: Matrix | WebGpuResidentTensor;
  qHeadCount: number;
  kHeadCount: number;
  headDim: number;
  positions: number[];
  qNormWeight?: ArrayLike<number>;
  kNormWeight?: ArrayLike<number>;
  eps?: number;
  ropeTheta?: number;
  gpu?: unknown;
  device?: unknown;
  requireWebGpu?: boolean;
  bufferCache?: WebGpuRuntimeBufferCache;
  traceMetadata?: Record<string, unknown>;
  commandBatch?: WebGpuDecodeCommandBatchLike;
}

export interface WebGpuQkvNormRopePairResidentTrace {
  backend: "webgpu";
  tokens: number;
  headDim: number;
  qHeadCount: number;
  kHeadCount: number;
  qHidden: number;
  kHidden: number;
  computeMs: number;
  outputResident: true;
  readback: false;
  inputResident: {
    q: boolean;
    k: boolean;
  };
  dispatchCount: number;
  pipelineCacheKey?: string;
  pipelineCacheHit?: boolean;
  metadata?: Record<string, unknown>;
}

export interface WebGpuQkvNormRopePairResidentResult {
  q: WebGpuResidentTensor;
  k: WebGpuResidentTensor;
  backend: "webgpu";
  trace: WebGpuQkvNormRopePairResidentTrace;
}

export interface WebGpuRmsNormResidentInput {
  hidden: Matrix | WebGpuResidentTensor;
  weight?: ArrayLike<number>;
  eps?: number;
  gpu?: unknown;
  device?: unknown;
  requireWebGpu?: boolean;
  bufferCache?: WebGpuRuntimeBufferCache;
  traceMetadata?: Record<string, unknown>;
  commandBatch?: WebGpuDecodeCommandBatchLike;
}

export interface WebGpuResidualAddResidentInput {
  left: Matrix | WebGpuResidentTensor;
  right: Matrix | WebGpuResidentTensor;
  gpu?: unknown;
  device?: unknown;
  requireWebGpu?: boolean;
  bufferCache?: WebGpuRuntimeBufferCache;
  traceMetadata?: Record<string, unknown>;
  commandBatch?: WebGpuDecodeCommandBatchLike;
}

export interface WebGpuResidualRmsNormPairResidentInput {
  left: Matrix | WebGpuResidentTensor;
  right: Matrix | WebGpuResidentTensor;
  weight: ArrayLike<number>;
  eps?: number;
  gpu?: unknown;
  device?: unknown;
  requireWebGpu?: boolean;
  bufferCache?: WebGpuRuntimeBufferCache;
  traceMetadata?: Record<string, unknown>;
  commandBatch?: WebGpuDecodeCommandBatchLike;
}

export interface WebGpuResidualRmsNormPairResidentTrace {
  backend: "webgpu";
  tokens: number;
  hidden: number;
  computeMs: number;
  outputResident: true;
  readback: false;
  inputResident: {
    left: boolean;
    right: boolean;
  };
  dispatchCount: number;
  pipelineCacheKey?: string;
  pipelineCacheHit?: boolean;
  metadata?: Record<string, unknown>;
}

export interface WebGpuResidualRmsNormPairResidentResult {
  summed: WebGpuResidentTensor;
  normed: WebGpuResidentTensor;
  backend: "webgpu";
  trace: WebGpuResidualRmsNormPairResidentTrace;
}

export interface WebGpuSparseAttentionResidentInput extends Omit<WebGpuSparseAttentionInput, "q" | "k" | "v" | "backendPreference"> {
  q: Matrix | WebGpuResidentTensor;
  k: Matrix | WebGpuResidentTensor;
  v: Matrix | WebGpuResidentTensor;
  traceMetadata?: Record<string, unknown>;
  commandBatch?: WebGpuDecodeCommandBatchLike;
}

export interface WebGpuSparseAttentionResidentTrace extends WebGpuSparseAttentionTrace {
  outputResident: true;
  readback: false;
  inputResident: {
    q: boolean;
    k: boolean;
    v: boolean;
  };
  metadata?: Record<string, unknown>;
}

export interface WebGpuSparseAttentionResidentResult {
  tensor: WebGpuResidentTensor;
  backend: "webgpu";
  trace: WebGpuSparseAttentionResidentTrace;
}

export type WebGpuMlpActivationKind = "silu_gated" | "gelu";

export interface WebGpuMlpInput {
  hidden: ArrayLike<number>;
  upProjection: DenseMatVecMatrix;
  downProjection: DenseMatVecMatrix;
  gateProjection?: DenseMatVecMatrix;
  backendPreference?: "webgpu" | "cpu";
  gpu?: unknown;
  device?: unknown;
  requireWebGpu?: boolean;
  bufferCache?: WebGpuRuntimeBufferCache;
  projectionCacheKey?: string;
  projectionCachePolicy?: "stable";
  traceMetadata?: Record<string, unknown>;
  commandBatch?: WebGpuDecodeCommandBatchLike;
}

export interface WebGpuMlpBatchInput extends Omit<WebGpuMlpInput, "hidden"> {
  hidden: Matrix;
}

export interface WebGpuMlpTrace {
  backend: "webgpu" | "cpu_reference";
  tokens?: number;
  inputSize: number;
  intermediateSize: number;
  outputSize: number;
  activationKind: WebGpuMlpActivationKind;
  computeMs: number;
  projectionCacheHits?: {
    upProjection: boolean;
    downProjection: boolean;
    gateProjection?: boolean;
  };
  pipelineCacheHits?: {
    intermediate: boolean;
    output: boolean;
  };
  metadata?: Record<string, unknown>;
}

export interface WebGpuMlpResult {
  values: number[];
  backend: "webgpu" | "cpu_reference";
  trace: WebGpuMlpTrace;
}

export interface WebGpuMlpBatchResult {
  output: Matrix;
  backend: "webgpu" | "cpu_reference";
  trace: WebGpuMlpTrace;
}

export interface WebGpuMlpBatchResidentInput extends Omit<WebGpuMlpBatchInput, "hidden" | "backendPreference"> {
  hidden: Matrix | WebGpuResidentTensor;
}

export interface WebGpuMlpBatchResidentTrace extends WebGpuMlpTrace {
  outputResident: true;
  readback: false;
  inputResident: boolean;
}

export interface WebGpuMlpBatchResidentResult {
  tensor: WebGpuResidentTensor;
  backend: "webgpu";
  trace: WebGpuMlpBatchResidentTrace;
}

export interface WebGpuDenseMatVecResidentInput extends Omit<WebGpuDenseMatVecInput, "vector" | "backendPreference"> {
  vector: WebGpuResidentTensor;
}

export interface WebGpuDenseMatVecResidentResult extends Omit<WebGpuDenseMatVecResult, "backend" | "trace"> {
  backend: "webgpu";
  trace: WebGpuDenseMatVecTrace & {
    vectorResident: true;
  };
}

export interface WebGpuDenseMatVecTopKResidentInput extends Omit<WebGpuDenseMatVecTopKInput, "vector" | "backendPreference"> {
  vector: WebGpuResidentTensor;
  forceFinalTopKReduction?: boolean;
}

export interface WebGpuDenseMatVecTopKResidentResult extends Omit<WebGpuDenseMatVecTopKResult, "backend" | "trace"> {
  backend: "webgpu";
  trace: WebGpuDenseMatVecTopKResult["trace"] & {
    vectorResident: true;
  };
}

interface RuntimeBufferCacheEntry {
  buffer: MinimalGpuBuffer;
  rows: number;
  cols: number;
  byteLength: number;
  usage: number;
}

interface RuntimeShaderModuleCacheEntry {
  module: unknown;
  code: string;
}

interface RuntimeComputePipelineCacheEntry {
  pipeline: MinimalGpuComputePipeline;
  moduleKey: string;
  entryPoint: string;
}

export class WebGpuRuntimeBufferCache {
  private denseMatrixBuffers = new WeakMap<object, Map<string, RuntimeBufferCacheEntry>>();
  private staticBuffers = new WeakMap<object, Map<string, RuntimeBufferCacheEntry>>();
  private shaderModules = new WeakMap<object, Map<string, RuntimeShaderModuleCacheEntry>>();
  private computePipelines = new WeakMap<object, Map<string, RuntimeComputePipelineCacheEntry>>();
  private readonly entries = new Set<RuntimeBufferCacheEntry>();

  getOrUploadFloatMatrix(input: {
    device: MinimalGpuDevice;
    key: string;
    rows: number;
    cols: number;
    usage: number;
    byteLength: number;
    dataFactory: () => Float32Array;
  }): { buffer: MinimalGpuBuffer; cacheHit: boolean } {
    const deviceKey = input.device as object;
    let deviceBuffers = this.denseMatrixBuffers.get(deviceKey);
    if (!deviceBuffers) {
      deviceBuffers = new Map();
      this.denseMatrixBuffers.set(deviceKey, deviceBuffers);
    }
    const existing = deviceBuffers.get(input.key);
    if (existing) {
      if (
        existing.rows !== input.rows
        || existing.cols !== input.cols
        || existing.byteLength !== input.byteLength
        || existing.usage !== input.usage
      ) {
        throw new Error(`WebGPU runtime buffer cache key ${input.key} was reused with an incompatible dense matrix shape.`);
      }
      return { buffer: existing.buffer, cacheHit: true };
    }
    const values = input.dataFactory();
    if (values.byteLength !== input.byteLength) {
      throw new Error(`WebGPU runtime buffer cache dataFactory for ${input.key} returned ${values.byteLength} bytes; expected ${input.byteLength}.`);
    }
    const buffer = createUploadedBuffer(input.device, values, input.usage);
    const entry = {
      buffer,
      rows: input.rows,
      cols: input.cols,
      byteLength: input.byteLength,
      usage: input.usage,
    };
    deviceBuffers.set(input.key, entry);
    this.entries.add(entry);
    return { buffer, cacheHit: false };
  }

  getOrUploadStaticBuffer(input: {
    device: MinimalGpuDevice;
    key: string;
    usage: number;
    byteLength: number;
    dataFactory: () => Float32Array | Int32Array | Uint32Array;
  }): { buffer: MinimalGpuBuffer; cacheHit: boolean } {
    const deviceKey = input.device as object;
    let deviceBuffers = this.staticBuffers.get(deviceKey);
    if (!deviceBuffers) {
      deviceBuffers = new Map();
      this.staticBuffers.set(deviceKey, deviceBuffers);
    }
    const existing = deviceBuffers.get(input.key);
    if (existing) {
      if (existing.byteLength !== input.byteLength || existing.usage !== input.usage) {
        throw new Error(`WebGPU runtime buffer cache key ${input.key} was reused with an incompatible static buffer shape.`);
      }
      return { buffer: existing.buffer, cacheHit: true };
    }
    const values = input.dataFactory();
    if (values.byteLength !== input.byteLength) {
      throw new Error(`WebGPU runtime buffer cache dataFactory for ${input.key} returned ${values.byteLength} bytes; expected ${input.byteLength}.`);
    }
    const entry = {
      buffer: createUploadedBuffer(input.device, values, input.usage),
      rows: 1,
      cols: input.byteLength,
      byteLength: input.byteLength,
      usage: input.usage,
    };
    deviceBuffers.set(input.key, entry);
    this.entries.add(entry);
    return { buffer: entry.buffer, cacheHit: false };
  }

  getOrCreateComputePipeline(input: {
    device: MinimalGpuDevice;
    key: string;
    moduleKey: string;
    code: string;
    entryPoint: string;
  }): { pipeline: MinimalGpuComputePipeline; cacheHit: boolean } {
    const deviceKey = input.device as object;
    let devicePipelines = this.computePipelines.get(deviceKey);
    if (!devicePipelines) {
      devicePipelines = new Map();
      this.computePipelines.set(deviceKey, devicePipelines);
    }
    const existing = devicePipelines.get(input.key);
    if (existing) {
      if (existing.moduleKey !== input.moduleKey || existing.entryPoint !== input.entryPoint) {
        throw new Error(`WebGPU runtime pipeline cache key ${input.key} was reused with an incompatible entry point.`);
      }
      return { pipeline: existing.pipeline, cacheHit: true };
    }
    const module = this.getOrCreateShaderModule({
      device: input.device,
      key: input.moduleKey,
      code: input.code,
    });
    const pipeline = input.device.createComputePipeline({
      layout: "auto",
      compute: { module: module.module, entryPoint: input.entryPoint },
    });
    devicePipelines.set(input.key, {
      pipeline,
      moduleKey: input.moduleKey,
      entryPoint: input.entryPoint,
    });
    return { pipeline, cacheHit: false };
  }

  clear(): void {
    for (const entry of this.entries) destroyBuffer(entry.buffer);
    this.entries.clear();
    this.denseMatrixBuffers = new WeakMap();
    this.staticBuffers = new WeakMap();
    this.shaderModules = new WeakMap();
    this.computePipelines = new WeakMap();
  }

  private getOrCreateShaderModule(input: {
    device: MinimalGpuDevice;
    key: string;
    code: string;
  }): { module: unknown; cacheHit: boolean } {
    const deviceKey = input.device as object;
    let deviceModules = this.shaderModules.get(deviceKey);
    if (!deviceModules) {
      deviceModules = new Map();
      this.shaderModules.set(deviceKey, deviceModules);
    }
    const existing = deviceModules.get(input.key);
    if (existing) {
      if (existing.code !== input.code) {
        throw new Error(`WebGPU runtime shader cache key ${input.key} was reused with different WGSL code.`);
      }
      return { module: existing.module, cacheHit: true };
    }
    const module = input.device.createShaderModule({ code: input.code });
    deviceModules.set(input.key, { module, code: input.code });
    return { module, cacheHit: false };
  }
}

export async function preloadStableWebGpuRuntimeBuffers(
  input: WebGpuStableRuntimePreloadInput,
): Promise<WebGpuStableRuntimePreloadResult> {
  const device = await resolveGpuDevice(input.device, input.gpu);
  if (!device) {
    if (input.requireWebGpu) throw new Error("WebGPU is not available for stable runtime buffer preload.");
    throw new Error("stable runtime buffer preload requires WebGPU.");
  }
  const startedAt = nowMs();
  const entries: WebGpuStableRuntimePreloadEntry[] = [];

  for (const descriptor of input.matrices ?? []) {
    const sourceRows = denseMatVecRowCount(descriptor.matrix);
    const sourceCols = denseMatVecColCount(descriptor.matrix);
    const cols = descriptor.cols ?? sourceCols;
    const rowIds = descriptor.rowIds
      ? [...descriptor.rowIds]
      : allRowIds(descriptor.rows ?? sourceRows);
    if (cols !== sourceCols) {
      throw new Error(`stable runtime matrix preload ${descriptor.key} expected ${cols} cols but source has ${sourceCols}.`);
    }
    if (rowIds.some((rowId) => rowId < 0 || rowId >= sourceRows)) {
      throw new Error(`stable runtime matrix preload ${descriptor.key} contains row ids outside 0-${sourceRows - 1}.`);
    }
    const rows = rowIds.length;
    const byteLength = rows * cols * Float32Array.BYTES_PER_ELEMENT;
    const uploaded = input.bufferCache.getOrUploadFloatMatrix({
      device,
      key: descriptor.key,
      rows,
      cols,
      usage: GPU_STORAGE | GPU_COPY_DST,
      byteLength,
      dataFactory: () => denseMatVecMatrixToFloat32Array(descriptor.matrix, { cols }, rowIds),
    });
    entries.push({
      kind: "matrix",
      key: descriptor.key,
      rows,
      cols,
      byteLength,
      cacheHit: uploaded.cacheHit,
    });
  }

  for (const descriptor of input.staticBuffers ?? []) {
    const usage = descriptor.usage
      ?? (descriptor.usageKind === "uniform" ? GPU_UNIFORM | GPU_COPY_DST : GPU_STORAGE | GPU_COPY_DST);
    const uploaded = input.bufferCache.getOrUploadStaticBuffer({
      device,
      key: descriptor.key,
      usage,
      byteLength: descriptor.data.byteLength,
      dataFactory: () => descriptor.data,
    });
    entries.push({
      kind: "static",
      key: descriptor.key,
      byteLength: descriptor.data.byteLength,
      cacheHit: uploaded.cacheHit,
    });
  }

  for (const kind of input.pipelines ?? []) {
    const pipeline = getOrCreateRuntimePreloadPipeline(input.bufferCache, device, kind);
    entries.push({
      kind: "pipeline",
      key: runtimePipelinePreloadCacheKey(kind),
      cacheHit: pipeline.cacheHit,
    });
  }

  const cacheHits = entries.filter((entry) => entry.cacheHit).length;
  return {
    backend: "webgpu",
    computeMs: nowMs() - startedAt,
    entries,
    uploadedEntries: entries.length - cacheHits,
    cacheHits,
  };
}

function getOrCreateRuntimePreloadPipeline(
  bufferCache: WebGpuRuntimeBufferCache,
  device: MinimalGpuDevice,
  kind: WebGpuRuntimePipelinePreloadKind,
): { cacheHit: boolean } {
  const descriptor = runtimePipelinePreloadDescriptor(kind);
  return getOrCreateComputePipeline({
    device,
    bufferCache,
    key: descriptor.key,
    moduleKey: descriptor.moduleKey,
    code: descriptor.code,
    entryPoint: descriptor.entryPoint,
  });
}

function runtimePipelinePreloadCacheKey(kind: WebGpuRuntimePipelinePreloadKind): string {
  return runtimePipelinePreloadDescriptor(kind).key;
}

function runtimePipelinePreloadDescriptor(kind: WebGpuRuntimePipelinePreloadKind): {
  key: string;
  moduleKey: string;
  code: string;
  entryPoint: string;
} {
  switch (kind) {
    case "sparse-attention":
      return { key: "sparse-attention:main", moduleKey: "sparse-attention", code: sparseAttentionWgsl, entryPoint: "main" };
    case "packed-sparse-attention":
      return { key: "packed-sparse-attention:packed_sparse_attention", moduleKey: "packed-sparse-attention", code: packedSparseAttentionWgsl, entryPoint: "packed_sparse_attention" };
    case "token-embedding-lookup-resident":
      return { key: "token-embedding-lookup:resident", moduleKey: "token-embedding-lookup", code: tokenEmbeddingLookupWgsl, entryPoint: "token_embedding_lookup" };
    case "rmsnorm-resident":
      return { key: "rmsnorm:resident", moduleKey: "rmsnorm", code: rmsNormWgsl, entryPoint: "rms_norm" };
    case "residual-rmsnorm-pair-resident":
      return { key: "residual-rmsnorm-pair:resident", moduleKey: "residual-rmsnorm-pair", code: RESIDUAL_RMSNORM_ONE_TOKEN_WGSL, entryPoint: "residual_rmsnorm_one_token" };
    case "residual-add-resident":
      return { key: "residual-add:resident", moduleKey: "residual-add", code: residualAddWgsl, entryPoint: "residual_add" };
    case "qwen-qkv-post-projection-resident":
      return { key: "qwen-qkv-post-projection:resident", moduleKey: "qwen-qkv-post-projection", code: qwenQkvPostProjectionWgsl, entryPoint: "qwen_qkv_post_projection" };
    case "qwen-qkv-norm-rope-pair-resident":
      return { key: "qwen-qkv-norm-rope-pair:resident", moduleKey: "qwen-qkv-norm-rope-pair", code: QWEN_QKV_NORM_ROPE_PAIR_WGSL, entryPoint: "qwen_qkv_norm_rope_pair" };
    case "packed-qkv-projection-resident":
      return { key: "packed-qkv-projection:resident", moduleKey: "packed-qkv-projection", code: PACKED_QKV_PROJECTION_WGSL, entryPoint: "packed_qkv_projection" };
    case "dense-matvec":
      return { key: "dense-matvec:main", moduleKey: "dense-matvec", code: denseMatVecWgsl, entryPoint: "main" };
    case "dense-matvec-top1-candidates":
      return { key: "dense-matvec-top1:dense_matvec_top1_candidates", moduleKey: "dense-matvec-top1", code: denseMatVecTop1CandidatesWgsl, entryPoint: "dense_matvec_top1_candidates" };
    case "dense-matvec-top1-reduce":
      return { key: "dense-matvec-top1:dense_matvec_top1_reduce", moduleKey: "dense-matvec-top1-reduce", code: denseMatVecTop1ReduceWgsl, entryPoint: "dense_matvec_top1_reduce" };
    case "dense-matvec-compact-topk-scores":
      return { key: "dense-matvec-compact-topk:dense_matvec_compact_topk_scores", moduleKey: "dense-matvec-compact-topk-scores", code: denseMatVecCompactTopKScoresWgsl, entryPoint: "dense_matvec_compact_topk_scores" };
    case "dense-matvec-compact-topk-reduce":
      return { key: "dense-matvec-compact-topk:dense_matvec_compact_topk_reduce", moduleKey: "dense-matvec-compact-topk-reduce", code: denseMatVecCompactTopKReduceWgsl, entryPoint: "dense_matvec_compact_topk_reduce" };
    case "dense-matmul":
      return { key: "dense-matmul:dense_matmul", moduleKey: "dense-matmul", code: denseMatMulWgsl, entryPoint: "dense_matmul" };
    case "mlp-batch-intermediate":
      return { key: "mlp-batch:mlp_batch_intermediate", moduleKey: "mlp-batch", code: mlpBatchWgsl, entryPoint: "mlp_batch_intermediate" };
    case "mlp-batch-output":
      return { key: "mlp-batch:mlp_batch_output", moduleKey: "mlp-batch", code: mlpBatchWgsl, entryPoint: "mlp_batch_output" };
    default: {
      const exhaustive: never = kind;
      throw new Error(`Unknown WebGPU runtime pipeline preload kind: ${exhaustive}`);
    }
  }
}

/**
 * Reference backend boundary for the future WebGPU SSA kernel path.
 *
 * This class intentionally does not claim native LLM SSA support yet. It can
 * execute tiny toy tensor handles through the WebGPU/CPU sparse-attention
 * kernel path, while real model-layer handles still require a backend exposing
 * Q/K/V tensors and KV ownership.
 */
export class WebGpuSsaReferenceBackend implements SSAKernelBackend {
  readonly config: SSAWebGpuConfig;
  private readonly fallback = new FallbackSSARuntime();

  constructor(private readonly options: WebGpuSsaBackendOptions = {}) {
    this.config = { ...DEFAULT_SSA_WEBGPU_CONFIG, ...options.config };
  }

  supportsNativeSSA(): boolean {
    return false;
  }

  async planSparseAttention(input: SSAPlanInput): Promise<SSAPlan> {
    const plan = await this.fallback.plan({
      ...input,
      blockSize: this.config.blockSize,
      topKBlocks: this.config.topKBlocks,
      localWindowBlocks: this.config.localWindowBlocks,
    });
    return {
      ...plan,
      mode: "webgpu_reference",
    };
  }

  async executeSparseForward(input: SparseForwardInput): Promise<SparseForwardOutput> {
    const q = readSsaToyTensorHandle(input.qHandle);
    const k = readSsaToyTensorHandle(input.kHandle);
    const v = readSsaToyTensorHandle(input.vHandle);
    const selectedKeyIndexesByQuery = buildSelectedKeyIndexesByQuery(q, k, input.routingPolicy);
    const startedAt = nowMs();
    const result = await runSparseAttentionWebGpu({
      q: q.matrix,
      k: k.matrix,
      v: v.matrix,
      selectedKeyIndexesByQuery,
      ...(this.options.backendPreference ? { backendPreference: this.options.backendPreference } : {}),
      ...(this.options.gpu ? { gpu: this.options.gpu } : {}),
      ...(this.options.device ? { device: this.options.device } : {}),
      ...(this.options.requireWebGpu ? { requireWebGpu: true } : {}),
    });
    const selectedBlockIds = collectSelectedBlockIds(input.routingPolicy.selectedBlockIdsByQueryBlock);
    const sparseTokenCountEstimate = estimateSparseTokensByQueryBlock(k, input.routingPolicy.selectedBlockIdsByQueryBlock);
    const outputHandle = createSsaToyTensorHandle({
      id: `ssa_out_${input.requestId}_${input.layerIndex}`,
      matrix: result.output,
      blockTokenRanges: q.blockTokenRanges,
    });
    const attentionMs = result.trace.computeMs || nowMs() - startedAt;
    const trace = {
      requestId: input.requestId,
      layerIndex: input.layerIndex,
      queryBlockIndex: 0,
      selectedBlockIds,
      pinnedBlockIds: input.routingPolicy.pinnedBlockIds,
      denseTokenCountEstimate: k.matrix.length,
      sparseTokenCountEstimate,
      routingMs: 0,
      gatherMs: 0,
      attentionMs,
      attentionBackend: result.backend,
      packedHeadBackends: [result.backend],
      packedHeadCount: 1,
    } as SSAKernelTrace & {
      attentionBackend: WebGpuSparseAttentionResult["backend"];
      packedHeadBackends: Array<WebGpuSparseAttentionResult["backend"]>;
      packedHeadCount: number;
    };
    return {
      requestId: input.requestId,
      layerIndex: input.layerIndex,
      outputHandle,
      selectedBlockIds,
      timingMs: { routingMs: 0, gatherMs: 0, attentionMs },
      trace,
    };
  }
}

export function createSsaToyTensorHandle(input: {
  id: string;
  matrix: Matrix;
  blockTokenRanges: Record<string, SsaToyTensorRange>;
}): SsaToyTensorHandle {
  validateMatrix(input.matrix, input.id);
  return {
    kind: "ssa_toy_tensor",
    id: input.id,
    matrix: input.matrix.map((row) => [...row]),
    blockTokenRanges: normalizeRanges(input.blockTokenRanges),
  };
}

export function readSsaToyTensorHandle(handle: unknown): SsaToyTensorHandle {
  if (!isToyTensorHandle(handle)) {
    throw new Error("WebGpuSsaReferenceBackend requires ssa_toy_tensor handles for toy sparse-forward execution.");
  }
  validateMatrix(handle.matrix, handle.id);
  return handle;
}

export async function runSparseAttentionWebGpu(input: WebGpuSparseAttentionInput): Promise<WebGpuSparseAttentionResult> {
  validateSparseAttentionInput(input);
  if (input.backendPreference === "cpu") {
    if (input.requireWebGpu) throw new Error("WebGPU is required for SSA sparse attention but backendPreference=cpu was requested.");
    return runSparseAttentionCpu(input);
  }

  const device = await resolveGpuDevice(input.device, input.gpu);
  if (!device) {
    if (input.requireWebGpu) throw new Error("WebGPU is not available for SSA sparse attention.");
    return runSparseAttentionCpu(input);
  }

  const startedAt = nowMs();
  const gpuResult = await runSparseAttentionOnGpuDevice(device, input);
  return {
    output: gpuResult.output,
    backend: "webgpu",
    trace: buildSparseAttentionTrace("webgpu", input, nowMs() - startedAt, gpuResult.pipelineCacheHit),
  };
}

export async function runPackedSparseAttentionWebGpu(input: WebGpuPackedSparseAttentionInput): Promise<WebGpuPackedSparseAttentionResult> {
  const shape = validatePackedSparseAttentionInput(input);
  if (input.backendPreference === "cpu") {
    if (input.requireWebGpu) throw new Error("WebGPU is required for packed SSA sparse attention but backendPreference=cpu was requested.");
    return runPackedSparseAttentionCpu(input, shape);
  }

  const device = await resolveGpuDevice(input.device, input.gpu);
  if (!device) {
    if (input.requireWebGpu) throw new Error("WebGPU is not available for packed SSA sparse attention.");
    return runPackedSparseAttentionCpu(input, shape);
  }

  const startedAt = nowMs();
  const gpuResult = await runPackedSparseAttentionOnGpuDevice(device, input, shape);
  return {
    output: gpuResult.output,
    backend: "webgpu",
    trace: withDecodeOptimizedPackedTrace(
      buildPackedSparseAttentionTrace("webgpu", input, shape, nowMs() - startedAt, gpuResult.pipelineCacheHit),
      gpuResult,
    ),
  };
}

export async function runPackedSparseAttentionResidentWebGpu(input: WebGpuPackedSparseAttentionResidentInput): Promise<WebGpuPackedSparseAttentionResidentResult> {
  const shape = validatePackedSparseAttentionResidentInput(input);
  const qResident = isWebGpuResidentTensor(input.q) ? input.q : undefined;
  const kResident = isWebGpuResidentTensor(input.k) ? input.k : undefined;
  const vResident = isWebGpuResidentTensor(input.v) ? input.v : undefined;
  const device = await resolveGpuDevice(input.device ?? qResident?.device ?? kResident?.device ?? vResident?.device, input.gpu);
  if (!device) throw new Error("WebGPU is not available for resident packed SSA sparse attention.");
  for (const [name, tensor] of [["q", qResident], ["k", kResident], ["v", vResident]] as const) {
    if (tensor && tensor.device !== device) throw new Error(`resident packed sparse attention ${name} tensor belongs to a different WebGPU device.`);
  }

  const startedAt = nowMs();
  const gpuResult = await runPackedSparseAttentionResidentOnGpuDevice(device, input, shape);
  return {
    tensor: gpuResult.tensor,
    backend: "webgpu",
    trace: (() => {
      const packedTrace = withDecodeOptimizedPackedTrace(
        buildPackedSparseAttentionTrace("webgpu", input, shape, nowMs() - startedAt, gpuResult.pipelineCacheHit),
        gpuResult,
      );
      return {
        ...packedTrace,
      outputResident: true,
      readback: false,
      inputResident: {
        q: Boolean(qResident),
        k: Boolean(kResident),
        v: Boolean(vResident),
      },
        ...(packedTrace.metadata || input.traceMetadata
          ? { metadata: { ...(packedTrace.metadata ?? {}), ...(input.traceMetadata ?? {}) } }
          : {}),
      };
    })(),
  };
}

export async function uploadWebGpuResidentTensor(input: WebGpuResidentTensorUploadInput): Promise<WebGpuResidentTensorResult> {
  const shape = validateActivationMatrixShape(input.matrix, "WebGPU resident tensor upload");
  const device = await resolveGpuDevice(input.device, input.gpu);
  if (!device) throw new Error("WebGPU is not available for resident tensor upload.");
  const startedAt = nowMs();
  const byteLength = shape.tokens * shape.hidden * Float32Array.BYTES_PER_ELEMENT;
  const buffer = createUploadedBuffer(
    device,
    new Float32Array(flattenMatrix(input.matrix)),
    GPU_STORAGE | GPU_COPY_DST | GPU_COPY_SRC,
  );
  return {
    tensor: createWebGpuResidentTensor({
      device,
      buffer,
      rows: shape.tokens,
      cols: shape.hidden,
      byteLength,
      retainedBuffers: [],
    }),
    backend: "webgpu",
    trace: buildResidentTensorTrace("webgpu-resident-upload", shape, nowMs() - startedAt, false, input.traceMetadata),
  };
}

export async function createWebGpuResidentRowCache(input: {
  matrix: Matrix;
  capacityRows?: number;
  gpu?: unknown;
  device?: unknown;
  requireWebGpu?: boolean;
}): Promise<WebGpuResidentRowCache> {
  const shape = validateActivationMatrixShape(input.matrix, "WebGPU resident row cache seed");
  const device = await resolveGpuDevice(input.device, input.gpu);
  if (!device) throw new Error("WebGPU is not available for resident row cache.");
  const capacityRows = Math.max(shape.tokens, Math.floor(input.capacityRows ?? Math.max(8, shape.tokens * 2)));
  const byteLength = capacityRows * shape.hidden * Float32Array.BYTES_PER_ELEMENT;
  const buffer = device.createBuffer({ size: alignTo4(byteLength), usage: GPU_STORAGE | GPU_COPY_SRC | GPU_COPY_DST });
  const seed = new Float32Array(flattenMatrix(input.matrix));
  if (seed.byteLength > 0) device.queue.writeBuffer(buffer, 0, seed.buffer, seed.byteOffset, seed.byteLength);
  const tensor = createWebGpuResidentTensor({
    device,
    buffer,
    rows: shape.tokens,
    cols: shape.hidden,
    byteLength: shape.tokens * shape.hidden * Float32Array.BYTES_PER_ELEMENT,
    retainedBuffers: [],
  });
  return {
    kind: "webgpu_resident_row_cache",
    tensor,
    rows: shape.tokens,
    cols: shape.hidden,
    capacityRows,
  };
}

export async function appendWebGpuResidentRowCache(input: {
  cache: WebGpuResidentRowCache;
  rows: Matrix | WebGpuResidentTensor;
  gpu?: unknown;
  device?: unknown;
  requireWebGpu?: boolean;
  commandBatch?: WebGpuDecodeCommandBatchLike;
}): Promise<WebGpuResidentRowCache> {
  const cacheShape = validateResidentTensor(input.cache.tensor);
  if (cacheShape.hidden !== input.cache.cols || input.cache.rows !== cacheShape.tokens) {
    throw new Error("WebGPU resident row cache shape does not match its tensor view.");
  }
  const device = await resolveGpuDevice(input.device ?? input.cache.tensor.device, input.gpu);
  if (!device) throw new Error("WebGPU is not available for resident row cache append.");
  if (input.cache.tensor.device !== device) {
    throw new Error("WebGPU resident row cache append requires the cache tensor device.");
  }
  const rowTensor = isWebGpuResidentTensor(input.rows) ? input.rows : undefined;
  const rowShape = rowTensor
    ? validateResidentTensor(rowTensor)
    : validateActivationMatrixShape(input.rows as Matrix, "WebGPU resident row cache append rows");
  if (rowTensor && rowTensor.device !== device) {
    throw new Error("WebGPU resident row cache append rows belong to a different WebGPU device.");
  }
  if (rowShape.hidden !== input.cache.cols) {
    throw new Error(`WebGPU resident row cache append hidden size ${rowShape.hidden} does not match cache cols ${input.cache.cols}.`);
  }
  const newRows = input.cache.rows + rowShape.tokens;
  const rowByteLength = rowShape.tokens * input.cache.cols * Float32Array.BYTES_PER_ELEMENT;
  const rowDestinationOffset = input.cache.rows * input.cache.cols * Float32Array.BYTES_PER_ELEMENT;
  const retainedBuffers = residentRetainedBuffers(input.cache.tensor);

  if (newRows <= input.cache.capacityRows) {
    if (rowTensor) {
      submitOrRecordCopy(device, input.commandBatch, {
        label: "resident-row-cache:append",
        record: (encoder) => {
          encoder.copyBufferToBuffer(readResidentTensorBuffer(rowTensor), 0, readResidentTensorBuffer(input.cache.tensor), rowDestinationOffset, rowByteLength);
        },
      });
    } else {
      const rowData = new Float32Array(flattenMatrix(input.rows as Matrix));
      device.queue.writeBuffer(readResidentTensorBuffer(input.cache.tensor), rowDestinationOffset, rowData.buffer, rowData.byteOffset, rowData.byteLength);
    }
    return {
      kind: "webgpu_resident_row_cache",
      tensor: createWebGpuResidentTensor({
        device,
        buffer: readResidentTensorBuffer(input.cache.tensor),
        rows: newRows,
        cols: input.cache.cols,
        byteLength: newRows * input.cache.cols * Float32Array.BYTES_PER_ELEMENT,
        retainedBuffers,
      }),
      rows: newRows,
      cols: input.cache.cols,
      capacityRows: input.cache.capacityRows,
    };
  }

  const newCapacityRows = Math.max(newRows, input.cache.capacityRows * 2);
  const newBufferByteLength = newCapacityRows * input.cache.cols * Float32Array.BYTES_PER_ELEMENT;
  const newBuffer = device.createBuffer({ size: alignTo4(newBufferByteLength), usage: GPU_STORAGE | GPU_COPY_SRC | GPU_COPY_DST });
  submitOrRecordCopy(device, input.commandBatch, {
    label: "resident-row-cache:grow",
    record: (encoder) => {
      encoder.copyBufferToBuffer(readResidentTensorBuffer(input.cache.tensor), 0, newBuffer, 0, input.cache.tensor.byteLength);
      if (rowTensor) {
        encoder.copyBufferToBuffer(readResidentTensorBuffer(rowTensor), 0, newBuffer, rowDestinationOffset, rowByteLength);
      }
    },
  });
  if (!rowTensor) {
    const rowData = new Float32Array(flattenMatrix(input.rows as Matrix));
    device.queue.writeBuffer(newBuffer, rowDestinationOffset, rowData.buffer, rowData.byteOffset, rowData.byteLength);
  }
  return {
    kind: "webgpu_resident_row_cache",
    tensor: createWebGpuResidentTensor({
      device,
      buffer: newBuffer,
      rows: newRows,
      cols: input.cache.cols,
      byteLength: newRows * input.cache.cols * Float32Array.BYTES_PER_ELEMENT,
      retainedBuffers: [readResidentTensorBuffer(input.cache.tensor), ...retainedBuffers],
    }),
    rows: newRows,
    cols: input.cache.cols,
    capacityRows: newCapacityRows,
  };
}

export function destroyWebGpuResidentRowCache(cache: WebGpuResidentRowCache | undefined): void {
  if (!cache) return;
  destroyWebGpuResidentTensor(cache.tensor);
}

export async function runTokenEmbeddingLookupResidentWebGpu(
  input: WebGpuTokenEmbeddingLookupResidentInput,
): Promise<WebGpuTokenEmbeddingLookupResidentResult> {
  const shape = validateTokenEmbeddingLookupInput(input);
  const device = await resolveGpuDevice(input.device, input.gpu);
  if (!device) throw new Error("WebGPU is not available for resident token embedding lookup.");

  const startedAt = nowMs();
  const gpuResult = await runTokenEmbeddingLookupResidentOnGpuDevice(device, input, shape);
  return {
    tensor: gpuResult.tensor,
    backend: "webgpu",
    trace: {
      ...buildResidentTensorTrace(
        "token-embedding-lookup:resident",
        { tokens: 1, hidden: shape.hidden },
        nowMs() - startedAt,
        true,
        input.traceMetadata,
        gpuResult.pipelineCacheHit,
      ),
      tokenId: shape.tokenId,
      vocabSize: shape.vocabSize,
      embeddingTileStart: gpuResult.embeddingTileStart,
      embeddingTileRows: gpuResult.embeddingTileRows,
      ...(input.projectionCacheKey
        ? {
            embeddingCacheKey: input.projectionCacheKey,
            embeddingCacheHit: gpuResult.embeddingCacheHit,
          }
        : {}),
    },
  };
}

async function runTokenEmbeddingLookupResidentOnGpuDevice(
  device: MinimalGpuDevice,
  input: WebGpuTokenEmbeddingLookupResidentInput,
  shape: { tokenId: number; vocabSize: number; hidden: number },
): Promise<{
  tensor: WebGpuResidentTensor;
  embeddingCacheHit: boolean;
  pipelineCacheHit: boolean;
  embeddingTileStart: number;
  embeddingTileRows: number;
}> {
  const embeddingTile = planTokenEmbeddingTile(shape, input.embeddingTileRows);
  const embeddingRowIds = rowIdsInRange(embeddingTile.start, embeddingTile.end);
  const embeddingByteLength = embeddingTile.rows * shape.hidden * Float32Array.BYTES_PER_ELEMENT;
  const embeddingDataFactory = () => denseMatVecMatrixToFloat32Array(
    input.tokenEmbedding,
    { cols: shape.hidden },
    embeddingRowIds,
  );
  const cachedEmbedding = input.bufferCache
    && input.projectionCacheKey
    && input.projectionCachePolicy === "stable"
    && !Array.isArray(input.tokenEmbedding)
    ? input.bufferCache.getOrUploadFloatMatrix({
        device,
        key: tokenEmbeddingTileCacheKey(input.projectionCacheKey, embeddingTile),
        rows: embeddingTile.rows,
        cols: shape.hidden,
        usage: GPU_STORAGE | GPU_COPY_DST,
        byteLength: embeddingByteLength,
        dataFactory: embeddingDataFactory,
      })
    : null;
  const embeddingBuffer = cachedEmbedding?.buffer
    ?? createUploadedBuffer(device, embeddingDataFactory(), GPU_STORAGE | GPU_COPY_DST);
  const paramsBuffer = createUploadedBuffer(
    device,
    new Uint32Array([shape.tokenId - embeddingTile.start, shape.hidden, embeddingTile.rows, 0]),
    GPU_UNIFORM | GPU_COPY_DST,
  );
  const outputByteLength = shape.hidden * Float32Array.BYTES_PER_ELEMENT;
  const outputBuffer = device.createBuffer({ size: outputByteLength, usage: GPU_STORAGE | GPU_COPY_SRC });

  try {
    const pipeline = getOrCreateComputePipeline({
      device,
      bufferCache: input.bufferCache,
      key: "token-embedding-lookup:resident",
      moduleKey: "token-embedding-lookup",
      code: tokenEmbeddingLookupWgsl,
      entryPoint: "token_embedding_lookup",
    });
    const bindGroup = device.createBindGroup({
      layout: pipeline.pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: embeddingBuffer } },
        { binding: 1, resource: { buffer: outputBuffer } },
        { binding: 2, resource: { buffer: paramsBuffer } },
      ],
    });
    submitOrRecordComputePass(device, input.commandBatch, {
      label: "token-embedding-lookup:resident",
      dispatches: 1,
      record: (encoder) => {
        const pass = encoder.beginComputePass();
        pass.setPipeline(pipeline.pipeline);
        pass.setBindGroup(0, bindGroup);
        pass.dispatchWorkgroups(Math.ceil(shape.hidden / 64));
        pass.end();
      },
    });
    const retainedBuffers: MinimalGpuBuffer[] = [paramsBuffer];
    if (!cachedEmbedding) retainedBuffers.push(embeddingBuffer);
    return {
      tensor: createWebGpuResidentTensor({
        device,
        buffer: outputBuffer,
        rows: 1,
        cols: shape.hidden,
        byteLength: outputByteLength,
        retainedBuffers,
      }),
      embeddingCacheHit: cachedEmbedding?.cacheHit ?? false,
      pipelineCacheHit: pipeline.cacheHit,
      embeddingTileStart: embeddingTile.start,
      embeddingTileRows: embeddingTile.rows,
    };
  } catch (error) {
    if (!cachedEmbedding) destroyBuffer(embeddingBuffer);
    destroyBuffer(paramsBuffer);
    destroyBuffer(outputBuffer);
    throw error;
  }
}

export async function runRmsNormResidentWebGpu(input: WebGpuRmsNormResidentInput): Promise<WebGpuResidentTensorResult> {
  const hiddenShape = isWebGpuResidentTensor(input.hidden)
    ? validateResidentTensor(input.hidden)
    : validateActivationMatrixShape(input.hidden, "resident RMSNorm hidden");
  if (input.weight && input.weight.length !== hiddenShape.hidden) {
    throw new Error(`resident RMSNorm weight length ${input.weight.length} must match hidden size ${hiddenShape.hidden}.`);
  }
  const residentInput = isWebGpuResidentTensor(input.hidden) ? input.hidden : undefined;
  const device = await resolveGpuDevice(input.device ?? residentInput?.device, input.gpu);
  if (!device) throw new Error("WebGPU is not available for resident RMSNorm.");
  if (residentInput && residentInput.device !== device) throw new Error("resident RMSNorm hidden tensor belongs to a different WebGPU device.");

  const startedAt = nowMs();
  const gpuResult = await runRmsNormResidentOnGpuDevice(device, input, hiddenShape);
  return {
    tensor: gpuResult.tensor,
    backend: "webgpu",
    trace: buildResidentTensorTrace(
      "rmsnorm:resident",
      hiddenShape,
      nowMs() - startedAt,
      Boolean(residentInput),
      input.traceMetadata,
      gpuResult.pipelineCacheHit,
    ),
  };
}

export async function runResidualAddResidentWebGpu(input: WebGpuResidualAddResidentInput): Promise<WebGpuResidentTensorResult> {
  const shape = validateResidentBinaryInput(input.left, input.right, "resident residual add");
  const leftResident = isWebGpuResidentTensor(input.left) ? input.left : undefined;
  const rightResident = isWebGpuResidentTensor(input.right) ? input.right : undefined;
  const device = await resolveGpuDevice(input.device ?? leftResident?.device ?? rightResident?.device, input.gpu);
  if (!device) throw new Error("WebGPU is not available for resident residual add.");
  if (leftResident && leftResident.device !== device) throw new Error("resident residual-add left tensor belongs to a different WebGPU device.");
  if (rightResident && rightResident.device !== device) throw new Error("resident residual-add right tensor belongs to a different WebGPU device.");

  const startedAt = nowMs();
  const gpuResult = await runResidualAddResidentOnGpuDevice(device, input, shape);
  return {
    tensor: gpuResult.tensor,
    backend: "webgpu",
    trace: buildResidentTensorTrace(
      "residual-add:resident",
      shape,
      nowMs() - startedAt,
      Boolean(leftResident || rightResident),
      input.traceMetadata,
      gpuResult.pipelineCacheHit,
    ),
  };
}

export async function runResidualRmsNormPairResidentWebGpu(
  input: WebGpuResidualRmsNormPairResidentInput,
): Promise<WebGpuResidualRmsNormPairResidentResult> {
  const shape = validateResidualRmsNormPairResidentInput(input);
  const leftResident = isWebGpuResidentTensor(input.left) ? input.left : undefined;
  const rightResident = isWebGpuResidentTensor(input.right) ? input.right : undefined;
  const device = await resolveGpuDevice(input.device ?? leftResident?.device ?? rightResident?.device, input.gpu);
  if (!device) throw new Error("WebGPU is not available for resident residual RMSNorm fusion.");
  if (leftResident && leftResident.device !== device) throw new Error("resident residual RMSNorm left tensor belongs to a different WebGPU device.");
  if (rightResident && rightResident.device !== device) throw new Error("resident residual RMSNorm right tensor belongs to a different WebGPU device.");

  const startedAt = nowMs();
  const gpuResult = await runResidualRmsNormPairResidentOnGpuDevice(device, input, shape);
  return {
    summed: gpuResult.summed,
    normed: gpuResult.normed,
    backend: "webgpu",
    trace: {
      backend: "webgpu",
      tokens: shape.tokens,
      hidden: shape.hidden,
      computeMs: nowMs() - startedAt,
      outputResident: true,
      readback: false,
      inputResident: {
        left: Boolean(leftResident),
        right: Boolean(rightResident),
      },
      dispatchCount: 1,
      pipelineCacheKey: "residual-rmsnorm-pair:resident",
      pipelineCacheHit: gpuResult.pipelineCacheHit,
      metadata: {
        ...(input.traceMetadata ?? {}),
        fusedStage: "residual_rmsnorm",
        residualRmsNormPair: true,
      },
    },
  };
}

export async function runQwenQkvPostProjectionResidentWebGpu(input: WebGpuQkvPostProjectionResidentInput): Promise<WebGpuResidentTensorResult> {
  const shape = validateQkvPostProjectionResidentInput(input);
  const residentProjected = isWebGpuResidentTensor(input.projected) ? input.projected : undefined;
  const device = await resolveGpuDevice(input.device ?? residentProjected?.device, input.gpu);
  if (!device) throw new Error("WebGPU is not available for resident Qwen Q/K post-projection.");
  if (residentProjected && residentProjected.device !== device) {
    throw new Error("resident Qwen Q/K post-projection tensor belongs to a different WebGPU device.");
  }

  const startedAt = nowMs();
  const gpuResult = await runQwenQkvPostProjectionResidentOnGpuDevice(device, input, shape);
  return {
    tensor: gpuResult.tensor,
    backend: "webgpu",
    trace: buildResidentTensorTrace(
      "qwen-qkv-post-projection:resident",
      shape,
      nowMs() - startedAt,
      Boolean(residentProjected),
      input.traceMetadata,
      gpuResult.pipelineCacheHit,
    ),
  };
}

export async function runQwenQkvNormRopePairResidentWebGpu(
  input: WebGpuQkvNormRopePairResidentInput,
): Promise<WebGpuQkvNormRopePairResidentResult> {
  const shape = validateQkvNormRopePairResidentInput(input);
  const qResident = isWebGpuResidentTensor(input.qProjected) ? input.qProjected : undefined;
  const kResident = isWebGpuResidentTensor(input.kProjected) ? input.kProjected : undefined;
  const device = await resolveGpuDevice(input.device ?? qResident?.device ?? kResident?.device, input.gpu);
  if (!device) throw new Error("WebGPU is not available for resident Qwen Q/K norm+RoPE fusion.");
  if (qResident && qResident.device !== device) {
    throw new Error("resident Qwen Q/K norm+RoPE Q tensor belongs to a different WebGPU device.");
  }
  if (kResident && kResident.device !== device) {
    throw new Error("resident Qwen Q/K norm+RoPE K tensor belongs to a different WebGPU device.");
  }

  const startedAt = nowMs();
  const gpuResult = await runQwenQkvNormRopePairResidentOnGpuDevice(device, input, shape);
  const metadata = {
    ...(input.traceMetadata ?? {}),
    fusedStage: "qkv_norm_rope_kv_append",
    qkvNormRopePair: true,
  };
  return {
    q: gpuResult.q,
    k: gpuResult.k,
    backend: "webgpu",
    trace: {
      backend: "webgpu",
      tokens: shape.tokens,
      headDim: input.headDim,
      qHeadCount: input.qHeadCount,
      kHeadCount: input.kHeadCount,
      qHidden: shape.qHidden,
      kHidden: shape.kHidden,
      computeMs: nowMs() - startedAt,
      outputResident: true,
      readback: false,
      inputResident: {
        q: Boolean(qResident),
        k: Boolean(kResident),
      },
      dispatchCount: 1,
      pipelineCacheKey: "qwen-qkv-norm-rope-pair:resident",
      pipelineCacheHit: gpuResult.pipelineCacheHit,
      metadata,
    },
  };
}

export async function runSparseAttentionResidentWebGpu(input: WebGpuSparseAttentionResidentInput): Promise<WebGpuSparseAttentionResidentResult> {
  const shape = validateSparseAttentionResidentInput(input);
  const qResident = isWebGpuResidentTensor(input.q) ? input.q : undefined;
  const kResident = isWebGpuResidentTensor(input.k) ? input.k : undefined;
  const vResident = isWebGpuResidentTensor(input.v) ? input.v : undefined;
  const device = await resolveGpuDevice(input.device ?? qResident?.device ?? kResident?.device ?? vResident?.device, input.gpu);
  if (!device) throw new Error("WebGPU is not available for resident SSA sparse attention.");
  for (const [name, tensor] of [["q", qResident], ["k", kResident], ["v", vResident]] as const) {
    if (tensor && tensor.device !== device) throw new Error(`resident sparse attention ${name} tensor belongs to a different WebGPU device.`);
  }

  const startedAt = nowMs();
  const gpuResult = await runSparseAttentionResidentOnGpuDevice(device, input, shape);
  return {
    tensor: gpuResult.tensor,
    backend: "webgpu",
    trace: {
      ...buildSparseAttentionTraceFromShape("webgpu", input, shape, nowMs() - startedAt, gpuResult.pipelineCacheHit),
      outputResident: true,
      readback: false,
      inputResident: {
        q: Boolean(qResident),
        k: Boolean(kResident),
        v: Boolean(vResident),
      },
      ...(input.traceMetadata ? { metadata: { ...input.traceMetadata } } : {}),
    },
  };
}

export async function runDenseMatVecWebGpu(input: WebGpuDenseMatVecInput): Promise<WebGpuDenseMatVecResult> {
  const shape = validateDenseMatVecInput(input);
  if (input.backendPreference === "cpu") {
    if (input.requireWebGpu) throw new Error("WebGPU is required for dense matvec but backendPreference=cpu was requested.");
    return runDenseMatVecCpu(input, shape);
  }

  const device = await resolveGpuDevice(input.device, input.gpu);
  if (!device) {
    if (input.requireWebGpu) throw new Error("WebGPU is not available for dense matvec.");
    return runDenseMatVecCpu(input, shape);
  }

  const startedAt = nowMs();
  const selectedRowIds = normalizeDenseMatVecRowIds(input.selectedRowIds, shape.rows);
  const gpuResult = await runDenseMatVecOnGpuDevice(device, input, shape, selectedRowIds);
  return {
    values: gpuResult.values,
    ...(input.selectedRowIds ? { selectedRowIds } : {}),
    backend: "webgpu",
    trace: buildDenseMatVecTrace(
      "webgpu",
      input,
      shape,
      selectedRowIds.length,
      nowMs() - startedAt,
      gpuResult.projectionCacheHit,
      gpuResult.pipelineCacheHit,
    ),
  };
}

export async function runDenseMatVecResidentWebGpu(input: WebGpuDenseMatVecResidentInput): Promise<WebGpuDenseMatVecResidentResult> {
  const shape = validateDenseMatVecResidentInput(input);
  const device = await resolveGpuDevice(input.device ?? input.vector.device, input.gpu);
  if (!device) throw new Error("WebGPU is not available for resident dense matvec.");
  if (input.vector.device !== device) throw new Error("resident dense matvec vector belongs to a different WebGPU device.");

  const startedAt = nowMs();
  const selectedRowIds = normalizeDenseMatVecRowIds(input.selectedRowIds, shape.rows);
  const gpuResult = await runDenseMatVecResidentOnGpuDevice(device, input, shape, selectedRowIds);
  return {
    values: gpuResult.values,
    ...(input.selectedRowIds ? { selectedRowIds } : {}),
    backend: "webgpu",
    trace: {
      ...buildDenseMatVecTrace(
        "webgpu",
        input,
        shape,
        selectedRowIds.length,
        nowMs() - startedAt,
        gpuResult.projectionCacheHit,
        gpuResult.pipelineCacheHit,
      ),
      vectorResident: true,
    },
  };
}

export async function runDenseMatVecTopKWebGpu(input: WebGpuDenseMatVecTopKInput): Promise<WebGpuDenseMatVecTopKResult> {
  if (!Number.isInteger(input.topK) || input.topK <= 0) {
    throw new Error("dense matvec topK must be a positive integer.");
  }
  const shape = validateDenseMatVecInput(input);
  const selectedInputRows = normalizeDenseMatVecRowIds(input.selectedRowIds, shape.rows);
  const suppressedRows = new Set(normalizeDenseMatVecSuppressedRowIds(input.suppressedRowIds, shape.rows));
  const selectableRowCount = selectedInputRows.filter((rowId) => !suppressedRows.has(rowId)).length;
  if (selectableRowCount <= 0) throw new Error("dense matvec topK has no unsuppressed rows to select.");
  const tileRows = normalizeDenseMatVecTileRows(input.tileRows, selectedInputRows.length);
  const topK = Math.min(input.topK, selectableRowCount);
  const ranked: Array<{ value: number; rowId: number }> = [];
  const traces: Array<WebGpuDenseMatVecTopKResult["trace"]> = [];
  const backends: WebGpuDenseMatVecTrace["backend"][] = [];

  for (let tileStart = 0; tileStart < selectedInputRows.length; tileStart += tileRows) {
    const tileInputRows = selectedInputRows.slice(tileStart, tileStart + tileRows);
    const tileInput = makeDenseMatVecTileInput(input, shape, tileInputRows, tileStart, tileRows);
    const reduced = topK === 1 && tileInput.backendPreference !== "cpu"
      ? await tryRunDenseMatVecTop1CandidatesWebGpu(tileInput, shape, tileInputRows)
      : null;
    const projected = reduced ?? await runDenseMatVecWebGpu(tileInput);
    traces.push(reduced?.trace ?? toFullReadbackTopKTrace(projected.trace, input.topK, tileRows, tileInputRows.length, projected.values.length));
    backends.push(projected.backend);
    const projectedRowIds = projected.selectedRowIds ?? tileInputRows;
    mergeDenseMatVecTopK(ranked, projected.values.map((value, index) => ({
      value: Number.isFinite(value) ? value : Number.NEGATIVE_INFINITY,
      rowId: projectedRowIds[index] ?? tileInputRows[index] ?? tileStart + index,
    })).filter((candidate) => !suppressedRows.has(candidate.rowId)), topK);
  }

  const computeMs = traces.reduce((sum, trace) => sum + trace.computeMs, 0);
  const cacheHits = traces
    .map((trace) => trace.projectionCacheHit)
    .filter((value): value is boolean => value !== undefined);
  const pipelineCacheHits = traces
    .map((trace) => trace.pipelineCacheHit)
    .filter((value): value is boolean => value !== undefined);
  const readbackStrategy = summarizeReadbackStrategy(traces.map((trace) => trace.readbackStrategy));
  const gpuReducedRows = sumTraceField(traces, "gpuReducedRows");
  const readbackRows = sumTraceField(traces, "readbackRows");
  const readbackBytes = sumTraceField(traces, "readbackBytes");
  const dispatchCount = sumTraceField(traces, "dispatchCount");
  return {
    values: ranked.map((item) => item.value),
    selectedRowIds: ranked.map((item) => item.rowId),
    backend: summarizeDenseMatVecTraceBackends(backends),
    trace: {
      backend: summarizeDenseMatVecTraceBackends(backends),
      rows: shape.rows,
      cols: shape.cols,
      selectedRows: ranked.length,
      computeMs,
      ...(input.projectionCacheKey
        ? {
            projectionCacheKey: input.projectionCacheKey,
            projectionCacheHit: cacheHits.length > 0 ? cacheHits.every(Boolean) : false,
          }
        : {}),
      ...(pipelineCacheHits.length > 0
        ? {
            pipelineCacheKey: "dense-matvec:main",
            pipelineCacheHit: pipelineCacheHits.every(Boolean),
          }
        : {}),
      topK: input.topK,
      tileRows,
      tiles: traces.length,
      scannedRows: selectedInputRows.length,
      materializedRows: ranked.length,
      ...(readbackStrategy ? { readbackStrategy } : {}),
      ...(gpuReducedRows !== undefined ? { gpuReducedRows } : {}),
      ...(readbackRows !== undefined ? { readbackRows } : {}),
      ...(readbackBytes !== undefined ? { readbackBytes } : {}),
      ...(dispatchCount !== undefined ? { dispatchCount } : {}),
      metadata: {
        ...(input.traceMetadata ?? {}),
        topKSelection: true,
        tiledTopK: true,
        suppressedRowCount: suppressedRows.size,
        ...(readbackStrategy === "gpu_top1_candidates" ? { gpuTopKReduction: true } : {}),
      },
      ...(suppressedRows.size > 0 ? { suppressedRowCount: suppressedRows.size } : {}),
    },
  };
}

export async function runDenseMatVecTopKResidentWebGpu(input: WebGpuDenseMatVecTopKResidentInput): Promise<WebGpuDenseMatVecTopKResidentResult> {
  if (!Number.isInteger(input.topK) || input.topK <= 0) {
    throw new Error("resident dense matvec topK must be a positive integer.");
  }
  const shape = validateDenseMatVecResidentInput(input);
  const selectedInputRows = normalizeDenseMatVecRowIds(input.selectedRowIds, shape.rows);
  const suppressedRows = new Set(normalizeDenseMatVecSuppressedRowIds(input.suppressedRowIds, shape.rows));
  const selectableRowCount = selectedInputRows.filter((rowId) => !suppressedRows.has(rowId)).length;
  if (selectableRowCount <= 0) throw new Error("resident dense matvec topK has no unsuppressed rows to select.");
  const tileRows = normalizeDenseMatVecTileRows(input.tileRows, selectedInputRows.length);
  const topK = Math.min(input.topK, selectableRowCount);
  if (topK === 1 && (selectedInputRows.length > tileRows || input.forceFinalTopKReduction === true)) {
    return await tryRunDenseMatVecTop1CandidatesResidentTiledWebGpu(input, shape, selectedInputRows, tileRows);
  }
  if (topK > 1) {
    return await runDenseMatVecCompactTopKTilesResidentWebGpu(input, shape, selectedInputRows, tileRows, topK);
  }
  const ranked: Array<{ value: number; rowId: number }> = [];
  const traces: Array<WebGpuDenseMatVecTopKResult["trace"] & { vectorResident?: true }> = [];

  for (let tileStart = 0; tileStart < selectedInputRows.length; tileStart += tileRows) {
    const tileInputRows = selectedInputRows.slice(tileStart, tileStart + tileRows);
    const tileInput = makeDenseMatVecResidentTileInput(input, shape, tileInputRows, tileStart, tileRows);
    const reduced = topK === 1
      ? await tryRunDenseMatVecTop1CandidatesResidentWebGpu(tileInput, shape, tileInputRows)
      : null;
    const projected = reduced ?? await runDenseMatVecResidentWebGpu(tileInput);
    traces.push(reduced?.trace ?? toFullReadbackTopKTrace(projected.trace, input.topK, tileRows, tileInputRows.length, projected.values.length, true));
    const projectedRowIds = projected.selectedRowIds ?? tileInputRows;
    mergeDenseMatVecTopK(ranked, projected.values.map((value, index) => ({
      value: Number.isFinite(value) ? value : Number.NEGATIVE_INFINITY,
      rowId: projectedRowIds[index] ?? tileInputRows[index] ?? tileStart + index,
    })).filter((candidate) => !suppressedRows.has(candidate.rowId)), topK);
  }

  const computeMs = traces.reduce((sum, trace) => sum + trace.computeMs, 0);
  const cacheHits = traces
    .map((trace) => trace.projectionCacheHit)
    .filter((value): value is boolean => value !== undefined);
  const pipelineCacheHits = traces
    .map((trace) => trace.pipelineCacheHit)
    .filter((value): value is boolean => value !== undefined);
  const readbackStrategy = summarizeReadbackStrategy(traces.map((trace) => trace.readbackStrategy));
  const gpuReducedRows = sumTraceField(traces, "gpuReducedRows");
  const readbackRows = sumTraceField(traces, "readbackRows");
  const readbackBytes = sumTraceField(traces, "readbackBytes");
  const dispatchCount = sumTraceField(traces, "dispatchCount");
  return {
    values: ranked.map((item) => item.value),
    selectedRowIds: ranked.map((item) => item.rowId),
    backend: "webgpu",
    trace: {
      backend: "webgpu",
      rows: shape.rows,
      cols: shape.cols,
      selectedRows: ranked.length,
      computeMs,
      ...(input.projectionCacheKey
        ? {
            projectionCacheKey: input.projectionCacheKey,
            projectionCacheHit: cacheHits.length > 0 ? cacheHits.every(Boolean) : false,
          }
        : {}),
      ...(pipelineCacheHits.length > 0
        ? {
            pipelineCacheKey: "dense-matvec:main",
            pipelineCacheHit: pipelineCacheHits.every(Boolean),
          }
        : {}),
      topK: input.topK,
      tileRows,
      tiles: traces.length,
      scannedRows: selectedInputRows.length,
      materializedRows: ranked.length,
      vectorResident: true,
      ...(readbackStrategy ? { readbackStrategy } : {}),
      ...(gpuReducedRows !== undefined ? { gpuReducedRows } : {}),
      ...(readbackRows !== undefined ? { readbackRows } : {}),
      ...(readbackBytes !== undefined ? { readbackBytes } : {}),
      ...(dispatchCount !== undefined ? { dispatchCount } : {}),
      metadata: {
        ...(input.traceMetadata ?? {}),
        topKSelection: true,
        tiledTopK: true,
        suppressedRowCount: suppressedRows.size,
        ...(readbackStrategy === "gpu_top1_candidates" ? { gpuTopKReduction: true } : {}),
      },
      ...(suppressedRows.size > 0 ? { suppressedRowCount: suppressedRows.size } : {}),
    },
  };
}

type DenseMatVecTop1CandidateTrace = WebGpuDenseMatVecTopKResult["trace"];

interface DenseMatVecTop1CandidateResult {
  values: number[];
  selectedRowIds: number[];
  backend: "webgpu";
  trace: DenseMatVecTop1CandidateTrace & { vectorResident?: true };
}

function toFullReadbackTopKTrace(
  trace: WebGpuDenseMatVecTrace & { vectorResident?: true },
  topK: number,
  tileRows: number,
  scannedRows: number,
  materializedRows: number,
  vectorResident = false,
): WebGpuDenseMatVecTopKResult["trace"] & { vectorResident?: true } {
  return {
    ...trace,
    topK,
    tileRows,
    tiles: 1,
    scannedRows,
    materializedRows,
    readbackStrategy: "full_logits",
    readbackRows: scannedRows,
    readbackBytes: scannedRows * Float32Array.BYTES_PER_ELEMENT,
    dispatchCount: 1,
    ...(vectorResident || trace.vectorResident ? { vectorResident: true } : {}),
  };
}

async function tryRunDenseMatVecTop1CandidatesWebGpu(
  input: WebGpuDenseMatVecInput,
  shape: { rows: number; cols: number },
  selectedRowIds: number[],
): Promise<DenseMatVecTop1CandidateResult | null> {
  const device = await resolveGpuDevice(input.device, input.gpu);
  if (!device) {
    if (input.requireWebGpu) throw new Error("WebGPU is not available for compact dense matvec top-1.");
    return null;
  }
  const vectorBuffer = createUploadedBuffer(device, new Float32Array(Array.from(input.vector)), GPU_STORAGE | GPU_COPY_DST);
  try {
    return await runDenseMatVecTop1CandidatesOnGpuDevice(device, input, shape, selectedRowIds, vectorBuffer, false);
  } finally {
    destroyBuffer(vectorBuffer);
  }
}

async function tryRunDenseMatVecTop1CandidatesResidentWebGpu(
  input: WebGpuDenseMatVecResidentInput,
  shape: { rows: number; cols: number },
  selectedRowIds: number[],
): Promise<DenseMatVecTop1CandidateResult | null> {
  const device = await resolveGpuDevice(input.device ?? input.vector.device, input.gpu);
  if (!device) throw new Error("WebGPU is not available for compact resident dense matvec top-1.");
  if (input.vector.device !== device) throw new Error("compact resident dense matvec top-1 vector belongs to a different WebGPU device.");
  return await runDenseMatVecTop1CandidatesOnGpuDevice(device, input, shape, selectedRowIds, readResidentTensorBuffer(input.vector), true);
}

async function tryRunDenseMatVecTop1CandidatesResidentTiledWebGpu(
  input: WebGpuDenseMatVecTopKResidentInput,
  shape: { rows: number; cols: number },
  selectedInputRows: number[],
  tileRows: number,
): Promise<WebGpuDenseMatVecTopKResidentResult> {
  const device = await resolveGpuDevice(input.device ?? input.vector.device, input.gpu);
  if (!device) throw new Error("WebGPU is not available for compact resident tiled dense matvec top-1.");
  if (input.vector.device !== device) throw new Error("compact resident tiled dense matvec top-1 vector belongs to a different WebGPU device.");
  const reduced = await runDenseMatVecTop1CandidateTilesOnGpuDevice(
    device,
    input,
    shape,
    selectedInputRows,
    tileRows,
    readResidentTensorBuffer(input.vector),
  );
  return {
    values: reduced.values,
    selectedRowIds: reduced.selectedRowIds,
    backend: "webgpu",
    trace: {
      ...reduced.trace,
      vectorResident: true,
    },
  };
}

interface DenseMatVecCompactTopKTileContext {
  matrixBuffer: MinimalGpuBuffer;
  rowIdsBuffer: MinimalGpuBuffer;
  paramsBuffer: MinimalGpuBuffer;
  outputRows: number;
  cachedMatrix: boolean;
  projectionCacheHit: boolean;
}

async function runDenseMatVecCompactTopKTilesResidentWebGpu(
  input: WebGpuDenseMatVecTopKResidentInput,
  shape: { rows: number; cols: number },
  selectedInputRows: number[],
  tileRows: number,
  topK: number,
): Promise<WebGpuDenseMatVecTopKResidentResult> {
  const device = await resolveGpuDevice(input.device ?? input.vector.device, input.gpu);
  if (!device) throw new Error("WebGPU is not available for compact resident tiled dense matvec top-k.");
  if (input.vector.device !== device) throw new Error("compact resident tiled dense matvec top-k vector belongs to a different WebGPU device.");
  const reduced = await runDenseMatVecCompactTopKTilesOnGpuDevice(
    device,
    input,
    shape,
    selectedInputRows,
    tileRows,
    topK,
    readResidentTensorBuffer(input.vector),
  );
  return {
    values: reduced.values,
    selectedRowIds: reduced.selectedRowIds,
    backend: "webgpu",
    trace: {
      ...reduced.trace,
      vectorResident: true,
    },
  };
}

async function runDenseMatVecCompactTopKTilesOnGpuDevice(
  device: MinimalGpuDevice,
  input: WebGpuDenseMatVecTopKResidentInput,
  shape: { rows: number; cols: number },
  selectedInputRows: number[],
  tileRows: number,
  topK: number,
  vectorBuffer: MinimalGpuBuffer,
): Promise<DenseMatVecTop1CandidateResult> {
  const startedAt = nowMs();
  const tileContexts: DenseMatVecCompactTopKTileContext[] = [];
  const temporaryBuffers: MinimalGpuBuffer[] = [];
  const suppressedRowIds = normalizeDenseMatVecSuppressedRowIds(input.suppressedRowIds, shape.rows);
  const suppressedRowIdsUpload = createSuppressedRowIdsBuffer(device, input.bufferCache, suppressedRowIds);
  const suppressedRowIdsBuffer = suppressedRowIdsUpload.buffer;
  let scoredRowsTotal = 0;
  let tileCount = 0;
  let scorePairsBuffer: MinimalGpuBuffer | undefined;
  let finalPairsBuffer: MinimalGpuBuffer | undefined;
  let readPairsBuffer: MinimalGpuBuffer | undefined;
  let reduceParamsBuffer: MinimalGpuBuffer | undefined;
  let reduceParamsCached = false;

  try {
    for (let tileStart = 0; tileStart < selectedInputRows.length; tileStart += tileRows) {
      const tileInputRows = selectedInputRows.slice(tileStart, tileStart + tileRows);
      const tileInput = makeDenseMatVecResidentTileInput(input, shape, tileInputRows, tileStart, tileRows);
      const outputRows = tileInputRows.length;
      const matrixByteLength = outputRows * shape.cols * Float32Array.BYTES_PER_ELEMENT;
      const matrixDataFactory = () => denseMatVecMatrixToFloat32Array(tileInput.matrix, shape, tileInputRows);
      const cachedMatrix = canCacheDenseMatVecProjection(tileInput)
        ? tileInput.bufferCache.getOrUploadFloatMatrix({
            device,
            key: `dense-matvec:${tileInput.projectionCacheKey}`,
            rows: outputRows,
            cols: shape.cols,
            usage: GPU_STORAGE | GPU_COPY_DST,
            byteLength: matrixByteLength,
            dataFactory: matrixDataFactory,
          })
        : null;
      const matrixBuffer = cachedMatrix?.buffer
        ?? createUploadedBuffer(device, matrixDataFactory(), GPU_STORAGE | GPU_COPY_DST);
      const tileDescriptorCache = canCacheDenseMatVecProjection(tileInput)
        ? {
            bufferCache: tileInput.bufferCache,
            projectionCacheKey: tileInput.projectionCacheKey,
          }
        : null;
      const cachedRowIds = tileDescriptorCache
        ? tileDescriptorCache.bufferCache.getOrUploadStaticBuffer({
            device,
            key: `dense-matvec-compact-topk-rowids:${tileDescriptorCache.projectionCacheKey}`,
            usage: GPU_STORAGE | GPU_COPY_DST,
            byteLength: tileInputRows.length * Int32Array.BYTES_PER_ELEMENT,
            dataFactory: () => new Int32Array(tileInputRows),
          })
        : null;
      const params = new Uint32Array([outputRows, shape.cols, scoredRowsTotal, suppressedRowIds.length]);
      const cachedParams = tileDescriptorCache
        ? tileDescriptorCache.bufferCache.getOrUploadStaticBuffer({
            device,
            key: `dense-matvec-compact-topk-params:${tileDescriptorCache.projectionCacheKey}:cols:${shape.cols}:offset:${scoredRowsTotal}:suppress:${suppressedRowIdsCacheKey(suppressedRowIds)}`,
            usage: GPU_UNIFORM | GPU_COPY_DST,
            byteLength: params.byteLength,
            dataFactory: () => params,
          })
        : null;
      const rowIdsBuffer = cachedRowIds?.buffer
        ?? createUploadedBuffer(device, new Int32Array(tileInputRows), GPU_STORAGE | GPU_COPY_DST);
      const paramsBuffer = cachedParams?.buffer
        ?? createUploadedBuffer(device, params, GPU_UNIFORM | GPU_COPY_DST);
      tileContexts.push({
        matrixBuffer,
        rowIdsBuffer,
        paramsBuffer,
        outputRows,
        cachedMatrix: Boolean(cachedMatrix),
        projectionCacheHit: cachedMatrix?.cacheHit ?? false,
      });
      if (!cachedMatrix) temporaryBuffers.push(matrixBuffer);
      if (!cachedRowIds) temporaryBuffers.push(rowIdsBuffer);
      if (!cachedParams) temporaryBuffers.push(paramsBuffer);
      scoredRowsTotal += outputRows;
      tileCount += 1;
    }

    const scorePairsByteLength = Math.max(1, scoredRowsTotal) * 2 * Float32Array.BYTES_PER_ELEMENT;
    const finalPairsByteLength = Math.max(1, topK) * 2 * Float32Array.BYTES_PER_ELEMENT;
    scorePairsBuffer = device.createBuffer({ size: scorePairsByteLength, usage: GPU_STORAGE | GPU_COPY_SRC });
    finalPairsBuffer = device.createBuffer({ size: finalPairsByteLength, usage: GPU_STORAGE | GPU_COPY_SRC });
    readPairsBuffer = device.createBuffer({ size: finalPairsByteLength, usage: GPU_MAP_READ | GPU_COPY_DST });
    const reduceParams = new Uint32Array([scoredRowsTotal, topK, 0, 0]);
    const selectedStart = selectedInputRows[0] ?? 0;
    const selectedEnd = (selectedInputRows.at(-1) ?? selectedStart) + 1;
    const cachedReduceParams = input.bufferCache && input.projectionCacheKey && isContiguousDenseMatVecRowSelection(selectedInputRows)
      ? input.bufferCache.getOrUploadStaticBuffer({
          device,
          key: `dense-matvec-compact-topk-reduce-params:${input.projectionCacheKey}:rows:${selectedStart}-${selectedEnd}:tileRows:${tileRows}:scoredRows:${scoredRowsTotal}:topK:${topK}`,
          usage: GPU_UNIFORM | GPU_COPY_DST,
          byteLength: reduceParams.byteLength,
          dataFactory: () => reduceParams,
        })
      : null;
    reduceParamsBuffer = cachedReduceParams?.buffer
      ?? createUploadedBuffer(device, reduceParams, GPU_UNIFORM | GPU_COPY_DST);
    reduceParamsCached = Boolean(cachedReduceParams);

    const scoresPipeline = getOrCreateComputePipeline({
      device,
      bufferCache: input.bufferCache,
      key: "dense-matvec-compact-topk:dense_matvec_compact_topk_scores",
      moduleKey: "dense-matvec-compact-topk-scores",
      code: denseMatVecCompactTopKScoresWgsl,
      entryPoint: "dense_matvec_compact_topk_scores",
    });
    const reducePipeline = getOrCreateComputePipeline({
      device,
      bufferCache: input.bufferCache,
      key: "dense-matvec-compact-topk:dense_matvec_compact_topk_reduce",
      moduleKey: "dense-matvec-compact-topk-reduce",
      code: denseMatVecCompactTopKReduceWgsl,
      entryPoint: "dense_matvec_compact_topk_reduce",
    });
    const encoder = device.createCommandEncoder();
    const scorePass = encoder.beginComputePass();
    scorePass.setPipeline(scoresPipeline.pipeline);
    for (const tile of tileContexts) {
      const bindGroup = device.createBindGroup({
        layout: scoresPipeline.pipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: tile.matrixBuffer } },
          { binding: 1, resource: { buffer: vectorBuffer } },
          { binding: 2, resource: { buffer: tile.rowIdsBuffer } },
          { binding: 3, resource: { buffer: scorePairsBuffer } },
          { binding: 4, resource: { buffer: tile.paramsBuffer } },
          { binding: 5, resource: { buffer: suppressedRowIdsBuffer } },
        ],
      });
      scorePass.setBindGroup(0, bindGroup);
      scorePass.dispatchWorkgroups(Math.max(1, Math.ceil(tile.outputRows / 64)));
    }
    scorePass.end();

    const reduceBindGroup = device.createBindGroup({
      layout: reducePipeline.pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: scorePairsBuffer } },
        { binding: 1, resource: { buffer: finalPairsBuffer } },
        { binding: 2, resource: { buffer: reduceParamsBuffer } },
      ],
    });
    const reducePass = encoder.beginComputePass();
    reducePass.setPipeline(reducePipeline.pipeline);
    reducePass.setBindGroup(0, reduceBindGroup);
    reducePass.dispatchWorkgroups(1);
    reducePass.end();
    encoder.copyBufferToBuffer(finalPairsBuffer, 0, readPairsBuffer, 0, finalPairsByteLength);
    device.queue.submit([encoder.finish()]);

    await readPairsBuffer.mapAsync(GPU_MAP_READ);
    const candidatePairs = new Float32Array(readPairsBuffer.getMappedRange()).slice(0, topK * 2);
    readPairsBuffer.unmap();
    const ranked: Array<{ value: number; rowId: number }> = [];
    const seenReadbackRows = new Set<number>();
    mergeDenseMatVecTopK(ranked, Array.from({ length: topK }, (_value, index) => ({
      value: Number.isFinite(candidatePairs[index * 2] ?? Number.NEGATIVE_INFINITY)
        ? candidatePairs[index * 2] ?? Number.NEGATIVE_INFINITY
        : Number.NEGATIVE_INFINITY,
      rowId: Math.trunc(candidatePairs[index * 2 + 1] ?? -1),
    })).filter((item) => {
      if (item.rowId < 0 || seenReadbackRows.has(item.rowId)) return false;
      seenReadbackRows.add(item.rowId);
      return true;
    }), topK);
    if (ranked.length < topK && selectedInputRows.length > 0) {
      const seenRankedRows = new Set(ranked.map((item) => item.rowId));
      for (const rowId of selectedInputRows) {
        if (seenRankedRows.has(rowId)) continue;
        ranked.push({ value: Number.NEGATIVE_INFINITY, rowId });
        seenRankedRows.add(rowId);
        if (ranked.length >= topK) break;
      }
    }

    const projectionCacheHits = tileContexts.map((tile) => tile.projectionCacheHit);
    const trace: DenseMatVecTop1CandidateTrace & { vectorResident?: true } = {
      ...buildDenseMatVecTrace(
        "webgpu",
        input,
        shape,
        ranked.length,
        nowMs() - startedAt,
        projectionCacheHits.length > 0 ? projectionCacheHits.every(Boolean) : false,
        scoresPipeline.cacheHit && reducePipeline.cacheHit,
      ),
      topK,
      tileRows,
      tiles: tileCount,
      scannedRows: selectedInputRows.length,
      materializedRows: ranked.length,
      readbackStrategy: "gpu_compact_topk",
      gpuReducedRows: selectedInputRows.length,
      readbackRows: ranked.length,
      readbackBytes: finalPairsByteLength,
      dispatchCount: tileCount + 1,
      vectorResident: true,
      metadata: {
        ...(input.traceMetadata ?? {}),
        topKSelection: true,
        tiledTopK: true,
        gpuCompactTopKReduction: true,
        suppressedRowCount: suppressedRowIds.length,
      },
      ...(suppressedRowIds.length > 0 ? { suppressedRowCount: suppressedRowIds.length } : {}),
    };
    return {
      values: ranked.map((item) => item.value),
      selectedRowIds: ranked.map((item) => item.rowId),
      backend: "webgpu",
      trace,
    };
  } finally {
    for (const buffer of temporaryBuffers) destroyBuffer(buffer);
    if (scorePairsBuffer) destroyBuffer(scorePairsBuffer);
    if (finalPairsBuffer) destroyBuffer(finalPairsBuffer);
    if (readPairsBuffer) destroyBuffer(readPairsBuffer);
    if (reduceParamsBuffer && !reduceParamsCached) destroyBuffer(reduceParamsBuffer);
    if (!suppressedRowIdsUpload.cached) destroyBuffer(suppressedRowIdsBuffer);
  }
}

interface DenseMatVecTop1TileContext {
  matrixBuffer: MinimalGpuBuffer;
  rowIdsBuffer: MinimalGpuBuffer;
  paramsBuffer: MinimalGpuBuffer;
  outputRows: number;
  candidateRows: number;
  cachedMatrix: boolean;
  projectionCacheHit: boolean;
}

async function runDenseMatVecTop1CandidateTilesOnGpuDevice(
  device: MinimalGpuDevice,
  input: WebGpuDenseMatVecTopKResidentInput,
  shape: { rows: number; cols: number },
  selectedInputRows: number[],
  tileRows: number,
  vectorBuffer: MinimalGpuBuffer,
): Promise<DenseMatVecTop1CandidateResult> {
  const startedAt = nowMs();
  const tileContexts: DenseMatVecTop1TileContext[] = [];
  const temporaryBuffers: MinimalGpuBuffer[] = [];
  const suppressedRowIds = normalizeDenseMatVecSuppressedRowIds(input.suppressedRowIds, shape.rows);
  const suppressedRowIdsUpload = createSuppressedRowIdsBuffer(device, input.bufferCache, suppressedRowIds);
  const suppressedRowIdsBuffer = suppressedRowIdsUpload.buffer;
  let candidateRowsTotal = 0;
  let tileCount = 0;
  let candidatePairsBuffer: MinimalGpuBuffer | undefined;
  let finalPairBuffer: MinimalGpuBuffer | undefined;
  let readPairsBuffer: MinimalGpuBuffer | undefined;
  let reduceParamsBuffer: MinimalGpuBuffer | undefined;
  let reduceParamsCached = false;

  try {
    for (let tileStart = 0; tileStart < selectedInputRows.length; tileStart += tileRows) {
      const tileInputRows = selectedInputRows.slice(tileStart, tileStart + tileRows);
      const tileInput = makeDenseMatVecResidentTileInput(input, shape, tileInputRows, tileStart, tileRows);
      const outputRows = tileInputRows.length;
      const matrixByteLength = outputRows * shape.cols * Float32Array.BYTES_PER_ELEMENT;
      const matrixDataFactory = () => denseMatVecMatrixToFloat32Array(tileInput.matrix, shape, tileInputRows);
      const cachedMatrix = canCacheDenseMatVecProjection(tileInput)
        ? tileInput.bufferCache.getOrUploadFloatMatrix({
            device,
            key: `dense-matvec:${tileInput.projectionCacheKey}`,
            rows: outputRows,
            cols: shape.cols,
            usage: GPU_STORAGE | GPU_COPY_DST,
            byteLength: matrixByteLength,
            dataFactory: matrixDataFactory,
          })
        : null;
      const matrixBuffer = cachedMatrix?.buffer
        ?? createUploadedBuffer(device, matrixDataFactory(), GPU_STORAGE | GPU_COPY_DST);
      const candidateRows = Math.max(1, Math.ceil(outputRows / 64));
      const tileDescriptorCache = canCacheDenseMatVecProjection(tileInput)
        ? {
            bufferCache: tileInput.bufferCache,
            projectionCacheKey: tileInput.projectionCacheKey,
          }
        : null;
      const cachedRowIds = tileDescriptorCache
        ? tileDescriptorCache.bufferCache.getOrUploadStaticBuffer({
            device,
            key: `dense-matvec-top1-rowids:${tileDescriptorCache.projectionCacheKey}`,
            usage: GPU_STORAGE | GPU_COPY_DST,
            byteLength: tileInputRows.length * Int32Array.BYTES_PER_ELEMENT,
            dataFactory: () => new Int32Array(tileInputRows),
          })
        : null;
      const params = new Uint32Array([outputRows, shape.cols, candidateRowsTotal, suppressedRowIds.length]);
      const cachedParams = tileDescriptorCache
        ? tileDescriptorCache.bufferCache.getOrUploadStaticBuffer({
            device,
            key: `dense-matvec-top1-params:${tileDescriptorCache.projectionCacheKey}:cols:${shape.cols}:offset:${candidateRowsTotal}:candidateRows:${candidateRows}:suppress:${suppressedRowIdsCacheKey(suppressedRowIds)}`,
            usage: GPU_UNIFORM | GPU_COPY_DST,
            byteLength: params.byteLength,
            dataFactory: () => params,
          })
        : null;
      const rowIdsBuffer = cachedRowIds?.buffer
        ?? createUploadedBuffer(device, new Int32Array(tileInputRows), GPU_STORAGE | GPU_COPY_DST);
      const paramsBuffer = cachedParams?.buffer
        ?? createUploadedBuffer(device, params, GPU_UNIFORM | GPU_COPY_DST);
      tileContexts.push({
        matrixBuffer,
        rowIdsBuffer,
        paramsBuffer,
        outputRows,
        candidateRows,
        cachedMatrix: Boolean(cachedMatrix),
        projectionCacheHit: cachedMatrix?.cacheHit ?? false,
      });
      if (!cachedMatrix) temporaryBuffers.push(matrixBuffer);
      if (!cachedRowIds) temporaryBuffers.push(rowIdsBuffer);
      if (!cachedParams) temporaryBuffers.push(paramsBuffer);
      candidateRowsTotal += candidateRows;
      tileCount += 1;
    }

    const candidatePairsByteLength = Math.max(1, candidateRowsTotal) * 2 * Float32Array.BYTES_PER_ELEMENT;
    candidatePairsBuffer = device.createBuffer({ size: candidatePairsByteLength, usage: GPU_STORAGE | GPU_COPY_SRC });
    finalPairBuffer = device.createBuffer({ size: 2 * Float32Array.BYTES_PER_ELEMENT, usage: GPU_STORAGE | GPU_COPY_SRC });
    readPairsBuffer = device.createBuffer({ size: 2 * Float32Array.BYTES_PER_ELEMENT, usage: GPU_MAP_READ | GPU_COPY_DST });
    const reduceParams = new Uint32Array([candidateRowsTotal, 0, 0, 0]);
    const selectedStart = selectedInputRows[0] ?? 0;
    const selectedEnd = (selectedInputRows.at(-1) ?? selectedStart) + 1;
    const cachedReduceParams = input.bufferCache && input.projectionCacheKey && isContiguousDenseMatVecRowSelection(selectedInputRows)
      ? input.bufferCache.getOrUploadStaticBuffer({
          device,
          key: `dense-matvec-top1-reduce-params:${input.projectionCacheKey}:rows:${selectedStart}-${selectedEnd}:tileRows:${tileRows}:candidateRows:${candidateRowsTotal}`,
          usage: GPU_UNIFORM | GPU_COPY_DST,
          byteLength: reduceParams.byteLength,
          dataFactory: () => reduceParams,
        })
      : null;
    reduceParamsBuffer = cachedReduceParams?.buffer
      ?? createUploadedBuffer(device, reduceParams, GPU_UNIFORM | GPU_COPY_DST);
    reduceParamsCached = Boolean(cachedReduceParams);

    const candidatePipeline = getOrCreateComputePipeline({
      device,
      bufferCache: input.bufferCache,
      key: "dense-matvec-top1:dense_matvec_top1_candidates",
      moduleKey: "dense-matvec-top1",
      code: denseMatVecTop1CandidatesWgsl,
      entryPoint: "dense_matvec_top1_candidates",
    });
    const reducePipeline = getOrCreateComputePipeline({
      device,
      bufferCache: input.bufferCache,
      key: "dense-matvec-top1:dense_matvec_top1_reduce",
      moduleKey: "dense-matvec-top1-reduce",
      code: denseMatVecTop1ReduceWgsl,
      entryPoint: "dense_matvec_top1_reduce",
    });
    const encoder = device.createCommandEncoder();
    const candidatePass = encoder.beginComputePass();
    candidatePass.setPipeline(candidatePipeline.pipeline);
    for (const tile of tileContexts) {
      const bindGroup = device.createBindGroup({
        layout: candidatePipeline.pipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: tile.matrixBuffer } },
          { binding: 1, resource: { buffer: vectorBuffer } },
          { binding: 2, resource: { buffer: tile.rowIdsBuffer } },
          { binding: 3, resource: { buffer: candidatePairsBuffer } },
          { binding: 4, resource: { buffer: tile.paramsBuffer } },
          { binding: 5, resource: { buffer: suppressedRowIdsBuffer } },
        ],
      });
      candidatePass.setBindGroup(0, bindGroup);
      candidatePass.dispatchWorkgroups(tile.candidateRows);
    }
    candidatePass.end();

    const reduceBindGroup = device.createBindGroup({
      layout: reducePipeline.pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: candidatePairsBuffer } },
        { binding: 1, resource: { buffer: finalPairBuffer } },
        { binding: 2, resource: { buffer: reduceParamsBuffer } },
      ],
    });
    const reducePass = encoder.beginComputePass();
    reducePass.setPipeline(reducePipeline.pipeline);
    reducePass.setBindGroup(0, reduceBindGroup);
    reducePass.dispatchWorkgroups(1);
    reducePass.end();
    encoder.copyBufferToBuffer(finalPairBuffer, 0, readPairsBuffer, 0, 2 * Float32Array.BYTES_PER_ELEMENT);
    device.queue.submit([encoder.finish()]);

    await readPairsBuffer.mapAsync(GPU_MAP_READ);
    const candidatePair = new Float32Array(readPairsBuffer.getMappedRange()).slice();
    readPairsBuffer.unmap();
    const ranked: Array<{ value: number; rowId: number }> = [];
    mergeDenseMatVecTopK(ranked, [{
      value: Number.isFinite(candidatePair[0] ?? Number.NEGATIVE_INFINITY)
        ? candidatePair[0] ?? Number.NEGATIVE_INFINITY
        : Number.NEGATIVE_INFINITY,
      rowId: Math.trunc(candidatePair[1] ?? -1),
    }].filter((item) => item.rowId >= 0), 1);
    if (ranked.length === 0 && selectedInputRows.length > 0) {
      ranked.push({ value: Number.NEGATIVE_INFINITY, rowId: selectedInputRows[0] ?? 0 });
    }

    const projectionCacheHits = tileContexts.map((tile) => tile.projectionCacheHit);
    const trace: DenseMatVecTop1CandidateTrace & { vectorResident?: true } = {
      ...buildDenseMatVecTrace(
        "webgpu",
        input,
        shape,
        ranked.length,
        nowMs() - startedAt,
        projectionCacheHits.length > 0 ? projectionCacheHits.every(Boolean) : false,
        candidatePipeline.cacheHit && reducePipeline.cacheHit,
      ),
      topK: 1,
      tileRows,
      tiles: tileCount,
      scannedRows: selectedInputRows.length,
      materializedRows: ranked.length,
      readbackStrategy: "gpu_top1_candidates",
      gpuReducedRows: selectedInputRows.length,
      readbackRows: 1,
      readbackBytes: 2 * Float32Array.BYTES_PER_ELEMENT,
      dispatchCount: tileCount + 1,
      vectorResident: true,
      metadata: {
        ...(input.traceMetadata ?? {}),
        topKSelection: true,
        tiledTopK: true,
        gpuTopKReduction: true,
        suppressedRowCount: suppressedRowIds.length,
      },
      ...(suppressedRowIds.length > 0 ? { suppressedRowCount: suppressedRowIds.length } : {}),
    };
    return {
      values: ranked.map((item) => item.value),
      selectedRowIds: ranked.map((item) => item.rowId),
      backend: "webgpu",
      trace,
    };
  } finally {
    for (const buffer of temporaryBuffers) destroyBuffer(buffer);
    if (candidatePairsBuffer) destroyBuffer(candidatePairsBuffer);
    if (finalPairBuffer) destroyBuffer(finalPairBuffer);
    if (readPairsBuffer) destroyBuffer(readPairsBuffer);
    if (reduceParamsBuffer && !reduceParamsCached) destroyBuffer(reduceParamsBuffer);
    if (!suppressedRowIdsUpload.cached) destroyBuffer(suppressedRowIdsBuffer);
  }
}

async function runDenseMatVecTop1CandidatesOnGpuDevice(
  device: MinimalGpuDevice,
  input: Pick<
    WebGpuDenseMatVecInput,
    "matrix" | "bufferCache" | "projectionCacheKey" | "projectionCachePolicy" | "traceMetadata" | "suppressedRowIds"
  >,
  shape: { rows: number; cols: number },
  selectedRowIds: number[],
  vectorBuffer: MinimalGpuBuffer,
  vectorResident: boolean,
): Promise<DenseMatVecTop1CandidateResult> {
  const startedAt = nowMs();
  const outputRows = selectedRowIds.length;
  const matrixByteLength = outputRows * shape.cols * Float32Array.BYTES_PER_ELEMENT;
  const matrixDataFactory = () => denseMatVecMatrixToFloat32Array(input.matrix, shape, selectedRowIds);
  const cachedMatrix = canCacheDenseMatVecProjection(input)
    ? input.bufferCache.getOrUploadFloatMatrix({
        device,
        key: `dense-matvec:${input.projectionCacheKey}`,
        rows: outputRows,
        cols: shape.cols,
        usage: GPU_STORAGE | GPU_COPY_DST,
        byteLength: matrixByteLength,
        dataFactory: matrixDataFactory,
      })
    : null;
  const matrixBuffer = cachedMatrix?.buffer
    ?? createUploadedBuffer(device, matrixDataFactory(), GPU_STORAGE | GPU_COPY_DST);
  const rowIdsBuffer = createUploadedBuffer(device, new Int32Array(selectedRowIds), GPU_STORAGE | GPU_COPY_DST);
  const suppressedRowIds = normalizeDenseMatVecSuppressedRowIds(input.suppressedRowIds, shape.rows);
  const suppressedRowIdsUpload = createSuppressedRowIdsBuffer(device, input.bufferCache, suppressedRowIds);
  const suppressedRowIdsBuffer = suppressedRowIdsUpload.buffer;
  const paramsBuffer = createUploadedBuffer(
    device,
    new Uint32Array([outputRows, shape.cols, 0, suppressedRowIds.length]),
    GPU_UNIFORM | GPU_COPY_DST,
  );
  const candidateRows = Math.max(1, Math.ceil(outputRows / 64));
  const candidatePairsByteLength = candidateRows * 2 * Float32Array.BYTES_PER_ELEMENT;
  const candidatePairsBuffer = device.createBuffer({ size: candidatePairsByteLength, usage: GPU_STORAGE | GPU_COPY_SRC });
  const readPairsBuffer = device.createBuffer({ size: candidatePairsByteLength, usage: GPU_MAP_READ | GPU_COPY_DST });

  try {
    const pipeline = getOrCreateComputePipeline({
      device,
      bufferCache: input.bufferCache,
      key: "dense-matvec-top1:dense_matvec_top1_candidates",
      moduleKey: "dense-matvec-top1",
      code: denseMatVecTop1CandidatesWgsl,
      entryPoint: "dense_matvec_top1_candidates",
    });
    const bindGroup = device.createBindGroup({
      layout: pipeline.pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: matrixBuffer } },
        { binding: 1, resource: { buffer: vectorBuffer } },
        { binding: 2, resource: { buffer: rowIdsBuffer } },
        { binding: 3, resource: { buffer: candidatePairsBuffer } },
        { binding: 4, resource: { buffer: paramsBuffer } },
        { binding: 5, resource: { buffer: suppressedRowIdsBuffer } },
      ],
    });
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline.pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(candidateRows);
    pass.end();
    encoder.copyBufferToBuffer(candidatePairsBuffer, 0, readPairsBuffer, 0, candidatePairsByteLength);
    device.queue.submit([encoder.finish()]);
    await readPairsBuffer.mapAsync(GPU_MAP_READ);
    const candidatePairs = new Float32Array(readPairsBuffer.getMappedRange()).slice();
    readPairsBuffer.unmap();
    const ranked: Array<{ value: number; rowId: number }> = [];
    mergeDenseMatVecTopK(ranked, Array.from({ length: candidateRows }, (_value, index) => ({
      value: Number.isFinite(candidatePairs[index * 2] ?? Number.NEGATIVE_INFINITY)
        ? candidatePairs[index * 2] ?? Number.NEGATIVE_INFINITY
        : Number.NEGATIVE_INFINITY,
      rowId: Math.trunc(candidatePairs[index * 2 + 1] ?? -1),
    })).filter((item) => item.rowId >= 0), 1);
    if (ranked.length === 0 && selectedRowIds.length > 0) {
      ranked.push({ value: Number.NEGATIVE_INFINITY, rowId: selectedRowIds[0] ?? 0 });
    }
    const trace: DenseMatVecTop1CandidateTrace & { vectorResident?: true } = {
      ...buildDenseMatVecTrace(
        "webgpu",
        input,
        shape,
        ranked.length,
        nowMs() - startedAt,
        cachedMatrix?.cacheHit ?? false,
        pipeline.cacheHit,
      ),
      topK: 1,
      tileRows: outputRows,
      tiles: 1,
      scannedRows: outputRows,
      materializedRows: ranked.length,
      readbackStrategy: "gpu_top1_candidates",
      gpuReducedRows: outputRows,
      readbackRows: candidateRows,
      readbackBytes: candidatePairsByteLength,
      dispatchCount: 1,
      ...(vectorResident ? { vectorResident: true } : {}),
      metadata: {
        ...(input.traceMetadata ?? {}),
        topKSelection: true,
        tiledTopK: true,
        gpuTopKReduction: true,
        suppressedRowCount: suppressedRowIds.length,
      },
      ...(suppressedRowIds.length > 0 ? { suppressedRowCount: suppressedRowIds.length } : {}),
    };
    return {
      values: ranked.map((item) => item.value),
      selectedRowIds: ranked.map((item) => item.rowId),
      backend: "webgpu",
      trace,
    };
  } finally {
    if (!cachedMatrix) destroyBuffer(matrixBuffer);
    destroyBuffer(rowIdsBuffer);
    if (!suppressedRowIdsUpload.cached) destroyBuffer(suppressedRowIdsBuffer);
    destroyBuffer(paramsBuffer);
    destroyBuffer(candidatePairsBuffer);
    destroyBuffer(readPairsBuffer);
  }
}

function summarizeReadbackStrategy(values: Array<WebGpuDenseMatVecTopKResult["trace"]["readbackStrategy"]>): WebGpuDenseMatVecTopKResult["trace"]["readbackStrategy"] | undefined {
  const strategies = values.filter((value): value is NonNullable<typeof value> => value !== undefined);
  if (strategies.length === 0) return undefined;
  return strategies.every((strategy) => strategy === "gpu_top1_candidates")
    ? "gpu_top1_candidates"
    : strategies.every((strategy) => strategy === "gpu_argmax_token_id")
      ? "gpu_argmax_token_id"
      : strategies.every((strategy) => strategy === "gpu_compact_topk")
        ? "gpu_compact_topk"
        : "full_logits";
}

function sumTraceField<K extends "gpuReducedRows" | "readbackRows" | "readbackBytes" | "dispatchCount">(
  traces: Array<Partial<Pick<WebGpuDenseMatVecTopKResult["trace"], K>>>,
  key: K,
): number | undefined {
  const values = traces.map((trace) => trace[key]).filter((value): value is number => typeof value === "number" && Number.isFinite(value));
  return values.length > 0 ? values.reduce((sum, value) => sum + value, 0) : undefined;
}

function normalizeDenseMatVecTileRows(value: number | undefined, totalRows: number): number {
  if (value !== undefined) {
    if (!Number.isInteger(value) || value <= 0) throw new Error("dense matvec topK tileRows must be a positive integer.");
    return Math.min(value, totalRows);
  }
  return Math.min(DEFAULT_DENSE_MATVEC_TOPK_TILE_ROWS, totalRows);
}

function makeDenseMatVecTileInput(
  input: WebGpuDenseMatVecTopKInput,
  shape: { rows: number; cols: number },
  selectedRowIds: number[],
  tileStart: number,
  tileRows: number,
): WebGpuDenseMatVecInput {
  const projectionCacheKey = makeDenseMatVecTileCacheKey(input.projectionCacheKey, selectedRowIds);
  return {
    vector: input.vector,
    matrix: input.matrix,
    selectedRowIds,
    ...(input.backendPreference ? { backendPreference: input.backendPreference } : {}),
    ...(input.gpu ? { gpu: input.gpu } : {}),
    ...(input.device ? { device: input.device } : {}),
    ...(input.requireWebGpu ? { requireWebGpu: true } : {}),
    ...(input.suppressedRowIds ? { suppressedRowIds: input.suppressedRowIds } : {}),
    ...(input.bufferCache ? { bufferCache: input.bufferCache } : {}),
    ...(projectionCacheKey ? { projectionCacheKey } : {}),
    ...(projectionCacheKey && input.projectionCachePolicy ? { projectionCachePolicy: input.projectionCachePolicy } : {}),
    traceMetadata: {
      ...(input.traceMetadata ?? {}),
      tileStart,
      tileRows,
      rowStart: selectedRowIds[0] ?? 0,
      rowEnd: (selectedRowIds.at(-1) ?? -1) + 1,
      totalRows: shape.rows,
      tiledTopK: true,
    },
  };
}

function makeDenseMatVecResidentTileInput(
  input: WebGpuDenseMatVecTopKResidentInput,
  shape: { rows: number; cols: number },
  selectedRowIds: number[],
  tileStart: number,
  tileRows: number,
): WebGpuDenseMatVecResidentInput {
  const projectionCacheKey = makeDenseMatVecTileCacheKey(input.projectionCacheKey, selectedRowIds);
  return {
    vector: input.vector,
    matrix: input.matrix,
    selectedRowIds,
    ...(input.gpu ? { gpu: input.gpu } : {}),
    ...(input.device ? { device: input.device } : {}),
    ...(input.requireWebGpu ? { requireWebGpu: true } : {}),
    ...(input.suppressedRowIds ? { suppressedRowIds: input.suppressedRowIds } : {}),
    ...(input.bufferCache ? { bufferCache: input.bufferCache } : {}),
    ...(projectionCacheKey ? { projectionCacheKey } : {}),
    ...(projectionCacheKey && input.projectionCachePolicy ? { projectionCachePolicy: input.projectionCachePolicy } : {}),
    traceMetadata: {
      ...(input.traceMetadata ?? {}),
      tileStart,
      tileRows,
      rowStart: selectedRowIds[0] ?? 0,
      rowEnd: (selectedRowIds.at(-1) ?? -1) + 1,
      totalRows: shape.rows,
      tiledTopK: true,
    },
  };
}

function makeDenseMatVecTileCacheKey(projectionCacheKey: string | undefined, rowIds: number[]): string | undefined {
  if (!projectionCacheKey || rowIds.length === 0 || !isContiguousDenseMatVecRowSelection(rowIds)) return undefined;
  const firstRowId = rowIds[0];
  if (firstRowId === undefined) return undefined;
  return `${projectionCacheKey}:rows:${firstRowId}-${(rowIds.at(-1) ?? firstRowId) + 1}`;
}

function suppressedRowIdsCacheKey(rowIds: number[]): string {
  return rowIds.length > 0 ? rowIds.join(",") : "none";
}

function createSuppressedRowIdsBuffer(
  device: MinimalGpuDevice,
  bufferCache: WebGpuRuntimeBufferCache | undefined,
  rowIds: number[],
): { buffer: MinimalGpuBuffer; cached: boolean } {
  const data = new Int32Array(rowIds.length > 0 ? rowIds : [-1]);
  if (bufferCache) {
    const uploaded = bufferCache.getOrUploadStaticBuffer({
      device,
      key: `dense-matvec-top1-suppressed-rowids:${suppressedRowIdsCacheKey(rowIds)}`,
      usage: GPU_STORAGE | GPU_COPY_DST,
      byteLength: data.byteLength,
      dataFactory: () => data,
    });
    return { buffer: uploaded.buffer, cached: true };
  }
  return {
    buffer: createUploadedBuffer(device, data, GPU_STORAGE | GPU_COPY_DST),
    cached: false,
  };
}

function isContiguousDenseMatVecRowSelection(rowIds: number[]): boolean {
  for (let index = 1; index < rowIds.length; index += 1) {
    if ((rowIds[index] ?? 0) !== (rowIds[index - 1] ?? 0) + 1) return false;
  }
  return true;
}

function mergeDenseMatVecTopK(
  ranked: Array<{ value: number; rowId: number }>,
  candidates: Array<{ value: number; rowId: number }>,
  topK: number,
): void {
  ranked.push(...candidates);
  ranked.sort((left, right) => right.value - left.value || left.rowId - right.rowId);
  if (ranked.length > topK) ranked.length = topK;
}

function summarizeDenseMatVecTraceBackends(backends: WebGpuDenseMatVecTrace["backend"][]): WebGpuDenseMatVecTrace["backend"] {
  return backends.length > 0 && backends.every((backend) => backend === "webgpu") ? "webgpu" : "cpu_reference";
}

export async function runDenseMatMulWebGpu(input: WebGpuDenseMatMulInput): Promise<WebGpuDenseMatMulResult> {
  const shape = validateDenseMatMulInput(input);
  if (input.backendPreference === "cpu") {
    if (input.requireWebGpu) throw new Error("WebGPU is required for dense matmul but backendPreference=cpu was requested.");
    return runDenseMatMulCpu(input, shape);
  }

  const device = await resolveGpuDevice(input.device, input.gpu);
  if (!device) {
    if (input.requireWebGpu) throw new Error("WebGPU is not available for dense matmul.");
    return runDenseMatMulCpu(input, shape);
  }

  const startedAt = nowMs();
  const gpuResult = await runDenseMatMulOnGpuDevice(device, input, shape);
  return {
    output: gpuResult.output,
    backend: "webgpu",
    trace: buildDenseMatMulTrace(
      "webgpu",
      input,
      shape,
      nowMs() - startedAt,
      gpuResult.projectionCacheHit,
      gpuResult.pipelineCacheHit,
    ),
  };
}

export async function runDenseMatMulResidentWebGpu(input: WebGpuDenseMatMulResidentInput): Promise<WebGpuDenseMatMulResidentResult> {
  const shape = validateDenseMatMulResidentInput(input);
  if (input.backendPreference === "cpu") {
    throw new Error("GPU-resident dense matmul requires WebGPU; backendPreference=cpu was requested.");
  }

  const residentInput = isWebGpuResidentTensor(input.activations) ? input.activations : undefined;
  const device = await resolveGpuDevice(input.device ?? residentInput?.device, input.gpu);
  if (!device) throw new Error("WebGPU is not available for GPU-resident dense matmul.");
  if (residentInput && residentInput.device !== device) {
    throw new Error("GPU-resident dense matmul activations belong to a different WebGPU device.");
  }

  const startedAt = nowMs();
  const gpuResult = await runDenseMatMulResidentOnGpuDevice(device, input, shape);
  return {
    tensor: gpuResult.tensor,
    backend: "webgpu",
    trace: {
      ...buildDenseMatMulTrace(
        "webgpu",
        input,
        shape,
        nowMs() - startedAt,
        gpuResult.projectionCacheHit,
        gpuResult.pipelineCacheHit,
      ),
      outputResident: true,
      readback: false,
      inputResident: Boolean(residentInput),
    },
  };
}

export async function runPackedQkvProjectionResidentWebGpu(
  input: WebGpuPackedQkvProjectionResidentInput,
): Promise<WebGpuPackedQkvProjectionResidentResult> {
  const shape = validatePackedQkvProjectionResidentInput(input);
  const residentInput = isWebGpuResidentTensor(input.hidden) ? input.hidden : undefined;
  const device = await resolveGpuDevice(input.device ?? residentInput?.device, input.gpu);
  if (!device) throw new Error("WebGPU is not available for packed resident QKV projection.");
  if (residentInput && residentInput.device !== device) {
    throw new Error("packed resident QKV projection hidden tensor belongs to a different WebGPU device.");
  }

  const startedAt = nowMs();
  const gpuResult = await runPackedQkvProjectionResidentOnGpuDevice(device, input, shape);
  return {
    q: gpuResult.q,
    k: gpuResult.k,
    v: gpuResult.v,
    backend: "webgpu",
    trace: {
      backend: "webgpu",
      tokens: shape.tokens,
      hidden: shape.hidden,
      qOutputSize: shape.qOutputSize,
      kOutputSize: shape.kOutputSize,
      vOutputSize: shape.vOutputSize,
      outputSize: shape.qOutputSize + shape.kOutputSize + shape.vOutputSize,
      computeMs: nowMs() - startedAt,
      outputResident: true,
      readback: false,
      inputResident: Boolean(residentInput),
      dispatchCount: 1,
      projectionCacheHits: gpuResult.projectionCacheHits,
      pipelineCacheKey: "packed-qkv-projection:resident",
      pipelineCacheHit: gpuResult.pipelineCacheHit,
      metadata: {
        ...(input.traceMetadata ?? {}),
        fusedStage: "packed_qkv_projection",
      },
    },
  };
}

export async function readWebGpuResidentTensor(tensor: WebGpuResidentTensor): Promise<Matrix> {
  const [matrix] = await readWebGpuResidentTensors([tensor]);
  if (!matrix) throw new Error("WebGPU resident tensor readback returned no matrix.");
  return matrix;
}

export async function readWebGpuResidentTensors(tensors: WebGpuResidentTensor[]): Promise<Matrix[]> {
  if (tensors.length === 0) return [];
  const first = tensors[0];
  if (!first) return [];
  validateResidentTensor(first);
  const device = await resolveGpuDevice(first.device, undefined);
  if (!device) throw new Error("WebGPU resident tensors cannot be read without their original WebGPU device.");
  const reads = tensors.map((tensor) => {
    validateResidentTensor(tensor);
    if (tensor.device !== first.device) {
      throw new Error("Batched WebGPU resident tensor readback requires tensors from the same WebGPU device.");
    }
    return {
      sourceBuffer: readResidentTensorBuffer(tensor),
      rows: tensor.rows,
      cols: tensor.cols,
      byteLength: tensor.byteLength,
    };
  });
  return readGpuBuffersAsMatrices(device, reads);
}

export function destroyWebGpuResidentTensor(tensor: WebGpuResidentTensor): void {
  if (!isWebGpuResidentTensor(tensor)) return;
  destroyBuffer(readResidentTensorBuffer(tensor));
  for (const retained of tensor.retainedBuffers ?? []) {
    if (isGpuBuffer(retained)) destroyBuffer(retained);
  }
}

export async function runMlpWebGpu(input: WebGpuMlpInput): Promise<WebGpuMlpResult> {
  const shape = validateMlpInput(input);
  if (input.backendPreference === "cpu") {
    if (input.requireWebGpu) throw new Error("WebGPU is required for MLP but backendPreference=cpu was requested.");
    return runMlpCpu(input, shape);
  }

  const device = await resolveGpuDevice(input.device, input.gpu);
  if (!device) {
    if (input.requireWebGpu) throw new Error("WebGPU is not available for MLP.");
    return runMlpCpu(input, shape);
  }

  const startedAt = nowMs();
  const gpuResult = await runMlpOnGpuDevice(device, input, shape);
  return {
    values: gpuResult.values,
    backend: "webgpu",
    trace: buildMlpTrace(
      "webgpu",
      input,
      shape,
      nowMs() - startedAt,
      gpuResult.projectionCacheHits,
      gpuResult.pipelineCacheHits,
    ),
  };
}

export async function runMlpBatchWebGpu(input: WebGpuMlpBatchInput): Promise<WebGpuMlpBatchResult> {
  const shape = validateMlpBatchInput(input);
  if (input.backendPreference === "cpu") {
    if (input.requireWebGpu) throw new Error("WebGPU is required for batched MLP but backendPreference=cpu was requested.");
    return runMlpBatchCpu(input, shape);
  }

  const device = await resolveGpuDevice(input.device, input.gpu);
  if (!device) {
    if (input.requireWebGpu) throw new Error("WebGPU is not available for batched MLP.");
    return runMlpBatchCpu(input, shape);
  }

  const startedAt = nowMs();
  const gpuResult = await runMlpBatchOnGpuDevice(device, input, shape);
  return {
    output: gpuResult.output,
    backend: "webgpu",
    trace: buildMlpTrace(
      "webgpu",
      input,
      shape,
      nowMs() - startedAt,
      gpuResult.projectionCacheHits,
      gpuResult.pipelineCacheHits,
    ),
  };
}

export async function runMlpBatchResidentWebGpu(input: WebGpuMlpBatchResidentInput): Promise<WebGpuMlpBatchResidentResult> {
  const shape = validateMlpBatchResidentInput(input);
  const residentInput = isWebGpuResidentTensor(input.hidden) ? input.hidden : undefined;
  const device = await resolveGpuDevice(input.device ?? residentInput?.device, input.gpu);
  if (!device) throw new Error("WebGPU is not available for resident batched MLP.");
  if (residentInput && residentInput.device !== device) throw new Error("resident batched MLP hidden tensor belongs to a different WebGPU device.");

  const startedAt = nowMs();
  const gpuResult = await runMlpBatchResidentOnGpuDevice(device, input, shape);
  return {
    tensor: gpuResult.tensor,
    backend: "webgpu",
    trace: {
      ...buildMlpTrace(
        "webgpu",
        input,
        shape,
        nowMs() - startedAt,
        gpuResult.projectionCacheHits,
        gpuResult.pipelineCacheHits,
      ),
      outputResident: true,
      readback: false,
      inputResident: Boolean(residentInput),
    },
  };
}

function runSparseAttentionCpu(input: WebGpuSparseAttentionInput): WebGpuSparseAttentionResult {
  const startedAt = nowMs();
  const output = sparseReferenceAttention({
    q: input.q,
    k: input.k,
    v: input.v,
    selectedKeyIndexesByQuery: input.selectedKeyIndexesByQuery,
    ...(input.causal !== undefined ? { causal: input.causal } : {}),
    ...(input.scale !== undefined ? { scale: input.scale } : {}),
  });
  return {
    output,
    backend: "cpu_reference",
    trace: buildSparseAttentionTrace("cpu_reference", input, nowMs() - startedAt),
  };
}

function runPackedSparseAttentionCpu(
  input: WebGpuPackedSparseAttentionInput,
  shape = validatePackedSparseAttentionInput(input),
): WebGpuPackedSparseAttentionResult {
  const startedAt = nowMs();
  const output = input.q.map(() => new Array(shape.outputSize).fill(0));
  for (let headIndex = 0; headIndex < shape.headCount; headIndex += 1) {
    const kvHeadIndex = mapQueryHeadToKeyValueHead(headIndex, shape.headCount, shape.keyValueHeadCount);
    const headOutput = sparseReferenceAttention({
      q: slicePackedHeadMatrix(input.q, headIndex, shape.headDim),
      k: slicePackedHeadMatrix(input.k, kvHeadIndex, shape.headDim),
      v: slicePackedHeadMatrix(input.v, kvHeadIndex, shape.headDim),
      selectedKeyIndexesByQuery: input.selectedKeyIndexesByQuery,
      ...(input.causal !== undefined ? { causal: input.causal } : {}),
      ...(input.scale !== undefined ? { scale: input.scale } : {}),
    });
    const headOffset = headIndex * shape.headDim;
    for (let queryIndex = 0; queryIndex < output.length; queryIndex += 1) {
      const outputRow = output[queryIndex] ?? [];
      const headRow = headOutput[queryIndex] ?? [];
      for (let dim = 0; dim < shape.headDim; dim += 1) {
        outputRow[headOffset + dim] = headRow[dim] ?? 0;
      }
    }
  }
  return {
    output,
    backend: "cpu_reference",
    trace: buildPackedSparseAttentionTrace("cpu_reference", input, shape, nowMs() - startedAt),
  };
}

function runDenseMatVecCpu(input: WebGpuDenseMatVecInput, shape = validateDenseMatVecInput(input)): WebGpuDenseMatVecResult {
  const startedAt = nowMs();
  const selectedRowIds = normalizeDenseMatVecRowIds(input.selectedRowIds, shape.rows);
  const values = selectedRowIds.map((rowId) => dotArrayLike(readDenseMatVecRow(input.matrix, rowId, shape.cols), input.vector));
  return {
    values,
    ...(input.selectedRowIds ? { selectedRowIds } : {}),
    backend: "cpu_reference",
    trace: buildDenseMatVecTrace("cpu_reference", input, shape, selectedRowIds.length, nowMs() - startedAt),
  };
}

function runDenseMatMulCpu(input: WebGpuDenseMatMulInput, shape = validateDenseMatMulInput(input)): WebGpuDenseMatMulResult {
  const startedAt = nowMs();
  const output = input.activations.map((activation) => (
    allRowIds(shape.outputSize).map((rowId) => dotArrayLike(readDenseMatVecRow(input.projection, rowId, shape.hidden), activation))
  ));
  return {
    output,
    backend: "cpu_reference",
    trace: buildDenseMatMulTrace("cpu_reference", input, shape, nowMs() - startedAt, false),
  };
}

function runMlpCpu(input: WebGpuMlpInput, shape = validateMlpInput(input)): WebGpuMlpResult {
  const startedAt = nowMs();
  const up = multiplyDenseMatrixByVector(input.upProjection, shape.intermediateSize, shape.inputSize, input.hidden);
  const intermediate = input.gateProjection
    ? multiplyDenseMatrixByVector(input.gateProjection, shape.intermediateSize, shape.inputSize, input.hidden)
      .map((gateValue, index) => silu(gateValue) * (up[index] ?? 0))
    : up.map(gelu);
  const values = multiplyDenseMatrixByVector(input.downProjection, shape.outputSize, shape.intermediateSize, intermediate);
  return {
    values,
    backend: "cpu_reference",
    trace: buildMlpTrace("cpu_reference", input, shape, nowMs() - startedAt),
  };
}

function runMlpBatchCpu(input: WebGpuMlpBatchInput, shape = validateMlpBatchInput(input)): WebGpuMlpBatchResult {
  const startedAt = nowMs();
  const output = input.hidden.map((row) => {
    const up = multiplyDenseMatrixByVector(input.upProjection, shape.intermediateSize, shape.inputSize, row);
    const intermediate = input.gateProjection
      ? multiplyDenseMatrixByVector(input.gateProjection, shape.intermediateSize, shape.inputSize, row)
        .map((gateValue, index) => silu(gateValue) * (up[index] ?? 0))
      : up.map(gelu);
    return multiplyDenseMatrixByVector(input.downProjection, shape.outputSize, shape.intermediateSize, intermediate);
  });
  return {
    output,
    backend: "cpu_reference",
    trace: buildMlpTrace("cpu_reference", input, shape, nowMs() - startedAt),
  };
}

async function runSparseAttentionOnGpuDevice(
  device: MinimalGpuDevice,
  input: WebGpuSparseAttentionInput,
): Promise<{ output: Matrix; pipelineCacheHit: boolean }> {
  const queryTokens = input.q.length;
  const keyTokens = input.k.length;
  const headDim = input.q[0]?.length ?? 0;
  const selected = flattenSelectedIndexes(input.selectedKeyIndexesByQuery);
  const outputByteLength = queryTokens * headDim * Float32Array.BYTES_PER_ELEMENT;
  const qBuffer = createUploadedBuffer(device, new Float32Array(flattenMatrix(input.q)), GPU_STORAGE | GPU_COPY_DST);
  const kBuffer = createUploadedBuffer(device, new Float32Array(flattenMatrix(input.k)), GPU_STORAGE | GPU_COPY_DST);
  const vBuffer = createUploadedBuffer(device, new Float32Array(flattenMatrix(input.v)), GPU_STORAGE | GPU_COPY_DST);
  const selectedBuffer = createUploadedBuffer(device, selected.values, GPU_STORAGE | GPU_COPY_DST);
  const paramsBuffer = createUploadedBuffer(device, createSparseAttentionParams(input, {
    queryTokens,
    keyTokens,
    headDim,
    maxSelectedPerQuery: selected.maxSelectedPerQuery,
  }), GPU_UNIFORM | GPU_COPY_DST);
  const outputBuffer = device.createBuffer({ size: outputByteLength, usage: GPU_STORAGE | GPU_COPY_SRC });
  const readBuffer = device.createBuffer({ size: outputByteLength, usage: GPU_MAP_READ | GPU_COPY_DST });

  try {
    const pipeline = getOrCreateComputePipeline({
      device,
      bufferCache: input.bufferCache,
      key: "sparse-attention:main",
      moduleKey: "sparse-attention",
      code: sparseAttentionWgsl,
      entryPoint: "main",
    });
    const bindGroup = device.createBindGroup({
      layout: pipeline.pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: qBuffer } },
        { binding: 1, resource: { buffer: kBuffer } },
        { binding: 2, resource: { buffer: vBuffer } },
        { binding: 3, resource: { buffer: selectedBuffer } },
        { binding: 4, resource: { buffer: outputBuffer } },
        { binding: 5, resource: { buffer: paramsBuffer } },
      ],
    });
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline.pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(Math.ceil(queryTokens / 8), Math.ceil(headDim / 8));
    pass.end();
    encoder.copyBufferToBuffer(outputBuffer, 0, readBuffer, 0, outputByteLength);
    device.queue.submit([encoder.finish()]);
    await readBuffer.mapAsync(GPU_MAP_READ);
    const data = new Float32Array(readBuffer.getMappedRange()).slice();
    readBuffer.unmap();
    return {
      output: unflattenMatrix([...data], queryTokens, headDim),
      pipelineCacheHit: pipeline.cacheHit,
    };
  } finally {
    destroyBuffer(qBuffer);
    destroyBuffer(kBuffer);
    destroyBuffer(vBuffer);
    destroyBuffer(selectedBuffer);
    destroyBuffer(paramsBuffer);
    destroyBuffer(outputBuffer);
    destroyBuffer(readBuffer);
  }
}

async function runPackedSparseAttentionOnGpuDevice(
  device: MinimalGpuDevice,
  input: WebGpuPackedSparseAttentionInput,
  shape: { queryTokens: number; keyTokens: number; headCount: number; keyValueHeadCount: number; headDim: number; outputSize: number },
): Promise<{ output: Matrix; pipelineCacheHit: boolean; decodeOptimized?: true }> {
  const selected = flattenSelectedIndexes(input.selectedKeyIndexesByQuery);
  if (shouldUseDecodeOptimizedPackedAttention(shape, selected.maxSelectedPerQuery)) {
    return runPackedSparseDecodeAttentionOnGpuDevice(device, input, shape, selected);
  }
  const outputByteLength = shape.queryTokens * shape.outputSize * Float32Array.BYTES_PER_ELEMENT;
  const qBuffer = createUploadedBuffer(device, new Float32Array(flattenMatrix(input.q)), GPU_STORAGE | GPU_COPY_DST);
  const kBuffer = createUploadedBuffer(device, new Float32Array(flattenMatrix(input.k)), GPU_STORAGE | GPU_COPY_DST);
  const vBuffer = createUploadedBuffer(device, new Float32Array(flattenMatrix(input.v)), GPU_STORAGE | GPU_COPY_DST);
  const selectedBuffer = createUploadedBuffer(device, selected.values, GPU_STORAGE | GPU_COPY_DST);
  const paramsBuffer = createUploadedBuffer(device, createPackedSparseAttentionParams(input, {
    queryTokens: shape.queryTokens,
    keyTokens: shape.keyTokens,
    headDim: shape.headDim,
    maxSelectedPerQuery: selected.maxSelectedPerQuery,
  }), GPU_UNIFORM | GPU_COPY_DST);
  const outputBuffer = device.createBuffer({ size: outputByteLength, usage: GPU_STORAGE | GPU_COPY_SRC });
  const readBuffer = device.createBuffer({ size: outputByteLength, usage: GPU_MAP_READ | GPU_COPY_DST });

  try {
    const pipeline = getOrCreateComputePipeline({
      device,
      bufferCache: input.bufferCache,
      key: "packed-sparse-attention:packed_sparse_attention",
      moduleKey: "packed-sparse-attention",
      code: packedSparseAttentionWgsl,
      entryPoint: "packed_sparse_attention",
    });
    const bindGroup = device.createBindGroup({
      layout: pipeline.pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: qBuffer } },
        { binding: 1, resource: { buffer: kBuffer } },
        { binding: 2, resource: { buffer: vBuffer } },
        { binding: 3, resource: { buffer: selectedBuffer } },
        { binding: 4, resource: { buffer: outputBuffer } },
        { binding: 5, resource: { buffer: paramsBuffer } },
      ],
    });
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline.pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(Math.ceil(shape.queryTokens / 8), Math.ceil(shape.outputSize / 8));
    pass.end();
    encoder.copyBufferToBuffer(outputBuffer, 0, readBuffer, 0, outputByteLength);
    device.queue.submit([encoder.finish()]);
    await readBuffer.mapAsync(GPU_MAP_READ);
    const data = new Float32Array(readBuffer.getMappedRange()).slice();
    readBuffer.unmap();
    return {
      output: unflattenMatrix([...data], shape.queryTokens, shape.outputSize),
      pipelineCacheHit: pipeline.cacheHit,
    };
  } finally {
    destroyBuffer(qBuffer);
    destroyBuffer(kBuffer);
    destroyBuffer(vBuffer);
    destroyBuffer(selectedBuffer);
    destroyBuffer(paramsBuffer);
    destroyBuffer(outputBuffer);
    destroyBuffer(readBuffer);
  }
}

async function runPackedSparseDecodeAttentionOnGpuDevice(
  device: MinimalGpuDevice,
  input: WebGpuPackedSparseAttentionInput,
  shape: { queryTokens: number; keyTokens: number; headCount: number; keyValueHeadCount: number; headDim: number; outputSize: number },
  selected: { values: Int32Array; maxSelectedPerQuery: number },
): Promise<{ output: Matrix; pipelineCacheHit: boolean; decodeOptimized: true }> {
  const outputByteLength = shape.queryTokens * shape.outputSize * Float32Array.BYTES_PER_ELEMENT;
  const scoreByteLength = shape.queryTokens * shape.headCount * selected.maxSelectedPerQuery * Float32Array.BYTES_PER_ELEMENT;
  const qBuffer = createUploadedBuffer(device, new Float32Array(flattenMatrix(input.q)), GPU_STORAGE | GPU_COPY_DST);
  const kBuffer = createUploadedBuffer(device, new Float32Array(flattenMatrix(input.k)), GPU_STORAGE | GPU_COPY_DST);
  const vBuffer = createUploadedBuffer(device, new Float32Array(flattenMatrix(input.v)), GPU_STORAGE | GPU_COPY_DST);
  const selectedBuffer = createUploadedBuffer(device, selected.values, GPU_STORAGE | GPU_COPY_DST);
  const paramsBuffer = createUploadedBuffer(device, createPackedSparseAttentionParams(input, {
    queryTokens: shape.queryTokens,
    keyTokens: shape.keyTokens,
    headDim: shape.headDim,
    maxSelectedPerQuery: selected.maxSelectedPerQuery,
  }), GPU_UNIFORM | GPU_COPY_DST);
  const scoresBuffer = device.createBuffer({ size: scoreByteLength, usage: GPU_STORAGE | GPU_COPY_SRC });
  const outputBuffer = device.createBuffer({ size: outputByteLength, usage: GPU_STORAGE | GPU_COPY_SRC });
  const readBuffer = device.createBuffer({ size: outputByteLength, usage: GPU_MAP_READ | GPU_COPY_DST });

  try {
    const pipelines = createDecodeOptimizedPackedAttentionPipelines(device, input.bufferCache);
    const scoreBindGroup = device.createBindGroup({
      layout: pipelines.score.pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: qBuffer } },
        { binding: 1, resource: { buffer: kBuffer } },
        { binding: 2, resource: { buffer: selectedBuffer } },
        { binding: 3, resource: { buffer: scoresBuffer } },
        { binding: 4, resource: { buffer: paramsBuffer } },
      ],
    });
    const outputBindGroup = device.createBindGroup({
      layout: pipelines.output.pipeline.getBindGroupLayout(1),
      entries: [
        { binding: 0, resource: { buffer: scoresBuffer } },
        { binding: 1, resource: { buffer: vBuffer } },
        { binding: 2, resource: { buffer: selectedBuffer } },
        { binding: 3, resource: { buffer: outputBuffer } },
        { binding: 4, resource: { buffer: paramsBuffer } },
      ],
    });
    const encoder = device.createCommandEncoder();
    const scorePass = encoder.beginComputePass();
    scorePass.setPipeline(pipelines.score.pipeline);
    scorePass.setBindGroup(0, scoreBindGroup);
    scorePass.dispatchWorkgroups(shape.queryTokens, Math.ceil(shape.headCount / 4), Math.ceil(selected.maxSelectedPerQuery / 64));
    scorePass.end();
    const outputPass = encoder.beginComputePass();
    outputPass.setPipeline(pipelines.output.pipeline);
    outputPass.setBindGroup(1, outputBindGroup);
    outputPass.dispatchWorkgroups(shape.queryTokens, Math.ceil(shape.outputSize / 8));
    outputPass.end();
    encoder.copyBufferToBuffer(outputBuffer, 0, readBuffer, 0, outputByteLength);
    device.queue.submit([encoder.finish()]);
    await readBuffer.mapAsync(GPU_MAP_READ);
    const data = new Float32Array(readBuffer.getMappedRange()).slice();
    readBuffer.unmap();
    return {
      output: unflattenMatrix([...data], shape.queryTokens, shape.outputSize),
      pipelineCacheHit: pipelines.score.cacheHit && pipelines.output.cacheHit,
      decodeOptimized: true,
    };
  } finally {
    destroyBuffer(qBuffer);
    destroyBuffer(kBuffer);
    destroyBuffer(vBuffer);
    destroyBuffer(selectedBuffer);
    destroyBuffer(paramsBuffer);
    destroyBuffer(scoresBuffer);
    destroyBuffer(outputBuffer);
    destroyBuffer(readBuffer);
  }
}

async function runPackedSparseAttentionResidentOnGpuDevice(
  device: MinimalGpuDevice,
  input: WebGpuPackedSparseAttentionResidentInput,
  shape: { queryTokens: number; keyTokens: number; headCount: number; keyValueHeadCount: number; headDim: number; outputSize: number },
): Promise<{ tensor: WebGpuResidentTensor; pipelineCacheHit: boolean; decodeOptimized?: true; qwenOneTokenAttention?: true; dispatchCount?: number }> {
  const selected = flattenSelectedIndexes(input.selectedKeyIndexesByQuery);
  if (shouldUseQwenOneTokenFullPrefixAttention(shape, selected)) {
    return runQwenOneTokenAttentionResidentOnGpuDevice(device, input, shape);
  }
  if (shouldUseDecodeOptimizedPackedAttention(shape, selected.maxSelectedPerQuery)) {
    return runPackedSparseDecodeAttentionResidentOnGpuDevice(device, input, shape, selected);
  }
  const outputByteLength = shape.queryTokens * shape.outputSize * Float32Array.BYTES_PER_ELEMENT;
  const qResident = isWebGpuResidentTensor(input.q) ? input.q : undefined;
  const kResident = isWebGpuResidentTensor(input.k) ? input.k : undefined;
  const vResident = isWebGpuResidentTensor(input.v) ? input.v : undefined;
  const qBuffer = qResident ? readResidentTensorBuffer(qResident) : createUploadedBuffer(device, new Float32Array(flattenMatrix(input.q as Matrix)), GPU_STORAGE | GPU_COPY_DST);
  const kBuffer = kResident ? readResidentTensorBuffer(kResident) : createUploadedBuffer(device, new Float32Array(flattenMatrix(input.k as Matrix)), GPU_STORAGE | GPU_COPY_DST);
  const vBuffer = vResident ? readResidentTensorBuffer(vResident) : createUploadedBuffer(device, new Float32Array(flattenMatrix(input.v as Matrix)), GPU_STORAGE | GPU_COPY_DST);
  const selectedBuffer = createUploadedBuffer(device, selected.values, GPU_STORAGE | GPU_COPY_DST);
  const paramsBuffer = createUploadedBuffer(device, createPackedSparseAttentionParams(input, {
    queryTokens: shape.queryTokens,
    keyTokens: shape.keyTokens,
    headDim: shape.headDim,
    maxSelectedPerQuery: selected.maxSelectedPerQuery,
  }), GPU_UNIFORM | GPU_COPY_DST);
  const outputBuffer = device.createBuffer({ size: outputByteLength, usage: GPU_STORAGE | GPU_COPY_SRC });

  try {
    const pipeline = getOrCreateComputePipeline({
      device,
      bufferCache: input.bufferCache,
      key: "packed-sparse-attention:packed_sparse_attention",
      moduleKey: "packed-sparse-attention",
      code: packedSparseAttentionWgsl,
      entryPoint: "packed_sparse_attention",
    });
    const bindGroup = device.createBindGroup({
      layout: pipeline.pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: qBuffer } },
        { binding: 1, resource: { buffer: kBuffer } },
        { binding: 2, resource: { buffer: vBuffer } },
        { binding: 3, resource: { buffer: selectedBuffer } },
        { binding: 4, resource: { buffer: outputBuffer } },
        { binding: 5, resource: { buffer: paramsBuffer } },
      ],
    });
    submitOrRecordComputePass(device, input.commandBatch, {
      label: "packed-sparse-attention:resident",
      dispatches: 1,
      record: (encoder) => {
        const pass = encoder.beginComputePass();
        pass.setPipeline(pipeline.pipeline);
        pass.setBindGroup(0, bindGroup);
        pass.dispatchWorkgroups(Math.ceil(shape.queryTokens / 8), Math.ceil(shape.outputSize / 8));
        pass.end();
      },
    });
    const retainedBuffers: MinimalGpuBuffer[] = [selectedBuffer, paramsBuffer];
    if (!qResident) retainedBuffers.push(qBuffer);
    if (!kResident) retainedBuffers.push(kBuffer);
    if (!vResident) retainedBuffers.push(vBuffer);
    return {
      tensor: createWebGpuResidentTensor({
        device,
        buffer: outputBuffer,
        rows: shape.queryTokens,
        cols: shape.outputSize,
        byteLength: outputByteLength,
        retainedBuffers,
      }),
      pipelineCacheHit: pipeline.cacheHit,
    };
  } catch (error) {
    if (!qResident) destroyBuffer(qBuffer);
    if (!kResident) destroyBuffer(kBuffer);
    if (!vResident) destroyBuffer(vBuffer);
    destroyBuffer(selectedBuffer);
    destroyBuffer(paramsBuffer);
    destroyBuffer(outputBuffer);
    throw error;
  }
}

async function runQwenOneTokenAttentionResidentOnGpuDevice(
  device: MinimalGpuDevice,
  input: WebGpuPackedSparseAttentionResidentInput,
  shape: { queryTokens: number; keyTokens: number; headCount: number; keyValueHeadCount: number; headDim: number; outputSize: number },
): Promise<{ tensor: WebGpuResidentTensor; pipelineCacheHit: boolean; decodeOptimized: true; qwenOneTokenAttention: true; dispatchCount: number }> {
  const outputByteLength = shape.outputSize * Float32Array.BYTES_PER_ELEMENT;
  const qResident = isWebGpuResidentTensor(input.q) ? input.q : undefined;
  const kResident = isWebGpuResidentTensor(input.k) ? input.k : undefined;
  const vResident = isWebGpuResidentTensor(input.v) ? input.v : undefined;
  const qBuffer = qResident ? readResidentTensorBuffer(qResident) : createUploadedBuffer(device, new Float32Array(flattenMatrix(input.q as Matrix)), GPU_STORAGE | GPU_COPY_DST);
  const kBuffer = kResident ? readResidentTensorBuffer(kResident) : createUploadedBuffer(device, new Float32Array(flattenMatrix(input.k as Matrix)), GPU_STORAGE | GPU_COPY_DST);
  const vBuffer = vResident ? readResidentTensorBuffer(vResident) : createUploadedBuffer(device, new Float32Array(flattenMatrix(input.v as Matrix)), GPU_STORAGE | GPU_COPY_DST);
  const paramsBuffer = createUploadedBuffer(device, createQwenOneTokenAttentionParams(input, shape), GPU_UNIFORM | GPU_COPY_DST);
  const outputBuffer = device.createBuffer({ size: outputByteLength, usage: GPU_STORAGE | GPU_COPY_SRC });

  try {
    const pipeline = getOrCreateComputePipeline({
      device,
      bufferCache: input.bufferCache,
      key: "packed-sparse-attention:qwen_one_token_attention",
      moduleKey: "qwen-one-token-attention",
      code: QWEN_ONE_TOKEN_ATTENTION_WGSL,
      entryPoint: "qwen_one_token_attention",
    });
    const bindGroup = device.createBindGroup({
      layout: pipeline.pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: qBuffer } },
        { binding: 1, resource: { buffer: kBuffer } },
        { binding: 2, resource: { buffer: vBuffer } },
        { binding: 3, resource: { buffer: outputBuffer } },
        { binding: 4, resource: { buffer: paramsBuffer } },
      ],
    });
    submitOrRecordComputePass(device, input.commandBatch, {
      label: "packed-sparse-attention:resident:qwen-one-token",
      dispatches: 1,
      record: (encoder) => {
        const pass = encoder.beginComputePass();
        pass.setPipeline(pipeline.pipeline);
        pass.setBindGroup(0, bindGroup);
        pass.dispatchWorkgroups(shape.headCount);
        pass.end();
      },
    });
    const retainedBuffers: MinimalGpuBuffer[] = [paramsBuffer];
    if (!qResident) retainedBuffers.push(qBuffer);
    if (!kResident) retainedBuffers.push(kBuffer);
    if (!vResident) retainedBuffers.push(vBuffer);
    return {
      tensor: createWebGpuResidentTensor({
        device,
        buffer: outputBuffer,
        rows: 1,
        cols: shape.outputSize,
        byteLength: outputByteLength,
        retainedBuffers,
      }),
      pipelineCacheHit: pipeline.cacheHit,
      decodeOptimized: true,
      qwenOneTokenAttention: true,
      dispatchCount: 1,
    };
  } catch (error) {
    if (!qResident) destroyBuffer(qBuffer);
    if (!kResident) destroyBuffer(kBuffer);
    if (!vResident) destroyBuffer(vBuffer);
    destroyBuffer(paramsBuffer);
    destroyBuffer(outputBuffer);
    throw error;
  }
}

async function runPackedSparseDecodeAttentionResidentOnGpuDevice(
  device: MinimalGpuDevice,
  input: WebGpuPackedSparseAttentionResidentInput,
  shape: { queryTokens: number; keyTokens: number; headCount: number; keyValueHeadCount: number; headDim: number; outputSize: number },
  selected: { values: Int32Array; maxSelectedPerQuery: number },
): Promise<{ tensor: WebGpuResidentTensor; pipelineCacheHit: boolean; decodeOptimized: true }> {
  const outputByteLength = shape.queryTokens * shape.outputSize * Float32Array.BYTES_PER_ELEMENT;
  const scoreByteLength = shape.queryTokens * shape.headCount * selected.maxSelectedPerQuery * Float32Array.BYTES_PER_ELEMENT;
  const qResident = isWebGpuResidentTensor(input.q) ? input.q : undefined;
  const kResident = isWebGpuResidentTensor(input.k) ? input.k : undefined;
  const vResident = isWebGpuResidentTensor(input.v) ? input.v : undefined;
  const qBuffer = qResident ? readResidentTensorBuffer(qResident) : createUploadedBuffer(device, new Float32Array(flattenMatrix(input.q as Matrix)), GPU_STORAGE | GPU_COPY_DST);
  const kBuffer = kResident ? readResidentTensorBuffer(kResident) : createUploadedBuffer(device, new Float32Array(flattenMatrix(input.k as Matrix)), GPU_STORAGE | GPU_COPY_DST);
  const vBuffer = vResident ? readResidentTensorBuffer(vResident) : createUploadedBuffer(device, new Float32Array(flattenMatrix(input.v as Matrix)), GPU_STORAGE | GPU_COPY_DST);
  const selectedBuffer = createUploadedBuffer(device, selected.values, GPU_STORAGE | GPU_COPY_DST);
  const paramsBuffer = createUploadedBuffer(device, createPackedSparseAttentionParams(input, {
    queryTokens: shape.queryTokens,
    keyTokens: shape.keyTokens,
    headDim: shape.headDim,
    maxSelectedPerQuery: selected.maxSelectedPerQuery,
  }), GPU_UNIFORM | GPU_COPY_DST);
  const scoresBuffer = device.createBuffer({ size: scoreByteLength, usage: GPU_STORAGE | GPU_COPY_SRC });
  const outputBuffer = device.createBuffer({ size: outputByteLength, usage: GPU_STORAGE | GPU_COPY_SRC });

  try {
    const pipelines = createDecodeOptimizedPackedAttentionPipelines(device, input.bufferCache);
    const scoreBindGroup = device.createBindGroup({
      layout: pipelines.score.pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: qBuffer } },
        { binding: 1, resource: { buffer: kBuffer } },
        { binding: 2, resource: { buffer: selectedBuffer } },
        { binding: 3, resource: { buffer: scoresBuffer } },
        { binding: 4, resource: { buffer: paramsBuffer } },
      ],
    });
    const outputBindGroup = device.createBindGroup({
      layout: pipelines.output.pipeline.getBindGroupLayout(1),
      entries: [
        { binding: 0, resource: { buffer: scoresBuffer } },
        { binding: 1, resource: { buffer: vBuffer } },
        { binding: 2, resource: { buffer: selectedBuffer } },
        { binding: 3, resource: { buffer: outputBuffer } },
        { binding: 4, resource: { buffer: paramsBuffer } },
      ],
    });
    submitOrRecordComputePass(device, input.commandBatch, {
      label: "packed-sparse-attention:resident:decode-optimized",
      dispatches: 2,
      record: (encoder) => {
        const scorePass = encoder.beginComputePass();
        scorePass.setPipeline(pipelines.score.pipeline);
        scorePass.setBindGroup(0, scoreBindGroup);
        scorePass.dispatchWorkgroups(shape.queryTokens, Math.ceil(shape.headCount / 4), Math.ceil(selected.maxSelectedPerQuery / 64));
        scorePass.end();
        const outputPass = encoder.beginComputePass();
        outputPass.setPipeline(pipelines.output.pipeline);
        outputPass.setBindGroup(1, outputBindGroup);
        outputPass.dispatchWorkgroups(shape.queryTokens, Math.ceil(shape.outputSize / 8));
        outputPass.end();
      },
    });
    const retainedBuffers: MinimalGpuBuffer[] = [selectedBuffer, paramsBuffer, scoresBuffer];
    if (!qResident) retainedBuffers.push(qBuffer);
    if (!kResident) retainedBuffers.push(kBuffer);
    if (!vResident) retainedBuffers.push(vBuffer);
    return {
      tensor: createWebGpuResidentTensor({
        device,
        buffer: outputBuffer,
        rows: shape.queryTokens,
        cols: shape.outputSize,
        byteLength: outputByteLength,
        retainedBuffers,
      }),
      pipelineCacheHit: pipelines.score.cacheHit && pipelines.output.cacheHit,
      decodeOptimized: true,
    };
  } catch (error) {
    if (!qResident) destroyBuffer(qBuffer);
    if (!kResident) destroyBuffer(kBuffer);
    if (!vResident) destroyBuffer(vBuffer);
    destroyBuffer(selectedBuffer);
    destroyBuffer(paramsBuffer);
    destroyBuffer(scoresBuffer);
    destroyBuffer(outputBuffer);
    throw error;
  }
}

async function runSparseAttentionResidentOnGpuDevice(
  device: MinimalGpuDevice,
  input: WebGpuSparseAttentionResidentInput,
  shape: { queryTokens: number; keyTokens: number; headDim: number },
): Promise<{ tensor: WebGpuResidentTensor; pipelineCacheHit: boolean }> {
  const selected = flattenSelectedIndexes(input.selectedKeyIndexesByQuery);
  const outputByteLength = shape.queryTokens * shape.headDim * Float32Array.BYTES_PER_ELEMENT;
  const qResident = isWebGpuResidentTensor(input.q) ? input.q : undefined;
  const kResident = isWebGpuResidentTensor(input.k) ? input.k : undefined;
  const vResident = isWebGpuResidentTensor(input.v) ? input.v : undefined;
  const qBuffer = qResident ? readResidentTensorBuffer(qResident) : createUploadedBuffer(device, new Float32Array(flattenMatrix(input.q as Matrix)), GPU_STORAGE | GPU_COPY_DST);
  const kBuffer = kResident ? readResidentTensorBuffer(kResident) : createUploadedBuffer(device, new Float32Array(flattenMatrix(input.k as Matrix)), GPU_STORAGE | GPU_COPY_DST);
  const vBuffer = vResident ? readResidentTensorBuffer(vResident) : createUploadedBuffer(device, new Float32Array(flattenMatrix(input.v as Matrix)), GPU_STORAGE | GPU_COPY_DST);
  const selectedBuffer = createUploadedBuffer(device, selected.values, GPU_STORAGE | GPU_COPY_DST);
  const paramsBuffer = createUploadedBuffer(device, createSparseAttentionParamsFromShape(input, {
    queryTokens: shape.queryTokens,
    keyTokens: shape.keyTokens,
    headDim: shape.headDim,
    maxSelectedPerQuery: selected.maxSelectedPerQuery,
  }), GPU_UNIFORM | GPU_COPY_DST);
  const outputBuffer = device.createBuffer({ size: outputByteLength, usage: GPU_STORAGE | GPU_COPY_SRC });

  try {
    const pipeline = getOrCreateComputePipeline({
      device,
      bufferCache: input.bufferCache,
      key: "sparse-attention:main",
      moduleKey: "sparse-attention",
      code: sparseAttentionWgsl,
      entryPoint: "main",
    });
    const bindGroup = device.createBindGroup({
      layout: pipeline.pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: qBuffer } },
        { binding: 1, resource: { buffer: kBuffer } },
        { binding: 2, resource: { buffer: vBuffer } },
        { binding: 3, resource: { buffer: selectedBuffer } },
        { binding: 4, resource: { buffer: outputBuffer } },
        { binding: 5, resource: { buffer: paramsBuffer } },
      ],
    });
    submitOrRecordComputePass(device, input.commandBatch, {
      label: "sparse-attention:resident",
      dispatches: 1,
      record: (encoder) => {
        const pass = encoder.beginComputePass();
        pass.setPipeline(pipeline.pipeline);
        pass.setBindGroup(0, bindGroup);
        pass.dispatchWorkgroups(Math.ceil(shape.queryTokens / 8), Math.ceil(shape.headDim / 8));
        pass.end();
      },
    });
    const retainedBuffers: MinimalGpuBuffer[] = [selectedBuffer, paramsBuffer];
    if (!qResident) retainedBuffers.push(qBuffer);
    if (!kResident) retainedBuffers.push(kBuffer);
    if (!vResident) retainedBuffers.push(vBuffer);
    return {
      tensor: createWebGpuResidentTensor({
        device,
        buffer: outputBuffer,
        rows: shape.queryTokens,
        cols: shape.headDim,
        byteLength: outputByteLength,
        retainedBuffers,
      }),
      pipelineCacheHit: pipeline.cacheHit,
    };
  } catch (error) {
    if (!qResident) destroyBuffer(qBuffer);
    if (!kResident) destroyBuffer(kBuffer);
    if (!vResident) destroyBuffer(vBuffer);
    destroyBuffer(selectedBuffer);
    destroyBuffer(paramsBuffer);
    destroyBuffer(outputBuffer);
    throw error;
  }
}

async function runRmsNormResidentOnGpuDevice(
  device: MinimalGpuDevice,
  input: WebGpuRmsNormResidentInput,
  shape: { tokens: number; hidden: number },
): Promise<{ tensor: WebGpuResidentTensor; pipelineCacheHit: boolean }> {
  const residentHidden = isWebGpuResidentTensor(input.hidden) ? input.hidden : undefined;
  const hiddenBuffer = residentHidden
    ? readResidentTensorBuffer(residentHidden)
    : createUploadedBuffer(device, new Float32Array(flattenMatrix(input.hidden as Matrix)), GPU_STORAGE | GPU_COPY_DST);
  const weightBuffer = createUploadedBuffer(
    device,
    new Float32Array(input.weight ? Array.from(input.weight) : new Array(shape.hidden).fill(1)),
    GPU_STORAGE | GPU_COPY_DST,
  );
  const paramsBuffer = createUploadedBuffer(device, createRmsNormParams(shape, input.eps ?? 1e-6), GPU_UNIFORM | GPU_COPY_DST);
  const byteLength = shape.tokens * shape.hidden * Float32Array.BYTES_PER_ELEMENT;
  const outputBuffer = device.createBuffer({ size: byteLength, usage: GPU_STORAGE | GPU_COPY_SRC });

  try {
    const pipeline = getOrCreateComputePipeline({
      device,
      bufferCache: input.bufferCache,
      key: "rmsnorm:resident",
      moduleKey: "rmsnorm",
      code: rmsNormWgsl,
      entryPoint: "rms_norm",
    });
    const bindGroup = device.createBindGroup({
      layout: pipeline.pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: hiddenBuffer } },
        { binding: 1, resource: { buffer: weightBuffer } },
        { binding: 2, resource: { buffer: outputBuffer } },
        { binding: 3, resource: { buffer: paramsBuffer } },
      ],
    });
    submitOrRecordComputePass(device, input.commandBatch, {
      label: "rmsnorm:resident",
      dispatches: 1,
      record: (encoder) => {
        const pass = encoder.beginComputePass();
        pass.setPipeline(pipeline.pipeline);
        pass.setBindGroup(0, bindGroup);
        pass.dispatchWorkgroups(Math.ceil(shape.hidden / 16), Math.ceil(shape.tokens / 16));
        pass.end();
      },
    });
    const retainedBuffers: MinimalGpuBuffer[] = [weightBuffer, paramsBuffer];
    if (!residentHidden) retainedBuffers.push(hiddenBuffer);
    return {
      tensor: createWebGpuResidentTensor({ device, buffer: outputBuffer, rows: shape.tokens, cols: shape.hidden, byteLength, retainedBuffers }),
      pipelineCacheHit: pipeline.cacheHit,
    };
  } catch (error) {
    if (!residentHidden) destroyBuffer(hiddenBuffer);
    destroyBuffer(weightBuffer);
    destroyBuffer(paramsBuffer);
    destroyBuffer(outputBuffer);
    throw error;
  }
}

async function runResidualAddResidentOnGpuDevice(
  device: MinimalGpuDevice,
  input: WebGpuResidualAddResidentInput,
  shape: { tokens: number; hidden: number },
): Promise<{ tensor: WebGpuResidentTensor; pipelineCacheHit: boolean }> {
  const leftResident = isWebGpuResidentTensor(input.left) ? input.left : undefined;
  const rightResident = isWebGpuResidentTensor(input.right) ? input.right : undefined;
  const leftBuffer = leftResident
    ? readResidentTensorBuffer(leftResident)
    : createUploadedBuffer(device, new Float32Array(flattenMatrix(input.left as Matrix)), GPU_STORAGE | GPU_COPY_DST);
  const rightBuffer = rightResident
    ? readResidentTensorBuffer(rightResident)
    : createUploadedBuffer(device, new Float32Array(flattenMatrix(input.right as Matrix)), GPU_STORAGE | GPU_COPY_DST);
  const paramsBuffer = createUploadedBuffer(device, new Uint32Array([shape.tokens, shape.hidden, 0, 0]), GPU_UNIFORM | GPU_COPY_DST);
  const byteLength = shape.tokens * shape.hidden * Float32Array.BYTES_PER_ELEMENT;
  const outputBuffer = device.createBuffer({ size: byteLength, usage: GPU_STORAGE | GPU_COPY_SRC });

  try {
    const pipeline = getOrCreateComputePipeline({
      device,
      bufferCache: input.bufferCache,
      key: "residual-add:resident",
      moduleKey: "residual-add",
      code: residualAddWgsl,
      entryPoint: "residual_add",
    });
    const bindGroup = device.createBindGroup({
      layout: pipeline.pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: leftBuffer } },
        { binding: 1, resource: { buffer: rightBuffer } },
        { binding: 2, resource: { buffer: outputBuffer } },
        { binding: 3, resource: { buffer: paramsBuffer } },
      ],
    });
    submitOrRecordComputePass(device, input.commandBatch, {
      label: "residual-add:resident",
      dispatches: 1,
      record: (encoder) => {
        const pass = encoder.beginComputePass();
        pass.setPipeline(pipeline.pipeline);
        pass.setBindGroup(0, bindGroup);
        pass.dispatchWorkgroups(Math.ceil(shape.hidden / 64), Math.ceil(shape.tokens));
        pass.end();
      },
    });
    const retainedBuffers: MinimalGpuBuffer[] = [paramsBuffer];
    if (!leftResident) retainedBuffers.push(leftBuffer);
    if (!rightResident) retainedBuffers.push(rightBuffer);
    return {
      tensor: createWebGpuResidentTensor({ device, buffer: outputBuffer, rows: shape.tokens, cols: shape.hidden, byteLength, retainedBuffers }),
      pipelineCacheHit: pipeline.cacheHit,
    };
  } catch (error) {
    if (!leftResident) destroyBuffer(leftBuffer);
    if (!rightResident) destroyBuffer(rightBuffer);
    destroyBuffer(paramsBuffer);
    destroyBuffer(outputBuffer);
    throw error;
  }
}

async function runResidualRmsNormPairResidentOnGpuDevice(
  device: MinimalGpuDevice,
  input: WebGpuResidualRmsNormPairResidentInput,
  shape: { tokens: number; hidden: number },
): Promise<{ summed: WebGpuResidentTensor; normed: WebGpuResidentTensor; pipelineCacheHit: boolean }> {
  const leftResident = isWebGpuResidentTensor(input.left) ? input.left : undefined;
  const rightResident = isWebGpuResidentTensor(input.right) ? input.right : undefined;
  const leftBuffer = leftResident
    ? readResidentTensorBuffer(leftResident)
    : createUploadedBuffer(device, new Float32Array(flattenMatrix(input.left as Matrix)), GPU_STORAGE | GPU_COPY_DST);
  const rightBuffer = rightResident
    ? readResidentTensorBuffer(rightResident)
    : createUploadedBuffer(device, new Float32Array(flattenMatrix(input.right as Matrix)), GPU_STORAGE | GPU_COPY_DST);
  const weightBuffer = createUploadedBuffer(device, new Float32Array(Array.from(input.weight)), GPU_STORAGE | GPU_COPY_DST);
  const paramsBuffer = createUploadedBuffer(device, createResidualRmsNormPairParams(shape, input.eps ?? 1e-6), GPU_UNIFORM | GPU_COPY_DST);
  const byteLength = shape.tokens * shape.hidden * Float32Array.BYTES_PER_ELEMENT;
  const summedBuffer = device.createBuffer({ size: byteLength, usage: GPU_STORAGE | GPU_COPY_SRC });
  const normedBuffer = device.createBuffer({ size: byteLength, usage: GPU_STORAGE | GPU_COPY_SRC });

  try {
    const pipeline = getOrCreateComputePipeline({
      device,
      bufferCache: input.bufferCache,
      key: "residual-rmsnorm-pair:resident",
      moduleKey: "residual-rmsnorm-pair",
      code: RESIDUAL_RMSNORM_ONE_TOKEN_WGSL,
      entryPoint: "residual_rmsnorm_one_token",
    });
    const bindGroup = device.createBindGroup({
      layout: pipeline.pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: leftBuffer } },
        { binding: 1, resource: { buffer: rightBuffer } },
        { binding: 2, resource: { buffer: weightBuffer } },
        { binding: 3, resource: { buffer: summedBuffer } },
        { binding: 4, resource: { buffer: normedBuffer } },
        { binding: 5, resource: { buffer: paramsBuffer } },
      ],
    });
    submitOrRecordComputePass(device, input.commandBatch, {
      label: "residual-rmsnorm-pair:resident",
      dispatches: 1,
      record: (encoder) => {
        const pass = encoder.beginComputePass();
        pass.setPipeline(pipeline.pipeline);
        pass.setBindGroup(0, bindGroup);
        pass.dispatchWorkgroups(1);
        pass.end();
      },
    });
    const retainedBuffers: MinimalGpuBuffer[] = [weightBuffer, paramsBuffer];
    if (!leftResident) retainedBuffers.push(leftBuffer);
    if (!rightResident) retainedBuffers.push(rightBuffer);
    return {
      summed: createWebGpuResidentTensor({
        device,
        buffer: summedBuffer,
        rows: shape.tokens,
        cols: shape.hidden,
        byteLength,
        retainedBuffers,
      }),
      normed: createWebGpuResidentTensor({
        device,
        buffer: normedBuffer,
        rows: shape.tokens,
        cols: shape.hidden,
        byteLength,
        retainedBuffers: [],
      }),
      pipelineCacheHit: pipeline.cacheHit,
    };
  } catch (error) {
    if (!leftResident) destroyBuffer(leftBuffer);
    if (!rightResident) destroyBuffer(rightBuffer);
    destroyBuffer(weightBuffer);
    destroyBuffer(paramsBuffer);
    destroyBuffer(summedBuffer);
    destroyBuffer(normedBuffer);
    throw error;
  }
}

async function runQwenQkvPostProjectionResidentOnGpuDevice(
  device: MinimalGpuDevice,
  input: WebGpuQkvPostProjectionResidentInput,
  shape: { tokens: number; hidden: number },
): Promise<{ tensor: WebGpuResidentTensor; pipelineCacheHit: boolean }> {
  const residentProjected = isWebGpuResidentTensor(input.projected) ? input.projected : undefined;
  const projectedBuffer = residentProjected
    ? readResidentTensorBuffer(residentProjected)
    : createUploadedBuffer(device, new Float32Array(flattenMatrix(input.projected as Matrix)), GPU_STORAGE | GPU_COPY_DST);
  const normWeightBuffer = createUploadedBuffer(
    device,
    new Float32Array(input.normWeight ? Array.from(input.normWeight) : new Array(input.headDim).fill(1)),
    GPU_STORAGE | GPU_COPY_DST,
  );
  const positionBuffer = createUploadedBuffer(device, new Uint32Array(input.positions), GPU_STORAGE | GPU_COPY_DST);
  const paramsBuffer = createUploadedBuffer(
    device,
    createQkvPostProjectionParams(input, shape),
    GPU_UNIFORM | GPU_COPY_DST,
  );
  const byteLength = shape.tokens * shape.hidden * Float32Array.BYTES_PER_ELEMENT;
  const outputBuffer = device.createBuffer({ size: byteLength, usage: GPU_STORAGE | GPU_COPY_SRC });

  try {
    const pipeline = getOrCreateComputePipeline({
      device,
      bufferCache: input.bufferCache,
      key: "qwen-qkv-post-projection:resident",
      moduleKey: "qwen-qkv-post-projection",
      code: qwenQkvPostProjectionWgsl,
      entryPoint: "qwen_qkv_post_projection",
    });
    const bindGroup = device.createBindGroup({
      layout: pipeline.pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: projectedBuffer } },
        { binding: 1, resource: { buffer: normWeightBuffer } },
        { binding: 2, resource: { buffer: positionBuffer } },
        { binding: 3, resource: { buffer: outputBuffer } },
        { binding: 4, resource: { buffer: paramsBuffer } },
      ],
    });
    submitOrRecordComputePass(device, input.commandBatch, {
      label: "qwen-qkv-post-projection:resident",
      dispatches: 1,
      record: (encoder) => {
        const pass = encoder.beginComputePass();
        pass.setPipeline(pipeline.pipeline);
        pass.setBindGroup(0, bindGroup);
        pass.dispatchWorkgroups(Math.ceil(shape.hidden / 16), Math.ceil(shape.tokens / 16));
        pass.end();
      },
    });
    const retainedBuffers: MinimalGpuBuffer[] = [normWeightBuffer, positionBuffer, paramsBuffer];
    if (!residentProjected) retainedBuffers.push(projectedBuffer);
    return {
      tensor: createWebGpuResidentTensor({ device, buffer: outputBuffer, rows: shape.tokens, cols: shape.hidden, byteLength, retainedBuffers }),
      pipelineCacheHit: pipeline.cacheHit,
    };
  } catch (error) {
    if (!residentProjected) destroyBuffer(projectedBuffer);
    destroyBuffer(normWeightBuffer);
    destroyBuffer(positionBuffer);
    destroyBuffer(paramsBuffer);
    destroyBuffer(outputBuffer);
    throw error;
  }
}

async function runQwenQkvNormRopePairResidentOnGpuDevice(
  device: MinimalGpuDevice,
  input: WebGpuQkvNormRopePairResidentInput,
  shape: { tokens: number; qHidden: number; kHidden: number },
): Promise<{ q: WebGpuResidentTensor; k: WebGpuResidentTensor; pipelineCacheHit: boolean }> {
  const qResident = isWebGpuResidentTensor(input.qProjected) ? input.qProjected : undefined;
  const kResident = isWebGpuResidentTensor(input.kProjected) ? input.kProjected : undefined;
  const qProjectedBuffer = qResident
    ? readResidentTensorBuffer(qResident)
    : createUploadedBuffer(device, new Float32Array(flattenMatrix(input.qProjected as Matrix)), GPU_STORAGE | GPU_COPY_DST);
  const kProjectedBuffer = kResident
    ? readResidentTensorBuffer(kResident)
    : createUploadedBuffer(device, new Float32Array(flattenMatrix(input.kProjected as Matrix)), GPU_STORAGE | GPU_COPY_DST);
  const qNormWeightBuffer = createUploadedBuffer(
    device,
    new Float32Array(input.qNormWeight ? Array.from(input.qNormWeight) : new Array(input.headDim).fill(1)),
    GPU_STORAGE | GPU_COPY_DST,
  );
  const kNormWeightBuffer = createUploadedBuffer(
    device,
    new Float32Array(input.kNormWeight ? Array.from(input.kNormWeight) : new Array(input.headDim).fill(1)),
    GPU_STORAGE | GPU_COPY_DST,
  );
  const positionBuffer = createUploadedBuffer(device, new Uint32Array(input.positions), GPU_STORAGE | GPU_COPY_DST);
  const paramsBuffer = createUploadedBuffer(
    device,
    createQkvNormRopePairParams(input, shape),
    GPU_UNIFORM | GPU_COPY_DST,
  );
  const qByteLength = shape.tokens * shape.qHidden * Float32Array.BYTES_PER_ELEMENT;
  const kByteLength = shape.tokens * shape.kHidden * Float32Array.BYTES_PER_ELEMENT;
  const qOutputBuffer = device.createBuffer({ size: qByteLength, usage: GPU_STORAGE | GPU_COPY_SRC });
  const kOutputBuffer = device.createBuffer({ size: kByteLength, usage: GPU_STORAGE | GPU_COPY_SRC });

  try {
    const pipeline = getOrCreateComputePipeline({
      device,
      bufferCache: input.bufferCache,
      key: "qwen-qkv-norm-rope-pair:resident",
      moduleKey: "qwen-qkv-norm-rope-pair",
      code: QWEN_QKV_NORM_ROPE_PAIR_WGSL,
      entryPoint: "qwen_qkv_norm_rope_pair",
    });
    const bindGroup = device.createBindGroup({
      layout: pipeline.pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: qProjectedBuffer } },
        { binding: 1, resource: { buffer: kProjectedBuffer } },
        { binding: 2, resource: { buffer: qNormWeightBuffer } },
        { binding: 3, resource: { buffer: kNormWeightBuffer } },
        { binding: 4, resource: { buffer: positionBuffer } },
        { binding: 5, resource: { buffer: qOutputBuffer } },
        { binding: 6, resource: { buffer: kOutputBuffer } },
        { binding: 7, resource: { buffer: paramsBuffer } },
      ],
    });
    submitOrRecordComputePass(device, input.commandBatch, {
      label: "qwen-qkv-norm-rope-pair:resident",
      dispatches: 1,
      record: (encoder) => {
        const pass = encoder.beginComputePass();
        pass.setPipeline(pipeline.pipeline);
        pass.setBindGroup(0, bindGroup);
        pass.dispatchWorkgroups(
          Math.ceil(Math.max(shape.qHidden, shape.kHidden) / 16),
          Math.ceil(shape.tokens / 16),
        );
        pass.end();
      },
    });
    const retainedBuffers: MinimalGpuBuffer[] = [qNormWeightBuffer, kNormWeightBuffer, positionBuffer, paramsBuffer];
    if (!qResident) retainedBuffers.push(qProjectedBuffer);
    if (!kResident) retainedBuffers.push(kProjectedBuffer);
    return {
      q: createWebGpuResidentTensor({
        device,
        buffer: qOutputBuffer,
        rows: shape.tokens,
        cols: shape.qHidden,
        byteLength: qByteLength,
        retainedBuffers,
      }),
      k: createWebGpuResidentTensor({
        device,
        buffer: kOutputBuffer,
        rows: shape.tokens,
        cols: shape.kHidden,
        byteLength: kByteLength,
        retainedBuffers: [],
      }),
      pipelineCacheHit: pipeline.cacheHit,
    };
  } catch (error) {
    if (!qResident) destroyBuffer(qProjectedBuffer);
    if (!kResident) destroyBuffer(kProjectedBuffer);
    destroyBuffer(qNormWeightBuffer);
    destroyBuffer(kNormWeightBuffer);
    destroyBuffer(positionBuffer);
    destroyBuffer(paramsBuffer);
    destroyBuffer(qOutputBuffer);
    destroyBuffer(kOutputBuffer);
    throw error;
  }
}

async function runDenseMatVecOnGpuDevice(
  device: MinimalGpuDevice,
  input: WebGpuDenseMatVecInput,
  shape: { rows: number; cols: number },
  selectedRowIds: number[],
): Promise<{ values: number[]; projectionCacheHit: boolean; pipelineCacheHit: boolean }> {
  const outputRows = selectedRowIds.length;
  const matrixByteLength = outputRows * shape.cols * Float32Array.BYTES_PER_ELEMENT;
  const matrixDataFactory = () => denseMatVecMatrixToFloat32Array(input.matrix, shape, selectedRowIds);
  const cachedMatrix = canCacheDenseMatVecProjection(input)
    ? input.bufferCache.getOrUploadFloatMatrix({
        device,
        key: `dense-matvec:${input.projectionCacheKey}`,
        rows: outputRows,
        cols: shape.cols,
        usage: GPU_STORAGE | GPU_COPY_DST,
        byteLength: matrixByteLength,
        dataFactory: matrixDataFactory,
      })
    : null;
  const matrixBuffer = cachedMatrix?.buffer
    ?? createUploadedBuffer(device, matrixDataFactory(), GPU_STORAGE | GPU_COPY_DST);
  const vectorBuffer = createUploadedBuffer(device, new Float32Array(Array.from(input.vector)), GPU_STORAGE | GPU_COPY_DST);
  const rowIdsBuffer = createUploadedBuffer(device, new Int32Array(selectedRowIds), GPU_STORAGE | GPU_COPY_DST);
  const paramsBuffer = createUploadedBuffer(
    device,
    new Uint32Array([outputRows, shape.cols, 0, 0]),
    GPU_UNIFORM | GPU_COPY_DST,
  );
  const outputByteLength = outputRows * Float32Array.BYTES_PER_ELEMENT;
  const outputBuffer = device.createBuffer({ size: outputByteLength, usage: GPU_STORAGE | GPU_COPY_SRC });
  const readBuffer = device.createBuffer({ size: outputByteLength, usage: GPU_MAP_READ | GPU_COPY_DST });

  try {
    const pipeline = getOrCreateComputePipeline({
      device,
      bufferCache: input.bufferCache,
      key: "dense-matvec:main",
      moduleKey: "dense-matvec",
      code: denseMatVecWgsl,
      entryPoint: "main",
    });
    const bindGroup = device.createBindGroup({
      layout: pipeline.pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: matrixBuffer } },
        { binding: 1, resource: { buffer: vectorBuffer } },
        { binding: 2, resource: { buffer: rowIdsBuffer } },
        { binding: 3, resource: { buffer: outputBuffer } },
        { binding: 4, resource: { buffer: paramsBuffer } },
      ],
    });
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline.pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(Math.ceil(outputRows / 64));
    pass.end();
    encoder.copyBufferToBuffer(outputBuffer, 0, readBuffer, 0, outputByteLength);
    device.queue.submit([encoder.finish()]);
    await readBuffer.mapAsync(GPU_MAP_READ);
    const data = new Float32Array(readBuffer.getMappedRange()).slice();
    readBuffer.unmap();
    return {
      values: [...data],
      projectionCacheHit: cachedMatrix?.cacheHit ?? false,
      pipelineCacheHit: pipeline.cacheHit,
    };
  } finally {
    if (!cachedMatrix) destroyBuffer(matrixBuffer);
    destroyBuffer(vectorBuffer);
    destroyBuffer(rowIdsBuffer);
    destroyBuffer(paramsBuffer);
    destroyBuffer(outputBuffer);
    destroyBuffer(readBuffer);
  }
}

async function runDenseMatVecResidentOnGpuDevice(
  device: MinimalGpuDevice,
  input: WebGpuDenseMatVecResidentInput,
  shape: { rows: number; cols: number },
  selectedRowIds: number[],
): Promise<{ values: number[]; projectionCacheHit: boolean; pipelineCacheHit: boolean }> {
  const outputRows = selectedRowIds.length;
  const matrixByteLength = outputRows * shape.cols * Float32Array.BYTES_PER_ELEMENT;
  const matrixDataFactory = () => denseMatVecMatrixToFloat32Array(input.matrix, shape, selectedRowIds);
  const cachedMatrix = canCacheDenseMatVecProjection(input)
    ? input.bufferCache.getOrUploadFloatMatrix({
        device,
        key: `dense-matvec:${input.projectionCacheKey}`,
        rows: outputRows,
        cols: shape.cols,
        usage: GPU_STORAGE | GPU_COPY_DST,
        byteLength: matrixByteLength,
        dataFactory: matrixDataFactory,
      })
    : null;
  const matrixBuffer = cachedMatrix?.buffer
    ?? createUploadedBuffer(device, matrixDataFactory(), GPU_STORAGE | GPU_COPY_DST);
  const vectorBuffer = readResidentTensorBuffer(input.vector);
  const rowIdsBuffer = createUploadedBuffer(device, new Int32Array(selectedRowIds), GPU_STORAGE | GPU_COPY_DST);
  const paramsBuffer = createUploadedBuffer(
    device,
    new Uint32Array([outputRows, shape.cols, 0, 0]),
    GPU_UNIFORM | GPU_COPY_DST,
  );
  const outputByteLength = outputRows * Float32Array.BYTES_PER_ELEMENT;
  const outputBuffer = device.createBuffer({ size: outputByteLength, usage: GPU_STORAGE | GPU_COPY_SRC });
  const readBuffer = device.createBuffer({ size: outputByteLength, usage: GPU_MAP_READ | GPU_COPY_DST });

  try {
    const pipeline = getOrCreateComputePipeline({
      device,
      bufferCache: input.bufferCache,
      key: "dense-matvec:main",
      moduleKey: "dense-matvec",
      code: denseMatVecWgsl,
      entryPoint: "main",
    });
    const bindGroup = device.createBindGroup({
      layout: pipeline.pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: matrixBuffer } },
        { binding: 1, resource: { buffer: vectorBuffer } },
        { binding: 2, resource: { buffer: rowIdsBuffer } },
        { binding: 3, resource: { buffer: outputBuffer } },
        { binding: 4, resource: { buffer: paramsBuffer } },
      ],
    });
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline.pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(Math.ceil(outputRows / 64));
    pass.end();
    encoder.copyBufferToBuffer(outputBuffer, 0, readBuffer, 0, outputByteLength);
    device.queue.submit([encoder.finish()]);
    await readBuffer.mapAsync(GPU_MAP_READ);
    const data = new Float32Array(readBuffer.getMappedRange()).slice();
    readBuffer.unmap();
    return {
      values: [...data],
      projectionCacheHit: cachedMatrix?.cacheHit ?? false,
      pipelineCacheHit: pipeline.cacheHit,
    };
  } finally {
    if (!cachedMatrix) destroyBuffer(matrixBuffer);
    destroyBuffer(rowIdsBuffer);
    destroyBuffer(paramsBuffer);
    destroyBuffer(outputBuffer);
    destroyBuffer(readBuffer);
  }
}

function canCacheDenseMatVecProjection(input: Pick<
  WebGpuDenseMatVecInput,
  "bufferCache" | "projectionCacheKey" | "projectionCachePolicy" | "matrix" | "selectedRowIds" | "traceMetadata"
>): input is Pick<WebGpuDenseMatVecInput, "matrix" | "selectedRowIds" | "traceMetadata"> & {
  bufferCache: WebGpuRuntimeBufferCache;
  projectionCacheKey: string;
  projectionCachePolicy: "stable";
} {
  return Boolean(input.bufferCache)
    && Boolean(input.projectionCacheKey)
    && input.projectionCachePolicy === "stable"
    && !Array.isArray(input.matrix)
    && (!input.selectedRowIds || input.traceMetadata?.tiledTopK === true);
}

async function runDenseMatMulOnGpuDevice(
  device: MinimalGpuDevice,
  input: WebGpuDenseMatMulInput,
  shape: { tokens: number; hidden: number; outputSize: number },
): Promise<{ output: Matrix; projectionCacheHit: boolean; pipelineCacheHit: boolean }> {
  const resident = await runDenseMatMulResidentOnGpuDevice(device, input, shape);
  try {
    return {
      output: await readWebGpuResidentTensor(resident.tensor),
      projectionCacheHit: resident.projectionCacheHit,
      pipelineCacheHit: resident.pipelineCacheHit,
    };
  } finally {
    destroyWebGpuResidentTensor(resident.tensor);
  }
}

async function runDenseMatMulResidentOnGpuDevice(
  device: MinimalGpuDevice,
  input: WebGpuDenseMatMulResidentInput,
  shape: { tokens: number; hidden: number; outputSize: number },
): Promise<{ tensor: WebGpuResidentTensor; projectionCacheHit: boolean; pipelineCacheHit: boolean }> {
  const projectionByteLength = shape.outputSize * shape.hidden * Float32Array.BYTES_PER_ELEMENT;
  const projectionDataFactory = () => denseMatVecMatrixToFloat32Array(
    input.projection,
    { cols: shape.hidden },
    allRowIds(shape.outputSize),
  );
  const cachedProjection = canCacheDenseMatMulProjection(input)
    ? input.bufferCache.getOrUploadFloatMatrix({
        device,
        key: `dense-matmul:${input.projectionCacheKey}`,
        rows: shape.outputSize,
        cols: shape.hidden,
        usage: GPU_STORAGE | GPU_COPY_DST,
        byteLength: projectionByteLength,
        dataFactory: projectionDataFactory,
      })
    : null;
  const projectionBuffer = cachedProjection?.buffer
    ?? createUploadedBuffer(device, projectionDataFactory(), GPU_STORAGE | GPU_COPY_DST);
  const residentActivation = isWebGpuResidentTensor(input.activations) ? input.activations : undefined;
  const activationMatrix = residentActivation ? undefined : input.activations as Matrix;
  const activationBuffer = residentActivation
    ? readResidentTensorBuffer(residentActivation)
    : createUploadedBuffer(device, new Float32Array(flattenMatrix(activationMatrix ?? [])), GPU_STORAGE | GPU_COPY_DST);
  const paramsBuffer = createUploadedBuffer(
    device,
    new Uint32Array([shape.tokens, shape.hidden, shape.outputSize, 0]),
    GPU_UNIFORM | GPU_COPY_DST,
  );
  const outputByteLength = shape.tokens * shape.outputSize * Float32Array.BYTES_PER_ELEMENT;
  const outputBuffer = device.createBuffer({ size: outputByteLength, usage: GPU_STORAGE | GPU_COPY_SRC });

  try {
    const pipeline = getOrCreateComputePipeline({
      device,
      bufferCache: input.bufferCache,
      key: "dense-matmul:dense_matmul",
      moduleKey: "dense-matmul",
      code: denseMatMulWgsl,
      entryPoint: "dense_matmul",
    });
    const bindGroup = device.createBindGroup({
      layout: pipeline.pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: activationBuffer } },
        { binding: 1, resource: { buffer: projectionBuffer } },
        { binding: 2, resource: { buffer: outputBuffer } },
        { binding: 3, resource: { buffer: paramsBuffer } },
      ],
    });
    submitOrRecordComputePass(device, input.commandBatch, {
      label: "dense-matmul:resident",
      dispatches: 1,
      record: (encoder) => {
        const pass = encoder.beginComputePass();
        pass.setPipeline(pipeline.pipeline);
        pass.setBindGroup(0, bindGroup);
        pass.dispatchWorkgroups(Math.ceil(shape.outputSize / 16), Math.ceil(shape.tokens / 16));
        pass.end();
      },
    });
    const retainedBuffers: MinimalGpuBuffer[] = [paramsBuffer];
    if (!residentActivation) retainedBuffers.push(activationBuffer);
    if (!cachedProjection) retainedBuffers.push(projectionBuffer);
    return {
      tensor: createWebGpuResidentTensor({
        device,
        buffer: outputBuffer,
        rows: shape.tokens,
        cols: shape.outputSize,
        byteLength: outputByteLength,
        retainedBuffers,
      }),
      projectionCacheHit: cachedProjection?.cacheHit ?? false,
      pipelineCacheHit: pipeline.cacheHit,
    };
  } catch (error) {
    if (!cachedProjection) destroyBuffer(projectionBuffer);
    if (!residentActivation) destroyBuffer(activationBuffer);
    destroyBuffer(paramsBuffer);
    destroyBuffer(outputBuffer);
    throw error;
  }
}

async function runPackedQkvProjectionResidentOnGpuDevice(
  device: MinimalGpuDevice,
  input: WebGpuPackedQkvProjectionResidentInput,
  shape: { tokens: number; hidden: number; qOutputSize: number; kOutputSize: number; vOutputSize: number },
): Promise<{
  q: WebGpuResidentTensor;
  k: WebGpuResidentTensor;
  v: WebGpuResidentTensor;
  projectionCacheHits: { qProjection: boolean; kProjection: boolean; vProjection: boolean };
  pipelineCacheHit: boolean;
}> {
  const residentHidden = isWebGpuResidentTensor(input.hidden) ? input.hidden : undefined;
  const hiddenBuffer = residentHidden
    ? readResidentTensorBuffer(residentHidden)
    : createUploadedBuffer(device, new Float32Array(flattenMatrix(input.hidden as Matrix)), GPU_STORAGE | GPU_COPY_DST);
  const qUpload = getOrUploadPackedQkvProjectionBuffer(device, input, input.qProjection, shape.qOutputSize, projectionKeyForPackedQkv(input, "qProjection"));
  const kUpload = getOrUploadPackedQkvProjectionBuffer(device, input, input.kProjection, shape.kOutputSize, projectionKeyForPackedQkv(input, "kProjection"));
  const vUpload = getOrUploadPackedQkvProjectionBuffer(device, input, input.vProjection, shape.vOutputSize, projectionKeyForPackedQkv(input, "vProjection"));
  const qByteLength = shape.tokens * shape.qOutputSize * Float32Array.BYTES_PER_ELEMENT;
  const kByteLength = shape.tokens * shape.kOutputSize * Float32Array.BYTES_PER_ELEMENT;
  const vByteLength = shape.tokens * shape.vOutputSize * Float32Array.BYTES_PER_ELEMENT;
  const qBuffer = device.createBuffer({ size: alignTo4(qByteLength), usage: GPU_STORAGE | GPU_COPY_SRC });
  const kBuffer = device.createBuffer({ size: alignTo4(kByteLength), usage: GPU_STORAGE | GPU_COPY_SRC });
  const vBuffer = device.createBuffer({ size: alignTo4(vByteLength), usage: GPU_STORAGE | GPU_COPY_SRC });
  const paramsBuffer = createUploadedBuffer(
    device,
    new Uint32Array([
      shape.hidden,
      shape.qOutputSize,
      shape.kOutputSize,
      shape.vOutputSize,
      shape.qOutputSize + shape.kOutputSize + shape.vOutputSize,
      shape.tokens,
      0,
      0,
    ]),
    GPU_UNIFORM | GPU_COPY_DST,
  );

  try {
    const pipeline = getOrCreateComputePipeline({
      device,
      bufferCache: input.bufferCache,
      key: "packed-qkv-projection:resident",
      moduleKey: "packed-qkv-projection",
      code: PACKED_QKV_PROJECTION_WGSL,
      entryPoint: "packed_qkv_projection",
    });
    const bindGroup = device.createBindGroup({
      layout: pipeline.pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: hiddenBuffer } },
        { binding: 1, resource: { buffer: qUpload.buffer } },
        { binding: 2, resource: { buffer: kUpload.buffer } },
        { binding: 3, resource: { buffer: vUpload.buffer } },
        { binding: 4, resource: { buffer: qBuffer } },
        { binding: 5, resource: { buffer: kBuffer } },
        { binding: 6, resource: { buffer: vBuffer } },
        { binding: 7, resource: { buffer: paramsBuffer } },
      ],
    });
    submitOrRecordComputePass(device, input.commandBatch, {
      label: "packed-qkv-projection:resident",
      dispatches: 1,
      record: (encoder) => {
        const pass = encoder.beginComputePass();
        pass.setPipeline(pipeline.pipeline);
        pass.setBindGroup(0, bindGroup);
        pass.dispatchWorkgroups(Math.ceil((shape.qOutputSize + shape.kOutputSize + shape.vOutputSize) / 128), shape.tokens);
        pass.end();
      },
    });
    const qRetainedBuffers: MinimalGpuBuffer[] = [paramsBuffer];
    const kRetainedBuffers: MinimalGpuBuffer[] = [];
    const vRetainedBuffers: MinimalGpuBuffer[] = [];
    if (!residentHidden) qRetainedBuffers.push(hiddenBuffer);
    if (!qUpload.cached) qRetainedBuffers.push(qUpload.buffer);
    if (!kUpload.cached) kRetainedBuffers.push(kUpload.buffer);
    if (!vUpload.cached) vRetainedBuffers.push(vUpload.buffer);
    return {
      q: createWebGpuResidentTensor({ device, buffer: qBuffer, rows: shape.tokens, cols: shape.qOutputSize, byteLength: qByteLength, retainedBuffers: qRetainedBuffers }),
      k: createWebGpuResidentTensor({ device, buffer: kBuffer, rows: shape.tokens, cols: shape.kOutputSize, byteLength: kByteLength, retainedBuffers: kRetainedBuffers }),
      v: createWebGpuResidentTensor({ device, buffer: vBuffer, rows: shape.tokens, cols: shape.vOutputSize, byteLength: vByteLength, retainedBuffers: vRetainedBuffers }),
      projectionCacheHits: {
        qProjection: qUpload.cacheHit,
        kProjection: kUpload.cacheHit,
        vProjection: vUpload.cacheHit,
      },
      pipelineCacheHit: pipeline.cacheHit,
    };
  } catch (error) {
    if (!residentHidden) destroyBuffer(hiddenBuffer);
    if (!qUpload.cached) destroyBuffer(qUpload.buffer);
    if (!kUpload.cached) destroyBuffer(kUpload.buffer);
    if (!vUpload.cached) destroyBuffer(vUpload.buffer);
    destroyBuffer(qBuffer);
    destroyBuffer(kBuffer);
    destroyBuffer(vBuffer);
    destroyBuffer(paramsBuffer);
    throw error;
  }
}

function getOrUploadPackedQkvProjectionBuffer(
  device: MinimalGpuDevice,
  input: WebGpuPackedQkvProjectionResidentInput,
  projection: DenseMatVecMatrix,
  rows: number,
  key: string | undefined,
): { buffer: MinimalGpuBuffer; cached: boolean; cacheHit: boolean } {
  const byteLength = rows * denseMatVecColCount(projection) * Float32Array.BYTES_PER_ELEMENT;
  const dataFactory = () => denseMatVecMatrixToFloat32Array(projection, { cols: denseMatVecColCount(projection) }, allRowIds(rows));
  const cached = input.bufferCache
    && key
    && input.projectionCachePolicy === "stable"
    && !Array.isArray(projection)
    ? input.bufferCache.getOrUploadFloatMatrix({
        device,
        key: `dense-matmul:${key}`,
        rows,
        cols: denseMatVecColCount(projection),
        usage: GPU_STORAGE | GPU_COPY_DST,
        byteLength,
        dataFactory,
      })
    : null;
  if (cached) {
    return { buffer: cached.buffer, cached: true, cacheHit: cached.cacheHit };
  }
  return {
    buffer: createUploadedBuffer(device, dataFactory(), GPU_STORAGE | GPU_COPY_DST),
    cached: false,
    cacheHit: false,
  };
}

function projectionKeyForPackedQkv(
  input: WebGpuPackedQkvProjectionResidentInput,
  projectionName: "qProjection" | "kProjection" | "vProjection",
): string | undefined {
  switch (projectionName) {
    case "qProjection": return input.qProjectionCacheKey ?? (input.projectionCacheKey ? `${input.projectionCacheKey}:qProjection` : undefined);
    case "kProjection": return input.kProjectionCacheKey ?? (input.projectionCacheKey ? `${input.projectionCacheKey}:kProjection` : undefined);
    case "vProjection": return input.vProjectionCacheKey ?? (input.projectionCacheKey ? `${input.projectionCacheKey}:vProjection` : undefined);
    default: {
      const exhaustive: never = projectionName;
      return exhaustive;
    }
  }
}

function canCacheDenseMatMulProjection(input: Pick<
  WebGpuDenseMatMulInput,
  "bufferCache" | "projectionCacheKey" | "projectionCachePolicy" | "projection"
>): input is Pick<WebGpuDenseMatMulInput, "projection"> & {
  bufferCache: WebGpuRuntimeBufferCache;
  projectionCacheKey: string;
  projectionCachePolicy: "stable";
} {
  return Boolean(input.bufferCache)
    && Boolean(input.projectionCacheKey)
    && input.projectionCachePolicy === "stable"
    && !Array.isArray(input.projection);
}

function createSparseAttentionParams(
  input: WebGpuSparseAttentionInput,
  shape: { queryTokens: number; keyTokens: number; headDim: number; maxSelectedPerQuery: number },
): Uint32Array {
  return createSparseAttentionParamsFromShape(input, shape);
}

function createSparseAttentionParamsFromShape(
  input: Pick<WebGpuSparseAttentionInput, "causal" | "scale">,
  shape: { queryTokens: number; keyTokens: number; headDim: number; maxSelectedPerQuery: number },
): Uint32Array {
  const buffer = new ArrayBuffer(32);
  const view = new DataView(buffer);
  view.setUint32(0, shape.queryTokens, true);
  view.setUint32(4, shape.keyTokens, true);
  view.setUint32(8, shape.headDim, true);
  view.setUint32(12, shape.maxSelectedPerQuery, true);
  // The runtime passes already-masked absolute selected key indexes. A shader
  // causal check only has the local query row, so it can incorrectly drop every
  // decode key after row 0. Keep masking in selectedKeyIndexesByQuery.
  view.setUint32(16, 0, true);
  view.setFloat32(20, input.scale ?? 1 / Math.sqrt(shape.headDim || 1), true);
  view.setUint32(24, 0, true);
  view.setUint32(28, 0, true);
  return new Uint32Array(buffer);
}

function createPackedSparseAttentionParams(
  input: Pick<WebGpuPackedSparseAttentionInput, "causal" | "scale" | "headCount" | "keyValueHeadCount">,
  shape: { queryTokens: number; keyTokens: number; headDim: number; maxSelectedPerQuery: number },
): Uint32Array {
  const buffer = new ArrayBuffer(32);
  const view = new DataView(buffer);
  view.setUint32(0, shape.queryTokens, true);
  view.setUint32(4, shape.keyTokens, true);
  view.setUint32(8, shape.headDim, true);
  view.setUint32(12, shape.maxSelectedPerQuery, true);
  // See createSparseAttentionParamsFromShape: selectedKeyIndexesByQuery is the
  // causal/sparse mask. Reapplying causal with local query coordinates breaks
  // single-token decode over an absolute KV cache.
  view.setUint32(16, 0, true);
  view.setFloat32(20, input.scale ?? 1 / Math.sqrt(shape.headDim || 1), true);
  view.setUint32(24, input.headCount, true);
  view.setUint32(28, input.keyValueHeadCount ?? input.headCount, true);
  return new Uint32Array(buffer);
}

function createQwenOneTokenAttentionParams(
  input: Pick<WebGpuPackedSparseAttentionInput, "scale">,
  shape: { keyTokens: number; headCount: number; keyValueHeadCount: number; headDim: number },
): Uint32Array {
  const buffer = new ArrayBuffer(32);
  const view = new DataView(buffer);
  view.setUint32(0, shape.headCount, true);
  view.setUint32(4, shape.keyValueHeadCount, true);
  view.setUint32(8, shape.headDim, true);
  view.setUint32(12, shape.keyTokens, true);
  view.setUint32(16, shape.keyValueHeadCount * shape.headDim, true);
  view.setFloat32(20, input.scale ?? 1 / Math.sqrt(shape.headDim || 1), true);
  view.setUint32(24, 0, true);
  view.setUint32(28, 0, true);
  return new Uint32Array(buffer);
}

function createRmsNormParams(shape: { tokens: number; hidden: number }, eps: number): Uint32Array {
  const buffer = new ArrayBuffer(16);
  const view = new DataView(buffer);
  view.setUint32(0, shape.tokens, true);
  view.setUint32(4, shape.hidden, true);
  view.setFloat32(8, eps, true);
  view.setUint32(12, 0, true);
  return new Uint32Array(buffer);
}

function createResidualRmsNormPairParams(shape: { hidden: number }, eps: number): Uint32Array {
  const buffer = new ArrayBuffer(16);
  const view = new DataView(buffer);
  view.setUint32(0, shape.hidden, true);
  view.setFloat32(4, eps, true);
  view.setUint32(8, 0, true);
  view.setUint32(12, 0, true);
  return new Uint32Array(buffer);
}

function createQkvPostProjectionParams(input: WebGpuQkvPostProjectionResidentInput, shape: { tokens: number; hidden: number }): Uint32Array {
  const buffer = new ArrayBuffer(32);
  const view = new DataView(buffer);
  view.setUint32(0, shape.tokens, true);
  view.setUint32(4, input.headCount, true);
  view.setUint32(8, input.headDim, true);
  view.setUint32(12, shape.hidden, true);
  view.setFloat32(16, input.eps ?? 1e-6, true);
  view.setFloat32(20, input.ropeTheta ?? 0, true);
  view.setUint32(24, input.normWeight ? 1 : 0, true);
  view.setUint32(28, input.ropeTheta !== undefined && input.ropeTheta > 0 ? 1 : 0, true);
  return new Uint32Array(buffer);
}

function createQkvNormRopePairParams(
  input: WebGpuQkvNormRopePairResidentInput,
  shape: { tokens: number; qHidden: number; kHidden: number },
): Uint32Array {
  const buffer = new ArrayBuffer(48);
  const view = new DataView(buffer);
  view.setUint32(0, shape.tokens, true);
  view.setUint32(4, input.qHeadCount, true);
  view.setUint32(8, input.kHeadCount, true);
  view.setUint32(12, input.headDim, true);
  view.setUint32(16, shape.qHidden, true);
  view.setUint32(20, shape.kHidden, true);
  view.setFloat32(24, input.eps ?? 1e-6, true);
  view.setFloat32(28, input.ropeTheta ?? 0, true);
  view.setUint32(32, input.qNormWeight ? 1 : 0, true);
  view.setUint32(36, input.kNormWeight ? 1 : 0, true);
  view.setUint32(40, input.ropeTheta !== undefined && input.ropeTheta > 0 ? 1 : 0, true);
  view.setUint32(44, 0, true);
  return new Uint32Array(buffer);
}

async function runMlpOnGpuDevice(
  device: MinimalGpuDevice,
  input: WebGpuMlpInput,
  shape: MlpShape,
): Promise<{
  values: number[];
  projectionCacheHits?: WebGpuMlpTrace["projectionCacheHits"];
  pipelineCacheHits?: WebGpuMlpTrace["pipelineCacheHits"];
}> {
  const hiddenBuffer = createUploadedBuffer(device, Float32Array.from(input.hidden), GPU_STORAGE | GPU_COPY_DST);
  const upProjection = getOrUploadStableProjectionBuffer({
    device,
    input,
    projection: input.upProjection,
    keySuffix: "upProjection",
    rows: shape.intermediateSize,
    cols: shape.inputSize,
  });
  const downProjection = getOrUploadStableProjectionBuffer({
    device,
    input,
    projection: input.downProjection,
    keySuffix: "downProjection",
    rows: shape.outputSize,
    cols: shape.intermediateSize,
  });
  const gateProjection = input.gateProjection
    ? getOrUploadStableProjectionBuffer({
        device,
        input,
        projection: input.gateProjection,
        keySuffix: "gateProjection",
        rows: shape.intermediateSize,
        cols: shape.inputSize,
      })
    : {
        buffer: createUploadedBuffer(device, new Float32Array([0]), GPU_STORAGE | GPU_COPY_DST),
        cacheHit: false,
        cached: false,
      };
  const paramsBuffer = createUploadedBuffer(
    device,
    new Uint32Array([
      shape.inputSize,
      shape.intermediateSize,
      shape.outputSize,
      shape.activationKind === "silu_gated" ? 1 : 0,
    ]),
    GPU_UNIFORM | GPU_COPY_DST,
  );
  const intermediateByteLength = shape.intermediateSize * Float32Array.BYTES_PER_ELEMENT;
  const outputByteLength = shape.outputSize * Float32Array.BYTES_PER_ELEMENT;
  const intermediateBuffer = device.createBuffer({ size: intermediateByteLength, usage: GPU_STORAGE | GPU_COPY_SRC });
  const outputBuffer = device.createBuffer({ size: outputByteLength, usage: GPU_STORAGE | GPU_COPY_SRC });
  const readBuffer = device.createBuffer({ size: outputByteLength, usage: GPU_MAP_READ | GPU_COPY_DST });
  const buffers = {
    hiddenBuffer,
    upBuffer: upProjection.buffer,
    gateBuffer: gateProjection.buffer,
    downBuffer: downProjection.buffer,
    intermediateBuffer,
    outputBuffer,
    paramsBuffer,
  };

  try {
    const intermediatePipeline = getOrCreateComputePipeline({
      device,
      bufferCache: input.bufferCache,
      key: "mlp:mlp_intermediate",
      moduleKey: "mlp",
      code: mlpWgsl,
      entryPoint: "mlp_intermediate",
    });
    const outputPipeline = getOrCreateComputePipeline({
      device,
      bufferCache: input.bufferCache,
      key: "mlp:mlp_output",
      moduleKey: "mlp",
      code: mlpWgsl,
      entryPoint: "mlp_output",
    });
    const intermediateBindGroup = createMlpBindGroup(device, intermediatePipeline.pipeline, buffers);
    const outputBindGroup = createMlpBindGroup(device, outputPipeline.pipeline, buffers);
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(intermediatePipeline.pipeline);
    pass.setBindGroup(0, intermediateBindGroup);
    pass.dispatchWorkgroups(Math.ceil(shape.intermediateSize / 64));
    pass.setPipeline(outputPipeline.pipeline);
    pass.setBindGroup(0, outputBindGroup);
    pass.dispatchWorkgroups(Math.ceil(shape.outputSize / 64));
    pass.end();
    encoder.copyBufferToBuffer(outputBuffer, 0, readBuffer, 0, outputByteLength);
    device.queue.submit([encoder.finish()]);
    await readBuffer.mapAsync(GPU_MAP_READ);
    const data = new Float32Array(readBuffer.getMappedRange()).slice();
    readBuffer.unmap();
    return {
      values: [...data],
      projectionCacheHits: {
        upProjection: upProjection.cacheHit,
        ...(input.gateProjection ? { gateProjection: gateProjection.cacheHit } : {}),
        downProjection: downProjection.cacheHit,
      },
      pipelineCacheHits: {
        intermediate: intermediatePipeline.cacheHit,
        output: outputPipeline.cacheHit,
      },
    };
  } finally {
    destroyBuffer(hiddenBuffer);
    if (!upProjection.cached) destroyBuffer(upProjection.buffer);
    if (!gateProjection.cached) destroyBuffer(gateProjection.buffer);
    if (!downProjection.cached) destroyBuffer(downProjection.buffer);
    destroyBuffer(paramsBuffer);
    destroyBuffer(intermediateBuffer);
    destroyBuffer(outputBuffer);
    destroyBuffer(readBuffer);
  }
}

async function runMlpBatchOnGpuDevice(
  device: MinimalGpuDevice,
  input: WebGpuMlpBatchInput,
  shape: MlpShape & { tokens: number },
): Promise<{
  output: Matrix;
  projectionCacheHits?: WebGpuMlpTrace["projectionCacheHits"];
  pipelineCacheHits?: WebGpuMlpTrace["pipelineCacheHits"];
}> {
  const hiddenBuffer = createUploadedBuffer(device, new Float32Array(flattenMatrix(input.hidden)), GPU_STORAGE | GPU_COPY_DST);
  const upProjection = getOrUploadStableProjectionBuffer({
    device,
    input,
    projection: input.upProjection,
    keySuffix: "upProjection",
    rows: shape.intermediateSize,
    cols: shape.inputSize,
  });
  const downProjection = getOrUploadStableProjectionBuffer({
    device,
    input,
    projection: input.downProjection,
    keySuffix: "downProjection",
    rows: shape.outputSize,
    cols: shape.intermediateSize,
  });
  const gateProjection = input.gateProjection
    ? getOrUploadStableProjectionBuffer({
        device,
        input,
        projection: input.gateProjection,
        keySuffix: "gateProjection",
        rows: shape.intermediateSize,
        cols: shape.inputSize,
      })
    : {
        buffer: createUploadedBuffer(device, new Float32Array([0]), GPU_STORAGE | GPU_COPY_DST),
        cacheHit: false,
        cached: false,
      };
  const paramsBuffer = createUploadedBuffer(
    device,
    new Uint32Array([
      shape.inputSize,
      shape.intermediateSize,
      shape.outputSize,
      shape.tokens,
      shape.activationKind === "silu_gated" ? 1 : 0,
      0,
      0,
      0,
    ]),
    GPU_UNIFORM | GPU_COPY_DST,
  );
  const intermediateByteLength = shape.tokens * shape.intermediateSize * Float32Array.BYTES_PER_ELEMENT;
  const outputByteLength = shape.tokens * shape.outputSize * Float32Array.BYTES_PER_ELEMENT;
  const intermediateBuffer = device.createBuffer({ size: intermediateByteLength, usage: GPU_STORAGE | GPU_COPY_SRC });
  const outputBuffer = device.createBuffer({ size: outputByteLength, usage: GPU_STORAGE | GPU_COPY_SRC });
  const readBuffer = device.createBuffer({ size: outputByteLength, usage: GPU_MAP_READ | GPU_COPY_DST });
  const buffers = {
    hiddenBuffer,
    upBuffer: upProjection.buffer,
    gateBuffer: gateProjection.buffer,
    downBuffer: downProjection.buffer,
    intermediateBuffer,
    outputBuffer,
    paramsBuffer,
  };

  try {
    const intermediatePipeline = getOrCreateComputePipeline({
      device,
      bufferCache: input.bufferCache,
      key: "mlp-batch:mlp_batch_intermediate",
      moduleKey: "mlp-batch",
      code: mlpBatchWgsl,
      entryPoint: "mlp_batch_intermediate",
    });
    const outputPipeline = getOrCreateComputePipeline({
      device,
      bufferCache: input.bufferCache,
      key: "mlp-batch:mlp_batch_output",
      moduleKey: "mlp-batch",
      code: mlpBatchWgsl,
      entryPoint: "mlp_batch_output",
    });
    const intermediateBindGroup = createMlpBindGroup(device, intermediatePipeline.pipeline, buffers);
    const outputBindGroup = createMlpBindGroup(device, outputPipeline.pipeline, buffers);
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(intermediatePipeline.pipeline);
    pass.setBindGroup(0, intermediateBindGroup);
    pass.dispatchWorkgroups(Math.ceil(shape.intermediateSize / 16), Math.ceil(shape.tokens / 16));
    pass.setPipeline(outputPipeline.pipeline);
    pass.setBindGroup(0, outputBindGroup);
    pass.dispatchWorkgroups(Math.ceil(shape.outputSize / 16), Math.ceil(shape.tokens / 16));
    pass.end();
    encoder.copyBufferToBuffer(outputBuffer, 0, readBuffer, 0, outputByteLength);
    device.queue.submit([encoder.finish()]);
    await readBuffer.mapAsync(GPU_MAP_READ);
    const data = new Float32Array(readBuffer.getMappedRange()).slice();
    readBuffer.unmap();
    return {
      output: unflattenMatrix([...data], shape.tokens, shape.outputSize),
      projectionCacheHits: {
        upProjection: upProjection.cacheHit,
        ...(input.gateProjection ? { gateProjection: gateProjection.cacheHit } : {}),
        downProjection: downProjection.cacheHit,
      },
      pipelineCacheHits: {
        intermediate: intermediatePipeline.cacheHit,
        output: outputPipeline.cacheHit,
      },
    };
  } finally {
    destroyBuffer(hiddenBuffer);
    if (!upProjection.cached) destroyBuffer(upProjection.buffer);
    if (!gateProjection.cached) destroyBuffer(gateProjection.buffer);
    if (!downProjection.cached) destroyBuffer(downProjection.buffer);
    destroyBuffer(paramsBuffer);
    destroyBuffer(intermediateBuffer);
    destroyBuffer(outputBuffer);
    destroyBuffer(readBuffer);
  }
}

async function runMlpBatchResidentOnGpuDevice(
  device: MinimalGpuDevice,
  input: WebGpuMlpBatchResidentInput,
  shape: MlpShape & { tokens: number },
): Promise<{
  tensor: WebGpuResidentTensor;
  projectionCacheHits?: WebGpuMlpTrace["projectionCacheHits"];
  pipelineCacheHits?: WebGpuMlpTrace["pipelineCacheHits"];
}> {
  const residentHidden = isWebGpuResidentTensor(input.hidden) ? input.hidden : undefined;
  const hiddenBuffer = residentHidden
    ? readResidentTensorBuffer(residentHidden)
    : createUploadedBuffer(device, new Float32Array(flattenMatrix(input.hidden as Matrix)), GPU_STORAGE | GPU_COPY_DST);
  const upProjection = getOrUploadStableProjectionBuffer({
    device,
    input,
    projection: input.upProjection,
    keySuffix: "upProjection",
    rows: shape.intermediateSize,
    cols: shape.inputSize,
  });
  const downProjection = getOrUploadStableProjectionBuffer({
    device,
    input,
    projection: input.downProjection,
    keySuffix: "downProjection",
    rows: shape.outputSize,
    cols: shape.intermediateSize,
  });
  const gateProjection = input.gateProjection
    ? getOrUploadStableProjectionBuffer({
        device,
        input,
        projection: input.gateProjection,
        keySuffix: "gateProjection",
        rows: shape.intermediateSize,
        cols: shape.inputSize,
      })
    : {
        buffer: createUploadedBuffer(device, new Float32Array([0]), GPU_STORAGE | GPU_COPY_DST),
        cacheHit: false,
        cached: false,
      };
  const paramsBuffer = createUploadedBuffer(
    device,
    new Uint32Array([
      shape.inputSize,
      shape.intermediateSize,
      shape.outputSize,
      shape.tokens,
      shape.activationKind === "silu_gated" ? 1 : 0,
      0,
      0,
      0,
    ]),
    GPU_UNIFORM | GPU_COPY_DST,
  );
  const intermediateByteLength = shape.tokens * shape.intermediateSize * Float32Array.BYTES_PER_ELEMENT;
  const outputByteLength = shape.tokens * shape.outputSize * Float32Array.BYTES_PER_ELEMENT;
  const intermediateBuffer = device.createBuffer({ size: intermediateByteLength, usage: GPU_STORAGE | GPU_COPY_SRC });
  const outputBuffer = device.createBuffer({ size: outputByteLength, usage: GPU_STORAGE | GPU_COPY_SRC });
  const buffers = {
    hiddenBuffer,
    upBuffer: upProjection.buffer,
    gateBuffer: gateProjection.buffer,
    downBuffer: downProjection.buffer,
    intermediateBuffer,
    outputBuffer,
    paramsBuffer,
  };

  try {
    const intermediatePipeline = getOrCreateComputePipeline({
      device,
      bufferCache: input.bufferCache,
      key: "mlp-batch:mlp_batch_intermediate",
      moduleKey: "mlp-batch",
      code: mlpBatchWgsl,
      entryPoint: "mlp_batch_intermediate",
    });
    const outputPipeline = getOrCreateComputePipeline({
      device,
      bufferCache: input.bufferCache,
      key: "mlp-batch:mlp_batch_output",
      moduleKey: "mlp-batch",
      code: mlpBatchWgsl,
      entryPoint: "mlp_batch_output",
    });
    const intermediateBindGroup = createMlpBindGroup(device, intermediatePipeline.pipeline, buffers);
    const outputBindGroup = createMlpBindGroup(device, outputPipeline.pipeline, buffers);
    submitOrRecordComputePass(device, input.commandBatch, {
      label: "mlp-batch:resident",
      dispatches: 2,
      record: (encoder) => {
        const pass = encoder.beginComputePass();
        pass.setPipeline(intermediatePipeline.pipeline);
        pass.setBindGroup(0, intermediateBindGroup);
        pass.dispatchWorkgroups(Math.ceil(shape.intermediateSize / 16), Math.ceil(shape.tokens / 16));
        pass.setPipeline(outputPipeline.pipeline);
        pass.setBindGroup(0, outputBindGroup);
        pass.dispatchWorkgroups(Math.ceil(shape.outputSize / 16), Math.ceil(shape.tokens / 16));
        pass.end();
      },
    });
    const retainedBuffers: MinimalGpuBuffer[] = [paramsBuffer, intermediateBuffer];
    if (!residentHidden) retainedBuffers.push(hiddenBuffer);
    if (!upProjection.cached) retainedBuffers.push(upProjection.buffer);
    if (!gateProjection.cached) retainedBuffers.push(gateProjection.buffer);
    if (!downProjection.cached) retainedBuffers.push(downProjection.buffer);
    return {
      tensor: createWebGpuResidentTensor({
        device,
        buffer: outputBuffer,
        rows: shape.tokens,
        cols: shape.outputSize,
        byteLength: outputByteLength,
        retainedBuffers,
      }),
      projectionCacheHits: {
        upProjection: upProjection.cacheHit,
        ...(input.gateProjection ? { gateProjection: gateProjection.cacheHit } : {}),
        downProjection: downProjection.cacheHit,
      },
      pipelineCacheHits: {
        intermediate: intermediatePipeline.cacheHit,
        output: outputPipeline.cacheHit,
      },
    };
  } catch (error) {
    if (!residentHidden) destroyBuffer(hiddenBuffer);
    if (!upProjection.cached) destroyBuffer(upProjection.buffer);
    if (!gateProjection.cached) destroyBuffer(gateProjection.buffer);
    if (!downProjection.cached) destroyBuffer(downProjection.buffer);
    destroyBuffer(paramsBuffer);
    destroyBuffer(intermediateBuffer);
    destroyBuffer(outputBuffer);
    throw error;
  }
}

type WebGpuMlpProjectionInput = Pick<
  WebGpuMlpInput,
  "bufferCache" | "projectionCacheKey" | "projectionCachePolicy"
>;

function getOrUploadStableProjectionBuffer(input: {
  device: MinimalGpuDevice;
  input: WebGpuMlpProjectionInput;
  projection: DenseMatVecMatrix;
  keySuffix: "upProjection" | "downProjection" | "gateProjection";
  rows: number;
  cols: number;
}): { buffer: MinimalGpuBuffer; cacheHit: boolean; cached: boolean } {
  const byteLength = input.rows * input.cols * Float32Array.BYTES_PER_ELEMENT;
  const dataFactory = () => denseMatVecMatrixToFloat32Array(
    input.projection,
    { cols: input.cols },
    allRowIds(input.rows),
  );
  if (
    input.input.bufferCache
    && input.input.projectionCacheKey
    && input.input.projectionCachePolicy === "stable"
    && !Array.isArray(input.projection)
  ) {
    const uploaded = input.input.bufferCache.getOrUploadFloatMatrix({
      device: input.device,
      key: `mlp:${input.input.projectionCacheKey}:${input.keySuffix}`,
      rows: input.rows,
      cols: input.cols,
      usage: GPU_STORAGE | GPU_COPY_DST,
      byteLength,
      dataFactory,
    });
    return { buffer: uploaded.buffer, cacheHit: uploaded.cacheHit, cached: true };
  }
  return {
    buffer: createUploadedBuffer(input.device, dataFactory(), GPU_STORAGE | GPU_COPY_DST),
    cacheHit: false,
    cached: false,
  };
}

function getOrCreateComputePipeline(input: {
  device: MinimalGpuDevice;
  bufferCache?: WebGpuRuntimeBufferCache | undefined;
  key: string;
  moduleKey: string;
  code: string;
  entryPoint: string;
}): { pipeline: MinimalGpuComputePipeline; cacheHit: boolean } {
  if (input.bufferCache) {
    return input.bufferCache.getOrCreateComputePipeline({
      device: input.device,
      key: input.key,
      moduleKey: input.moduleKey,
      code: input.code,
      entryPoint: input.entryPoint,
    });
  }
  const module = input.device.createShaderModule({ code: input.code });
  return {
    pipeline: input.device.createComputePipeline({
      layout: "auto",
      compute: { module, entryPoint: input.entryPoint },
    }),
    cacheHit: false,
  };
}

function createMlpBindGroup(
  device: MinimalGpuDevice,
  pipeline: MinimalGpuComputePipeline,
  buffers: {
    hiddenBuffer: MinimalGpuBuffer;
    upBuffer: MinimalGpuBuffer;
    gateBuffer: MinimalGpuBuffer;
    downBuffer: MinimalGpuBuffer;
    intermediateBuffer: MinimalGpuBuffer;
    outputBuffer: MinimalGpuBuffer;
    paramsBuffer: MinimalGpuBuffer;
  },
): unknown {
  return device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: buffers.hiddenBuffer } },
      { binding: 1, resource: { buffer: buffers.upBuffer } },
      { binding: 2, resource: { buffer: buffers.gateBuffer } },
      { binding: 3, resource: { buffer: buffers.downBuffer } },
      { binding: 4, resource: { buffer: buffers.intermediateBuffer } },
      { binding: 5, resource: { buffer: buffers.outputBuffer } },
      { binding: 6, resource: { buffer: buffers.paramsBuffer } },
    ],
  });
}

function buildSelectedKeyIndexesByQuery(
  q: SsaToyTensorHandle,
  k: SsaToyTensorHandle,
  policy: SparseForwardInput["routingPolicy"],
): number[][] {
  return q.matrix.map((_, queryTokenIndex) => {
    const queryBlockIndex = Math.floor(queryTokenIndex / policy.blockSize);
    const selectedBlockIds = policy.selectedBlockIdsByQueryBlock[queryBlockIndex] ?? policy.pinnedBlockIds;
    const indexes: number[] = [];
    for (const blockId of selectedBlockIds) {
      const range = k.blockTokenRanges[blockId];
      if (!range) continue;
      for (let token = range.tokenStart; token < range.tokenEnd; token += 1) {
        if (token >= 0 && token < k.matrix.length && !indexes.includes(token)) indexes.push(token);
      }
    }
    return indexes.sort((a, b) => a - b);
  });
}

function estimateSparseTokensByQueryBlock(
  k: SsaToyTensorHandle,
  selectedBlockIdsByQueryBlock: Record<number, string[]>,
): number {
  let total = 0;
  for (const selectedBlockIds of Object.values(selectedBlockIdsByQueryBlock)) {
    const seen = new Set<number>();
    for (const blockId of selectedBlockIds) {
      const range = k.blockTokenRanges[blockId];
      if (!range) continue;
      for (let token = range.tokenStart; token < range.tokenEnd; token += 1) seen.add(token);
    }
    total += seen.size;
  }
  return total;
}

function collectSelectedBlockIds(selectedBlockIdsByQueryBlock: Record<number, string[]>): string[] {
  const selected: string[] = [];
  for (const key of Object.keys(selectedBlockIdsByQueryBlock).sort((a, b) => Number(a) - Number(b))) {
    for (const id of selectedBlockIdsByQueryBlock[Number(key)] ?? []) {
      if (!selected.includes(id)) selected.push(id);
    }
  }
  return selected;
}

function buildSparseAttentionTrace(
  backend: WebGpuSparseAttentionTrace["backend"],
  input: WebGpuSparseAttentionInput,
  computeMs: number,
  pipelineCacheHit?: boolean,
): WebGpuSparseAttentionTrace {
  return {
    backend,
    queryTokens: input.q.length,
    keyTokens: input.k.length,
    headDim: input.q[0]?.length ?? 0,
    selectedIndexSlots: input.selectedKeyIndexesByQuery.reduce((sum, indexes) => sum + indexes.length, 0),
    maxSelectedPerQuery: Math.max(1, ...input.selectedKeyIndexesByQuery.map((indexes) => indexes.length)),
    computeMs,
    ...(pipelineCacheHit !== undefined
      ? {
          pipelineCacheKey: "sparse-attention:main",
          pipelineCacheHit,
        }
      : {}),
  };
}

function buildSparseAttentionTraceFromShape(
  backend: WebGpuSparseAttentionTrace["backend"],
  input: Pick<WebGpuSparseAttentionInput, "selectedKeyIndexesByQuery">,
  shape: { queryTokens: number; keyTokens: number; headDim: number },
  computeMs: number,
  pipelineCacheHit?: boolean,
): WebGpuSparseAttentionTrace {
  return {
    backend,
    queryTokens: shape.queryTokens,
    keyTokens: shape.keyTokens,
    headDim: shape.headDim,
    selectedIndexSlots: input.selectedKeyIndexesByQuery.reduce((sum, indexes) => sum + indexes.length, 0),
    maxSelectedPerQuery: Math.max(1, ...input.selectedKeyIndexesByQuery.map((indexes) => indexes.length)),
    computeMs,
    ...(pipelineCacheHit !== undefined
      ? {
          pipelineCacheKey: "sparse-attention:main",
          pipelineCacheHit,
        }
      : {}),
  };
}

function buildPackedSparseAttentionTrace(
  backend: WebGpuPackedSparseAttentionTrace["backend"],
  input: Pick<WebGpuPackedSparseAttentionInput, "selectedKeyIndexesByQuery">,
  shape: { queryTokens: number; keyTokens: number; headCount: number; keyValueHeadCount: number; headDim: number; outputSize: number },
  computeMs: number,
  pipelineCacheHit?: boolean,
): WebGpuPackedSparseAttentionTrace {
  return {
    backend,
    queryTokens: shape.queryTokens,
    keyTokens: shape.keyTokens,
    headDim: shape.headDim,
    selectedIndexSlots: input.selectedKeyIndexesByQuery.reduce((sum, indexes) => sum + indexes.length, 0),
    maxSelectedPerQuery: Math.max(1, ...input.selectedKeyIndexesByQuery.map((indexes) => indexes.length)),
    computeMs,
    packedHeads: true,
    headCount: shape.headCount,
    keyValueHeadCount: shape.keyValueHeadCount,
    outputSize: shape.outputSize,
    dispatchCount: backend === "webgpu" ? 1 : shape.headCount,
    ...(pipelineCacheHit !== undefined
      ? {
          pipelineCacheKey: "packed-sparse-attention:packed_sparse_attention",
          pipelineCacheHit,
        }
      : {}),
  };
}

function withDecodeOptimizedPackedTrace(
  trace: WebGpuPackedSparseAttentionTrace,
  result: { decodeOptimized?: true; pipelineCacheHit?: boolean; qwenOneTokenAttention?: true; dispatchCount?: number },
): WebGpuPackedSparseAttentionTrace {
  if (result.decodeOptimized !== true) return trace;
  const pipelineCacheKey = result.qwenOneTokenAttention === true
    ? "packed-sparse-attention:qwen_one_token_attention"
    : "packed-sparse-attention:decode_scores+decode_output";
  const oneTokenMetadata = trace.queryTokens === 1
    ? {
        metadata: {
          ...(trace.metadata ?? {}),
          fusedStage: "one_token_attention",
          oneTokenAttention: true,
          ...(result.qwenOneTokenAttention === true ? { qwenOneTokenAttention: true } : {}),
        },
      }
    : {};
  return {
    ...trace,
    dispatchCount: result.dispatchCount ?? 2,
    pipelineCacheKey,
    ...oneTokenMetadata,
    ...(result.pipelineCacheHit !== undefined ? { pipelineCacheHit: result.pipelineCacheHit } : {}),
  };
}

function shouldUseQwenOneTokenFullPrefixAttention(
  shape: { queryTokens: number; keyTokens: number; outputSize: number; headCount: number; headDim: number },
  selected: { values: Int32Array; maxSelectedPerQuery: number },
): boolean {
  if (shape.queryTokens !== 1) return false;
  if (shape.outputSize !== shape.headCount * shape.headDim) return false;
  if (shape.keyTokens < 64) return false;
  if (shape.keyTokens > QWEN_ONE_TOKEN_ATTENTION_MAX_SEQUENCE) return false;
  if (selected.maxSelectedPerQuery !== shape.keyTokens) return false;
  if (selected.values.length !== shape.keyTokens) return false;
  for (let index = 0; index < shape.keyTokens; index += 1) {
    if (selected.values[index] !== index) return false;
  }
  return true;
}

function shouldUseDecodeOptimizedPackedAttention(
  _shape: { queryTokens: number },
  maxSelectedPerQuery: number,
): boolean {
  return maxSelectedPerQuery >= 64;
}

function createDecodeOptimizedPackedAttentionPipelines(
  device: MinimalGpuDevice,
  bufferCache?: WebGpuRuntimeBufferCache,
): {
  score: { pipeline: MinimalGpuComputePipeline; cacheHit: boolean };
  output: { pipeline: MinimalGpuComputePipeline; cacheHit: boolean };
} {
  return {
    score: getOrCreateComputePipeline({
      device,
      bufferCache,
      key: "packed-sparse-attention:decode_scores",
      moduleKey: "packed-sparse-attention-decode",
      code: packedSparseAttentionDecodeWgsl,
      entryPoint: "packed_sparse_attention_decode_scores",
    }),
    output: getOrCreateComputePipeline({
      device,
      bufferCache,
      key: "packed-sparse-attention:decode_output",
      moduleKey: "packed-sparse-attention-decode",
      code: packedSparseAttentionDecodeWgsl,
      entryPoint: "packed_sparse_attention_decode_output",
    }),
  };
}

function buildResidentTensorTrace(
  pipelineCacheKey: string,
  shape: { tokens: number; hidden: number },
  computeMs: number,
  inputResident: boolean,
  metadata?: Record<string, unknown> | undefined,
  pipelineCacheHit?: boolean,
): WebGpuResidentTensorTrace {
  return {
    backend: "webgpu",
    tokens: shape.tokens,
    hidden: shape.hidden,
    computeMs,
    outputResident: true,
    readback: false,
    ...(inputResident ? { inputResident: true } : {}),
    pipelineCacheKey,
    ...(pipelineCacheHit !== undefined ? { pipelineCacheHit } : {}),
    ...(metadata ? { metadata: { ...metadata } } : {}),
  };
}

function validateSparseAttentionInput(input: WebGpuSparseAttentionInput): void {
  validateMatrix(input.q, "q");
  validateMatrix(input.k, "k");
  validateMatrix(input.v, "v");
  const headDim = input.q[0]?.length ?? 0;
  if (input.k.some((row) => row.length !== headDim) || input.v.some((row) => row.length !== headDim)) {
    throw new Error("q/k/v head dimensions must match for sparse attention.");
  }
  if (input.k.length !== input.v.length) throw new Error("k/v token counts must match for sparse attention.");
  if (input.selectedKeyIndexesByQuery.length !== input.q.length) {
    throw new Error("selectedKeyIndexesByQuery must include one row per query token.");
  }
}

function validateSparseAttentionResidentInput(input: WebGpuSparseAttentionResidentInput): { queryTokens: number; keyTokens: number; headDim: number } {
  const qShape = isWebGpuResidentTensor(input.q)
    ? validateResidentTensor(input.q)
    : validateActivationMatrixShape(input.q, "resident sparse attention q");
  const kShape = isWebGpuResidentTensor(input.k)
    ? validateResidentTensor(input.k)
    : validateActivationMatrixShape(input.k, "resident sparse attention k");
  const vShape = isWebGpuResidentTensor(input.v)
    ? validateResidentTensor(input.v)
    : validateActivationMatrixShape(input.v, "resident sparse attention v");
  if (kShape.hidden !== qShape.hidden || vShape.hidden !== qShape.hidden) {
    throw new Error("q/k/v head dimensions must match for resident sparse attention.");
  }
  if (kShape.tokens !== vShape.tokens) throw new Error("k/v token counts must match for resident sparse attention.");
  if (input.selectedKeyIndexesByQuery.length !== qShape.tokens) {
    throw new Error("selectedKeyIndexesByQuery must include one row per resident query token.");
  }
  return {
    queryTokens: qShape.tokens,
    keyTokens: kShape.tokens,
    headDim: qShape.hidden,
  };
}

function validatePackedSparseAttentionInput(input: WebGpuPackedSparseAttentionInput): {
  queryTokens: number;
  keyTokens: number;
  headCount: number;
  keyValueHeadCount: number;
  headDim: number;
  outputSize: number;
} {
  validateMatrix(input.q, "packed sparse attention q");
  validateMatrix(input.k, "packed sparse attention k");
  validateMatrix(input.v, "packed sparse attention v");
  if (!Number.isInteger(input.headCount) || input.headCount <= 0) throw new Error("packed sparse attention headCount must be positive.");
  if (!Number.isInteger(input.headDim) || input.headDim <= 0) throw new Error("packed sparse attention headDim must be positive.");
  const keyValueHeadCount = input.keyValueHeadCount ?? input.headCount;
  if (!Number.isInteger(keyValueHeadCount) || keyValueHeadCount <= 0) {
    throw new Error("packed sparse attention keyValueHeadCount must be positive.");
  }
  const queryWidth = input.headCount * input.headDim;
  const keyValueWidth = keyValueHeadCount * input.headDim;
  if (input.q.some((row) => row.length !== queryWidth)) {
    throw new Error("packed sparse attention q rows must equal headCount * headDim.");
  }
  if (input.k.some((row) => row.length !== keyValueWidth) || input.v.some((row) => row.length !== keyValueWidth)) {
    throw new Error("packed sparse attention k/v rows must equal keyValueHeadCount * headDim.");
  }
  if (input.k.length !== input.v.length) throw new Error("packed sparse attention k/v token counts must match.");
  if (input.selectedKeyIndexesByQuery.length !== input.q.length) {
    throw new Error("selectedKeyIndexesByQuery must include one row per packed query token.");
  }
  return {
    queryTokens: input.q.length,
    keyTokens: input.k.length,
    headCount: input.headCount,
    keyValueHeadCount,
    headDim: input.headDim,
    outputSize: queryWidth,
  };
}

function validatePackedSparseAttentionResidentInput(input: WebGpuPackedSparseAttentionResidentInput): {
  queryTokens: number;
  keyTokens: number;
  headCount: number;
  keyValueHeadCount: number;
  headDim: number;
  outputSize: number;
} {
  if (!Number.isInteger(input.headCount) || input.headCount <= 0) throw new Error("resident packed sparse attention headCount must be positive.");
  if (!Number.isInteger(input.headDim) || input.headDim <= 0) throw new Error("resident packed sparse attention headDim must be positive.");
  const keyValueHeadCount = input.keyValueHeadCount ?? input.headCount;
  if (!Number.isInteger(keyValueHeadCount) || keyValueHeadCount <= 0) {
    throw new Error("resident packed sparse attention keyValueHeadCount must be positive.");
  }
  const qShape = isWebGpuResidentTensor(input.q)
    ? validateResidentTensor(input.q)
    : validateActivationMatrixShape(input.q, "resident packed sparse attention q");
  const kShape = isWebGpuResidentTensor(input.k)
    ? validateResidentTensor(input.k)
    : validateActivationMatrixShape(input.k, "resident packed sparse attention k");
  const vShape = isWebGpuResidentTensor(input.v)
    ? validateResidentTensor(input.v)
    : validateActivationMatrixShape(input.v, "resident packed sparse attention v");
  const queryWidth = input.headCount * input.headDim;
  const keyValueWidth = keyValueHeadCount * input.headDim;
  if (qShape.hidden !== queryWidth) throw new Error("resident packed sparse attention q width must equal headCount * headDim.");
  if (kShape.hidden !== keyValueWidth || vShape.hidden !== keyValueWidth) {
    throw new Error("resident packed sparse attention k/v width must equal keyValueHeadCount * headDim.");
  }
  if (kShape.tokens !== vShape.tokens) throw new Error("resident packed sparse attention k/v token counts must match.");
  if (input.selectedKeyIndexesByQuery.length !== qShape.tokens) {
    throw new Error("selectedKeyIndexesByQuery must include one row per resident packed query token.");
  }
  return {
    queryTokens: qShape.tokens,
    keyTokens: kShape.tokens,
    headCount: input.headCount,
    keyValueHeadCount,
    headDim: input.headDim,
    outputSize: queryWidth,
  };
}

function validateDenseMatVecInput(input: WebGpuDenseMatVecInput): { rows: number; cols: number } {
  const rows = denseMatVecRowCount(input.matrix);
  const cols = denseMatVecColCount(input.matrix);
  if (!Number.isInteger(rows) || rows <= 0) throw new Error("dense matvec matrix must have at least one row.");
  if (!Number.isInteger(cols) || cols <= 0) throw new Error("dense matvec matrix rows must not be empty.");
  if (input.vector.length !== cols) {
    throw new Error(`dense matvec vector dimension ${input.vector.length} must match matrix column count ${cols}.`);
  }
  const rowsToValidate = input.selectedRowIds ?? Array.from({ length: rows }, (_, rowIndex) => rowIndex);
  if (input.selectedRowIds) {
    if (input.selectedRowIds.length === 0) throw new Error("dense matvec selectedRowIds must not be empty when provided.");
    for (const rowId of input.selectedRowIds) {
      if (!Number.isInteger(rowId) || rowId < 0 || rowId >= rows) {
        throw new Error(`dense matvec selected row ${rowId} is outside matrix row range 0..${rows - 1}.`);
      }
    }
  }
  if (Array.isArray(input.matrix)) {
    for (const rowIndex of rowsToValidate) {
      readDenseMatVecRow(input.matrix, rowIndex, cols);
    }
  }
  return { rows, cols };
}

function validateDenseMatVecResidentInput(input: WebGpuDenseMatVecResidentInput): { rows: number; cols: number } {
  const vectorShape = validateResidentTensor(input.vector);
  if (vectorShape.tokens !== 1) throw new Error("resident dense matvec vector must be a single-row tensor.");
  const rows = denseMatVecRowCount(input.matrix);
  const cols = denseMatVecColCount(input.matrix);
  if (!Number.isInteger(rows) || rows <= 0) throw new Error("resident dense matvec matrix must have at least one row.");
  if (!Number.isInteger(cols) || cols <= 0) throw new Error("resident dense matvec matrix rows must not be empty.");
  if (vectorShape.hidden !== cols) {
    throw new Error(`resident dense matvec vector dimension ${vectorShape.hidden} must match matrix column count ${cols}.`);
  }
  const rowsToValidate = input.selectedRowIds ?? Array.from({ length: rows }, (_, rowIndex) => rowIndex);
  if (input.selectedRowIds) {
    if (input.selectedRowIds.length === 0) throw new Error("resident dense matvec selectedRowIds must not be empty when provided.");
    for (const rowId of input.selectedRowIds) {
      if (!Number.isInteger(rowId) || rowId < 0 || rowId >= rows) {
        throw new Error(`resident dense matvec selected row ${rowId} is outside matrix row range 0..${rows - 1}.`);
      }
    }
  }
  if (Array.isArray(input.matrix)) {
    for (const rowIndex of rowsToValidate) readDenseMatVecRow(input.matrix, rowIndex, cols);
  }
  return { rows, cols };
}

function validateDenseMatMulInput(input: WebGpuDenseMatMulInput): { tokens: number; hidden: number; outputSize: number } {
  validateMatrix(input.activations, "dense matmul activations");
  const tokens = input.activations.length;
  const hidden = input.activations[0]?.length ?? 0;
  const outputSize = denseMatVecRowCount(input.projection);
  const projectionCols = denseMatVecColCount(input.projection);
  if (!Number.isInteger(outputSize) || outputSize <= 0) throw new Error("dense matmul projection must have at least one row.");
  if (projectionCols !== hidden) {
    throw new Error(`dense matmul projection column count ${projectionCols} must match activation hidden size ${hidden}.`);
  }
  if (Array.isArray(input.projection)) {
    for (const rowIndex of allRowIds(outputSize)) readDenseMatVecRow(input.projection, rowIndex, hidden);
  }
  return { tokens, hidden, outputSize };
}

function validateDenseMatMulResidentInput(input: WebGpuDenseMatMulResidentInput): { tokens: number; hidden: number; outputSize: number } {
  const activationShape = isWebGpuResidentTensor(input.activations)
    ? validateResidentTensor(input.activations)
    : validateActivationMatrixShape(input.activations, "GPU-resident dense matmul activations");
  const outputSize = denseMatVecRowCount(input.projection);
  const projectionCols = denseMatVecColCount(input.projection);
  if (!Number.isInteger(outputSize) || outputSize <= 0) throw new Error("dense matmul projection must have at least one row.");
  if (projectionCols !== activationShape.hidden) {
    throw new Error(`dense matmul projection column count ${projectionCols} must match activation hidden size ${activationShape.hidden}.`);
  }
  if (Array.isArray(input.projection)) {
    for (const rowIndex of allRowIds(outputSize)) readDenseMatVecRow(input.projection, rowIndex, activationShape.hidden);
  }
  return {
    tokens: activationShape.tokens,
    hidden: activationShape.hidden,
    outputSize,
  };
}

function validatePackedQkvProjectionResidentInput(input: WebGpuPackedQkvProjectionResidentInput): {
  tokens: number;
  hidden: number;
  qOutputSize: number;
  kOutputSize: number;
  vOutputSize: number;
} {
  const hiddenShape = isWebGpuResidentTensor(input.hidden)
    ? validateResidentTensor(input.hidden)
    : validateActivationMatrixShape(input.hidden, "packed resident QKV projection hidden");
  if (hiddenShape.tokens !== 1) {
    throw new Error("packed resident QKV projection currently supports one decode token at a time.");
  }
  const qOutputSize = validateProjectionForPackedQkv(input.qProjection, hiddenShape.hidden, "qProjection");
  const kOutputSize = validateProjectionForPackedQkv(input.kProjection, hiddenShape.hidden, "kProjection");
  const vOutputSize = validateProjectionForPackedQkv(input.vProjection, hiddenShape.hidden, "vProjection");
  return {
    tokens: hiddenShape.tokens,
    hidden: hiddenShape.hidden,
    qOutputSize,
    kOutputSize,
    vOutputSize,
  };
}

function validateProjectionForPackedQkv(
  projection: DenseMatVecMatrix,
  hidden: number,
  name: "qProjection" | "kProjection" | "vProjection",
): number {
  const outputSize = denseMatVecRowCount(projection);
  const cols = denseMatVecColCount(projection);
  if (!Number.isInteger(outputSize) || outputSize <= 0) throw new Error(`packed resident QKV ${name} must have at least one row.`);
  if (cols !== hidden) throw new Error(`packed resident QKV ${name} column count ${cols} must match hidden size ${hidden}.`);
  if (Array.isArray(projection)) {
    for (const rowIndex of allRowIds(outputSize)) readDenseMatVecRow(projection, rowIndex, hidden);
  }
  return outputSize;
}

function validateTokenEmbeddingLookupInput(input: WebGpuTokenEmbeddingLookupResidentInput): {
  tokenId: number;
  vocabSize: number;
  hidden: number;
} {
  const vocabSize = denseMatVecRowCount(input.tokenEmbedding);
  const hidden = denseMatVecColCount(input.tokenEmbedding);
  if (!Number.isInteger(vocabSize) || vocabSize <= 0) throw new Error("resident token embedding lookup requires at least one token row.");
  if (!Number.isInteger(hidden) || hidden <= 0) throw new Error("resident token embedding lookup requires non-empty embedding rows.");
  if (!Number.isInteger(input.tokenId) || input.tokenId < 0 || input.tokenId >= vocabSize) {
    throw new Error(`resident token embedding lookup tokenId ${input.tokenId} is outside embedding row range 0..${vocabSize - 1}.`);
  }
  if (Array.isArray(input.tokenEmbedding)) {
    for (const rowId of allRowIds(vocabSize)) readDenseMatVecRow(input.tokenEmbedding, rowId, hidden);
  }
  return {
    tokenId: input.tokenId,
    vocabSize,
    hidden,
  };
}

function validateActivationMatrixShape(matrix: Matrix, name: string): { tokens: number; hidden: number } {
  validateMatrix(matrix, name);
  return {
    tokens: matrix.length,
    hidden: matrix[0]?.length ?? 0,
  };
}

function validateResidentTensor(tensor: WebGpuResidentTensor): { tokens: number; hidden: number } {
  if (!isWebGpuResidentTensor(tensor)) throw new Error("Expected a WebGPU-resident tensor.");
  if (!Number.isInteger(tensor.rows) || tensor.rows <= 0) throw new Error("WebGPU-resident tensor rows must be a positive integer.");
  if (!Number.isInteger(tensor.cols) || tensor.cols <= 0) throw new Error("WebGPU-resident tensor cols must be a positive integer.");
  const expectedByteLength = tensor.rows * tensor.cols * Float32Array.BYTES_PER_ELEMENT;
  if (tensor.byteLength !== expectedByteLength) {
    throw new Error(`WebGPU-resident tensor byteLength ${tensor.byteLength} does not match ${tensor.rows}x${tensor.cols} f32 shape.`);
  }
  readResidentTensorBuffer(tensor);
  return {
    tokens: tensor.rows,
    hidden: tensor.cols,
  };
}

function validateResidentBinaryInput(
  left: Matrix | WebGpuResidentTensor,
  right: Matrix | WebGpuResidentTensor,
  name: string,
): { tokens: number; hidden: number } {
  const leftShape = isWebGpuResidentTensor(left)
    ? validateResidentTensor(left)
    : validateActivationMatrixShape(left, `${name} left`);
  const rightShape = isWebGpuResidentTensor(right)
    ? validateResidentTensor(right)
    : validateActivationMatrixShape(right, `${name} right`);
  if (leftShape.tokens !== rightShape.tokens || leftShape.hidden !== rightShape.hidden) {
    throw new Error(`${name} tensor shapes must match.`);
  }
  return leftShape;
}

function validateResidualRmsNormPairResidentInput(input: WebGpuResidualRmsNormPairResidentInput): { tokens: number; hidden: number } {
  const shape = validateResidentBinaryInput(input.left, input.right, "resident residual RMSNorm");
  if (shape.tokens !== 1) {
    throw new Error("resident residual RMSNorm fusion is decode-only and requires exactly one token.");
  }
  if (input.weight.length !== shape.hidden) {
    throw new Error(`resident residual RMSNorm weight length ${input.weight.length} must match hidden size ${shape.hidden}.`);
  }
  return shape;
}

function validateQkvPostProjectionResidentInput(input: WebGpuQkvPostProjectionResidentInput): { tokens: number; hidden: number } {
  const shape = isWebGpuResidentTensor(input.projected)
    ? validateResidentTensor(input.projected)
    : validateActivationMatrixShape(input.projected, "resident Qwen Q/K post-projection input");
  if (!Number.isInteger(input.headCount) || input.headCount <= 0) {
    throw new Error("resident Qwen Q/K post-projection headCount must be a positive integer.");
  }
  if (!Number.isInteger(input.headDim) || input.headDim <= 0) {
    throw new Error("resident Qwen Q/K post-projection headDim must be a positive integer.");
  }
  if (shape.hidden !== input.headCount * input.headDim) {
    throw new Error(
      `resident Qwen Q/K post-projection hidden size ${shape.hidden} must equal headCount ${input.headCount} * headDim ${input.headDim}.`,
    );
  }
  if (input.positions.length !== shape.tokens) {
    throw new Error(`resident Qwen Q/K post-projection positions length ${input.positions.length} must match token count ${shape.tokens}.`);
  }
  for (const position of input.positions) {
    if (!Number.isInteger(position) || position < 0) {
      throw new Error("resident Qwen Q/K post-projection positions must be non-negative integers.");
    }
  }
  if (input.normWeight && input.normWeight.length !== input.headDim) {
    throw new Error(`resident Qwen Q/K post-projection norm weight length ${input.normWeight.length} must match headDim ${input.headDim}.`);
  }
  if (input.ropeTheta !== undefined && input.ropeTheta > 0 && input.headDim % 2 !== 0) {
    throw new Error("resident Qwen Q/K RoPE requires an even headDim.");
  }
  return shape;
}

function validateQkvNormRopePairResidentInput(
  input: WebGpuQkvNormRopePairResidentInput,
): { tokens: number; qHidden: number; kHidden: number } {
  const qShape = isWebGpuResidentTensor(input.qProjected)
    ? validateResidentTensor(input.qProjected)
    : validateActivationMatrixShape(input.qProjected, "resident Qwen Q/K norm+RoPE Q input");
  const kShape = isWebGpuResidentTensor(input.kProjected)
    ? validateResidentTensor(input.kProjected)
    : validateActivationMatrixShape(input.kProjected, "resident Qwen Q/K norm+RoPE K input");
  if (qShape.tokens !== kShape.tokens) {
    throw new Error("resident Qwen Q/K norm+RoPE Q and K token counts must match.");
  }
  if (!Number.isInteger(input.qHeadCount) || input.qHeadCount <= 0) {
    throw new Error("resident Qwen Q/K norm+RoPE qHeadCount must be a positive integer.");
  }
  if (!Number.isInteger(input.kHeadCount) || input.kHeadCount <= 0) {
    throw new Error("resident Qwen Q/K norm+RoPE kHeadCount must be a positive integer.");
  }
  if (!Number.isInteger(input.headDim) || input.headDim <= 0) {
    throw new Error("resident Qwen Q/K norm+RoPE headDim must be a positive integer.");
  }
  if (qShape.hidden !== input.qHeadCount * input.headDim) {
    throw new Error(
      `resident Qwen Q/K norm+RoPE q hidden size ${qShape.hidden} must equal qHeadCount ${input.qHeadCount} * headDim ${input.headDim}.`,
    );
  }
  if (kShape.hidden !== input.kHeadCount * input.headDim) {
    throw new Error(
      `resident Qwen Q/K norm+RoPE k hidden size ${kShape.hidden} must equal kHeadCount ${input.kHeadCount} * headDim ${input.headDim}.`,
    );
  }
  if (input.positions.length !== qShape.tokens) {
    throw new Error(`resident Qwen Q/K norm+RoPE positions length ${input.positions.length} must match token count ${qShape.tokens}.`);
  }
  for (const position of input.positions) {
    if (!Number.isInteger(position) || position < 0) {
      throw new Error("resident Qwen Q/K norm+RoPE positions must be non-negative integers.");
    }
  }
  if (input.qNormWeight && input.qNormWeight.length !== input.headDim) {
    throw new Error(`resident Qwen Q/K norm+RoPE qNormWeight length ${input.qNormWeight.length} must match headDim ${input.headDim}.`);
  }
  if (input.kNormWeight && input.kNormWeight.length !== input.headDim) {
    throw new Error(`resident Qwen Q/K norm+RoPE kNormWeight length ${input.kNormWeight.length} must match headDim ${input.headDim}.`);
  }
  if (input.ropeTheta !== undefined && input.ropeTheta > 0 && input.headDim % 2 !== 0) {
    throw new Error("resident Qwen Q/K norm+RoPE requires an even headDim.");
  }
  return { tokens: qShape.tokens, qHidden: qShape.hidden, kHidden: kShape.hidden };
}

interface MlpShape {
  tokens?: number;
  inputSize: number;
  intermediateSize: number;
  outputSize: number;
  activationKind: WebGpuMlpActivationKind;
}

function validateMlpInput(input: WebGpuMlpInput): MlpShape {
  const inputSize = input.hidden.length;
  if (!Number.isInteger(inputSize) || inputSize <= 0) throw new Error("MLP hidden vector must not be empty.");
  const intermediateSize = denseMatVecRowCount(input.upProjection);
  const upCols = denseMatVecColCount(input.upProjection);
  if (!Number.isInteger(intermediateSize) || intermediateSize <= 0) throw new Error("MLP upProjection must have at least one row.");
  if (upCols !== inputSize) {
    throw new Error(`MLP upProjection column count ${upCols} must match hidden size ${inputSize}.`);
  }
  if (Array.isArray(input.upProjection)) {
    for (const rowIndex of allRowIds(intermediateSize)) readDenseMatVecRow(input.upProjection, rowIndex, inputSize);
  }

  const outputSize = denseMatVecRowCount(input.downProjection);
  const downCols = denseMatVecColCount(input.downProjection);
  if (!Number.isInteger(outputSize) || outputSize <= 0) throw new Error("MLP downProjection must have at least one row.");
  if (downCols !== intermediateSize) {
    throw new Error(`MLP downProjection column count ${downCols} must match intermediate size ${intermediateSize}.`);
  }
  if (Array.isArray(input.downProjection)) {
    for (const rowIndex of allRowIds(outputSize)) readDenseMatVecRow(input.downProjection, rowIndex, intermediateSize);
  }

  if (input.gateProjection) {
    const gateRows = denseMatVecRowCount(input.gateProjection);
    const gateCols = denseMatVecColCount(input.gateProjection);
    if (gateRows !== intermediateSize || gateCols !== inputSize) {
      throw new Error("MLP gateProjection shape must match upProjection rows and hidden size.");
    }
    if (Array.isArray(input.gateProjection)) {
      for (const rowIndex of allRowIds(intermediateSize)) readDenseMatVecRow(input.gateProjection, rowIndex, inputSize);
    }
  }

  return {
    inputSize,
    intermediateSize,
    outputSize,
    activationKind: input.gateProjection ? "silu_gated" : "gelu",
  };
}

function validateMlpBatchInput(input: WebGpuMlpBatchInput): MlpShape & { tokens: number } {
  validateMatrix(input.hidden, "MLP hidden batch");
  const firstRow = input.hidden[0];
  if (!firstRow) throw new Error("MLP hidden batch must not be empty.");
  const shape = validateMlpInput({
    ...input,
    hidden: firstRow,
  });
  return {
    ...shape,
    tokens: input.hidden.length,
  };
}

function validateMlpBatchResidentInput(input: WebGpuMlpBatchResidentInput): MlpShape & { tokens: number } {
  const hiddenShape = isWebGpuResidentTensor(input.hidden)
    ? validateResidentTensor(input.hidden)
    : validateActivationMatrixShape(input.hidden, "resident MLP hidden batch");
  const inputSize = hiddenShape.hidden;
  const intermediateSize = denseMatVecRowCount(input.upProjection);
  const upCols = denseMatVecColCount(input.upProjection);
  if (!Number.isInteger(intermediateSize) || intermediateSize <= 0) throw new Error("resident MLP upProjection must have at least one row.");
  if (upCols !== inputSize) {
    throw new Error(`resident MLP upProjection column count ${upCols} must match hidden size ${inputSize}.`);
  }
  if (Array.isArray(input.upProjection)) {
    for (const rowIndex of allRowIds(intermediateSize)) readDenseMatVecRow(input.upProjection, rowIndex, inputSize);
  }

  const outputSize = denseMatVecRowCount(input.downProjection);
  const downCols = denseMatVecColCount(input.downProjection);
  if (!Number.isInteger(outputSize) || outputSize <= 0) throw new Error("resident MLP downProjection must have at least one row.");
  if (downCols !== intermediateSize) {
    throw new Error(`resident MLP downProjection column count ${downCols} must match intermediate size ${intermediateSize}.`);
  }
  if (Array.isArray(input.downProjection)) {
    for (const rowIndex of allRowIds(outputSize)) readDenseMatVecRow(input.downProjection, rowIndex, intermediateSize);
  }

  if (input.gateProjection) {
    const gateRows = denseMatVecRowCount(input.gateProjection);
    const gateCols = denseMatVecColCount(input.gateProjection);
    if (gateRows !== intermediateSize || gateCols !== inputSize) {
      throw new Error("resident MLP gateProjection shape must match upProjection rows and hidden size.");
    }
    if (Array.isArray(input.gateProjection)) {
      for (const rowIndex of allRowIds(intermediateSize)) readDenseMatVecRow(input.gateProjection, rowIndex, inputSize);
    }
  }

  return {
    tokens: hiddenShape.tokens,
    inputSize,
    intermediateSize,
    outputSize,
    activationKind: input.gateProjection ? "silu_gated" : "gelu",
  };
}

function validateMatrix(matrix: Matrix, name: string): void {
  if (matrix.length === 0) throw new Error(`${name} matrix must not be empty.`);
  const dim = matrix[0]?.length ?? 0;
  if (dim === 0) throw new Error(`${name} matrix rows must not be empty.`);
  for (const row of matrix) {
    if (row.length !== dim) throw new Error(`${name} matrix rows must all have dimension ${dim}.`);
  }
}

function normalizeRanges(ranges: Record<string, SsaToyTensorRange>): Record<string, SsaToyTensorRange> {
  const normalized: Record<string, SsaToyTensorRange> = {};
  for (const [id, range] of Object.entries(ranges)) {
    normalized[id] = {
      tokenStart: Math.max(0, Math.floor(range.tokenStart)),
      tokenEnd: Math.max(0, Math.floor(range.tokenEnd)),
    };
  }
  return normalized;
}

function isToyTensorHandle(handle: unknown): handle is SsaToyTensorHandle {
  return typeof handle === "object"
    && handle !== null
    && (handle as SsaToyTensorHandle).kind === "ssa_toy_tensor"
    && typeof (handle as SsaToyTensorHandle).id === "string"
    && Array.isArray((handle as SsaToyTensorHandle).matrix)
    && typeof (handle as SsaToyTensorHandle).blockTokenRanges === "object"
    && (handle as SsaToyTensorHandle).blockTokenRanges !== null;
}

function flattenSelectedIndexes(indexesByQuery: number[][]): { values: Int32Array; maxSelectedPerQuery: number } {
  const maxSelectedPerQuery = Math.max(1, ...indexesByQuery.map((indexes) => indexes.length));
  const values = new Int32Array(indexesByQuery.length * maxSelectedPerQuery);
  values.fill(-1);
  for (let query = 0; query < indexesByQuery.length; query += 1) {
    const indexes = indexesByQuery[query] ?? [];
    for (let slot = 0; slot < Math.min(indexes.length, maxSelectedPerQuery); slot += 1) {
      values[query * maxSelectedPerQuery + slot] = Math.trunc(indexes[slot] ?? -1);
    }
  }
  return { values, maxSelectedPerQuery };
}

function flattenMatrix(matrix: Matrix): number[] {
  return matrix.flatMap((row) => row);
}

function slicePackedHeadMatrix(matrix: Matrix, headIndex: number, headDim: number): Matrix {
  const start = headIndex * headDim;
  return matrix.map((row) => row.slice(start, start + headDim));
}

function mapQueryHeadToKeyValueHead(headIndex: number, headCount: number, keyValueHeadCount: number): number {
  return Math.min(keyValueHeadCount - 1, Math.floor((headIndex * keyValueHeadCount) / Math.max(1, headCount)));
}

function denseMatVecMatrixToFloat32Array(matrix: DenseMatVecMatrix, shape: { cols: number }, rowIds: number[]): Float32Array {
  const optimized = !Array.isArray(matrix) ? matrix.toFloat32Array?.(rowIds) : undefined;
  if (optimized) {
    const expectedLength = rowIds.length * shape.cols;
    if (optimized.length !== expectedLength) {
      throw new Error(`dense matrix descriptor returned ${optimized.length} f32 values; expected ${expectedLength}.`);
    }
    return optimized;
  }
  const values = new Float32Array(rowIds.length * shape.cols);
  let outputOffset = 0;
  for (const rowId of rowIds) {
    values.set(Array.from(readDenseMatVecRow(matrix, rowId, shape.cols)), outputOffset);
    outputOffset += shape.cols;
  }
  return values;
}

function denseMatVecRowCount(matrix: DenseMatVecMatrix): number {
  return Array.isArray(matrix) ? matrix.length : matrix.rowCount;
}

function denseMatVecColCount(matrix: DenseMatVecMatrix): number {
  return Array.isArray(matrix) ? matrix[0]?.length ?? 0 : matrix.colCount;
}

function readDenseMatVecRow(matrix: DenseMatVecMatrix, rowIndex: number, cols: number): ArrayLike<number> {
  const row = Array.isArray(matrix) ? matrix[rowIndex] : matrix.row(rowIndex);
  if (!row) throw new Error(`dense matvec matrix is missing row ${rowIndex}.`);
  if (row.length !== cols) throw new Error(`dense matvec matrix row ${rowIndex} must have dimension ${cols}.`);
  return row;
}

function normalizeDenseMatVecRowIds(selectedRowIds: number[] | undefined, rows: number): number[] {
  return selectedRowIds ? [...selectedRowIds] : Array.from({ length: rows }, (_, index) => index);
}

function normalizeDenseMatVecSuppressedRowIds(suppressedRowIds: number[] | undefined, rows: number): number[] {
  if (!suppressedRowIds || suppressedRowIds.length === 0) return [];
  return [...new Set(suppressedRowIds
    .filter((rowId) => Number.isInteger(rowId) && rowId >= 0 && rowId < rows)
    .map((rowId) => Math.trunc(rowId)))]
    .sort((left, right) => left - right);
}

function allRowIds(rows: number): number[] {
  return Array.from({ length: rows }, (_, index) => index);
}

function rowIdsInRange(start: number, end: number): number[] {
  return Array.from({ length: Math.max(0, end - start) }, (_value, index) => start + index);
}

function planTokenEmbeddingTile(
  shape: { tokenId: number; vocabSize: number },
  requestedTileRows: number | undefined,
): { start: number; end: number; rows: number } {
  const tileRows = typeof requestedTileRows === "number" && Number.isInteger(requestedTileRows) && requestedTileRows > 0
    ? Math.trunc(requestedTileRows)
    : DEFAULT_TOKEN_EMBEDDING_TILE_ROWS;
  const start = Math.floor(shape.tokenId / tileRows) * tileRows;
  const end = Math.min(shape.vocabSize, start + tileRows);
  return { start, end, rows: Math.max(0, end - start) };
}

function tokenEmbeddingTileCacheKey(
  projectionCacheKey: string,
  tile: { start: number; end: number },
): string {
  return `token-embedding:${projectionCacheKey}:rows:${tile.start}-${tile.end}`;
}

function dotArrayLike(left: ArrayLike<number>, right: ArrayLike<number>): number {
  let total = 0;
  for (let index = 0; index < left.length; index += 1) total += (left[index] ?? 0) * (right[index] ?? 0);
  return total;
}

function multiplyDenseMatrixByVector(
  matrix: DenseMatVecMatrix,
  rows: number,
  cols: number,
  vector: ArrayLike<number>,
): number[] {
  return allRowIds(rows).map((rowId) => dotArrayLike(readDenseMatVecRow(matrix, rowId, cols), vector));
}

function buildDenseMatVecTrace(
  backend: WebGpuDenseMatVecTrace["backend"],
  input: Pick<WebGpuDenseMatVecInput, "projectionCacheKey" | "traceMetadata">,
  shape: { rows: number; cols: number },
  selectedRows: number,
  computeMs: number,
  projectionCacheHit = false,
  pipelineCacheHit?: boolean,
): WebGpuDenseMatVecTrace {
  return {
    backend,
    rows: shape.rows,
    cols: shape.cols,
    selectedRows,
    computeMs,
    ...(input.projectionCacheKey ? { projectionCacheKey: input.projectionCacheKey, projectionCacheHit } : {}),
    ...(pipelineCacheHit !== undefined
      ? {
          pipelineCacheKey: "dense-matvec:main",
          pipelineCacheHit,
        }
      : {}),
    ...(input.traceMetadata ? { metadata: { ...input.traceMetadata } } : {}),
  };
}

function buildDenseMatMulTrace(
  backend: WebGpuDenseMatMulTrace["backend"],
  input: Pick<WebGpuDenseMatMulInput, "projectionCacheKey" | "traceMetadata">,
  shape: { tokens: number; hidden: number; outputSize: number },
  computeMs: number,
  projectionCacheHit: boolean,
  pipelineCacheHit?: boolean,
): WebGpuDenseMatMulTrace {
  return {
    backend,
    tokens: shape.tokens,
    hidden: shape.hidden,
    outputSize: shape.outputSize,
    computeMs,
    ...(input.projectionCacheKey ? { projectionCacheKey: input.projectionCacheKey, projectionCacheHit } : {}),
    ...(pipelineCacheHit !== undefined
      ? {
          pipelineCacheKey: "dense-matmul:dense_matmul",
          pipelineCacheHit,
        }
      : {}),
    ...(input.traceMetadata ? { metadata: { ...input.traceMetadata } } : {}),
  };
}

function buildMlpTrace(
  backend: WebGpuMlpTrace["backend"],
  input: Pick<WebGpuMlpInput, "traceMetadata">,
  shape: MlpShape,
  computeMs: number,
  projectionCacheHits?: WebGpuMlpTrace["projectionCacheHits"],
  pipelineCacheHits?: WebGpuMlpTrace["pipelineCacheHits"],
): WebGpuMlpTrace {
  return {
    backend,
    ...(shape.tokens !== undefined ? { tokens: shape.tokens } : {}),
    inputSize: shape.inputSize,
    intermediateSize: shape.intermediateSize,
    outputSize: shape.outputSize,
    activationKind: shape.activationKind,
    computeMs,
    ...(projectionCacheHits ? { projectionCacheHits } : {}),
    ...(pipelineCacheHits ? { pipelineCacheHits } : {}),
    ...(input.traceMetadata ? { metadata: { ...input.traceMetadata } } : {}),
  };
}

function gelu(value: number): number {
  return 0.5 * value * (1 + Math.tanh(Math.sqrt(2 / Math.PI) * (value + 0.044715 * value ** 3)));
}

function silu(value: number): number {
  return value / (1 + Math.exp(-value));
}

function unflattenMatrix(values: number[], rows: number, cols: number): Matrix {
  const out: Matrix = [];
  for (let r = 0; r < rows; r += 1) {
    out.push(values.slice(r * cols, (r + 1) * cols));
  }
  return out;
}

let residentTensorId = 0;

function createWebGpuResidentTensor(input: {
  device: MinimalGpuDevice;
  buffer: MinimalGpuBuffer;
  rows: number;
  cols: number;
  byteLength: number;
  retainedBuffers: MinimalGpuBuffer[];
}): WebGpuResidentTensor {
  residentTensorId += 1;
  return {
    kind: "webgpu_resident_tensor",
    id: `webgpu_resident_${residentTensorId}`,
    rows: input.rows,
    cols: input.cols,
    byteLength: input.byteLength,
    buffer: input.buffer,
    device: input.device,
    ...(input.retainedBuffers.length > 0 ? { retainedBuffers: input.retainedBuffers } : {}),
  };
}

async function readGpuBufferAsMatrix(
  device: MinimalGpuDevice,
  sourceBuffer: MinimalGpuBuffer,
  rows: number,
  cols: number,
  byteLength: number,
): Promise<Matrix> {
  const [matrix] = await readGpuBuffersAsMatrices(device, [{ sourceBuffer, rows, cols, byteLength }]);
  if (!matrix) throw new Error("WebGPU buffer readback returned no matrix.");
  return matrix;
}

async function readGpuBuffersAsMatrices(
  device: MinimalGpuDevice,
  reads: Array<{
    sourceBuffer: MinimalGpuBuffer;
    rows: number;
    cols: number;
    byteLength: number;
  }>,
): Promise<Matrix[]> {
  if (reads.length === 0) return [];
  const readBuffers = reads.map((read) => (
    device.createBuffer({ size: read.byteLength, usage: GPU_MAP_READ | GPU_COPY_DST })
  ));
  try {
    const encoder = device.createCommandEncoder();
    reads.forEach((read, index) => {
      const readBuffer = readBuffers[index];
      if (!readBuffer) throw new Error("WebGPU batched readback buffer allocation failed.");
      encoder.copyBufferToBuffer(read.sourceBuffer, 0, readBuffer, 0, read.byteLength);
    });
    device.queue.submit([encoder.finish()]);
    await Promise.all(readBuffers.map((readBuffer) => readBuffer.mapAsync(GPU_MAP_READ)));
    return reads.map((read, index) => {
      const readBuffer = readBuffers[index];
      if (!readBuffer) throw new Error("WebGPU batched readback buffer was not allocated.");
      const data = new Float32Array(readBuffer.getMappedRange()).slice();
      readBuffer.unmap();
      return unflattenMatrix([...data], read.rows, read.cols);
    });
  } finally {
    for (const readBuffer of readBuffers) {
      destroyBuffer(readBuffer);
    }
  }
}

function isWebGpuResidentTensor(value: unknown): value is WebGpuResidentTensor {
  return typeof value === "object"
    && value !== null
    && (value as WebGpuResidentTensor).kind === "webgpu_resident_tensor"
    && typeof (value as WebGpuResidentTensor).id === "string";
}

function readResidentTensorBuffer(tensor: WebGpuResidentTensor): MinimalGpuBuffer {
  if (!isGpuBuffer(tensor.buffer)) throw new Error("WebGPU-resident tensor has an invalid GPU buffer.");
  return tensor.buffer;
}

function residentRetainedBuffers(tensor: WebGpuResidentTensor): MinimalGpuBuffer[] {
  return (tensor.retainedBuffers ?? []).filter(isGpuBuffer);
}

const resolvedGpuDevices = new WeakMap<object, Promise<MinimalGpuDevice | null>>();

async function resolveGpuDevice(device: unknown, gpu: unknown): Promise<MinimalGpuDevice | null> {
  if (isGpuDevice(device)) return device;
  const runtimeGpu = isGpu(gpu) ? gpu : getGlobalGpu();
  if (!runtimeGpu) return null;
  const runtimeGpuKey = runtimeGpu as object;
  const existing = resolvedGpuDevices.get(runtimeGpuKey);
  if (existing) return existing;
  const promise = runtimeGpu.requestAdapter()
    .then(async (adapter) => (await adapter?.requestDevice?.()) ?? null)
    .catch(() => null);
  resolvedGpuDevices.set(runtimeGpuKey, promise);
  const resolved = await promise;
  if (!resolved) resolvedGpuDevices.delete(runtimeGpuKey);
  return resolved;
}

function getGlobalGpu(): MinimalGpu | undefined {
  const root = globalThis as typeof globalThis & { navigator?: { gpu?: MinimalGpu } };
  return root.navigator?.gpu;
}

function createUploadedBuffer(device: MinimalGpuDevice, data: Float32Array | Int32Array | Uint32Array, usage: number): MinimalGpuBuffer {
  const buffer = device.createBuffer({ size: alignTo4(data.byteLength), usage });
  device.queue.writeBuffer(buffer, 0, data.buffer, data.byteOffset, data.byteLength);
  return buffer;
}

function submitOrRecordComputePass(
  device: MinimalGpuDevice,
  batch: WebGpuDecodeCommandBatchLike | undefined,
  input: {
    label: string;
    dispatches?: number;
    record: (encoder: MinimalGpuCommandEncoder) => void;
  },
): void {
  if (batch) {
    batch.recordComputePass(input);
    return;
  }
  const encoder = device.createCommandEncoder();
  input.record(encoder);
  device.queue.submit([encoder.finish()]);
}

function submitOrRecordCopy(
  device: MinimalGpuDevice,
  batch: WebGpuDecodeCommandBatchLike | undefined,
  input: {
    label: string;
    record: (encoder: MinimalGpuCommandEncoder) => void;
  },
): void {
  if (batch) {
    if (!batch.recordCopy) {
      batch.recordComputePass({
        label: input.label,
        dispatches: 0,
        record: input.record,
      });
      return;
    }
    batch.recordCopy(input);
    return;
  }
  const encoder = device.createCommandEncoder();
  input.record(encoder);
  device.queue.submit([encoder.finish()]);
}

function alignTo4(value: number): number {
  return Math.max(4, Math.ceil(value / 4) * 4);
}

function destroyBuffer(buffer: MinimalGpuBuffer): void {
  buffer.destroy?.();
}

function isGpu(value: unknown): value is MinimalGpu {
  return typeof value === "object" && value !== null && typeof (value as MinimalGpu).requestAdapter === "function";
}

function isGpuDevice(value: unknown): value is MinimalGpuDevice {
  return typeof value === "object"
    && value !== null
    && typeof (value as MinimalGpuDevice).createBuffer === "function"
    && typeof (value as MinimalGpuDevice).createShaderModule === "function";
}

function isGpuBuffer(value: unknown): value is MinimalGpuBuffer {
  return typeof value === "object"
    && value !== null
    && typeof (value as MinimalGpuBuffer).mapAsync === "function"
    && typeof (value as MinimalGpuBuffer).getMappedRange === "function";
}

function nowMs(): number {
  return typeof performance !== "undefined" ? performance.now() : Date.now();
}

const GPU_MAP_READ = 1;
const GPU_COPY_SRC = 4;
const GPU_COPY_DST = 8;
const GPU_UNIFORM = 64;
const GPU_STORAGE = 128;
const QWEN_ONE_TOKEN_ATTENTION_MAX_SEQUENCE = 1024;

const tokenEmbeddingLookupWgsl = `
struct Params {
  tokenId: u32,
  hidden: u32,
  vocabSize: u32,
  _pad0: u32,
};

@group(0) @binding(0) var<storage, read> embedding: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(64)
fn token_embedding_lookup(@builtin(global_invocation_id) globalId: vec3<u32>) {
  let col = globalId.x;
  if (col >= params.hidden || params.tokenId >= params.vocabSize) {
    return;
  }
  output[col] = embedding[(params.tokenId * params.hidden) + col];
}
`;

const denseMatVecWgsl = `
struct Params {
  outputRows: u32,
  cols: u32,
  _pad0: u32,
  _pad1: u32,
};

@group(0) @binding(0) var<storage, read> matrix: array<f32>;
@group(0) @binding(1) var<storage, read> vector: array<f32>;
@group(0) @binding(2) var<storage, read> rowIds: array<i32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;
@group(0) @binding(4) var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) globalId: vec3<u32>) {
  let outputRow = globalId.x;
  if (outputRow >= params.outputRows) {
    return;
  }
  let originalRowId = rowIds[outputRow];
  if (originalRowId < 0) {
    return;
  }
  var sum = 0.0;
  for (var col = 0u; col < params.cols; col = col + 1u) {
    sum = sum + matrix[outputRow * params.cols + col] * vector[col];
  }
  output[outputRow] = sum;
}
`;

const denseMatVecTop1CandidatesWgsl = `
struct Params {
  outputRows: u32,
  cols: u32,
  candidateOffset: u32,
  suppressedCount: u32,
};

@group(0) @binding(0) var<storage, read> matrix: array<f32>;
@group(0) @binding(1) var<storage, read> vector: array<f32>;
@group(0) @binding(2) var<storage, read> rowIds: array<i32>;
@group(0) @binding(3) var<storage, read_write> candidatePairs: array<f32>;
@group(0) @binding(4) var<uniform> params: Params;
@group(0) @binding(5) var<storage, read> suppressedRowIds: array<i32>;

var<workgroup> workValues: array<f32, 64>;
var<workgroup> workRowIds: array<i32, 64>;

fn better_candidate(leftValue: f32, leftRowId: i32, rightValue: f32, rightRowId: i32) -> bool {
  if (rightRowId < 0) {
    return false;
  }
  if (leftRowId < 0) {
    return true;
  }
  if (rightValue > leftValue) {
    return true;
  }
  if (rightValue == leftValue && rightRowId < leftRowId) {
    return true;
  }
  return false;
}

fn is_suppressed_row(rowId: i32) -> bool {
  if (rowId < 0) {
    return true;
  }
  for (var index = 0u; index < params.suppressedCount; index = index + 1u) {
    if (suppressedRowIds[index] == rowId) {
      return true;
    }
  }
  return false;
}

@compute @workgroup_size(64)
fn dense_matvec_top1_candidates(
  @builtin(local_invocation_id) localId: vec3<u32>,
  @builtin(workgroup_id) workgroupId: vec3<u32>,
) {
  let local = localId.x;
  let outputRow = workgroupId.x * 64u + local;
  var score = -3.4028234663852886e38;
  var rowId = -1;
  if (outputRow < params.outputRows && !is_suppressed_row(rowIds[outputRow])) {
    var sum = 0.0;
    for (var col = 0u; col < params.cols; col = col + 1u) {
      sum = sum + matrix[outputRow * params.cols + col] * vector[col];
    }
    score = sum;
    rowId = rowIds[outputRow];
  }
  workValues[local] = score;
  workRowIds[local] = rowId;
  workgroupBarrier();

  var stride = 32u;
  loop {
    if (stride == 0u) {
      break;
    }
    if (local < stride) {
      let other = local + stride;
      if (better_candidate(workValues[local], workRowIds[local], workValues[other], workRowIds[other])) {
        workValues[local] = workValues[other];
        workRowIds[local] = workRowIds[other];
      }
    }
    workgroupBarrier();
    stride = stride / 2u;
  }

  if (local == 0u) {
    let candidateIndex = params.candidateOffset + workgroupId.x;
    candidatePairs[candidateIndex * 2u] = workValues[0];
    candidatePairs[candidateIndex * 2u + 1u] = f32(workRowIds[0]);
  }
}
`;

const denseMatVecTop1ReduceWgsl = `
struct ReduceParams {
  candidateRows: u32,
  _pad0: u32,
  _pad1: u32,
  _pad2: u32,
};

@group(0) @binding(0) var<storage, read> candidatePairs: array<f32>;
@group(0) @binding(1) var<storage, read_write> finalPair: array<f32>;
@group(0) @binding(2) var<uniform> params: ReduceParams;

fn reduce_better_candidate(leftValue: f32, leftRowId: i32, rightValue: f32, rightRowId: i32) -> bool {
  if (rightRowId < 0) {
    return false;
  }
  if (leftRowId < 0) {
    return true;
  }
  if (rightValue > leftValue) {
    return true;
  }
  if (rightValue == leftValue && rightRowId < leftRowId) {
    return true;
  }
  return false;
}

@compute @workgroup_size(1)
fn dense_matvec_top1_reduce(@builtin(global_invocation_id) globalId: vec3<u32>) {
  if (globalId.x > 0u) {
    return;
  }
  var bestValue = -3.4028234663852886e38;
  var bestRowId = -1;
  for (var index = 0u; index < params.candidateRows; index = index + 1u) {
    let value = candidatePairs[index * 2u];
    let rowId = i32(candidatePairs[index * 2u + 1u]);
    if (reduce_better_candidate(bestValue, bestRowId, value, rowId)) {
      bestValue = value;
      bestRowId = rowId;
    }
  }
  finalPair[0] = bestValue;
  finalPair[1] = f32(bestRowId);
}
`;

const denseMatVecCompactTopKScoresWgsl = `
struct Params {
  outputRows: u32,
  cols: u32,
  scoreOffset: u32,
  suppressedCount: u32,
};

@group(0) @binding(0) var<storage, read> matrix: array<f32>;
@group(0) @binding(1) var<storage, read> vector: array<f32>;
@group(0) @binding(2) var<storage, read> rowIds: array<i32>;
@group(0) @binding(3) var<storage, read_write> scorePairs: array<f32>;
@group(0) @binding(4) var<uniform> params: Params;
@group(0) @binding(5) var<storage, read> suppressedRowIds: array<i32>;

fn compact_topk_is_suppressed_row(rowId: i32) -> bool {
  if (rowId < 0) {
    return true;
  }
  for (var index = 0u; index < params.suppressedCount; index = index + 1u) {
    if (suppressedRowIds[index] == rowId) {
      return true;
    }
  }
  return false;
}

@compute @workgroup_size(64)
fn dense_matvec_compact_topk_scores(@builtin(global_invocation_id) globalId: vec3<u32>) {
  let outputRow = globalId.x;
  if (outputRow >= params.outputRows) {
    return;
  }
  let pairIndex = (params.scoreOffset + outputRow) * 2u;
  var score = -3.4028234663852886e38;
  var rowId = -1;
  let originalRowId = rowIds[outputRow];
  if (!compact_topk_is_suppressed_row(originalRowId)) {
    var sum = 0.0;
    for (var col = 0u; col < params.cols; col = col + 1u) {
      sum = sum + matrix[outputRow * params.cols + col] * vector[col];
    }
    score = sum;
    rowId = originalRowId;
  }
  scorePairs[pairIndex] = score;
  scorePairs[pairIndex + 1u] = f32(rowId);
}
`;

const denseMatVecCompactTopKReduceWgsl = `
struct ReduceParams {
  candidateRows: u32,
  topK: u32,
  _pad0: u32,
  _pad1: u32,
};

@group(0) @binding(0) var<storage, read> scorePairs: array<f32>;
@group(0) @binding(1) var<storage, read_write> finalPairs: array<f32>;
@group(0) @binding(2) var<uniform> params: ReduceParams;

fn compact_topk_better_candidate(leftValue: f32, leftRowId: i32, rightValue: f32, rightRowId: i32) -> bool {
  if (rightRowId < 0) {
    return false;
  }
  if (leftRowId < 0) {
    return true;
  }
  if (rightValue > leftValue) {
    return true;
  }
  if (rightValue == leftValue && rightRowId < leftRowId) {
    return true;
  }
  return false;
}

@compute @workgroup_size(1)
fn dense_matvec_compact_topk_reduce(@builtin(global_invocation_id) globalId: vec3<u32>) {
  if (globalId.x > 0u) {
    return;
  }
  var selectedRowIds: array<i32, 256>;
  let boundedTopK = min(params.topK, 256u);
  for (var rank = 0u; rank < boundedTopK; rank = rank + 1u) {
    var bestValue = -3.4028234663852886e38;
    var bestRowId = -1;
    for (var index = 0u; index < params.candidateRows; index = index + 1u) {
      let value = scorePairs[index * 2u];
      let rowId = i32(scorePairs[index * 2u + 1u]);
      var alreadySelected = false;
      for (var selectedIndex = 0u; selectedIndex < rank; selectedIndex = selectedIndex + 1u) {
        if (selectedRowIds[selectedIndex] == rowId) {
          alreadySelected = true;
        }
      }
      if (!alreadySelected && compact_topk_better_candidate(bestValue, bestRowId, value, rowId)) {
        bestValue = value;
        bestRowId = rowId;
      }
    }
    selectedRowIds[rank] = bestRowId;
    finalPairs[rank * 2u] = bestValue;
    finalPairs[rank * 2u + 1u] = f32(bestRowId);
  }
}
`;

const denseMatMulWgsl = `
struct DenseMatMulParams {
  tokens: u32,
  hidden: u32,
  outputSize: u32,
  _pad0: u32,
};

@group(0) @binding(0) var<storage, read> activations: array<f32>;
@group(0) @binding(1) var<storage, read> projection: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> params: DenseMatMulParams;

@compute @workgroup_size(16, 16)
fn dense_matmul(@builtin(global_invocation_id) globalId: vec3<u32>) {
  let outCol = globalId.x;
  let token = globalId.y;
  if (token >= params.tokens || outCol >= params.outputSize) {
    return;
  }
  var sum = 0.0;
  for (var hiddenIndex = 0u; hiddenIndex < params.hidden; hiddenIndex = hiddenIndex + 1u) {
    let activation = activations[token * params.hidden + hiddenIndex];
    let weight = projection[outCol * params.hidden + hiddenIndex];
    sum = sum + activation * weight;
  }
  output[token * params.outputSize + outCol] = sum;
}
`;

const qwenQkvPostProjectionWgsl = `
struct QkvPostProjectionParams {
  tokens: u32,
  headCount: u32,
  headDim: u32,
  hidden: u32,
  eps: f32,
  ropeTheta: f32,
  normEnabled: u32,
  ropeEnabled: u32,
};

@group(0) @binding(0) var<storage, read> projected: array<f32>;
@group(0) @binding(1) var<storage, read> normWeight: array<f32>;
@group(0) @binding(2) var<storage, read> positions: array<u32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;
@group(0) @binding(4) var<uniform> params: QkvPostProjectionParams;

fn normalized_head_value(token: u32, head: u32, dim: u32) -> f32 {
  let headStart = token * params.hidden + head * params.headDim;
  let offset = headStart + dim;
  var value = projected[offset];
  if (params.normEnabled != 0u) {
    var meanSquare = 0.0;
    for (var headDimIndex = 0u; headDimIndex < params.headDim; headDimIndex = headDimIndex + 1u) {
      let sample = projected[headStart + headDimIndex];
      meanSquare = meanSquare + sample * sample;
    }
    meanSquare = meanSquare / f32(params.headDim);
    value = value * inverseSqrt(meanSquare + params.eps) * normWeight[dim];
  }
  return value;
}

@compute @workgroup_size(16, 16)
fn qwen_qkv_post_projection(@builtin(global_invocation_id) globalId: vec3<u32>) {
  let col = globalId.x;
  let token = globalId.y;
  if (token >= params.tokens || col >= params.hidden) {
    return;
  }
  let head = col / params.headDim;
  let dim = col % params.headDim;
  var value = normalized_head_value(token, head, dim);
  if (params.ropeEnabled != 0u) {
    let halfDim = params.headDim / 2u;
    let rotaryDim = dim % halfDim;
    var pairDim = dim - halfDim;
    if (dim < halfDim) {
      pairDim = dim + halfDim;
    }
    let pairValue = normalized_head_value(token, head, pairDim);
    let frequency = pow(params.ropeTheta, -f32(rotaryDim) / f32(halfDim));
    let angle = f32(positions[token]) * frequency;
    let c = cos(angle);
    let s = sin(angle);
    if (dim < halfDim) {
      value = value * c - pairValue * s;
    } else {
      value = pairValue * s + value * c;
    }
  }
  output[token * params.hidden + col] = value;
}
`;

const rmsNormWgsl = `
struct RmsNormParams {
  tokens: u32,
  hidden: u32,
  eps: f32,
  _pad0: u32,
};

@group(0) @binding(0) var<storage, read> hidden: array<f32>;
@group(0) @binding(1) var<storage, read> weight: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> params: RmsNormParams;

@compute @workgroup_size(16, 16)
fn rms_norm(@builtin(global_invocation_id) globalId: vec3<u32>) {
  let col = globalId.x;
  let token = globalId.y;
  if (token >= params.tokens || col >= params.hidden) {
    return;
  }
  var meanSquare = 0.0;
  for (var hiddenIndex = 0u; hiddenIndex < params.hidden; hiddenIndex = hiddenIndex + 1u) {
    let value = hidden[token * params.hidden + hiddenIndex];
    meanSquare = meanSquare + value * value;
  }
  meanSquare = meanSquare / f32(params.hidden);
  let scale = inverseSqrt(meanSquare + params.eps);
  let offset = token * params.hidden + col;
  output[offset] = hidden[offset] * scale * weight[col];
}
`;

const residualAddWgsl = `
struct ResidualAddParams {
  tokens: u32,
  hidden: u32,
  _pad0: u32,
  _pad1: u32,
};

@group(0) @binding(0) var<storage, read> left: array<f32>;
@group(0) @binding(1) var<storage, read> right: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> params: ResidualAddParams;

@compute @workgroup_size(64)
fn residual_add(@builtin(global_invocation_id) globalId: vec3<u32>) {
  let col = globalId.x;
  let token = globalId.y;
  if (token >= params.tokens || col >= params.hidden) {
    return;
  }
  let offset = token * params.hidden + col;
  output[offset] = left[offset] + right[offset];
}
`;

const mlpWgsl = `
struct MlpParams {
  inputSize: u32,
  intermediateSize: u32,
  outputSize: u32,
  activationKind: u32,
};

@group(0) @binding(0) var<storage, read> hidden: array<f32>;
@group(0) @binding(1) var<storage, read> upProjection: array<f32>;
@group(0) @binding(2) var<storage, read> gateProjection: array<f32>;
@group(0) @binding(3) var<storage, read> downProjection: array<f32>;
@group(0) @binding(4) var<storage, read_write> intermediate: array<f32>;
@group(0) @binding(5) var<storage, read_write> output: array<f32>;
@group(0) @binding(6) var<uniform> params: MlpParams;

fn gelu_activation(value: f32) -> f32 {
  return 0.5 * value * (1.0 + tanh(0.7978845608 * (value + 0.044715 * value * value * value)));
}

fn silu_activation(value: f32) -> f32 {
  return value / (1.0 + exp(-value));
}

@compute @workgroup_size(64)
fn mlp_intermediate(@builtin(global_invocation_id) globalId: vec3<u32>) {
  if (arrayLength(&downProjection) == 0u || arrayLength(&output) == 0u) {
    return;
  }
  let row = globalId.x;
  if (row >= params.intermediateSize) {
    return;
  }
  var upSum = 0.0;
  for (var col = 0u; col < params.inputSize; col = col + 1u) {
    upSum = upSum + upProjection[row * params.inputSize + col] * hidden[col];
  }
  if (params.activationKind == 1u) {
    var gateSum = 0.0;
    for (var col = 0u; col < params.inputSize; col = col + 1u) {
      gateSum = gateSum + gateProjection[row * params.inputSize + col] * hidden[col];
    }
    intermediate[row] = silu_activation(gateSum) * upSum;
    return;
  }
  intermediate[row] = gelu_activation(upSum);
}

@compute @workgroup_size(64)
fn mlp_output(@builtin(global_invocation_id) globalId: vec3<u32>) {
  if (arrayLength(&hidden) == 0u || arrayLength(&upProjection) == 0u || arrayLength(&gateProjection) == 0u) {
    return;
  }
  let row = globalId.x;
  if (row >= params.outputSize) {
    return;
  }
  var sum = 0.0;
  for (var col = 0u; col < params.intermediateSize; col = col + 1u) {
    sum = sum + downProjection[row * params.intermediateSize + col] * intermediate[col];
  }
  output[row] = sum;
}
`;

const mlpBatchWgsl = `
struct MlpBatchParams {
  inputSize: u32,
  intermediateSize: u32,
  outputSize: u32,
  tokens: u32,
  activationKind: u32,
  _pad0: u32,
  _pad1: u32,
  _pad2: u32,
};

@group(0) @binding(0) var<storage, read> hidden: array<f32>;
@group(0) @binding(1) var<storage, read> upProjection: array<f32>;
@group(0) @binding(2) var<storage, read> gateProjection: array<f32>;
@group(0) @binding(3) var<storage, read> downProjection: array<f32>;
@group(0) @binding(4) var<storage, read_write> intermediate: array<f32>;
@group(0) @binding(5) var<storage, read_write> output: array<f32>;
@group(0) @binding(6) var<uniform> params: MlpBatchParams;

fn gelu_activation(value: f32) -> f32 {
  return 0.5 * value * (1.0 + tanh(0.7978845608 * (value + 0.044715 * value * value * value)));
}

fn silu_activation(value: f32) -> f32 {
  return value / (1.0 + exp(-value));
}

@compute @workgroup_size(16, 16)
fn mlp_batch_intermediate(@builtin(global_invocation_id) globalId: vec3<u32>) {
  if (arrayLength(&downProjection) == 0u || arrayLength(&output) == 0u) {
    return;
  }
  let row = globalId.x;
  let token = globalId.y;
  if (row >= params.intermediateSize || token >= params.tokens) {
    return;
  }
  var upSum = 0.0;
  for (var col = 0u; col < params.inputSize; col = col + 1u) {
    upSum = upSum + upProjection[row * params.inputSize + col] * hidden[token * params.inputSize + col];
  }
  let intermediateOffset = token * params.intermediateSize + row;
  if (params.activationKind == 1u) {
    var gateSum = 0.0;
    for (var col = 0u; col < params.inputSize; col = col + 1u) {
      gateSum = gateSum + gateProjection[row * params.inputSize + col] * hidden[token * params.inputSize + col];
    }
    intermediate[intermediateOffset] = silu_activation(gateSum) * upSum;
    return;
  }
  intermediate[intermediateOffset] = gelu_activation(upSum);
}

@compute @workgroup_size(16, 16)
fn mlp_batch_output(@builtin(global_invocation_id) globalId: vec3<u32>) {
  if (arrayLength(&hidden) == 0u || arrayLength(&upProjection) == 0u || arrayLength(&gateProjection) == 0u) {
    return;
  }
  let row = globalId.x;
  let token = globalId.y;
  if (row >= params.outputSize || token >= params.tokens) {
    return;
  }
  var sum = 0.0;
  for (var col = 0u; col < params.intermediateSize; col = col + 1u) {
    sum = sum + downProjection[row * params.intermediateSize + col] * intermediate[token * params.intermediateSize + col];
  }
  output[token * params.outputSize + row] = sum;
}
`;

interface MinimalGpu {
  requestAdapter(): Promise<MinimalGpuAdapter | null>;
}

interface MinimalGpuAdapter {
  requestDevice(): Promise<MinimalGpuDevice>;
}

interface MinimalGpuDevice {
  queue: {
    writeBuffer(buffer: MinimalGpuBuffer, bufferOffset: number, data: ArrayBufferLike, dataOffset?: number, size?: number): void;
    submit(commandBuffers: unknown[]): void;
  };
  createBuffer(descriptor: { size: number; usage: number }): MinimalGpuBuffer;
  createShaderModule(descriptor: { code: string }): unknown;
  createComputePipeline(descriptor: { layout: "auto"; compute: { module: unknown; entryPoint: string } }): MinimalGpuComputePipeline;
  createBindGroup(descriptor: { layout: unknown; entries: Array<{ binding: number; resource: { buffer: MinimalGpuBuffer } }> }): unknown;
  createCommandEncoder(): MinimalGpuCommandEncoder;
}

interface MinimalGpuBuffer {
  mapAsync(mode: number): Promise<void>;
  getMappedRange(): ArrayBuffer;
  unmap(): void;
  destroy?(): void;
}

interface MinimalGpuComputePipeline {
  getBindGroupLayout(index: number): unknown;
}

interface MinimalGpuCommandEncoder {
  beginComputePass(): MinimalGpuComputePass;
  copyBufferToBuffer(source: MinimalGpuBuffer, sourceOffset: number, destination: MinimalGpuBuffer, destinationOffset: number, size: number): void;
  finish(): unknown;
}

interface MinimalGpuComputePass {
  setPipeline(pipeline: MinimalGpuComputePipeline): void;
  setBindGroup(index: number, bindGroup: unknown): void;
  dispatchWorkgroups(x: number, y?: number, z?: number): void;
  end(): void;
}
