import type { BackendTensorHandle, KVBlock, KVTier } from "./kvswap";
import { KVTensorPagingRegistry, type KVTensorPagingEvent } from "./kvTensorPaging";
import type { SSALayerRoutingPolicy } from "./ssa";
import type { Matrix, Vector } from "./ssa_webgpu";
import {
  appendWebGpuResidentRowCache,
  createWebGpuResidentRowCache,
  createSsaToyTensorHandle,
  destroyWebGpuResidentTensor,
  destroyWebGpuResidentRowCache,
  preloadStableWebGpuRuntimeBuffers,
  readSsaToyTensorHandle,
  readWebGpuResidentTensor,
  readWebGpuResidentTensors,
  runDenseMatVecTopKResidentWebGpu,
  runDenseMatMulResidentWebGpu,
  runDenseMatMulWebGpu,
  runDenseMatVecTopKWebGpu,
  runDenseMatVecWebGpu,
  runMlpBatchResidentWebGpu,
  runMlpBatchWebGpu,
  runMlpWebGpu,
  runPackedQkvProjectionResidentWebGpu,
  runPackedSparseAttentionResidentWebGpu,
  runPackedSparseAttentionWebGpu,
  planPrefillChunks,
  projectCompactTopKDecodeTokensWebGpu,
  projectGreedyDecodeTokenWebGpu,
  runQwenQkvNormRopePairResidentWebGpu,
  runQwenQkvPostProjectionResidentWebGpu,
  runResidualAddResidentWebGpu,
  runResidualRmsNormPairResidentWebGpu,
  runRmsNormResidentWebGpu,
  runTokenEmbeddingLookupResidentWebGpu,
  uploadWebGpuResidentTensor,
  WebGpuRuntimeBufferCache,
  WebGpuSsaReferenceBackend,
  type WebGpuDenseMatMulTrace,
  type WebGpuDenseMatMulResidentTrace,
  type WebGpuDenseMatVecTrace,
  type WebGpuMlpTrace,
  type WebGpuPackedQkvProjectionResidentTrace,
  type WebGpuQkvNormRopePairResidentTrace,
  type WebGpuResidualRmsNormPairResidentTrace,
  type PrefillChunkPlan,
  type WebGpuResidentRowCache,
  type WebGpuResidentTensor,
  type WebGpuResidentTensorTrace,
  type WebGpuRuntimePipelinePreloadKind,
  type WebGpuStableMatrixPreloadDescriptor,
  type WebGpuStableRuntimePreloadEntry,
  type WebGpuStableStaticPreloadDescriptor,
  type WebGpuSparseAttentionResult,
  type SsaToyTensorHandle,
  type WebGpuSsaBackendOptions,
} from "./ssa_webgpu";
import type {
  NativeSSABackendContract,
  SparseLayerForwardInput,
  SparseLayerForwardOutput,
  SSADecodeInput,
  SSADecodeOutput,
  SSAKernelTrace,
  SSAPrefillHandle,
  SSAPrefillPolicy,
} from "./ssa_webgpu/types";
import { executeTSPSchedule, type TSPExecutionTraceRecord } from "./tspExecutor";
import type { TSPScheduleStep } from "./tsp";
import { summarizeDecodeHotPath, type DecodePerfSummary } from "./perf";
import { WebGpuDecodeCommandBatch } from "./ssa_webgpu/fusedDecodeLayer/commandBatcher";
import type { WebGpuDecodeCommandBatchTrace } from "./ssa_webgpu/fusedDecodeLayer/types";
import { sampleFromCompactTopK, type CompactTopKSamplingResult } from "./quality/topKSampler";

export interface UnlockedBrowserTransformerLayerWeights {
  inputLayerNorm?: RuntimeVector;
  qProj: RuntimeMatrix;
  kProj: RuntimeMatrix;
  vProj: RuntimeMatrix;
  oProj: RuntimeMatrix;
  qNorm?: RuntimeVector;
  kNorm?: RuntimeVector;
  postAttentionLayerNorm?: RuntimeVector;
  mlpGateProj?: RuntimeMatrix;
  mlpUpProj?: RuntimeMatrix;
  mlpDownProj?: RuntimeMatrix;
}

export interface UnlockedBrowserTransformerWeights {
  modelId: string;
  architecture: "qwen3_decoder_control" | "llama_decoder_control" | "smollm_decoder_control";
  vocabSize: number;
  hiddenSize: number;
  headDim: number;
  numAttentionHeads?: number;
  numKeyValueHeads?: number;
  maxPositionEmbeddings?: number;
  ropeTheta?: number;
  tieWordEmbeddings?: boolean;
  rmsNormEps?: number;
  tokenEmbedding: RuntimeMatrix;
  outputProjection: RuntimeMatrix;
  finalNorm?: RuntimeVector;
  layers: UnlockedBrowserTransformerLayerWeights[];
}

export type RuntimeVector = Vector | Float32Array;
export type RuntimeMatrix = Matrix | F32Matrix | F16Matrix;

export class F32Matrix {
  readonly rowCount: number;
  readonly colCount: number;

  constructor(
    private readonly buffer: ArrayBuffer,
    private readonly byteOffset: number,
    rowCount: number,
    colCount: number,
  ) {
    if (!Number.isInteger(rowCount) || rowCount <= 0) throw new Error("F32Matrix rowCount must be positive.");
    if (!Number.isInteger(colCount) || colCount <= 0) throw new Error("F32Matrix colCount must be positive.");
    if (byteOffset % Float32Array.BYTES_PER_ELEMENT !== 0) throw new Error("F32Matrix byteOffset must align to f32 values.");
    const requiredBytes = rowCount * colCount * Float32Array.BYTES_PER_ELEMENT;
    if (byteOffset < 0 || byteOffset + requiredBytes > buffer.byteLength) {
      throw new Error("F32Matrix buffer does not contain enough f32 values for the requested shape.");
    }
    this.rowCount = rowCount;
    this.colCount = colCount;
  }

  row(index: number): Float32Array | undefined {
    if (!Number.isInteger(index) || index < 0 || index >= this.rowCount) return undefined;
    return new Float32Array(
      this.buffer,
      this.byteOffset + index * this.colCount * Float32Array.BYTES_PER_ELEMENT,
      this.colCount,
    );
  }

  toFloat32Array(rowIds?: number[]): Float32Array {
    if (!rowIds || isFullContiguousRowSelection(rowIds, this.rowCount)) {
      return new Float32Array(this.buffer, this.byteOffset, this.rowCount * this.colCount);
    }
    const values = new Float32Array(rowIds.length * this.colCount);
    let outputOffset = 0;
    for (const rowId of rowIds) {
      const row = this.row(rowId);
      if (!row) throw new Error(`F32Matrix is missing row ${rowId}.`);
      values.set(row, outputOffset);
      outputOffset += this.colCount;
    }
    return values;
  }
}

export class F16Matrix {
  readonly rowCount: number;
  readonly colCount: number;
  private decodedValues?: Float32Array;

  constructor(
    private readonly buffer: ArrayBuffer,
    private readonly byteOffset: number,
    rowCount: number,
    colCount: number,
  ) {
    if (!Number.isInteger(rowCount) || rowCount <= 0) throw new Error("F16Matrix rowCount must be positive.");
    if (!Number.isInteger(colCount) || colCount <= 0) throw new Error("F16Matrix colCount must be positive.");
    if (byteOffset % Uint16Array.BYTES_PER_ELEMENT !== 0) throw new Error("F16Matrix byteOffset must align to f16 values.");
    const requiredBytes = rowCount * colCount * Uint16Array.BYTES_PER_ELEMENT;
    if (byteOffset < 0 || byteOffset + requiredBytes > buffer.byteLength) {
      throw new Error("F16Matrix buffer does not contain enough f16 values for the requested shape.");
    }
    this.rowCount = rowCount;
    this.colCount = colCount;
  }

  row(index: number): Float32Array | undefined {
    if (!Number.isInteger(index) || index < 0 || index >= this.rowCount) return undefined;
    const start = index * this.colCount;
    if (this.decodedValues) return this.decodedValues.subarray(start, start + this.colCount);
    return this.decodeRow(index);
  }

  toFloat32Array(rowIds?: number[]): Float32Array {
    if (!rowIds || isFullContiguousRowSelection(rowIds, this.rowCount)) {
      if (this.decodedValues) return this.decodedValues;
      const source = new Uint16Array(this.buffer, this.byteOffset, this.rowCount * this.colCount);
      const values = new Float32Array(source.length);
      for (let index = 0; index < source.length; index += 1) {
        values[index] = float16BitsToFloat32(source[index] ?? 0);
      }
      this.decodedValues = values;
      return values;
    }
    const values = new Float32Array(rowIds.length * this.colCount);
    let outputOffset = 0;
    for (const rowId of rowIds) {
      const row = this.decodedValues
        ? this.decodedValues.subarray(rowId * this.colCount, (rowId + 1) * this.colCount)
        : this.decodeRow(rowId);
      if (!row) throw new Error(`F16Matrix is missing row ${rowId}.`);
      values.set(row, outputOffset);
      outputOffset += this.colCount;
    }
    return values;
  }

  private decodeRow(index: number): Float32Array | undefined {
    if (!Number.isInteger(index) || index < 0 || index >= this.rowCount) return undefined;
    const source = new Uint16Array(
      this.buffer,
      this.byteOffset + index * this.colCount * Uint16Array.BYTES_PER_ELEMENT,
      this.colCount,
    );
    const values = new Float32Array(this.colCount);
    for (let offset = 0; offset < source.length; offset += 1) {
      values[offset] = float16BitsToFloat32(source[offset] ?? 0);
    }
    return values;
  }
}

const DEFAULT_LOGIT_RESIDENCY_TILE_ROWS = 8192;
const TOKEN_EMBEDDING_LOOKUP_TILE_ROWS = 4096;

function isFullContiguousRowSelection(rowIds: number[], rowCount: number): boolean {
  if (rowIds.length !== rowCount) return false;
  for (let index = 0; index < rowIds.length; index += 1) {
    if (rowIds[index] !== index) return false;
  }
  return true;
}

function float16BitsToFloat32(bits: number): number {
  const sign = bits & 0x8000 ? -1 : 1;
  const exponent = (bits >> 10) & 0x1f;
  const fraction = bits & 0x03ff;
  if (exponent === 0) {
    return sign * 2 ** -14 * (fraction / 1024);
  }
  if (exponent === 0x1f) {
    return fraction ? Number.NaN : sign * Infinity;
  }
  return sign * 2 ** (exponent - 15) * (1 + fraction / 1024);
}

function isPackedRuntimeMatrix(matrix: RuntimeMatrix): matrix is F32Matrix | F16Matrix {
  return matrix instanceof F32Matrix || matrix instanceof F16Matrix;
}

function hasPackedRuntimeProjection(...matrices: Array<RuntimeMatrix | undefined>): boolean {
  return matrices.some((matrix) => matrix !== undefined && isPackedRuntimeMatrix(matrix));
}

export interface UnlockedBrowserTransformerOptions extends Pick<WebGpuSsaBackendOptions, "backendPreference" | "device" | "gpu" | "requireWebGpu"> {
  weights: UnlockedBrowserTransformerWeights;
  initialNonPinnedTier?: Extract<KVTier, "vram" | "ram" | "disk">;
  bufferCache?: WebGpuRuntimeBufferCache;
}

export interface UnlockedBrowserModelResidencyWarmupInput {
  layerCount?: number;
  logitTileRows?: number | null;
  logitTopK?: number;
}

export interface UnlockedBrowserModelResidencyWarmupResult {
  mode: "direct_projection_preload";
  backend: "webgpu";
  layerCount: number;
  computeMs: number;
  entries: WebGpuStableRuntimePreloadEntry[];
  uploadedEntries: number;
  cacheHits: number;
  logitTileRows: number;
  logitTiles: number;
}

export interface UnlockedBrowserLayerTensorHandles {
  qHandle: SsaToyTensorHandle;
  kHandle: SsaToyTensorHandle;
  vHandle: SsaToyTensorHandle;
}

export interface UnlockedBrowserLayerState {
  hidden: Matrix;
  q: Matrix;
  k: Matrix;
  v: Matrix;
  compactK?: Matrix;
  compactV?: Matrix;
  residentCompactK?: WebGpuResidentRowCache;
  residentCompactV?: WebGpuResidentRowCache;
  projectedTokenCount: number;
}

export interface UnlockedBrowserKvCacheHandle {
  kind: "unlocked_browser_transformer_kv_cache";
  id: string;
  modelId: string;
  requestId: string;
  tokenIds: number[];
  blockTokenRanges: Record<string, { tokenStart: number; tokenEnd: number }>;
  kvBlocks: KVBlock[];
  layers: Record<number, UnlockedBrowserLayerTensorHandles>;
  layerStates: Record<number, UnlockedBrowserLayerState>;
  prefillProof?: UnlockedBrowserPrefillBackendProof;
}

export interface UnlockedBrowserDecodeHandle {
  kind: "unlocked_browser_transformer_logits";
  id: string;
  requestId: string;
  logits: number[];
  logitTokenIds?: number[];
  backendProof?: UnlockedBrowserDecodeBackendProof;
  tspTrace: TSPExecutionTraceRecord[];
  kvPagingEvents: KVTensorPagingEvent[];
}

export interface UnlockedBrowserDecodeBackendProof {
  logitProjection?: {
    backend: WebGpuDenseMatVecTrace["backend"];
    selectedRowIds?: number[];
    fullRowCount: number;
    selectedRowCount: number;
    purpose: "candidate_logit_projection" | "full_vocab_logit_projection" | "full_vocab_topk_logit_projection" | "greedy_argmax_logit_projection" | "compact_topk_logit_projection";
    trace: WebGpuDenseMatVecTrace & {
      vectorResident?: true;
      topK?: number;
      tileRows?: number;
      tiles?: number;
      scannedRows?: number;
      materializedRows?: number;
      readbackStrategy?: "full_logits" | "gpu_top1_candidates" | "gpu_argmax_token_id" | "gpu_compact_topk";
      gpuReducedRows?: number;
      readbackRows?: number;
      readbackBytes?: number;
      dispatchCount?: number;
      compactTopK?: number;
    };
  };
  sampling?: UnlockedBrowserDecodeSamplingProof;
  projectionLayers?: UnlockedBrowserDecodeProjectionLayerProof[];
  oProjectionLayers?: UnlockedBrowserDecodeOProjectionProof[];
  mlpLayers?: UnlockedBrowserDecodeMlpProof[];
  residualRmsNormLayers?: UnlockedBrowserDecodeResidualRmsNormProof[];
  residencyLayers?: UnlockedBrowserDecodeResidencyProof[];
  decodePerf?: DecodePerfSummary;
}

export interface UnlockedBrowserDecodeSamplingProof {
  strategy: CompactTopKSamplingResult["strategy"];
  selectedTokenId: number;
  selectedScore: number;
  selectedRank: number;
  effectiveCandidateCount: number;
  compactLogitTopK: number;
  temperature: number;
  topP: number;
  repetitionPenalty: number;
  greedyDecodeUsed: boolean;
  suppressedTokenCount: number;
  recentTokenCount: number;
}

export interface UnlockedBrowserDecodeResidencyProof {
  layerIndex: number;
  residentLayerPath: boolean;
  activationUploadBytes: number;
  activationUploadCount: number;
  hiddenReadbackCount: number;
  finalHiddenUsedForLogits: boolean;
}

type DecodeLogitProjectionResult = {
  logits: number[];
  logitTokenIds?: number[];
  selectedTokenId?: number;
  backendProof?: UnlockedBrowserDecodeBackendProof;
};

type FinalResidentDecodeLogitsInput = {
  outputProjection: RuntimeMatrix;
  finalNorm?: RuntimeVector;
  topK: number;
  tileRows: number | null;
  suppressedTokenIds: number[];
  sampling?: DecodeSamplingInput;
};

interface DecodeSamplingInput {
  temperature?: number;
  topP?: number;
  repetitionPenalty?: number;
  recentTokenIds?: number[];
  seed?: number;
}

export interface UnlockedBrowserSpeculativeVerificationInput {
  requestId: string;
  previousTokenId: number;
  draftTokenIds: number[];
  kvCacheHandle: unknown;
  policy: SSALayerRoutingPolicy[];
  logitCandidateTokenIds?: number[];
  logitTopK?: number;
  logitTileRows?: number;
  suppressedTokenIds?: number[];
}

export interface UnlockedBrowserSpeculativeVerificationOutput {
  requestId: string;
  targetTokenIds: number[];
  committedInputTokenIds: number[];
  acceptedTokens: number;
  rejectedTokens: number;
  correctedTokenId?: number;
  verifyLatencyMs: number;
  verifiedTokenCount: number;
  targetDecodeCalls: number;
  decodeOutput: SSADecodeOutput;
}

export interface UnlockedBrowserPrefillBackendProof {
  layers: UnlockedBrowserPrefillLayerProof[];
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

export interface UnlockedBrowserPrefillProjectionProof {
  backend: WebGpuDenseMatMulTrace["backend"];
  trace: WebGpuDenseMatMulTrace;
}

export interface UnlockedBrowserPrefillResidentTensorProof {
  backend: WebGpuResidentTensorTrace["backend"];
  trace: WebGpuResidentTensorTrace;
}

export interface UnlockedBrowserPrefillLayerProof {
  layerIndex: number;
  qProjection: UnlockedBrowserPrefillProjectionProof;
  kProjection: UnlockedBrowserPrefillProjectionProof;
  vProjection: UnlockedBrowserPrefillProjectionProof;
  qPostProjection?: UnlockedBrowserPrefillResidentTensorProof;
  kPostProjection?: UnlockedBrowserPrefillResidentTensorProof;
  oProjection: UnlockedBrowserPrefillProjectionProof;
  mlp?: UnlockedBrowserPrefillMlpProof;
  attentionBackend: WebGpuSparseAttentionResult["backend"] | "mixed";
  packedHeadBackends: WebGpuSparseAttentionResult["backend"][];
  packedHeadCount: number;
  keyValueHeadCount?: number;
  keyValueCompressionRatio?: number;
  selectedKeyRows: number;
  prefillChunkDispatch?: "single_dispatch" | "chunked_dispatch";
  attentionDispatchCount?: number;
  awaitedDispatchBreaks?: number;
}

export interface UnlockedBrowserPrefillMlpProof {
  backend: WebGpuMlpTrace["backend"] | "mixed";
  rowCount: number;
  lastTrace: WebGpuMlpTrace;
}

export interface UnlockedBrowserDecodeMlpProof {
  layerIndex: number;
  backend: WebGpuMlpTrace["backend"];
  trace: WebGpuMlpTrace;
}

export interface UnlockedBrowserDecodeResidualRmsNormProof {
  layerIndex: number;
  backend: WebGpuResidualRmsNormPairResidentTrace["backend"];
  trace: WebGpuResidualRmsNormPairResidentTrace;
}

export interface UnlockedBrowserDecodeOProjectionProof {
  layerIndex: number;
  backend: WebGpuDenseMatVecTrace["backend"] | WebGpuDenseMatMulTrace["backend"];
  trace: WebGpuDenseMatVecTrace | WebGpuDenseMatMulTrace;
}

export interface UnlockedBrowserDecodeProjectionProof {
  backend: WebGpuDenseMatVecTrace["backend"] | WebGpuDenseMatMulTrace["backend"];
  trace: WebGpuDenseMatVecTrace | WebGpuDenseMatMulTrace;
}

export interface UnlockedBrowserDecodeProjectionLayerProof {
  layerIndex: number;
  qProjection: UnlockedBrowserDecodeProjectionProof;
  kProjection: UnlockedBrowserDecodeProjectionProof;
  vProjection: UnlockedBrowserDecodeProjectionProof;
  oProjection: UnlockedBrowserDecodeProjectionProof;
  qPostProjection?: UnlockedBrowserPrefillResidentTensorProof;
  kPostProjection?: UnlockedBrowserPrefillResidentTensorProof;
  qkvNormRopePair?: {
    backend: WebGpuQkvNormRopePairResidentTrace["backend"];
    trace: WebGpuQkvNormRopePairResidentTrace;
  };
  qkvReadbackCount?: number;
}

export interface UnlockedBrowserModelPlan {
  primary: {
    modelId: string;
    family: "qwen3";
    reason: string;
    browserRuntime: "custom-webgpu-transformer";
  };
  alternates: Array<{ modelId: string; family: string; reason: string }>;
  rejectedFamilies: Array<{ family: string; reason: string }>;
}

export class UnlockedBrowserTransformerBackend implements NativeSSABackendContract {
  readonly backendName = "unlocked-browser-transformer";
  readonly supportsQkvAccess = true;
  readonly supportsLayerSparseRouting = true;
  readonly supportsPinnedKvBlocks = true;
  readonly supportsDenseReferenceMode = true;

  private readonly kernelBackend: WebGpuSsaReferenceBackend;
  private readonly sparseAttentionOptions: Pick<WebGpuSsaBackendOptions, "backendPreference" | "device" | "gpu" | "requireWebGpu">;
  private readonly weights: UnlockedBrowserTransformerWeights;
  private readonly initialNonPinnedTier: Extract<KVTier, "vram" | "ram" | "disk">;
  private readonly bufferCache: WebGpuRuntimeBufferCache;
  private readonly ownsBufferCache: boolean;
  private modelId: string | null = null;
  private disposed = false;

  constructor(options: UnlockedBrowserTransformerOptions) {
    validateWeights(options.weights);
    this.weights = cloneWeights(options.weights);
    this.initialNonPinnedTier = options.initialNonPinnedTier ?? "ram";
    this.bufferCache = options.bufferCache ?? new WebGpuRuntimeBufferCache();
    this.ownsBufferCache = options.bufferCache === undefined;
    this.sparseAttentionOptions = {
      ...(options.backendPreference ? { backendPreference: options.backendPreference } : {}),
      ...(options.device ? { device: options.device } : {}),
      ...(options.gpu ? { gpu: options.gpu } : {}),
      ...(options.requireWebGpu ? { requireWebGpu: true } : {}),
    };
    this.kernelBackend = new WebGpuSsaReferenceBackend({
      ...this.sparseAttentionOptions,
      config: { headDim: options.weights.headDim },
    });
  }

  async initializeModel(modelId: string): Promise<void> {
    this.assertNotDisposed();
    if (modelId !== this.weights.modelId) {
      throw new Error(`UnlockedBrowserTransformerBackend expected modelId ${this.weights.modelId}, received ${modelId}.`);
    }
    this.modelId = modelId;
  }

  async warmModelResidency(
    input: UnlockedBrowserModelResidencyWarmupInput = {},
  ): Promise<UnlockedBrowserModelResidencyWarmupResult> {
    this.assertReady();
    if (this.sparseAttentionOptions.backendPreference === "cpu") {
      throw new Error("UnlockedBrowserTransformerBackend direct model residency warmup requires WebGPU.");
    }
    const layerCount = normalizeWarmupLayerCount(input.layerCount, this.weights.layers.length);
    const matrices: WebGpuStableMatrixPreloadDescriptor[] = [];
    const staticBuffers: WebGpuStableStaticPreloadDescriptor[] = [];
    const pipelines = new Set<WebGpuRuntimePipelinePreloadKind>([
      "sparse-attention",
      "packed-sparse-attention",
      "token-embedding-lookup-resident",
      "dense-matmul",
      "packed-qkv-projection-resident",
      "qwen-qkv-norm-rope-pair-resident",
      "qwen-qkv-post-projection-resident",
      "rmsnorm-resident",
      "residual-add-resident",
      "mlp-batch-intermediate",
      "mlp-batch-output",
    ]);
    addTokenEmbeddingLookupPreload(matrices, this.weights.tokenEmbedding);

    for (let layerIndex = 0; layerIndex < layerCount; layerIndex += 1) {
      const layer = this.requireLayer(layerIndex);
      addPackedProjectionPreload(matrices, projectionCacheKey(layerIndex, "qProj"), layer.qProj, "dense-matmul");
      addPackedProjectionPreload(matrices, projectionCacheKey(layerIndex, "kProj"), layer.kProj, "dense-matmul");
      addPackedProjectionPreload(matrices, projectionCacheKey(layerIndex, "vProj"), layer.vProj, "dense-matmul");
      addPackedProjectionPreload(matrices, projectionCacheKey(layerIndex, "oProj"), layer.oProj, "dense-matmul");
      if (layer.mlpUpProj && layer.mlpDownProj) {
        const mlpKey = projectionCacheKey(layerIndex, "mlp");
        addPackedProjectionPreload(matrices, `${mlpKey}:upProjection`, layer.mlpUpProj, "mlp");
        addPackedProjectionPreload(matrices, `${mlpKey}:downProjection`, layer.mlpDownProj, "mlp");
        if (layer.mlpGateProj) addPackedProjectionPreload(matrices, `${mlpKey}:gateProjection`, layer.mlpGateProj, "mlp");
      }
    }

    const logitTopK = normalizeLogitTopK(input.logitTopK, this.weights.vocabSize);
    const outputRows = matrixRowCount(this.weights.outputProjection);
    const outputCols = matrixColCount(this.weights.outputProjection);
    const logitTileRows = normalizeResidencyLogitTileRows(input.logitTileRows, outputRows);
    let logitTiles = 0;
    if (isPackedRuntimeMatrix(this.weights.outputProjection)) {
      const logitTileRowIds: number[][] = [];
      for (let rowStart = 0; rowStart < outputRows; rowStart += logitTileRows) {
        const rowEnd = Math.min(outputRows, rowStart + logitTileRows);
        const rowIds = Array.from({ length: rowEnd - rowStart }, (_value, offset) => rowStart + offset);
        logitTileRowIds.push(rowIds);
        matrices.push({
          key: `dense-matvec:outputProjection:rows:${rowStart}-${rowEnd}`,
          matrix: this.weights.outputProjection,
          rowIds,
        });
        logitTiles += 1;
      }
      pipelines.add(logitTopK === 1 ? "dense-matvec-top1-candidates" : "dense-matvec");
      if (logitTopK === 1 && outputRows > logitTileRows) {
        pipelines.add("dense-matvec-top1-reduce");
        let candidateRowsTotal = 0;
        for (const rowIds of logitTileRowIds) {
          const rowStart = rowIds[0] ?? 0;
          const rowEnd = (rowIds.at(-1) ?? rowStart) + 1;
          const candidateRows = Math.max(1, Math.ceil(rowIds.length / 64));
          staticBuffers.push({
            key: `dense-matvec-top1-rowids:outputProjection:rows:${rowStart}-${rowEnd}`,
            data: new Int32Array(rowIds),
          });
          staticBuffers.push({
            key: `dense-matvec-top1-params:outputProjection:rows:${rowStart}-${rowEnd}:cols:${outputCols}:offset:${candidateRowsTotal}:candidateRows:${candidateRows}`,
            data: new Uint32Array([rowIds.length, outputCols, candidateRowsTotal, 0]),
            usageKind: "uniform",
          });
          candidateRowsTotal += candidateRows;
        }
        staticBuffers.push({
          key: `dense-matvec-top1-reduce-params:outputProjection:rows:0-${outputRows}:tileRows:${logitTileRows}:candidateRows:${candidateRowsTotal}`,
          data: new Uint32Array([candidateRowsTotal, 0, 0, 0]),
          usageKind: "uniform",
        });
      }
    }

    const preload = await preloadStableWebGpuRuntimeBuffers({
      bufferCache: this.bufferCache,
      matrices,
      staticBuffers,
      pipelines: [...pipelines],
      ...this.sparseAttentionOptions,
      requireWebGpu: true,
    });
    return {
      mode: "direct_projection_preload",
      backend: preload.backend,
      layerCount,
      computeMs: preload.computeMs,
      entries: preload.entries,
      uploadedEntries: preload.uploadedEntries,
      cacheHits: preload.cacheHits,
      logitTileRows,
      logitTiles,
    };
  }

  async prefill(inputTokenIds: Int32Array, policy: SSAPrefillPolicy): Promise<SSAPrefillHandle> {
    this.assertReady();
    if (inputTokenIds.length === 0) throw new Error("UnlockedBrowserTransformerBackend prefill requires at least one token.");
    if (policy.layerPolicies.length === 0) throw new Error("UnlockedBrowserTransformerBackend prefill requires at least one layer policy.");

    const tokenIds = [...inputTokenIds];
    const selectedBlockCount = maxSelectedBlockCount(policy.layerPolicies);
    const prefillPlan = planPrefillChunks(tokenIds.length, {
      operation: "prefill_chunk",
      headDim: this.weights.headDim,
      ...(selectedBlockCount > 0 ? { selectedBlockCount } : {}),
      blockSize: maxPolicyBlockSize(policy.layerPolicies),
    });
    let hidden = tokenIds.map((tokenId) => tokenEmbedding(this.weights, tokenId));
    const layers: Record<number, UnlockedBrowserLayerTensorHandles> = {};
    const layerStates: Record<number, UnlockedBrowserLayerState> = {};
    const traces: SSAKernelTrace[] = [];
    const kvBlocks: KVBlock[] = [];
    const prefillProofLayers: UnlockedBrowserPrefillLayerProof[] = [];
    let cacheBlockTokenRanges: UnlockedBrowserKvCacheHandle["blockTokenRanges"] | null = null;
    const positions = tokenIds.map((_, index) => index);

    for (let position = 0; position < policy.layerPolicies.length; position += 1) {
      const layerPolicy = policy.layerPolicies[position] as SSALayerRoutingPolicy | undefined;
      if (!layerPolicy) continue;
      const layerIndex = normalizeLayerIndex(layerPolicy.layerIndex, position);
      const layer = this.requireLayer(layerIndex);
      const blockTokenRanges = buildBlockTokenRanges(tokenIds.length, layerPolicy);
      cacheBlockTokenRanges ??= blockTokenRanges;
      const attentionInput = prepareAttentionInput(hidden, layer, this.weights.rmsNormEps);
      const projected = await projectAttentionTensors({
        attentionInput,
        layer,
        weights: this.weights,
        positions,
        options: this.sparseAttentionOptions,
        bufferCache: this.bufferCache,
        requestId: policy.requestId,
        layerIndex,
        phase: "prefill",
      });
      layers[layerIndex] = createLayerHandles({
        requestId: policy.requestId,
        layerIndex,
        q: projected.q,
        k: projected.k,
        v: projected.v,
        blockTokenRanges,
      });
      const prefillAttention = await runPrefillAttention(
        projected.q,
        projected.attentionK,
        projected.attentionV,
        projected.geometry,
        this.sparseAttentionOptions,
        this.bufferCache,
        prefillPlan,
      );
      const attentionOutput = await applyAttentionOutput(
        hidden,
        prefillAttention.output,
        layer,
        this.weights.rmsNormEps,
        {
          options: this.sparseAttentionOptions,
          bufferCache: this.bufferCache,
          requestId: policy.requestId,
          layerIndex,
          phase: "prefill",
        },
      );
      hidden = attentionOutput.hidden;
      const compactK = clonePlainMatrix(projected.attentionK);
      const compactV = clonePlainMatrix(projected.attentionV);
      const layerState: UnlockedBrowserLayerState = {
        hidden: clonePlainMatrix(hidden),
        q: clonePlainMatrix(projected.q),
        k: clonePlainMatrix(projected.k),
        v: clonePlainMatrix(projected.v),
        compactK,
        compactV,
        projectedTokenCount: tokenIds.length,
      };
      if (this.sparseAttentionOptions.requireWebGpu === true || this.sparseAttentionOptions.backendPreference === "webgpu") {
        try {
          const residentCompactK = await createWebGpuResidentRowCache({
            matrix: compactK,
            capacityRows: Math.max(tokenIds.length + 64, tokenIds.length * 2),
            ...(this.sparseAttentionOptions.device ? { device: this.sparseAttentionOptions.device } : {}),
            ...(this.sparseAttentionOptions.gpu ? { gpu: this.sparseAttentionOptions.gpu } : {}),
            ...(this.sparseAttentionOptions.requireWebGpu ? { requireWebGpu: true } : {}),
          });
          const residentCompactV = await createWebGpuResidentRowCache({
            matrix: compactV,
            capacityRows: Math.max(tokenIds.length + 64, tokenIds.length * 2),
            ...(this.sparseAttentionOptions.device ? { device: this.sparseAttentionOptions.device } : {}),
            ...(this.sparseAttentionOptions.gpu ? { gpu: this.sparseAttentionOptions.gpu } : {}),
            ...(this.sparseAttentionOptions.requireWebGpu ? { requireWebGpu: true } : {}),
          });
          layerState.residentCompactK = residentCompactK;
          layerState.residentCompactV = residentCompactV;
        } catch (error) {
          if (this.sparseAttentionOptions.requireWebGpu === true) throw error;
        }
      }
      layerStates[layerIndex] = layerState;
      kvBlocks.push(...buildKvBlocks({
        layerIndex,
        blockTokenRanges,
        policy: layerPolicy,
        initialNonPinnedTier: this.initialNonPinnedTier,
      }));
      traces.push({
        ...buildTrace(policy.requestId, layerIndex, layerPolicy, tokenIds.length),
        attentionMs: prefillAttention.attentionMs,
        prefillChunkCount: prefillPlan.prefillChunkCount,
        prefillChunkSize: prefillPlan.prefillChunkSize,
        shapeBucket: prefillPlan.shapeBucket,
        pipelineCacheKey: prefillPlan.pipelineCacheKey,
        prefillDispatchTargetMs: prefillPlan.dispatchBudgetMs,
        maxDispatchEstimatedMs: prefillPlan.maxDispatchEstimatedMs,
        prefillChunkDispatch: prefillAttention.prefillChunkDispatch,
        attentionDispatchCount: prefillAttention.attentionDispatchCount,
        awaitedDispatchBreaks: prefillAttention.awaitedDispatchBreaks,
      });
      prefillProofLayers.push({
        layerIndex,
        qProjection: toPrefillProjectionProof(projected.proofs.qProjection),
        kProjection: toPrefillProjectionProof(projected.proofs.kProjection),
        vProjection: toPrefillProjectionProof(projected.proofs.vProjection),
        ...(projected.proofs.qPostProjection ? { qPostProjection: toPrefillResidentTensorProof(projected.proofs.qPostProjection) } : {}),
        ...(projected.proofs.kPostProjection ? { kPostProjection: toPrefillResidentTensorProof(projected.proofs.kPostProjection) } : {}),
        oProjection: toPrefillProjectionProof(attentionOutput.oProjection),
        ...(attentionOutput.mlp ? { mlp: attentionOutput.mlp } : {}),
        attentionBackend: prefillAttention.attentionBackend,
        packedHeadBackends: prefillAttention.packedHeadBackends,
        packedHeadCount: prefillAttention.packedHeadCount,
        keyValueHeadCount: prefillAttention.keyValueHeadCount,
        keyValueCompressionRatio: prefillAttention.keyValueCompressionRatio,
        selectedKeyRows: prefillAttention.selectedKeyRows,
        prefillChunkDispatch: prefillAttention.prefillChunkDispatch,
        attentionDispatchCount: prefillAttention.attentionDispatchCount,
        awaitedDispatchBreaks: prefillAttention.awaitedDispatchBreaks,
      });
    }

    const prefillDispatchProof = summarizePrefillDispatchProof(prefillProofLayers);

    return {
      requestId: policy.requestId,
      tokenCount: tokenIds.length,
      prefillChunkCount: prefillPlan.prefillChunkCount,
      prefillChunkSize: prefillPlan.prefillChunkSize,
      shapeBucket: prefillPlan.shapeBucket,
      pipelineCacheKey: prefillPlan.pipelineCacheKey,
      prefillDispatchTargetMs: prefillPlan.dispatchBudgetMs,
      maxDispatchEstimatedMs: prefillPlan.maxDispatchEstimatedMs,
      prefillChunkDispatch: prefillDispatchProof.prefillChunkDispatch,
      attentionDispatchCount: prefillDispatchProof.attentionDispatchCount,
      awaitedDispatchBreaks: prefillDispatchProof.awaitedDispatchBreaks,
      kvCacheHandle: {
        kind: "unlocked_browser_transformer_kv_cache",
        id: `unlocked:${policy.requestId}:kv`,
        modelId: this.modelId as string,
        requestId: policy.requestId,
        tokenIds,
        blockTokenRanges: cacheBlockTokenRanges ?? {},
        kvBlocks,
        layers,
        layerStates,
        prefillProof: {
          layers: prefillProofLayers,
          prefillChunkCount: prefillPlan.prefillChunkCount,
          prefillChunkSize: prefillPlan.prefillChunkSize,
          shapeBucket: prefillPlan.shapeBucket,
          pipelineCacheKey: prefillPlan.pipelineCacheKey,
          prefillDispatchTargetMs: prefillPlan.dispatchBudgetMs,
          maxDispatchEstimatedMs: prefillPlan.maxDispatchEstimatedMs,
          prefillChunkDispatch: prefillDispatchProof.prefillChunkDispatch,
          attentionDispatchCount: prefillDispatchProof.attentionDispatchCount,
          awaitedDispatchBreaks: prefillDispatchProof.awaitedDispatchBreaks,
        },
      } satisfies UnlockedBrowserKvCacheHandle,
      traces,
    };
  }

  async executeSparseLayer(input: SparseLayerForwardInput): Promise<SparseLayerForwardOutput> {
    this.assertReady();
    const layer = this.requireLayer(input.layerIndex);
    const geometry = getAttentionGeometry(this.weights, layer);
    if (geometry.attentionHeads > 1) {
      return await executePackedHeadSparseLayer(input, geometry, this.sparseAttentionOptions, this.bufferCache);
    }
    const sparse = await this.kernelBackend.executeSparseForward({
      requestId: input.requestId,
      layerIndex: input.layerIndex,
      qHandle: input.qHandle,
      kHandle: input.kHandle,
      vHandle: input.vHandle,
      routingPolicy: input.policy,
    });
    return {
      requestId: sparse.requestId,
      layerIndex: sparse.layerIndex,
      outputHandle: sparse.outputHandle,
      trace: sparse.trace ?? buildTrace(input.requestId, input.layerIndex, input.policy, readSsaToyTensorHandle(input.kHandle).matrix.length),
    };
  }

  async decode(input: SSADecodeInput): Promise<SSADecodeOutput> {
    this.assertReady();
    const cache = readUnlockedBrowserKvCacheHandle(input.kvCacheHandle);
    if (cache.requestId !== input.requestId) {
      throw new Error(`UnlockedBrowserTransformerBackend KV cache requestId mismatch: cache ${cache.requestId} cannot decode request ${input.requestId}.`);
    }
    if (cache.modelId !== this.modelId) {
      throw new Error(`UnlockedBrowserTransformerBackend KV cache modelId mismatch: cache ${cache.modelId} cannot decode model ${this.modelId}.`);
    }

    cache.tokenIds.push(input.inputTokenId);
    cache.kvBlocks.length = 0;
    cache.blockTokenRanges = {};
    const registry = new KVTensorPagingRegistry({ now: Date.now(), defaultEvictionTier: "disk" });
    const traces: SSAKernelTrace[] = [];
    const kvPagingEvents: KVTensorPagingEvent[] = [];
    const tspTrace: TSPExecutionTraceRecord[] = [];
    const preferResidentDecode = this.sparseAttentionOptions.requireWebGpu === true
      || this.sparseAttentionOptions.backendPreference === "webgpu";
    const decodeResidentTensors: WebGpuResidentTensor[] = [];
    const rememberDecodeResident = <T extends WebGpuResidentTensor>(tensor: T): T => {
      if (!decodeResidentTensors.includes(tensor)) decodeResidentTensors.push(tensor);
      return tensor;
    };
    const rememberDecodeResidentResult = <T extends { tensor: WebGpuResidentTensor }>(result: T): T => {
      rememberDecodeResident(result.tensor);
      return result;
    };
    let currentHidden = tokenEmbedding(this.weights, input.inputTokenId);
    let currentHiddenResident: WebGpuResidentTensor | undefined;
    let strictResidentDecode = false;
    if (preferResidentDecode) {
      try {
        currentHiddenResident = rememberDecodeResident((await runTokenEmbeddingLookupResidentWebGpu({
          tokenId: positiveModulo(input.inputTokenId, this.weights.vocabSize),
          tokenEmbedding: this.weights.tokenEmbedding,
          ...(this.sparseAttentionOptions.device ? { device: this.sparseAttentionOptions.device } : {}),
          ...(this.sparseAttentionOptions.gpu ? { gpu: this.sparseAttentionOptions.gpu } : {}),
          ...(this.sparseAttentionOptions.requireWebGpu ? { requireWebGpu: true } : {}),
          bufferCache: this.bufferCache,
          projectionCacheKey: "tokenEmbedding",
          projectionCachePolicy: "stable",
          traceMetadata: {
            runtime: "unlocked-browser-transformer",
            requestId: input.requestId,
            phase: "decode",
            purpose: "decode_token_embedding_lookup",
            residentDecodeLayerPath: true,
          },
        })).tensor);
        currentHidden = [];
        strictResidentDecode = true;
      } catch (error) {
        if (this.sparseAttentionOptions.requireWebGpu === true) throw error;
      }
    }
    let logits: number[] = [];
    let logitTokenIds: number[] | undefined;
    let backendProof: UnlockedBrowserDecodeBackendProof | undefined;
    const projectionProofs: UnlockedBrowserDecodeProjectionLayerProof[] = [];
    const oProjectionProofs: UnlockedBrowserDecodeOProjectionProof[] = [];
    const mlpProofs: UnlockedBrowserDecodeMlpProof[] = [];
    const residualRmsNormProofs: UnlockedBrowserDecodeResidualRmsNormProof[] = [];
    const residencyProofs: UnlockedBrowserDecodeResidencyProof[] = [];
    const logitCandidateTokenIds = normalizeLogitCandidateTokenIds(input.logitCandidateTokenIds, this.weights.vocabSize);
    const suppressedTokenIds = normalizeSuppressedTokenIds(input.suppressedTokenIds, this.weights.vocabSize);
    const logitTopK = normalizeLogitTopK(input.logitTopK, this.weights.vocabSize);
    const logitTileRows = normalizeLogitTileRows(input.logitTileRows, this.weights.vocabSize);
    const decodeSampling = normalizeDecodeSamplingInput(input);
    let executedLayer = false;
    const newTokenPosition = cache.tokenIds.length - 1;
    const decodeCommandBatchTraces: WebGpuDecodeCommandBatchTrace[] = [];
    const tokenCommandBatchDevice = strictResidentDecode
      ? resolveDecodeCommandBatchDevice(
          this.sparseAttentionOptions.device,
          currentHiddenResident,
        )
      : undefined;
    const tokenCommandBatch = strictResidentDecode && tokenCommandBatchDevice
      ? new WebGpuDecodeCommandBatch(tokenCommandBatchDevice, {
          requestId: input.requestId,
          tokenIndex: newTokenPosition,
          label: "qwen-decode-token",
        })
      : undefined;

    for (let position = 0; position < input.policy.length; position += 1) {
      const layerPolicy = input.policy[position] as SSALayerRoutingPolicy | undefined;
      if (!layerPolicy) continue;
      const layerIndex = normalizeLayerIndex(layerPolicy.layerIndex, position);
      const layer = this.requireLayer(layerIndex);
      const commandBatch = tokenCommandBatch;
      const layerState = cache.layerStates[layerIndex];
      if (!layerState) {
        throw new Error(`UnlockedBrowserTransformerBackend has no per-layer KV state for layer ${layerIndex}.`);
      }
      const blockTokenRanges = buildBlockTokenRanges(cache.tokenIds.length, layerPolicy);
      if (Object.keys(cache.blockTokenRanges).length === 0) cache.blockTokenRanges = blockTokenRanges;
      const attentionInput = currentHiddenResident
        ? (layer.inputLayerNorm
            ? rememberDecodeResident((await runRmsNormResidentWebGpu({
                hidden: currentHiddenResident,
                weight: layer.inputLayerNorm as RuntimeVector,
                ...(this.weights.rmsNormEps !== undefined ? { eps: this.weights.rmsNormEps } : {}),
                ...(this.sparseAttentionOptions.device ? { device: this.sparseAttentionOptions.device } : {}),
                ...(this.sparseAttentionOptions.gpu ? { gpu: this.sparseAttentionOptions.gpu } : {}),
                requireWebGpu: true,
                bufferCache: this.bufferCache,
                traceMetadata: {
                  runtime: "unlocked-browser-transformer",
                  requestId: input.requestId,
                  layerIndex,
                  phase: "decode",
                  purpose: "decode_input_rmsnorm",
                  residentDecodeLayerPath: true,
                },
                ...(commandBatch ? { commandBatch } : {}),
              })).tensor)
            : currentHiddenResident)
        : prepareAttentionInput([currentHidden], layer, this.weights.rmsNormEps);
      const projected = await projectAttentionTensors({
        attentionInput,
        layer,
        weights: this.weights,
        positions: [newTokenPosition],
        options: this.sparseAttentionOptions,
        bufferCache: this.bufferCache,
        requestId: input.requestId,
        layerIndex,
        phase: "decode",
        ...(commandBatch ? { commandBatch } : {}),
      });
      if (projected.resident) {
        rememberDecodeResident(projected.resident.q);
        rememberDecodeResident(projected.resident.attentionK);
        rememberDecodeResident(projected.resident.attentionV);
      }
      layerState.q.push([...(projected.q[0] ?? [])]);
      layerState.k.push([...(projected.k[0] ?? [])]);
      layerState.v.push([...(projected.v[0] ?? [])]);
      if (projected.resident && strictResidentDecode) {
        if (!layerState.compactK || !layerState.compactV) {
          layerState.compactK = layerState.k;
          layerState.compactV = layerState.v;
        }
        const residentProjected = projected as ProjectedAttentionTensors & {
          resident: NonNullable<ProjectedAttentionTensors["resident"]>;
        };
        await appendResidentDecodeKvCache({
          layerState,
          projected: residentProjected,
          options: this.sparseAttentionOptions,
          ...(commandBatch ? { commandBatch } : {}),
        });
      } else if (layerState.compactK && layerState.compactV) {
        layerState.compactK.push([...(projected.attentionK[0] ?? [])]);
        layerState.compactV.push([...(projected.attentionV[0] ?? [])]);
      } else if ((projected.attentionK[0]?.length ?? 0) === (projected.k[0]?.length ?? 0)) {
        layerState.compactK = layerState.k;
        layerState.compactV = layerState.v;
      }
      layerState.projectedTokenCount = layerState.k.length;
      const handles = createLayerHandles({
        requestId: input.requestId,
        layerIndex,
        q: layerState.q,
        k: layerState.k,
        v: layerState.v,
        blockTokenRanges,
        cloneMatrices: false,
      });
      const attentionHandles = createAttentionLayerHandles({
        requestId: input.requestId,
        layerIndex,
        q: layerState.q,
        k: layerState.compactK,
        v: layerState.compactV,
        fallback: handles,
        blockTokenRanges,
      });
      cache.layers[layerIndex] = handles;
      const kvBlocks = buildKvBlocks({
        layerIndex,
        blockTokenRanges,
        policy: layerPolicy,
        initialNonPinnedTier: this.initialNonPinnedTier,
      });
      cache.kvBlocks.push(...kvBlocks);
      for (const block of kvBlocks) registry.registerBlock(block, block.tensorHandles);
      let sparseOutput: SparseLayerForwardOutput | null = null;
      const selectedKvBlockIds = collectSelectedBlockIds(layerPolicy).map((blockId) => `layer${layerIndex}:${blockId}`);
      const schedule = buildDecodeSchedule(layerIndex, cache.tokenIds.length);

      const layerTspTrace = await executeTSPSchedule(schedule, {
        kv_prefetch: () => {
          const readiness = registry.ensureBlocksAvailableForSparseAttention(selectedKvBlockIds);
          kvPagingEvents.push(...readiness.events);
        },
        attention: async () => {
          sparseOutput = await this.executeSparseLayer({
            requestId: input.requestId,
            layerIndex,
            qHandle: attentionHandles.qHandle,
            kHandle: attentionHandles.kHandle,
            vHandle: attentionHandles.vHandle,
            ...(projected.resident ? { residentQ: projected.resident.q } : {}),
            ...(layerState.residentCompactK ? { residentK: layerState.residentCompactK.tensor } : {}),
            ...(layerState.residentCompactV ? { residentV: layerState.residentCompactV.tensor } : {}),
            ...(commandBatch ? { commandBatch } : {}),
            policy: layerPolicy,
            queryTokenIndexes: [layerState.q.length - 1],
            preferResidentOutput: true,
          });
          traces.push(sparseOutput.trace);
        },
        mlp: async () => {
          if (!sparseOutput) throw new Error("UnlockedBrowserTransformerBackend TSP schedule ran MLP before attention.");
          const residentAttention = readResidentSparseTensorHandle(sparseOutput.outputHandle);
          const attentionMatrix = residentAttention ? null : readSsaToyTensorHandle(sparseOutput.outputHandle).matrix;
          const latestAttention = residentAttention?.tensor ?? attentionMatrix?.[attentionMatrix.length - 1] ?? projected.v[0] ?? currentHidden;
          const isFinalDecodeLayer = position === input.policy.length - 1;
          const canUseResidentFinalLogits = !commandBatch && !logitCandidateTokenIds && isFinalDecodeLayer;
          const keepHiddenResident = Boolean(currentHiddenResident && !logitCandidateTokenIds);
          const nextHidden = await applyAttentionOutputForDecode({
            hidden: currentHiddenResident ?? currentHidden,
            attention: latestAttention,
            layer,
            rmsNormEps: this.weights.rmsNormEps,
            options: this.sparseAttentionOptions,
            bufferCache: this.bufferCache,
            requestId: input.requestId,
            layerIndex,
            ...(keepHiddenResident || canUseResidentFinalLogits ? { materializeHidden: false } : {}),
            ...(canUseResidentFinalLogits
              ? {
                  finalLogits: {
                    outputProjection: this.weights.outputProjection,
                    ...(this.weights.finalNorm ? { finalNorm: this.weights.finalNorm } : {}),
                    topK: logitTopK,
                    tileRows: logitTileRows,
                    suppressedTokenIds,
                    ...(decodeSampling ? { sampling: decodeSampling } : {}),
                  },
                }
              : {}),
            ...(commandBatch ? { commandBatch } : {}),
          });
          if (nextHidden.hidden) {
            currentHidden = nextHidden.hidden;
            currentHiddenResident = undefined;
          } else if (nextHidden.hiddenResident) {
            currentHiddenResident = rememberDecodeResident(nextHidden.hiddenResident);
          } else if (!isFinalDecodeLayer || !nextHidden.finalLogits) {
            throw new Error("UnlockedBrowserTransformerBackend decode did not materialize hidden state before it was needed.");
          }
          if (nextHidden.oProjectionProof) {
            oProjectionProofs.push(nextHidden.oProjectionProof);
            projectionProofs.push(createDecodeProjectionLayerProof(
              layerIndex,
              projected.proofs,
              nextHidden.oProjectionProof,
            ));
          }
          if (nextHidden.mlpProof) mlpProofs.push(nextHidden.mlpProof);
          if (nextHidden.residualRmsNormProof) residualRmsNormProofs.push(nextHidden.residualRmsNormProof);
          if (nextHidden.residencyProof) residencyProofs.push(nextHidden.residencyProof);
          if (nextHidden.hidden) layerState.hidden.push([...currentHidden]);
          if (isFinalDecodeLayer && !commandBatch) {
            const projectedLogits = nextHidden.finalLogits
              ?? (currentHiddenResident && !logitCandidateTokenIds
                ? await projectDecodeLogitsFromResidentHidden({
                    hidden: currentHiddenResident,
                    outputProjection: this.weights.outputProjection,
                    ...(this.weights.finalNorm ? { finalNorm: this.weights.finalNorm } : {}),
                    rmsNormEps: this.weights.rmsNormEps,
                    topK: logitTopK,
                    tileRows: logitTileRows,
                    suppressedTokenIds,
                    options: this.sparseAttentionOptions,
                    bufferCache: this.bufferCache,
                    requestId: input.requestId,
                    layerIndex,
                    rememberResidentTensor: rememberDecodeResidentResult,
                    finalHiddenReadbackSkipped: true,
                    ...(decodeSampling ? { sampling: decodeSampling } : {}),
                  })
                : await projectDecodeLogits(
                    applyOptionalRmsNorm(
                      currentHidden,
                      this.weights.finalNorm,
                      this.weights.rmsNormEps,
                    ),
                    this.weights.outputProjection,
                    logitCandidateTokenIds,
                    suppressedTokenIds,
                    logitTopK,
                    logitTileRows,
                    this.sparseAttentionOptions,
                    this.bufferCache,
                    input.requestId,
                  ));
            logits = projectedLogits.logits;
            logitTokenIds = projectedLogits.logitTokenIds;
            backendProof = mergeDecodeBackendProof(
              projectedLogits.backendProof,
              projectionProofs,
              mlpProofs,
              oProjectionProofs,
              residencyProofs,
              residualRmsNormProofs,
            );
          }
        },
      }, {
        metadata: {
          backend: this.backendName,
          modelId: this.modelId,
          requestId: input.requestId,
          layerIndex,
        },
      });
      tspTrace.push(...layerTspTrace);
      executedLayer = true;
    }

    if (!executedLayer) throw new Error("UnlockedBrowserTransformerBackend decode requires at least one layer policy.");
    if (tokenCommandBatch) {
      decodeCommandBatchTraces.push(await tokenCommandBatch.submitOnce());
    }
	    if (logits.length === 0) {
      const projectedLogits = currentHiddenResident && !logitCandidateTokenIds
        ? await projectDecodeLogitsFromResidentHidden({
            hidden: currentHiddenResident,
            outputProjection: this.weights.outputProjection,
            ...(this.weights.finalNorm ? { finalNorm: this.weights.finalNorm } : {}),
            rmsNormEps: this.weights.rmsNormEps,
            topK: logitTopK,
            tileRows: logitTileRows,
            suppressedTokenIds,
            options: this.sparseAttentionOptions,
            bufferCache: this.bufferCache,
            requestId: input.requestId,
            layerIndex: input.policy.length - 1,
            rememberResidentTensor: rememberDecodeResidentResult,
            finalHiddenReadbackSkipped: true,
            ...(decodeSampling ? { sampling: decodeSampling } : {}),
          })
        : await projectDecodeLogits(
            applyOptionalRmsNorm(currentHidden, this.weights.finalNorm, this.weights.rmsNormEps),
            this.weights.outputProjection,
            logitCandidateTokenIds,
            suppressedTokenIds,
            logitTopK,
            logitTileRows,
            this.sparseAttentionOptions,
            this.bufferCache,
            input.requestId,
          );
      logits = projectedLogits.logits;
      logitTokenIds = projectedLogits.logitTokenIds;
	      backendProof = mergeDecodeBackendProof(
	        projectedLogits.backendProof,
	        projectionProofs,
	        mlpProofs,
	        oProjectionProofs,
	        residencyProofs,
	        residualRmsNormProofs,
	      );
	    }
	    backendProof = attachDecodePerfToBackendProof(backendProof, {
	      requestId: input.requestId,
	      generatedTokens: 1,
	      decodeCallCount: 1,
	      traces,
	      kvDecodeReused: true,
	      commandBatchTraces: decodeCommandBatchTraces,
	    });
	    const bestLogitIndex = backendProof?.sampling
      ? Math.max(0, logitTokenIds?.indexOf(backendProof.sampling.selectedTokenId) ?? 0)
      : argmax(logits);
    for (const tensor of decodeResidentTensors.reverse()) destroyWebGpuResidentTensor(tensor);
    return {
      requestId: input.requestId,
      tokenId: backendProof?.sampling?.selectedTokenId ?? logitTokenIds?.[bestLogitIndex] ?? bestLogitIndex,
      logitsHandle: {
        kind: "unlocked_browser_transformer_logits",
        id: `unlocked:${input.requestId}:decode:${input.inputTokenId}:${cache.tokenIds.length}`,
        requestId: input.requestId,
        logits,
        ...(logitTokenIds ? { logitTokenIds } : {}),
        ...(backendProof ? { backendProof } : {}),
        tspTrace,
        kvPagingEvents,
      } satisfies UnlockedBrowserDecodeHandle,
      traces,
    };
  }

  async verifySpeculativeDraft(input: UnlockedBrowserSpeculativeVerificationInput): Promise<UnlockedBrowserSpeculativeVerificationOutput> {
    this.assertReady();
    const cache = readUnlockedBrowserKvCacheHandle(input.kvCacheHandle);
    if (cache.requestId !== input.requestId) {
      throw new Error(`UnlockedBrowserTransformerBackend KV cache requestId mismatch: cache ${cache.requestId} cannot verify request ${input.requestId}.`);
    }
    if (cache.modelId !== this.modelId) {
      throw new Error(`UnlockedBrowserTransformerBackend KV cache modelId mismatch: cache ${cache.modelId} cannot verify model ${this.modelId}.`);
    }
    if (input.draftTokenIds.length === 0) {
      throw new Error("UnlockedBrowserTransformerBackend speculative verification requires at least one draft token.");
    }
    if (input.policy.length === 0) {
      throw new Error("UnlockedBrowserTransformerBackend speculative verification requires at least one layer policy.");
    }

    const started = nowMs();
    const verificationInputTokenIds = [input.previousTokenId, ...input.draftTokenIds.slice(0, -1)];
    const continuation = await runSpeculativeContinuation({
      weights: this.weights,
      modelId: this.modelId as string,
      cache,
      inputTokenIds: verificationInputTokenIds,
      requestId: input.requestId,
      policy: input.policy,
      logitCandidateTokenIds: normalizeLogitCandidateTokenIds(input.logitCandidateTokenIds, this.weights.vocabSize),
      suppressedTokenIds: normalizeSuppressedTokenIds(input.suppressedTokenIds, this.weights.vocabSize),
      logitTopK: normalizeLogitTopK(input.logitTopK, this.weights.vocabSize),
      logitTileRows: normalizeLogitTileRows(input.logitTileRows, this.weights.vocabSize),
      sparseAttentionOptions: this.sparseAttentionOptions,
      bufferCache: this.bufferCache,
      initialNonPinnedTier: this.initialNonPinnedTier,
    });
    const targetTokenIds = continuation.logitRows.map((logits, rowIndex) => {
      const bestLogitIndex = argmax(logits);
      return continuation.logitTokenIdsByRow[rowIndex]?.[bestLogitIndex] ?? bestLogitIndex;
    });
    const acceptedTokens = countAcceptedDraftPrefix(input.draftTokenIds, targetTokenIds);
    const rejectedTokens = input.draftTokenIds.length - acceptedTokens;
    const commitCount = acceptedTokens === input.draftTokenIds.length
      ? verificationInputTokenIds.length
      : Math.min(verificationInputTokenIds.length, acceptedTokens + 1);
    commitSpeculativeContinuation({
      cache,
      requestId: input.requestId,
      inputTokenIds: verificationInputTokenIds.slice(0, commitCount),
      continuation,
      commitCount,
      policy: input.policy,
      initialNonPinnedTier: this.initialNonPinnedTier,
    });
    const proofIndex = Math.max(0, Math.min(targetTokenIds.length - 1, acceptedTokens));
    const logits = continuation.logitRows[proofIndex] ?? [];
    const logitTokenIds = continuation.logitTokenIdsByRow[proofIndex];
    const backendProof = mergeDecodeBackendProof(
      continuation.logitProofs[proofIndex],
      continuation.projectionProofs,
      continuation.mlpProofs,
      continuation.oProjectionProofs,
    );
    const correctedTokenId = rejectedTokens > 0 ? targetTokenIds[acceptedTokens] : undefined;
    return {
      requestId: input.requestId,
      targetTokenIds,
      committedInputTokenIds: verificationInputTokenIds.slice(0, commitCount),
      acceptedTokens,
      rejectedTokens,
      ...(correctedTokenId !== undefined ? { correctedTokenId } : {}),
      verifyLatencyMs: Math.max(0, nowMs() - started),
      verifiedTokenCount: targetTokenIds.length,
      targetDecodeCalls: 1,
      decodeOutput: {
        requestId: input.requestId,
        tokenId: targetTokenIds[proofIndex] ?? 0,
        logitsHandle: {
          kind: "unlocked_browser_transformer_logits",
          id: `unlocked:${input.requestId}:mtp:${verificationInputTokenIds.join(".")}:${cache.tokenIds.length}`,
          requestId: input.requestId,
          logits,
          ...(logitTokenIds ? { logitTokenIds } : {}),
          ...(backendProof ? { backendProof } : {}),
          tspTrace: continuation.tspTrace,
          kvPagingEvents: continuation.kvPagingEvents,
        } satisfies UnlockedBrowserDecodeHandle,
        traces: continuation.traces,
      },
    };
  }

  async dispose(): Promise<void> {
    this.disposed = true;
    this.modelId = null;
    if (this.ownsBufferCache) this.bufferCache.clear();
  }

  private requireLayer(layerIndex: number): UnlockedBrowserTransformerLayerWeights {
    const layer = this.weights.layers[layerIndex];
    if (!layer) throw new Error(`UnlockedBrowserTransformerBackend has no layer ${layerIndex}.`);
    return layer;
  }

  private assertReady(): void {
    this.assertNotDisposed();
    if (!this.modelId) throw new Error("UnlockedBrowserTransformerBackend must initializeModel(modelId) before use.");
  }

  private assertNotDisposed(): void {
    if (this.disposed) throw new Error("UnlockedBrowserTransformerBackend has been disposed.");
  }
}

export function readUnlockedBrowserKvCacheHandle(handle: unknown): UnlockedBrowserKvCacheHandle {
  if (!isUnlockedBrowserKvCacheHandle(handle)) {
    throw new Error("Expected an UnlockedBrowserTransformerBackend KV cache handle.");
  }
  return handle;
}

export function getUnlockedBrowserLayerTensorHandles(handle: unknown, layerIndex: number): UnlockedBrowserLayerTensorHandles {
  const cache = readUnlockedBrowserKvCacheHandle(handle);
  const layer = cache.layers[layerIndex];
  if (!layer) throw new Error(`UnlockedBrowserTransformerBackend has no Q/K/V handles for layer ${layerIndex}.`);
  return layer;
}

export function readUnlockedBrowserDecodeHandle(handle: unknown): UnlockedBrowserDecodeHandle {
  if (!isRecord(handle) || handle.kind !== "unlocked_browser_transformer_logits" || !Array.isArray(handle.logits)) {
    throw new Error("Expected an UnlockedBrowserTransformerBackend logits handle.");
  }
  return handle as unknown as UnlockedBrowserDecodeHandle;
}

function readResidentSparseTensorHandle(handle: unknown): ResidentSparseTensorHandle | null {
  if (!isRecord(handle) || handle.kind !== "unlocked_browser_transformer_resident_sparse_tensor") return null;
  const tensor = handle.tensor as Partial<WebGpuResidentTensor> | undefined;
  if (!tensor || tensor.kind !== "webgpu_resident_tensor" || typeof tensor.id !== "string") {
    throw new Error("Expected an UnlockedBrowserTransformerBackend resident sparse tensor handle.");
  }
  return handle as unknown as ResidentSparseTensorHandle;
}

export function createRecommendedUnlockedBrowserModelPlan(): UnlockedBrowserModelPlan {
  return {
    primary: {
      modelId: "Qwen/Qwen3-0.6B",
      family: "qwen3",
      browserRuntime: "custom-webgpu-transformer",
      reason: "Qwen3 0.6B is the smallest strong browser-control target with a modern transformer shape and enough instruction-following ability to justify full SSA/KV/TSP investment.",
    },
    alternates: [
      {
        modelId: "HuggingFaceTB/SmolLM2-1.7B-Instruct",
        family: "smollm2",
        reason: "Good browser-size fallback if Qwen conversion or licensing blocks a fully controlled asset.",
      },
      {
        modelId: "meta-llama/Llama-3.2-1B-Instruct",
        family: "llama3.2",
        reason: "Common baseline with mature tokenizer/runtime support, but less attractive than Qwen for the unlocked primary path.",
      },
    ],
    rejectedFamilies: [
      {
        family: "diffusion",
        reason: "Diffusion does not preserve the autoregressive KV cache and target-verifier loop that SSA/KV/TSP/MTP are built to control.",
      },
      {
        family: "opaque-browser-chat-api",
        reason: "Opaque browser inference APIs do not expose model-layer Q/K/V tensors or KV-cache ownership.",
      },
    ],
  };
}

function createLayerHandles(input: {
  requestId: string;
  layerIndex: number;
  q: Matrix;
  k: Matrix;
  v: Matrix;
  blockTokenRanges: UnlockedBrowserKvCacheHandle["blockTokenRanges"];
  cloneMatrices?: boolean;
}): UnlockedBrowserLayerTensorHandles {
  const createHandle = input.cloneMatrices === false ? createUnclonedSsaToyTensorHandle : createSsaToyTensorHandle;
  return {
    qHandle: createHandle({
      id: `unlocked:${input.requestId}:layer${input.layerIndex}:q`,
      matrix: input.q,
      blockTokenRanges: input.blockTokenRanges,
    }),
    kHandle: createHandle({
      id: `unlocked:${input.requestId}:layer${input.layerIndex}:k`,
      matrix: input.k,
      blockTokenRanges: input.blockTokenRanges,
    }),
    vHandle: createHandle({
      id: `unlocked:${input.requestId}:layer${input.layerIndex}:v`,
      matrix: input.v,
      blockTokenRanges: input.blockTokenRanges,
    }),
  };
}

function createAttentionLayerHandles(input: {
  requestId: string;
  layerIndex: number;
  q: Matrix;
  k: Matrix | undefined;
  v: Matrix | undefined;
  fallback: UnlockedBrowserLayerTensorHandles;
  blockTokenRanges: UnlockedBrowserKvCacheHandle["blockTokenRanges"];
}): UnlockedBrowserLayerTensorHandles {
  if (!input.k || !input.v || input.k.length !== input.q.length || input.v.length !== input.q.length) {
    return input.fallback;
  }
  return {
    qHandle: input.fallback.qHandle,
    kHandle: createUnclonedSsaToyTensorHandle({
      id: `unlocked:${input.requestId}:layer${input.layerIndex}:k:attention`,
      matrix: input.k,
      blockTokenRanges: input.blockTokenRanges,
    }),
    vHandle: createUnclonedSsaToyTensorHandle({
      id: `unlocked:${input.requestId}:layer${input.layerIndex}:v:attention`,
      matrix: input.v,
      blockTokenRanges: input.blockTokenRanges,
    }),
  };
}

function createUnclonedSsaToyTensorHandle(input: {
  id: string;
  matrix: Matrix;
  blockTokenRanges: Record<string, { tokenStart: number; tokenEnd: number }>;
}): SsaToyTensorHandle {
  return {
    kind: "ssa_toy_tensor",
    id: input.id,
    matrix: input.matrix,
    blockTokenRanges: normalizeBlockTokenRanges(input.blockTokenRanges),
  };
}

function normalizeBlockTokenRanges(
  ranges: Record<string, { tokenStart: number; tokenEnd: number }>,
): Record<string, { tokenStart: number; tokenEnd: number }> {
  const normalized: Record<string, { tokenStart: number; tokenEnd: number }> = {};
  for (const [id, range] of Object.entries(ranges)) {
    normalized[id] = {
      tokenStart: Math.max(0, Math.floor(range.tokenStart)),
      tokenEnd: Math.max(0, Math.floor(range.tokenEnd)),
    };
  }
  return normalized;
}

type PackedSparseAttentionBackend = WebGpuSparseAttentionResult["backend"] | "mixed";

type PackedHeadSparseTrace = SSAKernelTrace & {
  attentionBackend: PackedSparseAttentionBackend;
  packedHeadBackends: WebGpuSparseAttentionResult["backend"][];
  packedHeadCount: number;
  keyValueHeadCount?: number;
  keyValueCompressionRatio?: number;
  queryTokenCount?: number;
  attentionDispatchCount?: number;
  attentionPipelineCacheKey?: string;
  attentionFusedStage?: "one_token_attention";
  outputResident?: true;
  readback?: false;
  inputResident?: {
    q: boolean;
    k: boolean;
    v: boolean;
  };
};

interface ResidentSparseTensorHandle {
  kind: "unlocked_browser_transformer_resident_sparse_tensor";
  id: string;
  tensor: WebGpuResidentTensor;
  blockTokenRanges: Record<string, { tokenStart: number; tokenEnd: number }>;
}

async function executePackedHeadSparseLayer(
  input: SparseLayerForwardInput,
  geometry: AttentionGeometry,
  options: Pick<WebGpuSsaBackendOptions, "backendPreference" | "device" | "gpu" | "requireWebGpu">,
  bufferCache: WebGpuRuntimeBufferCache,
): Promise<SparseLayerForwardOutput> {
  const startedAt = nowMs();
  const q = readSsaToyTensorHandle(input.qHandle);
  const k = readSsaToyTensorHandle(input.kHandle);
  const v = readSsaToyTensorHandle(input.vHandle);
  const residentQ = isResidentTensor(input.residentQ) ? input.residentQ : undefined;
  const residentK = isResidentTensor(input.residentK) ? input.residentK : undefined;
  const residentV = isResidentTensor(input.residentV) ? input.residentV : undefined;
  const commandBatch = input.commandBatch instanceof WebGpuDecodeCommandBatch
    ? input.commandBatch
    : undefined;
  const queryTokenIndexes = normalizeSparseQueryTokenIndexes(input.queryTokenIndexes, q.matrix.length);
  const queryRows = residentQ
    ? createZeroMatrix(queryTokenIndexes.length, geometry.qProjectionSize)
    : queryTokenIndexes.map((tokenIndex) => q.matrix[tokenIndex] ?? []);
  const lastQueryTokenIndex = queryTokenIndexes[queryTokenIndexes.length - 1] ?? q.matrix.length - 1;
  const lastQueryBlockIndex = queryBlockIndexForToken(lastQueryTokenIndex, input.policy);
  const selectedKeyIndexesByQuery = queryTokenIndexes.map((queryTokenIndex) => (
    buildSelectedKeyIndexesForQuery(q, k, input.policy, queryTokenIndex)
      .filter((keyIndex) => keyIndex <= queryTokenIndex)
  ));
  const packedInput = {
    q: residentQ ?? queryRows,
    k: residentK ?? k.matrix,
    v: residentV ?? v.matrix,
    selectedKeyIndexesByQuery,
    causal: false,
    headCount: geometry.attentionHeads,
    keyValueHeadCount: residentK ? geometry.keyValueHeads : packedKeyValueHeadCount(k.matrix, geometry),
    headDim: geometry.headDim,
    ...(options.backendPreference ? { backendPreference: options.backendPreference } : {}),
    ...(options.device ? { device: options.device } : {}),
    ...(options.gpu ? { gpu: options.gpu } : {}),
    ...(options.requireWebGpu ? { requireWebGpu: true } : {}),
    ...(commandBatch ? { commandBatch } : {}),
    bufferCache,
  };
  const packed = input.preferResidentOutput && options.backendPreference !== "cpu"
    ? await runResidentPackedSparseAttentionOrFallback(packedInput, options.requireWebGpu === true)
    : await runPackedSparseAttentionWebGpu({
        ...packedInput,
        q: queryRows,
        k: k.matrix,
        v: v.matrix,
      });
  const output = "output" in packed ? packed.output : undefined;
  const residentOutput = "tensor" in packed ? packed.tensor : undefined;
  const packedHeadBackends = repeatBackend(packed.backend, geometry.attentionHeads);
  const selectedBlockIds = selectedBlockIdsForQueryBlock(input.policy, lastQueryBlockIndex);
  const attentionMs = packed.trace.computeMs || nowMs() - startedAt;
  const attentionBackend = summarizePackedBackends(packedHeadBackends);
  const outputHandle = output
    ? createSsaToyTensorHandle({
        id: `unlocked:${input.requestId}:layer${input.layerIndex}:packed_sparse_out`,
        matrix: output,
        blockTokenRanges: q.blockTokenRanges,
      })
    : createResidentSparseTensorHandle({
        id: `unlocked:${input.requestId}:layer${input.layerIndex}:packed_sparse_out:resident`,
        tensor: residentOutput,
        blockTokenRanges: q.blockTokenRanges,
      });
  const trace: PackedHeadSparseTrace = {
    requestId: input.requestId,
    layerIndex: input.layerIndex,
    queryBlockIndex: lastQueryBlockIndex,
    selectedBlockIds,
    pinnedBlockIds: [...input.policy.pinnedBlockIds].sort(compareBlockIds),
    denseTokenCountEstimate: k.matrix.length,
    sparseTokenCountEstimate: selectedKeyIndexesByQuery[selectedKeyIndexesByQuery.length - 1]?.length ?? 0,
    routingMs: 0,
    gatherMs: 0,
    attentionMs,
    attentionBackend,
    packedHeadBackends,
    packedHeadCount: geometry.attentionHeads,
    keyValueHeadCount: packed.trace.keyValueHeadCount,
    keyValueCompressionRatio: keyValueCompressionRatio(packed.trace.keyValueHeadCount, geometry.attentionHeads),
    queryTokenCount: queryRows.length,
    attentionDispatchCount: packed.trace.dispatchCount,
    ...(packed.trace.pipelineCacheKey ? { attentionPipelineCacheKey: packed.trace.pipelineCacheKey } : {}),
    ...(packed.trace.metadata?.fusedStage === "one_token_attention" ? { attentionFusedStage: "one_token_attention" as const } : {}),
    ...("outputResident" in packed.trace && packed.trace.outputResident
      ? {
          outputResident: true,
          readback: false,
          inputResident: packed.trace.inputResident,
        }
      : {}),
  };
  return {
    requestId: input.requestId,
    layerIndex: input.layerIndex,
    outputHandle,
    trace,
  };
}

async function runResidentPackedSparseAttentionOrFallback(
  input: Parameters<typeof runPackedSparseAttentionResidentWebGpu>[0],
  requireWebGpu: boolean,
): Promise<
  Awaited<ReturnType<typeof runPackedSparseAttentionResidentWebGpu>>
  | Awaited<ReturnType<typeof runPackedSparseAttentionWebGpu>>
> {
  try {
    return await runPackedSparseAttentionResidentWebGpu(input);
  } catch (error) {
    if (requireWebGpu || !isWebGpuUnavailableError(error)) throw error;
    if (isResidentTensor(input.q) || isResidentTensor(input.k) || isResidentTensor(input.v)) throw error;
    return runPackedSparseAttentionWebGpu({
      ...input,
      q: input.q,
      k: input.k,
      v: input.v,
    });
  }
}

function isWebGpuUnavailableError(error: unknown): boolean {
  return error instanceof Error && /WebGPU is not available/i.test(error.message);
}

function createResidentSparseTensorHandle(input: {
  id: string;
  tensor: WebGpuResidentTensor | undefined;
  blockTokenRanges: Record<string, { tokenStart: number; tokenEnd: number }>;
}): ResidentSparseTensorHandle {
  if (!input.tensor) throw new Error("Packed sparse attention did not return a resident output tensor.");
  return {
    kind: "unlocked_browser_transformer_resident_sparse_tensor",
    id: input.id,
    tensor: input.tensor,
    blockTokenRanges: normalizeBlockTokenRanges(input.blockTokenRanges),
  };
}

function summarizePackedBackends(backends: WebGpuSparseAttentionResult["backend"][]): PackedSparseAttentionBackend {
  const first = backends[0];
  if (!first) return "cpu_reference";
  return backends.every((backend) => backend === first) ? first : "mixed";
}

function repeatBackend(backend: WebGpuSparseAttentionResult["backend"], count: number): WebGpuSparseAttentionResult["backend"][] {
  return Array.from({ length: Math.max(0, count) }, () => backend);
}

function packedKeyValueHeadCount(k: Matrix, geometry: AttentionGeometry): number {
  const rowWidth = k[0]?.length ?? geometry.keyValueHeads * geometry.headDim;
  return Math.max(1, Math.floor(rowWidth / Math.max(1, geometry.headDim)));
}

function keyValueCompressionRatio(keyValueHeadCount: number, attentionHeadCount: number): number {
  if (attentionHeadCount <= 0) return 1;
  return Math.max(0, Math.min(1, keyValueHeadCount / attentionHeadCount));
}

function normalizeSparseQueryTokenIndexes(value: number[] | undefined, queryRowCount: number): number[] {
  if (!value || value.length === 0) return Array.from({ length: queryRowCount }, (_item, index) => index);
  const indexes = value
    .filter((index) => Number.isInteger(index) && index >= 0 && index < queryRowCount)
    .filter((index, position, all) => all.indexOf(index) === position)
    .sort((a, b) => a - b);
  return indexes.length > 0 ? indexes : Array.from({ length: queryRowCount }, (_item, index) => index);
}

function buildKvBlocks(input: {
  layerIndex: number;
  blockTokenRanges: Record<string, { tokenStart: number; tokenEnd: number }>;
  policy: SSALayerRoutingPolicy;
  initialNonPinnedTier: Extract<KVTier, "vram" | "ram" | "disk">;
}): KVBlock[] {
  const pinned = new Set(input.policy.pinnedBlockIds);
  return Object.entries(input.blockTokenRanges).map(([blockId, range], index) => {
    const fullId = `layer${input.layerIndex}:${blockId}`;
    const isPinned = pinned.has(blockId);
    return {
      id: fullId,
      layer: input.layerIndex,
      startToken: range.tokenStart,
      endToken: range.tokenEnd,
      tier: isPinned ? "vram" : input.initialNonPinnedTier,
      pinned: isPinned,
      importance: isPinned ? 1 : Math.max(0.1, 0.8 - index * 0.1),
      lastAccessAt: Date.now() - index,
      estimatedBytes: Math.max(1, range.tokenEnd - range.tokenStart) * input.policy.blockSize * Float32Array.BYTES_PER_ELEMENT,
      tensorHandles: {
        key: tensorHandle(`${fullId}:key`),
        value: tensorHandle(`${fullId}:value`),
      },
      compressedKeySummary: new Float32Array([range.tokenStart, range.tokenEnd]),
    };
  });
}

function tensorHandle(id: string): BackendTensorHandle {
  return {
    backend: "unlocked-browser-transformer",
    id,
    dtype: "f32",
  };
}

function buildDecodeSchedule(layerIndex: number, tokenCount: number): TSPScheduleStep[] {
  return [
    { id: `prefetch_l${layerIndex}`, kind: "kv_prefetch", sequenceShard: 0, tensorShard: 0, tokenStart: 0, tokenEnd: tokenCount },
    { id: `attention_l${layerIndex}`, kind: "attention", sequenceShard: 0, tensorShard: 0, tokenStart: 0, tokenEnd: tokenCount },
    { id: `mlp_l${layerIndex}`, kind: "mlp", sequenceShard: 0, tensorShard: 0, tokenStart: 0, tokenEnd: tokenCount },
  ];
}

function buildTrace(
  requestId: string,
  layerIndex: number,
  policy: SSALayerRoutingPolicy,
  denseTokenCount: number,
): SSAKernelTrace {
  const selectedBlockIds = collectSelectedBlockIds(policy);
  return {
    requestId,
    layerIndex,
    queryBlockIndex: 0,
    selectedBlockIds,
    pinnedBlockIds: [...policy.pinnedBlockIds].sort(compareBlockIds),
    denseTokenCountEstimate: denseTokenCount,
    sparseTokenCountEstimate: selectedBlockIds.length * Math.max(1, policy.blockSize),
    routingMs: 0,
    gatherMs: 0,
    attentionMs: 0,
  };
}

function buildBlockTokenRanges(
  tokenCount: number,
  layerPolicy: SSALayerRoutingPolicy,
): UnlockedBrowserKvCacheHandle["blockTokenRanges"] {
  const blockSize = Math.max(1, layerPolicy.blockSize);
  const blockIds = collectKnownBlockIds(layerPolicy);
  const blockCount = Math.ceil(tokenCount / blockSize);
  const ranges: UnlockedBrowserKvCacheHandle["blockTokenRanges"] = {};
  for (let blockIndex = 0; blockIndex < blockCount; blockIndex += 1) {
    ranges[`b${blockIndex}`] = {
      tokenStart: Math.min(tokenCount, blockIndex * blockSize),
      tokenEnd: Math.min(tokenCount, (blockIndex + 1) * blockSize),
    };
  }
  for (const id of blockIds) {
    if (ranges[id]) continue;
    const tokenBlockIndex = blockIndexFromId(id);
    if (tokenBlockIndex === undefined) continue;
    ranges[id] = {
      tokenStart: Math.min(tokenCount, tokenBlockIndex * blockSize),
      tokenEnd: Math.min(tokenCount, (tokenBlockIndex + 1) * blockSize),
    };
  }
  return ranges;
}

function collectKnownBlockIds(policy: SSALayerRoutingPolicy): string[] {
  return collectSelectedBlockIds(policy).sort(compareBlockIds);
}

function maxSelectedBlockCount(policies: SSALayerRoutingPolicy[]): number {
  return policies.reduce((max, policy) => Math.max(max, collectSelectedBlockIds(policy).length), 0);
}

function maxPolicyBlockSize(policies: SSALayerRoutingPolicy[]): number {
  return policies.reduce((max, policy) => Math.max(max, policy.blockSize), 1);
}

function collectSelectedBlockIds(policy: SSALayerRoutingPolicy): string[] {
  const ids: string[] = [];
  for (const id of policy.pinnedBlockIds) pushUnique(ids, id);
  for (const selected of Object.values(policy.selectedBlockIdsByQueryBlock)) {
    for (const id of selected) pushUnique(ids, id);
  }
  return ids.sort(compareBlockIds);
}

function pushUnique(values: string[], value: string): void {
  if (!values.includes(value)) values.push(value);
}

function compareBlockIds(a: string, b: string): number {
  const aNumber = blockIndexFromId(a);
  const bNumber = blockIndexFromId(b);
  if (aNumber !== undefined && bNumber !== undefined) return aNumber - bNumber;
  return a.localeCompare(b);
}

function blockIndexFromId(id: string): number | undefined {
  const match = /^b(\d+)$/.exec(id);
  if (!match) return undefined;
  const index = Number(match[1]);
  return Number.isSafeInteger(index) && index >= 0 ? index : undefined;
}

function normalizeLayerIndex(layerIndex: number, position: number): number {
  return layerIndex < 0 ? position : layerIndex;
}

function tokenEmbedding(weights: UnlockedBrowserTransformerWeights, tokenId: number): Vector {
  const embedding = matrixRow(weights.tokenEmbedding, positiveModulo(tokenId, weights.vocabSize));
  if (!embedding) throw new Error(`Missing token embedding for token ${tokenId}.`);
  return Array.from(embedding);
}

function projectMatrix(input: Matrix, projection: RuntimeMatrix): Matrix {
  return input.map((row) => matVec(row, projection));
}

function matVec(vector: RuntimeVector, matrix: RuntimeMatrix): Vector {
  const output: Vector = new Array(matrixRowCount(matrix));
  for (let rowIndex = 0; rowIndex < output.length; rowIndex += 1) {
    output[rowIndex] = dot(matrixRow(matrix, rowIndex) ?? [], vector);
  }
  return output;
}

type UnlockedBrowserTransformerLayerWithMlp = UnlockedBrowserTransformerLayerWeights & {
  mlpUpProj: RuntimeMatrix;
  mlpDownProj: RuntimeMatrix;
};

function hasMlp(layer: UnlockedBrowserTransformerLayerWeights): layer is UnlockedBrowserTransformerLayerWithMlp {
  return Boolean(layer.mlpUpProj && layer.mlpDownProj);
}

function runMlp(hidden: Vector, layer: UnlockedBrowserTransformerLayerWithMlp): Vector {
  if (layer.mlpGateProj) {
    const gate = matVec(hidden, layer.mlpGateProj).map(silu);
    const up = matVec(hidden, layer.mlpUpProj);
    return matVec(multiplyVectors(gate, up), layer.mlpDownProj);
  }
  const up = matVec(hidden, layer.mlpUpProj).map(gelu);
  return matVec(up, layer.mlpDownProj);
}

function projectToLogits(hidden: RuntimeVector, outputProjection: RuntimeMatrix): number[] {
  const logits = new Array(matrixRowCount(outputProjection));
  for (let rowIndex = 0; rowIndex < logits.length; rowIndex += 1) {
    logits[rowIndex] = dot(matrixRow(outputProjection, rowIndex) ?? [], hidden);
  }
  return logits;
}

async function projectDecodeLogits(
  hidden: RuntimeVector,
  outputProjection: RuntimeMatrix,
  candidateTokenIds: number[] | null,
  suppressedTokenIds: number[],
  topK: number,
  tileRows: number | null,
  options: Pick<WebGpuSsaBackendOptions, "backendPreference" | "device" | "gpu" | "requireWebGpu">,
  bufferCache: WebGpuRuntimeBufferCache,
  requestId: string,
): Promise<DecodeLogitProjectionResult> {
  const purpose = candidateTokenIds ? "candidate_logit_projection" : "full_vocab_topk_logit_projection";
  if (!candidateTokenIds) {
    if (options.backendPreference !== "cpu") {
      const greedy = await tryProjectGreedyDecodeLogits(
        hidden,
        outputProjection,
        suppressedTokenIds,
        tileRows,
        options,
        bufferCache,
        requestId,
        { residentHiddenUpload: false },
      );
      if (greedy) return greedy;
      const residentHidden = await tryProjectResidentDecodeLogits(
        hidden,
        outputProjection,
        topK,
        tileRows,
        suppressedTokenIds,
        options,
        bufferCache,
        requestId,
        "full_vocab_topk_logit_projection",
      );
      if (residentHidden) return residentHidden;
    }
    const projected = await runDenseMatVecTopKWebGpu({
      vector: hidden,
      matrix: outputProjection,
      topK,
      ...(tileRows ? { tileRows } : {}),
      ...(suppressedTokenIds.length > 0 ? { suppressedRowIds: suppressedTokenIds } : {}),
      ...(options.backendPreference ? { backendPreference: options.backendPreference } : {}),
      ...(options.device ? { device: options.device } : {}),
      ...(options.gpu ? { gpu: options.gpu } : {}),
      ...(options.requireWebGpu ? { requireWebGpu: true } : {}),
      ...(isPackedRuntimeMatrix(outputProjection)
        ? {
            bufferCache,
            projectionCacheKey: "outputProjection",
            projectionCachePolicy: "stable" as const,
          }
        : {}),
      traceMetadata: {
        runtime: "unlocked-browser-transformer",
        requestId,
        purpose,
      },
    });
    return {
      logits: projected.values,
      logitTokenIds: projected.selectedRowIds,
      backendProof: {
        logitProjection: {
          backend: projected.backend,
          selectedRowIds: projected.selectedRowIds,
          fullRowCount: matrixRowCount(outputProjection),
          selectedRowCount: projected.trace.selectedRows,
          purpose,
          trace: projected.trace,
        },
      },
    };
  }
  const suppressedTokenSet = new Set(suppressedTokenIds);
  const selectedCandidateTokenIds = candidateTokenIds.filter((tokenId) => !suppressedTokenSet.has(tokenId));
  if (selectedCandidateTokenIds.length === 0) throw new Error("candidate logit projection has no unsuppressed token ids.");
  const projected = await runDenseMatVecWebGpu({
    vector: hidden,
    matrix: outputProjection,
    selectedRowIds: selectedCandidateTokenIds,
    ...(options.backendPreference ? { backendPreference: options.backendPreference } : {}),
    ...(options.device ? { device: options.device } : {}),
    ...(options.gpu ? { gpu: options.gpu } : {}),
    ...(options.requireWebGpu ? { requireWebGpu: true } : {}),
    traceMetadata: {
      runtime: "unlocked-browser-transformer",
      requestId,
      purpose,
    },
  });
  return {
    logits: projected.values,
    logitTokenIds: selectedCandidateTokenIds,
    backendProof: {
      logitProjection: {
        backend: projected.backend,
        selectedRowIds: projected.selectedRowIds ?? selectedCandidateTokenIds,
        fullRowCount: matrixRowCount(outputProjection),
        selectedRowCount: projected.trace.selectedRows,
        purpose,
        trace: projected.trace,
      },
    },
  };
}

async function tryProjectGreedyDecodeLogits(
  hidden: RuntimeVector,
  outputProjection: RuntimeMatrix,
  suppressedTokenIds: number[],
  tileRows: number | null,
  options: Pick<WebGpuSsaBackendOptions, "backendPreference" | "device" | "gpu" | "requireWebGpu">,
  bufferCache: WebGpuRuntimeBufferCache,
  requestId: string,
  metadata: Record<string, unknown> = {},
): Promise<DecodeLogitProjectionResult | null> {
  if (options.backendPreference === "cpu") return null;
  try {
    const projected = await projectGreedyDecodeTokenWebGpu({
      hidden,
      outputProjection,
      suppressedTokenIds,
      tileRows,
      options,
      requireResidentHidden: options.requireWebGpu === true || options.backendPreference === "webgpu",
      ...(isPackedRuntimeMatrix(outputProjection)
        ? {
            bufferCache,
            projectionCacheKey: "outputProjection",
          }
        : {}),
      requestId,
      traceMetadata: {
        runtime: "unlocked-browser-transformer",
        requestId,
        phase: "decode",
        ...metadata,
      },
    });
    return greedyProjectionToDecodeLogits(projected, outputProjection);
  } catch (error) {
    if (options.requireWebGpu) throw error;
    return null;
  }
}

function greedyProjectionToDecodeLogits(
  projected: Awaited<ReturnType<typeof projectGreedyDecodeTokenWebGpu>>,
  outputProjection: RuntimeMatrix,
): DecodeLogitProjectionResult {
  return {
    logits: [projected.score],
    logitTokenIds: [projected.tokenId],
    backendProof: {
      logitProjection: {
        backend: projected.backend,
        selectedRowIds: [projected.tokenId],
        fullRowCount: matrixRowCount(outputProjection),
        selectedRowCount: 1,
        purpose: "greedy_argmax_logit_projection",
        trace: projected.trace,
      },
    },
  };
}

function compactProjectionToDecodeLogits(
  projected: Awaited<ReturnType<typeof projectCompactTopKDecodeTokensWebGpu>>,
  outputProjection: RuntimeMatrix,
  sampling: Required<Pick<DecodeSamplingInput, "temperature" | "topP" | "repetitionPenalty">> & DecodeSamplingInput,
  suppressedTokenIds: number[],
): DecodeLogitProjectionResult {
  const sampled = sampleFromCompactTopK(projected.candidates, {
    temperature: sampling.temperature,
    topP: sampling.topP,
    repetitionPenalty: sampling.repetitionPenalty,
    recentTokenIds: sampling.recentTokenIds ?? [],
    suppressedTokenIds,
    ...(sampling.seed !== undefined ? { seed: sampling.seed } : {}),
  });
  return {
    logits: projected.candidates.map((candidate) => candidate.score),
    logitTokenIds: projected.candidates.map((candidate) => candidate.tokenId),
    selectedTokenId: sampled.tokenId,
    backendProof: {
      logitProjection: {
        backend: projected.backend,
        selectedRowIds: projected.candidates.map((candidate) => candidate.tokenId),
        fullRowCount: matrixRowCount(outputProjection),
        selectedRowCount: projected.candidates.length,
        purpose: "compact_topk_logit_projection",
        trace: projected.trace,
      },
      sampling: {
        strategy: sampled.strategy,
        selectedTokenId: sampled.tokenId,
        selectedScore: sampled.score,
        selectedRank: sampled.selectedRank,
        effectiveCandidateCount: sampled.effectiveCandidateCount,
        compactLogitTopK: projected.trace.compactTopK,
        temperature: sampling.temperature,
        topP: sampling.topP,
        repetitionPenalty: sampling.repetitionPenalty,
        greedyDecodeUsed: sampled.strategy === "greedy",
        suppressedTokenCount: suppressedTokenIds.length,
        recentTokenCount: sampling.recentTokenIds?.length ?? 0,
      },
    },
  };
}

async function tryProjectResidentDecodeLogits(
  hidden: RuntimeVector,
  outputProjection: RuntimeMatrix,
  topK: number,
  tileRows: number | null,
  suppressedTokenIds: number[],
  options: Pick<WebGpuSsaBackendOptions, "backendPreference" | "device" | "gpu" | "requireWebGpu">,
  bufferCache: WebGpuRuntimeBufferCache,
  requestId: string,
  purpose: "full_vocab_topk_logit_projection",
): Promise<DecodeLogitProjectionResult | null> {
  let residentHidden: Awaited<ReturnType<typeof uploadWebGpuResidentTensor>> | undefined;
  try {
    residentHidden = await uploadWebGpuResidentTensor({
      matrix: [Array.from(hidden)],
      ...(options.device ? { device: options.device } : {}),
      ...(options.gpu ? { gpu: options.gpu } : {}),
      ...(options.requireWebGpu ? { requireWebGpu: true } : {}),
      traceMetadata: {
        runtime: "unlocked-browser-transformer",
        requestId,
        purpose: "decode_hidden_resident_upload",
      },
    });
    const projected = await runDenseMatVecTopKResidentWebGpu({
      vector: residentHidden.tensor,
      matrix: outputProjection,
      topK,
      ...(tileRows ? { tileRows } : {}),
      ...(suppressedTokenIds.length > 0 ? { suppressedRowIds: suppressedTokenIds } : {}),
      ...(options.device ? { device: options.device } : {}),
      ...(options.gpu ? { gpu: options.gpu } : {}),
      ...(options.requireWebGpu ? { requireWebGpu: true } : {}),
      ...(isPackedRuntimeMatrix(outputProjection)
        ? {
            bufferCache,
            projectionCacheKey: "outputProjection",
            projectionCachePolicy: "stable" as const,
          }
        : {}),
      traceMetadata: {
        runtime: "unlocked-browser-transformer",
        requestId,
        purpose,
        residentHiddenUpload: true,
      },
    });
    return {
      logits: projected.values,
      logitTokenIds: projected.selectedRowIds,
      backendProof: {
        logitProjection: {
          backend: projected.backend,
          selectedRowIds: projected.selectedRowIds,
          fullRowCount: matrixRowCount(outputProjection),
          selectedRowCount: projected.trace.selectedRows,
          purpose,
          trace: projected.trace,
        },
      },
    };
  } catch (error) {
    if (options.requireWebGpu) throw error;
    return null;
  } finally {
    if (residentHidden) destroyWebGpuResidentTensor(residentHidden.tensor);
  }
}

async function projectDecodeLogitsFromResidentHidden(
  input: {
    hidden: WebGpuResidentTensor;
    outputProjection: RuntimeMatrix;
    finalNorm?: RuntimeVector;
    rmsNormEps: number | undefined;
    topK: number;
    tileRows: number | null;
    suppressedTokenIds: number[];
    options: Pick<WebGpuSsaBackendOptions, "backendPreference" | "device" | "gpu" | "requireWebGpu">;
    bufferCache: WebGpuRuntimeBufferCache;
    requestId: string;
    layerIndex: number;
    rememberResidentTensor: <T extends { tensor: Parameters<typeof destroyWebGpuResidentTensor>[0] }>(result: T) => T;
    finalHiddenReadbackSkipped?: boolean;
    sampling?: DecodeSamplingInput;
  },
): Promise<DecodeLogitProjectionResult> {
  const purpose = "full_vocab_topk_logit_projection";
  const normalizedHidden = input.finalNorm
    ? input.rememberResidentTensor(await runRmsNormResidentWebGpu({
        hidden: input.hidden,
        weight: input.finalNorm,
        ...(input.rmsNormEps !== undefined ? { eps: input.rmsNormEps } : {}),
        ...(input.options.device ? { device: input.options.device } : {}),
        ...(input.options.gpu ? { gpu: input.options.gpu } : {}),
        ...(input.options.requireWebGpu ? { requireWebGpu: true } : {}),
        ...(input.bufferCache ? { bufferCache: input.bufferCache } : {}),
        traceMetadata: {
          runtime: "unlocked-browser-transformer",
          requestId: input.requestId,
          layerIndex: input.layerIndex,
          phase: "decode",
          purpose: "decode_final_rmsnorm",
          residentDecodeFinalLogits: true,
        },
      })).tensor
    : input.hidden;
  if (input.options.backendPreference !== "cpu" && shouldUseCompactTopKSampling(input.topK, input.sampling)) {
    const compact = await projectCompactTopKDecodeTokensWebGpu({
      hidden: normalizedHidden,
      outputProjection: input.outputProjection,
      topK: input.topK,
      suppressedTokenIds: input.suppressedTokenIds,
      tileRows: input.tileRows,
      options: input.options,
      requireResidentHidden: input.options.requireWebGpu === true || input.options.backendPreference === "webgpu",
      ...(isPackedRuntimeMatrix(input.outputProjection)
        ? {
            bufferCache: input.bufferCache,
            projectionCacheKey: "outputProjection",
          }
        : {}),
      requestId: input.requestId,
      traceMetadata: {
        runtime: "unlocked-browser-transformer",
        requestId: input.requestId,
        layerIndex: input.layerIndex,
        phase: "decode",
        residentDecodeFinalLogits: true,
        finalNormResident: Boolean(input.finalNorm),
        ...(input.finalHiddenReadbackSkipped ? { finalHiddenReadbackSkipped: true } : {}),
      },
    });
    return compactProjectionToDecodeLogits(
      compact,
      input.outputProjection,
      resolveDecodeSampling(input.sampling),
      input.suppressedTokenIds,
    );
  }
  if (input.options.backendPreference !== "cpu") {
    const greedy = await projectGreedyDecodeTokenWebGpu({
      hidden: normalizedHidden,
      outputProjection: input.outputProjection,
      suppressedTokenIds: input.suppressedTokenIds,
      tileRows: input.tileRows,
      options: input.options,
      requireResidentHidden: input.options.requireWebGpu === true || input.options.backendPreference === "webgpu",
      ...(isPackedRuntimeMatrix(input.outputProjection)
        ? {
            bufferCache: input.bufferCache,
            projectionCacheKey: "outputProjection",
          }
        : {}),
      requestId: input.requestId,
      traceMetadata: {
        runtime: "unlocked-browser-transformer",
        requestId: input.requestId,
        layerIndex: input.layerIndex,
        phase: "decode",
        residentDecodeFinalLogits: true,
        finalNormResident: Boolean(input.finalNorm),
        ...(input.finalHiddenReadbackSkipped ? { finalHiddenReadbackSkipped: true } : {}),
      },
    });
    return greedyProjectionToDecodeLogits(greedy, input.outputProjection);
  }
  const projected = await runDenseMatVecTopKResidentWebGpu({
    vector: normalizedHidden,
    matrix: input.outputProjection,
    topK: input.topK,
    ...(input.tileRows ? { tileRows: input.tileRows } : {}),
    ...(input.suppressedTokenIds.length > 0 ? { suppressedRowIds: input.suppressedTokenIds } : {}),
    ...(input.options.device ? { device: input.options.device } : {}),
    ...(input.options.gpu ? { gpu: input.options.gpu } : {}),
    ...(input.options.requireWebGpu ? { requireWebGpu: true } : {}),
    ...(isPackedRuntimeMatrix(input.outputProjection)
      ? {
          bufferCache: input.bufferCache,
          projectionCacheKey: "outputProjection",
          projectionCachePolicy: "stable" as const,
        }
      : {}),
    traceMetadata: {
      runtime: "unlocked-browser-transformer",
      requestId: input.requestId,
      layerIndex: input.layerIndex,
      purpose,
      residentDecodeFinalLogits: true,
      finalNormResident: Boolean(input.finalNorm),
      ...(input.finalHiddenReadbackSkipped ? { finalHiddenReadbackSkipped: true } : {}),
    },
  });
  return {
    logits: projected.values,
    logitTokenIds: projected.selectedRowIds,
    backendProof: {
      logitProjection: {
        backend: projected.backend,
        selectedRowIds: projected.selectedRowIds,
        fullRowCount: matrixRowCount(input.outputProjection),
        selectedRowCount: projected.trace.selectedRows,
        purpose,
        trace: projected.trace,
      },
    },
  };
}

function mergeDecodeBackendProof(
  logitProof: UnlockedBrowserDecodeBackendProof | undefined,
  projectionProofs: UnlockedBrowserDecodeProjectionLayerProof[],
  mlpProofs: UnlockedBrowserDecodeMlpProof[],
  oProjectionProofs: UnlockedBrowserDecodeOProjectionProof[],
  residencyProofs: UnlockedBrowserDecodeResidencyProof[] = [],
  residualRmsNormProofs: UnlockedBrowserDecodeResidualRmsNormProof[] = [],
): UnlockedBrowserDecodeBackendProof | undefined {
  const proof: UnlockedBrowserDecodeBackendProof = {
    ...(logitProof ?? {}),
  };
  if (projectionProofs.length > 0) {
    proof.projectionLayers = projectionProofs.map((projectionProof) => ({
      layerIndex: projectionProof.layerIndex,
      qProjection: cloneDecodeProjectionProof(projectionProof.qProjection),
      kProjection: cloneDecodeProjectionProof(projectionProof.kProjection),
      vProjection: cloneDecodeProjectionProof(projectionProof.vProjection),
      oProjection: cloneDecodeProjectionProof(projectionProof.oProjection),
      ...(projectionProof.qPostProjection ? { qPostProjection: projectionProof.qPostProjection } : {}),
      ...(projectionProof.kPostProjection ? { kPostProjection: projectionProof.kPostProjection } : {}),
      ...(projectionProof.qkvNormRopePair ? { qkvNormRopePair: projectionProof.qkvNormRopePair } : {}),
      ...(projectionProof.qkvReadbackCount !== undefined ? { qkvReadbackCount: projectionProof.qkvReadbackCount } : {}),
    }));
  }
  if (oProjectionProofs.length > 0) {
    proof.oProjectionLayers = oProjectionProofs.map((oProjectionProof) => ({
      layerIndex: oProjectionProof.layerIndex,
      backend: oProjectionProof.backend,
      trace: oProjectionProof.trace,
    }));
  }
  if (mlpProofs.length > 0) {
    proof.mlpLayers = mlpProofs.map((mlpProof) => ({
      layerIndex: mlpProof.layerIndex,
      backend: mlpProof.backend,
      trace: mlpProof.trace,
    }));
  }
  if (residualRmsNormProofs.length > 0) {
    proof.residualRmsNormLayers = residualRmsNormProofs.map((residualProof) => ({
      layerIndex: residualProof.layerIndex,
      backend: residualProof.backend,
      trace: residualProof.trace,
    }));
  }
  if (residencyProofs.length > 0) {
    proof.residencyLayers = residencyProofs.map((residencyProof) => ({ ...residencyProof }));
  }
  return proof.logitProjection
    || proof.projectionLayers
    || proof.oProjectionLayers
    || proof.mlpLayers
    || proof.residualRmsNormLayers
    || proof.residencyLayers
    ? proof
    : undefined;
}

function attachDecodePerfToBackendProof(
  proof: UnlockedBrowserDecodeBackendProof | undefined,
  input: {
    requestId: string;
    generatedTokens: number;
    decodeCallCount: number;
    traces: SSAKernelTrace[];
    kvDecodeReused: boolean;
    commandBatchTraces?: WebGpuDecodeCommandBatchTrace[];
  },
): UnlockedBrowserDecodeBackendProof {
  const logitTrace = proof?.logitProjection?.trace;
  const uploadEstimate = estimateDecodeWeightUploads(proof);
  const activationMovement = summarizeDecodeActivationMovement(proof);
  const decodeDispatchCount = estimateDecodeDispatchCount(proof, input.traces);
  const fusedPackedQkvLayerCount = estimateFusedPackedQkvLayerCount(proof);
  const fusedQkvNormRopeKvAppendLayerCount = estimateFusedQkvNormRopeKvAppendLayerCount(proof);
  const fusedOneTokenAttentionLayerCount = estimateFusedOneTokenAttentionLayerCount(input.traces);
  const fusedResidualRmsNormLayerCount = estimateFusedResidualRmsNormLayerCount(proof);
  const decodeSubmitCount = input.commandBatchTraces && input.commandBatchTraces.length > 0
    ? estimateBatchedDecodeSubmitCount(input.commandBatchTraces, proof)
    : estimateDecodeSubmitCount(decodeDispatchCount);
  return {
    ...(proof ?? {}),
    decodePerf: summarizeDecodeHotPath({
      requestId: input.requestId,
      generatedTokens: input.generatedTokens,
      decodeCallCount: input.decodeCallCount,
      ...(proof?.logitProjection?.backend !== undefined ? { logitProjectionBackend: proof.logitProjection.backend } : {}),
      ...(logitTrace?.readbackStrategy !== undefined ? { logitProjectionReadbackStrategy: logitTrace.readbackStrategy } : {}),
      ...(logitTrace?.readbackRows !== undefined ? { logitProjectionReadbackRows: logitTrace.readbackRows } : {}),
      ...(logitTrace?.readbackBytes !== undefined ? { logitProjectionReadbackBytes: logitTrace.readbackBytes } : {}),
      logitProjectionDispatchCount: decodeDispatchCount,
      decodeSubmitCount,
      cpuFallbackUsed: decodeProofUsesCpuFallback(proof, input.traces),
      weightUploadBytesDuringDecode: uploadEstimate.bytes,
      weightUploadCountDuringDecode: uploadEstimate.count,
      activationUploadBytesDuringDecode: activationMovement.activationUploadBytes,
      activationUploadCountDuringDecode: activationMovement.activationUploadCount,
      hiddenReadbackCountDuringDecode: activationMovement.hiddenReadbackCount,
      f32ExpansionCountDuringDecode: 0,
      f32ExpansionBytesDuringDecode: 0,
      prefillExecutionsDuringDecode: 0,
      residentDecodeLayerCount: activationMovement.residentLayerCount,
      totalDecodeLayerCount: activationMovement.totalLayerCount,
      residentFinalHiddenUsedForLogits: activationMovement.finalHiddenUsedForLogits,
      kvDecodeReused: input.kvDecodeReused,
      fusedPackedQkvLayerCount,
      fusedQkvNormRopeKvAppendLayerCount,
      fusedOneTokenAttentionLayerCount,
      fusedResidualRmsNormLayerCount,
      fusedMlpLayerCount: 0,
      fusedFullLayerCount: 0,
    }),
  };
}

function estimateDecodeDispatchCount(
  proof: UnlockedBrowserDecodeBackendProof | undefined,
  traces: SSAKernelTrace[],
): number {
  const logitDispatches = proof?.logitProjection?.trace.dispatchCount ?? 0;
  const projectionDispatches = estimateDecodeProjectionDispatchCount(proof);
  const mlpDispatches = estimateDecodeMlpDispatchCount(proof);
  const attentionDispatches = traces.reduce((sum, trace) => {
    const dispatchCount = (trace as typeof trace & { attentionDispatchCount?: number }).attentionDispatchCount;
    return sum + (Number.isFinite(dispatchCount) ? Math.max(0, Math.floor(dispatchCount ?? 0)) : 0);
  }, 0);
  return logitDispatches + projectionDispatches + mlpDispatches + attentionDispatches;
}

function estimateDecodeSubmitCount(dispatchCount: number): number {
  return Math.max(0, Math.floor(dispatchCount));
}

function estimateBatchedDecodeSubmitCount(
  commandBatchTraces: WebGpuDecodeCommandBatchTrace[],
  proof: UnlockedBrowserDecodeBackendProof | undefined,
): number {
  const layerSubmits = commandBatchTraces.reduce((sum, trace) => sum + Math.max(0, Math.floor(trace.submitCount ?? 0)), 0);
  const logitSubmit = (proof?.logitProjection?.trace.dispatchCount ?? 0) > 0 ? 1 : 0;
  return layerSubmits + logitSubmit;
}

function estimateDecodeProjectionDispatchCount(proof: UnlockedBrowserDecodeBackendProof | undefined): number {
  const projectionLayers = proof?.projectionLayers ?? [];
  if (projectionLayers.length > 0) {
    return projectionLayers.reduce((sum, layer) => {
      const qkvDispatches = isPackedQkvProjectionTrace(layer.qProjection.trace)
        ? estimateDenseBackendDispatch(layer.qProjection.trace)
        : estimateDenseBackendDispatch(layer.qProjection.trace)
          + estimateDenseBackendDispatch(layer.kProjection.trace)
          + estimateDenseBackendDispatch(layer.vProjection.trace);
      return sum + qkvDispatches + estimateDenseBackendDispatch(layer.oProjection.trace);
    }, 0);
  }
  return (proof?.oProjectionLayers ?? []).reduce((sum, layer) => sum + estimateDenseBackendDispatch(layer.trace), 0);
}

function isPackedQkvProjectionTrace(trace: WebGpuDenseMatVecTrace | WebGpuDenseMatMulTrace): boolean {
  const metadata = trace.metadata as Record<string, unknown> | undefined;
  return metadata?.fusedStage === "packed_qkv_projection" || metadata?.packedQkvProjection === true;
}

function estimateFusedPackedQkvLayerCount(proof: UnlockedBrowserDecodeBackendProof | undefined): number {
  return (proof?.projectionLayers ?? []).reduce((sum, layer) => {
    const metadata = layer.qProjection.trace.metadata as Record<string, unknown> | undefined;
    return sum + (metadata?.fusedStage === "packed_qkv_projection" || metadata?.packedQkvProjection === true ? 1 : 0);
  }, 0);
}

function estimateFusedQkvNormRopeKvAppendLayerCount(proof: UnlockedBrowserDecodeBackendProof | undefined): number {
  return (proof?.projectionLayers ?? []).reduce((sum, layer) => {
    const metadata = layer.qkvNormRopePair?.trace.metadata as Record<string, unknown> | undefined;
    return sum + (
      metadata?.fusedStage === "qkv_norm_rope_kv_append"
      || metadata?.qkvNormRopePair === true
        ? 1
        : 0
    );
  }, 0);
}

function estimateFusedOneTokenAttentionLayerCount(traces: SSAKernelTrace[]): number {
  return traces.reduce((sum, trace) => {
    const attentionTrace = trace as typeof trace & {
      attentionFusedStage?: string;
      queryTokenCount?: number;
      attentionPipelineCacheKey?: string;
      attentionBackend?: "webgpu" | "cpu_reference" | "mixed";
    };
    if (attentionTrace.attentionFusedStage === "one_token_attention") return sum + 1;
    if (
      attentionTrace.queryTokenCount === 1
      && attentionTrace.attentionBackend === "webgpu"
      && attentionTrace.attentionPipelineCacheKey === "packed-sparse-attention:decode_scores+decode_output"
    ) {
      return sum + 1;
    }
    return sum;
  }, 0);
}

function estimateFusedResidualRmsNormLayerCount(proof: UnlockedBrowserDecodeBackendProof | undefined): number {
  return (proof?.residualRmsNormLayers ?? []).reduce((sum, layer) => {
    const metadata = layer.trace.metadata as Record<string, unknown> | undefined;
    return sum + (metadata?.fusedStage === "residual_rmsnorm" || metadata?.residualRmsNormPair === true ? 1 : 0);
  }, 0);
}

function estimateDecodeMlpDispatchCount(proof: UnlockedBrowserDecodeBackendProof | undefined): number {
  return (proof?.mlpLayers ?? []).reduce((sum, layer) => (
    sum + (layer.backend === "webgpu" ? 2 : 0)
  ), 0);
}

function estimateDenseBackendDispatch(trace: WebGpuDenseMatVecTrace | WebGpuDenseMatMulTrace): number {
  if (trace.backend !== "webgpu") return 0;
  const dispatchCount = (trace as typeof trace & { dispatchCount?: number }).dispatchCount;
  return Number.isFinite(dispatchCount) ? Math.max(0, Math.floor(dispatchCount ?? 0)) : 1;
}

function decodeProofUsesCpuFallback(
  proof: UnlockedBrowserDecodeBackendProof | undefined,
  traces: SSAKernelTrace[],
): boolean {
  if (!proof) return false;
  if (proof.logitProjection?.backend === "cpu_reference") return true;
  if ((proof.mlpLayers ?? []).some((layer) => layer.backend === "cpu_reference")) return true;
  if ((proof.oProjectionLayers ?? []).some((layer) => layer.backend === "cpu_reference")) return true;
  if ((proof.projectionLayers ?? []).some((layer) => (
    layer.qProjection.backend === "cpu_reference"
    || layer.kProjection.backend === "cpu_reference"
    || layer.vProjection.backend === "cpu_reference"
    || layer.oProjection.backend === "cpu_reference"
  ))) return true;
  return traces.some((trace) => {
    const attentionTrace = trace as typeof trace & {
      attentionBackend?: "webgpu" | "cpu_reference" | "mixed";
      packedHeadBackends?: Array<"webgpu" | "cpu_reference">;
    };
    return attentionTrace.attentionBackend === "cpu_reference"
      || attentionTrace.attentionBackend === "mixed"
      || (attentionTrace.packedHeadBackends ?? []).includes("cpu_reference");
  });
}

function estimateDecodeWeightUploads(proof: UnlockedBrowserDecodeBackendProof | undefined): { bytes: number; count: number } {
  if (!proof) return { bytes: 0, count: 0 };
  const estimates: number[] = [];
  if (proof.logitProjection?.trace) estimates.push(estimateDenseTraceUploadBytes(proof.logitProjection.trace));
  for (const layer of proof.projectionLayers ?? []) {
    estimates.push(
      estimateDenseTraceUploadBytes(layer.qProjection.trace),
      estimateDenseTraceUploadBytes(layer.kProjection.trace),
      estimateDenseTraceUploadBytes(layer.vProjection.trace),
      estimateDenseTraceUploadBytes(layer.oProjection.trace),
    );
  }
  for (const layer of proof.oProjectionLayers ?? []) estimates.push(estimateDenseTraceUploadBytes(layer.trace));
  for (const layer of proof.mlpLayers ?? []) estimates.push(estimateMlpTraceUploadBytes(layer.trace));
  const positive = estimates.filter((bytes) => bytes > 0);
  return {
    bytes: positive.reduce((sum, bytes) => sum + bytes, 0),
    count: positive.length,
  };
}

function summarizeDecodeActivationMovement(proof: UnlockedBrowserDecodeBackendProof | undefined): {
  activationUploadBytes: number;
  activationUploadCount: number;
  hiddenReadbackCount: number;
  residentLayerCount: number;
  totalLayerCount: number;
  finalHiddenUsedForLogits: boolean;
} {
  const residencyLayers = proof?.residencyLayers ?? [];
  const qkvReadbackCount = (proof?.projectionLayers ?? [])
    .reduce((sum, layer) => sum + Math.max(0, Math.floor(layer.qkvReadbackCount ?? 0)), 0);
  if (residencyLayers.length === 0) {
    return {
      activationUploadBytes: 0,
      activationUploadCount: 0,
      hiddenReadbackCount: qkvReadbackCount,
      residentLayerCount: 0,
      totalLayerCount: 0,
      finalHiddenUsedForLogits: proof?.logitProjection?.trace.metadata?.hiddenResident === true,
    };
  }
  const logitProjectionUsedResidentHidden = proof?.logitProjection?.trace.metadata?.hiddenResident === true;
  return {
    activationUploadBytes: residencyLayers.reduce((sum, layer) => sum + layer.activationUploadBytes, 0),
    activationUploadCount: residencyLayers.reduce((sum, layer) => sum + layer.activationUploadCount, 0),
    hiddenReadbackCount: qkvReadbackCount + residencyLayers.reduce((sum, layer) => sum + layer.hiddenReadbackCount, 0),
    residentLayerCount: residencyLayers.filter((layer) => layer.residentLayerPath).length,
    totalLayerCount: residencyLayers.length,
    finalHiddenUsedForLogits: logitProjectionUsedResidentHidden
      && (
        residencyLayers.some((layer) => layer.finalHiddenUsedForLogits)
        || residencyLayers.every((layer) => layer.residentLayerPath)
      ),
  };
}

function estimateDenseTraceUploadBytes(
  trace: WebGpuDenseMatVecTrace | WebGpuDenseMatMulTrace,
): number {
  if (trace.backend !== "webgpu" || !trace.projectionCacheKey || trace.projectionCacheHit !== false) return 0;
  if (isDenseMatMulTraceLike(trace)) return trace.outputSize * trace.hidden * Float32Array.BYTES_PER_ELEMENT;
  return trace.rows * trace.cols * Float32Array.BYTES_PER_ELEMENT;
}

function estimateMlpTraceUploadBytes(trace: WebGpuMlpTrace): number {
  if (trace.backend !== "webgpu" || !trace.projectionCacheHits) return 0;
  let bytes = 0;
  if (trace.projectionCacheHits.upProjection === false) bytes += trace.inputSize * trace.intermediateSize * Float32Array.BYTES_PER_ELEMENT;
  if (trace.projectionCacheHits.downProjection === false) bytes += trace.intermediateSize * trace.outputSize * Float32Array.BYTES_PER_ELEMENT;
  if (trace.projectionCacheHits.gateProjection === false) bytes += trace.inputSize * trace.intermediateSize * Float32Array.BYTES_PER_ELEMENT;
  return bytes;
}

function isDenseMatMulTraceLike(trace: WebGpuDenseMatVecTrace | WebGpuDenseMatMulTrace): trace is WebGpuDenseMatMulTrace {
  return "outputSize" in trace && "hidden" in trace;
}

function createDecodeProjectionLayerProof(
  layerIndex: number,
  qkvProofs: ProjectedAttentionTensors["proofs"],
  oProjectionProof: UnlockedBrowserDecodeOProjectionProof,
): UnlockedBrowserDecodeProjectionLayerProof {
  return {
    layerIndex,
    qProjection: {
      backend: qkvProofs.qProjection.backend,
      trace: qkvProofs.qProjection,
    },
    kProjection: {
      backend: qkvProofs.kProjection.backend,
      trace: qkvProofs.kProjection,
    },
    vProjection: {
      backend: qkvProofs.vProjection.backend,
      trace: qkvProofs.vProjection,
    },
    oProjection: {
      backend: oProjectionProof.backend,
      trace: oProjectionProof.trace,
    },
    ...(qkvProofs.qPostProjection ? { qPostProjection: toPrefillResidentTensorProof(qkvProofs.qPostProjection) } : {}),
    ...(qkvProofs.kPostProjection ? { kPostProjection: toPrefillResidentTensorProof(qkvProofs.kPostProjection) } : {}),
    ...(qkvProofs.qkvNormRopePair
      ? {
          qkvNormRopePair: {
            backend: qkvProofs.qkvNormRopePair.backend,
            trace: qkvProofs.qkvNormRopePair,
          },
        }
      : {}),
    ...(qkvProofs.qkvReadbackCount !== undefined ? { qkvReadbackCount: qkvProofs.qkvReadbackCount } : {}),
  };
}

function cloneDecodeProjectionProof(
  proof: UnlockedBrowserDecodeProjectionProof,
): UnlockedBrowserDecodeProjectionProof {
  return {
    backend: proof.backend,
    trace: proof.trace,
  };
}

function normalizeLogitCandidateTokenIds(value: number[] | undefined, vocabSize: number): number[] | null {
  if (!value || value.length === 0) return null;
  const seen = new Set<number>();
  const normalized: number[] = [];
  for (const tokenId of value) {
    if (!Number.isInteger(tokenId)) continue;
    const normalizedTokenId = positiveModulo(tokenId, vocabSize);
    if (seen.has(normalizedTokenId)) continue;
    seen.add(normalizedTokenId);
    normalized.push(normalizedTokenId);
  }
  return normalized.length > 0 ? normalized : null;
}

function normalizeSuppressedTokenIds(value: number[] | undefined, vocabSize: number): number[] {
  if (!value || value.length === 0) return [];
  const seen = new Set<number>();
  const normalized: number[] = [];
  for (const tokenId of value) {
    if (!Number.isInteger(tokenId)) continue;
    const normalizedTokenId = positiveModulo(tokenId, vocabSize);
    if (seen.has(normalizedTokenId)) continue;
    seen.add(normalizedTokenId);
    normalized.push(normalizedTokenId);
  }
  return normalized;
}

function normalizeLogitTopK(value: number | undefined, vocabSize: number): number {
  if (value === undefined) return 1;
  if (!Number.isInteger(value) || value <= 0) throw new Error("UnlockedBrowserTransformerBackend logitTopK must be a positive integer.");
  return Math.min(value, vocabSize);
}

function normalizeLogitTileRows(value: number | undefined, vocabSize: number): number | null {
  if (value === undefined) return null;
  if (!Number.isInteger(value) || value <= 0) throw new Error("UnlockedBrowserTransformerBackend logitTileRows must be a positive integer.");
  return Math.min(value, vocabSize);
}

function normalizeDecodeSamplingInput(input: SSADecodeInput): DecodeSamplingInput | undefined {
  const temperature = normalizeOptionalNonNegative(input.samplingTemperature);
  const topP = normalizeOptionalProbability(input.samplingTopP);
  const repetitionPenalty = normalizeOptionalPositive(input.repetitionPenalty);
  const recentTokenIds = input.recentTokenIds?.filter((tokenId) => Number.isInteger(tokenId));
  const seed = Number.isInteger(input.samplingSeed) ? input.samplingSeed : undefined;
  const hasSamplingSignal = temperature !== undefined
    || topP !== undefined
    || repetitionPenalty !== undefined
    || (recentTokenIds?.length ?? 0) > 0
    || seed !== undefined;
  if (!hasSamplingSignal) return undefined;
  return {
    ...(temperature !== undefined ? { temperature } : {}),
    ...(topP !== undefined ? { topP } : {}),
    ...(repetitionPenalty !== undefined ? { repetitionPenalty } : {}),
    ...(recentTokenIds && recentTokenIds.length > 0 ? { recentTokenIds } : {}),
    ...(seed !== undefined ? { seed } : {}),
  };
}

function shouldUseCompactTopKSampling(topK: number, sampling: DecodeSamplingInput | undefined): boolean {
  return topK > 1 && sampling !== undefined;
}

function resolveDecodeSampling(
  sampling: DecodeSamplingInput | undefined,
): Required<Pick<DecodeSamplingInput, "temperature" | "topP" | "repetitionPenalty">> & DecodeSamplingInput {
  return {
    ...(sampling ?? {}),
    temperature: sampling?.temperature ?? 0.7,
    topP: sampling?.topP ?? 0.9,
    repetitionPenalty: sampling?.repetitionPenalty ?? 1.05,
  };
}

function normalizeOptionalNonNegative(value: number | undefined): number | undefined {
  if (value === undefined) return undefined;
  if (!Number.isFinite(value) || value < 0) throw new Error("UnlockedBrowserTransformerBackend samplingTemperature must be a non-negative number.");
  return value;
}

function normalizeOptionalPositive(value: number | undefined): number | undefined {
  if (value === undefined) return undefined;
  if (!Number.isFinite(value) || value <= 0) throw new Error("UnlockedBrowserTransformerBackend repetitionPenalty must be a positive number.");
  return value;
}

function normalizeOptionalProbability(value: number | undefined): number | undefined {
  if (value === undefined) return undefined;
  if (!Number.isFinite(value) || value <= 0 || value > 1) throw new Error("UnlockedBrowserTransformerBackend samplingTopP must be in (0, 1].");
  return value;
}

function prepareAttentionInput(hidden: Matrix, layer: UnlockedBrowserTransformerLayerWeights, rmsNormEps?: number): Matrix {
  return layer.inputLayerNorm
    ? hidden.map((row) => applyRmsNorm(row, layer.inputLayerNorm as RuntimeVector, rmsNormEps))
    : hidden;
}

async function applyAttentionOutput(
  hidden: Matrix,
  attention: Matrix,
  layer: UnlockedBrowserTransformerLayerWeights,
  rmsNormEps?: number,
  kernel?: ProjectionKernelContext,
): Promise<{ hidden: Matrix; oProjection: WebGpuDenseMatMulTrace; mlp?: UnlockedBrowserPrefillMlpProof }> {
  const projected = await runDenseMatMulWebGpu({
    activations: attention,
    projection: layer.oProj,
    ...(kernel?.options.backendPreference ? { backendPreference: kernel.options.backendPreference } : {}),
    ...(kernel?.options.device ? { device: kernel.options.device } : {}),
    ...(kernel?.options.gpu ? { gpu: kernel.options.gpu } : {}),
    ...(kernel?.options.requireWebGpu ? { requireWebGpu: true } : {}),
    ...(kernel?.bufferCache && isPackedRuntimeMatrix(layer.oProj)
      ? {
          bufferCache: kernel.bufferCache,
          projectionCacheKey: projectionCacheKey(kernel.layerIndex, "oProj"),
          projectionCachePolicy: "stable" as const,
        }
      : {}),
    traceMetadata: {
      runtime: "unlocked-browser-transformer",
      ...(kernel ? { requestId: kernel.requestId, layerIndex: kernel.layerIndex, phase: kernel.phase } : {}),
      purpose: `${kernel?.phase ?? "prefill"}_o_projection`,
    },
  });
  const projectedAttention = projected.output;
  if (!usesQwenResidualBlock(layer)) {
    if (hasMlp(layer) && kernel) {
      const mlp = await runPrefillMlp(projectedAttention, layer, kernel.options, kernel.bufferCache, kernel.requestId, kernel.layerIndex);
      return { hidden: mlp.hidden, oProjection: projected.trace, mlp: mlp.proof };
    }
    return {
      hidden: hasMlp(layer) ? projectedAttention.map((row) => runMlp(row, layer)) : projectedAttention,
      oProjection: projected.trace,
    };
  }
  const afterAttention = addMatrices(hidden, projectedAttention);
  if (!hasMlp(layer)) return { hidden: afterAttention, oProjection: projected.trace };
  const mlpInput = layer.postAttentionLayerNorm
    ? afterAttention.map((row) => applyRmsNorm(row, layer.postAttentionLayerNorm as RuntimeVector, rmsNormEps))
    : afterAttention;
  if (!kernel) {
    const mlpOutput = mlpInput.map((row) => runMlp(row, layer));
    return { hidden: addMatrices(afterAttention, mlpOutput), oProjection: projected.trace };
  }
  const mlp = await runPrefillMlp(mlpInput, layer, kernel.options, kernel.bufferCache, kernel.requestId, kernel.layerIndex);
  return { hidden: addMatrices(afterAttention, mlp.hidden), oProjection: projected.trace, mlp: mlp.proof };
}

async function runPrefillMlp(
  hiddenRows: Matrix,
  layer: UnlockedBrowserTransformerLayerWithMlp,
  options: Pick<WebGpuSsaBackendOptions, "backendPreference" | "device" | "gpu" | "requireWebGpu">,
  bufferCache: WebGpuRuntimeBufferCache,
  requestId: string,
  layerIndex: number,
): Promise<{ hidden: Matrix; proof: UnlockedBrowserPrefillMlpProof }> {
  const hidden: Matrix = [];
  const result = await runMlpBatchWebGpu({
    hidden: hiddenRows,
    upProjection: layer.mlpUpProj,
    downProjection: layer.mlpDownProj,
    ...(layer.mlpGateProj ? { gateProjection: layer.mlpGateProj } : {}),
    ...(options.backendPreference ? { backendPreference: options.backendPreference } : {}),
    ...(options.device ? { device: options.device } : {}),
    ...(options.gpu ? { gpu: options.gpu } : {}),
    ...(options.requireWebGpu ? { requireWebGpu: true } : {}),
    ...(hasPackedRuntimeProjection(layer.mlpUpProj, layer.mlpDownProj, layer.mlpGateProj)
      ? {
          bufferCache,
          projectionCacheKey: projectionCacheKey(layerIndex, "mlp"),
          projectionCachePolicy: "stable" as const,
        }
      : {}),
    traceMetadata: {
      runtime: "unlocked-browser-transformer",
      requestId,
      layerIndex,
      phase: "prefill",
      purpose: "prefill_mlp_batch",
    },
  });
  hidden.push(...result.output);
  return {
    hidden,
    proof: {
      backend: result.backend,
      rowCount: hiddenRows.length,
      lastTrace: result.trace,
    },
  };
}

async function applyAttentionOutputForDecode(input: {
  hidden: Vector | WebGpuResidentTensor;
  attention: Vector | WebGpuResidentTensor;
  layer: UnlockedBrowserTransformerLayerWeights;
  rmsNormEps: number | undefined;
  options: Pick<WebGpuSsaBackendOptions, "backendPreference" | "device" | "gpu" | "requireWebGpu">;
  bufferCache: WebGpuRuntimeBufferCache;
  requestId: string;
  layerIndex: number;
  finalLogits?: FinalResidentDecodeLogitsInput;
  materializeHidden?: boolean;
  commandBatch?: WebGpuDecodeCommandBatch;
}): Promise<{
  hidden?: Vector;
  hiddenResident?: WebGpuResidentTensor;
  oProjectionProof: UnlockedBrowserDecodeOProjectionProof;
  mlpProof?: UnlockedBrowserDecodeMlpProof;
  residualRmsNormProof?: UnlockedBrowserDecodeResidualRmsNormProof;
  finalLogits?: DecodeLogitProjectionResult;
  residencyProof?: UnlockedBrowserDecodeResidencyProof;
}> {
  if (input.options.backendPreference !== "cpu") {
    const resident = await tryApplyAttentionOutputForDecodeResident(input);
    if (resident) return resident;
  }
  const hiddenVector = isResidentTensor(input.hidden)
    ? firstResidentRow(await readWebGpuResidentTensor(input.hidden))
    : input.hidden;
  const attentionVector = isResidentTensor(input.attention)
    ? firstResidentRow(await readWebGpuResidentTensor(input.attention))
    : input.attention;
  if (isResidentTensor(input.attention)) destroyWebGpuResidentTensor(input.attention);
  const fallbackResidencyProof = createDecodeResidencyProof({
    layerIndex: input.layerIndex,
    residentLayerPath: false,
    activationUploadBytes: 0,
    activationUploadCount: 0,
    hiddenReadbackCount: (isResidentTensor(input.hidden) ? 1 : 0) + (isResidentTensor(input.attention) ? 1 : 0),
    finalHiddenUsedForLogits: false,
  });
  const projected = await runDenseMatVecWebGpu({
    vector: attentionVector,
    matrix: input.layer.oProj,
    ...(input.options.backendPreference ? { backendPreference: input.options.backendPreference } : {}),
    ...(input.options.device ? { device: input.options.device } : {}),
    ...(input.options.gpu ? { gpu: input.options.gpu } : {}),
    ...(input.options.requireWebGpu ? { requireWebGpu: true } : {}),
    ...(isPackedRuntimeMatrix(input.layer.oProj)
      ? {
          bufferCache: input.bufferCache,
          projectionCacheKey: projectionCacheKey(input.layerIndex, "oProj"),
          projectionCachePolicy: "stable" as const,
        }
      : {}),
    traceMetadata: {
      runtime: "unlocked-browser-transformer",
      requestId: input.requestId,
      layerIndex: input.layerIndex,
      phase: "decode",
      purpose: "decode_o_projection",
    },
  });
  const oProjectionProof = {
    layerIndex: input.layerIndex,
    backend: projected.backend,
    trace: projected.trace,
  };
  const projectedAttention = projected.values;
  if (!usesQwenResidualBlock(input.layer)) {
    if (!hasMlp(input.layer)) return { hidden: projectedAttention, oProjectionProof, residencyProof: fallbackResidencyProof };
    const mlp = await runDecodeMlp(projectedAttention, input.layer, input.options, input.bufferCache, input.requestId, input.layerIndex);
    return { hidden: mlp.hidden, oProjectionProof, mlpProof: mlp.mlpProof, residencyProof: fallbackResidencyProof };
  }
  const afterAttention = addVectors(hiddenVector, projectedAttention);
  if (!hasMlp(input.layer)) return { hidden: afterAttention, oProjectionProof, residencyProof: fallbackResidencyProof };
  const mlpInput = input.layer.postAttentionLayerNorm
    ? applyRmsNorm(afterAttention, input.layer.postAttentionLayerNorm as RuntimeVector, input.rmsNormEps)
    : afterAttention;
  const mlp = await runDecodeMlp(mlpInput, input.layer, input.options, input.bufferCache, input.requestId, input.layerIndex);
  return {
    hidden: addVectors(afterAttention, mlp.hidden),
    oProjectionProof,
    mlpProof: mlp.mlpProof,
    residencyProof: fallbackResidencyProof,
  };
}

async function tryApplyAttentionOutputForDecodeResident(input: {
  hidden: Vector | WebGpuResidentTensor;
  attention: Vector | WebGpuResidentTensor;
  layer: UnlockedBrowserTransformerLayerWeights;
  rmsNormEps: number | undefined;
  options: Pick<WebGpuSsaBackendOptions, "backendPreference" | "device" | "gpu" | "requireWebGpu">;
  bufferCache: WebGpuRuntimeBufferCache;
  requestId: string;
  layerIndex: number;
  finalLogits?: FinalResidentDecodeLogitsInput;
  materializeHidden?: boolean;
  commandBatch?: WebGpuDecodeCommandBatch;
}): Promise<{
  hidden?: Vector;
  hiddenResident?: WebGpuResidentTensor;
  oProjectionProof: UnlockedBrowserDecodeOProjectionProof;
  mlpProof?: UnlockedBrowserDecodeMlpProof;
  residualRmsNormProof?: UnlockedBrowserDecodeResidualRmsNormProof;
  finalLogits?: DecodeLogitProjectionResult;
  residencyProof?: UnlockedBrowserDecodeResidencyProof;
} | null> {
  const residentTensors: Array<{ tensor: Parameters<typeof destroyWebGpuResidentTensor>[0] }> = [];
  const hiddenResidentInput = isResidentTensor(input.hidden) ? input.hidden : undefined;
  const hiddenVectorInput = hiddenResidentInput ? undefined : input.hidden as Vector;
  const activationUploadBytes = (
    (hiddenVectorInput ? hiddenVectorInput.length : 0)
    + (isResidentTensor(input.attention) ? 0 : input.attention.length)
  ) * Float32Array.BYTES_PER_ELEMENT;
  const activationUploadCount = (hiddenResidentInput ? 0 : 1) + (isResidentTensor(input.attention) ? 0 : 1);
  const hiddenReadbackCount = input.materializeHidden === false ? 0 : 1;
  const remember = <T extends { tensor: Parameters<typeof destroyWebGpuResidentTensor>[0] }>(result: T): T => {
    residentTensors.push({ tensor: result.tensor });
    return result;
  };
  const retain = <T extends Parameters<typeof destroyWebGpuResidentTensor>[0]>(tensor: T): T => {
    const index = residentTensors.findIndex((entry) => entry.tensor === tensor);
    if (index >= 0) residentTensors.splice(index, 1);
    return tensor;
  };
  const buildResidencyProof = (inputFinalHiddenUsedForLogits: boolean): UnlockedBrowserDecodeResidencyProof => createDecodeResidencyProof({
    layerIndex: input.layerIndex,
    residentLayerPath: true,
    activationUploadBytes,
    activationUploadCount,
    hiddenReadbackCount,
    finalHiddenUsedForLogits: inputFinalHiddenUsedForLogits,
  });
  const projectFinalLogits = async (hidden: WebGpuResidentTensor): Promise<DecodeLogitProjectionResult | undefined> => {
    if (!input.finalLogits) return undefined;
    return await projectDecodeLogitsFromResidentHidden({
      hidden,
      outputProjection: input.finalLogits.outputProjection,
      ...(input.finalLogits.finalNorm ? { finalNorm: input.finalLogits.finalNorm } : {}),
      rmsNormEps: input.rmsNormEps,
      topK: input.finalLogits.topK,
      tileRows: input.finalLogits.tileRows,
      suppressedTokenIds: input.finalLogits.suppressedTokenIds,
      ...(input.finalLogits.sampling ? { sampling: input.finalLogits.sampling } : {}),
      options: input.options,
      bufferCache: input.bufferCache,
      requestId: input.requestId,
      layerIndex: input.layerIndex,
      rememberResidentTensor: remember,
      ...(input.materializeHidden === false ? { finalHiddenReadbackSkipped: true } : {}),
    });
  };
  try {
    const hiddenResident = hiddenResidentInput
      ? { tensor: hiddenResidentInput }
      : remember(await uploadWebGpuResidentTensor({
          matrix: [hiddenVectorInput ?? []],
          ...(input.options.device ? { device: input.options.device } : {}),
          ...(input.options.gpu ? { gpu: input.options.gpu } : {}),
          ...(input.options.requireWebGpu ? { requireWebGpu: true } : {}),
          traceMetadata: {
            runtime: "unlocked-browser-transformer",
            requestId: input.requestId,
            layerIndex: input.layerIndex,
            phase: "decode",
            purpose: "decode_hidden_resident_upload",
            residentDecodeLayerPath: true,
          },
        }));
    const attentionResident = isResidentTensor(input.attention)
      ? remember({ tensor: input.attention })
      : remember(await uploadWebGpuResidentTensor({
          matrix: [input.attention],
          ...(input.options.device ? { device: input.options.device } : {}),
          ...(input.options.gpu ? { gpu: input.options.gpu } : {}),
          ...(input.options.requireWebGpu ? { requireWebGpu: true } : {}),
          traceMetadata: {
            runtime: "unlocked-browser-transformer",
            requestId: input.requestId,
            layerIndex: input.layerIndex,
            phase: "decode",
            purpose: "decode_attention_resident_upload",
            residentDecodeLayerPath: true,
          },
        }));
    const projected = remember(await runDenseMatMulResidentWebGpu({
      activations: attentionResident.tensor,
      projection: input.layer.oProj,
      ...(input.options.device ? { device: input.options.device } : {}),
      ...(input.options.gpu ? { gpu: input.options.gpu } : {}),
      ...(input.options.requireWebGpu ? { requireWebGpu: true } : {}),
      ...(isPackedRuntimeMatrix(input.layer.oProj)
        ? {
            bufferCache: input.bufferCache,
            projectionCacheKey: projectionCacheKey(input.layerIndex, "oProj"),
            projectionCachePolicy: "stable" as const,
          }
        : {}),
      traceMetadata: {
        runtime: "unlocked-browser-transformer",
        requestId: input.requestId,
        layerIndex: input.layerIndex,
        phase: "decode",
        purpose: "decode_o_projection",
        residentDecodeLayerPath: true,
      },
      ...(input.commandBatch ? { commandBatch: input.commandBatch } : {}),
    }));
    const oProjectionProof = {
      layerIndex: input.layerIndex,
      backend: projected.backend,
      trace: projected.trace,
    };

    if (!usesQwenResidualBlock(input.layer)) {
      if (!hasMlp(input.layer)) {
        const finalLogits = await projectFinalLogits(projected.tensor);
        const residencyProof = buildResidencyProof(Boolean(finalLogits && input.materializeHidden === false));
        return {
          ...(input.materializeHidden === false
            ? { hiddenResident: retain(projected.tensor) }
            : { hidden: firstResidentRow(await readWebGpuResidentTensor(projected.tensor)) }),
          oProjectionProof,
          ...(finalLogits ? { finalLogits } : {}),
          residencyProof,
        };
      }
      const mlp = remember(await runMlpBatchResidentWebGpu({
        hidden: projected.tensor,
        upProjection: input.layer.mlpUpProj,
        downProjection: input.layer.mlpDownProj,
        ...(input.layer.mlpGateProj ? { gateProjection: input.layer.mlpGateProj } : {}),
        ...(input.options.device ? { device: input.options.device } : {}),
        ...(input.options.gpu ? { gpu: input.options.gpu } : {}),
        ...(input.options.requireWebGpu ? { requireWebGpu: true } : {}),
        ...(hasPackedRuntimeProjection(input.layer.mlpUpProj, input.layer.mlpDownProj, input.layer.mlpGateProj)
          ? {
              bufferCache: input.bufferCache,
              projectionCacheKey: projectionCacheKey(input.layerIndex, "mlp"),
              projectionCachePolicy: "stable" as const,
            }
          : {}),
        traceMetadata: {
          runtime: "unlocked-browser-transformer",
          requestId: input.requestId,
          layerIndex: input.layerIndex,
          purpose: "decode_mlp",
          residentDecodeLayerPath: true,
        },
        ...(input.commandBatch ? { commandBatch: input.commandBatch } : {}),
      }));
      const finalLogits = await projectFinalLogits(mlp.tensor);
      const residencyProof = buildResidencyProof(Boolean(finalLogits && input.materializeHidden === false));
      return {
        ...(input.materializeHidden === false
          ? { hiddenResident: retain(mlp.tensor) }
          : { hidden: firstResidentRow(await readWebGpuResidentTensor(mlp.tensor)) }),
        oProjectionProof,
        mlpProof: {
          layerIndex: input.layerIndex,
          backend: mlp.backend,
          trace: mlp.trace,
        },
        ...(finalLogits ? { finalLogits } : {}),
        residencyProof,
      };
    }

    let residualRmsNormProof: UnlockedBrowserDecodeResidualRmsNormProof | undefined;
    const canFusePostAttentionResidualNorm = hasMlp(input.layer) && Boolean(input.layer.postAttentionLayerNorm);
    const afterAttention = canFusePostAttentionResidualNorm
      ? undefined
      : remember(await runResidualAddResidentWebGpu({
          left: hiddenResident.tensor,
          right: projected.tensor,
          ...(input.options.device ? { device: input.options.device } : {}),
          ...(input.options.gpu ? { gpu: input.options.gpu } : {}),
          ...(input.options.requireWebGpu ? { requireWebGpu: true } : {}),
          ...(input.bufferCache ? { bufferCache: input.bufferCache } : {}),
          traceMetadata: {
            runtime: "unlocked-browser-transformer",
            requestId: input.requestId,
            layerIndex: input.layerIndex,
            phase: "decode",
            purpose: "decode_attention_residual",
            residentDecodeLayerPath: true,
          },
          ...(input.commandBatch ? { commandBatch: input.commandBatch } : {}),
        }));
    const fusedResidualNorm = canFusePostAttentionResidualNorm
      ? await runResidualRmsNormPairResidentWebGpu({
          left: hiddenResident.tensor,
          right: projected.tensor,
          weight: input.layer.postAttentionLayerNorm as RuntimeVector,
          ...(input.rmsNormEps !== undefined ? { eps: input.rmsNormEps } : {}),
          ...(input.options.device ? { device: input.options.device } : {}),
          ...(input.options.gpu ? { gpu: input.options.gpu } : {}),
          ...(input.options.requireWebGpu ? { requireWebGpu: true } : {}),
          ...(input.bufferCache ? { bufferCache: input.bufferCache } : {}),
          traceMetadata: {
            runtime: "unlocked-browser-transformer",
            requestId: input.requestId,
            layerIndex: input.layerIndex,
            phase: "decode",
            purpose: "decode_attention_residual_rmsnorm",
            residentDecodeLayerPath: true,
          },
          ...(input.commandBatch ? { commandBatch: input.commandBatch } : {}),
        })
      : undefined;
    if (fusedResidualNorm) {
      remember({ tensor: fusedResidualNorm.summed });
      remember({ tensor: fusedResidualNorm.normed });
      residualRmsNormProof = {
        layerIndex: input.layerIndex,
        backend: fusedResidualNorm.backend,
        trace: fusedResidualNorm.trace,
      };
    }
    const afterAttentionTensor = fusedResidualNorm?.summed ?? afterAttention?.tensor;
    if (!afterAttentionTensor) {
      throw new Error("Qwen decode residual path did not produce a resident after-attention tensor.");
    }
    if (!hasMlp(input.layer)) {
      const finalLogits = await projectFinalLogits(afterAttentionTensor);
      const residencyProof = buildResidencyProof(Boolean(finalLogits && input.materializeHidden === false));
      return {
        ...(input.materializeHidden === false
          ? { hiddenResident: retain(afterAttentionTensor) }
          : { hidden: firstResidentRow(await readWebGpuResidentTensor(afterAttentionTensor)) }),
        oProjectionProof,
        ...(residualRmsNormProof ? { residualRmsNormProof } : {}),
        ...(finalLogits ? { finalLogits } : {}),
        residencyProof,
      };
    }
    const mlpInput = input.layer.postAttentionLayerNorm
      ? fusedResidualNorm?.normed ?? remember(await runRmsNormResidentWebGpu({
          hidden: afterAttentionTensor,
          weight: input.layer.postAttentionLayerNorm as RuntimeVector,
          ...(input.rmsNormEps !== undefined ? { eps: input.rmsNormEps } : {}),
          ...(input.options.device ? { device: input.options.device } : {}),
          ...(input.options.gpu ? { gpu: input.options.gpu } : {}),
          ...(input.options.requireWebGpu ? { requireWebGpu: true } : {}),
          ...(input.bufferCache ? { bufferCache: input.bufferCache } : {}),
          traceMetadata: {
            runtime: "unlocked-browser-transformer",
            requestId: input.requestId,
            layerIndex: input.layerIndex,
            phase: "decode",
            purpose: "decode_post_attention_rmsnorm",
            residentDecodeLayerPath: true,
          },
          ...(input.commandBatch ? { commandBatch: input.commandBatch } : {}),
        })).tensor
      : afterAttentionTensor;
    const mlp = remember(await runMlpBatchResidentWebGpu({
      hidden: mlpInput,
      upProjection: input.layer.mlpUpProj,
      downProjection: input.layer.mlpDownProj,
      ...(input.layer.mlpGateProj ? { gateProjection: input.layer.mlpGateProj } : {}),
      ...(input.options.device ? { device: input.options.device } : {}),
      ...(input.options.gpu ? { gpu: input.options.gpu } : {}),
      ...(input.options.requireWebGpu ? { requireWebGpu: true } : {}),
      ...(hasPackedRuntimeProjection(input.layer.mlpUpProj, input.layer.mlpDownProj, input.layer.mlpGateProj)
        ? {
            bufferCache: input.bufferCache,
            projectionCacheKey: projectionCacheKey(input.layerIndex, "mlp"),
            projectionCachePolicy: "stable" as const,
          }
        : {}),
      traceMetadata: {
        runtime: "unlocked-browser-transformer",
        requestId: input.requestId,
        layerIndex: input.layerIndex,
        purpose: "decode_mlp",
        residentDecodeLayerPath: true,
      },
      ...(input.commandBatch ? { commandBatch: input.commandBatch } : {}),
    }));
    const finalHidden = remember(await runResidualAddResidentWebGpu({
      left: afterAttentionTensor,
      right: mlp.tensor,
      ...(input.options.device ? { device: input.options.device } : {}),
      ...(input.options.gpu ? { gpu: input.options.gpu } : {}),
      ...(input.options.requireWebGpu ? { requireWebGpu: true } : {}),
      ...(input.bufferCache ? { bufferCache: input.bufferCache } : {}),
      traceMetadata: {
        runtime: "unlocked-browser-transformer",
        requestId: input.requestId,
        layerIndex: input.layerIndex,
        phase: "decode",
        purpose: "decode_mlp_residual",
        residentDecodeLayerPath: true,
      },
      ...(input.commandBatch ? { commandBatch: input.commandBatch } : {}),
    }));
    const finalLogits = await projectFinalLogits(finalHidden.tensor);
    const residencyProof = buildResidencyProof(Boolean(finalLogits && input.materializeHidden === false));
    return {
      ...(input.materializeHidden === false
        ? { hiddenResident: retain(finalHidden.tensor) }
        : { hidden: firstResidentRow(await readWebGpuResidentTensor(finalHidden.tensor)) }),
      oProjectionProof,
      mlpProof: {
        layerIndex: input.layerIndex,
        backend: mlp.backend,
        trace: mlp.trace,
      },
      ...(residualRmsNormProof ? { residualRmsNormProof } : {}),
      ...(finalLogits ? { finalLogits } : {}),
      residencyProof,
    };
  } catch (error) {
    if (input.options.requireWebGpu) throw error;
    return null;
  } finally {
    const tensorsToDestroy = residentTensors.splice(0).reverse().map(({ tensor }) => tensor);
    if (input.commandBatch) {
      input.commandBatch.deferAfterSubmit(() => {
        for (const tensor of tensorsToDestroy) destroyWebGpuResidentTensor(tensor);
      });
    } else {
      for (const tensor of tensorsToDestroy) destroyWebGpuResidentTensor(tensor);
    }
  }
}

function firstResidentRow(matrix: Matrix): Vector {
  return [...(matrix[0] ?? [])];
}

function createZeroMatrix(rows: number, cols: number): Matrix {
  return Array.from({ length: Math.max(0, rows) }, () => new Array(Math.max(0, cols)).fill(0));
}

async function appendResidentDecodeKvCache(input: {
  layerState: UnlockedBrowserLayerState;
  projected: ProjectedAttentionTensors & {
    resident: NonNullable<ProjectedAttentionTensors["resident"]>;
  };
  options: Pick<WebGpuSsaBackendOptions, "device" | "gpu" | "requireWebGpu">;
  commandBatch?: WebGpuDecodeCommandBatch;
}): Promise<void> {
  const compactK = input.layerState.compactK ?? [];
  const compactV = input.layerState.compactV ?? [];
  if (!input.layerState.residentCompactK) {
    input.layerState.residentCompactK = await createWebGpuResidentRowCache({
      matrix: compactK,
      capacityRows: Math.max(compactK.length + 64, compactK.length * 2),
      ...(input.options.device ? { device: input.options.device } : {}),
      ...(input.options.gpu ? { gpu: input.options.gpu } : {}),
      ...(input.options.requireWebGpu ? { requireWebGpu: true } : {}),
    });
  }
  if (!input.layerState.residentCompactV) {
    input.layerState.residentCompactV = await createWebGpuResidentRowCache({
      matrix: compactV,
      capacityRows: Math.max(compactV.length + 64, compactV.length * 2),
      ...(input.options.device ? { device: input.options.device } : {}),
      ...(input.options.gpu ? { gpu: input.options.gpu } : {}),
      ...(input.options.requireWebGpu ? { requireWebGpu: true } : {}),
    });
  }
  input.layerState.residentCompactK = await appendWebGpuResidentRowCache({
    cache: input.layerState.residentCompactK,
    rows: input.projected.resident.attentionK,
    ...(input.options.device ? { device: input.options.device } : {}),
    ...(input.options.gpu ? { gpu: input.options.gpu } : {}),
    ...(input.options.requireWebGpu ? { requireWebGpu: true } : {}),
    ...(input.commandBatch ? { commandBatch: input.commandBatch } : {}),
  });
  input.layerState.residentCompactV = await appendWebGpuResidentRowCache({
    cache: input.layerState.residentCompactV,
    rows: input.projected.resident.attentionV,
    ...(input.options.device ? { device: input.options.device } : {}),
    ...(input.options.gpu ? { gpu: input.options.gpu } : {}),
    ...(input.options.requireWebGpu ? { requireWebGpu: true } : {}),
    ...(input.commandBatch ? { commandBatch: input.commandBatch } : {}),
  });
  compactK.push([...(input.projected.attentionK[0] ?? [])]);
  compactV.push([...(input.projected.attentionV[0] ?? [])]);
  input.layerState.compactK = compactK;
  input.layerState.compactV = compactV;
}

function createDecodeResidencyProof(input: UnlockedBrowserDecodeResidencyProof): UnlockedBrowserDecodeResidencyProof {
  return {
    layerIndex: input.layerIndex,
    residentLayerPath: input.residentLayerPath,
    activationUploadBytes: Math.max(0, Math.floor(input.activationUploadBytes)),
    activationUploadCount: Math.max(0, Math.floor(input.activationUploadCount)),
    hiddenReadbackCount: Math.max(0, Math.floor(input.hiddenReadbackCount)),
    finalHiddenUsedForLogits: input.finalHiddenUsedForLogits,
  };
}

function isResidentTensor(value: unknown): value is WebGpuResidentTensor {
  return typeof value === "object"
    && value !== null
    && (value as WebGpuResidentTensor).kind === "webgpu_resident_tensor"
    && typeof (value as WebGpuResidentTensor).id === "string";
}

function isWebGpuDeviceLike(value: unknown): value is ConstructorParameters<typeof WebGpuDecodeCommandBatch>[0] {
  return typeof value === "object"
    && value !== null
    && "queue" in value
    && "createCommandEncoder" in value
    && typeof (value as { createCommandEncoder?: unknown }).createCommandEncoder === "function"
    && typeof (value as { queue?: { submit?: unknown } }).queue?.submit === "function";
}

function resolveDecodeCommandBatchDevice(
  configuredDevice: unknown,
  residentHidden: WebGpuResidentTensor | undefined,
): ConstructorParameters<typeof WebGpuDecodeCommandBatch>[0] | undefined {
  if (isWebGpuDeviceLike(configuredDevice)) return configuredDevice;
  if (isWebGpuDeviceLike(residentHidden?.device)) return residentHidden.device;
  return undefined;
}

async function runDecodeMlp(
  hidden: Vector,
  layer: UnlockedBrowserTransformerLayerWithMlp,
  options: Pick<WebGpuSsaBackendOptions, "backendPreference" | "device" | "gpu" | "requireWebGpu">,
  bufferCache: WebGpuRuntimeBufferCache,
  requestId: string,
  layerIndex: number,
): Promise<{ hidden: Vector; mlpProof: UnlockedBrowserDecodeMlpProof }> {
  const result = await runMlpWebGpu({
    hidden,
    upProjection: layer.mlpUpProj,
    downProjection: layer.mlpDownProj,
    ...(layer.mlpGateProj ? { gateProjection: layer.mlpGateProj } : {}),
    ...(options.backendPreference ? { backendPreference: options.backendPreference } : {}),
    ...(options.device ? { device: options.device } : {}),
    ...(options.gpu ? { gpu: options.gpu } : {}),
      ...(options.requireWebGpu ? { requireWebGpu: true } : {}),
    ...(hasPackedRuntimeProjection(layer.mlpUpProj, layer.mlpDownProj, layer.mlpGateProj)
      ? {
          bufferCache,
          projectionCacheKey: projectionCacheKey(layerIndex, "mlp"),
          projectionCachePolicy: "stable" as const,
        }
      : {}),
    traceMetadata: {
      runtime: "unlocked-browser-transformer",
      requestId,
      layerIndex,
      purpose: "decode_mlp",
    },
  });
  return {
    hidden: result.values,
    mlpProof: {
      layerIndex,
      backend: result.backend,
      trace: result.trace,
    },
  };
}

function usesQwenResidualBlock(layer: UnlockedBrowserTransformerLayerWeights): boolean {
  return Boolean(layer.inputLayerNorm || layer.postAttentionLayerNorm || layer.qNorm || layer.kNorm || layer.mlpGateProj);
}

interface AttentionGeometry {
  attentionHeads: number;
  keyValueHeads: number;
  headDim: number;
  qProjectionSize: number;
  kvProjectionSize: number;
  expandedProjectionSize: number;
}

interface ProjectedAttentionTensors {
  q: Matrix;
  k: Matrix;
  v: Matrix;
  attentionK: Matrix;
  attentionV: Matrix;
  resident?: {
    q: WebGpuResidentTensor;
    attentionK: WebGpuResidentTensor;
    attentionV: WebGpuResidentTensor;
  };
  geometry: AttentionGeometry;
  proofs: {
    qProjection: WebGpuDenseMatMulTrace;
    kProjection: WebGpuDenseMatMulTrace;
    vProjection: WebGpuDenseMatMulTrace;
    qPostProjection?: WebGpuResidentTensorTrace;
    kPostProjection?: WebGpuResidentTensorTrace;
    qkvReadbackCount?: number;
    packedQkvProjection?: WebGpuPackedQkvProjectionResidentTrace;
    qkvNormRopePair?: WebGpuQkvNormRopePairResidentTrace;
  };
}

interface ProjectionKernelContext {
  options: Pick<WebGpuSsaBackendOptions, "backendPreference" | "device" | "gpu" | "requireWebGpu">;
  bufferCache: WebGpuRuntimeBufferCache;
  requestId: string;
  layerIndex: number;
  phase: "prefill" | "decode";
  commandBatch?: WebGpuDecodeCommandBatch;
}

async function projectAttentionTensors(input: {
  attentionInput: Matrix | WebGpuResidentTensor;
  layer: UnlockedBrowserTransformerLayerWeights;
  weights: UnlockedBrowserTransformerWeights;
  positions: number[];
} & ProjectionKernelContext): Promise<ProjectedAttentionTensors> {
  const geometry = getAttentionGeometry(input.weights, input.layer);
  if (input.options.backendPreference !== "cpu") {
    const resident = await tryProjectAttentionTensorsResident(input, geometry);
    if (resident) return resident;
  }
  const attentionInput = isResidentTensor(input.attentionInput)
    ? await readWebGpuResidentTensor(input.attentionInput)
    : input.attentionInput;
  const qProjection = await projectWithDenseMatMul(attentionInput, input.layer.qProj, input, "qProj");
  const kProjection = await projectWithDenseMatMul(attentionInput, input.layer.kProj, input, "kProj");
  const vProjection = await projectWithDenseMatMul(attentionInput, input.layer.vProj, input, "vProj");
  const q = applyRotaryPositionEmbedding(
    normalizePackedHeads(qProjection.output, input.layer.qNorm, geometry.headDim, input.weights.rmsNormEps),
    geometry.attentionHeads,
    geometry.headDim,
    input.positions,
    input.weights.ropeTheta,
  );
  const attentionK = applyRotaryPositionEmbedding(
    normalizePackedHeads(kProjection.output, input.layer.kNorm, geometry.headDim, input.weights.rmsNormEps),
    geometry.keyValueHeads,
    geometry.headDim,
    input.positions,
    input.weights.ropeTheta,
  );
  const attentionV = vProjection.output.map((row) => [...row]);
  return {
    q,
    k: expandGroupedKeyValueHeads(attentionK, geometry),
    v: expandGroupedKeyValueHeads(attentionV, geometry),
    attentionK,
    attentionV,
    geometry,
    proofs: {
      qProjection: qProjection.trace,
      kProjection: kProjection.trace,
      vProjection: vProjection.trace,
    },
  };
}

async function tryProjectAttentionTensorsResident(
  input: {
    attentionInput: Matrix | WebGpuResidentTensor;
    layer: UnlockedBrowserTransformerLayerWeights;
    weights: UnlockedBrowserTransformerWeights;
    positions: number[];
  } & ProjectionKernelContext,
  geometry: AttentionGeometry,
): Promise<ProjectedAttentionTensors | null> {
  const residentTensors: WebGpuResidentTensor[] = [];
  const remember = <T extends { tensor: WebGpuResidentTensor }>(result: T): T => {
    residentTensors.push(result.tensor);
    return result;
  };
  const retain = <T extends WebGpuResidentTensor>(tensor: T): T => {
    const index = residentTensors.indexOf(tensor);
    if (index >= 0) residentTensors.splice(index, 1);
    return tensor;
  };

  try {
    const attentionInput = isResidentTensor(input.attentionInput)
      ? { tensor: input.attentionInput }
      : remember(await uploadWebGpuResidentTensor({
          matrix: input.attentionInput,
          ...(input.options.device ? { device: input.options.device } : {}),
          ...(input.options.gpu ? { gpu: input.options.gpu } : {}),
          ...(input.options.requireWebGpu ? { requireWebGpu: true } : {}),
          traceMetadata: {
            runtime: "unlocked-browser-transformer",
            requestId: input.requestId,
            layerIndex: input.layerIndex,
            phase: input.phase,
            residentQkvPostProjection: true,
            purpose: `${input.phase}_attention_input_upload`,
          },
        }));
    const packedQkvProjection = input.phase === "decode" && input.options.requireWebGpu === true
      ? await runPackedQkvProjectionResidentWebGpu({
          hidden: attentionInput.tensor,
          qProjection: input.layer.qProj,
          kProjection: input.layer.kProj,
          vProjection: input.layer.vProj,
          ...(input.options.device ? { device: input.options.device } : {}),
          ...(input.options.gpu ? { gpu: input.options.gpu } : {}),
          requireWebGpu: true,
          bufferCache: input.bufferCache,
          qProjectionCacheKey: projectionCacheKey(input.layerIndex, "qProj"),
          kProjectionCacheKey: projectionCacheKey(input.layerIndex, "kProj"),
          vProjectionCacheKey: projectionCacheKey(input.layerIndex, "vProj"),
          projectionCachePolicy: "stable" as const,
          traceMetadata: {
            runtime: "unlocked-browser-transformer",
            requestId: input.requestId,
            layerIndex: input.layerIndex,
            phase: input.phase,
            residentQkvPostProjection: true,
            purpose: `${input.phase}_packed_qkv_projection`,
          },
          ...(input.commandBatch ? { commandBatch: input.commandBatch } : {}),
        })
      : null;
    if (packedQkvProjection) {
      remember({ tensor: packedQkvProjection.q });
      remember({ tensor: packedQkvProjection.k });
      remember({ tensor: packedQkvProjection.v });
    }
    const qProjection = packedQkvProjection
      ? { tensor: packedQkvProjection.q, trace: packedDenseMatMulTrace(packedQkvProjection.trace, "qProjection") }
      : remember(await projectWithDenseMatMulResident(attentionInput.tensor, input.layer.qProj, input, "qProj"));
    const kProjection = packedQkvProjection
      ? { tensor: packedQkvProjection.k, trace: packedDenseMatMulTrace(packedQkvProjection.trace, "kProjection") }
      : remember(await projectWithDenseMatMulResident(attentionInput.tensor, input.layer.kProj, input, "kProj"));
    const vProjection = packedQkvProjection
      ? { tensor: packedQkvProjection.v, trace: packedDenseMatMulTrace(packedQkvProjection.trace, "vProjection") }
      : remember(await projectWithDenseMatMulResident(attentionInput.tensor, input.layer.vProj, input, "vProj"));
    const qkvNormRopePair = input.phase === "decode" && input.options.requireWebGpu === true
      ? await runQwenQkvNormRopePairResidentWebGpu({
          qProjected: qProjection.tensor,
          kProjected: kProjection.tensor,
          qHeadCount: geometry.attentionHeads,
          kHeadCount: geometry.keyValueHeads,
          headDim: geometry.headDim,
          positions: input.positions,
          ...(input.layer.qNorm ? { qNormWeight: input.layer.qNorm } : {}),
          ...(input.layer.kNorm ? { kNormWeight: input.layer.kNorm } : {}),
          ...(input.weights.ropeTheta ? { ropeTheta: input.weights.ropeTheta } : {}),
          ...(input.weights.rmsNormEps ? { eps: input.weights.rmsNormEps } : {}),
          ...(input.options.device ? { device: input.options.device } : {}),
          ...(input.options.gpu ? { gpu: input.options.gpu } : {}),
          requireWebGpu: true,
          bufferCache: input.bufferCache,
          traceMetadata: {
            runtime: "unlocked-browser-transformer",
            requestId: input.requestId,
            layerIndex: input.layerIndex,
            phase: input.phase,
            residentQkvPostProjection: true,
            purpose: `${input.phase}_qkv_norm_rope_pair`,
          },
          ...(input.commandBatch ? { commandBatch: input.commandBatch } : {}),
        })
      : null;
    if (qkvNormRopePair) {
      remember({ tensor: qkvNormRopePair.q });
      remember({ tensor: qkvNormRopePair.k });
    }
    const qPostProjection = qkvNormRopePair
      ? {
          tensor: qkvNormRopePair.q,
          trace: qkvNormRopePairResidentTrace(qkvNormRopePair.trace, "q_post_projection"),
        }
      : remember(await runQwenQkvPostProjectionResidentWebGpu({
          projected: qProjection.tensor,
          headCount: geometry.attentionHeads,
          headDim: geometry.headDim,
          positions: input.positions,
          ...(input.layer.qNorm ? { normWeight: input.layer.qNorm } : {}),
          ...(input.weights.ropeTheta ? { ropeTheta: input.weights.ropeTheta } : {}),
          ...(input.weights.rmsNormEps ? { eps: input.weights.rmsNormEps } : {}),
          ...(input.options.device ? { device: input.options.device } : {}),
          ...(input.options.gpu ? { gpu: input.options.gpu } : {}),
          ...(input.options.requireWebGpu ? { requireWebGpu: true } : {}),
          bufferCache: input.bufferCache,
          traceMetadata: {
            runtime: "unlocked-browser-transformer",
            requestId: input.requestId,
            layerIndex: input.layerIndex,
            phase: input.phase,
            residentQkvPostProjection: true,
            purpose: `${input.phase}_q_post_projection`,
          },
          ...(input.commandBatch ? { commandBatch: input.commandBatch } : {}),
        }));
    const kPostProjection = qkvNormRopePair
      ? {
          tensor: qkvNormRopePair.k,
          trace: qkvNormRopePairResidentTrace(qkvNormRopePair.trace, "k_post_projection"),
        }
      : remember(await runQwenQkvPostProjectionResidentWebGpu({
          projected: kProjection.tensor,
          headCount: geometry.keyValueHeads,
          headDim: geometry.headDim,
          positions: input.positions,
          ...(input.layer.kNorm ? { normWeight: input.layer.kNorm } : {}),
          ...(input.weights.ropeTheta ? { ropeTheta: input.weights.ropeTheta } : {}),
          ...(input.weights.rmsNormEps ? { eps: input.weights.rmsNormEps } : {}),
          ...(input.options.device ? { device: input.options.device } : {}),
          ...(input.options.gpu ? { gpu: input.options.gpu } : {}),
          ...(input.options.requireWebGpu ? { requireWebGpu: true } : {}),
          bufferCache: input.bufferCache,
          traceMetadata: {
            runtime: "unlocked-browser-transformer",
            requestId: input.requestId,
            layerIndex: input.layerIndex,
            phase: input.phase,
            residentQkvPostProjection: true,
            purpose: `${input.phase}_k_post_projection`,
          },
          ...(input.commandBatch ? { commandBatch: input.commandBatch } : {}),
        }));
    if (input.phase === "decode" && input.options.requireWebGpu === true) {
      const qRows = input.positions.length;
      const qPlaceholder = createZeroMatrix(qRows, geometry.qProjectionSize);
      const kPlaceholder = createZeroMatrix(qRows, geometry.kvProjectionSize);
      const vPlaceholder = createZeroMatrix(qRows, geometry.kvProjectionSize);
      return {
        q: qPlaceholder,
        k: expandGroupedKeyValueHeads(kPlaceholder, geometry),
        v: expandGroupedKeyValueHeads(vPlaceholder, geometry),
        attentionK: kPlaceholder,
        attentionV: vPlaceholder,
        resident: {
          q: retain(qPostProjection.tensor),
          attentionK: retain(kPostProjection.tensor),
          attentionV: retain(vProjection.tensor),
        },
        geometry,
        proofs: {
          qProjection: qProjection.trace,
          kProjection: kProjection.trace,
          vProjection: vProjection.trace,
          qPostProjection: qPostProjection.trace,
          kPostProjection: kPostProjection.trace,
          qkvReadbackCount: 0,
          ...(packedQkvProjection ? { packedQkvProjection: packedQkvProjection.trace } : {}),
          ...(qkvNormRopePair ? { qkvNormRopePair: qkvNormRopePair.trace } : {}),
        },
      };
    }
    const [q, k, v] = await readWebGpuResidentTensors([
      qPostProjection.tensor,
      kPostProjection.tensor,
      vProjection.tensor,
    ]);
    if (!q || !k || !v) {
      throw new Error("resident Q/K/V batched readback did not return all projected tensors.");
    }
    return {
      q,
      k: expandGroupedKeyValueHeads(k, geometry),
      v: expandGroupedKeyValueHeads(v, geometry),
      attentionK: k,
      attentionV: v,
      geometry,
      proofs: {
        qProjection: qProjection.trace,
        kProjection: kProjection.trace,
        vProjection: vProjection.trace,
        qPostProjection: qPostProjection.trace,
        kPostProjection: kPostProjection.trace,
        qkvReadbackCount: 3,
        ...(packedQkvProjection ? { packedQkvProjection: packedQkvProjection.trace } : {}),
        ...(qkvNormRopePair ? { qkvNormRopePair: qkvNormRopePair.trace } : {}),
      },
    };
  } catch (error) {
    if (input.options.requireWebGpu) throw error;
    return null;
  } finally {
    const tensorsToDestroy = residentTensors.splice(0).reverse();
    if (input.commandBatch) {
      input.commandBatch.deferAfterSubmit(() => {
        for (const tensor of tensorsToDestroy) destroyWebGpuResidentTensor(tensor);
      });
    } else {
      for (const tensor of tensorsToDestroy) destroyWebGpuResidentTensor(tensor);
    }
  }
}

async function projectWithDenseMatMul(
  activations: Matrix,
  projection: RuntimeMatrix,
  context: ProjectionKernelContext,
  projectionName: "qProj" | "kProj" | "vProj" | "oProj",
): Promise<{ output: Matrix; trace: WebGpuDenseMatMulTrace }> {
  const result = await runDenseMatMulWebGpu({
    activations,
    projection,
    ...(context.options.backendPreference ? { backendPreference: context.options.backendPreference } : {}),
    ...(context.options.device ? { device: context.options.device } : {}),
    ...(context.options.gpu ? { gpu: context.options.gpu } : {}),
    ...(context.options.requireWebGpu ? { requireWebGpu: true } : {}),
    ...(isPackedRuntimeMatrix(projection)
      ? {
          bufferCache: context.bufferCache,
          projectionCacheKey: projectionCacheKey(context.layerIndex, projectionName),
          projectionCachePolicy: "stable" as const,
        }
      : {}),
    traceMetadata: {
      runtime: "unlocked-browser-transformer",
      requestId: context.requestId,
      layerIndex: context.layerIndex,
      phase: context.phase,
      purpose: `${context.phase}_${projectionNameToPurpose(projectionName)}_projection`,
    },
  });
  return { output: result.output, trace: result.trace };
}

async function projectWithDenseMatMulResident(
  activations: Matrix | WebGpuResidentTensor,
  projection: RuntimeMatrix,
  context: ProjectionKernelContext,
  projectionName: "qProj" | "kProj" | "vProj" | "oProj",
): Promise<{ tensor: WebGpuResidentTensor; trace: WebGpuDenseMatMulResidentTrace }> {
  const result = await runDenseMatMulResidentWebGpu({
    activations,
    projection,
    ...(context.options.backendPreference ? { backendPreference: context.options.backendPreference } : {}),
    ...(context.options.device ? { device: context.options.device } : {}),
    ...(context.options.gpu ? { gpu: context.options.gpu } : {}),
    ...(context.options.requireWebGpu ? { requireWebGpu: true } : {}),
    ...(isPackedRuntimeMatrix(projection)
      ? {
          bufferCache: context.bufferCache,
          projectionCacheKey: projectionCacheKey(context.layerIndex, projectionName),
          projectionCachePolicy: "stable" as const,
        }
      : {}),
    traceMetadata: {
      runtime: "unlocked-browser-transformer",
      requestId: context.requestId,
      layerIndex: context.layerIndex,
      phase: context.phase,
      residentQkvPostProjection: true,
      purpose: `${context.phase}_${projectionNameToPurpose(projectionName)}_projection`,
    },
    ...(context.commandBatch ? { commandBatch: context.commandBatch } : {}),
  });
  return { tensor: result.tensor, trace: result.trace };
}

function packedDenseMatMulTrace(
  trace: WebGpuPackedQkvProjectionResidentTrace,
  projectionName: "qProjection" | "kProjection" | "vProjection",
): WebGpuDenseMatMulResidentTrace {
  const outputSize = projectionName === "qProjection"
    ? trace.qOutputSize
    : projectionName === "kProjection"
      ? trace.kOutputSize
      : trace.vOutputSize;
  const dispatchCount = projectionName === "qProjection" ? trace.dispatchCount : 0;
  return {
    backend: trace.backend,
    tokens: trace.tokens,
    hidden: trace.hidden,
    outputSize,
    computeMs: trace.computeMs,
    outputResident: true,
    readback: false,
    inputResident: trace.inputResident,
    pipelineCacheKey: trace.pipelineCacheKey,
    pipelineCacheHit: trace.pipelineCacheHit,
    metadata: {
      ...(trace.metadata ?? {}),
      projectionName,
      packedQkvProjection: true,
    },
    projectionCacheHit: projectionName === "qProjection"
      ? trace.projectionCacheHits?.qProjection
      : projectionName === "kProjection"
        ? trace.projectionCacheHits?.kProjection
        : trace.projectionCacheHits?.vProjection,
    dispatchCount,
  } as WebGpuDenseMatMulResidentTrace;
}

function qkvNormRopePairResidentTrace(
  trace: WebGpuQkvNormRopePairResidentTrace,
  purpose: "q_post_projection" | "k_post_projection",
): WebGpuResidentTensorTrace {
  return {
    backend: trace.backend,
    tokens: trace.tokens,
    hidden: purpose === "q_post_projection" ? trace.qHidden : trace.kHidden,
    computeMs: trace.computeMs,
    outputResident: true,
    readback: false,
    inputResident: purpose === "q_post_projection" ? trace.inputResident.q : trace.inputResident.k,
    ...(trace.pipelineCacheKey !== undefined ? { pipelineCacheKey: trace.pipelineCacheKey } : {}),
    ...(trace.pipelineCacheHit !== undefined ? { pipelineCacheHit: trace.pipelineCacheHit } : {}),
    metadata: {
      ...(trace.metadata ?? {}),
      purpose,
      qkvNormRopePairProjection: purpose,
    },
  };
}

function projectionNameToPurpose(name: "qProj" | "kProj" | "vProj" | "oProj"): "q" | "k" | "v" | "o" {
  return name.slice(0, 1) as "q" | "k" | "v" | "o";
}

function projectionCacheKey(layerIndex: number, projectionName: "qProj" | "kProj" | "vProj" | "oProj" | "mlp"): string {
  return `layer${layerIndex}.${projectionName}`;
}

function addPackedProjectionPreload(
  descriptors: WebGpuStableMatrixPreloadDescriptor[],
  cacheKey: string,
  matrix: RuntimeMatrix,
  namespace: "dense-matmul" | "mlp",
): void {
  if (!isPackedRuntimeMatrix(matrix)) return;
  descriptors.push({
    key: `${namespace}:${cacheKey}`,
    matrix,
  });
}

function addTokenEmbeddingLookupPreload(
  descriptors: WebGpuStableMatrixPreloadDescriptor[],
  matrix: RuntimeMatrix,
): void {
  const rowCount = matrixRowCount(matrix);
  const rows = Math.min(rowCount, TOKEN_EMBEDDING_LOOKUP_TILE_ROWS);
  if (rows <= 0) return;
  descriptors.push({
    key: `token-embedding:tokenEmbedding:rows:0-${rows}`,
    matrix,
    rowIds: Array.from({ length: rows }, (_value, rowId) => rowId),
  });
}

function normalizeWarmupLayerCount(value: number | undefined, maxLayerCount: number): number {
  if (value === undefined) return maxLayerCount;
  if (!Number.isInteger(value) || value <= 0) throw new Error("UnlockedBrowserTransformerBackend warmup layerCount must be a positive integer.");
  return Math.min(value, maxLayerCount);
}

function normalizeResidencyLogitTileRows(value: number | null | undefined, vocabSize: number): number {
  if (value === null || value === undefined) return Math.min(DEFAULT_LOGIT_RESIDENCY_TILE_ROWS, vocabSize);
  if (!Number.isInteger(value) || value <= 0) throw new Error("UnlockedBrowserTransformerBackend warmup logitTileRows must be a positive integer.");
  return Math.min(value, vocabSize);
}

interface PrefillAttentionKernelResult {
  output: Matrix;
  attentionBackend: WebGpuSparseAttentionResult["backend"] | "mixed";
  packedHeadBackends: WebGpuSparseAttentionResult["backend"][];
  packedHeadCount: number;
  keyValueHeadCount: number;
  keyValueCompressionRatio: number;
  selectedKeyRows: number;
  attentionMs: number;
  prefillChunkDispatch: "single_dispatch" | "chunked_dispatch";
  attentionDispatchCount: number;
  awaitedDispatchBreaks: number;
}

interface SpeculativeContinuationInput {
  weights: UnlockedBrowserTransformerWeights;
  modelId: string;
  cache: UnlockedBrowserKvCacheHandle;
  inputTokenIds: number[];
  requestId: string;
  policy: SSALayerRoutingPolicy[];
  logitCandidateTokenIds: number[] | null;
  suppressedTokenIds: number[];
  logitTopK: number;
  logitTileRows: number | null;
  sparseAttentionOptions: Pick<WebGpuSsaBackendOptions, "backendPreference" | "device" | "gpu" | "requireWebGpu">;
  bufferCache: WebGpuRuntimeBufferCache;
  initialNonPinnedTier: Extract<KVTier, "vram" | "ram" | "disk">;
}

interface SpeculativeLayerContinuationRows {
  q: Matrix;
  k: Matrix;
  v: Matrix;
  compactK?: Matrix;
  compactV?: Matrix;
  hidden: Matrix;
}

interface SpeculativeContinuationResult {
  inputTokenIds: number[];
  layerRows: Record<number, SpeculativeLayerContinuationRows>;
  logitRows: number[][];
  logitTokenIdsByRow: Array<number[] | undefined>;
  logitProofs: Array<UnlockedBrowserDecodeBackendProof | undefined>;
  projectionProofs: UnlockedBrowserDecodeProjectionLayerProof[];
  oProjectionProofs: UnlockedBrowserDecodeOProjectionProof[];
  mlpProofs: UnlockedBrowserDecodeMlpProof[];
  traces: SSAKernelTrace[];
  kvPagingEvents: KVTensorPagingEvent[];
  tspTrace: TSPExecutionTraceRecord[];
}

async function runSpeculativeContinuation(input: SpeculativeContinuationInput): Promise<SpeculativeContinuationResult> {
  let hidden = input.inputTokenIds.map((tokenId) => tokenEmbedding(input.weights, tokenId));
  const layerRows: Record<number, SpeculativeLayerContinuationRows> = {};
  const traces: SSAKernelTrace[] = [];
  const kvPagingEvents: KVTensorPagingEvent[] = [];
  const tspTrace: TSPExecutionTraceRecord[] = [];
  const projectionProofs: UnlockedBrowserDecodeProjectionLayerProof[] = [];
  const oProjectionProofs: UnlockedBrowserDecodeOProjectionProof[] = [];
  const mlpProofs: UnlockedBrowserDecodeMlpProof[] = [];
  const logitRows: number[][] = [];
  const logitTokenIdsByRow: Array<number[] | undefined> = [];
  const logitProofs: Array<UnlockedBrowserDecodeBackendProof | undefined> = [];
  let executedLayer = false;

  for (let position = 0; position < input.policy.length; position += 1) {
    const layerPolicy = input.policy[position] as SSALayerRoutingPolicy | undefined;
    if (!layerPolicy) continue;
    const layerIndex = normalizeLayerIndex(layerPolicy.layerIndex, position);
    const layer = input.weights.layers[layerIndex];
    if (!layer) throw new Error(`UnlockedBrowserTransformerBackend has no layer ${layerIndex}.`);
    const layerState = input.cache.layerStates[layerIndex];
    if (!layerState) {
      throw new Error(`UnlockedBrowserTransformerBackend has no per-layer KV state for layer ${layerIndex}.`);
    }
    const baseTokenCount = layerState.k.length;
    const totalTokenCount = baseTokenCount + input.inputTokenIds.length;
    const blockTokenRanges = buildBlockTokenRanges(totalTokenCount, layerPolicy);
    const attentionInput = prepareAttentionInput(hidden, layer, input.weights.rmsNormEps);
    let continuationRows: SpeculativeLayerContinuationRows | null = null;
    let attentionOutput: PrefillAttentionKernelResult | null = null;
    let attentionProjectionProofs: ProjectedAttentionTensors["proofs"] | null = null;
    const kvBlocks = buildKvBlocks({
      layerIndex,
      blockTokenRanges,
      policy: layerPolicy,
      initialNonPinnedTier: input.initialNonPinnedTier,
    });
    const registry = new KVTensorPagingRegistry({ now: Date.now(), defaultEvictionTier: "disk" });
    for (const block of kvBlocks) registry.registerBlock(block, block.tensorHandles);
    const selectedKvBlockIds = collectSelectedBlockIds(layerPolicy).map((blockId) => `layer${layerIndex}:${blockId}`);
    const schedule = buildDecodeSchedule(layerIndex, totalTokenCount);
    const layerTspTrace = await executeTSPSchedule(schedule, {
      kv_prefetch: () => {
        const readiness = registry.ensureBlocksAvailableForSparseAttention(selectedKvBlockIds);
        kvPagingEvents.push(...readiness.events);
      },
      attention: async () => {
        const projected = await projectAttentionTensors({
          attentionInput,
          layer,
          weights: input.weights,
          positions: input.inputTokenIds.map((_tokenId, offset) => baseTokenCount + offset),
          options: input.sparseAttentionOptions,
          bufferCache: input.bufferCache,
          requestId: input.requestId,
          layerIndex,
          phase: "decode",
        });
        attentionProjectionProofs = projected.proofs;
        attentionOutput = await runContinuationAttention({
          q: projected.q,
          k: [...(layerState.compactK ?? layerState.k), ...projected.attentionK],
          v: [...(layerState.compactV ?? layerState.v), ...projected.attentionV],
          geometry: projected.geometry,
          baseTokenCount,
          policy: layerPolicy,
          blockTokenRanges,
          options: input.sparseAttentionOptions,
          bufferCache: input.bufferCache,
        });
        continuationRows = {
          q: projected.q,
          k: projected.k,
          v: projected.v,
          compactK: projected.attentionK,
          compactV: projected.attentionV,
          hidden: [],
        };
        const trace: PackedHeadSparseTrace = {
          ...buildTrace(input.requestId, layerIndex, layerPolicy, totalTokenCount),
          queryBlockIndex: queryBlockIndexForToken(totalTokenCount - 1, layerPolicy),
          attentionMs: attentionOutput.attentionMs,
          attentionBackend: attentionOutput.attentionBackend,
          packedHeadBackends: attentionOutput.packedHeadBackends,
          packedHeadCount: attentionOutput.packedHeadCount,
          keyValueHeadCount: attentionOutput.keyValueHeadCount,
          keyValueCompressionRatio: attentionOutput.keyValueCompressionRatio,
        };
        traces.push(trace);
      },
      mlp: async () => {
        if (!attentionOutput || !continuationRows) {
          throw new Error("UnlockedBrowserTransformerBackend speculative TSP schedule ran MLP before attention.");
        }
        const nextHidden = await applyAttentionOutput(
          hidden,
          attentionOutput.output,
          layer,
          input.weights.rmsNormEps,
          {
            options: input.sparseAttentionOptions,
            bufferCache: input.bufferCache,
            requestId: input.requestId,
            layerIndex,
            phase: "decode",
          },
        );
        hidden = nextHidden.hidden;
        continuationRows.hidden = clonePlainMatrix(hidden);
        layerRows[layerIndex] = continuationRows;
        oProjectionProofs.push({
          layerIndex,
          backend: nextHidden.oProjection.backend,
          trace: nextHidden.oProjection,
        });
        if (attentionProjectionProofs) {
          projectionProofs.push(createDecodeProjectionLayerProof(
            layerIndex,
            attentionProjectionProofs,
            {
              layerIndex,
              backend: nextHidden.oProjection.backend,
              trace: nextHidden.oProjection,
            },
          ));
        }
        if (nextHidden.mlp) {
          mlpProofs.push({
            layerIndex,
            backend: nextHidden.mlp.lastTrace.backend,
            trace: nextHidden.mlp.lastTrace,
          });
        }
        if (position === input.policy.length - 1) {
          for (const row of hidden) {
            const lastHidden = applyOptionalRmsNorm(row, input.weights.finalNorm, input.weights.rmsNormEps);
            const projectedLogits = await projectDecodeLogits(
              lastHidden,
              input.weights.outputProjection,
              input.logitCandidateTokenIds,
              input.suppressedTokenIds,
              input.logitTopK,
              input.logitTileRows,
              input.sparseAttentionOptions,
              input.bufferCache,
              input.requestId,
            );
            logitRows.push(projectedLogits.logits);
            logitTokenIdsByRow.push(projectedLogits.logitTokenIds);
            logitProofs.push(projectedLogits.backendProof);
          }
        }
      },
    }, {
      metadata: {
        backend: "unlocked-browser-transformer",
        modelId: input.modelId,
        requestId: input.requestId,
        layerIndex,
        mode: "speculative_continuation",
        tokenStart: baseTokenCount,
        tokenEnd: totalTokenCount,
      },
    });
    tspTrace.push(...layerTspTrace);
    executedLayer = true;
  }

  if (!executedLayer) throw new Error("UnlockedBrowserTransformerBackend speculative verification requires at least one executable layer policy.");
  if (logitRows.length === 0) {
    for (const row of hidden) {
      const lastHidden = applyOptionalRmsNorm(row, input.weights.finalNorm, input.weights.rmsNormEps);
      const projectedLogits = await projectDecodeLogits(
        lastHidden,
        input.weights.outputProjection,
        input.logitCandidateTokenIds,
        input.suppressedTokenIds,
        input.logitTopK,
        input.logitTileRows,
        input.sparseAttentionOptions,
        input.bufferCache,
        input.requestId,
      );
      logitRows.push(projectedLogits.logits);
      logitTokenIdsByRow.push(projectedLogits.logitTokenIds);
      logitProofs.push(projectedLogits.backendProof);
    }
  }

  return {
    inputTokenIds: [...input.inputTokenIds],
    layerRows,
    logitRows,
    logitTokenIdsByRow,
    logitProofs,
    projectionProofs,
    oProjectionProofs,
    mlpProofs,
    traces,
    kvPagingEvents,
    tspTrace,
  };
}

async function runContinuationAttention(input: {
  q: Matrix;
  k: Matrix;
  v: Matrix;
  geometry: AttentionGeometry;
  baseTokenCount: number;
  policy: SSALayerRoutingPolicy;
  blockTokenRanges: Record<string, { tokenStart: number; tokenEnd: number }>;
  options: Pick<WebGpuSsaBackendOptions, "backendPreference" | "device" | "gpu" | "requireWebGpu">;
  bufferCache: WebGpuRuntimeBufferCache;
}): Promise<PrefillAttentionKernelResult> {
  const selectedKeyIndexesByQuery = input.q.map((_row, queryOffset) => (
    buildSelectedKeyIndexesForAbsoluteQuery(input.k, input.blockTokenRanges, input.policy, input.baseTokenCount + queryOffset)
      .filter((keyIndex) => keyIndex <= input.baseTokenCount + queryOffset)
  ));
  const selectedKeyRows = selectedKeyIndexesByQuery.reduce((sum, indexes) => sum + indexes.length, 0);
  const packed = await runPackedSparseAttentionWebGpu({
    q: input.q,
    k: input.k,
    v: input.v,
    selectedKeyIndexesByQuery,
    causal: false,
    headCount: input.geometry.attentionHeads,
    keyValueHeadCount: packedKeyValueHeadCount(input.k, input.geometry),
    headDim: input.geometry.headDim,
    ...(input.options.backendPreference ? { backendPreference: input.options.backendPreference } : {}),
    ...(input.options.device ? { device: input.options.device } : {}),
    ...(input.options.gpu ? { gpu: input.options.gpu } : {}),
    ...(input.options.requireWebGpu ? { requireWebGpu: true } : {}),
    bufferCache: input.bufferCache,
  });
  const output = packed.output;
  const packedHeadBackends = repeatBackend(packed.backend, input.geometry.attentionHeads);

  return {
    output,
    attentionBackend: summarizePackedBackends(packedHeadBackends),
    packedHeadBackends,
    packedHeadCount: input.geometry.attentionHeads,
    keyValueHeadCount: packed.trace.keyValueHeadCount,
    keyValueCompressionRatio: keyValueCompressionRatio(packed.trace.keyValueHeadCount, input.geometry.attentionHeads),
    selectedKeyRows,
    attentionMs: packed.trace.computeMs,
    prefillChunkDispatch: "single_dispatch",
    attentionDispatchCount: packed.trace.dispatchCount,
    awaitedDispatchBreaks: 0,
  };
}

function buildSelectedKeyIndexesForAbsoluteQuery(
  keyRows: Matrix,
  blockTokenRanges: Record<string, { tokenStart: number; tokenEnd: number }>,
  policy: SSALayerRoutingPolicy,
  queryTokenIndex: number,
): number[] {
  const queryBlockIndex = queryBlockIndexForToken(queryTokenIndex, policy);
  const selectedBlockIds = selectedBlockIdsForQueryBlock(policy, queryBlockIndex);
  const indexes: number[] = [];
  for (const blockId of selectedBlockIds) {
    const range = blockTokenRanges[blockId];
    if (!range) continue;
    for (let token = range.tokenStart; token < range.tokenEnd; token += 1) {
      if (token >= 0 && token < keyRows.length && !indexes.includes(token)) indexes.push(token);
    }
  }
  return indexes.sort((a, b) => a - b);
}

function countAcceptedDraftPrefix(draftTokenIds: number[], targetTokenIds: number[]): number {
  let accepted = 0;
  while (accepted < draftTokenIds.length && draftTokenIds[accepted] === targetTokenIds[accepted]) accepted += 1;
  return accepted;
}

function commitSpeculativeContinuation(input: {
  cache: UnlockedBrowserKvCacheHandle;
  requestId: string;
  inputTokenIds: number[];
  continuation: SpeculativeContinuationResult;
  commitCount: number;
  policy: SSALayerRoutingPolicy[];
  initialNonPinnedTier: Extract<KVTier, "vram" | "ram" | "disk">;
}): void {
  if (input.commitCount <= 0) return;
  input.cache.tokenIds.push(...input.inputTokenIds);
  input.cache.kvBlocks.length = 0;
  input.cache.blockTokenRanges = {};
  for (let position = 0; position < input.policy.length; position += 1) {
    const layerPolicy = input.policy[position] as SSALayerRoutingPolicy | undefined;
    if (!layerPolicy) continue;
    const layerIndex = normalizeLayerIndex(layerPolicy.layerIndex, position);
    const rows = input.continuation.layerRows[layerIndex];
    const layerState = input.cache.layerStates[layerIndex];
    if (!rows || !layerState) continue;
    layerState.q.push(...rows.q.slice(0, input.commitCount).map((row) => [...row]));
    layerState.k.push(...rows.k.slice(0, input.commitCount).map((row) => [...row]));
    layerState.v.push(...rows.v.slice(0, input.commitCount).map((row) => [...row]));
    if (layerState.compactK && rows.compactK) {
      layerState.compactK.push(...rows.compactK.slice(0, input.commitCount).map((row) => [...row]));
    }
    if (layerState.compactV && rows.compactV) {
      layerState.compactV.push(...rows.compactV.slice(0, input.commitCount).map((row) => [...row]));
    }
    layerState.hidden.push(...rows.hidden.slice(0, input.commitCount).map((row) => [...row]));
    layerState.projectedTokenCount = layerState.k.length;
    const blockTokenRanges = buildBlockTokenRanges(input.cache.tokenIds.length, layerPolicy);
    if (Object.keys(input.cache.blockTokenRanges).length === 0) input.cache.blockTokenRanges = blockTokenRanges;
    input.cache.layers[layerIndex] = createLayerHandles({
      requestId: input.requestId,
      layerIndex,
      q: layerState.q,
      k: layerState.k,
      v: layerState.v,
      blockTokenRanges,
      cloneMatrices: false,
    });
    input.cache.kvBlocks.push(...buildKvBlocks({
      layerIndex,
      blockTokenRanges,
      policy: layerPolicy,
      initialNonPinnedTier: input.initialNonPinnedTier,
    }));
  }
}

async function runPrefillAttention(
  q: Matrix,
  k: Matrix,
  v: Matrix,
  geometry: AttentionGeometry,
  options: Pick<WebGpuSsaBackendOptions, "backendPreference" | "device" | "gpu" | "requireWebGpu">,
  bufferCache: WebGpuRuntimeBufferCache,
  prefillPlan?: PrefillChunkPlan,
): Promise<PrefillAttentionKernelResult> {
  const selectedKeyIndexesByQuery = buildDenseCausalSelectedKeyIndexes(q.length);
  const selectedKeyRows = selectedKeyIndexesByQuery.reduce((sum, indexes) => sum + indexes.length, 0);
  const output = q.map(() => new Array(geometry.expandedProjectionSize).fill(0));
  const packedHeadBackends: WebGpuSparseAttentionResult["backend"][] = [];
  let attentionMs = 0;
  let attentionDispatchCount = 0;
  let awaitedDispatchBreaks = 0;
  let keyValueHeadCount = packedKeyValueHeadCount(k, geometry);
  const chunks = resolvePrefillAttentionChunks(q.length, prefillPlan);

  for (let chunkIndex = 0; chunkIndex < chunks.length; chunkIndex += 1) {
    const chunk = chunks[chunkIndex] ?? { tokenStart: 0, tokenEnd: q.length };
    const qChunk = q.slice(chunk.tokenStart, chunk.tokenEnd);
    const selectedChunk = selectedKeyIndexesByQuery.slice(chunk.tokenStart, chunk.tokenEnd);
    const packed = await runPackedSparseAttentionWebGpu({
      q: qChunk,
      k,
      v,
      selectedKeyIndexesByQuery: selectedChunk,
      causal: false,
      headCount: geometry.attentionHeads,
      keyValueHeadCount: packedKeyValueHeadCount(k, geometry),
      headDim: geometry.headDim,
      ...(options.backendPreference ? { backendPreference: options.backendPreference } : {}),
      ...(options.device ? { device: options.device } : {}),
      ...(options.gpu ? { gpu: options.gpu } : {}),
      ...(options.requireWebGpu ? { requireWebGpu: true } : {}),
      bufferCache,
    });
    attentionDispatchCount += packed.trace.dispatchCount;
    packedHeadBackends.push(...repeatBackend(packed.backend, geometry.attentionHeads));
    keyValueHeadCount = packed.trace.keyValueHeadCount;
    attentionMs += packed.trace.computeMs;
    for (let localQueryIndex = 0; localQueryIndex < packed.output.length; localQueryIndex += 1) {
      output[chunk.tokenStart + localQueryIndex] = [...(packed.output[localQueryIndex] ?? [])];
    }
    if (chunkIndex < chunks.length - 1) {
      awaitedDispatchBreaks += 1;
      await yieldToBrowserScheduler();
    }
  }

  return {
    output,
    attentionBackend: summarizePackedBackends(packedHeadBackends),
    packedHeadBackends,
    packedHeadCount: geometry.attentionHeads,
    keyValueHeadCount,
    keyValueCompressionRatio: keyValueCompressionRatio(keyValueHeadCount, geometry.attentionHeads),
    selectedKeyRows,
    attentionMs,
    prefillChunkDispatch: chunks.length > 1 ? "chunked_dispatch" : "single_dispatch",
    attentionDispatchCount,
    awaitedDispatchBreaks,
  };
}

function resolvePrefillAttentionChunks(
  tokenCount: number,
  prefillPlan: PrefillChunkPlan | undefined,
): Array<{ tokenStart: number; tokenEnd: number }> {
  const planned = prefillPlan?.chunks
    .map((chunk) => ({
      tokenStart: Math.max(0, Math.min(tokenCount, chunk.tokenStart)),
      tokenEnd: Math.max(0, Math.min(tokenCount, chunk.tokenEnd)),
    }))
    .filter((chunk) => chunk.tokenEnd > chunk.tokenStart) ?? [];
  return planned.length > 0 ? planned : [{ tokenStart: 0, tokenEnd: tokenCount }];
}

async function yieldToBrowserScheduler(): Promise<void> {
  const scheduler = (globalThis as typeof globalThis & {
    scheduler?: { yield?: () => Promise<void> };
  }).scheduler;
  if (typeof scheduler?.yield === "function") {
    await scheduler.yield();
    return;
  }
  await new Promise<void>((resolve) => setTimeout(resolve, 0));
}

function buildDenseCausalSelectedKeyIndexes(tokenCount: number): number[][] {
  return Array.from({ length: tokenCount }, (_, queryIndex) => (
    Array.from({ length: queryIndex + 1 }, (_value, keyIndex) => keyIndex)
  ));
}

function toPrefillProjectionProof(trace: WebGpuDenseMatMulTrace): UnlockedBrowserPrefillProjectionProof {
  return { backend: trace.backend, trace };
}

function toPrefillResidentTensorProof(trace: WebGpuResidentTensorTrace): UnlockedBrowserPrefillResidentTensorProof {
  return { backend: trace.backend, trace };
}

function summarizePrefillDispatchProof(layers: UnlockedBrowserPrefillLayerProof[]): {
  prefillChunkDispatch: "single_dispatch" | "chunked_dispatch";
  attentionDispatchCount: number;
  awaitedDispatchBreaks: number;
} {
  return {
    prefillChunkDispatch: layers.some((layer) => layer.prefillChunkDispatch === "chunked_dispatch")
      ? "chunked_dispatch"
      : "single_dispatch",
    attentionDispatchCount: layers.reduce((sum, layer) => sum + (layer.attentionDispatchCount ?? 0), 0),
    awaitedDispatchBreaks: layers.reduce((sum, layer) => sum + (layer.awaitedDispatchBreaks ?? 0), 0),
  };
}

function densePackedSelfAttention(q: Matrix, k: Matrix, v: Matrix, geometry: AttentionGeometry): Matrix {
  return q.map((query, queryIndex) => {
    const output = new Array(geometry.expandedProjectionSize).fill(0);
    for (let headIndex = 0; headIndex < geometry.attentionHeads; headIndex += 1) {
      const headOffset = headIndex * geometry.headDim;
      const legalKeyIndexes = k.map((_, index) => index).filter((keyIndex) => keyIndex <= queryIndex);
      const scores = legalKeyIndexes.map((keyIndex) => (
        dotSlices(query, k[keyIndex] ?? [], headOffset, headOffset, geometry.headDim) / Math.sqrt(geometry.headDim)
      ));
      const weights = softmax(scores);
      for (let selectedIndex = 0; selectedIndex < legalKeyIndexes.length; selectedIndex += 1) {
        const weight = weights[selectedIndex] ?? 0;
        const value = v[legalKeyIndexes[selectedIndex] ?? 0] ?? [];
        for (let dim = 0; dim < geometry.headDim; dim += 1) {
          const outputIndex = headOffset + dim;
          output[outputIndex] = (output[outputIndex] ?? 0) + weight * (value[outputIndex] ?? 0);
        }
      }
    }
    return output;
  });
}

function buildSelectedKeyIndexesForQuery(
  _q: SsaToyTensorHandle,
  k: SsaToyTensorHandle,
  policy: SSALayerRoutingPolicy,
  queryTokenIndex: number,
): number[] {
  const queryBlockIndex = queryBlockIndexForToken(queryTokenIndex, policy);
  const selectedBlockIds = selectedBlockIdsForQueryBlock(policy, queryBlockIndex);
  const indexes: number[] = [];
  for (const blockId of selectedBlockIds) {
    const range = k.blockTokenRanges[blockId];
    if (!range) continue;
    for (let token = range.tokenStart; token < range.tokenEnd; token += 1) {
      if (token >= 0 && token < k.matrix.length && !indexes.includes(token)) indexes.push(token);
    }
  }
  return indexes.sort((a, b) => a - b);
}

function queryBlockIndexForToken(queryTokenIndex: number, policy: SSALayerRoutingPolicy): number {
  return Math.floor(queryTokenIndex / Math.max(1, policy.blockSize));
}

function selectedBlockIdsForQueryBlock(policy: SSALayerRoutingPolicy, queryBlockIndex: number): string[] {
  return [...(policy.selectedBlockIdsByQueryBlock[queryBlockIndex] ?? policy.pinnedBlockIds)].sort(compareBlockIds);
}

function normalizePackedHeads(projected: Matrix, norm: RuntimeVector | undefined, headDim: number, rmsNormEps?: number): Matrix {
  if (!norm) return projected;
  return projected.map((row) => {
    const normalized: Vector = [];
    const headCount = Math.max(1, row.length / headDim);
    for (let headIndex = 0; headIndex < headCount; headIndex += 1) {
      const start = headIndex * headDim;
      normalized.push(...applyRmsNorm(row.slice(start, start + headDim), norm, rmsNormEps));
    }
    return normalized;
  });
}

function applyRotaryPositionEmbedding(
  projected: Matrix,
  headCount: number,
  headDim: number,
  positions: number[],
  ropeTheta: number | undefined,
): Matrix {
  if (ropeTheta === undefined) return projected;
  if (headDim % 2 !== 0) throw new Error("Qwen RoPE requires an even headDim.");
  const halfDim = headDim / 2;
  return projected.map((row, rowIndex) => {
    const position = positions[rowIndex] ?? rowIndex;
    const rotated = [...row];
    for (let headIndex = 0; headIndex < headCount; headIndex += 1) {
      const headOffset = headIndex * headDim;
      for (let dim = 0; dim < halfDim; dim += 1) {
        const firstIndex = headOffset + dim;
        const secondIndex = headOffset + dim + halfDim;
        const frequency = 1 / ropeTheta ** (dim / halfDim);
        const angle = position * frequency;
        const cos = Math.cos(angle);
        const sin = Math.sin(angle);
        const first = row[firstIndex] ?? 0;
        const second = row[secondIndex] ?? 0;
        rotated[firstIndex] = first * cos - second * sin;
        rotated[secondIndex] = first * sin + second * cos;
      }
    }
    return rotated;
  });
}

function expandGroupedKeyValueHeads(matrix: Matrix, geometry: AttentionGeometry): Matrix {
  if (geometry.attentionHeads === geometry.keyValueHeads) return matrix.map((row) => [...row]);
  const headsPerKv = geometry.attentionHeads / geometry.keyValueHeads;
  return matrix.map((row) => {
    const expanded: Vector = [];
    for (let attentionHead = 0; attentionHead < geometry.attentionHeads; attentionHead += 1) {
      const kvHead = Math.floor(attentionHead / headsPerKv);
      const sourceOffset = kvHead * geometry.headDim;
      for (let dim = 0; dim < geometry.headDim; dim += 1) {
        expanded.push(row[sourceOffset + dim] ?? 0);
      }
    }
    return expanded;
  });
}

function dotSlices(a: ArrayLike<number>, b: ArrayLike<number>, aOffset: number, bOffset: number, length: number): number {
  let sum = 0;
  for (let index = 0; index < length; index += 1) {
    sum += (a[aOffset + index] ?? 0) * (b[bOffset + index] ?? 0);
  }
  return sum;
}

function dot(a: ArrayLike<number>, b: ArrayLike<number>): number {
  let sum = 0;
  for (let index = 0; index < a.length; index += 1) {
    sum += (a[index] ?? 0) * (b[index] ?? 0);
  }
  return sum;
}

function softmax(values: Vector): Vector {
  const max = Math.max(...values);
  const exp = values.map((value) => Math.exp(value - max));
  const total = exp.reduce((sum, value) => sum + value, 0) || 1;
  return exp.map((value) => value / total);
}

function gelu(value: number): number {
  return 0.5 * value * (1 + Math.tanh(Math.sqrt(2 / Math.PI) * (value + 0.044715 * value ** 3)));
}

function silu(value: number): number {
  return value / (1 + Math.exp(-value));
}

function applyOptionalRmsNorm(vector: Vector, norm: RuntimeVector | undefined, eps?: number): Vector {
  return norm ? applyRmsNorm(vector, norm, eps) : vector;
}

function applyRmsNorm(vector: RuntimeVector, norm: RuntimeVector, eps = 1e-6): Vector {
  let squareSum = 0;
  for (let index = 0; index < vector.length; index += 1) {
    const value = vector[index] ?? 0;
    squareSum += value * value;
  }
  const meanSquare = squareSum / Math.max(1, vector.length);
  const scale = 1 / Math.sqrt(meanSquare + eps);
  return Array.from(vector, (value, index) => value * scale * (norm[index] ?? 1));
}

function addMatrices(a: Matrix, b: Matrix): Matrix {
  return a.map((row, rowIndex) => {
    const other = b[rowIndex] ?? [];
    return row.map((value, colIndex) => value + (other[colIndex] ?? 0));
  });
}

function addVectors(a: Vector, b: Vector): Vector {
  return a.map((value, index) => value + (b[index] ?? 0));
}

function multiplyVectors(a: Vector, b: Vector): Vector {
  return a.map((value, index) => value * (b[index] ?? 0));
}

function argmax(values: number[]): number {
  if (values.length === 0) return 0;
  let bestIndex = 0;
  let bestValue = values[0] ?? Number.NEGATIVE_INFINITY;
  for (let index = 1; index < values.length; index += 1) {
    const value = values[index] ?? Number.NEGATIVE_INFINITY;
    if (value > bestValue) {
      bestIndex = index;
      bestValue = value;
    }
  }
  return bestIndex;
}

function positiveModulo(value: number, modulus: number): number {
  return ((value % modulus) + modulus) % modulus;
}

function nowMs(): number {
  return typeof performance !== "undefined" ? performance.now() : Date.now();
}

function matrixRowCount(matrix: RuntimeMatrix): number {
  return isPackedRuntimeMatrix(matrix) ? matrix.rowCount : matrix.length;
}

function matrixColCount(matrix: RuntimeMatrix): number {
  if (isPackedRuntimeMatrix(matrix)) return matrix.colCount;
  return matrix[0]?.length ?? 0;
}

function matrixRow(matrix: RuntimeMatrix, rowIndex: number): ArrayLike<number> | undefined {
  return isPackedRuntimeMatrix(matrix) ? matrix.row(rowIndex) : matrix[rowIndex];
}

function validateWeights(weights: UnlockedBrowserTransformerWeights): void {
  if (!weights.modelId.trim()) throw new Error("Unlocked browser transformer weights require a modelId.");
  if (!Number.isInteger(weights.vocabSize) || weights.vocabSize <= 0) throw new Error("Unlocked browser transformer vocabSize must be positive.");
  if (!Number.isInteger(weights.hiddenSize) || weights.hiddenSize <= 0) throw new Error("Unlocked browser transformer hiddenSize must be positive.");
  if (!Number.isInteger(weights.headDim) || weights.headDim <= 0) throw new Error("Unlocked browser transformer headDim must be positive.");
  if (weights.numAttentionHeads !== undefined && (!Number.isInteger(weights.numAttentionHeads) || weights.numAttentionHeads <= 0)) {
    throw new Error("Unlocked browser transformer numAttentionHeads must be positive when provided.");
  }
  if (weights.numKeyValueHeads !== undefined && (!Number.isInteger(weights.numKeyValueHeads) || weights.numKeyValueHeads <= 0)) {
    throw new Error("Unlocked browser transformer numKeyValueHeads must be positive when provided.");
  }
  if (weights.layers.length === 0) throw new Error("Unlocked browser transformer requires at least one layer.");
  assertMatrixShape(weights.tokenEmbedding, weights.vocabSize, weights.hiddenSize, "tokenEmbedding");
  assertMatrixShape(weights.outputProjection, weights.vocabSize, weights.hiddenSize, "outputProjection");
  if (weights.finalNorm) assertVectorShape(weights.finalNorm, weights.hiddenSize, "finalNorm");
  weights.layers.forEach((layer, index) => {
    const geometry = getAttentionGeometry(weights, layer);
    if (layer.inputLayerNorm) assertVectorShape(layer.inputLayerNorm, weights.hiddenSize, `layers[${index}].inputLayerNorm`);
    assertMatrixShape(layer.qProj, geometry.qProjectionSize, weights.hiddenSize, `layers[${index}].qProj`);
    assertMatrixShape(layer.kProj, geometry.kvProjectionSize, weights.hiddenSize, `layers[${index}].kProj`);
    assertMatrixShape(layer.vProj, geometry.kvProjectionSize, weights.hiddenSize, `layers[${index}].vProj`);
    assertMatrixShape(layer.oProj, weights.hiddenSize, geometry.expandedProjectionSize, `layers[${index}].oProj`);
    if (layer.qNorm) assertVectorShape(layer.qNorm, weights.headDim, `layers[${index}].qNorm`);
    if (layer.kNorm) assertVectorShape(layer.kNorm, weights.headDim, `layers[${index}].kNorm`);
    if (layer.postAttentionLayerNorm) assertVectorShape(layer.postAttentionLayerNorm, weights.hiddenSize, `layers[${index}].postAttentionLayerNorm`);
    validateMlpShapes(layer, index, weights.hiddenSize);
  });
}

function getAttentionGeometry(
  weights: Pick<UnlockedBrowserTransformerWeights, "headDim" | "numAttentionHeads" | "numKeyValueHeads">,
  layer: UnlockedBrowserTransformerLayerWeights,
): AttentionGeometry {
  const qRows = matrixRowCount(layer.qProj);
  const kRows = matrixRowCount(layer.kProj);
  const attentionHeads = weights.numAttentionHeads ?? rowsToHeadCount(qRows, weights.headDim, "qProj");
  const keyValueHeads = weights.numKeyValueHeads ?? rowsToHeadCount(kRows, weights.headDim, "kProj");
  if (attentionHeads % keyValueHeads !== 0) {
    throw new Error("Unlocked browser transformer numAttentionHeads must be divisible by numKeyValueHeads for GQA expansion.");
  }
  return {
    attentionHeads,
    keyValueHeads,
    headDim: weights.headDim,
    qProjectionSize: attentionHeads * weights.headDim,
    kvProjectionSize: keyValueHeads * weights.headDim,
    expandedProjectionSize: attentionHeads * weights.headDim,
  };
}

function rowsToHeadCount(rows: number, headDim: number, name: string): number {
  if (rows % headDim !== 0) throw new Error(`${name} rows must be divisible by headDim.`);
  return Math.max(1, rows / headDim);
}

function assertMatrixShape(matrix: RuntimeMatrix, rows: number, cols: number, name: string): void {
  if (matrixRowCount(matrix) !== rows) throw new Error(`${name} must have ${rows} rows.`);
  if (matrixColCount(matrix) !== cols) throw new Error(`${name} must have ${cols} columns.`);
}

function assertVectorShape(vector: RuntimeVector, length: number, name: string): void {
  if (vector.length !== length) throw new Error(`${name} must have ${length} values.`);
}

function validateMlpShapes(layer: UnlockedBrowserTransformerLayerWeights, index: number, hiddenSize: number): void {
  if (!layer.mlpUpProj && !layer.mlpDownProj && !layer.mlpGateProj) return;
  if (!layer.mlpUpProj || !layer.mlpDownProj) {
    throw new Error(`layers[${index}] MLP requires both mlpUpProj and mlpDownProj.`);
  }
  assertMatrixCols(layer.mlpUpProj, hiddenSize, `layers[${index}].mlpUpProj`);
  const intermediateSize = matrixRowCount(layer.mlpUpProj);
  if (layer.mlpGateProj) assertMatrixShape(layer.mlpGateProj, intermediateSize, hiddenSize, `layers[${index}].mlpGateProj`);
  assertMatrixShape(layer.mlpDownProj, hiddenSize, intermediateSize, `layers[${index}].mlpDownProj`);
}

function assertMatrixCols(matrix: RuntimeMatrix, cols: number, name: string): void {
  if (matrixRowCount(matrix) === 0) throw new Error(`${name} must have at least one row.`);
  if (matrixColCount(matrix) !== cols) throw new Error(`${name} must have ${cols} columns.`);
}

function cloneVector(vector: RuntimeVector): RuntimeVector {
  return vector instanceof Float32Array ? new Float32Array(vector) : [...vector];
}

function cloneMatrix(matrix: RuntimeMatrix): RuntimeMatrix {
  return isPackedRuntimeMatrix(matrix) ? matrix : matrix.map((row) => [...row]);
}

function clonePlainMatrix(matrix: Matrix): Matrix {
  return matrix.map((row) => [...row]);
}

function cloneWeights(weights: UnlockedBrowserTransformerWeights): UnlockedBrowserTransformerWeights {
  return {
    ...weights,
    ...(weights.finalNorm ? { finalNorm: cloneVector(weights.finalNorm) } : {}),
    tokenEmbedding: cloneMatrix(weights.tokenEmbedding),
    outputProjection: cloneMatrix(weights.outputProjection),
    layers: weights.layers.map((layer) => ({
      ...(layer.inputLayerNorm ? { inputLayerNorm: cloneVector(layer.inputLayerNorm) } : {}),
      qProj: cloneMatrix(layer.qProj),
      kProj: cloneMatrix(layer.kProj),
      vProj: cloneMatrix(layer.vProj),
      oProj: cloneMatrix(layer.oProj),
      ...(layer.qNorm ? { qNorm: cloneVector(layer.qNorm) } : {}),
      ...(layer.kNorm ? { kNorm: cloneVector(layer.kNorm) } : {}),
      ...(layer.postAttentionLayerNorm ? { postAttentionLayerNorm: cloneVector(layer.postAttentionLayerNorm) } : {}),
      ...(layer.mlpGateProj ? { mlpGateProj: cloneMatrix(layer.mlpGateProj) } : {}),
      ...(layer.mlpUpProj ? { mlpUpProj: cloneMatrix(layer.mlpUpProj) } : {}),
      ...(layer.mlpDownProj ? { mlpDownProj: cloneMatrix(layer.mlpDownProj) } : {}),
    })),
  };
}

function isUnlockedBrowserKvCacheHandle(handle: unknown): handle is UnlockedBrowserKvCacheHandle {
  return isRecord(handle)
    && handle.kind === "unlocked_browser_transformer_kv_cache"
    && typeof handle.modelId === "string"
    && typeof handle.requestId === "string"
    && Array.isArray(handle.tokenIds)
    && Array.isArray(handle.kvBlocks)
    && isRecord(handle.layers);
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}
