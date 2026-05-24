import type { SSALayerRoutingPolicy } from "./ssa";
import { denseReferenceAttention, type Matrix } from "./ssa_webgpu/denseReference";
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
import {
  createSsaToyTensorHandle,
  readSsaToyTensorHandle,
  WebGpuSsaReferenceBackend,
  type SsaToyTensorHandle,
  type WebGpuSsaBackendOptions,
} from "./ssa_webgpu/webgpuSsaBackend";

export interface NativeEdgeReferenceBackendOptions extends Pick<WebGpuSsaBackendOptions, "backendPreference" | "device" | "gpu"> {
  headDim?: number;
  denseReferenceTolerance?: number;
}

export interface NativeEdgeLayerTensorHandles {
  qHandle: SsaToyTensorHandle;
  kHandle: SsaToyTensorHandle;
  vHandle: SsaToyTensorHandle;
}

export interface NativeEdgeKvCacheHandle {
  kind: "native_edge_kv_cache";
  id: string;
  modelId: string;
  requestId: string;
  tokenIds: number[];
  blockTokenRanges: Record<string, { tokenStart: number; tokenEnd: number }>;
  layers: Record<number, NativeEdgeLayerTensorHandles>;
}

export interface NativeEdgeDenseReferenceValidation {
  passed: boolean;
  maxAbsDiff: number;
  tolerance: number;
  denseOutput: Matrix;
  sparseOutput: Matrix;
}

export interface NativeEdgeSparseLayerForwardOutput extends SparseLayerForwardOutput {
  selectedBlockIds: string[];
  denseReference?: NativeEdgeDenseReferenceValidation;
}

export interface NativeEdgeDecodeHandle {
  kind: "native_edge_decode_logits";
  id: string;
  requestId: string;
  logits: number[];
}

/**
 * Deterministic native-backend bridge for SSA integration tests and backend swapping.
 *
 * This intentionally uses backend-owned toy tensor handles. It does not claim
 * opaque-runtime native SSA; it proves the app-facing NativeSSABackendContract
 * can drive one layer through the shared SSA sparse-kernel path once a real
 * model backend exposes Q/K/V tensors.
 */
export class NativeEdgeReferenceBackend implements NativeSSABackendContract {
  readonly backendName = "native-edge-reference";
  readonly supportsQkvAccess = true;
  readonly supportsLayerSparseRouting = true;
  readonly supportsPinnedKvBlocks = true;
  readonly supportsDenseReferenceMode = true;

  private readonly kernelBackend: WebGpuSsaReferenceBackend;
  private readonly headDim: number;
  private readonly denseReferenceTolerance: number;
  private modelId: string | null = null;
  private disposed = false;

  constructor(options: NativeEdgeReferenceBackendOptions = {}) {
    this.headDim = options.headDim ?? 4;
    this.denseReferenceTolerance = options.denseReferenceTolerance ?? 1e-6;
    this.kernelBackend = new WebGpuSsaReferenceBackend({
      ...(options.backendPreference ? { backendPreference: options.backendPreference } : {}),
      ...(options.device ? { device: options.device } : {}),
      ...(options.gpu ? { gpu: options.gpu } : {}),
      config: { headDim: this.headDim },
    });
  }

  async initializeModel(modelId: string): Promise<void> {
    this.assertNotDisposed();
    if (modelId.trim().length === 0) throw new Error("NativeEdgeReferenceBackend requires a non-empty modelId.");
    this.modelId = modelId;
  }

  async prefill(inputTokenIds: Int32Array, policy: SSAPrefillPolicy): Promise<SSAPrefillHandle> {
    this.assertReady();
    if (inputTokenIds.length === 0) throw new Error("NativeEdgeReferenceBackend prefill requires at least one token.");
    if (policy.layerPolicies.length === 0) throw new Error("NativeEdgeReferenceBackend prefill requires at least one layer policy.");

    const tokenIds = [...inputTokenIds];
    let cacheBlockTokenRanges: NativeEdgeKvCacheHandle["blockTokenRanges"] | null = null;
    const layers: Record<number, NativeEdgeLayerTensorHandles> = {};
    const traces: SSAKernelTrace[] = [];

    for (let layerPosition = 0; layerPosition < policy.layerPolicies.length; layerPosition += 1) {
      const layerPolicy = policy.layerPolicies[layerPosition] as SSALayerRoutingPolicy | undefined;
      if (!layerPolicy) continue;
      const layerIndex = layerPolicy.layerIndex < 0 ? layerPosition : layerPolicy.layerIndex;
      const blockTokenRanges = buildBlockTokenRanges(tokenIds.length, layerPolicy);
      cacheBlockTokenRanges ??= blockTokenRanges;
      layers[layerIndex] = createLayerTensorHandles({
        requestId: policy.requestId,
        layerIndex,
        tokenIds,
        headDim: this.headDim,
        blockTokenRanges,
      });
      traces.push(buildTrace({
        requestId: policy.requestId,
        layerIndex,
        policy: normalizePolicy(layerPolicy),
        denseTokenCount: tokenIds.length,
      }));
    }

    return {
      requestId: policy.requestId,
      tokenCount: tokenIds.length,
      kvCacheHandle: {
        kind: "native_edge_kv_cache",
        id: `native_edge:${policy.requestId}:kv`,
        modelId: this.modelId as string,
        requestId: policy.requestId,
        tokenIds,
        blockTokenRanges: cacheBlockTokenRanges ?? {},
        layers,
      } satisfies NativeEdgeKvCacheHandle,
      traces,
    };
  }

  async executeSparseLayer(input: SparseLayerForwardInput): Promise<NativeEdgeSparseLayerForwardOutput> {
    this.assertReady();
    const normalizedPolicy = normalizePolicy(input.policy);
    const sparse = await this.kernelBackend.executeSparseForward({
      requestId: input.requestId,
      layerIndex: input.layerIndex,
      qHandle: input.qHandle,
      kHandle: input.kHandle,
      vHandle: input.vHandle,
      routingPolicy: normalizedPolicy,
    });
    const selectedBlockIds = collectSelectedBlockIds(normalizedPolicy);
    const output: NativeEdgeSparseLayerForwardOutput = {
      requestId: sparse.requestId,
      layerIndex: sparse.layerIndex,
      outputHandle: sparse.outputHandle,
      trace: {
        ...(sparse.trace ?? buildTrace({
          requestId: input.requestId,
          layerIndex: input.layerIndex,
          policy: normalizedPolicy,
          denseTokenCount: readSsaToyTensorHandle(input.kHandle).matrix.length,
        })),
        selectedBlockIds,
        pinnedBlockIds: normalizedPolicy.pinnedBlockIds,
      },
      selectedBlockIds,
    };

    if (input.policy.denseFallback) {
      output.denseReference = buildDenseReferenceValidation({
        qHandle: input.qHandle,
        kHandle: input.kHandle,
        vHandle: input.vHandle,
        sparseOutputHandle: output.outputHandle,
        tolerance: this.denseReferenceTolerance,
      });
    }

    return output;
  }

  async decode(input: SSADecodeInput): Promise<SSADecodeOutput> {
    this.assertReady();
    const cache = readNativeEdgeKvCacheHandle(input.kvCacheHandle);
    if (cache.requestId !== input.requestId) {
      throw new Error(`NativeEdgeReferenceBackend KV cache requestId mismatch: cache ${cache.requestId} cannot decode request ${input.requestId}.`);
    }
    if (cache.modelId !== this.modelId) {
      throw new Error(`NativeEdgeReferenceBackend KV cache modelId mismatch: cache ${cache.modelId} cannot decode model ${this.modelId}.`);
    }
    const seed = hashNumbers([...cache.tokenIds, input.inputTokenId, hashString(cache.modelId)]);
    const logits = Array.from({ length: 8 }, (_, index) => deterministicScalar(seed + index * 17, index, 0));
    const tokenId = positiveModulo(Math.round(logits.reduce((sum, value, index) => sum + value * (index + 1), input.inputTokenId)), 32_000);
    const denseTokenCount = cache.tokenIds.length + 1;

    return {
      requestId: input.requestId,
      tokenId,
      logitsHandle: {
        kind: "native_edge_decode_logits",
        id: `native_edge:${input.requestId}:decode:${input.inputTokenId}`,
        requestId: input.requestId,
        logits,
      } satisfies NativeEdgeDecodeHandle,
      traces: input.policy.map((policy, index) => buildTrace({
        requestId: input.requestId,
        layerIndex: policy.layerIndex < 0 ? index : policy.layerIndex,
        policy: normalizePolicy(policy),
        denseTokenCount,
      })),
    };
  }

  async dispose(): Promise<void> {
    this.disposed = true;
    this.modelId = null;
  }

  private assertReady(): void {
    this.assertNotDisposed();
    if (!this.modelId) throw new Error("NativeEdgeReferenceBackend must initializeModel(modelId) before use.");
  }

  private assertNotDisposed(): void {
    if (this.disposed) throw new Error("NativeEdgeReferenceBackend has been disposed.");
  }
}

export function readNativeEdgeKvCacheHandle(handle: unknown): NativeEdgeKvCacheHandle {
  if (!isNativeEdgeKvCacheHandle(handle)) {
    throw new Error("Expected a NativeEdgeReferenceBackend KV cache handle.");
  }
  return handle;
}

export function getNativeEdgeLayerTensorHandles(handle: unknown, layerIndex: number): NativeEdgeLayerTensorHandles {
  const cache = readNativeEdgeKvCacheHandle(handle);
  const layer = cache.layers[layerIndex];
  if (!layer) throw new Error(`NativeEdgeReferenceBackend has no Q/K/V handles for layer ${layerIndex}.`);
  return layer;
}

export function readNativeEdgeDecodeHandle(handle: unknown): NativeEdgeDecodeHandle {
  if (!isNativeEdgeDecodeHandle(handle)) {
    throw new Error("Expected a NativeEdgeReferenceBackend decode logits handle.");
  }
  return handle;
}

function createLayerTensorHandles(input: {
  requestId: string;
  layerIndex: number;
  tokenIds: number[];
  headDim: number;
  blockTokenRanges: NativeEdgeKvCacheHandle["blockTokenRanges"];
}): NativeEdgeLayerTensorHandles {
  const qMatrix = input.tokenIds.map((tokenId, tokenIndex) => (
    Array.from({ length: input.headDim }, (_, dim) => deterministicScalar(tokenId, tokenIndex, input.layerIndex + dim))
  ));
  const vMatrix = input.tokenIds.map((tokenId, tokenIndex) => (
    Array.from({ length: input.headDim }, (_, dim) => deterministicScalar(tokenId + 31, tokenIndex, input.layerIndex + dim + 11))
  ));
  return {
    qHandle: createSsaToyTensorHandle({
      id: `native_edge:${input.requestId}:layer${input.layerIndex}:q`,
      matrix: qMatrix,
      blockTokenRanges: input.blockTokenRanges,
    }),
    kHandle: createSsaToyTensorHandle({
      id: `native_edge:${input.requestId}:layer${input.layerIndex}:k`,
      matrix: qMatrix,
      blockTokenRanges: input.blockTokenRanges,
    }),
    vHandle: createSsaToyTensorHandle({
      id: `native_edge:${input.requestId}:layer${input.layerIndex}:v`,
      matrix: vMatrix,
      blockTokenRanges: input.blockTokenRanges,
    }),
  };
}

function buildBlockTokenRanges(
  tokenCount: number,
  layerPolicy: SSALayerRoutingPolicy,
): NativeEdgeKvCacheHandle["blockTokenRanges"] {
  const blockSize = Math.max(1, layerPolicy.blockSize);
  const blockIds = collectKnownBlockIds([layerPolicy]);
  const blockCount = Math.max(blockIds.length, Math.ceil(tokenCount / blockSize));
  const ranges: NativeEdgeKvCacheHandle["blockTokenRanges"] = {};
  for (let blockIndex = 0; blockIndex < blockCount; blockIndex += 1) {
    const id = blockIds[blockIndex] ?? `b${blockIndex}`;
    ranges[id] = {
      tokenStart: Math.min(tokenCount, blockIndex * blockSize),
      tokenEnd: Math.min(tokenCount, (blockIndex + 1) * blockSize),
    };
  }
  return ranges;
}

function collectKnownBlockIds(layerPolicies: SSALayerRoutingPolicy[]): string[] {
  const ids: string[] = [];
  for (const policy of layerPolicies) {
    for (const id of policy.pinnedBlockIds) pushUnique(ids, id);
    for (const selected of Object.values(policy.selectedBlockIdsByQueryBlock)) {
      for (const id of selected) pushUnique(ids, id);
    }
  }
  return ids.sort(compareBlockIds);
}

function normalizePolicy(policy: SSALayerRoutingPolicy): SSALayerRoutingPolicy {
  const selectedBlockIdsByQueryBlock: Record<number, string[]> = {};
  for (const [queryBlockIndex, selected] of Object.entries(policy.selectedBlockIdsByQueryBlock)) {
    const merged: string[] = [];
    for (const id of policy.pinnedBlockIds) pushUnique(merged, id);
    for (const id of selected) pushUnique(merged, id);
    selectedBlockIdsByQueryBlock[Number(queryBlockIndex)] = merged;
  }
  if (Object.keys(selectedBlockIdsByQueryBlock).length === 0) {
    selectedBlockIdsByQueryBlock[0] = [...policy.pinnedBlockIds];
  }
  return {
    ...policy,
    pinnedBlockIds: [...policy.pinnedBlockIds],
    selectedBlockIdsByQueryBlock,
  };
}

function buildDenseReferenceValidation(input: {
  qHandle: unknown;
  kHandle: unknown;
  vHandle: unknown;
  sparseOutputHandle: unknown;
  tolerance: number;
}): NativeEdgeDenseReferenceValidation {
  const q = readSsaToyTensorHandle(input.qHandle);
  const k = readSsaToyTensorHandle(input.kHandle);
  const v = readSsaToyTensorHandle(input.vHandle);
  const sparseOutput = readSsaToyTensorHandle(input.sparseOutputHandle).matrix;
  const denseOutput = denseReferenceAttention(q.matrix, k.matrix, v.matrix);
  const maxAbsDiff = matrixMaxAbsDiff(denseOutput, sparseOutput);
  return {
    passed: maxAbsDiff <= input.tolerance,
    maxAbsDiff,
    tolerance: input.tolerance,
    denseOutput,
    sparseOutput,
  };
}

function buildTrace(input: {
  requestId: string;
  layerIndex: number;
  policy: SSALayerRoutingPolicy;
  denseTokenCount: number;
}): SSAKernelTrace {
  return {
    requestId: input.requestId,
    layerIndex: input.layerIndex,
    queryBlockIndex: 0,
    selectedBlockIds: collectSelectedBlockIds(input.policy),
    pinnedBlockIds: input.policy.pinnedBlockIds,
    denseTokenCountEstimate: input.denseTokenCount,
    sparseTokenCountEstimate: estimateSparseTokens(input.policy),
    routingMs: 0,
    gatherMs: 0,
    attentionMs: 0,
  };
}

function collectSelectedBlockIds(policy: SSALayerRoutingPolicy): string[] {
  const ids: string[] = [];
  for (const selected of Object.values(policy.selectedBlockIdsByQueryBlock)) {
    for (const id of selected) pushUnique(ids, id);
  }
  for (const id of policy.pinnedBlockIds) pushUnique(ids, id);
  return ids.sort(compareBlockIds);
}

function estimateSparseTokens(policy: SSALayerRoutingPolicy): number {
  let total = 0;
  for (const selected of Object.values(policy.selectedBlockIdsByQueryBlock)) {
    total += new Set(selected).size * policy.blockSize;
  }
  return total;
}

function matrixMaxAbsDiff(a: Matrix, b: Matrix): number {
  let max = 0;
  for (let row = 0; row < Math.max(a.length, b.length); row += 1) {
    const aRow = a[row] ?? [];
    const bRow = b[row] ?? [];
    for (let col = 0; col < Math.max(aRow.length, bRow.length); col += 1) {
      max = Math.max(max, Math.abs((aRow[col] ?? 0) - (bRow[col] ?? 0)));
    }
  }
  return max;
}

function deterministicScalar(tokenId: number, tokenIndex: number, channel: number): number {
  const value = positiveModulo((tokenId + 1) * 17 + (tokenIndex + 1) * 31 + (channel + 1) * 13, 97);
  return Number(((value + 1) / 97).toFixed(6));
}

function hashNumbers(values: number[]): number {
  return values.reduce((hash, value) => positiveModulo(hash * 33 + Math.trunc(value), 2_147_483_647), 5381);
}

function hashString(value: string): number {
  return hashNumbers([...value].map((char) => char.charCodeAt(0)));
}

function positiveModulo(value: number, modulo: number): number {
  return ((value % modulo) + modulo) % modulo;
}

function pushUnique(values: string[], value: string): void {
  if (!values.includes(value)) values.push(value);
}

function compareBlockIds(a: string, b: string): number {
  const aNumber = Number(a.match(/\d+$/)?.[0]);
  const bNumber = Number(b.match(/\d+$/)?.[0]);
  if (Number.isFinite(aNumber) && Number.isFinite(bNumber) && aNumber !== bNumber) return aNumber - bNumber;
  return a.localeCompare(b);
}

function isNativeEdgeKvCacheHandle(handle: unknown): handle is NativeEdgeKvCacheHandle {
  return typeof handle === "object"
    && handle !== null
    && (handle as NativeEdgeKvCacheHandle).kind === "native_edge_kv_cache"
    && typeof (handle as NativeEdgeKvCacheHandle).id === "string"
    && typeof (handle as NativeEdgeKvCacheHandle).modelId === "string"
    && typeof (handle as NativeEdgeKvCacheHandle).requestId === "string"
    && Array.isArray((handle as NativeEdgeKvCacheHandle).tokenIds)
    && typeof (handle as NativeEdgeKvCacheHandle).layers === "object"
    && (handle as NativeEdgeKvCacheHandle).layers !== null;
}

function isNativeEdgeDecodeHandle(handle: unknown): handle is NativeEdgeDecodeHandle {
  return typeof handle === "object"
    && handle !== null
    && (handle as NativeEdgeDecodeHandle).kind === "native_edge_decode_logits"
    && typeof (handle as NativeEdgeDecodeHandle).id === "string"
    && typeof (handle as NativeEdgeDecodeHandle).requestId === "string"
    && Array.isArray((handle as NativeEdgeDecodeHandle).logits);
}
