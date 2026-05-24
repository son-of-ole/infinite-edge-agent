import {
  destroyWebGpuResidentTensor,
  runDenseMatVecTopKResidentWebGpu,
  uploadWebGpuResidentTensor,
  type DenseMatVecMatrix,
  type WebGpuDenseMatVecTopKResult,
  type WebGpuResidentTensor,
  type WebGpuRuntimeBufferCache,
  type WebGpuSsaBackendOptions,
} from "./webgpuSsaBackend";
import type { Matrix } from "./denseReference";

export interface GpuGreedyLogitProjectionInput {
  hidden: ArrayLike<number> | WebGpuResidentTensor;
  outputProjection: DenseMatVecMatrix;
  suppressedTokenIds?: number[];
  tileRows?: number | null;
  options?: Pick<WebGpuSsaBackendOptions, "backendPreference" | "device" | "gpu" | "requireWebGpu">;
  bufferCache?: WebGpuRuntimeBufferCache;
  projectionCacheKey?: string;
  requestId?: string;
  traceMetadata?: Record<string, unknown>;
  /**
   * Strict decode mode: final hidden must already be GPU-resident.
   * This prevents a hidden-vector CPU -> GPU upload during every generated token.
   */
  requireResidentHidden?: boolean;
}

export interface GpuGreedyLogitProjectionResult {
  tokenId: number;
  score: number;
  backend: "webgpu";
  trace: WebGpuDenseMatVecTopKResult["trace"] & {
    readbackStrategy: "gpu_argmax_token_id";
  };
}

export function shouldUseGpuGreedyLogitProjection(input: {
  backendPreference?: "webgpu" | "cpu";
  requireWebGpu?: boolean;
  hasCandidateTokenIds?: boolean;
  forceGpuGreedyLogits?: boolean;
}): boolean {
  if (input.hasCandidateTokenIds) return false;
  if (input.backendPreference === "cpu") return false;
  return input.forceGpuGreedyLogits === true
    || input.requireWebGpu === true
    || input.backendPreference === "webgpu";
}

export async function projectGreedyDecodeTokenWebGpu(
  input: GpuGreedyLogitProjectionInput,
): Promise<GpuGreedyLogitProjectionResult> {
  const options = input.options ?? {};
  const existingResidentHidden = isResidentTensor(input.hidden) ? input.hidden : undefined;
  if (!existingResidentHidden && input.requireResidentHidden === true) {
    throw new Error(
      "Strict WebGPU decode requires resident final hidden for GPU argmax; refused decode_hidden_resident_upload_for_gpu_argmax.",
    );
  }
  const uploadedHidden = existingResidentHidden
    ? undefined
    : await uploadWebGpuResidentTensor({
        matrix: [Array.from(input.hidden as ArrayLike<number>)] as Matrix,
        ...(options.device ? { device: options.device } : {}),
        ...(options.gpu ? { gpu: options.gpu } : {}),
        ...(options.requireWebGpu ? { requireWebGpu: true } : {}),
        traceMetadata: {
          ...(input.traceMetadata ?? {}),
          purpose: "decode_hidden_resident_upload_for_gpu_argmax",
        },
      });
  const residentHidden = existingResidentHidden ?? uploadedHidden?.tensor;
  if (!residentHidden) throw new Error("GPU greedy logit projection requires a resident hidden vector.");

  try {
    const projected = await runDenseMatVecTopKResidentWebGpu({
      vector: residentHidden,
      matrix: input.outputProjection,
      topK: 1,
      forceFinalTopKReduction: true,
      ...(input.tileRows ? { tileRows: input.tileRows } : {}),
      ...((input.suppressedTokenIds?.length ?? 0) > 0 ? { suppressedRowIds: input.suppressedTokenIds } : {}),
      ...(options.device ? { device: options.device } : {}),
      ...(options.gpu ? { gpu: options.gpu } : {}),
      ...(options.requireWebGpu ? { requireWebGpu: true } : {}),
      ...(input.bufferCache && input.projectionCacheKey
        ? {
            bufferCache: input.bufferCache,
            projectionCacheKey: input.projectionCacheKey,
            projectionCachePolicy: "stable" as const,
          }
        : {}),
      traceMetadata: {
        ...(input.traceMetadata ?? {}),
        purpose: "greedy_argmax_logit_projection",
        gpuArgmaxTokenId: true,
        hiddenResident: Boolean(existingResidentHidden),
        ...(input.requestId ? { requestId: input.requestId } : {}),
      },
    });
    const maybeTokenId = projected.selectedRowIds[0];
    if (typeof maybeTokenId !== "number" || !Number.isInteger(maybeTokenId) || maybeTokenId < 0) {
      throw new Error("GPU greedy logit projection did not return a valid token id.");
    }
    const tokenId = maybeTokenId;
    const score = projected.values[0] ?? Number.NEGATIVE_INFINITY;
    return {
      tokenId,
      score,
      backend: "webgpu",
      trace: {
        ...projected.trace,
        readbackStrategy: "gpu_argmax_token_id",
        readbackRows: 1,
        readbackBytes: Math.min(projected.trace.readbackBytes ?? 8, 16),
        materializedRows: 1,
        metadata: {
          ...(projected.trace.metadata ?? {}),
          gpuArgmaxTokenId: true,
          fullLogitsMaterialized: false,
        },
      },
    };
  } finally {
    if (uploadedHidden) destroyWebGpuResidentTensor(uploadedHidden.tensor);
  }
}

function isResidentTensor(value: unknown): value is WebGpuResidentTensor {
  return typeof value === "object"
    && value !== null
    && (value as WebGpuResidentTensor).kind === "webgpu_resident_tensor";
}
