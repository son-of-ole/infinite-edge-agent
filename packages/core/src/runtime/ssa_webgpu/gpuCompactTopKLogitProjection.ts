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

export interface GpuCompactTopKLogitProjectionInput {
  hidden: ArrayLike<number> | WebGpuResidentTensor;
  outputProjection: DenseMatVecMatrix;
  topK: number;
  suppressedTokenIds?: number[];
  tileRows?: number | null;
  options?: Pick<WebGpuSsaBackendOptions, "backendPreference" | "device" | "gpu" | "requireWebGpu">;
  bufferCache?: WebGpuRuntimeBufferCache;
  projectionCacheKey?: string;
  requestId?: string;
  traceMetadata?: Record<string, unknown>;
  requireResidentHidden?: boolean;
}

export interface GpuCompactTopKLogitProjectionResult {
  candidates: Array<{ tokenId: number; score: number }>;
  backend: "webgpu";
  trace: WebGpuDenseMatVecTopKResult["trace"] & {
    readbackStrategy: "gpu_compact_topk" | "gpu_argmax_token_id";
    compactTopK: number;
  };
}

export function normalizeCompactLogitTopK(value: number | null | undefined, fallback = 40): number {
  if (typeof value !== "number" || !Number.isFinite(value)) return fallback;
  return Math.max(1, Math.min(256, Math.floor(value)));
}

export async function projectCompactTopKDecodeTokensWebGpu(
  input: GpuCompactTopKLogitProjectionInput,
): Promise<GpuCompactTopKLogitProjectionResult> {
  const options = input.options ?? {};
  const topK = normalizeCompactLogitTopK(input.topK);
  const existingResidentHidden = isResidentTensor(input.hidden) ? input.hidden : undefined;

  if (!existingResidentHidden && input.requireResidentHidden === true) {
    throw new Error("Strict WebGPU decode requires resident final hidden for compact top-k logit projection.");
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
          purpose: "decode_hidden_resident_upload_for_gpu_compact_topk",
        },
      });
  const residentHidden = existingResidentHidden ?? uploadedHidden?.tensor;
  if (!residentHidden) throw new Error("GPU compact top-k logit projection requires a resident hidden vector.");

  try {
    const projected = await runDenseMatVecTopKResidentWebGpu({
      vector: residentHidden,
      matrix: input.outputProjection,
      topK,
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
        purpose: topK === 1 ? "greedy_argmax_logit_projection" : "compact_topk_logit_projection",
        gpuCompactTopK: topK,
        hiddenResident: Boolean(existingResidentHidden),
        ...(input.requestId ? { requestId: input.requestId } : {}),
      },
    });

    const candidates = projected.selectedRowIds.map((tokenId, index) => ({
      tokenId,
      score: projected.values[index] ?? Number.NEGATIVE_INFINITY,
    }));
    if (candidates.length === 0 || candidates.some((candidate) => !Number.isInteger(candidate.tokenId) || candidate.tokenId < 0)) {
      throw new Error("GPU compact top-k logit projection returned invalid token candidates.");
    }

    return {
      candidates,
      backend: "webgpu",
      trace: {
        ...projected.trace,
        readbackStrategy: topK === 1 ? "gpu_argmax_token_id" : "gpu_compact_topk",
        compactTopK: topK,
        readbackRows: candidates.length,
        readbackBytes: Math.min(projected.trace.readbackBytes ?? candidates.length * 8, candidates.length * 16),
        materializedRows: candidates.length,
        metadata: {
          ...(projected.trace.metadata ?? {}),
          gpuCompactTopK: topK,
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
