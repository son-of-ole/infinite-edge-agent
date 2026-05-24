import { buildFusedDecodePlan, fusedStageEnabled } from "./fusedDecodeLayerPlan";
import { WebGpuDecodeCommandBatch, type WebGpuDeviceLike } from "./commandBatcher";
import type { FusedDecodeLayerInput, FusedDecodeLayerOutput, FusedDecodeLayerTrace } from "./types";

/**
 * v10 fused layer coordinator.
 *
 * This is intentionally a coordinator scaffold, not a drop-in replacement for the existing v9 kernels.
 * Wire each record* callback to the existing resident WebGPU helpers first, then replace callback internals
 * with the WGSL kernels in ./wgsl one stage at a time.
 */
export interface QwenFusedDecodeLayerOps {
  recordInputRmsNorm: (batch: WebGpuDecodeCommandBatch, input: FusedDecodeLayerInput) => void;
  recordPackedQkvProjection?: (batch: WebGpuDecodeCommandBatch, input: FusedDecodeLayerInput) => void;
  recordSeparateQkvProjection: (batch: WebGpuDecodeCommandBatch, input: FusedDecodeLayerInput) => void;
  recordQkvNormRopeKvAppend?: (batch: WebGpuDecodeCommandBatch, input: FusedDecodeLayerInput) => void;
  recordSeparateQkvNormRopeKvAppend: (batch: WebGpuDecodeCommandBatch, input: FusedDecodeLayerInput) => void;
  recordOneTokenAttention?: (batch: WebGpuDecodeCommandBatch, input: FusedDecodeLayerInput) => void;
  recordSeparateAttention: (batch: WebGpuDecodeCommandBatch, input: FusedDecodeLayerInput) => void;
  recordOProjection: (batch: WebGpuDecodeCommandBatch, input: FusedDecodeLayerInput) => void;
  recordResidualRmsNorm?: (batch: WebGpuDecodeCommandBatch, input: FusedDecodeLayerInput) => void;
  recordSeparateResidualRmsNorm: (batch: WebGpuDecodeCommandBatch, input: FusedDecodeLayerInput) => void;
  recordSwiGluMlp?: (batch: WebGpuDecodeCommandBatch, input: FusedDecodeLayerInput) => void;
  recordSeparateMlp: (batch: WebGpuDecodeCommandBatch, input: FusedDecodeLayerInput) => void;
  getOutputHidden: (input: FusedDecodeLayerInput) => FusedDecodeLayerOutput["hidden"];
}

export async function runQwenFusedDecodeLayerWebGpu(
  input: FusedDecodeLayerInput,
  ops: QwenFusedDecodeLayerOps,
): Promise<FusedDecodeLayerOutput> {
  const plan = buildFusedDecodePlan(input.flags ? { flags: input.flags } : {});
  const enabledStages = plan.enabledStages;
  const fallbackStages = plan.disabledStages;
  const device = input.options.device;
  if (!isWebGpuDeviceLike(device)) throw new Error("runQwenFusedDecodeLayerWebGpu requires a GPUDevice.");

  const batch = new WebGpuDecodeCommandBatch(device, {
    requestId: input.requestId,
    tokenIndex: input.tokenIndex,
    layerIndex: input.layerIndex,
    label: `qwen-fused-decode-layer:${input.layerIndex}`,
  });

  ops.recordInputRmsNorm(batch, input);

  if (fusedStageEnabled(plan, "packed_qkv_projection") && ops.recordPackedQkvProjection) {
    ops.recordPackedQkvProjection(batch, input);
  } else {
    ops.recordSeparateQkvProjection(batch, input);
  }

  if (fusedStageEnabled(plan, "qkv_norm_rope_kv_append") && ops.recordQkvNormRopeKvAppend) {
    ops.recordQkvNormRopeKvAppend(batch, input);
  } else {
    ops.recordSeparateQkvNormRopeKvAppend(batch, input);
  }

  if (fusedStageEnabled(plan, "one_token_attention") && ops.recordOneTokenAttention) {
    ops.recordOneTokenAttention(batch, input);
  } else {
    ops.recordSeparateAttention(batch, input);
  }

  ops.recordOProjection(batch, input);

  if (fusedStageEnabled(plan, "residual_rmsnorm") && ops.recordResidualRmsNorm) {
    ops.recordResidualRmsNorm(batch, input);
  } else {
    ops.recordSeparateResidualRmsNorm(batch, input);
  }

  if (fusedStageEnabled(plan, "swiglu_mlp") && ops.recordSwiGluMlp) {
    ops.recordSwiGluMlp(batch, input);
  } else {
    ops.recordSeparateMlp(batch, input);
  }

  const batchTrace = await batch.submitOnce();
  const trace: FusedDecodeLayerTrace = {
    requestId: input.requestId,
    tokenIndex: input.tokenIndex,
    layerIndex: input.layerIndex,
    enabledStages,
    fallbackStages,
    submitCount: batchTrace.submitCount,
    dispatchCount: batchTrace.dispatchCount,
    passCount: batchTrace.passCount,
    residentInput: input.hidden.kind === "webgpu_resident_tensor",
    residentOutput: true,
    forbiddenSyncDetected: batchTrace.forbiddenSyncDetected,
    labels: batchTrace.labels,
  };

  return {
    hidden: ops.getOutputHidden(input),
    trace,
  };
}

function isWebGpuDeviceLike(value: unknown): value is WebGpuDeviceLike {
  return typeof value === "object"
    && value !== null
    && "queue" in value
    && "createCommandEncoder" in value
    && typeof (value as { createCommandEncoder?: unknown }).createCommandEncoder === "function"
    && typeof (value as { queue?: { submit?: unknown } }).queue?.submit === "function";
}
