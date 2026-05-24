import type { WebGpuResidentTensor, WebGpuRuntimeBufferCache, WebGpuSsaBackendOptions } from "../webgpuSsaBackend";

export type FusedDecodeStage =
  | "command_batching"
  | "packed_qkv_projection"
  | "qkv_norm_rope_kv_append"
  | "one_token_attention"
  | "residual_rmsnorm"
  | "swiglu_mlp"
  | "full_layer_plan";

export interface FusedDecodeFeatureFlags {
  commandBatching: boolean;
  packedQkvProjection: boolean;
  qkvNormRopeKvAppend: boolean;
  oneTokenAttention: boolean;
  residualRmsNorm: boolean;
  swigluMlp: boolean;
  fullLayerPlan: boolean;
  requireParityBeforeEnable: boolean;
}

export const DEFAULT_FUSED_DECODE_FLAGS: FusedDecodeFeatureFlags = {
  commandBatching: true,
  packedQkvProjection: false,
  qkvNormRopeKvAppend: false,
  oneTokenAttention: false,
  residualRmsNorm: false,
  swigluMlp: false,
  fullLayerPlan: false,
  requireParityBeforeEnable: true,
};

export interface FusedDecodeRollbackFlags {
  disableCommandBatching?: boolean;
  disablePackedQkvProjection?: boolean;
  disableQkvNormRopeKvAppend?: boolean;
  disableOneTokenAttention?: boolean;
  disableResidualRmsNorm?: boolean;
  disableSwiGluMlp?: boolean;
  disableFullLayerPlan?: boolean;
}

export interface QwenDecodeLayerShape {
  layerIndex: number;
  hiddenSize: number;
  intermediateSize: number;
  numAttentionHeads: number;
  numKeyValueHeads: number;
  headDim: number;
  vocabSize?: number;
  maxSequenceLength?: number;
}

export interface FusedDecodeLayerInput {
  requestId: string;
  tokenIndex: number;
  position: number;
  layerIndex: number;
  hidden: WebGpuResidentTensor;
  shape: QwenDecodeLayerShape;
  options: Pick<WebGpuSsaBackendOptions, "device" | "gpu" | "requireWebGpu" | "backendPreference">;
  bufferCache: WebGpuRuntimeBufferCache;
  flags?: Partial<FusedDecodeFeatureFlags>;
  traceMetadata?: Record<string, unknown>;
}

export interface FusedDecodeLayerOutput {
  hidden: WebGpuResidentTensor;
  trace: FusedDecodeLayerTrace;
}

export interface FusedDecodeLayerTrace {
  requestId: string;
  tokenIndex: number;
  layerIndex: number;
  enabledStages: FusedDecodeStage[];
  fallbackStages: FusedDecodeStage[];
  submitCount: number;
  dispatchCount: number;
  passCount: number;
  residentInput: boolean;
  residentOutput: boolean;
  forbiddenSyncDetected: boolean;
  maxAbsError?: number;
  meanAbsError?: number;
  selectedTokenParity?: boolean;
  labels: string[];
}

export interface WebGpuDecodeCommandBatchTrace {
  requestId: string;
  tokenIndex: number;
  layerIndex?: number;
  passCount: number;
  commandBufferCount?: number;
  dispatchCount: number;
  submitCount: number;
  submitted: boolean;
  labels: string[];
  forbiddenSyncDetected: boolean;
}

export interface FusedLayerParityFailure {
  requestId: string;
  tokenIndex: number;
  layerIndex: number;
  stage: FusedDecodeStage | string;
  maxAbsError: number;
  meanAbsError: number;
  referenceTopTokenId?: number;
  candidateTopTokenId?: number;
  firstBadElement?: number;
  rollbackFlag: keyof FusedDecodeRollbackFlags;
}

export interface FusedDecodeStagePlan {
  stage: FusedDecodeStage;
  enabled: boolean;
  reason: string;
  rollbackFlag: keyof FusedDecodeRollbackFlags;
}

export interface FusedDecodePlan {
  flags: FusedDecodeFeatureFlags;
  stages: FusedDecodeStagePlan[];
  enabledStages: FusedDecodeStage[];
  disabledStages: FusedDecodeStage[];
}

export function mergeFusedDecodeFlags(flags?: Partial<FusedDecodeFeatureFlags>): FusedDecodeFeatureFlags {
  return {
    ...DEFAULT_FUSED_DECODE_FLAGS,
    ...(flags ?? {}),
  };
}
