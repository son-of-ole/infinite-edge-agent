import type { FusedDecodeLayerTrace, FusedLayerParityFailure, FusedDecodeStage } from "./types";

export interface NumericVectorLike {
  readonly length: number;
  [index: number]: number;
}

export interface FusedLayerParityInput {
  requestId: string;
  tokenIndex: number;
  layerIndex: number;
  stage: FusedDecodeStage | string;
  reference: NumericVectorLike;
  candidate: NumericVectorLike;
  referenceTopTokenId?: number;
  candidateTopTokenId?: number;
  maxAbsErrorThreshold?: number;
  meanAbsErrorThreshold?: number;
}

export interface FusedLayerParityResult {
  passed: boolean;
  maxAbsError: number;
  meanAbsError: number;
  firstBadElement?: number;
  failure?: FusedLayerParityFailure;
}

export function compareFusedLayerVectors(input: FusedLayerParityInput): FusedLayerParityResult {
  if (input.reference.length !== input.candidate.length) {
    return buildFailure(input, Number.POSITIVE_INFINITY, Number.POSITIVE_INFINITY, 0);
  }
  let maxAbsError = 0;
  let sumAbsError = 0;
  let firstBadElement: number | undefined;
  const maxThreshold = input.maxAbsErrorThreshold ?? 1.5e-1;
  const meanThreshold = input.meanAbsErrorThreshold ?? 5e-2;

  for (let index = 0; index < input.reference.length; index += 1) {
    const ref = input.reference[index] ?? 0;
    const got = input.candidate[index] ?? 0;
    const err = Math.abs(ref - got);
    if (!Number.isFinite(err) && firstBadElement === undefined) firstBadElement = index;
    if (err > maxAbsError) maxAbsError = err;
    sumAbsError += Number.isFinite(err) ? err : Number.POSITIVE_INFINITY;
    if (firstBadElement === undefined && err > maxThreshold) firstBadElement = index;
  }

  const meanAbsError = input.reference.length === 0 ? 0 : sumAbsError / input.reference.length;
  const tokenMismatch = input.referenceTopTokenId !== undefined
    && input.candidateTopTokenId !== undefined
    && input.referenceTopTokenId !== input.candidateTopTokenId;
  const passed = maxAbsError <= maxThreshold && meanAbsError <= meanThreshold && !tokenMismatch;

  if (passed) return { passed, maxAbsError, meanAbsError };
  return buildFailure(input, maxAbsError, meanAbsError, firstBadElement);
}

export function attachParityToTrace(
  trace: FusedDecodeLayerTrace,
  parity: FusedLayerParityResult,
): FusedDecodeLayerTrace {
  return {
    ...trace,
    maxAbsError: parity.maxAbsError,
    meanAbsError: parity.meanAbsError,
    selectedTokenParity: parity.failure?.referenceTopTokenId === undefined
      || parity.failure.candidateTopTokenId === undefined
      || parity.failure.referenceTopTokenId === parity.failure.candidateTopTokenId,
  };
}

function buildFailure(
  input: FusedLayerParityInput,
  maxAbsError: number,
  meanAbsError: number,
  firstBadElement?: number,
): FusedLayerParityResult {
  return {
    passed: false,
    maxAbsError,
    meanAbsError,
    ...(firstBadElement !== undefined ? { firstBadElement } : {}),
    failure: {
      requestId: input.requestId,
      tokenIndex: input.tokenIndex,
      layerIndex: input.layerIndex,
      stage: input.stage,
      maxAbsError,
      meanAbsError,
      ...(input.referenceTopTokenId !== undefined ? { referenceTopTokenId: input.referenceTopTokenId } : {}),
      ...(input.candidateTopTokenId !== undefined ? { candidateTopTokenId: input.candidateTopTokenId } : {}),
      ...(firstBadElement !== undefined ? { firstBadElement } : {}),
      rollbackFlag: rollbackFlagForStage(input.stage),
    },
  };
}

function rollbackFlagForStage(stage: FusedDecodeStage | string): FusedLayerParityFailure["rollbackFlag"] {
  switch (stage) {
    case "command_batching": return "disableCommandBatching";
    case "packed_qkv_projection": return "disablePackedQkvProjection";
    case "qkv_norm_rope_kv_append": return "disableQkvNormRopeKvAppend";
    case "one_token_attention": return "disableOneTokenAttention";
    case "residual_rmsnorm": return "disableResidualRmsNorm";
    case "swiglu_mlp": return "disableSwiGluMlp";
    case "full_layer_plan": return "disableFullLayerPlan";
    default: return "disableFullLayerPlan";
  }
}
