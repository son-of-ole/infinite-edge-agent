import type { DecodePerfGateOptions, DecodePerfGateResult, DecodePerfSummary } from "./decodePerfTypes";

export const DEFAULT_BROWSER_DECODE_SPEED_GATE: Required<DecodePerfGateOptions> = {
  minTokensPerSecond: 1,
  requireNoCpuFallback: true,
  requireNoCpuValidation: true,
  requireNoDecodeWeightUploads: true,
  requireNoDecodeActivationUploads: true,
  requireNoHiddenReadbacks: true,
  requireNoDecodeF32Expansion: true,
  requireNoFullLogitsReadback: true,
  requireNoPrefillDuringDecode: true,
  requireResidentDecodeCoverage: true,
  requireResidentFinalHiddenForLogits: true,
};

export function evaluateDecodeSpeedGate(
  summary: DecodePerfSummary,
  options: DecodePerfGateOptions = {},
): DecodePerfGateResult {
  const gate = { ...DEFAULT_BROWSER_DECODE_SPEED_GATE, ...options };
  const reasons: string[] = [];
  if (summary.tokensPerSecond !== null && summary.tokensPerSecond < gate.minTokensPerSecond) {
    reasons.push(`tokens_per_second_below_${formatGateNumber(gate.minTokensPerSecond)}`);
  }
  if (gate.requireNoCpuFallback && summary.cpuFallbackUsed) reasons.push("cpu_fallback_used");
  if (gate.requireNoCpuValidation && summary.cpuValidationUsed) reasons.push("cpu_validation_used");
  if (gate.requireNoDecodeWeightUploads && summary.weightUploadBytesDuringDecode > 0) reasons.push("decode_weight_upload");
  if (gate.requireNoDecodeActivationUploads && summary.activationUploadBytesDuringDecode > 0) reasons.push("decode_activation_upload");
  if (gate.requireNoHiddenReadbacks && summary.hiddenReadbackCountDuringDecode > 0) reasons.push("hidden_readback");
  if (gate.requireNoDecodeF32Expansion && summary.f32ExpansionCountDuringDecode > 0) reasons.push("decode_f32_expansion");
  if (gate.requireNoFullLogitsReadback && summary.fullLogitsReadbackCount > 0) reasons.push("full_logits_readback");
  if (gate.requireNoPrefillDuringDecode && summary.prefillCountPerGeneratedToken > 0) reasons.push("prefill_during_decode");
  if (gate.requireResidentDecodeCoverage && summary.residentDecodeLayerCoverage < 1) reasons.push("resident_decode_coverage_below_1");
  if (gate.requireResidentFinalHiddenForLogits && summary.residentFinalHiddenUsedForLogits !== true) {
    reasons.push("resident_final_hidden_not_used");
  }
  return {
    passed: reasons.length === 0,
    reasons,
    summary,
  };
}

function formatGateNumber(value: number): string {
  return Number.isInteger(value) ? String(value) : String(value).replace(/[^0-9]+/g, "_");
}
