export interface DecodePerfSummary {
  requestId?: string;
  generatedTokenCount: number;
  decodeCallCount: number;
  decodeSubmitCount: number;
  dispatchCount: number;
  decodeSubmitCountPerToken: number;
  decodeDispatchCountPerToken: number;
  decodeDispatchCountPerLayerPerToken: number;
  readbackCount: number;
  totalReadbackRows: number;
  totalReadbackBytes: number;
  fullLogitsReadbackCount: number;
  compactLogitReadbackCount: number;
  weightUploadBytesDuringDecode: number;
  weightUploadCountDuringDecode: number;
  activationUploadBytesDuringDecode: number;
  activationUploadCountDuringDecode: number;
  hiddenReadbackCountDuringDecode: number;
  f32ExpansionCountDuringDecode: number;
  f32ExpansionBytesDuringDecode: number;
  cpuFallbackUsed: boolean;
  cpuValidationUsed: boolean;
  prefillExecutionsDuringDecode: number;
  prefillCountPerGeneratedToken: number;
  residentDecodeLayerCount: number;
  totalDecodeLayerCount: number;
  residentDecodeLayerCoverage: number;
  residentFinalHiddenUsedForLogits: boolean;
  kvDecodeReused: boolean;
  fusedPackedQkvLayerCount: number;
  fusedQkvNormRopeKvAppendLayerCount: number;
  fusedOneTokenAttentionLayerCount: number;
  fusedResidualRmsNormLayerCount: number;
  fusedMlpLayerCount: number;
  fusedFullLayerCount: number;
  fusedLayerCoverage: number;
  tokensPerSecond: number | null;
}

export interface DecodePerfGateResult {
  passed: boolean;
  reasons: string[];
  summary: DecodePerfSummary;
}

export interface DecodePerfGateOptions {
  minTokensPerSecond?: number;
  requireNoCpuFallback?: boolean;
  requireNoCpuValidation?: boolean;
  requireNoDecodeWeightUploads?: boolean;
  requireNoDecodeActivationUploads?: boolean;
  requireNoHiddenReadbacks?: boolean;
  requireNoDecodeF32Expansion?: boolean;
  requireNoFullLogitsReadback?: boolean;
  requireNoPrefillDuringDecode?: boolean;
  requireResidentDecodeCoverage?: boolean;
  requireResidentFinalHiddenForLogits?: boolean;
}

export interface DecodePerfTracerInit {
  requestId?: string;
}

export interface DecodePerfSnapshotOptions {
  tokensPerSecond?: number | null;
}

export interface DecodeHotPathProofInput {
  requestId?: string;
  tokensPerSecond?: number | null;
  generatedTokens?: number;
  decodeCallCount?: number;
  logitProjectionBackend?: "webgpu" | "cpu_reference" | string;
  logitProjectionReadbackStrategy?: "full_logits" | "gpu_top1_candidates" | "gpu_argmax_token_id" | "gpu_compact_topk" | string;
  logitProjectionReadbackRows?: number;
  logitProjectionReadbackBytes?: number;
  logitProjectionDispatchCount?: number;
  decodeSubmitCount?: number;
  cpuFallbackUsed?: boolean;
  cpuValidationUsed?: boolean;
  weightUploadBytesDuringDecode?: number;
  weightUploadCountDuringDecode?: number;
  activationUploadBytesDuringDecode?: number;
  activationUploadCountDuringDecode?: number;
  hiddenReadbackCountDuringDecode?: number;
  f32ExpansionCountDuringDecode?: number;
  f32ExpansionBytesDuringDecode?: number;
  prefillExecutionsDuringDecode?: number;
  residentDecodeLayerCount?: number;
  totalDecodeLayerCount?: number;
  residentFinalHiddenUsedForLogits?: boolean;
  kvDecodeReused?: boolean;
  fusedPackedQkvLayerCount?: number;
  fusedQkvNormRopeKvAppendLayerCount?: number;
  fusedOneTokenAttentionLayerCount?: number;
  fusedResidualRmsNormLayerCount?: number;
  fusedMlpLayerCount?: number;
  fusedFullLayerCount?: number;
}
