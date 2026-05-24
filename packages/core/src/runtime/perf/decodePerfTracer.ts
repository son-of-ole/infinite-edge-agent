import type {
  DecodeHotPathProofInput,
  DecodePerfSnapshotOptions,
  DecodePerfSummary,
  DecodePerfTracerInit,
} from "./decodePerfTypes";

export class DecodePerfTracer {
  private generatedTokenCount = 0;
  private decodeCallCount = 0;
  private decodeSubmitCount = 0;
  private dispatchCount = 0;
  private readbackCount = 0;
  private totalReadbackRows = 0;
  private totalReadbackBytes = 0;
  private fullLogitsReadbackCount = 0;
  private compactLogitReadbackCount = 0;
  private weightUploadBytesDuringDecode = 0;
  private weightUploadCountDuringDecode = 0;
  private activationUploadBytesDuringDecode = 0;
  private activationUploadCountDuringDecode = 0;
  private hiddenReadbackCountDuringDecode = 0;
  private f32ExpansionCountDuringDecode = 0;
  private f32ExpansionBytesDuringDecode = 0;
  private cpuFallbackUsed = false;
  private cpuValidationUsed = false;
  private prefillExecutionsDuringDecode = 0;
  private residentDecodeLayerCount = 0;
  private totalDecodeLayerCount = 0;
  private residentFinalHiddenUsedForLogits = false;
  private kvDecodeReused = false;
  private fusedPackedQkvLayerCount = 0;
  private fusedQkvNormRopeKvAppendLayerCount = 0;
  private fusedOneTokenAttentionLayerCount = 0;
  private fusedResidualRmsNormLayerCount = 0;
  private fusedMlpLayerCount = 0;
  private fusedFullLayerCount = 0;

  constructor(private readonly init: DecodePerfTracerInit = {}) {}

  recordGeneratedToken(count = 1): void {
    this.generatedTokenCount += nonNegativeInteger(count);
  }

  recordDecodeCall(count = 1): void {
    this.decodeCallCount += nonNegativeInteger(count);
  }

  recordSubmits(count: number): void {
    this.decodeSubmitCount += nonNegativeInteger(count);
  }

  recordDispatches(count: number): void {
    this.dispatchCount += nonNegativeInteger(count);
  }

  recordReadback(input: { rows?: number; bytes?: number }): void {
    this.readbackCount += 1;
    this.totalReadbackRows += nonNegativeInteger(input.rows ?? 0);
    this.totalReadbackBytes += nonNegativeInteger(input.bytes ?? 0);
  }

  recordFullLogitsReadback(input: { rows?: number; bytes?: number }): void {
    this.fullLogitsReadbackCount += 1;
    this.recordReadback(input);
  }

  recordCompactLogitReadback(input: { rows?: number; bytes?: number }): void {
    this.compactLogitReadbackCount += 1;
    this.recordReadback(input);
  }

  recordWeightUpload(input: { bytes?: number; reason?: string }): void {
    this.weightUploadCountDuringDecode += 1;
    this.weightUploadBytesDuringDecode += nonNegativeInteger(input.bytes ?? 0);
    void input.reason;
  }

  recordActivationUpload(input: { bytes?: number; reason?: string }): void {
    this.activationUploadCountDuringDecode += 1;
    this.activationUploadBytesDuringDecode += nonNegativeInteger(input.bytes ?? 0);
    void input.reason;
  }

  recordHiddenReadback(count = 1): void {
    this.hiddenReadbackCountDuringDecode += nonNegativeInteger(count);
  }

  recordF32Expansion(input: { bytes?: number; reason?: string }): void {
    this.f32ExpansionCountDuringDecode += 1;
    this.f32ExpansionBytesDuringDecode += nonNegativeInteger(input.bytes ?? 0);
    void input.reason;
  }

  recordCpuFallback(): void {
    this.cpuFallbackUsed = true;
  }

  recordCpuValidation(): void {
    this.cpuValidationUsed = true;
  }

  recordPrefillDuringDecode(count = 1): void {
    this.prefillExecutionsDuringDecode += nonNegativeInteger(count);
  }

  recordKvDecodeReuse(): void {
    this.kvDecodeReused = true;
  }

  recordResidentDecodeLayers(input: { residentLayerCount?: number; totalLayerCount?: number }): void {
    this.residentDecodeLayerCount += nonNegativeInteger(input.residentLayerCount ?? 0);
    this.totalDecodeLayerCount += nonNegativeInteger(input.totalLayerCount ?? 0);
  }

  recordResidentFinalHiddenUsedForLogits(used: boolean): void {
    this.residentFinalHiddenUsedForLogits = used;
  }

  recordFusedDecodeLayers(input: {
    packedQkvLayerCount?: number | undefined;
    qkvNormRopeKvAppendLayerCount?: number | undefined;
    oneTokenAttentionLayerCount?: number | undefined;
    residualRmsNormLayerCount?: number | undefined;
    mlpLayerCount?: number | undefined;
    fullLayerCount?: number | undefined;
  }): void {
    this.fusedPackedQkvLayerCount += nonNegativeInteger(input.packedQkvLayerCount ?? 0);
    this.fusedQkvNormRopeKvAppendLayerCount += nonNegativeInteger(input.qkvNormRopeKvAppendLayerCount ?? 0);
    this.fusedOneTokenAttentionLayerCount += nonNegativeInteger(input.oneTokenAttentionLayerCount ?? 0);
    this.fusedResidualRmsNormLayerCount += nonNegativeInteger(input.residualRmsNormLayerCount ?? 0);
    this.fusedMlpLayerCount += nonNegativeInteger(input.mlpLayerCount ?? 0);
    this.fusedFullLayerCount += nonNegativeInteger(input.fullLayerCount ?? 0);
  }

  snapshot(options: DecodePerfSnapshotOptions = {}): DecodePerfSummary {
    const generatedTokenCount = Math.max(0, this.generatedTokenCount);
    const generatedTokenDenom = Math.max(1, generatedTokenCount);
    const layerVisitDenom = Math.max(1, this.totalDecodeLayerCount);
    const residentDecodeLayerCoverage = this.totalDecodeLayerCount > 0
      ? this.residentDecodeLayerCount / this.totalDecodeLayerCount
      : 0;
    const fusedStageHits = this.fusedPackedQkvLayerCount
      + this.fusedQkvNormRopeKvAppendLayerCount
      + this.fusedOneTokenAttentionLayerCount
      + this.fusedResidualRmsNormLayerCount
      + this.fusedMlpLayerCount
      + this.fusedFullLayerCount;
    const fusedLayerCoverage = fusedStageHits / Math.max(1, this.totalDecodeLayerCount * 6);
    return {
      ...(this.init.requestId ? { requestId: this.init.requestId } : {}),
      generatedTokenCount,
      decodeCallCount: this.decodeCallCount,
      decodeSubmitCount: this.decodeSubmitCount,
      dispatchCount: this.dispatchCount,
      decodeSubmitCountPerToken: round(this.decodeSubmitCount / generatedTokenDenom),
      decodeDispatchCountPerToken: round(this.dispatchCount / generatedTokenDenom),
      decodeDispatchCountPerLayerPerToken: round(this.dispatchCount / layerVisitDenom),
      readbackCount: this.readbackCount,
      totalReadbackRows: this.totalReadbackRows,
      totalReadbackBytes: this.totalReadbackBytes,
      fullLogitsReadbackCount: this.fullLogitsReadbackCount,
      compactLogitReadbackCount: this.compactLogitReadbackCount,
      weightUploadBytesDuringDecode: this.weightUploadBytesDuringDecode,
      weightUploadCountDuringDecode: this.weightUploadCountDuringDecode,
      activationUploadBytesDuringDecode: this.activationUploadBytesDuringDecode,
      activationUploadCountDuringDecode: this.activationUploadCountDuringDecode,
      hiddenReadbackCountDuringDecode: this.hiddenReadbackCountDuringDecode,
      f32ExpansionCountDuringDecode: this.f32ExpansionCountDuringDecode,
      f32ExpansionBytesDuringDecode: this.f32ExpansionBytesDuringDecode,
      cpuFallbackUsed: this.cpuFallbackUsed,
      cpuValidationUsed: this.cpuValidationUsed,
      prefillExecutionsDuringDecode: this.prefillExecutionsDuringDecode,
      prefillCountPerGeneratedToken: generatedTokenCount > 0
        ? this.prefillExecutionsDuringDecode / generatedTokenCount
        : this.prefillExecutionsDuringDecode,
      residentDecodeLayerCount: this.residentDecodeLayerCount,
      totalDecodeLayerCount: this.totalDecodeLayerCount,
      residentDecodeLayerCoverage,
      residentFinalHiddenUsedForLogits: this.residentFinalHiddenUsedForLogits,
      kvDecodeReused: this.kvDecodeReused,
      fusedPackedQkvLayerCount: this.fusedPackedQkvLayerCount,
      fusedQkvNormRopeKvAppendLayerCount: this.fusedQkvNormRopeKvAppendLayerCount,
      fusedOneTokenAttentionLayerCount: this.fusedOneTokenAttentionLayerCount,
      fusedResidualRmsNormLayerCount: this.fusedResidualRmsNormLayerCount,
      fusedMlpLayerCount: this.fusedMlpLayerCount,
      fusedFullLayerCount: this.fusedFullLayerCount,
      fusedLayerCoverage: round(fusedLayerCoverage),
      tokensPerSecond: normalizeNullableNumber(options.tokensPerSecond),
    };
  }
}

export function summarizeDecodeHotPath(input: DecodeHotPathProofInput): DecodePerfSummary {
  const tracer = new DecodePerfTracer({ ...(input.requestId ? { requestId: input.requestId } : {}) });
  tracer.recordGeneratedToken(input.generatedTokens ?? 0);
  tracer.recordDecodeCall(input.decodeCallCount ?? 0);
  tracer.recordSubmits(input.decodeSubmitCount ?? 0);
  tracer.recordDispatches(input.logitProjectionDispatchCount ?? 0);
  if (input.logitProjectionBackend === "cpu_reference" || input.cpuFallbackUsed === true) {
    tracer.recordCpuFallback();
  }
  if (input.cpuValidationUsed === true) tracer.recordCpuValidation();
  if ((input.logitProjectionReadbackBytes ?? 0) > 0 || (input.logitProjectionReadbackRows ?? 0) > 0) {
    const readbackStats = {
      ...(input.logitProjectionReadbackRows !== undefined ? { rows: input.logitProjectionReadbackRows } : {}),
      ...(input.logitProjectionReadbackBytes !== undefined ? { bytes: input.logitProjectionReadbackBytes } : {}),
    };
    if (input.logitProjectionReadbackStrategy === "full_logits") {
      tracer.recordFullLogitsReadback(readbackStats);
    } else if (
      input.logitProjectionReadbackStrategy === "gpu_top1_candidates"
      || input.logitProjectionReadbackStrategy === "gpu_argmax_token_id"
      || input.logitProjectionReadbackStrategy === "gpu_compact_topk"
    ) {
      tracer.recordCompactLogitReadback(readbackStats);
    } else {
      tracer.recordReadback(readbackStats);
    }
  }
  if ((input.weightUploadCountDuringDecode ?? 0) > 0 || (input.weightUploadBytesDuringDecode ?? 0) > 0) {
    for (let index = 0; index < Math.max(1, input.weightUploadCountDuringDecode ?? 0); index += 1) {
      tracer.recordWeightUpload({ bytes: index === 0 ? input.weightUploadBytesDuringDecode ?? 0 : 0 });
    }
  }
  if ((input.activationUploadCountDuringDecode ?? 0) > 0 || (input.activationUploadBytesDuringDecode ?? 0) > 0) {
    for (let index = 0; index < Math.max(1, input.activationUploadCountDuringDecode ?? 0); index += 1) {
      tracer.recordActivationUpload({ bytes: index === 0 ? input.activationUploadBytesDuringDecode ?? 0 : 0 });
    }
  }
  if ((input.hiddenReadbackCountDuringDecode ?? 0) > 0) {
    tracer.recordHiddenReadback(input.hiddenReadbackCountDuringDecode);
  }
  if ((input.f32ExpansionCountDuringDecode ?? 0) > 0 || (input.f32ExpansionBytesDuringDecode ?? 0) > 0) {
    for (let index = 0; index < Math.max(1, input.f32ExpansionCountDuringDecode ?? 0); index += 1) {
      tracer.recordF32Expansion({ bytes: index === 0 ? input.f32ExpansionBytesDuringDecode ?? 0 : 0 });
    }
  }
  if ((input.prefillExecutionsDuringDecode ?? 0) > 0) {
    tracer.recordPrefillDuringDecode(input.prefillExecutionsDuringDecode);
  }
  tracer.recordResidentDecodeLayers({
    residentLayerCount: input.residentDecodeLayerCount ?? 0,
    totalLayerCount: input.totalDecodeLayerCount ?? 0,
  });
  tracer.recordResidentFinalHiddenUsedForLogits(input.residentFinalHiddenUsedForLogits === true);
  tracer.recordFusedDecodeLayers({
    packedQkvLayerCount: input.fusedPackedQkvLayerCount,
    qkvNormRopeKvAppendLayerCount: input.fusedQkvNormRopeKvAppendLayerCount,
    oneTokenAttentionLayerCount: input.fusedOneTokenAttentionLayerCount,
    residualRmsNormLayerCount: input.fusedResidualRmsNormLayerCount,
    mlpLayerCount: input.fusedMlpLayerCount,
    fullLayerCount: input.fusedFullLayerCount,
  });
  if (input.kvDecodeReused === true) tracer.recordKvDecodeReuse();
  return tracer.snapshot({
    ...(input.tokensPerSecond !== undefined ? { tokensPerSecond: input.tokensPerSecond } : {}),
  });
}

function nonNegativeInteger(value: number): number {
  if (!Number.isFinite(value)) return 0;
  return Math.max(0, Math.floor(value));
}

function normalizeNullableNumber(value: number | null | undefined): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function round(value: number): number {
  return Math.round(value * 1000) / 1000;
}
