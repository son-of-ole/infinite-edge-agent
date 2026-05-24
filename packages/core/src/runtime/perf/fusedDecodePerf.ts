export interface FusedDecodePerfSummary {
  decodeSubmitCount: number;
  decodeSubmitCountPerToken: number;
  decodeDispatchCountPerToken: number;
  decodeDispatchCountPerLayerPerToken: number;
  fusedPackedQkvLayerCount: number;
  fusedQkvNormRopeKvAppendLayerCount: number;
  fusedOneTokenAttentionLayerCount: number;
  fusedResidualRmsNormLayerCount: number;
  fusedMlpLayerCount: number;
  fusedFullLayerCount: number;
  fusedLayerCoverage: number;
}

export interface FusedDecodePerfAccumulatorInput {
  layerCount: number;
}

export class FusedDecodePerfAccumulator {
  private generatedTokens = 0;
  private submitCount = 0;
  private dispatchCount = 0;
  private visitedLayerCount = 0;
  private packedQkvLayerCount = 0;
  private qkvNormRopeKvAppendLayerCount = 0;
  private oneTokenAttentionLayerCount = 0;
  private residualRmsNormLayerCount = 0;
  private mlpLayerCount = 0;
  private fullLayerCount = 0;

  constructor(private readonly input: FusedDecodePerfAccumulatorInput) {}

  recordGeneratedToken(): void {
    this.generatedTokens += 1;
  }

  recordLayer(input: {
    submitCount: number;
    dispatchCount: number;
    packedQkv?: boolean;
    qkvNormRopeKvAppend?: boolean;
    oneTokenAttention?: boolean;
    residualRmsNorm?: boolean;
    mlp?: boolean;
    fullLayer?: boolean;
  }): void {
    this.visitedLayerCount += 1;
    this.submitCount += Math.max(0, Math.floor(input.submitCount));
    this.dispatchCount += Math.max(0, Math.floor(input.dispatchCount));
    if (input.packedQkv) this.packedQkvLayerCount += 1;
    if (input.qkvNormRopeKvAppend) this.qkvNormRopeKvAppendLayerCount += 1;
    if (input.oneTokenAttention) this.oneTokenAttentionLayerCount += 1;
    if (input.residualRmsNorm) this.residualRmsNormLayerCount += 1;
    if (input.mlp) this.mlpLayerCount += 1;
    if (input.fullLayer) this.fullLayerCount += 1;
  }

  snapshot(): FusedDecodePerfSummary {
    const tokenCount = Math.max(1, this.generatedTokens);
    const layerDenom = Math.max(1, this.generatedTokens * this.input.layerCount);
    const fusedStageHits = this.packedQkvLayerCount
      + this.qkvNormRopeKvAppendLayerCount
      + this.oneTokenAttentionLayerCount
      + this.residualRmsNormLayerCount
      + this.mlpLayerCount
      + this.fullLayerCount;
    const maxStageHits = Math.max(1, this.visitedLayerCount * 6);
    return {
      decodeSubmitCount: this.submitCount,
      decodeSubmitCountPerToken: round(this.submitCount / tokenCount),
      decodeDispatchCountPerToken: round(this.dispatchCount / tokenCount),
      decodeDispatchCountPerLayerPerToken: round(this.dispatchCount / layerDenom),
      fusedPackedQkvLayerCount: this.packedQkvLayerCount,
      fusedQkvNormRopeKvAppendLayerCount: this.qkvNormRopeKvAppendLayerCount,
      fusedOneTokenAttentionLayerCount: this.oneTokenAttentionLayerCount,
      fusedResidualRmsNormLayerCount: this.residualRmsNormLayerCount,
      fusedMlpLayerCount: this.mlpLayerCount,
      fusedFullLayerCount: this.fullLayerCount,
      fusedLayerCoverage: round(fusedStageHits / maxStageHits),
    };
  }
}

export function evaluateFusedDecodeSpeedGate(input: {
  summary: FusedDecodePerfSummary;
  layerCount: number;
  meanTokensPerSecond: number | null;
  minTokensPerSecond?: number;
}): { passed: boolean; failures: string[] } {
  const failures: string[] = [];
  const floor = input.minTokensPerSecond ?? 2;
  if ((input.meanTokensPerSecond ?? 0) < floor) failures.push("tokens_per_second_below_floor");
  if (input.summary.decodeSubmitCountPerToken > input.layerCount + 2) failures.push("too_many_submits_per_token");
  if (input.summary.decodeDispatchCountPerLayerPerToken <= 0) failures.push("missing_dispatch_counter");
  return { passed: failures.length === 0, failures };
}

function round(value: number): number {
  return Math.round(value * 1000) / 1000;
}
