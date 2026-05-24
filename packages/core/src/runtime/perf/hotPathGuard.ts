import type { DecodePerfGateOptions, DecodePerfSummary } from "./decodePerfTypes";
import { evaluateDecodeSpeedGate } from "./speedGates";

export class DecodeHotPathViolationError extends Error {
  readonly reasons: string[];
  readonly summary: DecodePerfSummary;

  constructor(reasons: string[], summary: DecodePerfSummary) {
    super(`Decode hot path violated: ${reasons.join(", ")}`);
    this.name = "DecodeHotPathViolationError";
    this.reasons = reasons;
    this.summary = summary;
  }
}

export function assertDecodeHotPath(summary: DecodePerfSummary, options?: DecodePerfGateOptions): void {
  const result = evaluateDecodeSpeedGate(summary, options);
  if (!result.passed) throw new DecodeHotPathViolationError(result.reasons, summary);
}
