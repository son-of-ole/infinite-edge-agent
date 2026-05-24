import { describe, expect, it } from "vitest";
import {
  applySpeculativeVerification,
  evaluateSpeculationAutoDisable,
  measureSpeculativeMetrics,
  resolveSpeculativeDecodingConfig,
  shouldDisableSpeculation,
  SpeculativeModelRegistry,
} from "./speculative";

describe("MTP speculative decoding contracts", () => {
  it("resolves compatible draft/target pairs and falls back target-only otherwise", () => {
    const registry = new SpeculativeModelRegistry([
      { modelId: "draft-small", role: "draft", tokenizerId: "qwen3", maxSpeculativeTokens: 3, targetModelIds: ["target-main"] },
      { modelId: "target-main", role: "target", tokenizerId: "qwen3" },
      { modelId: "target-other", role: "target", tokenizerId: "other" },
    ]);

    expect(resolveSpeculativeDecodingConfig({
      enabled: true,
      draftModelId: "draft-small",
      targetModelId: "target-main",
      numSpeculativeTokens: 8,
    }, registry)).toMatchObject({
      mode: "draft_verify",
      draftModelId: "draft-small",
      numSpeculativeTokens: 3,
    });

    expect(resolveSpeculativeDecodingConfig({
      enabled: true,
      draftModelId: "draft-small",
      targetModelId: "target-other",
    }, registry)).toMatchObject({
      mode: "target_only",
      numSpeculativeTokens: 0,
    });
  });

  it("streams accepted tokens and one target correction on rejection", () => {
    const result = applySpeculativeVerification(
      [{ token: "A" }, { token: "B" }, { token: "C" }],
      [
        { token: "A", accepted: true },
        { token: "X", accepted: false, replacement: "X" },
      ],
    );

    expect(result).toEqual({
      streamedTokens: ["A", "X"],
      acceptedTokens: 1,
      rejectedTokens: 2,
      correctedToken: "X",
    });
  });

  it("measures acceptance and auto-disables when speculation is worse", () => {
    const metrics = measureSpeculativeMetrics({
      draftTokens: 4,
      acceptedTokens: 1,
      rejectedTokens: 3,
      draftLatencyMs: 30,
      verifyLatencyMs: 90,
      targetOnlyLatencyMs: 100,
      minAcceptanceRate: 0.45,
      disableWhenLatencyWorse: true,
    });

    expect(metrics).toMatchObject({
      acceptanceRate: 0.25,
      netSpeedupRatio: 0.8333333333333334,
      disabledReason: "acceptance_rate_below_threshold",
    });
    expect(shouldDisableSpeculation({ ...metrics, acceptanceRate: 0.8, netSpeedupRatio: 0.9 }, 0.45, false)).toBeUndefined();
  });

  it("auto-disables after repeated slower requests", () => {
    const samples = [
      sample({ acceptanceRate: 0.9, netSpeedupRatio: 0.8 }),
      sample({ acceptanceRate: 0.8, netSpeedupRatio: 0.9 }),
      sample({ acceptanceRate: 0.7, netSpeedupRatio: 0.95 }),
    ];

    expect(evaluateSpeculationAutoDisable(samples, {
      minAcceptanceRate: 0.45,
      disableWhenLatencyWorse: true,
    }, 3)).toMatchObject({
      disabled: true,
      disabledReason: "speculation_slower_than_target_only",
      consecutiveWorseRequests: 3,
      sampleCount: 3,
    });
  });

  it("does not count low-acceptance fast samples as consecutive slower requests", () => {
    const samples = [
      sample({ acceptedTokens: 100, rejectedTokens: 0, acceptanceRate: 1, netSpeedupRatio: 2 }),
      sample({ acceptedTokens: 100, rejectedTokens: 0, acceptanceRate: 1, netSpeedupRatio: 2 }),
      sample({ acceptedTokens: 100, rejectedTokens: 0, acceptanceRate: 1, netSpeedupRatio: 2 }),
      sample({ acceptedTokens: 1, rejectedTokens: 3, acceptanceRate: 0.25, netSpeedupRatio: 2 }),
      sample({ acceptedTokens: 1, rejectedTokens: 3, acceptanceRate: 0.25, netSpeedupRatio: 2 }),
      sample({ acceptedTokens: 1, rejectedTokens: 3, acceptanceRate: 0.25, netSpeedupRatio: 2 }),
    ];

    expect(evaluateSpeculationAutoDisable(samples, {
      minAcceptanceRate: 0.45,
      disableWhenLatencyWorse: true,
    }, 3)).toMatchObject({
      disabled: false,
      rollingAcceptanceRate: 303 / 312,
      consecutiveWorseRequests: 0,
      sampleCount: 6,
    });
  });
});

function sample(overrides: Partial<ReturnType<typeof measureSpeculativeMetrics>> = {}) {
  return {
    draftTokens: 4,
    acceptedTokens: 4,
    rejectedTokens: 0,
    acceptanceRate: 1,
    draftLatencyMs: 10,
    verifyLatencyMs: 10,
    netSpeedupRatio: 2,
    ...overrides,
  };
}
