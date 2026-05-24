import { describe, expect, it } from "vitest";
import {
  evaluateFusedDecodeSpeedGate,
  FusedDecodePerfAccumulator,
} from "./index";

describe("fused decode performance counters", () => {
  it("summarizes submit and dispatch fragmentation per token and layer", () => {
    const perf = new FusedDecodePerfAccumulator({ layerCount: 2 });
    perf.recordGeneratedToken();
    perf.recordLayer({ submitCount: 1, dispatchCount: 4, packedQkv: true });
    perf.recordLayer({ submitCount: 1, dispatchCount: 5, packedQkv: true, oneTokenAttention: true });

    expect(perf.snapshot()).toMatchObject({
      decodeSubmitCount: 2,
      decodeSubmitCountPerToken: 2,
      decodeDispatchCountPerToken: 9,
      decodeDispatchCountPerLayerPerToken: 4.5,
      fusedPackedQkvLayerCount: 2,
      fusedOneTokenAttentionLayerCount: 1,
    });
  });

  it("fails the fused speed gate when token throughput is below the production floor", () => {
    const perf = new FusedDecodePerfAccumulator({ layerCount: 28 });
    perf.recordGeneratedToken();
    for (let layer = 0; layer < 28; layer += 1) {
      perf.recordLayer({ submitCount: 1, dispatchCount: 12 });
    }

    const gate = evaluateFusedDecodeSpeedGate({
      summary: perf.snapshot(),
      layerCount: 28,
      meanTokensPerSecond: 1.45,
      minTokensPerSecond: 2,
    });

    expect(gate).toMatchObject({
      passed: false,
      failures: expect.arrayContaining(["tokens_per_second_below_floor"]),
    });
  });
});
