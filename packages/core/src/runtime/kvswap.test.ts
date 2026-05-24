import { describe, expect, it } from "vitest";
import {
  buildKVSwapPressureTelemetry,
  createLowRankKeySummary,
  planKVSwap,
  scoreLowRankAttention,
  type KVBlock,
  type KVSwapPolicy,
} from "./kvswap";

describe("KVSwap planner", () => {
  it("pins protected blocks, prefetches predicted blocks, and evicts deterministic cold VRAM blocks", () => {
    const policy: KVSwapPolicy = {
      mode: "metadata_only",
      now: 1_000_000,
      vramPressureThreshold: 0.8,
      ramPressureThreshold: 0.85,
      vramBudgetBytes: 3_000,
      ramBudgetBytes: 5_000,
    };
    const decision = planKVSwap(makeBlocks(), policy, 1_200, ["kv_needed_ram"]);

    expect(decision.pinBlockIds).toEqual(["kv_system", "kv_gac_pin"]);
    expect(decision.prefetchBlockIds).toEqual(["kv_needed_ram"]);
    expect(decision.evictBlockIds).toEqual(["kv_cold", "kv_warm"]);
    expect(decision.reasons.kv_cold).toEqual(expect.arrayContaining(["not_pinned", "low_importance_or_old", "tier:vram"]));
    expect(decision.reasons.kv_system).toEqual(expect.arrayContaining(["pinned", "tensor_key:test-backend"]));
    expect(decision.pressureTelemetry).toMatchObject({
      vramBytes: 3_600,
      ramBytes: 900,
      targetFreeBytes: 1_200,
      projectedVramBytesAfterEviction: 1_900,
      backendTensorHandleCount: 1,
    });
  });

  it("reports pressure telemetry in metadata-only mode", () => {
    const telemetry = buildKVSwapPressureTelemetry(makeBlocks(), {
      mode: "metadata_only",
      now: 1_000_000,
      vramPressureThreshold: 0.8,
      ramPressureThreshold: 0.85,
      vramBudgetBytes: 4_000,
      ramBudgetBytes: 2_000,
    }, 500, 700);

    expect(telemetry).toMatchObject({
      vramPressureRatio: 0.9,
      ramPressureRatio: 0.45,
      projectedVramBytesAfterEviction: 2_900,
    });
  });

  it("carries predictive prefetch reasons and confidence into the decision proof", () => {
    const decision = planKVSwap(makeBlocks(), {
      mode: "predictive",
      now: 1_000_000,
      vramPressureThreshold: 0.8,
      ramPressureThreshold: 0.85,
    }, 0, ["kv_needed_ram"], [
      {
        blockId: "kv_needed_ram",
        confidence: 0.82,
        reasons: ["mtp_branch", "gac:low_risk_representative"],
      },
    ]);

    expect(decision.prefetchBlockIds).toEqual(["kv_needed_ram"]);
    expect(decision.predictivePrefetchBlockIds).toEqual(["kv_needed_ram"]);
    expect(decision.prefetchConfidenceByBlockId.kv_needed_ram).toBe(0.82);
    expect(decision.predictionReasons.kv_needed_ram).toEqual(["mtp_branch", "gac:low_risk_representative"]);
    expect(decision.reasons.kv_needed_ram).toEqual(expect.arrayContaining([
      "predicted_needed",
      "prediction:mtp_branch",
      "prediction:gac:low_risk_representative",
    ]));
  });

  it("scores approximate attention deterministically from compressed low-rank key summaries", () => {
    const hot = createLowRankKeySummary({
      blockId: "kv_hot_summary",
      layer: 2,
      headGroupId: "kv-heads:0-3",
      projectionId: "qwen3:k-low-rank:v1",
      values: [0.95, 0.42, -0.1, 0.05],
      checksum: "sha256:hot",
      qualityScore: 0.92,
    });
    const cold = createLowRankKeySummary({
      blockId: "kv_cold_summary",
      layer: 2,
      headGroupId: "kv-heads:0-3",
      projectionId: "qwen3:k-low-rank:v1",
      values: [-0.75, -0.3, 0.4, -0.2],
      checksum: "sha256:cold",
      qualityScore: 0.9,
    });
    const query = {
      rank: 4,
      projectionId: "qwen3:k-low-rank:v1",
      layer: 2,
      headGroupId: "kv-heads:0-3",
      values: [1, 0.35, -0.05, 0.02],
    };

    const hotScore = scoreLowRankAttention(hot, query);
    const coldScore = scoreLowRankAttention(cold, query);

    expect(hotScore).toMatchObject({
      blockId: "kv_hot_summary",
      rank: 4,
      projectionId: "qwen3:k-low-rank:v1",
      layer: 2,
      headGroupId: "kv-heads:0-3",
      qualityScore: 0.92,
    });
    expect(hotScore.score).toBeGreaterThan(0.85);
    expect(hotScore.score).toBeGreaterThan(coldScore.score + 0.55);
    expect(scoreLowRankAttention(hot, query).score).toBe(hotScore.score);
  });

  it("plans predictive prefetch from approximate attention instead of exact prompt identity", () => {
    const policy: KVSwapPolicy = {
      mode: "predictive",
      now: 1_000_000,
      vramPressureThreshold: 0.8,
      ramPressureThreshold: 0.85,
    };
    const decision = planKVSwap(makeLowRankBlocks(), policy, 0, [], [], {
      querySummary: {
        rank: 4,
        projectionId: "qwen3:k-low-rank:v1",
        layer: 0,
        headGroupId: "kv-heads:0-3",
        values: [0.9, 0.35, -0.05, 0.1],
      },
      maxBlocks: 2,
      minScore: 0.2,
      actualAttentionBlockIds: ["kv_predict_hot"],
    });

    expect(decision.predictivePrefetchBlockIds).toEqual(["kv_predict_hot", "kv_predict_warm"]);
    expect(decision.prefetchBlockIds).toEqual(["kv_predict_hot", "kv_predict_warm"]);
    expect((decision.predictedHotBlocks ?? []).map((block) => block.blockId)).toEqual(["kv_predict_hot", "kv_predict_warm"]);
    expect(decision.prefetchedBlocks ?? []).toEqual(["kv_predict_hot", "kv_predict_warm"]);
    expect(decision.lowRankSummaryRank).toBe(4);
    expect(decision.prefetchHitRate).toBe(1);
    expect(decision.prefetchBytes).toBe(2_200);
    expect(decision.prefetchStrategy).toBe("predictive_prefetch");
    expect(decision.reasons.kv_predict_hot).toEqual(expect.arrayContaining([
      "prediction:low_rank_attention",
      "prediction:approx_attention_score",
    ]));
    expect(decision.prefetchConfidenceByBlockId.kv_predict_hot).toBeGreaterThan(0.8);
  });
});

function makeBlocks(): KVBlock[] {
  return [
    {
      id: "kv_system",
      layer: 0,
      startToken: 0,
      endToken: 16,
      tier: "vram",
      pinned: true,
      importance: 1,
      lastAccessAt: 999_999,
      estimatedBytes: 1_000,
      tensorHandles: { key: { backend: "test-backend", id: "k_system", bytes: 500 } },
    },
    {
      id: "kv_gac_pin",
      layer: 0,
      startToken: 16,
      endToken: 32,
      tier: "vram",
      pinned: false,
      importance: 0.7,
      lastAccessAt: 999_998,
      estimatedBytes: 900,
      gacPriority: { blockId: "kv_gac_pin", tier: "PIN_HOT", priorityScore: 1, reasonCodes: ["identity_pin"] },
    },
    {
      id: "kv_cold",
      layer: 0,
      startToken: 32,
      endToken: 48,
      tier: "vram",
      pinned: false,
      importance: 0.05,
      lastAccessAt: 0,
      estimatedBytes: 700,
    },
    {
      id: "kv_warm",
      layer: 0,
      startToken: 48,
      endToken: 64,
      tier: "vram",
      pinned: false,
      importance: 0.2,
      lastAccessAt: 10,
      estimatedBytes: 1_000,
    },
    {
      id: "kv_needed_ram",
      layer: 0,
      startToken: 64,
      endToken: 80,
      tier: "ram",
      pinned: false,
      importance: 0.8,
      lastAccessAt: 999_000,
      estimatedBytes: 900,
    },
  ];
}

function makeLowRankBlocks(): KVBlock[] {
  return [
    {
      id: "kv_predict_hot",
      layer: 0,
      startToken: 0,
      endToken: 16,
      tier: "ram",
      pinned: false,
      importance: 0.7,
      lastAccessAt: 999_900,
      estimatedBytes: 1_000,
      lowRankKeySummary: createLowRankKeySummary({
        blockId: "kv_predict_hot",
        layer: 0,
        headGroupId: "kv-heads:0-3",
        projectionId: "qwen3:k-low-rank:v1",
        values: [0.92, 0.38, -0.08, 0.09],
        checksum: "sha256:predict-hot",
        qualityScore: 0.95,
      }),
    },
    {
      id: "kv_predict_warm",
      layer: 0,
      startToken: 16,
      endToken: 32,
      tier: "disk",
      pinned: false,
      importance: 0.6,
      lastAccessAt: 999_800,
      estimatedBytes: 1_200,
      lowRankKeySummary: createLowRankKeySummary({
        blockId: "kv_predict_warm",
        layer: 0,
        headGroupId: "kv-heads:0-3",
        projectionId: "qwen3:k-low-rank:v1",
        values: [0.68, 0.25, -0.02, 0.08],
        checksum: "sha256:predict-warm",
        qualityScore: 0.88,
      }),
    },
    {
      id: "kv_predict_cold",
      layer: 0,
      startToken: 32,
      endToken: 48,
      tier: "ram",
      pinned: false,
      importance: 0.5,
      lastAccessAt: 999_700,
      estimatedBytes: 900,
      lowRankKeySummary: createLowRankKeySummary({
        blockId: "kv_predict_cold",
        layer: 0,
        headGroupId: "kv-heads:0-3",
        projectionId: "qwen3:k-low-rank:v1",
        values: [-0.9, -0.3, 0.25, -0.1],
        checksum: "sha256:predict-cold",
        qualityScore: 0.9,
      }),
    },
  ];
}
