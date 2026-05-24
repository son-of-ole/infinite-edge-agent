import { describe, expect, it } from "vitest";
import { buildPredictiveRuntimePlan } from "./predictiveRuntime";

describe("Predictive Runtime Plan", () => {
  it("predicts GAC retrievals, KV hot pages, sparse blocks, and MTP branch pressure", () => {
    const plan = buildPredictiveRuntimePlan({
      requestId: "req_predict",
      userMessage: "Continue the browser runtime work.",
      activeBlocks: [
        {
          id: "pin_1",
          text: "Pinned identity memory.",
          tokenStart: 0,
          tokenEnd: 8,
          priority: 1,
          source: "retrieved",
          tags: ["retrieved"],
          gac: {
            blockId: "pin_1",
            memoryClass: "PINNED_EXACT",
            rawMemoryId: "raw_pin_1",
            identityRisk: 0.95,
            pinStrength: 1,
            mustAttend: true,
          },
        },
        {
          id: "rep_1",
          text: "Representative summary.",
          tokenStart: 8,
          tokenEnd: 20,
          priority: 0.72,
          source: "retrieved",
          tags: ["retrieved"],
          gac: {
            blockId: "rep_1",
            memoryClass: "LOW_RISK_REPRESENTATIVE",
            rawMemoryId: "raw_rep_1",
            representativeId: "representative_1",
          },
        },
      ],
      ssaPlan: {
        mode: "fallback_sparse_planner",
        targetProfile: "subq_compatible_public_ssa_path",
        selectedBlockIds: ["pin_1", "rep_1"],
        pinnedBlockIds: ["pin_1"],
        droppedBlockIds: [],
        routingReasons: {},
        layerPolicies: [],
        estimatedDenseTokens: 20,
        estimatedSparseTokens: 12,
        sparsityRatio: 0.4,
        routingTrace: [],
        kernelTraces: [],
      },
      kvBlocks: [
        {
          id: "kv_pin_1",
          layer: 0,
          startToken: 0,
          endToken: 8,
          tier: "vram",
          pinned: true,
          importance: 1,
          lastAccessAt: 100,
          sourceBlockId: "pin_1",
          estimatedBytes: 1024,
        },
        {
          id: "kv_rep_1",
          layer: 0,
          startToken: 8,
          endToken: 20,
          tier: "ram",
          pinned: false,
          importance: 0.72,
          lastAccessAt: 90,
          sourceBlockId: "rep_1",
          estimatedBytes: 2048,
        },
      ],
      speculativeConfig: {
        enabled: true,
        mode: "draft_verify",
        draftModelId: "browser/ngram-drafter",
        targetModelId: "Qwen/Qwen3-0.6B",
        numSpeculativeTokens: 4,
        minAcceptanceRate: 0,
        disableWhenLatencyWorse: false,
      },
      tokenBudget: 4096,
      vramBudgetBytes: 8_000_000_000,
      ramBudgetBytes: 16_000_000_000,
    });

    expect(plan.predictedRetrievals).toEqual(expect.arrayContaining([
      expect.objectContaining({
        query: expect.stringContaining("raw_pin_1"),
        targetMemoryClass: "PINNED_EXACT",
        expectedRawMemoryIds: ["raw_pin_1"],
      }),
      expect.objectContaining({
        query: expect.stringContaining("representative_1"),
        targetMemoryClass: "LOW_RISK_REPRESENTATIVE",
        expectedRepresentativeIds: ["representative_1"],
      }),
    ]));
    expect(plan.kvHotPages).toEqual(expect.arrayContaining([
      expect.objectContaining({ blockId: "kv_pin_1", tier: "PIN_HOT", source: "gac" }),
      expect.objectContaining({ blockId: "kv_rep_1", source: "ssa" }),
    ]));
    expect(plan.sparseBlocks.map((block) => block.blockId)).toEqual(["pin_1", "rep_1"]);
    expect(plan.mtpBranches).toEqual([
      expect.objectContaining({
        branchId: "mtp_req_predict_accept",
        draftModelId: "browser/ngram-drafter",
        maxDraftTokens: 4,
        dependsOnBlockIds: ["kv_pin_1", "kv_rep_1"],
      }),
    ]);
    expect(plan.cacheBudget.prefetchBlockIds).toEqual(["kv_pin_1", "kv_rep_1"]);
    expect(plan.confidence).toBeGreaterThan(0);
  });
});
