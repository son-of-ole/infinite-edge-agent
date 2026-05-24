import { describe, expect, it } from "vitest";
import {
  buildFallbackTSPPlan,
  buildTSPSchedule,
  createModelProfileRegistry,
  detectDeviceProfile,
  estimateTSPMemory,
  getDefaultModelProfile,
  type ModelProfile,
} from "./tsp";

describe("TSP planner", () => {
  it("detects deterministic device profiles from explicit hints", () => {
    const device = detectDeviceProfile({
      name: "ci-webgpu",
      backend: "webgpu",
      deviceMemoryGb: 16,
      hardwareConcurrency: 10,
      maxBufferSizeBytes: 512 * 1024 * 1024,
    });

    expect(device).toMatchObject({
      name: "ci-webgpu",
      backend: "webgpu",
      ramBudgetBytes: 17_179_869_184,
      vramBudgetBytes: 8_589_934_592,
      maxBufferSizeBytes: 536_870_912,
    });
  });

  it("registers and resolves model profiles", () => {
    const registry = createModelProfileRegistry([tinyModel()]);
    registry.register({ ...tinyModel(), modelId: "custom/tiny-draft" });

    expect(registry.require("test/tiny").layers).toBe(4);
    expect(registry.list().map((profile) => profile.modelId)).toEqual(["custom/tiny-draft", "test/tiny"]);
    expect(getDefaultModelProfile("Qwen/Qwen3-0.6B")?.family).toBe("qwen3");
  });

  it("estimates memory pressure and builds folded schedules without exceeding usable VRAM", () => {
    const device = {
      name: "low-memory-ci",
      vramBudgetBytes: 256 * 1024 * 1024,
      ramBudgetBytes: 2 * 1024 * 1024 * 1024,
      backend: "wasm" as const,
    };
    const input = {
      device,
      model: tinyModel({ parameterBytes: 128 * 1024 * 1024 }),
      requestedContextTokens: 131_072,
      batchSize: 1,
      kvPrecisionBytes: 2,
      activationPrecisionBytes: 2,
      safetyMarginRatio: 0.2,
    };

    const estimate = estimateTSPMemory(input);
    const plan = buildFallbackTSPPlan(input);

    expect(estimate.fitsSingleWindow).toBe(false);
    expect(plan.degradationReason).toBe("requested_context_exceeds_safe_single_window");
    expect(plan.estimatedVramBytes).toBeLessThanOrEqual(estimate.usableVramBytes);
    expect(plan.sequenceShards).toBeGreaterThan(1);
    expect(plan.schedule.map((step) => step.kind)).toEqual(expect.arrayContaining(["kv_prefetch", "attention", "activation_checkpoint", "mlp"]));
  });

  it("builds eval-style schedules deterministically", () => {
    const schedule = buildTSPSchedule({ requestedContextTokens: 100, sequenceShards: 2, tensorShards: 2 });

    expect(schedule.map((step) => step.id)).toEqual([
      "prefetch_s0_t0",
      "attn_s0_t0",
      "checkpoint_s0_t0",
      "mlp_s0_t0",
      "prefetch_s0_t1",
      "attn_s0_t1",
      "checkpoint_s0_t1",
      "mlp_s0_t1",
      "prefetch_s1_t0",
      "attn_s1_t0",
      "checkpoint_s1_t0",
      "mlp_s1_t0",
      "prefetch_s1_t1",
      "attn_s1_t1",
      "checkpoint_s1_t1",
      "mlp_s1_t1",
    ]);
  });
});

function tinyModel(overrides: Partial<ModelProfile> = {}): ModelProfile {
  return {
    modelId: "test/tiny",
    parameterBytes: 16 * 1024 * 1024,
    layers: 4,
    hiddenSize: 64,
    kvHeads: 4,
    contextWindowTokens: 4096,
    family: "test",
    ...overrides,
  };
}
