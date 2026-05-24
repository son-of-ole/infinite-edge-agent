import { describe, expect, it } from "vitest";
import type { TSPScheduleStep } from "./tsp";
import { executeTSPSchedule } from "./tspExecutor";

describe("executeTSPSchedule", () => {
  it("runs schedule steps in deterministic sequence/tensor order with step metadata", async () => {
    const calls: string[] = [];
    const schedule: TSPScheduleStep[] = [
      makeStep("mlp_s1_t0", "mlp", 1, 0),
      makeStep("attn_s0_t0", "attention", 0, 0),
      makeStep("prefetch_s0_t0", "kv_prefetch", 0, 0),
      makeStep("checkpoint_s0_t0", "activation_checkpoint", 0, 0),
    ];

    const trace = await executeTSPSchedule(schedule, {
      kv_prefetch: async ({ step, metadata }) => calls.push(`${metadata.requestId}:${step.id}`),
      attention: async ({ step, order }) => calls.push(`${order}:${step.id}`),
      activation_checkpoint: async ({ step }) => calls.push(step.id),
      mlp: async ({ step }) => calls.push(step.id),
    }, {
      metadata: { requestId: "req_tsp_exec" },
    });

    expect(calls).toEqual([
      "0:attn_s0_t0",
      "req_tsp_exec:prefetch_s0_t0",
      "checkpoint_s0_t0",
      "mlp_s1_t0",
    ]);
    expect(trace.map((record) => record.stepId)).toEqual([
      "attn_s0_t0",
      "prefetch_s0_t0",
      "checkpoint_s0_t0",
      "mlp_s1_t0",
    ]);
    expect(trace[1]).toMatchObject({
      kind: "kv_prefetch",
      sequenceShard: 0,
      tensorShard: 0,
      originalIndex: 2,
      status: "ok",
      metadata: { requestId: "req_tsp_exec" },
    });
  });

  it("fails clearly when a required callback is missing", async () => {
    await expect(executeTSPSchedule([
      makeStep("attn_s0_t0", "attention", 0, 0),
    ], {
      kv_prefetch: async () => undefined,
      activation_checkpoint: async () => undefined,
      mlp: async () => undefined,
    })).rejects.toThrow('TSP executor missing required callback for step kind "attention" (step: attn_s0_t0)');
  });

  it("preflights callbacks before executing any scheduled step", async () => {
    const calls: string[] = [];

    await expect(executeTSPSchedule([
      makeStep("prefetch_s0_t0", "kv_prefetch", 0, 0),
      makeStep("attn_s0_t0", "attention", 0, 0),
    ], {
      kv_prefetch: async ({ step }) => calls.push(step.id),
      activation_checkpoint: async () => undefined,
      mlp: async () => undefined,
    })).rejects.toThrow('TSP executor missing required callback for step kind "attention" (step: attn_s0_t0)');

    expect(calls).toEqual([]);
  });

  it("passes frozen metadata snapshots to callbacks and trace records", async () => {
    const frozenStates: boolean[] = [];
    const seenMetadata: Record<string, unknown>[] = [];

    const trace = await executeTSPSchedule([
      makeStep("prefetch_s0_t0", "kv_prefetch", 0, 0),
      makeStep("attn_s0_t0", "attention", 0, 0),
    ], {
      kv_prefetch: async ({ metadata }) => {
        frozenStates.push(Object.isFrozen(metadata));
        seenMetadata.push(metadata);
        try {
          metadata.requestId = "mutated";
        } catch {
          // Frozen metadata should reject mutation in strict runtimes.
        }
      },
      attention: async ({ metadata }) => {
        frozenStates.push(Object.isFrozen(metadata));
        seenMetadata.push(metadata);
      },
      activation_checkpoint: async () => undefined,
      mlp: async () => undefined,
    }, {
      metadata: { requestId: "req_metadata", nested: { stable: true } },
    });

    expect(frozenStates).toEqual([true, true]);
    expect(seenMetadata[0]).not.toBe(seenMetadata[1]);
    expect(trace[0]?.metadata).toEqual({ requestId: "req_metadata", nested: { stable: true } });
    expect(trace[1]?.metadata).toEqual({ requestId: "req_metadata", nested: { stable: true } });
    expect(trace[0]?.metadata).not.toBe(trace[1]?.metadata);
  });

  it("preserves planner-emitted order for equal scheduling coordinates", async () => {
    const calls: string[] = [];

    const trace = await executeTSPSchedule([
      makeStep("z_second_name_but_first_plan_step", "attention", 0, 0),
      makeStep("a_first_name_but_second_plan_step", "attention", 0, 0),
    ], {
      kv_prefetch: async () => undefined,
      attention: async ({ step }) => calls.push(step.id),
      activation_checkpoint: async () => undefined,
      mlp: async () => undefined,
    });

    expect(calls).toEqual([
      "z_second_name_but_first_plan_step",
      "a_first_name_but_second_plan_step",
    ]);
    expect(trace.map((record) => record.originalIndex)).toEqual([0, 1]);
  });

  it("preserves mixed-kind planner order within equal shard coordinates", async () => {
    const calls: string[] = [];
    const schedule: TSPScheduleStep[] = [
      makeStep("attn_before_prefetch_same_shard", "attention", 0, 0, { tokenStart: 32, tokenEnd: 48 }),
      makeStep("prefetch_after_attention_same_shard", "kv_prefetch", 0, 0, { tokenStart: 0, tokenEnd: 16 }),
      makeStep("mlp_after_prefetch_same_shard", "mlp", 0, 0, { tokenStart: 16, tokenEnd: 32 }),
    ];

    const trace = await executeTSPSchedule(schedule, {
      kv_prefetch: async ({ step }) => calls.push(step.id),
      attention: async ({ step }) => calls.push(step.id),
      activation_checkpoint: async () => undefined,
      mlp: async ({ step }) => calls.push(step.id),
    });

    expect(calls).toEqual([
      "attn_before_prefetch_same_shard",
      "prefetch_after_attention_same_shard",
      "mlp_after_prefetch_same_shard",
    ]);
    expect(trace.map((record) => record.originalIndex)).toEqual([0, 1, 2]);
  });
});

function makeStep(
  id: string,
  kind: TSPScheduleStep["kind"],
  sequenceShard: number,
  tensorShard: number,
  overrides: Partial<TSPScheduleStep> = {},
): TSPScheduleStep {
  return {
    id,
    kind,
    sequenceShard,
    tensorShard,
    tokenStart: sequenceShard * 8,
    tokenEnd: sequenceShard * 8 + 8,
    ...overrides,
  };
}
