# 15 — TSP Runtime Spec

## Role

TSP is the memory scheduling layer for long-context inference. It computes how tensor shards, sequence shards, activation windows, KV blocks, and batch sizes should be planned for a constrained device.

## Why it is first-class

Long context fails not only from attention complexity but also from memory pressure. TSP is included from the start so the runtime always knows the device budget and has a plan for folding tensor and sequence work onto available execution resources.

## Modes

| Mode | Meaning |
|---|---|
| `fallback_budget_planner` | Estimate safe context size and batch schedule without custom kernel control |
| `webgpu_custom` | WebGPU backend uses explicit compute schedules |
| `native_edge` | Desktop/native runtime executes folded TP+SP schedule |
| `server_edge` | Optional private edge appliance executes TSP plan |

## Inputs

```ts
interface TSPPlanInput {
  device: DeviceProfile;
  model: ModelProfile;
  requestedContextTokens: number;
  batchSize: number;
  kvPrecisionBytes: number;
  activationPrecisionBytes: number;
  safetyMarginRatio: number;
}
```

## Outputs

```ts
interface TSPPlan {
  mode: TSPMode;
  sequenceShards: number;
  tensorShards: number;
  activationWindowTokens: number;
  maxSafeContextTokens: number;
  estimatedVramBytes: number;
  estimatedRamBytes: number;
  schedule: TSPScheduleStep[];
  degradationReason?: string;
}
```

## Planner rules

1. Detect or configure GPU limits.
2. Estimate model weight footprint.
3. Estimate KV cache growth.
4. Estimate activation pressure by context and batch.
5. Choose sequence shard count.
6. Choose tensor shard count.
7. Communicate max safe context to Context Runtime.
8. Communicate cache pressure thresholds to KVSwap.

## TSP to Context Runtime contract

The context runtime must not blindly pack all memory into the prompt. It asks TSP for the max safe active context budget:

```text
TSPPlan.maxSafeContextTokens
  -> Context Runtime token budget
  -> SSA selected blocks
  -> KVSwap tier thresholds
```

## Acceptance gates

- Planner never exceeds configured memory budget.
- Planner reports degradation reason when requested context cannot fit.
- Context Runtime obeys TSP budget.
- Production backend can consume `TSPScheduleStep[]` without changing app code.

## Current implementation

`executeTSPSchedule` consumes `TSPScheduleStep[]` through backend callbacks for `kv_prefetch`, `attention`, `activation_checkpoint`, and `mlp`. It preflights missing callbacks before execution, preserves planner order within equal sequence/tensor shard coordinates, snapshots immutable metadata, and emits per-step trace records for production audits.
