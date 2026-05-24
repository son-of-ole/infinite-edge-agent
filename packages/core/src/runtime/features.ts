export type RuntimeFeatureName =
  | "memoryProvider"
  | "contextRuntime"
  | "ssa"
  | "tsp"
  | "mtp"
  | "kvswap"
  | "inferenceBackend";

export type RuntimeFeatureState =
  | "required"
  | "enabled"
  | "fallback"
  | "disabled_for_test"
  | "unavailable";

export interface RuntimeFeatureStatus {
  name: RuntimeFeatureName;
  state: RuntimeFeatureState;
  mode: string;
  reason?: string;
  impact?: string;
  metrics?: Record<string, string | number | boolean>;
}

export class RuntimeFeatureRegistry {
  private readonly statuses = new Map<RuntimeFeatureName, RuntimeFeatureStatus>();

  constructor(initial?: RuntimeFeatureStatus[]) {
    for (const status of initial ?? []) this.set(status);
  }

  set(status: RuntimeFeatureStatus): void {
    this.statuses.set(status.name, status);
  }

  get(name: RuntimeFeatureName): RuntimeFeatureStatus | undefined {
    return this.statuses.get(name);
  }

  list(): RuntimeFeatureStatus[] {
    return Array.from(this.statuses.values()).sort((a, b) => a.name.localeCompare(b.name));
  }

  assertTierZeroReady(): void {
    const required: RuntimeFeatureName[] = [
      "memoryProvider",
      "contextRuntime",
      "ssa",
      "tsp",
      "mtp",
      "kvswap",
      "inferenceBackend",
    ];

    const missing = required.filter((name) => !this.statuses.has(name));
    if (missing.length > 0) {
      throw new Error(`Missing Tier-0 runtime feature status: ${missing.join(", ")}`);
    }
  }

  assertProductionReady(): void {
    this.assertTierZeroReady();
    const disabled = this.list().filter((status) => status.state === "disabled_for_test");
    if (disabled.length > 0) {
      throw new Error(`Tier-0 runtime features disabled for test in production: ${disabled.map((status) => status.name).join(", ")}`);
    }
  }
}

export function assertProductionRuntimeFeatures(features: RuntimeFeatureStatus[]): void {
  new RuntimeFeatureRegistry(features).assertProductionReady();
}

export function createDefaultRuntimeFeatureRegistry(): RuntimeFeatureRegistry {
  return new RuntimeFeatureRegistry([
    {
      name: "memoryProvider",
      state: "fallback",
      mode: "indexeddb_until_sidecar_connected",
      reason: "Production memory provider availability is detected at runtime.",
    },
    { name: "contextRuntime", state: "enabled", mode: "ledger_and_prompt_packing" },
    {
      name: "ssa",
      state: "fallback",
      mode: "fallback_sparse_planner",
      reason: "Native SSA attention backend is not selected.",
      impact: "The runtime emits sparse block plans and traces, but browser inference still executes dense attention over the packed context.",
    },
    {
      name: "tsp",
      state: "fallback",
      mode: "fallback_budget_planner",
      reason: "Native folded TP+SP backend is not selected.",
      impact: "The runtime reports safe context budgets and folded schedules, but the browser backend does not execute tensor shards directly.",
    },
    {
      name: "mtp",
      state: "fallback",
      mode: "target_only",
      reason: "Draft model is not configured.",
      impact: "Generation remains target-only until a tokenizer-compatible draft model and verifier backend are configured.",
    },
    {
      name: "kvswap",
      state: "fallback",
      mode: "metadata_only",
      reason: "Inference backend has not exposed KV tensor handles.",
      impact: "KV blocks are planned, pinned, evicted, and prefetched as metadata; real tensor paging waits on backend handles.",
    },
    { name: "inferenceBackend", state: "enabled", mode: "browser_runtime" },
  ]);
}
