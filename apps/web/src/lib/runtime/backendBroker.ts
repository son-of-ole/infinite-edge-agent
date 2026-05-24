export type BrowserBackendAdapterKind =
  | "compiled-browser"
  | "mediapipe-edge"
  | "custom-webgpu-kernel-lab"
  | "wasm-fallback";

export type BrowserBackendProductionRole =
  | "production_candidate"
  | "research_kernel_lab"
  | "fallback";

export type BrowserBackendTask =
  | "grounded_answer"
  | "final_answer"
  | "memory_label"
  | "context_triage"
  | "kernel_research"
  | "strict_custom_proof";

export interface BrowserBackendCapabilities {
  supportsCompiledGraph: boolean;
  supportsWebGpu: boolean;
  supportsSparseAttentionRouting: boolean;
  supportsKvSwapPersistence: boolean;
  supportsSpeculativeDecode: boolean;
  supportsLocalOnly: boolean;
  supportsGroundedAnswers: boolean;
}

export interface BrowserBackendRegistryEntry {
  backendId: string;
  label: string;
  adapterKind: BrowserBackendAdapterKind;
  productionRole: BrowserBackendProductionRole;
  deployDefault: boolean;
  modelIds: string[];
  qualityClass: "answer" | "control" | "research";
  speedClass: "fast" | "medium" | "experimental";
  privacyClass: "local" | "local_or_remote";
  defaultTasks: BrowserBackendTask[];
  capabilities: BrowserBackendCapabilities;
}

export interface BrowserBackendSelectionInput {
  task: BrowserBackendTask;
  availableBackendIds?: string[];
  preferredBackendId?: string;
  preferredModelId?: string;
}

export interface BrowserBackendSelection {
  backendId: string;
  modelId: string;
  productionRole: BrowserBackendProductionRole;
  deployReadyCandidate: boolean;
  reason: string;
  fallbackChain: string[];
  proofRequirements: string[];
}

export const BROWSER_BACKEND_REGISTRY: BrowserBackendRegistryEntry[] = [
  {
    backendId: "compiled-browser-webllm",
    label: "Compiled Browser WebLLM Backend",
    adapterKind: "compiled-browser",
    productionRole: "production_candidate",
    deployDefault: true,
    modelIds: ["Qwen3-0.6B-q4f16_1-MLC"],
    qualityClass: "answer",
    speedClass: "fast",
    privacyClass: "local",
    defaultTasks: ["grounded_answer", "final_answer"],
    capabilities: {
      supportsCompiledGraph: true,
      supportsWebGpu: true,
      supportsSparseAttentionRouting: false,
      supportsKvSwapPersistence: false,
      supportsSpeculativeDecode: false,
      supportsLocalOnly: true,
      supportsGroundedAnswers: true,
    },
  },
  {
    backendId: "unlocked-browser-transformer",
    label: "Custom WebGPU Kernel Lab",
    adapterKind: "custom-webgpu-kernel-lab",
    productionRole: "research_kernel_lab",
    deployDefault: false,
    modelIds: ["Qwen/Qwen3-0.6B"],
    qualityClass: "research",
    speedClass: "experimental",
    privacyClass: "local",
    defaultTasks: ["kernel_research", "strict_custom_proof"],
    capabilities: {
      supportsCompiledGraph: false,
      supportsWebGpu: true,
      supportsSparseAttentionRouting: true,
      supportsKvSwapPersistence: true,
      supportsSpeculativeDecode: false,
      supportsLocalOnly: true,
      supportsGroundedAnswers: true,
    },
  },
  {
    backendId: "wasm-small-core",
    label: "WASM Small Core Fallback",
    adapterKind: "wasm-fallback",
    productionRole: "fallback",
    deployDefault: false,
    modelIds: ["small-core-control"],
    qualityClass: "control",
    speedClass: "medium",
    privacyClass: "local",
    defaultTasks: ["memory_label", "context_triage"],
    capabilities: {
      supportsCompiledGraph: false,
      supportsWebGpu: false,
      supportsSparseAttentionRouting: false,
      supportsKvSwapPersistence: false,
      supportsSpeculativeDecode: false,
      supportsLocalOnly: true,
      supportsGroundedAnswers: false,
    },
  },
];

export function getBrowserBackendRegistryEntry(backendId: string): BrowserBackendRegistryEntry | undefined {
  return BROWSER_BACKEND_REGISTRY.find((entry) => entry.backendId === backendId);
}

export function selectBrowserBackend(input: BrowserBackendSelectionInput): BrowserBackendSelection {
  const available = new Set(input.availableBackendIds ?? BROWSER_BACKEND_REGISTRY.map((entry) => entry.backendId));
  const preferred = input.preferredBackendId ? getAvailableEntry(input.preferredBackendId, available) : undefined;
  const selected = preferred ?? selectByTask(input.task, available);
  if (!selected) {
    throw new Error(`No registered browser backend can run task "${input.task}" with the available backend set.`);
  }
  const modelId = input.preferredModelId && selected.modelIds.includes(input.preferredModelId)
    ? input.preferredModelId
    : selected.modelIds[0] ?? "unknown-model";
  return {
    backendId: selected.backendId,
    modelId,
    productionRole: selected.productionRole,
    deployReadyCandidate: selected.productionRole === "production_candidate",
    reason: selectionReason(input.task, selected),
    fallbackChain: fallbackChainFor(selected.backendId, available),
    proofRequirements: proofRequirementsFor(selected, input.task),
  };
}

function selectByTask(task: BrowserBackendTask, available: Set<string>): BrowserBackendRegistryEntry | undefined {
  if (task === "kernel_research" || task === "strict_custom_proof") {
    return getAvailableEntry("unlocked-browser-transformer", available);
  }
  const production = BROWSER_BACKEND_REGISTRY.find((entry) =>
    available.has(entry.backendId)
    && entry.productionRole === "production_candidate"
    && entry.defaultTasks.includes(task));
  if (production) return production;
  return BROWSER_BACKEND_REGISTRY.find((entry) =>
    available.has(entry.backendId)
    && entry.defaultTasks.includes(task));
}

function getAvailableEntry(backendId: string, available: Set<string>): BrowserBackendRegistryEntry | undefined {
  if (!available.has(backendId)) return undefined;
  return getBrowserBackendRegistryEntry(backendId);
}

function selectionReason(task: BrowserBackendTask, entry: BrowserBackendRegistryEntry): string {
  if (entry.backendId === "unlocked-browser-transformer") return "kernel_lab_required";
  if (entry.productionRole === "production_candidate" && task === "grounded_answer") return "compiled_first_grounded_answer";
  if (entry.productionRole === "production_candidate") return "compiled_first_answer";
  return `${entry.adapterKind}_${task}`;
}

function fallbackChainFor(backendId: string, available: Set<string>): string[] {
  const order = ["compiled-browser-webllm", "unlocked-browser-transformer", "wasm-small-core"];
  return order.filter((candidate) =>
    candidate !== backendId
    && (available.has(candidate) || candidate === "wasm-small-core"));
}

function proofRequirementsFor(entry: BrowserBackendRegistryEntry, task: BrowserBackendTask): string[] {
  if (entry.productionRole === "production_candidate") {
    return [
      "memory_grounding",
      "quality_canaries",
      "speed_floor",
      "backend_trace",
    ];
  }
  if (entry.productionRole === "research_kernel_lab") {
    return [
      "strict_webgpu",
      "decode_hot_path",
      "kernel_parity",
      task === "strict_custom_proof" ? "kv_reuse" : "research_trace",
    ];
  }
  return ["task_bounds", "fallback_trace"];
}
