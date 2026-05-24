import registryJson from "./models.registry.json";

export type ModelRegistryProductionRole = "production_candidate" | "research_kernel_lab" | "fallback";
export type ModelRegistryQualityClass = "answer" | "control" | "research";
export type ModelRegistrySpeedClass = "fast" | "medium" | "experimental";
export type ModelRegistryPrivacyClass = "local" | "local_or_remote";

export interface EdgeTensorProtocolContract {
  schema: "edge-tensor-protocol/v1";
  architecture: string;
  dtype: string;
  residency: string;
  kvCache: {
    layout: string;
    persistence: string[];
  };
  kernels: Record<string, boolean>;
  proof: Record<string, boolean>;
}

export interface ModelRegistryEntry {
  modelId: string;
  label: string;
  backendId: "compiled-browser-webllm" | "unlocked-browser-transformer" | "wasm-small-core";
  productionRole: ModelRegistryProductionRole;
  artifactKind: string;
  artifactUrl: string;
  contextLimit: number;
  expectedMemory: string;
  qualityClass: ModelRegistryQualityClass;
  speedClass: ModelRegistrySpeedClass;
  privacyClass: ModelRegistryPrivacyClass;
  defaultTasks: string[];
  notes: string;
  estimatedDownload: string;
  tensorAbi: EdgeTensorProtocolContract;
}

export interface ModelRegistry {
  schema: "edge-model-registry/v1";
  models: ModelRegistryEntry[];
}

export interface LocalModelOptionFromRegistry {
  id: string;
  label: string;
  backend: ModelRegistryEntry["backendId"];
  notes: string;
  estimatedDownload: string;
  modelAssetPath?: string;
}

export const MODEL_REGISTRY = registryJson as ModelRegistry;

export function getModelRegistryEntry(
  backendId: ModelRegistryEntry["backendId"],
  modelId: string,
): ModelRegistryEntry | undefined {
  return MODEL_REGISTRY.models.find((entry) => entry.backendId === backendId && entry.modelId === modelId);
}

export function listLocalModelOptionsFromRegistry(): LocalModelOptionFromRegistry[] {
  return MODEL_REGISTRY.models
    .filter((entry) => entry.backendId !== "wasm-small-core")
    .sort((left, right) => roleSort(left.productionRole) - roleSort(right.productionRole))
    .map((entry) => ({
      id: entry.modelId,
      label: entry.label,
      backend: entry.backendId,
      notes: entry.notes,
      estimatedDownload: entry.estimatedDownload,
      ...(entry.artifactUrl.startsWith("/") ? { modelAssetPath: entry.artifactUrl } : {}),
    }));
}

function roleSort(role: ModelRegistryProductionRole): number {
  switch (role) {
    case "production_candidate": return 0;
    case "research_kernel_lab": return 1;
    case "fallback": return 2;
  }
}
