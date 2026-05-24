import type { MemoryProviderCapabilities, MemoryProviderMode } from "./types";

export const DEFAULT_BROWSER_VECTOR_DIMENSION = 384;

export function normalizeMemoryProviderMode(mode: string | undefined): MemoryProviderMode | "sidecar" {
  const normalized = mode?.trim() || "browser-vector";
  if (normalized === "sidecar") return "sidecar";
  if (normalized === "indexeddb") return "indexeddb";
  if (normalized === "browser-vector") return "browser-vector";
  if (normalized === "remote-http") return "remote-http";
  if (normalized === "lancedb-sidecar") return "lancedb-sidecar";
  if (normalized === "unavailable") return "unavailable";
  return "browser-vector";
}

export function getMemoryProviderCapabilities(mode: MemoryProviderMode): MemoryProviderCapabilities {
  if (mode === "browser-vector" || mode === "indexeddb") {
    return {
      mode,
      storage: "indexeddb",
      localOnly: true,
      vectorSearch: true,
      deterministicSearch: true,
      metadataFilters: true,
      vectorDimension: DEFAULT_BROWSER_VECTOR_DIMENSION,
      persistent: true,
      importExport: true,
      contextPackTracePersistence: true,
      remoteSync: false,
    };
  }
  if (mode === "lancedb-sidecar") {
    return {
      mode,
      storage: "lancedb",
      localOnly: true,
      vectorSearch: true,
      deterministicSearch: true,
      metadataFilters: true,
      vectorDimension: DEFAULT_BROWSER_VECTOR_DIMENSION,
      persistent: true,
      importExport: true,
      contextPackTracePersistence: true,
      remoteSync: false,
    };
  }
  if (mode === "remote-http") {
    return {
      mode,
      storage: "remote-http",
      localOnly: false,
      vectorSearch: true,
      deterministicSearch: true,
      metadataFilters: true,
      vectorDimension: DEFAULT_BROWSER_VECTOR_DIMENSION,
      persistent: true,
      importExport: true,
      contextPackTracePersistence: true,
      remoteSync: true,
    };
  }
  return {
    mode,
    storage: "none",
    localOnly: false,
    vectorSearch: false,
    deterministicSearch: false,
    metadataFilters: false,
    persistent: false,
    importExport: false,
    contextPackTracePersistence: false,
    remoteSync: false,
  };
}
