import { getMemoryProviderCapabilities, type MemoryProviderCapabilities, type MemoryProviderMode, type MemoryStore } from "@infinite-edge-agent/core";
import { IndexedDbMemoryStore } from "./indexedDbMemoryStore";
import { RemoteMemoryStore } from "./remoteMemoryStore";
import { SidecarMemoryStore } from "./sidecarMemoryStore";

export interface HybridMemoryOptions {
  provider: MemoryProviderMode | "sidecar";
  allowFallback?: boolean;
  remoteUrl: string;
  remoteToken?: string;
  remoteCredentials?: RequestCredentials;
  remoteTenantId?: string;
  remoteCellId?: string;
  useSidecar: boolean;
  sidecarUrl: string;
}

export async function createMemoryStore(options: HybridMemoryOptions): Promise<{ store: MemoryStore; mode: MemoryProviderMode; capabilities: MemoryProviderCapabilities }> {
  const allowFallback = options.allowFallback !== false;
  if (options.provider === "remote-http") {
    const remote = new RemoteMemoryStore({
      baseUrl: options.remoteUrl,
      credentials: options.remoteCredentials ?? "same-origin",
      ...(options.remoteToken ? { token: options.remoteToken } : {}),
      ...(options.remoteTenantId ? { tenantId: options.remoteTenantId } : {}),
      ...(options.remoteCellId ? { cellId: options.remoteCellId } : {}),
    });
    if (await remote.health()) {
      return withCapabilities(remote, "remote-http");
    }
    if (!allowFallback) {
      throw new Error("Remote memory provider is unavailable and fallback is disabled.");
    }
    console.warn("Remote memory provider unavailable. Falling back to browser-vector because fallback is enabled.");
  }

  if (options.provider === "sidecar" || options.useSidecar) {
    const sidecar = new SidecarMemoryStore(options.sidecarUrl);
    if (await sidecar.health()) {
      return withCapabilities(sidecar, "lancedb-sidecar");
    }
    if (!allowFallback && (options.provider === "sidecar" || options.useSidecar)) {
      throw new Error("Memory sidecar is unavailable and fallback is disabled.");
    }
    console.warn("LanceDB sidecar requested but unavailable. Falling back to browser-vector.");
  }
  if (options.provider === "indexeddb") {
    return withCapabilities(new IndexedDbMemoryStore(), "indexeddb");
  }
  return withCapabilities(new IndexedDbMemoryStore(), "browser-vector");
}

function withCapabilities(store: MemoryStore, mode: MemoryProviderMode) {
  return { store, mode, capabilities: getMemoryProviderCapabilities(mode) };
}
