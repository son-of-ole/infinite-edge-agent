import { getBrowserBackendRegistryEntry, type BrowserBackendProductionRole } from "./backendBroker";

export interface ProductionReadinessInput {
  memoryProvider: string;
  allowMemoryFallback: boolean;
  llmBackend: string;
  unlockedModelManifestPath?: string;
  unlockedModelManifestSha256?: string;
  unlockedManifestFormat?: string;
  unlockedAllowFixture?: boolean;
  unlockedBackendPreference?: string;
  requireWebGpuKernels?: boolean;
  requireUnlockedRuntime?: boolean;
  compiledBackendAdapterAvailable?: boolean;
  remoteMemoryUrl: string;
  remoteMemoryTenantId?: string | undefined;
  remoteMemoryCellId?: string | undefined;
  hasPublicRemoteMemoryToken?: boolean;
  production: boolean;
}

export interface ProductionReadinessReport {
  ready: boolean;
  blockers: string[];
  warnings: string[];
  backendId?: string;
  backendProductionRole?: BrowserBackendProductionRole;
  deployBackendId?: string;
  researchBackendId?: string;
}

export function evaluateProductionReadiness(input: ProductionReadinessInput): ProductionReadinessReport {
  const blockers: string[] = [];
  const warnings: string[] = [];
  const backendEntry = getBrowserBackendRegistryEntry(input.llmBackend);

  if (input.memoryProvider === "remote-http" && !input.remoteMemoryUrl.trim() && !input.allowMemoryFallback) {
    blockers.push("Remote memory requires VITE_REMOTE_MEMORY_URL.");
  } else if (input.memoryProvider === "remote-http" && !input.remoteMemoryUrl.trim()) {
    warnings.push("Remote memory URL is not configured; startup will use browser-vector memory because fallback is enabled.");
  }

  if (input.memoryProvider === "remote-http" && input.remoteMemoryUrl.trim()) {
    const proxyCanInjectScope = isSameOriginRelativeUrl(input.remoteMemoryUrl);
    if (!proxyCanInjectScope && !input.remoteMemoryTenantId?.trim() && !input.allowMemoryFallback) {
      blockers.push("Remote memory requires VITE_REMOTE_MEMORY_TENANT_ID unless a same-origin proxy injects scope.");
    } else if (!proxyCanInjectScope && !input.remoteMemoryTenantId?.trim()) {
      warnings.push("Remote memory tenant scope is not configured; startup can fall back to browser-vector memory if the remote provider is unavailable.");
    }
    if (!proxyCanInjectScope && !input.remoteMemoryCellId?.trim() && !input.allowMemoryFallback) {
      blockers.push("Remote memory requires VITE_REMOTE_MEMORY_CELL_ID unless a same-origin proxy injects scope.");
    } else if (!proxyCanInjectScope && !input.remoteMemoryCellId?.trim()) {
      warnings.push("Remote memory cell scope is not configured; startup can fall back to browser-vector memory if the remote provider is unavailable.");
    }
    if (proxyCanInjectScope && (!input.remoteMemoryTenantId?.trim() || !input.remoteMemoryCellId?.trim())) {
      warnings.push("Remote memory is using a same-origin proxy without browser tenant/cell scope; the proxy must inject and authorize tenant and cell values before forwarding.");
    }
  }

  if (
    input.memoryProvider === "remote-http"
    && input.hasPublicRemoteMemoryToken
    && (input.production || !isLocalOrSameOriginRemoteUrl(input.remoteMemoryUrl))
  ) {
    blockers.push("VITE_REMOTE_MEMORY_TOKEN is browser-bundled and cannot be used as a production remote memory secret. Use a same-origin authenticated proxy or secure cookie/session layer instead.");
  }

  if (input.production && (input.memoryProvider === "browser-vector" || input.memoryProvider === "indexeddb")) {
    warnings.push(input.memoryProvider === "browser-vector"
      ? "Using browser-native vector memory; data stays local to each browser profile and remote sync is disabled."
      : "Using browser IndexedDB memory compatibility alias; data stays local to each browser profile.");
  }

  if (input.production && input.allowMemoryFallback && input.memoryProvider !== "indexeddb" && input.memoryProvider !== "browser-vector") {
    warnings.push("Remote memory fallback is enabled; production may continue with local browser memory if the remote API is unavailable.");
  }

  if (input.requireUnlockedRuntime && input.llmBackend !== "unlocked-browser-transformer") {
    blockers.push("Full-control SSA/KV/TSP runtime requires VITE_LLM_BACKEND=unlocked-browser-transformer.");
  }

  if (!backendEntry) {
    blockers.push(`Unsupported inference backend "${input.llmBackend}". Backend Broker only ships registered browser backends.`);
  }

  if (input.llmBackend === "unlocked-browser-transformer") {
    warnings.push("unlocked-browser-transformer is registered as the Custom WebGPU Kernel Lab; production deploy readiness should be proven by a compiled production backend unless this backend passes the production speed and quality gates.");
    const strictUnlocked = input.production || input.requireUnlockedRuntime === true;
    if (strictUnlocked && !input.unlockedModelManifestPath?.trim()) {
      blockers.push(input.production
        ? "Unlocked browser transformer production requires VITE_UNLOCKED_MODEL_MANIFEST_PATH for converted model weights."
        : "Unlocked browser transformer requires VITE_UNLOCKED_MODEL_MANIFEST_PATH when full-control runtime is required.");
    }
    if (strictUnlocked && !input.unlockedModelManifestSha256?.trim()) {
      blockers.push(input.production
        ? "Unlocked browser transformer production requires VITE_UNLOCKED_MODEL_MANIFEST_SHA256 for manifest integrity."
        : "Unlocked browser transformer requires VITE_UNLOCKED_MODEL_MANIFEST_SHA256 when full-control runtime is required.");
    }
    if (strictUnlocked && input.unlockedManifestFormat !== "sharded") {
      blockers.push(input.production
        ? "Unlocked browser transformer production requires VITE_UNLOCKED_MANIFEST_FORMAT=sharded."
        : "Unlocked browser transformer requires VITE_UNLOCKED_MANIFEST_FORMAT=sharded when full-control runtime is required.");
    }
    if (strictUnlocked && input.unlockedBackendPreference !== "webgpu") {
      blockers.push(input.production
        ? "Unlocked browser transformer production requires VITE_UNLOCKED_BACKEND_PREFERENCE=webgpu."
        : "Unlocked browser transformer requires VITE_UNLOCKED_BACKEND_PREFERENCE=webgpu when full-control runtime is required.");
    }
    if (strictUnlocked && input.requireWebGpuKernels !== true) {
      blockers.push(input.production
        ? "Unlocked browser transformer production requires VITE_REQUIRE_WEBGPU_KERNELS=true so CPU-reference fallback cannot run."
        : "Unlocked browser transformer requires VITE_REQUIRE_WEBGPU_KERNELS=true when full-control runtime is required.");
    }
    if (input.unlockedModelManifestSha256?.trim()) {
      try {
        normalizeSha256(input.unlockedModelManifestSha256);
      } catch (error) {
        blockers.push(error instanceof Error ? error.message : String(error));
      }
    }
    if (input.unlockedAllowFixture && input.production) {
      blockers.push("VITE_UNLOCKED_ALLOW_FIXTURE cannot be true in production.");
    } else if (input.unlockedAllowFixture && input.requireUnlockedRuntime) {
      blockers.push("VITE_UNLOCKED_ALLOW_FIXTURE cannot be true when full-control runtime is required.");
    }
    if (!strictUnlocked && input.unlockedAllowFixture && !input.unlockedModelManifestPath?.trim()) {
      warnings.push("Unlocked browser transformer is using fixture weights for local tensor-control proof; production requires VITE_UNLOCKED_MODEL_MANIFEST_PATH.");
    }
  }

  if (backendEntry && input.llmBackend !== "unlocked-browser-transformer" && backendEntry.productionRole !== "production_candidate") {
    blockers.push(`Inference backend "${input.llmBackend}" is registered as ${backendEntry.productionRole}, not a production answer backend.`);
  }
  if (
    backendEntry
    && backendEntry.productionRole === "production_candidate"
    && input.compiledBackendAdapterAvailable !== true
  ) {
    blockers.push(`Compiled production backend "${backendEntry.backendId}" is registered but its runtime adapter is not available in this build.`);
  }

  return {
    ready: blockers.length === 0,
    blockers,
    warnings,
    ...(backendEntry
      ? {
          backendId: backendEntry.backendId,
          backendProductionRole: backendEntry.productionRole,
          ...(backendEntry.productionRole === "production_candidate" ? { deployBackendId: backendEntry.backendId } : {}),
          ...(backendEntry.productionRole === "research_kernel_lab" ? { researchBackendId: backendEntry.backendId } : {}),
        }
      : {}),
  };
}

export function markInitializationFailure(
  report: ProductionReadinessReport,
  message: string,
): ProductionReadinessReport {
  const blocker = message.trim();
  if (!blocker) return report;
  return {
    ready: false,
    blockers: report.blockers.includes(blocker)
      ? report.blockers
      : [...report.blockers, blocker],
    warnings: report.warnings,
  };
}

function isSameOriginRelativeUrl(value: string): boolean {
  return value.startsWith("/") && !value.startsWith("//");
}

function isLocalOrSameOriginRemoteUrl(value: string): boolean {
  if (!value.trim()) return true;
  if (isSameOriginRelativeUrl(value)) return true;
  try {
    const url = new URL(value);
    return url.hostname === "localhost" || url.hostname === "127.0.0.1" || url.hostname === "::1";
  } catch {
    return false;
  }
}

function normalizeSha256(value: string): string {
  const normalized = value.trim().toLowerCase();
  if (!/^[a-f0-9]{64}$/.test(normalized)) {
    throw new Error("VITE_UNLOCKED_MODEL_MANIFEST_SHA256 must be a 64-character hexadecimal SHA-256 digest.");
  }
  return normalized;
}
