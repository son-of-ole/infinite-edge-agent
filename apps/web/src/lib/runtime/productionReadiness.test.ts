import { describe, expect, it } from "vitest";
import { evaluateProductionReadiness, markInitializationFailure, type ProductionReadinessInput } from "./productionReadiness";

const unlockedReadyInput = {
  memoryProvider: "browser-vector",
  allowMemoryFallback: false,
  llmBackend: "unlocked-browser-transformer",
  remoteMemoryUrl: "",
  unlockedModelManifestPath: "/models/qwen3-unlocked/manifest.json",
  unlockedModelManifestSha256: "b".repeat(64),
  unlockedManifestFormat: "sharded",
  unlockedAllowFixture: false,
  unlockedBackendPreference: "webgpu",
  requireWebGpuKernels: true,
  requireUnlockedRuntime: true,
  production: true,
} satisfies ProductionReadinessInput;

function evaluate(overrides: Partial<ProductionReadinessInput> = {}) {
  return evaluateProductionReadiness({ ...unlockedReadyInput, ...overrides });
}

describe("evaluateProductionReadiness", () => {
  it("accepts unlocked production only with manifest path, manifest hash, and fixture mode disabled", () => {
    const report = evaluate();

    expect(report.ready).toBe(true);
    expect(report.blockers).toEqual([]);
    expect(report.warnings).toContain("Using browser-native vector memory; data stays local to each browser profile and remote sync is disabled.");
  });

  it("blocks production remote memory when a browser-bundled bearer token is configured", () => {
    const report = evaluate({
      memoryProvider: "remote-http",
      remoteMemoryUrl: "https://example.test/api/edge-ai",
      remoteMemoryTenantId: "tenant_1",
      remoteMemoryCellId: "cell_1",
      hasPublicRemoteMemoryToken: true,
    });

    expect(report.ready).toBe(false);
    expect(report.blockers).toContain("VITE_REMOTE_MEMORY_TOKEN is browser-bundled and cannot be used as a production remote memory secret. Use a same-origin authenticated proxy or secure cookie/session layer instead.");
  });

  it("allows production remote memory through a configured proxy or cookie session without a public token", () => {
    const report = evaluate({
      memoryProvider: "remote-http",
      remoteMemoryUrl: "/api/edge-ai",
      hasPublicRemoteMemoryToken: false,
    });

    expect(report.ready).toBe(true);
    expect(report.blockers).toEqual([]);
    expect(report.warnings).toContain("Remote memory is using a same-origin proxy without browser tenant/cell scope; the proxy must inject and authorize tenant and cell values before forwarding.");
  });

  it("blocks remote memory without tenant and cell scope when no same-origin proxy can inject it", () => {
    const report = evaluate({
      memoryProvider: "remote-http",
      remoteMemoryUrl: "https://example.test/api/edge-ai",
      hasPublicRemoteMemoryToken: false,
    });

    expect(report.ready).toBe(false);
    expect(report.blockers).toContain("Remote memory requires VITE_REMOTE_MEMORY_TENANT_ID unless a same-origin proxy injects scope.");
    expect(report.blockers).toContain("Remote memory requires VITE_REMOTE_MEMORY_CELL_ID unless a same-origin proxy injects scope.");
  });

  it("does not block remote memory config gaps when browser-vector fallback is enabled", () => {
    const missingUrl = evaluate({
      memoryProvider: "remote-http",
      allowMemoryFallback: true,
      remoteMemoryUrl: "",
    });
    const missingScope = evaluate({
      memoryProvider: "remote-http",
      allowMemoryFallback: true,
      remoteMemoryUrl: "https://example.test/api/edge-ai",
    });

    expect(missingUrl.ready).toBe(true);
    expect(missingUrl.blockers).toEqual([]);
    expect(missingUrl.warnings).toContain("Remote memory URL is not configured; startup will use browser-vector memory because fallback is enabled.");
    expect(missingScope.ready).toBe(true);
    expect(missingScope.blockers).toEqual([]);
    expect(missingScope.warnings).toContain("Remote memory tenant scope is not configured; startup can fall back to browser-vector memory if the remote provider is unavailable.");
    expect(missingScope.warnings).toContain("Remote memory cell scope is not configured; startup can fall back to browser-vector memory if the remote provider is unavailable.");
  });

  it("keeps IndexedDB as a production-compatible browser memory alias", () => {
    const report = evaluate({
      memoryProvider: "indexeddb",
    });

    expect(report.ready).toBe(true);
    expect(report.warnings).toContain("Using browser IndexedDB memory compatibility alias; data stays local to each browser profile.");
  });

  it("blocks remote memory when the URL is missing and fallback is disabled", () => {
    const report = evaluate({
      memoryProvider: "remote-http",
      allowMemoryFallback: false,
      remoteMemoryUrl: "",
    });

    expect(report.ready).toBe(false);
    expect(report.blockers).toContain("Remote memory requires VITE_REMOTE_MEMORY_URL.");
  });

  it("blocks unlocked-required development when fixture weights would be used instead of real Qwen", () => {
    const report = evaluate({
      memoryProvider: "indexeddb",
      unlockedModelManifestPath: "",
      unlockedModelManifestSha256: "",
      unlockedManifestFormat: "",
      unlockedAllowFixture: true,
      requireUnlockedRuntime: true,
      production: false,
    });

    expect(report.ready).toBe(false);
    expect(report.blockers).toContain("Unlocked browser transformer requires VITE_UNLOCKED_MODEL_MANIFEST_PATH when full-control runtime is required.");
    expect(report.blockers).toContain("Unlocked browser transformer requires VITE_UNLOCKED_MODEL_MANIFEST_SHA256 when full-control runtime is required.");
    expect(report.blockers).toContain("Unlocked browser transformer requires VITE_UNLOCKED_MANIFEST_FORMAT=sharded when full-control runtime is required.");
    expect(report.blockers).toContain("VITE_UNLOCKED_ALLOW_FIXTURE cannot be true when full-control runtime is required.");
  });

  it("blocks unlocked production without real manifest weights", () => {
    const report = evaluate({
      memoryProvider: "indexeddb",
      unlockedModelManifestPath: "",
      unlockedModelManifestSha256: "",
      unlockedManifestFormat: "",
      unlockedAllowFixture: true,
    });

    expect(report.ready).toBe(false);
    expect(report.blockers).toContain("Unlocked browser transformer production requires VITE_UNLOCKED_MODEL_MANIFEST_PATH for converted model weights.");
    expect(report.blockers).toContain("Unlocked browser transformer production requires VITE_UNLOCKED_MODEL_MANIFEST_SHA256 for manifest integrity.");
    expect(report.blockers).toContain("Unlocked browser transformer production requires VITE_UNLOCKED_MANIFEST_FORMAT=sharded.");
    expect(report.blockers).toContain("VITE_UNLOCKED_ALLOW_FIXTURE cannot be true in production.");
  });

  it("blocks malformed unlocked manifest SHA-256 values", () => {
    const report = evaluate({
      unlockedModelManifestSha256: "not-a-sha",
    });

    expect(report.ready).toBe(false);
    expect(report.blockers).toContain("VITE_UNLOCKED_MODEL_MANIFEST_SHA256 must be a 64-character hexadecimal SHA-256 digest.");
  });

  it("blocks strict unlocked production when WebGPU kernels are not required", () => {
    const report = evaluate({
      requireWebGpuKernels: false,
    });

    expect(report.ready).toBe(false);
    expect(report.blockers).toContain("Unlocked browser transformer production requires VITE_REQUIRE_WEBGPU_KERNELS=true so CPU-reference fallback cannot run.");
  });

  it("blocks every opaque inference backend when full unlocked runtime is required", () => {
    const report = evaluate({
      llmBackend: "opaque-browser-chat",
      production: false,
    });

    expect(report.ready).toBe(false);
    expect(report.blockers).toContain("Full-control SSA/KV/TSP runtime requires VITE_LLM_BACKEND=unlocked-browser-transformer.");
    expect(report.blockers).toContain('Unsupported inference backend "opaque-browser-chat". Backend Broker only ships registered browser backends.');
  });

  it("allows registered compiled browser backends to be the production answer path when full-control kernel lab proof is not required", () => {
    const report = evaluate({
      llmBackend: "compiled-browser-webllm",
      compiledBackendAdapterAvailable: true,
      requireUnlockedRuntime: false,
      unlockedModelManifestPath: "",
      unlockedModelManifestSha256: "",
      unlockedManifestFormat: "",
      unlockedBackendPreference: "",
      requireWebGpuKernels: false,
    });

    expect(report.ready).toBe(true);
    expect(report.deployBackendId).toBe("compiled-browser-webllm");
    expect(report.backendProductionRole).toBe("production_candidate");
    expect(report.researchBackendId).toBeUndefined();
  });

  it("blocks registered compiled browser backends until their runtime adapter is actually installed", () => {
    const report = evaluate({
      llmBackend: "compiled-browser-webllm",
      compiledBackendAdapterAvailable: false,
      requireUnlockedRuntime: false,
      unlockedModelManifestPath: "",
      unlockedModelManifestSha256: "",
      unlockedManifestFormat: "",
      unlockedBackendPreference: "",
      requireWebGpuKernels: false,
    });

    expect(report.ready).toBe(false);
    expect(report.blockers).toContain('Compiled production backend "compiled-browser-webllm" is registered but its runtime adapter is not available in this build.');
  });

  it("reports the custom unlocked runtime as a research Kernel Lab backend for deploy-readiness decisions", () => {
    const report = evaluate();

    expect(report.ready).toBe(true);
    expect(report.deployBackendId).toBeUndefined();
    expect(report.researchBackendId).toBe("unlocked-browser-transformer");
    expect(report.backendProductionRole).toBe("research_kernel_lab");
    expect(report.warnings).toContain("unlocked-browser-transformer is registered as the Custom WebGPU Kernel Lab; production deploy readiness should be proven by a compiled production backend unless this backend passes the production speed and quality gates.");
  });

  it("marks runtime initialization failures as readiness blockers without dropping warnings", () => {
    const report = markInitializationFailure({
      ready: true,
      blockers: [],
      warnings: ["existing warning"],
    }, "Unlocked model manifest is not served.");

    expect(report).toEqual({
      ready: false,
      blockers: ["Unlocked model manifest is not served."],
      warnings: ["existing warning"],
    });
  });
});
