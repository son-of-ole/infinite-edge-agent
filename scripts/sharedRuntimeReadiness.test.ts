import { mkdtemp, readFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import {
  buildSharedRuntimeReadinessArtifact,
  evaluateSharedRuntimeReadiness,
  writeSharedRuntimeReadinessArtifact,
} from "./sharedRuntimeReadiness";
import { evaluateBackendReadinessMatrix } from "./backendReadinessMatrix";
import { evaluateHostedDeploymentProfile } from "./hostedDeploymentProfile";

const completeHostedEnv = {
  VITE_LLM_BACKEND: "compiled-browser-webllm",
  VITE_DEFAULT_MODEL: "Qwen3-0.6B-q4f16_1-MLC",
  VITE_COMPILED_WEBLLM_ENABLED: "true",
  VITE_REQUIRE_UNLOCKED_RUNTIME: "false",
  VITE_MTP_ENABLED: "false",
  VITE_BENCHMARK_TELEMETRY_ENABLED: "true",
  VITE_BENCHMARK_TELEMETRY_URL: "/api/benchmark-runs",
  BENCHMARK_TELEMETRY_ENABLED: "true",
  BENCHMARK_TELEMETRY_STORAGE: "postgres",
  BENCHMARK_TELEMETRY_DATABASE_URL: "postgres://example.test/infinite_edge_agent",
  BENCHMARK_TELEMETRY_ADMIN_TOKEN: "admin-token",
  BENCHMARK_TELEMETRY_RATE_LIMIT_MAX: "60",
  BENCHMARK_TELEMETRY_RATE_LIMIT_WINDOW_MS: "600000",
  HOSTED_PRODUCTION_BENCHMARK_URL:
    "https://agent.example.com/__bench/browser-runtime?backend=compiled-browser-webllm&modelId=Qwen3-0.6B-q4f16_1-MLC&memoryGrounding=montana_capital&expectedExact=Helena&submitTelemetry=true&qwenThinkingMode=disabled",
};

function makePassingBackendMatrix() {
  return evaluateBackendReadinessMatrix({
    hostedProfile: evaluateHostedDeploymentProfile(completeHostedEnv),
  });
}

describe("evaluateSharedRuntimeReadiness", () => {
  it("proves shared memory/context runtime coverage across compiled deploy and Kernel Lab backends", () => {
    const report = evaluateSharedRuntimeReadiness({
      backendMatrix: makePassingBackendMatrix(),
    });

    expect(report.passed).toBe(true);
    expect(report.blockers).toEqual([]);
    expect(report.coveredBackendIds).toEqual(["compiled-browser-webllm", "unlocked-browser-transformer", "wasm-small-core"]);
    expect(report.memoryProviders).toEqual(expect.arrayContaining([
      expect.objectContaining({
        mode: "browser-vector",
        vectorSearch: true,
        deterministicSearch: true,
        contextPackTracePersistence: true,
      }),
      expect.objectContaining({
        mode: "remote-http",
        vectorSearch: true,
        deterministicSearch: true,
        contextPackTracePersistence: true,
      }),
      expect.objectContaining({
        mode: "lancedb-sidecar",
        vectorSearch: true,
        deterministicSearch: true,
        contextPackTracePersistence: true,
      }),
    ]));
    expect(report.contextRuntime).toMatchObject({
      sharedAcrossBackends: true,
      requiresContextPackTraceStore: true,
      writesContextPackTraceBeforeGeneration: true,
      persistsRuntimeTraceAfterGeneration: true,
      usesRetrievedMemoryBeforeBackendSelection: true,
      passesBackendProfileIntoRuntimePlan: true,
    });
    expect(report.modelRegistryAlignment).toMatchObject({
      aligned: true,
      modelCount: 3,
      publicOptionCount: 2,
      publicDeployOptionCount: 1,
      publicKernelLabOptionCount: 1,
    });
  });

  it("fails when backend-specific deploy readiness is not proven", () => {
    const failingMatrix = evaluateBackendReadinessMatrix({
      hostedProfile: evaluateHostedDeploymentProfile({
        ...completeHostedEnv,
        VITE_LLM_BACKEND: "unlocked-browser-transformer",
      }),
    });
    const report = evaluateSharedRuntimeReadiness({ backendMatrix: failingMatrix });

    expect(report.passed).toBe(false);
    expect(report.blockers).toContain("Shared runtime readiness requires a deploy-ready compiled backend in the backend readiness matrix.");
  });

  it("builds a release-gate friendly shared runtime artifact", () => {
    const report = evaluateSharedRuntimeReadiness({ backendMatrix: makePassingBackendMatrix() });
    const artifact = buildSharedRuntimeReadinessArtifact(report, "2026-05-30T18:00:00.000Z");

    expect(artifact).toMatchObject({
      name: "shared-runtime-readiness",
      createdAt: "2026-05-30T18:00:00.000Z",
      passed: true,
      summary: {
        sharedRuntimeReadinessPassed: true,
        sharedRuntimeCoveredBackendCount: 3,
        sharedRuntimeDeployBackendId: "compiled-browser-webllm",
        sharedRuntimeKernelLabBackendId: "unlocked-browser-transformer",
        sharedRuntimeFallbackBackendId: "wasm-small-core",
        sharedRuntimeBackendRoleBoundaryPassed: true,
        sharedRuntimeMemoryProviderCount: 3,
        sharedRuntimeModelRegistryAligned: true,
        sharedRuntimeModelRegistryModelCount: 3,
        sharedRuntimePublicModelOptionCount: 2,
        sharedRuntimePublicDeployOptionCount: 1,
        sharedRuntimePublicKernelLabOptionCount: 1,
        sharedRuntimeContextTraceRequired: true,
        sharedRuntimeContextTraceBeforeGeneration: true,
        sharedRuntimeTracePersistedAfterGeneration: true,
        sharedRuntimeBackendProfilePassedToPlan: true,
      },
    });
  });

  it("writes latest and timestamped shared runtime readiness artifacts", async () => {
    const artifactDir = await mkdtemp(join(tmpdir(), "shared-runtime-readiness-"));
    const report = evaluateSharedRuntimeReadiness({ backendMatrix: makePassingBackendMatrix() });

    const written = await writeSharedRuntimeReadinessArtifact(report, {
      artifactDir,
      createdAt: "2026-05-30T18:00:00.000Z",
    });

    expect(written.latestPath).toBe(join(artifactDir, "shared-runtime-readiness-latest.json"));
    expect(written.resultPath).toBe(join(artifactDir, "shared-runtime-readiness", "2026-05-30T18-00-00-000Z.json"));

    const latest = JSON.parse(await readFile(written.latestPath, "utf8")) as ReturnType<typeof buildSharedRuntimeReadinessArtifact>;

    expect(latest.passed).toBe(true);
    expect(latest.summary.sharedRuntimeCoveredBackendCount).toBe(3);
  });
});
