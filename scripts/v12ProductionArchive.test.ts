import { mkdtemp, readFile, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import {
  buildV12ProductionArchiveArtifact,
  runV12ProductionArchive,
} from "./v12ProductionArchive";

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

function makePassingHostedBenchmarkArtifact() {
  const brokerSelection = {
    backendId: "compiled-browser-webllm",
    modelId: "Qwen3-0.6B-q4f16_1-MLC",
    productionRole: "production_candidate",
    deployReadyCandidate: true,
    reason: "compiled_first_grounded_answer",
    fallbackChain: ["unlocked-browser-transformer", "wasm-small-core"],
    proofRequirements: ["memory_grounding", "quality_canaries", "speed_floor", "backend_trace"],
  };

  return {
    name: "browser-preview-benchmark",
    createdAt: "2026-05-30T21:00:00.000Z",
    passed: true,
    summary: {
      runtimeBackendId: "compiled-browser-webllm",
      deployBackendId: "compiled-browser-webllm",
      productionDeployReadyPassed: true,
      compiledBackendReadyPassed: true,
      memoryGroundingRequired: true,
      memoryGroundingPassed: true,
      memoryExpectedHitPassed: true,
      memoryContextRebuildPassed: true,
      memoryAnswerOnlyPassed: true,
      directModelFactualProofUsed: false,
      expectedExactCheckCount: 1,
      expectedExactPassed: true,
      technicalProofOnly: false,
      productionQualityPassed: true,
      productionSpeedFloorPassed: true,
      productionSpeedTokensPerSecond: 2.7,
      productionSpeedFloorTokensPerSecond: 2,
      meanTokensPerSecond: 2.7,
      strictWebGpuPassed: true,
      cpuFallbackUsed: false,
      backendBrokerTraceCount: 1,
      backendBrokerSelectionPassed: true,
      backendBrokerSelectedBackendId: brokerSelection.backendId,
      backendBrokerSelectedModelId: brokerSelection.modelId,
      backendBrokerProductionRole: brokerSelection.productionRole,
      backendBrokerDeployReadyCandidate: brokerSelection.deployReadyCandidate,
      backendBrokerReason: brokerSelection.reason,
      backendBrokerProofRequirements: brokerSelection.proofRequirements,
    },
    runs: [
      {
        response: "Helena",
        runtimeTrace: {
          backend: "compiled-browser-webllm",
          brokerSelection,
        },
      },
    ],
  };
}

describe("v12 production archive", () => {
  it("fails when a hosted benchmark artifact is not supplied", async () => {
    const artifactDir = await mkdtemp(join(tmpdir(), "v12-production-archive-missing-"));

    const result = await runV12ProductionArchive({
      env: completeHostedEnv,
      artifactDir,
      createdAt: "2026-05-30T22:00:00.000Z",
    });

    expect(result.archive.passed).toBe(false);
    expect(result.archive.summary).toMatchObject({
      v12ProductionArchivePassed: false,
      v12ProductionHostedBenchmarkProofPassed: false,
      v12ProductionDeployBackendId: null,
      v12ProductionKernelLabBackendId: "unlocked-browser-transformer",
    });
    expect(result.archive.archive.blockers).toContain("backend_readiness_matrix: Compiled production backend is not deploy-ready because hosted benchmark proof is required and missing or failed.");
    expect(result.archive.archive.blockers).toContain("hosted_benchmark_proof: required but no HOSTED_BENCHMARK_ARTIFACT_PATH or report was provided.");
  });

  it("writes a strict production archive with hosted benchmark proof included", async () => {
    const artifactDir = await mkdtemp(join(tmpdir(), "v12-production-archive-"));
    const hostedBenchmarkPath = join(artifactDir, "browser-runtime-bench-latest.json");
    await writeFile(hostedBenchmarkPath, `${JSON.stringify(makePassingHostedBenchmarkArtifact(), null, 2)}\n`);

    const result = await runV12ProductionArchive({
      env: {
        ...completeHostedEnv,
        HOSTED_BENCHMARK_ARTIFACT_PATH: hostedBenchmarkPath,
      },
      artifactDir,
      createdAt: "2026-05-30T22:00:00.000Z",
    });

    expect(result.latestPath).toBe(join(artifactDir, "v12-production-archive-latest.json"));
    expect(result.resultPath).toBe(join(artifactDir, "v12-production-archive", "2026-05-30T22-00-00-000Z.json"));
    expect(result.archive).toMatchObject({
      name: "v12-production-archive",
      passed: true,
      summary: {
        v12ProductionArchivePassed: true,
        v12ProductionSuitePassed: true,
        v12ProductionDeployBackendId: "compiled-browser-webllm",
        v12ProductionKernelLabBackendId: "unlocked-browser-transformer",
        v12ProductionHostedBenchmarkProofRequired: true,
        v12ProductionHostedBenchmarkProofPassed: true,
        v12ProductionBackendReadinessProofBound: true,
        v12ProductionArtifactCount: 7,
        v12ProductionSuiteArtifactCount: 6,
        v12ProductionChildArtifactCount: 5,
        v12ProductionHostedBenchmarkRuntimeBackendId: "compiled-browser-webllm",
        v12ProductionHostedBenchmarkDeployBackendId: "compiled-browser-webllm",
        v12ProductionCompiledBackendReadyPassed: true,
        v12ProductionDeployReadyPassed: true,
        v12ProductionMemoryGroundingPassed: true,
        v12ProductionExpectedExactPassed: true,
        v12ProductionSpeedFloorPassed: true,
        v12ProductionMeanTokensPerSecond: 2.7,
        v12ProductionDirectModelFactualProofUsed: false,
        v12ProductionTechnicalProofOnly: false,
        v12ProductionCpuFallbackUsed: false,
        v12ProductionStrictWebGpuPassed: true,
        v12ProductionBackendBrokerSelectionPassed: true,
        v12ProductionBackendBrokerTraceCount: 1,
        v12ProductionBrokerSelectedBackendId: "compiled-browser-webllm",
        v12ProductionBrokerSelectedModelId: "Qwen3-0.6B-q4f16_1-MLC",
        v12ProductionBrokerProductionRole: "production_candidate",
        v12ProductionBrokerDeployReadyCandidate: true,
      },
    });

    const latest = JSON.parse(await readFile(result.latestPath, "utf8")) as ReturnType<typeof buildV12ProductionArchiveArtifact>;

    expect(latest.archive.suiteResult.childArtifacts.hostedBenchmarkProof?.passed).toBe(true);
    expect(latest.archive.suiteLatestPath).toBe(join(artifactDir, "v12-readiness-suite-latest.json"));
  });
});
