import { mkdtemp, readFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import {
  buildBackendReadinessMatrixArtifact,
  evaluateBackendReadinessMatrix,
  writeBackendReadinessMatrixArtifact,
} from "./backendReadinessMatrix";
import { evaluateHostedBenchmarkProof } from "./hostedBenchmarkProof";
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

function makePassingHostedBenchmarkProofReport(options: { expectedSourceGitSha?: string | null } = {}) {
  const brokerSelection = {
    backendId: "compiled-browser-webllm",
    modelId: "Qwen3-0.6B-q4f16_1-MLC",
    productionRole: "production_candidate",
    deployReadyCandidate: true,
    reason: "compiled_first_grounded_answer",
    fallbackChain: ["unlocked-browser-transformer", "wasm-small-core"],
    proofRequirements: ["memory_grounding", "quality_canaries", "speed_floor", "backend_trace"],
  };

  return evaluateHostedBenchmarkProof({
    artifact: {
      name: "browser-preview-benchmark",
      createdAt: "2026-05-30T21:00:00.000Z",
      schemaVersion: 2,
      passed: true,
      summary: {
        v12ProductionProofSchemaVersion: 2,
        v12ProductionProofSourceGitSha: "abc123",
        runtimeBackendId: "compiled-browser-webllm",
        modelId: "Qwen3-0.6B-q4f16_1-MLC",
        runtimeModelId: "Qwen3-0.6B-q4f16_1-MLC",
        runtimeBackendRole: "production_candidate",
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
        expectedExactPassCount: 1,
        expectedExactPassed: true,
        technicalProofOnly: false,
        productionQualityPassed: true,
        productionSpeedFloorPassed: true,
        productionSpeedTokensPerSecond: 2.7,
        productionSpeedFloorTokensPerSecond: 2,
        meanTokensPerSecond: 2.7,
        benchmarkTelemetryRequested: true,
        benchmarkTelemetryConfigured: true,
        benchmarkTelemetrySubmitted: true,
        benchmarkTelemetryStatus: 202,
        strictWebGpuPassed: true,
        cpuFallbackUsed: false,
        benchmarkGpuLabelEvidencePassed: true,
        benchmarkGpuVendor: "apple",
        benchmarkGpuDescription: "Apple M3",
        benchmarkWebGlRenderer: "ANGLE Metal Renderer: Apple M3",
        backendBrokerTraceCount: 1,
        backendBrokerSelectionPassed: true,
        backendBrokerSelectedBackendId: brokerSelection.backendId,
        backendBrokerSelectedModelId: brokerSelection.modelId,
        backendBrokerProductionRole: brokerSelection.productionRole,
        backendBrokerDeployReadyCandidate: brokerSelection.deployReadyCandidate,
        backendBrokerReason: brokerSelection.reason,
        backendBrokerProofRequirements: brokerSelection.proofRequirements,
        backendBrokerDeployBackendId: "compiled-browser-webllm",
        backendBrokerKernelLabBackendId: "unlocked-browser-transformer",
        backendBrokerFallbackBackendId: "wasm-small-core",
        backendBrokerFallbackBackendCount: 1,
        backendBrokerFallbackDeployReadyCandidate: false,
        backendBrokerRoleBoundaryPassed: true,
      },
      runs: [
        {
          response: "Helena",
          expectedAnswerOnlyPassed: true,
          memoryGrounding: {
            mode: "seeded_browser_vector_context_rebuild",
            caseId: "montana_capital",
            corpusCount: 16,
            retrievedMemoryIds: ["bench_memory_montana_capital"],
            includedMemoryIds: ["bench_memory_montana_capital"],
            expectedMemoryIds: ["bench_memory_montana_capital"],
            expectedMemoryHitPassed: true,
            contextRebuildPassed: true,
            answerOnlyExpected: true,
            answerOnlyPassed: true,
            contextEstimatedTokens: 42,
            retrievalMs: 2,
            contextRebuildMs: 1,
            retrievalRank: 1,
            retrievalScore: 0.99,
            retrievalTopScoreMargin: 0.4,
          },
          runtimeTrace: {
            backend: "compiled-browser-webllm",
            brokerSelection,
          },
        },
      ],
    },
    expectedSourceGitSha: options.expectedSourceGitSha,
  });
}

describe("evaluateBackendReadinessMatrix", () => {
  it("marks the compiled backend as deploy-ready only when hosted profile proof passes", () => {
    const hostedProfile = evaluateHostedDeploymentProfile(completeHostedEnv);
    const matrix = evaluateBackendReadinessMatrix({ hostedProfile });

    expect(matrix.passed).toBe(true);
    expect(matrix.deployBackendId).toBe("compiled-browser-webllm");
    expect(matrix.researchBackendIds).toEqual(["unlocked-browser-transformer"]);
    expect(matrix.backends).toEqual(expect.arrayContaining([
      expect.objectContaining({
        backendId: "compiled-browser-webllm",
        productionRole: "production_candidate",
        readinessStatus: "deploy_ready",
        deployReady: true,
        proofSource: "hosted_deployment_profile",
      }),
      expect.objectContaining({
        backendId: "unlocked-browser-transformer",
        productionRole: "research_kernel_lab",
        readinessStatus: "research_only",
        deployReady: false,
        proofSource: "kernel_lab_research_gates",
      }),
      expect.objectContaining({
        backendId: "wasm-small-core",
        productionRole: "fallback",
        readinessStatus: "fallback_only",
        deployReady: false,
      }),
    ]));
  });

  it("does not mark the compiled backend deploy-ready without hosted benchmark proof when proof is required", () => {
    const hostedProfile = evaluateHostedDeploymentProfile(completeHostedEnv);
    const matrix = evaluateBackendReadinessMatrix({
      hostedProfile,
      requireHostedBenchmarkProof: true,
    });

    expect(matrix.passed).toBe(false);
    expect(matrix.deployBackendId).toBeNull();
    expect(matrix.blockers).toContain("Compiled production backend is not deploy-ready because hosted benchmark proof is required and missing or failed.");
    expect(matrix.backends.find((backend) => backend.backendId === "compiled-browser-webllm")).toMatchObject({
      readinessStatus: "blocked",
      deployReady: false,
      proofSource: "hosted_deployment_profile+hosted_benchmark_proof",
      blockers: expect.arrayContaining([
        "Hosted benchmark proof is required to mark compiled-browser-webllm deploy-ready.",
      ]),
      proofRequirements: expect.arrayContaining([
        "hosted_benchmark_artifact_passed",
      ]),
    });
    expect(buildBackendReadinessMatrixArtifact(matrix, "2026-05-30T17:45:00.000Z").summary).toMatchObject({
      backendReadinessCompiledHostedProfilePassed: true,
      backendReadinessCompiledDeployReady: false,
      backendReadinessHostedBenchmarkProofSourceBound: false,
    });
  });

  it("marks the compiled backend deploy-ready when required hosted benchmark proof passes", () => {
    const hostedProfile = evaluateHostedDeploymentProfile(completeHostedEnv);
    const matrix = evaluateBackendReadinessMatrix({
      hostedProfile,
      hostedBenchmarkProof: makePassingHostedBenchmarkProofReport({ expectedSourceGitSha: "abc123" }),
      requireHostedBenchmarkProof: true,
    });

    expect(matrix.passed).toBe(true);
    expect(matrix.deployBackendId).toBe("compiled-browser-webllm");
    expect(matrix.backends.find((backend) => backend.backendId === "compiled-browser-webllm")).toMatchObject({
      readinessStatus: "deploy_ready",
      deployReady: true,
      proofSource: "hosted_deployment_profile+hosted_benchmark_proof",
      proofRequirements: expect.arrayContaining([
        "hosted_benchmark_artifact_passed",
        "hosted_benchmark_artifact_source_bound",
      ]),
      hostedBenchmarkProofSourceGitSha: "abc123",
      hostedBenchmarkExpectedSourceGitSha: "abc123",
      hostedBenchmarkProofSourceBound: true,
    });

    expect(buildBackendReadinessMatrixArtifact(matrix, "2026-05-30T17:30:00.000Z").summary).toMatchObject({
      backendReadinessProofBoundToHostedBenchmark: true,
      backendReadinessHostedBenchmarkProofSourceGitSha: "abc123",
      backendReadinessHostedBenchmarkExpectedSourceGitSha: "abc123",
      backendReadinessHostedBenchmarkProofSourceBound: true,
    });
  });

  it("does not mark the compiled backend deploy-ready when required hosted benchmark proof is not source-bound", () => {
    const hostedProfile = evaluateHostedDeploymentProfile(completeHostedEnv);
    const matrix = evaluateBackendReadinessMatrix({
      hostedProfile,
      hostedBenchmarkProof: makePassingHostedBenchmarkProofReport(),
      requireHostedBenchmarkProof: true,
    });

    expect(matrix.passed).toBe(false);
    expect(matrix.deployBackendId).toBeNull();
    expect(matrix.blockers).toContain("Compiled production backend is not deploy-ready because hosted benchmark proof is required and missing, failed, or not source-bound.");
    expect(matrix.backends.find((backend) => backend.backendId === "compiled-browser-webllm")).toMatchObject({
      readinessStatus: "blocked",
      deployReady: false,
      hostedBenchmarkProofSourceGitSha: "abc123",
      hostedBenchmarkExpectedSourceGitSha: null,
      hostedBenchmarkProofSourceBound: false,
      blockers: expect.arrayContaining([
        "Hosted benchmark proof must be source-bound to the expected deployment commit.",
      ]),
    });
  });

  it("fails the matrix when the hosted profile does not prove the compiled production backend", () => {
    const hostedProfile = evaluateHostedDeploymentProfile({
      ...completeHostedEnv,
      VITE_LLM_BACKEND: "unlocked-browser-transformer",
    });
    const matrix = evaluateBackendReadinessMatrix({ hostedProfile });

    expect(matrix.passed).toBe(false);
    expect(matrix.deployBackendId).toBeNull();
    expect(matrix.blockers).toContain("Compiled production backend is not deploy-ready because hosted deployment profile failed.");
    expect(matrix.backends.find((backend) => backend.backendId === "compiled-browser-webllm")).toMatchObject({
      readinessStatus: "blocked",
      deployReady: false,
    });
  });

  it("builds a release-gate friendly artifact with backend role and readiness summary fields", () => {
    const matrix = evaluateBackendReadinessMatrix({
      hostedProfile: evaluateHostedDeploymentProfile(completeHostedEnv),
    });
    const artifact = buildBackendReadinessMatrixArtifact(matrix, "2026-05-30T17:00:00.000Z");

    expect(artifact).toMatchObject({
      name: "backend-readiness-matrix",
      createdAt: "2026-05-30T17:00:00.000Z",
      passed: true,
      summary: {
        backendReadinessMatrixPassed: true,
        backendReadinessDeployBackendId: "compiled-browser-webllm",
        backendReadinessProductionCandidateCount: 1,
        backendReadinessDeployReadyCount: 1,
        backendReadinessResearchBackendCount: 1,
        backendReadinessKernelLabBackendId: "unlocked-browser-transformer",
        backendReadinessFallbackBackendCount: 1,
        backendReadinessFallbackBackendId: "wasm-small-core",
        backendReadinessFallbackDeployReadyCount: 0,
        backendReadinessRoleBoundaryPassed: true,
        backendReadinessModelRegistryAligned: true,
        backendReadinessModelRegistryModelCount: 3,
        backendReadinessPublicModelOptionCount: 2,
        backendReadinessPublicDeployOptionCount: 1,
        backendReadinessPublicKernelLabOptionCount: 1,
        backendReadinessCompiledHostedProfilePassed: true,
        backendReadinessCompiledDeployReady: true,
      },
    });
  });

  it("writes latest and timestamped backend readiness artifacts", async () => {
    const artifactDir = await mkdtemp(join(tmpdir(), "backend-readiness-artifacts-"));
    const matrix = evaluateBackendReadinessMatrix({
      hostedProfile: evaluateHostedDeploymentProfile(completeHostedEnv),
    });

    const written = await writeBackendReadinessMatrixArtifact(matrix, {
      artifactDir,
      createdAt: "2026-05-30T17:00:00.000Z",
    });

    expect(written.latestPath).toBe(join(artifactDir, "backend-readiness-matrix-latest.json"));
    expect(written.resultPath).toBe(join(artifactDir, "backend-readiness-matrix", "2026-05-30T17-00-00-000Z.json"));

    const latest = JSON.parse(await readFile(written.latestPath, "utf8")) as ReturnType<typeof buildBackendReadinessMatrixArtifact>;

    expect(latest.passed).toBe(true);
    expect(latest.summary.backendReadinessDeployBackendId).toBe("compiled-browser-webllm");
  });
});
