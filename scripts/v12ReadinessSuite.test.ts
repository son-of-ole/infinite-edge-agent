import { mkdtemp, readFile, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import {
  buildV12ReadinessSuiteArtifact,
  evaluateV12ReadinessSuite,
  runV12ReadinessSuite,
} from "./v12ReadinessSuite";
import { evaluateHostedBenchmarkProof } from "./hostedBenchmarkProof";

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

describe("v12 readiness suite", () => {
  it("evaluates all final-state readiness proofs as one suite", () => {
    const suite = evaluateV12ReadinessSuite({ env: completeHostedEnv });

    expect(suite).toMatchObject({
      passed: true,
      blockers: [],
      deployBackendId: "compiled-browser-webllm",
      kernelLabBackendId: "unlocked-browser-transformer",
      childArtifactCount: 4,
      totalArtifactCount: 5,
      hostedProfilePassed: true,
      backendReadinessPassed: true,
      sharedRuntimePassed: true,
      v12ReadinessBundlePassed: true,
      hostedBenchmarkProofRequired: false,
      hostedBenchmarkProofPassed: null,
    });
  });

  it("fails honestly when hosted benchmark proof is required but no artifact is provided", () => {
    const suite = evaluateV12ReadinessSuite({
      env: {
        ...completeHostedEnv,
        RELEASE_REQUIRE_HOSTED_BENCHMARK_PROOF: "true",
      },
    });

    expect(suite.passed).toBe(false);
    expect(suite.hostedBenchmarkProofRequired).toBe(true);
    expect(suite.hostedBenchmarkProofPassed).toBe(false);
    expect(suite.blockers).toContain("hosted_benchmark_proof: required but no HOSTED_BENCHMARK_ARTIFACT_PATH or report was provided.");
  });

  it("passes required hosted benchmark proof into backend-specific readiness", () => {
    const suite = evaluateV12ReadinessSuite({
      env: {
        ...completeHostedEnv,
        RELEASE_REQUIRE_HOSTED_BENCHMARK_PROOF: "true",
      },
      hostedBenchmarkProof: evaluateHostedBenchmarkProof({
        artifact: makePassingHostedBenchmarkArtifact(),
      }),
    });

    expect(suite.passed).toBe(true);
    expect(suite.hostedBenchmarkProofRequired).toBe(true);
    expect(suite.hostedBenchmarkProofPassed).toBe(true);
    expect(suite.backendReadinessPassed).toBe(true);
    expect(suite.backendMatrix.backends.find((backend) => backend.backendId === "compiled-browser-webllm")).toMatchObject({
      readinessStatus: "deploy_ready",
      proofSource: "hosted_deployment_profile+hosted_benchmark_proof",
      proofRequirements: expect.arrayContaining([
        "hosted_benchmark_artifact_passed",
      ]),
    });
  });

  it("builds release-summary fields for the whole v12 final-state proof set", () => {
    const artifact = buildV12ReadinessSuiteArtifact(
      evaluateV12ReadinessSuite({ env: completeHostedEnv }),
      {
        createdAt: "2026-05-30T20:00:00.000Z",
        childArtifacts: {
          hostedDeploymentProfile: {
            name: "hosted-deployment-profile",
            passed: true,
            latestPath: ".artifacts/evals/hosted-deployment-profile-latest.json",
            resultPath: ".artifacts/evals/hosted-deployment-profile/2026-05-30T20-00-00-000Z.json",
          },
          backendReadinessMatrix: {
            name: "backend-readiness-matrix",
            passed: true,
            latestPath: ".artifacts/evals/backend-readiness-matrix-latest.json",
            resultPath: ".artifacts/evals/backend-readiness-matrix/2026-05-30T20-00-00-000Z.json",
          },
          sharedRuntimeReadiness: {
            name: "shared-runtime-readiness",
            passed: true,
            latestPath: ".artifacts/evals/shared-runtime-readiness-latest.json",
            resultPath: ".artifacts/evals/shared-runtime-readiness/2026-05-30T20-00-00-000Z.json",
          },
          v12ReadinessBundle: {
            name: "v12-readiness-bundle",
            passed: true,
            latestPath: ".artifacts/evals/v12-readiness-bundle-latest.json",
            resultPath: ".artifacts/evals/v12-readiness-bundle/2026-05-30T20-00-00-000Z.json",
          },
        },
      },
    );

    expect(artifact).toMatchObject({
      name: "v12-readiness-suite",
      createdAt: "2026-05-30T20:00:00.000Z",
      passed: true,
      summary: {
        v12SuitePassed: true,
        v12SuiteArtifactCount: 5,
        v12SuiteChildArtifactCount: 4,
        v12SuiteDeployBackendId: "compiled-browser-webllm",
        v12SuiteKernelLabBackendId: "unlocked-browser-transformer",
        v12SuiteRequirementCount: 5,
        v12SuitePassedRequirementCount: 5,
        v12SuiteHostedProfilePassed: true,
        v12SuiteBackendReadinessPassed: true,
        v12SuiteSharedRuntimePassed: true,
        v12SuiteReadinessBundlePassed: true,
        v12SuiteHostedBenchmarkProofRequired: false,
        v12SuiteHostedBenchmarkProofPassed: null,
      },
    });
  });

  it("writes latest and timestamped suite plus child artifacts with one timestamp", async () => {
    const artifactDir = await mkdtemp(join(tmpdir(), "v12-readiness-suite-"));

    const result = await runV12ReadinessSuite({
      env: completeHostedEnv,
      artifactDir,
      createdAt: "2026-05-30T20:00:00.000Z",
    });

    expect(result.latestPath).toBe(join(artifactDir, "v12-readiness-suite-latest.json"));
    expect(result.resultPath).toBe(join(artifactDir, "v12-readiness-suite", "2026-05-30T20-00-00-000Z.json"));
    expect(result.childArtifacts.hostedDeploymentProfile.resultPath).toBe(join(artifactDir, "hosted-deployment-profile", "2026-05-30T20-00-00-000Z.json"));
    expect(result.childArtifacts.backendReadinessMatrix.resultPath).toBe(join(artifactDir, "backend-readiness-matrix", "2026-05-30T20-00-00-000Z.json"));
    expect(result.childArtifacts.sharedRuntimeReadiness.resultPath).toBe(join(artifactDir, "shared-runtime-readiness", "2026-05-30T20-00-00-000Z.json"));
    expect(result.childArtifacts.v12ReadinessBundle.resultPath).toBe(join(artifactDir, "v12-readiness-bundle", "2026-05-30T20-00-00-000Z.json"));

    const latest = JSON.parse(await readFile(result.latestPath, "utf8")) as ReturnType<typeof buildV12ReadinessSuiteArtifact>;

    expect(latest.passed).toBe(true);
    expect(latest.summary.v12SuiteArtifactCount).toBe(5);
    expect(latest.suite.childArtifacts.v12ReadinessBundle.latestPath).toBe(join(artifactDir, "v12-readiness-bundle-latest.json"));
  });

  it("writes hosted benchmark proof as part of the suite when a saved browser artifact path is provided", async () => {
    const artifactDir = await mkdtemp(join(tmpdir(), "v12-readiness-suite-hosted-proof-"));
    const hostedBenchmarkPath = join(artifactDir, "source-browser-runtime-bench-latest.json");
    await writeFile(hostedBenchmarkPath, `${JSON.stringify(makePassingHostedBenchmarkArtifact(), null, 2)}\n`);

    const result = await runV12ReadinessSuite({
      env: {
        ...completeHostedEnv,
        HOSTED_BENCHMARK_ARTIFACT_PATH: hostedBenchmarkPath,
      },
      artifactDir,
      createdAt: "2026-05-30T20:30:00.000Z",
    });

    expect(result.suite.passed).toBe(true);
    expect(result.suite.childArtifactCount).toBe(5);
    expect(result.suite.totalArtifactCount).toBe(6);
    expect(result.suite.hostedBenchmarkProofPassed).toBe(true);
    expect(result.childArtifacts.hostedBenchmarkProof?.resultPath).toBe(join(artifactDir, "hosted-benchmark-proof", "2026-05-30T20-30-00-000Z.json"));

    const latest = JSON.parse(await readFile(result.latestPath, "utf8")) as ReturnType<typeof buildV12ReadinessSuiteArtifact>;

    expect(latest.summary).toMatchObject({
      v12SuiteArtifactCount: 6,
      v12SuiteChildArtifactCount: 5,
      v12SuiteHostedBenchmarkProofRequired: false,
      v12SuiteHostedBenchmarkProofPassed: true,
    });
    expect(latest.suite.childArtifacts.hostedBenchmarkProof?.latestPath).toBe(join(artifactDir, "hosted-benchmark-proof-latest.json"));
  });
});
