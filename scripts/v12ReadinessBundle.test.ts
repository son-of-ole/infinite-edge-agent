import { mkdtemp, readFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import {
  buildV12ReadinessBundleArtifact,
  evaluateV12ReadinessBundle,
  writeV12ReadinessBundleArtifact,
} from "./v12ReadinessBundle";
import { evaluateHostedDeploymentProfile } from "./hostedDeploymentProfile";
import { evaluateBackendReadinessMatrix } from "./backendReadinessMatrix";
import { evaluateSharedRuntimeReadiness } from "./sharedRuntimeReadiness";

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

function makePassingInputs() {
  const hostedProfile = evaluateHostedDeploymentProfile(completeHostedEnv);
  const backendMatrix = evaluateBackendReadinessMatrix({ hostedProfile });
  const sharedRuntime = evaluateSharedRuntimeReadiness({ backendMatrix });
  return { hostedProfile, backendMatrix, sharedRuntime };
}

describe("evaluateV12ReadinessBundle", () => {
  it("passes only when hosted profile, backend matrix, and shared runtime proofs all pass", () => {
    const bundle = evaluateV12ReadinessBundle(makePassingInputs());

    expect(bundle.passed).toBe(true);
    expect(bundle.blockers).toEqual([]);
    expect(bundle.deployBackendId).toBe("compiled-browser-webllm");
    expect(bundle.kernelLabBackendId).toBe("unlocked-browser-transformer");
    expect(bundle.fallbackBackendId).toBe("wasm-small-core");
    expect(bundle.backendRoleBoundaryPassed).toBe(true);
    expect(bundle.requirements).toEqual(expect.arrayContaining([
      expect.objectContaining({ id: "backend_broker", passed: true }),
      expect.objectContaining({ id: "compiled_production_backend", passed: true }),
      expect.objectContaining({ id: "custom_webgpu_kernel_lab", passed: true }),
      expect.objectContaining({ id: "shared_memory_context_runtime", passed: true }),
      expect.objectContaining({ id: "backend_specific_readiness", passed: true }),
    ]));
  });

  it("fails with requirement-level blockers when any required proof is missing", () => {
    const hostedProfile = evaluateHostedDeploymentProfile({
      ...completeHostedEnv,
      VITE_LLM_BACKEND: "unlocked-browser-transformer",
    });
    const backendMatrix = evaluateBackendReadinessMatrix({ hostedProfile });
    const sharedRuntime = evaluateSharedRuntimeReadiness({ backendMatrix });
    const bundle = evaluateV12ReadinessBundle({ hostedProfile, backendMatrix, sharedRuntime });

    expect(bundle.passed).toBe(false);
    expect(bundle.blockers).toEqual(expect.arrayContaining([
      "compiled_production_backend: Hosted deployment profile did not pass for compiled-browser-webllm.",
      "backend_specific_readiness: Backend readiness matrix did not pass.",
      "shared_memory_context_runtime: Shared runtime readiness did not pass.",
    ]));
  });

  it("builds a release-gate friendly v12 final-state readiness artifact", () => {
    const artifact = buildV12ReadinessBundleArtifact(
      evaluateV12ReadinessBundle(makePassingInputs()),
      "2026-05-30T19:00:00.000Z",
    );

    expect(artifact).toMatchObject({
      name: "v12-readiness-bundle",
      createdAt: "2026-05-30T19:00:00.000Z",
      passed: true,
      summary: {
        v12ReadinessPassed: true,
        v12DeployBackendId: "compiled-browser-webllm",
        v12KernelLabBackendId: "unlocked-browser-transformer",
        v12FallbackBackendId: "wasm-small-core",
        v12BackendRoleBoundaryPassed: true,
        v12RequirementCount: 5,
        v12PassedRequirementCount: 5,
        v12HostedProfilePassed: true,
        v12BackendReadinessPassed: true,
        v12SharedRuntimePassed: true,
      },
    });
  });

  it("writes latest and timestamped v12 readiness artifacts", async () => {
    const artifactDir = await mkdtemp(join(tmpdir(), "v12-readiness-bundle-"));

    const written = await writeV12ReadinessBundleArtifact(
      evaluateV12ReadinessBundle(makePassingInputs()),
      {
        artifactDir,
        createdAt: "2026-05-30T19:00:00.000Z",
      },
    );

    expect(written.latestPath).toBe(join(artifactDir, "v12-readiness-bundle-latest.json"));
    expect(written.resultPath).toBe(join(artifactDir, "v12-readiness-bundle", "2026-05-30T19-00-00-000Z.json"));

    const latest = JSON.parse(await readFile(written.latestPath, "utf8")) as ReturnType<typeof buildV12ReadinessBundleArtifact>;

    expect(latest.passed).toBe(true);
    expect(latest.summary.v12DeployBackendId).toBe("compiled-browser-webllm");
    expect(latest.summary.v12FallbackBackendId).toBe("wasm-small-core");
  });
});
