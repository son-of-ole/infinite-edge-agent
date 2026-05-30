import { mkdtemp, readFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import {
  buildHostedDeploymentProfileArtifact,
  evaluateHostedDeploymentProfile,
  writeHostedDeploymentProfileArtifact,
} from "./hostedDeploymentProfile";

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

describe("evaluateHostedDeploymentProfile", () => {
  it("passes only when the hosted profile uses the compiled production backend, durable telemetry, and grounded Chrome proof URL", () => {
    const report = evaluateHostedDeploymentProfile(completeHostedEnv);

    expect(report.passed).toBe(true);
    expect(report.blockers).toEqual([]);
    expect(report.profile).toMatchObject({
      llmBackend: "compiled-browser-webllm",
      compiledWebLlmEnabled: true,
      telemetryEnabled: true,
      telemetryStorage: "postgres",
      telemetryAdminProtected: true,
      telemetryRateLimited: true,
      benchmarkBackend: "compiled-browser-webllm",
      benchmarkExpectedExact: "Helena",
      benchmarkRequiresSubmitTelemetry: true,
      mtpProductionDisabled: true,
    });
  });

  it("fails an unconfigured hosted profile with actionable blockers", () => {
    const report = evaluateHostedDeploymentProfile({});

    expect(report.passed).toBe(false);
    expect(report.blockers).toEqual(expect.arrayContaining([
      "Hosted production requires VITE_LLM_BACKEND=compiled-browser-webllm.",
      "Hosted production requires VITE_COMPILED_WEBLLM_ENABLED=true.",
      "Hosted benchmark telemetry must be enabled on both browser and collector.",
      "Hosted production requires HOSTED_PRODUCTION_BENCHMARK_URL or BROWSER_RUNTIME_BENCH_PREVIEW_URL.",
    ]));
  });

  it("rejects hosted telemetry when storage is local JSONL instead of durable Postgres", () => {
    const report = evaluateHostedDeploymentProfile({
      ...completeHostedEnv,
      BENCHMARK_TELEMETRY_STORAGE: "jsonl",
    });

    expect(report.passed).toBe(false);
    expect(report.blockers).toContain("Hosted telemetry must use BENCHMARK_TELEMETRY_STORAGE=postgres.");
  });

  it("rejects hosted telemetry when dashboard and export routes are not admin protected", () => {
    const report = evaluateHostedDeploymentProfile({
      ...completeHostedEnv,
      BENCHMARK_TELEMETRY_ADMIN_TOKEN: "",
    });

    expect(report.passed).toBe(false);
    expect(report.blockers).toContain("Hosted telemetry requires BENCHMARK_TELEMETRY_ADMIN_TOKEN for list, summary, dashboard, and CSV export routes.");
  });

  it("rejects benchmark URLs that prove the research Kernel Lab instead of the compiled backend", () => {
    const report = evaluateHostedDeploymentProfile({
      ...completeHostedEnv,
      HOSTED_PRODUCTION_BENCHMARK_URL:
        "https://agent.example.com/__bench/browser-runtime?backend=unlocked-browser-transformer&memoryGrounding=montana_capital&expectedExact=Helena&submitTelemetry=true",
    });

    expect(report.passed).toBe(false);
    expect(report.blockers).toContain("Hosted production benchmark URL must set backend=compiled-browser-webllm.");
  });

  it.each([
    "http://agent.example.com/__bench/browser-runtime?backend=compiled-browser-webllm&memoryGrounding=montana_capital&expectedExact=Helena&submitTelemetry=true",
    "https://localhost/__bench/browser-runtime?backend=compiled-browser-webllm&memoryGrounding=montana_capital&expectedExact=Helena&submitTelemetry=true",
    "https://192.168.1.5/__bench/browser-runtime?backend=compiled-browser-webllm&memoryGrounding=montana_capital&expectedExact=Helena&submitTelemetry=true",
  ])("rejects hosted benchmark URLs that are not public HTTPS: %s", (url) => {
    const report = evaluateHostedDeploymentProfile({
      ...completeHostedEnv,
      HOSTED_PRODUCTION_BENCHMARK_URL: url,
    });

    expect(report.passed).toBe(false);
    expect(report.blockers).toContain("Hosted production benchmark URL must use a public HTTPS origin.");
  });

  it("rejects generated benchmark URLs when VITE_DEPLOY_URL is not public HTTPS", () => {
    const report = evaluateHostedDeploymentProfile({
      ...completeHostedEnv,
      HOSTED_PRODUCTION_BENCHMARK_URL: "",
      VITE_DEPLOY_URL: "http://localhost:5173",
    });

    expect(report.passed).toBe(false);
    expect(report.blockers).toContain("Hosted production benchmark URL must use a public HTTPS origin.");
  });

  it("rejects benchmark URLs that do not prove grounded exact Montana retrieval with telemetry submission", () => {
    const report = evaluateHostedDeploymentProfile({
      ...completeHostedEnv,
      HOSTED_PRODUCTION_BENCHMARK_URL:
        "https://agent.example.com/__bench/browser-runtime?backend=compiled-browser-webllm&expected=Helena",
    });

    expect(report.passed).toBe(false);
    expect(report.blockers).toEqual(expect.arrayContaining([
      "Hosted production benchmark URL must require memoryGrounding=montana_capital or memoryGroundingProfile=qa_corpus_v1.",
      "Hosted production benchmark URL must set expectedExact=Helena.",
      "Hosted production benchmark URL must opt in with submitTelemetry=true or benchmarkTelemetry=true.",
    ]));
  });

  it("emits a generated production benchmark URL when the deploy URL is known", () => {
    const report = evaluateHostedDeploymentProfile({
      ...completeHostedEnv,
      HOSTED_PRODUCTION_BENCHMARK_URL: "",
      VITE_DEPLOY_URL: "https://agent.example.com/",
    });

    expect(report.passed).toBe(true);
    expect(report.expectedBenchmarkUrl).toBe(
      "https://agent.example.com/__bench/browser-runtime?backend=compiled-browser-webllm&modelId=Qwen3-0.6B-q4f16_1-MLC&memoryGrounding=montana_capital&expectedExact=Helena&submitTelemetry=true&qwenThinkingMode=disabled",
    );
  });

  it("builds a release-gate friendly artifact with backend-specific hosted readiness summary fields", () => {
    const report = evaluateHostedDeploymentProfile(completeHostedEnv);
    const artifact = buildHostedDeploymentProfileArtifact(report, "2026-05-30T16:00:00.000Z");

    expect(artifact).toMatchObject({
      name: "hosted-deployment-profile",
      createdAt: "2026-05-30T16:00:00.000Z",
      passed: true,
      summary: {
        hostedProfilePassed: true,
        hostedProfileBackend: "compiled-browser-webllm",
        hostedProfileDefaultModel: "Qwen3-0.6B-q4f16_1-MLC",
        hostedProfileCompiledWebLlmEnabled: true,
        hostedProfileMtpProductionDisabled: true,
        hostedProfileTelemetryEnabled: true,
        hostedProfileTelemetryStorage: "postgres",
        hostedProfileTelemetryAdminProtected: true,
        hostedProfileTelemetryRateLimited: true,
        hostedProfileBenchmarkBackend: "compiled-browser-webllm",
        hostedProfileBenchmarkMemoryGrounding: "montana_capital",
        hostedProfileBenchmarkExpectedExact: "Helena",
        hostedProfileBenchmarkRequiresSubmitTelemetry: true,
      },
    });
  });

  it("writes latest and timestamped hosted deployment profile artifacts", async () => {
    const artifactDir = await mkdtemp(join(tmpdir(), "hosted-profile-artifacts-"));
    const report = evaluateHostedDeploymentProfile(completeHostedEnv);

    const written = await writeHostedDeploymentProfileArtifact(report, {
      artifactDir,
      createdAt: "2026-05-30T16:00:00.000Z",
    });

    expect(written.latestPath).toBe(join(artifactDir, "hosted-deployment-profile-latest.json"));
    expect(written.resultPath).toBe(join(artifactDir, "hosted-deployment-profile", "2026-05-30T16-00-00-000Z.json"));

    const latest = JSON.parse(await readFile(written.latestPath, "utf8")) as ReturnType<typeof buildHostedDeploymentProfileArtifact>;
    const timestamped = JSON.parse(await readFile(written.resultPath, "utf8")) as ReturnType<typeof buildHostedDeploymentProfileArtifact>;

    expect(latest).toEqual(timestamped);
    expect(latest.passed).toBe(true);
    expect(latest.summary.hostedProfileBackend).toBe("compiled-browser-webllm");
  });
});
