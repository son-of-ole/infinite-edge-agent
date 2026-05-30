import { describe, expect, it } from "vitest";
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
});
