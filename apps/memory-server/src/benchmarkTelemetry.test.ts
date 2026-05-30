import { mkdtemp, readFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import Fastify from "fastify";
import { describe, expect, it } from "vitest";
import {
  JsonlBenchmarkTelemetryStore,
  registerBenchmarkTelemetryRoutes,
  type BenchmarkTelemetryPayload
} from "./benchmarkTelemetry";

describe("benchmark telemetry server", () => {
  it("persists sanitized benchmark runs without raw prompt, response, expected text, or user agent", async () => {
    const dir = await mkdtemp(join(tmpdir(), "edge-ai-benchmark-telemetry-"));
    const store = new JsonlBenchmarkTelemetryStore({ dir, maxArtifactBytes: 64_000 });

    const record = await store.save(makePayload());
    const stored = await readFile(join(dir, "runs.jsonl"), "utf8");

    expect(record).toMatchObject({
      schemaVersion: 1,
      runId: "bench_test",
      artifactBytes: expect.any(Number)
    });
    expect(stored).not.toContain("private benchmark prompt");
    expect(stored).not.toContain("Helena");
    expect(stored).not.toContain("raw-token");
    expect(stored).not.toContain("Mozilla/5.0");
    expect(stored).toContain("[redacted]");
    expect(stored).toContain("\"matched\":true");

    await expect(store.summary()).resolves.toMatchObject({
      count: 1,
      meanTokensPerSecond: 8.25,
      backendCounts: { "compiled-browser-webllm": 1 },
      browserCounts: { Chrome: 1 },
      osCounts: { macOS: 1 },
      webgpuAvailableCount: 1,
      productionDeployReadyCount: 1
    });
  });

  it("exposes opt-in POST, list, and summary routes when enabled", async () => {
    const dir = await mkdtemp(join(tmpdir(), "edge-ai-benchmark-telemetry-route-"));
    const app = Fastify();
    registerBenchmarkTelemetryRoutes(app, {
      enabled: true,
      prefix: "/api/benchmark-runs",
      dir,
      maxArtifactBytes: 64_000
    });

    const post = await app.inject({
      method: "POST",
      url: "/api/benchmark-runs",
      payload: makePayload()
    });
    expect(post.statusCode).toBe(202);
    expect(post.json()).toMatchObject({
      ok: true,
      runId: "bench_test",
      artifactBytes: expect.any(Number)
    });

    const list = await app.inject({ method: "GET", url: "/api/benchmark-runs?limit=5" });
    expect(list.statusCode).toBe(200);
    expect(list.json()).toMatchObject({
      runs: [expect.objectContaining({ runId: "bench_test" })]
    });

    const summary = await app.inject({ method: "GET", url: "/api/benchmark-runs/summary" });
    expect(summary.statusCode).toBe(200);
    expect(summary.json()).toMatchObject({
      count: 1,
      productionDeployReadyCount: 1
    });

    await app.close();
  });

  it("exports dashboard HTML and CSV rows for hosted benchmark review", async () => {
    const dir = await mkdtemp(join(tmpdir(), "edge-ai-benchmark-telemetry-export-"));
    const app = Fastify();
    registerBenchmarkTelemetryRoutes(app, {
      enabled: true,
      prefix: "/api/benchmark-runs",
      dir,
      maxArtifactBytes: 64_000
    });

    await app.inject({
      method: "POST",
      url: "/api/benchmark-runs",
      payload: makePayload()
    });

    const dashboard = await app.inject({ method: "GET", url: "/api/benchmark-runs/dashboard" });
    expect(dashboard.statusCode).toBe(200);
    expect(dashboard.headers["content-type"]).toContain("text/html");
    expect(dashboard.body).toContain("Infinite Edge Agent Benchmark Runs");
    expect(dashboard.body).toContain("compiled-browser-webllm");
    expect(dashboard.body).toContain("8.25");
    expect(dashboard.body).not.toContain("private benchmark prompt");
    expect(dashboard.body).not.toContain("Helena");

    const csv = await app.inject({ method: "GET", url: "/api/benchmark-runs/export.csv" });
    expect(csv.statusCode).toBe(200);
    expect(csv.headers["content-type"]).toContain("text/csv");
    expect(csv.body).toContain("run_id,received_at,created_at,backend_id,model_id");
    expect(csv.body).toContain("bench_test");
    expect(csv.body).toContain("compiled-browser-webllm");
    expect(csv.body).toContain("8.25");
    expect(csv.body).not.toContain("private benchmark prompt");
    expect(csv.body).not.toContain("Helena");

    await app.close();
  });

  it("does not register public telemetry routes unless explicitly enabled", async () => {
    const app = Fastify();
    registerBenchmarkTelemetryRoutes(app, {
      enabled: false,
      prefix: "/api/benchmark-runs",
      dir: ".data/unused-benchmark-runs"
    });

    const response = await app.inject({
      method: "POST",
      url: "/api/benchmark-runs",
      payload: makePayload()
    });
    expect(response.statusCode).toBe(404);
    await app.close();
  });
});

function makePayload(): BenchmarkTelemetryPayload {
  return {
    runId: "bench_test",
    createdAt: "2026-05-30T00:00:00.000Z",
    appVersion: "0.1.0",
    gitSha: "abc123",
    deployUrl: "https://agent.example.com",
    benchmarkProfile: "full",
    backendId: "compiled-browser-webllm",
    modelId: "Qwen3-0.6B-q4f16_1-MLC",
    device: {
      os: "macOS",
      browserName: "Chrome",
      browserVersion: "125.0.0.0",
      userAgentHash: "1234abcd",
      mobile: false,
      hardwareConcurrency: 10,
      deviceMemoryGb: 8,
      screen: { width: 1440, height: 900 },
      webgpuAvailable: true
    },
    summary: {
      initLoadMs: 100,
      timeToFirstTokenMs: 20,
      tokensPerSecond: 8.25,
      memoryGroundingPassed: true,
      expectedExactPassed: true,
      productionDeployReadyPassed: true,
      compiledBackendReadyPassed: true
    },
    artifactJson: {
      runs: [{
        prompt: "private benchmark prompt",
        response: "Helena",
        expectedSubstrings: ["Helena"],
        expectedSubstringMatches: ["Helena"],
        expectedExact: ["Helena"],
        expectedExactMatches: [{ expected: "Helena", matched: true }],
        tokenDiagnostics: { generatedTokenTexts: ["raw-token"] },
        runtimeTrace: { userAgent: "Mozilla/5.0" }
      }]
    }
  };
}
