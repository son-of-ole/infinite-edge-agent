import { mkdtemp, readFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import Fastify from "fastify";
import { describe, expect, it } from "vitest";
import {
  JsonlBenchmarkTelemetryStore,
  SqlBenchmarkTelemetryStore,
  createBenchmarkTelemetryStore,
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

  it("persists benchmark runs through a durable SQL client abstraction", async () => {
    const client = new CapturingSqlClient();
    const store = new SqlBenchmarkTelemetryStore({ client });

    const saved = await store.save(makePayload());
    const listed = await store.list({ limit: 10 });
    const summary = await store.summary({ limit: 10 });

    expect(saved).toMatchObject({
      schemaVersion: 1,
      runId: "bench_test",
      backendId: "compiled-browser-webllm"
    });
    expect(listed).toHaveLength(1);
    expect(listed[0]).toMatchObject({
      runId: "bench_test",
      summary: { tokensPerSecond: 8.25 }
    });
    expect(summary).toMatchObject({
      count: 1,
      productionDeployReadyCount: 1
    });
    expect(client.sql).toEqual(expect.arrayContaining([
      expect.stringContaining("create table if not exists benchmark_runs"),
      expect.stringContaining("insert into benchmark_runs"),
      expect.stringContaining("select * from benchmark_runs")
    ]));
    expect(JSON.stringify(client.rows)).not.toContain("private benchmark prompt");
    expect(JSON.stringify(client.rows)).not.toContain("Helena");
  });

  it("selects SQL storage only when explicitly configured with a database url", () => {
    expect(createBenchmarkTelemetryStore({
      storage: "jsonl",
      dir: ".data/benchmark-runs"
    })).toBeInstanceOf(JsonlBenchmarkTelemetryStore);
    expect(() => createBenchmarkTelemetryStore({
      storage: "postgres",
      dir: ".data/benchmark-runs"
    })).toThrow(/requires BENCHMARK_TELEMETRY_DATABASE_URL or DATABASE_URL/);
  });

  it("rate limits telemetry submissions by client identity", async () => {
    const dir = await mkdtemp(join(tmpdir(), "edge-ai-benchmark-telemetry-rate-limit-"));
    const app = Fastify();
    registerBenchmarkTelemetryRoutes(app, {
      enabled: true,
      prefix: "/api/benchmark-runs",
      dir,
      submitRateLimit: {
        max: 1,
        windowMs: 60_000
      }
    });

    const first = await app.inject({
      method: "POST",
      url: "/api/benchmark-runs",
      headers: { "x-forwarded-for": "203.0.113.10" },
      payload: makePayload()
    });
    expect(first.statusCode).toBe(202);

    const second = await app.inject({
      method: "POST",
      url: "/api/benchmark-runs",
      headers: { "x-forwarded-for": "203.0.113.10" },
      payload: { ...makePayload(), runId: "bench_test_second" }
    });
    expect(second.statusCode).toBe(429);
    expect(second.json()).toMatchObject({
      errorCode: "BENCHMARK_TELEMETRY_RATE_LIMITED"
    });

    await app.close();
  });

  it("requires a submit token when public submission token gating is configured", async () => {
    const dir = await mkdtemp(join(tmpdir(), "edge-ai-benchmark-telemetry-submit-token-"));
    const app = Fastify();
    registerBenchmarkTelemetryRoutes(app, {
      enabled: true,
      prefix: "/api/benchmark-runs",
      dir,
      submitToken: "submit-secret"
    });

    const unauthorized = await app.inject({
      method: "POST",
      url: "/api/benchmark-runs",
      payload: makePayload()
    });
    expect(unauthorized.statusCode).toBe(401);
    expect(unauthorized.json()).toMatchObject({
      errorCode: "BENCHMARK_TELEMETRY_SUBMIT_UNAUTHORIZED"
    });

    const authorized = await app.inject({
      method: "POST",
      url: "/api/benchmark-runs",
      headers: { authorization: "Bearer submit-secret" },
      payload: makePayload()
    });
    expect(authorized.statusCode).toBe(202);

    await app.close();
  });

  it("requires an admin token for list, summary, dashboard, and CSV export when configured", async () => {
    const dir = await mkdtemp(join(tmpdir(), "edge-ai-benchmark-telemetry-admin-token-"));
    const app = Fastify();
    registerBenchmarkTelemetryRoutes(app, {
      enabled: true,
      prefix: "/api/benchmark-runs",
      dir,
      adminToken: "admin-secret"
    });
    await app.inject({
      method: "POST",
      url: "/api/benchmark-runs",
      payload: makePayload()
    });

    for (const url of [
      "/api/benchmark-runs",
      "/api/benchmark-runs/summary",
      "/api/benchmark-runs/dashboard",
      "/api/benchmark-runs/export.csv"
    ]) {
      const unauthorized = await app.inject({ method: "GET", url });
      expect(unauthorized.statusCode).toBe(401);
      expect(unauthorized.json()).toMatchObject({
        errorCode: "BENCHMARK_TELEMETRY_ADMIN_UNAUTHORIZED"
      });

      const authorized = await app.inject({
        method: "GET",
        url,
        headers: { authorization: "Bearer admin-secret" }
      });
      expect(authorized.statusCode).toBe(200);
    }

    await app.close();
  });
});

class CapturingSqlClient {
  readonly sql: string[] = [];
  readonly rows: Array<Record<string, unknown>> = [];

  async query(sql: string, values: readonly unknown[] = []): Promise<{ rows: Array<Record<string, unknown>> }> {
    this.sql.push(sql);
    if (sql.includes("insert into benchmark_runs")) {
      const row = {
        id: values[0],
        received_at: values[1],
        created_at: values[2],
        app_version: values[3],
        git_sha: values[4],
        deploy_url: values[5],
        benchmark_profile: values[6],
        backend_id: values[7],
        model_id: values[8],
        os: values[9],
        browser_name: values[10],
        browser_version: values[11],
        user_agent_hash: values[12],
        mobile: values[13],
        hardware_concurrency: values[14],
        device_memory_gb: values[15],
        screen_width: values[16],
        screen_height: values[17],
        webgpu_available: values[18],
        init_load_ms: values[19],
        time_to_first_token_ms: values[20],
        tokens_per_second: values[21],
        memory_grounding_passed: values[22],
        expected_exact_passed: values[23],
        compiled_backend_ready_passed: values[24],
        production_deploy_ready_passed: values[25],
        artifact_json: values[26],
        artifact_bytes: values[27],
        schema_version: 1
      };
      this.rows.splice(0, this.rows.length, row);
      return { rows: [row] };
    }
    if (sql.includes("select * from benchmark_runs")) {
      return { rows: [...this.rows] };
    }
    return { rows: [] };
  }
}

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
