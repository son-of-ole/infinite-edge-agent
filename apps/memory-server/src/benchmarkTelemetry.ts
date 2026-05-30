import { appendFile, mkdir, readFile } from "node:fs/promises";
import { join } from "node:path";
import type { FastifyInstance, FastifyReply, FastifyRequest } from "fastify";
import { z } from "zod";

const MAX_RUNS_LIST_LIMIT = 500;
const REDACTED = "[redacted]";
const REDACTED_KEYS = new Set([
  "prompt",
  "response",
  "expectedSubstrings",
  "expectedSubstringMatches",
  "expectedExact",
  "tokenDiagnostics",
  "messages",
  "content",
  "rawPrompt",
  "rawResponse",
  "userAgent"
]);

const nullableString = z.string().max(2048).nullable();
const nullableNumber = z.number().finite().nullable();
const nullableBoolean = z.boolean().nullable();

export const benchmarkTelemetryPayloadSchema = z.object({
  runId: z.string().min(1).max(160),
  createdAt: z.string().min(1).max(80),
  appVersion: nullableString,
  gitSha: nullableString,
  deployUrl: nullableString,
  benchmarkProfile: nullableString,
  backendId: nullableString,
  modelId: nullableString,
  device: z.object({
    os: z.string().max(80),
    browserName: z.string().max(80),
    browserVersion: nullableString,
    userAgentHash: z.string().max(128),
    mobile: z.boolean(),
    hardwareConcurrency: z.number().int().positive().max(1024).nullable(),
    deviceMemoryGb: z.number().positive().max(4096).nullable(),
    screen: z.object({
      width: z.number().int().positive().max(100_000).nullable(),
      height: z.number().int().positive().max(100_000).nullable()
    }),
    webgpuAvailable: z.boolean()
  }).strict(),
  summary: z.object({
    initLoadMs: nullableNumber,
    timeToFirstTokenMs: nullableNumber,
    tokensPerSecond: nullableNumber,
    memoryGroundingPassed: nullableBoolean,
    expectedExactPassed: nullableBoolean,
    productionDeployReadyPassed: nullableBoolean,
    compiledBackendReadyPassed: nullableBoolean
  }).strict(),
  artifactJson: z.unknown()
}).strict();

export type BenchmarkTelemetryPayload = z.infer<typeof benchmarkTelemetryPayloadSchema>;

export interface StoredBenchmarkTelemetryRun extends BenchmarkTelemetryPayload {
  schemaVersion: 1;
  receivedAt: string;
  artifactBytes: number;
}

export interface BenchmarkTelemetryStore {
  save(payload: BenchmarkTelemetryPayload): Promise<StoredBenchmarkTelemetryRun>;
  list(options?: { limit?: number }): Promise<StoredBenchmarkTelemetryRun[]>;
  summary(options?: { limit?: number }): Promise<BenchmarkTelemetrySummary>;
}

export interface BenchmarkTelemetrySummary {
  count: number;
  latestReceivedAt: string | null;
  meanTokensPerSecond: number | null;
  backendCounts: Record<string, number>;
  browserCounts: Record<string, number>;
  osCounts: Record<string, number>;
  webgpuAvailableCount: number;
  productionDeployReadyCount: number;
}

export interface JsonlBenchmarkTelemetryStoreOptions {
  dir: string;
  maxArtifactBytes?: number;
}

export interface SqlBenchmarkTelemetryClient {
  query(sql: string, values?: readonly unknown[]): Promise<{ rows: Array<Record<string, unknown>> }>;
}

export interface SqlBenchmarkTelemetryStoreOptions {
  client: SqlBenchmarkTelemetryClient;
  maxArtifactBytes?: number;
}

export interface BenchmarkTelemetryStoreFactoryOptions extends JsonlBenchmarkTelemetryStoreOptions {
  storage?: "jsonl" | "postgres";
  databaseUrl?: string;
}

export interface RegisterBenchmarkTelemetryRoutesOptions extends BenchmarkTelemetryStoreFactoryOptions {
  enabled: boolean;
  prefix: string;
  store?: BenchmarkTelemetryStore;
}

export class JsonlBenchmarkTelemetryStore implements BenchmarkTelemetryStore {
  private readonly filePath: string;
  private readonly maxArtifactBytes: number;

  constructor(private readonly options: JsonlBenchmarkTelemetryStoreOptions) {
    this.filePath = join(options.dir, "runs.jsonl");
    this.maxArtifactBytes = options.maxArtifactBytes ?? 1024 * 1024;
  }

  async save(payload: BenchmarkTelemetryPayload): Promise<StoredBenchmarkTelemetryRun> {
    const sanitized = sanitizeBenchmarkTelemetryPayload(payload);
    const artifactBytes = Buffer.byteLength(JSON.stringify(sanitized.artifactJson), "utf8");
    if (artifactBytes > this.maxArtifactBytes) {
      throw new BenchmarkTelemetryError(
        "BENCHMARK_TELEMETRY_ARTIFACT_TOO_LARGE",
        `Benchmark artifact is ${artifactBytes} bytes; max is ${this.maxArtifactBytes}.`,
        413
      );
    }
    const record: StoredBenchmarkTelemetryRun = {
      ...sanitized,
      schemaVersion: 1,
      receivedAt: new Date().toISOString(),
      artifactBytes
    };
    await mkdir(this.options.dir, { recursive: true });
    await appendFile(this.filePath, `${JSON.stringify(record)}\n`, "utf8");
    return record;
  }

  async list(options: { limit?: number } = {}): Promise<StoredBenchmarkTelemetryRun[]> {
    const limit = normalizeListLimit(options.limit);
    let text = "";
    try {
      text = await readFile(this.filePath, "utf8");
    } catch (error) {
      if (isMissingFileError(error)) return [];
      throw error;
    }
    const records = text
      .split("\n")
      .filter(Boolean)
      .map((line) => JSON.parse(line) as StoredBenchmarkTelemetryRun);
    return records.slice(-limit).reverse();
  }

  async summary(options: { limit?: number } = {}): Promise<BenchmarkTelemetrySummary> {
    return summarizeBenchmarkTelemetryRuns(await this.list(options));
  }
}

export class SqlBenchmarkTelemetryStore implements BenchmarkTelemetryStore {
  private readonly maxArtifactBytes: number;
  private schemaReady: Promise<void> | null = null;

  constructor(private readonly options: SqlBenchmarkTelemetryStoreOptions) {
    this.maxArtifactBytes = options.maxArtifactBytes ?? 1024 * 1024;
  }

  async save(payload: BenchmarkTelemetryPayload): Promise<StoredBenchmarkTelemetryRun> {
    await this.ensureSchema();
    const sanitized = sanitizeBenchmarkTelemetryPayload(payload);
    const artifactBytes = Buffer.byteLength(JSON.stringify(sanitized.artifactJson), "utf8");
    if (artifactBytes > this.maxArtifactBytes) {
      throw new BenchmarkTelemetryError(
        "BENCHMARK_TELEMETRY_ARTIFACT_TOO_LARGE",
        `Benchmark artifact is ${artifactBytes} bytes; max is ${this.maxArtifactBytes}.`,
        413
      );
    }
    const receivedAt = new Date().toISOString();
    const result = await this.options.client.query(INSERT_BENCHMARK_RUN_SQL, [
      sanitized.runId,
      receivedAt,
      sanitized.createdAt,
      sanitized.appVersion,
      sanitized.gitSha,
      sanitized.deployUrl,
      sanitized.benchmarkProfile,
      sanitized.backendId,
      sanitized.modelId,
      sanitized.device.os,
      sanitized.device.browserName,
      sanitized.device.browserVersion,
      sanitized.device.userAgentHash,
      sanitized.device.mobile,
      sanitized.device.hardwareConcurrency,
      sanitized.device.deviceMemoryGb,
      sanitized.device.screen.width,
      sanitized.device.screen.height,
      sanitized.device.webgpuAvailable,
      sanitized.summary.initLoadMs,
      sanitized.summary.timeToFirstTokenMs,
      sanitized.summary.tokensPerSecond,
      sanitized.summary.memoryGroundingPassed,
      sanitized.summary.expectedExactPassed,
      sanitized.summary.compiledBackendReadyPassed,
      sanitized.summary.productionDeployReadyPassed,
      sanitized.artifactJson,
      artifactBytes
    ]);
    return rowToStoredBenchmarkTelemetryRun(result.rows[0] ?? {
      id: sanitized.runId,
      received_at: receivedAt,
      created_at: sanitized.createdAt,
      app_version: sanitized.appVersion,
      git_sha: sanitized.gitSha,
      deploy_url: sanitized.deployUrl,
      benchmark_profile: sanitized.benchmarkProfile,
      backend_id: sanitized.backendId,
      model_id: sanitized.modelId,
      os: sanitized.device.os,
      browser_name: sanitized.device.browserName,
      browser_version: sanitized.device.browserVersion,
      user_agent_hash: sanitized.device.userAgentHash,
      mobile: sanitized.device.mobile,
      hardware_concurrency: sanitized.device.hardwareConcurrency,
      device_memory_gb: sanitized.device.deviceMemoryGb,
      screen_width: sanitized.device.screen.width,
      screen_height: sanitized.device.screen.height,
      webgpu_available: sanitized.device.webgpuAvailable,
      init_load_ms: sanitized.summary.initLoadMs,
      time_to_first_token_ms: sanitized.summary.timeToFirstTokenMs,
      tokens_per_second: sanitized.summary.tokensPerSecond,
      memory_grounding_passed: sanitized.summary.memoryGroundingPassed,
      expected_exact_passed: sanitized.summary.expectedExactPassed,
      compiled_backend_ready_passed: sanitized.summary.compiledBackendReadyPassed,
      production_deploy_ready_passed: sanitized.summary.productionDeployReadyPassed,
      artifact_json: sanitized.artifactJson,
      artifact_bytes: artifactBytes,
      schema_version: 1
    });
  }

  async list(options: { limit?: number } = {}): Promise<StoredBenchmarkTelemetryRun[]> {
    await this.ensureSchema();
    const limit = normalizeListLimit(options.limit);
    const result = await this.options.client.query(
      "select * from benchmark_runs order by received_at desc limit $1",
      [limit]
    );
    return result.rows.map(rowToStoredBenchmarkTelemetryRun);
  }

  async summary(options: { limit?: number } = {}): Promise<BenchmarkTelemetrySummary> {
    return summarizeBenchmarkTelemetryRuns(await this.list(options));
  }

  private async ensureSchema(): Promise<void> {
    this.schemaReady ??= this.options.client.query(CREATE_BENCHMARK_RUNS_TABLE_SQL).then(() => undefined);
    await this.schemaReady;
  }
}

export function createBenchmarkTelemetryStore(
  options: BenchmarkTelemetryStoreFactoryOptions
): BenchmarkTelemetryStore {
  if (options.storage === "postgres") {
    const databaseUrl = options.databaseUrl?.trim();
    if (!databaseUrl) {
      throw new Error("Postgres benchmark telemetry requires BENCHMARK_TELEMETRY_DATABASE_URL or DATABASE_URL.");
    }
    return new SqlBenchmarkTelemetryStore({
      client: createPostgresTelemetryClient(databaseUrl),
      ...(options.maxArtifactBytes !== undefined ? { maxArtifactBytes: options.maxArtifactBytes } : {})
    });
  }
  return new JsonlBenchmarkTelemetryStore(options);
}

export function registerBenchmarkTelemetryRoutes(
  app: FastifyInstance,
  options: RegisterBenchmarkTelemetryRoutesOptions
): void {
  if (!options.enabled) return;
  const prefix = normalizePrefix(options.prefix);
  const store = options.store ?? createBenchmarkTelemetryStore(options);

  app.post(prefix, async (request, reply) => {
    const parsed = benchmarkTelemetryPayloadSchema.safeParse(request.body);
    if (!parsed.success) {
      return reply.status(400).send({ error: parsed.error.flatten() });
    }
    try {
      const record = await store.save(parsed.data);
      return reply.status(202).send({
        ok: true,
        runId: record.runId,
        receivedAt: record.receivedAt,
        artifactBytes: record.artifactBytes
      });
    } catch (error) {
      return sendBenchmarkTelemetryError(error, reply);
    }
  });

  app.get(`${prefix}/dashboard`, async (request, reply) => {
    const limit = readQueryLimit(request);
    const runs = await store.list({ ...(limit !== undefined ? { limit } : {}) });
    const summary = summarizeBenchmarkTelemetryRuns(runs);
    return reply
      .type("text/html; charset=utf-8")
      .send(renderBenchmarkTelemetryDashboard(runs, summary));
  });

  app.get(`${prefix}/export.csv`, async (request, reply) => {
    const limit = readQueryLimit(request);
    const runs = await store.list({ ...(limit !== undefined ? { limit } : {}) });
    return reply
      .header("content-disposition", "attachment; filename=\"benchmark-runs.csv\"")
      .type("text/csv; charset=utf-8")
      .send(renderBenchmarkTelemetryCsv(runs));
  });

  app.get(prefix, async (request) => {
    const limit = readQueryLimit(request);
    return { runs: await store.list({ ...(limit !== undefined ? { limit } : {}) }) };
  });

  app.get(`${prefix}/summary`, async (request) => {
    const limit = readQueryLimit(request);
    return await store.summary({ ...(limit !== undefined ? { limit } : {}) });
  });
}

export function sanitizeBenchmarkTelemetryPayload(
  payload: BenchmarkTelemetryPayload
): BenchmarkTelemetryPayload {
  return {
    ...payload,
    artifactJson: sanitizeBenchmarkArtifactJson(payload.artifactJson)
  };
}

export function summarizeBenchmarkTelemetryRuns(
  runs: readonly StoredBenchmarkTelemetryRun[]
): BenchmarkTelemetrySummary {
  const tokenSpeeds = runs
    .map((run) => run.summary.tokensPerSecond)
    .filter((value): value is number => typeof value === "number" && Number.isFinite(value));
  return {
    count: runs.length,
    latestReceivedAt: runs[0]?.receivedAt ?? null,
    meanTokensPerSecond: tokenSpeeds.length > 0
      ? roundMetric(tokenSpeeds.reduce((total, value) => total + value, 0) / tokenSpeeds.length)
      : null,
    backendCounts: countBy(runs.map((run) => run.backendId ?? "unknown")),
    browserCounts: countBy(runs.map((run) => run.device.browserName || "unknown")),
    osCounts: countBy(runs.map((run) => run.device.os || "unknown")),
    webgpuAvailableCount: runs.filter((run) => run.device.webgpuAvailable).length,
    productionDeployReadyCount: runs.filter((run) => run.summary.productionDeployReadyPassed === true).length
  };
}

function sanitizeBenchmarkArtifactJson(value: unknown): unknown {
  if (Array.isArray(value)) return value.map(sanitizeBenchmarkArtifactJson);
  if (!value || typeof value !== "object") return value;
  const sanitized: Record<string, unknown> = {};
  for (const [key, child] of Object.entries(value)) {
    if (REDACTED_KEYS.has(key)) {
      sanitized[key] = key === "tokenDiagnostics" ? undefined : REDACTED;
      continue;
    }
    if (key === "expectedExactMatches" && Array.isArray(child)) {
      sanitized[key] = child.map((match) => ({
        matched: Boolean(isRecord(match) ? match.matched : false)
      }));
      continue;
    }
    sanitized[key] = sanitizeBenchmarkArtifactJson(child);
  }
  return sanitized;
}

const CREATE_BENCHMARK_RUNS_TABLE_SQL = `
create table if not exists benchmark_runs (
  id text primary key,
  received_at timestamptz not null,
  created_at timestamptz not null,
  app_version text,
  git_sha text,
  deploy_url text,
  benchmark_profile text,
  backend_id text,
  model_id text,
  os text not null,
  browser_name text not null,
  browser_version text,
  user_agent_hash text not null,
  mobile boolean not null,
  hardware_concurrency integer,
  device_memory_gb numeric,
  screen_width integer,
  screen_height integer,
  webgpu_available boolean not null,
  init_load_ms numeric,
  time_to_first_token_ms numeric,
  tokens_per_second numeric,
  memory_grounding_passed boolean,
  expected_exact_passed boolean,
  compiled_backend_ready_passed boolean,
  production_deploy_ready_passed boolean,
  artifact_json jsonb not null,
  artifact_bytes integer not null,
  schema_version integer not null default 1
);
create index if not exists benchmark_runs_received_at_idx on benchmark_runs (received_at desc);
create index if not exists benchmark_runs_backend_id_idx on benchmark_runs (backend_id);
`;

const INSERT_BENCHMARK_RUN_SQL = `
insert into benchmark_runs (
  id,
  received_at,
  created_at,
  app_version,
  git_sha,
  deploy_url,
  benchmark_profile,
  backend_id,
  model_id,
  os,
  browser_name,
  browser_version,
  user_agent_hash,
  mobile,
  hardware_concurrency,
  device_memory_gb,
  screen_width,
  screen_height,
  webgpu_available,
  init_load_ms,
  time_to_first_token_ms,
  tokens_per_second,
  memory_grounding_passed,
  expected_exact_passed,
  compiled_backend_ready_passed,
  production_deploy_ready_passed,
  artifact_json,
  artifact_bytes,
  schema_version
) values (
  $1, $2, $3, $4, $5, $6, $7, $8, $9, $10,
  $11, $12, $13, $14, $15, $16, $17, $18, $19, $20,
  $21, $22, $23, $24, $25, $26, $27, $28, 1
) on conflict (id) do update set
  received_at = excluded.received_at,
  app_version = excluded.app_version,
  git_sha = excluded.git_sha,
  deploy_url = excluded.deploy_url,
  benchmark_profile = excluded.benchmark_profile,
  backend_id = excluded.backend_id,
  model_id = excluded.model_id,
  os = excluded.os,
  browser_name = excluded.browser_name,
  browser_version = excluded.browser_version,
  user_agent_hash = excluded.user_agent_hash,
  mobile = excluded.mobile,
  hardware_concurrency = excluded.hardware_concurrency,
  device_memory_gb = excluded.device_memory_gb,
  screen_width = excluded.screen_width,
  screen_height = excluded.screen_height,
  webgpu_available = excluded.webgpu_available,
  init_load_ms = excluded.init_load_ms,
  time_to_first_token_ms = excluded.time_to_first_token_ms,
  tokens_per_second = excluded.tokens_per_second,
  memory_grounding_passed = excluded.memory_grounding_passed,
  expected_exact_passed = excluded.expected_exact_passed,
  compiled_backend_ready_passed = excluded.compiled_backend_ready_passed,
  production_deploy_ready_passed = excluded.production_deploy_ready_passed,
  artifact_json = excluded.artifact_json,
  artifact_bytes = excluded.artifact_bytes,
  schema_version = excluded.schema_version
returning *;
`;

function rowToStoredBenchmarkTelemetryRun(row: Record<string, unknown>): StoredBenchmarkTelemetryRun {
  return {
    runId: readRowString(row.id, "unknown"),
    createdAt: readDateString(row.created_at),
    appVersion: readNullableString(row.app_version),
    gitSha: readNullableString(row.git_sha),
    deployUrl: readNullableString(row.deploy_url),
    benchmarkProfile: readNullableString(row.benchmark_profile),
    backendId: readNullableString(row.backend_id),
    modelId: readNullableString(row.model_id),
    device: {
      os: readRowString(row.os, "Unknown"),
      browserName: readRowString(row.browser_name, "Unknown"),
      browserVersion: readNullableString(row.browser_version),
      userAgentHash: readRowString(row.user_agent_hash, ""),
      mobile: readBoolean(row.mobile),
      hardwareConcurrency: readNullableNumber(row.hardware_concurrency),
      deviceMemoryGb: readNullableNumber(row.device_memory_gb),
      screen: {
        width: readNullableNumber(row.screen_width),
        height: readNullableNumber(row.screen_height)
      },
      webgpuAvailable: readBoolean(row.webgpu_available)
    },
    summary: {
      initLoadMs: readNullableNumber(row.init_load_ms),
      timeToFirstTokenMs: readNullableNumber(row.time_to_first_token_ms),
      tokensPerSecond: readNullableNumber(row.tokens_per_second),
      memoryGroundingPassed: readNullableBoolean(row.memory_grounding_passed),
      expectedExactPassed: readNullableBoolean(row.expected_exact_passed),
      compiledBackendReadyPassed: readNullableBoolean(row.compiled_backend_ready_passed),
      productionDeployReadyPassed: readNullableBoolean(row.production_deploy_ready_passed)
    },
    artifactJson: readArtifactJson(row.artifact_json),
    schemaVersion: 1,
    receivedAt: readDateString(row.received_at),
    artifactBytes: Math.trunc(readNullableNumber(row.artifact_bytes) ?? 0)
  };
}

function renderBenchmarkTelemetryDashboard(
  runs: readonly StoredBenchmarkTelemetryRun[],
  summary: BenchmarkTelemetrySummary
): string {
  const rows = runs.map((run) => `
      <tr>
        <td>${escapeHtml(run.receivedAt)}</td>
        <td>${escapeHtml(run.backendId ?? "unknown")}</td>
        <td>${escapeHtml(run.modelId ?? "unknown")}</td>
        <td>${escapeHtml(run.device.os)}</td>
        <td>${escapeHtml(run.device.browserName)}</td>
        <td>${formatMetric(run.summary.tokensPerSecond)}</td>
        <td>${formatBoolean(run.summary.memoryGroundingPassed)}</td>
        <td>${formatBoolean(run.summary.expectedExactPassed)}</td>
        <td>${formatBoolean(run.summary.productionDeployReadyPassed)}</td>
      </tr>`).join("");
  return `<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Infinite Edge Agent Benchmark Runs</title>
    <style>
      body { font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 2rem; color: #111827; background: #f8fafc; }
      main { max-width: 1180px; margin: 0 auto; }
      h1 { font-size: 1.75rem; margin-bottom: 0.25rem; }
      .summary { display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 0.75rem; margin: 1.5rem 0; }
      .metric { border: 1px solid #d1d5db; border-radius: 8px; padding: 0.875rem; background: white; }
      .metric strong { display: block; font-size: 1.35rem; margin-top: 0.25rem; }
      table { width: 100%; border-collapse: collapse; background: white; border: 1px solid #d1d5db; }
      th, td { text-align: left; border-bottom: 1px solid #e5e7eb; padding: 0.65rem; font-size: 0.9rem; }
      th { background: #f3f4f6; }
      a { color: #075985; }
    </style>
  </head>
  <body>
    <main>
      <h1>Infinite Edge Agent Benchmark Runs</h1>
      <p>Sanitized hosted browser benchmark telemetry. Raw prompts, responses, expected answers, token diagnostics, and user agents are not rendered here.</p>
      <p><a href="./export.csv">Download CSV</a></p>
      <section class="summary">
        <div class="metric">Runs<strong>${summary.count}</strong></div>
        <div class="metric">Mean tokens/sec<strong>${formatMetric(summary.meanTokensPerSecond)}</strong></div>
        <div class="metric">WebGPU available<strong>${summary.webgpuAvailableCount}</strong></div>
        <div class="metric">Deploy-ready runs<strong>${summary.productionDeployReadyCount}</strong></div>
      </section>
      <table>
        <thead>
          <tr>
            <th>Received</th>
            <th>Backend</th>
            <th>Model</th>
            <th>OS</th>
            <th>Browser</th>
            <th>Tok/s</th>
            <th>Memory</th>
            <th>Exact</th>
            <th>Deploy</th>
          </tr>
        </thead>
        <tbody>${rows || "<tr><td colspan=\"9\">No benchmark runs saved yet.</td></tr>"}</tbody>
      </table>
    </main>
  </body>
</html>`;
}

function renderBenchmarkTelemetryCsv(runs: readonly StoredBenchmarkTelemetryRun[]): string {
  const header = [
    "run_id",
    "received_at",
    "created_at",
    "backend_id",
    "model_id",
    "app_version",
    "git_sha",
    "deploy_url",
    "os",
    "browser_name",
    "browser_version",
    "mobile",
    "hardware_concurrency",
    "device_memory_gb",
    "webgpu_available",
    "tokens_per_second",
    "time_to_first_token_ms",
    "init_load_ms",
    "memory_grounding_passed",
    "expected_exact_passed",
    "compiled_backend_ready_passed",
    "production_deploy_ready_passed"
  ];
  const rows = runs.map((run) => [
    run.runId,
    run.receivedAt,
    run.createdAt,
    run.backendId,
    run.modelId,
    run.appVersion,
    run.gitSha,
    run.deployUrl,
    run.device.os,
    run.device.browserName,
    run.device.browserVersion,
    run.device.mobile,
    run.device.hardwareConcurrency,
    run.device.deviceMemoryGb,
    run.device.webgpuAvailable,
    run.summary.tokensPerSecond,
    run.summary.timeToFirstTokenMs,
    run.summary.initLoadMs,
    run.summary.memoryGroundingPassed,
    run.summary.expectedExactPassed,
    run.summary.compiledBackendReadyPassed,
    run.summary.productionDeployReadyPassed
  ].map(csvCell).join(","));
  return `${header.join(",")}\n${rows.join("\n")}${rows.length > 0 ? "\n" : ""}`;
}

function sendBenchmarkTelemetryError(error: unknown, reply: FastifyReply): FastifyReply {
  if (error instanceof BenchmarkTelemetryError) {
    return reply.status(error.statusCode).send({
      errorCode: error.code,
      message: error.message
    });
  }
  return reply.status(500).send({
    errorCode: "BENCHMARK_TELEMETRY_WRITE_FAILED",
    message: error instanceof Error ? error.message : String(error)
  });
}

function readQueryLimit(request: FastifyRequest): number | undefined {
  const query = request.query as { limit?: string } | undefined;
  if (!query?.limit) return undefined;
  const parsed = Number(query.limit);
  if (!Number.isSafeInteger(parsed) || parsed <= 0) return undefined;
  return normalizeListLimit(parsed);
}

function normalizeListLimit(limit: number | undefined): number {
  if (!limit || !Number.isSafeInteger(limit) || limit <= 0) return 100;
  return Math.min(limit, MAX_RUNS_LIST_LIMIT);
}

function normalizePrefix(prefix: string): string {
  if (!prefix || prefix === "/") return "";
  return `/${prefix.replace(/^\/+|\/+$/g, "")}`;
}

function isMissingFileError(error: unknown): boolean {
  return Boolean(error && typeof error === "object" && "code" in error && error.code === "ENOENT");
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return Boolean(value && typeof value === "object" && !Array.isArray(value));
}

function readRowString(value: unknown, fallback: string): string {
  if (typeof value === "string") return value;
  if (value instanceof Date) return value.toISOString();
  return fallback;
}

function readNullableString(value: unknown): string | null {
  return typeof value === "string" ? value : null;
}

function readDateString(value: unknown): string {
  if (value instanceof Date) return value.toISOString();
  if (typeof value === "string") return value;
  return new Date(0).toISOString();
}

function readBoolean(value: unknown): boolean {
  return value === true;
}

function readNullableBoolean(value: unknown): boolean | null {
  return typeof value === "boolean" ? value : null;
}

function readNullableNumber(value: unknown): number | null {
  if (typeof value === "number" && Number.isFinite(value)) return value;
  if (typeof value === "string" && value.trim()) {
    const parsed = Number(value);
    return Number.isFinite(parsed) ? parsed : null;
  }
  return null;
}

function readArtifactJson(value: unknown): unknown {
  if (typeof value !== "string") return value ?? {};
  try {
    return JSON.parse(value);
  } catch {
    return {};
  }
}

function countBy(values: readonly string[]): Record<string, number> {
  const counts: Record<string, number> = {};
  for (const value of values) counts[value] = (counts[value] ?? 0) + 1;
  return counts;
}

function roundMetric(value: number): number {
  return Math.round(value * 1000) / 1000;
}

function formatMetric(value: number | null): string {
  return value === null ? "n/a" : String(value);
}

function formatBoolean(value: boolean | null): string {
  if (value === null) return "n/a";
  return value ? "pass" : "fail";
}

function escapeHtml(value: string): string {
  return value
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll("\"", "&quot;")
    .replaceAll("'", "&#39;");
}

function csvCell(value: unknown): string {
  if (value === null || value === undefined) return "";
  const text = String(value);
  if (!/[",\n\r]/.test(text)) return text;
  return `"${text.replaceAll("\"", "\"\"")}"`;
}

function createPostgresTelemetryClient(databaseUrl: string): SqlBenchmarkTelemetryClient {
  let poolPromise: Promise<{ query: SqlBenchmarkTelemetryClient["query"] }> | null = null;
  return {
    async query(sql, values) {
      poolPromise ??= loadPostgresPool(databaseUrl);
      const pool = await poolPromise;
      return pool.query(sql, values);
    }
  };
}

async function loadPostgresPool(databaseUrl: string): Promise<{ query: SqlBenchmarkTelemetryClient["query"] }> {
  const dynamicImport = new Function("specifier", "return import(specifier)") as (specifier: string) => Promise<unknown>;
  const pgModule = await dynamicImport("pg").catch((error) => {
    throw new Error(
      `Postgres benchmark telemetry requires the optional "pg" package to be installed. ${error instanceof Error ? error.message : String(error)}`
    );
  });
  const Pool = isRecord(pgModule)
    ? pgModule.Pool as (new (options: { connectionString: string }) => { query: SqlBenchmarkTelemetryClient["query"] }) | undefined
    : undefined;
  if (typeof Pool !== "function") {
    throw new Error("Postgres benchmark telemetry could not find pg.Pool.");
  }
  return new Pool({ connectionString: databaseUrl }) as { query: SqlBenchmarkTelemetryClient["query"] };
}

class BenchmarkTelemetryError extends Error {
  constructor(
    readonly code: string,
    message: string,
    readonly statusCode: number
  ) {
    super(message);
  }
}
