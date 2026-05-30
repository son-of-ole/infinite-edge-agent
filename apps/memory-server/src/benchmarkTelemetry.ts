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

export interface RegisterBenchmarkTelemetryRoutesOptions extends JsonlBenchmarkTelemetryStoreOptions {
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

export function registerBenchmarkTelemetryRoutes(
  app: FastifyInstance,
  options: RegisterBenchmarkTelemetryRoutesOptions
): void {
  if (!options.enabled) return;
  const prefix = normalizePrefix(options.prefix);
  const store = options.store ?? new JsonlBenchmarkTelemetryStore(options);

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

function countBy(values: readonly string[]): Record<string, number> {
  const counts: Record<string, number> = {};
  for (const value of values) counts[value] = (counts[value] ?? 0) + 1;
  return counts;
}

function roundMetric(value: number): number {
  return Math.round(value * 1000) / 1000;
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
