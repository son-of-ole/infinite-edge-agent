import {
  benchmarkTelemetryPayloadSchema,
  createBenchmarkTelemetryStore,
  type BenchmarkTelemetryStore,
} from "../apps/memory-server/src/benchmarkTelemetry.js";

interface VercelRequestLike {
  method?: string;
  headers: Record<string, string | string[] | undefined>;
  query: Record<string, string | string[] | undefined>;
  body?: unknown;
  socket?: { remoteAddress?: string };
}

interface VercelResponseLike {
  status(code: number): VercelResponseLike;
  setHeader(name: string, value: string): void;
  json(value: unknown): void;
  send(value: unknown): void;
  end(value?: string): void;
}

interface RateLimitBucket {
  count: number;
  resetAt: number;
}

const stores = new Map<string, BenchmarkTelemetryStore>();
const rateLimitBuckets = new Map<string, RateLimitBucket>();

export default async function handler(req: VercelRequestLike, res: VercelResponseLike): Promise<void> {
  if (req.method === "OPTIONS") {
    writeCorsHeaders(res);
    res.status(204).end();
    return;
  }

  if (process.env.BENCHMARK_TELEMETRY_ENABLED !== "true") {
    res.status(404).json({ errorCode: "BENCHMARK_TELEMETRY_DISABLED" });
    return;
  }

  writeCorsHeaders(res);

  if (req.method === "POST") {
    await handlePost(req, res);
    return;
  }

  if (req.method === "GET") {
    await handleList(req, res);
    return;
  }

  res.status(405).json({ errorCode: "METHOD_NOT_ALLOWED" });
}

async function handlePost(req: VercelRequestLike, res: VercelResponseLike): Promise<void> {
  const submitAuth = enforceBearerToken(req, process.env.BENCHMARK_TELEMETRY_SUBMIT_TOKEN);
  if (!submitAuth.ok) {
    res.status(401).json({
      errorCode: "BENCHMARK_TELEMETRY_SUBMIT_UNAUTHORIZED",
      message: "Benchmark telemetry submit token is required.",
    });
    return;
  }

  const rateLimit = consumeRateLimit(readClientIdentity(req));
  if (rateLimit.allowed === false) {
    res.setHeader("retry-after", String(Math.ceil(rateLimit.retryAfterMs / 1000)));
    res.status(429).json({
      errorCode: "BENCHMARK_TELEMETRY_RATE_LIMITED",
      message: "Benchmark telemetry submission rate limit exceeded.",
      retryAfterMs: rateLimit.retryAfterMs,
    });
    return;
  }

  const parsed = benchmarkTelemetryPayloadSchema.safeParse(readRequestBody(req.body));
  if (!parsed.success) {
    res.status(400).json({ error: parsed.error.flatten() });
    return;
  }

  try {
    const record = await getStore().save(parsed.data);
    res.status(202).json({
      ok: true,
      runId: record.runId,
      receivedAt: record.receivedAt,
      artifactBytes: record.artifactBytes,
    });
  } catch (error) {
    res.status(500).json({
      errorCode: "BENCHMARK_TELEMETRY_WRITE_FAILED",
      message: error instanceof Error ? error.message : String(error),
    });
  }
}

async function handleList(req: VercelRequestLike, res: VercelResponseLike): Promise<void> {
  const adminAuth = enforceBearerToken(req, process.env.BENCHMARK_TELEMETRY_ADMIN_TOKEN);
  if (!adminAuth.ok) {
    res.status(401).json({
      errorCode: "BENCHMARK_TELEMETRY_ADMIN_UNAUTHORIZED",
      message: "Benchmark telemetry admin token is required.",
    });
    return;
  }

  try {
    const limit = readLimit(req.query.limit);
    res.status(200).json({ runs: await getStore().list({ ...(limit ? { limit } : {}) }) });
  } catch (error) {
    res.status(500).json({
      errorCode: "BENCHMARK_TELEMETRY_READ_FAILED",
      message: error instanceof Error ? error.message : String(error),
    });
  }
}

function getStore(): BenchmarkTelemetryStore {
  const storage = process.env.BENCHMARK_TELEMETRY_STORAGE === "postgres" ? "postgres" : "jsonl";
  const databaseUrl = process.env.BENCHMARK_TELEMETRY_DATABASE_URL ?? process.env.DATABASE_URL;
  const key = `${storage}:${databaseUrl ?? ""}`;
  const cached = stores.get(key);
  if (cached) return cached;
  const store = createBenchmarkTelemetryStore({
    storage,
    databaseUrl,
    dir: process.env.BENCHMARK_TELEMETRY_DIR ?? ".data/benchmark-runs",
    maxArtifactBytes: readPositiveInt(process.env.BENCHMARK_TELEMETRY_MAX_ARTIFACT_BYTES) ?? 1024 * 1024,
  });
  stores.set(key, store);
  return store;
}

function writeCorsHeaders(res: VercelResponseLike): void {
  res.setHeader("access-control-allow-methods", "GET,POST,OPTIONS");
  res.setHeader("access-control-allow-headers", "authorization,content-type,x-benchmark-telemetry-token");
}

function readRequestBody(body: unknown): unknown {
  if (typeof body !== "string") return body;
  try {
    return JSON.parse(body);
  } catch {
    return body;
  }
}

function enforceBearerToken(
  req: VercelRequestLike,
  token: string | undefined,
): { ok: true } | { ok: false } {
  if (!token) return { ok: true };
  return readBearerToken(req) === token ? { ok: true } : { ok: false };
}

function readBearerToken(req: VercelRequestLike): string | null {
  const authorization = req.headers.authorization;
  const authorizationValue = Array.isArray(authorization) ? authorization[0] : authorization;
  if (authorizationValue?.startsWith("Bearer ")) return authorizationValue.slice("Bearer ".length);
  const headerToken = req.headers["x-benchmark-telemetry-token"];
  return Array.isArray(headerToken) ? headerToken[0] ?? null : headerToken ?? null;
}

function consumeRateLimit(identity: string): { allowed: true } | { allowed: false; retryAfterMs: number } {
  const max = readPositiveInt(process.env.BENCHMARK_TELEMETRY_RATE_LIMIT_MAX) ?? 60;
  const windowMs = readPositiveInt(process.env.BENCHMARK_TELEMETRY_RATE_LIMIT_WINDOW_MS) ?? 10 * 60 * 1000;
  const now = Date.now();
  const current = rateLimitBuckets.get(identity);
  if (!current || current.resetAt <= now) {
    rateLimitBuckets.set(identity, { count: 1, resetAt: now + windowMs });
    return { allowed: true };
  }
  if (current.count >= max) {
    return { allowed: false, retryAfterMs: Math.max(0, current.resetAt - now) };
  }
  current.count += 1;
  return { allowed: true };
}

function readClientIdentity(req: VercelRequestLike): string {
  const forwarded = req.headers["x-forwarded-for"];
  const forwardedValue = Array.isArray(forwarded) ? forwarded[0] : forwarded;
  if (forwardedValue?.trim()) return forwardedValue.split(",")[0]?.trim() || "unknown";
  const realIp = req.headers["x-real-ip"];
  const realIpValue = Array.isArray(realIp) ? realIp[0] : realIp;
  return realIpValue?.trim() || req.socket?.remoteAddress || "unknown";
}

function readLimit(value: string | string[] | undefined): number | undefined {
  const raw = Array.isArray(value) ? value[0] : value;
  if (!raw) return undefined;
  const parsed = Number(raw);
  return Number.isSafeInteger(parsed) && parsed > 0 ? Math.min(parsed, 500) : undefined;
}

function readPositiveInt(value: string | undefined): number | undefined {
  if (!value?.trim()) return undefined;
  const parsed = Number(value);
  return Number.isSafeInteger(parsed) && parsed > 0 ? parsed : undefined;
}
