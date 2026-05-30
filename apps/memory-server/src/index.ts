import cors from "@fastify/cors";
import Fastify from "fastify";
import type { FastifyInstance, FastifyReply, FastifyRequest } from "fastify";
import type { GacListOptions, MemoryChunk, MemoryDeleteOptions, MemorySearchOptions, RawMemorySearchOptions, RuntimeTrace, RuntimeTraceListOptions } from "@infinite-edge-agent/core";
import { createAesGcmStringCodec } from "./encryption.js";
import { registerBenchmarkTelemetryRoutes } from "./benchmarkTelemetry.js";
import { LanceMemoryStore } from "./lancedbStore.js";
import { shouldExposeLocalRoutes } from "./localRoutes.js";
import {
  filterMemoryChunksForScope,
  hasConfiguredRemoteSecurity,
  resolveRemoteScope,
  restoreRequestedSearchLimit,
  scopedClearOptions,
  scopedDeleteOptions,
  scopedGacListOptions,
  scopedGacRecords,
  scopedRuntimeTrace,
  scopedRuntimeTraceListOptions,
  scopedSearchOptions,
  stampMemoryChunksForScope,
  type RemoteScope
} from "./remoteScope.js";
import {
  deleteMemoryRequestSchema,
  importMemoryRequestSchema,
  rawMemorySearchRequestSchema,
  searchRequestSchema,
  upsertRequestSchema,
  writeContextPackTracesRequestSchema,
  writeClusterMetricsRequestSchema,
  writeConsolidationRunsRequestSchema,
  writeIdentityPinsRequestSchema,
  writeMemoryLineageRequestSchema,
  writeMemoryClustersRequestSchema,
  writeMemoryContradictionsRequestSchema,
  writeModelMemoryActionsRequestSchema,
  writeMemoryRepresentativesRequestSchema,
  writeRawMemoryRequestSchema,
  writeRetrievalAuditsRequestSchema,
  writeSourceDocumentsRequestSchema,
  writeTrainingExamplesRequestSchema,
  writeRuntimeTraceRequestSchema
} from "./types.js";

const host = process.env.MEMORY_SERVER_HOST ?? "127.0.0.1";
const port = Number(process.env.MEMORY_SERVER_PORT ?? 8787);
const dbUri = process.env.MEMORY_DB_URI ?? ".data/lancedb";
const tableName = process.env.MEMORY_TABLE ?? "memory_chunks";
const expectedVectorDimension = readPositiveInt(process.env.MEMORY_VECTOR_DIMENSION, 384);
const apiPrefix = normalizePrefix(process.env.MEMORY_API_PREFIX ?? "/api/edge-ai");
const authToken = process.env.MEMORY_SERVER_TOKEN;
const expectedTenantId = process.env.MEMORY_TENANT_ID;
const expectedCellId = process.env.MEMORY_CELL_ID;
const corsOrigin = parseCorsOrigin(process.env.MEMORY_CORS_ORIGIN);
const memoryEncryptionKey = process.env.MEMORY_ENCRYPTION_KEY;
const benchmarkTelemetryEnabled = process.env.BENCHMARK_TELEMETRY_ENABLED === "true";
const benchmarkTelemetryPrefix = process.env.BENCHMARK_TELEMETRY_PREFIX ?? "/api/benchmark-runs";
const benchmarkTelemetryDir = process.env.BENCHMARK_TELEMETRY_DIR ?? ".data/benchmark-runs";
const benchmarkTelemetryMaxArtifactBytes = readPositiveInt(
  process.env.BENCHMARK_TELEMETRY_MAX_ARTIFACT_BYTES,
  1024 * 1024
);

const store = new LanceMemoryStore(dbUri, tableName, {
  ...(memoryEncryptionKey ? { stringCodec: createAesGcmStringCodec(memoryEncryptionKey) } : {})
});
const server = Fastify({
  logger: true,
  bodyLimit: readPositiveInt(process.env.MEMORY_SERVER_BODY_LIMIT_BYTES, 2 * 1024 * 1024)
});

await server.register(cors, {
  origin: corsOrigin,
  methods: ["GET", "POST", "DELETE", "OPTIONS"]
});

if (shouldExposeLocalRoutes({
  host,
  ...(process.env.MEMORY_EXPOSE_LOCAL_ROUTES !== undefined ? { value: process.env.MEMORY_EXPOSE_LOCAL_ROUTES } : {})
})) {
  registerMemoryRoutes(server, "", "lancedb-sidecar");
}
if (apiPrefix) {
  registerMemoryRoutes(server, apiPrefix, "remote-http");
}
registerBenchmarkTelemetryRoutes(server, {
  enabled: benchmarkTelemetryEnabled,
  prefix: benchmarkTelemetryPrefix,
  dir: benchmarkTelemetryDir,
  maxArtifactBytes: benchmarkTelemetryMaxArtifactBytes
});

await server.listen({ host, port });

function registerMemoryRoutes(app: FastifyInstance, prefix: string, mode: "lancedb-sidecar" | "remote-http"): void {
  const routeOptions = mode === "remote-http" ? { preHandler: enforceRemoteAuth } : {};

  app.get(`${prefix}/health`, routeOptions, async () => ({
    ok: true,
    mode,
    apiPrefix: mode === "remote-http" ? prefix : undefined,
    dbUri,
    tableName,
    expectedVectorDimension,
    encryptionEnabled: Boolean(memoryEncryptionKey)
  }));

  app.get(`${prefix}/memory/status`, routeOptions, async () => ({
    ...(await store.getDatabaseStatus()),
    mode,
    apiPrefix: mode === "remote-http" ? prefix : undefined,
    dbUri,
    tableName,
    expectedVectorDimension,
    encryptionEnabled: Boolean(memoryEncryptionKey)
  }));

  app.post(`${prefix}/memory/repair`, routeOptions, async () => ({
    ...(await store.getDatabaseStatus({ repair: true, expectedVectorDimension })),
    mode,
    apiPrefix: mode === "remote-http" ? prefix : undefined,
    dbUri,
    tableName,
    expectedVectorDimension,
    encryptionEnabled: Boolean(memoryEncryptionKey)
  }));

  app.post(`${prefix}/memory/upsert`, routeOptions, async (request, reply) => {
    const parsed = upsertRequestSchema.safeParse(request.body);
    if (!parsed.success) {
      return reply.status(400).send({ error: parsed.error.flatten() });
    }
    const chunks = parsed.data.chunks.map(toMemoryChunk);
    const dimensionError = validateMemoryChunkDimensions(chunks);
    if (dimensionError) return reply.status(400).send(dimensionError);
    const scoped = bindMemoryWriteScope(chunks, request, reply, mode);
    if (!scoped.ok) return scoped.response;
    await store.upsert(scoped.value);
    return { ok: true, count: parsed.data.chunks.length };
  });

  app.post(`${prefix}/memory/search`, routeOptions, async (request, reply) => {
    const parsed = searchRequestSchema.safeParse(request.body);
    if (!parsed.success) {
      return reply.status(400).send({ error: parsed.error.flatten() });
    }
    const dimensionError = validateEmbeddingDimension(parsed.data.embedding);
    if (dimensionError) return reply.status(400).send(dimensionError);
    const options = toSearchOptions(parsed.data.options);
    if (mode === "remote-http") {
      const scope = getRemoteScope(request, reply);
      if (!scope) return reply;
      const hits = await store.search(parsed.data.embedding, scopedSearchOptions(options, scope));
      return { hits: restoreRequestedSearchLimit(filterMemoryChunksForScope(hits, scope), options?.limit) };
    }
    const hits = await store.search(parsed.data.embedding, options);
    return { hits };
  });

  app.delete(`${prefix}/memory`, routeOptions, async (request, reply) => {
    if (mode === "remote-http") {
      const scope = getRemoteScope(request, reply);
      if (!scope) return reply;
      const count = await store.deleteMemory(scopedClearOptions(scope));
      return { ok: true, count };
    }
    await store.clear();
    return { ok: true };
  });

  app.delete(`${prefix}/memory/query`, routeOptions, async (request, reply) => {
    const parsed = deleteMemoryRequestSchema.safeParse(request.body);
    if (!parsed.success) {
      return reply.status(400).send({ error: parsed.error.flatten() });
    }
    const options = toDeleteOptions(parsed.data.options);
    const scoped = mode === "remote-http" ? getRemoteScope(request, reply) : null;
    if (mode === "remote-http" && !scoped) return reply;
    const count = await store.deleteMemory(scoped ? scopedDeleteOptions(options, scoped) : options);
    return { ok: true, count };
  });

  app.get(`${prefix}/memory/export`, routeOptions, async (request, reply) => {
    const query = request.query as { sessionId?: string; limit?: string; tenantId?: string; cellId?: string };
    const scope = mode === "remote-http" ? getRemoteScope(request, reply) : null;
    if (mode === "remote-http" && !scope) return reply;
    const requestedLimit = parseQueryLimit(query.limit);
    const chunks = await store.listMemoryChunks({
      ...(query.sessionId ? { sessionId: query.sessionId } : {}),
      ...(scope ? { tenantId: scope.tenantId, cellId: scope.cellId } : {}),
      ...(!scope && query.tenantId ? { tenantId: query.tenantId } : {}),
      ...(!scope && query.cellId ? { cellId: query.cellId } : {}),
      ...(requestedLimit !== undefined ? { limit: requestedLimit } : {})
    });
    if (mode === "remote-http" && scope) {
      return { chunks: filterMemoryChunksForScope(chunks, scope, requestedLimit) };
    }
    return { chunks };
  });

  app.post(`${prefix}/memory/import`, routeOptions, async (request, reply) => {
    const parsed = importMemoryRequestSchema.safeParse(request.body);
    if (!parsed.success) {
      return reply.status(400).send({ error: parsed.error.flatten() });
    }
    const chunks = parsed.data.chunks.map(toMemoryChunk);
    const dimensionError = validateMemoryChunkDimensions(chunks);
    if (dimensionError) return reply.status(400).send(dimensionError);
    const scoped = bindMemoryWriteScope(chunks, request, reply, mode);
    if (!scoped.ok) return scoped.response;
    await store.importMemoryChunks(scoped.value);
    return { ok: true, count: parsed.data.chunks.length };
  });

  app.post(`${prefix}/runtime/traces`, routeOptions, async (request, reply) => {
    const parsed = writeRuntimeTraceRequestSchema.safeParse(request.body);
    if (!parsed.success) {
      return reply.status(400).send({ error: parsed.error.flatten() });
    }
    const scope = mode === "remote-http" ? getRemoteScope(request, reply) : null;
    if (mode === "remote-http" && !scope) return reply;
    let trace = toRuntimeTrace(parsed.data.trace);
    if (scope) {
      try {
        trace = scopedRuntimeTrace(trace, scope);
      } catch (error) {
        return reply.status(403).send({
          errorCode: "RUNTIME_TRACE_SCOPE_MISMATCH",
          message: error instanceof Error ? error.message : String(error),
          retryable: false,
          policyViolation: true,
          traceId: makeRouteTraceId("runtime_trace_scope_mismatch")
        });
      }
    }
    await store.writeRuntimeTrace(trace);
    return { ok: true, traceId: trace.traceId };
  });

  app.get(`${prefix}/runtime/traces`, routeOptions, async (request, reply) => {
    const scope = mode === "remote-http" ? getRemoteScope(request, reply) : null;
    if (mode === "remote-http" && !scope) return reply;
    let options = toRuntimeTraceListOptions(request.query);
    if (scope) {
      try {
        options = scopedRuntimeTraceListOptions(options, scope);
      } catch (error) {
        return reply.status(403).send({
          errorCode: "RUNTIME_TRACE_SCOPE_MISMATCH",
          message: error instanceof Error ? error.message : String(error),
          retryable: false,
          policyViolation: true,
          traceId: makeRouteTraceId("runtime_trace_scope_mismatch")
        });
      }
    }
    const traces = await store.listRuntimeTraces(options);
    return { traces };
  });

  app.post(`${prefix}/gac/raw-memory`, routeOptions, async (request, reply) => {
    const parsed = writeRawMemoryRequestSchema.safeParse(request.body);
    if (!parsed.success) return reply.status(400).send({ error: parsed.error.flatten() });
    const scoped = bindGacWriteScope(parsed.data.records, request, reply, mode);
    if (!scoped.ok) return scoped.response;
    return store.writeRawMemory(scoped.value);
  });

  app.get(`${prefix}/gac/raw-memory`, routeOptions, async (request, reply) => {
    const scoped = bindGacListScope(toGacListOptions(request.query), request, reply, mode);
    if (!scoped.ok) return scoped.response;
    return { records: await store.listRawMemory(scoped.value) };
  });

  app.post(`${prefix}/gac/raw-memory/search`, routeOptions, async (request, reply) => {
    const parsed = rawMemorySearchRequestSchema.safeParse(request.body);
    if (!parsed.success) return reply.status(400).send({ error: parsed.error.flatten() });
    const scoped = bindRawMemorySearchScope(toRawMemorySearchOptions(parsed.data.options), request, reply, mode);
    if (!scoped.ok) return scoped.response;
    return { records: await store.searchRawMemory(scoped.value), fallback: "raw_memory" };
  });

  app.post(`${prefix}/gac/identity-pins`, routeOptions, async (request, reply) => {
    const parsed = writeIdentityPinsRequestSchema.safeParse(request.body);
    if (!parsed.success) return reply.status(400).send({ error: parsed.error.flatten() });
    const scoped = bindGacWriteScope(parsed.data.records, request, reply, mode);
    if (!scoped.ok) return scoped.response;
    return store.writeIdentityPins(scoped.value);
  });

  app.get(`${prefix}/gac/identity-pins`, routeOptions, async (request, reply) => {
    const scoped = bindGacListScope(toGacListOptions(request.query), request, reply, mode);
    if (!scoped.ok) return scoped.response;
    return { records: await store.listIdentityPins(scoped.value) };
  });

  app.post(`${prefix}/gac/clusters`, routeOptions, async (request, reply) => {
    const parsed = writeMemoryClustersRequestSchema.safeParse(request.body);
    if (!parsed.success) return reply.status(400).send({ error: parsed.error.flatten() });
    const scoped = bindGacWriteScope(parsed.data.records, request, reply, mode);
    if (!scoped.ok) return scoped.response;
    return store.writeMemoryClusters(scoped.value);
  });

  app.get(`${prefix}/gac/clusters`, routeOptions, async (request, reply) => {
    const scoped = bindGacListScope(toGacListOptions(request.query), request, reply, mode);
    if (!scoped.ok) return scoped.response;
    return { records: await store.listMemoryClusters(scoped.value) };
  });

  app.post(`${prefix}/gac/cluster-metrics`, routeOptions, async (request, reply) => {
    const parsed = writeClusterMetricsRequestSchema.safeParse(request.body);
    if (!parsed.success) return reply.status(400).send({ error: parsed.error.flatten() });
    const scoped = bindGacWriteScope(parsed.data.records, request, reply, mode);
    if (!scoped.ok) return scoped.response;
    return store.writeClusterMetrics(scoped.value);
  });

  app.get(`${prefix}/gac/cluster-metrics`, routeOptions, async (request, reply) => {
    const scoped = bindGacListScope(toGacListOptions(request.query), request, reply, mode);
    if (!scoped.ok) return scoped.response;
    return { records: await store.listClusterMetrics(scoped.value) };
  });

  app.post(`${prefix}/gac/representatives`, routeOptions, async (request, reply) => {
    const parsed = writeMemoryRepresentativesRequestSchema.safeParse(request.body);
    if (!parsed.success) return reply.status(400).send({ error: parsed.error.flatten() });
    const scopedRecords = bindGacWriteScope(parsed.data.records, request, reply, mode);
    if (!scopedRecords.ok) return scopedRecords.response;
    const scopedLineage = bindGacWriteScope(parsed.data.lineage ?? [], request, reply, mode);
    if (!scopedLineage.ok) return scopedLineage.response;
    try {
      return await store.writeMemoryRepresentatives(scopedRecords.value, scopedLineage.value);
    } catch (error) {
      if (error instanceof Error && error.message.includes("MISSING_LINEAGE")) {
        return reply.status(400).send({
          errorCode: "MISSING_LINEAGE",
          message: error.message,
          retryable: false,
          policyViolation: true,
          traceId: makeRouteTraceId("gac_memory_representative")
        });
      }
      throw error;
    }
  });

  app.get(`${prefix}/gac/representatives`, routeOptions, async (request, reply) => {
    const scoped = bindGacListScope(toGacListOptions(request.query), request, reply, mode);
    if (!scoped.ok) return scoped.response;
    return { records: await store.listMemoryRepresentatives(scoped.value) };
  });

  app.post(`${prefix}/gac/lineage`, routeOptions, async (request, reply) => {
    const parsed = writeMemoryLineageRequestSchema.safeParse(request.body);
    if (!parsed.success) return reply.status(400).send({ error: parsed.error.flatten() });
    const scoped = bindGacWriteScope(parsed.data.records, request, reply, mode);
    if (!scoped.ok) return scoped.response;
    return store.writeMemoryLineage(scoped.value);
  });

  app.get(`${prefix}/gac/lineage`, routeOptions, async (request, reply) => {
    const scoped = bindGacListScope(toGacListOptions(request.query), request, reply, mode);
    if (!scoped.ok) return scoped.response;
    return { records: await store.listMemoryLineage(scoped.value) };
  });

  app.post(`${prefix}/gac/consolidation-runs`, routeOptions, async (request, reply) => {
    const parsed = writeConsolidationRunsRequestSchema.safeParse(request.body);
    if (!parsed.success) return reply.status(400).send({ error: parsed.error.flatten() });
    const scoped = bindGacWriteScope(parsed.data.records, request, reply, mode);
    if (!scoped.ok) return scoped.response;
    return store.writeConsolidationRuns(scoped.value);
  });

  app.get(`${prefix}/gac/consolidation-runs`, routeOptions, async (request, reply) => {
    const scoped = bindGacListScope(toGacListOptions(request.query), request, reply, mode);
    if (!scoped.ok) return scoped.response;
    return { records: await store.listConsolidationRuns(scoped.value) };
  });

  app.post(`${prefix}/gac/retrieval-audits`, routeOptions, async (request, reply) => {
    const parsed = writeRetrievalAuditsRequestSchema.safeParse(request.body);
    if (!parsed.success) return reply.status(400).send({ error: parsed.error.flatten() });
    const scoped = bindGacWriteScope(parsed.data.records, request, reply, mode);
    if (!scoped.ok) return scoped.response;
    return store.writeRetrievalAudits(scoped.value);
  });

  app.get(`${prefix}/gac/retrieval-audits`, routeOptions, async (request, reply) => {
    const scoped = bindGacListScope(toGacListOptions(request.query), request, reply, mode);
    if (!scoped.ok) return scoped.response;
    return { records: await store.listRetrievalAudits(scoped.value) };
  });

  app.post(`${prefix}/gac/context-pack-traces`, routeOptions, async (request, reply) => {
    const parsed = writeContextPackTracesRequestSchema.safeParse(request.body);
    if (!parsed.success) return reply.status(400).send({ error: parsed.error.flatten() });
    const scoped = bindGacWriteScope(parsed.data.records, request, reply, mode);
    if (!scoped.ok) return scoped.response;
    return store.writeContextPackTraces(scoped.value);
  });

  app.get(`${prefix}/gac/context-pack-traces`, routeOptions, async (request, reply) => {
    const scoped = bindGacListScope(toGacListOptions(request.query), request, reply, mode);
    if (!scoped.ok) return scoped.response;
    return { traces: await store.listContextPackTraces(scoped.value) };
  });

  app.post(`${prefix}/gac/model-memory-actions`, routeOptions, async (request, reply) => {
    const parsed = writeModelMemoryActionsRequestSchema.safeParse(request.body);
    if (!parsed.success) return reply.status(400).send({ error: parsed.error.flatten() });
    const scoped = bindGacWriteScope(parsed.data.records, request, reply, mode);
    if (!scoped.ok) return scoped.response;
    return store.writeModelMemoryActions(scoped.value);
  });

  app.get(`${prefix}/gac/model-memory-actions`, routeOptions, async (request, reply) => {
    const scoped = bindGacListScope(toGacListOptions(request.query), request, reply, mode);
    if (!scoped.ok) return scoped.response;
    return { records: await store.listModelMemoryActions(scoped.value) };
  });

  app.post(`${prefix}/gac/contradictions`, routeOptions, async (request, reply) => {
    const parsed = writeMemoryContradictionsRequestSchema.safeParse(request.body);
    if (!parsed.success) return reply.status(400).send({ error: parsed.error.flatten() });
    const scoped = bindGacWriteScope(parsed.data.records, request, reply, mode);
    if (!scoped.ok) return scoped.response;
    return store.writeMemoryContradictions(scoped.value);
  });

  app.get(`${prefix}/gac/contradictions`, routeOptions, async (request, reply) => {
    const scoped = bindGacListScope(toGacListOptions(request.query), request, reply, mode);
    if (!scoped.ok) return scoped.response;
    return { records: await store.listMemoryContradictions(scoped.value) };
  });

  app.post(`${prefix}/gac/source-documents`, routeOptions, async (request, reply) => {
    const parsed = writeSourceDocumentsRequestSchema.safeParse(request.body);
    if (!parsed.success) return reply.status(400).send({ error: parsed.error.flatten() });
    const scoped = bindGacWriteScope(parsed.data.records, request, reply, mode);
    if (!scoped.ok) return scoped.response;
    return store.writeSourceDocuments(scoped.value);
  });

  app.get(`${prefix}/gac/source-documents`, routeOptions, async (request, reply) => {
    const scoped = bindGacListScope(toGacListOptions(request.query), request, reply, mode);
    if (!scoped.ok) return scoped.response;
    return { records: await store.listSourceDocuments(scoped.value) };
  });

  app.post(`${prefix}/gac/training-examples`, routeOptions, async (request, reply) => {
    const parsed = writeTrainingExamplesRequestSchema.safeParse(request.body);
    if (!parsed.success) return reply.status(400).send({ error: parsed.error.flatten() });
    const scoped = bindGacWriteScope(parsed.data.records, request, reply, mode);
    if (!scoped.ok) return scoped.response;
    return store.writeTrainingExamples(scoped.value);
  });

  app.get(`${prefix}/gac/training-examples`, routeOptions, async (request, reply) => {
    const scoped = bindGacListScope(toGacListOptions(request.query), request, reply, mode);
    if (!scoped.ok) return scoped.response;
    return { records: await store.listTrainingExamples(scoped.value) };
  });
}

async function enforceRemoteAuth(request: FastifyRequest, reply: FastifyReply): Promise<void> {
  if (!hasConfiguredRemoteSecurity({ token: authToken, tenantId: expectedTenantId, cellId: expectedCellId })) {
    await reply.status(503).send({
      error: "Remote memory API requires MEMORY_SERVER_TOKEN, MEMORY_TENANT_ID, and MEMORY_CELL_ID."
    });
    return;
  }

  if (authToken) {
    const authorization = request.headers.authorization;
    if (authorization !== `Bearer ${authToken}`) {
      await reply.status(401).send({ error: "Unauthorized" });
      return;
    }
  }

  const tenantId = request.headers["x-edge-agent-tenant"];
  if (expectedTenantId && tenantId !== expectedTenantId) {
    await reply.status(403).send({ error: "Tenant is not allowed." });
    return;
  }

  const cellId = request.headers["x-edge-agent-cell"];
  if (expectedCellId && cellId !== expectedCellId) {
    await reply.status(403).send({ error: "Cell is not allowed." });
    return;
  }
}

function toMemoryChunk(chunk: typeof upsertRequestSchema._type.chunks[number]): MemoryChunk {
  return {
    id: chunk.id,
    text: chunk.text,
    embedding: chunk.embedding,
    sessionId: chunk.sessionId,
    source: chunk.source,
    ...(chunk.role ? { role: chunk.role } : {}),
    createdAt: chunk.createdAt,
    updatedAt: chunk.updatedAt,
    tags: chunk.tags,
    metadata: chunk.metadata,
    tokenCount: chunk.tokenCount
  };
}

function toSearchOptions(options: typeof searchRequestSchema._type.options): MemorySearchOptions | undefined {
  if (!options) return undefined;
  return {
    ...(options.limit !== undefined ? { limit: options.limit } : {}),
    ...(options.minScore !== undefined ? { minScore: options.minScore } : {}),
    ...(options.sessionId !== undefined ? { sessionId: options.sessionId } : {}),
    ...(options.tags !== undefined ? { tags: options.tags } : {}),
    ...(options.maxAgeMs !== undefined ? { maxAgeMs: options.maxAgeMs } : {})
  };
}

function toDeleteOptions(options: typeof deleteMemoryRequestSchema._type.options): MemoryDeleteOptions {
  return {
    ...(options.sessionId !== undefined ? { sessionId: options.sessionId } : {}),
    ...(options.tags !== undefined ? { tags: options.tags } : {})
  };
}

function toGacListOptions(query: unknown): GacListOptions {
  const params = query as {
    tenantId?: string;
    cellId?: string;
    sessionId?: string;
    rawMemoryId?: string;
    representativeId?: string;
    contextPackId?: string;
    limit?: string;
  };
  const limit = parseQueryLimit(params.limit);
  return {
    ...(params.tenantId ? { tenantId: params.tenantId } : {}),
    ...(params.cellId ? { cellId: params.cellId } : {}),
    ...(params.sessionId ? { sessionId: params.sessionId } : {}),
    ...(params.rawMemoryId ? { rawMemoryId: params.rawMemoryId } : {}),
    ...(params.representativeId ? { representativeId: params.representativeId } : {}),
    ...(params.contextPackId ? { contextPackId: params.contextPackId } : {}),
    ...(limit !== undefined ? { limit } : {})
  };
}

function toRawMemorySearchOptions(options: typeof rawMemorySearchRequestSchema._type.options): RawMemorySearchOptions {
  if (!options) return {};
  return {
    ...(options.tenantId ? { tenantId: options.tenantId } : {}),
    ...(options.cellId ? { cellId: options.cellId } : {}),
    ...(options.sessionId ? { sessionId: options.sessionId } : {}),
    ...(options.rawMemoryId ? { rawMemoryId: options.rawMemoryId } : {}),
    ...(options.limit !== undefined ? { limit: options.limit } : {}),
    ...(options.queryText ? { queryText: options.queryText } : {}),
    ...(options.includeDeleted !== undefined ? { includeDeleted: options.includeDeleted } : {})
  };
}

function toRuntimeTraceListOptions(query: unknown): RuntimeTraceListOptions {
  const params = query as {
    tenantId?: string;
    cellId?: string;
    sessionId?: string;
    limit?: string;
  };
  const limit = parseQueryLimit(params.limit);
  return {
    ...(params.tenantId ? { tenantId: params.tenantId } : {}),
    ...(params.cellId ? { cellId: params.cellId } : {}),
    ...(params.sessionId ? { sessionId: params.sessionId } : {}),
    ...(limit !== undefined ? { limit } : {})
  };
}

function toRuntimeTrace(trace: typeof writeRuntimeTraceRequestSchema._type.trace): RuntimeTrace {
  return {
    traceId: trace.traceId,
    requestId: trace.requestId,
    sessionId: trace.sessionId,
    ...(trace.tenantId !== undefined ? { tenantId: trace.tenantId } : {}),
    ...(trace.cellId !== undefined ? { cellId: trace.cellId } : {}),
    modelId: trace.modelId,
    backend: trace.backend,
    createdAt: trace.createdAt,
    runtime: trace.runtime
  };
}

function bindGacWriteScope<T extends { tenantId: string; cellId: string }>(
  records: T[],
  request: FastifyRequest,
  reply: FastifyReply,
  mode: "lancedb-sidecar" | "remote-http"
): { ok: true; value: T[] } | { ok: false; response: FastifyReply } {
  if (mode !== "remote-http") return { ok: true, value: records };
  const scope = getRemoteScope(request, reply);
  if (!scope) return { ok: false, response: reply };
  try {
    return { ok: true, value: scopedGacRecords(records, scope) };
  } catch (error) {
    return {
      ok: false,
      response: reply.status(403).send({
        errorCode: "GAC_SCOPE_MISMATCH",
        message: error instanceof Error ? error.message : String(error),
        retryable: false,
        policyViolation: true,
        traceId: makeRouteTraceId("gac_scope_mismatch")
      })
    };
  }
}

function bindGacListScope(
  options: GacListOptions,
  request: FastifyRequest,
  reply: FastifyReply,
  mode: "lancedb-sidecar" | "remote-http"
): { ok: true; value: GacListOptions } | { ok: false; response: FastifyReply } {
  if (mode !== "remote-http") return { ok: true, value: options };
  const scope = resolveRemoteScope({
    tenant: request.headers["x-edge-agent-tenant"],
    cell: request.headers["x-edge-agent-cell"]
  }, {
    ...(expectedTenantId ? { tenantId: expectedTenantId } : {}),
    ...(expectedCellId ? { cellId: expectedCellId } : {})
  });
  if (!scope) {
    return {
      ok: false,
      response: reply.status(400).send({
        errorCode: "GAC_SCOPE_REQUIRED",
        message: "Remote GAC routes require tenant and cell scope.",
        retryable: false,
        policyViolation: true,
        traceId: makeRouteTraceId("gac_scope_required")
      })
    };
  }
  try {
    return { ok: true, value: scopedGacListOptions(options, scope) };
  } catch (error) {
    return {
      ok: false,
      response: reply.status(403).send({
        errorCode: "GAC_SCOPE_MISMATCH",
        message: error instanceof Error ? error.message : String(error),
        retryable: false,
        policyViolation: true,
        traceId: makeRouteTraceId("gac_scope_mismatch")
      })
    };
  }
}

function bindRawMemorySearchScope(
  options: RawMemorySearchOptions,
  request: FastifyRequest,
  reply: FastifyReply,
  mode: "lancedb-sidecar" | "remote-http"
): { ok: true; value: RawMemorySearchOptions } | { ok: false; response: FastifyReply } {
  const scoped = bindGacListScope(options, request, reply, mode);
  if (!scoped.ok) return scoped;
  return {
    ok: true,
    value: {
      ...options,
      ...scoped.value
    }
  };
}

function bindMemoryWriteScope(
  chunks: MemoryChunk[],
  request: FastifyRequest,
  reply: FastifyReply,
  mode: "lancedb-sidecar" | "remote-http"
): { ok: true; value: MemoryChunk[] } | { ok: false; response: FastifyReply } {
  if (mode !== "remote-http") return { ok: true, value: chunks };
  const scope = getRemoteScope(request, reply);
  if (!scope) return { ok: false, response: reply };
  return { ok: true, value: stampMemoryChunksForScope(chunks, scope) };
}

function getRemoteScope(request: FastifyRequest, reply: FastifyReply): RemoteScope | null {
  const scope = resolveRemoteScope({
    tenant: request.headers["x-edge-agent-tenant"],
    cell: request.headers["x-edge-agent-cell"]
  }, {
    ...(expectedTenantId ? { tenantId: expectedTenantId } : {}),
    ...(expectedCellId ? { cellId: expectedCellId } : {})
  });
  if (scope) return scope;
  reply.status(400).send({
    errorCode: "GAC_SCOPE_REQUIRED",
    message: "Remote memory routes require configured tenant and cell scope.",
    retryable: false,
    policyViolation: true,
    traceId: makeRouteTraceId("gac_scope_required")
  });
  return null;
}

function makeRouteTraceId(scope: string): string {
  if (typeof crypto !== "undefined" && "randomUUID" in crypto) {
    return `${scope}_${crypto.randomUUID()}`;
  }
  return `${scope}_${Date.now()}`;
}

function validateMemoryChunkDimensions(chunks: MemoryChunk[]): { errorCode: string; message: string } | null {
  const invalid = chunks.find((chunk) => chunk.embedding.length !== expectedVectorDimension);
  if (!invalid) return null;
  return {
    errorCode: "VECTOR_DIMENSION_MISMATCH",
    message: `Memory chunk ${invalid.id} embedding dimension ${invalid.embedding.length} does not match MEMORY_VECTOR_DIMENSION=${expectedVectorDimension}.`
  };
}

function validateEmbeddingDimension(embedding: number[]): { errorCode: string; message: string } | null {
  if (embedding.length === expectedVectorDimension) return null;
  return {
    errorCode: "VECTOR_DIMENSION_MISMATCH",
    message: `Query embedding dimension ${embedding.length} does not match MEMORY_VECTOR_DIMENSION=${expectedVectorDimension}.`
  };
}

function parseQueryLimit(value: string | undefined): number | undefined {
  if (!value) return undefined;
  const parsed = Number(value);
  if (!Number.isSafeInteger(parsed) || parsed <= 0) return undefined;
  return Math.min(parsed, 1000);
}

function normalizePrefix(prefix: string): string {
  if (!prefix || prefix === "/") return "";
  return `/${prefix.replace(/^\/+|\/+$/g, "")}`;
}

function parseCorsOrigin(value: string | undefined): boolean | string[] {
  if (value === "*") return true;
  const origins = value?.split(",").map((origin) => origin.trim()).filter(Boolean);
  if (origins && origins.length > 0) return origins;
  return ["http://127.0.0.1:5173", "http://localhost:5173"];
}

function readPositiveInt(value: string | undefined, fallback: number): number {
  if (!value) return fallback;
  const parsed = Number(value);
  if (!Number.isSafeInteger(parsed) || parsed <= 0) {
    throw new Error(`Invalid positive integer env value: ${value}`);
  }
  return parsed;
}
