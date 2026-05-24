import type {
  ClusterMetricRecord,
  ConsolidationRunRecord,
  ContextPackTraceRecord,
  GacListOptions,
  GacMemoryStore,
  GacWriteResult,
  IdentityPinRecord,
  MemoryChunk,
  MemoryClusterRecord,
  MemoryContradictionRecord,
  MemoryDeleteOptions,
  MemoryLineageRecord,
  MemoryRepresentativeRecord,
  MemorySearchHit,
  MemorySearchOptions,
  MemorySnapshotStore,
  MemoryStore,
  ModelMemoryActionRecord,
  RawMemoryRecord,
  RetrievalAuditRecord,
  RuntimeTrace,
  RuntimeTraceListOptions,
  RuntimeTraceSnapshotStore,
  RuntimeTraceStore,
  SourceDocumentRecord,
  TrainingExampleRecord
} from "@infinite-edge-agent/core";

export interface RemoteMemoryStoreOptions {
  baseUrl: string;
  token?: string;
  tenantId?: string;
  cellId?: string;
  credentials?: RequestCredentials;
  healthTimeoutMs?: number;
}

export class RemoteMemoryStore implements MemoryStore, MemorySnapshotStore, RuntimeTraceStore, RuntimeTraceSnapshotStore, GacMemoryStore {
  private readonly baseUrl: string;

  constructor(private readonly options: RemoteMemoryStoreOptions) {
    this.baseUrl = options.baseUrl.replace(/\/$/, "");
  }

  async health(): Promise<boolean> {
    const controller = typeof AbortController !== "undefined" ? new AbortController() : null;
    const timeoutMs = this.options.healthTimeoutMs ?? 5000;
    const timeout = controller
      ? setTimeout(() => controller.abort(), timeoutMs)
      : null;
    try {
      const response = await this.request("/health", {
        method: "GET",
        ...(controller ? { signal: controller.signal } : {}),
      });
      if (!response.ok) return false;
      const contentType = response.headers.get("content-type") ?? "";
      if (contentType.toLowerCase().includes("text/html")) return false;
      if (contentType && !contentType.toLowerCase().includes("application/json")) return false;
      const data = await response.json().catch(() => null) as { ok?: unknown; mode?: unknown } | null;
      return data?.ok === true && (data.mode === undefined || data.mode === "remote-http");
    } catch {
      return false;
    } finally {
      if (timeout) clearTimeout(timeout);
    }
  }

  async upsert(chunks: MemoryChunk[]): Promise<void> {
    const response = await this.request("/memory/upsert", {
      method: "POST",
      body: JSON.stringify({ chunks })
    });
    if (!response.ok) throw new Error(`Remote memory upsert failed: ${response.status}`);
  }

  async search(queryEmbedding: number[], options: MemorySearchOptions = {}): Promise<MemorySearchHit[]> {
    const response = await this.request("/memory/search", {
      method: "POST",
      body: JSON.stringify({ embedding: queryEmbedding, options })
    });
    if (!response.ok) throw new Error(`Remote memory search failed: ${response.status}`);
    const data = (await response.json()) as { hits?: MemorySearchHit[] };
    return data.hits ?? [];
  }

  async clear(): Promise<void> {
    const response = await this.request("/memory", { method: "DELETE" });
    if (!response.ok) throw new Error(`Remote memory clear failed: ${response.status}`);
  }

  async deleteMemory(options: MemoryDeleteOptions): Promise<number> {
    const response = await this.request("/memory/query", {
      method: "DELETE",
      body: JSON.stringify({ options })
    });
    if (!response.ok) throw new Error(`Remote memory delete failed: ${response.status}`);
    const data = (await response.json()) as { count?: number };
    return data.count ?? 0;
  }

  async listMemoryChunks(options: { sessionId?: string; limit?: number; tenantId?: string; cellId?: string } = {}): Promise<MemoryChunk[]> {
    const params = new URLSearchParams();
    if (options.sessionId) params.set("sessionId", options.sessionId);
    if (options.tenantId) params.set("tenantId", options.tenantId);
    if (options.cellId) params.set("cellId", options.cellId);
    if (options.limit !== undefined) params.set("limit", String(options.limit));
    const suffix = params.size > 0 ? `?${params.toString()}` : "";
    const response = await this.request(`/memory/export${suffix}`, { method: "GET" });
    if (!response.ok) throw new Error(`Remote memory export failed: ${response.status}`);
    const data = (await response.json()) as { chunks?: MemoryChunk[] };
    return data.chunks ?? [];
  }

  async importMemoryChunks(chunks: MemoryChunk[]): Promise<void> {
    const response = await this.request("/memory/import", {
      method: "POST",
      body: JSON.stringify({ chunks })
    });
    if (!response.ok) throw new Error(`Remote memory import failed: ${response.status}`);
  }

  async writeRuntimeTrace(trace: RuntimeTrace): Promise<void> {
    const response = await this.request("/runtime/traces", {
      method: "POST",
      body: JSON.stringify({ trace })
    });
    if (!response.ok) throw new Error(`Remote runtime trace write failed: ${response.status}`);
  }

  async listRuntimeTraces(options: RuntimeTraceListOptions = {}): Promise<RuntimeTrace[]> {
    const params = new URLSearchParams();
    if (options.sessionId) params.set("sessionId", options.sessionId);
    if (options.tenantId) params.set("tenantId", options.tenantId);
    if (options.cellId) params.set("cellId", options.cellId);
    if (options.limit !== undefined) params.set("limit", String(options.limit));
    const suffix = params.size > 0 ? `?${params.toString()}` : "";
    const response = await this.request(`/runtime/traces${suffix}`, { method: "GET" });
    if (!response.ok) throw new Error(`Remote runtime trace read failed: ${response.status}`);
    const data = (await response.json()) as { traces?: RuntimeTrace[] };
    return data.traces ?? [];
  }

  async importRuntimeTraces(traces: RuntimeTrace[]): Promise<void> {
    for (const trace of traces) {
      await this.writeRuntimeTrace(trace);
    }
  }

  async writeRawMemory(records: RawMemoryRecord[]): Promise<GacWriteResult> {
    return this.writeGacRecords("/gac/raw-memory", records);
  }

  async listRawMemory(options: GacListOptions = {}): Promise<RawMemoryRecord[]> {
    return this.listGacRecords<RawMemoryRecord>("/gac/raw-memory", options);
  }

  async writeIdentityPins(records: IdentityPinRecord[]): Promise<GacWriteResult> {
    return this.writeGacRecords("/gac/identity-pins", records);
  }

  async listIdentityPins(options: GacListOptions = {}): Promise<IdentityPinRecord[]> {
    return this.listGacRecords<IdentityPinRecord>("/gac/identity-pins", options);
  }

  async writeMemoryClusters(records: MemoryClusterRecord[]): Promise<GacWriteResult> {
    return this.writeGacRecords("/gac/clusters", records);
  }

  async listMemoryClusters(options: GacListOptions = {}): Promise<MemoryClusterRecord[]> {
    return this.listGacRecords<MemoryClusterRecord>("/gac/clusters", options);
  }

  async writeClusterMetrics(records: ClusterMetricRecord[]): Promise<GacWriteResult> {
    return this.writeGacRecords("/gac/cluster-metrics", records);
  }

  async listClusterMetrics(options: GacListOptions = {}): Promise<ClusterMetricRecord[]> {
    return this.listGacRecords<ClusterMetricRecord>("/gac/cluster-metrics", options);
  }

  async writeMemoryRepresentatives(records: MemoryRepresentativeRecord[], lineage?: MemoryLineageRecord[]): Promise<GacWriteResult> {
    const response = await this.request("/gac/representatives", {
      method: "POST",
      body: JSON.stringify({ records, ...(lineage ? { lineage } : {}) })
    });
    if (!response.ok) throw new Error(`Remote GAC representatives write failed: ${response.status}`);
    return response.json() as Promise<GacWriteResult>;
  }

  async listMemoryRepresentatives(options: GacListOptions = {}): Promise<MemoryRepresentativeRecord[]> {
    return this.listGacRecords<MemoryRepresentativeRecord>("/gac/representatives", options);
  }

  async writeMemoryLineage(records: MemoryLineageRecord[]): Promise<GacWriteResult> {
    return this.writeGacRecords("/gac/lineage", records);
  }

  async listMemoryLineage(options: GacListOptions = {}): Promise<MemoryLineageRecord[]> {
    return this.listGacRecords<MemoryLineageRecord>("/gac/lineage", options);
  }

  async writeConsolidationRuns(records: ConsolidationRunRecord[]): Promise<GacWriteResult> {
    return this.writeGacRecords("/gac/consolidation-runs", records);
  }

  async listConsolidationRuns(options: GacListOptions = {}): Promise<ConsolidationRunRecord[]> {
    return this.listGacRecords<ConsolidationRunRecord>("/gac/consolidation-runs", options);
  }

  async writeRetrievalAudits(records: RetrievalAuditRecord[]): Promise<GacWriteResult> {
    return this.writeGacRecords("/gac/retrieval-audits", records);
  }

  async listRetrievalAudits(options: GacListOptions = {}): Promise<RetrievalAuditRecord[]> {
    return this.listGacRecords<RetrievalAuditRecord>("/gac/retrieval-audits", options);
  }

  async writeContextPackTraces(records: ContextPackTraceRecord[]): Promise<GacWriteResult> {
    return this.writeGacRecords("/gac/context-pack-traces", records);
  }

  async listContextPackTraces(options: GacListOptions = {}): Promise<ContextPackTraceRecord[]> {
    const response = await this.request(`/gac/context-pack-traces${toQueryString(options)}`, { method: "GET" });
    if (!response.ok) throw new Error(`Remote GAC context pack trace read failed: ${response.status}`);
    const data = (await response.json()) as { traces?: ContextPackTraceRecord[] };
    return data.traces ?? [];
  }

  async writeModelMemoryActions(records: ModelMemoryActionRecord[]): Promise<GacWriteResult> {
    return this.writeGacRecords("/gac/model-memory-actions", records);
  }

  async listModelMemoryActions(options: GacListOptions = {}): Promise<ModelMemoryActionRecord[]> {
    return this.listGacRecords<ModelMemoryActionRecord>("/gac/model-memory-actions", options);
  }

  async writeMemoryContradictions(records: MemoryContradictionRecord[]): Promise<GacWriteResult> {
    return this.writeGacRecords("/gac/contradictions", records);
  }

  async listMemoryContradictions(options: GacListOptions = {}): Promise<MemoryContradictionRecord[]> {
    return this.listGacRecords<MemoryContradictionRecord>("/gac/contradictions", options);
  }

  async writeSourceDocuments(records: SourceDocumentRecord[]): Promise<GacWriteResult> {
    return this.writeGacRecords("/gac/source-documents", records);
  }

  async listSourceDocuments(options: GacListOptions = {}): Promise<SourceDocumentRecord[]> {
    return this.listGacRecords<SourceDocumentRecord>("/gac/source-documents", options);
  }

  async writeTrainingExamples(records: TrainingExampleRecord[]): Promise<GacWriteResult> {
    return this.writeGacRecords("/gac/training-examples", records);
  }

  async listTrainingExamples(options: GacListOptions = {}): Promise<TrainingExampleRecord[]> {
    return this.listGacRecords<TrainingExampleRecord>("/gac/training-examples", options);
  }

  private request(path: string, init: RequestInit): Promise<Response> {
    const headers = toHeaderRecord(init.headers);
    if (init.body !== undefined && init.body !== null && !hasHeader(headers, "Content-Type")) {
      headers["Content-Type"] = "application/json";
    }
    return fetch(`${this.baseUrl}${path}`, {
      ...init,
      credentials: this.options.credentials ?? "same-origin",
      headers: {
        ...(this.options.token ? { Authorization: `Bearer ${this.options.token}` } : {}),
        ...(this.options.tenantId ? { "X-Edge-Agent-Tenant": this.options.tenantId } : {}),
        ...(this.options.cellId ? { "X-Edge-Agent-Cell": this.options.cellId } : {}),
        ...headers
      }
    });
  }

  private async writeGacRecords<T>(path: string, records: T[]): Promise<GacWriteResult> {
    const response = await this.request(path, {
      method: "POST",
      body: JSON.stringify({ records })
    });
    if (!response.ok) throw new Error(`Remote GAC write failed for ${path}: ${response.status}`);
    return response.json() as Promise<GacWriteResult>;
  }

  private async listGacRecords<T>(path: string, options: GacListOptions): Promise<T[]> {
    const response = await this.request(`${path}${toQueryString(options)}`, { method: "GET" });
    if (!response.ok) throw new Error(`Remote GAC read failed for ${path}: ${response.status}`);
    const data = (await response.json()) as { records?: T[] };
    return data.records ?? [];
  }
}

function toHeaderRecord(headers: HeadersInit | undefined): Record<string, string> {
  if (!headers) return {};
  if (headers instanceof Headers) return Object.fromEntries(headers.entries());
  if (Array.isArray(headers)) return Object.fromEntries(headers);
  return { ...headers };
}

function hasHeader(headers: Record<string, string>, name: string): boolean {
  return Object.keys(headers).some((key) => key.toLowerCase() === name.toLowerCase());
}

function toQueryString(options: GacListOptions): string {
  const params = new URLSearchParams();
  if (options.tenantId) params.set("tenantId", options.tenantId);
  if (options.cellId) params.set("cellId", options.cellId);
  if (options.sessionId) params.set("sessionId", options.sessionId);
  if (options.rawMemoryId) params.set("rawMemoryId", options.rawMemoryId);
  if (options.representativeId) params.set("representativeId", options.representativeId);
  if (options.contextPackId) params.set("contextPackId", options.contextPackId);
  if (options.limit !== undefined) params.set("limit", String(options.limit));
  return params.size > 0 ? `?${params.toString()}` : "";
}
