import type { GacListOptions, MemoryChunk, MemoryDeleteOptions, MemorySearchOptions, RuntimeTrace, RuntimeTraceListOptions } from "@infinite-edge-agent/core";

export interface RemoteScope {
  tenantId: string;
  cellId: string;
}

export function hasConfiguredRemoteSecurity(config: { token?: string | undefined; tenantId?: string | undefined; cellId?: string | undefined }): boolean {
  return Boolean(config.token && config.tenantId && config.cellId);
}

export function resolveRemoteScope(
  headers: { tenant?: string | string[] | undefined; cell?: string | string[] | undefined },
  expected: { tenantId?: string; cellId?: string } = {}
): RemoteScope | null {
  const tenantId = expected.tenantId ?? firstHeader(headers.tenant);
  const cellId = expected.cellId ?? firstHeader(headers.cell);
  if (!tenantId || !cellId) return null;
  return { tenantId, cellId };
}

export function scopedGacRecords<T extends { tenantId: string; cellId: string }>(
  records: T[],
  scope: RemoteScope
): T[] {
  return records.map((record) => {
    assertScopeMatch(record.tenantId, scope.tenantId, "tenantId");
    assertScopeMatch(record.cellId, scope.cellId, "cellId");
    return { ...record, tenantId: scope.tenantId, cellId: scope.cellId };
  });
}

export function scopedGacListOptions(options: GacListOptions, scope: RemoteScope): GacListOptions {
  assertScopeMatch(options.tenantId, scope.tenantId, "tenantId");
  assertScopeMatch(options.cellId, scope.cellId, "cellId");
  return {
    ...options,
    tenantId: scope.tenantId,
    cellId: scope.cellId
  };
}

export function scopedRuntimeTrace(trace: RuntimeTrace, scope: RemoteScope): RuntimeTrace {
  assertScopeMatch(trace.tenantId, scope.tenantId, "tenantId", "RUNTIME_TRACE_SCOPE_MISMATCH");
  assertScopeMatch(trace.cellId, scope.cellId, "cellId", "RUNTIME_TRACE_SCOPE_MISMATCH");
  return {
    ...trace,
    tenantId: scope.tenantId,
    cellId: scope.cellId
  };
}

export function scopedRuntimeTraceListOptions(options: RuntimeTraceListOptions, scope: RemoteScope): RuntimeTraceListOptions {
  assertScopeMatch(options.tenantId, scope.tenantId, "tenantId", "RUNTIME_TRACE_SCOPE_MISMATCH");
  assertScopeMatch(options.cellId, scope.cellId, "cellId", "RUNTIME_TRACE_SCOPE_MISMATCH");
  return {
    ...options,
    tenantId: scope.tenantId,
    cellId: scope.cellId
  };
}

export function stampMemoryChunksForScope(chunks: MemoryChunk[], scope: RemoteScope): MemoryChunk[] {
  return chunks.map((chunk) => ({
    ...chunk,
    tags: withScopeTags(stripReservedScopeTags(chunk.tags), scope),
    metadata: {
      ...chunk.metadata,
      edgeTenantId: scope.tenantId,
      edgeCellId: scope.cellId
    }
  }));
}

export function scopedSearchOptions(options: MemorySearchOptions | undefined, scope: RemoteScope): MemorySearchOptions {
  const limit = options?.limit ?? 8;
  return {
    ...options,
    limit: Math.max(limit * 10, limit),
    tags: stripReservedScopeTags(options?.tags ?? []),
    tenantId: scope.tenantId,
    cellId: scope.cellId
  };
}

export function scopedDeleteOptions(options: MemoryDeleteOptions, scope: RemoteScope): MemoryDeleteOptions {
  return {
    ...options,
    tags: stripReservedScopeTags(options.tags ?? []),
    tenantId: scope.tenantId,
    cellId: scope.cellId
  };
}

export function scopedClearOptions(scope: RemoteScope): MemoryDeleteOptions {
  return {
    tenantId: scope.tenantId,
    cellId: scope.cellId
  };
}

export function filterMemoryChunksForScope<T extends MemoryChunk>(chunks: T[], scope: RemoteScope, limit?: number): T[] {
  return chunks
    .filter((chunk) => chunk.metadata.edgeTenantId === scope.tenantId && chunk.metadata.edgeCellId === scope.cellId)
    .slice(0, limit ?? chunks.length);
}

export function restoreRequestedSearchLimit<T>(hits: T[], requestedLimit: number | undefined): T[] {
  return hits.slice(0, requestedLimit ?? 8);
}

function assertScopeMatch(actual: string | undefined, expected: string, field: "tenantId" | "cellId", code = "GAC_SCOPE_MISMATCH"): void {
  if (actual !== undefined && actual !== expected) {
    throw new Error(`${code}: ${field} must match authenticated scope.`);
  }
}

function withScopeTags(tags: string[], scope: RemoteScope): string[] {
  return [...new Set([...tags, ...scopeTags(scope)])];
}

function stripReservedScopeTags(tags: string[]): string[] {
  return tags.filter((tag) => !tag.startsWith("edge-tenant:") && !tag.startsWith("edge-cell:"));
}

function scopeTags(scope: RemoteScope): string[] {
  return [`edge-tenant:${scope.tenantId}`, `edge-cell:${scope.cellId}`];
}

function firstHeader(value: string | string[] | undefined): string | undefined {
  if (Array.isArray(value)) return value[0];
  return value;
}
