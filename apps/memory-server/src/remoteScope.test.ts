import { describe, expect, it } from "vitest";
import {
  filterMemoryChunksForScope,
  hasConfiguredRemoteSecurity,
  resolveRemoteScope,
  scopedClearOptions,
  scopedDeleteOptions,
  scopedGacListOptions,
  scopedGacRecords,
  scopedRuntimeTrace,
  scopedRuntimeTraceListOptions,
  scopedSearchOptions,
  stampMemoryChunksForScope
} from "./remoteScope";

describe("remote GAC scope binding", () => {
  it("requires token, tenant, and cell configuration before remote routes can serve scoped data", () => {
    expect(hasConfiguredRemoteSecurity({
      token: "token",
      tenantId: "tenant_header",
      cellId: "cell_header"
    })).toBe(true);
    expect(hasConfiguredRemoteSecurity({ token: "token", tenantId: "tenant_header" })).toBe(false);
    expect(hasConfiguredRemoteSecurity({ tenantId: "tenant_header", cellId: "cell_header" })).toBe(false);
  });

  it("derives scope from authenticated tenant and cell headers", () => {
    expect(resolveRemoteScope({
      tenant: "tenant_header",
      cell: "cell_header"
    })).toEqual({
      tenantId: "tenant_header",
      cellId: "cell_header"
    });
  });

  it("rejects records that try to write outside the authenticated scope", () => {
    expect(() => scopedGacRecords([
      { id: "trace_1", tenantId: "other_tenant", cellId: "cell_header" }
    ], {
      tenantId: "tenant_header",
      cellId: "cell_header"
    })).toThrow("GAC_SCOPE_MISMATCH");
  });

  it("binds list queries to authenticated scope and rejects escape filters", () => {
    expect(scopedGacListOptions({ sessionId: "session_1" }, {
      tenantId: "tenant_header",
      cellId: "cell_header"
    })).toMatchObject({
      tenantId: "tenant_header",
      cellId: "cell_header",
      sessionId: "session_1"
    });

    expect(() => scopedGacListOptions({ tenantId: "other_tenant" }, {
      tenantId: "tenant_header",
      cellId: "cell_header"
    })).toThrow("GAC_SCOPE_MISMATCH");
  });

  it("stamps remote memory chunks with reserved scope metadata and tags", () => {
    const [chunk] = stampMemoryChunksForScope([{
      ...makeChunk("chunk_1", {}),
      tags: ["user", "edge-tenant:forged", "edge-cell:forged"]
    }], {
      tenantId: "tenant_header",
      cellId: "cell_header"
    });

    expect(chunk).toMatchObject({
      metadata: {
        edgeTenantId: "tenant_header",
        edgeCellId: "cell_header"
      },
      tags: expect.arrayContaining(["edge-tenant:tenant_header", "edge-cell:cell_header"])
    });
    expect(chunk?.tags).not.toContain("edge-tenant:forged");
    expect(chunk?.tags).not.toContain("edge-cell:forged");
  });

  it("adds trusted scope fields to remote search/delete/clear operations", () => {
    const scope = { tenantId: "tenant_header", cellId: "cell_header" };

    expect(scopedSearchOptions({ limit: 2, tags: ["user", "edge-tenant:forged"] }, scope)).toMatchObject({
      limit: 20,
      tags: ["user"],
      tenantId: "tenant_header",
      cellId: "cell_header"
    });
    expect(scopedDeleteOptions({ sessionId: "session_1" }, scope)).toEqual({
      sessionId: "session_1",
      tags: [],
      tenantId: "tenant_header",
      cellId: "cell_header"
    });
    expect(scopedClearOptions(scope)).toEqual({
      tenantId: "tenant_header",
      cellId: "cell_header"
    });
  });

  it("filters exported remote chunks to authenticated scope", () => {
    expect(filterMemoryChunksForScope([
      makeChunk("match", { edgeTenantId: "tenant_header", edgeCellId: "cell_header" }),
      makeChunk("wrong_tenant", { edgeTenantId: "other", edgeCellId: "cell_header" }),
      makeChunk("wrong_cell", { edgeTenantId: "tenant_header", edgeCellId: "other" })
    ], {
      tenantId: "tenant_header",
      cellId: "cell_header"
    })).toEqual([
      expect.objectContaining({ id: "match" })
    ]);
  });

  it("stamps runtime trace writes and rejects forged runtime trace scope", () => {
    const scope = { tenantId: "tenant_header", cellId: "cell_header" };

    expect(scopedRuntimeTrace(makeTrace({}), scope)).toMatchObject({
      tenantId: "tenant_header",
      cellId: "cell_header"
    });
    expect(() => scopedRuntimeTrace(makeTrace({ tenantId: "other_tenant" }), scope)).toThrow("RUNTIME_TRACE_SCOPE_MISMATCH");
    expect(() => scopedRuntimeTrace(makeTrace({ cellId: "other_cell" }), scope)).toThrow("RUNTIME_TRACE_SCOPE_MISMATCH");
  });

  it("binds runtime trace list queries to authenticated scope and rejects escape filters", () => {
    const scope = { tenantId: "tenant_header", cellId: "cell_header" };

    expect(scopedRuntimeTraceListOptions({ sessionId: "session_1", limit: 5 }, scope)).toEqual({
      sessionId: "session_1",
      limit: 5,
      tenantId: "tenant_header",
      cellId: "cell_header"
    });
    expect(() => scopedRuntimeTraceListOptions({ sessionId: "session_1", tenantId: "other_tenant" }, scope)).toThrow("RUNTIME_TRACE_SCOPE_MISMATCH");
  });
});

function makeChunk(id: string, metadata: Record<string, unknown>) {
  return {
    id,
    text: id,
    embedding: [0.1, 0.2],
    sessionId: "session_1",
    source: "chat" as const,
    role: "user" as const,
    createdAt: "2026-05-11T00:00:00.000Z",
    updatedAt: "2026-05-11T00:00:00.000Z",
    tags: ["user"],
    metadata,
    tokenCount: 1,
  };
}

function makeTrace(scope: { tenantId?: string; cellId?: string }) {
  return {
    traceId: "trace_1",
    requestId: "request_1",
    sessionId: "session_1",
    modelId: "model_1",
    backend: "backend_1",
    createdAt: "2026-05-11T00:00:00.000Z",
    runtime: {},
    ...scope
  };
}
