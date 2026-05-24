import type { ContextPackTraceRecord, MemoryChunk, RuntimeTrace } from "@infinite-edge-agent/core";
import { afterEach, describe, expect, it, vi } from "vitest";
import { RemoteMemoryStore } from "./remoteMemoryStore";

const chunk: MemoryChunk = {
  id: "chunk_1",
  text: "Remember Qwen unlocked uses a configurable remote memory API.",
  embedding: [0.1, 0.2],
  sessionId: "session_1",
  source: "chat",
  role: "user",
  createdAt: "2026-05-11T00:00:00.000Z",
  updatedAt: "2026-05-11T00:00:00.000Z",
  tags: ["user"],
  metadata: {},
  tokenCount: 8,
};

const trace: RuntimeTrace = {
  traceId: "trace_1",
  requestId: "request_1",
  sessionId: "session_1",
  modelId: "Qwen/Qwen3-0.6B",
  backend: "unlocked-browser-transformer",
  createdAt: "2026-05-11T00:00:00.000Z",
  runtime: { ok: true },
};

const contextPackTrace: ContextPackTraceRecord = {
  id: "pack_trace_1",
  traceId: "trace_1",
  tenantId: "tenant_1",
  cellId: "cell_1",
  sessionId: "session_1",
  queryId: "request_1",
  contextPackId: "pack_1",
  rawMemoryIds: ["chunk_1"],
  representativeIds: [],
  identityPinIds: [],
  tokenBudget: 4096,
  estimatedTokens: 512,
  packingStrategy: "advanced-runtime",
  includedMemoryIds: ["chunk_1"],
  omittedMemoryIds: [],
  createdAt: "2026-05-11T00:00:00.000Z",
};

describe("RemoteMemoryStore", () => {
  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("checks health with tenant, cell, and bearer headers", async () => {
    const fetchMock = mockFetch({ ok: true, body: { ok: true, mode: "remote-http" } });
    const store = makeStore();

    await expect(store.health()).resolves.toBe(true);

    expect(fetchMock).toHaveBeenCalledWith("https://example.test/api/edge-ai/health", expect.objectContaining({
      method: "GET",
      credentials: "same-origin",
      headers: expect.objectContaining({
        Authorization: "Bearer test-token",
        "X-Edge-Agent-Tenant": "tenant_1",
        "X-Edge-Agent-Cell": "cell_1",
      }),
    }));
    expect(getHeaders(fetchMock, 0)).not.toHaveProperty("Content-Type");
  });

  it("rejects app-shell HTML health responses from static hosts", async () => {
    mockFetch({
      ok: true,
      body: "<html>app shell</html>",
      contentType: "text/html; charset=utf-8",
    });
    const store = makeStore();

    await expect(store.health()).resolves.toBe(false);
  });

  it("rejects malformed or wrong-mode remote health payloads", async () => {
    mockFetchSequence([
      { ok: true, body: { ok: true, mode: "lancedb-sidecar" } },
      { ok: true, body: { ready: true } },
      { ok: true, body: "not-json", contentType: "application/json" },
    ]);
    const store = makeStore();

    await expect(store.health()).resolves.toBe(false);
    await expect(store.health()).resolves.toBe(false);
    await expect(store.health()).resolves.toBe(false);
  });

  it("upserts chunks using the remote memory contract", async () => {
    const fetchMock = mockFetch({ ok: true, body: { ok: true, mode: "remote-http" } });
    const store = makeStore();

    await store.upsert([chunk]);

    const firstCall = getFetchCall(fetchMock, 0);
    const request = firstCall[1];
    expect(firstCall[0]).toBe("https://example.test/api/edge-ai/memory/upsert");
    expect(firstCall[1].headers).toMatchObject({ "Content-Type": "application/json" });
    expect(JSON.parse(String(request.body))).toEqual({ chunks: [chunk] });
  });

  it("can opt into credentialed proxy/session requests without a bearer token", async () => {
    const fetchMock = mockFetch({ ok: true, body: { ok: true, mode: "remote-http" } });
    const store = new RemoteMemoryStore({
      baseUrl: "https://example.test/api/edge-ai",
      credentials: "include",
      tenantId: "tenant_1",
      cellId: "cell_1",
    });

    await expect(store.health()).resolves.toBe(true);

    const firstCall = getFetchCall(fetchMock, 0);
    expect(firstCall[1].credentials).toBe("include");
    expect(getHeaders(fetchMock, 0)).not.toHaveProperty("Authorization");
  });

  it("clears memory without sending an empty JSON content-type", async () => {
    const fetchMock = mockFetch({ ok: true });
    const store = makeStore();

    await store.clear();

    const firstCall = getFetchCall(fetchMock, 0);
    expect(firstCall[0]).toBe("https://example.test/api/edge-ai/memory");
    expect(firstCall[1].method).toBe("DELETE");
    expect(firstCall[1].body).toBeUndefined();
    expect(getHeaders(fetchMock, 0)).not.toHaveProperty("Content-Type");
  });

  it("parses search hits", async () => {
    mockFetch({ ok: true, body: { hits: [{ ...chunk, score: 0.92 }] } });
    const store = makeStore();

    const hits = await store.search([0.1, 0.2], { limit: 1 });

    expect(hits).toHaveLength(1);
    expect(hits[0]?.score).toBe(0.92);
  });

  it("persists and reads runtime traces", async () => {
    const fetchMock = mockFetchSequence([
      { ok: true },
      { ok: true, body: { traces: [trace] } },
    ]);
    const store = makeStore();

    await store.writeRuntimeTrace(trace);
    const traces = await store.listRuntimeTraces({ sessionId: "session_1", limit: 1 });

    const firstCall = getFetchCall(fetchMock, 0);
    const secondCall = getFetchCall(fetchMock, 1);
    expect(firstCall[0]).toBe("https://example.test/api/edge-ai/runtime/traces");
    expect(secondCall[0]).toBe("https://example.test/api/edge-ai/runtime/traces?sessionId=session_1&limit=1");
    expect(traces).toEqual([trace]);
  });

  it("exports and imports memory chunks", async () => {
    const fetchMock = mockFetchSequence([
      { ok: true, body: { chunks: [chunk] } },
      { ok: true },
    ]);
    const store = makeStore();

    const chunks = await store.listMemoryChunks({ sessionId: "session_1", tenantId: "tenant_1", cellId: "cell_1", limit: 1 });
    await store.importMemoryChunks(chunks);

    const firstCall = getFetchCall(fetchMock, 0);
    const secondCall = getFetchCall(fetchMock, 1);
    expect(firstCall[0]).toBe("https://example.test/api/edge-ai/memory/export?sessionId=session_1&tenantId=tenant_1&cellId=cell_1&limit=1");
    expect(secondCall[0]).toBe("https://example.test/api/edge-ai/memory/import");
    expect(JSON.parse(String(secondCall[1].body))).toEqual({ chunks: [chunk] });
  });

  it("deletes scoped memory through the remote query route", async () => {
    const fetchMock = mockFetch({ ok: true, body: { count: 3 } });
    const store = makeStore();

    await expect(store.deleteMemory({ sessionId: "session_1", tags: ["project:edge-ai"] })).resolves.toBe(3);

    const firstCall = getFetchCall(fetchMock, 0);
    expect(firstCall[0]).toBe("https://example.test/api/edge-ai/memory/query");
    expect(firstCall[1].method).toBe("DELETE");
    expect(firstCall[1].headers).toMatchObject({ "Content-Type": "application/json" });
    expect(JSON.parse(String(firstCall[1].body))).toEqual({
      options: { sessionId: "session_1", tags: ["project:edge-ai"] },
    });
  });

  it("persists and lists context pack traces through the GAC endpoint", async () => {
    const fetchMock = mockFetchSequence([
      { ok: true, body: { ok: true, count: 1, traceId: "trace_1" } },
      { ok: true, body: { traces: [contextPackTrace] } },
    ]);
    const store = makeStore();

    await expect(store.writeContextPackTraces([contextPackTrace])).resolves.toEqual({
      ok: true,
      count: 1,
      traceId: "trace_1",
    });
    await expect(store.listContextPackTraces({ sessionId: "session_1", contextPackId: "pack_1" })).resolves.toEqual([
      contextPackTrace,
    ]);

    const writeCall = getFetchCall(fetchMock, 0);
    const listCall = getFetchCall(fetchMock, 1);
    expect(writeCall[0]).toBe("https://example.test/api/edge-ai/gac/context-pack-traces");
    expect(JSON.parse(String(writeCall[1].body))).toEqual({ records: [contextPackTrace] });
    expect(listCall[0]).toBe("https://example.test/api/edge-ai/gac/context-pack-traces?sessionId=session_1&contextPackId=pack_1");
  });
});

function makeStore(): RemoteMemoryStore {
  return new RemoteMemoryStore({
    baseUrl: "https://example.test/api/edge-ai",
    token: "test-token",
    tenantId: "tenant_1",
    cellId: "cell_1",
  });
}

function getHeaders(fetchMock: ReturnType<typeof mockFetchSequence>, index: number): Record<string, string> {
  return getFetchCall(fetchMock, index)[1].headers as Record<string, string>;
}

function mockFetch(response: MockFetchResponse) {
  return mockFetchSequence([response]);
}

interface MockFetchResponse {
  ok: boolean;
  body?: unknown;
  contentType?: string;
}

function mockFetchSequence(responses: MockFetchResponse[]) {
  const fetchMock = vi.fn(async () => {
    const next = responses.shift() ?? { ok: true };
    return {
      ok: next.ok,
      status: next.ok ? 200 : 500,
      headers: new Headers({
        "content-type": next.contentType ?? "application/json",
      }),
      json: async () => {
        if (typeof next.body === "string") throw new SyntaxError("Unexpected token");
        return next.body ?? {};
      },
    } as Response;
  });
  vi.stubGlobal("fetch", fetchMock);
  return fetchMock;
}

function getFetchCall(fetchMock: ReturnType<typeof mockFetchSequence>, index: number): [string, RequestInit] {
  const calls = fetchMock.mock.calls as unknown as Array<[string, RequestInit]>;
  const call = calls[index];
  if (!call || typeof call[0] !== "string") throw new Error(`Missing fetch call ${index}`);
  return call;
}
