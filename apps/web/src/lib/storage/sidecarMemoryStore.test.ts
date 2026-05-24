import type { MemoryChunk } from "@infinite-edge-agent/core";
import { afterEach, describe, expect, it, vi } from "vitest";
import { SidecarMemoryStore } from "./sidecarMemoryStore";

const chunk: MemoryChunk = {
  id: "chunk_1",
  text: "Sidecar memory",
  embedding: [0.1, 0.2],
  sessionId: "session_1",
  source: "chat",
  role: "user",
  createdAt: "2026-05-11T00:00:00.000Z",
  updatedAt: "2026-05-11T00:00:00.000Z",
  tags: ["user"],
  metadata: {},
  tokenCount: 2,
};

describe("SidecarMemoryStore request headers", () => {
  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("does not send JSON content-type for empty-body health requests", async () => {
    const fetchMock = mockFetch({ ok: true });
    const store = new SidecarMemoryStore("http://127.0.0.1:8787");

    await expect(store.health()).resolves.toBe(true);

    const call = getFetchCall(fetchMock, 0);
    expect(call[0]).toBe("http://127.0.0.1:8787/health");
    expect(call[1].method).toBe("GET");
    expect(call[1].body).toBeUndefined();
    expect(call[1].headers).not.toHaveProperty("Content-Type");
  });

  it("does not send JSON content-type for empty-body clear requests", async () => {
    const fetchMock = mockFetch({ ok: true });
    const store = new SidecarMemoryStore("http://127.0.0.1:8787");

    await store.clear();

    const call = getFetchCall(fetchMock, 0);
    expect(call[0]).toBe("http://127.0.0.1:8787/memory");
    expect(call[1].method).toBe("DELETE");
    expect(call[1].body).toBeUndefined();
    expect(call[1].headers).not.toHaveProperty("Content-Type");
  });

  it("sends JSON content-type when a body is present", async () => {
    const fetchMock = mockFetch({ ok: true });
    const store = new SidecarMemoryStore("http://127.0.0.1:8787");

    await store.upsert([chunk]);

    const call = getFetchCall(fetchMock, 0);
    expect(call[0]).toBe("http://127.0.0.1:8787/memory/upsert");
    expect(call[1].headers).toMatchObject({ "Content-Type": "application/json" });
  });

  it("passes tenant and cell scope when exporting memory rows", async () => {
    const fetchMock = mockFetch({ ok: true, body: { chunks: [] } });
    const store = new SidecarMemoryStore("http://127.0.0.1:8787");

    await store.listMemoryChunks({ tenantId: "tenant_1", cellId: "cell_1", sessionId: "session_1", limit: 5 });

    const call = getFetchCall(fetchMock, 0);
    expect(call[0]).toBe("http://127.0.0.1:8787/memory/export?sessionId=session_1&tenantId=tenant_1&cellId=cell_1&limit=5");
  });
});

function mockFetch(response: { ok: boolean; body?: unknown }) {
  const fetchMock = vi.fn(async () => ({
    ok: response.ok,
    status: response.ok ? 200 : 500,
    json: async () => response.body ?? {},
  } as Response));
  vi.stubGlobal("fetch", fetchMock);
  return fetchMock;
}

function getFetchCall(fetchMock: ReturnType<typeof mockFetch>, index: number): [string, RequestInit] {
  const calls = fetchMock.mock.calls as unknown as Array<[string, RequestInit]>;
  const call = calls[index];
  if (!call || typeof call[0] !== "string") throw new Error(`Missing fetch call ${index}`);
  return call;
}
