import { afterEach, describe, expect, it, vi } from "vitest";
import { IndexedDbMemoryStore } from "./indexedDbMemoryStore";
import { createMemoryStore } from "./hybridMemoryClient";

vi.mock("idb", () => ({
  openDB: vi.fn(async () => ({
    transaction: vi.fn(() => ({ store: { put: vi.fn() }, done: Promise.resolve() })),
  })),
}));

describe("createMemoryStore remote-http", () => {
  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("uses session credentials without forwarding a browser-bundled bearer token", async () => {
    const fetchMock = vi.fn(async () => ({
      ok: true,
      status: 200,
      headers: new Headers({ "Content-Type": "application/json" }),
      json: async () => ({ ok: true, mode: "remote-http" }),
    } as Response));
    vi.stubGlobal("fetch", fetchMock);

    const result = await createMemoryStore({
      provider: "remote-http",
      allowFallback: false,
      remoteUrl: "/api/edge-ai",
      remoteCredentials: "same-origin",
      remoteTenantId: "tenant_1",
      remoteCellId: "cell_1",
      useSidecar: false,
      sidecarUrl: "http://127.0.0.1:8787",
    });

    expect(result.mode).toBe("remote-http");
    const call = getFetchCall(fetchMock, 0);
    expect(call[0]).toBe("/api/edge-ai/health");
    expect(call[1].credentials).toBe("same-origin");
    expect(call[1].headers).toMatchObject({
      "X-Edge-Agent-Tenant": "tenant_1",
      "X-Edge-Agent-Cell": "cell_1",
    });
    expect(call[1].headers).not.toHaveProperty("Authorization");
  });

  it("can forward a dev-only remote bearer token for secured local testing", async () => {
    const fetchMock = vi.fn(async () => ({
      ok: true,
      status: 200,
      headers: new Headers({ "Content-Type": "application/json" }),
      json: async () => ({ ok: true, mode: "remote-http" }),
    } as Response));
    vi.stubGlobal("fetch", fetchMock);

    const result = await createMemoryStore({
      provider: "remote-http",
      allowFallback: false,
      remoteUrl: "http://127.0.0.1:8787/api/edge-ai",
      remoteToken: "dev-token",
      remoteCredentials: "omit",
      remoteTenantId: "tenant_1",
      remoteCellId: "cell_1",
      useSidecar: false,
      sidecarUrl: "http://127.0.0.1:8787",
    });

    expect(result.mode).toBe("remote-http");
    const call = getFetchCall(fetchMock, 0);
    expect(call[1].credentials).toBe("omit");
    expect(call[1].headers).toMatchObject({
      Authorization: "Bearer dev-token",
      "X-Edge-Agent-Tenant": "tenant_1",
      "X-Edge-Agent-Cell": "cell_1",
    });
  });

  it("uses browser-vector as the named IndexedDB-backed local provider", async () => {
    const result = await createMemoryStore({
      provider: "browser-vector",
      allowFallback: false,
      remoteUrl: "",
      useSidecar: false,
      sidecarUrl: "http://127.0.0.1:8787",
    });

    expect(result.mode).toBe("browser-vector");
    expect(result.store).toBeInstanceOf(IndexedDbMemoryStore);
    expect(result.capabilities).toMatchObject({
      mode: "browser-vector",
      storage: "indexeddb",
      localOnly: true,
      vectorSearch: true,
      deterministicSearch: true,
      metadataFilters: true,
      vectorDimension: 384,
      persistent: true,
      importExport: true,
      contextPackTracePersistence: true,
      remoteSync: false,
    });
  });

  it("keeps indexeddb as a compatibility alias for the same browser store", async () => {
    const result = await createMemoryStore({
      provider: "indexeddb",
      allowFallback: false,
      remoteUrl: "",
      useSidecar: false,
      sidecarUrl: "http://127.0.0.1:8787",
    });

    expect(result.mode).toBe("indexeddb");
    expect(result.store).toBeInstanceOf(IndexedDbMemoryStore);
    expect(result.capabilities.storage).toBe("indexeddb");
  });

  it("falls back from an unavailable remote provider to browser-vector unless fallback is explicitly disabled", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => ({ ok: false, status: 503 } as Response)));
    const warn = vi.spyOn(console, "warn").mockImplementation(() => undefined);

    const result = await createMemoryStore({
      provider: "remote-http",
      remoteUrl: "/api/edge-ai",
      useSidecar: false,
      sidecarUrl: "http://127.0.0.1:8787",
    });

    expect(result.mode).toBe("browser-vector");
    expect(result.store).toBeInstanceOf(IndexedDbMemoryStore);
    expect(result.capabilities.vectorDimension).toBe(384);
    expect(warn).toHaveBeenCalledWith("Remote memory provider unavailable. Falling back to browser-vector because fallback is enabled.");
  });

  it("falls back from an unavailable explicit sidecar provider to browser-vector unless fallback is explicitly disabled", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => ({ ok: false, status: 503 } as Response)));
    const warn = vi.spyOn(console, "warn").mockImplementation(() => undefined);

    const result = await createMemoryStore({
      provider: "sidecar",
      remoteUrl: "",
      useSidecar: false,
      sidecarUrl: "http://127.0.0.1:8787",
    });

    expect(result.mode).toBe("browser-vector");
    expect(result.store).toBeInstanceOf(IndexedDbMemoryStore);
    expect(result.capabilities).toMatchObject({
      mode: "browser-vector",
      storage: "indexeddb",
      vectorDimension: 384,
      remoteSync: false,
    });
    expect(warn).toHaveBeenCalledWith("LanceDB sidecar requested but unavailable. Falling back to browser-vector.");
  });

  it("still lets operators disable remote fallback for hard-fail deployments", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => ({ ok: false, status: 503 } as Response)));

    await expect(createMemoryStore({
      provider: "remote-http",
      allowFallback: false,
      remoteUrl: "/api/edge-ai",
      useSidecar: false,
      sidecarUrl: "http://127.0.0.1:8787",
    })).rejects.toThrow("Remote memory provider is unavailable and fallback is disabled.");
  });

  it("still lets operators disable sidecar fallback for hard-fail deployments", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => ({ ok: false, status: 503 } as Response)));

    await expect(createMemoryStore({
      provider: "sidecar",
      allowFallback: false,
      remoteUrl: "",
      useSidecar: false,
      sidecarUrl: "http://127.0.0.1:8787",
    })).rejects.toThrow("Memory sidecar is unavailable and fallback is disabled.");
  });

  it("hard-fails legacy optional sidecar mode when fallback is explicitly disabled", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => ({ ok: false, status: 503 } as Response)));

    await expect(createMemoryStore({
      provider: "browser-vector",
      allowFallback: false,
      remoteUrl: "",
      useSidecar: true,
      sidecarUrl: "http://127.0.0.1:8787",
    })).rejects.toThrow("Memory sidecar is unavailable and fallback is disabled.");
  });
});

function getFetchCall(fetchMock: ReturnType<typeof vi.fn>, index: number): [string, RequestInit] {
  const calls = fetchMock.mock.calls as unknown as Array<[string, RequestInit]>;
  const call = calls[index];
  if (!call || typeof call[0] !== "string") throw new Error(`Missing fetch call ${index}`);
  return call;
}
