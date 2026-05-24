import { describe, expect, it } from "vitest";
import {
  createBrowserKVSwapPersistence,
  KV_SWAP_STORAGE_VERSION,
  MemoryKVSwapPersistence,
  type KVSwapPersistenceStore,
  type SerializedKVSwapBlock,
} from "./kvSwapPersistence";
import { MAX_KV_SWAP_BINARY_RECORD_BYTES } from "./kvSwapBinaryCodec";
import { handleKVSwapPersistenceWorkerMessage } from "../../workers/kvSwapPersistence.worker";

describe("KVSwap persistence", () => {
  it("round-trips serialized KV blocks with compressed key summaries", async () => {
    const store = new MemoryKVSwapPersistence({ namespace: "tenant:cell:session", maxBlocks: 8, maxBytes: 1024 * 1024 });
    const block = makeBlock("layer0:b0");

    const persist = await store.save(block);
    const loaded = await store.load("tenant:cell:session", "layer0:b0");
    const listed = await store.list("tenant:cell:session");
    const hydrated = await store.hydrate("tenant:cell:session");

    expect(persist).toMatchObject({ operation: "persist", mode: "memory", ok: true });
    expect(loaded.block).toMatchObject({ id: "layer0:b0", keyRows: [[1, 2], [3, 4]] });
    expect(listed.metadata).toEqual([
      expect.objectContaining({
        id: "layer0:b0",
        byteLength: 64,
        summaryRank: 2,
      }),
    ]);
    expect(hydrated.event).toMatchObject({ operation: "hydrate", ok: true });
    expect(hydrated.blocks).toEqual([
      expect.objectContaining({
        version: KV_SWAP_STORAGE_VERSION,
        id: "layer0:b0",
        keyRows: [[1, 2], [3, 4]],
        valueRows: [[5, 6], [7, 8]],
        compressedKeySummary: [0.25, 0.75],
        summaryRank: 2,
      }),
    ]);
  });

  it("falls back to memory when OPFS and IndexedDB are unavailable", async () => {
    const store = await createBrowserKVSwapPersistence(
      { namespace: "fallback", preferOpfs: true },
      { navigator: { storage: {} as StorageManager } },
    );

    expect(store.mode).toBe("memory");
    expect(store.health()).toMatchObject({ enabled: true, mode: "memory", namespace: "fallback" });
  });

  it("falls back from denied OPFS to an IndexedDB adapter when IndexedDB is available", async () => {
    const indexedDbStore = makeIndexedDbTestStore("fallback-idb");
    const store = await createBrowserKVSwapPersistence(
      { namespace: "fallback-idb", preferOpfs: true },
      {
        navigator: {
          storage: {
            getDirectory: async () => {
              throw new Error("opfs denied");
            },
          } as unknown as StorageManager,
        },
        indexedDB: {} as IDBFactory,
        createIndexedDbPersistence: async () => indexedDbStore,
      },
    );

    expect(store).toBe(indexedDbStore);
    expect(store.mode).toBe("indexeddb");
  });

  it("uses OPFS when available before falling back", async () => {
    const directory = new FakeOpfsDirectory();
    const store = await createBrowserKVSwapPersistence(
      { namespace: "opfs-session", preferOpfs: true },
      {
        navigator: {
          storage: {
            getDirectory: async () => ({
              getDirectoryHandle: async () => directory,
            }),
          } as unknown as StorageManager,
        },
      },
    );

    await store.save(makeBlock("layer0:b0", "opfs-session"));
    expect(store.mode).toBe("opfs");
    expect((await store.load("opfs-session", "layer0:b0")).block).toMatchObject({ id: "layer0:b0" });
  });

  it("uses binary sync-handle OPFS before async JSON OPFS when worker-owned handles are available", async () => {
    const directory = new FakeOpfsDirectory({ syncAccessHandle: true });
    const locks = new FakeLockManager();
    const store = await createBrowserKVSwapPersistence(
      { namespace: "binary-opfs", preferOpfs: true },
      {
        navigator: {
          storage: {
            getDirectory: async () => ({
              getDirectoryHandle: async () => directory,
            }),
          } as unknown as StorageManager,
          locks,
        },
      },
    );

    const persist = await store.save({
      ...makeBlock("layer0:b0", "binary-opfs"),
      promptTokenIds: [151643, 220],
      tokenIds: [151643, 220],
      queryRows: [[0.1, 0.2], [0.3, 0.4]],
      hiddenRows: [[0.5, 0.6], [0.7, 0.8]],
      checksum: "summary-checksum",
    });
    const loaded = await store.load("binary-opfs", "layer0:b0");

    expect(store.mode).toBe("opfs");
    expect(directory.fileNames()).toContain("binary-opfs__bGF5ZXIwOmIw.bin");
    expect(directory.fileNames()).not.toContain("binary-opfs__bGF5ZXIwOmIw.json");
    expect(loaded.block).toMatchObject({
      id: "layer0:b0",
      tokenIds: [151643, 220],
      queryRows: [[0.1, 0.2], [0.3, 0.4]],
      hiddenRows: [[0.5, 0.6], [0.7, 0.8]],
      checksum: "summary-checksum",
    });
    expect(persist).toMatchObject({
      mode: "opfs",
      binary: true,
      syncAccessHandle: true,
      webLocks: true,
      tabCoordination: "web_locks",
      bytesRead: 0,
      bytesWritten: expect.any(Number),
    });
    expect(persist.bytesWritten).toBeGreaterThan(64);
    expect(loaded.event).toMatchObject({
      binary: true,
      syncAccessHandle: true,
      webLocks: true,
      tabCoordination: "web_locks",
      bytesRead: expect.any(Number),
    });
    expect(loaded.event.bytesRead).toBeGreaterThan(64);
    expect(store.health()).toMatchObject({
      mode: "opfs",
      binary: true,
      syncAccessHandle: true,
      webLocks: true,
      tabCoordination: "web_locks",
      bytesRead: expect.any(Number),
      bytesWritten: expect.any(Number),
    });
    expect(locks.requests).toEqual(expect.arrayContaining(["kvswap:binary-opfs"]));
  });

  it("falls back to async JSON OPFS when sync handles exist without Web Locks or worker routing", async () => {
    const directory = new FakeOpfsDirectory({ syncAccessHandle: true });
    const store = await createBrowserKVSwapPersistence(
      { namespace: "uncoordinated-opfs", preferOpfs: true },
      {
        navigator: {
          storage: {
            getDirectory: async () => ({
              getDirectoryHandle: async () => directory,
            }),
          } as unknown as StorageManager,
        },
      },
    );

    const persist = await store.save(makeBlock("layer0:b0", "uncoordinated-opfs"));

    expect(store.mode).toBe("opfs");
    expect(directory.fileNames()).toContain("uncoordinated-opfs__bGF5ZXIwOmIw.json");
    expect(directory.fileNames()).not.toContain("uncoordinated-opfs__bGF5ZXIwOmIw.bin");
    expect(persist).toMatchObject({
      binary: false,
      syncAccessHandle: false,
      tabCoordination: "async_opfs_fallback",
    });
  });

  it("allows binary sync-handle OPFS through an explicit single worker route", async () => {
    const directory = new FakeOpfsDirectory({ syncAccessHandle: true });
    const store = await makeOpfsStore("single-worker-binary-opfs", directory, { singleWorkerRoute: true });

    const persist = await store.save(makeBlock("layer0:b0", "single-worker-binary-opfs"));

    expect(directory.fileNames()).toContain("single-worker-binary-opfs__bGF5ZXIwOmIw.bin");
    expect(persist).toMatchObject({
      binary: true,
      syncAccessHandle: true,
      webLocks: false,
      tabCoordination: "single_worker_route",
    });
    expect(store.health()).toMatchObject({
      binary: true,
      syncAccessHandle: true,
      webLocks: false,
      tabCoordination: "single_worker_route",
    });
  });

  it("routes binary OPFS through the KV persistence worker as a single owner", async () => {
    const directory = new FakeOpfsDirectory({ syncAccessHandle: true });
    const ready = await handleKVSwapPersistenceWorkerMessage(
      {
        id: "init-1",
        type: "init",
        options: { namespace: "worker-binary-opfs", preferOpfs: true },
      },
      {
        navigator: {
          storage: {
            getDirectory: async () => ({
              getDirectoryHandle: async () => directory,
            }),
          } as unknown as StorageManager,
        },
      },
    );
    const saved = await handleKVSwapPersistenceWorkerMessage({
      id: "save-1",
      type: "save",
      block: makeBlock("layer0:b0", "worker-binary-opfs"),
    });

    expect(ready).toMatchObject({
      type: "ready",
      health: {
        binary: true,
        syncAccessHandle: true,
        webLocks: false,
        tabCoordination: "single_worker_route",
      },
    });
    expect(saved).toMatchObject({
      type: "event",
      event: {
        binary: true,
        syncAccessHandle: true,
        webLocks: false,
        tabCoordination: "single_worker_route",
      },
    });
    expect(directory.fileNames()).toContain("worker-binary-opfs__bGF5ZXIwOmIw.bin");
  });

  it("serializes overlapping KV persistence worker messages in the single-worker route", async () => {
    const directory = new FakeOpfsDirectory({ syncAccessHandle: true, writeDelayMs: 5 });
    await handleKVSwapPersistenceWorkerMessage(
      {
        id: "init-serialized",
        type: "init",
        options: { namespace: "serialized-worker-opfs", preferOpfs: true },
      },
      {
        navigator: {
          storage: {
            getDirectory: async () => ({
              getDirectoryHandle: async () => directory,
            }),
          } as unknown as StorageManager,
        },
      },
    );

    const first = handleKVSwapPersistenceWorkerMessage({
      id: "save-first",
      type: "save",
      block: makeBlock("first", "serialized-worker-opfs"),
    });
    const second = handleKVSwapPersistenceWorkerMessage({
      id: "save-second",
      type: "save",
      block: makeBlock("second", "serialized-worker-opfs"),
    });

    await expect(Promise.all([first, second])).resolves.toEqual([
      expect.objectContaining({ id: "save-first", type: "event" }),
      expect.objectContaining({ id: "save-second", type: "event" }),
    ]);
    expect(directory.syncWriteOrder()).toEqual([
      "serialized-worker-opfs__Zmlyc3Q.bin",
      "serialized-worker-opfs__c2Vjb25k.bin",
    ]);
  });

  it("keeps KV persistence worker stores isolated by namespace after multiple init messages", async () => {
    const directory = new FakeOpfsDirectory({ syncAccessHandle: true });
    await handleKVSwapPersistenceWorkerMessage(
      { id: "init-a", type: "init", options: { namespace: "worker-namespace-a", preferOpfs: true } },
      {
        navigator: {
          storage: {
            getDirectory: async () => ({
              getDirectoryHandle: async () => directory,
            }),
          } as unknown as StorageManager,
        },
      },
    );
    await handleKVSwapPersistenceWorkerMessage(
      { id: "init-b", type: "init", options: { namespace: "worker-namespace-b", preferOpfs: true } },
      {
        navigator: {
          storage: {
            getDirectory: async () => ({
              getDirectoryHandle: async () => directory,
            }),
          } as unknown as StorageManager,
        },
      },
    );

    const saveA = await handleKVSwapPersistenceWorkerMessage({
      id: "save-a-after-b",
      type: "save",
      block: makeBlock("a", "worker-namespace-a"),
    });
    const saveB = await handleKVSwapPersistenceWorkerMessage({
      id: "save-b",
      type: "save",
      block: makeBlock("b", "worker-namespace-b"),
    });

    expect(saveA).toMatchObject({ id: "save-a-after-b", type: "event" });
    expect(saveB).toMatchObject({ id: "save-b", type: "event" });
    expect(directory.fileNames()).toEqual(expect.arrayContaining([
      "worker-namespace-a__YQ.bin",
      "worker-namespace-b__Yg.bin",
    ]));
  });

  it("quarantines corrupt binary OPFS records and keeps valid records listed", async () => {
    const directory = new FakeOpfsDirectory({ syncAccessHandle: true });
    const store = await makeOpfsStore("corrupt-binary-opfs", directory, { singleWorkerRoute: true });
    await store.save(makeBlock("valid", "corrupt-binary-opfs"));
    await directory.writeRawBinary("corrupt-binary-opfs", "corrupt", new Uint8Array([0, 1, 2, 3]));

    const listed = await store.list("corrupt-binary-opfs");

    expect(listed.records.map((block) => block.id)).toEqual(["valid"]);
    expect(listed.event).toMatchObject({
      ok: false,
      binary: true,
      syncAccessHandle: true,
      reason: expect.stringContaining("bad_magic"),
    });
    expect(directory.fileNames()).toHaveLength(1);
  });

  it("quarantines oversized binary OPFS records before reading the file body", async () => {
    const directory = new FakeOpfsDirectory({ syncAccessHandle: true });
    const store = await makeOpfsStore("oversized-binary-opfs", directory, { singleWorkerRoute: true });
    await store.save(makeBlock("valid", "oversized-binary-opfs"));
    await directory.writeRawBinary("oversized-binary-opfs", "huge", new Uint8Array([0, 1, 2, 3]));
    directory.overrideSize("oversized-binary-opfs__aHVnZQ.bin", MAX_KV_SWAP_BINARY_RECORD_BYTES + 1);

    const listed = await store.list("oversized-binary-opfs");

    expect(listed.records.map((block) => block.id)).toEqual(["valid"]);
    expect(listed.event).toMatchObject({
      ok: false,
      reason: expect.stringContaining("binary_record_too_large"),
    });
    expect(directory.reads()).not.toContain("oversized-binary-opfs__aHVnZQ.bin");
  });

  it("quarantines persisted KV records with unsupported versions", async () => {
    const store = new MemoryKVSwapPersistence({ namespace: "invalid-version" });
    await store.save({
      ...makeBlock("bad", "invalid-version"),
      version: 999 as typeof KV_SWAP_STORAGE_VERSION,
    });

    const listed = await store.list("invalid-version");
    expect(listed).toMatchObject({
      records: [],
      event: { ok: false, reason: expect.stringContaining("quarantined") },
    });
    expect((await store.load("invalid-version", "bad")).block).toBeNull();
  });

  it("preserves quarantine health on hydrate", async () => {
    const store = new MemoryKVSwapPersistence({ namespace: "hydrate-invalid" });
    await store.save({
      ...makeBlock("bad", "hydrate-invalid"),
      version: 999 as typeof KV_SWAP_STORAGE_VERSION,
    });

    const hydrated = await store.hydrate("hydrate-invalid");

    expect(hydrated.blocks).toEqual([]);
    expect(hydrated.event).toMatchObject({
      operation: "hydrate",
      ok: false,
      reason: expect.stringContaining("quarantined"),
    });
    expect(store.health().lastOperation).toMatchObject({ operation: "hydrate", ok: false });
  });

  it("quarantines malformed records without crashing list and load", async () => {
    const store = new MemoryKVSwapPersistence({ namespace: "malformed" });
    await store.save({
      ...makeBlock("bad-shape", "malformed"),
      keyRows: [["not-a-number"]] as unknown as number[][],
    });

    const listed = await store.list("malformed");

    expect(listed.records).toEqual([]);
    expect(listed.event).toMatchObject({ ok: false, reason: expect.stringContaining("invalid_keyRows") });
    expect((await store.load("malformed", "bad-shape")).block).toBeNull();
  });

  it("rejects invalid numeric payload semantics", async () => {
    const store = new MemoryKVSwapPersistence({ namespace: "bad-numbers" });
    await store.save({ ...makeBlock("negative-bytes", "bad-numbers"), byteLength: -1 });
    await store.save({ ...makeBlock("bad-range", "bad-numbers"), startToken: 4, endToken: 3 });

    const listed = await store.list("bad-numbers");

    expect(listed.records).toEqual([]);
    expect(listed.event).toMatchObject({
      ok: false,
      reason: expect.stringMatching(/invalid_byteLength|invalid_endToken/),
    });
  });

  it("quarantines malformed exact-reuse metadata in persisted records", async () => {
    const store = new MemoryKVSwapPersistence({ namespace: "bad-reuse-metadata" });
    await store.save({
      ...makeBlock("bad-phase", "bad-reuse-metadata"),
      phase: "wrong" as "prefill",
    });
    await store.save({
      ...makeBlock("bad-prompt-token", "bad-reuse-metadata"),
      promptTokenIds: [1, -1],
      prefillTokenCount: 2,
    });
    await store.save({
      ...makeBlock("bad-prefill-count", "bad-reuse-metadata"),
      promptTokenIds: [1, 2],
      prefillTokenCount: 3,
    });

    const listed = await store.list("bad-reuse-metadata");

    expect(listed.records).toEqual([]);
    expect(listed.event).toMatchObject({
      ok: false,
      reason: expect.stringContaining("quarantined"),
    });
  });

  it("quarantines corrupt OPFS JSON and keeps valid records listed", async () => {
    const directory = new FakeOpfsDirectory();
    const store = await makeOpfsStore("corrupt-opfs", directory);
    await store.save(makeBlock("valid", "corrupt-opfs"));
    await directory.writeRaw("corrupt-opfs", "corrupt", "{not json");

    const listed = await store.list("corrupt-opfs");

    expect(listed.records.map((block) => block.id)).toEqual(["valid"]);
    expect(listed.event).toMatchObject({ ok: false, reason: expect.stringContaining("corrupt_json") });
    expect(directory.fileNames()).toHaveLength(1);
  });

  it("quarantines malformed IndexedDB-like records without crashing", async () => {
    const store = makeIndexedDbTestStore("malformed-idb", { maxBlocks: 8, maxBytes: 1024 * 1024 });
    await store.save(makeBlock("valid", "malformed-idb"));
    await store.save({
      ...makeBlock("malformed", "malformed-idb"),
      valueRows: [null] as unknown as number[][],
    });

    const listed = await store.list("malformed-idb");

    expect(listed.records.map((block) => block.id)).toEqual(["valid"]);
    expect(listed.event).toMatchObject({ ok: false, reason: expect.stringContaining("invalid_valueRows") });
  });

  it("exercises the real IndexedDB adapter path for quarantine and quota reporting", async () => {
    const fakeDb = new FakeIndexedDb();
    const store = await createBrowserKVSwapPersistence(
      { namespace: "real-idb", preferOpfs: false, maxBlocks: 1, maxBytes: 1 },
      {
        indexedDB: {} as IDBFactory,
        openIndexedDb: makeFakeOpenDb(fakeDb),
      },
    );
    await store.persist([makeBlock("pinned", "real-idb", 1, true)]);
    await store.persist([makeBlock("cold", "real-idb", 2, false)]);
    await fakeDb.putRaw({
      ...makeBlock("malformed", "real-idb"),
      storageKey: "real-idb:malformed",
      valueRows: [null] as unknown as number[][],
    });

    const listed = await store.list("real-idb");
    const hydrated = await store.hydrate("real-idb");

    expect(store.mode).toBe("indexeddb");
    expect(listed.records.map((block) => block.id)).toEqual(["pinned"]);
    expect(listed.event).toMatchObject({ ok: false, reason: expect.stringContaining("invalid_valueRows") });
    expect(hydrated.event).toMatchObject({ ok: true });
    expect(fakeDb.rows().map((row) => row.id)).toEqual(["pinned"]);
  });

  it("clears and deletes only the requested namespace", async () => {
    const store = new MemoryKVSwapPersistence({ namespace: "a", maxBlocks: 8, maxBytes: 1024 * 1024 });
    await store.persist([makeBlock("keep", "a"), makeBlock("delete", "a")]);
    await store.persist([makeBlock("other", "b")]);

    const deleted = await store.delete("a", "delete");
    expect(deleted).toMatchObject({ operation: "delete", blockIds: ["delete"], ok: true });
    expect((await store.hydrate("a")).blocks.map((block) => block.id)).toEqual(["keep"]);

    const clear = await store.clear("a");
    expect(clear).toMatchObject({ operation: "clear", blockIds: ["keep"], ok: true });
    expect((await store.hydrate("a")).blocks).toEqual([]);
    expect((await store.hydrate("b")).blocks.map((block) => block.id)).toEqual(["other"]);
  });

  it("evicts older blocks over quota and reports health metadata", async () => {
    const store = new MemoryKVSwapPersistence({ namespace: "quota", maxBlocks: 1, maxBytes: 1024 * 1024 });
    await store.persist([makeBlock("old", "quota", 1)]);
    await store.persist([makeBlock("new", "quota", 2)]);

    expect((await store.hydrate("quota")).blocks.map((block) => block.id)).toEqual(["new"]);
    expect(store.health().lastOperation).toMatchObject({
      operation: "hydrate",
      mode: "memory",
      ok: true,
      quotaBytes: 1024 * 1024,
    });
  });

  it("never evicts pinned blocks over quota in memory and IndexedDB-like stores", async () => {
    const memory = new MemoryKVSwapPersistence({ namespace: "pinned", maxBlocks: 1, maxBytes: 1 });
    await memory.persist([makeBlock("pinned", "pinned", 1, true)]);
    const memoryPersist = await memory.persist([makeBlock("cold", "pinned", 2, false)]);

    expect((await memory.list("pinned")).records.map((block) => block.id)).toEqual(["pinned"]);
    expect(memoryPersist).toMatchObject({ ok: false, reason: expect.stringContaining("protected_over_quota") });

    const indexedDb = makeIndexedDbTestStore("pinned-idb", { maxBlocks: 1, maxBytes: 1 });
    await indexedDb.persist([makeBlock("pinned", "pinned-idb", 1, true)]);
    const indexedDbPersist = await indexedDb.persist([makeBlock("cold", "pinned-idb", 2, false)]);

    expect((await indexedDb.list("pinned-idb")).records.map((block) => block.id)).toEqual(["pinned"]);
    expect(indexedDbPersist).toMatchObject({ ok: false, reason: expect.stringContaining("protected_over_quota") });
  });

  it("never evicts pinned blocks over quota in OPFS stores", async () => {
    const directory = new FakeOpfsDirectory();
    const store = await makeOpfsStore("pinned-opfs", directory, { maxBlocks: 1, maxBytes: 1 });
    await store.persist([makeBlock("pinned", "pinned-opfs", 1, true)]);
    const persist = await store.persist([makeBlock("cold", "pinned-opfs", 2, false)]);

    expect((await store.list("pinned-opfs")).records.map((block) => block.id)).toEqual(["pinned"]);
    expect(persist).toMatchObject({ ok: false, reason: expect.stringContaining("protected_over_quota") });
  });

  it("uses collision-safe OPFS filenames while preserving raw ids in records and clear events", async () => {
    const directory = new FakeOpfsDirectory();
    const store = await makeOpfsStore("encoded-opfs", directory);
    await store.persist([
      makeBlock("layer/a:b", "encoded-opfs"),
      makeBlock("layer_a_b", "encoded-opfs"),
    ]);

    expect(directory.fileNames()).toHaveLength(2);
    expect((await store.list("encoded-opfs")).records.map((block) => block.id).sort()).toEqual(["layer/a:b", "layer_a_b"]);
    const cleared = await store.clear("encoded-opfs");
    expect(cleared.blockIds.sort()).toEqual(["layer/a:b", "layer_a_b"]);
  });
});

class FakeOpfsDirectory {
  readonly kind = "directory";
  readonly name = "fake";
  private readonly files = new Map<string, Uint8Array>();
  private readonly writes: string[] = [];
  private readonly readNames: string[] = [];
  private readonly sizeOverrides = new Map<string, number>();

  constructor(private readonly options: { syncAccessHandle?: boolean; writeDelayMs?: number } = {}) {}

  async getFileHandle(name: string, options?: { create?: boolean }): Promise<FakeOpfsFileHandle> {
    if (!this.files.has(name) && !options?.create) {
      throw new DOMException("Not found", "NotFoundError");
    }
    if (!this.files.has(name)) this.files.set(name, new Uint8Array());
    return new FakeOpfsFileHandle(name, this.files, this.writes, this.readNames, this.sizeOverrides, this.options);
  }

  async removeEntry(name: string): Promise<void> {
    this.files.delete(name);
  }

  async writeRaw(namespace: string, blockId: string, text: string): Promise<void> {
    const handle = await this.getFileHandle(`${namespace}__${stableTestBlockFileId(blockId)}.json`, { create: true });
    const writable = await handle.createWritable();
    await writable.write(text);
    await writable.close();
  }

  async writeRawBinary(namespace: string, blockId: string, bytes: Uint8Array): Promise<void> {
    const handle = await this.getFileHandle(`${namespace}__${stableTestBlockFileId(blockId)}.bin`, { create: true });
    const access = await handle.createSyncAccessHandle();
    access.write(bytes, { at: 0 });
    access.truncate(bytes.byteLength);
    access.flush();
    access.close();
  }

  fileNames(): string[] {
    return [...this.files.keys()];
  }

  async *entries(): AsyncIterable<[string, FileSystemHandle]> {
    for (const name of this.files.keys()) {
      yield [name, new FakeOpfsFileHandle(name, this.files, this.writes, this.readNames, this.sizeOverrides, this.options) as unknown as FileSystemHandle];
    }
  }

  syncWriteOrder(): string[] {
    return this.writes.filter((name) => !name.includes("__sync_access_probe__"));
  }

  overrideSize(name: string, size: number): void {
    this.sizeOverrides.set(name, size);
  }

  reads(): string[] {
    return [...this.readNames];
  }
}

async function makeOpfsStore(
  namespace: string,
  directory: FakeOpfsDirectory,
  options: { maxBlocks?: number; maxBytes?: number; singleWorkerRoute?: boolean } = {},
): Promise<KVSwapPersistenceStore> {
  const { singleWorkerRoute, ...storeOptions } = options;
  return await createBrowserKVSwapPersistence(
    { namespace, preferOpfs: true, ...storeOptions },
    {
      navigator: {
        storage: {
          getDirectory: async () => ({
            getDirectoryHandle: async () => directory,
          }),
        } as unknown as StorageManager,
      },
      ...(singleWorkerRoute ? { singleWorkerRoute: true } : {}),
    },
  );
}

function stableTestBlockFileId(blockId: string): string {
  const bytes = new TextEncoder().encode(blockId);
  let binary = "";
  for (const byte of bytes) binary += String.fromCharCode(byte);
  return btoa(binary).replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/g, "");
}

function makeFakeOpenDb(fakeDb: FakeIndexedDb) {
  return (async (_name: string, _version: number, callbacks?: { upgrade?: (database: unknown) => void }) => {
    callbacks?.upgrade?.({
      objectStoreNames: {
        contains: () => true,
      },
      createObjectStore: () => ({
        createIndex: () => undefined,
      }),
    });
    return fakeDb;
  }) as unknown as typeof import("idb").openDB;
}

class FakeIndexedDb {
  private readonly store = new Map<string, StoredTestRow>();

  async get(_storeName: string, key: string): Promise<StoredTestRow | undefined> {
    return this.store.get(key);
  }

  async delete(_storeName: string, key: string): Promise<void> {
    this.store.delete(key);
  }

  async getAllFromIndex(_storeName: string, _indexName: string, namespace: string): Promise<StoredTestRow[]> {
    return [...this.store.values()].filter((row) => row.namespace === namespace);
  }

  transaction(_storeName: string, _mode: "readwrite") {
    return {
      store: {
        put: async (row: StoredTestRow) => {
          this.store.set(row.storageKey, row);
        },
        delete: async (key: string) => {
          this.store.delete(key);
        },
      },
      done: Promise.resolve(),
    };
  }

  async putRaw(row: StoredTestRow): Promise<void> {
    this.store.set(row.storageKey, row);
  }

  rows(): StoredTestRow[] {
    return [...this.store.values()];
  }
}

type StoredTestRow = SerializedKVSwapBlock & { storageKey: string };

function makeIndexedDbTestStore(namespace: string, options: { maxBlocks?: number; maxBytes?: number } = {}): KVSwapPersistenceStore {
  const memory = new MemoryKVSwapPersistence({ namespace, ...options });
  return {
    mode: "indexeddb",
    save: (block) => memory.save(block),
    load: (targetNamespace, blockId) => memory.load(targetNamespace, blockId),
    delete: (targetNamespace, blockId) => memory.delete(targetNamespace, blockId),
    list: (targetNamespace) => memory.list(targetNamespace),
    persist: (blocks) => memory.persist(blocks),
    hydrate: (targetNamespace) => memory.hydrate(targetNamespace),
    evict: (targetNamespace, blockIds) => memory.evict(targetNamespace, blockIds),
    clear: (targetNamespace) => memory.clear(targetNamespace),
    health: () => ({ ...memory.health(), mode: "indexeddb" }),
  };
}

class FakeOpfsFileHandle {
  readonly kind = "file";

  constructor(
    readonly name: string,
    private readonly files: Map<string, Uint8Array>,
    private readonly writes: string[],
    private readonly readNames: string[],
    private readonly sizeOverrides: Map<string, number>,
    private readonly options: { syncAccessHandle?: boolean; writeDelayMs?: number } = {},
  ) {}

  async createWritable(): Promise<{ write(value: string): Promise<void>; close(): Promise<void> }> {
    return {
      write: async (value: string) => {
        this.files.set(this.name, new TextEncoder().encode(value));
      },
      close: async () => undefined,
    };
  }

  async getFile(): Promise<{ text(): Promise<string> }> {
    return {
      text: async () => new TextDecoder().decode(this.files.get(this.name) ?? new Uint8Array()),
    };
  }

  async createSyncAccessHandle(): Promise<FakeSyncAccessHandle> {
    if (!this.options.syncAccessHandle) throw new TypeError("sync access handles unavailable");
    return new FakeSyncAccessHandle(this.name, this.files, this.writes, this.readNames, this.sizeOverrides, this.options);
  }
}

class FakeSyncAccessHandle {
  constructor(
    readonly name: string,
    private readonly files: Map<string, Uint8Array>,
    private readonly writes: string[],
    private readonly readNames: string[],
    private readonly sizeOverrides: Map<string, number>,
    private readonly options: { writeDelayMs?: number } = {},
  ) {}

  getSize(): number {
    const override = this.sizeOverrides.get(this.name);
    if (override !== undefined) return override;
    return this.files.get(this.name)?.byteLength ?? 0;
  }

  read(buffer: Uint8Array, options: { at?: number } = {}): number {
    this.readNames.push(this.name);
    const source = this.files.get(this.name) ?? new Uint8Array();
    const at = options.at ?? 0;
    const slice = source.subarray(at, at + buffer.byteLength);
    buffer.set(slice);
    return slice.byteLength;
  }

  write(buffer: Uint8Array, options: { at?: number } = {}): number {
    if (this.options.writeDelayMs) busyWait(this.options.writeDelayMs);
    this.writes.push(this.name);
    const at = options.at ?? 0;
    const current = this.files.get(this.name) ?? new Uint8Array();
    const nextLength = Math.max(current.byteLength, at + buffer.byteLength);
    const next = new Uint8Array(nextLength);
    next.set(current);
    next.set(buffer, at);
    this.files.set(this.name, next);
    return buffer.byteLength;
  }

  truncate(size: number): void {
    const current = this.files.get(this.name) ?? new Uint8Array();
    this.files.set(this.name, current.slice(0, size));
  }

  flush(): void {}

  close(): void {}
}

class FakeLockManager {
  readonly requests: string[] = [];

  async request<T>(name: string, _options: { mode: "exclusive" }, callback: () => T | Promise<T>): Promise<T> {
    this.requests.push(name);
    return await callback();
  }
}

function busyWait(ms: number): void {
  const started = Date.now();
  while (Date.now() - started < ms) {
    // Test-only synchronous pause to expose missing worker serialization.
  }
}

function makeBlock(id: string, namespace = "tenant:cell:session", lastAccessAt = 100, pinned = false): SerializedKVSwapBlock {
  return {
    version: KV_SWAP_STORAGE_VERSION,
    namespace,
    id,
    modelId: "Qwen/Qwen3-0.6B",
    requestId: "req",
    layer: 0,
    startToken: 0,
    endToken: 2,
    pinned,
    importance: 0.5,
    estimatedBytes: 16,
    compressedKeySummary: [0.25, 0.75],
    summaryRank: 2,
    keyRows: [[1, 2], [3, 4]],
    valueRows: [[5, 6], [7, 8]],
    createdAt: "2026-05-17T00:00:00.000Z",
    updatedAt: "2026-05-17T00:00:00.000Z",
    lastAccessAt,
    byteLength: 64,
  };
}
