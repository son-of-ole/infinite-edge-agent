import { openDB, type IDBPDatabase } from "idb";
import { decodeKVSwapBlockBinary, encodeKVSwapBlockBinary, MAX_KV_SWAP_BINARY_RECORD_BYTES } from "./kvSwapBinaryCodec";

export const KV_SWAP_STORAGE_VERSION = 1;
const DEFAULT_DB_NAME = "infinite-edge-kvswap";
const DEFAULT_STORE_NAME = "kv_blocks";
const DEFAULT_OPFS_DIR = "infinite-edge-kvswap";

export type KVSwapPersistenceMode = "disabled" | "opfs" | "indexeddb" | "memory";
export type KVSwapPersistenceOperation = "hydrate" | "persist" | "reuse" | "evict" | "clear" | "save" | "load" | "delete" | "list";
export type KVSwapTabCoordination = "none" | "memory_local" | "indexeddb_transaction" | "async_opfs_fallback" | "single_worker_route" | "web_locks";
export type KVSwapPrefetchStrategy = "none" | "exact_reuse" | "predictive_prefetch" | "miss_stall";
export type KVSwapLowRankQuerySource = "persisted_q_rows" | "token_id_fallback";

export interface SerializedKVLowRankKeySummary {
  blockId: string;
  rank: number;
  projectionId: string;
  layer: number;
  headGroupId: string;
  checksum: string;
  qualityScore: number;
  values: number[];
}

export type SerializedKVSwapKernelBackend = "webgpu" | "cpu_reference" | "mixed";

export interface SerializedKVSwapPrefillLayerProof {
  layerIndex: number;
  qProjection: SerializedKVSwapKernelBackend;
  kProjection: SerializedKVSwapKernelBackend;
  vProjection: SerializedKVSwapKernelBackend;
  oProjection: SerializedKVSwapKernelBackend;
  mlpBackend?: SerializedKVSwapKernelBackend;
  mlpActivationKind?: string;
  mlpRowCount?: number;
  attentionBackend: SerializedKVSwapKernelBackend;
  packedHeadBackends: SerializedKVSwapKernelBackend[];
  packedHeadCount: number;
  selectedKeyRows?: number;
  prefillChunkDispatch?: "single_dispatch" | "chunked_dispatch";
  attentionDispatchCount?: number;
  awaitedDispatchBreaks?: number;
}

export interface SerializedKVSwapPrefillProof {
  layers: SerializedKVSwapPrefillLayerProof[];
  prefillChunkCount?: number;
  prefillChunkSize?: number;
  shapeBucket?: string;
  pipelineCacheKey?: string;
  prefillDispatchTargetMs?: number;
  maxDispatchEstimatedMs?: number;
  prefillChunkDispatch?: "single_dispatch" | "chunked_dispatch";
  attentionDispatchCount?: number;
  awaitedDispatchBreaks?: number;
}

export interface KVSwapPredictiveHotBlockTrace {
  blockId: string;
  runtimeBlockId?: string;
  score: number;
  rank: number;
  projectionId: string;
  layer: number;
  headGroupId: string;
  checksum: string;
  qualityScore: number;
  tier: "vram" | "ram" | "disk";
  estimatedBytes: number;
  source: "low_rank_attention";
}

export interface SerializedKVSwapBlock {
  version: typeof KV_SWAP_STORAGE_VERSION;
  namespace: string;
  id: string;
  modelId: string;
  requestId: string;
  runtimeBlockId?: string;
  phase?: "prefill" | "decode";
  modelFingerprint?: string;
  promptTokenHash?: string;
  promptTokenIds?: number[];
  prefillTokenCount?: number;
  runtimeLayerCount?: number;
  policyHash?: string;
  layer: number;
  startToken: number;
  endToken: number;
  pinned: boolean;
  importance: number;
  estimatedBytes: number;
  checksum?: string;
  summaryRank?: number;
  compressedKeySummary?: number[] | string;
  lowRankKeySummary?: SerializedKVLowRankKeySummary;
  prefillProof?: SerializedKVSwapPrefillProof;
  tokenIds?: number[];
  queryRows?: number[][];
  keyRows: number[][];
  valueRows: number[][];
  compactKeyRows?: number[][];
  compactValueRows?: number[][];
  hiddenRows?: number[][];
  createdAt: string;
  updatedAt: string;
  lastAccessAt: number;
  byteLength: number;
}

export interface KVSwapPersistenceTraceEvent {
  operation: KVSwapPersistenceOperation;
  mode: KVSwapPersistenceMode;
  detail?: string;
  ok: boolean;
  namespace: string;
  blockIds: string[];
  bytes: number;
  quotaBytes?: number;
  usageBytes?: number;
  reason?: string;
  binary?: boolean;
  syncAccessHandle?: boolean;
  webLocks?: boolean;
  lockWaitMs?: number;
  bytesRead?: number;
  bytesWritten?: number;
  tabCoordination?: KVSwapTabCoordination;
  prefetchStrategy?: KVSwapPrefetchStrategy;
  lowRankSummaryRank?: number;
  lowRankQuerySource?: KVSwapLowRankQuerySource;
  predictedHotBlocks?: KVSwapPredictiveHotBlockTrace[];
  prefetchedBlocks?: string[];
  prefetchHitRate?: number;
  prefetchBytes?: number;
  prefetchLatencyMs?: number;
  attentionStallMs?: number;
  kvPersistDeferred?: boolean;
  kvPersistCriticalPathMs?: number;
  kvPersistFlushMs?: number;
  kvPersistPendingBlockCount?: number;
  at: string;
}

export interface KVSwapPersistenceHealth {
  enabled: boolean;
  mode: KVSwapPersistenceMode;
  detail?: string;
  namespace: string;
  decodeReuse: boolean;
  lastOperation?: KVSwapPersistenceTraceEvent;
  quotaBytes?: number;
  usageBytes?: number;
  binary?: boolean;
  syncAccessHandle?: boolean;
  webLocks?: boolean;
  lockWaitMs?: number;
  bytesRead?: number;
  bytesWritten?: number;
  tabCoordination?: KVSwapTabCoordination;
  prefetchStrategy?: KVSwapPrefetchStrategy;
  lowRankSummaryRank?: number;
  lowRankQuerySource?: KVSwapLowRankQuerySource;
  predictedHotBlocks?: KVSwapPredictiveHotBlockTrace[];
  prefetchedBlocks?: string[];
  prefetchHitRate?: number;
  prefetchBytes?: number;
  prefetchLatencyMs?: number;
  attentionStallMs?: number;
  kvPersistDeferred?: boolean;
  kvPersistCriticalPathMs?: number;
  kvPersistFlushMs?: number;
  kvPersistPendingBlockCount?: number;
}

export interface KVSwapPersistedBlockMetadata {
  version: number;
  namespace: string;
  id: string;
  modelId: string;
  requestId: string;
  layer: number;
  startToken: number;
  endToken: number;
  pinned: boolean;
  importance: number;
  estimatedBytes: number;
  checksum?: string;
  summaryRank?: number;
  createdAt: string;
  updatedAt: string;
  lastAccessAt: number;
  byteLength: number;
}

export interface KVSwapPersistenceListResult {
  records: SerializedKVSwapBlock[];
  metadata: KVSwapPersistedBlockMetadata[];
  event: KVSwapPersistenceTraceEvent;
}

export interface KVSwapPersistenceStore {
  readonly mode: KVSwapPersistenceMode;
  save(block: SerializedKVSwapBlock): Promise<KVSwapPersistenceTraceEvent>;
  load(namespace: string, blockId: string): Promise<{ block: SerializedKVSwapBlock | null; event: KVSwapPersistenceTraceEvent }>;
  delete(namespace: string, blockId: string): Promise<KVSwapPersistenceTraceEvent>;
  list(namespace: string): Promise<KVSwapPersistenceListResult>;
  persist(blocks: SerializedKVSwapBlock[]): Promise<KVSwapPersistenceTraceEvent>;
  hydrate(namespace: string): Promise<{ blocks: SerializedKVSwapBlock[]; event: KVSwapPersistenceTraceEvent }>;
  evict(namespace: string, blockIds: string[]): Promise<KVSwapPersistenceTraceEvent>;
  clear(namespace: string): Promise<KVSwapPersistenceTraceEvent>;
  health(): KVSwapPersistenceHealth;
}

export interface BrowserKVSwapPersistenceOptions {
  namespace: string;
  preferOpfs?: boolean;
  maxBlocks?: number;
  maxBytes?: number;
}

export interface BrowserKVSwapPersistenceEnvironment {
  navigator?: Pick<Navigator, "storage"> & { locks?: KVSwapLockManager };
  indexedDB?: IDBFactory;
  createIndexedDbPersistence?: (options: BrowserKVSwapPersistenceOptions) => Promise<KVSwapPersistenceStore>;
  openIndexedDb?: typeof openDB;
  singleWorkerRoute?: boolean;
}

export interface KVSwapLockManager {
  request<T>(name: string, options: { mode: "exclusive" }, callback: () => T | Promise<T>): Promise<T>;
}

export async function createBrowserKVSwapPersistence(
  options: BrowserKVSwapPersistenceOptions,
  env: BrowserKVSwapPersistenceEnvironment = globalThis,
): Promise<KVSwapPersistenceStore> {
  const namespace = normalizeNamespace(options.namespace);
  const baseOptions = { ...options, namespace };
  if (options.preferOpfs !== false && typeof env.navigator?.storage?.getDirectory === "function") {
    try {
      try {
        return await BinaryOpfsKVSwapPersistence.create(
          baseOptions,
          env.navigator.storage,
          env.navigator.locks,
          env.singleWorkerRoute === true,
        );
      } catch {
        return await OpfsKVSwapPersistence.create(baseOptions, env.navigator.storage);
      }
    } catch {
      // Fall through to IndexedDB; private browsing and quota prompts can deny OPFS.
    }
  }
  if (env.indexedDB) {
    if (env.createIndexedDbPersistence) return await env.createIndexedDbPersistence(baseOptions);
    return await IndexedDbKVSwapPersistence.create(baseOptions, env.openIndexedDb);
  }
  return new MemoryKVSwapPersistence(baseOptions);
}

export class MemoryKVSwapPersistence implements KVSwapPersistenceStore {
  readonly mode: KVSwapPersistenceMode = "memory";
  private readonly namespace: string;
  private readonly maxBlocks: number;
  private readonly maxBytes: number;
  private readonly blocks = new Map<string, SerializedKVSwapBlock>();
  private lastOperation: KVSwapPersistenceTraceEvent | undefined;

  constructor(options: BrowserKVSwapPersistenceOptions) {
    this.namespace = normalizeNamespace(options.namespace);
    this.maxBlocks = options.maxBlocks ?? 512;
    this.maxBytes = options.maxBytes ?? 256 * 1024 * 1024;
  }

  async save(block: SerializedKVSwapBlock): Promise<KVSwapPersistenceTraceEvent> {
    return await this.persist([block]);
  }

  async load(namespace: string, blockId: string): Promise<{ block: SerializedKVSwapBlock | null; event: KVSwapPersistenceTraceEvent }> {
    const normalized = normalizeNamespace(namespace);
    const block = this.blocks.get(storageKey(normalized, blockId)) ?? null;
    const validation = block ? validateSerializedBlock(block) : { ok: true as const, block: null };
    if (!validation.ok) {
      this.blocks.delete(storageKey(normalized, blockId));
      const event = this.record("load", [], false, `quarantined:${validation.reason}`);
      return { block: null, event };
    }
    const event = this.record("load", validation.block ? [validation.block.id] : [], true);
    return { block: validation.block, event };
  }

  async delete(namespace: string, blockId: string): Promise<KVSwapPersistenceTraceEvent> {
    const normalized = normalizeNamespace(namespace);
    this.blocks.delete(storageKey(normalized, blockId));
    return this.record("delete", [blockId], true);
  }

  async list(namespace: string): Promise<KVSwapPersistenceListResult> {
    const normalized = normalizeNamespace(namespace);
    const records: SerializedKVSwapBlock[] = [];
    const quarantined: string[] = [];
    for (const [key, block] of this.blocks.entries()) {
      if (block.namespace !== normalized) continue;
      const validation = validateSerializedBlock(block);
      if (validation.ok) {
        records.push(validation.block);
      } else {
        quarantined.push(`${block.id ?? key}:${validation.reason}`);
        this.blocks.delete(key);
      }
    }
    const event = this.record(
      "list",
      records.map((block) => block.id),
      quarantined.length === 0,
      quarantined.length ? `quarantined:${quarantined.join(",")}` : undefined,
    );
    return { records, metadata: records.map(toMetadata), event };
  }

  async persist(blocks: SerializedKVSwapBlock[]): Promise<KVSwapPersistenceTraceEvent> {
    for (const block of blocks) {
      const namespace = normalizeNamespace(block.namespace || this.namespace);
      this.blocks.set(storageKey(namespace, block.id), { ...block, namespace });
    }
    const quota = this.evictOverBudget();
    return this.record(
      "persist",
      blocks.map((block) => block.id),
      quota.ok,
      quota.reason,
      quota.evictedBlockIds,
    );
  }

  async hydrate(namespace: string): Promise<{ blocks: SerializedKVSwapBlock[]; event: KVSwapPersistenceTraceEvent }> {
    const listed = await this.list(namespace);
    const event = this.record(
      "hydrate",
      listed.records.map((block) => block.id),
      listed.event.ok,
      listed.event.reason,
    );
    return { blocks: listed.records, event };
  }

  async evict(namespace: string, blockIds: string[]): Promise<KVSwapPersistenceTraceEvent> {
    for (const blockId of blockIds) await this.delete(namespace, blockId);
    return this.record("evict", blockIds, true);
  }

  async clear(namespace: string): Promise<KVSwapPersistenceTraceEvent> {
    const normalized = normalizeNamespace(namespace);
    const blockIds: string[] = [];
    for (const [key, block] of this.blocks.entries()) {
      if (block.namespace !== normalized) continue;
      blockIds.push(block.id);
      this.blocks.delete(key);
    }
    return this.record("clear", blockIds, true);
  }

  health(): KVSwapPersistenceHealth {
    const usageBytes = this.usageBytes();
    return {
      enabled: true,
      mode: this.mode,
      detail: "memory-local",
      namespace: this.namespace,
      decodeReuse: false,
      usageBytes,
      quotaBytes: this.maxBytes,
      binary: false,
      syncAccessHandle: false,
      webLocks: false,
      lockWaitMs: 0,
      bytesRead: this.lastOperation?.bytesRead ?? 0,
      bytesWritten: this.lastOperation?.bytesWritten ?? 0,
      tabCoordination: "memory_local",
      ...(this.lastOperation ? { lastOperation: this.lastOperation } : {}),
    };
  }

  private evictOverBudget(): QuotaEvictionResult {
    const blocks = [...this.blocks.values()];
    const quota = planQuotaEviction(blocks, this.maxBlocks, this.maxBytes);
    for (const blockId of quota.evictedBlockIds) {
      const block = blocks.find((candidate) => candidate.id === blockId);
      if (block) this.blocks.delete(storageKey(block.namespace, block.id));
    }
    return quota;
  }

  private usageBytes(): number {
    return [...this.blocks.values()].reduce((sum, block) => sum + block.byteLength, 0);
  }

  private record(
    operation: KVSwapPersistenceOperation,
    blockIds: string[],
    ok: boolean,
    reason?: string,
    evictedBlockIds: string[] = [],
  ): KVSwapPersistenceTraceEvent {
    const event = makeTraceEvent({
      operation,
      mode: this.mode,
      namespace: this.namespace,
      blockIds: [...blockIds, ...evictedBlockIds],
      bytes: this.usageBytes(),
      quotaBytes: this.maxBytes,
      usageBytes: this.usageBytes(),
      ok,
      ...(reason !== undefined ? { reason } : {}),
    });
    this.lastOperation = event;
    return event;
  }
}

class IndexedDbKVSwapPersistence implements KVSwapPersistenceStore {
  readonly mode: KVSwapPersistenceMode = "indexeddb";
  private lastOperation: KVSwapPersistenceTraceEvent | undefined;

  private constructor(
    private readonly options: Required<BrowserKVSwapPersistenceOptions>,
    private readonly db: IDBPDatabase,
  ) {}

  static async create(
    options: BrowserKVSwapPersistenceOptions,
    openIndexedDb: typeof openDB = openDB,
  ): Promise<IndexedDbKVSwapPersistence> {
    const normalized = withDefaults(options);
    const db = await openIndexedDb(DEFAULT_DB_NAME, 1, {
      upgrade(database) {
        if (!database.objectStoreNames.contains(DEFAULT_STORE_NAME)) {
          const store = database.createObjectStore(DEFAULT_STORE_NAME, { keyPath: "storageKey" });
          store.createIndex("namespace", "namespace");
          store.createIndex("lastAccessAt", "lastAccessAt");
        }
      },
    });
    return new IndexedDbKVSwapPersistence(normalized, db);
  }

  async save(block: SerializedKVSwapBlock): Promise<KVSwapPersistenceTraceEvent> {
    return await this.persist([block]);
  }

  async load(namespace: string, blockId: string): Promise<{ block: SerializedKVSwapBlock | null; event: KVSwapPersistenceTraceEvent }> {
    const normalized = normalizeNamespace(namespace);
    const row = await this.db.get(DEFAULT_STORE_NAME, storageKey(normalized, blockId)) as StoredKVSwapBlock | undefined;
    const validation = row ? validateSerializedBlock(stripStorageKey(row)) : { ok: true as const, block: null };
    if (!validation.ok) {
      await this.db.delete(DEFAULT_STORE_NAME, storageKey(normalized, blockId));
      const event = await this.record("load", [], false, `quarantined:${validation.reason}`);
      return { block: null, event };
    }
    const block = validation.block;
    const event = await this.record("load", block ? [block.id] : [], true);
    return { block, event };
  }

  async delete(namespace: string, blockId: string): Promise<KVSwapPersistenceTraceEvent> {
    const normalized = normalizeNamespace(namespace);
    await this.db.delete(DEFAULT_STORE_NAME, storageKey(normalized, blockId));
    return await this.record("delete", [blockId], true);
  }

  async list(namespace: string): Promise<KVSwapPersistenceListResult> {
    const normalized = normalizeNamespace(namespace);
    const rows = await this.db.getAllFromIndex(DEFAULT_STORE_NAME, "namespace", normalized);
    const { records, quarantined } = await this.validateStoredRows(rows as StoredKVSwapBlock[]);
    const event = await this.record(
      "list",
      records.map((block) => block.id),
      quarantined.length === 0,
      quarantined.length ? `quarantined:${quarantined.join(",")}` : undefined,
    );
    return { records, metadata: records.map(toMetadata), event };
  }

  async persist(blocks: SerializedKVSwapBlock[]): Promise<KVSwapPersistenceTraceEvent> {
    const tx = this.db.transaction(DEFAULT_STORE_NAME, "readwrite");
    for (const block of blocks) {
      await tx.store.put({ ...block, namespace: this.options.namespace, storageKey: storageKey(this.options.namespace, block.id) });
    }
    await tx.done;
    const quota = await this.evictOverBudget();
    return await this.record(
      "persist",
      [...blocks.map((block) => block.id), ...quota.evictedBlockIds],
      quota.ok,
      quota.reason,
    );
  }

  async hydrate(namespace: string): Promise<{ blocks: SerializedKVSwapBlock[]; event: KVSwapPersistenceTraceEvent }> {
    const listed = await this.list(namespace);
    const event = await this.record(
      "hydrate",
      listed.records.map((block) => block.id),
      listed.event.ok,
      listed.event.reason,
    );
    return { blocks: listed.records, event };
  }

  async evict(namespace: string, blockIds: string[]): Promise<KVSwapPersistenceTraceEvent> {
    for (const blockId of blockIds) await this.delete(namespace, blockId);
    return await this.record("evict", blockIds, true);
  }

  async clear(namespace: string): Promise<KVSwapPersistenceTraceEvent> {
    const normalized = normalizeNamespace(namespace);
    const rows = await this.db.getAllFromIndex(DEFAULT_STORE_NAME, "namespace", normalized);
    const tx = this.db.transaction(DEFAULT_STORE_NAME, "readwrite");
    for (const row of rows as StoredKVSwapBlock[]) await tx.store.delete(row.storageKey);
    await tx.done;
    return await this.record("clear", rows.map((row) => (row as StoredKVSwapBlock).id), true);
  }

  health(): KVSwapPersistenceHealth {
    return {
      enabled: true,
      mode: this.mode,
      detail: "indexeddb-transaction",
      namespace: this.options.namespace,
      decodeReuse: false,
      binary: false,
      syncAccessHandle: false,
      webLocks: false,
      lockWaitMs: 0,
      bytesRead: this.lastOperation?.bytesRead ?? 0,
      bytesWritten: this.lastOperation?.bytesWritten ?? 0,
      tabCoordination: "indexeddb_transaction",
      ...(this.lastOperation ? { lastOperation: this.lastOperation } : {}),
    };
  }

  private async evictOverBudget(): Promise<QuotaEvictionResult> {
    const rows = await this.db.getAllFromIndex(DEFAULT_STORE_NAME, "namespace", this.options.namespace) as StoredKVSwapBlock[];
    const quota = planQuotaEviction(rows, this.options.maxBlocks, this.options.maxBytes);
    for (const blockId of quota.evictedBlockIds) {
      const row = rows.find((candidate) => candidate.id === blockId);
      if (row) await this.db.delete(DEFAULT_STORE_NAME, row.storageKey);
    }
    return quota;
  }

  private async record(operation: KVSwapPersistenceOperation, blockIds: string[], ok: boolean, reason?: string): Promise<KVSwapPersistenceTraceEvent> {
    const usageBytes = await this.namespaceUsageBytes();
    const estimate = await storageEstimate();
    const event = makeTraceEvent({
      operation,
      mode: this.mode,
      namespace: this.options.namespace,
      blockIds,
      bytes: usageBytes,
      usageBytes: estimate.usage ?? usageBytes,
      quotaBytes: estimate.quota ?? this.options.maxBytes,
      ok,
      ...(reason !== undefined ? { reason } : {}),
    });
    this.lastOperation = event;
    return event;
  }

  private async namespaceUsageBytes(): Promise<number> {
    const rows = await this.db.getAllFromIndex(DEFAULT_STORE_NAME, "namespace", this.options.namespace) as StoredKVSwapBlock[];
    return rows.reduce((sum, row) => sum + row.byteLength, 0);
  }

  private async validateStoredRows(rows: StoredKVSwapBlock[]): Promise<{ records: SerializedKVSwapBlock[]; quarantined: string[] }> {
    const records: SerializedKVSwapBlock[] = [];
    const quarantined: string[] = [];
    for (const row of rows) {
      const validation = validateSerializedBlock(stripStorageKey(row));
      if (validation.ok) {
        records.push(validation.block);
      } else {
        quarantined.push(`${row.id ?? row.storageKey}:${validation.reason}`);
        await this.db.delete(DEFAULT_STORE_NAME, row.storageKey);
      }
    }
    return { records, quarantined };
  }
}

class BinaryOpfsKVSwapPersistence implements KVSwapPersistenceStore {
  readonly mode: KVSwapPersistenceMode = "opfs";
  private lastOperation: KVSwapPersistenceTraceEvent | undefined;
  private totalBytesRead = 0;
  private totalBytesWritten = 0;
  private lastLockWaitMs = 0;
  private lastWebLocks = false;

  private constructor(
    private readonly options: Required<BrowserKVSwapPersistenceOptions>,
    private readonly directory: FileSystemDirectoryHandle,
    private readonly locks?: KVSwapLockManager,
    private readonly singleWorkerRoute = false,
  ) {}

  static async create(
    options: BrowserKVSwapPersistenceOptions,
    storage: StorageManager,
    locks?: KVSwapLockManager,
    singleWorkerRoute = false,
  ): Promise<BinaryOpfsKVSwapPersistence> {
    if (!locks && !singleWorkerRoute) throw new Error("opfs_binary_requires_coordination");
    const root = await storage.getDirectory();
    const directory = await root.getDirectoryHandle(DEFAULT_OPFS_DIR, { create: true });
    await assertSyncAccessHandle(directory, normalizeNamespace(options.namespace));
    return new BinaryOpfsKVSwapPersistence(withDefaults(options), directory, locks, singleWorkerRoute);
  }

  async save(block: SerializedKVSwapBlock): Promise<KVSwapPersistenceTraceEvent> {
    return await this.persist([block]);
  }

  async load(namespace: string, blockId: string): Promise<{ block: SerializedKVSwapBlock | null; event: KVSwapPersistenceTraceEvent }> {
    const normalized = normalizeNamespace(namespace);
    let block: SerializedKVSwapBlock | null = null;
    let ok = true;
    let reason: string | undefined;
    let blockIds: string[] = [];
    let bytesRead = 0;

    const coordination = await this.withCoordination(async () => {
      try {
        const handle = await this.directory.getFileHandle(binaryFileName(normalized, blockId));
        const read = await readSyncFile(handle);
        bytesRead += read.bytesRead;
        const decoded = decodeKVSwapBlockBinary(read.bytes);
        const validation = decoded.ok ? validateSerializedBlock(decoded.block) : { ok: false as const, reason: decoded.reason };
        if (!validation.ok) {
          await this.directory.removeEntry(binaryFileName(normalized, blockId)).catch(() => undefined);
          ok = false;
          reason = `quarantined:${validation.reason}`;
          return;
        }
        block = validation.block;
        blockIds = [validation.block.id];
      } catch (error) {
        if (isNotFoundError(error)) return;
        if (isBinaryRecordError(error)) {
          await this.directory.removeEntry(binaryFileName(normalized, blockId)).catch(() => undefined);
          ok = false;
          reason = `quarantined:${error.message}`;
          return;
        }
        throw error;
      }
    });

    const event = await this.record("load", blockIds, ok, reason, {
      ...coordination,
      bytesRead,
    });
    return { block, event };
  }

  async delete(namespace: string, blockId: string): Promise<KVSwapPersistenceTraceEvent> {
    const normalized = normalizeNamespace(namespace);
    const coordination = await this.withCoordination(async () => {
      await this.directory.removeEntry(binaryFileName(normalized, blockId)).catch(() => undefined);
    });
    return await this.record("delete", [blockId], true, undefined, coordination);
  }

  async list(namespace: string): Promise<KVSwapPersistenceListResult> {
    const normalized = normalizeNamespace(namespace);
    const records: SerializedKVSwapBlock[] = [];
    const quarantined: string[] = [];
    let bytesRead = 0;

    const coordination = await this.withCoordination(async () => {
      for await (const [name, handle] of opfsEntries(this.directory)) {
        if (handle.kind !== "file" || !name.startsWith(`${normalized}__`) || !name.endsWith(".bin")) continue;
        try {
          const read = await readSyncFile(handle as FileSystemFileHandle);
          bytesRead += read.bytesRead;
          const decoded = decodeKVSwapBlockBinary(read.bytes);
          const validation = decoded.ok ? validateSerializedBlock(decoded.block) : { ok: false as const, reason: decoded.reason };
          if (validation.ok) {
            records.push(validation.block);
          } else {
            quarantined.push(`${name}:${validation.reason}`);
            await this.directory.removeEntry(name).catch(() => undefined);
          }
        } catch (error) {
          if (!isBinaryRecordError(error)) throw error;
          quarantined.push(`${name}:${error.message}`);
          await this.directory.removeEntry(name).catch(() => undefined);
        }
      }
    });

    const event = await this.record(
      "list",
      records.map((candidate) => candidate.id),
      quarantined.length === 0,
      quarantined.length ? `quarantined:${quarantined.join(",")}` : undefined,
      {
        ...coordination,
        bytesRead,
      },
    );
    return { records, metadata: records.map(toMetadata), event };
  }

  async persist(blocks: SerializedKVSwapBlock[]): Promise<KVSwapPersistenceTraceEvent> {
    let bytesWritten = 0;
    const coordination = await this.withCoordination(async () => {
      for (const block of blocks) {
        const namespacedBlock = { ...block, namespace: this.options.namespace };
        const bytes = encodeKVSwapBlockBinary(namespacedBlock);
        const handle = await this.directory.getFileHandle(binaryFileName(this.options.namespace, block.id), { create: true });
        bytesWritten += await writeSyncFile(handle, bytes);
      }
    });
    const quota = await this.evictOverBudget();
    return await this.record(
      "persist",
      [...blocks.map((block) => block.id), ...quota.evictedBlockIds],
      quota.ok,
      quota.reason,
      {
        ...coordination,
        bytesWritten,
      },
    );
  }

  async hydrate(namespace: string): Promise<{ blocks: SerializedKVSwapBlock[]; event: KVSwapPersistenceTraceEvent }> {
    const listed = await this.list(namespace);
    const event = await this.record(
      "hydrate",
      listed.records.map((block) => block.id),
      listed.event.ok,
      listed.event.reason,
      {
        webLocks: listed.event.webLocks ?? false,
        lockWaitMs: listed.event.lockWaitMs ?? 0,
        bytesRead: listed.event.bytesRead ?? 0,
        tabCoordination: listed.event.tabCoordination ?? this.defaultTabCoordination(),
        countBytes: false,
      },
    );
    return { blocks: listed.records, event };
  }

  async evict(namespace: string, blockIds: string[]): Promise<KVSwapPersistenceTraceEvent> {
    const normalized = normalizeNamespace(namespace);
    const coordination = await this.withCoordination(async () => {
      for (const blockId of blockIds) {
        await this.directory.removeEntry(binaryFileName(normalized, blockId)).catch(() => undefined);
      }
    });
    return await this.record("evict", blockIds, true, undefined, coordination);
  }

  async clear(namespace: string): Promise<KVSwapPersistenceTraceEvent> {
    const normalized = normalizeNamespace(namespace);
    const listed = await this.list(normalized);
    const coordination = await this.withCoordination(async () => {
      for await (const [name, handle] of opfsEntries(this.directory)) {
        if (handle.kind !== "file" || !name.startsWith(`${normalized}__`) || !name.endsWith(".bin")) continue;
        await this.directory.removeEntry(name).catch(() => undefined);
      }
    });
    return await this.record(
      "clear",
      listed.records.map((block) => block.id),
      listed.event.ok,
      listed.event.reason,
      {
        ...coordination,
        bytesRead: listed.event.bytesRead ?? 0,
        countBytes: false,
      },
    );
  }

  health(): KVSwapPersistenceHealth {
    return {
      enabled: true,
      mode: this.mode,
      detail: "opfs-sync-binary",
      namespace: this.options.namespace,
      decodeReuse: false,
      binary: true,
      syncAccessHandle: true,
      webLocks: this.lastWebLocks || Boolean(this.locks),
      lockWaitMs: this.lastLockWaitMs,
      bytesRead: this.totalBytesRead,
      bytesWritten: this.totalBytesWritten,
      tabCoordination: this.defaultTabCoordination(),
      ...(this.lastOperation ? { lastOperation: this.lastOperation } : {}),
    };
  }

  private async evictOverBudget(): Promise<QuotaEvictionResult> {
    const { blocks } = await this.hydrate(this.options.namespace);
    const quota = planQuotaEviction(blocks, this.options.maxBlocks, this.options.maxBytes);
    const evictBlockIds = new Set(quota.evictedBlockIds);
    if (evictBlockIds.size === 0) return quota;
    await this.withCoordination(async () => {
      for (const block of blocks) {
        if (!evictBlockIds.has(block.id)) continue;
        await this.directory.removeEntry(binaryFileName(this.options.namespace, block.id)).catch(() => undefined);
      }
    });
    return quota;
  }

  private async record(
    operation: KVSwapPersistenceOperation,
    blockIds: string[],
    ok: boolean,
    reason?: string,
    detail: BinaryOpfsTraceDetail = {},
  ): Promise<KVSwapPersistenceTraceEvent> {
    const bytesRead = detail.bytesRead ?? 0;
    const bytesWritten = detail.bytesWritten ?? 0;
    if (detail.countBytes !== false) {
      this.totalBytesRead += bytesRead;
      this.totalBytesWritten += bytesWritten;
    }
    this.lastLockWaitMs = detail.lockWaitMs ?? 0;
    this.lastWebLocks = detail.webLocks ?? false;

    const estimate = await storageEstimate();
    const event = makeTraceEvent({
      operation,
      mode: this.mode,
      detail: "opfs-sync-binary",
      namespace: this.options.namespace,
      blockIds,
      bytes: estimate.usage ?? 0,
      ...(estimate.usage !== undefined ? { usageBytes: estimate.usage } : {}),
      quotaBytes: estimate.quota ?? this.options.maxBytes,
      ok,
      ...(reason !== undefined ? { reason } : {}),
      binary: true,
      syncAccessHandle: true,
      webLocks: detail.webLocks ?? false,
      lockWaitMs: detail.lockWaitMs ?? 0,
      bytesRead,
      bytesWritten,
      tabCoordination: detail.tabCoordination ?? this.defaultTabCoordination(),
    });
    this.lastOperation = event;
    return event;
  }

  private async withCoordination<T>(operation: () => T | Promise<T>): Promise<BinaryOpfsTraceDetail> {
    if (!this.locks) {
      if (!this.singleWorkerRoute) throw new Error("opfs_binary_requires_coordination");
      await operation();
      return { webLocks: false, lockWaitMs: 0, tabCoordination: "single_worker_route" };
    }

    const startedAt = nowMs();
    let acquiredAt = startedAt;
    await this.locks.request(`kvswap:${this.options.namespace}`, { mode: "exclusive" }, async () => {
      acquiredAt = nowMs();
      await operation();
    });
    return {
      webLocks: true,
      lockWaitMs: Math.max(0, acquiredAt - startedAt),
      tabCoordination: "web_locks",
    };
  }

  private defaultTabCoordination(): KVSwapTabCoordination {
    if (this.locks) return "web_locks";
    return this.singleWorkerRoute ? "single_worker_route" : "async_opfs_fallback";
  }
}

class OpfsKVSwapPersistence implements KVSwapPersistenceStore {
  readonly mode: KVSwapPersistenceMode = "opfs";
  private lastOperation: KVSwapPersistenceTraceEvent | undefined;

  private constructor(
    private readonly options: Required<BrowserKVSwapPersistenceOptions>,
    private readonly directory: FileSystemDirectoryHandle,
  ) {}

  static async create(options: BrowserKVSwapPersistenceOptions, storage: StorageManager): Promise<OpfsKVSwapPersistence> {
    const root = await storage.getDirectory();
    const directory = await root.getDirectoryHandle(DEFAULT_OPFS_DIR, { create: true });
    return new OpfsKVSwapPersistence(withDefaults(options), directory);
  }

  async save(block: SerializedKVSwapBlock): Promise<KVSwapPersistenceTraceEvent> {
    return await this.persist([block]);
  }

  async load(namespace: string, blockId: string): Promise<{ block: SerializedKVSwapBlock | null; event: KVSwapPersistenceTraceEvent }> {
    const normalized = normalizeNamespace(namespace);
    try {
      const handle = await this.directory.getFileHandle(fileName(normalized, blockId));
      const file = await handle.getFile();
      const parsed = safeJsonParse(await file.text());
      const validation = parsed.ok ? validateSerializedBlock(parsed.value) : { ok: false as const, reason: parsed.reason };
      if (!validation.ok) {
        await this.directory.removeEntry(fileName(normalized, blockId)).catch(() => undefined);
        const event = await this.record("load", [], false, `quarantined:${validation.reason}`);
        return { block: null, event };
      }
      const block = validation.block;
      const event = await this.record("load", [block.id], true);
      return { block, event };
    } catch (error) {
      if (isNotFoundError(error)) {
        const event = await this.record("load", [], true);
        return { block: null, event };
      }
      throw error;
    }
  }

  async delete(namespace: string, blockId: string): Promise<KVSwapPersistenceTraceEvent> {
    const normalized = normalizeNamespace(namespace);
    await this.directory.removeEntry(fileName(normalized, blockId)).catch(() => undefined);
    return await this.record("delete", [blockId], true);
  }

  async list(namespace: string): Promise<KVSwapPersistenceListResult> {
    const normalized = normalizeNamespace(namespace);
    const records: SerializedKVSwapBlock[] = [];
    const quarantined: string[] = [];
    for await (const [, handle] of opfsEntries(this.directory)) {
      if (handle.kind !== "file" || !handle.name.startsWith(`${normalized}__`)) continue;
      const file = await (handle as FileSystemFileHandle).getFile();
      const parsed = safeJsonParse(await file.text());
      const validation = parsed.ok ? validateSerializedBlock(parsed.value) : { ok: false as const, reason: parsed.reason };
      if (validation.ok) {
        records.push(validation.block);
      } else {
        quarantined.push(`${handle.name}:${validation.reason}`);
        await this.directory.removeEntry(handle.name).catch(() => undefined);
      }
    }
    const event = await this.record(
      "list",
      records.map((block) => block.id),
      quarantined.length === 0,
      quarantined.length ? `quarantined:${quarantined.join(",")}` : undefined,
    );
    return { records, metadata: records.map(toMetadata), event };
  }

  async persist(blocks: SerializedKVSwapBlock[]): Promise<KVSwapPersistenceTraceEvent> {
    for (const block of blocks) {
      const handle = await this.directory.getFileHandle(fileName(this.options.namespace, block.id), { create: true });
      const writable = await handle.createWritable();
      await writable.write(JSON.stringify({ ...block, namespace: this.options.namespace }));
      await writable.close();
    }
    const quota = await this.evictOverBudget();
    return await this.record(
      "persist",
      [...blocks.map((block) => block.id), ...quota.evictedBlockIds],
      quota.ok,
      quota.reason,
    );
  }

  async hydrate(namespace: string): Promise<{ blocks: SerializedKVSwapBlock[]; event: KVSwapPersistenceTraceEvent }> {
    const listed = await this.list(namespace);
    const event = await this.record(
      "hydrate",
      listed.records.map((block) => block.id),
      listed.event.ok,
      listed.event.reason,
    );
    return { blocks: listed.records, event };
  }

  async evict(namespace: string, blockIds: string[]): Promise<KVSwapPersistenceTraceEvent> {
    for (const blockId of blockIds) {
      await this.delete(namespace, blockId);
    }
    return await this.record("evict", blockIds, true);
  }

  async clear(namespace: string): Promise<KVSwapPersistenceTraceEvent> {
    const normalized = normalizeNamespace(namespace);
    const listed = await this.list(normalized);
    for await (const [name, handle] of opfsEntries(this.directory)) {
      if (handle.kind !== "file" || !name.startsWith(`${normalized}__`)) continue;
      await this.directory.removeEntry(name).catch(() => undefined);
    }
    return await this.record(
      "clear",
      listed.records.map((block) => block.id),
      listed.event.ok,
      listed.event.reason,
    );
  }

  health(): KVSwapPersistenceHealth {
    return {
      enabled: true,
      mode: this.mode,
      detail: "opfs-json-async",
      namespace: this.options.namespace,
      decodeReuse: false,
      binary: false,
      syncAccessHandle: false,
      webLocks: false,
      lockWaitMs: 0,
      bytesRead: this.lastOperation?.bytesRead ?? 0,
      bytesWritten: this.lastOperation?.bytesWritten ?? 0,
      tabCoordination: "async_opfs_fallback",
      ...(this.lastOperation ? { lastOperation: this.lastOperation } : {}),
    };
  }

  private async evictOverBudget(): Promise<QuotaEvictionResult> {
    const { blocks } = await this.hydrate(this.options.namespace);
    const quota = planQuotaEviction(blocks, this.options.maxBlocks, this.options.maxBytes);
    for (const blockId of quota.evictedBlockIds) {
      const block = blocks.find((candidate) => candidate.id === blockId);
      if (!block) continue;
      await this.directory.removeEntry(fileName(this.options.namespace, block.id)).catch(() => undefined);
    }
    return quota;
  }

  private async record(operation: KVSwapPersistenceOperation, blockIds: string[], ok: boolean, reason?: string): Promise<KVSwapPersistenceTraceEvent> {
    const estimate = await storageEstimate();
    const event = makeTraceEvent({
      operation,
      mode: this.mode,
      namespace: this.options.namespace,
      blockIds,
      bytes: estimate.usage ?? 0,
      ...(estimate.usage !== undefined ? { usageBytes: estimate.usage } : {}),
      quotaBytes: estimate.quota ?? this.options.maxBytes,
      ok,
      ...(reason !== undefined ? { reason } : {}),
    });
    this.lastOperation = event;
    return event;
  }
}

type StoredKVSwapBlock = SerializedKVSwapBlock & { storageKey: string };

interface QuotaEvictionResult {
  ok: boolean;
  evictedBlockIds: string[];
  reason?: string;
}

interface BinaryOpfsTraceDetail {
  webLocks?: boolean;
  lockWaitMs?: number;
  bytesRead?: number;
  bytesWritten?: number;
  tabCoordination?: KVSwapTabCoordination;
  countBytes?: boolean;
}

interface SyncAccessHandleLike {
  getSize(): number;
  read(buffer: Uint8Array, options?: { at?: number }): number;
  write(buffer: Uint8Array, options?: { at?: number }): number;
  truncate(size: number): void;
  flush(): void;
  close(): void;
}

type SyncCapableFileHandle = FileSystemFileHandle & {
  createSyncAccessHandle?: () => SyncAccessHandleLike | Promise<SyncAccessHandleLike>;
};

function stripStorageKey(row: StoredKVSwapBlock): SerializedKVSwapBlock {
  const { storageKey: _storageKey, ...block } = row;
  return block;
}

function validateSerializedBlock(value: unknown): { ok: true; block: SerializedKVSwapBlock } | { ok: false; reason: string } {
  if (!isRecord(value)) return { ok: false, reason: "record_not_object" };
  if (value.version !== KV_SWAP_STORAGE_VERSION) return { ok: false, reason: `unsupported_version:${String(value.version)}` };
  const requiredStrings = ["namespace", "id", "modelId", "requestId", "createdAt", "updatedAt"] as const;
  for (const key of requiredStrings) {
    if (typeof value[key] !== "string" || value[key].length === 0) return { ok: false, reason: `invalid_${key}` };
  }
  const requiredNumbers = ["layer", "startToken", "endToken", "importance", "estimatedBytes", "lastAccessAt", "byteLength"] as const;
  for (const key of requiredNumbers) {
    if (typeof value[key] !== "number" || !Number.isFinite(value[key])) return { ok: false, reason: `invalid_${key}` };
  }
  const layer = value.layer as number;
  const startToken = value.startToken as number;
  const endToken = value.endToken as number;
  const importance = value.importance as number;
  const estimatedBytes = value.estimatedBytes as number;
  const lastAccessAt = value.lastAccessAt as number;
  const byteLength = value.byteLength as number;
  if (!Number.isInteger(layer) || layer < 0) return { ok: false, reason: "invalid_layer" };
  if (!Number.isInteger(startToken) || startToken < 0) return { ok: false, reason: "invalid_startToken" };
  if (!Number.isInteger(endToken) || endToken < 0 || endToken < startToken) return { ok: false, reason: "invalid_endToken" };
  if (importance < 0) return { ok: false, reason: "invalid_importance" };
  if (estimatedBytes < 0) return { ok: false, reason: "invalid_estimatedBytes" };
  if (lastAccessAt < 0) return { ok: false, reason: "invalid_lastAccessAt" };
  if (byteLength < 0) return { ok: false, reason: "invalid_byteLength" };
  if (typeof value.pinned !== "boolean") return { ok: false, reason: "invalid_pinned" };
  if (value.phase !== undefined && value.phase !== "prefill" && value.phase !== "decode") return { ok: false, reason: "invalid_phase" };
  if (value.runtimeBlockId !== undefined && typeof value.runtimeBlockId !== "string") return { ok: false, reason: "invalid_runtimeBlockId" };
  if (value.modelFingerprint !== undefined && typeof value.modelFingerprint !== "string") return { ok: false, reason: "invalid_modelFingerprint" };
  if (value.promptTokenHash !== undefined && typeof value.promptTokenHash !== "string") return { ok: false, reason: "invalid_promptTokenHash" };
  if (value.policyHash !== undefined && typeof value.policyHash !== "string") return { ok: false, reason: "invalid_policyHash" };
  if (value.promptTokenIds !== undefined && !isIntegerArray(value.promptTokenIds)) return { ok: false, reason: "invalid_promptTokenIds" };
  if (value.prefillTokenCount !== undefined && !isNonNegativeInteger(value.prefillTokenCount)) return { ok: false, reason: "invalid_prefillTokenCount" };
  if (value.runtimeLayerCount !== undefined && !isNonNegativeInteger(value.runtimeLayerCount)) return { ok: false, reason: "invalid_runtimeLayerCount" };
  if (value.promptTokenIds !== undefined && value.prefillTokenCount !== undefined && value.promptTokenIds.length !== value.prefillTokenCount) {
    return { ok: false, reason: "prompt_prefill_count_mismatch" };
  }
  if (value.tokenIds !== undefined && !isIntegerArray(value.tokenIds)) return { ok: false, reason: "invalid_tokenIds" };
  if (value.queryRows !== undefined && !isNumberMatrix(value.queryRows)) return { ok: false, reason: "invalid_queryRows" };
  if (!isNumberMatrix(value.keyRows)) return { ok: false, reason: "invalid_keyRows" };
  if (!isNumberMatrix(value.valueRows)) return { ok: false, reason: "invalid_valueRows" };
  if (value.keyRows.length !== value.valueRows.length) return { ok: false, reason: "kv_row_mismatch" };
  if (value.queryRows !== undefined && value.queryRows.length !== value.keyRows.length) return { ok: false, reason: "query_key_row_mismatch" };
  if (value.hiddenRows !== undefined && !isNumberMatrix(value.hiddenRows)) return { ok: false, reason: "invalid_hiddenRows" };
  if (value.hiddenRows !== undefined && value.hiddenRows.length !== value.keyRows.length) return { ok: false, reason: "hidden_key_row_mismatch" };
  if (value.tokenIds !== undefined && value.tokenIds.length !== value.keyRows.length) return { ok: false, reason: "token_key_row_mismatch" };
  if (value.compressedKeySummary !== undefined && typeof value.compressedKeySummary !== "string" && !isNumberArray(value.compressedKeySummary)) {
    return { ok: false, reason: "invalid_compressedKeySummary" };
  }
  if (value.lowRankKeySummary !== undefined && !isSerializedLowRankKeySummary(value.lowRankKeySummary)) {
    return { ok: false, reason: "invalid_lowRankKeySummary" };
  }
  if (value.prefillProof !== undefined && !isSerializedPrefillProof(value.prefillProof)) {
    return { ok: false, reason: "invalid_prefillProof" };
  }
  if (value.summaryRank !== undefined && (typeof value.summaryRank !== "number" || !Number.isFinite(value.summaryRank))) {
    return { ok: false, reason: "invalid_summaryRank" };
  }
  if (value.summaryRank !== undefined && (!Number.isInteger(value.summaryRank) || value.summaryRank < 0)) {
    return { ok: false, reason: "invalid_summaryRank" };
  }
  return { ok: true, block: value as unknown as SerializedKVSwapBlock };
}

function isNonNegativeInteger(value: unknown): boolean {
  return typeof value === "number" && Number.isInteger(value) && value >= 0;
}

function isSerializedPrefillProof(value: unknown): value is SerializedKVSwapPrefillProof {
  if (!isRecord(value) || !Array.isArray(value.layers)) return false;
  if (!value.layers.every(isSerializedPrefillLayerProof)) return false;
  if (value.prefillChunkCount !== undefined && !isNonNegativeInteger(value.prefillChunkCount)) return false;
  if (value.prefillChunkSize !== undefined && !isNonNegativeInteger(value.prefillChunkSize)) return false;
  if (value.shapeBucket !== undefined && typeof value.shapeBucket !== "string") return false;
  if (value.pipelineCacheKey !== undefined && typeof value.pipelineCacheKey !== "string") return false;
  if (value.prefillDispatchTargetMs !== undefined && (typeof value.prefillDispatchTargetMs !== "number" || !Number.isFinite(value.prefillDispatchTargetMs))) return false;
  if (value.maxDispatchEstimatedMs !== undefined && (typeof value.maxDispatchEstimatedMs !== "number" || !Number.isFinite(value.maxDispatchEstimatedMs))) return false;
  if (value.prefillChunkDispatch !== undefined && !isPrefillChunkDispatch(value.prefillChunkDispatch)) return false;
  if (value.attentionDispatchCount !== undefined && !isNonNegativeInteger(value.attentionDispatchCount)) return false;
  if (value.awaitedDispatchBreaks !== undefined && !isNonNegativeInteger(value.awaitedDispatchBreaks)) return false;
  return true;
}

function isSerializedPrefillLayerProof(value: unknown): value is SerializedKVSwapPrefillLayerProof {
  if (!isRecord(value)) return false;
  if (!isNonNegativeInteger(value.layerIndex)) return false;
  if (!isSerializedKernelBackend(value.qProjection)) return false;
  if (!isSerializedKernelBackend(value.kProjection)) return false;
  if (!isSerializedKernelBackend(value.vProjection)) return false;
  if (!isSerializedKernelBackend(value.oProjection)) return false;
  if (value.mlpBackend !== undefined && !isSerializedKernelBackend(value.mlpBackend)) return false;
  if (value.mlpActivationKind !== undefined && typeof value.mlpActivationKind !== "string") return false;
  if (value.mlpRowCount !== undefined && !isNonNegativeInteger(value.mlpRowCount)) return false;
  if (!isSerializedKernelBackend(value.attentionBackend)) return false;
  if (!Array.isArray(value.packedHeadBackends) || !value.packedHeadBackends.every(isSerializedKernelBackend)) return false;
  if (!isNonNegativeInteger(value.packedHeadCount)) return false;
  if (value.selectedKeyRows !== undefined && !isNonNegativeInteger(value.selectedKeyRows)) return false;
  if (value.prefillChunkDispatch !== undefined && !isPrefillChunkDispatch(value.prefillChunkDispatch)) return false;
  if (value.attentionDispatchCount !== undefined && !isNonNegativeInteger(value.attentionDispatchCount)) return false;
  if (value.awaitedDispatchBreaks !== undefined && !isNonNegativeInteger(value.awaitedDispatchBreaks)) return false;
  return true;
}

function isSerializedKernelBackend(value: unknown): value is SerializedKVSwapKernelBackend {
  return value === "webgpu" || value === "cpu_reference" || value === "mixed";
}

function isPrefillChunkDispatch(value: unknown): value is "single_dispatch" | "chunked_dispatch" {
  return value === "single_dispatch" || value === "chunked_dispatch";
}

function planQuotaEviction(
  blocks: Array<Pick<SerializedKVSwapBlock, "id" | "pinned" | "lastAccessAt" | "byteLength">>,
  maxBlocks: number,
  maxBytes: number,
): QuotaEvictionResult {
  const evictedBlockIds: string[] = [];
  const remaining = [...blocks];
  const evictable = remaining
    .filter((block) => !block.pinned)
    .sort((a, b) => a.lastAccessAt - b.lastAccessAt);

  while (isOverQuota(remaining, maxBlocks, maxBytes) && evictable.length > 0) {
    const block = evictable.shift();
    if (!block) break;
    evictedBlockIds.push(block.id);
    const index = remaining.findIndex((candidate) => candidate.id === block.id);
    if (index >= 0) remaining.splice(index, 1);
  }

  const over = quotaOverage(remaining, maxBlocks, maxBytes);
  const reason = over.length > 0 ? `protected_over_quota:${over.join("+")}` : undefined;
  return {
    ok: over.length === 0,
    evictedBlockIds,
    ...(reason ? { reason } : {}),
  };
}

function isOverQuota(
  blocks: Array<Pick<SerializedKVSwapBlock, "byteLength">>,
  maxBlocks: number,
  maxBytes: number,
): boolean {
  return quotaOverage(blocks, maxBlocks, maxBytes).length > 0;
}

function quotaOverage(
  blocks: Array<Pick<SerializedKVSwapBlock, "byteLength">>,
  maxBlocks: number,
  maxBytes: number,
): string[] {
  const over: string[] = [];
  if (blocks.length > maxBlocks) over.push("blocks");
  const bytes = blocks.reduce((sum, block) => sum + Math.max(0, block.byteLength), 0);
  if (bytes > maxBytes) over.push("bytes");
  return over;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function isNumberArray(value: unknown): value is number[] {
  return Array.isArray(value) && value.every((item) => typeof item === "number" && Number.isFinite(item));
}

function isNumberMatrix(value: unknown): value is number[][] {
  return Array.isArray(value) && value.every(isNumberArray);
}

function isIntegerArray(value: unknown): value is number[] {
  return Array.isArray(value) && value.every((item) => Number.isInteger(item) && item >= 0);
}

function isSerializedLowRankKeySummary(value: unknown): value is SerializedKVLowRankKeySummary {
  if (!isRecord(value)) return false;
  return typeof value.blockId === "string"
    && value.blockId.length > 0
    && typeof value.rank === "number"
    && Number.isInteger(value.rank)
    && value.rank >= 0
    && typeof value.projectionId === "string"
    && value.projectionId.length > 0
    && typeof value.layer === "number"
    && Number.isInteger(value.layer)
    && value.layer >= 0
    && typeof value.headGroupId === "string"
    && value.headGroupId.length > 0
    && typeof value.checksum === "string"
    && value.checksum.length > 0
    && typeof value.qualityScore === "number"
    && Number.isFinite(value.qualityScore)
    && value.qualityScore >= 0
    && value.qualityScore <= 1
    && isNumberArray(value.values)
    && value.values.length === value.rank;
}

function safeJsonParse(text: string): { ok: true; value: unknown } | { ok: false; reason: string } {
  try {
    return { ok: true, value: JSON.parse(text) as unknown };
  } catch {
    return { ok: false, reason: "corrupt_json" };
  }
}

function toMetadata(block: SerializedKVSwapBlock): KVSwapPersistedBlockMetadata {
  return {
    version: block.version,
    namespace: block.namespace,
    id: block.id,
    modelId: block.modelId,
    requestId: block.requestId,
    layer: block.layer,
    startToken: block.startToken,
    endToken: block.endToken,
    pinned: block.pinned,
    importance: block.importance,
    estimatedBytes: block.estimatedBytes,
    ...(block.checksum ? { checksum: block.checksum } : {}),
    ...(block.summaryRank !== undefined ? { summaryRank: block.summaryRank } : {}),
    createdAt: block.createdAt,
    updatedAt: block.updatedAt,
    lastAccessAt: block.lastAccessAt,
    byteLength: block.byteLength,
  };
}

function makeTraceEvent(input: Omit<KVSwapPersistenceTraceEvent, "at">): KVSwapPersistenceTraceEvent {
  const binary = input.binary ?? false;
  const syncAccessHandle = input.syncAccessHandle ?? false;
  const webLocks = input.webLocks ?? false;
  return {
    ...input,
    detail: input.detail ?? defaultTraceDetail(input.mode, binary, syncAccessHandle),
    binary,
    syncAccessHandle,
    webLocks,
    lockWaitMs: input.lockWaitMs ?? 0,
    bytesRead: input.bytesRead ?? 0,
    bytesWritten: input.bytesWritten ?? 0,
    tabCoordination: input.tabCoordination ?? defaultTabCoordination(input.mode, binary),
    at: new Date().toISOString(),
  };
}

function defaultTraceDetail(mode: KVSwapPersistenceMode, binary: boolean, syncAccessHandle: boolean): string {
  if (mode === "opfs" && binary && syncAccessHandle) return "opfs-sync-binary";
  if (mode === "opfs") return "opfs-json-async";
  if (mode === "indexeddb") return "indexeddb-transaction";
  if (mode === "memory") return "memory-local";
  return "disabled";
}

function defaultTabCoordination(mode: KVSwapPersistenceMode, binary: boolean): KVSwapTabCoordination {
  if (mode === "opfs" && binary) return "single_worker_route";
  if (mode === "opfs") return "async_opfs_fallback";
  if (mode === "indexeddb") return "indexeddb_transaction";
  if (mode === "memory") return "memory_local";
  return "none";
}

function withDefaults(options: BrowserKVSwapPersistenceOptions): Required<BrowserKVSwapPersistenceOptions> {
  return {
    namespace: normalizeNamespace(options.namespace),
    preferOpfs: options.preferOpfs ?? true,
    maxBlocks: options.maxBlocks ?? 512,
    maxBytes: options.maxBytes ?? 256 * 1024 * 1024,
  };
}

export function normalizeKVSwapNamespace(namespace: string): string {
  const normalized = namespace.trim().replace(/[^a-zA-Z0-9._-]/g, "_").slice(0, 160);
  return normalized || "default";
}

function normalizeNamespace(namespace: string): string {
  return normalizeKVSwapNamespace(namespace);
}

function storageKey(namespace: string, blockId: string): string {
  return `${normalizeNamespace(namespace)}:${blockId}`;
}

function fileName(namespace: string, blockId: string): string {
  return `${normalizeNamespace(namespace)}__${stableBlockFileId(blockId)}.json`;
}

function binaryFileName(namespace: string, blockId: string): string {
  return `${normalizeNamespace(namespace)}__${stableBlockFileId(blockId)}.bin`;
}

function stableBlockFileId(blockId: string): string {
  const bytes = new TextEncoder().encode(blockId);
  let binary = "";
  for (const byte of bytes) binary += String.fromCharCode(byte);
  return btoa(binary).replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/g, "");
}

async function assertSyncAccessHandle(directory: FileSystemDirectoryHandle, namespace: string): Promise<void> {
  const probeName = binaryFileName(namespace, "__sync_access_probe__");
  const handle = await directory.getFileHandle(probeName, { create: true }) as SyncCapableFileHandle;
  try {
    const access = await openSyncAccessHandle(handle);
    access.close();
  } finally {
    await directory.removeEntry(probeName).catch(() => undefined);
  }
}

async function readSyncFile(handle: FileSystemFileHandle): Promise<{ bytes: Uint8Array; bytesRead: number }> {
  const access = await openSyncAccessHandle(handle as SyncCapableFileHandle);
  try {
    const size = Math.max(0, access.getSize());
    if (size > MAX_KV_SWAP_BINARY_RECORD_BYTES) {
      throw new Error(`binary_record_too_large:${size}/${MAX_KV_SWAP_BINARY_RECORD_BYTES}`);
    }
    const buffer = new Uint8Array(size);
    const bytesRead = size > 0 ? access.read(buffer, { at: 0 }) : 0;
    return { bytes: buffer.slice(0, bytesRead), bytesRead };
  } finally {
    access.close();
  }
}

async function writeSyncFile(handle: FileSystemFileHandle, bytes: Uint8Array): Promise<number> {
  const access = await openSyncAccessHandle(handle as SyncCapableFileHandle);
  try {
    access.truncate(0);
    const bytesWritten = bytes.byteLength > 0 ? access.write(bytes, { at: 0 }) : 0;
    access.truncate(bytesWritten);
    access.flush();
    return bytesWritten;
  } finally {
    access.close();
  }
}

async function openSyncAccessHandle(handle: SyncCapableFileHandle): Promise<SyncAccessHandleLike> {
  if (typeof handle.createSyncAccessHandle !== "function") {
    throw new Error("sync_access_handle_unavailable");
  }
  return await handle.createSyncAccessHandle();
}

function opfsEntries(directory: FileSystemDirectoryHandle): AsyncIterable<[string, FileSystemHandle]> {
  return (directory as FileSystemDirectoryHandle & {
    entries(): AsyncIterable<[string, FileSystemHandle]>;
  }).entries();
}

function isNotFoundError(error: unknown): boolean {
  return error instanceof DOMException && error.name === "NotFoundError";
}

function isBinaryRecordError(error: unknown): error is Error {
  return error instanceof Error && error.message.startsWith("binary_record_too_large:");
}

async function storageEstimate(): Promise<StorageEstimate> {
  try {
    return await navigator.storage?.estimate?.() ?? {};
  } catch {
    return {};
  }
}

function nowMs(): number {
  return typeof performance !== "undefined" && typeof performance.now === "function" ? performance.now() : Date.now();
}
