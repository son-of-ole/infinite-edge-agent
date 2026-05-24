import type {
  KVBackendTensorHandles,
  BackendTensorHandle,
  KVBlock,
  KVLowRankKeySummary,
  KVSwapDecision,
  KVTier,
} from "./kvswap";

export type KVTensorResidencyTier = "hot" | "warm" | "cold";

export interface KVTensorPagingRegistryOptions {
  now?: number;
  defaultEvictionTier?: Extract<KVTier, "ram" | "disk">;
}

export interface KVTensorPagingEvent {
  blockId: string;
  fromTier: KVTier;
  toTier: KVTier;
  residencyTier: KVTensorResidencyTier;
  skipped?: boolean;
  reason?: string;
  serializedBytes?: number;
}

export interface KVTensorSparseAttentionReadiness {
  availableBlockIds: string[];
  events: KVTensorPagingEvent[];
}

export interface SerializedKVTensorBlock {
  version: 1;
  serializedAt: number;
  block: SerializedKVTensorBlockPayload;
}

export type SerializedCompressedKeySummary =
  | string
  | { kind: "float32array"; values: number[] };

export type SerializedLowRankKeySummary = Omit<KVLowRankKeySummary, "values"> & {
  values: number[];
};

export type SerializedKVTensorBlockPayload = Omit<KVBlock, "compressedKeySummary" | "lowRankKeySummary"> & {
  compressedKeySummary?: SerializedCompressedKeySummary;
  lowRankKeySummary?: SerializedLowRankKeySummary;
}

export interface SerializeKVTensorBlockOptions {
  serializedAt?: number;
}

export class KVTensorPagingRegistry {
  private readonly blocks = new Map<string, KVBlock>();
  private readonly diskBlocks = new Map<string, string>();
  private readonly defaultEvictionTier: Extract<KVTier, "ram" | "disk">;
  private clock: number;

  constructor(options: KVTensorPagingRegistryOptions = {}) {
    this.clock = options.now ?? 0;
    this.defaultEvictionTier = options.defaultEvictionTier ?? "ram";
  }

  registerBlock(block: KVBlock, tensorHandles?: KVBackendTensorHandles): KVBlock {
    const registered = cloneBlock({
      ...block,
      ...(tensorHandles ? { tensorHandles } : {}),
    });
    if (registered.tier === "disk") {
      this.diskBlocks.set(registered.id, serializeKVTensorBlock(registered, { serializedAt: this.clock }));
      delete registered.tensorHandles;
    }
    this.blocks.set(registered.id, registered);
    return cloneBlock(registered);
  }

  getBlock(id: string): KVBlock | undefined {
    const block = this.blocks.get(id);
    return block ? cloneBlock(block) : undefined;
  }

  listBlocks(): KVBlock[] {
    return [...this.blocks.values()]
      .sort((a, b) => a.id.localeCompare(b.id))
      .map(cloneBlock);
  }

  getResidencyTier(id: string): KVTensorResidencyTier | undefined {
    const block = this.blocks.get(id);
    return block ? toResidencyTier(block.tier) : undefined;
  }

  pageBlock(id: string, targetTier: KVTier): KVTensorPagingEvent {
    const block = this.requireBlock(id);
    const fromTier = block.tier;

    if (block.pinned && targetTier !== "vram") {
      return {
        blockId: id,
        fromTier,
        toTier: fromTier,
        residencyTier: toResidencyTier(fromTier),
        skipped: true,
        reason: "pinned",
      };
    }

    if (fromTier === targetTier) {
      if (targetTier === "vram") block.lastAccessAt = this.clock;
      return {
        blockId: id,
        fromTier,
        toTier: targetTier,
        residencyTier: toResidencyTier(targetTier),
        skipped: true,
        reason: "already_available",
      };
    }

    let serializedBytes: number | undefined;
    if (targetTier === "disk") {
      const serialized = serializeKVTensorBlock(block, { serializedAt: this.clock });
      this.diskBlocks.set(id, serialized);
      serializedBytes = serialized.length;
      delete block.tensorHandles;
    } else if (fromTier === "disk" && !block.tensorHandles) {
      const restored = deserializeKVTensorBlock(this.requireSerializedBlock(id));
      if (restored.tensorHandles) block.tensorHandles = restored.tensorHandles;
      if (restored.checksum) block.checksum = restored.checksum;
    }

    block.tier = targetTier;
    if (targetTier === "vram") block.lastAccessAt = this.clock;

    return {
      blockId: id,
      fromTier,
      toTier: targetTier,
      residencyTier: toResidencyTier(targetTier),
      ...(serializedBytes === undefined ? {} : { serializedBytes }),
    };
  }

  applyKVSwapDecision(
    decision: KVSwapDecision,
    options: { evictionTier?: Extract<KVTier, "ram" | "disk"> } = {},
  ): KVTensorPagingEvent[] {
    const events: KVTensorPagingEvent[] = [];
    const evictionTier = options.evictionTier ?? this.defaultEvictionTier;

    for (const id of decision.pinBlockIds) {
      const block = this.blocks.get(id);
      if (block) block.pinned = true;
    }

    for (const id of decision.evictBlockIds) {
      events.push(this.pageBlock(id, evictionTier));
    }

    for (const id of decision.prefetchBlockIds) {
      const event = this.pageBlock(id, "vram");
      const predictedHotBlocks = decision.predictedHotBlocks ?? [];
      if (!event.skipped) {
        event.reason = predictedHotBlocks.some((block) => block.blockId === id && block.source === "low_rank_attention")
          ? "low_rank_predictive_prefetch"
          : (decision.predictivePrefetchBlockIds.includes(id) ? "predictive_prefetch" : "planned_prefetch");
      }
      if (!event.skipped) events.push(event);
    }

    return events;
  }

  ensureBlocksAvailableForSparseAttention(blockIds: string[]): KVTensorSparseAttentionReadiness {
    const events: KVTensorPagingEvent[] = [];
    const availableBlockIds: string[] = [];

    for (const id of blockIds) {
      const event = this.pageBlock(id, "vram");
      if (!event.skipped) events.push(event);
      const block = this.requireBlock(id);
      if (block.tier !== "vram") {
        throw new Error(`KV tensor block ${id} is not available in VRAM before sparse attention.`);
      }
      availableBlockIds.push(id);
    }

    return { availableBlockIds, events };
  }

  private requireBlock(id: string): KVBlock {
    const block = this.blocks.get(id);
    if (!block) throw new Error(`Unknown KV tensor block: ${id}`);
    return block;
  }

  private requireSerializedBlock(id: string): string {
    const serialized = this.diskBlocks.get(id);
    if (!serialized) throw new Error(`KV tensor block ${id} has no disk serialization record.`);
    return serialized;
  }
}

export function serializeKVTensorBlock(
  block: KVBlock,
  options: SerializeKVTensorBlockOptions = {},
): string {
  const record: SerializedKVTensorBlock = {
    version: 1,
    serializedAt: options.serializedAt ?? Date.now(),
    block: serializeBlockPayload(block),
  };
  return JSON.stringify(record);
}

export function deserializeKVTensorBlock(serialized: string): KVBlock {
  const parsed = JSON.parse(serialized) as Partial<SerializedKVTensorBlock>;
  if (parsed.version !== 1 || !parsed.block || typeof parsed.block.id !== "string") {
    throw new Error("Unsupported KV tensor disk block serialization format.");
  }
  return deserializeBlockPayload(parsed.block);
}

function toResidencyTier(tier: KVTier): KVTensorResidencyTier {
  if (tier === "vram") return "hot";
  if (tier === "ram") return "warm";
  return "cold";
}

function cloneBlock(block: KVBlock): KVBlock {
  return {
    ...block,
    ...(block.tensorHandles ? {
      tensorHandles: {
        ...(block.tensorHandles.key ? { key: cloneTensorHandle(block.tensorHandles.key) } : {}),
        ...(block.tensorHandles.value ? { value: cloneTensorHandle(block.tensorHandles.value) } : {}),
      },
    } : {}),
    ...(block.compressedKeySummary instanceof Float32Array ? {
      compressedKeySummary: new Float32Array(block.compressedKeySummary),
    } : {}),
    ...(block.lowRankKeySummary ? { lowRankKeySummary: cloneLowRankKeySummary(block.lowRankKeySummary) } : {}),
  };
}

function serializeBlockPayload(block: KVBlock): SerializedKVTensorBlockPayload {
  const cloned = cloneBlock(block);
  const { compressedKeySummary, lowRankKeySummary, ...rest } = cloned;
  return {
    ...rest,
    ...(compressedKeySummary instanceof Float32Array ? {
      compressedKeySummary: {
        kind: "float32array",
        values: Array.from(compressedKeySummary),
      },
    } : typeof compressedKeySummary === "string" ? {
      compressedKeySummary,
    } : {}),
    ...(lowRankKeySummary ? { lowRankKeySummary: serializeLowRankKeySummary(lowRankKeySummary) } : {}),
  };
}

function deserializeBlockPayload(payload: SerializedKVTensorBlockPayload): KVBlock {
  const { compressedKeySummary, lowRankKeySummary, ...rest } = payload;
  return cloneBlock({
    ...rest,
    ...(compressedKeySummary === undefined ? {} : {
      compressedKeySummary: deserializeCompressedKeySummary(compressedKeySummary),
    }),
    ...(lowRankKeySummary === undefined ? {} : {
      lowRankKeySummary: deserializeLowRankKeySummary(lowRankKeySummary),
    }),
  });
}

function deserializeCompressedKeySummary(summary: SerializedCompressedKeySummary): Float32Array | string {
  if (typeof summary === "string") return summary;
  if (summary.kind === "float32array" && Array.isArray(summary.values)) {
    return new Float32Array(summary.values);
  }
  throw new Error("Unsupported KV tensor compressed key summary serialization format.");
}

function serializeLowRankKeySummary(summary: KVLowRankKeySummary): SerializedLowRankKeySummary {
  return {
    ...summary,
    values: Array.from(summary.values),
  };
}

function deserializeLowRankKeySummary(summary: SerializedLowRankKeySummary): KVLowRankKeySummary {
  return cloneLowRankKeySummary({
    ...summary,
    values: new Float32Array(summary.values),
  });
}

function cloneLowRankKeySummary(summary: KVLowRankKeySummary): KVLowRankKeySummary {
  return {
    ...summary,
    values: new Float32Array(summary.values),
  };
}

function cloneTensorHandle(handle: BackendTensorHandle): BackendTensorHandle {
  return {
    ...handle,
    ...(handle.shape ? { shape: [...handle.shape] } : {}),
  };
}
