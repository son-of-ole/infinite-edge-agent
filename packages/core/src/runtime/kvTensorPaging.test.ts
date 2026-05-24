import { describe, expect, it } from "vitest";
import type { KVBlock, KVSwapDecision } from "./kvswap";
import {
  deserializeKVTensorBlock,
  KVTensorPagingRegistry,
  serializeKVTensorBlock,
} from "./kvTensorPaging";

describe("KVTensorPagingRegistry", () => {
  it("prefetches selected SSA blocks into VRAM before sparse attention", () => {
    const registry = new KVTensorPagingRegistry({ now: 10 });
    registry.registerBlock(makeBlock("kv_hot", "vram", { pinned: true }));
    registry.registerBlock(makeBlock("kv_selected_ram", "ram"));
    registry.registerBlock(makeBlock("kv_selected_disk", "disk"));

    const decision = makeDecision({
      pinBlockIds: ["kv_hot"],
      prefetchBlockIds: ["kv_selected_disk"],
    });
    registry.applyKVSwapDecision(decision);
    const readiness = registry.ensureBlocksAvailableForSparseAttention(["kv_selected_ram", "kv_selected_disk"]);

    expect(readiness.availableBlockIds).toEqual(["kv_selected_ram", "kv_selected_disk"]);
    expect(registry.getBlock("kv_selected_ram")?.tier).toBe("vram");
    expect(registry.getBlock("kv_selected_disk")?.tier).toBe("vram");
    expect(readiness.events.map((event) => `${event.blockId}:${event.fromTier}->${event.toTier}`)).toEqual([
      "kv_selected_ram:ram->vram",
    ]);
    expect(registry.getBlock("kv_selected_disk")?.lastAccessAt).toBe(10);
  });

  it("does not evict pinned blocks even when a pressure decision asks for them", () => {
    const registry = new KVTensorPagingRegistry({ now: 20, defaultEvictionTier: "disk" });
    registry.registerBlock(makeBlock("kv_pinned", "vram", { pinned: true }));
    registry.registerBlock(makeBlock("kv_cold", "vram", { importance: 0.01 }));

    const events = registry.applyKVSwapDecision(makeDecision({
      evictBlockIds: ["kv_pinned", "kv_cold"],
    }));

    expect(registry.getBlock("kv_pinned")?.tier).toBe("vram");
    expect(registry.getBlock("kv_cold")?.tier).toBe("disk");
    expect(events).toEqual(expect.arrayContaining([
      expect.objectContaining({ blockId: "kv_pinned", skipped: true, reason: "pinned" }),
      expect.objectContaining({ blockId: "kv_cold", fromTier: "vram", toTier: "disk" }),
    ]));
  });

  it("round-trips a paged block through disk serialization", () => {
    const block = makeBlock("kv_disk_roundtrip", "vram", {
      checksum: "sha256:test",
      tensorHandles: {
        key: { backend: "native-edge-reference", id: "key_1", dtype: "f16", shape: [2, 4], bytes: 16 },
        value: { backend: "native-edge-reference", id: "value_1", dtype: "f16", shape: [2, 4], bytes: 16 },
      },
    });
    const serialized = serializeKVTensorBlock(block, { serializedAt: 123 });
    const restored = deserializeKVTensorBlock(serialized);

    expect(restored).toEqual(block);

    const registry = new KVTensorPagingRegistry({ now: 30 });
    registry.registerBlock(block);
    registry.pageBlock("kv_disk_roundtrip", "disk");
    expect(registry.getBlock("kv_disk_roundtrip")?.tensorHandles).toBeUndefined();

    registry.pageBlock("kv_disk_roundtrip", "vram");
    expect(registry.getBlock("kv_disk_roundtrip")?.tensorHandles).toEqual(block.tensorHandles);
  });

  it("round-trips compressed Float32Array key summaries through disk serialization", () => {
    const block = makeBlock("kv_summary_roundtrip", "vram", {
      compressedKeySummary: new Float32Array([0.25, -1.5, 3]),
      summaryRank: 3,
    });

    const restored = deserializeKVTensorBlock(serializeKVTensorBlock(block, { serializedAt: 456 }));

    expect(restored.compressedKeySummary).toBeInstanceOf(Float32Array);
    expect(Array.from(restored.compressedKeySummary as Float32Array)).toEqual([0.25, -1.5, 3]);
    expect(restored.summaryRank).toBe(3);
  });
});

function makeBlock(id: string, tier: KVBlock["tier"], overrides: Partial<KVBlock> = {}): KVBlock {
  return {
    id,
    layer: 0,
    startToken: 0,
    endToken: 8,
    tier,
    pinned: false,
    importance: 0.5,
    lastAccessAt: 1,
    estimatedBytes: 512,
    tensorHandles: {
      key: { backend: "native-edge-reference", id: `${id}:key`, dtype: "f16", shape: [8, 4], bytes: 128 },
      value: { backend: "native-edge-reference", id: `${id}:value`, dtype: "f16", shape: [8, 4], bytes: 128 },
    },
    ...overrides,
  };
}

function makeDecision(overrides: Partial<KVSwapDecision>): KVSwapDecision {
  return {
    pinBlockIds: [],
    evictBlockIds: [],
    prefetchBlockIds: [],
    predictivePrefetchBlockIds: [],
    predictionReasons: {},
    prefetchConfidenceByBlockId: {},
    reasons: {},
    estimatedBytesFreed: 0,
    pressureTelemetry: {
      mode: "predictive",
      vramBytes: 0,
      ramBytes: 0,
      diskBytes: 0,
      totalBytes: 0,
      vramPressureRatio: 0,
      ramPressureRatio: 0,
      targetFreeBytes: 0,
      projectedVramBytesAfterEviction: 0,
      backendTensorHandleCount: 0,
    },
    ...overrides,
  };
}
