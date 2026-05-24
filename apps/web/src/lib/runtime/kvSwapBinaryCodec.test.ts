import { describe, expect, it } from "vitest";
import { KV_SWAP_STORAGE_VERSION, type SerializedKVSwapBlock } from "./kvSwapPersistence";
import {
  KV_SWAP_BINARY_CODEC_VERSION,
  KV_SWAP_BINARY_MAGIC,
  MAX_KV_SWAP_BINARY_RECORD_BYTES,
  decodeKVSwapBlockBinary,
  encodeKVSwapBlockBinary,
} from "./kvSwapBinaryCodec";

describe("KVSwap binary codec", () => {
  it("round-trips serialized KV blocks with identity, tensor rows, token ids, and summary metadata", () => {
    const block = makeBlock();

    const bytes = encodeKVSwapBlockBinary(block);
    const decoded = decodeKVSwapBlockBinary(bytes);

    expect(bytes.byteLength).toBeGreaterThan(64);
    expect(decoded).toMatchObject({ ok: true });
    if (!decoded.ok) throw new Error(decoded.reason);
    expect(decoded.block).toEqual(block);
    expect(decoded.bytesRead).toBe(bytes.byteLength);
    expect(decoded.checksum).toMatch(/^[0-9a-f]{8}$/);
  });

  it("returns a clear failure reason when checksum validation fails", () => {
    const bytes = encodeKVSwapBlockBinary(makeBlock());
    const corrupt = new Uint8Array(bytes);
    const lastIndex = corrupt.length - 1;
    corrupt[lastIndex] = (corrupt[lastIndex] ?? 0) ^ 0xff;

    const decoded = decodeKVSwapBlockBinary(corrupt);

    expect(decoded).toMatchObject({
      ok: false,
      reason: expect.stringContaining("checksum_mismatch"),
    });
  });

  it("returns a clear failure reason when the binary version is unsupported", () => {
    const bytes = encodeKVSwapBlockBinary(makeBlock());
    const unsupportedVersion = new Uint8Array(bytes);
    new DataView(unsupportedVersion.buffer, unsupportedVersion.byteOffset, unsupportedVersion.byteLength).setUint16(4, 999, true);

    const decoded = decodeKVSwapBlockBinary(unsupportedVersion);

    expect(decoded).toEqual({
      ok: false,
      reason: "unsupported_binary_version:999",
    });
  });

  it("rejects oversized payload lengths before allocating nested arrays", () => {
    const bytes = new Uint8Array(16);
    const view = new DataView(bytes.buffer);
    view.setUint32(0, KV_SWAP_BINARY_MAGIC, true);
    view.setUint16(4, KV_SWAP_BINARY_CODEC_VERSION, true);
    view.setUint16(6, 16, true);
    view.setUint32(8, MAX_KV_SWAP_BINARY_RECORD_BYTES, true);

    const decoded = decodeKVSwapBlockBinary(bytes);

    expect(decoded).toEqual({
      ok: false,
      reason: `binary_payload_too_large:${MAX_KV_SWAP_BINARY_RECORD_BYTES}/${MAX_KV_SWAP_BINARY_RECORD_BYTES - 16}`,
    });
  });
});

function makeBlock(): SerializedKVSwapBlock {
  return {
    version: KV_SWAP_STORAGE_VERSION,
    namespace: "tenant_cell_session",
    id: "layer0:b0",
    modelId: "Qwen/Qwen3-0.6B",
    requestId: "req-123",
    runtimeBlockId: "runtime-layer0-b0",
    phase: "prefill",
    modelFingerprint: "sha256-model",
    promptTokenHash: "prompt-hash",
    promptTokenIds: [151643, 220, 198],
    prefillTokenCount: 3,
    runtimeLayerCount: 28,
    policyHash: "policy-a",
    layer: 0,
    startToken: 0,
    endToken: 3,
    pinned: true,
    importance: 0.875,
    estimatedBytes: 192,
    checksum: "record-checksum",
    summaryRank: 2,
    compressedKeySummary: [0.125, 0.25, 0.5, 0.75],
    prefillProof: {
      layers: [{
        layerIndex: 0,
        qProjection: "webgpu",
        kProjection: "webgpu",
        vProjection: "webgpu",
        oProjection: "webgpu",
        mlpBackend: "webgpu",
        mlpActivationKind: "silu_gated",
        mlpRowCount: 3,
        attentionBackend: "webgpu",
        packedHeadBackends: ["webgpu", "webgpu"],
        packedHeadCount: 2,
        selectedKeyRows: 3,
        prefillChunkDispatch: "chunked_dispatch",
        attentionDispatchCount: 4,
        awaitedDispatchBreaks: 1,
      }],
      prefillChunkCount: 2,
      prefillChunkSize: 1024,
      shapeBucket: "prompt<=2048:selected<=16:headDim<=128:tileRows<=8192:precision=f32",
      pipelineCacheKey: "prefill_chunk:prompt<=2048:selected<=16:headDim<=128:tileRows<=8192:precision=f32",
      maxDispatchEstimatedMs: 0.8,
      prefillChunkDispatch: "chunked_dispatch",
      attentionDispatchCount: 4,
      awaitedDispatchBreaks: 1,
    },
    tokenIds: [151643, 220, 198],
    queryRows: [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]],
    keyRows: [[1, 2], [3, 4], [5, 6]],
    valueRows: [[7, 8], [9, 10], [11, 12]],
    compactKeyRows: [[1], [3], [5]],
    compactValueRows: [[7], [9], [11]],
    hiddenRows: [[13, 14], [15, 16], [17, 18]],
    createdAt: "2026-05-20T00:00:00.000Z",
    updatedAt: "2026-05-20T00:00:01.000Z",
    lastAccessAt: 123456789,
    byteLength: 192,
  };
}
