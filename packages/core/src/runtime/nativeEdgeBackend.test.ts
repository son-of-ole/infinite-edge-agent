import { describe, expect, it } from "vitest";
import type { SSALayerRoutingPolicy } from "./ssa";
import {
  getNativeEdgeLayerTensorHandles,
  NativeEdgeReferenceBackend,
  readNativeEdgeDecodeHandle,
  readNativeEdgeKvCacheHandle,
} from "./nativeEdgeBackend";
import { readSsaToyTensorHandle } from "./ssa_webgpu";

describe("NativeEdgeReferenceBackend", () => {
  it("prefills deterministic Q/K/V handles and executes one layer through sparse SSA", async () => {
    const backend = new NativeEdgeReferenceBackend({ backendPreference: "cpu", headDim: 2 });
    await backend.initializeModel("native-edge-reference:test");

    const policy = makePolicy({ denseFallback: false });
    const prefill = await backend.prefill(new Int32Array([7, 11, 13, 17]), {
      requestId: "req_native_sparse",
      layerPolicies: [policy],
    });
    const kvCache = readNativeEdgeKvCacheHandle(prefill.kvCacheHandle);
    const handles = getNativeEdgeLayerTensorHandles(prefill.kvCacheHandle, 0);
    const output = await backend.executeSparseLayer({
      requestId: prefill.requestId,
      layerIndex: 0,
      qHandle: handles.qHandle,
      kHandle: handles.kHandle,
      vHandle: handles.vHandle,
      policy,
    });

    expect(kvCache).toMatchObject({
      kind: "native_edge_kv_cache",
      modelId: "native-edge-reference:test",
      requestId: "req_native_sparse",
      tokenIds: [7, 11, 13, 17],
    });
    expect(readSsaToyTensorHandle(handles.qHandle).id).toBe("native_edge:req_native_sparse:layer0:q");
    expect(readSsaToyTensorHandle(handles.kHandle).matrix).toEqual(readSsaToyTensorHandle(handles.qHandle).matrix);
    expect(readSsaToyTensorHandle(handles.vHandle).matrix).not.toEqual(readSsaToyTensorHandle(handles.qHandle).matrix);
    expect(readSsaToyTensorHandle(output.outputHandle).matrix).toHaveLength(4);
    expect(output.trace).toMatchObject({
      requestId: "req_native_sparse",
      layerIndex: 0,
      selectedBlockIds: ["b0", "b1"],
      pinnedBlockIds: ["b0"],
      denseTokenCountEstimate: 4,
      sparseTokenCountEstimate: 6,
    });
  });

  it("keeps pinned blocks in selected IDs and traces", async () => {
    const backend = new NativeEdgeReferenceBackend({ backendPreference: "cpu", headDim: 2 });
    await backend.initializeModel("native-edge-reference:test");

    const policy = makePolicy({
      pinnedBlockIds: ["b0"],
      selectedBlockIdsByQueryBlock: {
        0: ["b1"],
        1: ["b1"],
      },
    });
    const prefill = await backend.prefill(new Int32Array([2, 3, 5, 7]), {
      requestId: "req_native_pinned",
      layerPolicies: [policy],
    });
    const handles = getNativeEdgeLayerTensorHandles(prefill.kvCacheHandle, 0);
    const output = await backend.executeSparseLayer({
      requestId: "req_native_pinned",
      layerIndex: 0,
      qHandle: handles.qHandle,
      kHandle: handles.kHandle,
      vHandle: handles.vHandle,
      policy,
    });

    expect(output.selectedBlockIds).toEqual(["b0", "b1"]);
    expect(output.trace.selectedBlockIds).toEqual(["b0", "b1"]);
    expect(output.trace.pinnedBlockIds).toEqual(["b0"]);
  });

  it("validates dense reference mode on a tiny all-block fixture", async () => {
    const backend = new NativeEdgeReferenceBackend({ backendPreference: "cpu", headDim: 2 });
    await backend.initializeModel("native-edge-reference:test");

    const policy = makePolicy({ denseFallback: true });
    const prefill = await backend.prefill(new Int32Array([1, 1, 2, 3]), {
      requestId: "req_native_dense",
      layerPolicies: [policy],
    });
    const handles = getNativeEdgeLayerTensorHandles(prefill.kvCacheHandle, 0);
    const output = await backend.executeSparseLayer({
      requestId: "req_native_dense",
      layerIndex: 0,
      qHandle: handles.qHandle,
      kHandle: handles.kHandle,
      vHandle: handles.vHandle,
      policy,
    });

    expect(output.denseReference).toMatchObject({
      passed: true,
      maxAbsDiff: 0,
    });
    expect(output.denseReference?.denseOutput).toEqual(readSsaToyTensorHandle(output.outputHandle).matrix);
  });

  it("decodes deterministically from the backend-owned KV cache", async () => {
    const backend = new NativeEdgeReferenceBackend({ backendPreference: "cpu", headDim: 2 });
    await backend.initializeModel("native-edge-reference:test");

    const prefill = await backend.prefill(new Int32Array([4, 4, 8, 8]), {
      requestId: "req_native_decode",
      layerPolicies: [makePolicy()],
    });
    const first = await backend.decode({
      requestId: "req_native_decode",
      inputTokenId: 9,
      kvCacheHandle: prefill.kvCacheHandle,
      policy: [makePolicy()],
    });
    const second = await backend.decode({
      requestId: "req_native_decode",
      inputTokenId: 9,
      kvCacheHandle: prefill.kvCacheHandle,
      policy: [makePolicy()],
    });

    expect(first.tokenId).toBe(second.tokenId);
    expect(readNativeEdgeDecodeHandle(first.logitsHandle)).toEqual(readNativeEdgeDecodeHandle(second.logitsHandle));
    expect(first.traces[0]).toMatchObject({ requestId: "req_native_decode", layerIndex: 0 });
  });

  it("rejects decode when the KV cache belongs to another request or model", async () => {
    const backend = new NativeEdgeReferenceBackend({ backendPreference: "cpu", headDim: 2 });
    await backend.initializeModel("native-edge-reference:test-a");
    const prefill = await backend.prefill(new Int32Array([4, 4, 8, 8]), {
      requestId: "req_native_decode_a",
      layerPolicies: [makePolicy()],
    });

    await expect(backend.decode({
      requestId: "req_native_decode_b",
      inputTokenId: 9,
      kvCacheHandle: prefill.kvCacheHandle,
      policy: [makePolicy()],
    })).rejects.toThrow("KV cache requestId mismatch");

    const otherBackend = new NativeEdgeReferenceBackend({ backendPreference: "cpu", headDim: 2 });
    await otherBackend.initializeModel("native-edge-reference:test-b");
    await expect(otherBackend.decode({
      requestId: "req_native_decode_a",
      inputTokenId: 9,
      kvCacheHandle: prefill.kvCacheHandle,
      policy: [makePolicy()],
    })).rejects.toThrow("KV cache modelId mismatch");
  });

  it("keeps block token ranges layer-specific when layer policies use different block sizes", async () => {
    const backend = new NativeEdgeReferenceBackend({ backendPreference: "cpu", headDim: 2 });
    await backend.initializeModel("native-edge-reference:test");

    const prefill = await backend.prefill(new Int32Array([1, 2, 3, 4, 5, 6]), {
      requestId: "req_native_multilayer_ranges",
      layerPolicies: [
        makePolicy({
          layerIndex: 0,
          blockSize: 2,
          selectedBlockIdsByQueryBlock: {
            0: ["b0"],
            1: ["b1"],
            2: ["b2"],
          },
        }),
        makePolicy({
          layerIndex: 1,
          blockSize: 3,
          pinnedBlockIds: [],
          selectedBlockIdsByQueryBlock: {
            0: ["b0"],
            1: ["b1"],
          },
        }),
      ],
    });

    const layer0 = getNativeEdgeLayerTensorHandles(prefill.kvCacheHandle, 0);
    const layer1 = getNativeEdgeLayerTensorHandles(prefill.kvCacheHandle, 1);

    expect(readSsaToyTensorHandle(layer0.qHandle).blockTokenRanges).toEqual({
      b0: { tokenStart: 0, tokenEnd: 2 },
      b1: { tokenStart: 2, tokenEnd: 4 },
      b2: { tokenStart: 4, tokenEnd: 6 },
    });
    expect(readSsaToyTensorHandle(layer1.qHandle).blockTokenRanges).toEqual({
      b0: { tokenStart: 0, tokenEnd: 3 },
      b1: { tokenStart: 3, tokenEnd: 6 },
    });
  });

  it("dispose prevents later use with a clear error", async () => {
    const backend = new NativeEdgeReferenceBackend({ backendPreference: "cpu" });
    await backend.initializeModel("native-edge-reference:test");
    await backend.dispose();

    await expect(backend.prefill(new Int32Array([1]), {
      requestId: "req_after_dispose",
      layerPolicies: [makePolicy()],
    })).rejects.toThrow("NativeEdgeReferenceBackend has been disposed");
  });
});

function makePolicy(overrides: Partial<SSALayerRoutingPolicy> = {}): SSALayerRoutingPolicy {
  return {
    layerIndex: 0,
    blockSize: 2,
    topKBlocks: 1,
    localWindowBlocks: 0,
    pinnedBlockIds: ["b0"],
    selectedBlockIdsByQueryBlock: {
      0: ["b0"],
      1: ["b0", "b1"],
    },
    denseFallback: false,
    ...overrides,
  };
}
