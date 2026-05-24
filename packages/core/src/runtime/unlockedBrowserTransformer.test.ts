import { describe, expect, it } from "vitest";
import type { SSALayerRoutingPolicy } from "./ssa";
import {
  createRecommendedUnlockedBrowserModelPlan,
  getUnlockedBrowserLayerTensorHandles,
  readUnlockedBrowserDecodeHandle,
  readUnlockedBrowserKvCacheHandle,
  UnlockedBrowserTransformerBackend,
  F16Matrix,
  F32Matrix,
  type UnlockedBrowserTransformerWeights,
} from "./unlockedBrowserTransformer";
import { createSsaToyTensorHandle, readSsaToyTensorHandle, WebGpuRuntimeBufferCache } from "./ssa_webgpu";

describe("UnlockedBrowserTransformerBackend", () => {
  it("prefills browser-owned Q/K/V tensors from model weights and owns KV block handles", async () => {
    const backend = new UnlockedBrowserTransformerBackend({
      weights: makeFixtureWeights(),
      backendPreference: "cpu",
    });
    await backend.initializeModel("fixture/unlocked-qwen-control");

    const policy = makePolicy();
    const prefill = await backend.prefill(new Int32Array([0, 1, 2, 3]), {
      requestId: "req_unlocked_prefill",
      layerPolicies: [policy],
    });
    const cache = readUnlockedBrowserKvCacheHandle(prefill.kvCacheHandle);
    const handles = getUnlockedBrowserLayerTensorHandles(prefill.kvCacheHandle, 0);

    expect(cache).toMatchObject({
      kind: "unlocked_browser_transformer_kv_cache",
      modelId: "fixture/unlocked-qwen-control",
      requestId: "req_unlocked_prefill",
      tokenIds: [0, 1, 2, 3],
    });
    expect(cache.kvBlocks.map((block) => block.id)).toEqual(["layer0:b0", "layer0:b1"]);
    expect(cache.kvBlocks.every((block) => block.tensorHandles?.key && block.tensorHandles.value)).toBe(true);
    expect(readSsaToyTensorHandle(handles.qHandle).matrix).toEqual([
      [1, 0],
      [0, 1],
      [1, 1],
      [0.5, -0.5],
    ]);
    expect(readSsaToyTensorHandle(handles.kHandle).matrix).toEqual(readSsaToyTensorHandle(handles.qHandle).matrix);
    expect(readSsaToyTensorHandle(handles.vHandle).matrix).not.toEqual(readSsaToyTensorHandle(handles.qHandle).matrix);
  });

  it("records prefill projection and causal attention kernel proofs on the KV cache handle", async () => {
    const backend = new UnlockedBrowserTransformerBackend({
      weights: makePackedHeadFixtureWeights(),
      backendPreference: "cpu",
    });
    await backend.initializeModel("fixture/unlocked-packed-heads");

    const prefill = await backend.prefill(new Int32Array([0, 1, 2]), {
      requestId: "req_unlocked_prefill_proof",
      layerPolicies: [makePolicy({ blockSize: 1 })],
    });
    const cache = readUnlockedBrowserKvCacheHandle(prefill.kvCacheHandle);

    expect(cache.prefillProof?.layers).toEqual([
      expect.objectContaining({
        layerIndex: 0,
        qProjection: expect.objectContaining({
          backend: "cpu_reference",
          trace: expect.objectContaining({ tokens: 3, metadata: expect.objectContaining({ purpose: "prefill_q_projection" }) }),
        }),
        kProjection: expect.objectContaining({ backend: "cpu_reference" }),
        vProjection: expect.objectContaining({ backend: "cpu_reference" }),
        oProjection: expect.objectContaining({
          backend: "cpu_reference",
          trace: expect.objectContaining({ metadata: expect.objectContaining({ purpose: "prefill_o_projection" }) }),
        }),
        attentionBackend: "cpu_reference",
        packedHeadBackends: ["cpu_reference", "cpu_reference"],
        packedHeadCount: 2,
        selectedKeyRows: 6,
      }),
    ]);
  });

  it("splits long prefill attention into bounded chunk dispatches with static shape proof", async () => {
    const backend = new UnlockedBrowserTransformerBackend({
      weights: makePackedHeadFixtureWeights(),
      backendPreference: "cpu",
    });
    await backend.initializeModel("fixture/unlocked-packed-heads");

    const tokenIds = new Int32Array(Array.from({ length: 1100 }, (_value, index) => index % 4));
    const prefill = await backend.prefill(tokenIds, {
      requestId: "req_unlocked_chunked_prefill",
      layerPolicies: [makePolicy({ blockSize: 1 })],
    });
    const cache = readUnlockedBrowserKvCacheHandle(prefill.kvCacheHandle);

    expect(prefill.prefillChunkCount).toBeGreaterThan(1);
    expect(prefill.prefillChunkSize).toBeLessThanOrEqual(1024);
    expect(prefill).toMatchObject({
      prefillChunkDispatch: "chunked_dispatch",
      shapeBucket: expect.stringContaining("prompt<="),
      pipelineCacheKey: expect.stringContaining("prefill_chunk:"),
      maxDispatchEstimatedMs: expect.any(Number),
    });
    expect(cache.prefillProof).toMatchObject({
      prefillChunkDispatch: "chunked_dispatch",
      prefillChunkCount: prefill.prefillChunkCount,
      prefillChunkSize: prefill.prefillChunkSize,
    });
    expect(cache.prefillProof?.layers[0]).toMatchObject({
      attentionBackend: "cpu_reference",
      packedHeadCount: 2,
      prefillChunkDispatch: "chunked_dispatch",
      attentionDispatchCount: (prefill.prefillChunkCount ?? 0) * 2,
    });
  });

  it("decodes through sparse SSA, KV tensor paging, and TSP callbacks while appending KV state", async () => {
    const backend = new UnlockedBrowserTransformerBackend({
      weights: makeFixtureWeights(),
      backendPreference: "cpu",
      initialNonPinnedTier: "disk",
    });
    await backend.initializeModel("fixture/unlocked-qwen-control");

    const policy = makePolicy();
    const prefill = await backend.prefill(new Int32Array([0, 1, 2, 3]), {
      requestId: "req_unlocked_decode",
      layerPolicies: [policy],
    });
    const first = await backend.decode({
      requestId: "req_unlocked_decode",
      inputTokenId: 1,
      kvCacheHandle: prefill.kvCacheHandle,
      policy: [policy],
    });
    const second = await backend.decode({
      requestId: "req_unlocked_decode",
      inputTokenId: 1,
      kvCacheHandle: prefill.kvCacheHandle,
      policy: [policy],
    });
    const firstHandle = readUnlockedBrowserDecodeHandle(first.logitsHandle);
    const secondHandle = readUnlockedBrowserDecodeHandle(second.logitsHandle);
    const cache = readUnlockedBrowserKvCacheHandle(prefill.kvCacheHandle);

    expect(firstHandle.tspTrace.map((step) => step.kind)).toEqual(["kv_prefetch", "attention", "mlp"]);
    expect(secondHandle.tspTrace.map((step) => step.kind)).toEqual(["kv_prefetch", "attention", "mlp"]);
    expect(firstHandle.kvPagingEvents.length).toBeGreaterThan(0);
    expect(cache.tokenIds).toHaveLength(6);
    expect(readSsaToyTensorHandle(getUnlockedBrowserLayerTensorHandles(prefill.kvCacheHandle, 0).qHandle).matrix).toHaveLength(6);
    expect(first.traces[0]).toMatchObject({
      requestId: "req_unlocked_decode",
      layerIndex: 0,
      selectedBlockIds: ["b0", "b1"],
    });
  });

  it("reports CPU fallback MLP kernel proof when decode layers have MLP weights", async () => {
    const backend = new UnlockedBrowserTransformerBackend({
      weights: makeFixtureWeights(),
      backendPreference: "cpu",
    });
    await backend.initializeModel("fixture/unlocked-qwen-control");

    const policy = makePolicy();
    const prefill = await backend.prefill(new Int32Array([0, 1]), {
      requestId: "req_unlocked_mlp_proof",
      layerPolicies: [policy],
    });
    const decoded = await backend.decode({
      requestId: "req_unlocked_mlp_proof",
      inputTokenId: 1,
      kvCacheHandle: prefill.kvCacheHandle,
      policy: [policy],
    });
    const handle = readUnlockedBrowserDecodeHandle(decoded.logitsHandle);

    expect(handle.backendProof?.mlpLayers).toEqual([
      expect.objectContaining({
        layerIndex: 0,
        backend: "cpu_reference",
        trace: expect.objectContaining({
          backend: "cpu_reference",
          inputSize: 2,
          intermediateSize: 2,
          outputSize: 2,
          activationKind: "gelu",
        }),
      }),
    ]);
  });

  it("does not emit MLP proof for decode layers without MLP weights", async () => {
    const backend = new UnlockedBrowserTransformerBackend({
      weights: makePackedHeadFixtureWeights(),
      backendPreference: "cpu",
    });
    await backend.initializeModel("fixture/unlocked-packed-heads");

    const policy = makePolicy({ blockSize: 1 });
    const prefill = await backend.prefill(new Int32Array([0, 1]), {
      requestId: "req_unlocked_no_mlp_proof",
      layerPolicies: [policy],
    });
    const decoded = await backend.decode({
      requestId: "req_unlocked_no_mlp_proof",
      inputTokenId: 2,
      kvCacheHandle: prefill.kvCacheHandle,
      policy: [policy],
    });
    const handle = readUnlockedBrowserDecodeHandle(decoded.logitsHandle);

    expect(handle.backendProof?.mlpLayers).toBeUndefined();
  });

  it("selects decode tokens from full-vocab top-k instead of a heuristic candidate set", async () => {
    const weights = makeFixtureWeights();
    weights.outputProjection = [
      [1, 0],
      [0, 1],
      [10, 10],
      [-1, 0.5],
    ];
    const backend = new UnlockedBrowserTransformerBackend({
      weights,
      backendPreference: "cpu",
    });
    await backend.initializeModel("fixture/unlocked-qwen-control");

    const policy = makePolicy();
    const prefill = await backend.prefill(new Int32Array([0, 1]), {
      requestId: "req_full_vocab_topk_logits",
      layerPolicies: [policy],
    });
    const decoded = await backend.decode({
      requestId: "req_full_vocab_topk_logits",
      inputTokenId: 1,
      kvCacheHandle: prefill.kvCacheHandle,
      policy: [policy],
      logitTopK: 2,
    });
    const handle = readUnlockedBrowserDecodeHandle(decoded.logitsHandle);

    expect(handle.logitTokenIds?.[0]).toBe(2);
    expect(decoded.tokenId).toBe(2);
    expect(handle.logits).toHaveLength(2);
    expect(handle.backendProof?.logitProjection).toMatchObject({
      backend: "cpu_reference",
      fullRowCount: 4,
      selectedRowCount: 2,
      purpose: "full_vocab_topk_logit_projection",
      trace: expect.objectContaining({
        scannedRows: 4,
        materializedRows: 2,
        metadata: expect.objectContaining({
          purpose: "full_vocab_topk_logit_projection",
        }),
      }),
    });
  });

  it("suppresses disabled thinking token ids during full-vocab top-k selection", async () => {
    const weights = makeFixtureWeights();
    weights.outputProjection = [
      [1, 0],
      [0, 1],
      [10, 10],
      [-1, 0.5],
    ];
    const backend = new UnlockedBrowserTransformerBackend({
      weights,
      backendPreference: "cpu",
    });
    await backend.initializeModel("fixture/unlocked-qwen-control");

    const policy = makePolicy();
    const prefill = await backend.prefill(new Int32Array([0, 1]), {
      requestId: "req_suppressed_thinking_logits",
      layerPolicies: [policy],
    });
    const decoded = await backend.decode({
      requestId: "req_suppressed_thinking_logits",
      inputTokenId: 1,
      kvCacheHandle: prefill.kvCacheHandle,
      policy: [policy],
      logitTopK: 2,
      suppressedTokenIds: [2],
    });
    const handle = readUnlockedBrowserDecodeHandle(decoded.logitsHandle);

    expect(handle.logitTokenIds).not.toContain(2);
    expect(decoded.tokenId).not.toBe(2);
    expect(handle.backendProof?.logitProjection?.trace.metadata).toMatchObject({
      suppressedRowCount: 1,
    });
  });

  it("keeps f16 packed matrices lazy while preserving backend projection math", async () => {
    const weights = makeFixtureWeights();
    weights.tokenEmbedding = toF16Matrix(weights.tokenEmbedding as number[][]);
    weights.outputProjection = toF16Matrix(weights.outputProjection as number[][]);
    weights.layers = weights.layers.map((layer) => ({
      ...layer,
      qProj: toF16Matrix(layer.qProj as number[][]),
      kProj: toF16Matrix(layer.kProj as number[][]),
      vProj: toF16Matrix(layer.vProj as number[][]),
      oProj: toF16Matrix(layer.oProj as number[][]),
      mlpUpProj: toF16Matrix(layer.mlpUpProj as number[][]),
      mlpDownProj: toF16Matrix(layer.mlpDownProj as number[][]),
    }));

    const packed = weights.outputProjection as F16Matrix;
    expect(Array.from(packed.row(2) ?? [])).toEqual([1, 1]);
    expect((packed as unknown as { decodedValues?: Float32Array }).decodedValues).toBeUndefined();
    expect(Array.from(packed.toFloat32Array([3]))).toEqual([-1, 0.5]);
    expect((packed as unknown as { decodedValues?: Float32Array }).decodedValues).toBeUndefined();
    expect(packed.toFloat32Array()).toBe(packed.toFloat32Array());
    expect((packed as unknown as { decodedValues?: Float32Array }).decodedValues).toBeInstanceOf(Float32Array);

    const backend = new UnlockedBrowserTransformerBackend({
      weights,
      backendPreference: "cpu",
    });
    await backend.initializeModel("fixture/unlocked-qwen-control");

    const prefill = await backend.prefill(new Int32Array([0, 1]), {
      requestId: "req_f16_packed_runtime",
      layerPolicies: [makePolicy()],
    });
    const decoded = await backend.decode({
      requestId: "req_f16_packed_runtime",
      inputTokenId: 1,
      kvCacheHandle: prefill.kvCacheHandle,
      policy: [makePolicy()],
      logitTopK: 2,
    });
    const handle = readUnlockedBrowserDecodeHandle(decoded.logitsHandle);

    expect(handle.backendProof?.logitProjection).toMatchObject({
      backend: "cpu_reference",
      fullRowCount: 4,
      selectedRowCount: 2,
    });
    expect(decoded.tokenId).toBeGreaterThanOrEqual(0);
  });

  it("reports full-vocab logit projection proof through the dense matvec kernel boundary", async () => {
    const backend = new UnlockedBrowserTransformerBackend({
      weights: makeFixtureWeights(),
      backendPreference: "cpu",
    });
    await backend.initializeModel("fixture/unlocked-qwen-control");

    const policy = makePolicy();
    const prefill = await backend.prefill(new Int32Array([0, 1]), {
      requestId: "req_full_vocab_logits_proof",
      layerPolicies: [policy],
    });
    const decoded = await backend.decode({
      requestId: "req_full_vocab_logits_proof",
      inputTokenId: 1,
      kvCacheHandle: prefill.kvCacheHandle,
      policy: [policy],
    });
    const handle = readUnlockedBrowserDecodeHandle(decoded.logitsHandle);

    expect(handle.logits).toHaveLength(1);
    expect(handle.logitTokenIds).toHaveLength(1);
    expect(handle.backendProof?.logitProjection).toMatchObject({
      backend: "cpu_reference",
      fullRowCount: 4,
      selectedRowCount: 1,
      purpose: "full_vocab_topk_logit_projection",
      trace: expect.objectContaining({
        selectedRows: 1,
        scannedRows: 4,
        materializedRows: 1,
        metadata: expect.objectContaining({
          purpose: "full_vocab_topk_logit_projection",
        }),
      }),
    });
  });

  it("executes every configured transformer layer before returning decode logits", async () => {
    const weights = makeFixtureWeights();
    weights.layers.push({
      qProj: [
        [0.5, 0],
        [0, 0.5],
      ],
      kProj: [
        [1, 0],
        [0, 1],
      ],
      vProj: [
        [1, 0],
        [0, 1],
      ],
      oProj: [
        [1, 0],
        [0, 1],
      ],
      mlpUpProj: [
        [1, 0],
        [0, 1],
      ],
      mlpDownProj: [
        [1, 0],
        [0, 1],
      ],
    });
    const backend = new UnlockedBrowserTransformerBackend({
      weights,
      backendPreference: "cpu",
      initialNonPinnedTier: "disk",
    });
    await backend.initializeModel("fixture/unlocked-qwen-control");

    const prefill = await backend.prefill(new Int32Array([0, 1, 2, 3]), {
      requestId: "req_unlocked_layers",
      layerPolicies: [makePolicy({ layerIndex: 0 }), makePolicy({ layerIndex: 1 })],
    });
    const decoded = await backend.decode({
      requestId: "req_unlocked_layers",
      inputTokenId: 2,
      kvCacheHandle: prefill.kvCacheHandle,
      policy: [makePolicy({ layerIndex: 0 }), makePolicy({ layerIndex: 1 })],
    });
    const handle = readUnlockedBrowserDecodeHandle(decoded.logitsHandle);

    expect(decoded.traces.map((trace) => trace.layerIndex)).toEqual([0, 1]);
    expect(handle.tspTrace.map((step) => step.kind)).toEqual([
      "kv_prefetch",
      "attention",
      "mlp",
      "kv_prefetch",
      "attention",
      "mlp",
    ]);
    expect(readUnlockedBrowserKvCacheHandle(prefill.kvCacheHandle).kvBlocks.map((block) => block.layer)).toEqual([
      0,
      0,
      0,
      1,
      1,
      1,
    ]);
  });

  it("uses causal dense attention when prefill advances hidden states between decoder layers", async () => {
    const identityLayer = {
      qProj: [
        [1, 0],
        [0, 1],
      ],
      kProj: [
        [1, 0],
        [0, 1],
      ],
      vProj: [
        [1, 0],
        [0, 1],
      ],
      oProj: [
        [1, 0],
        [0, 1],
      ],
    };
    const backend = new UnlockedBrowserTransformerBackend({
      weights: {
        modelId: "fixture/unlocked-causal-prefill",
        architecture: "qwen3_decoder_control",
        vocabSize: 4,
        hiddenSize: 2,
        headDim: 2,
        tokenEmbedding: [
          [1, 0],
          [0, 1],
          [1, 1],
          [0, 0],
        ],
        outputProjection: [
          [1, 0],
          [0, 1],
          [1, 1],
          [-1, -1],
        ],
        layers: [
          identityLayer,
          identityLayer,
        ],
      },
      backendPreference: "cpu",
    });
    await backend.initializeModel("fixture/unlocked-causal-prefill");

    const prefill = await backend.prefill(new Int32Array([0, 1]), {
      requestId: "req_unlocked_causal_prefill",
      layerPolicies: [makePolicy({ layerIndex: 0, blockSize: 1 }), makePolicy({ layerIndex: 1, blockSize: 1 })],
    });
    const layer1 = getUnlockedBrowserLayerTensorHandles(prefill.kvCacheHandle, 1);

    expect(readSsaToyTensorHandle(layer1.qHandle).matrix[0]).toEqual([1, 0]);
  });

  it("maps sparse block ids to their numeric token ranges rather than selected-array position", async () => {
    const backend = new UnlockedBrowserTransformerBackend({
      weights: makeFixtureWeights(),
      backendPreference: "cpu",
    });
    await backend.initializeModel("fixture/unlocked-qwen-control");

    const prefill = await backend.prefill(new Int32Array(Array.from({ length: 20 }, (_, index) => index % 4)), {
      requestId: "req_unlocked_block_ranges",
      layerPolicies: [
        makePolicy({
          blockSize: 2,
          selectedBlockIdsByQueryBlock: {
            9: ["b9"],
          },
        }),
      ],
    });
    const cache = readUnlockedBrowserKvCacheHandle(prefill.kvCacheHandle);

    expect(cache.blockTokenRanges.b9).toEqual({ tokenStart: 18, tokenEnd: 20 });
  });

  it("runs optional Qwen RMSNorm, head norm, final norm, and gated MLP tensors", async () => {
    const weights = makeFixtureWeights();
    weights.rmsNormEps = 1e-6;
    weights.finalNorm = [1, 1];
    weights.layers[0] = {
      inputLayerNorm: [2, 1],
      qProj: [
        [1, 0],
        [0, 1],
      ],
      kProj: [
        [1, 0],
        [0, 1],
      ],
      vProj: [
        [1, 0],
        [0, 1],
      ],
      oProj: [
        [1, 0],
        [0, 1],
      ],
      qNorm: [1, 1],
      kNorm: [1, 1],
      postAttentionLayerNorm: [1, 1],
      mlpGateProj: [
        [1, 0],
        [0, 1],
        [1, 1],
      ],
      mlpUpProj: [
        [1, 0],
        [0, 1],
        [1, 1],
      ],
      mlpDownProj: [
        [1, 0, 0.5],
        [0, 1, -0.5],
      ],
    };
    const backend = new UnlockedBrowserTransformerBackend({
      weights,
      backendPreference: "cpu",
      initialNonPinnedTier: "disk",
    });
    await backend.initializeModel("fixture/unlocked-qwen-control");

    const policy = makePolicy();
    const prefill = await backend.prefill(new Int32Array([0, 1, 2]), {
      requestId: "req_unlocked_qwen_math",
      layerPolicies: [policy],
    });
    const decoded = await backend.decode({
      requestId: "req_unlocked_qwen_math",
      inputTokenId: 3,
      kvCacheHandle: prefill.kvCacheHandle,
      policy: [policy],
      logitTopK: weights.vocabSize,
    });
    const handle = readUnlockedBrowserDecodeHandle(decoded.logitsHandle);
    const vMatrix = readSsaToyTensorHandle(getUnlockedBrowserLayerTensorHandles(prefill.kvCacheHandle, 0).vHandle).matrix;

    expect(handle.tspTrace.map((step) => step.kind)).toEqual(["kv_prefetch", "attention", "mlp"]);
    expect(handle.logits).toHaveLength(weights.vocabSize);
    expect(handle.logits.every(Number.isFinite)).toBe(true);
    expect(vMatrix[0]?.[0]).toBeCloseTo(2.8284, 3);
    expect(vMatrix[1]?.[1]).toBeCloseTo(1.4142, 3);
    expect(readSsaToyTensorHandle(getUnlockedBrowserLayerTensorHandles(prefill.kvCacheHandle, 0).qHandle).matrix).toHaveLength(4);
  });

  it("does not synthesize an MLP residual for partial Qwen norm-only manifests", async () => {
    const weights = makeFixtureWeights();
    weights.rmsNormEps = 1e-6;
    weights.layers[0] = {
      inputLayerNorm: [1, 1],
      qProj: [
        [1, 0],
        [0, 1],
      ],
      kProj: [
        [1, 0],
        [0, 1],
      ],
      vProj: [
        [1, 0],
        [0, 1],
      ],
      oProj: [
        [0, 0],
        [0, 0],
      ],
      postAttentionLayerNorm: [1, 1],
    };
    const backend = new UnlockedBrowserTransformerBackend({
      weights,
      backendPreference: "cpu",
      initialNonPinnedTier: "disk",
    });
    await backend.initializeModel("fixture/unlocked-qwen-control");

    const policy = makePolicy();
    const prefill = await backend.prefill(new Int32Array([0]), {
      requestId: "req_unlocked_qwen_norm_only",
      layerPolicies: [policy],
    });
    const decoded = await backend.decode({
      requestId: "req_unlocked_qwen_norm_only",
      inputTokenId: 1,
      kvCacheHandle: prefill.kvCacheHandle,
      policy: [policy],
      logitTopK: weights.vocabSize,
    });
    const handle = readUnlockedBrowserDecodeHandle(decoded.logitsHandle);
    const logitsByTokenId = new Map((handle.logitTokenIds ?? []).map((tokenId, index) => [tokenId, handle.logits[index] ?? Number.NaN]));

    expect(logitsByTokenId.get(0)).toBeCloseTo(0, 6);
    expect(logitsByTokenId.get(1)).toBeCloseTo(1, 6);
    expect(logitsByTokenId.get(2)).toBeCloseTo(1, 6);
    expect(logitsByTokenId.get(3)).toBeCloseTo(0.5, 6);
  });

  it("validates Qwen gated MLP tensor shapes before model initialization", () => {
    const weights = makeFixtureWeights();
    const layer = weights.layers[0];
    if (!layer) throw new Error("Expected fixture layer.");
    weights.layers[0] = {
      ...layer,
      mlpGateProj: [
        [1, 0],
        [0, 1],
        [1, 1],
      ],
      mlpUpProj: [
        [1, 0],
        [0, 1],
        [1, 1],
      ],
      mlpDownProj: [
        [1, 0],
        [0, 1],
      ],
    };

    expect(() => new UnlockedBrowserTransformerBackend({
      weights,
      backendPreference: "cpu",
    })).toThrow("mlpDownProj");
  });

  it("selects Qwen-class transformer assets over diffusion or opaque chat APIs for the unlocked route", () => {
    const plan = createRecommendedUnlockedBrowserModelPlan();

    expect(plan.primary.family).toBe("qwen3");
    expect(plan.primary.reason).toContain("smallest strong browser-control target");
    expect(plan.rejectedFamilies).toEqual(expect.arrayContaining([
      expect.objectContaining({ family: "diffusion", reason: expect.stringContaining("KV cache") }),
      expect.objectContaining({ family: "opaque-browser-chat-api", reason: expect.stringContaining("Q/K/V") }),
    ]));
  });

  it("executes full Qwen GQA projections with RoPE and expands KV heads for SSA", async () => {
    const weights = makeFullGqaFixtureWeights();
    const backend = new UnlockedBrowserTransformerBackend({
      weights,
      backendPreference: "cpu",
      initialNonPinnedTier: "disk",
    });
    await backend.initializeModel("fixture/unlocked-qwen-full-gqa");

    const prefill = await backend.prefill(new Int32Array([0, 1]), {
      requestId: "req_unlocked_full_gqa",
      layerPolicies: [makePolicy({ blockSize: 1 })],
    });
    const handles = getUnlockedBrowserLayerTensorHandles(prefill.kvCacheHandle, 0);
    const q = readSsaToyTensorHandle(handles.qHandle).matrix;
    const k = readSsaToyTensorHandle(handles.kHandle).matrix;
    const v = readSsaToyTensorHandle(handles.vHandle).matrix;

    expect(q[0]).toHaveLength(4);
    expect(k[0]).toHaveLength(4);
    expect(v[0]).toHaveLength(4);
    expect(k[0]?.slice(0, 2)).toEqual(k[0]?.slice(2, 4));
    expect(v[0]?.slice(0, 2)).toEqual(v[0]?.slice(2, 4));
    expect(k[1]?.slice(0, 2)).not.toEqual([0, 1]);

    const decoded = await backend.decode({
      requestId: "req_unlocked_full_gqa",
      inputTokenId: 2,
      kvCacheHandle: prefill.kvCacheHandle,
      policy: [makePolicy({ blockSize: 1 })],
      logitTopK: weights.vocabSize,
    });
    const handle = readUnlockedBrowserDecodeHandle(decoded.logitsHandle);

    expect(handle.logits).toHaveLength(weights.vocabSize);
    expect(handle.logits.every(Number.isFinite)).toBe(true);
    expect(handle.tspTrace.map((step) => step.kind)).toEqual(["kv_prefetch", "attention", "mlp"]);
  });

  it("appends decode KV rows from per-layer state instead of rebuilding the whole prefix", async () => {
    const backend = new UnlockedBrowserTransformerBackend({
      weights: makeFullGqaFixtureWeights(),
      backendPreference: "cpu",
      initialNonPinnedTier: "disk",
    });
    await backend.initializeModel("fixture/unlocked-qwen-full-gqa");

    const prefill = await backend.prefill(new Int32Array([0, 1]), {
      requestId: "req_unlocked_incremental_kv",
      layerPolicies: [makePolicy({ blockSize: 1 })],
    });
    const cache = readUnlockedBrowserKvCacheHandle(prefill.kvCacheHandle);
    const beforeK = getUnlockedBrowserLayerTensorHandles(prefill.kvCacheHandle, 0).kHandle;
    const stateK = cache.layerStates[0]?.k;

    await backend.decode({
      requestId: "req_unlocked_incremental_kv",
      inputTokenId: 2,
      kvCacheHandle: prefill.kvCacheHandle,
      policy: [makePolicy({ blockSize: 1 })],
    });
    const afterK = getUnlockedBrowserLayerTensorHandles(prefill.kvCacheHandle, 0).kHandle;

    expect(afterK).not.toBe(beforeK);
    expect(readSsaToyTensorHandle(afterK).matrix).toBe(stateK);
    expect(readSsaToyTensorHandle(afterK).matrix).toHaveLength(3);
    expect(cache.layerStates[0]?.projectedTokenCount).toBe(3);
  });

  it("verifies MTP draft windows as one continuation pass and commits only valid KV input rows", async () => {
    const backend = new UnlockedBrowserTransformerBackend({
      weights: makeFullGqaFixtureWeights(),
      backendPreference: "cpu",
      initialNonPinnedTier: "disk",
    });
    await backend.initializeModel("fixture/unlocked-qwen-full-gqa");

    const policy = makePolicy({
      blockSize: 1,
      topKBlocks: 4,
      pinnedBlockIds: ["b0"],
      selectedBlockIdsByQueryBlock: {
        0: ["b0"],
        1: ["b0", "b1"],
        2: ["b0", "b1", "b2"],
        3: ["b0", "b1", "b2", "b3"],
      },
    });
    const prefill = await backend.prefill(new Int32Array([0, 1]), {
      requestId: "req_unlocked_mtp_batch",
      layerPolicies: [policy],
    });
    const cache = readUnlockedBrowserKvCacheHandle(prefill.kvCacheHandle);
    const result = await backend.verifySpeculativeDraft({
      requestId: "req_unlocked_mtp_batch",
      previousTokenId: 2,
      draftTokenIds: [999, 998],
      kvCacheHandle: prefill.kvCacheHandle,
      policy: [policy],
    });

    expect(result.targetDecodeCalls).toBe(1);
    expect(result.verifiedTokenCount).toBe(2);
    expect(result.rejectedTokens).toBe(2);
    expect(result.committedInputTokenIds).toEqual([2]);
    expect(cache.tokenIds).toEqual([0, 1, 2]);
    expect(cache.layerStates[0]?.k).toHaveLength(3);
    expect(decodedPackedAttentionBackend(result.decodeOutput)).toMatchObject({
      attentionBackend: "cpu_reference",
      packedHeadBackends: ["cpu_reference", "cpu_reference"],
      packedHeadCount: 2,
    });
    expect(readUnlockedBrowserDecodeHandle(result.decodeOutput.logitsHandle).tspTrace.map((step) => step.kind)).toEqual(["kv_prefetch", "attention", "mlp"]);
  });

  it("runs sparse decode attention per packed head instead of flattening all heads into one score", async () => {
    const backend = new UnlockedBrowserTransformerBackend({
      weights: makePackedHeadFixtureWeights(),
      backendPreference: "cpu",
      initialNonPinnedTier: "disk",
    });
    await backend.initializeModel("fixture/unlocked-packed-heads");

    const policy = makePolicy({
      blockSize: 1,
      topKBlocks: 3,
      pinnedBlockIds: ["b0"],
      selectedBlockIdsByQueryBlock: {
        0: ["b0"],
        1: ["b0", "b1"],
        2: ["b0", "b1", "b2"],
      },
    });
    const prefill = await backend.prefill(new Int32Array([0, 1]), {
      requestId: "req_unlocked_packed_heads",
      layerPolicies: [policy],
    });
    const decoded = await backend.decode({
      requestId: "req_unlocked_packed_heads",
      inputTokenId: 2,
      kvCacheHandle: prefill.kvCacheHandle,
      policy: [policy],
    });
    const cache = readUnlockedBrowserKvCacheHandle(prefill.kvCacheHandle);
    const latestHidden = cache.layerStates[0]?.hidden.at(-1);

    expect(latestHidden?.[0]).toBeCloseTo(1 / 3, 5);
    expect(latestHidden?.[1]).toBeCloseTo(1 / 3, 5);
    expect(latestHidden?.[2]).toBeCloseTo(0.5035, 3);
    expect(latestHidden?.[0]).not.toBeCloseTo(0.274, 2);
    expect(decodedTraceSparseTokenCount(decoded)).toBe(3);
    expect(decodedPackedAttentionBackend(decoded)).toMatchObject({
      attentionBackend: "cpu_reference",
      packedHeadBackends: ["cpu_reference", "cpu_reference"],
      packedHeadCount: 2,
    });
  });

  it("runs packed decode attention only for the newly appended query row", async () => {
    const backend = new UnlockedBrowserTransformerBackend({
      weights: makePackedHeadFixtureWeights(),
      backendPreference: "cpu",
      initialNonPinnedTier: "disk",
    });
    await backend.initializeModel("fixture/unlocked-packed-heads");

    const policy = makePolicy({
      blockSize: 1,
      topKBlocks: 3,
      pinnedBlockIds: ["b0"],
      selectedBlockIdsByQueryBlock: {
        0: ["b0"],
        1: ["b0", "b1"],
        2: ["b0", "b1", "b2"],
      },
    });
    const prefill = await backend.prefill(new Int32Array([0, 1]), {
      requestId: "req_unlocked_decode_single_query_attention",
      layerPolicies: [policy],
    });
    const decoded = await backend.decode({
      requestId: "req_unlocked_decode_single_query_attention",
      inputTokenId: 2,
      kvCacheHandle: prefill.kvCacheHandle,
      policy: [policy],
    });

    expect(decoded.traces[0]).toMatchObject({
      queryTokenCount: 1,
      denseTokenCountEstimate: 3,
      sparseTokenCountEstimate: 3,
      attentionBackend: "cpu_reference",
      packedHeadCount: 2,
    });
  });

  it("preserves one sparse output row per packed-head query when executeSparseLayer is called directly", async () => {
    const backend = new UnlockedBrowserTransformerBackend({
      weights: makePackedHeadFixtureWeights(),
      backendPreference: "cpu",
    });
    await backend.initializeModel("fixture/unlocked-packed-heads");

    const blockTokenRanges = {
      b0: { tokenStart: 0, tokenEnd: 1 },
      b1: { tokenStart: 1, tokenEnd: 2 },
      b2: { tokenStart: 2, tokenEnd: 3 },
    };
    const policy = makePolicy({
      blockSize: 1,
      selectedBlockIdsByQueryBlock: {
        0: ["b0"],
        1: ["b0", "b1"],
        2: ["b1", "b2"],
      },
    });
    const sparse = await backend.executeSparseLayer({
      requestId: "req_unlocked_direct_packed_sparse",
      layerIndex: 0,
      qHandle: createSsaToyTensorHandle({
        id: "direct_packed_q",
        matrix: [
          [1, 0, 0, 1],
          [0, 1, 1, 0],
          [1, 1, 1, 1],
        ],
        blockTokenRanges,
      }),
      kHandle: createSsaToyTensorHandle({
        id: "direct_packed_k",
        matrix: [
          [1, 0, 0, 1],
          [0, 1, 1, 0],
          [1, 1, 1, 1],
        ],
        blockTokenRanges,
      }),
      vHandle: createSsaToyTensorHandle({
        id: "direct_packed_v",
        matrix: [
          [1, 0, 0, 1],
          [0, 1, 1, 0],
          [0.5, 0.5, 0.5, 0.5],
        ],
        blockTokenRanges,
      }),
      policy,
    });

    const output = readSsaToyTensorHandle(sparse.outputHandle).matrix;

    expect(output).toHaveLength(3);
    expect(output.every((row) => row.length === 4)).toBe(true);
    expect(sparse.trace).toMatchObject({
      queryBlockIndex: 2,
      selectedBlockIds: ["b1", "b2"],
      sparseTokenCountEstimate: 2,
    });
  });

  it("reports packed decode trace route for the actual later query block", async () => {
    const backend = new UnlockedBrowserTransformerBackend({
      weights: makePackedHeadFixtureWeights(),
      backendPreference: "cpu",
      initialNonPinnedTier: "disk",
    });
    await backend.initializeModel("fixture/unlocked-packed-heads");

    const policy = makePolicy({
      blockSize: 2,
      topKBlocks: 1,
      pinnedBlockIds: ["b0"],
      selectedBlockIdsByQueryBlock: {
        0: ["b0"],
        1: ["b1"],
        2: ["b2"],
      },
    });
    const prefill = await backend.prefill(new Int32Array([0, 1, 2]), {
      requestId: "req_unlocked_packed_trace_route",
      layerPolicies: [policy],
    });
    const decoded = await backend.decode({
      requestId: "req_unlocked_packed_trace_route",
      inputTokenId: 3,
      kvCacheHandle: prefill.kvCacheHandle,
      policy: [policy],
    });

    expect(decoded.traces[0]).toMatchObject({
      queryBlockIndex: 1,
      selectedBlockIds: ["b1"],
      sparseTokenCountEstimate: 2,
    });
  });

  it("reuses stable WebGPU buffers for decode O, MLP, and output projection tensors", async () => {
    const weights = makeFixtureWeights();
    const layer = weights.layers[0];
    if (!layer?.mlpUpProj || !layer.mlpDownProj) throw new Error("Expected fixture MLP weights.");
    weights.outputProjection = toF32Matrix(weights.outputProjection as number[][]);
    weights.layers[0] = {
      ...layer,
      qProj: toF32Matrix(layer.qProj as number[][]),
      kProj: toF32Matrix(layer.kProj as number[][]),
      vProj: toF32Matrix(layer.vProj as number[][]),
      oProj: toF32Matrix(layer.oProj as number[][]),
      mlpUpProj: toF32Matrix(layer.mlpUpProj as number[][]),
      mlpDownProj: toF32Matrix(layer.mlpDownProj as number[][]),
    };
    const backend = new UnlockedBrowserTransformerBackend({
      weights,
      device: createCapturingGpuDevice(),
      initialNonPinnedTier: "disk",
    });
    await backend.initializeModel("fixture/unlocked-qwen-control");

    const policy = makePolicy();
    const prefill = await backend.prefill(new Int32Array([0, 1]), {
      requestId: "req_unlocked_decode_projection_cache",
      layerPolicies: [policy],
    });
    await backend.decode({
      requestId: "req_unlocked_decode_projection_cache",
      inputTokenId: 1,
      kvCacheHandle: prefill.kvCacheHandle,
      policy: [policy],
    });
    const second = await backend.decode({
      requestId: "req_unlocked_decode_projection_cache",
      inputTokenId: 1,
      kvCacheHandle: prefill.kvCacheHandle,
      policy: [policy],
    });
    const handle = readUnlockedBrowserDecodeHandle(second.logitsHandle);

    expect(handle.backendProof?.oProjectionLayers).toEqual([
      expect.objectContaining({
        layerIndex: 0,
        backend: "webgpu",
        trace: expect.objectContaining({
          projectionCacheHit: true,
          pipelineCacheHit: true,
        }),
      }),
    ]);
    expect(handle.backendProof?.projectionLayers).toEqual([
      expect.objectContaining({
        layerIndex: 0,
        qProjection: expect.objectContaining({
          backend: "webgpu",
          trace: expect.objectContaining({
            projectionCacheHit: true,
            pipelineCacheHit: true,
          }),
        }),
        kProjection: expect.objectContaining({
          backend: "webgpu",
          trace: expect.objectContaining({
            projectionCacheHit: true,
            pipelineCacheHit: true,
          }),
        }),
        vProjection: expect.objectContaining({
          backend: "webgpu",
          trace: expect.objectContaining({
            projectionCacheHit: true,
            pipelineCacheHit: true,
          }),
        }),
        oProjection: expect.objectContaining({
          backend: "webgpu",
          trace: expect.objectContaining({
            projectionCacheHit: true,
            pipelineCacheHit: true,
          }),
        }),
      }),
    ]);
    expect(handle.backendProof?.mlpLayers?.[0]?.trace.projectionCacheHits).toEqual({
      upProjection: true,
      downProjection: true,
    });
    expect(handle.backendProof?.mlpLayers?.[0]?.trace.pipelineCacheHits).toEqual({
      intermediate: true,
      output: true,
    });
    expect(handle.backendProof?.logitProjection?.trace.projectionCacheHit).toBe(true);
    expect(handle.backendProof?.logitProjection?.trace.pipelineCacheHit).toBe(true);
  });

  it("directly warms packed WebGPU model residency before the first strict decode", async () => {
    const weights = makeFixtureWeights();
    const layer = weights.layers[0];
    if (!layer?.mlpUpProj || !layer.mlpDownProj) throw new Error("Expected fixture MLP weights.");
    weights.outputProjection = toF32Matrix(weights.outputProjection as number[][]);
    weights.layers[0] = {
      ...layer,
      qProj: toF32Matrix(layer.qProj as number[][]),
      kProj: toF32Matrix(layer.kProj as number[][]),
      vProj: toF32Matrix(layer.vProj as number[][]),
      oProj: toF32Matrix(layer.oProj as number[][]),
      mlpUpProj: toF32Matrix(layer.mlpUpProj as number[][]),
      mlpDownProj: toF32Matrix(layer.mlpDownProj as number[][]),
    };
    const device = createCapturingGpuDevice();
    const backend = new UnlockedBrowserTransformerBackend({
      weights,
      device,
      initialNonPinnedTier: "disk",
    });
    await backend.initializeModel("fixture/unlocked-qwen-control");

    const warmup = await backend.warmModelResidency({
      layerCount: 1,
      logitTileRows: 2,
    });
    const mapReadsAfterWarmup = device.mapReadCount;

    expect(warmup).toMatchObject({
      mode: "direct_projection_preload",
      backend: "webgpu",
      layerCount: 1,
      logitTileRows: 2,
      logitTiles: 2,
    });
    expect(warmup.uploadedEntries).toBeGreaterThan(0);
    expect(warmup.entries.map((entry) => entry.key)).toEqual(expect.arrayContaining([
      "dense-matmul:layer0.qProj",
      "dense-matmul:layer0.kProj",
      "dense-matmul:layer0.vProj",
      "dense-matmul:layer0.oProj",
      "mlp:layer0.mlp:upProjection",
      "mlp:layer0.mlp:downProjection",
      "dense-matvec:outputProjection:rows:0-2",
      "dense-matvec:outputProjection:rows:2-4",
    ]));
    expect(mapReadsAfterWarmup).toBe(0);

    const policy = makePolicy();
    const prefill = await backend.prefill(new Int32Array([0, 1]), {
      requestId: "req_unlocked_direct_residency_warm",
      layerPolicies: [policy],
    });
    const decoded = await backend.decode({
      requestId: "req_unlocked_direct_residency_warm",
      inputTokenId: 1,
      kvCacheHandle: prefill.kvCacheHandle,
      policy: [policy],
      logitTileRows: 2,
    });
    const handle = readUnlockedBrowserDecodeHandle(decoded.logitsHandle);

    expect(handle.backendProof?.projectionLayers?.[0]?.qProjection.trace.projectionCacheHit).toBe(true);
    expect(handle.backendProof?.projectionLayers?.[0]?.kProjection.trace.projectionCacheHit).toBe(true);
    expect(handle.backendProof?.projectionLayers?.[0]?.vProjection.trace.projectionCacheHit).toBe(true);
    expect(handle.backendProof?.projectionLayers?.[0]?.oProjection.trace.projectionCacheHit).toBe(true);
    expect(handle.backendProof?.mlpLayers?.[0]?.trace.projectionCacheHits).toEqual({
      upProjection: true,
      downProjection: true,
    });
    expect(handle.backendProof?.logitProjection?.trace.projectionCacheHit).toBe(true);
  });

  it("preloads compact top-k output projection tiles with the same default tile size used by decode", async () => {
    const vocabSize = 10_000;
    const hiddenSize = 2;
    const values = new Float32Array(vocabSize * hiddenSize);
    for (let row = 0; row < vocabSize; row += 1) {
      values[row * hiddenSize] = row;
      values[row * hiddenSize + 1] = vocabSize - row;
    }
    const weights = makeFixtureWeights();
    weights.vocabSize = vocabSize;
    weights.tokenEmbedding = new F32Matrix(values.buffer.slice(0), 0, vocabSize, hiddenSize);
    weights.outputProjection = new F32Matrix(values.buffer.slice(0), 0, vocabSize, hiddenSize);
    const backend = new UnlockedBrowserTransformerBackend({
      weights,
      device: createCapturingGpuDevice(),
      initialNonPinnedTier: "disk",
    });
    await backend.initializeModel("fixture/unlocked-qwen-control");

    const warmup = await backend.warmModelResidency({
      layerCount: 1,
      logitTopK: 40,
    });

    expect(warmup.entries.map((entry) => entry.key)).toEqual(expect.arrayContaining([
      "dense-matvec:outputProjection:rows:0-8192",
      "dense-matvec:outputProjection:rows:8192-10000",
    ]));
    expect(warmup.entries.map((entry) => entry.key)).not.toContain("dense-matvec:outputProjection:rows:0-10000");
  });

  it("can reuse WebGPU model buffers across fresh strict browser backends", async () => {
    const sharedBufferCache = new WebGpuRuntimeBufferCache();
    const device = createCapturingGpuDevice();
    const policy = makePolicy();
    const buildBackend = async (requestId: string) => {
      const weights = makeFixtureWeights();
      const layer = weights.layers[0];
      if (!layer?.mlpUpProj || !layer.mlpDownProj) throw new Error("Expected fixture MLP weights.");
      weights.outputProjection = toF32Matrix(weights.outputProjection as number[][]);
      weights.layers[0] = {
        ...layer,
        qProj: toF32Matrix(layer.qProj as number[][]),
        kProj: toF32Matrix(layer.kProj as number[][]),
        vProj: toF32Matrix(layer.vProj as number[][]),
        oProj: toF32Matrix(layer.oProj as number[][]),
        mlpUpProj: toF32Matrix(layer.mlpUpProj as number[][]),
        mlpDownProj: toF32Matrix(layer.mlpDownProj as number[][]),
      };
      const backend = new UnlockedBrowserTransformerBackend({
        weights,
        device,
        bufferCache: sharedBufferCache,
      });
      await backend.initializeModel("fixture/unlocked-qwen-control");
      const prefill = await backend.prefill(new Int32Array([0, 1]), {
        requestId,
        layerPolicies: [policy],
      });
      const decoded = await backend.decode({
        requestId,
        inputTokenId: 1,
        kvCacheHandle: prefill.kvCacheHandle,
        policy: [policy],
      });
      return readUnlockedBrowserDecodeHandle(decoded.logitsHandle);
    };

    await buildBackend("req_unlocked_shared_model_cache_first");
    const second = await buildBackend("req_unlocked_shared_model_cache_second");

    expect(second.backendProof?.projectionLayers?.[0]?.qProjection.trace.projectionCacheHit).toBe(true);
    expect(second.backendProof?.projectionLayers?.[0]?.kProjection.trace.projectionCacheHit).toBe(true);
    expect(second.backendProof?.projectionLayers?.[0]?.vProjection.trace.projectionCacheHit).toBe(true);
    expect(second.backendProof?.projectionLayers?.[0]?.oProjection.trace.projectionCacheHit).toBe(true);
    expect(second.backendProof?.mlpLayers?.[0]?.trace.projectionCacheHits).toEqual({
      upProjection: true,
      downProjection: true,
    });
    expect(second.backendProof?.logitProjection?.trace.projectionCacheHit).toBe(true);
  });

  it("projects final decode logits from the resident WebGPU layer output without re-uploading hidden state", async () => {
    const weights = makeFixtureWeights();
    weights.finalNorm = [1, 1];
    weights.outputProjection = toF32Matrix(weights.outputProjection as number[][]);
    const backend = new UnlockedBrowserTransformerBackend({
      weights,
      device: createCapturingGpuDevice(),
    });
    await backend.initializeModel("fixture/unlocked-qwen-control");

    const policy = makePolicy();
    const prefill = await backend.prefill(new Int32Array([0, 1]), {
      requestId: "req_resident_logit_projection",
      layerPolicies: [policy],
    });
    const decoded = await backend.decode({
      requestId: "req_resident_logit_projection",
      inputTokenId: 1,
      kvCacheHandle: prefill.kvCacheHandle,
      policy: [policy],
      logitTopK: 2,
    });
    const handle = readUnlockedBrowserDecodeHandle(decoded.logitsHandle);

    expect(handle.backendProof?.logitProjection).toMatchObject({
      backend: "webgpu",
      purpose: "greedy_argmax_logit_projection",
      trace: expect.objectContaining({
        vectorResident: true,
        readbackStrategy: "gpu_argmax_token_id",
        readbackRows: 1,
        readbackBytes: expect.any(Number),
        metadata: expect.objectContaining({
          residentDecodeFinalLogits: true,
          finalNormResident: true,
          purpose: "greedy_argmax_logit_projection",
          gpuArgmaxTokenId: true,
        }),
      }),
    });
    expect(handle.backendProof?.logitProjection?.trace.readbackBytes).toBeLessThanOrEqual(16);
    expect(handle.backendProof?.logitProjection?.trace.metadata).not.toHaveProperty("residentHiddenUpload");
  });

  it("attaches decode hot-path performance counters to the backend proof", async () => {
    const weights = makeFixtureWeights();
    weights.finalNorm = [1, 1];
    weights.outputProjection = toF32Matrix(weights.outputProjection as number[][]);
    const backend = new UnlockedBrowserTransformerBackend({
      weights,
      device: createCapturingGpuDevice(),
    });
    await backend.initializeModel("fixture/unlocked-qwen-control");

    const policy = makePolicy();
    const prefill = await backend.prefill(new Int32Array([0, 1]), {
      requestId: "req_decode_perf_counters",
      layerPolicies: [policy],
    });
    const decoded = await backend.decode({
      requestId: "req_decode_perf_counters",
      inputTokenId: 1,
      kvCacheHandle: prefill.kvCacheHandle,
      policy: [policy],
      logitTopK: 1,
    });
    const handle = readUnlockedBrowserDecodeHandle(decoded.logitsHandle);

    expect(handle.backendProof?.decodePerf).toMatchObject({
      generatedTokenCount: 1,
      decodeCallCount: 1,
      fullLogitsReadbackCount: 0,
      compactLogitReadbackCount: 1,
      cpuFallbackUsed: false,
      cpuValidationUsed: false,
      prefillExecutionsDuringDecode: 0,
      prefillCountPerGeneratedToken: 0,
      kvDecodeReused: true,
    });
    expect(handle.backendProof?.decodePerf?.dispatchCount).toBeGreaterThan(0);
    expect(handle.backendProof?.decodePerf?.totalReadbackBytes).toBeLessThanOrEqual(8);
  });

  it("uses GPU argmax token-id projection for strict WebGPU decode even when logitTopK is larger than one", async () => {
    const weights = makeFixtureWeights();
    weights.finalNorm = [1, 1];
    weights.outputProjection = toF32Matrix(weights.outputProjection as number[][]);
    const backend = new UnlockedBrowserTransformerBackend({
      weights,
      device: createCapturingGpuDevice(),
      requireWebGpu: true,
    });
    await backend.initializeModel("fixture/unlocked-qwen-control");

    const policy = makePolicy();
    const prefill = await backend.prefill(new Int32Array([0, 1]), {
      requestId: "req_gpu_argmax_decode",
      layerPolicies: [policy],
    });
    const decoded = await backend.decode({
      requestId: "req_gpu_argmax_decode",
      inputTokenId: 1,
      kvCacheHandle: prefill.kvCacheHandle,
      policy: [policy],
      logitTopK: 64,
    });
    const handle = readUnlockedBrowserDecodeHandle(decoded.logitsHandle);

    expect(handle.logits).toHaveLength(1);
    expect(handle.logitTokenIds).toHaveLength(1);
    expect(decoded.tokenId).toBe(handle.logitTokenIds?.[0]);
    expect(handle.backendProof?.logitProjection).toMatchObject({
      backend: "webgpu",
      selectedRowCount: 1,
      purpose: "greedy_argmax_logit_projection",
      trace: expect.objectContaining({
        readbackStrategy: "gpu_argmax_token_id",
        readbackRows: 1,
        readbackBytes: expect.any(Number),
        metadata: expect.objectContaining({
          gpuArgmaxTokenId: true,
        }),
      }),
    });
    expect(handle.backendProof?.logitProjection?.trace.readbackBytes).toBeLessThanOrEqual(16);
    expect(handle.backendProof?.decodePerf).toMatchObject({
      fullLogitsReadbackCount: 0,
      compactLogitReadbackCount: 1,
      totalReadbackRows: 1,
      kvDecodeReused: true,
    });
  });

  it("uses compact top-k sampling for strict WebGPU decode when sampling is requested", async () => {
    const weights = makeFixtureWeights();
    weights.finalNorm = [1, 1];
    weights.outputProjection = toF32Matrix(weights.outputProjection as number[][]);
    const backend = new UnlockedBrowserTransformerBackend({
      weights,
      device: createCapturingGpuDevice(),
      requireWebGpu: true,
    });
    await backend.initializeModel("fixture/unlocked-qwen-control");

    const policy = makePolicy();
    const prefill = await backend.prefill(new Int32Array([0, 1]), {
      requestId: "req_gpu_compact_topk_decode",
      layerPolicies: [policy],
    });
    const decoded = await backend.decode({
      requestId: "req_gpu_compact_topk_decode",
      inputTokenId: 1,
      kvCacheHandle: prefill.kvCacheHandle,
      policy: [policy],
      logitTopK: 4,
      samplingTemperature: 0.7,
      samplingTopP: 0.9,
      repetitionPenalty: 1.05,
      recentTokenIds: [1],
      samplingSeed: 1234,
    });
    const handle = readUnlockedBrowserDecodeHandle(decoded.logitsHandle);

    expect(handle.logits).toHaveLength(4);
    expect(handle.logitTokenIds).toHaveLength(4);
    expect(handle.backendProof?.logitProjection).toMatchObject({
      backend: "webgpu",
      selectedRowCount: 4,
      purpose: "compact_topk_logit_projection",
      trace: expect.objectContaining({
        readbackStrategy: "gpu_compact_topk",
        readbackRows: 4,
        readbackBytes: expect.any(Number),
        metadata: expect.objectContaining({
          gpuCompactTopK: 4,
        }),
      }),
    });
    expect(handle.backendProof?.sampling).toMatchObject({
      compactLogitTopK: 4,
      temperature: 0.7,
      topP: 0.9,
      repetitionPenalty: 1.05,
    });
    expect(typeof handle.backendProof?.sampling?.greedyDecodeUsed).toBe("boolean");
    expect(decoded.tokenId).toBe(handle.backendProof?.sampling?.selectedTokenId);
    expect(handle.backendProof?.decodePerf).toMatchObject({
      fullLogitsReadbackCount: 0,
      compactLogitReadbackCount: 1,
      totalReadbackRows: 4,
      kvDecodeReused: true,
    });
  });

  it("keeps the real Qwen decode attention/MLP residual path resident on WebGPU before layer materialization", async () => {
    const weights = makeFixtureWeights();
    weights.rmsNormEps = 1e-6;
    const layer = weights.layers[0];
    if (!layer?.mlpUpProj || !layer.mlpDownProj) throw new Error("Expected fixture MLP weights.");
    weights.outputProjection = toF32Matrix(weights.outputProjection as number[][]);
    weights.layers[0] = {
      inputLayerNorm: [1, 1],
      qProj: toF32Matrix(layer.qProj as number[][]),
      kProj: toF32Matrix(layer.kProj as number[][]),
      vProj: toF32Matrix(layer.vProj as number[][]),
      oProj: toF32Matrix(layer.oProj as number[][]),
      postAttentionLayerNorm: [1, 1],
      mlpGateProj: toF32Matrix(layer.mlpUpProj as number[][]),
      mlpUpProj: toF32Matrix(layer.mlpUpProj as number[][]),
      mlpDownProj: toF32Matrix(layer.mlpDownProj as number[][]),
    };
    const backend = new UnlockedBrowserTransformerBackend({
      weights,
      device: createCapturingGpuDevice(),
    });
    await backend.initializeModel("fixture/unlocked-qwen-control");

    const policy = makePolicy();
    const prefill = await backend.prefill(new Int32Array([0, 1]), {
      requestId: "req_resident_decode_layer_path",
      layerPolicies: [policy],
    });
    const decoded = await backend.decode({
      requestId: "req_resident_decode_layer_path",
      inputTokenId: 1,
      kvCacheHandle: prefill.kvCacheHandle,
      policy: [policy],
      logitTopK: 2,
    });
    const handle = readUnlockedBrowserDecodeHandle(decoded.logitsHandle);

    expect(handle.backendProof?.oProjectionLayers?.[0]?.trace).toMatchObject({
      outputResident: true,
      readback: false,
      metadata: expect.objectContaining({
        residentDecodeLayerPath: true,
        purpose: "decode_o_projection",
      }),
    });
    expect(handle.backendProof?.projectionLayers?.[0]).toMatchObject({
      layerIndex: 0,
      qProjection: {
        backend: "webgpu",
        trace: expect.objectContaining({
          metadata: expect.objectContaining({ purpose: "decode_q_projection" }),
        }),
      },
      kProjection: {
        backend: "webgpu",
        trace: expect.objectContaining({
          metadata: expect.objectContaining({ purpose: "decode_k_projection" }),
        }),
      },
      vProjection: {
        backend: "webgpu",
        trace: expect.objectContaining({
          metadata: expect.objectContaining({ purpose: "decode_v_projection" }),
        }),
      },
      oProjection: {
        backend: "webgpu",
        trace: expect.objectContaining({
          metadata: expect.objectContaining({ purpose: "decode_o_projection" }),
        }),
      },
    });
    expect(handle.backendProof?.mlpLayers?.[0]?.trace).toMatchObject({
      outputResident: true,
      readback: false,
      inputResident: true,
      metadata: expect.objectContaining({
        residentDecodeLayerPath: true,
        purpose: "decode_mlp",
      }),
    });
    expect(handle.backendProof?.logitProjection?.trace).toMatchObject({
      vectorResident: true,
    });
  });

  it("does not read back final hidden when resident final logits already satisfy decode", async () => {
    const weights = makeFixtureWeights();
    weights.rmsNormEps = 1e-6;
    const layer = weights.layers[0];
    if (!layer?.mlpUpProj || !layer.mlpDownProj) throw new Error("Expected fixture MLP weights.");
    weights.outputProjection = toF32Matrix(weights.outputProjection as number[][]);
    weights.layers[0] = {
      inputLayerNorm: [1, 1],
      qProj: toF32Matrix(layer.qProj as number[][]),
      kProj: toF32Matrix(layer.kProj as number[][]),
      vProj: toF32Matrix(layer.vProj as number[][]),
      oProj: toF32Matrix(layer.oProj as number[][]),
      postAttentionLayerNorm: [1, 1],
      mlpGateProj: toF32Matrix(layer.mlpUpProj as number[][]),
      mlpUpProj: toF32Matrix(layer.mlpUpProj as number[][]),
      mlpDownProj: toF32Matrix(layer.mlpDownProj as number[][]),
    };
    const device = createCapturingGpuDevice();
    const backend = new UnlockedBrowserTransformerBackend({
      weights,
      device,
    });
    await backend.initializeModel("fixture/unlocked-qwen-control");

    const policy = makePolicy();
    const prefill = await backend.prefill(new Int32Array([0, 1]), {
      requestId: "req_skip_final_hidden_readback",
      layerPolicies: [policy],
    });
    const readbacksBeforeDecode = device.mapReadCount;
    const decoded = await backend.decode({
      requestId: "req_skip_final_hidden_readback",
      inputTokenId: 1,
      kvCacheHandle: prefill.kvCacheHandle,
      policy: [policy],
      logitTopK: 1,
    });
    const handle = readUnlockedBrowserDecodeHandle(decoded.logitsHandle);

    expect(device.mapReadCount - readbacksBeforeDecode).toBe(5);
    expect(handle.backendProof?.logitProjection?.trace.metadata).toMatchObject({
      finalHiddenReadbackSkipped: true,
    });
  });

  it("keeps packed decode attention resident into the final logit path", async () => {
    const weights = makePackedHeadFixtureWeights();
    const device = createCapturingGpuDevice();
    const backend = new UnlockedBrowserTransformerBackend({
      weights,
      device,
      requireWebGpu: true,
    });
    await backend.initializeModel("fixture/unlocked-packed-heads");

    const policy = makePolicy();
    const prefill = await backend.prefill(new Int32Array([0, 1]), {
      requestId: "req_resident_packed_attention",
      layerPolicies: [policy],
    });
    const readbacksBeforeDecode = device.mapReadCount;
    const decoded = await backend.decode({
      requestId: "req_resident_packed_attention",
      inputTokenId: 1,
      kvCacheHandle: prefill.kvCacheHandle,
      policy: [policy],
      logitTopK: 1,
    });
    const handle = readUnlockedBrowserDecodeHandle(decoded.logitsHandle);

    expect(device.mapReadCount - readbacksBeforeDecode).toBe(1);
    expect(decoded.traces[0]).toMatchObject({
      attentionBackend: "webgpu",
      outputResident: true,
      readback: false,
      inputResident: {
        q: true,
        k: true,
        v: true,
      },
    });
    expect(handle.backendProof?.oProjectionLayers?.[0]?.trace).toMatchObject({
      inputResident: true,
      outputResident: true,
      readback: false,
    });
    expect(handle.backendProof?.logitProjection?.trace).toMatchObject({
      vectorResident: true,
    });
    expect(handle.backendProof?.decodePerf).toMatchObject({
      activationUploadBytesDuringDecode: 0,
      activationUploadCountDuringDecode: 0,
      hiddenReadbackCountDuringDecode: 0,
      residentFinalHiddenUsedForLogits: true,
      residentDecodeLayerCoverage: 1,
    });
  });

  it("carries strict packed decode hidden state resident between layers", async () => {
    const weights = makePackedHeadFixtureWeights();
    const firstLayer = weights.layers[0];
    if (!firstLayer) throw new Error("Expected packed fixture layer.");
    weights.layers = [firstLayer, { ...firstLayer }];
    const device = createCapturingGpuDevice();
    const backend = new UnlockedBrowserTransformerBackend({
      weights,
      device,
      requireWebGpu: true,
    });
    await backend.initializeModel("fixture/unlocked-packed-heads");

    const prefill = await backend.prefill(new Int32Array([0, 1]), {
      requestId: "req_strict_resident_hidden_across_layers",
      layerPolicies: [
        makePolicy({ layerIndex: 0 }),
        makePolicy({ layerIndex: 1 }),
      ],
    });
    const decoded = await backend.decode({
      requestId: "req_strict_resident_hidden_across_layers",
      inputTokenId: 1,
      kvCacheHandle: prefill.kvCacheHandle,
      policy: [
        makePolicy({ layerIndex: 0 }),
        makePolicy({ layerIndex: 1 }),
      ],
      logitTopK: 1,
    });
    const handle = readUnlockedBrowserDecodeHandle(decoded.logitsHandle);

    expect(handle.backendProof?.decodePerf).toMatchObject({
      activationUploadBytesDuringDecode: 0,
      activationUploadCountDuringDecode: 0,
      hiddenReadbackCountDuringDecode: 0,
      residentFinalHiddenUsedForLogits: true,
      residentDecodeLayerCoverage: 1,
    });
  });

  it("batches strict decode queue submissions to one submit per layer plus logits", async () => {
    const weights = makePackedHeadFixtureWeights();
    const firstLayer = weights.layers[0];
    if (!firstLayer) throw new Error("Expected packed fixture layer.");
    const qwenLayer = {
      ...firstLayer,
      inputLayerNorm: [1, 1, 1, 1],
      postAttentionLayerNorm: [1, 1, 1, 1],
      mlpUpProj: firstLayer.qProj,
      mlpDownProj: firstLayer.oProj,
      mlpGateProj: firstLayer.vProj,
    };
    weights.layers = [qwenLayer, { ...qwenLayer }];
    const backend = new UnlockedBrowserTransformerBackend({
      weights,
      device: createCapturingGpuDevice(),
      requireWebGpu: true,
    });
    await backend.initializeModel("fixture/unlocked-packed-heads");

    const prefill = await backend.prefill(new Int32Array([0, 1]), {
      requestId: "req_strict_submit_batching",
      layerPolicies: [
        makePolicy({ layerIndex: 0 }),
        makePolicy({ layerIndex: 1 }),
      ],
    });
    const decoded = await backend.decode({
      requestId: "req_strict_submit_batching",
      inputTokenId: 1,
      kvCacheHandle: prefill.kvCacheHandle,
      policy: [
        makePolicy({ layerIndex: 0 }),
        makePolicy({ layerIndex: 1 }),
      ],
      logitTopK: 1,
    });
    const handle = readUnlockedBrowserDecodeHandle(decoded.logitsHandle);

    expect(handle.backendProof?.decodePerf).toMatchObject({
      generatedTokenCount: 1,
      totalDecodeLayerCount: 2,
      residentDecodeLayerCoverage: 1,
      fusedPackedQkvLayerCount: 2,
      fusedQkvNormRopeKvAppendLayerCount: 2,
      fusedResidualRmsNormLayerCount: 2,
    });
    expect(handle.backendProof?.decodePerf?.fusedLayerCoverage).toBeGreaterThan(0);
    expect(handle.backendProof?.decodePerf?.decodeSubmitCountPerToken).toBe(2);
    expect(handle.backendProof?.decodePerf?.dispatchCount).toBeGreaterThan(
      handle.backendProof?.decodePerf?.decodeSubmitCount ?? 0,
    );
    expect(handle.backendProof?.decodePerf?.decodeDispatchCountPerLayerPerToken).toBeLessThanOrEqual(6);
    expect(handle.backendProof?.decodePerf?.fusedLayerCoverage).toBeGreaterThanOrEqual(0.5);
    expect(handle.backendProof?.residualRmsNormLayers?.[0]?.trace.metadata).toMatchObject({
      fusedStage: "residual_rmsnorm",
      residualRmsNormPair: true,
    });
  });

  it("reports one-token packed attention fusion for large strict decode contexts while leaving MLP fusion off", async () => {
    const weights = makePackedHeadFixtureWeights();
    const firstLayer = weights.layers[0];
    if (!firstLayer) throw new Error("Expected packed fixture layer.");
    weights.layers = [{
      ...firstLayer,
      inputLayerNorm: [1, 1, 1, 1],
      postAttentionLayerNorm: [1, 1, 1, 1],
      mlpUpProj: firstLayer.qProj,
      mlpDownProj: firstLayer.oProj,
      mlpGateProj: firstLayer.vProj,
    }];
    const backend = new UnlockedBrowserTransformerBackend({
      weights,
      device: createCapturingGpuDevice(),
      requireWebGpu: true,
    });
    await backend.initializeModel("fixture/unlocked-packed-heads");

    const policy = makePolicy({
      blockSize: 128,
      topKBlocks: 1,
      pinnedBlockIds: ["b0"],
      selectedBlockIdsByQueryBlock: { 0: ["b0"] },
    });
    const prefill = await backend.prefill(
      new Int32Array(Array.from({ length: 80 }, (_value, index) => index % weights.vocabSize)),
      {
        requestId: "req_strict_one_token_attention",
        layerPolicies: [policy],
      },
    );
    const decoded = await backend.decode({
      requestId: "req_strict_one_token_attention",
      inputTokenId: 1,
      kvCacheHandle: prefill.kvCacheHandle,
      policy: [policy],
      logitTopK: 1,
    });
    const handle = readUnlockedBrowserDecodeHandle(decoded.logitsHandle);

    expect(decoded.traces[0]).toMatchObject({
      attentionDispatchCount: 1,
      queryTokenCount: 1,
    });
    expect(handle.backendProof?.decodePerf).toMatchObject({
      fusedOneTokenAttentionLayerCount: 1,
      fusedMlpLayerCount: 0,
      totalDecodeLayerCount: 1,
    });
  });

  it("batches strict decode through the resident tensor device when no explicit device is configured", async () => {
    const weights = makePackedHeadFixtureWeights();
    const firstLayer = weights.layers[0];
    if (!firstLayer) throw new Error("Expected packed fixture layer.");
    weights.layers = [firstLayer, { ...firstLayer }];
    const device = createCapturingGpuDevice();
    const backend = new UnlockedBrowserTransformerBackend({
      weights,
      gpu: {
        requestAdapter: async () => ({
          requestDevice: async () => device,
        }),
      },
      requireWebGpu: true,
    });
    await backend.initializeModel("fixture/unlocked-packed-heads");

    const prefill = await backend.prefill(new Int32Array([0, 1]), {
      requestId: "req_strict_submit_batching_resolved_device",
      layerPolicies: [
        makePolicy({ layerIndex: 0 }),
        makePolicy({ layerIndex: 1 }),
      ],
    });
    const decoded = await backend.decode({
      requestId: "req_strict_submit_batching_resolved_device",
      inputTokenId: 1,
      kvCacheHandle: prefill.kvCacheHandle,
      policy: [
        makePolicy({ layerIndex: 0 }),
        makePolicy({ layerIndex: 1 }),
      ],
      logitTopK: 1,
    });
    const handle = readUnlockedBrowserDecodeHandle(decoded.logitsHandle);

    expect(handle.backendProof?.decodePerf?.decodeSubmitCountPerToken).toBeLessThanOrEqual(4);
    expect(handle.backendProof?.decodePerf).toMatchObject({
      generatedTokenCount: 1,
      totalDecodeLayerCount: 2,
      residentDecodeLayerCoverage: 1,
    });
  });

  it("keeps Qwen Q/K head normalization and RoPE resident after WebGPU projection during prefill", async () => {
    const weights = makeFullGqaFixtureWeights();
    const layer = weights.layers[0];
    if (!layer) throw new Error("Expected full GQA fixture layer.");
    weights.outputProjection = toF32Matrix(weights.outputProjection as number[][]);
    weights.layers[0] = {
      ...layer,
      qProj: toF32Matrix(layer.qProj as number[][]),
      kProj: toF32Matrix(layer.kProj as number[][]),
      vProj: toF32Matrix(layer.vProj as number[][]),
      oProj: toF32Matrix(layer.oProj as number[][]),
      mlpUpProj: toF32Matrix(layer.mlpUpProj as number[][]),
      mlpDownProj: toF32Matrix(layer.mlpDownProj as number[][]),
    };
    const backend = new UnlockedBrowserTransformerBackend({
      weights,
      device: createCapturingGpuDevice(),
    });
    await backend.initializeModel("fixture/unlocked-qwen-full-gqa");

    const prefill = await backend.prefill(new Int32Array([0, 1]), {
      requestId: "req_resident_qk_post_projection",
      layerPolicies: [makePolicy({ blockSize: 1 })],
    });
    const cache = readUnlockedBrowserKvCacheHandle(prefill.kvCacheHandle);
    const proofLayer = cache.prefillProof?.layers[0] as any;

    expect(proofLayer?.qPostProjection?.trace).toMatchObject({
      outputResident: true,
      readback: false,
      inputResident: true,
      metadata: expect.objectContaining({
        residentQkvPostProjection: true,
        purpose: "prefill_q_post_projection",
      }),
    });
    expect(proofLayer?.kPostProjection?.trace).toMatchObject({
      outputResident: true,
      readback: false,
      inputResident: true,
      metadata: expect.objectContaining({
        residentQkvPostProjection: true,
        purpose: "prefill_k_post_projection",
      }),
    });
  });

  it("keeps grouped-query KV compact when feeding packed WebGPU attention", async () => {
    const weights = makeFullGqaFixtureWeights();
    const layer = weights.layers[0];
    if (!layer) throw new Error("Expected full GQA fixture layer.");
    weights.outputProjection = toF32Matrix(weights.outputProjection as number[][]);
    weights.layers[0] = {
      ...layer,
      qProj: toF32Matrix(layer.qProj as number[][]),
      kProj: toF32Matrix(layer.kProj as number[][]),
      vProj: toF32Matrix(layer.vProj as number[][]),
      oProj: toF32Matrix(layer.oProj as number[][]),
      mlpUpProj: toF32Matrix(layer.mlpUpProj as number[][]),
      mlpDownProj: toF32Matrix(layer.mlpDownProj as number[][]),
    };
    const backend = new UnlockedBrowserTransformerBackend({
      weights,
      device: createCapturingGpuDevice(),
    });
    await backend.initializeModel("fixture/unlocked-qwen-full-gqa");

    const policy = makePolicy({ blockSize: 1 });
    const prefill = await backend.prefill(new Int32Array([0, 1]), {
      requestId: "req_compact_gqa_webgpu_prefill",
      layerPolicies: [policy],
    });
    const cache = readUnlockedBrowserKvCacheHandle(prefill.kvCacheHandle);
    const proofLayer = cache.prefillProof?.layers[0] as any;
    const handles = getUnlockedBrowserLayerTensorHandles(prefill.kvCacheHandle, 0);

    expect(readSsaToyTensorHandle(handles.kHandle).matrix[0]).toHaveLength(4);
    expect(proofLayer).toMatchObject({
      packedHeadCount: 2,
      keyValueHeadCount: 1,
      keyValueCompressionRatio: 0.5,
    });

    const decoded = await backend.decode({
      requestId: "req_compact_gqa_webgpu_prefill",
      inputTokenId: 2,
      kvCacheHandle: prefill.kvCacheHandle,
      policy: [policy],
      logitTopK: 1,
    });

    expect(decoded.traces[0]).toMatchObject({
      attentionBackend: "webgpu",
      packedHeadCount: 2,
      keyValueHeadCount: 1,
      keyValueCompressionRatio: 0.5,
    });
  });

  it("routes prefill Qwen MLP rows through the batched WebGPU MLP kernel with stable projection cache reuse", async () => {
    const weights = makeFixtureWeights();
    const layer = weights.layers[0];
    if (!layer?.mlpUpProj || !layer.mlpDownProj) throw new Error("Expected fixture MLP weights.");
    weights.layers[0] = {
      ...layer,
      qProj: toF32Matrix(layer.qProj as number[][]),
      kProj: toF32Matrix(layer.kProj as number[][]),
      vProj: toF32Matrix(layer.vProj as number[][]),
      oProj: toF32Matrix(layer.oProj as number[][]),
      mlpUpProj: toF32Matrix(layer.mlpUpProj as number[][]),
      mlpDownProj: toF32Matrix(layer.mlpDownProj as number[][]),
    };
    const backend = new UnlockedBrowserTransformerBackend({
      weights,
      device: createCapturingGpuDevice(),
    });
    await backend.initializeModel("fixture/unlocked-qwen-control");

    await backend.prefill(new Int32Array([0, 1, 2]), {
      requestId: "req_unlocked_prefill_mlp_webgpu_warm",
      layerPolicies: [makePolicy()],
    });
    const prefill = await backend.prefill(new Int32Array([0, 1, 2]), {
      requestId: "req_unlocked_prefill_mlp_webgpu_reuse",
      layerPolicies: [makePolicy()],
    });
    const cache = readUnlockedBrowserKvCacheHandle(prefill.kvCacheHandle);

    expect(cache.prefillProof?.layers[0]?.mlp).toEqual(expect.objectContaining({
      backend: "webgpu",
      rowCount: 3,
      lastTrace: expect.objectContaining({
        tokens: 3,
        metadata: expect.objectContaining({
          purpose: "prefill_mlp_batch",
        }),
        projectionCacheHits: {
          upProjection: true,
          downProjection: true,
        },
        pipelineCacheHits: {
          intermediate: true,
          output: true,
        },
      }),
    }));
  });

  it("chunks large full-vocab WebGPU logit projection uploads instead of binding the entire matrix at once", async () => {
    const weights = makeFixtureWeights();
    weights.vocabSize = 8193;
    weights.tokenEmbedding = Array.from({ length: 8193 }, (_, index) => [index % 2, (index + 1) % 2]);
    weights.outputProjection = toF32Matrix(Array.from({ length: 8193 }, (_, index) => [index % 3, index % 5]));
    const device = createCapturingGpuDevice();
    const backend = new UnlockedBrowserTransformerBackend({
      weights,
      device,
    });
    await backend.initializeModel("fixture/unlocked-qwen-control");

    const policy = makePolicy();
    const prefill = await backend.prefill(new Int32Array([0, 1]), {
      requestId: "req_unlocked_chunked_logits",
      layerPolicies: [policy],
    });
    const decoded = await backend.decode({
      requestId: "req_unlocked_chunked_logits",
      inputTokenId: 1,
      kvCacheHandle: prefill.kvCacheHandle,
      policy: [policy],
    });
    const handle = readUnlockedBrowserDecodeHandle(decoded.logitsHandle);

    expect(handle.logits).toHaveLength(1);
    expect(handle.logitTokenIds).toHaveLength(1);
    expect(handle.backendProof?.logitProjection).toMatchObject({
      backend: "webgpu",
      fullRowCount: 8193,
      selectedRowCount: 1,
      purpose: "greedy_argmax_logit_projection",
      trace: expect.objectContaining({
        scannedRows: 8193,
        tileRows: 8192,
        tiles: 2,
        materializedRows: 1,
        readbackStrategy: "gpu_argmax_token_id",
        readbackRows: 1,
      }),
    });
    expect(handle.backendProof?.logitProjection?.trace.readbackBytes).toBeLessThanOrEqual(16);
    expect(Math.max(...device.floatUploads.map((upload) => upload.length))).toBeLessThanOrEqual(8192 * 2);
  });
});

function decodedTraceSparseTokenCount(decoded: Awaited<ReturnType<UnlockedBrowserTransformerBackend["decode"]>>): number {
  return decoded.traces[0]?.sparseTokenCountEstimate ?? 0;
}

function decodedPackedAttentionBackend(decoded: Awaited<ReturnType<UnlockedBrowserTransformerBackend["decode"]>>): {
  attentionBackend?: string;
  packedHeadBackends?: string[];
  packedHeadCount?: number;
} {
  return (decoded.traces[0] ?? {}) as {
    attentionBackend?: string;
    packedHeadBackends?: string[];
    packedHeadCount?: number;
  };
}

function makePolicy(overrides: Partial<SSALayerRoutingPolicy> = {}): SSALayerRoutingPolicy {
  return {
    layerIndex: 0,
    blockSize: 2,
    topKBlocks: 2,
    localWindowBlocks: 0,
    pinnedBlockIds: ["b0"],
    selectedBlockIdsByQueryBlock: {
      0: ["b0", "b1"],
      1: ["b0", "b1"],
    },
    denseFallback: true,
    ...overrides,
  };
}

function makeFixtureWeights(): UnlockedBrowserTransformerWeights {
  return {
    modelId: "fixture/unlocked-qwen-control",
    architecture: "qwen3_decoder_control",
    vocabSize: 4,
    hiddenSize: 2,
    headDim: 2,
    tokenEmbedding: [
      [1, 0],
      [0, 1],
      [1, 1],
      [0.5, -0.5],
    ],
    outputProjection: [
      [1, 0],
      [0, 1],
      [1, 1],
      [-1, 0.5],
    ],
    layers: [
      {
        qProj: [
          [1, 0],
          [0, 1],
        ],
        kProj: [
          [1, 0],
          [0, 1],
        ],
        vProj: [
          [0.5, 0.5],
          [1, -1],
        ],
        oProj: [
          [1, 0],
          [0, 1],
        ],
        mlpUpProj: [
          [1, 0],
          [0, 1],
        ],
        mlpDownProj: [
          [1, 0],
          [0, 1],
        ],
      },
    ],
  };
}

function makeFullGqaFixtureWeights(): UnlockedBrowserTransformerWeights & {
  numAttentionHeads: number;
  numKeyValueHeads: number;
  ropeTheta: number;
  maxPositionEmbeddings: number;
  tieWordEmbeddings: boolean;
} {
  return {
    modelId: "fixture/unlocked-qwen-full-gqa",
    architecture: "qwen3_decoder_control",
    vocabSize: 4,
    hiddenSize: 2,
    headDim: 2,
    numAttentionHeads: 2,
    numKeyValueHeads: 1,
    ropeTheta: 10,
    maxPositionEmbeddings: 128,
    tieWordEmbeddings: true,
    tokenEmbedding: [
      [1, 0],
      [0, 1],
      [1, 1],
      [0.5, -0.5],
    ],
    outputProjection: [
      [1, 0],
      [0, 1],
      [1, 1],
      [-1, 0.5],
    ],
    finalNorm: [1, 1],
    layers: [
      {
        inputLayerNorm: [1, 1],
        qProj: [
          [1, 0],
          [0, 1],
          [1, 1],
          [1, -1],
        ],
        kProj: [
          [1, 0],
          [0, 1],
        ],
        vProj: [
          [0.5, 0.5],
          [1, -1],
        ],
        oProj: [
          [1, 0, 0.5, 0],
          [0, 1, 0, 0.5],
        ],
        qNorm: [1, 1],
        kNorm: [1, 1],
        postAttentionLayerNorm: [1, 1],
        mlpGateProj: [
          [1, 0],
          [0, 1],
        ],
        mlpUpProj: [
          [1, 0],
          [0, 1],
        ],
        mlpDownProj: [
          [1, 0],
          [0, 1],
        ],
      },
    ],
  };
}

function makePackedHeadFixtureWeights(): UnlockedBrowserTransformerWeights {
  const identity4 = [
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1],
  ];
  return {
    modelId: "fixture/unlocked-packed-heads",
    architecture: "qwen3_decoder_control",
    vocabSize: 4,
    hiddenSize: 4,
    headDim: 2,
    numAttentionHeads: 2,
    numKeyValueHeads: 2,
    tokenEmbedding: [
      [1, 0, 0, 0],
      [0, 1, 0, 0],
      [0, 0, 1, 0],
      [0, 0, 0, 1],
    ],
    outputProjection: identity4,
    layers: [
      {
        qProj: identity4,
        kProj: identity4,
        vProj: identity4,
        oProj: identity4,
      },
    ],
  };
}

function toF32Matrix(matrix: number[][]): F32Matrix {
  const rows = matrix.length;
  const cols = matrix[0]?.length ?? 0;
  const values = new Float32Array(matrix.flatMap((row) => row));
  return new F32Matrix(values.buffer, 0, rows, cols);
}

function toF16Matrix(matrix: number[][]): F16Matrix {
  const rows = matrix.length;
  const cols = matrix[0]?.length ?? 0;
  const values = new Uint16Array(matrix.flatMap((row) => row).map(float32ToFloat16Bits));
  return new F16Matrix(values.buffer, 0, rows, cols);
}

function float32ToFloat16Bits(value: number): number {
  if (Number.isNaN(value)) return 0x7e00;
  if (value === Infinity) return 0x7c00;
  if (value === -Infinity) return 0xfc00;
  const sign = value < 0 || Object.is(value, -0) ? 0x8000 : 0;
  const absolute = Math.abs(value);
  if (absolute === 0) return sign;
  if (absolute >= 65504) return sign | 0x7bff;
  if (absolute < 0.00006103515625) return sign | Math.round(absolute / 0.000000059604644775390625);
  const exponent = Math.floor(Math.log2(absolute));
  const fraction = absolute / 2 ** exponent - 1;
  const halfExponent = exponent + 15;
  const mantissa = Math.round(fraction * 1024);
  return sign | (halfExponent << 10) | (mantissa & 0x3ff);
}

function createCapturingGpuDevice(): {
  floatUploads: number[][];
  mapReadCount: number;
  queue: {
    writeBuffer(buffer: CapturingGpuBuffer, bufferOffset: number, data: ArrayBufferLike, dataOffset?: number, size?: number): void;
    submit(): void;
  };
  createBuffer(descriptor: { size: number; usage: number }): CapturingGpuBuffer;
  createShaderModule(): unknown;
  createComputePipeline(): { getBindGroupLayout(index: number): unknown };
  createBindGroup(): unknown;
  createCommandEncoder(): {
    beginComputePass(): {
      setPipeline(): void;
      setBindGroup(): void;
      dispatchWorkgroups(): void;
      end(): void;
    };
    copyBufferToBuffer(source: CapturingGpuBuffer, sourceOffset: number, destination: CapturingGpuBuffer, destinationOffset: number, size: number): void;
    finish(): unknown;
  };
} {
  const floatUploads: number[][] = [];
  const device = {
    floatUploads,
    mapReadCount: 0,
    queue: {
      writeBuffer: (
        _buffer: CapturingGpuBuffer,
        _bufferOffset: number,
        data: ArrayBufferLike,
        dataOffset = 0,
        size = data.byteLength - dataOffset,
      ) => {
        const bytes = data.slice(dataOffset, dataOffset + size);
        if (size % Float32Array.BYTES_PER_ELEMENT === 0) floatUploads.push([...new Float32Array(bytes)]);
      },
      submit: () => undefined,
    },
    createBuffer(descriptor: { size: number; usage: number }) {
      return new CapturingGpuBuffer(descriptor.size, () => {
        device.mapReadCount += 1;
      });
    },
    createShaderModule() {
      return {};
    },
    createComputePipeline() {
      return { getBindGroupLayout: () => ({}) };
    },
    createBindGroup() {
      return {};
    },
    createCommandEncoder() {
      return {
        beginComputePass: () => ({
          setPipeline: () => undefined,
          setBindGroup: () => undefined,
          dispatchWorkgroups: () => undefined,
          end: () => undefined,
        }),
        copyBufferToBuffer: (
          _source: CapturingGpuBuffer,
          _sourceOffset: number,
          destination: CapturingGpuBuffer,
          _destinationOffset: number,
          size: number,
        ) => {
          destination.bytes = new ArrayBuffer(size);
        },
        finish: () => ({}),
      };
    },
  };
  return device;
}

class CapturingGpuBuffer {
  bytes: ArrayBuffer;

  constructor(size: number, private readonly onMapRead: () => void = () => undefined) {
    this.bytes = new ArrayBuffer(size);
  }

  async mapAsync(): Promise<void> {
    this.onMapRead();
    return undefined;
  }

  getMappedRange(): ArrayBuffer {
    return this.bytes;
  }

  unmap(): void {
    return undefined;
  }

  destroy(): void {
    return undefined;
  }
}
