import { describe, expect, it } from "vitest";
import { sparseReferenceAttention } from "./sparseReference";
import { projectGreedyDecodeTokenWebGpu } from "./gpuGreedyLogitProjection";
import {
  createSsaToyTensorHandle,
  destroyWebGpuResidentTensor,
  readSsaToyTensorHandle,
  readWebGpuResidentTensor,
  readWebGpuResidentTensors,
  runDenseMatVecTopKResidentWebGpu,
  runDenseMatMulResidentWebGpu,
  runDenseMatMulWebGpu,
  runMlpBatchResidentWebGpu,
  runDenseMatVecTopKWebGpu,
  runDenseMatVecWebGpu,
  runMlpBatchWebGpu,
  runMlpWebGpu,
  runPackedQkvProjectionResidentWebGpu,
  runPackedSparseAttentionWebGpu,
  runPackedSparseAttentionResidentWebGpu,
  runQwenQkvNormRopePairResidentWebGpu,
  runResidualAddResidentWebGpu,
  runRmsNormResidentWebGpu,
  runSparseAttentionResidentWebGpu,
  runSparseAttentionWebGpu,
  runTokenEmbeddingLookupResidentWebGpu,
  uploadWebGpuResidentTensor,
  WebGpuRuntimeBufferCache,
  WebGpuSsaReferenceBackend,
} from "./webgpuSsaBackend";

describe("WebGPU SSA toy kernels", () => {
  it("refuses decode hidden uploads for strict GPU greedy argmax", async () => {
    const device = createCapturingGpuDevice(1);

    await expect(projectGreedyDecodeTokenWebGpu({
      hidden: [1, 0],
      outputProjection: [[1, 0], [0, 1]],
      options: {
        device,
        requireWebGpu: true,
      },
      requireResidentHidden: true,
    })).rejects.toThrow(/requires resident final hidden/i);
  });

  it("looks up decode token embeddings directly into a resident GPU tensor", async () => {
    const device = createCapturingGpuDevice(1);
    const bufferCache = new WebGpuRuntimeBufferCache();

    const result = await runTokenEmbeddingLookupResidentWebGpu({
      tokenId: 2,
      tokenEmbedding: [
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
      ],
      device,
      bufferCache,
      projectionCacheKey: "fixture-token-embedding",
      projectionCachePolicy: "stable",
      requireWebGpu: true,
      traceMetadata: { purpose: "decode_token_embedding_lookup" },
    });

    expect(result.backend).toBe("webgpu");
    expect(result.tensor).toMatchObject({
      kind: "webgpu_resident_tensor",
      rows: 1,
      cols: 3,
      byteLength: 3 * Float32Array.BYTES_PER_ELEMENT,
    });
    expect(result.trace).toMatchObject({
      backend: "webgpu",
      tokens: 1,
      hidden: 3,
      outputResident: true,
      readback: false,
      tokenId: 2,
      vocabSize: 3,
      metadata: { purpose: "decode_token_embedding_lookup" },
    });
    expect(device.pipelineEntryPoints).toContain("token_embedding_lookup");
    expect(device.shaderCodes.some((code) => code.includes("fn token_embedding_lookup"))).toBe(true);
    expect(device.mapReadCount).toBe(0);

    destroyWebGpuResidentTensor(result.tensor);
    bufferCache.clear();
  });

  it("tiles token embedding lookup so strict decode never binds the full vocabulary matrix", async () => {
    const device = createCapturingGpuDevice(1);
    const bufferCache = new WebGpuRuntimeBufferCache();

    const result = await runTokenEmbeddingLookupResidentWebGpu({
      tokenId: 2,
      tokenEmbedding: [
        [1, 10, 100],
        [2, 20, 200],
        [3, 30, 300],
        [4, 40, 400],
        [5, 50, 500],
      ],
      embeddingTileRows: 2,
      device,
      bufferCache,
      projectionCacheKey: "fixture-token-embedding",
      projectionCachePolicy: "stable",
      requireWebGpu: true,
    });

    expect(device.floatUploads[0]).toEqual([3, 30, 300, 4, 40, 400]);
    expect(device.uniformUploads[0]?.uint32.slice(0, 4)).toEqual([0, 3, 2, 0]);
    expect(result.trace).toMatchObject({
      tokenId: 2,
      vocabSize: 5,
      embeddingTileStart: 2,
      embeddingTileRows: 2,
    });

    destroyWebGpuResidentTensor(result.tensor);
    bufferCache.clear();
  });

  it("projects Q/K/V in one resident packed decode dispatch without activation readback", async () => {
    const device = createCapturingGpuDevice(1);
    const bufferCache = new WebGpuRuntimeBufferCache();
    const hidden = await uploadWebGpuResidentTensor({
      matrix: [[1, 2, 3]],
      device,
    });

    const result = await runPackedQkvProjectionResidentWebGpu({
      hidden: hidden.tensor,
      qProjection: [
        [1, 0, 0],
        [0, 1, 0],
      ],
      kProjection: [
        [0, 0, 1],
      ],
      vProjection: [
        [1, 1, 1],
      ],
      device,
      bufferCache,
      projectionCacheKey: "fixture-layer0-qkv",
      projectionCachePolicy: "stable",
      requireWebGpu: true,
      traceMetadata: { purpose: "decode_packed_qkv_projection" },
    });

    expect(result.backend).toBe("webgpu");
    expect(result.q).toMatchObject({ kind: "webgpu_resident_tensor", rows: 1, cols: 2 });
    expect(result.k).toMatchObject({ kind: "webgpu_resident_tensor", rows: 1, cols: 1 });
    expect(result.v).toMatchObject({ kind: "webgpu_resident_tensor", rows: 1, cols: 1 });
    expect(result.trace).toMatchObject({
      backend: "webgpu",
      tokens: 1,
      hidden: 3,
      qOutputSize: 2,
      kOutputSize: 1,
      vOutputSize: 1,
      dispatchCount: 1,
      outputResident: true,
      readback: false,
      inputResident: true,
      metadata: {
        purpose: "decode_packed_qkv_projection",
        fusedStage: "packed_qkv_projection",
      },
    });
    expect(device.pipelineEntryPoints).toContain("packed_qkv_projection");
    expect(device.mapReadCount).toBe(0);

    destroyWebGpuResidentTensor(result.q);
    destroyWebGpuResidentTensor(result.k);
    destroyWebGpuResidentTensor(result.v);
    destroyWebGpuResidentTensor(hidden.tensor);
    bufferCache.clear();
  });

  it("normalizes and rotates resident Q/K tensors in one fused decode dispatch", async () => {
    const device = createCapturingGpuDevice(1);
    const bufferCache = new WebGpuRuntimeBufferCache();
    const q = await uploadWebGpuResidentTensor({
      matrix: [[1, 2, 3, 4]],
      device,
    });
    const k = await uploadWebGpuResidentTensor({
      matrix: [[5, 6, 7, 8]],
      device,
    });

    const result = await runQwenQkvNormRopePairResidentWebGpu({
      qProjected: q.tensor,
      kProjected: k.tensor,
      qHeadCount: 2,
      kHeadCount: 2,
      headDim: 2,
      positions: [3],
      qNormWeight: [1, 1],
      kNormWeight: [1, 1],
      ropeTheta: 10_000,
      device,
      bufferCache,
      requireWebGpu: true,
      traceMetadata: { purpose: "decode_qkv_norm_rope_pair" },
    });

    expect(result.backend).toBe("webgpu");
    expect(result.q).toMatchObject({ kind: "webgpu_resident_tensor", rows: 1, cols: 4 });
    expect(result.k).toMatchObject({ kind: "webgpu_resident_tensor", rows: 1, cols: 4 });
    expect(result.trace).toMatchObject({
      backend: "webgpu",
      tokens: 1,
      headDim: 2,
      qHeadCount: 2,
      kHeadCount: 2,
      dispatchCount: 1,
      outputResident: true,
      readback: false,
      inputResident: {
        q: true,
        k: true,
      },
      metadata: {
        purpose: "decode_qkv_norm_rope_pair",
        fusedStage: "qkv_norm_rope_kv_append",
        qkvNormRopePair: true,
      },
    });
    expect(device.pipelineEntryPoints).toContain("qwen_qkv_norm_rope_pair");
    expect(device.mapReadCount).toBe(0);

    destroyWebGpuResidentTensor(result.q);
    destroyWebGpuResidentTensor(result.k);
    destroyWebGpuResidentTensor(q.tensor);
    destroyWebGpuResidentTensor(k.tensor);
    bufferCache.clear();
  });

  it("matches CPU sparse attention when the WebGPU runner falls back to the reference path", async () => {
    const q = [
      [1, 0],
      [0, 1],
      [1, 1],
    ];
    const k = [
      [1, 0],
      [0, 1],
      [1, 1],
    ];
    const v = [
      [10, 0],
      [0, 20],
      [30, 30],
    ];
    const selectedKeyIndexesByQuery = [[0], [0, 1], [1, 2]];

    const result = await runSparseAttentionWebGpu({
      q,
      k,
      v,
      selectedKeyIndexesByQuery,
      backendPreference: "cpu",
    });

    expect(result.backend).toBe("cpu_reference");
    expect(result.output).toEqual(sparseReferenceAttention({ q, k, v, selectedKeyIndexesByQuery }));
    expect(result.trace).toMatchObject({
      queryTokens: 3,
      keyTokens: 3,
      headDim: 2,
      selectedIndexSlots: 5,
    });
  });

  it("selects deterministic top-k dense matvec rows without exposing full logits to callers", async () => {
    const result = await runDenseMatVecTopKWebGpu({
      vector: [2, -1, 0.5],
      matrix: [
        [1, 0, 0],
        [0, 1, 0],
        [0.5, 0.5, 1],
        [-1, 2, 0],
      ],
      topK: 2,
      backendPreference: "cpu",
      traceMetadata: { purpose: "full_vocab_topk_logit_projection" },
    });

    expect(result.backend).toBe("cpu_reference");
    expect(result.selectedRowIds).toEqual([0, 2]);
    expect(result.values).toEqual([2, 1]);
    expect(result.trace).toMatchObject({
      topK: 2,
      tileRows: 4,
      tiles: 1,
      scannedRows: 4,
      materializedRows: 2,
      selectedRows: 2,
      metadata: {
        purpose: "full_vocab_topk_logit_projection",
        topKSelection: true,
        tiledTopK: true,
      },
    });
  });

  it("scans dense matvec top-k in tiles while keeping global row ids", async () => {
    const readRows: number[] = [];
    const matrix = [
      [1, 0],
      [0, 4],
      [3, 0],
      [0, 2],
      [5, 0],
    ];
    const result = await runDenseMatVecTopKWebGpu({
      vector: [1, 1],
      matrix: {
        rowCount: matrix.length,
        colCount: 2,
        row: (index: number) => {
          readRows.push(index);
          return matrix[index];
        },
      },
      topK: 2,
      tileRows: 2,
      backendPreference: "cpu",
    });

    expect(readRows).toEqual([0, 1, 2, 3, 4]);
    expect(result.selectedRowIds).toEqual([4, 1]);
    expect(result.values).toEqual([5, 4]);
    expect(result.trace).toMatchObject({
      topK: 2,
      tileRows: 2,
      tiles: 3,
      scannedRows: 5,
      materializedRows: 2,
      selectedRows: 2,
    });
  });

  it("uses deterministic lower-token tie breaking and clamps top-k to row count", async () => {
    const result = await runDenseMatVecTopKWebGpu({
      vector: [1],
      matrix: [[2], [2], [1]],
      topK: 99,
      tileRows: 1,
      backendPreference: "cpu",
    });

    expect(result.selectedRowIds).toEqual([0, 1, 2]);
    expect(result.values).toEqual([2, 2, 1]);
    expect(result.trace.selectedRows).toBe(3);
  });

  it("rejects invalid dense matvec top-k controls", async () => {
    await expect(runDenseMatVecTopKWebGpu({
      vector: [1],
      matrix: [[1]],
      topK: 0,
      backendPreference: "cpu",
    })).rejects.toThrow(/topK must be a positive integer/i);

    await expect(runDenseMatVecTopKWebGpu({
      vector: [1],
      matrix: [[1]],
      topK: 1,
      tileRows: 0,
      backendPreference: "cpu",
    })).rejects.toThrow(/tileRows must be a positive integer/i);
  });

  it("returns query-width zero rows for sparse CPU fallback when selected indexes are empty or invalid", async () => {
    const result = await runSparseAttentionWebGpu({
      q: [
        [1, 0],
        [0, 1],
      ],
      k: [
        [1, 0],
        [0, 1],
      ],
      v: [
        [10, 0],
        [0, 20],
      ],
      selectedKeyIndexesByQuery: [[], [99, -1]],
      backendPreference: "cpu",
    });

    expect(result.output).toEqual([
      [0, 0],
      [0, 0],
    ]);
  });

  it("matches dense matvec CPU fallback for selected projection rows", async () => {
    const matrix = [
      [1, 0, 0],
      [0, 1, 0],
      [0.5, 0.5, 1],
      [-1, 2, 0],
    ];
    const result = await runDenseMatVecWebGpu({
      vector: [2, -1, 0.5],
      matrix: {
        rowCount: matrix.length,
        colCount: matrix[0]?.length ?? 0,
        row: (index: number) => {
          if (index !== 1 && index !== 3) throw new Error(`unselected row ${index} should not be read`);
          return matrix[index];
        },
      },
      selectedRowIds: [3, 1],
      backendPreference: "cpu",
    });

    expect(result.backend).toBe("cpu_reference");
    expect(result.values).toEqual([-4, -1]);
    expect(result.selectedRowIds).toEqual([3, 1]);
    expect(result.trace).toMatchObject({
      backend: "cpu_reference",
      rows: 4,
      cols: 3,
      selectedRows: 2,
    });
  });

  it("matches dense matmul CPU fallback for token projection matrices", async () => {
    const result = await runDenseMatMulWebGpu({
      activations: [
        [2, -1, 0.5],
        [0, 1, 2],
      ],
      projection: [
        [1, 0, 0],
        [0, 1, 0],
        [0.5, 0.5, 1],
        [-1, 2, 0],
      ],
      backendPreference: "cpu",
      traceMetadata: { purpose: "prefill_q_projection" },
    });

    expect(result.backend).toBe("cpu_reference");
    expect(result.output).toEqual([
      [2, -1, 1, -4],
      [0, 1, 2.5, 2],
    ]);
    expect(result.trace).toMatchObject({
      backend: "cpu_reference",
      tokens: 2,
      hidden: 3,
      outputSize: 4,
      metadata: { purpose: "prefill_q_projection" },
    });
  });

  it("matches gated SiLU MLP CPU fallback semantics", async () => {
    const result = await runMlpWebGpu({
      hidden: [0.5, -1],
      gateProjection: [
        [1, 0],
        [0, -1],
        [1, 1],
      ],
      upProjection: [
        [2, 0],
        [0, 0.5],
        [1, -1],
      ],
      downProjection: [
        [1, 0, -0.25],
        [0.5, -1, 0.75],
      ],
      backendPreference: "cpu",
      traceMetadata: { requestId: "req_mlp_gated" },
    });
    const gate = [silu(0.5), silu(1), silu(-0.5)];
    const up = [1, -0.5, 1.5];
    const intermediate = gate.map((value, index) => value * (up[index] ?? 0));
    const [i0 = 0, i1 = 0, i2 = 0] = intermediate;

    expect(result.backend).toBe("cpu_reference");
    expectVectorCloseTo(result.values, [
      i0 - 0.25 * i2,
      0.5 * i0 - i1 + 0.75 * i2,
    ]);
    expect(result.trace).toMatchObject({
      backend: "cpu_reference",
      inputSize: 2,
      intermediateSize: 3,
      outputSize: 2,
      activationKind: "silu_gated",
      metadata: { requestId: "req_mlp_gated" },
    });
  });

  it("matches GeLU MLP CPU fallback semantics", async () => {
    const result = await runMlpWebGpu({
      hidden: [1, -0.25],
      upProjection: [
        [1, 1],
        [-0.5, 2],
      ],
      downProjection: [
        [2, -1],
        [0.25, 0.5],
      ],
      backendPreference: "cpu",
    });
    const intermediate = [gelu(0.75), gelu(-1)];
    const [i0 = 0, i1 = 0] = intermediate;

    expect(result.backend).toBe("cpu_reference");
    expectVectorCloseTo(result.values, [
      2 * i0 - i1,
      0.25 * i0 + 0.5 * i1,
    ]);
    expect(result.trace).toMatchObject({
      backend: "cpu_reference",
      inputSize: 2,
      intermediateSize: 2,
      outputSize: 2,
      activationKind: "gelu",
    });
  });

  it("matches single-row MLP CPU fallback semantics for batched prefill rows", async () => {
    const hiddenRows = [
      [0.5, -1],
      [1, 0.25],
    ];
    const gateProjection = [
      [1, 0],
      [0, -1],
      [1, 1],
    ];
    const upProjection = [
      [2, 0],
      [0, 0.5],
      [1, -1],
    ];
    const downProjection = [
      [1, 0, -0.25],
      [0.5, -1, 0.75],
    ];
    const expected = await Promise.all(hiddenRows.map(async (hidden) => (
      await runMlpWebGpu({
        hidden,
        gateProjection,
        upProjection,
        downProjection,
        backendPreference: "cpu",
      })
    ).values));

    const result = await runMlpBatchWebGpu({
      hidden: hiddenRows,
      gateProjection,
      upProjection,
      downProjection,
      backendPreference: "cpu",
      traceMetadata: { purpose: "prefill_mlp_batch" },
    });

    expect(result.backend).toBe("cpu_reference");
    expect(result.trace).toMatchObject({
      backend: "cpu_reference",
      tokens: 2,
      inputSize: 2,
      intermediateSize: 3,
      outputSize: 2,
      activationKind: "silu_gated",
      metadata: { purpose: "prefill_mlp_batch" },
    });
    expectMatrixCloseTo(result.output, expected);
  });

  it("uploads only compact selected rows for dense matvec WebGPU preparation", async () => {
    const matrix = [
      [1, 0, 0],
      [0, 1, 0],
      [0.5, 0.5, 1],
      [-1, 2, 0],
    ];
    const device = createCapturingGpuDevice(2);

    const result = await runDenseMatVecWebGpu({
      vector: [2, -1, 0.5],
      matrix: {
        rowCount: matrix.length,
        colCount: matrix[0]?.length ?? 0,
        row: (index: number) => {
          if (index !== 1 && index !== 3) throw new Error(`unselected row ${index} should not be read`);
          return matrix[index];
        },
      },
      selectedRowIds: [3, 1],
      device,
    });

    expect(result.backend).toBe("webgpu");
    expect(result.selectedRowIds).toEqual([3, 1]);
    expect(result.trace).toMatchObject({
      backend: "webgpu",
      rows: 4,
      cols: 3,
      selectedRows: 2,
    });
    expect(device.floatUploads[0]).toEqual([-1, 2, 0, 0, 1, 0]);
  });

  it("uses descriptor-provided Float32Array views for large stable projections", async () => {
    const device = createCapturingGpuDevice(2);
    let descriptorReads = 0;
    const result = await runDenseMatVecWebGpu({
      vector: [2, -1],
      matrix: {
        rowCount: 2,
        colCount: 2,
        row: () => {
          throw new Error("large descriptor rows should not be materialized for full projection uploads");
        },
        toFloat32Array: () => {
          descriptorReads += 1;
          return new Float32Array([1, 0, 0, 1]);
        },
      },
      device,
    });

    expect(result.backend).toBe("webgpu");
    expect(descriptorReads).toBe(1);
    expect(device.floatUploads[0]).toEqual([1, 0, 0, 1]);
  });

  it("does not cache mutable plain dense matmul projection arrays even when a caller key is present", async () => {
    const device = createCapturingGpuDevice(2);
    const bufferCache = new WebGpuRuntimeBufferCache();
    const projection = [
      [1, 0, 0],
      [0, 1, 0],
      [0.5, 0.5, 1],
    ];

    const first = await runDenseMatMulWebGpu({
      activations: [[2, -1, 0.5]],
      projection,
      device,
      bufferCache,
      projectionCacheKey: "layer0.mutable_q_proj",
      traceMetadata: { purpose: "prefill_q_projection" },
    });
    projection[0] = [9, 9, 9];
    const second = await runDenseMatMulWebGpu({
      activations: [[0, 1, 2]],
      projection,
      device,
      bufferCache,
      projectionCacheKey: "layer0.mutable_q_proj",
      traceMetadata: { purpose: "prefill_q_projection" },
    });

    expect(first.trace.projectionCacheHit).toBe(false);
    expect(second.trace.projectionCacheHit).toBe(false);
    expect(device.floatUploads.filter((upload) => upload.length === 9)).toHaveLength(2);
  });

  it("reuses stable dense matmul descriptor buffers within a live WebGPU runtime cache", async () => {
    const device = createCapturingGpuDevice(2);
    const bufferCache = new WebGpuRuntimeBufferCache();
    let projectionRowReads = 0;
    const matrix = [
      [1, 0, 0],
      [0, 1, 0],
      [0.5, 0.5, 1],
    ];
    const projection = {
      rowCount: matrix.length,
      colCount: matrix[0]?.length ?? 0,
      row: (index: number) => {
        projectionRowReads += 1;
        return matrix[index];
      },
    };

    const first = await runDenseMatMulWebGpu({
      activations: [[2, -1, 0.5]],
      projection,
      device,
      bufferCache,
      projectionCacheKey: "layer0.q_proj",
      projectionCachePolicy: "stable",
      traceMetadata: { purpose: "prefill_q_projection" },
    });
    const firstProjectionReads = projectionRowReads;
    projectionRowReads = 0;
    const second = await runDenseMatMulWebGpu({
      activations: [[0, 1, 2]],
      projection,
      device,
      bufferCache,
      projectionCacheKey: "layer0.q_proj",
      projectionCachePolicy: "stable",
      traceMetadata: { purpose: "prefill_q_projection" },
    });

    expect(first.backend).toBe("webgpu");
    expect(second.backend).toBe("webgpu");
    expect(first.trace.projectionCacheHit).toBe(false);
    expect(second.trace.projectionCacheHit).toBe(true);
    expect(first.trace.pipelineCacheHit).toBe(false);
    expect(second.trace.pipelineCacheHit).toBe(true);
    expect(firstProjectionReads).toBe(3);
    expect(projectionRowReads).toBe(0);
    expect(device.floatUploads.filter((upload) => upload.length === 9)).toHaveLength(1);
    expect(device.shaderCodes.some((code) => code.includes("fn dense_matmul"))).toBe(true);
    expect(device.pipelineEntryPoints.filter((entryPoint) => entryPoint === "dense_matmul")).toHaveLength(1);
  });

  it("keeps dense matmul tensors GPU-resident across chained kernels until explicit readback", async () => {
    const device = createCapturingGpuDevice(2);
    const bufferCache = new WebGpuRuntimeBufferCache();

    const first = await runDenseMatMulResidentWebGpu({
      activations: [[2, -1, 0.5]],
      projection: [
        [1, 0, 0],
        [0, 1, 0],
      ],
      device,
      bufferCache,
      traceMetadata: { purpose: "resident_q_projection" },
    });
    const second = await runDenseMatMulResidentWebGpu({
      activations: first.tensor,
      projection: [
        [1, 0],
        [0, 1],
      ],
      device,
      bufferCache,
      traceMetadata: { purpose: "resident_o_projection" },
    });

    expect(first.backend).toBe("webgpu");
    expect(second.backend).toBe("webgpu");
    expect(first.tensor).toMatchObject({
      kind: "webgpu_resident_tensor",
      rows: 1,
      cols: 2,
    });
    expect(second.tensor).toMatchObject({
      kind: "webgpu_resident_tensor",
      rows: 1,
      cols: 2,
    });
    expect(first.trace).toMatchObject({
      backend: "webgpu",
      outputResident: true,
      readback: false,
      metadata: { purpose: "resident_q_projection" },
    });
    expect(second.trace).toMatchObject({
      backend: "webgpu",
      outputResident: true,
      readback: false,
      metadata: { purpose: "resident_o_projection" },
    });
    expect(device.mapReadCount).toBe(0);

    const materialized = await readWebGpuResidentTensor(second.tensor);

    expect(materialized).toHaveLength(1);
    expect(materialized[0]).toHaveLength(2);
    expect(device.mapReadCount).toBe(1);
    destroyWebGpuResidentTensor(first.tensor);
    destroyWebGpuResidentTensor(second.tensor);
  });

  it("batches resident tensor readbacks into one queue submission", async () => {
    const device = createCapturingGpuDevice(2);
    const first = await uploadWebGpuResidentTensor({
      matrix: [[1, 2]],
      device,
    });
    const second = await uploadWebGpuResidentTensor({
      matrix: [[3, 4]],
      device,
    });
    const submitsBeforeReadback = device.submitCount;

    const materialized = await readWebGpuResidentTensors([first.tensor, second.tensor]);

    expect(materialized).toHaveLength(2);
    expect(materialized[0]).toHaveLength(1);
    expect(materialized[0]?.[0]).toHaveLength(2);
    expect(materialized[1]).toHaveLength(1);
    expect(materialized[1]?.[0]).toHaveLength(2);
    expect(device.submitCount - submitsBeforeReadback).toBe(1);
    expect(device.mapReadCount).toBe(2);
    destroyWebGpuResidentTensor(first.tensor);
    destroyWebGpuResidentTensor(second.tensor);
  });

  it("keeps the toy transformer layer path GPU-resident until final top-k logits", async () => {
    const device = createCapturingGpuDevice(2);
    const bufferCache = new WebGpuRuntimeBufferCache();
    const hidden = await uploadWebGpuResidentTensor({
      matrix: [[0.5, -1]],
      device,
      traceMetadata: { purpose: "embedding_hidden_upload" },
    });
    const norm = await runRmsNormResidentWebGpu({
      hidden: hidden.tensor,
      weight: [1, 1],
      eps: 1e-6,
      device,
      bufferCache,
      traceMetadata: { purpose: "input_rms_norm" },
    });
    const q = await runDenseMatMulResidentWebGpu({
      activations: norm.tensor,
      projection: [
        [1, 0],
        [0, 1],
      ],
      device,
      bufferCache,
      traceMetadata: { purpose: "resident_q_projection" },
    });
    const attention = await runSparseAttentionResidentWebGpu({
      q: q.tensor,
      k: q.tensor,
      v: q.tensor,
      selectedKeyIndexesByQuery: [[0]],
      device,
      bufferCache,
      traceMetadata: { purpose: "resident_sparse_attention" },
    });
    const projectedAttention = await runDenseMatMulResidentWebGpu({
      activations: attention.tensor,
      projection: [
        [1, 0],
        [0, 1],
      ],
      device,
      bufferCache,
      traceMetadata: { purpose: "resident_o_projection" },
    });
    const afterAttention = await runResidualAddResidentWebGpu({
      left: hidden.tensor,
      right: projectedAttention.tensor,
      device,
      bufferCache,
      traceMetadata: { purpose: "attention_residual" },
    });
    const mlp = await runMlpBatchResidentWebGpu({
      hidden: afterAttention.tensor,
      upProjection: [
        [1, 0],
        [0, 1],
      ],
      gateProjection: [
        [1, 0],
        [0, 1],
      ],
      downProjection: [
        [1, 0],
        [0, 1],
      ],
      device,
      bufferCache,
      traceMetadata: { purpose: "resident_mlp_batch" },
    });
    const afterMlp = await runResidualAddResidentWebGpu({
      left: afterAttention.tensor,
      right: mlp.tensor,
      device,
      bufferCache,
      traceMetadata: { purpose: "mlp_residual" },
    });

    expect([
      hidden.trace,
      norm.trace,
      attention.trace,
      afterAttention.trace,
      mlp.trace,
      afterMlp.trace,
    ]).toEqual(expect.arrayContaining([
      expect.objectContaining({ outputResident: true, readback: false }),
    ]));
    expect(device.mapReadCount).toBe(0);

    const logits = await runDenseMatVecTopKResidentWebGpu({
      vector: afterMlp.tensor,
      matrix: [
        [1, 0],
        [0, 1],
        [1, 1],
      ],
      topK: 2,
      device,
      bufferCache,
      traceMetadata: { purpose: "resident_full_vocab_topk_logit_projection" },
    });

    expect(logits.backend).toBe("webgpu");
    expect(logits.selectedRowIds).toEqual([0, 1]);
    expect(logits.trace).toMatchObject({
      backend: "webgpu",
      topK: 2,
      materializedRows: 2,
      vectorResident: true,
      metadata: { purpose: "resident_full_vocab_topk_logit_projection" },
    });
    expect(device.mapReadCount).toBe(1);

    for (const tensor of [
      hidden.tensor,
      norm.tensor,
      q.tensor,
      attention.tensor,
      projectedAttention.tensor,
      afterAttention.tensor,
      mlp.tensor,
      afterMlp.tensor,
    ]) {
      destroyWebGpuResidentTensor(tensor);
    }
  });

  it("reduces resident full-vocab top-1 logits on the GPU before readback", async () => {
    const device = createCapturingGpuDevice(3);
    const bufferCache = new WebGpuRuntimeBufferCache();
    const hidden = await uploadWebGpuResidentTensor({
      matrix: [[1, 1]],
      device,
    });
    const matrix = {
      rowCount: 130,
      colCount: 2,
      row: (index: number) => [index, 130 - index],
    };

    const result = await runDenseMatVecTopKResidentWebGpu({
      vector: hidden.tensor,
      matrix,
      topK: 1,
      tileRows: 130,
      device,
      bufferCache,
      projectionCacheKey: "output_projection",
      projectionCachePolicy: "stable",
      traceMetadata: { purpose: "resident_full_vocab_topk_logit_projection" },
    });

    expect(result.backend).toBe("webgpu");
    expect(result.values).toHaveLength(1);
    expect(result.selectedRowIds).toHaveLength(1);
    expect(result.trace).toMatchObject({
      backend: "webgpu",
      topK: 1,
      scannedRows: 130,
      materializedRows: 1,
      vectorResident: true,
      readbackStrategy: "gpu_top1_candidates",
      gpuReducedRows: 130,
      readbackRows: 3,
      readbackBytes: 24,
      metadata: {
        purpose: "resident_full_vocab_topk_logit_projection",
        gpuTopKReduction: true,
      },
    });
    expect(device.pipelineEntryPoints).toContain("dense_matvec_top1_candidates");
    expect(device.shaderCodes.some((code) => code.includes("fn dense_matvec_top1_candidates"))).toBe(true);
    expect(device.mapReadCount).toBe(1);

    destroyWebGpuResidentTensor(hidden.tensor);
  });

  it("keeps full-vocab top-1 suppression inside the GPU candidate shader", async () => {
    const device = createCapturingGpuDevice(3);
    const bufferCache = new WebGpuRuntimeBufferCache();
    const hidden = await uploadWebGpuResidentTensor({
      matrix: [[1, 1]],
      device,
    });

    const result = await runDenseMatVecTopKResidentWebGpu({
      vector: hidden.tensor,
      matrix: [
        [10, 10],
        [1, 0],
        [0, 1],
      ],
      topK: 1,
      suppressedRowIds: [0],
      device,
      bufferCache,
      traceMetadata: { purpose: "resident_full_vocab_topk_logit_projection" },
    });

    expect(result.backend).toBe("webgpu");
    expect(result.trace).toMatchObject({
      readbackStrategy: "gpu_top1_candidates",
      suppressedRowCount: 1,
      metadata: {
        gpuTopKReduction: true,
        suppressedRowCount: 1,
      },
    });
    const shader = device.shaderCodes.find((code) => code.includes("fn dense_matvec_top1_candidates"));
    expect(shader).toContain("suppressedRowIds");
    expect(shader).toContain("is_suppressed_row");

    destroyWebGpuResidentTensor(hidden.tensor);
  });

  it("reduces tiled resident full-vocab top-1 logits to one final GPU readback", async () => {
    const device = createCapturingGpuDevice(1);
    const bufferCache = new WebGpuRuntimeBufferCache();
    const hidden = await uploadWebGpuResidentTensor({
      matrix: [[1, 1]],
      device,
    });
    const matrix = {
      rowCount: 130,
      colCount: 2,
      row: (index: number) => [index, 130 - index],
    };

    const result = await runDenseMatVecTopKResidentWebGpu({
      vector: hidden.tensor,
      matrix,
      topK: 1,
      tileRows: 64,
      device,
      bufferCache,
      projectionCacheKey: "output_projection",
      projectionCachePolicy: "stable",
      traceMetadata: { purpose: "resident_full_vocab_topk_logit_projection" },
    });

    expect(result.backend).toBe("webgpu");
    expect(result.values).toHaveLength(1);
    expect(result.selectedRowIds).toHaveLength(1);
    expect(result.trace).toMatchObject({
      backend: "webgpu",
      topK: 1,
      tileRows: 64,
      tiles: 3,
      scannedRows: 130,
      materializedRows: 1,
      vectorResident: true,
      readbackStrategy: "gpu_top1_candidates",
      gpuReducedRows: 130,
      readbackRows: 1,
      readbackBytes: 8,
      dispatchCount: 4,
      metadata: {
        purpose: "resident_full_vocab_topk_logit_projection",
        gpuTopKReduction: true,
      },
    });
    expect(device.pipelineEntryPoints).toEqual(expect.arrayContaining([
      "dense_matvec_top1_candidates",
      "dense_matvec_top1_reduce",
    ]));
    expect(device.shaderCodes.some((code) =>
      code.includes("fn dense_matvec_top1_candidates")
      && code.includes("candidateOffset: u32")
      && code.includes("params.candidateOffset")
    )).toBe(true);
    expect(device.mapReadCount).toBe(1);

    destroyWebGpuResidentTensor(hidden.tensor);
  });

  it("reduces resident full-vocab top-k logits to compact GPU readback", async () => {
    const device = createCapturingGpuDevice(4);
    const bufferCache = new WebGpuRuntimeBufferCache();
    const hidden = await uploadWebGpuResidentTensor({
      matrix: [[1, 1]],
      device,
    });
    const matrix = {
      rowCount: 130,
      colCount: 2,
      row: (index: number) => [index, 130 - index],
    };

    const result = await runDenseMatVecTopKResidentWebGpu({
      vector: hidden.tensor,
      matrix,
      topK: 4,
      tileRows: 64,
      device,
      bufferCache,
      projectionCacheKey: "output_projection",
      projectionCachePolicy: "stable",
      traceMetadata: { purpose: "resident_full_vocab_compact_topk_logit_projection" },
    });

    expect(result.backend).toBe("webgpu");
    expect(result.values).toHaveLength(4);
    expect(result.selectedRowIds).toHaveLength(4);
    expect(result.trace).toMatchObject({
      backend: "webgpu",
      topK: 4,
      tileRows: 64,
      tiles: 3,
      scannedRows: 130,
      materializedRows: 4,
      vectorResident: true,
      readbackStrategy: "gpu_compact_topk",
      gpuReducedRows: 130,
      readbackRows: 4,
      readbackBytes: 32,
      dispatchCount: 4,
      metadata: {
        purpose: "resident_full_vocab_compact_topk_logit_projection",
        gpuCompactTopKReduction: true,
      },
    });
    expect(device.pipelineEntryPoints).toEqual(expect.arrayContaining([
      "dense_matvec_compact_topk_scores",
      "dense_matvec_compact_topk_reduce",
    ]));
    expect(device.mapReadCount).toBe(1);

    destroyWebGpuResidentTensor(hidden.tensor);
  });

  it("reuses stable tiled top-1 row-id and parameter buffers across decode calls", async () => {
    const device = createCapturingGpuDevice(1);
    const bufferCache = new WebGpuRuntimeBufferCache();
    const hidden = await uploadWebGpuResidentTensor({
      matrix: [[1, 1]],
      device,
    });
    const matrix = {
      rowCount: 130,
      colCount: 2,
      row: (index: number) => [index, 130 - index],
    };

    await runDenseMatVecTopKResidentWebGpu({
      vector: hidden.tensor,
      matrix,
      topK: 1,
      tileRows: 64,
      device,
      bufferCache,
      projectionCacheKey: "output_projection",
      projectionCachePolicy: "stable",
    });
    const writesAfterFirstDecode = device.writeBufferCount;

    await runDenseMatVecTopKResidentWebGpu({
      vector: hidden.tensor,
      matrix,
      topK: 1,
      tileRows: 64,
      device,
      bufferCache,
      projectionCacheKey: "output_projection",
      projectionCachePolicy: "stable",
    });

    expect(device.writeBufferCount - writesAfterFirstDecode).toBe(0);

    destroyWebGpuResidentTensor(hidden.tensor);
    bufferCache.clear();
  });

  it("reuses stable dense matvec descriptor buffers for full-row projections", async () => {
    const device = createCapturingGpuDevice(3);
    const bufferCache = new WebGpuRuntimeBufferCache();
    let projectionRowReads = 0;
    const matrix = [
      [1, 0, 0],
      [0, 1, 0],
      [0.5, 0.5, 1],
    ];
    const projection = {
      rowCount: matrix.length,
      colCount: matrix[0]?.length ?? 0,
      row: (index: number) => {
        projectionRowReads += 1;
        return matrix[index];
      },
    };

    const first = await runDenseMatVecWebGpu({
      vector: [2, -1, 0.5],
      matrix: projection,
      device,
      bufferCache,
      projectionCacheKey: "output_projection",
      projectionCachePolicy: "stable",
    });
    const firstProjectionReads = projectionRowReads;
    projectionRowReads = 0;
    const second = await runDenseMatVecWebGpu({
      vector: [0, 1, 2],
      matrix: projection,
      device,
      bufferCache,
      projectionCacheKey: "output_projection",
      projectionCachePolicy: "stable",
    });

    expect(first.backend).toBe("webgpu");
    expect(second.backend).toBe("webgpu");
    expect(first.trace.projectionCacheHit).toBe(false);
    expect(second.trace.projectionCacheHit).toBe(true);
    expect(first.trace.pipelineCacheHit).toBe(false);
    expect(second.trace.pipelineCacheHit).toBe(true);
    expect(firstProjectionReads).toBe(3);
    expect(projectionRowReads).toBe(0);
    expect(device.floatUploads.filter((upload) => upload.length === 9)).toHaveLength(1);
    expect(device.pipelineEntryPoints.filter((entryPoint) => entryPoint === "main")).toHaveLength(1);
  });

  it("reuses stable MLP projection buffers within a live WebGPU runtime cache", async () => {
    const device = createCapturingGpuDevice(2);
    const bufferCache = new WebGpuRuntimeBufferCache();
    let upReads = 0;
    let gateReads = 0;
    let downReads = 0;
    const upProjection = makeCountingMatrix([
      [1, 0],
      [0, 1],
      [1, 1],
    ], () => { upReads += 1; });
    const gateProjection = makeCountingMatrix([
      [0.5, 0],
      [0, 0.5],
      [1, -1],
    ], () => { gateReads += 1; });
    const downProjection = makeCountingMatrix([
      [1, 0, 0.25],
      [0, 1, -0.5],
    ], () => { downReads += 1; });

    const first = await runMlpWebGpu({
      hidden: [0.25, -0.5],
      upProjection,
      gateProjection,
      downProjection,
      device,
      bufferCache,
      projectionCacheKey: "layer0.mlp",
      projectionCachePolicy: "stable",
    });
    const firstReads = { upReads, gateReads, downReads };
    upReads = 0;
    gateReads = 0;
    downReads = 0;
    const second = await runMlpWebGpu({
      hidden: [0.5, 0.25],
      upProjection,
      gateProjection,
      downProjection,
      device,
      bufferCache,
      projectionCacheKey: "layer0.mlp",
      projectionCachePolicy: "stable",
    });

    expect(first.trace.projectionCacheHits).toEqual({
      upProjection: false,
      gateProjection: false,
      downProjection: false,
    });
    expect(first.trace.pipelineCacheHits).toEqual({
      intermediate: false,
      output: false,
    });
    expect(second.trace.projectionCacheHits).toEqual({
      upProjection: true,
      gateProjection: true,
      downProjection: true,
    });
    expect(second.trace.pipelineCacheHits).toEqual({
      intermediate: true,
      output: true,
    });
    expect(firstReads).toEqual({ upReads: 3, gateReads: 3, downReads: 2 });
    expect({ upReads, gateReads, downReads }).toEqual({ upReads: 0, gateReads: 0, downReads: 0 });
    expect(device.pipelineEntryPoints.filter((entryPoint) => entryPoint === "mlp_intermediate")).toHaveLength(1);
    expect(device.pipelineEntryPoints.filter((entryPoint) => entryPoint === "mlp_output")).toHaveLength(1);
  });

  it("reuses sparse-attention pipelines within a live WebGPU runtime cache", async () => {
    const device = createCapturingGpuDevice(1);
    const bufferCache = new WebGpuRuntimeBufferCache();
    const input = {
      q: [[1, 0]],
      k: [[1, 0]],
      v: [[2, 3]],
      selectedKeyIndexesByQuery: [[0]],
      device,
      bufferCache,
    };

    const first = await runSparseAttentionWebGpu(input);
    const second = await runSparseAttentionWebGpu(input);

    expect(first.trace.pipelineCacheHit).toBe(false);
    expect(second.trace.pipelineCacheHit).toBe(true);
    expect(second.trace.pipelineCacheKey).toBe("sparse-attention:main");
    expect(device.pipelineEntryPoints.filter((entryPoint) => entryPoint === "main")).toHaveLength(1);
    expect(device.shaderCodes).toHaveLength(1);
  });

  it("fuses packed-head sparse attention into one WebGPU dispatch and readback", async () => {
    const device = createCapturingGpuDevice(4);

    const result = await runPackedSparseAttentionWebGpu({
      q: [[1, 0, 0, 1]],
      k: [
        [1, 0, 0, 1],
        [0, 1, 1, 0],
      ],
      v: [
        [2, 3, 5, 7],
        [11, 13, 17, 19],
      ],
      selectedKeyIndexesByQuery: [[0, 1]],
      headCount: 2,
      headDim: 2,
      device,
    });

    expect(result.backend).toBe("webgpu");
    expect(result.trace).toMatchObject({
      packedHeads: true,
      headCount: 2,
      keyValueHeadCount: 2,
      outputSize: 4,
      dispatchCount: 1,
      pipelineCacheKey: "packed-sparse-attention:packed_sparse_attention",
    });
    expect(device.pipelineEntryPoints).toEqual(["packed_sparse_attention"]);
    expect(device.mapReadCount).toBe(1);
    expect(device.shaderCodes[0]).toContain("fn packed_sparse_attention");
    expect(device.shaderCodes[0]).toContain("kv_head_count");
  });

  it("keeps packed-head sparse attention output resident when requested", async () => {
    const device = createCapturingGpuDevice(4);

    const result = await runPackedSparseAttentionResidentWebGpu({
      q: [[1, 0, 0, 1]],
      k: [
        [1, 0, 0, 1],
        [0, 1, 1, 0],
      ],
      v: [
        [2, 3, 5, 7],
        [11, 13, 17, 19],
      ],
      selectedKeyIndexesByQuery: [[0, 1]],
      headCount: 2,
      headDim: 2,
      device,
    });

    expect(result.backend).toBe("webgpu");
    expect(result.tensor).toMatchObject({
      kind: "webgpu_resident_tensor",
      rows: 1,
      cols: 4,
    });
    expect(result.trace).toMatchObject({
      packedHeads: true,
      outputResident: true,
      readback: false,
      inputResident: {
        q: false,
        k: false,
        v: false,
      },
      dispatchCount: 1,
      pipelineCacheKey: "packed-sparse-attention:packed_sparse_attention",
    });
    expect(device.mapReadCount).toBe(0);
  });

  it("uses one-dispatch Qwen attention for full-prefix large single-query contexts", async () => {
    const device = createCapturingGpuDevice(4);
    const keyRows = Array.from({ length: 80 }, (_value, index) => [
      index % 3,
      (index + 1) % 5,
      (index + 2) % 7,
      (index + 3) % 11,
    ]);

    const result = await runPackedSparseAttentionResidentWebGpu({
      q: [[1, 0, 0, 1]],
      k: keyRows,
      v: keyRows,
      selectedKeyIndexesByQuery: [keyRows.map((_row, index) => index)],
      headCount: 2,
      headDim: 2,
      device,
    });

    expect(result.trace).toMatchObject({
      packedHeads: true,
      outputResident: true,
      readback: false,
      dispatchCount: 1,
      pipelineCacheKey: "packed-sparse-attention:qwen_one_token_attention",
      metadata: {
        fusedStage: "one_token_attention",
        oneTokenAttention: true,
        qwenOneTokenAttention: true,
      },
    });
    expect(device.pipelineEntryPoints).toEqual([
      "qwen_one_token_attention",
    ]);
    expect(device.dispatchWorkgroupCount).toBe(1);
    expect(device.shaderCodes[0]).toContain("fn qwen_one_token_attention");
    expect(device.shaderCodes[0]).toContain("var<workgroup> scores");
    expect(device.shaderCodes[0]).toContain("attention_out");
    expect(device.mapReadCount).toBe(0);
  });

  it("uses the score-reuse packed attention pipeline for large prefill chunks", async () => {
    const device = createCapturingGpuDevice(16);
    const keyRows = Array.from({ length: 80 }, (_value, index) => [
      index % 3,
      (index + 1) % 5,
      (index + 2) % 7,
      (index + 3) % 11,
    ]);

    const result = await runPackedSparseAttentionWebGpu({
      q: keyRows.slice(0, 4),
      k: keyRows,
      v: keyRows,
      selectedKeyIndexesByQuery: Array.from({ length: 4 }, () => keyRows.map((_row, index) => index)),
      causal: false,
      headCount: 2,
      headDim: 2,
      device,
    });

    expect(result.trace).toMatchObject({
      packedHeads: true,
      dispatchCount: 2,
      pipelineCacheKey: "packed-sparse-attention:decode_scores+decode_output",
    });
    expect(device.pipelineEntryPoints).toEqual([
      "packed_sparse_attention_decode_scores",
      "packed_sparse_attention_decode_output",
    ]);
    expect(device.dispatchWorkgroupCount).toBe(2);
    expect(device.mapReadCount).toBe(1);
  });

  it("reuses an implicit GPU device for repeated kernel calls against the same GPU adapter", async () => {
    const device = createCapturingGpuDevice(2);
    let requestDeviceCalls = 0;
    const gpu = {
      requestAdapter: async () => ({
        requestDevice: async () => {
          requestDeviceCalls += 1;
          return device;
        },
      }),
    };

    await runDenseMatVecWebGpu({
      vector: [1, 0],
      matrix: [
        [1, 0],
        [0, 1],
      ],
      gpu,
    });
    await runDenseMatVecWebGpu({
      vector: [0, 1],
      matrix: [
        [1, 0],
        [0, 1],
      ],
      gpu,
    });

    expect(requestDeviceCalls).toBe(1);
  });

  it("passes explicit sparse-attention scale into the WebGPU params and shader", async () => {
    const device = createCapturingGpuDevice(2);

    const result = await runSparseAttentionWebGpu({
      q: [[1, 0]],
      k: [[1, 0]],
      v: [[2, 3]],
      selectedKeyIndexesByQuery: [[0]],
      scale: 0.25,
      device,
    });

    expect(result.backend).toBe("webgpu");
    expect(device.uniformUploads.some((upload) => upload.float32.includes(0.25))).toBe(true);
    expect(device.shaderCodes.some((code) => code.includes("params.scale"))).toBe(true);
    expect(device.shaderCodes.some((code) => code.includes("inverseSqrt(f32(params.head_dim))"))).toBe(false);
  });

  it("uploads MLP projections and prepares the MLP shader path for WebGPU execution", async () => {
    const device = createCapturingGpuDevice(2);

    const result = await runMlpWebGpu({
      hidden: [0.25, -0.5],
      upProjection: [
        [1, 0],
        [0, 1],
        [1, 1],
      ],
      gateProjection: [
        [0.5, 0],
        [0, 0.5],
        [1, -1],
      ],
      downProjection: [
        [1, 0, 0.25],
        [0, 1, -0.5],
      ],
      device,
    });

    expect(result.backend).toBe("webgpu");
    expect(result.trace).toMatchObject({
      backend: "webgpu",
      inputSize: 2,
      intermediateSize: 3,
      outputSize: 2,
      activationKind: "silu_gated",
    });
    expect(device.floatUploads).toEqual(expect.arrayContaining([
      [0.25, -0.5],
      [1, 0, 0, 1, 1, 1],
      [1, 0, 0.25, 0, 1, -0.5],
      [0.5, 0, 0, 0.5, 1, -1],
    ]));
    expect(device.shaderCodes.some((code) => code.includes("fn mlp_intermediate"))).toBe(true);
    expect(device.shaderCodes.some((code) => code.includes("fn mlp_output"))).toBe(true);
    const mlpShader = device.shaderCodes.find((code) => code.includes("fn mlp_intermediate") && code.includes("fn mlp_output"));
    expect(mlpShader).toBeDefined();
    for (const entryPoint of ["mlp_intermediate", "mlp_output"]) {
      expectEntryPointReferences(mlpShader ?? "", entryPoint, [
        "hidden",
        "upProjection",
        "gateProjection",
        "downProjection",
        "intermediate",
        "output",
        "params",
      ]);
    }
  });

  it("uploads batched MLP prefill rows and prepares the packed WebGPU shader path", async () => {
    const device = createCapturingGpuDevice(4);

    const result = await runMlpBatchWebGpu({
      hidden: [
        [0.25, -0.5],
        [0.5, 0.25],
      ],
      upProjection: [
        [1, 0],
        [0, 1],
        [1, 1],
      ],
      gateProjection: [
        [0.5, 0],
        [0, 0.5],
        [1, -1],
      ],
      downProjection: [
        [1, 0, 0.25],
        [0, 1, -0.5],
      ],
      device,
    });

    expect(result.backend).toBe("webgpu");
    expect(result.trace).toMatchObject({
      backend: "webgpu",
      tokens: 2,
      inputSize: 2,
      intermediateSize: 3,
      outputSize: 2,
      activationKind: "silu_gated",
    });
    expect(device.floatUploads).toEqual(expect.arrayContaining([
      [0.25, -0.5, 0.5, 0.25],
      [1, 0, 0, 1, 1, 1],
      [1, 0, 0.25, 0, 1, -0.5],
      [0.5, 0, 0, 0.5, 1, -1],
    ]));
    expect(device.uniformUploads.some((upload) => (
      upload.uint32[0] === 2
      && upload.uint32[1] === 3
      && upload.uint32[2] === 2
      && upload.uint32[3] === 2
      && upload.uint32[4] === 1
    ))).toBe(true);
    const mlpShader = device.shaderCodes.find((code) => code.includes("fn mlp_batch_intermediate") && code.includes("fn mlp_batch_output"));
    expect(mlpShader).toBeDefined();
    for (const entryPoint of ["mlp_batch_intermediate", "mlp_batch_output"]) {
      expectEntryPointReferences(mlpShader ?? "", entryPoint, [
        "hidden",
        "upProjection",
        "gateProjection",
        "downProjection",
        "intermediate",
        "output",
        "params",
      ]);
    }
  });

  it("reuses stable batched MLP projection buffers and pipelines within a live WebGPU runtime cache", async () => {
    const device = createCapturingGpuDevice(4);
    const bufferCache = new WebGpuRuntimeBufferCache();
    let upReads = 0;
    let gateReads = 0;
    let downReads = 0;
    const upProjection = makeCountingMatrix([
      [1, 0],
      [0, 1],
      [1, 1],
    ], () => { upReads += 1; });
    const gateProjection = makeCountingMatrix([
      [0.5, 0],
      [0, 0.5],
      [1, -1],
    ], () => { gateReads += 1; });
    const downProjection = makeCountingMatrix([
      [1, 0, 0.25],
      [0, 1, -0.5],
    ], () => { downReads += 1; });

    const first = await runMlpBatchWebGpu({
      hidden: [[0.25, -0.5], [0.5, 0.25]],
      upProjection,
      gateProjection,
      downProjection,
      device,
      bufferCache,
      projectionCacheKey: "layer0.mlp",
      projectionCachePolicy: "stable",
    });
    const firstReads = { upReads, gateReads, downReads };
    upReads = 0;
    gateReads = 0;
    downReads = 0;
    const second = await runMlpBatchWebGpu({
      hidden: [[0.75, -0.25], [0.125, 0.875]],
      upProjection,
      gateProjection,
      downProjection,
      device,
      bufferCache,
      projectionCacheKey: "layer0.mlp",
      projectionCachePolicy: "stable",
    });

    expect(first.trace.projectionCacheHits).toEqual({
      upProjection: false,
      gateProjection: false,
      downProjection: false,
    });
    expect(first.trace.pipelineCacheHits).toEqual({
      intermediate: false,
      output: false,
    });
    expect(second.trace.projectionCacheHits).toEqual({
      upProjection: true,
      gateProjection: true,
      downProjection: true,
    });
    expect(second.trace.pipelineCacheHits).toEqual({
      intermediate: true,
      output: true,
    });
    expect(firstReads).toEqual({ upReads: 3, gateReads: 3, downReads: 2 });
    expect({ upReads, gateReads, downReads }).toEqual({ upReads: 0, gateReads: 0, downReads: 0 });
    expect(device.pipelineEntryPoints.filter((entryPoint) => entryPoint === "mlp_batch_intermediate")).toHaveLength(1);
    expect(device.pipelineEntryPoints.filter((entryPoint) => entryPoint === "mlp_batch_output")).toHaveLength(1);
  });

  it("executes a routed toy sparse layer through the WebGPU SSA backend contract", async () => {
    const backend = new WebGpuSsaReferenceBackend({ backendPreference: "cpu" });
    const q = createSsaToyTensorHandle({
      id: "q",
      matrix: [
        [1, 0],
        [0, 1],
        [1, 1],
        [1, 0.5],
      ],
      blockTokenRanges: {
        b0: { tokenStart: 0, tokenEnd: 2 },
        b1: { tokenStart: 2, tokenEnd: 4 },
      },
    });
    const k = createSsaToyTensorHandle({
      id: "k",
      matrix: [
        [1, 0],
        [0, 1],
        [1, 1],
        [1, 0.5],
      ],
      blockTokenRanges: {
        b0: { tokenStart: 0, tokenEnd: 2 },
        b1: { tokenStart: 2, tokenEnd: 4 },
      },
    });
    const v = createSsaToyTensorHandle({
      id: "v",
      matrix: [
        [10, 0],
        [0, 20],
        [30, 30],
        [40, 10],
      ],
      blockTokenRanges: {
        b0: { tokenStart: 0, tokenEnd: 2 },
        b1: { tokenStart: 2, tokenEnd: 4 },
      },
    });

    const output = await backend.executeSparseForward({
      requestId: "req_webgpu_toy",
      layerIndex: 0,
      qHandle: q,
      kHandle: k,
      vHandle: v,
      routingPolicy: {
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
      },
    });

    const selectedKeyIndexesByQuery = [
      [0, 1],
      [0, 1],
      [0, 1, 2, 3],
      [0, 1, 2, 3],
    ];
    const expected = sparseReferenceAttention({
      q: q.matrix,
      k: k.matrix,
      v: v.matrix,
      selectedKeyIndexesByQuery,
    });

    expect(output.selectedBlockIds).toEqual(["b0", "b1"]);
    expect(readSsaToyTensorHandle(output.outputHandle).matrix).toEqual(expected);
    expect(output.trace).toMatchObject({
      requestId: "req_webgpu_toy",
      layerIndex: 0,
      queryBlockIndex: 0,
      selectedBlockIds: ["b0", "b1"],
      pinnedBlockIds: ["b0"],
      denseTokenCountEstimate: 4,
      sparseTokenCountEstimate: 6,
    });
  });
});

function createCapturingGpuDevice(outputRows: number): {
  floatUploads: number[][];
  intUploads: number[][];
  uniformUploads: Array<{ uint32: number[]; float32: number[] }>;
  writeBufferCount: number;
  submitCount: number;
  dispatchWorkgroupCount: number;
  shaderCodes: string[];
  pipelineEntryPoints: string[];
  mapReadCount: number;
  queue: {
    writeBuffer(buffer: CapturingGpuBuffer, bufferOffset: number, data: ArrayBufferLike, dataOffset?: number, size?: number): void;
    submit(commandBuffers: unknown[]): void;
  };
  createBuffer(descriptor: { size: number; usage: number }): CapturingGpuBuffer;
  createShaderModule(descriptor: { code: string }): unknown;
  createComputePipeline(descriptor?: { compute?: { entryPoint?: string } }): { getBindGroupLayout(index: number): unknown };
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
  const intUploads: number[][] = [];
  const uniformUploads: Array<{ uint32: number[]; float32: number[] }> = [];
  const shaderCodes: string[] = [];
  const pipelineEntryPoints: string[] = [];
  const device = {
    floatUploads,
    intUploads,
    uniformUploads,
    writeBufferCount: 0,
    submitCount: 0,
    dispatchWorkgroupCount: 0,
    shaderCodes,
    pipelineEntryPoints,
    mapReadCount: 0,
    queue: {
      writeBuffer(
        _buffer: CapturingGpuBuffer,
        _bufferOffset: number,
        data: ArrayBufferLike,
        dataOffset = 0,
        size = data.byteLength - dataOffset,
      ) {
        device.writeBufferCount += 1;
        const bytes = data.slice(dataOffset, dataOffset + size);
        if (size % Float32Array.BYTES_PER_ELEMENT === 0) floatUploads.push([...new Float32Array(bytes)]);
        if (size % Int32Array.BYTES_PER_ELEMENT === 0) intUploads.push([...new Int32Array(bytes)]);
        if (size === 16 || size === 32) {
          uniformUploads.push({
            uint32: [...new Uint32Array(bytes)],
            float32: [...new Float32Array(bytes)],
          });
        }
      },
      submit() {
        device.submitCount += 1;
        // The fake device only validates upload shape; it does not execute WGSL.
      },
    },
    createBuffer(descriptor: { size: number; usage: number }) {
      return new CapturingGpuBuffer(descriptor.size, () => {
        device.mapReadCount += 1;
      });
    },
    createShaderModule(descriptor: { code: string }) {
      shaderCodes.push(descriptor.code);
      return {};
    },
    createComputePipeline(descriptor?: { compute?: { entryPoint?: string } }) {
      pipelineEntryPoints.push(descriptor?.compute?.entryPoint ?? "unknown");
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
          dispatchWorkgroups: () => {
            device.dispatchWorkgroupCount += 1;
          },
          end: () => undefined,
        }),
        copyBufferToBuffer: (
          _source: CapturingGpuBuffer,
          _sourceOffset: number,
          destination: CapturingGpuBuffer,
          _destinationOffset: number,
          size: number,
        ) => {
          destination.bytes = new ArrayBuffer(size || outputRows * Float32Array.BYTES_PER_ELEMENT);
        },
        finish: () => ({}),
      };
    },
  };
  return device;
}

function makeCountingMatrix(matrix: number[][], onRead: () => void): {
  rowCount: number;
  colCount: number;
  row(index: number): number[] | undefined;
} {
  return {
    rowCount: matrix.length,
    colCount: matrix[0]?.length ?? 0,
    row(index: number) {
      onRead();
      return matrix[index];
    },
  };
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

function expectVectorCloseTo(actual: number[], expected: number[]): void {
  expect(actual).toHaveLength(expected.length);
  for (let index = 0; index < expected.length; index += 1) {
    expect(actual[index]).toBeCloseTo(expected[index] ?? 0, 6);
  }
}

function expectMatrixCloseTo(actual: number[][], expected: number[][]): void {
  expect(actual).toHaveLength(expected.length);
  for (let row = 0; row < expected.length; row += 1) {
    expectVectorCloseTo(actual[row] ?? [], expected[row] ?? []);
  }
}

function gelu(value: number): number {
  return 0.5 * value * (1 + Math.tanh(Math.sqrt(2 / Math.PI) * (value + 0.044715 * value ** 3)));
}

function silu(value: number): number {
  return value / (1 + Math.exp(-value));
}

function expectEntryPointReferences(shader: string, entryPoint: string, resources: string[]): void {
  const body = extractWgslFunctionBody(shader, entryPoint);
  for (const resource of resources) {
    expect(body, `${entryPoint} should reference ${resource} so auto layout includes its binding`).toContain(resource);
  }
}

function extractWgslFunctionBody(shader: string, entryPoint: string): string {
  const signatureIndex = shader.indexOf(`fn ${entryPoint}`);
  expect(signatureIndex).toBeGreaterThanOrEqual(0);
  const bodyStart = shader.indexOf("{", signatureIndex);
  expect(bodyStart).toBeGreaterThanOrEqual(0);
  let depth = 0;
  for (let index = bodyStart; index < shader.length; index += 1) {
    const char = shader[index];
    if (char === "{") depth += 1;
    if (char === "}") {
      depth -= 1;
      if (depth === 0) return shader.slice(bodyStart + 1, index);
    }
  }
  throw new Error(`Could not extract WGSL body for ${entryPoint}.`);
}
