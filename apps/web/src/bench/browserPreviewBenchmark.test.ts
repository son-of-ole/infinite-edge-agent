import { describe, expect, it, vi } from "vitest";
import {
  buildDeterministicLongPrompt,
  buildBrowserPreviewBenchmarkPayload,
  MAX_LONG_PROMPT_REPEAT,
  MAX_LONG_PROMPT_SEED_WORDS,
  MAX_LONG_PROMPT_TARGET_TOKENS,
  type BrowserPreviewBenchmarkRun,
} from "./browserPreviewBenchmark";
import {
  buildBrowserPreviewRunPrompts,
  buildBrowserPreviewBenchmarkFailurePayload,
  buildStopAfterSequences,
  buildWebGpuProof,
  calculateBrowserPreviewRunTiming,
  isCoherentBrowserPreviewResponse,
  readBrowserPreviewBenchmarkRequest,
  runBrowserPreviewBenchmarkWithExclusiveLock,
} from "./browserPreviewBenchmarkRoute";

describe("browser preview benchmark payload", () => {
  it("summarizes coherent response, runtime trace, predictive counts, backend coverage, MTP, and KV persistence events", () => {
    const run: BrowserPreviewBenchmarkRun = {
      promptId: "prompt-1",
      prompt: "Explain persistent runtime intelligence.",
      response: "[unlocked:ssa-kv-tsp] Persistent runtime intelligence keeps model, memory, and runtime state distinct.",
      coherent: true,
      expectedSubstrings: [],
      expectedSubstringMatches: [],
      metrics: {
        initLoadMs: 20,
        warmupMs: 7,
        prefillMs: 8,
        timeToFirstTokenMs: 11,
        decodeLatencyMs: 6,
        tokensPerSecond: 166.67,
        generatedTokens: 1,
      },
      runtimeTrace: {
        backend: "unlocked-browser-transformer",
        tensorControl: true,
        tspSteps: ["kv_prefetch", "attention", "mlp"],
        kvPagingEvents: 1,
        selectedBlockIds: ["b0", "b1"],
      },
      predictive: {
        promptTokenCount: 9,
        generatedTokenCount: 1,
        selectedBlockCount: 2,
        kvPagingEventCount: 1,
        tspStepCount: 3,
      },
      webGpu: {
        available: true,
        requestedBackendPreference: "webgpu",
        logitProjectionBackend: "webgpu",
        logitProjectionReadbackStrategy: "gpu_top1_candidates",
        logitProjectionGpuReducedRows: 151936,
        logitProjectionReadbackRows: 2374,
        logitProjectionReadbackBytes: 18992,
        logitProjectionDispatchCount: 38,
        logitProjectionTiles: 38,
        logitProjectionTileRows: 4096,
        cpuFallbackUsed: false,
        noCpuFallback: true,
        requestedGates: ["logits"],
        passedGates: ["logits"],
        failedGates: [],
        positiveKernelProof: true,
      },
      mtp: {
        mode: "draft_verify",
        draftModelId: "browser/qwen-prefix-drafter",
        draftSource: "qwen_prefix_draft",
        latencyDisablePolicy: "paired_benchmark_required",
        acceptedTokens: 1,
        rejectedTokens: 0,
        acceptanceRate: 1,
        numSpeculativeTokens: 2,
        verifiedTokenCount: 2,
        targetDecodeCalls: 1,
        verifierStrategy: "batched_continuation",
      },
      kvPersistence: {
        enabled: true,
        mode: "indexeddb",
        eventCount: 2,
        persistEvents: 1,
        hydrateEvents: 1,
        reuseEvents: 0,
        prefetchStrategy: "predictive_prefetch",
        lowRankSummaryRank: 4,
        predictedHotBlocks: ["b0", "b1"],
        prefetchedBlocks: ["b0"],
        prefetchHitRate: 0.5,
        prefetchBytes: 2048,
        prefetchLatencyMs: 3.5,
        attentionStallMs: 0,
      },
    };
    Object.assign(run, {
      prefillChunkCount: 2,
      prefillChunkSize: 1024,
      shapeBucket: "prompt<=1024:selected<=512:headDim<=128:tileRows<=8192:precision=f32",
      pipelineCacheKey: "prefill_chunk:prompt<=1024:selected<=512:headDim<=128:tileRows<=8192:precision=f32",
      maxDispatchEstimatedMs: 7.5,
    });

    const payload = buildBrowserPreviewBenchmarkPayload({
      createdAt: "2026-05-18T00:00:00.000Z",
      profile: "ci",
      runs: [run],
      strictWebGpuRequested: true,
    });

    expect(payload.passed).toBe(true);
    expect(payload.summary).toMatchObject({
      profile: "ci",
      memoryQueryMode: "direct_model_no_memory_retrieval",
      promptCount: 1,
      coherentResponseCount: 1,
      runtimeTraceCount: 1,
      meanWarmupMs: 7,
      predictiveSelectedBlockCount: 2,
      predictiveKvPagingEventCount: 1,
      webGpuAvailable: true,
      strictWebGpuRequested: true,
      cpuFallbackUsed: false,
      noCpuFallback: true,
      positiveWebGpuKernelProof: true,
      logitProjectionReadbackStrategy: "gpu_top1_candidates",
      logitProjectionGpuReducedRows: 151936,
      logitProjectionReadbackRows: 2374,
      logitProjectionReadbackBytes: 18992,
      logitProjectionDispatchCount: 38,
      logitProjectionTiles: 38,
      logitProjectionTileRows: 4096,
      mtpMode: "draft_verify",
      mtpMaxSpeculativeTokens: 2,
      mtpVerifiedTokenCount: 2,
      mtpTargetDecodeCalls: 1,
      mtpVerifierStrategy: "batched_continuation",
      kvPersistenceEventCount: 2,
      kvPrefetchStrategy: "predictive_prefetch",
      kvPredictivePrefetchRunCount: 1,
      kvExactReuseRunCount: 0,
      kvMissStallRunCount: 0,
      kvLowRankSummaryRank: 4,
      kvPredictedHotBlockCount: 2,
      kvPrefetchedBlockCount: 1,
      kvPrefetchHitRate: 0.5,
      kvPrefetchBytes: 2048,
      kvPrefetchLatencyMs: 3.5,
      kvAttentionStallMs: 0,
      prefillChunkCount: 2,
      prefillChunkSize: 1024,
      shapeBucket: "prompt<=1024:selected<=512:headDim<=128:tileRows<=8192:precision=f32",
      pipelineCacheKey: "prefill_chunk:prompt<=1024:selected<=512:headDim<=128:tileRows<=8192:precision=f32",
      maxDispatchEstimatedMs: 7.5,
      technicalProofOnly: false,
      productionQualityPassed: false,
      productionDeployReadyPassed: false,
    });
    expect(payload.runs[0] as BrowserPreviewBenchmarkRun & { prefillChunkCount: number }).toMatchObject({
      prefillChunkCount: 2,
      prefillChunkSize: 1024,
    });
  });

  it("uses suffix stops instead of newline stops for compiled exact-output canaries", () => {
    const stops = buildStopAfterSequences(
      { backendId: "compiled-browser-webllm" } as never,
      { expectedSubstrings: [], expectedExact: ["Helena"] },
      true,
    );

    expect(stops).toEqual(expect.arrayContaining([".", "!", "?"]));
    expect(stops).not.toContain("\n");
    expect(stops).not.toContain("Helena");
  });

  it("fails strict WebGPU preview when no CPU fallback is seen but positive WebGPU proof is missing", () => {
    const run = makeRun({
      webGpu: {
        available: true,
        requestedBackendPreference: "webgpu",
        logitProjectionBackend: "unknown",
        cpuFallbackUsed: false,
        noCpuFallback: true,
        requestedGates: ["logits"],
        passedGates: [],
        failedGates: ["logits"],
        positiveKernelProof: false,
      },
    });

    const payload = buildBrowserPreviewBenchmarkPayload({
      createdAt: "2026-05-18T00:00:00.000Z",
      profile: "full",
      runs: [run],
      strictWebGpuRequested: true,
    });

    expect(payload.passed).toBe(false);
    expect(payload.summary).toMatchObject({
      strictWebGpuRequested: true,
      noCpuFallback: true,
      positiveWebGpuKernelProof: false,
      strictWebGpuPassed: false,
    });
  });

  it("fails strict browser WebGPU attention proof when packed-head proof count is incomplete", () => {
    const webGpu = buildWebGpuProof({
      tensorControl: true,
      tspSteps: [],
      selectedBlockIds: [],
      kvPagingEvents: 0,
      tokenId: 1,
      expectedLayerCount: 1,
      mlpKernelBackends: [{ layerIndex: 0, backend: "webgpu", activationKind: "silu_gated" }],
      prefillMlpKernelBackends: [{ layerIndex: 0, backend: "webgpu", activationKind: "silu_gated", rowCount: 4 }],
      logitProjectionBackend: "webgpu",
      logitProjectionPurpose: "full_vocab_topk_logit_projection",
      logitProjectionSelectedRows: 4,
      logitProjectionFullRows: 151936,
      prefillProjectionBackends: [
        { layerIndex: 0, qProjection: "webgpu", kProjection: "webgpu", vProjection: "webgpu", oProjection: "webgpu" },
      ],
      decodeProjectionBackends: [
        { layerIndex: 0, qProjection: "webgpu", kProjection: "webgpu", vProjection: "webgpu", oProjection: "webgpu", projectionKind: "matmul", tokens: 1 },
      ],
      prefillAttentionBackends: [
        { layerIndex: 0, attentionBackend: "webgpu", packedHeadBackends: ["webgpu"], packedHeadCount: 2 },
      ],
      decodeAttentionBackends: [
        { layerIndex: 0, attentionBackend: "webgpu", packedHeadBackends: ["webgpu"], packedHeadCount: 1 },
      ],
    }, "webgpu", ["mlp", "logits", "attention", "projection"]);

    expect(webGpu.cpuFallbackUsed).toBe(true);
    expect(webGpu.noCpuFallback).toBe(false);
    expect(webGpu.passedGates).toEqual(["mlp", "logits", "projection"]);
    expect(webGpu.failedGates).toEqual(["attention"]);
    expect(webGpu.positiveKernelProof).toBe(false);
  });

  it("passes strict browser WebGPU attention proof when chunked prefill records repeated packed-head dispatches", () => {
    const webGpu = buildWebGpuProof({
      tensorControl: true,
      tspSteps: [],
      selectedBlockIds: [],
      kvPagingEvents: 0,
      tokenId: 1,
      expectedLayerCount: 1,
      mlpKernelBackends: [{ layerIndex: 0, backend: "webgpu", activationKind: "silu_gated" }],
      prefillMlpKernelBackends: [{ layerIndex: 0, backend: "webgpu", activationKind: "silu_gated", rowCount: 64 }],
      logitProjectionBackend: "webgpu",
      logitProjectionPurpose: "full_vocab_topk_logit_projection",
      logitProjectionSelectedRows: 4,
      logitProjectionFullRows: 151936,
      prefillProjectionBackends: [
        { layerIndex: 0, qProjection: "webgpu", kProjection: "webgpu", vProjection: "webgpu", oProjection: "webgpu" },
      ],
      decodeProjectionBackends: [
        { layerIndex: 0, qProjection: "webgpu", kProjection: "webgpu", vProjection: "webgpu", oProjection: "webgpu", projectionKind: "matmul", tokens: 1 },
      ],
      prefillAttentionBackends: [
        {
          layerIndex: 0,
          attentionBackend: "webgpu",
          packedHeadBackends: ["webgpu", "webgpu", "webgpu", "webgpu"],
          packedHeadCount: 2,
        },
      ],
      decodeAttentionBackends: [
        { layerIndex: 0, attentionBackend: "webgpu", packedHeadBackends: ["webgpu", "webgpu"], packedHeadCount: 2 },
      ],
    }, "webgpu", ["mlp", "logits", "attention", "projection"]);

    expect(webGpu.cpuFallbackUsed).toBe(false);
    expect(webGpu.noCpuFallback).toBe(true);
    expect(webGpu.passedGates).toEqual(["mlp", "logits", "attention", "projection"]);
    expect(webGpu.failedGates).toEqual([]);
    expect(webGpu.positiveKernelProof).toBe(true);
  });

  it("surfaces decode hot-path counters from the unlocked decode proof", () => {
    const webGpu = buildWebGpuProof({
      tensorControl: true,
      tspSteps: [],
      selectedBlockIds: [],
      kvPagingEvents: 0,
      tokenId: 1,
      logitProjectionBackend: "webgpu",
      logitProjectionReadbackStrategy: "gpu_argmax_token_id",
      logitProjectionReadbackRows: 1,
      logitProjectionReadbackBytes: 8,
      decodePerf: {
        requestId: "req_hot_path_browser",
        generatedTokenCount: 1,
        decodeCallCount: 1,
        decodeSubmitCount: 10,
        dispatchCount: 19,
        decodeSubmitCountPerToken: 10,
        decodeDispatchCountPerToken: 19,
        decodeDispatchCountPerLayerPerToken: 19,
        readbackCount: 1,
        totalReadbackRows: 1,
        totalReadbackBytes: 8,
        fullLogitsReadbackCount: 0,
        compactLogitReadbackCount: 1,
        weightUploadBytesDuringDecode: 0,
        weightUploadCountDuringDecode: 0,
        activationUploadBytesDuringDecode: 0,
        activationUploadCountDuringDecode: 0,
        hiddenReadbackCountDuringDecode: 0,
        f32ExpansionCountDuringDecode: 0,
        f32ExpansionBytesDuringDecode: 0,
        cpuFallbackUsed: false,
        cpuValidationUsed: false,
        prefillExecutionsDuringDecode: 0,
        prefillCountPerGeneratedToken: 0,
        residentDecodeLayerCount: 1,
        totalDecodeLayerCount: 1,
        residentDecodeLayerCoverage: 1,
        residentFinalHiddenUsedForLogits: true,
        kvDecodeReused: true,
        fusedPackedQkvLayerCount: 0,
        fusedQkvNormRopeKvAppendLayerCount: 0,
        fusedOneTokenAttentionLayerCount: 0,
        fusedResidualRmsNormLayerCount: 0,
        fusedMlpLayerCount: 0,
        fusedFullLayerCount: 0,
        fusedLayerCoverage: 0,
        tokensPerSecond: 1.4,
      },
    }, "webgpu", []);

    expect(webGpu).toMatchObject({
      logitProjectionReadbackStrategy: "gpu_argmax_token_id",
      logitProjectionReadbackRows: 1,
      logitProjectionReadbackBytes: 8,
      decodeDispatchCount: 19,
      decodeSubmitCount: 10,
      decodeSubmitCountPerToken: 10,
      decodeDispatchCountPerToken: 19,
      decodeDispatchCountPerLayerPerToken: 19,
      decodeReadbackCount: 1,
      decodeReadbackBytes: 8,
      fullLogitsReadbackCount: 0,
      compactLogitReadbackCount: 1,
      weightUploadBytesDuringDecode: 0,
      activationUploadBytesDuringDecode: 0,
      hiddenReadbackCountDuringDecode: 0,
      f32ExpansionCountDuringDecode: 0,
      cpuValidationUsed: false,
      prefillCountPerGeneratedToken: 0,
      residentDecodeLayerCoverage: 1,
      residentFinalHiddenUsedForLogits: true,
      fusedLayerCoverage: 0,
    });
  });

  it("summarizes v11 fused decode counters and reports submit fragmentation before fused-stage coverage", () => {
    const run = makeRun();
    Object.assign(run.metrics, {
      tokensPerSecond: 1.45,
      generatedTokens: 2,
      decodeLatencyMs: 1_379,
    });
    Object.assign(run.webGpu, {
      decodeSubmitCount: 80,
      decodeDispatchCount: 60,
      totalDecodeLayerCount: 56,
      residentDecodeLayerCoverage: 1,
      residentFinalHiddenUsedForLogits: true,
      fusedPackedQkvLayerCount: 0,
      fusedQkvNormRopeKvAppendLayerCount: 0,
      fusedOneTokenAttentionLayerCount: 0,
      fusedResidualRmsNormLayerCount: 0,
      fusedMlpLayerCount: 0,
      fusedFullLayerCount: 0,
      fusedLayerCoverage: 0,
    });

    const payload = buildBrowserPreviewBenchmarkPayload({
      createdAt: "2026-05-18T00:00:00.000Z",
      profile: "full",
      runs: [run],
      strictWebGpuRequested: true,
    });

    expect(payload.summary).toMatchObject({
      decodeSubmitCount: 80,
      decodeSubmitCountPerToken: 40,
      decodeDispatchCount: 60,
      decodeDispatchCountPerToken: 30,
      decodeDispatchCountPerLayerPerToken: 1.071,
      v11CommandBatchingPassed: false,
      fusedPackedQkvLayerCount: 0,
      fusedFullLayerCount: 0,
      fusedLayerCoverage: 0,
      parityRecordCount: 0,
      parityPassedCount: 0,
      parityFailedCount: 0,
      productionSpeedFloorPassed: false,
      primarySpeedBottleneck: "submit_fragmentation",
    });
  });

  it("fails benchmark payload when strict decode hot path falls back to full logits readback", () => {
    const run = makeRun({
      webGpu: {
        available: true,
        requestedBackendPreference: "webgpu",
        logitProjectionBackend: "webgpu",
        cpuFallbackUsed: false,
        noCpuFallback: true,
        requestedGates: [],
        passedGates: [],
        failedGates: [],
        positiveKernelProof: true,
        fullLogitsReadbackCount: 1,
        compactLogitReadbackCount: 0,
        weightUploadBytesDuringDecode: 0,
        activationUploadBytesDuringDecode: 0,
        hiddenReadbackCountDuringDecode: 0,
        f32ExpansionCountDuringDecode: 0,
        prefillCountPerGeneratedToken: 0,
        residentDecodeLayerCoverage: 1,
        residentFinalHiddenUsedForLogits: true,
      },
    });
    const payload = buildBrowserPreviewBenchmarkPayload({
      createdAt: "2026-05-23T00:00:00.000Z",
      profile: "balanced",
      strictWebGpuRequested: true,
      runs: [run],
    });

    expect(payload.passed).toBe(false);
    expect(payload.summary).toMatchObject({
      decodeHotPathPassed: false,
      decodeHotPathFailureCount: 1,
      fullLogitsReadbackCount: 1,
    });
  });

  it("fails benchmark payload when strict decode moves activations or misses resident final hidden", () => {
    const run = makeRun({
      webGpu: {
        available: true,
        requestedBackendPreference: "webgpu",
        logitProjectionBackend: "webgpu",
        cpuFallbackUsed: false,
        noCpuFallback: true,
        requestedGates: [],
        passedGates: [],
        failedGates: [],
        positiveKernelProof: true,
        fullLogitsReadbackCount: 0,
        compactLogitReadbackCount: 8,
        weightUploadBytesDuringDecode: 0,
        activationUploadBytesDuringDecode: 4096,
        activationUploadCountDuringDecode: 8,
        hiddenReadbackCountDuringDecode: 7,
        f32ExpansionCountDuringDecode: 0,
        prefillCountPerGeneratedToken: 0,
        residentDecodeLayerCoverage: 0.875,
        residentFinalHiddenUsedForLogits: false,
      },
    });
    const payload = buildBrowserPreviewBenchmarkPayload({
      createdAt: "2026-05-23T00:00:00.000Z",
      profile: "balanced",
      strictWebGpuRequested: true,
      runs: [run],
    });

    expect(payload.passed).toBe(false);
    expect(payload.summary).toMatchObject({
      decodeHotPathPassed: false,
      decodeHotPathFailureCount: 1,
      primarySpeedBottleneck: "activation_upload_during_decode",
      activationUploadBytesDuringDecode: 4096,
      hiddenReadbackCountDuringDecode: 7,
      residentDecodeLayerCoverage: 0.875,
      residentFinalHiddenUsedForLogits: false,
    });
  });

  it("fails strict browser WebGPU attention proof when packed-head evidence is empty", () => {
    const webGpu = buildWebGpuProof({
      tensorControl: true,
      tspSteps: [],
      selectedBlockIds: [],
      kvPagingEvents: 0,
      tokenId: 1,
      prefillAttentionBackends: [
        { layerIndex: 0, attentionBackend: "webgpu", packedHeadBackends: [], packedHeadCount: 0 },
      ],
      decodeAttentionBackends: [
        { layerIndex: 0, attentionBackend: "webgpu", packedHeadBackends: [], packedHeadCount: 0 },
      ],
    }, "webgpu", ["attention"]);

    expect(webGpu.cpuFallbackUsed).toBe(true);
    expect(webGpu.noCpuFallback).toBe(false);
    expect(webGpu.passedGates).toEqual([]);
    expect(webGpu.failedGates).toEqual(["attention"]);
    expect(webGpu.positiveKernelProof).toBe(false);
  });

  it("fails strict browser projection proof when prefill Q/K/V/O projection evidence is missing", () => {
    const webGpu = buildWebGpuProof({
      tensorControl: true,
      tspSteps: [],
      selectedBlockIds: [],
      kvPagingEvents: 0,
      tokenId: 1,
      decodeProjectionBackends: [
        { layerIndex: 0, oProjection: "webgpu", projectionKind: "matmul", tokens: 1 },
      ],
    }, "webgpu", ["projection"]);

    expect(webGpu.passedGates).toEqual([]);
    expect(webGpu.failedGates).toEqual(["projection"]);
    expect(webGpu.positiveKernelProof).toBe(false);
  });

  it("fails strict browser projection proof when decode Q/K/V evidence is missing", () => {
    const webGpu = buildWebGpuProof({
      tensorControl: true,
      tspSteps: [],
      selectedBlockIds: [],
      kvPagingEvents: 0,
      tokenId: 1,
      expectedLayerCount: 1,
      prefillProjectionBackends: [
        { layerIndex: 0, qProjection: "webgpu", kProjection: "webgpu", vProjection: "webgpu", oProjection: "webgpu" },
      ],
      decodeProjectionBackends: [
        { layerIndex: 0, oProjection: "webgpu", projectionKind: "matmul", tokens: 1 },
      ],
    }, "webgpu", ["projection"]);

    expect(webGpu.passedGates).toEqual([]);
    expect(webGpu.failedGates).toEqual(["projection"]);
    expect(webGpu.positiveKernelProof).toBe(false);
  });

  it("can require browser KV reuse instead of only reporting KV events", () => {
    const withoutReuse = buildBrowserPreviewBenchmarkPayload({
      createdAt: "2026-05-18T00:00:00.000Z",
      profile: "full",
      runs: [makeRun({ kvPersistence: { enabled: true, mode: "indexeddb", eventCount: 1, persistEvents: 1, hydrateEvents: 0, reuseEvents: 0 } })],
      strictWebGpuRequested: false,
      requireKvReuse: true,
    });
    const withReuse = buildBrowserPreviewBenchmarkPayload({
      createdAt: "2026-05-18T00:00:00.000Z",
      profile: "full",
      runs: [makeRun({ kvPersistence: { enabled: true, mode: "indexeddb", eventCount: 2, persistEvents: 1, hydrateEvents: 1, reuseEvents: 1 } })],
      strictWebGpuRequested: false,
      requireKvReuse: true,
    });

    expect(withoutReuse.passed).toBe(false);
    expect(withoutReuse.summary.kvReusePassed).toBe(false);
    expect(withReuse.passed).toBe(true);
    expect(withReuse.summary.kvReusePassed).toBe(true);
  });

  it("can require low-rank predictive KV prefetch instead of accepting exact-only hydration proof", () => {
    const exactOnly = buildBrowserPreviewBenchmarkPayload({
      createdAt: "2026-05-18T00:00:00.000Z",
      profile: "full",
      runs: [makeRun({
        kvPersistence: {
          enabled: true,
          mode: "indexeddb",
          eventCount: 2,
          persistEvents: 1,
          hydrateEvents: 1,
          reuseEvents: 1,
          prefetchStrategy: "exact_reuse",
          predictedHotBlocks: [],
          prefetchedBlocks: ["kv0"],
        },
      })],
      strictWebGpuRequested: false,
      requireKvPredictivePrefetch: true,
    });
    const predictive = buildBrowserPreviewBenchmarkPayload({
      createdAt: "2026-05-18T00:00:00.000Z",
      profile: "full",
      runs: [makeRun({
        kvPersistence: {
          enabled: true,
          mode: "indexeddb",
          eventCount: 3,
          persistEvents: 1,
          hydrateEvents: 1,
          reuseEvents: 0,
          prefetchStrategy: "predictive_prefetch",
          lowRankSummaryRank: 4,
          lowRankQuerySource: "persisted_q_rows",
          predictedHotBlocks: ["kv-hot"],
          prefetchedBlocks: ["kv-hot"],
        },
      })],
      strictWebGpuRequested: false,
      requireKvPredictivePrefetch: true,
    });

    expect(exactOnly.passed).toBe(false);
    expect(exactOnly.summary).toMatchObject({
      requireKvPredictivePrefetch: true,
      kvPredictivePrefetchPassed: false,
      kvPredictivePrefetchRunCount: 0,
      kvPredictedHotBlockCount: 0,
      kvPrefetchedBlockCount: 1,
    });
    expect(predictive.passed).toBe(true);
    expect(predictive.summary).toMatchObject({
      requireKvPredictivePrefetch: true,
      kvPredictivePrefetchPassed: true,
      kvPredictivePrefetchRunCount: 1,
      kvLowRankSummaryRank: 4,
      kvLowRankQuerySource: "persisted_q_rows",
      kvPredictedHotBlockCount: 1,
      kvPrefetchedBlockCount: 1,
    });
  });

  it("fails closed when another browser benchmark tab already owns the exclusive lock", async () => {
    const task = vi.fn(async () => "ran");
    const lockCalls: Array<{ name: string; options: unknown }> = [];
    const locks = {
      request: async <T>(
        name: string,
        options: { mode: "exclusive"; ifAvailable: true },
        callback: (lock: unknown | null) => T | Promise<T>,
      ): Promise<T> => {
        lockCalls.push({ name, options });
        return callback(null);
      },
    };

    await expect(runBrowserPreviewBenchmarkWithExclusiveLock(locks, task)).rejects.toThrow(/already running/i);
    expect(task).not.toHaveBeenCalled();
    expect(lockCalls).toEqual([{ name: "edge-ai-browser-runtime-benchmark", options: { mode: "exclusive", ifAvailable: true } }]);
  });

  it("runs browser benchmarks normally when Web Locks are unavailable", async () => {
    await expect(runBrowserPreviewBenchmarkWithExclusiveLock(undefined, async () => "ran")).resolves.toBe("ran");
  });

  it("fails marker-only responses even when the run claims coherence and has strict WebGPU plus KV reuse proof", () => {
    const payload = buildBrowserPreviewBenchmarkPayload({
      createdAt: "2026-05-18T00:00:00.000Z",
      profile: "full",
      runs: [makeRun({
        response: "[unlocked:ssa-kv-tsp]",
        coherent: true,
        metrics: {
          initLoadMs: 20,
          prefillMs: 8,
          timeToFirstTokenMs: 11,
          decodeLatencyMs: 6,
          tokensPerSecond: 166.67,
          generatedTokens: 8,
        },
        predictive: {
          promptTokenCount: 9,
          generatedTokenCount: 8,
          selectedBlockCount: 2,
          kvPagingEventCount: 1,
          tspStepCount: 3,
        },
        kvPersistence: {
          enabled: true,
          mode: "indexeddb",
          eventCount: 3,
          persistEvents: 1,
          hydrateEvents: 1,
          reuseEvents: 1,
        },
      })],
      strictWebGpuRequested: true,
      requireKvReuse: true,
      minGeneratedTokens: 8,
    });

    expect(payload.passed).toBe(false);
    expect(payload.summary).toMatchObject({
      coherentResponseCount: 0,
      visibleResponseQualityPassed: false,
      strictWebGpuPassed: true,
      kvReusePassed: true,
      minGeneratedTokensPassed: true,
    });
  });

  it("lets technical long-prompt proof bypass prose quality but still fails runaway repetition", () => {
    const payload = buildBrowserPreviewBenchmarkPayload({
      createdAt: "2026-05-18T00:00:00.000Z",
      profile: "balanced",
      runs: [makeRun({
        response: "[unlocked:ssa-kv-tsp] languageaheadaheadahead",
        coherent: false,
        metrics: {
          initLoadMs: 20,
          prefillMs: 120,
          timeToFirstTokenMs: 130,
          decodeLatencyMs: 80,
          tokensPerSecond: 100,
          generatedTokens: 8,
        },
      })],
      strictWebGpuRequested: true,
      minGeneratedTokens: 8,
      technicalProofOnly: true,
    });

    expect(payload.passed).toBe(false);
    expect(payload.summary.visibleResponseQualityPassed).toBe(false);
    expect(payload.summary.technicalProofOnly).toBe(true);
    expect(payload.summary.productionQualityPassed).toBe(false);
    expect(payload.summary.productionDeployReadyPassed).toBe(false);
    expect(payload.summary.stopQualityPassed).toBe(false);
    expect(payload.summary.stopQualityFailureCount).toBe(1);
    expect(payload.summary.runawayRepetitionPassed).toBe(false);
    expect(payload.summary.runawayRepetitionFailureCount).toBe(1);
    expect(payload.summary.strictWebGpuPassed).toBe(true);
    expect(payload.summary.minGeneratedTokensPassed).toBe(true);
  });

  it("fails marker-only output even in technical proof mode", () => {
    const payload = buildBrowserPreviewBenchmarkPayload({
      createdAt: "2026-05-18T00:00:00.000Z",
      profile: "balanced",
      runs: [makeRun({
        response: "[unlocked:ssa-kv-tsp]",
        coherent: false,
        metrics: {
          initLoadMs: 20,
          prefillMs: 120,
          timeToFirstTokenMs: 130,
          decodeLatencyMs: 80,
          tokensPerSecond: 100,
          generatedTokens: 8,
        },
      })],
      strictWebGpuRequested: true,
      minGeneratedTokens: 8,
      technicalProofOnly: true,
    });

    expect(payload.passed).toBe(false);
    expect(payload.summary.visibleResponseQualityPassed).toBe(false);
    expect(payload.summary.technicalProofOnly).toBe(true);
    expect(payload.summary.productionQualityPassed).toBe(false);
    expect(payload.summary.productionDeployReadyPassed).toBe(false);
    expect(payload.summary.markerOnlyResponsePassed).toBe(false);
    expect(payload.summary.markerOnlyResponseFailureCount).toBe(1);
    expect(payload.summary.strictWebGpuPassed).toBe(true);
    expect(payload.summary.minGeneratedTokensPassed).toBe(true);
  });

  it("does not let hidden thinking text satisfy visible quality or expected substring gates", () => {
    const payload = buildBrowserPreviewBenchmarkPayload({
      createdAt: "2026-05-18T00:00:00.000Z",
      profile: "full",
      runs: [makeRun({
        prompt: "What is the capital of Utah?",
        response: "<think>Salt Lake City is the answer with lots of coherent hidden text.</think>The capital of Utah is Great Falls.",
        coherent: true,
        expectedSubstrings: ["Salt Lake"],
        expectedSubstringMatches: ["Salt Lake"],
      })],
      strictWebGpuRequested: false,
    });

    expect(payload.passed).toBe(false);
    expect(payload.summary).toMatchObject({
      coherentResponseCount: 1,
      visibleResponseQualityPassed: true,
      expectedSubstringPassCount: 0,
      expectedSubstringsPassed: false,
    });
  });

  it("can require expected substrings for arbitrary prompt quality proof", () => {
    const missingExpected = buildBrowserPreviewBenchmarkPayload({
      createdAt: "2026-05-18T00:00:00.000Z",
      profile: "full",
      runs: [makeRun({
        prompt: "What is the capital of Utah?",
        response: "The capital of Utah is Great Falls.",
        expectedSubstrings: ["Salt Lake"],
        expectedSubstringMatches: [],
      })],
      strictWebGpuRequested: false,
    });
    const withExpected = buildBrowserPreviewBenchmarkPayload({
      createdAt: "2026-05-18T00:00:00.000Z",
      profile: "full",
      runs: [makeRun({
        prompt: "What is the capital of Utah?",
        response: "The capital of Utah is Salt Lake City.",
        expectedSubstrings: ["Salt Lake"],
        expectedSubstringMatches: ["Salt Lake"],
      })],
      strictWebGpuRequested: false,
    });

    expect(missingExpected.passed).toBe(false);
    expect(missingExpected.summary.expectedSubstringsPassed).toBe(false);
    expect(withExpected.passed).toBe(true);
    expect(withExpected.summary.expectedSubstringsPassed).toBe(true);
  });

  it("tracks exact-output checks separately from substring quality checks", () => {
    const exactPass = buildBrowserPreviewBenchmarkPayload({
      createdAt: "2026-05-23T00:00:00.000Z",
      profile: "full",
      runs: [makeRun({
        prompt: "Return exactly: edge-runtime-ok",
        response: "[unlocked:ssa-kv-tsp] edge-runtime-ok \n",
        expectedExact: ["edge-runtime-ok"],
      })],
      strictWebGpuRequested: false,
    });
    const exactFail = buildBrowserPreviewBenchmarkPayload({
      createdAt: "2026-05-23T00:00:00.000Z",
      profile: "full",
      runs: [makeRun({
        prompt: "Return exactly: edge-runtime-ok",
        response: "[unlocked:ssa-kv-tsp] edge-runtime-ok.",
        expectedExact: ["edge-runtime-ok"],
      })],
      strictWebGpuRequested: false,
    });

    expect(exactPass.summary).toMatchObject({
      expectedExactCheckCount: 1,
      expectedExactPassCount: 1,
      expectedExactPassed: true,
    });
    expect(exactPass.runs[0]?.expectedExactMatches).toEqual([{ expected: "edge-runtime-ok", matched: true }]);
    expect(exactFail.passed).toBe(false);
    expect(exactFail.summary).toMatchObject({
      expectedExactCheckCount: 1,
      expectedExactPassCount: 0,
      expectedExactPassed: false,
    });
    expect(exactFail.runs[0]?.expectedExactMatches).toEqual([{ expected: "edge-runtime-ok", matched: false }]);
  });


  it("lets browser preview runs request production-style prompt and generation budgets", () => {
    const request = readBrowserPreviewBenchmarkRequest(new URL(
      "http://localhost:5173/__bench/browser-runtime"
      + "?profile=full"
      + "&prompt=Write%20about%20Earth."
      + "&generationTokens=24"
      + "&maxRuntimePromptTokens=4096"
      + "&maxRuntimeLayers=28"
      + "&requireKvReuse=true"
      + "&kvNamespace=test-bench-namespace"
      + "&minGeneratedTokens=16"
      + "&expected=Earth"
      + "&qwenThinkingMode=enabled"
      + "&strictWebGpu=true",
    ));

    expect(request).toMatchObject({
      strictWebGpuRequested: true,
      webGpuGates: ["mlp", "logits", "attention", "projection"],
      requireKvReuse: true,
      kvNamespace: "test-bench-namespace",
      qwenThinkingMode: "enabled",
      minGeneratedTokens: 16,
      prompts: [{ id: "prompt-1", text: "Write about Earth.", expectedSubstrings: ["Earth"] }],
      profile: {
        profile: "full",
        caps: {
          maxRuntimePromptTokens: 4096,
          maxRuntimeLayers: 28,
          logitCandidateLimit: null,
          maxGenerationTokens: 24,
        },
      },
    });
  });

  it("parses exact-output expectations without mixing them into substring checks", () => {
    const request = readBrowserPreviewBenchmarkRequest(new URL(
      "http://localhost:5173/__bench/browser-runtime"
      + "?prompt=Return%20exactly%3A%20edge-runtime-ok"
      + "&expected=runtime"
      + "&expectedExact=edge-runtime-ok",
    ));

    expect(request.prompts).toEqual([{
      id: "prompt-1",
      text: "Return exactly: edge-runtime-ok",
      expectedSubstrings: ["runtime"],
      expectedExact: ["edge-runtime-ok"],
    }]);
  });

  it("parses opt-in benchmark telemetry requests without enabling upload by default", () => {
    const request = readBrowserPreviewBenchmarkRequest(new URL(
      "http://localhost:5173/__bench/browser-runtime"
      + "?memoryGrounding=montana_capital"
      + "&submitTelemetry=true",
    ));

    expect(request.benchmarkTelemetryRequested).toBe(true);
    expect(request.benchmarkTelemetryConfig).toMatchObject({
      enabled: false,
      url: "",
    });
  });

  it("builds a non-exact seed run before the target prompt for predictive KVSwap proof", () => {
    const request = readBrowserPreviewBenchmarkRequest(new URL(
      "http://localhost:5173/__bench/browser-runtime"
      + "?prompt=alpha%20beta%20runtime%20cache"
      + "&expected=alpha%20beta"
      + "&requireKvPredictivePrefetch=true"
      + "&kvNamespace=predictive-proof-namespace",
    ));
    const prompts = buildBrowserPreviewRunPrompts(request);

    expect(request.requireKvPredictivePrefetch).toBe(true);
    expect(prompts).toHaveLength(2);
    expect(prompts[0]).toMatchObject({
      id: "prompt-1-kv-predictive-seed",
      expectedSubstrings: [],
    });
    expect(prompts[0]?.text).toContain("alpha beta runtime cache");
    expect(prompts[0]?.text).not.toBe(prompts[1]?.text);
    expect(prompts[1]).toEqual({
      id: "prompt-1",
      text: "alpha beta runtime cache",
      expectedSubstrings: ["alpha beta"],
    });
  });

  it("keeps predictive KV seed runs grounded when memory QA proof is enabled", () => {
    const request = readBrowserPreviewBenchmarkRequest(new URL(
      "http://localhost:5173/__bench/browser-runtime"
      + "?memoryGrounding=qa_corpus_v1"
      + "&memoryCorpusSize=64"
      + "&requireKvPredictivePrefetch=true",
    ));
    const prompts = buildBrowserPreviewRunPrompts(request);

    expect(prompts[0]).toMatchObject({
      id: "qa-1-kv-predictive-seed",
      expectedSubstrings: ["Helena"],
    });
    expect(prompts[0]?.text).toContain("Cedar Ridge operations dossier");
    expect(prompts[0]?.text).toContain("Use the persisted answer token only.");
    expect(prompts[0]?.text).not.toBe(request.prompts[0]?.text);
    expect(prompts[0]?.text).not.toContain("runtime cache warm seed");
  });

  it("lets browser preview runs request model-residency warmup as reported init work", () => {
    const request = readBrowserPreviewBenchmarkRequest(new URL(
      "http://localhost:5173/__bench/browser-runtime"
      + "?prompt=Warm%20the%20full%20lane."
      + "&warmModelResidency=true",
    ));

    expect(request.warmModelResidency).toBe(true);
    expect(request.warmModelResidencyMode).toBe("pipeline_preload");
  });

  it("defaults strict browser WebGPU benchmarks to pipeline preload warmup", () => {
    const request = readBrowserPreviewBenchmarkRequest(new URL(
      "http://localhost:5173/__bench/browser-runtime"
      + "?prompt=Warm%20strict%20lane."
      + "&strictWebGpu=true",
    ));

    expect(request.warmModelResidency).toBe(true);
    expect(request.warmModelResidencyMode).toBe("pipeline_preload");
  });

  it("allows strict browser WebGPU benchmarks to explicitly disable warmup", () => {
    const request = readBrowserPreviewBenchmarkRequest(new URL(
      "http://localhost:5173/__bench/browser-runtime"
      + "?prompt=Cold%20strict%20lane."
      + "&strictWebGpu=true"
      + "&warmModelResidency=false",
    ));

    expect(request.warmModelResidency).toBe(false);
  });

  it("lets browser preview benchmarks explicitly request a target-probe warmup", () => {
    const request = readBrowserPreviewBenchmarkRequest(new URL(
      "http://localhost:5173/__bench/browser-runtime"
      + "?prompt=Warm%20the%20full%20lane."
      + "&warmModelResidency=true"
      + "&warmupMode=target_probe",
    ));

    expect(request.warmModelResidency).toBe(true);
    expect(request.warmModelResidencyMode).toBe("target_probe");
  });

  it("forces full-vocab top-k logits when strict WebGPU logits proof is requested", () => {
    const request = readBrowserPreviewBenchmarkRequest(new URL(
      "http://localhost:5173/__bench/browser-runtime"
      + "?profile=balanced"
      + "&prompt=Strict%20proof"
      + "&logitCandidateLimit=1024"
      + "&strictWebGpu=true",
    ));

    expect(request.webGpuGates).toContain("logits");
    expect(request.profile.caps.logitCandidateLimit).toBeNull();
    expect(request.profile.capStatus.logits).toBe(false);
    expect(request.logitTopK).toBe(1);
    expect(request.logitTileRows).toBe(32_768);
  });

  it("keeps requested layer caps for strict expected-substring speed proof only when explicitly requested", () => {
    const request = readBrowserPreviewBenchmarkRequest(new URL(
      "http://localhost:5173/__bench/browser-runtime"
      + "?profile=balanced"
      + "&maxRuntimeLayers=8"
      + "&prompt=Answer%20with%20the%20exact%20sentence%3A%20Salt%20Lake%20City%20is%20ready."
      + "&expected=Salt%20Lake"
      + "&strictWebGpu=true"
      + "&strictExpectedLayers=requested",
    ));

    expect(request.profile.caps.maxRuntimeLayers).toBe(8);
    expect(request.profile.capStatus.layers).toBe(true);
    expect(request.strictExpectedLayerMode).toBe("requested");
  });

  it("can explicitly lift capped layers for strict expected-substring production quality proof", () => {
    const request = readBrowserPreviewBenchmarkRequest(new URL(
      "http://localhost:5173/__bench/browser-runtime"
      + "?profile=balanced"
      + "&maxRuntimeLayers=8"
      + "&prompt=Answer%20with%20the%20exact%20sentence%3A%20Salt%20Lake%20City%20is%20ready."
      + "&expected=Salt%20Lake"
      + "&strictWebGpu=true"
      + "&strictExpectedLayers=full",
    ));

    expect(request.profile.caps.maxRuntimeLayers).toBeNull();
    expect(request.profile.capStatus.layers).toBe(false);
    expect(request.strictExpectedLayerMode).toBe("full");
  });

  it("lets browser preview runs request explicit logit top-k and tile controls", () => {
    const request = readBrowserPreviewBenchmarkRequest(new URL(
      "http://localhost:5173/__bench/browser-runtime"
      + "?prompt=Strict%20speed"
      + "&logitTopK=1"
      + "&logitTileRows=8192",
    ));

    expect(request.logitTopK).toBe(1);
    expect(request.logitTileRows).toBe(8192);
  });

  it("uses the measured 32k logit tile size for explicit greedy speed proofs", () => {
    const request = readBrowserPreviewBenchmarkRequest(new URL(
      "http://localhost:5173/__bench/browser-runtime"
      + "?prompt=Strict%20speed"
      + "&logitTopK=1",
    ));

    expect(request.logitTopK).toBe(1);
    expect(request.logitTileRows).toBe(32_768);
  });

  it("can generate deterministic long-prompt input and request strict long-prompt proof", () => {
    const request = readBrowserPreviewBenchmarkRequest(new URL(
      "http://localhost:5173/__bench/browser-runtime"
      + "?longPromptTargetTokens=12"
      + "&strictLongPrompt=true",
    ));

    expect(request.prompts).toEqual([
      {
        id: "long-prompt-1",
        text: "alpha beta gamma delta runtime memory context tensor alpha beta gamma delta",
        expectedSubstrings: [],
      },
    ]);
    expect((request as typeof request & {
      longPromptTargetTokens?: number;
      strictLongPromptProof?: boolean;
    }).longPromptTargetTokens).toBe(12);
    expect((request as typeof request & { strictLongPromptProof?: boolean }).strictLongPromptProof).toBe(true);
  });

  it("clamps oversized deterministic long-prompt URL controls before materializing prompt text", () => {
    const request = readBrowserPreviewBenchmarkRequest(new URL(
      "http://localhost:5173/__bench/browser-runtime"
      + "?longPromptTargetTokens=999999999",
    ));

    expect((request as typeof request & { longPromptTargetTokens?: number }).longPromptTargetTokens)
      .toBe(MAX_LONG_PROMPT_TARGET_TOKENS);
    expect(request.prompts[0]?.text.split(/\s+/)).toHaveLength(MAX_LONG_PROMPT_TARGET_TOKENS);
  });

  it("bounds repeat-only long prompts by seed words and total generated words", () => {
    const hugeSeed = Array.from({ length: MAX_LONG_PROMPT_TARGET_TOKENS + 1000 }, (_value, index) => `seed${index}`).join(" ");
    const prompt = buildDeterministicLongPrompt({
      repeat: MAX_LONG_PROMPT_REPEAT,
      seedText: hugeSeed,
    });
    const words = prompt.split(/\s+/);

    expect(words).toHaveLength(MAX_LONG_PROMPT_TARGET_TOKENS);
    expect(new Set(words.slice(0, MAX_LONG_PROMPT_SEED_WORDS)).size).toBe(MAX_LONG_PROMPT_SEED_WORDS);
    expect(words[MAX_LONG_PROMPT_SEED_WORDS]).toBe("seed0");
  });

  it("uses compact promptSeed for deterministic browser long-prompt generation", () => {
    const request = readBrowserPreviewBenchmarkRequest(new URL(
      "http://localhost:5173/__bench/browser-runtime"
      + "?promptSeed=red%20blue"
      + "&longPromptRepeat=3",
    ));

    expect(request.prompts).toEqual([
      {
        id: "long-prompt-1",
        text: "red blue red blue red blue",
        expectedSubstrings: [],
      },
    ]);
  });

  it("includes chunk-plan metadata and a fail-closed reason in browser preview failure payloads", () => {
    const request = readBrowserPreviewBenchmarkRequest(new URL(
      "http://localhost:5173/__bench/browser-runtime"
      + "?longPromptTargetTokens=5000"
      + "&strictLongPrompt=true",
    ));
    const error = Object.assign(
      new Error("Strict long-prompt proof requires chunked prefill dispatch, but chunked dispatch is not implemented yet."),
      {
        prefillChunkCount: 5,
        prefillChunkSize: 1024,
        shapeBucket: "prompt<=1024:selected<=512:headDim<=128:tileRows<=8192:precision=f32",
        pipelineCacheKey: "prefill_chunk:prompt<=1024:selected<=512:headDim<=128:tileRows<=8192:precision=f32",
        maxDispatchEstimatedMs: 7.5,
        prefillChunkReason: "chunked_prefill_dispatch_not_implemented",
      },
    );

    const payload = buildBrowserPreviewBenchmarkFailurePayload(request, error);

    expect(payload.passed).toBe(false);
    expect(payload.summary).toMatchObject({
      error: "Strict long-prompt proof requires chunked prefill dispatch, but chunked dispatch is not implemented yet.",
      prefillChunkCount: 5,
      prefillChunkSize: 1024,
      shapeBucket: "prompt<=1024:selected<=512:headDim<=128:tileRows<=8192:precision=f32",
      pipelineCacheKey: "prefill_chunk:prompt<=1024:selected<=512:headDim<=128:tileRows<=8192:precision=f32",
      maxDispatchEstimatedMs: 7.5,
      prefillChunkReason: "chunked_prefill_dispatch_not_implemented",
    });
  });

  it("reads repeated prompt and expected params without splitting literal pipes or commas", () => {
    const request = readBrowserPreviewBenchmarkRequest(new URL(
      "http://localhost:5173/__bench/browser-runtime"
      + "?prompt=Compare%20A%20%7C%20B%20literally."
      + "&prompt=Name%20the%20Utah%20capital."
      + "&expected=A%20%7C%20B"
      + "&expected=Salt%20Lake%2C%20Utah",
    ));

    expect(request.prompts).toEqual([
      { id: "prompt-1", text: "Compare A | B literally.", expectedSubstrings: ["A | B"] },
      { id: "prompt-2", text: "Name the Utah capital.", expectedSubstrings: ["Salt Lake, Utah"] },
    ]);
  });

  it("reads expectedJson params as delimiter-safe expected substring arrays per prompt", () => {
    const request = readBrowserPreviewBenchmarkRequest(new URL(
      "http://localhost:5173/__bench/browser-runtime"
      + "?prompt=Check%20literal%20values."
      + "&prompt=Name%20the%20Utah%20capital."
      + `&expectedJson=${encodeURIComponent(JSON.stringify(["A | B", "foo, bar"]))}`
      + `&expectedJson=${encodeURIComponent(JSON.stringify(["Salt Lake, Utah"]))}`,
    ));

    expect(request.prompts).toEqual([
      { id: "prompt-1", text: "Check literal values.", expectedSubstrings: ["A | B", "foo, bar"] },
      { id: "prompt-2", text: "Name the Utah capital.", expectedSubstrings: ["Salt Lake, Utah"] },
    ]);
  });

  it("lets browser preview runs set an explicit timeout budget", () => {
    const request = readBrowserPreviewBenchmarkRequest(new URL(
      "http://localhost:5173/__bench/browser-runtime?prompt=What%20is%20Earth%3F&timeoutMs=45000",
    ));

    expect(request.timeoutMs).toBe(45_000);
  });

  it("lets browser preview runs disable MTP to isolate target-only prefill and decode cost", () => {
    const request = readBrowserPreviewBenchmarkRequest(new URL(
      "http://localhost:5173/__bench/browser-runtime?prompt=alpha%20beta&mtp=false",
    ));

    expect(request.mtpEnabled).toBe(false);
  });

  it("inherits production WebGPU-only requirements without needing a separate strict preview flag", () => {
    const request = readBrowserPreviewBenchmarkRequest(new URL(
      "http://localhost:5173/__bench/browser-runtime?profile=full&prompt=alpha%20beta",
    ));

    expect(request.strictWebGpuRequested).toBe(true);
    expect(request.webGpuGates).toEqual(["mlp", "logits", "attention", "projection"]);
  });

  it("rejects degenerate browser-visible output even when tokens streamed", () => {
    expect(isCoherentBrowserPreviewResponse("[unlocked:ssa-kv-tsp]", 8)).toBe(false);
    expect(isCoherentBrowserPreviewResponse("[unlocked:ssa-kv-tsp] . , ;", 8)).toBe(false);
    expect(isCoherentBrowserPreviewResponse("[unlocked:ssa-kv-tsp] **", 8)).toBe(false);
    expect(isCoherentBrowserPreviewResponse("<|im_end|>", 8)).toBe(false);
    expect(isCoherentBrowserPreviewResponse("<think>Salt Lake City is coherent hidden content.</think>.", 8)).toBe(false);
    expect(isCoherentBrowserPreviewResponse("The", 8)).toBe(false);
    expect(isCoherentBrowserPreviewResponse("no no no no", 8)).toBe(false);
    expect(isCoherentBrowserPreviewResponse("The capital of Montana is Helena. m m m", 9)).toBe(false);
    expect(isCoherentBrowserPreviewResponse("Salt Lake City is the capital of Utah.", 8)).toBe(true);
    expect(isCoherentBrowserPreviewResponse("[unlocked:ssa-kv-tsp]Helena.", 2, ["Helena"])).toBe(true);
    expect(isCoherentBrowserPreviewResponse("[unlocked:ssa-kv-tsp]Helena m m m", 5, ["Helena"])).toBe(false);
  });

  it("forces full-layer quality mode when strict expected-answer proof is requested", () => {
    const request = readBrowserPreviewBenchmarkRequest(new URL(
      "http://localhost:5173/__bench/browser-runtime"
      + "?profile=balanced"
      + "&strictWebGpu=true"
      + "&prompt=What%20is%20the%20capital%20of%20Montana%3F"
      + "&expected=Helena",
    ));

    expect(request.strictExpectedLayerMode).toBe("full");
    expect(request.profile.caps.maxRuntimeLayers).toBeNull();
    expect(request.profile.capStatus.layers).toBe(false);
    expect(request.profile.caps.logitCandidateLimit).toBeNull();
  });

  it("enables a seeded browser-vector context rebuild benchmark for grounded answer proof", () => {
    const request = readBrowserPreviewBenchmarkRequest(new URL(
      "http://localhost:5173/__bench/browser-runtime"
      + "?profile=balanced"
      + "&strictWebGpu=true"
      + "&memoryGrounding=montana_capital",
    ));

    expect(request.memoryGroundingCase).toBe("montana_capital");
    expect(request.prompts[0]).toMatchObject({
      text: "Using retrieved memory only, in the Cedar Ridge operations dossier, which city is listed as the Montana field office hub? Answer with only the city.",
      expectedSubstrings: ["Helena"],
    });
    expect(request.strictExpectedLayerMode).toBe("full");
    expect(request.profile.caps.maxRuntimeLayers).toBeNull();
  });

  it("expands the large QA corpus grounding mode into multiple answer-quality prompts", () => {
    const request = readBrowserPreviewBenchmarkRequest(new URL(
      "http://localhost:5173/__bench/browser-runtime"
      + "?strictWebGpu=true"
      + "&memoryGrounding=qa_corpus_v1"
      + "&memoryCorpusSize=64",
    ));

    expect(request.memoryGroundingCase).toBe("qa_corpus_v1");
    expect(request.memoryGroundingCorpusSize).toBe(64);
    expect(request.prompts.map((prompt) => prompt.expectedSubstrings[0])).toEqual([
      "Helena",
      "edge-runtime-ok",
      "amber",
      "Vela-42",
      "Nora Vale",
      "37 days",
    ]);
    expect(request.prompts.every((prompt) => prompt.text.includes("MEMORY_FACT_"))).toBe(false);
    expect(request.prompts[2]?.text).toContain("Desert Lantern onboarding memo");
    expect(request.profile.caps.maxRuntimeLayers).toBeNull();
  });

  it("can bound QA corpus model generations without shrinking the seeded corpus size request", () => {
    const request = readBrowserPreviewBenchmarkRequest(new URL(
      "http://localhost:5173/__bench/browser-runtime"
      + "?strictWebGpu=true"
      + "&memoryGrounding=qa_corpus_v1"
      + "&memoryCorpusSize=64"
      + "&memoryPromptLimit=2",
    ));

    expect(request.memoryGroundingCase).toBe("qa_corpus_v1");
    expect(request.memoryGroundingCorpusSize).toBe(64);
    expect(request.prompts.map((prompt) => prompt.expectedSubstrings[0])).toEqual([
      "Helena",
      "edge-runtime-ok",
    ]);
    expect(request.prompts).toHaveLength(2);
    expect(request.profile.caps.maxRuntimeLayers).toBeNull();
  });

  it("expands the large synthetic grounding mode into corpus-spanning answer prompts", () => {
    const request = readBrowserPreviewBenchmarkRequest(new URL(
      "http://localhost:5173/__bench/browser-runtime"
      + "?strictWebGpu=true"
      + "&memoryGrounding=large_synthetic_v1"
      + "&memoryCorpusSize=1024",
    ));

    expect(request.memoryGroundingCase).toBe("large_synthetic_v1");
    expect(request.memoryGroundingCorpusSize).toBe(1024);
    expect(request.prompts).toHaveLength(5);
    expect(request.prompts.map((prompt) => prompt.expectedSubstrings[0])).toEqual([
      "Aster-0001",
      "Aster-0016",
      "Aster-0032",
      "Aster-0048",
      "Aster-0064",
    ]);
    expect(request.prompts[4]?.text).toContain("Helix Ledger synthetic corpus");
    expect(request.prompts[4]?.text).toContain("synth0064");
  });

  it("supports a database-only grounding audit without forcing model initialization", () => {
    const request = readBrowserPreviewBenchmarkRequest(new URL(
      "http://localhost:5173/__bench/browser-runtime?memoryGroundingAuditOnly=true&memoryCorpusSize=1024",
    ));

    expect(request.memoryGroundingAuditOnly).toBe(true);
    expect(request.memoryGroundingCase).toBe("large_synthetic_v1");
    expect(request.prompts).toHaveLength(5);
    expect(request.prompts[0]?.expectedSubstrings).toEqual(["Aster-0001"]);
  });

  it("parses the compiled browser production backend separately from the Kernel Lab route", () => {
    const request = readBrowserPreviewBenchmarkRequest(new URL(
      "http://localhost:5173/__bench/browser-runtime"
      + "?backend=compiled-browser-webllm"
      + "&modelId=Qwen3-0.6B-q4f16_1-MLC"
      + "&memoryGrounding=montana_capital"
      + "&expectedExact=Helena",
    ));

    expect(request.backendId).toBe("compiled-browser-webllm");
    expect(request.modelId).toBe("Qwen3-0.6B-q4f16_1-MLC");
    expect(request.strictWebGpuRequested).toBe(false);
    expect(request.requireKvReuse).toBe(false);
    expect(request.prompts[0]).toMatchObject({
      expectedSubstrings: ["Helena"],
      expectedExact: ["Helena"],
    });
  });

  it("makes memory retrieval/context rebuild correctness part of browser benchmark pass/fail", () => {
    const groundedRun = makeRun({
      expectedSubstrings: ["Helena"],
      response: "[unlocked:ssa-kv-tsp]Helena.",
      expectedAnswerOnlyPassed: true,
      webGpu: {
        available: true,
        requestedBackendPreference: "webgpu",
        logitProjectionBackend: "webgpu",
        cpuFallbackUsed: false,
        noCpuFallback: true,
        requestedGates: [],
        passedGates: [],
        failedGates: [],
        positiveKernelProof: true,
        decodeSubmitCount: 120,
        decodeDispatchCount: 336,
        totalDecodeLayerCount: 112,
        residentFinalHiddenUsedForLogits: true,
        residentDecodeLayerCoverage: 1,
        fullLogitsReadbackCount: 0,
        activationUploadBytesDuringDecode: 0,
        hiddenReadbackCountDuringDecode: 0,
      },
      memoryGrounding: {
        mode: "seeded_browser_vector_context_rebuild",
        caseId: "montana_capital",
        corpusCount: 16,
        retrievedMemoryIds: ["bench_memory_montana_capital"],
        includedMemoryIds: ["bench_memory_montana_capital"],
        expectedMemoryIds: ["bench_memory_montana_capital"],
        expectedMemoryHitPassed: true,
        contextRebuildPassed: true,
        answerOnlyExpected: true,
        answerOnlyPassed: true,
        contextEstimatedTokens: 90,
        retrievalMs: 2,
        contextRebuildMs: 1,
        retrievalAudit: makePassedQaRetrievalAudit(),
      },
    });
    const payload = buildBrowserPreviewBenchmarkPayload({
      createdAt: "2026-05-23T00:00:00.000Z",
      profile: "full",
      strictWebGpuRequested: true,
      runs: [groundedRun, groundedRun],
    });

    expect(payload.passed).toBe(true);
    expect(payload.summary).toMatchObject({
      memoryQueryMode: "seeded_browser_vector_context_rebuild",
      memoryGroundingRequired: true,
      memoryGroundingPassed: true,
      memoryGroundingCoveragePassed: true,
      memoryAnswerOnlyPassed: true,
      stopQualityPassed: true,
      stopQualityFailureCount: 0,
      memorySeededCorpusCount: 16,
      memoryRetrievedCount: 2,
      memoryIncludedCount: 2,
      technicalProofOnly: false,
      productionQualityPassed: true,
      productionDeployReadyPassed: false,
    });
  });

  it("summarizes memory retrieval audit quality by query class", () => {
    const groundedRun = makeRun({
      expectedSubstrings: ["Helena"],
      response: "[unlocked:ssa-kv-tsp]Helena.",
      expectedAnswerOnlyPassed: true,
      memoryGrounding: {
        mode: "seeded_browser_vector_context_rebuild",
        caseId: "qa_corpus_v1",
        corpusCount: 64,
        retrievedMemoryIds: ["bench_memory_montana_capital"],
        includedMemoryIds: ["bench_memory_montana_capital"],
        expectedMemoryIds: ["bench_memory_montana_capital"],
        expectedMemoryHitPassed: true,
        contextRebuildPassed: true,
        answerOnlyExpected: true,
        answerOnlyPassed: true,
        contextEstimatedTokens: 120,
        retrievalMs: 2,
        contextRebuildMs: 1,
        retrievalAudit: {
          corpusCount: 64,
          queryCount: 23,
          top1CorrectCount: 23,
          reciprocalRankSum: 23,
          recallAt1: 1,
          mrr: 1,
          canonicalQueryCount: 6,
          canonicalTop1CorrectCount: 6,
          canonicalRecallAt1: 1,
          canonicalMrr: 1,
          aliasQueryCount: 5,
          aliasTop1CorrectCount: 5,
          aliasRecallAt1: 1,
          aliasMrr: 1,
          generatedParaphraseQueryCount: 12,
          generatedParaphraseTop1CorrectCount: 12,
          generatedParaphraseRecallAt1: 1,
          generatedParaphraseMrr: 1,
          queryClassBreakdown: [
            { queryClass: "canonical", queryCount: 6, top1CorrectCount: 6, recallAt1: 1, mrr: 1 },
            { queryClass: "alias", queryCount: 5, top1CorrectCount: 5, recallAt1: 1, mrr: 1 },
            { queryClass: "generated_paraphrase", queryCount: 12, top1CorrectCount: 12, recallAt1: 1, mrr: 1 },
          ],
          minTopScoreMargin: 0.12,
          meanExpectedHitRank: 1,
          passed: true,
          elapsedMs: 3,
          minRequiredRecallAt1: 1,
        },
      },
    });

    const payload = buildBrowserPreviewBenchmarkPayload({
      createdAt: "2026-05-23T00:00:00.000Z",
      profile: "full",
      strictWebGpuRequested: true,
      runs: [groundedRun],
    });

    expect(payload.summary).toMatchObject({
      memoryRetrievalAuditQueryCount: 23,
      memoryCanonicalQueryCount: 6,
      memoryCanonicalRecallAt1: 1,
      memoryAliasQueryCount: 5,
      memoryAliasRecallAt1: 1,
      memoryGeneratedParaphraseQueryCount: 12,
      memoryGeneratedParaphraseRecallAt1: 1,
      memoryGeneratedParaphraseMrr: 1,
    });
  });

  it("requires grounded memory, exact output, KV reuse, and v11 command batching for deploy readiness", () => {
    const groundedRun = makeRun({
      expectedSubstrings: ["Helena"],
      expectedExact: ["Helena"],
      expectedExactMatches: [{ expected: "Helena", matched: true }],
      response: "[unlocked:ssa-kv-tsp]Helena",
      expectedAnswerOnlyPassed: true,
      metrics: {
        initLoadMs: 20,
        prefillMs: 8,
        timeToFirstTokenMs: 11,
        decodeLatencyMs: 6,
        tokensPerSecond: 166.67,
        generatedTokens: 4,
      },
      webGpu: {
        available: true,
        requestedBackendPreference: "webgpu",
        logitProjectionBackend: "webgpu",
        cpuFallbackUsed: false,
        noCpuFallback: true,
        requestedGates: [],
        passedGates: [],
        failedGates: [],
        positiveKernelProof: true,
        decodeSubmitCount: 120,
        decodeDispatchCount: 336,
        totalDecodeLayerCount: 112,
        residentFinalHiddenUsedForLogits: true,
        residentDecodeLayerCoverage: 1,
        fullLogitsReadbackCount: 0,
        activationUploadBytesDuringDecode: 0,
        hiddenReadbackCountDuringDecode: 0,
      },
      kvPersistence: {
        reuseEvents: 1,
      },
      memoryGrounding: {
        mode: "seeded_browser_vector_context_rebuild",
        caseId: "montana_capital",
        corpusCount: 16,
        retrievedMemoryIds: ["bench_memory_montana_capital"],
        includedMemoryIds: ["bench_memory_montana_capital"],
        expectedMemoryIds: ["bench_memory_montana_capital"],
        expectedMemoryHitPassed: true,
        contextRebuildPassed: true,
        answerOnlyExpected: true,
        answerOnlyPassed: true,
        contextEstimatedTokens: 90,
        retrievalMs: 2,
        contextRebuildMs: 1,
        retrievalAudit: makePassedQaRetrievalAudit(),
      },
    });

    const payload = buildBrowserPreviewBenchmarkPayload({
      createdAt: "2026-05-23T00:00:00.000Z",
      profile: "full",
      strictWebGpuRequested: true,
      requireKvReuse: true,
      runs: [groundedRun],
    });

    expect(payload.summary).toMatchObject({
      expectedExactPassed: true,
      memoryGroundingRequired: true,
      memoryGroundingPassed: true,
      directModelFactualProofUsed: false,
      v11CommandBatchingPassed: true,
      productionLayerCoverageRequired: true,
      productionLayerCoveragePassed: true,
      productionLayerVisitsPerToken: 28,
      diagnosticCappedLayerRun: false,
      groundedProductionReadyPassed: true,
      customKernelLabReadyPassed: true,
      researchBackendId: "unlocked-browser-transformer",
      deployBackendId: null,
      productionDeployReadyPassed: false,
    });
  });

  it("keeps capped layer diagnostic runs out of production readiness even when other gates pass", () => {
    const groundedRun = makeRun({
      expectedSubstrings: ["Helena"],
      expectedExact: ["Helena"],
      expectedExactMatches: [{ expected: "Helena", matched: true }],
      response: "[unlocked:ssa-kv-tsp]Helena",
      expectedAnswerOnlyPassed: true,
      metrics: {
        initLoadMs: 20,
        prefillMs: 8,
        timeToFirstTokenMs: 11,
        decodeLatencyMs: 6,
        tokensPerSecond: 166.67,
        generatedTokens: 4,
      },
      webGpu: {
        available: true,
        requestedBackendPreference: "webgpu",
        logitProjectionBackend: "webgpu",
        cpuFallbackUsed: false,
        noCpuFallback: true,
        requestedGates: [],
        passedGates: [],
        failedGates: [],
        positiveKernelProof: true,
        decodeSubmitCount: 40,
        decodeDispatchCount: 96,
        totalDecodeLayerCount: 32,
        residentFinalHiddenUsedForLogits: true,
        residentDecodeLayerCoverage: 1,
        fullLogitsReadbackCount: 0,
        activationUploadBytesDuringDecode: 0,
        hiddenReadbackCountDuringDecode: 0,
      },
      kvPersistence: {
        reuseEvents: 1,
      },
      memoryGrounding: {
        mode: "seeded_browser_vector_context_rebuild",
        caseId: "montana_capital",
        corpusCount: 16,
        retrievedMemoryIds: ["bench_memory_montana_capital"],
        includedMemoryIds: ["bench_memory_montana_capital"],
        expectedMemoryIds: ["bench_memory_montana_capital"],
        expectedMemoryHitPassed: true,
        contextRebuildPassed: true,
        answerOnlyExpected: true,
        answerOnlyPassed: true,
        contextEstimatedTokens: 90,
        retrievalMs: 2,
        contextRebuildMs: 1,
        retrievalAudit: makePassedQaRetrievalAudit(),
      },
    });

    const payload = buildBrowserPreviewBenchmarkPayload({
      createdAt: "2026-05-23T00:00:00.000Z",
      profile: "balanced",
      strictWebGpuRequested: true,
      requireKvReuse: true,
      runs: [groundedRun],
    });

    expect(payload.summary).toMatchObject({
      expectedExactPassed: true,
      memoryGroundingPassed: true,
      v11CommandBatchingPassed: true,
      productionSpeedFloorPassed: true,
      productionLayerCoverageRequired: true,
      productionLayerCoveragePassed: false,
      productionLayerVisitsPerToken: 8,
      diagnosticCappedLayerRun: true,
      productionQualityPassed: false,
      groundedProductionReadyPassed: false,
      productionDeployReadyPassed: false,
    });
  });

  it("rejects deploy readiness when generated-paraphrase retrieval proof is missing", () => {
    const groundedRun = makeRun({
      expectedSubstrings: ["Helena"],
      expectedExact: ["Helena"],
      expectedExactMatches: [{ expected: "Helena", matched: true }],
      response: "[unlocked:ssa-kv-tsp]Helena",
      expectedAnswerOnlyPassed: true,
      metrics: {
        initLoadMs: 20,
        prefillMs: 8,
        timeToFirstTokenMs: 11,
        decodeLatencyMs: 6,
        tokensPerSecond: 166.67,
        generatedTokens: 4,
      },
      webGpu: {
        available: true,
        requestedBackendPreference: "webgpu",
        logitProjectionBackend: "webgpu",
        cpuFallbackUsed: false,
        noCpuFallback: true,
        requestedGates: [],
        passedGates: [],
        failedGates: [],
        positiveKernelProof: true,
        decodeSubmitCount: 6,
        decodeDispatchCount: 12,
        totalDecodeLayerCount: 4,
        residentFinalHiddenUsedForLogits: true,
        residentDecodeLayerCoverage: 1,
        fullLogitsReadbackCount: 0,
        activationUploadBytesDuringDecode: 0,
        hiddenReadbackCountDuringDecode: 0,
      },
      kvPersistence: {
        reuseEvents: 1,
      },
      memoryGrounding: {
        mode: "seeded_browser_vector_context_rebuild",
        caseId: "qa_corpus_v1",
        corpusCount: 64,
        retrievedMemoryIds: ["bench_memory_montana_capital"],
        includedMemoryIds: ["bench_memory_montana_capital"],
        expectedMemoryIds: ["bench_memory_montana_capital"],
        expectedMemoryHitPassed: true,
        contextRebuildPassed: true,
        answerOnlyExpected: true,
        answerOnlyPassed: true,
        contextEstimatedTokens: 120,
        retrievalMs: 2,
        contextRebuildMs: 1,
        retrievalAudit: {
          corpusCount: 64,
          queryCount: 11,
          top1CorrectCount: 11,
          reciprocalRankSum: 11,
          recallAt1: 1,
          mrr: 1,
          canonicalQueryCount: 6,
          canonicalTop1CorrectCount: 6,
          canonicalRecallAt1: 1,
          canonicalMrr: 1,
          aliasQueryCount: 5,
          aliasTop1CorrectCount: 5,
          aliasRecallAt1: 1,
          aliasMrr: 1,
          minTopScoreMargin: 0.12,
          meanExpectedHitRank: 1,
          passed: true,
          elapsedMs: 3,
          minRequiredRecallAt1: 1,
        },
      },
    });

    const payload = buildBrowserPreviewBenchmarkPayload({
      createdAt: "2026-05-23T00:00:00.000Z",
      profile: "full",
      strictWebGpuRequested: true,
      requireKvReuse: true,
      runs: [groundedRun],
    });

    expect(payload.summary).toMatchObject({
      memoryGroundingPassed: true,
      memoryRetrievalAuditPassed: true,
      memoryGeneratedParaphraseRequired: true,
      memoryGeneratedParaphrasePassed: false,
      groundedProductionReadyPassed: false,
      productionDeployReadyPassed: false,
    });
  });

  it("uses warm-resident runs, not cold init/warmup runs, for the production speed floor", () => {
    const coldRun = makeRun({
      expectedSubstrings: ["edge-runtime-ok"],
      expectedExact: ["edge-runtime-ok"],
      expectedExactMatches: [{ expected: "edge-runtime-ok", matched: true }],
      response: "[unlocked:ssa-kv-tsp]edge-runtime-ok",
      expectedAnswerOnlyPassed: true,
      metrics: {
        initLoadMs: 20_000,
        warmupMs: 16_000,
        warmupBlockingMs: 16_000,
        prefillMs: 7_000,
        timeToFirstTokenMs: 7_000,
        decodeLatencyMs: 7_900,
        tokensPerSecond: 0.38,
        generatedTokens: 3,
      },
      webGpu: {
        available: true,
        requestedBackendPreference: "webgpu",
        logitProjectionBackend: "webgpu",
        cpuFallbackUsed: false,
        noCpuFallback: true,
        requestedGates: [],
        passedGates: [],
        failedGates: [],
        positiveKernelProof: true,
        decodeSubmitCount: 87,
        decodeDispatchCount: 522,
        totalDecodeLayerCount: 84,
        residentFinalHiddenUsedForLogits: true,
        residentDecodeLayerCoverage: 1,
        fullLogitsReadbackCount: 0,
        activationUploadBytesDuringDecode: 0,
        hiddenReadbackCountDuringDecode: 0,
        fusedPackedQkvLayerCount: 84,
      },
      kvPersistence: { reuseEvents: 0 },
      memoryGrounding: {
        mode: "seeded_browser_vector_context_rebuild",
        caseId: "qa_corpus_v1",
        corpusCount: 64,
        retrievedMemoryIds: ["bench_memory_edge_runtime_sentinel"],
        includedMemoryIds: ["bench_memory_edge_runtime_sentinel"],
        expectedMemoryIds: ["bench_memory_edge_runtime_sentinel"],
        expectedMemoryHitPassed: true,
        contextRebuildPassed: true,
        answerOnlyExpected: true,
        answerOnlyPassed: true,
        contextEstimatedTokens: 143,
        retrievalMs: 5,
        contextRebuildMs: 1,
        retrievalAudit: makePassedQaRetrievalAudit(),
      },
    });
    const warmRun = makeRun({
      ...coldRun,
      metrics: {
        initLoadMs: 0,
        warmupMs: 0,
        warmupBlockingMs: 0,
        prefillMs: 1_200,
        timeToFirstTokenMs: 1_200,
        decodeLatencyMs: 1_300,
        tokensPerSecond: 2.31,
        generatedTokens: 3,
      },
      kvPersistence: { reuseEvents: 1 },
    });

    const payload = buildBrowserPreviewBenchmarkPayload({
      createdAt: "2026-05-23T00:00:00.000Z",
      profile: "full",
      strictWebGpuRequested: true,
      requireKvReuse: true,
      runs: [coldRun, warmRun],
    });

    expect(payload.summary).toMatchObject({
      meanTokensPerSecond: 1.345,
      warmResidentRunCount: 1,
      meanWarmResidentTokensPerSecond: 2.31,
      productionSpeedMeasurement: "warm_resident",
      productionSpeedFloorPassed: true,
      groundedProductionReadyPassed: true,
      customKernelLabReadyPassed: true,
      productionDeployReadyPassed: false,
    });
  });

  it("passes deploy readiness for compiled browser backends with grounded exact quality and backend trace", () => {
    const compiledRun = makeRun({
      response: "Helena",
      expectedSubstrings: ["Helena"],
      expectedSubstringMatches: ["Helena"],
      expectedExact: ["Helena"],
      expectedExactMatches: [{ expected: "Helena", matched: true }],
      expectedAnswerOnlyPassed: true,
      runtimeTrace: {
        backend: "compiled-browser-webllm",
        tensorControl: false,
        tspSteps: [],
        kvPagingEvents: 0,
        selectedBlockIds: [],
      },
      predictive: {
        promptTokenCount: 0,
        generatedTokenCount: 1,
        selectedBlockCount: 0,
        kvPagingEventCount: 0,
        tspStepCount: 0,
      },
      webGpu: {
        available: true,
        requestedBackendPreference: "compiled-browser",
        logitProjectionBackend: "backend_native",
        cpuFallbackUsed: false,
        noCpuFallback: true,
        requestedGates: [],
        passedGates: [],
        failedGates: [],
        positiveKernelProof: true,
      },
      mtp: {
        mode: "none",
      },
      kvPersistence: {
        enabled: false,
        mode: "backend_native",
        eventCount: 0,
        persistEvents: 0,
        hydrateEvents: 0,
        reuseEvents: 0,
      },
      metrics: {
        initLoadMs: 100,
        prefillMs: 100,
        timeToFirstTokenMs: 100,
        decodeLatencyMs: 300,
        tokensPerSecond: 3.33,
        generatedTokens: 1,
      },
      memoryGrounding: {
        mode: "seeded_browser_vector_context_rebuild",
        caseId: "montana_capital",
        corpusCount: 16,
        retrievedMemoryIds: ["bench_memory_montana_capital"],
        includedMemoryIds: ["bench_memory_montana_capital"],
        expectedMemoryIds: ["bench_memory_montana_capital"],
        expectedMemoryHitPassed: true,
        contextRebuildPassed: true,
        answerOnlyExpected: true,
        answerOnlyPassed: true,
        contextEstimatedTokens: 90,
        retrievalMs: 2,
        contextRebuildMs: 1,
      },
    });

    const payload = buildBrowserPreviewBenchmarkPayload({
      createdAt: "2026-05-24T00:00:00.000Z",
      profile: "full",
      strictWebGpuRequested: false,
      minGeneratedTokens: 1,
      runs: [compiledRun],
    });

    expect(payload.passed).toBe(true);
    expect(payload.summary).toMatchObject({
      runtimeBackendId: "compiled-browser-webllm",
      runtimeBackendRole: "production_candidate",
      compiledBackendReadyPassed: true,
      groundedProductionReadyPassed: true,
      customKernelLabReadyPassed: false,
      deployBackendId: "compiled-browser-webllm",
      researchBackendId: null,
      productionDeployReadyPassed: true,
      strictWebGpuRequested: false,
      kvReusePassed: true,
      v11CommandBatchingPassed: true,
    });
  });

  it("makes large-corpus retrieval audit quality part of browser benchmark pass/fail", () => {
    const groundedRun = makeRun({
      expectedSubstrings: ["Aster-0048"],
      response: "[unlocked:ssa-kv-tsp]Aster-0048",
      expectedAnswerOnlyPassed: true,
      memoryGrounding: {
        mode: "seeded_browser_vector_context_rebuild",
        caseId: "large_synthetic_v1",
        corpusCount: 1024,
        retrievedMemoryIds: ["bench_memory_synthetic_0048"],
        includedMemoryIds: ["bench_memory_synthetic_0048"],
        expectedMemoryIds: ["bench_memory_synthetic_0048"],
        expectedMemoryHitPassed: true,
        contextRebuildPassed: true,
        answerOnlyExpected: true,
        answerOnlyPassed: true,
        contextEstimatedTokens: 120,
        retrievalMs: 3,
        contextRebuildMs: 2,
        retrievalRank: 1,
        retrievalScore: 0.99,
        retrievalTopScoreMargin: 0.4,
        retrievalAudit: {
          corpusCount: 1024,
          queryCount: 64,
          top1CorrectCount: 64,
          reciprocalRankSum: 64,
          recallAt1: 1,
          mrr: 1,
          minTopScoreMargin: 0.35,
          meanExpectedHitRank: 1,
          passed: true,
          elapsedMs: 12,
          minRequiredRecallAt1: 1,
        },
      },
    });
    const payload = buildBrowserPreviewBenchmarkPayload({
      createdAt: "2026-05-23T00:00:00.000Z",
      profile: "full",
      strictWebGpuRequested: true,
      runs: [groundedRun],
    });

    expect(payload.passed).toBe(true);
    expect(payload.summary).toMatchObject({
      memoryGroundingPassed: true,
      memoryRetrievalAuditRequired: true,
      memoryRetrievalAuditPassed: true,
      memoryRetrievalAuditQueryCount: 64,
      memoryRetrievalAuditTop1CorrectCount: 64,
      memoryRecallAt1: 1,
      memoryMrr: 1,
      memoryMinTopScoreMargin: 0.35,
      memoryMeanExpectedHitRank: 1,
      memoryExpectedHitMeanRank: 1,
      memoryExpectedHitMinTopScoreMargin: 0.4,
    });
  });

  it("fails browser benchmark payload when the large-corpus retrieval audit misses rank-1 hits", () => {
    const payload = buildBrowserPreviewBenchmarkPayload({
      createdAt: "2026-05-23T00:00:00.000Z",
      profile: "full",
      strictWebGpuRequested: true,
      runs: [makeRun({
        expectedSubstrings: ["Aster-0048"],
        response: "[unlocked:ssa-kv-tsp]Aster-0048",
        expectedAnswerOnlyPassed: true,
        memoryGrounding: {
          mode: "seeded_browser_vector_context_rebuild",
          caseId: "large_synthetic_v1",
          corpusCount: 1024,
          retrievedMemoryIds: ["bench_memory_synthetic_0048"],
          includedMemoryIds: ["bench_memory_synthetic_0048"],
          expectedMemoryIds: ["bench_memory_synthetic_0048"],
          expectedMemoryHitPassed: true,
          contextRebuildPassed: true,
          answerOnlyExpected: true,
          answerOnlyPassed: true,
          contextEstimatedTokens: 120,
          retrievalMs: 3,
          contextRebuildMs: 2,
          retrievalRank: 1,
          retrievalAudit: {
            corpusCount: 1024,
            queryCount: 64,
            top1CorrectCount: 63,
            reciprocalRankSum: 63,
            recallAt1: 0.984,
            mrr: 0.984,
            minTopScoreMargin: -0.02,
            meanExpectedHitRank: 1.125,
            passed: false,
            elapsedMs: 12,
            minRequiredRecallAt1: 1,
          },
        },
      })],
    });

    expect(payload.passed).toBe(false);
    expect(payload.summary).toMatchObject({
      memoryGroundingPassed: false,
      memoryRetrievalAuditRequired: true,
      memoryRetrievalAuditPassed: false,
      memoryRecallAt1: 0.984,
    });
  });

  it("fails browser benchmark payload when seeded memory is not retrieved and packed", () => {
    const payload = buildBrowserPreviewBenchmarkPayload({
      createdAt: "2026-05-23T00:00:00.000Z",
      profile: "full",
      strictWebGpuRequested: true,
      runs: [makeRun({
        expectedSubstrings: ["Helena"],
        response: "[unlocked:ssa-kv-tsp]Helena.",
        expectedAnswerOnlyPassed: true,
        memoryGrounding: {
          mode: "seeded_browser_vector_context_rebuild",
          caseId: "montana_capital",
          corpusCount: 4,
          retrievedMemoryIds: ["bench_memory_utah_capital"],
          includedMemoryIds: ["bench_memory_utah_capital"],
          expectedMemoryIds: ["bench_memory_montana_capital"],
          expectedMemoryHitPassed: false,
          contextRebuildPassed: false,
          answerOnlyExpected: true,
          answerOnlyPassed: true,
          contextEstimatedTokens: 90,
          retrievalMs: 2,
          contextRebuildMs: 1,
        },
      })],
    });

    expect(payload.passed).toBe(false);
    expect(payload.summary).toMatchObject({
      memoryGroundingRequired: true,
      memoryGroundingPassed: false,
      memoryExpectedHitPassed: false,
    });
  });

  it("fails grounded payloads when any run is missing memory grounding proof", () => {
    const payload = buildBrowserPreviewBenchmarkPayload({
      createdAt: "2026-05-23T00:00:00.000Z",
      profile: "full",
      strictWebGpuRequested: true,
      runs: [
        makeRun({
          expectedSubstrings: ["Helena"],
          response: "[unlocked:ssa-kv-tsp]Helena.",
          expectedAnswerOnlyPassed: true,
          memoryGrounding: {
            mode: "seeded_browser_vector_context_rebuild",
            caseId: "montana_capital",
            corpusCount: 16,
            retrievedMemoryIds: ["bench_memory_montana_capital"],
            includedMemoryIds: ["bench_memory_montana_capital"],
            expectedMemoryIds: ["bench_memory_montana_capital"],
            expectedMemoryHitPassed: true,
            contextRebuildPassed: true,
            answerOnlyExpected: true,
            answerOnlyPassed: true,
            contextEstimatedTokens: 90,
            retrievalMs: 2,
            contextRebuildMs: 1,
          },
        }),
        makeRun(),
      ],
    });

    expect(payload.passed).toBe(false);
    expect(payload.summary).toMatchObject({
      memoryGroundingRequired: true,
      memoryGroundingCoveragePassed: false,
      memoryGroundingPassed: false,
    });
  });

  it("fails grounded payloads when the answer is not answer-only", () => {
    const payload = buildBrowserPreviewBenchmarkPayload({
      createdAt: "2026-05-23T00:00:00.000Z",
      profile: "full",
      strictWebGpuRequested: true,
      runs: [makeRun({
        expectedSubstrings: ["Helena"],
        response: "[unlocked:ssa-kv-tsp]Helena. ✅. (Answer",
        expectedAnswerOnlyPassed: false,
        memoryGrounding: {
          mode: "seeded_browser_vector_context_rebuild",
          caseId: "montana_capital",
          corpusCount: 16,
          retrievedMemoryIds: ["bench_memory_montana_capital"],
          includedMemoryIds: ["bench_memory_montana_capital"],
          expectedMemoryIds: ["bench_memory_montana_capital"],
          expectedMemoryHitPassed: true,
          contextRebuildPassed: true,
          answerOnlyExpected: true,
          answerOnlyPassed: false,
          contextEstimatedTokens: 90,
          retrievalMs: 2,
          contextRebuildMs: 1,
        },
      })],
    });

    expect(payload.passed).toBe(false);
    expect(payload.summary).toMatchObject({
      memoryAnswerOnlyPassed: false,
      memoryGroundingPassed: false,
      stopQualityPassed: false,
      stopQualityFailureCount: 1,
    });
  });

  it("fails stop quality for repeated-token loops even when expected text appears", () => {
    const payload = buildBrowserPreviewBenchmarkPayload({
      createdAt: "2026-05-23T00:00:00.000Z",
      profile: "full",
      strictWebGpuRequested: true,
      runs: [makeRun({
        expectedSubstrings: ["Helena"],
        response: "[unlocked:ssa-kv-tsp]Helena Helena Helena Helena",
        coherent: false,
        expectedAnswerOnlyPassed: false,
        memoryGrounding: {
          mode: "seeded_browser_vector_context_rebuild",
          caseId: "montana_capital",
          corpusCount: 16,
          retrievedMemoryIds: ["bench_memory_montana_capital"],
          includedMemoryIds: ["bench_memory_montana_capital"],
          expectedMemoryIds: ["bench_memory_montana_capital"],
          expectedMemoryHitPassed: true,
          contextRebuildPassed: true,
          answerOnlyExpected: true,
          answerOnlyPassed: false,
          contextEstimatedTokens: 90,
          retrievalMs: 2,
          contextRebuildMs: 1,
        },
      })],
    });

    expect(payload.passed).toBe(false);
    expect(payload.summary).toMatchObject({
      expectedSubstringsPassed: true,
      stopQualityPassed: false,
      stopQualityFailureCount: 1,
    });
  });

  it("uses total generation wall time when an accepted MTP batch streams all tokens in the first chunk", () => {
    expect(calculateBrowserPreviewRunTiming({
      start: 0,
      firstChunkAt: 100,
      firstGeneratedAt: 100,
      endAt: 100,
      generatedTokens: 4,
    })).toEqual({
      prefillMs: 100,
      timeToFirstTokenMs: 100,
      decodeLatencyMs: 100,
      tokensPerSecond: 40,
    });
  });
});

function makeRun(
  overrides: Partial<Omit<BrowserPreviewBenchmarkRun, "kvPersistence" | "mtp">> & {
    kvPersistence?: Partial<BrowserPreviewBenchmarkRun["kvPersistence"]>;
    mtp?: Partial<BrowserPreviewBenchmarkRun["mtp"]>;
  } = {},
): BrowserPreviewBenchmarkRun {
  const { kvPersistence: kvPersistenceOverride, mtp: mtpOverride, ...runOverrides } = overrides;
  return {
    promptId: "prompt-1",
    prompt: "Explain persistent runtime intelligence.",
    response: "[unlocked:ssa-kv-tsp] Persistent runtime intelligence keeps model, memory, and runtime state distinct.",
    coherent: true,
    expectedSubstrings: [],
    expectedSubstringMatches: [],
    metrics: {
      initLoadMs: 20,
      prefillMs: 8,
      timeToFirstTokenMs: 11,
      decodeLatencyMs: 6,
      tokensPerSecond: 166.67,
      generatedTokens: 4,
    },
    runtimeTrace: {
      backend: "unlocked-browser-transformer",
      tensorControl: true,
      tspSteps: ["kv_prefetch", "attention", "mlp"],
      kvPagingEvents: 1,
      selectedBlockIds: ["b0", "b1"],
    },
    predictive: {
      promptTokenCount: 9,
      generatedTokenCount: 4,
      selectedBlockCount: 2,
      kvPagingEventCount: 1,
      tspStepCount: 3,
    },
    webGpu: {
      available: true,
      requestedBackendPreference: "webgpu",
      logitProjectionBackend: "webgpu",
      cpuFallbackUsed: false,
      noCpuFallback: true,
      requestedGates: [],
      passedGates: [],
      failedGates: [],
      positiveKernelProof: true,
    },
    mtp: {
      mode: "draft_verify",
      draftModelId: "browser/qwen-prefix-drafter",
      draftSource: "qwen_prefix_draft",
      latencyDisablePolicy: "paired_benchmark_required",
      acceptedTokens: 1,
      rejectedTokens: 0,
      acceptanceRate: 1,
      numSpeculativeTokens: 2,
      verifiedTokenCount: 2,
      targetDecodeCalls: 1,
      verifierStrategy: "batched_continuation",
      ...mtpOverride,
    },
    kvPersistence: {
      enabled: true,
      mode: "indexeddb",
      eventCount: 2,
      persistEvents: 1,
      hydrateEvents: 1,
      reuseEvents: 0,
      prefetchStrategy: "none",
      predictedHotBlocks: [],
      prefetchedBlocks: [],
      ...kvPersistenceOverride,
    },
    ...runOverrides,
  };
}

function makePassedQaRetrievalAudit(): NonNullable<NonNullable<BrowserPreviewBenchmarkRun["memoryGrounding"]>["retrievalAudit"]> {
  return {
    corpusCount: 64,
    queryCount: 23,
    top1CorrectCount: 23,
    reciprocalRankSum: 23,
    recallAt1: 1,
    mrr: 1,
    canonicalQueryCount: 6,
    canonicalTop1CorrectCount: 6,
    canonicalRecallAt1: 1,
    canonicalMrr: 1,
    aliasQueryCount: 5,
    aliasTop1CorrectCount: 5,
    aliasRecallAt1: 1,
    aliasMrr: 1,
    generatedParaphraseQueryCount: 12,
    generatedParaphraseTop1CorrectCount: 12,
    generatedParaphraseRecallAt1: 1,
    generatedParaphraseMrr: 1,
    queryClassBreakdown: [
      { queryClass: "canonical", queryCount: 6, top1CorrectCount: 6, recallAt1: 1, mrr: 1 },
      { queryClass: "alias", queryCount: 5, top1CorrectCount: 5, recallAt1: 1, mrr: 1 },
      { queryClass: "generated_paraphrase", queryCount: 12, top1CorrectCount: 12, recallAt1: 1, mrr: 1 },
    ],
    minTopScoreMargin: 0.12,
    meanExpectedHitRank: 1,
    passed: true,
    elapsedMs: 3,
    minRequiredRecallAt1: 1,
  };
}
