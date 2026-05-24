import { describe, expect, it } from "vitest";
import {
  buildBrowserRuntimeBenchmarkFailureArtifact,
  buildBrowserRuntimeBenchmarkArtifact,
  buildMarkdownSummary,
  calculateTokensPerSecond,
  evaluateExpectedSubstringMatches,
  evaluateBrowserRuntimeBenchmarkThresholds,
  parseBrowserRuntimeBenchmarkThresholds,
  resolveBrowserRuntimeBenchmarkGenerationTokens,
  parseQwenThinkingMode,
  readBrowserPreviewWebGpuGates,
  readBenchmarkWebGpuGates,
  readBrowserRuntimeBenchmarkArgs,
  shouldRequireWebGpuForBenchmark,
  readBrowserPreviewBenchmark,
  calculateBrowserRuntimeRunTiming,
  type BrowserRuntimeBenchmarkRun,
} from "./browserRuntimeBenchmark";
import type { UnlockedWebGpuCoverageSummary } from "./unlockedWebGpuCoverage";

const run: BrowserRuntimeBenchmarkRun = {
  promptId: "fixture-short",
  prompt: "alpha beta",
  response: "[unlocked:ssa-kv-tsp] beta",
  metrics: {
    initLoadMs: 12,
    prefillMs: 7,
    timeToFirstTokenMs: 7,
    decodeLatencyMs: 5,
    tokensPerSecond: 200,
    generatedTokens: 1,
  },
  mtp: {
    mode: "draft_verify",
    draftModelId: "browser/qwen-prefix-drafter",
    draftSource: "qwen_prefix_draft",
    latencyDisablePolicy: "paired_benchmark_required",
    verifierStrategy: "batched_continuation",
    acceptedTokens: 1,
    rejectedTokens: 0,
    acceptanceRate: 1,
    numSpeculativeTokens: 2,
    verifiedTokenCount: 2,
    targetDecodeCalls: 1,
    committedInputTokens: 2,
  },
  kvPersistence: {
    enabled: true,
    mode: "indexeddb",
    eventCount: 3,
    persistEvents: 1,
    hydrateEvents: 1,
    reuseEvents: 0,
    prefetchStrategy: "predictive_prefetch",
    lowRankSummaryRank: 4,
    predictedHotBlocks: ["block-0", "block-1"],
    prefetchedBlocks: ["block-0"],
    prefetchHitRate: 0.5,
    prefetchBytes: 2048,
    prefetchLatencyMs: 3.5,
    attentionStallMs: 0,
  },
  decodePerf: {
    requestId: "bench-fixture",
    generatedTokenCount: 1,
    decodeCallCount: 1,
    decodeSubmitCount: 7,
    dispatchCount: 9,
    decodeSubmitCountPerToken: 7,
    decodeDispatchCountPerToken: 9,
    decodeDispatchCountPerLayerPerToken: 9,
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
    tokensPerSecond: 200,
  },
};

const browserPreviewKvSummary = {
  technicalProofOnly: false,
  productionQualityPassed: false,
  productionDeployReadyPassed: false,
  kvPersistenceEventCount: 3,
  kvPersistEventCount: 1,
  kvHydrateEventCount: 1,
  kvReuseEventCount: 0,
  kvPrefetchStrategy: "predictive_prefetch",
  kvExactReuseRunCount: 0,
  kvPredictivePrefetchRunCount: 1,
  kvMissStallRunCount: 0,
  kvNoPrefetchRunCount: 0,
  kvLowRankSummaryRank: 4,
  kvPredictedHotBlockCount: 2,
  kvPrefetchedBlockCount: 1,
  kvPrefetchHitRate: 0.5,
  kvPrefetchBytes: 2048,
  kvPrefetchLatencyMs: 3.5,
  kvAttentionStallMs: 0,
};

const browserPreviewMtpSummary = {
  technicalProofOnly: false,
  productionQualityPassed: false,
  productionDeployReadyPassed: false,
  mtpMaxSpeculativeTokens: 2,
  mtpMeanSpeculativeTokens: 2,
  mtpVerifiedTokenCount: 2,
  mtpTargetDecodeCalls: 1,
  mtpVerifierStrategy: "batched_continuation",
};

describe("browser runtime benchmark artifacts", () => {
  it("keeps benchmark generation budget explicit instead of silently using the full-profile runtime caps", () => {
    expect(resolveBrowserRuntimeBenchmarkGenerationTokens({
      mode: "configured",
      explicitValue: undefined,
    })).toBe(16);
    expect(resolveBrowserRuntimeBenchmarkGenerationTokens({
      mode: "generated-fixture",
      explicitValue: undefined,
    })).toBe(4);
    expect(resolveBrowserRuntimeBenchmarkGenerationTokens({
      mode: "configured",
      explicitValue: "64",
    })).toBe(64);
  });

  it("defaults configured strict browser preview proof to meaningful prompts, expected text, token floor, and KV reuse", () => {
    const args = readBrowserRuntimeBenchmarkArgs([
      "--browser-preview-url",
      "http://127.0.0.1:5173/__bench/browser-runtime",
      "--manifest-path",
      "apps/web/public/models/qwen3-0.6b-unlocked/manifest.json",
      "--manifest-sha256",
      "abc123",
      "--require-browser-preview",
      "--browser-preview-require-strict-webgpu",
    ], {});

    expect(args.prompts).toEqual([
      {
        id: "prompt-1",
        text: "What is the capital of Utah? Answer in one clear sentence.",
        expectedSubstrings: ["Salt Lake"],
      },
      {
        id: "prompt-2",
        text: "Write two clear sentences about Earth.",
        expectedSubstrings: ["Earth"],
      },
    ]);
    expect(args.generationTokenBudget).toBe(16);
    expect(args.browserPreviewMinGeneratedTokens).toBe(8);
    expect(args.browserPreviewRequireKvReuse).toBe(true);
    expect(args.browserPreviewWebGpuGates).toEqual(["mlp", "logits", "attention", "projection"]);
  });

  it("defaults production browser runtime benchmarks to target-only unless MTP is explicitly enabled", () => {
    expect(readBrowserRuntimeBenchmarkArgs([], {}).mtpEnabled).toBe(false);
    expect(readBrowserRuntimeBenchmarkArgs([], { VITE_MTP_ENABLED: "false" }).mtpEnabled).toBe(false);
    expect(readBrowserRuntimeBenchmarkArgs(["--mtp-enabled"], {}).mtpEnabled).toBe(true);
    expect(readBrowserRuntimeBenchmarkArgs([], { VITE_MTP_ENABLED: "true" }).mtpEnabled).toBe(true);
    expect(readBrowserRuntimeBenchmarkArgs(["--mtp-enabled", "--mtp-disabled"], {}).mtpEnabled).toBe(false);
  });

  it("can generate a deterministic long prompt and request strict long-prompt proof", () => {
    const args = readBrowserRuntimeBenchmarkArgs([
      "--long-prompt-target-tokens",
      "12",
      "--require-long-prompt-proof",
    ], {});

    expect(args.prompts).toEqual([
      {
        id: "long-prompt-1",
        text: "alpha beta gamma delta runtime memory context tensor alpha beta gamma delta",
        expectedSubstrings: [],
      },
    ]);
    expect(args.longPromptTargetTokens).toBe(12);
    expect(args.strictLongPromptProof).toBe(true);
  });

  it("clamps oversized long-prompt CLI controls before generating prompt text", () => {
    const args = readBrowserRuntimeBenchmarkArgs([
      "--long-prompt-target-tokens",
      "999999999",
    ], {});

    expect(args.longPromptTargetTokens).toBe(8192);
    expect(args.prompts[0]?.text.split(/\s+/)).toHaveLength(8192);
  });

  it("preserves the original bounded long-prompt seed separately from expanded prompt text", () => {
    const args = readBrowserRuntimeBenchmarkArgs([
      "--prompts",
      "red blue green|ignored",
      "--long-prompt-target-tokens",
      "70",
    ], {});

    expect(args.longPromptSeed).toBe("red blue green");
    expect(args.prompts[0]?.text.split(/\s+/).slice(63, 69)).toEqual([
      "red",
      "blue",
      "green",
      "red",
      "blue",
      "green",
    ]);
  });

  it("defaults Qwen benchmark runs to fast visible no-think mode unless thinking is explicit", () => {
    expect(parseQwenThinkingMode(undefined)).toBe("disabled");
    expect(parseQwenThinkingMode("bad")).toBe("disabled");
    expect(parseQwenThinkingMode("enabled")).toBe("enabled");
    expect(parseQwenThinkingMode("disabled")).toBe("disabled");
  });

  it("keeps runtime, backend, MTP, and CPU/WebGPU coverage fields in the release artifact", () => {
    const artifact = buildBrowserRuntimeBenchmarkArtifact({
      createdAt: "2026-05-17T00:00:00.000Z",
      mode: "generated-fixture",
      manifestUrl: "file:///tmp/manifest.json",
      modelId: "Qwen/Qwen3-0.6B",
      requestedBackendPreference: "cpu",
      runtimeProfile: {
        profile: "ci",
        caps: {
          maxRuntimePromptTokens: 4,
          maxRuntimeLayers: 1,
          maxGenerationTokens: 1,
          logitCandidateLimit: 64,
        },
        capStatus: {
          prompt: true,
          layers: true,
          generation: true,
          logits: true,
        },
      },
      memoryMode: "browser-local-fixture",
      backendProofs: {
        tensorControl: true,
        tspSteps: ["kv_prefetch", "attention", "mlp"],
        kvPagingEvents: 0,
      },
      webGpuCoverage: {
        cpuFallbackUsed: true,
        logitProjection: { backend: "cpu_reference", purpose: "candidate_logit_projection", selectedRows: 4, fullRows: 4 },
        mlpLayersByBackend: { webgpu: 0, cpu_reference: 1, mixed: 0, unknown: 0 },
        prefillMlpLayers: [],
        decodeMlpLayers: [{ layerIndex: 0, backend: "cpu_reference", activationKind: "silu_gated" }],
        mlpLayers: [{ layerIndex: 0, backend: "cpu_reference", activationKind: "silu_gated" }],
        prefillProjectionBackends: {
          qProjection: { webgpu: 0, cpu_reference: 1, mixed: 0, unknown: 0 },
          kProjection: { webgpu: 0, cpu_reference: 1, mixed: 0, unknown: 0 },
          vProjection: { webgpu: 0, cpu_reference: 1, mixed: 0, unknown: 0 },
          oProjection: { webgpu: 0, cpu_reference: 1, mixed: 0, unknown: 0 },
          layers: [{ layerIndex: 0, qProjection: "cpu_reference", kProjection: "cpu_reference", vProjection: "cpu_reference", oProjection: "cpu_reference" }],
        },
        decodeProjectionBackends: {
          qProjection: { webgpu: 0, cpu_reference: 1, mixed: 0, unknown: 0 },
          kProjection: { webgpu: 0, cpu_reference: 1, mixed: 0, unknown: 0 },
          vProjection: { webgpu: 0, cpu_reference: 1, mixed: 0, unknown: 0 },
          oProjection: { webgpu: 0, cpu_reference: 1, mixed: 0, unknown: 0 },
          layers: [{ layerIndex: 0, qProjection: "cpu_reference", kProjection: "cpu_reference", vProjection: "cpu_reference", oProjection: "cpu_reference", projectionKind: "matvec", tokens: null, selectedRows: 4 }],
        },
        attentionBackends: {
          prefill: { webgpu: 0, cpu_reference: 1, mixed: 0, unknown: 0 },
          decode: { webgpu: 0, cpu_reference: 1, mixed: 0, unknown: 0 },
          packedHeads: { webgpu: 0, cpu_reference: 2, mixed: 0, unknown: 0 },
          prefillLayers: [{ layerIndex: 0, attentionBackend: "cpu_reference", packedHeadBackends: ["cpu_reference"], packedHeadCount: 1, packedHeadComplete: true }],
          decodeLayers: [{ layerIndex: 0, attentionBackend: "cpu_reference", packedHeadBackends: ["cpu_reference"], packedHeadCount: 1, packedHeadComplete: true }],
          incompletePackedHeadProofs: 0,
        },
      },
      runs: [run],
      thresholds: [],
      mtpAcceleration: {
        mode: "skipped",
        requested: false,
        passed: true,
        reason: "not_requested",
      },
      browserPreview: {
        mode: "skipped",
        requested: false,
        reason: "not_requested",
      },
      generationTokenBudgetUsed: 16,
      browserPreviewRequired: false,
    });

    expect(artifact.passed).toBe(true);
    expect(artifact.browserPreview).toEqual({
      mode: "skipped",
      requested: false,
      reason: "not_requested",
    });
    expect(artifact.summary).toMatchObject({
      profile: "ci",
      memoryMode: "browser-local-fixture",
      memoryQueryMode: "direct_model_no_memory_retrieval",
      promptCount: 1,
      browserPreviewMode: "skipped",
      browserPreviewRequested: false,
      browserPreviewPassed: false,
      browserPreviewRequired: false,
      meanInitLoadMs: 12,
      meanPrefillMs: 7,
      meanTimeToFirstTokenMs: 7,
      meanDecodeLatencyMs: 5,
      meanTokensPerSecond: 200,
      decodeSubmitCount: 7,
      decodeSubmitCountPerToken: 7,
      decodeDispatchCount: 9,
      decodeDispatchCountPerToken: 9,
      decodeDispatchCountPerLayerPerToken: 9,
      v11CommandBatchingPassed: false,
      fusedLayerCoverage: 0,
      parityRecordCount: 0,
      parityPassedCount: 0,
      parityFailedCount: 0,
      mtpMode: "draft_verify",
      mtpAcceptanceRate: 1,
      mtpMaxSpeculativeTokens: 2,
      mtpMeanSpeculativeTokens: 2,
      mtpVerifiedTokenCount: 2,
      mtpTargetDecodeCalls: 1,
      mtpVerifierStrategy: "batched_continuation",
      mtpAccelerationMode: "skipped",
      mtpAccelerationRequested: false,
      mtpAccelerationPassed: true,
      kvPersistenceEventCount: 3,
      kvPersistEventCount: 1,
      kvHydrateEventCount: 1,
      kvReuseEventCount: 0,
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
      cpuFallbackUsed: true,
      logitProjectionBackend: "cpu_reference",
      generationTokenBudgetUsed: 16,
    });
    expect(artifact.webGpuCoverage.cpuFallbackUsed).toBe(true);
    expect(artifact.runs[0]?.mtp).toMatchObject({
      draftModelId: "browser/qwen-prefix-drafter",
      draftSource: "qwen_prefix_draft",
      latencyDisablePolicy: "paired_benchmark_required",
    });
    expect(buildMarkdownSummary(artifact)).toContain("- KV prefetch strategy: predictive_prefetch");
    expect(buildMarkdownSummary(artifact)).toContain("- Decode fragmentation: submits/token=7, dispatches/token=9, dispatches/layer/token=9, fusedCoverage=0");
  });

  it("carries prefill chunk metadata through benchmark runs and summaries", () => {
    const chunkedRun = {
      ...run,
      prefillChunkCount: 4,
      prefillChunkSize: 1024,
      shapeBucket: "prompt<=1024:selected<=512:headDim<=128:tileRows<=8192:precision=f32",
      pipelineCacheKey: "prefill_chunk:prompt<=1024:selected<=512:headDim<=128:tileRows<=8192:precision=f32",
      maxDispatchEstimatedMs: 7.25,
    } as BrowserRuntimeBenchmarkRun & {
      prefillChunkCount: number;
      prefillChunkSize: number;
      shapeBucket: string;
      pipelineCacheKey: string;
      maxDispatchEstimatedMs: number;
    };

    const artifact = buildBrowserRuntimeBenchmarkArtifact({
      createdAt: "2026-05-20T00:00:00.000Z",
      mode: "configured",
      manifestUrl: "file:///tmp/manifest.json",
      modelId: "Qwen/Qwen3-0.6B",
      requestedBackendPreference: "webgpu",
      runtimeProfile: {
        profile: "full",
        caps: {
          maxRuntimePromptTokens: null,
          maxRuntimeLayers: null,
          maxGenerationTokens: null,
          logitCandidateLimit: null,
        },
        capStatus: {
          prompt: false,
          layers: false,
          generation: false,
          logits: false,
        },
      },
      memoryMode: "browser-local",
      backendProofs: {
        tensorControl: true,
        tspSteps: ["kv_prefetch", "attention", "mlp"],
        kvPagingEvents: 1,
      },
      webGpuCoverage: emptyWebGpuCoverage(),
      runs: [chunkedRun],
      thresholds: [],
      mtpAcceleration: {
        mode: "skipped",
        requested: false,
        passed: true,
        reason: "not_requested",
      },
      browserPreview: {
        mode: "skipped",
        requested: false,
        reason: "not_requested",
      },
    });

    expect(artifact.runs[0] as typeof chunkedRun).toMatchObject({
      prefillChunkCount: 4,
      prefillChunkSize: 1024,
      shapeBucket: "prompt<=1024:selected<=512:headDim<=128:tileRows<=8192:precision=f32",
      pipelineCacheKey: "prefill_chunk:prompt<=1024:selected<=512:headDim<=128:tileRows<=8192:precision=f32",
      maxDispatchEstimatedMs: 7.25,
    });
    expect(artifact.summary).toMatchObject({
      prefillChunkCount: 4,
      prefillChunkSize: 1024,
      shapeBucket: "prompt<=1024:selected<=512:headDim<=128:tileRows<=8192:precision=f32",
      pipelineCacheKey: "prefill_chunk:prompt<=1024:selected<=512:headDim<=128:tileRows<=8192:precision=f32",
      maxDispatchEstimatedMs: 7.25,
    });
    expect(buildMarkdownSummary(artifact)).toContain(
      "- Prefill shape bucket: prompt<=1024:selected<=512:headDim<=128:tileRows<=8192:precision=f32",
    );
  });

  it("fails strict artifacts when browser preview proof is required but skipped", () => {
    const artifact = buildBrowserRuntimeBenchmarkArtifact({
      createdAt: "2026-05-17T00:00:00.000Z",
      mode: "configured",
      manifestUrl: "file:///tmp/manifest.json",
      modelId: "Qwen/Qwen3-0.6B",
      requestedBackendPreference: "webgpu",
      runtimeProfile: {
        profile: "full",
        caps: {
          maxRuntimePromptTokens: null,
          maxRuntimeLayers: null,
          maxGenerationTokens: null,
          logitCandidateLimit: null,
        },
        capStatus: {
          prompt: false,
          layers: false,
          generation: false,
          logits: false,
        },
      },
      memoryMode: "browser-local",
      backendProofs: {
        tensorControl: true,
        tspSteps: ["kv_prefetch", "attention", "mlp"],
        kvPagingEvents: 1,
      },
      webGpuCoverage: emptyWebGpuCoverage(),
      runs: [run],
      thresholds: [],
      mtpAcceleration: {
        mode: "skipped",
        requested: false,
        passed: true,
        reason: "not_requested",
      },
      browserPreview: {
        mode: "skipped",
        requested: false,
        reason: "not_requested",
      },
      browserPreviewRequired: true,
    });

    expect(artifact.passed).toBe(false);
    expect(artifact.summary).toMatchObject({
      browserPreviewMode: "skipped",
      browserPreviewRequested: false,
      browserPreviewPassed: false,
      browserPreviewRequired: true,
      blockingThresholdFailures: 1,
    });
  });

  it("fails required production browser previews that are not deploy-ready", () => {
    const artifact = buildBrowserRuntimeBenchmarkArtifact({
      createdAt: "2026-05-23T00:00:00.000Z",
      mode: "configured",
      manifestUrl: "file:///tmp/manifest.json",
      modelId: "Qwen/Qwen3-0.6B",
      requestedBackendPreference: "webgpu",
      runtimeProfile: {
        profile: "full",
        caps: {
          maxRuntimePromptTokens: null,
          maxRuntimeLayers: null,
          maxGenerationTokens: null,
          logitCandidateLimit: null,
        },
        capStatus: {
          prompt: false,
          layers: false,
          generation: false,
          logits: false,
        },
      },
      memoryMode: "browser-local",
      backendProofs: {
        tensorControl: true,
        tspSteps: ["kv_prefetch", "attention", "mlp"],
        kvPagingEvents: 1,
      },
      webGpuCoverage: emptyWebGpuCoverage(),
      runs: [run],
      thresholds: [],
      mtpAcceleration: {
        mode: "skipped",
        requested: false,
        passed: true,
        reason: "not_requested",
      },
      browserPreview: {
        mode: "completed",
        requested: true,
        url: "http://127.0.0.1:5173/__bench/browser-runtime",
        passed: true,
        summary: {
          meanInitLoadMs: 20,
          meanPrefillMs: 8,
          meanTimeToFirstTokenMs: 11,
          meanDecodeLatencyMs: 6,
          meanTokensPerSecond: 3,
          mtpMode: "target_only",
          technicalProofOnly: false,
          productionQualityPassed: true,
          groundedProductionReadyPassed: false,
          productionDeployReadyPassed: false,
          memoryGroundingRequired: true,
          memoryGroundingPassed: true,
          memoryGeneratedParaphraseRequired: true,
          memoryGeneratedParaphrasePassed: false,
          memoryGeneratedParaphraseQueryCount: 0,
          memoryGeneratedParaphraseRecallAt1: null,
          ...browserPreviewMtpSummary,
          ...browserPreviewKvSummary,
        },
        runs: [run],
      },
      browserPreviewRequired: true,
    });

    expect(artifact.passed).toBe(false);
    expect(artifact.summary).toMatchObject({
      browserPreviewMode: "completed",
      browserPreviewPassed: true,
      browserPreviewRequired: true,
      browserPreviewProductionDeployReadyPassed: false,
      browserPreviewGroundedProductionReadyPassed: false,
      browserPreviewMemoryGeneratedParaphrasePassed: false,
      blockingThresholdFailures: 1,
    });
  });

  it("does not require deploy-ready browser summary for technical audit-only previews", () => {
    const artifact = buildBrowserRuntimeBenchmarkArtifact({
      createdAt: "2026-05-23T00:00:00.000Z",
      mode: "generated-fixture",
      manifestUrl: "file:///tmp/manifest.json",
      modelId: "Qwen/Qwen3-0.6B",
      requestedBackendPreference: "cpu",
      runtimeProfile: {
        profile: "ci",
        caps: {
          maxRuntimePromptTokens: 4,
          maxRuntimeLayers: 1,
          maxGenerationTokens: 1,
          logitCandidateLimit: 64,
        },
        capStatus: {
          prompt: true,
          layers: true,
          generation: true,
          logits: true,
        },
      },
      memoryMode: "browser-local-fixture",
      backendProofs: {
        tensorControl: true,
        tspSteps: ["kv_prefetch", "attention", "mlp"],
        kvPagingEvents: 1,
      },
      webGpuCoverage: emptyWebGpuCoverage(),
      runs: [run],
      thresholds: [],
      mtpAcceleration: {
        mode: "skipped",
        requested: false,
        passed: true,
        reason: "not_requested",
      },
      browserPreview: {
        mode: "completed",
        requested: true,
        url: "http://127.0.0.1:5173/__bench/browser-runtime?memoryGroundingAuditOnly=true",
        passed: true,
        summary: {
          ...browserPreviewMtpSummary,
          ...browserPreviewKvSummary,
          meanInitLoadMs: 0,
          meanPrefillMs: 0,
          meanTimeToFirstTokenMs: 50,
          meanDecodeLatencyMs: 0,
          meanTokensPerSecond: null,
          mtpMode: "target_only",
          technicalProofOnly: true,
          productionQualityPassed: false,
          productionDeployReadyPassed: false,
          memoryGroundingAuditOnly: true,
          memoryGroundingRequired: true,
          memoryGroundingPassed: true,
          memoryGeneratedParaphraseRequired: true,
          memoryGeneratedParaphrasePassed: true,
          memoryGeneratedParaphraseQueryCount: 12,
          memoryGeneratedParaphraseTop1CorrectCount: 12,
          memoryGeneratedParaphraseRecallAt1: 1,
          memoryGeneratedParaphraseMrr: 1,
        },
        runs: [],
      },
      browserPreviewRequired: true,
    });

    expect(artifact.passed).toBe(true);
    expect(artifact.summary).toMatchObject({
      browserPreviewPassed: true,
      browserPreviewProductionDeployReadyPassed: false,
      browserPreviewMemoryGeneratedParaphrasePassed: true,
      blockingThresholdFailures: 0,
    });
  });

  it("fails direct benchmark quality gates when required expected text is absent", () => {
    const artifact = buildBrowserRuntimeBenchmarkArtifact({
      createdAt: "2026-05-18T00:00:00.000Z",
      mode: "configured",
      manifestUrl: "file:///tmp/manifest.json",
      modelId: "Qwen/Qwen3-0.6B",
      requestedBackendPreference: "webgpu",
      runtimeProfile: {
        profile: "full",
        caps: {
          maxRuntimePromptTokens: null,
          maxRuntimeLayers: null,
          maxGenerationTokens: null,
          logitCandidateLimit: null,
        },
        capStatus: {
          prompt: false,
          layers: false,
          generation: false,
          logits: false,
        },
      },
      memoryMode: "browser-vector",
      backendProofs: {
        tensorControl: true,
        tspSteps: ["kv_prefetch", "attention", "mlp"],
        kvPagingEvents: 1,
      },
      webGpuCoverage: emptyWebGpuCoverage(),
      runs: [{
        ...run,
        promptId: "utah",
        prompt: "What is the capital of Utah? Answer in one clear sentence.",
        response: "The capital of Utah is Great Falls.",
        expectedSubstrings: ["Salt Lake"],
        expectedSubstringMatches: [{ expected: "Salt Lake", matched: false }],
      }],
      thresholds: [],
      mtpAcceleration: {
        mode: "skipped",
        requested: false,
        passed: true,
        reason: "not_requested",
      },
      browserPreview: {
        mode: "skipped",
        requested: false,
        reason: "not_requested",
      },
      browserPreviewRequired: false,
    });

    expect(artifact.passed).toBe(false);
    expect(artifact.summary).toMatchObject({
      expectedSubstringCheckCount: 1,
      expectedSubstringPassCount: 0,
      expectedSubstringsPassed: false,
      memoryQueryMode: "direct_model_no_memory_retrieval",
      blockingThresholdFailures: 1,
    });
  });

  it("does not let hidden thinking text satisfy direct expected substring checks", () => {
    expect(evaluateExpectedSubstringMatches(
      "<think>Salt Lake City is the hidden answer.</think>The capital of Utah is Great Falls.",
      ["Salt Lake"],
    )).toEqual([{ expected: "Salt Lake", matched: false }]);
  });

  it("fails requested MTP acceleration gates when paired target-only comparison does not clear floors", () => {
    const artifact = buildBrowserRuntimeBenchmarkArtifact({
      createdAt: "2026-05-17T00:00:00.000Z",
      mode: "configured",
      manifestUrl: "file:///tmp/manifest.json",
      modelId: "Qwen/Qwen3-0.6B",
      requestedBackendPreference: "cpu",
      runtimeProfile: {
        profile: "ci",
        caps: {
          maxRuntimePromptTokens: 4,
          maxRuntimeLayers: 1,
          maxGenerationTokens: 1,
          logitCandidateLimit: 64,
        },
        capStatus: {
          prompt: true,
          layers: true,
          generation: true,
          logits: true,
        },
      },
      memoryMode: "browser-local",
      backendProofs: {
        tensorControl: true,
        tspSteps: ["kv_prefetch", "attention", "mlp"],
        kvPagingEvents: 1,
      },
      webGpuCoverage: emptyWebGpuCoverage(),
      runs: [{
        ...run,
        mtp: {
          mode: "draft_verify",
          acceptedTokens: 0,
          rejectedTokens: 1,
          acceptanceRate: 0,
        },
      }],
      thresholds: [],
      mtpAcceleration: {
        mode: "completed",
        requested: true,
        passed: false,
        minAcceptanceRate: 0.25,
        minNetSpeedup: 1.05,
        targetOnlyRuns: [{ ...run, mtp: { mode: "target_only", acceptedTokens: 0, rejectedTokens: 0, acceptanceRate: 0 } }],
        targetOnlyMeanTokensPerSecond: 200,
        draftVerifyMeanTokensPerSecond: 100,
        acceptanceRate: 0,
        netSpeedupRatio: 0.5,
        failedReasons: ["MTP acceptance rate 0 is below 0.25", "MTP net speedup ratio 0.5 is below 1.05"],
      },
      browserPreview: {
        mode: "skipped",
        requested: false,
        reason: "not_requested",
      },
    });

    expect(artifact.passed).toBe(false);
    expect(artifact.summary).toMatchObject({
      mtpAccelerationMode: "completed",
      mtpAccelerationRequested: true,
      mtpAccelerationPassed: false,
      mtpNetSpeedupRatio: 0.5,
      mtpAccelerationFailureCount: 2,
      blockingThresholdFailures: 1,
    });
  });

  it("fails strict WebGPU benchmark gates when the unlocked proof used CPU fallback", () => {
    const artifact = buildBrowserRuntimeBenchmarkArtifact({
      createdAt: "2026-05-17T00:00:00.000Z",
      mode: "configured",
      manifestUrl: "file:///tmp/manifest.json",
      modelId: "Qwen/Qwen3-0.6B",
      requestedBackendPreference: "webgpu",
      runtimeProfile: {
        profile: "full",
        caps: {
          maxRuntimePromptTokens: null,
          maxRuntimeLayers: null,
          maxGenerationTokens: null,
          logitCandidateLimit: null,
        },
        capStatus: {
          prompt: false,
          layers: false,
          generation: false,
          logits: false,
        },
      },
      memoryMode: "browser-local",
      backendProofs: {
        tensorControl: true,
        tspSteps: ["kv_prefetch", "attention", "mlp"],
        kvPagingEvents: 1,
      },
      webGpuCoverage: {
        ...emptyWebGpuCoverage(),
        cpuFallbackUsed: true,
        logitProjection: { backend: "cpu_reference", purpose: "full_vocab_topk_logit_projection", selectedRows: 4, fullRows: 4 },
      },
      webGpuGates: ["logits"],
      runs: [run],
      thresholds: [],
      mtpAcceleration: {
        mode: "skipped",
        requested: false,
        passed: true,
        reason: "not_requested",
      },
      browserPreview: {
        mode: "skipped",
        requested: false,
        reason: "not_requested",
      },
    });

    expect(artifact.passed).toBe(false);
    expect(artifact.webGpuGate).toMatchObject({
      required: true,
      gates: ["logits"],
      passed: false,
      failedReasons: ["logit projection backend is cpu_reference"],
    });
    expect(artifact.summary).toMatchObject({
      strictWebGpuRequired: true,
      strictWebGpuPassed: false,
      strictWebGpuGateCount: 1,
      strictWebGpuFailureCount: 1,
      blockingThresholdFailures: 1,
    });
  });

  it("passes strict WebGPU benchmark gates when every requested proof is WebGPU-backed", () => {
    const artifact = buildBrowserRuntimeBenchmarkArtifact({
      createdAt: "2026-05-17T00:00:00.000Z",
      mode: "configured",
      manifestUrl: "file:///tmp/manifest.json",
      modelId: "Qwen/Qwen3-0.6B",
      requestedBackendPreference: "webgpu",
      runtimeProfile: {
        profile: "full",
        caps: {
          maxRuntimePromptTokens: null,
          maxRuntimeLayers: null,
          maxGenerationTokens: null,
          logitCandidateLimit: null,
        },
        capStatus: {
          prompt: false,
          layers: false,
          generation: false,
          logits: false,
        },
      },
      memoryMode: "browser-local",
      backendProofs: {
        tensorControl: true,
        tspSteps: ["kv_prefetch", "attention", "mlp"],
        kvPagingEvents: 1,
      },
      webGpuCoverage: emptyWebGpuCoverage(),
      webGpuGates: ["mlp", "logits", "attention", "projection"],
      runs: [run],
      thresholds: [],
      mtpAcceleration: {
        mode: "skipped",
        requested: false,
        passed: true,
        reason: "not_requested",
      },
      browserPreview: {
        mode: "skipped",
        requested: false,
        reason: "not_requested",
      },
    });

    expect(artifact.webGpuCoverage.prefillMlpLayers).toEqual([
      { layerIndex: 0, backend: "webgpu", activationKind: "silu_gated" },
    ]);
    expect(artifact.webGpuCoverage.decodeMlpLayers).toEqual([
      { layerIndex: 0, backend: "webgpu", activationKind: "silu_gated" },
    ]);
    expect(artifact.passed).toBe(true);
    expect(artifact.webGpuGate).toMatchObject({
      required: true,
      gates: ["mlp", "logits", "attention", "projection"],
      passed: true,
      failedReasons: [],
    });
    expect(artifact.summary).toMatchObject({
      strictWebGpuRequired: true,
      strictWebGpuPassed: true,
      strictWebGpuGateCount: 4,
      strictWebGpuFailureCount: 0,
      blockingThresholdFailures: 0,
    });
  });

  it("keeps thresholds non-blocking unless strict benchmark mode is enabled", () => {
    const thresholds = parseBrowserRuntimeBenchmarkThresholds({
      BROWSER_RUNTIME_BENCH_MAX_TTFT_MS: "5",
    });

    expect(evaluateBrowserRuntimeBenchmarkThresholds([run], thresholds, false)).toEqual([
      { name: "maxTimeToFirstTokenMs", threshold: 5, observed: 7, passed: true, blocking: false },
    ]);
    expect(evaluateBrowserRuntimeBenchmarkThresholds([run], thresholds, true)).toEqual([
      { name: "maxTimeToFirstTokenMs", threshold: 5, observed: 7, passed: false, blocking: true },
    ]);
  });

  it("marks tokens/sec unavailable instead of inventing throughput when decode latency is zero", () => {
    expect(calculateTokensPerSecond(2, 0)).toBeNull();
    expect(calculateTokensPerSecond(2, 0.004)).toBe(500000);
  });

  it("uses total generation wall time when an accepted MTP batch streams all tokens in the first chunk", () => {
    expect(calculateBrowserRuntimeRunTiming({
      streamStart: 0,
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

  it("fails strict throughput thresholds when tokens/sec is unavailable", () => {
    const unavailableRun: BrowserRuntimeBenchmarkRun = {
      ...run,
      metrics: {
        ...run.metrics,
        decodeLatencyMs: 0,
        tokensPerSecond: null,
      },
    };
    const thresholds = parseBrowserRuntimeBenchmarkThresholds({
      BROWSER_RUNTIME_BENCH_MIN_TOKENS_PER_SEC: "1",
    });

    expect(evaluateBrowserRuntimeBenchmarkThresholds([unavailableRun], thresholds, true)).toEqual([
      { name: "minTokensPerSecond", threshold: 1, observed: null, passed: false, blocking: true },
    ]);
  });

  it("requires WebGPU execution when any strict WebGPU benchmark gate is active", () => {
    expect(shouldRequireWebGpuForBenchmark([])).toBe(false);
    expect(shouldRequireWebGpuForBenchmark(["logits"])).toBe(true);
    expect(shouldRequireWebGpuForBenchmark(["mlp", "attention", "projection"])).toBe(true);
  });

  it("treats the app WebGPU-only kernel setting as all strict benchmark gates", () => {
    expect(readBenchmarkWebGpuGates(new Set(), {
      VITE_REQUIRE_WEBGPU_KERNELS: "true",
    })).toEqual(["mlp", "logits", "attention", "projection"]);
    expect(readBrowserPreviewWebGpuGates(new Set(), {
      VITE_REQUIRE_WEBGPU_KERNELS: "true",
    }, [])).toEqual(["mlp", "logits", "attention", "projection"]);
  });

  it("can require all WebGPU gates only for the browser-preview proof", () => {
    expect(readBrowserPreviewWebGpuGates(new Set(), {}, ["logits"])).toEqual(["logits"]);
    expect(readBrowserPreviewWebGpuGates(new Set(), {
      BROWSER_RUNTIME_BENCH_PREVIEW_REQUIRE_STRICT_WEBGPU: "true",
    }, [])).toEqual(["mlp", "logits", "attention", "projection"]);
  });

  it("builds a failed artifact when strict WebGPU aborts before decode proof", () => {
    const artifact = buildBrowserRuntimeBenchmarkFailureArtifact({
      createdAt: "2026-05-17T00:00:00.000Z",
      mode: "configured",
      manifestUrl: "file:///tmp/manifest.json",
      modelId: "Qwen/Qwen3-0.6B",
      requestedBackendPreference: "webgpu",
      runtimeProfile: {
        profile: "full",
        caps: {
          maxRuntimePromptTokens: null,
          maxRuntimeLayers: null,
          maxGenerationTokens: null,
          logitCandidateLimit: null,
        },
        capStatus: {
          prompt: false,
          layers: false,
          generation: false,
          logits: false,
        },
      },
      memoryMode: "browser-local",
      thresholds: [],
      webGpuGates: ["mlp", "logits", "attention", "projection"],
      error: new Error("WebGPU is not available for dense matmul."),
      browserPreviewRequired: true,
      browserPreviewUrl: "",
    });

    expect(artifact.passed).toBe(false);
    expect(artifact.fatalError?.message).toBe("WebGPU is not available for dense matmul.");
    expect(artifact.webGpuCoverage.cpuFallbackUsed).toBe(false);
    expect(artifact.summary).toMatchObject({
      fatalError: true,
      fatalErrorMessage: "WebGPU is not available for dense matmul.",
      cpuFallbackUsed: false,
      strictWebGpuRequired: true,
      strictWebGpuPassed: false,
      promptCount: 0,
      browserPreviewMode: "failed",
      browserPreviewRequired: true,
      browserPreviewReason: "browser preview proof is required but no browser preview URL was provided.",
      blockingThresholdFailures: 3,
    });
  });

  it("keeps chunk-plan fields in fatal failure artifacts for timeout-safe long-prompt proof", () => {
    const error = Object.assign(
      new Error("Strict long-prompt proof requires chunked prefill dispatch, but chunked dispatch is not implemented yet."),
      {
        prefillChunkCount: 3,
        prefillChunkSize: 1024,
        shapeBucket: "prompt<=1024:selected<=512:headDim<=128:tileRows<=8192:precision=f32",
        pipelineCacheKey: "prefill_chunk:prompt<=1024:selected<=512:headDim<=128:tileRows<=8192:precision=f32",
        maxDispatchEstimatedMs: 6.5,
        prefillChunkReason: "chunked_prefill_dispatch_not_implemented",
      },
    );

    const artifact = buildBrowserRuntimeBenchmarkFailureArtifact({
      createdAt: "2026-05-20T00:00:00.000Z",
      mode: "configured",
      manifestUrl: "file:///tmp/manifest.json",
      modelId: "Qwen/Qwen3-0.6B",
      requestedBackendPreference: "webgpu",
      runtimeProfile: {
        profile: "full",
        caps: {
          maxRuntimePromptTokens: null,
          maxRuntimeLayers: null,
          maxGenerationTokens: null,
          logitCandidateLimit: null,
        },
        capStatus: {
          prompt: false,
          layers: false,
          generation: false,
          logits: false,
        },
      },
      memoryMode: "browser-local",
      thresholds: [],
      webGpuGates: [],
      error,
      browserPreviewRequired: true,
      browserPreviewUrl: "http://127.0.0.1:5173/__bench/browser-runtime",
    });

    expect(artifact.passed).toBe(false);
    expect(artifact.browserPreview).toMatchObject({
      mode: "failed",
      reason: expect.stringContaining("benchmark aborted before browser proof completed"),
    });
    expect(artifact.summary).toMatchObject({
      fatalError: true,
      prefillChunkCount: 3,
      prefillChunkSize: 1024,
      shapeBucket: "prompt<=1024:selected<=512:headDim<=128:tileRows<=8192:precision=f32",
      pipelineCacheKey: "prefill_chunk:prompt<=1024:selected<=512:headDim<=128:tileRows<=8192:precision=f32",
      maxDispatchEstimatedMs: 6.5,
      prefillChunkReason: "chunked_prefill_dispatch_not_implemented",
    });
  });

  it("can include a requested browser-preview benchmark snapshot from the preview URL contract", async () => {
    const originalFetch = globalThis.fetch;
    globalThis.fetch = async (input: RequestInfo | URL): Promise<Response> => {
      const url = new URL(String(input));
      expect(url.searchParams.get("profile")).toBe("ci");
      expect(url.searchParams.getAll("prompt")).toEqual([]);
      expect(url.searchParams.has("prompts")).toBe(false);
      expect(url.searchParams.get("webGpuGates")).toBe("logits");
      expect(url.searchParams.get("generationTokens")).toBe("16");
      expect(url.searchParams.get("timeoutMs")).toBe("45000");
      expect(url.searchParams.get("minGeneratedTokens")).toBe("8");
      expect(url.searchParams.get("requireKvReuse")).toBe("true");
      expect(url.searchParams.get("requireKvPredictivePrefetch")).toBe("true");
      expect(url.searchParams.get("memoryGrounding")).toBe("qa_corpus_v1");
      expect(url.searchParams.get("memoryCorpusSize")).toBe("64");
      expect(url.searchParams.get("memoryPromptLimit")).toBe("2");
      expect(url.searchParams.get("strictLongPrompt")).toBe("true");
      expect(url.searchParams.getAll("expectedJson")).toEqual([]);
      expect(url.searchParams.getAll("expectedExactJson")).toEqual([]);
      expect(url.searchParams.has("expected")).toBe(false);
      expect(url.searchParams.has("expectedSubstrings")).toBe(false);
      expect(url.searchParams.get("qwenThinkingMode")).toBe("enabled");
      return new Response(JSON.stringify({
        passed: true,
        summary: {
          meanInitLoadMs: 20,
          meanPrefillMs: 8,
          meanTimeToFirstTokenMs: 11,
          meanDecodeLatencyMs: 6,
          meanTokensPerSecond: 166,
          mtpMode: "draft_verify",
          prefillChunkCount: 2,
          prefillChunkSize: 1024,
          shapeBucket: "prompt<=1024:selected<=512:headDim<=128:tileRows<=8192:precision=f32",
          pipelineCacheKey: "prefill_chunk:prompt<=1024:selected<=512:headDim<=128:tileRows<=8192:precision=f32",
          maxDispatchEstimatedMs: 7.5,
          ...browserPreviewMtpSummary,
          ...browserPreviewKvSummary,
        },
        runs: [{
          ...run,
          prefillChunkCount: 2,
          prefillChunkSize: 1024,
          shapeBucket: "prompt<=1024:selected<=512:headDim<=128:tileRows<=8192:precision=f32",
          pipelineCacheKey: "prefill_chunk:prompt<=1024:selected<=512:headDim<=128:tileRows<=8192:precision=f32",
          maxDispatchEstimatedMs: 7.5,
        }],
      }), { status: 200, headers: { "content-type": "application/json" } });
    };
    try {
      await expect(readBrowserPreviewBenchmark({
        url: "http://localhost:5173/__bench/browser-runtime",
        runtimeProfile: "ci",
        prompts: [{ id: "fixture-short", text: "alpha beta", expectedSubstrings: ["Salt Lake"] }],
        webGpuGates: ["logits"],
        generationTokenBudget: 16,
        timeoutMs: 45_000,
        minGeneratedTokens: 8,
        requireKvReuse: true,
        requireKvPredictivePrefetch: true,
        memoryGroundingCase: "qa_corpus_v1",
        memoryGroundingCorpusSize: 64,
        memoryGroundingPromptLimit: 2,
        qwenThinkingMode: "enabled",
        strictLongPromptProof: true,
      } as Parameters<typeof readBrowserPreviewBenchmark>[0] & {
        strictLongPromptProof: boolean;
      })).resolves.toMatchObject({
        mode: "completed",
        requested: true,
        url: "http://localhost:5173/__bench/browser-runtime",
        passed: true,
        summary: {
          meanInitLoadMs: 20,
          meanPrefillMs: 8,
          meanTimeToFirstTokenMs: 11,
          meanDecodeLatencyMs: 6,
          meanTokensPerSecond: 166,
          mtpMode: "draft_verify",
          prefillChunkCount: 2,
          prefillChunkSize: 1024,
          shapeBucket: "prompt<=1024:selected<=512:headDim<=128:tileRows<=8192:precision=f32",
          pipelineCacheKey: "prefill_chunk:prompt<=1024:selected<=512:headDim<=128:tileRows<=8192:precision=f32",
          maxDispatchEstimatedMs: 7.5,
        },
      });
    } finally {
      globalThis.fetch = originalFetch;
    }
  });

  it("forwards exact-output expectations to browser-preview snapshots", async () => {
    const originalFetch = globalThis.fetch;
    globalThis.fetch = async (input: RequestInfo | URL): Promise<Response> => {
      const url = new URL(String(input));
      expect(url.searchParams.getAll("prompt")).toEqual(["Return exactly: edge-runtime-ok"]);
      expect(url.searchParams.getAll("expectedJson")).toEqual([]);
      expect(url.searchParams.getAll("expectedExactJson")).toEqual([
        JSON.stringify(["edge-runtime-ok"]),
      ]);
      return new Response(JSON.stringify({
        passed: true,
        summary: {
          meanInitLoadMs: 20,
          meanPrefillMs: 8,
          meanTimeToFirstTokenMs: 11,
          meanDecodeLatencyMs: 6,
          meanTokensPerSecond: 166,
          mtpMode: "target_only",
          technicalProofOnly: false,
          productionQualityPassed: true,
          productionDeployReadyPassed: true,
          expectedExactCheckCount: 1,
          expectedExactPassCount: 1,
          expectedExactPassed: true,
          ...browserPreviewMtpSummary,
          ...browserPreviewKvSummary,
        },
        runs: [{
          ...run,
          prompt: "Return exactly: edge-runtime-ok",
          response: "[unlocked:ssa-kv-tsp]edge-runtime-ok",
          expectedExact: ["edge-runtime-ok"],
          expectedExactMatches: [{ expected: "edge-runtime-ok", matched: true }],
        }],
      }), { status: 200, headers: { "content-type": "application/json" } });
    };
    try {
      await expect(readBrowserPreviewBenchmark({
        url: "http://localhost:5173/__bench/browser-runtime",
        runtimeProfile: "ci",
        prompts: [{ id: "exact-literal", text: "Return exactly: edge-runtime-ok", expectedExact: ["edge-runtime-ok"] }],
      })).resolves.toMatchObject({
        mode: "completed",
        requested: true,
        passed: true,
        summary: {
          expectedExactPassed: true,
        },
      });
    } finally {
      globalThis.fetch = originalFetch;
    }
  });

  it("forwards browser-preview long-prompt controls compactly instead of expanding prompt query text", async () => {
    const originalFetch = globalThis.fetch;
    globalThis.fetch = async (input: RequestInfo | URL): Promise<Response> => {
      const url = new URL(String(input));
      expect(url.searchParams.get("longPromptTargetTokens")).toBe("4096");
      expect(url.searchParams.get("longPromptRepeat")).toBe("4");
      expect(url.searchParams.getAll("prompt")).toEqual([]);
      expect(url.searchParams.get("promptSeed")).toBe("source seed");
      expect(url.searchParams.get("strictLongPrompt")).toBe("true");
      return new Response(JSON.stringify({
        passed: false,
        summary: {
          meanInitLoadMs: 20,
          meanPrefillMs: 8,
          meanTimeToFirstTokenMs: 11,
          meanDecodeLatencyMs: 6,
          meanTokensPerSecond: 166,
          mtpMode: "draft_verify",
          prefillChunkCount: 4,
          prefillChunkSize: 1024,
          shapeBucket: "prompt<=1024:selected<=512:headDim<=128:tileRows<=8192:precision=f32",
          pipelineCacheKey: "prefill_chunk:prompt<=1024:selected<=512:headDim<=128:tileRows<=8192:precision=f32",
          maxDispatchEstimatedMs: 7.5,
          ...browserPreviewMtpSummary,
          ...browserPreviewKvSummary,
          prefillChunkReason: "chunked_prefill_dispatch_not_implemented",
        },
        runs: [run],
      }), { status: 200, headers: { "content-type": "application/json" } });
    };
    try {
      await expect(readBrowserPreviewBenchmark({
        url: "http://localhost:5173/__bench/browser-runtime",
        runtimeProfile: "full",
        prompts: [{ id: "long-prompt-1", text: "expanded text should not be forwarded" }],
        strictLongPromptProof: true,
        longPromptTargetTokens: 4096,
        longPromptRepeat: 4,
        longPromptSeed: "source seed",
      })).resolves.toMatchObject({
        mode: "completed",
        passed: false,
        summary: {
          prefillChunkReason: "chunked_prefill_dispatch_not_implemented",
        },
      });
    } finally {
      globalThis.fetch = originalFetch;
    }
  });

  it("does not infer a compact browser-preview seed from already-expanded long prompt text", async () => {
    const originalFetch = globalThis.fetch;
    globalThis.fetch = async (input: RequestInfo | URL): Promise<Response> => {
      const url = new URL(String(input));
      expect(url.searchParams.get("longPromptRepeat")).toBe("4");
      expect(url.searchParams.getAll("prompt")).toEqual([]);
      expect(url.searchParams.has("promptSeed")).toBe(false);
      return new Response(JSON.stringify({
        passed: true,
        summary: {
          meanInitLoadMs: 20,
          meanPrefillMs: 8,
          meanTimeToFirstTokenMs: 11,
          meanDecodeLatencyMs: 6,
          meanTokensPerSecond: 166,
          mtpMode: "draft_verify",
          ...browserPreviewMtpSummary,
          ...browserPreviewKvSummary,
        },
        runs: [run],
      }), { status: 200, headers: { "content-type": "application/json" } });
    };
    try {
      await expect(readBrowserPreviewBenchmark({
        url: "http://localhost:5173/__bench/browser-runtime",
        runtimeProfile: "full",
        prompts: [{ id: "long-prompt-1", text: "alpha beta gamma delta alpha beta gamma delta" }],
        longPromptRepeat: 4,
      })).resolves.toMatchObject({ mode: "completed", passed: true });
    } finally {
      globalThis.fetch = originalFetch;
    }
  });

  it("forwards browser-preview prompts and expected substrings without delimiter coupling", async () => {
    const originalFetch = globalThis.fetch;
    globalThis.fetch = async (input: RequestInfo | URL): Promise<Response> => {
      const url = new URL(String(input));
      expect(url.searchParams.getAll("prompt")).toEqual([
        "Compare A | B literally.",
        "Name the Utah capital.",
      ]);
      expect(url.searchParams.getAll("expectedJson")).toEqual([
        JSON.stringify(["A | B", "foo, bar"]),
        JSON.stringify(["Salt Lake, Utah"]),
      ]);
      expect(url.searchParams.has("expected")).toBe(false);
      expect(url.searchParams.has("prompts")).toBe(false);
      expect(url.searchParams.has("expectedSubstrings")).toBe(false);
      return new Response(JSON.stringify({
        passed: true,
        summary: {
          meanInitLoadMs: 20,
          meanPrefillMs: 8,
          meanTimeToFirstTokenMs: 11,
          meanDecodeLatencyMs: 6,
          meanTokensPerSecond: 166,
          mtpMode: "draft_verify",
          ...browserPreviewMtpSummary,
          ...browserPreviewKvSummary,
        },
        runs: [run],
      }), { status: 200, headers: { "content-type": "application/json" } });
    };
    try {
      await expect(readBrowserPreviewBenchmark({
        url: "http://localhost:5173/__bench/browser-runtime",
        runtimeProfile: "ci",
        prompts: [
          { id: "pipe", text: "Compare A | B literally.", expectedSubstrings: ["A | B", "foo, bar"] },
          { id: "comma", text: "Name the Utah capital.", expectedSubstrings: ["Salt Lake, Utah"] },
        ],
      })).resolves.toMatchObject({
        mode: "completed",
        requested: true,
        passed: true,
      });
    } finally {
      globalThis.fetch = originalFetch;
    }
  });

  it("can read a browser-rendered preview payload from an HTML script tag", async () => {
    const originalFetch = globalThis.fetch;
    globalThis.fetch = async (): Promise<Response> => new Response(`
      <!doctype html>
      <html>
        <body>
          <script
            id="browser-preview-benchmark-payload"
            type="application/json"
          >${JSON.stringify({
            passed: true,
            summary: {
              meanInitLoadMs: 20,
              meanPrefillMs: 8,
              meanTimeToFirstTokenMs: 11,
              meanDecodeLatencyMs: 6,
              meanTokensPerSecond: 166,
              mtpMode: "draft_verify",
              ...browserPreviewMtpSummary,
          ...browserPreviewKvSummary,
            },
            runs: [run],
          })}</script>
        </body>
      </html>
    `, { status: 200, headers: { "content-type": "text/html" } });
    try {
      await expect(readBrowserPreviewBenchmark({
        url: "http://localhost:5173/__bench/browser-runtime",
        runtimeProfile: "ci",
        prompts: [{ id: "fixture-short", text: "alpha beta" }],
      })).resolves.toMatchObject({
        mode: "completed",
        requested: true,
        passed: true,
        summary: {
          meanTokensPerSecond: 166,
          mtpMode: "draft_verify",
        },
      });
    } finally {
      globalThis.fetch = originalFetch;
    }
  });

  it("uses a browser-executed reader when the preview URL returns the SPA shell", async () => {
    const originalFetch = globalThis.fetch;
    let fetchedUrl = "";
    let browserExecutedUrl = "";
    globalThis.fetch = async (input: RequestInfo | URL): Promise<Response> => {
      fetchedUrl = String(input);
      return new Response(`
        <!doctype html>
        <html>
          <body>
            <div id="root"></div>
            <script type="module" src="/src/main.tsx"></script>
          </body>
        </html>
      `, { status: 200, headers: { "content-type": "text/html" } });
    };
    try {
      await expect(readBrowserPreviewBenchmark({
        url: "http://localhost:5173/",
        runtimeProfile: "ci",
        prompts: [{ id: "fixture-short", text: "alpha beta" }],
        browserPayloadReader: async ({ url }) => {
          browserExecutedUrl = url;
          return {
            passed: true,
            summary: {
              meanInitLoadMs: 20,
              meanPrefillMs: 8,
              meanTimeToFirstTokenMs: 11,
              meanDecodeLatencyMs: 6,
              meanTokensPerSecond: 166,
              mtpMode: "draft_verify",
              ...browserPreviewMtpSummary,
          ...browserPreviewKvSummary,
            },
            runs: [run],
          };
        },
      } as Parameters<typeof readBrowserPreviewBenchmark>[0] & {
        browserPayloadReader: (input: { url: string }) => Promise<{
          passed: boolean;
          summary: Record<string, number | string | boolean | null>;
          runs: BrowserRuntimeBenchmarkRun[];
        }>;
      })).resolves.toMatchObject({
        mode: "completed",
        requested: true,
        passed: true,
      });
      expect(new URL(fetchedUrl).pathname).toBe("/__bench/browser-runtime");
      expect(new URL(browserExecutedUrl).pathname).toBe("/__bench/browser-runtime");
      expect(new URL(browserExecutedUrl).searchParams.getAll("prompt")).toEqual(["alpha beta"]);
    } finally {
      globalThis.fetch = originalFetch;
    }
  });

  it("accepts browser-preview benchmark summaries with unavailable throughput", async () => {
    const originalFetch = globalThis.fetch;
    globalThis.fetch = async (): Promise<Response> => new Response(JSON.stringify({
      passed: false,
      summary: {
        meanInitLoadMs: 20,
        meanPrefillMs: 8,
        meanTimeToFirstTokenMs: 11,
        meanDecodeLatencyMs: 0,
        meanTokensPerSecond: null,
        mtpMode: "draft_verify",
        ...browserPreviewMtpSummary,
          ...browserPreviewKvSummary,
      },
      runs: [{
        ...run,
        metrics: {
          ...run.metrics,
          decodeLatencyMs: 0,
          tokensPerSecond: null,
        },
      }],
    }), { status: 200, headers: { "content-type": "application/json" } });
    try {
      await expect(readBrowserPreviewBenchmark({
        url: "http://localhost:5173/__bench/browser-runtime",
        runtimeProfile: "ci",
        prompts: [{ id: "fixture-short", text: "alpha beta" }],
      })).resolves.toMatchObject({
        mode: "completed",
        requested: true,
        passed: false,
        summary: {
          meanTokensPerSecond: null,
          mtpMode: "draft_verify",
        },
      });
    } finally {
      globalThis.fetch = originalFetch;
    }
  });

  it("accepts browser-preview failure payloads with null timing summaries", async () => {
    const originalFetch = globalThis.fetch;
    globalThis.fetch = async (): Promise<Response> => new Response(JSON.stringify({
      passed: false,
      summary: {
        meanInitLoadMs: null,
        meanPrefillMs: null,
        meanTimeToFirstTokenMs: null,
        meanDecodeLatencyMs: null,
        meanTokensPerSecond: null,
        mtpMode: "none",
        technicalProofOnly: false,
        productionQualityPassed: false,
        productionDeployReadyPassed: false,
        error: "WebGPU unavailable",
      },
      runs: [],
    }), { status: 200, headers: { "content-type": "application/json" } });
    try {
      await expect(readBrowserPreviewBenchmark({
        url: "http://localhost:5173/__bench/browser-runtime",
        runtimeProfile: "ci",
        prompts: [{ id: "fixture-short", text: "alpha beta" }],
      })).resolves.toMatchObject({
        mode: "completed",
        requested: true,
        passed: false,
        summary: {
          meanInitLoadMs: null,
          mtpMode: "none",
          error: "WebGPU unavailable",
        },
      });
    } finally {
      globalThis.fetch = originalFetch;
    }
  });

  it("fails requested browser-preview snapshots that are missing required fields", async () => {
    const originalFetch = globalThis.fetch;
    globalThis.fetch = async (): Promise<Response> => new Response(JSON.stringify({
      summary: { meanTimeToFirstTokenMs: 11 },
      runs: [],
    }), { status: 200, headers: { "content-type": "application/json" } });
    try {
      await expect(readBrowserPreviewBenchmark({
        url: "http://localhost:5173/__bench/browser-runtime",
        runtimeProfile: "ci",
        prompts: [{ id: "fixture-short", text: "alpha beta" }],
      })).resolves.toMatchObject({
        mode: "failed",
        requested: true,
        passed: false,
        reason: expect.stringContaining("boolean passed"),
      });
    } finally {
      globalThis.fetch = originalFetch;
    }
  });

  it("fails requested browser-preview snapshots that claim pass without any run proof", async () => {
    const originalFetch = globalThis.fetch;
    globalThis.fetch = async (): Promise<Response> => new Response(JSON.stringify({
      passed: true,
      summary: {
        meanInitLoadMs: 20,
        meanPrefillMs: 8,
        meanTimeToFirstTokenMs: 11,
        meanDecodeLatencyMs: 6,
        meanTokensPerSecond: 166,
        mtpMode: "draft_verify",
        technicalProofOnly: false,
        productionQualityPassed: false,
        productionDeployReadyPassed: false,
      },
      runs: [],
    }), { status: 200, headers: { "content-type": "application/json" } });
    try {
      await expect(readBrowserPreviewBenchmark({
        url: "http://localhost:5173/__bench/browser-runtime",
        runtimeProfile: "ci",
        prompts: [{ id: "fixture-short", text: "alpha beta" }],
      })).resolves.toMatchObject({
        mode: "failed",
        requested: true,
        passed: false,
        reason: expect.stringContaining("at least one run"),
      });
    } finally {
      globalThis.fetch = originalFetch;
    }
  });

  it("accepts browser-preview audit-only snapshots without generation runs", async () => {
    const originalFetch = globalThis.fetch;
    globalThis.fetch = async (): Promise<Response> => new Response(JSON.stringify({
      passed: true,
      summary: {
        meanInitLoadMs: 0,
        meanPrefillMs: 0,
        meanTimeToFirstTokenMs: 2069.35,
        meanDecodeLatencyMs: 0,
        meanTokensPerSecond: null,
        mtpMode: "target_only",
        technicalProofOnly: true,
        productionQualityPassed: false,
        productionDeployReadyPassed: false,
        memoryGroundingAuditOnly: true,
        memoryGroundingPassed: true,
        memorySeededCorpusCount: 1024,
        memoryRetrievalAuditPassed: true,
        memoryRetrievalAuditQueryCount: 64,
        memoryRetrievalAuditTop1CorrectCount: 64,
        memoryRecallAt1: 1,
        memoryGeneratedParaphraseRequired: true,
        memoryGeneratedParaphrasePassed: true,
        memoryGeneratedParaphraseQueryCount: 12,
        memoryGeneratedParaphraseTop1CorrectCount: 12,
        memoryGeneratedParaphraseRecallAt1: 1,
        memoryGeneratedParaphraseMrr: 1,
        ...browserPreviewMtpSummary,
        ...browserPreviewKvSummary,
      },
      runs: [],
    }), { status: 200, headers: { "content-type": "application/json" } });
    try {
      await expect(readBrowserPreviewBenchmark({
        url: "http://localhost:5173/__bench/browser-runtime?memoryGroundingAuditOnly=true",
        runtimeProfile: "ci",
        prompts: [{ id: "fixture-short", text: "alpha beta" }],
      })).resolves.toMatchObject({
        mode: "completed",
        requested: true,
        passed: true,
        summary: {
          memoryGroundingAuditOnly: true,
          memoryRetrievalAuditPassed: true,
          memoryRecallAt1: 1,
        },
        runs: [],
      });
    } finally {
      globalThis.fetch = originalFetch;
    }
  });

  it("rejects grounded browser-preview snapshots that omit generated-paraphrase proof fields", async () => {
    const originalFetch = globalThis.fetch;
    globalThis.fetch = async (): Promise<Response> => new Response(JSON.stringify({
      passed: true,
      summary: {
        meanInitLoadMs: 20,
        meanPrefillMs: 8,
        meanTimeToFirstTokenMs: 11,
        meanDecodeLatencyMs: 6,
        meanTokensPerSecond: 166,
        mtpMode: "target_only",
        technicalProofOnly: false,
        productionQualityPassed: true,
        productionDeployReadyPassed: false,
        memoryGroundingRequired: true,
        memoryGroundingPassed: true,
        ...browserPreviewMtpSummary,
        ...browserPreviewKvSummary,
      },
      runs: [run],
    }), { status: 200, headers: { "content-type": "application/json" } });
    try {
      await expect(readBrowserPreviewBenchmark({
        url: "http://localhost:5173/__bench/browser-runtime",
        runtimeProfile: "ci",
        prompts: [{ id: "fixture-short", text: "alpha beta" }],
      })).resolves.toMatchObject({
        mode: "failed",
        requested: true,
        passed: false,
        reason: expect.stringContaining("memoryGeneratedParaphraseRequired"),
      });
    } finally {
      globalThis.fetch = originalFetch;
    }
  });

  it("fails requested browser-preview snapshots with invalid metric field types", async () => {
    const originalFetch = globalThis.fetch;
    globalThis.fetch = async (): Promise<Response> => new Response(JSON.stringify({
      passed: true,
      summary: {
        meanInitLoadMs: "fast",
        meanPrefillMs: 8,
        meanTimeToFirstTokenMs: 11,
        meanDecodeLatencyMs: 6,
        meanTokensPerSecond: 166,
        mtpMode: "draft_verify",
      },
      runs: [run],
    }), { status: 200, headers: { "content-type": "application/json" } });
    try {
      await expect(readBrowserPreviewBenchmark({
        url: "http://localhost:5173/__bench/browser-runtime",
        runtimeProfile: "ci",
        prompts: [{ id: "fixture-short", text: "alpha beta" }],
      })).resolves.toMatchObject({
        mode: "failed",
        requested: true,
        passed: false,
        reason: expect.stringContaining("meanInitLoadMs"),
      });
    } finally {
      globalThis.fetch = originalFetch;
    }
  });

  it("fails requested browser-preview snapshots missing KV prefetch summary proof fields", async () => {
    const originalFetch = globalThis.fetch;
    globalThis.fetch = async (): Promise<Response> => new Response(JSON.stringify({
      passed: true,
      summary: {
        meanInitLoadMs: 20,
        meanPrefillMs: 8,
        meanTimeToFirstTokenMs: 11,
        meanDecodeLatencyMs: 6,
        meanTokensPerSecond: 166,
        mtpMode: "draft_verify",
        ...browserPreviewMtpSummary,
      },
      runs: [run],
    }), { status: 200, headers: { "content-type": "application/json" } });
    try {
      await expect(readBrowserPreviewBenchmark({
        url: "http://localhost:5173/__bench/browser-runtime",
        runtimeProfile: "ci",
        prompts: [{ id: "fixture-short", text: "alpha beta" }],
      })).resolves.toMatchObject({
        mode: "failed",
        requested: true,
        passed: false,
        reason: expect.stringContaining("kvPrefetchStrategy"),
      });
    } finally {
      globalThis.fetch = originalFetch;
    }
  });

  it("fails requested browser-preview snapshots missing MTP verifier summary proof fields", async () => {
    const originalFetch = globalThis.fetch;
    globalThis.fetch = async (): Promise<Response> => new Response(JSON.stringify({
      passed: true,
      summary: {
        meanInitLoadMs: 20,
        meanPrefillMs: 8,
        meanTimeToFirstTokenMs: 11,
        meanDecodeLatencyMs: 6,
        meanTokensPerSecond: 166,
        mtpMode: "draft_verify",
        ...browserPreviewKvSummary,
      },
      runs: [run],
    }), { status: 200, headers: { "content-type": "application/json" } });
    try {
      await expect(readBrowserPreviewBenchmark({
        url: "http://localhost:5173/__bench/browser-runtime",
        runtimeProfile: "ci",
        prompts: [{ id: "fixture-short", text: "alpha beta" }],
      })).resolves.toMatchObject({
        mode: "failed",
        requested: true,
        passed: false,
        reason: expect.stringContaining("mtpVerifierStrategy"),
      });
    } finally {
      globalThis.fetch = originalFetch;
    }
  });

  it("fails requested browser-preview snapshots missing per-run KV prefetch proof fields", async () => {
    const originalFetch = globalThis.fetch;
    const { kvPersistence: _kvPersistence, ...runWithoutKvPersistence } = run;
    globalThis.fetch = async (): Promise<Response> => new Response(JSON.stringify({
      passed: true,
      summary: {
        meanInitLoadMs: 20,
        meanPrefillMs: 8,
        meanTimeToFirstTokenMs: 11,
        meanDecodeLatencyMs: 6,
        meanTokensPerSecond: 166,
        mtpMode: "draft_verify",
        ...browserPreviewMtpSummary,
          ...browserPreviewKvSummary,
      },
      runs: [runWithoutKvPersistence],
    }), { status: 200, headers: { "content-type": "application/json" } });
    try {
      await expect(readBrowserPreviewBenchmark({
        url: "http://localhost:5173/__bench/browser-runtime",
        runtimeProfile: "ci",
        prompts: [{ id: "fixture-short", text: "alpha beta" }],
      })).resolves.toMatchObject({
        mode: "failed",
        requested: true,
        passed: false,
        reason: expect.stringContaining("runs[0].kvPersistence"),
      });
    } finally {
      globalThis.fetch = originalFetch;
    }
  });

  it("fails requested browser-preview snapshots missing per-run MTP verifier proof fields", async () => {
    const originalFetch = globalThis.fetch;
    const runWithoutMtpProof = {
      ...run,
      mtp: {
        mode: "draft_verify",
        acceptedTokens: 1,
        rejectedTokens: 0,
        acceptanceRate: 1,
      },
    };
    globalThis.fetch = async (): Promise<Response> => new Response(JSON.stringify({
      passed: true,
      summary: {
        meanInitLoadMs: 20,
        meanPrefillMs: 8,
        meanTimeToFirstTokenMs: 11,
        meanDecodeLatencyMs: 6,
        meanTokensPerSecond: 166,
        mtpMode: "draft_verify",
        ...browserPreviewMtpSummary,
        ...browserPreviewKvSummary,
      },
      runs: [runWithoutMtpProof],
    }), { status: 200, headers: { "content-type": "application/json" } });
    try {
      await expect(readBrowserPreviewBenchmark({
        url: "http://localhost:5173/__bench/browser-runtime",
        runtimeProfile: "ci",
        prompts: [{ id: "fixture-short", text: "alpha beta" }],
      })).resolves.toMatchObject({
        mode: "failed",
        requested: true,
        passed: false,
        reason: expect.stringContaining("runs[0].mtp.numSpeculativeTokens"),
      });
    } finally {
      globalThis.fetch = originalFetch;
    }
  });

  it("fails requested browser-preview snapshots with invalid run payloads", async () => {
    const originalFetch = globalThis.fetch;
    globalThis.fetch = async (): Promise<Response> => new Response(JSON.stringify({
      passed: true,
      summary: {
        meanInitLoadMs: 20,
        meanPrefillMs: 8,
        meanTimeToFirstTokenMs: 11,
        meanDecodeLatencyMs: 6,
        meanTokensPerSecond: 166,
        mtpMode: "draft_verify",
        ...browserPreviewMtpSummary,
          ...browserPreviewKvSummary,
      },
      runs: [{
        promptId: "bad",
        prompt: "alpha beta",
        response: "[unlocked:ssa-kv-tsp]",
        metrics: {
          initLoadMs: 12,
          prefillMs: 7,
          timeToFirstTokenMs: 7,
          decodeLatencyMs: "soon",
          tokensPerSecond: 200,
          generatedTokens: 1,
        },
        mtp: {
          mode: "draft_verify",
          acceptedTokens: 1,
          rejectedTokens: 0,
          acceptanceRate: 1,
        },
        kvPersistence: run.kvPersistence,
      }],
    }), { status: 200, headers: { "content-type": "application/json" } });
    try {
      await expect(readBrowserPreviewBenchmark({
        url: "http://localhost:5173/__bench/browser-runtime",
        runtimeProfile: "ci",
        prompts: [{ id: "fixture-short", text: "alpha beta" }],
      })).resolves.toMatchObject({
        mode: "failed",
        requested: true,
        passed: false,
        reason: expect.stringContaining("runs[0].metrics.decodeLatencyMs"),
      });
    } finally {
      globalThis.fetch = originalFetch;
    }
  });
});

function emptyWebGpuCoverage(): UnlockedWebGpuCoverageSummary {
  return {
    expectedLayerCount: 1,
    executedLayerCount: 1,
    cpuFallbackUsed: false,
    logitProjection: { backend: "webgpu" as const, purpose: "full_vocab_topk_logit_projection", selectedRows: 4, fullRows: 4 },
    mlpLayersByBackend: { webgpu: 2, cpu_reference: 0, mixed: 0, unknown: 0 },
    prefillMlpLayers: [{ layerIndex: 0, backend: "webgpu" as const, activationKind: "silu_gated" as const }],
    decodeMlpLayers: [{ layerIndex: 0, backend: "webgpu" as const, activationKind: "silu_gated" as const }],
    mlpLayers: [
      { layerIndex: 0, backend: "webgpu" as const, activationKind: "silu_gated" as const },
      { layerIndex: 0, backend: "webgpu" as const, activationKind: "silu_gated" as const },
    ],
    prefillProjectionBackends: {
      qProjection: { webgpu: 1, cpu_reference: 0, mixed: 0, unknown: 0 },
      kProjection: { webgpu: 1, cpu_reference: 0, mixed: 0, unknown: 0 },
      vProjection: { webgpu: 1, cpu_reference: 0, mixed: 0, unknown: 0 },
      oProjection: { webgpu: 1, cpu_reference: 0, mixed: 0, unknown: 0 },
      layers: [{ layerIndex: 0, qProjection: "webgpu" as const, kProjection: "webgpu" as const, vProjection: "webgpu" as const, oProjection: "webgpu" as const }],
    },
    decodeProjectionBackends: {
      qProjection: { webgpu: 1, cpu_reference: 0, mixed: 0, unknown: 0 },
      kProjection: { webgpu: 1, cpu_reference: 0, mixed: 0, unknown: 0 },
      vProjection: { webgpu: 1, cpu_reference: 0, mixed: 0, unknown: 0 },
      oProjection: { webgpu: 1, cpu_reference: 0, mixed: 0, unknown: 0 },
      layers: [{ layerIndex: 0, qProjection: "webgpu" as const, kProjection: "webgpu" as const, vProjection: "webgpu" as const, oProjection: "webgpu" as const, projectionKind: "matvec" as const, tokens: null, selectedRows: 4 }],
    },
    attentionBackends: {
      prefill: { webgpu: 1, cpu_reference: 0, mixed: 0, unknown: 0 },
      decode: { webgpu: 1, cpu_reference: 0, mixed: 0, unknown: 0 },
      packedHeads: { webgpu: 1, cpu_reference: 0, mixed: 0, unknown: 0 },
      prefillLayers: [{ layerIndex: 0, attentionBackend: "webgpu" as const, packedHeadBackends: ["webgpu" as const], packedHeadCount: 1, packedHeadComplete: true }],
      decodeLayers: [{ layerIndex: 0, attentionBackend: "webgpu" as const, packedHeadBackends: ["webgpu" as const], packedHeadCount: 1, packedHeadComplete: true }],
      incompletePackedHeadProofs: 0,
    },
  };
}
