import { readFileSync } from "node:fs";
import { afterEach, describe, expect, it, vi } from "vitest";
import {
  UnlockedBrowserTransformerClient,
  __unlockedBrowserTransformerClientTestHooks,
} from "./unlockedBrowserTransformerClient";
import {
  KV_SWAP_STORAGE_VERSION,
  MemoryKVSwapPersistence,
  type BrowserKVSwapPersistenceOptions,
  type KVSwapPersistenceStore,
  type KVSwapPersistenceTraceEvent,
  type SerializedKVSwapBlock,
} from "../runtime/kvSwapPersistence";

describe("UnlockedBrowserTransformerClient", () => {
  afterEach(() => {
    vi.restoreAllMocks();
    vi.unstubAllGlobals();
  });

  it("streams from the browser-owned transformer backend without opaque chat APIs", async () => {
    const client = new UnlockedBrowserTransformerClient({
      modelId: "Qwen/Qwen3-0.6B",
      allowFixtureWeights: true,
      backendPreference: "cpu",
      qwenThinkingMode: "enabled",
    });

    await client.init();
    const chunks: string[] = [];
    for await (const chunk of client.streamChat([{ role: "user", content: "test full control" }], { maxTokens: 4, includeProofMarker: true })) {
      chunks.push(chunk);
    }

    expect(client.backendId).toBe("unlocked-browser-transformer");
    expect(chunks.join("")).toContain("[unlocked:ssa-kv-tsp]");
    expect(client.lastDecodeProof).toMatchObject({
      tspSteps: ["kv_prefetch", "attention", "mlp"],
      kvPagingEvents: expect.any(Number),
      tensorControl: true,
      logitProjectionBackend: "cpu_reference",
      logitProjectionFullRows: 16,
      logitProjectionPurpose: "full_vocab_topk_logit_projection",
      logitProjectionSelectedRows: 16,
      prefillAttentionBackends: [
        expect.objectContaining({
          packedHeadBackends: ["cpu_reference"],
          packedHeadCount: 1,
        }),
      ],
      decodeAttentionBackends: [
        expect.objectContaining({
          packedHeadBackends: ["cpu_reference"],
          packedHeadCount: 1,
        }),
      ],
      decodeProjectionBackends: [
        expect.objectContaining({
          layerIndex: 0,
          qProjection: "cpu_reference",
          kProjection: "cpu_reference",
          vProjection: "cpu_reference",
          oProjection: "cpu_reference",
        }),
      ],
    });
  });

  it("streams user-facing assistant text by default while keeping tensor-control proof metadata", async () => {
    const client = new UnlockedBrowserTransformerClient({
      modelId: "Qwen/Qwen3-0.6B",
      allowFixtureWeights: true,
      backendPreference: "cpu",
      maxGenerationTokens: 2,
    });

    await client.init();
    const chunks: string[] = [];
    for await (const chunk of client.streamChat([{ role: "user", content: "test full control" }])) {
      chunks.push(chunk);
    }

    expect(chunks.join("")).not.toContain("[unlocked:ssa-kv-tsp]");
    expect(chunks.join("").trim().length).toBeGreaterThan(0);
    expect(client.lastDecodeProof).toMatchObject({
      tensorControl: true,
      tspSteps: ["kv_prefetch", "attention", "mlp"],
    });
    expect(client.lastGeneratedTokenTexts).toHaveLength(client.lastGeneratedTokenIds.length);
    expect(client.lastGeneratedTokenTexts.join("")).toBe(chunks.join(""));
  });

  it("accumulates decode performance proof across all target-only generated tokens", async () => {
    const client = new UnlockedBrowserTransformerClient({
      modelId: "Qwen/Qwen3-0.6B",
      allowFixtureWeights: true,
      backendPreference: "cpu",
      maxGenerationTokens: 3,
      mtp: {
        enabled: false,
      },
    });

    await client.init();
    await collectStream(client.streamChat([{ role: "user", content: "count every token" }], { maxTokens: 3 }));

    expect(client.lastDecodeProof?.decodePerf).toMatchObject({
      generatedTokenCount: 3,
      decodeCallCount: 3,
    });
  });

  it("can warm target model residency during init without counting it as an MTP run", async () => {
    const client = new UnlockedBrowserTransformerClient({
      modelId: "Qwen/Qwen3-0.6B",
      allowFixtureWeights: true,
      backendPreference: "cpu",
      maxGenerationTokens: 1,
      warmModelResidency: true,
      warmModelResidencyMode: "target_probe",
      mtp: {
        enabled: false,
      },
    });

    await client.init();

    expect(client.lastWarmupMs).toBeGreaterThanOrEqual(0);
    expect(client.lastWarmupProof).toMatchObject({
      tensorControl: true,
      mtp: {
        mode: "target_only",
      },
    });
    expect(client.lastWarmupProof?.prefillProjectionBackends?.length).toBeGreaterThan(0);
    expect(client.lastWarmupProof?.decodeProjectionBackends?.length).toBeGreaterThan(0);
    expect(client.lastWarmupProof?.logitProjectionSelectedRows).toBeGreaterThan(0);
  });

  it("defaults warm residency to pipeline preload without target probe readbacks", async () => {
    const client = new UnlockedBrowserTransformerClient({
      modelId: "Qwen/Qwen3-0.6B",
      allowFixtureWeights: true,
      backendPreference: "cpu",
      maxGenerationTokens: 1,
      warmModelResidency: true,
      mtp: {
        enabled: false,
      },
    });

    await client.init();

    expect(client.lastWarmupMode).toBe("pipeline_preload");
    expect(client.lastWarmupMs).toBeGreaterThanOrEqual(0);
    expect(client.lastWarmupProof).toMatchObject({
      tensorControl: true,
      warmupMode: "pipeline_preload",
      warmupBlockingMs: expect.any(Number),
      residentReadbackCount: 0,
      mtp: {
        mode: "target_only",
      },
    });
    expect(client.lastWarmupProof?.decodeProjectionBackends).toBeUndefined();
  });

  it("disposes the browser transformer client and rejects later generation attempts", async () => {
    const client = new UnlockedBrowserTransformerClient({
      modelId: "Qwen/Qwen3-0.6B",
      allowFixtureWeights: true,
      backendPreference: "cpu",
      maxGenerationTokens: 1,
    });

    await client.init();
    await client.dispose({ clearSharedBuffers: true });

    expect(client.lastPromptTokenIds).toEqual([]);
    expect(client.lastGeneratedTokenIds).toEqual([]);
    await expect(collectStream(client.streamChat([{ role: "user", content: "after dispose" }]))).rejects.toThrow(
      /init\(\) must complete/i,
    );
  });

  it("surfaces CPU fallback MLP kernel proof from fixture weights", async () => {
    const client = new UnlockedBrowserTransformerClient({
      modelId: "Qwen/Qwen3-0.6B",
      allowFixtureWeights: true,
      backendPreference: "cpu",
      maxGenerationTokens: 1,
    });

    await client.init();
    for await (const _chunk of client.streamChat([{ role: "user", content: "test mlp proof" }])) {
      // Drain the stream so lastDecodeProof is populated.
    }

    expect(client.lastDecodeProof?.mlpKernelBackends).toEqual([
      expect.objectContaining({
        layerIndex: 0,
        backend: "cpu_reference",
        activationKind: "gelu",
      }),
    ]);
    expect(client.lastDecodeProof?.prefillMlpKernelBackends).toEqual([
      expect.objectContaining({
        layerIndex: 0,
        backend: "cpu_reference",
        activationKind: "gelu",
        rowCount: expect.any(Number),
      }),
    ]);
  });

  it("surfaces prefill shape-bucket planning metadata in decode proof", async () => {
    const client = new UnlockedBrowserTransformerClient({
      modelId: "Qwen/Qwen3-0.6B",
      allowFixtureWeights: true,
      backendPreference: "cpu",
      maxGenerationTokens: 1,
    });

    await client.init();
    await collectStream(client.streamChat([{ role: "user", content: "test prefill planning proof" }]));

    expect(client.lastDecodeProof).toMatchObject({
      prefillChunkCount: expect.any(Number),
      prefillChunkSize: expect.any(Number),
      shapeBucket: expect.stringContaining("prompt<="),
      pipelineCacheKey: expect.stringContaining("prefill_chunk:"),
      maxDispatchEstimatedMs: expect.any(Number),
    });
    expect(client.lastDecodeProof?.prefillChunkCount).toBeGreaterThanOrEqual(1);
    expect(client.lastDecodeProof?.prefillChunkSize).toBeGreaterThanOrEqual(1);
  });

  it("executes strict long-prompt proof with chunked prefill dispatch instead of planning-only failure", async () => {
    vi.stubGlobal("fetch", async () => new Response(JSON.stringify(makeManifest({
      tokenizer: {
        kind: "qwen-bpe",
        tokens: ["<unk>", "a", "b", "c"],
        merges: [],
        unknownTokenId: 0,
      },
      vocabSize: 4,
    })), {
      status: 200,
      headers: { "content-type": "application/json" },
    }));
    const client = new UnlockedBrowserTransformerClient({
      modelId: "Qwen/Qwen3-0.6B",
      manifestPath: "/models/qwen3-unlocked/manifest.json",
      allowFixtureWeights: false,
      backendPreference: "cpu",
      maxGenerationTokens: 1,
      strictChunkedPrefill: true,
    } as ConstructorParameters<typeof UnlockedBrowserTransformerClient>[0] & {
      strictChunkedPrefill: boolean;
    });

    await client.init();
    const prompt = "abc".repeat(800);

    await expect(collectStream(client.streamChat([{ role: "user", content: prompt }]))).resolves.toBeTypeOf("string");
    expect(client.lastDecodeProof).toMatchObject({
      prefillChunkCount: expect.any(Number),
      prefillChunkSize: expect.any(Number),
      shapeBucket: expect.stringContaining("prompt<="),
      pipelineCacheKey: expect.stringContaining("prefill_chunk:"),
      maxDispatchEstimatedMs: expect.any(Number),
      prefillChunkDispatch: "chunked_dispatch",
    });
    expect(client.lastDecodeProof?.prefillChunkCount).toBeGreaterThan(1);
    expect(client.lastDecodeProof).not.toHaveProperty("prefillChunkReason");
  }, 10_000);

  it("applies interactive runtime budgets without disabling the tensor-control decode proof", async () => {
    const client = new UnlockedBrowserTransformerClient({
      modelId: "Qwen/Qwen3-0.6B",
      allowFixtureWeights: true,
      backendPreference: "cpu",
      maxRuntimePromptTokens: 2,
      maxRuntimeLayers: 1,
      logitCandidateLimit: 3,
      maxGenerationTokens: 1,
    });

    await client.init();
    const chunks: string[] = [];
    for await (const chunk of client.streamChat([{ role: "user", content: "alpha beta gamma delta" }])) {
      chunks.push(chunk);
    }

    expect(chunks.join("").trim().length).toBeGreaterThan(0);
    expect(chunks).toHaveLength(1);
    expect(client.lastDecodeProof).toMatchObject({
      tensorControl: true,
      tspSteps: ["kv_prefetch", "attention", "mlp"],
      logitProjectionBackend: "cpu_reference",
      logitProjectionPurpose: "candidate_logit_projection",
    });
  });

  it("does not impose hidden default or 32-token generation caps", async () => {
    const client = new UnlockedBrowserTransformerClient({
      modelId: "Qwen/Qwen3-0.6B",
      allowFixtureWeights: true,
      backendPreference: "cpu",
    });

    await client.init();
    await expect(collectStream(client.streamChat([{ role: "user", content: "needs explicit generation budget" }]))).rejects.toThrow(
      /requires an explicit maxTokens or maxGenerationTokens/i,
    );

    const chunks: string[] = [];
    for await (const chunk of client.streamChat([{ role: "user", content: "more than old clamp" }], { maxTokens: 33 })) {
      chunks.push(chunk);
    }

    expect(chunks).toHaveLength(33);
  });

  it("keeps the full chat prompt visible during decode for normal interactive prompts", () => {
    const policy = __unlockedBrowserTransformerClientTestHooks.buildClientPolicy(640, 0);
    const latestQueryBlock = Math.ceil(640 / policy.blockSize) - 1;
    const explicitQueryBlocks = Object.keys(policy.selectedBlockIdsByQueryBlock).map(Number);

    expect(policy.blockSize).toBe(16);
    expect(policy.topKBlocks).toBe(latestQueryBlock + 1);
    expect(explicitQueryBlocks).toContain(latestQueryBlock);
    expect(explicitQueryBlocks).toContain(latestQueryBlock - 1);
    expect(policy.selectedBlockIdsByQueryBlock[latestQueryBlock]).toEqual(
      Array.from({ length: latestQueryBlock + 1 }, (_value, index) => `b${index}`),
    );
  });

  it("uses larger SSA blocks to preserve full-context semantics while shrinking long-prompt selected blocks", () => {
    const policy = __unlockedBrowserTransformerClientTestHooks.buildClientPolicy(6000, 0);
    const latestQueryBlock = Math.ceil(6000 / policy.blockSize) - 1;
    const selected = policy.selectedBlockIdsByQueryBlock[latestQueryBlock] ?? [];

    expect(policy.blockSize).toBe(16);
    expect(policy.topKBlocks).toBe(375);
    expect(selected).toHaveLength(375);
    expect(selected[0]).toBe("b0");
    expect(selected.at(-1)).toBe(`b${latestQueryBlock}`);
    expect(selected).toEqual(Array.from({ length: latestQueryBlock + 1 }, (_value, index) => `b${index}`));
  });

  it("keeps full-context routing exactly through the Qwen context boundary", () => {
    const boundaryPolicy = __unlockedBrowserTransformerClientTestHooks.buildClientPolicy(40_960, 0);
    const beyondPolicy = __unlockedBrowserTransformerClientTestHooks.buildClientPolicy(40_961, 0);
    const boundaryLatest = Math.ceil(40_960 / boundaryPolicy.blockSize) - 1;
    const beyondLatest = Math.ceil(40_961 / beyondPolicy.blockSize) - 1;

    expect(boundaryPolicy.topKBlocks).toBe(boundaryLatest + 1);
    expect(boundaryPolicy.selectedBlockIdsByQueryBlock[boundaryLatest]).toHaveLength(boundaryLatest + 1);
    expect(boundaryPolicy.selectedBlockIdsByQueryBlock[boundaryLatest]?.at(-1)).toBe(`b${boundaryLatest}`);
    expect(beyondPolicy.topKBlocks).toBe(256);
    expect(beyondPolicy.selectedBlockIdsByQueryBlock[beyondLatest]).toHaveLength(256);
    expect(beyondPolicy.selectedBlockIdsByQueryBlock[beyondLatest]).toContain(`b${beyondLatest}`);
  });

  it("keeps trailing continuation query blocks routed for MTP verification", () => {
    const policy = __unlockedBrowserTransformerClientTestHooks.buildClientPolicy(28, 0);
    const latestQueryBlock = Math.ceil(28 / policy.blockSize) - 1;
    const previousContinuationBlock = latestQueryBlock - 1;

    expect(policy.selectedBlockIdsByQueryBlock[previousContinuationBlock]).toEqual(
      Array.from({ length: previousContinuationBlock + 1 }, (_value, index) => `b${index}`),
    );
    expect(policy.selectedBlockIdsByQueryBlock[latestQueryBlock]).toEqual(
      Array.from({ length: latestQueryBlock + 1 }, (_value, index) => `b${index}`),
    );
  });

  it("keeps the SSA sparse window bounded for very long prompts", () => {
    const tokenCount = 41_000;
    const policy = __unlockedBrowserTransformerClientTestHooks.buildClientPolicy(tokenCount, 0);
    const latestQueryBlock = Math.ceil(tokenCount / policy.blockSize) - 1;
    const explicitQueryBlocks = Object.keys(policy.selectedBlockIdsByQueryBlock).map(Number);

    expect(policy.blockSize).toBe(16);
    expect(policy.topKBlocks).toBe(256);
    expect(explicitQueryBlocks).toContain(latestQueryBlock);
    expect(explicitQueryBlocks).toContain(latestQueryBlock - 1);
    expect(policy.selectedBlockIdsByQueryBlock[latestQueryBlock]).toHaveLength(256);
    expect(policy.selectedBlockIdsByQueryBlock[latestQueryBlock]?.[0]).toBe("b0");
    expect(policy.selectedBlockIdsByQueryBlock[latestQueryBlock]).toContain(`b${latestQueryBlock}`);
    expect(policy.selectedBlockIdsByQueryBlock[latestQueryBlock]?.some((blockId) => /^b1\d\d\d$/.test(blockId))).toBe(true);
  });

  it("requires a manifest path when fixture weights are disabled", async () => {
    const client = new UnlockedBrowserTransformerClient({
      modelId: "Qwen/Qwen3-0.6B",
      allowFixtureWeights: false,
      backendPreference: "cpu",
    });

    await expect(client.init()).rejects.toThrow("VITE_UNLOCKED_MODEL_MANIFEST_PATH");
  });

  it("fails early when strict WebGPU proof is requested with a CPU backend preference", async () => {
    const client = new UnlockedBrowserTransformerClient({
      modelId: "Qwen/Qwen3-0.6B",
      allowFixtureWeights: true,
      backendPreference: "cpu",
      requireWebGpu: true,
      maxGenerationTokens: 1,
    });

    await client.init();

    await expect(collectStream(client.streamChat([{ role: "user", content: "strict webgpu" }]))).rejects.toThrow(
      /WebGPU is required/i,
    );
  });

  it("requires tokenizer metadata for non-fixture manifests", async () => {
    vi.stubGlobal("fetch", async () => new Response(JSON.stringify(makeManifest({ tokenizer: undefined })), {
      status: 200,
      headers: { "content-type": "application/json" },
    }));
    const client = new UnlockedBrowserTransformerClient({
      modelId: "Qwen/Qwen3-0.6B",
      manifestPath: "/models/qwen3-unlocked/manifest.json",
      allowFixtureWeights: false,
      backendPreference: "cpu",
    });

    await expect(client.init()).rejects.toThrow("tokenizer.tokens");
  });

  it("uses manifest tokenizer tokens instead of the fixture proof vocabulary", async () => {
    vi.stubGlobal("fetch", async () => new Response(JSON.stringify(makeManifest()), {
      status: 200,
      headers: { "content-type": "application/json" },
    }));
    const client = new UnlockedBrowserTransformerClient({
      modelId: "Qwen/Qwen3-0.6B",
      manifestPath: "/models/qwen3-unlocked/manifest.json",
      allowFixtureWeights: false,
      backendPreference: "cpu",
    });

    await client.init();
    const chunks: string[] = [];
    for await (const chunk of client.streamChat([{ role: "user", content: "alpha beta" }], { maxTokens: 2 })) {
      chunks.push(chunk);
    }

    expect(chunks.join("")).toContain("MANIFEST_TOKEN");
  });

  it("streams byte-level BPE UTF-8 across token boundaries without replacement characters", () => {
    const tokenizer = __unlockedBrowserTransformerClientTestHooks.normalizeTokenizer({
      kind: "qwen-bpe",
      tokens: ["<unk>", "Ã", "©"],
      unknownTokenId: 0,
      merges: [],
      specialTokens: [],
    }, 3);
    const streamDecoder = tokenizer.createStreamDecoder?.();

    expect(streamDecoder).toBeDefined();
    expect(streamDecoder?.decodeToken(1)).toBe("");
    expect(streamDecoder?.decodeToken(2)).toBe("é");
    expect(streamDecoder?.flush()).toBe("");
  });

  it("uses full-vocab top-k projection unless the caller supplies a resolved logit candidate cap", async () => {
    vi.stubGlobal("fetch", async () => new Response(JSON.stringify(makeManifest({ vocabSize: 1024 })), {
      status: 200,
      headers: { "content-type": "application/json" },
    }));
    const client = new UnlockedBrowserTransformerClient({
      modelId: "Qwen/Qwen3-0.6B",
      manifestPath: "/models/qwen3-unlocked/manifest.json",
      allowFixtureWeights: false,
      backendPreference: "cpu",
    });

    await client.init();
    for await (const _chunk of client.streamChat([{ role: "user", content: "alpha beta gamma" }], { maxTokens: 1 })) {
      // Drain the stream so lastDecodeProof is populated.
    }

    expect(client.lastDecodeProof).toMatchObject({
      logitProjectionPurpose: "full_vocab_topk_logit_projection",
      logitProjectionFullRows: 1024,
    });
    expect(client.lastDecodeProof?.logitProjectionSelectedRows).toBe(64);
  });

  it("suppresses a visible token after two consecutive repeats so generation can escape loops", async () => {
    vi.stubGlobal("fetch", async () => new Response(JSON.stringify(makeRepeatingManifest()), {
      status: 200,
      headers: { "content-type": "application/json" },
    }));
    const client = new UnlockedBrowserTransformerClient({
      modelId: "Qwen/Qwen3-0.6B",
      manifestPath: "/models/qwen3-unlocked/manifest.json",
      allowFixtureWeights: false,
      backendPreference: "cpu",
      logitTopK: 4,
      maxGenerationTokens: 3,
    });

    await client.init();
    const response = await collectStream(client.streamChat([{ role: "user", content: "repeat probe" }]));

    expect(response).toBe(" m m n");
    expect(client.lastGeneratedTokenTexts).toEqual([" m", " m", " n"]);
    expect(client.lastGeneratedTokenIds).toEqual([1, 1, 2]);
    expect(client.lastGenerationStopReason).toBe("max_tokens");
  });

  it("can stop immediately after an expected grounded answer instead of streaming answer-plus-junk", async () => {
    vi.stubGlobal("fetch", async () => new Response(JSON.stringify(makeManifest()), {
      status: 200,
      headers: { "content-type": "application/json" },
    }));
    const client = new UnlockedBrowserTransformerClient({
      modelId: "Qwen/Qwen3-0.6B",
      manifestPath: "/models/qwen3-unlocked/manifest.json",
      allowFixtureWeights: false,
      backendPreference: "cpu",
      logitTopK: 4,
      maxGenerationTokens: 4,
    });

    await client.init();
    const response = await collectStream(client.streamChat([{ role: "user", content: "answer only" }], {
      maxTokens: 4,
      stopAfterSequences: ["MANIFEST_TOKEN"],
    }));

    expect(response).toBe(" MANIFEST_TOKEN");
    expect(client.lastGeneratedTokenIds).toHaveLength(1);
    expect(client.lastGenerationStopReason).toBe("stop_after_sequence");
  });

  it("clears stale decode proof before yielding a new proof marker", async () => {
    const client = new UnlockedBrowserTransformerClient({
      modelId: "Qwen/Qwen3-0.6B",
      allowFixtureWeights: true,
      backendPreference: "cpu",
      maxGenerationTokens: 1,
    });

    await client.init();
    await collectStream(client.streamChat([{ role: "user", content: "first proof" }]));
    expect(client.lastDecodeProof).not.toBeNull();

    const iterator = client.streamChat([{ role: "user", content: "second proof" }], {
      includeProofMarker: true,
      maxTokens: 1,
    })[Symbol.asyncIterator]();
    const first = await iterator.next();

    expect(first).toMatchObject({ done: false, value: "[unlocked:ssa-kv-tsp]" });
    expect(client.lastDecodeProof).toBeNull();
    await expect(iterator.next()).resolves.toMatchObject({ done: false });
    await expect(iterator.next()).resolves.toMatchObject({ done: true });
  });

  it("records browser MTP verifier metadata and correction accounting", async () => {
    vi.stubGlobal("fetch", async () => new Response(JSON.stringify(makeManifest()), {
      status: 200,
      headers: { "content-type": "application/json" },
    }));
    const client = new UnlockedBrowserTransformerClient({
      modelId: "Qwen/Qwen3-0.6B",
      manifestPath: "/models/qwen3-unlocked/manifest.json",
      allowFixtureWeights: false,
      backendPreference: "cpu",
      mtp: {
        enabled: true,
        draftModelId: "browser/ngram-drafter",
        numSpeculativeTokens: 3,
        minAcceptanceRate: 0,
        disableWhenLatencyWorse: false,
      },
    });

    await client.init();
    const chunks: string[] = [];
    for await (const chunk of client.streamChat([{ role: "user", content: "alpha beta" }], { maxTokens: 1 })) {
      chunks.push(chunk);
    }

    expect(chunks.join("")).toContain("MANIFEST_TOKEN");
    expect(chunks).toHaveLength(1);
    expect(client.lastDecodeProof?.mtp).toMatchObject({
      mode: "draft_verify",
      draftModelId: "browser/ngram-drafter",
      draftSource: "local_tokenizer_ngram",
      verifierStrategy: "batched_continuation",
      verifierBackend: "unlocked-browser-transformer",
      acceptedTokens: 0,
      rejectedTokens: 1,
      correctedTokens: 1,
      numSpeculativeTokens: 1,
      verifiedTokenCount: 1,
      targetDecodeCalls: 1,
    });
    expect(client.lastDecodeProof?.mtp?.draftTokenIds).toHaveLength(1);
    expect(client.lastDecodeProof?.mtp?.acceptanceRate).toBe(0);
    expect(client.lastDecodeProof?.decodeAttentionBackends).toEqual([
      expect.objectContaining({
        layerIndex: 0,
        attentionBackend: "cpu_reference",
        packedHeadBackends: ["cpu_reference"],
        packedHeadCount: 1,
      }),
    ]);
  });

  it("does not run the local browser drafter when configured with an arbitrary draft model id", async () => {
    vi.stubGlobal("fetch", async () => new Response(JSON.stringify(makeManifest()), {
      status: 200,
      headers: { "content-type": "application/json" },
    }));
    const client = new UnlockedBrowserTransformerClient({
      modelId: "Qwen/Qwen3-0.6B",
      manifestPath: "/models/qwen3-unlocked/manifest.json",
      allowFixtureWeights: false,
      backendPreference: "cpu",
      mtp: {
        enabled: true,
        draftModelId: "some/neural-drafter",
        numSpeculativeTokens: 4,
      },
    });

    await client.init();
    const chunks: string[] = [];
    for await (const chunk of client.streamChat([{ role: "user", content: "MANIFEST_TOKEN" }], { maxTokens: 2 })) {
      chunks.push(chunk);
    }

    expect(chunks).toHaveLength(2);
    expect(client.lastDecodeProof?.mtp).toMatchObject({
      mode: "target_only",
      draftModelId: "some/neural-drafter",
      draftTokenIds: [],
      acceptedTokens: 0,
      rejectedTokens: 0,
      correctedTokens: 0,
      disabledReason: "unsupported_draft_model_id",
    });
    expect(client.lastDecodeProof?.mtp).not.toHaveProperty("draftSource");
  });

  it("accepts browser MTP drafts through target verification and stays within maxTokens", async () => {
    vi.stubGlobal("fetch", async () => new Response(JSON.stringify(makeManifest()), {
      status: 200,
      headers: { "content-type": "application/json" },
    }));
    const client = new UnlockedBrowserTransformerClient({
      modelId: "Qwen/Qwen3-0.6B",
      manifestPath: "/models/qwen3-unlocked/manifest.json",
      allowFixtureWeights: false,
      backendPreference: "cpu",
      mtp: {
        enabled: true,
        draftModelId: "browser/ngram-drafter",
        numSpeculativeTokens: 8,
        minAcceptanceRate: 0,
      },
    });

    await client.init();
    const chunks: string[] = [];
    for await (const chunk of client.streamChat([{ role: "user", content: "MANIFEST_TOKEN" }], { maxTokens: 2 })) {
      chunks.push(chunk);
    }

    expect(chunks).toHaveLength(2);
    expect(client.lastDecodeProof?.mtp).toMatchObject({
      mode: "draft_verify",
      verifierStrategy: "batched_continuation",
      acceptedTokens: 2,
      rejectedTokens: 0,
      correctedTokens: 0,
      numSpeculativeTokens: 2,
      verifiedTokenCount: 2,
      targetDecodeCalls: 1,
      acceptanceRate: 1,
      latencyDisablePolicy: "unsupported_without_target_baseline",
    });
    expect(client.lastDecodeProof?.mtp?.disabledReason).toBeUndefined();
    expect(client.lastDecodeProof?.mtp?.metrics?.disabledReason).toBeUndefined();
  });

  it("runs a tokenizer-compatible Qwen prefix drafter instead of the static n-gram draft source", async () => {
    vi.stubGlobal("fetch", async () => new Response(JSON.stringify(makeManifest()), {
      status: 200,
      headers: { "content-type": "application/json" },
    }));
    const client = new UnlockedBrowserTransformerClient({
      modelId: "Qwen/Qwen3-0.6B",
      manifestPath: "/models/qwen3-unlocked/manifest.json",
      allowFixtureWeights: false,
      backendPreference: "cpu",
      mtp: {
        enabled: true,
        draftModelId: "browser/qwen-prefix-drafter",
        numSpeculativeTokens: 2,
        minAcceptanceRate: 0,
        disableWhenLatencyWorse: false,
        draftLayerCount: 1,
      },
    });

    await client.init();
    const chunks: string[] = [];
    for await (const chunk of client.streamChat([{ role: "user", content: "alpha beta" }], { maxTokens: 2 })) {
      chunks.push(chunk);
    }

    expect(chunks).toHaveLength(2);
    expect(chunks.join("")).toContain("MANIFEST_TOKEN");
    expect(client.lastDecodeProof?.mtp).toMatchObject({
      mode: "draft_verify",
      draftModelId: "browser/qwen-prefix-drafter",
      draftSource: "qwen_prefix_draft",
      verifierStrategy: "batched_continuation",
      acceptedTokens: 2,
      rejectedTokens: 0,
      correctedTokens: 0,
      numSpeculativeTokens: 2,
      verifiedTokenCount: 2,
      targetDecodeCalls: 1,
      acceptanceRate: 1,
    });
    expect(client.lastDecodeProof?.mtp?.metrics?.draftLatencyMs).toBeGreaterThanOrEqual(0);
  });

  it("auto-disables one-token Qwen prefix drafting because it cannot beat target-only browser decode", async () => {
    vi.stubGlobal("fetch", async () => new Response(JSON.stringify(makeManifest()), {
      status: 200,
      headers: { "content-type": "application/json" },
    }));
    const client = new UnlockedBrowserTransformerClient({
      modelId: "Qwen/Qwen3-0.6B",
      manifestPath: "/models/qwen3-unlocked/manifest.json",
      allowFixtureWeights: false,
      backendPreference: "cpu",
      mtp: {
        enabled: true,
        draftModelId: "browser/qwen-prefix-drafter",
        numSpeculativeTokens: 1,
        minAcceptanceRate: 0,
        disableWhenLatencyWorse: true,
        draftLayerCount: 1,
      },
    });

    await client.init();
    await collectStream(client.streamChat([{ role: "user", content: "alpha beta" }], { maxTokens: 2 }));

    expect(client.lastDecodeProof?.mtp).toMatchObject({
      mode: "target_only",
      draftModelId: "browser/qwen-prefix-drafter",
      draftTokenIds: [],
      acceptedTokens: 0,
      rejectedTokens: 0,
      latencyDisablePolicy: "paired_benchmark_required",
      disabledReason: "speculation_slower_than_target_only",
    });
  });

  it("preserves target-only fallback proof metadata when browser MTP is disabled", async () => {
    vi.stubGlobal("fetch", async () => new Response(JSON.stringify(makeManifest()), {
      status: 200,
      headers: { "content-type": "application/json" },
    }));
    const client = new UnlockedBrowserTransformerClient({
      modelId: "Qwen/Qwen3-0.6B",
      manifestPath: "/models/qwen3-unlocked/manifest.json",
      allowFixtureWeights: false,
      backendPreference: "cpu",
      mtp: {
        enabled: false,
        draftModelId: "browser/ngram-drafter",
        numSpeculativeTokens: 4,
      },
    });

    await client.init();
    const chunks: string[] = [];
    for await (const chunk of client.streamChat([{ role: "user", content: "MANIFEST_TOKEN" }], { maxTokens: 2 })) {
      chunks.push(chunk);
    }

    expect(chunks).toHaveLength(2);
    expect(client.lastDecodeProof?.mtp).toMatchObject({
      mode: "target_only",
      draftModelId: "browser/ngram-drafter",
      draftTokenIds: [],
      acceptedTokens: 0,
      rejectedTokens: 0,
      correctedTokens: 0,
      disabledReason: "mtp_disabled",
    });
  });

  it("reuses persisted KV rows for the same prompt and marks decode reuse in proof health", async () => {
    const client = new UnlockedBrowserTransformerClient({
      modelId: "Qwen/Qwen3-0.6B",
      allowFixtureWeights: true,
      backendPreference: "cpu",
      maxGenerationTokens: 1,
      kvPersistence: {
        enabled: true,
        namespace: "reuse-test",
        preferOpfs: false,
      },
    });

    await client.init();
    await collectStream(client.streamChat([{ role: "user", content: "reuse this prompt" }]));
    expect(client.lastDecodeProof?.kvPersistence?.decodeReuse).toBe(false);
    await client.flushKvPersistence();

    await collectStream(client.streamChat([{ role: "user", content: "reuse this prompt" }]));

    expect(client.lastDecodeProof?.kvPersistence).toMatchObject({
      enabled: true,
      mode: "memory",
      namespace: "reuse-test",
      decodeReuse: true,
    });
    expect(client.lastDecodeProof?.kvPersistence?.events).toEqual(expect.arrayContaining([
      expect.objectContaining({
        operation: "reuse",
        ok: true,
        reason: "decode_reuse_prefill_skipped",
      }),
    ]));
    expect(client.lastDecodeProof?.prefillMlpKernelBackends).toEqual([
      expect.objectContaining({ layerIndex: 0, backend: "cpu_reference" }),
    ]);
    expect(client.lastDecodeProof?.prefillProjectionBackends).toEqual([
      expect.objectContaining({
        layerIndex: 0,
        qProjection: "cpu_reference",
        kProjection: "cpu_reference",
        vProjection: "cpu_reference",
        oProjection: "cpu_reference",
      }),
    ]);
    expect(client.lastDecodeProof?.prefillAttentionBackends).toEqual([
      expect.objectContaining({
        layerIndex: 0,
        attentionBackend: "cpu_reference",
        packedHeadBackends: ["cpu_reference"],
        packedHeadCount: 1,
      }),
    ]);
  });

  it("restores compact GQA KV rows for exact reuse so strict decode appends matching resident rows", () => {
    const namespace = "compact-gqa-reuse-test";
    const tokenIds = [101, 102];
    const now = new Date().toISOString();
    const block: SerializedKVSwapBlock = {
      version: KV_SWAP_STORAGE_VERSION,
      namespace,
      id: "block-0",
      modelId: "Qwen/Qwen3-0.6B",
      requestId: "req-compact-gqa",
      runtimeBlockId: "layer0:b0",
      phase: "prefill",
      modelFingerprint: "fixture-fingerprint",
      promptTokenHash: "391a58f2",
      promptTokenIds: tokenIds,
      prefillTokenCount: tokenIds.length,
      runtimeLayerCount: 1,
      policyHash: "client-v4:block16:full40960:sparse256:anchors8:trail2:tokens2:layers1",
      layer: 0,
      startToken: 0,
      endToken: 2,
      pinned: false,
      importance: 1,
      estimatedBytes: 128,
      tokenIds,
      queryRows: [[1, 0, 0, 1], [0, 1, 1, 0]],
      keyRows: [[10, 11, 12, 13], [14, 15, 16, 17]],
      valueRows: [[20, 21, 22, 23], [24, 25, 26, 27]],
      compactKeyRows: [[10, 11], [14, 15]],
      compactValueRows: [[20, 21], [24, 25]],
      hiddenRows: [[0.1, 0.2], [0.3, 0.4]],
      createdAt: now,
      updatedAt: now,
      lastAccessAt: Date.now(),
      byteLength: 128,
    };

    const reused = __unlockedBrowserTransformerClientTestHooks.buildReusedKvCacheHandle({
      blocks: [block],
      namespace,
      modelId: "Qwen/Qwen3-0.6B",
      modelFingerprint: "fixture-fingerprint",
      requestId: "req-compact-gqa",
      tokenIds,
      layerCount: 1,
    });

    expect(reused?.layerStates[0]?.k[0]).toHaveLength(4);
    expect(reused?.layerStates[0]?.compactK).toEqual([[10, 11], [14, 15]]);
    expect(reused?.layerStates[0]?.compactV).toEqual([[20, 21], [24, 25]]);
  });

  it("defers prefill KV persistence without blocking stream completion and flushes on demand", async () => {
    vi.stubGlobal("indexedDB", {} as IDBFactory);
    const store = new DelayedPersistKVSwapPersistence({
      namespace: "deferred-persist-test",
      maxBlocks: 32,
      maxBytes: 1024 * 1024,
    });
    vi.stubGlobal("createIndexedDbPersistence", makeIndexedDbPersistenceForStore(store));
    const client = new UnlockedBrowserTransformerClient({
      modelId: "Qwen/Qwen3-0.6B",
      allowFixtureWeights: true,
      backendPreference: "cpu",
      maxGenerationTokens: 1,
      kvPersistence: {
        enabled: true,
        namespace: "deferred-persist-test",
        preferOpfs: false,
      },
    });

    await client.init();
    const iterator = client.streamChat([{ role: "user", content: "defer this prompt" }], { maxTokens: 1 })[Symbol.asyncIterator]();
    const first = await iterator.next();

    expect(first.done).toBe(false);
    expect(String(first.value).length).toBeGreaterThan(0);
    expect(store.persistStarted).toBe(false);
    expect(client.lastDecodeProof?.kvPersistence).toMatchObject({
      kvPersistDeferred: true,
      kvPersistCriticalPathMs: 0,
    });

    await expect(iterator.next()).resolves.toMatchObject({ done: true });
    expect(store.persistStarted).toBe(false);
    let flushed = false;
    const flushPromise = client.flushKvPersistence().then(() => {
      flushed = true;
    });
    await Promise.resolve();
    expect(store.persistStarted).toBe(true);
    expect(flushed).toBe(false);
    store.resolvePersist();
    await flushPromise;
    expect(client.lastDecodeProof?.kvPersistence).toMatchObject({
      kvPersistDeferred: true,
      kvPersistFlushMs: expect.any(Number),
    });
    expect(client.lastDecodeProof?.kvPersistence?.kvPersistPendingBlockCount).toBeGreaterThan(0);
    expect(client.lastDecodeProof?.kvPersistence?.events).toEqual(expect.arrayContaining([
      expect.objectContaining({
        operation: "persist",
        ok: true,
      }),
    ]));
  });

  it("flushes pending deferred KV persistence before direct client disposal clears state", async () => {
    vi.stubGlobal("indexedDB", {} as IDBFactory);
    const namespace = "dispose-flush-test";
    const store = new DelayedPersistKVSwapPersistence({
      namespace,
      maxBlocks: 32,
      maxBytes: 1024 * 1024,
    });
    vi.stubGlobal("createIndexedDbPersistence", makeIndexedDbPersistenceForStore(store));
    const client = new UnlockedBrowserTransformerClient({
      modelId: "Qwen/Qwen3-0.6B",
      allowFixtureWeights: true,
      backendPreference: "cpu",
      maxGenerationTokens: 1,
      kvPersistence: {
        enabled: true,
        namespace,
        preferOpfs: false,
      },
    });

    await client.init();
    await collectStream(client.streamChat([{ role: "user", content: "dispose should flush this prompt" }], { maxTokens: 1 }));
    expect(store.persistStarted).toBe(false);

    let disposed = false;
    const disposePromise = client.dispose().then(() => {
      disposed = true;
    });
    await Promise.resolve();

    expect(store.persistStarted).toBe(true);
    expect(disposed).toBe(false);
    store.resolvePersist();
    await disposePromise;

    const hydrated = await store.hydrate(namespace);
    expect(hydrated.blocks.length).toBeGreaterThan(0);
    expect(hydrated.blocks.every((block) => block.id.includes(":prefill:"))).toBe(true);
  });

  it("hydrates exact-match prefill KV in a fresh client using the same persistence namespace", async () => {
    vi.stubGlobal("indexedDB", {} as IDBFactory);
    vi.stubGlobal("createIndexedDbPersistence", makeSharedIndexedDbPersistence());
    const namespace = "reload-reuse-test";
    const prompt = "reload should reuse this prompt";
    const firstClient = makeFixtureKvClient(namespace);

    await firstClient.init();
    await collectStream(firstClient.streamChat([{ role: "user", content: prompt }]));
    await firstClient.flushKvPersistence();
    const firstPersistEvents = firstClient.lastDecodeProof?.kvPersistence?.events.filter((event) => event.operation === "persist") ?? [];
    expect(firstPersistEvents).toHaveLength(1);
    expect(firstPersistEvents[0]?.blockIds.every((blockId) => blockId.includes(":prefill:"))).toBe(true);

    const secondClient = makeFixtureKvClient(namespace);
    await secondClient.init();

    expect(secondClient.getKvPersistenceHealth().lastOperation).toMatchObject({
      operation: "hydrate",
      mode: "indexeddb",
      namespace,
      ok: true,
      blockIds: expect.arrayContaining([expect.stringContaining(":prefill:")]),
    });

    await collectStream(secondClient.streamChat([{ role: "user", content: prompt }]));

    expect(secondClient.lastDecodeProof?.kvPersistence).toMatchObject({
      enabled: true,
      mode: "indexeddb",
      namespace,
      decodeReuse: true,
    });
    expect(secondClient.lastDecodeProof?.kvPersistence?.events).toEqual(expect.arrayContaining([
      expect.objectContaining({
        operation: "hydrate",
        mode: "indexeddb",
        ok: true,
      }),
      expect.objectContaining({
        operation: "reuse",
        mode: "indexeddb",
        ok: true,
        reason: "decode_reuse_prefill_skipped",
      }),
    ]));
    expect(secondClient.lastDecodeProof?.prefillProjectionBackends).toEqual([
      expect.objectContaining({
        layerIndex: 0,
        qProjection: "cpu_reference",
        kProjection: "cpu_reference",
        vProjection: "cpu_reference",
        oProjection: "cpu_reference",
      }),
    ]);
    expect(secondClient.lastDecodeProof?.prefillAttentionBackends).toEqual([
      expect.objectContaining({
        layerIndex: 0,
        attentionBackend: "cpu_reference",
        packedHeadBackends: ["cpu_reference"],
        packedHeadCount: 1,
      }),
    ]);
    expect(secondClient.lastDecodeProof?.kvPersistence?.events.filter((event) => event.operation === "persist")).toHaveLength(0);
  });

  it("rejects exact KV reuse when persisted blocks carry an old routing policy hash", async () => {
    vi.stubGlobal("indexedDB", {} as IDBFactory);
    const namespace = "old-policy-reuse-test";
    const prompt = "same prompt but old policy hash";
    const store = new MemoryKVSwapPersistence({
      namespace,
      maxBlocks: 64,
      maxBytes: 1024 * 1024,
    });
    vi.stubGlobal("createIndexedDbPersistence", makeIndexedDbPersistenceForStore(store));
    const firstClient = makeFixtureKvClient(namespace);

    await firstClient.init();
    await collectStream(firstClient.streamChat([{ role: "user", content: prompt }]));
    await firstClient.flushKvPersistence();
    const persisted = await store.list(namespace);
    expect(persisted.records.length).toBeGreaterThan(0);
    await store.persist(persisted.records.map((block) => ({
      ...block,
      policyHash: "client-v3:block2:stale",
    })));

    const secondClient = makeFixtureKvClient(namespace);
    await secondClient.init();
    await collectStream(secondClient.streamChat([{ role: "user", content: prompt }]));

    expect(secondClient.lastDecodeProof?.kvPersistence).toMatchObject({
      enabled: true,
      mode: "indexeddb",
      namespace,
      decodeReuse: false,
    });
    expect(secondClient.lastDecodeProof?.kvPersistence?.events.filter((event) => event.operation === "reuse")).toHaveLength(0);
  });

  it("normalizes browser-style KV namespaces before exact-match reuse", async () => {
    vi.stubGlobal("indexedDB", {} as IDBFactory);
    vi.stubGlobal("createIndexedDbPersistence", makeSharedIndexedDbPersistence());
    const namespace = "local:browser:browser-preview-bench";
    const prompt = "browser preview should reuse this prompt";
    const firstClient = makeFixtureKvClient(namespace);

    await firstClient.init();
    await collectStream(firstClient.streamChat([{ role: "user", content: prompt }]));
    await firstClient.flushKvPersistence();

    const secondClient = makeFixtureKvClient(namespace);
    await secondClient.init();
    await collectStream(secondClient.streamChat([{ role: "user", content: prompt }]));

    expect(secondClient.lastDecodeProof?.kvPersistence).toMatchObject({
      enabled: true,
      mode: "indexeddb",
      namespace: "local_browser_browser-preview-bench",
      decodeReuse: true,
    });
    expect(secondClient.lastDecodeProof?.kvPersistence?.events).toEqual(expect.arrayContaining([
      expect.objectContaining({
        operation: "reuse",
        mode: "indexeddb",
        ok: true,
      }),
    ]));
  });

  it("does not carry decode reuse proof from one prompt into a later different prompt", async () => {
    vi.stubGlobal("indexedDB", {} as IDBFactory);
    vi.stubGlobal("createIndexedDbPersistence", makeSharedIndexedDbPersistence());
    const client = makeFixtureKvClient("reuse-reset-test");

    await client.init();
    await collectStream(client.streamChat([{ role: "user", content: "prompt that will be reused" }]));
    await client.flushKvPersistence();
    await collectStream(client.streamChat([{ role: "user", content: "prompt that will be reused" }]));
    expect(client.lastDecodeProof?.kvPersistence?.decodeReuse).toBe(true);

    await collectStream(client.streamChat([{ role: "user", content: "prompt that must not reuse stale proof" }]));

    expect(client.lastDecodeProof?.kvPersistence?.decodeReuse).toBe(false);
    expect(client.lastDecodeProof?.kvPersistence?.events.filter((event) => event.operation === "reuse")).toHaveLength(0);
  });

  it("schedules predictive KV prefetch loads for persisted blocks when exact prompt reuse misses", async () => {
    const client = new UnlockedBrowserTransformerClient({
      modelId: "Qwen/Qwen3-0.6B",
      allowFixtureWeights: true,
      backendPreference: "cpu",
      maxGenerationTokens: 1,
      kvPersistence: {
        enabled: true,
        namespace: "predictive-prefetch-test",
        preferOpfs: false,
      },
    });

    await client.init();
    await collectStream(client.streamChat([{ role: "user", content: "alpha beta runtime cache" }]));
    await client.flushKvPersistence();
    await collectStream(client.streamChat([{ role: "user", content: "alpha beta followup cache" }], {
      awaitKvPredictivePrefetchProof: true,
    }));

    const proof = client.lastDecodeProof?.kvPersistence;
    expect(proof).toMatchObject({
      enabled: true,
      mode: "memory",
      namespace: "predictive-prefetch-test",
      decodeReuse: false,
      prefetchStrategy: "predictive_prefetch",
      lowRankSummaryRank: expect.any(Number),
      lowRankQuerySource: "persisted_q_rows",
      predictedHotBlocks: expect.any(Array),
      prefetchedBlocks: expect.any(Array),
      prefetchHitRate: expect.any(Number),
      prefetchBytes: expect.any(Number),
      prefetchLatencyMs: expect.any(Number),
      attentionStallMs: expect.any(Number),
    });
    expect(proof?.predictedHotBlocks?.length).toBeGreaterThan(0);
    expect(proof?.prefetchedBlocks?.length).toBeGreaterThan(0);
    expect(proof?.prefetchBytes).toBeGreaterThan(0);
    expect(proof?.events).toEqual(expect.arrayContaining([
      expect.objectContaining({
        operation: "load",
        ok: true,
        reason: "predictive_prefetch",
        predictedHotBlocks: expect.any(Array),
        prefetchedBlocks: expect.any(Array),
        prefetchStrategy: "predictive_prefetch",
        lowRankQuerySource: "persisted_q_rows",
      }),
    ]));
  });

  it("does not await predictive KV loads before the first streamed token", async () => {
    vi.stubGlobal("indexedDB", {} as IDBFactory);
    const store = new DelayedLoadKVSwapPersistence({
      namespace: "nonblocking-predictive-prefetch-test",
      maxBlocks: 64,
      maxBytes: 1024 * 1024,
    });
    vi.stubGlobal("createIndexedDbPersistence", makeIndexedDbPersistenceForStore(store));
    const client = new UnlockedBrowserTransformerClient({
      modelId: "Qwen/Qwen3-0.6B",
      allowFixtureWeights: true,
      backendPreference: "cpu",
      maxGenerationTokens: 1,
      kvPersistence: {
        enabled: true,
        namespace: "nonblocking-predictive-prefetch-test",
        preferOpfs: false,
      },
    });

    await client.init();
    await collectStream(client.streamChat([{ role: "user", content: "alpha beta runtime cache" }]));
    await client.flushKvPersistence();

    const iterator = client.streamChat([{ role: "user", content: "alpha beta followup cache" }], {
      maxTokens: 1,
      awaitKvPredictivePrefetchProof: true,
    })[Symbol.asyncIterator]();
    const first = await iterator.next();

    expect(first.done).toBe(false);
    expect(String(first.value).length).toBeGreaterThan(0);
    expect(store.loadStarted).toBe(false);

    let done = false;
    const donePromise = iterator.next().then((result) => {
      done = true;
      return result;
    });
    for (let index = 0; index < 5 && !store.loadStarted; index += 1) {
      await new Promise((resolve) => setTimeout(resolve, 0));
    }
    expect(store.loadStarted).toBe(true);
    expect(store.pendingLoadCount).toBeGreaterThan(0);
    expect(done).toBe(false);
    expect(store.loadSettled).toBe(false);
    store.resolveLoads();
    await expect(donePromise).resolves.toMatchObject({ done: true });
    await store.waitForLoadsToSettle();
    for (let index = 0; index < 5 && client.lastDecodeProof?.kvPersistence?.prefetchStrategy !== "predictive_prefetch"; index += 1) {
      await new Promise((resolve) => setTimeout(resolve, 0));
    }

    expect(client.lastDecodeProof?.kvPersistence).toMatchObject({
      prefetchStrategy: "predictive_prefetch",
      lowRankQuerySource: "persisted_q_rows",
    });
  });

  it("builds low-rank predictive KV queries from persisted Q projection rows when tokens overlap", () => {
    const query = __unlockedBrowserTransformerClientTestHooks.buildClientLowRankQuerySummary(
      [101, 202, 303],
      4,
      [
        {
          version: 1,
          namespace: "predictive-query-test",
          id: "persisted:block0",
          modelId: "Qwen/Qwen3-0.6B",
          requestId: "request_seed",
          phase: "prefill",
          runtimeBlockId: "layer0:b0",
          modelFingerprint: "fingerprint",
          layer: 0,
          startToken: 0,
          endToken: 2,
          pinned: false,
          importance: 0.9,
          estimatedBytes: 128,
          tokenIds: [101, 202],
          queryRows: [
            [2, 2, 4, 4, 8, 8, 16, 16],
            [4, 4, 6, 6, 10, 10, 18, 18],
          ],
          keyRows: [[0], [0]],
          valueRows: [[0], [0]],
          createdAt: "2026-05-22T00:00:00.000Z",
          updatedAt: "2026-05-22T00:00:00.000Z",
          lastAccessAt: 0,
          byteLength: 128,
        },
        {
          version: 1,
          namespace: "predictive-query-test",
          id: "persisted:block1",
          modelId: "Qwen/Qwen3-0.6B",
          requestId: "request_seed",
          phase: "prefill",
          runtimeBlockId: "layer1:b0",
          modelFingerprint: "fingerprint",
          layer: 1,
          startToken: 0,
          endToken: 1,
          pinned: false,
          importance: 0.9,
          estimatedBytes: 128,
          tokenIds: [101],
          queryRows: [[100, 100, 100, 100, 100, 100, 100, 100]],
          keyRows: [[0]],
          valueRows: [[0]],
          createdAt: "2026-05-22T00:00:00.000Z",
          updatedAt: "2026-05-22T00:00:00.000Z",
          lastAccessAt: 0,
          byteLength: 128,
        },
      ],
    );

    expect(query).toMatchObject({
      rank: 4,
      layer: 0,
      projectionId: "unlocked-browser:key-low-rank:v1",
      headGroupId: "all_heads",
    });
    expect(Array.from(query.values)).toEqual([3, 5, 9, 17]);
  });

  it("persists prefill KV once by default instead of rewriting the full cache after every decode token", async () => {
    const client = new UnlockedBrowserTransformerClient({
      modelId: "Qwen/Qwen3-0.6B",
      allowFixtureWeights: true,
      backendPreference: "cpu",
      maxGenerationTokens: 3,
      mtp: {
        enabled: false,
      },
      kvPersistence: {
        enabled: true,
        namespace: "decode-pressure-test",
        preferOpfs: false,
      },
    });

    await client.init();
    await collectStream(client.streamChat([{ role: "user", content: "persist pressure prompt" }]));
    await client.flushKvPersistence();

    const persistEvents = client.lastDecodeProof?.kvPersistence?.events.filter((event) => event.operation === "persist") ?? [];
    expect(persistEvents).toHaveLength(1);
    expect(persistEvents[0]).toMatchObject({
      operation: "persist",
      ok: true,
    });
    expect(persistEvents[0]?.blockIds.every((blockId) => blockId.includes(":prefill:"))).toBe(true);
  });

  it("encodes qwen-bpe manifests with byte-level merges instead of lowercased vocab word matching", () => {
    const tokenizer = __unlockedBrowserTransformerClientTestHooks.normalizeTokenizer({
      kind: "qwen-bpe",
      tokens: ["<unk>", "A", "l", "p", "h", "a", "Al", "Alp", "Alph", "Alpha", "alpha"],
      merges: [["A", "l"], "Al p", ["Alp", "h"], ["Alph", "a"]],
      unknownTokenId: 0,
    }, 11);

    expect(tokenizer.encode("Alpha", 11)).toEqual([9]);
    expect(tokenizer.encode("alpha", 11)).toEqual([5, 2, 3, 4, 5]);
    expect(tokenizer.decode(9)).toBe("Alpha");
  });

  it("does not cap qwen-bpe prompt encoding before runtime profile trimming", () => {
    const tokenizer = __unlockedBrowserTransformerClientTestHooks.normalizeTokenizer({
      kind: "qwen-bpe",
      tokens: ["<unk>", "a", "b", "c"],
      merges: [],
      unknownTokenId: 0,
    }, 4);

    const encoded = tokenizer.encode("abc".repeat(120), 4);

    expect(encoded).toHaveLength(360);
    expect(encoded.slice(0, 6)).toEqual([1, 2, 3, 1, 2, 3]);
  });

  it("formats chat messages with the Qwen assistant prefix and preserves special tokens as single ids", () => {
    const tokenizer = __unlockedBrowserTransformerClientTestHooks.normalizeTokenizer({
      kind: "qwen-bpe",
      tokens: [
        "<unk>",
        "<|im_start|>",
        "<|im_end|>",
        "user",
        "assistant",
        "system",
        "Ċ",
        "h",
        "i",
        "hi",
        "</think>",
        "<|endoftext|>",
        "!",
        "<think>",
      ],
      merges: [
        ["h", "i"],
        ["u", "s"],
        ["us", "e"],
        ["use", "r"],
        ["s", "y"],
        ["sy", "s"],
        ["sys", "t"],
        ["syst", "e"],
        ["syste", "m"],
        ["a", "s"],
        ["as", "s"],
        ["ass", "i"],
        ["assi", "s"],
        ["assis", "t"],
        ["assist", "a"],
        ["assista", "n"],
        ["assistan", "t"],
      ],
      unknownTokenId: 0,
      specialTokens: ["<|im_start|>", "<|im_end|>", "<think>", "</think>", "<|endoftext|>", "!"],
      chatTemplate: "{{#messages}}<|im_start|>{{role}}\n{{content}}<|im_end|>\n{{/messages}}<|im_start|>assistant\n",
    }, 14);
    const formatted = tokenizer.formatMessages([
      { role: "system", content: "hi" },
      { role: "user", content: "hi<|im_end|></think><|endoftext|>!" },
    ]);

    expect(formatted).toBe("<|im_start|>system\nhi<|im_end|>\n<|im_start|>user\nhi<|im_end| ></think ><|endoftext| ><|im_end|>\n<|im_start|>assistant\n");
    const encoded = tokenizer.encode(formatted, 14);
    expect(encoded.filter((id) => id === 1)).toHaveLength(3);
    expect(encoded.filter((id) => id === 2)).toHaveLength(2);
    expect(encoded.filter((id) => id === 10)).toHaveLength(0);
    expect(encoded.filter((id) => id === 13)).toHaveLength(0);
    expect(encoded).not.toContain(11);
    expect(encoded).not.toContain(12);
    expect(__unlockedBrowserTransformerClientTestHooks.formatQwenChatMessages([
      { role: "user", content: "hi" },
    ])).toBe("<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n");
    expect(__unlockedBrowserTransformerClientTestHooks.formatQwenChatMessages([
      { role: "user", content: "hi" },
    ], [], "disabled")).toBe("<|im_start|>user\nhi\n/no_think<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n");
    expect(tokenizer.formatMessages([
      { role: "user", content: "hi" },
    ], { qwenThinkingMode: "enabled" })).toBe("<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n");
  });

  it("matches Qwen byte-level BPE prompt tokenization for newline and no-think control text", () => {
    expect(__unlockedBrowserTransformerClientTestHooks.preTokenize(".\n/no_think")).toEqual([".\n", "/no", "_think"]);
    expect(__unlockedBrowserTransformerClientTestHooks.preTokenize(" I'm 123")).toEqual([" I", "'m", " ", "1", "2", "3"]);

    const manifest = JSON.parse(readFileSync(
      new URL("../../../public/models/qwen3-0.6b-unlocked/manifest.json", import.meta.url),
      "utf8",
    ));
    const tokenizer = __unlockedBrowserTransformerClientTestHooks.normalizeTokenizer(manifest.tokenizer, manifest.vocabSize);
    const formatted = __unlockedBrowserTransformerClientTestHooks.formatQwenChatMessages([
      { role: "user", content: "Answer with the exact sentence: Salt Lake City is ready." },
    ], manifest.tokenizer.specialTokens, "disabled");

    expect(tokenizer.encode("Salt Lake City is ready.", manifest.vocabSize)).toEqual([47318, 11678, 4311, 374, 5527, 13]);
    expect(tokenizer.encode("What is the capital of Montana?", manifest.vocabSize)).toEqual([3838, 374, 279, 6722, 315, 36005, 30]);
    expect(tokenizer.encode(formatted, manifest.vocabSize)).toEqual([
      151644, 872, 198, 16141, 448, 279, 4734, 11652, 25, 27040, 11678, 4311, 374, 5527, 624,
      33100, 5854, 766, 151645, 198, 151644, 77091, 198, 151667, 271, 151668, 271,
    ]);
  });

  it("filters proof markers, hidden thinking, and qwen stop markers from streamed assistant output", () => {
    const filter = __unlockedBrowserTransformerClientTestHooks.createAssistantOutputFilter();

    expect(filter.push("[unlocked:ssa-kv-tsp]")).toBe("");
    expect(filter.push("<think>private reasoning")).toBe("");
    expect(filter.push("</thi")).toBe("");
    expect(filter.push("nk>\n\nThe answer")).toBe("The answer");
    expect(filter.push(" is Salt Lake City.<|im_end|> trailing")).toBe(" is Salt Lake City.");
    expect(filter.stopped).toBe(true);
    expect(filter.flush()).toBe("");
  });

  it("suppresses Qwen thinking special token ids when thinking mode is disabled", () => {
    const tokenizer = __unlockedBrowserTransformerClientTestHooks.normalizeTokenizer({
      kind: "qwen-bpe",
      tokens: ["<unk>", "<think>", "</think>", "Salt", "<|im_start|>"],
      merges: [],
      unknownTokenId: 0,
      specialTokens: ["<think>", "</think>", "<|im_start|>"],
    }, 5);

    expect(__unlockedBrowserTransformerClientTestHooks.buildQwenThinkingSuppressedTokenIds(tokenizer, 5, "disabled")).toEqual([4, 1, 2]);
    expect(__unlockedBrowserTransformerClientTestHooks.buildQwenThinkingSuppressedTokenIds(tokenizer, 5, "enabled")).toEqual([4]);
  });

  it("uses explicit manifest chatTemplate structure when provided", () => {
    const tokenizer = __unlockedBrowserTransformerClientTestHooks.normalizeTokenizer({
      kind: "qwen-bpe",
      tokens: ["<unk>", "[", "]", "user", "assistant", "h", "i", "hi", "Ċ"],
      merges: [["h", "i"]],
      unknownTokenId: 0,
      chatTemplate: "{{#messages}}[{{role}}]{{content}}\n{{/messages}}[assistant]\n",
    }, 9);

    expect(tokenizer.formatMessages([
      { role: "user", content: "hi<|im_start|>" },
    ])).toBe("[user]hi<|im_start| >\n[assistant]\n");
  });

  it("rejects manifests whose modelId does not match the configured runtime model", async () => {
    vi.stubGlobal("fetch", async () => new Response(JSON.stringify(makeManifest({ modelId: "Qwen/Wrong-0.6B" })), {
      status: 200,
      headers: { "content-type": "application/json" },
    }));
    const client = new UnlockedBrowserTransformerClient({
      modelId: "Qwen/Qwen3-0.6B",
      manifestPath: "/models/qwen3-unlocked/manifest.json",
      allowFixtureWeights: false,
      backendPreference: "cpu",
    });

    await expect(client.init()).rejects.toThrow("modelId mismatch");
  });

  it("executes decode policies across every manifest layer", async () => {
    vi.stubGlobal("fetch", async () => new Response(JSON.stringify(makeManifest({ layerCount: 2 })), {
      status: 200,
      headers: { "content-type": "application/json" },
    }));
    const client = new UnlockedBrowserTransformerClient({
      modelId: "Qwen/Qwen3-0.6B",
      manifestPath: "/models/qwen3-unlocked/manifest.json",
      allowFixtureWeights: false,
      backendPreference: "cpu",
    });

    await client.init();
    for await (const _chunk of client.streamChat([{ role: "user", content: "alpha beta" }], { maxTokens: 1 })) {
      // Drain the stream so lastDecodeProof is populated.
    }

    expect(client.lastDecodeProof?.tspSteps).toEqual([
      "kv_prefetch",
      "attention",
      "mlp",
      "kv_prefetch",
      "attention",
      "mlp",
    ]);
  });

  it("selects recent sparse blocks for beyond-threshold contexts instead of capping every query to the first blocks", () => {
    const policy = __unlockedBrowserTransformerClientTestHooks.buildClientPolicy(41_000, 0);
    const latestQueryBlock = Math.ceil(41_000 / policy.blockSize) - 1;
    const selected = policy.selectedBlockIdsByQueryBlock[latestQueryBlock] ?? [];

    expect(selected).toHaveLength(256);
    expect(selected).toContain(`b${latestQueryBlock}`);
    expect(selected).toContain(`b${latestQueryBlock - 1}`);
    expect(selected.some((blockId) => Number(blockId.slice(1)) > 1000)).toBe(true);
  });

  it("loads manifest matrices from typed-array shard descriptors relative to the manifest URL", async () => {
    const shardBytes = makeF32Shard([
      ...flatten([
        [1, 0],
        [0, 1],
        [1, 1],
        [0.5, 0.5],
      ]),
      ...flatten([
        [0, 0],
        [0, 0],
        [10, 10],
        [-1, -1],
      ]),
      ...flatten([
        [1, 0],
        [0, 1],
      ]),
    ]);
    const shardSha256 = await sha256Hex(shardBytes);
    const fetched: string[] = [];
    vi.stubGlobal("fetch", async (input: RequestInfo | URL) => {
      const url = String(input);
      fetched.push(url);
      if (url.endsWith("/manifest.json")) {
        return new Response(JSON.stringify(makeShardedManifest({ shardSha256 })), {
          status: 200,
          headers: { "content-type": "application/json" },
        });
      }
      if (url.endsWith("/weights.bin")) {
        return new Response(shardBytes, {
          status: 200,
          headers: { "content-type": "application/octet-stream" },
        });
      }
      return new Response("not found", { status: 404 });
    });
    const client = new UnlockedBrowserTransformerClient({
      modelId: "Qwen/Qwen3-0.6B",
      manifestPath: "https://cdn.example.test/models/qwen3/manifest.json",
      allowFixtureWeights: false,
      backendPreference: "cpu",
    });

    await client.init();
    const chunks: string[] = [];
    for await (const chunk of client.streamChat([{ role: "user", content: "alpha beta" }], { maxTokens: 1 })) {
      chunks.push(chunk);
    }

    expect(fetched).toContain("https://cdn.example.test/models/qwen3/weights.bin");
    expect(fetched.filter((url) => url.endsWith("/weights.bin"))).toHaveLength(1);
    expect(chunks.join("")).toContain("MANIFEST_TOKEN");
  });

  it("loads f16-packed shard descriptors from production-style manifests", async () => {
    const shardValues = [
      ...flatten([
        [1, 0],
        [0, 1],
        [1, 1],
        [0.5, 0.5],
      ]),
      ...flatten([
        [0, 0],
        [0, 0],
        [10, 10],
        [-1, -1],
      ]),
      ...flatten([
        [1, 0],
        [0, 1],
      ]),
    ];
    const shardBytes = makeF16Shard(shardValues);
    const shardSha256 = await sha256Hex(shardBytes);
    const digestSpy = vi.spyOn(globalThis.crypto.subtle, "digest");
    vi.stubGlobal("fetch", async (input: RequestInfo | URL) => {
      const url = String(input);
      if (url.endsWith("/manifest.json")) {
        return new Response(JSON.stringify(makeShardedManifest({ shardSha256, tensorFormat: "f16" })), {
          status: 200,
          headers: { "content-type": "application/json" },
        });
      }
      if (url.endsWith("/weights.bin")) {
        return new Response(shardBytes, {
          status: 200,
          headers: { "content-type": "application/octet-stream" },
        });
      }
      return new Response("not found", { status: 404 });
    });
    const client = new UnlockedBrowserTransformerClient({
      modelId: "Qwen/Qwen3-0.6B",
      manifestPath: "https://cdn.example.test/models/qwen3/manifest.json",
      allowFixtureWeights: false,
      backendPreference: "cpu",
    });

    await client.init();
    const chunks = await collectStream(client.streamChat([{ role: "user", content: "alpha beta" }], { maxTokens: 1 }));

    expect(digestSpy).toHaveBeenCalledTimes(1);
    digestSpy.mockRestore();
    expect(chunks).toContain("MANIFEST_TOKEN");
  });

  it("decodes f16 bit-pattern edge cases for packed shard loading", () => {
    const decode = __unlockedBrowserTransformerClientTestHooks.float16BitsToFloat32;
    expect(Object.is(decode(0x0000), 0)).toBe(true);
    expect(Object.is(decode(0x8000), -0)).toBe(true);
    expect(decode(0x0001)).toBe(2 ** -24);
    expect(decode(0x03ff)).toBeCloseTo(1023 * 2 ** -24, 12);
    expect(decode(0x3c00)).toBe(1);
    expect(decode(0x7bff)).toBe(65504);
    expect(decode(0x7c00)).toBe(Infinity);
    expect(decode(0xfc00)).toBe(-Infinity);
    expect(Number.isNaN(decode(0x7e00))).toBe(true);
  });

  it("loads optional Qwen norm vectors and gated MLP shard descriptors", async () => {
    const shardBytes = makeF32Shard([
      ...flatten([
        [1, 0],
        [0, 1],
        [1, 1],
        [0.5, 0.5],
      ]),
      ...flatten([
        [0, 0],
        [0, 0],
        [10, 10],
        [-1, -1],
      ]),
      ...flatten([
        [1, 0],
        [0, 1],
      ]),
      1, 1,
      1, 1,
      1, 1,
      1, 1,
      1, 1,
      ...flatten([
        [1, 0],
        [0, 1],
        [1, 1],
      ]),
      ...flatten([
        [1, 0],
        [0, 1],
        [1, 1],
      ]),
      ...flatten([
        [1, 0, 0.5],
        [0, 1, -0.5],
      ]),
    ]);
    const shardSha256 = await sha256Hex(shardBytes);
    vi.stubGlobal("fetch", async (input: RequestInfo | URL) => {
      const url = String(input);
      if (url.endsWith("/manifest.json")) {
        return new Response(JSON.stringify(makeShardedManifest({ shardSha256, qwenMath: true })), {
          status: 200,
          headers: { "content-type": "application/json" },
        });
      }
      if (url.endsWith("/weights.bin")) {
        return new Response(shardBytes, {
          status: 200,
          headers: { "content-type": "application/octet-stream" },
        });
      }
      return new Response("not found", { status: 404 });
    });
    const client = new UnlockedBrowserTransformerClient({
      modelId: "Qwen/Qwen3-0.6B",
      manifestPath: "https://cdn.example.test/models/qwen3/manifest.json",
      allowFixtureWeights: false,
      backendPreference: "cpu",
    });

    await client.init();
    const chunks: string[] = [];
    for await (const chunk of client.streamChat([{ role: "user", content: "alpha beta" }], { maxTokens: 1 })) {
      chunks.push(chunk);
    }

    expect(chunks.join("").trim().length).toBeGreaterThan(0);
    expect(client.lastDecodeProof?.tensorControl).toBe(true);
  });

  it("loads full Qwen GQA metadata and full projection shard shapes", async () => {
    const shardBytes = makeF32Shard([
      ...flatten([
        [1, 0],
        [0, 1],
        [1, 1],
        [0.5, 0.5],
      ]),
      ...flatten([
        [0, 0],
        [0, 0],
        [10, 10],
        [-1, -1],
      ]),
      ...flatten([
        [1, 0],
        [0, 1],
        [1, 1],
        [1, -1],
      ]),
      ...flatten([
        [1, 0],
        [0, 1],
      ]),
      ...flatten([
        [0.5, 0.5],
        [1, -1],
      ]),
      ...flatten([
        [1, 0, 0.5, 0],
        [0, 1, 0, 0.5],
      ]),
    ]);
    const shardSha256 = await sha256Hex(shardBytes);
    vi.stubGlobal("fetch", async (input: RequestInfo | URL) => {
      const url = String(input);
      if (url.endsWith("/manifest.json")) {
        return new Response(JSON.stringify(makeFullGqaShardedManifest({ shardSha256 })), {
          status: 200,
          headers: { "content-type": "application/json" },
        });
      }
      return new Response(shardBytes, {
        status: 200,
        headers: { "content-type": "application/octet-stream" },
      });
    });
    const client = new UnlockedBrowserTransformerClient({
      modelId: "Qwen/Qwen3-0.6B",
      manifestPath: "https://cdn.example.test/models/qwen3/manifest.json",
      allowFixtureWeights: false,
      backendPreference: "cpu",
    });

    await client.init();
    for await (const _chunk of client.streamChat([{ role: "user", content: "alpha beta" }], { maxTokens: 1 })) {
      // Drain the stream so lastDecodeProof is populated.
    }

    expect(client.lastDecodeProof?.tensorControl).toBe(true);
  });

  it("rejects full Qwen manifests whose attention metadata disagrees with shard shapes", async () => {
    const shardBytes = makeF32Shard([
      ...flatten([
        [1, 0],
        [0, 1],
        [1, 1],
        [0.5, 0.5],
      ]),
      ...flatten([
        [0, 0],
        [0, 0],
        [10, 10],
        [-1, -1],
      ]),
      ...flatten([
        [1, 0],
        [0, 1],
        [1, 1],
        [1, -1],
      ]),
      ...flatten([
        [1, 0],
        [0, 1],
        [1, 1],
        [1, -1],
      ]),
      ...flatten([
        [0.5, 0.5],
        [1, -1],
        [0.25, 0.25],
        [0.5, -0.5],
      ]),
      ...flatten([
        [1, 0, 0.5, 0],
        [0, 1, 0, 0.5],
      ]),
    ]);
    const shardSha256 = await sha256Hex(shardBytes);
    vi.stubGlobal("fetch", async (input: RequestInfo | URL) => {
      const url = String(input);
      if (url.endsWith("/manifest.json")) {
        return new Response(JSON.stringify(makeFullGqaShardedManifest({ shardSha256, mismatchedKvShape: true })), {
          status: 200,
          headers: { "content-type": "application/json" },
        });
      }
      return new Response(shardBytes, {
        status: 200,
        headers: { "content-type": "application/octet-stream" },
      });
    });
    const client = new UnlockedBrowserTransformerClient({
      modelId: "Qwen/Qwen3-0.6B",
      manifestPath: "https://cdn.example.test/models/qwen3/manifest.json",
      allowFixtureWeights: false,
      backendPreference: "cpu",
    });

    await expect(client.init()).rejects.toThrow("kProj must have 2 rows");
  });

  it("rejects vector fields with matrix shard shapes", async () => {
    const shardBytes = makeF32Shard([
      ...flatten([
        [1, 0],
        [0, 1],
        [1, 1],
        [0.5, 0.5],
      ]),
      ...flatten([
        [0, 0],
        [0, 0],
        [10, 10],
        [-1, -1],
      ]),
      ...flatten([
        [1, 0],
        [0, 1],
      ]),
      1, 1,
    ]);
    const shardSha256 = await sha256Hex(shardBytes);
    vi.stubGlobal("fetch", async (input: RequestInfo | URL) => {
      const url = String(input);
      if (url.endsWith("/manifest.json")) {
        return new Response(JSON.stringify(makeShardedManifest({
          shardSha256,
          qwenMath: true,
          vectorShape: [1, 2],
        })), {
          status: 200,
          headers: { "content-type": "application/json" },
        });
      }
      return new Response(shardBytes, {
        status: 200,
        headers: { "content-type": "application/octet-stream" },
      });
    });
    const client = new UnlockedBrowserTransformerClient({
      modelId: "Qwen/Qwen3-0.6B",
      manifestPath: "https://cdn.example.test/models/qwen3/manifest.json",
      allowFixtureWeights: false,
      backendPreference: "cpu",
    });

    await expect(client.init()).rejects.toThrow("requires a 1D vector shape");
  });

  it("rejects shard descriptors whose byte range cannot satisfy the requested matrix shape", async () => {
    const shardBytes = makeF32Shard([1, 2]);
    const shardSha256 = await sha256Hex(shardBytes);
    vi.stubGlobal("fetch", async (input: RequestInfo | URL) => {
      const url = String(input);
      if (url.endsWith("/manifest.json")) {
        return new Response(JSON.stringify(makeShardedManifest({ shardSha256 })), {
          status: 200,
          headers: { "content-type": "application/json" },
        });
      }
      return new Response(shardBytes, {
        status: 200,
        headers: { "content-type": "application/octet-stream" },
      });
    });
    const client = new UnlockedBrowserTransformerClient({
      modelId: "Qwen/Qwen3-0.6B",
      manifestPath: "https://cdn.example.test/models/qwen3/manifest.json",
      allowFixtureWeights: false,
      backendPreference: "cpu",
    });

    await expect(client.init()).rejects.toThrow("does not contain enough f32 values");
  });

  it("rejects shard descriptors without integrity hashes", async () => {
    vi.stubGlobal("fetch", async () => new Response(JSON.stringify(makeShardedManifest({ shardSha256: "" })), {
      status: 200,
      headers: { "content-type": "application/json" },
    }));
    const client = new UnlockedBrowserTransformerClient({
      modelId: "Qwen/Qwen3-0.6B",
      manifestPath: "https://cdn.example.test/models/qwen3/manifest.json",
      allowFixtureWeights: false,
      backendPreference: "cpu",
    });

    await expect(client.init()).rejects.toThrow("sha256");
  });

  it("rejects shard descriptors with mismatched integrity hashes", async () => {
    vi.stubGlobal("fetch", async (input: RequestInfo | URL) => {
      const url = String(input);
      if (url.endsWith("/manifest.json")) {
        return new Response(JSON.stringify(makeShardedManifest({ shardSha256: "a".repeat(64) })), {
          status: 200,
          headers: { "content-type": "application/json" },
        });
      }
      return new Response(makeF32Shard([1, 2, 3, 4, 5, 6]), {
        status: 200,
        headers: { "content-type": "application/octet-stream" },
      });
    });
    const client = new UnlockedBrowserTransformerClient({
      modelId: "Qwen/Qwen3-0.6B",
      manifestPath: "https://cdn.example.test/models/qwen3/manifest.json",
      allowFixtureWeights: false,
      backendPreference: "cpu",
    });

    await expect(client.init()).rejects.toThrow("does not match expected SHA-256");
  });

  it("rejects unaligned shard byte offsets", async () => {
    vi.stubGlobal("fetch", async () => new Response(JSON.stringify(makeShardedManifest({
      shardSha256: "b".repeat(64),
      byteOffset: 1,
    })), {
      status: 200,
      headers: { "content-type": "application/json" },
    }));
    const client = new UnlockedBrowserTransformerClient({
      modelId: "Qwen/Qwen3-0.6B",
      manifestPath: "https://cdn.example.test/models/qwen3/manifest.json",
      allowFixtureWeights: false,
      backendPreference: "cpu",
    });

    await expect(client.init()).rejects.toThrow("byteOffset must align");
  });
});

async function collectStream(stream: AsyncGenerator<string, string, void>): Promise<string> {
  const chunks: string[] = [];
  for await (const chunk of stream) chunks.push(chunk);
  return chunks.join("");
}

function makeFixtureKvClient(namespace: string): UnlockedBrowserTransformerClient {
  return new UnlockedBrowserTransformerClient({
    modelId: "Qwen/Qwen3-0.6B",
    allowFixtureWeights: true,
    backendPreference: "cpu",
    maxGenerationTokens: 1,
    kvPersistence: {
      enabled: true,
      namespace,
      preferOpfs: false,
    },
  });
}

function makeSharedIndexedDbPersistence(): (options: BrowserKVSwapPersistenceOptions) => Promise<KVSwapPersistenceStore> {
  const stores = new Map<string, MemoryKVSwapPersistence>();
  return async (options) => {
    const namespace = options.namespace.trim() || "default";
    let store = stores.get(namespace);
    if (!store) {
      store = new MemoryKVSwapPersistence({ ...options, namespace });
      stores.set(namespace, store);
    }
    const withIndexedDbMode = (event: KVSwapPersistenceTraceEvent): KVSwapPersistenceTraceEvent => ({
      ...event,
      mode: "indexeddb",
    });
    return {
      mode: "indexeddb",
      save: async (block) => withIndexedDbMode(await store.save(block)),
      load: async (targetNamespace, blockId) => {
        const result = await store.load(targetNamespace, blockId);
        return { ...result, event: withIndexedDbMode(result.event) };
      },
      delete: async (targetNamespace, blockId) => withIndexedDbMode(await store.delete(targetNamespace, blockId)),
      list: async (targetNamespace) => {
        const result = await store.list(targetNamespace);
        return { ...result, event: withIndexedDbMode(result.event) };
      },
      persist: async (blocks) => withIndexedDbMode(await store.persist(blocks)),
      hydrate: async (targetNamespace) => {
        const result = await store.hydrate(targetNamespace);
        return { ...result, event: withIndexedDbMode(result.event) };
      },
      evict: async (targetNamespace, blockIds) => withIndexedDbMode(await store.evict(targetNamespace, blockIds)),
      clear: async (targetNamespace) => withIndexedDbMode(await store.clear(targetNamespace)),
      health: () => {
        const health = store.health();
        return {
          ...health,
          mode: "indexeddb",
          ...(health.lastOperation ? { lastOperation: withIndexedDbMode(health.lastOperation) } : {}),
        };
      },
    };
  };
}

class DelayedPersistKVSwapPersistence extends MemoryKVSwapPersistence {
  persistStarted = false;
  private resolvePersistCallback: (() => void) | null = null;

  async persist(blocks: Parameters<MemoryKVSwapPersistence["persist"]>[0]): Promise<KVSwapPersistenceTraceEvent> {
    this.persistStarted = true;
    await new Promise<void>((resolve) => {
      this.resolvePersistCallback = resolve;
    });
    return super.persist(blocks);
  }

  resolvePersist(): void {
    this.resolvePersistCallback?.();
  }
}

class DelayedLoadKVSwapPersistence extends MemoryKVSwapPersistence {
  loadStarted = false;
  loadSettled = true;
  private pendingLoadResolvers: Array<() => void> = [];
  private settleWaiters: Array<() => void> = [];

  get pendingLoadCount(): number {
    return this.pendingLoadResolvers.length;
  }

  async load(namespace: string, blockId: string): Promise<{ block: SerializedKVSwapBlock | null; event: KVSwapPersistenceTraceEvent }> {
    this.loadStarted = true;
    this.loadSettled = false;
    await new Promise<void>((resolve) => {
      this.pendingLoadResolvers.push(resolve);
    });
    const result = await super.load(namespace, blockId);
    if (this.pendingLoadResolvers.length === 0) {
      this.loadSettled = true;
      for (const waiter of this.settleWaiters.splice(0)) waiter();
    }
    return result;
  }

  resolveLoads(): void {
    for (const resolve of this.pendingLoadResolvers.splice(0)) resolve();
  }

  async waitForLoadsToSettle(): Promise<void> {
    if (this.loadSettled) return;
    await new Promise<void>((resolve) => {
      this.settleWaiters.push(resolve);
    });
  }
}

function makeIndexedDbPersistenceForStore(store: KVSwapPersistenceStore): () => Promise<KVSwapPersistenceStore> {
  const withIndexedDbMode = (event: KVSwapPersistenceTraceEvent): KVSwapPersistenceTraceEvent => ({
    ...event,
    mode: "indexeddb",
  });
  return async () => ({
    mode: "indexeddb",
    save: async (block) => withIndexedDbMode(await store.save(block)),
    load: async (targetNamespace, blockId) => {
      const result = await store.load(targetNamespace, blockId);
      return { ...result, event: withIndexedDbMode(result.event) };
    },
    delete: async (targetNamespace, blockId) => withIndexedDbMode(await store.delete(targetNamespace, blockId)),
    list: async (targetNamespace) => {
      const result = await store.list(targetNamespace);
      return { ...result, event: withIndexedDbMode(result.event) };
    },
    persist: async (blocks) => withIndexedDbMode(await store.persist(blocks)),
    hydrate: async (targetNamespace) => {
      const result = await store.hydrate(targetNamespace);
      return { ...result, event: withIndexedDbMode(result.event) };
    },
    evict: async (targetNamespace, blockIds) => withIndexedDbMode(await store.evict(targetNamespace, blockIds)),
    clear: async (targetNamespace) => withIndexedDbMode(await store.clear(targetNamespace)),
    health: () => {
      const health = store.health();
      return {
        ...health,
        mode: "indexeddb",
        ...(health.lastOperation ? { lastOperation: withIndexedDbMode(health.lastOperation) } : {}),
      };
    },
  });
}

function makeManifest(overrides: { tokenizer?: unknown; layerCount?: number; modelId?: string; vocabSize?: number } = {}) {
  const vocabSize = overrides.vocabSize ?? 4;
  const layer = {
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
    mlpUpProj: [
      [1, 0],
      [0, 1],
    ],
    mlpDownProj: [
      [1, 0],
      [0, 1],
    ],
  };
  return {
    schemaVersion: 1,
    modelId: overrides.modelId ?? "Qwen/Qwen3-0.6B",
    architecture: "qwen3_decoder_control",
    vocabSize,
    hiddenSize: 2,
    headDim: 2,
    tokenEmbedding: Array.from({ length: vocabSize }, (_value, index) => (
      [
        index === 0 ? 1 : index / vocabSize,
        index === 1 ? 1 : (vocabSize - index) / vocabSize,
      ]
    )),
    outputProjection: Array.from({ length: vocabSize }, (_value, index) => (
      index === 2 ? [10, 10] : [index / vocabSize, -index / vocabSize]
    )),
    layers: Array.from({ length: overrides.layerCount ?? 1 }, () => layer),
    ...(overrides.tokenizer === undefined
      ? {}
      : {
          tokenizer: overrides.tokenizer,
        }),
    ...(!("tokenizer" in overrides)
      ? {
          tokenizer: {
            kind: "vocab",
            tokens: Array.from({ length: vocabSize }, (_value, index) => (
              ["alpha", "beta", "MANIFEST_TOKEN", "delta"][index] ?? `tok${index}`
            )),
            unknownTokenId: 0,
          },
        }
      : {}),
  };
}

function makeRepeatingManifest() {
  const identity = [
    [1, 0],
    [0, 1],
  ];
  return {
    schemaVersion: 1,
    modelId: "Qwen/Qwen3-0.6B",
    architecture: "qwen3_decoder_control",
    vocabSize: 4,
    hiddenSize: 2,
    headDim: 2,
    tokenEmbedding: [
      [1, 1],
      [1, 1],
      [1, 1],
      [1, 1],
    ],
    outputProjection: [
      [0, 0],
      [10, 10],
      [9, 9],
      [1, 1],
    ],
    tokenizer: {
      kind: "vocab",
      tokens: ["unk", "m", "n", "."],
      unknownTokenId: 0,
    },
    layers: [
      {
        qProj: identity,
        kProj: identity,
        vProj: identity,
        oProj: identity,
      },
    ],
  };
}

function makeShardedManifest(options: { shardSha256?: string; byteOffset?: number; qwenMath?: boolean; vectorShape?: number[]; tensorFormat?: "f32" | "f16" } = {}) {
  const shardSha256 = options.shardSha256 ?? "b".repeat(64);
  const byteOffset = options.byteOffset;
  const vectorShape = options.vectorShape ?? [2];
  const tensorFormat = options.tensorFormat ?? "f32";
  const qwenMath = options.qwenMath
    ? {
        rmsNormEps: 0.000001,
        finalNorm: shardTensor(20, vectorShape, shardSha256, undefined, tensorFormat),
      }
    : {};
  const qwenLayerMath = options.qwenMath
    ? {
        inputLayerNorm: shardTensor(22, vectorShape, shardSha256, undefined, tensorFormat),
        qNorm: shardTensor(24, vectorShape, shardSha256, undefined, tensorFormat),
        kNorm: shardTensor(26, vectorShape, shardSha256, undefined, tensorFormat),
        postAttentionLayerNorm: shardTensor(28, vectorShape, shardSha256, undefined, tensorFormat),
        mlpGateProj: shardTensor(30, [3, 2], shardSha256, undefined, tensorFormat),
        mlpUpProj: shardTensor(36, [3, 2], shardSha256, undefined, tensorFormat),
        mlpDownProj: shardTensor(42, [2, 3], shardSha256, undefined, tensorFormat),
      }
    : {
        mlpUpProj: shardTensor(16, [2, 2], shardSha256, undefined, tensorFormat),
        mlpDownProj: shardTensor(16, [2, 2], shardSha256, undefined, tensorFormat),
      };
  return {
    schemaVersion: 1,
    modelId: "Qwen/Qwen3-0.6B",
    architecture: "qwen3_decoder_control",
    vocabSize: 4,
    hiddenSize: 2,
    headDim: 2,
    ...qwenMath,
    ...(tensorFormat === "f16"
      ? {
          tensorStorage: {
            format: "f16-packed",
            dtype: "f16",
            shardKind: "f16-shard",
            byteWidth: 2,
            productionTarget: "webgpu-packed",
            runtimeRepresentation: "packed-f16-runtime-lazy-decode",
            packedRuntimeCompute: false,
          },
        }
      : {}),
    tokenEmbedding: shardTensor(0, [4, 2], shardSha256, byteOffset, tensorFormat),
    outputProjection: shardTensor(8, [4, 2], shardSha256, undefined, tensorFormat),
    tokenizer: {
      kind: "vocab",
      tokens: ["alpha", "beta", "MANIFEST_TOKEN", "delta"],
      unknownTokenId: 0,
    },
    layers: [
      {
        qProj: shardTensor(16, [2, 2], shardSha256, undefined, tensorFormat),
        kProj: shardTensor(16, [2, 2], shardSha256, undefined, tensorFormat),
        vProj: shardTensor(16, [2, 2], shardSha256, undefined, tensorFormat),
        oProj: shardTensor(16, [2, 2], shardSha256, undefined, tensorFormat),
        ...qwenLayerMath,
      },
    ],
  };
}

function makeFullGqaShardedManifest(options: { shardSha256: string; mismatchedKvShape?: boolean }) {
  const kvShape = options.mismatchedKvShape ? [4, 2] : [2, 2];
  const vOffset = options.mismatchedKvShape ? 32 : 28;
  const oOffset = options.mismatchedKvShape ? 40 : 32;
  return {
    schemaVersion: 1,
    modelId: "Qwen/Qwen3-0.6B",
    architecture: "qwen3_decoder_control",
    vocabSize: 4,
    hiddenSize: 2,
    headDim: 2,
    numAttentionHeads: 2,
    numKeyValueHeads: 1,
    maxPositionEmbeddings: 128,
    ropeTheta: 10,
    tieWordEmbeddings: true,
    tokenEmbedding: shardTensor(0, [4, 2], options.shardSha256),
    outputProjection: shardTensor(8, [4, 2], options.shardSha256),
    tokenizer: {
      kind: "vocab",
      tokens: ["alpha", "beta", "MANIFEST_TOKEN", "delta"],
      unknownTokenId: 0,
    },
    layers: [
      {
        qProj: shardTensor(16, [4, 2], options.shardSha256),
        kProj: shardTensor(24, kvShape, options.shardSha256),
        vProj: shardTensor(vOffset, kvShape, options.shardSha256),
        oProj: shardTensor(oOffset, [2, 4], options.shardSha256),
      },
    ],
  };
}

function shardTensor(floatOffset: number, shape: number[], sha256: string, byteOffset?: number, tensorFormat: "f32" | "f16" = "f32") {
  return {
    kind: `${tensorFormat}-shard`,
    uri: "weights.bin",
    byteOffset: byteOffset ?? floatOffset * (tensorFormat === "f16" ? Uint16Array.BYTES_PER_ELEMENT : Float32Array.BYTES_PER_ELEMENT),
    shape,
    dtype: tensorFormat,
    ...(sha256 ? { sha256 } : {}),
  };
}

function makeF32Shard(values: number[]): ArrayBuffer {
  const array = new Float32Array(values);
  return array.buffer.slice(array.byteOffset, array.byteOffset + array.byteLength);
}

function makeF16Shard(values: number[]): ArrayBuffer {
  const array = new Uint16Array(values.map(float32ToFloat16Bits));
  return array.buffer.slice(array.byteOffset, array.byteOffset + array.byteLength);
}

function float32ToFloat16Bits(value: number): number {
  if (Number.isNaN(value)) return 0x7e00;
  if (value === Infinity) return 0x7c00;
  if (value === -Infinity) return 0xfc00;
  const sign = value < 0 || Object.is(value, -0) ? 0x8000 : 0;
  const abs = Math.abs(value);
  if (abs === 0) return sign;
  if (abs >= 65504) return sign | 0x7bff;
  if (abs < 2 ** -24) return sign;
  if (abs < 2 ** -14) return sign | Math.round(abs / 2 ** -24);
  const exponent = Math.floor(Math.log2(abs));
  const mantissa = Math.round((abs / 2 ** exponent - 1) * 1024);
  if (mantissa === 1024) return sign | ((exponent + 16) << 10);
  return sign | ((exponent + 15) << 10) | (mantissa & 0x03ff);
}

function flatten(matrix: number[][]): number[] {
  return matrix.flatMap((row) => row);
}

async function sha256Hex(value: ArrayBuffer): Promise<string> {
  const digest = await crypto.subtle.digest("SHA-256", value);
  return [...new Uint8Array(digest)].map((byte) => byte.toString(16).padStart(2, "0")).join("");
}
