import type { RawMemoryRecord } from "../types";
import { describe, expect, it } from "vitest";
import {
  buildAdvancedRuntimeGenerationPlan,
  buildRuntimeFeatureCapabilitySnapshot,
} from "./advancedRuntimeCoordinator";

describe("buildAdvancedRuntimeGenerationPlan", () => {
  it("keeps the current user request pinned and exposes Tier-0 fallback status", async () => {
    const plan = await buildAdvancedRuntimeGenerationPlan({
      requestId: "req_test",
      sessionId: "session_test",
      systemPrompt: "You are local.",
      userMessage: "Remember that Qwen3 0.6B is the target model.",
      recentMessages: [],
      retrievedMemory: [
        {
          id: "mem_1",
          text: "Use LanceDB as primary memory when the sidecar is available.",
          embedding: [1, 0],
          score: 0.9,
          sessionId: "session_test",
          source: "chat",
          createdAt: "2026-05-11T00:00:00.000Z",
          updatedAt: "2026-05-11T00:00:00.000Z",
          tags: ["architecture"],
          metadata: {},
          tokenCount: 16,
        },
      ],
      config: {
        modelId: "Qwen/Qwen3-0.6B",
        embeddingModelId: "Xenova/all-MiniLM-L6-v2",
        memoryTopK: 8,
        maxRetrievedMemoryTokens: 512,
        maxRecentConversationTokens: 512,
        maxPromptTokens: 4096,
      },
      backend: {
        id: "unlocked-browser-transformer",
        label: "Unlocked Qwen3 0.6B",
        mode: "custom",
      },
      model: {
        modelId: "Qwen/Qwen3-0.6B",
        parameterBytes: 4_000_000_000,
        effectiveParameterCount: 1_910_000_000,
        layers: 30,
        hiddenSize: 2048,
        kvHeads: 8,
        contextWindowTokens: 32_000,
      },
      device: {
        name: "test-device",
        vramBudgetBytes: 8_000_000_000,
        ramBudgetBytes: 16_000_000_000,
      },
      memoryMode: "indexeddb",
      memoryStore: makeContextTraceStore(),
      now: new Date("2026-05-11T12:00:00.000Z"),
    });

    expect(plan.packed.includedMemoryIds).toEqual(["mem_1"]);
    expect(plan.contextPlan.pinnedAnchorIds).toEqual(["req_test:current_user"]);
    expect(plan.ssaPlan.pinnedBlockIds).toContain("req_test:current_user");
    expect(plan.trace.runtime).toMatchObject({
      backend: { mode: "custom" },
      mtp: { mode: "target_only", targetModelId: "Qwen/Qwen3-0.6B" },
    });
    expect(plan.features.map((feature) => feature.name)).toEqual([
      "contextRuntime",
      "inferenceBackend",
      "kvswap",
      "memoryProvider",
      "mtp",
      "ssa",
      "tsp",
    ]);
    expect(plan.features.find((feature) => feature.name === "memoryProvider")).toMatchObject({
      state: "fallback",
      mode: "indexeddb",
    });
  });

  it("reports remote HTTP as the active production memory provider", async () => {
    const plan = await buildAdvancedRuntimeGenerationPlan({
      requestId: "req_remote",
      sessionId: "session_test",
      systemPrompt: "You are local.",
      userMessage: "Use a remote memory API.",
      recentMessages: [],
      retrievedMemory: [],
      config: {
        modelId: "Qwen/Qwen3-0.6B",
        embeddingModelId: "Xenova/all-MiniLM-L6-v2",
        memoryTopK: 8,
        maxRetrievedMemoryTokens: 512,
        maxRecentConversationTokens: 512,
        maxPromptTokens: 4096,
      },
      backend: {
        id: "unlocked-browser-transformer",
        label: "Unlocked Qwen3 0.6B",
        mode: "custom",
      },
      model: {
        modelId: "Qwen/Qwen3-0.6B",
        parameterBytes: 4_000_000_000,
        effectiveParameterCount: 1_910_000_000,
        layers: 30,
        hiddenSize: 2048,
        kvHeads: 8,
        contextWindowTokens: 32_000,
      },
      device: {
        name: "test-device",
        vramBudgetBytes: 8_000_000_000,
        ramBudgetBytes: 16_000_000_000,
      },
      memoryMode: "remote-http",
      memoryStore: makeContextTraceStore(),
      now: new Date("2026-05-11T12:00:00.000Z"),
    });

    expect(plan.features.find((feature) => feature.name === "memoryProvider")).toMatchObject({
      state: "enabled",
      mode: "remote-http",
      reason: "Remote HTTP memory provider is active.",
    });
  });

  it("reports browser-vector as a production-ready browser-native memory provider", async () => {
    const plan = await buildAdvancedRuntimeGenerationPlan({
      requestId: "req_browser_vector",
      sessionId: "session_test",
      systemPrompt: "You are local.",
      userMessage: "Use browser-native memory.",
      recentMessages: [],
      retrievedMemory: [],
      config: {
        modelId: "Qwen/Qwen3-0.6B",
        embeddingModelId: "Xenova/all-MiniLM-L6-v2",
        memoryTopK: 8,
        maxRetrievedMemoryTokens: 512,
        maxRecentConversationTokens: 512,
        maxPromptTokens: 4096,
      },
      backend: {
        id: "unlocked-browser-transformer",
        label: "Unlocked Qwen3 0.6B",
        mode: "custom",
      },
      model: {
        modelId: "Qwen/Qwen3-0.6B",
        parameterBytes: 4_000_000_000,
        effectiveParameterCount: 1_910_000_000,
        layers: 30,
        hiddenSize: 2048,
        kvHeads: 8,
        contextWindowTokens: 32_000,
      },
      device: {
        name: "test-device",
        vramBudgetBytes: 8_000_000_000,
        ramBudgetBytes: 16_000_000_000,
      },
      memoryMode: "browser-vector",
      memoryStore: makeContextTraceStore(),
      now: new Date("2026-05-11T12:00:00.000Z"),
    });

    expect(plan.features.find((feature) => feature.name === "memoryProvider")).toMatchObject({
      state: "enabled",
      mode: "browser-vector",
      reason: "Browser-native vector memory provider is active.",
      metrics: {
        storage: "indexeddb",
        localOnly: true,
        vectorSearch: true,
        deterministicSearch: true,
        metadataFilters: true,
        vectorDimension: 384,
        persistent: true,
        importExport: true,
        contextPackTracePersistence: true,
        remoteSync: false,
      },
    });
  });

  it("reports unlocked browser transformer capability as the production SSA/KV/TSP path", async () => {
    const plan = await buildAdvancedRuntimeGenerationPlan({
      requestId: "req_unlocked",
      sessionId: "session_test",
      systemPrompt: "You are local.",
      userMessage: "Use full control browser tensors.",
      recentMessages: [],
      retrievedMemory: [],
      config: {
        modelId: "Qwen/Qwen3-0.6B",
        embeddingModelId: "Xenova/all-MiniLM-L6-v2",
        memoryTopK: 8,
        maxRetrievedMemoryTokens: 512,
        maxRecentConversationTokens: 512,
        maxPromptTokens: 4096,
      },
      backend: {
        id: "unlocked-browser-transformer",
        label: "Unlocked Browser Transformer",
        mode: "custom",
        capabilities: {
          qkvAccess: true,
          layerSparseRouting: true,
          pinnedKvBlocks: true,
          kvTensorPaging: true,
          tspScheduleExecution: true,
        },
      },
      model: {
        modelId: "Qwen/Qwen3-0.6B",
        parameterBytes: 1_200_000_000,
        effectiveParameterCount: 600_000_000,
        layers: 28,
        hiddenSize: 1024,
        kvHeads: 8,
        contextWindowTokens: 32_000,
      },
      device: {
        name: "browser-webgpu-unlocked",
        vramBudgetBytes: 4_000_000_000,
        ramBudgetBytes: 8_000_000_000,
        backend: "webgpu",
      },
      memoryMode: "indexeddb",
      memoryStore: makeContextTraceStore(),
      now: new Date("2026-05-11T12:00:00.000Z"),
    });

    expect(plan.features.find((feature) => feature.name === "inferenceBackend")).toMatchObject({
      state: "enabled",
      mode: "custom",
      reason: "Full-control browser transformer backend owns model tensors and runtime scheduling.",
    });
    expect(plan.features.find((feature) => feature.name === "ssa")).toMatchObject({
      state: "enabled",
      mode: "backend_native_sparse_execution",
      reason: "Backend exposes Q/K/V tensors for sparse layer execution.",
    });
    expect(plan.features.find((feature) => feature.name === "kvswap")).toMatchObject({
      state: "enabled",
      mode: "tensor_paging",
      reason: "Backend owns KV tensor handles and can page selected blocks before sparse attention.",
    });
    expect(plan.features.find((feature) => feature.name === "tsp")).toMatchObject({
      state: "enabled",
      mode: "backend_schedule_execution",
      reason: "Backend executes planner-emitted sequence/tensor schedule callbacks.",
    });
    expect(plan.features.find((feature) => feature.name === "mtp")).toMatchObject({
      state: "fallback",
      mode: "target_only",
      reason: "No compatible draft model is configured yet.",
    });
  });

  it("reports MTP enabled only with a compatible configured draft and verifier-capable backend", async () => {
    const base = {
      requestId: "req_unlocked_mtp",
      sessionId: "session_test",
      systemPrompt: "You are local.",
      userMessage: "Use browser verifier speculation.",
      recentMessages: [],
      retrievedMemory: [],
      config: {
        modelId: "Qwen/Qwen3-0.6B",
        embeddingModelId: "Xenova/all-MiniLM-L6-v2",
        memoryTopK: 8,
        maxRetrievedMemoryTokens: 512,
        maxRecentConversationTokens: 512,
        maxPromptTokens: 4096,
        mtp: {
          enabled: true,
          draftModelId: "browser/ngram-drafter",
          numSpeculativeTokens: 6,
          minAcceptanceRate: 0.5,
          disableWhenLatencyWorse: true,
          targetTokenizerId: "qwen3",
          draftModelProfiles: [
            {
              modelId: "browser/ngram-drafter",
              role: "draft" as const,
              tokenizerId: "qwen3",
              maxSpeculativeTokens: 4,
              targetModelIds: ["Qwen/Qwen3-0.6B"],
            },
          ],
        },
      },
      model: {
        modelId: "Qwen/Qwen3-0.6B",
        parameterBytes: 1_200_000_000,
        effectiveParameterCount: 600_000_000,
        layers: 28,
        hiddenSize: 1024,
        kvHeads: 8,
        contextWindowTokens: 32_000,
      },
      device: {
        name: "browser-webgpu-unlocked",
        vramBudgetBytes: 4_000_000_000,
        ramBudgetBytes: 8_000_000_000,
        backend: "webgpu" as const,
      },
      memoryMode: "indexeddb" as const,
      memoryStore: makeContextTraceStore(),
      now: new Date("2026-05-11T12:00:00.000Z"),
    };
    const verifierCapable = await buildAdvancedRuntimeGenerationPlan({
      ...base,
      backend: {
        id: "unlocked-browser-transformer",
        label: "Unlocked Browser Transformer",
        mode: "custom",
        capabilities: {
          qkvAccess: true,
          layerSparseRouting: true,
          pinnedKvBlocks: true,
          kvTensorPaging: true,
          tspScheduleExecution: true,
          speculativeVerifierBatching: true,
        },
      },
    });
    const missingVerifier = await buildAdvancedRuntimeGenerationPlan({
      ...base,
      requestId: "req_unlocked_mtp_no_verifier",
      backend: {
        id: "unlocked-browser-transformer",
        label: "Unlocked Browser Transformer",
        mode: "custom",
        capabilities: {
          qkvAccess: true,
          layerSparseRouting: true,
          pinnedKvBlocks: true,
          kvTensorPaging: true,
          tspScheduleExecution: true,
        },
      },
    });

    expect(verifierCapable.speculativeConfig).toMatchObject({
      enabled: true,
      mode: "draft_verify",
      draftModelId: "browser/ngram-drafter",
      targetModelId: "Qwen/Qwen3-0.6B",
      numSpeculativeTokens: 4,
      minAcceptanceRate: 0.5,
      disableWhenLatencyWorse: true,
    });
    expect(verifierCapable.trace.runtime.mtp).toMatchObject({
      mode: "draft_verify",
      draftModelId: "browser/ngram-drafter",
      numSpeculativeTokens: 4,
    });
    expect(verifierCapable.features.find((feature) => feature.name === "mtp")).toMatchObject({
      state: "enabled",
      mode: "verifier_batching",
    });
    expect(missingVerifier.speculativeConfig).toMatchObject({
      mode: "target_only",
      draftModelId: "browser/ngram-drafter",
      numSpeculativeTokens: 0,
    });
    expect(missingVerifier.features.find((feature) => feature.name === "mtp")).toMatchObject({
      state: "fallback",
      mode: "target_only",
      reason: "Backend does not expose batched target verification for draft branches.",
    });
  });

  it("reports unlocked startup capability status before the first decode trace", () => {
    const features = buildRuntimeFeatureCapabilitySnapshot({
      backend: {
        id: "unlocked-browser-transformer",
        label: "Unlocked Browser Transformer",
        mode: "custom",
        capabilities: {
          qkvAccess: true,
          layerSparseRouting: true,
          pinnedKvBlocks: true,
          kvTensorPaging: true,
          tspScheduleExecution: true,
          speculativeVerifierBatching: true,
        },
      },
      model: {
        modelId: "Qwen/Qwen3-0.6B",
        parameterBytes: 1_200_000_000,
        effectiveParameterCount: 600_000_000,
        layers: 28,
        hiddenSize: 1024,
        kvHeads: 8,
        contextWindowTokens: 40_960,
      },
      memoryMode: "browser-vector",
      mtp: {
        enabled: true,
        draftModelId: "browser/ngram-drafter",
        numSpeculativeTokens: 4,
        minAcceptanceRate: 0.45,
        disableWhenLatencyWorse: true,
        targetTokenizerId: "qwen3",
        draftModelProfiles: [
          {
            modelId: "browser/ngram-drafter",
            role: "draft",
            tokenizerId: "qwen3",
            maxSpeculativeTokens: 4,
            targetModelIds: ["Qwen/Qwen3-0.6B"],
          },
        ],
      },
    });

    expect(features.find((feature) => feature.name === "memoryProvider")).toMatchObject({
      state: "enabled",
      mode: "browser-vector",
    });
    expect(features.find((feature) => feature.name === "ssa")).toMatchObject({
      state: "enabled",
      mode: "backend_native_sparse_execution",
    });
    expect(features.find((feature) => feature.name === "tsp")).toMatchObject({
      state: "enabled",
      mode: "backend_schedule_execution",
    });
    expect(features.find((feature) => feature.name === "kvswap")).toMatchObject({
      state: "enabled",
      mode: "tensor_paging",
    });
    expect(features.find((feature) => feature.name === "mtp")).toMatchObject({
      state: "enabled",
      mode: "verifier_batching",
    });
  });

  it("writes a non-blocking GAC context pack trace when the memory store exposes the capability", async () => {
    const writes: unknown[] = [];
    const plan = await buildAdvancedRuntimeGenerationPlan({
      requestId: "req_gac",
      sessionId: "session_test",
      systemPrompt: "You are local.",
      userMessage: "Use persisted GAC traces.",
      recentMessages: [],
      retrievedMemory: [
        {
          id: "mem_1",
          text: "Context packs must store provenance.",
          embedding: [1, 0],
          score: 0.9,
          sessionId: "session_test",
          source: "chat",
          createdAt: "2026-05-11T00:00:00.000Z",
          updatedAt: "2026-05-11T00:00:00.000Z",
          tags: ["gac"],
          metadata: {},
          tokenCount: 8,
        },
      ],
      config: {
        modelId: "Qwen/Qwen3-0.6B",
        embeddingModelId: "Xenova/all-MiniLM-L6-v2",
        memoryTopK: 8,
        maxRetrievedMemoryTokens: 512,
        maxRecentConversationTokens: 512,
        maxPromptTokens: 4096,
      },
      backend: {
        id: "unlocked-browser-transformer",
        label: "Unlocked Qwen3 0.6B",
        mode: "custom",
      },
      model: {
        modelId: "Qwen/Qwen3-0.6B",
        parameterBytes: 4_000_000_000,
        effectiveParameterCount: 1_910_000_000,
        layers: 30,
        hiddenSize: 2048,
        kvHeads: 8,
        contextWindowTokens: 32_000,
      },
      device: {
        name: "test-device",
        vramBudgetBytes: 8_000_000_000,
        ramBudgetBytes: 16_000_000_000,
      },
      memoryMode: "remote-http",
      memoryStore: {
        writeContextPackTraces: async (records) => {
          writes.push(records);
          return { ok: true, count: records.length, traceId: records[0]?.traceId ?? "trace_missing" };
        },
        listContextPackTraces: async () => [],
      },
      now: new Date("2026-05-11T12:00:00.000Z"),
    });

    expect(writes).toEqual([
      [
        expect.objectContaining({
          traceId: plan.trace.traceId,
          sessionId: "session_test",
          queryId: "req_gac",
          rawMemoryIds: ["mem_1"],
          includedMemoryIds: ["mem_1"],
          tokenBudget: 4096,
          packingStrategy: "advanced-runtime",
        }),
      ],
    ]);
    expect(plan.trace.runtime.context).toMatchObject({
      contextPackTraceId: expect.any(String),
      contextPackTraceWrite: "ok",
    });
  });

  it("requires a context-pack trace store for model calls", async () => {
    await expect(buildAdvancedRuntimeGenerationPlan({
      requestId: "req_gac_missing_store",
      sessionId: "session_test",
      systemPrompt: "You are local.",
      userMessage: "Generation must not run without context-pack trace persistence.",
      recentMessages: [],
      retrievedMemory: [],
      config: {
        modelId: "Qwen/Qwen3-0.6B",
        embeddingModelId: "Xenova/all-MiniLM-L6-v2",
        memoryTopK: 8,
        maxRetrievedMemoryTokens: 512,
        maxRecentConversationTokens: 512,
        maxPromptTokens: 4096,
      },
      backend: {
        id: "unlocked-browser-transformer",
        label: "Unlocked Qwen3 0.6B",
        mode: "custom",
      },
      model: {
        modelId: "Qwen/Qwen3-0.6B",
        parameterBytes: 4_000_000_000,
        effectiveParameterCount: 1_910_000_000,
        layers: 30,
        hiddenSize: 2048,
        kvHeads: 8,
        contextWindowTokens: 32_000,
      },
      device: {
        name: "test-device",
        vramBudgetBytes: 8_000_000_000,
        ramBudgetBytes: 16_000_000_000,
      },
      memoryMode: "remote-http",
      now: new Date("2026-05-11T12:00:00.000Z"),
    })).rejects.toThrow("GAC context pack trace persistence is required");
  });

  it("blocks generation when GAC context pack trace persistence fails", async () => {
    await expect(buildAdvancedRuntimeGenerationPlan({
      requestId: "req_gac_warn",
      sessionId: "session_test",
      systemPrompt: "You are local.",
      userMessage: "Generation must continue.",
      recentMessages: [],
      retrievedMemory: [],
      config: {
        modelId: "Qwen/Qwen3-0.6B",
        embeddingModelId: "Xenova/all-MiniLM-L6-v2",
        memoryTopK: 8,
        maxRetrievedMemoryTokens: 512,
        maxRecentConversationTokens: 512,
        maxPromptTokens: 4096,
      },
      backend: {
        id: "unlocked-browser-transformer",
        label: "Unlocked Qwen3 0.6B",
        mode: "custom",
      },
      model: {
        modelId: "Qwen/Qwen3-0.6B",
        parameterBytes: 4_000_000_000,
        effectiveParameterCount: 1_910_000_000,
        layers: 30,
        hiddenSize: 2048,
        kvHeads: 8,
        contextWindowTokens: 32_000,
      },
      device: {
        name: "test-device",
        vramBudgetBytes: 8_000_000_000,
        ramBudgetBytes: 16_000_000_000,
      },
      memoryMode: "remote-http",
      memoryStore: {
        writeContextPackTraces: async () => {
          throw new Error("trace write unavailable");
        },
        listContextPackTraces: async () => [],
      },
      now: new Date("2026-05-11T12:00:00.000Z"),
    })).rejects.toThrow("GAC context pack trace persistence failed: trace write unavailable");
  });

  it("drops model-visible representatives that do not carry raw lineage metadata", async () => {
    const writes: unknown[] = [];
    const plan = await buildAdvancedRuntimeGenerationPlan({
      requestId: "req_rep_guard",
      sessionId: "session_test",
      systemPrompt: "You are local.",
      userMessage: "Use only grounded memory.",
      recentMessages: [],
      retrievedMemory: [
        {
          id: "rep_missing_lineage",
          text: "Unsafe representative should not be shown.",
          embedding: [1, 0],
          score: 0.99,
          sessionId: "session_test",
          source: "summary",
          createdAt: "2026-05-11T00:00:00.000Z",
          updatedAt: "2026-05-11T00:00:00.000Z",
          tags: ["gac"],
          metadata: {
            representativeId: "rep_1",
            modelVisible: true,
            factual: true,
          },
          tokenCount: 8,
        },
        {
          id: "raw_1",
          text: "Grounded raw memory should be shown.",
          embedding: [1, 0],
          score: 0.9,
          sessionId: "session_test",
          source: "chat",
          createdAt: "2026-05-11T00:00:00.000Z",
          updatedAt: "2026-05-11T00:00:00.000Z",
          tags: ["gac"],
          metadata: { rawMemoryId: "raw_1", memoryClass: "HIGH_RISK_RAW" },
          tokenCount: 8,
        },
      ],
      config: {
        modelId: "Qwen/Qwen3-0.6B",
        embeddingModelId: "Xenova/all-MiniLM-L6-v2",
        memoryTopK: 8,
        maxRetrievedMemoryTokens: 512,
        maxRecentConversationTokens: 512,
        maxPromptTokens: 4096,
      },
      backend: {
        id: "unlocked-browser-transformer",
        label: "Unlocked Qwen3 0.6B",
        mode: "custom",
      },
      model: {
        modelId: "Qwen/Qwen3-0.6B",
        parameterBytes: 4_000_000_000,
        effectiveParameterCount: 1_910_000_000,
        layers: 30,
        hiddenSize: 2048,
        kvHeads: 8,
        contextWindowTokens: 32_000,
      },
      device: {
        name: "test-device",
        vramBudgetBytes: 8_000_000_000,
        ramBudgetBytes: 16_000_000_000,
      },
      memoryMode: "indexeddb",
      memoryStore: {
        writeContextPackTraces: async (records) => {
          writes.push(records);
          return { ok: true, count: records.length, traceId: records[0]?.traceId ?? "trace_missing" };
        },
        listContextPackTraces: async () => [],
      },
      now: new Date("2026-05-11T12:00:00.000Z"),
    });

    expect(plan.packed.includedMemoryIds).toEqual(["raw_1"]);
    expect(plan.packed.messages[0]?.content).not.toContain("Unsafe representative should not be shown.");
    expect(plan.trace.runtime.context).toMatchObject({
      gacDropReasons: {
        rep_missing_lineage: ["representative_missing_lineage"],
      },
    });
    expect(writes).toEqual([
      [
        expect.objectContaining({
          rawMemoryIds: ["raw_1"],
          includedMemoryIds: ["raw_1"],
          omittedMemoryIds: expect.arrayContaining(["rep_missing_lineage"]),
        }),
      ],
    ]);
  });

  it("carries GAC metadata into SSA/KVSwap planner inputs and persisted context-pack traces", async () => {
    const writes: unknown[] = [];
    const plan = await buildAdvancedRuntimeGenerationPlan({
      requestId: "req_gac_metadata",
      sessionId: "session_test",
      systemPrompt: "You are local.",
      userMessage: "Prioritize pinned facts.",
      recentMessages: [],
      retrievedMemory: [
        {
          id: "pin_1",
          text: "Pinned exact memory.",
          embedding: [1, 0],
          score: 0.95,
          sessionId: "session_test",
          source: "chat",
          createdAt: "2026-05-11T00:00:00.000Z",
          updatedAt: "2026-05-11T00:00:00.000Z",
          tags: ["gac"],
          metadata: {
            rawMemoryId: "raw_pin_1",
            identityPinId: "identity_pin_1",
            memoryClass: "PINNED_EXACT",
            identityRisk: 0.98,
            pinStrength: 1,
            sourceTrust: 0.91,
            mustAttend: true,
          },
          tokenCount: 8,
        },
      ],
      config: {
        modelId: "Qwen/Qwen3-0.6B",
        embeddingModelId: "Xenova/all-MiniLM-L6-v2",
        memoryTopK: 8,
        maxRetrievedMemoryTokens: 512,
        maxRecentConversationTokens: 512,
        maxPromptTokens: 4096,
      },
      backend: {
        id: "unlocked-browser-transformer",
        label: "Unlocked Qwen3 0.6B",
        mode: "custom",
      },
      model: {
        modelId: "Qwen/Qwen3-0.6B",
        parameterBytes: 4_000_000_000,
        effectiveParameterCount: 1_910_000_000,
        layers: 30,
        hiddenSize: 2048,
        kvHeads: 8,
        contextWindowTokens: 32_000,
      },
      device: {
        name: "test-device",
        vramBudgetBytes: 8_000_000_000,
        ramBudgetBytes: 16_000_000_000,
      },
      memoryMode: "remote-http",
      memoryStore: {
        writeContextPackTraces: async (records) => {
          writes.push(records);
          return { ok: true, count: records.length, traceId: records[0]?.traceId ?? "trace_missing" };
        },
        listContextPackTraces: async () => [],
      },
      now: new Date("2026-05-11T12:00:00.000Z"),
    });

    expect(plan.ssaPlan.routingReasons.pin_1).toEqual(expect.arrayContaining([
      "gac_memory_class:PINNED_EXACT",
      "gac_must_attend",
    ]));
    expect(plan.kvswapDecision.reasons.kv_pin_1).toEqual(expect.arrayContaining([
      "gac_tier:PIN_HOT",
      "gac:must_attend",
      "prediction:gac",
    ]));
    expect(plan.trace.runtime.predictive).toMatchObject({
      requestId: "req_gac_metadata",
      kvHotPages: expect.arrayContaining([
        expect.objectContaining({
          blockId: "kv_pin_1",
          tier: "PIN_HOT",
        }),
      ]),
      predictedRetrievals: expect.arrayContaining([
        expect.objectContaining({
          expectedRawMemoryIds: ["raw_pin_1"],
        }),
      ]),
    });
    expect(plan.kvswapDecision.predictivePrefetchBlockIds).toEqual(expect.arrayContaining(["kv_pin_1"]));
    expect(writes).toEqual([
      [
        expect.objectContaining({
          predictivePlanId: "pred_req_gac_metadata",
          ssaRoutingBlocks: [
            expect.objectContaining({
              blockId: "pin_1",
              memoryClass: "PINNED_EXACT",
              rawMemoryId: "raw_pin_1",
              identityRisk: 0.98,
              pinStrength: 1,
              sourceTrust: 0.91,
              mustAttend: true,
            }),
          ],
          kvSwapPriorities: [
            expect.objectContaining({
              blockId: "kv_pin_1",
              tier: "PIN_HOT",
              priorityScore: 1,
              reasonCodes: expect.arrayContaining(["identity_pin", "must_attend"]),
            }),
          ],
        }),
      ],
    ]);
  });

  it("uses prior traces, identity pins, and retrieval audits for learned context rebuilding", async () => {
    const writes: unknown[] = [];
    const plan = await buildAdvancedRuntimeGenerationPlan({
      requestId: "req_learned_rebuild",
      tenantId: "tenant_1",
      cellId: "cell_1",
      sessionId: "session_test",
      systemPrompt: "You are local.",
      userMessage: "Recover the durable constraints.",
      recentMessages: [],
      retrievedMemory: [
        {
          id: "ordinary_high",
          text: "Ordinary high-score memory that should not outrank identity repair.",
          embedding: [1, 0],
          score: 0.5,
          sessionId: "session_test",
          source: "chat",
          createdAt: "2026-05-11T00:00:00.000Z",
          updatedAt: "2026-05-11T00:00:00.000Z",
          tags: ["ordinary"],
          metadata: {},
          tokenCount: 10,
        },
        {
          id: "pin_low",
          text: "Pinned exact memory: never compress raw identity constraints.",
          embedding: [0.9, 0.1],
          score: 0.05,
          sessionId: "session_test",
          source: "chat",
          createdAt: "2026-05-11T00:00:00.000Z",
          updatedAt: "2026-05-11T00:00:00.000Z",
          tags: ["gac"],
          metadata: {
            rawMemoryId: "raw_pin",
            identityPinId: "pin_1",
            memoryClass: "PINNED_EXACT",
            mustAttend: true,
            pinStrength: 1,
            identityRisk: 0.99,
          },
          tokenCount: 10,
        },
        {
          id: "missed_raw",
          text: "Raw memory that failed a prior identity retrieval audit.",
          embedding: [0.8, 0.2],
          score: 0.2,
          sessionId: "session_test",
          source: "chat",
          createdAt: "2026-05-11T00:00:00.000Z",
          updatedAt: "2026-05-11T00:00:00.000Z",
          tags: ["gac"],
          metadata: {
            rawMemoryId: "raw_failed",
            memoryClass: "HIGH_RISK_RAW",
            identityRisk: 0.7,
          },
          tokenCount: 10,
        },
      ],
      config: {
        modelId: "Qwen/Qwen3-0.6B",
        embeddingModelId: "Xenova/all-MiniLM-L6-v2",
        memoryTopK: 8,
        maxRetrievedMemoryTokens: 20,
        maxRecentConversationTokens: 0,
        maxPromptTokens: 4096,
      },
      backend: {
        id: "unlocked-browser-transformer",
        label: "Unlocked Qwen3 0.6B",
        mode: "custom",
      },
      model: {
        modelId: "Qwen/Qwen3-0.6B",
        parameterBytes: 4_000_000_000,
        effectiveParameterCount: 1_910_000_000,
        layers: 30,
        hiddenSize: 2048,
        kvHeads: 8,
        contextWindowTokens: 32_000,
      },
      device: {
        name: "test-device",
        vramBudgetBytes: 8_000_000_000,
        ramBudgetBytes: 16_000_000_000,
      },
      memoryMode: "remote-http",
      memoryStore: {
        writeContextPackTraces: async (records) => {
          writes.push(records);
          return { ok: true, count: records.length, traceId: records[0]?.traceId ?? "trace_missing" };
        },
        listContextPackTraces: async () => [{
          id: "ctx_prior",
          traceId: "trace_prior",
          tenantId: "tenant_1",
          cellId: "cell_1",
          sessionId: "session_test",
          queryId: "req_prior",
          contextPackId: "pack_prior",
          rawMemoryIds: ["raw_failed"],
          representativeIds: [],
          identityPinIds: [],
          tokenBudget: 4096,
          packingStrategy: "advanced-runtime",
          includedMemoryIds: ["missed_raw"],
          createdAt: "2026-05-11T00:00:00.000Z",
        }],
        listIdentityPins: async () => [{
          id: "pin_1",
          tenantId: "tenant_1",
          cellId: "cell_1",
          sessionId: "session_test",
          rawMemoryId: "raw_pin",
          pinReason: "user_instruction",
          pinStrength: 1,
          createdBy: "policy",
          createdAt: "2026-05-11T00:00:00.000Z",
        }],
        listRetrievalAudits: async () => [{
          id: "audit_1",
          tenantId: "tenant_1",
          cellId: "cell_1",
          sessionId: "session_test",
          queryText: "Retrieve raw_failed",
          expectedRawMemoryId: "raw_failed",
          retrievedRawMemoryIds: ["ordinary_high"],
          retrievedRepresentativeIds: [],
          identityPreserved: false,
          failureMode: "over_pruned",
          createdAt: "2026-05-11T00:00:00.000Z",
        }],
      },
      now: new Date("2026-05-11T12:00:00.000Z"),
    });

    expect(plan.packed.includedMemoryIds).toEqual(["pin_low", "missed_raw"]);
    expect(plan.contextPlan.droppedFrameIds).toContain("ordinary_high");
    expect(plan.ssaPlan.selectedBlockIds).not.toContain("ordinary_high");
    expect(plan.kvswapDecision.reasons).not.toHaveProperty("kv_ordinary_high");
    expect(plan.trace.runtime.context).toMatchObject({
      contextRebuildLearning: {
        contextTraceCount: 1,
        retrievalAuditCount: 1,
        boostedMemoryIds: expect.arrayContaining(["pin_low", "missed_raw"]),
        protectedMemoryIds: ["pin_low"],
      },
      memoryPriorityMap: {
        pin_low: expect.objectContaining({
          protected: true,
          reasons: expect.arrayContaining(["identity_pin", "must_attend"]),
        }),
        missed_raw: expect.objectContaining({
          reasons: expect.arrayContaining(["retrieval_audit_failure_repair", "prior_context_inclusion"]),
        }),
      },
    });
    expect(writes).toEqual([
      [
        expect.objectContaining({
          includedMemoryIds: ["pin_low", "missed_raw"],
          rawMemoryIds: expect.arrayContaining(["raw_pin", "raw_failed"]),
          identityPinIds: ["pin_1"],
        }),
      ],
    ]);
  });

  it("recovers exact pinned and failed-audit raw memories even when vector search misses them", async () => {
    const writes: unknown[] = [];
    const rawReads: unknown[] = [];
    const plan = await buildAdvancedRuntimeGenerationPlan({
      requestId: "req_raw_recovery",
      tenantId: "tenant_1",
      cellId: "cell_1",
      sessionId: "session_test",
      systemPrompt: "You are local.",
      userMessage: "Recover missed durable memory.",
      recentMessages: [],
      retrievedMemory: [
        {
          id: "ordinary_only",
          text: "Vector search returned only an ordinary memory.",
          embedding: [1, 0],
          score: 0.75,
          sessionId: "session_test",
          source: "chat",
          createdAt: "2026-05-11T00:00:00.000Z",
          updatedAt: "2026-05-11T00:00:00.000Z",
          tags: ["ordinary"],
          metadata: {},
          tokenCount: 10,
        },
      ],
      config: {
        modelId: "Qwen/Qwen3-0.6B",
        embeddingModelId: "Xenova/all-MiniLM-L6-v2",
        memoryTopK: 8,
        maxRetrievedMemoryTokens: 20,
        maxRecentConversationTokens: 0,
        maxPromptTokens: 4096,
      },
      backend: {
        id: "unlocked-browser-transformer",
        label: "Unlocked Qwen3 0.6B",
        mode: "custom",
      },
      model: {
        modelId: "Qwen/Qwen3-0.6B",
        parameterBytes: 4_000_000_000,
        effectiveParameterCount: 1_910_000_000,
        layers: 30,
        hiddenSize: 2048,
        kvHeads: 8,
        contextWindowTokens: 32_000,
      },
      device: {
        name: "test-device",
        vramBudgetBytes: 8_000_000_000,
        ramBudgetBytes: 16_000_000_000,
      },
      memoryMode: "remote-http",
      memoryStore: {
        writeContextPackTraces: async (records) => {
          writes.push(records);
          return { ok: true, count: records.length, traceId: records[0]?.traceId ?? "trace_missing" };
        },
        listContextPackTraces: async () => [],
        listIdentityPins: async () => [{
          id: "pin_1",
          tenantId: "tenant_1",
          cellId: "cell_1",
          sessionId: "session_test",
          rawMemoryId: "raw_pin",
          pinReason: "user_instruction",
          pinStrength: 1,
          createdBy: "policy",
          createdAt: "2026-05-11T00:00:00.000Z",
        }],
        listRetrievalAudits: async () => [{
          id: "audit_1",
          tenantId: "tenant_1",
          cellId: "cell_1",
          sessionId: "session_test",
          queryText: "Retrieve raw_failed",
          expectedRawMemoryId: "raw_failed",
          retrievedRawMemoryIds: ["ordinary_only"],
          retrievedRepresentativeIds: [],
          identityPreserved: false,
          failureMode: "over_pruned",
          createdAt: "2026-05-11T00:00:00.000Z",
        }],
        listRawMemory: async (options) => {
          rawReads.push(options);
          const records = [
            {
              id: "raw_pin",
              tenantId: "tenant_1",
              cellId: "cell_1",
              sessionId: "session_test",
              sourceType: "chat",
              text: "Recovered pinned exact memory.",
              memoryKind: "instruction",
              importance: 1,
              identityRiskSeed: 0.99,
              createdAt: "2026-05-11T00:00:00.000Z",
              updatedAt: "2026-05-11T00:00:00.000Z",
              retentionClass: "pinned",
              hash: "hash_pin",
            },
            {
              id: "raw_failed",
              tenantId: "tenant_1",
              cellId: "cell_1",
              sessionId: "session_test",
              sourceType: "chat",
              text: "Recovered raw memory that failed a prior audit.",
              memoryKind: "fact",
              importance: 0.8,
              identityRiskSeed: 0.75,
              createdAt: "2026-05-11T00:00:00.000Z",
              updatedAt: "2026-05-11T00:00:00.000Z",
              retentionClass: "normal",
              hash: "hash_failed",
            },
          ] satisfies RawMemoryRecord[];
          return options?.rawMemoryId
            ? records.filter((record) => record.id === options.rawMemoryId)
            : [];
        },
      },
      now: new Date("2026-05-11T12:00:00.000Z"),
    });

    expect(plan.packed.includedMemoryIds).toEqual(["raw_pin", "raw_failed"]);
    expect(plan.packed.messages[0]?.content).toContain("Recovered pinned exact memory.");
    expect(plan.trace.runtime.context).toMatchObject({
      rawMemoryRecovery: {
        requestedRawMemoryIds: ["raw_pin", "raw_failed"],
        recoveredRawMemoryIds: ["raw_pin", "raw_failed"],
      },
      memoryPriorityMap: {
        raw_pin: expect.objectContaining({
          protected: true,
          reasons: expect.arrayContaining(["identity_pin", "pinned_exact"]),
        }),
        raw_failed: expect.objectContaining({
          reasons: expect.arrayContaining(["retrieval_audit_failure_repair"]),
        }),
      },
    });
    expect(rawReads).toEqual([
      expect.objectContaining({
        tenantId: "tenant_1",
        cellId: "cell_1",
        sessionId: "session_test",
        rawMemoryId: "raw_pin",
      }),
      expect.objectContaining({
        tenantId: "tenant_1",
        cellId: "cell_1",
        sessionId: "session_test",
        rawMemoryId: "raw_failed",
      }),
    ]);
    expect(writes).toEqual([
      [
        expect.objectContaining({
          includedMemoryIds: ["raw_pin", "raw_failed"],
          rawMemoryIds: expect.arrayContaining(["raw_pin", "raw_failed"]),
          identityPinIds: ["pin_1"],
        }),
      ],
    ]);
  });

  it("persists array lineage ids from included representative memories", async () => {
    const writes: unknown[] = [];
    await buildAdvancedRuntimeGenerationPlan({
      requestId: "req_array_lineage",
      tenantId: "tenant_1",
      cellId: "cell_1",
      sessionId: "session_test",
      systemPrompt: "You are local.",
      userMessage: "Use representative lineage.",
      recentMessages: [],
      retrievedMemory: [
        {
          id: "rep_array",
          text: "Representative with array lineage.",
          embedding: [1, 0],
          score: 0.95,
          sessionId: "session_test",
          source: "summary",
          createdAt: "2026-05-11T00:00:00.000Z",
          updatedAt: "2026-05-11T00:00:00.000Z",
          tags: ["gac"],
          metadata: {
            rawMemoryIds: ["raw_a", "raw_b"],
            representativeIds: ["rep_a", "rep_b"],
            identityPinIds: ["pin_a", "pin_b"],
            memoryClass: "LOW_RISK_REPRESENTATIVE",
          },
          tokenCount: 8,
        },
      ],
      config: {
        modelId: "Qwen/Qwen3-0.6B",
        embeddingModelId: "Xenova/all-MiniLM-L6-v2",
        memoryTopK: 8,
        maxRetrievedMemoryTokens: 64,
        maxRecentConversationTokens: 0,
        maxPromptTokens: 4096,
      },
      backend: {
        id: "unlocked-browser-transformer",
        label: "Unlocked Qwen3 0.6B",
        mode: "custom",
      },
      model: {
        modelId: "Qwen/Qwen3-0.6B",
        parameterBytes: 4_000_000_000,
        effectiveParameterCount: 1_910_000_000,
        layers: 30,
        hiddenSize: 2048,
        kvHeads: 8,
        contextWindowTokens: 32_000,
      },
      device: {
        name: "test-device",
        vramBudgetBytes: 8_000_000_000,
        ramBudgetBytes: 16_000_000_000,
      },
      memoryMode: "remote-http",
      memoryStore: {
        writeContextPackTraces: async (records) => {
          writes.push(records);
          return { ok: true, count: records.length, traceId: records[0]?.traceId ?? "trace_missing" };
        },
        listContextPackTraces: async () => [],
      },
      now: new Date("2026-05-11T12:00:00.000Z"),
    });

    expect(writes).toEqual([
      [
        expect.objectContaining({
          rawMemoryIds: ["raw_a", "raw_b"],
          representativeIds: ["rep_a", "rep_b"],
          identityPinIds: ["pin_a", "pin_b"],
        }),
      ],
    ]);
  });

  it("surfaces low-rank predictive KVSwap prefetch fields in runtime traces", async () => {
    const plan = await buildAdvancedRuntimeGenerationPlan({
      requestId: "req_low_rank_kvswap",
      sessionId: "session_test",
      systemPrompt: "You are local.",
      userMessage: "Use the browser vector memory from the earlier runtime notes.",
      recentMessages: [],
      retrievedMemory: [
        {
          id: "mem_browser_vector",
          text: "Browser vector memory keeps runtime traces local and searchable.",
          embedding: [0.9, 0.1],
          score: 0.87,
          sessionId: "session_test",
          source: "chat",
          createdAt: "2026-05-11T00:00:00.000Z",
          updatedAt: "2026-05-11T00:00:00.000Z",
          tags: ["runtime"],
          metadata: {},
          tokenCount: 12,
        },
        {
          id: "mem_sparse_attention",
          text: "Sparse attention routes hot KV pages before the attention step.",
          embedding: [0.7, 0.3],
          score: 0.74,
          sessionId: "session_test",
          source: "chat",
          createdAt: "2026-05-11T00:00:00.000Z",
          updatedAt: "2026-05-11T00:00:00.000Z",
          tags: ["runtime"],
          metadata: {},
          tokenCount: 12,
        },
      ],
      config: {
        modelId: "Qwen/Qwen3-0.6B",
        embeddingModelId: "Xenova/all-MiniLM-L6-v2",
        memoryTopK: 8,
        maxRetrievedMemoryTokens: 512,
        maxRecentConversationTokens: 512,
        maxPromptTokens: 4096,
      },
      backend: {
        id: "unlocked-browser-transformer",
        label: "Unlocked Qwen3 0.6B",
        mode: "custom",
        capabilities: {
          qkvAccess: true,
          layerSparseRouting: true,
          pinnedKvBlocks: true,
          kvTensorPaging: true,
          tspScheduleExecution: true,
        },
      },
      model: {
        modelId: "Qwen/Qwen3-0.6B",
        parameterBytes: 1_200_000_000,
        effectiveParameterCount: 600_000_000,
        layers: 28,
        hiddenSize: 1024,
        kvHeads: 8,
        contextWindowTokens: 32_000,
      },
      device: {
        name: "browser-webgpu-unlocked",
        vramBudgetBytes: 4_000_000_000,
        ramBudgetBytes: 8_000_000_000,
      },
      memoryMode: "browser-vector",
      memoryStore: makeContextTraceStore(),
      now: new Date("2026-05-11T12:00:00.000Z"),
    });

    expect(plan.kvswapDecision.lowRankSummaryRank).toBe(4);
    const predictedHotBlocks = plan.kvswapDecision.predictedHotBlocks ?? [];
    const prefetchedBlocks = plan.kvswapDecision.prefetchedBlocks ?? [];
    expect(predictedHotBlocks.length).toBeGreaterThan(0);
    expect(predictedHotBlocks[0]).toMatchObject({
      source: "low_rank_attention",
      projectionId: "advanced-runtime:text-low-rank:v1",
    });
    expect(plan.kvswapDecision.prefetchStrategy).toBe("predictive_prefetch");
    expect(prefetchedBlocks.length).toBeGreaterThan(0);
    expect(plan.trace.runtime.kvswap).toMatchObject({
      lowRankSummaryRank: 4,
      predictedHotBlocks: expect.arrayContaining([
        expect.objectContaining({ source: "low_rank_attention" }),
      ]),
      prefetchStrategy: "predictive_prefetch",
      prefetchHitRate: expect.any(Number),
      prefetchBytes: expect.any(Number),
      prefetchLatencyMs: expect.any(Number),
      attentionStallMs: expect.any(Number),
    });
  });
});

function makeContextTraceStore() {
  return {
    writeContextPackTraces: async (records: Array<{ traceId: string }>) => ({
      ok: true as const,
      count: records.length,
      traceId: records[0]?.traceId ?? "trace_missing",
    }),
    listContextPackTraces: async () => [],
  };
}
