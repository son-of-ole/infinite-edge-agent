import { describe, expect, it } from "vitest";
import {
  BROWSER_QWEN_PREFIX_MTP_DRAFT_MODEL_ID,
  BROWSER_NGRAM_MTP_DRAFT_MODEL_ID,
  makeBrowserMtpClientOptions,
  makeBackendProfile,
  makeRuntimeMtpConfig,
  normalizeMemoryProviderMode,
  resolveBenchmarkTelemetryConfig,
  resolveCompiledWebLlmEnabled,
  resolveDefaultModel,
  resolveAgentMaxPromptTokens,
  resolveMtpEnabled,
  resolveRequireWebGpuKernels,
  resolveUnlockedModelAssetConfig,
  resolveMemoryFallback,
  resolveInteractiveRuntimeLimits,
  resolveQwenThinkingMode,
  resolveUnlockedRuntimeProfile,
  assertUnlockedFullProfile,
} from "./config";

describe("unlocked model asset config", () => {
  it("defaults the browser app to the real local Qwen manifest instead of fixture weights", () => {
    expect(resolveUnlockedModelAssetConfig({}, false)).toEqual({
      manifestPath: "/models/qwen3-0.6b-unlocked/manifest.json",
      manifestSha256: "6f89087e755568187951956a39077a3d9e5538768399e982febf76364e4caebd",
      manifestFormat: "sharded",
      allowFixture: false,
      backendPreference: "webgpu",
    });
  });

  it("keeps fixture mode explicit so CI proofs cannot be confused with live Qwen", () => {
    expect(resolveUnlockedModelAssetConfig({
      VITE_UNLOCKED_ALLOW_FIXTURE: "true",
    }, false)).toMatchObject({
      manifestPath: "",
      manifestSha256: "",
      allowFixture: true,
    });
  });
});

describe("strict WebGPU kernel config", () => {
  it("requires WebGPU kernels by default for the unlocked browser runtime", () => {
    expect(resolveRequireWebGpuKernels({}, "unlocked-browser-transformer")).toBe(true);
  });

  it("keeps CPU-reference development opt-in explicit", () => {
    expect(resolveRequireWebGpuKernels({ VITE_REQUIRE_WEBGPU_KERNELS: "false" }, "unlocked-browser-transformer")).toBe(false);
    expect(resolveRequireWebGpuKernels({ VITE_REQUIRE_WEBGPU_KERNELS: "true" }, "unlocked-browser-transformer")).toBe(true);
    expect(resolveRequireWebGpuKernels({}, "some-other-runtime")).toBe(false);
  });
});

describe("backend profile resolution", () => {
  it("describes compiled browser backends separately from the custom WebGPU Kernel Lab", () => {
    expect(makeBackendProfile("compiled-browser-webllm")).toMatchObject({
      id: "compiled-browser-webllm",
      label: "Compiled Browser WebLLM Backend",
      mode: "custom",
      capabilities: expect.objectContaining({
        qkvAccess: false,
        layerSparseRouting: false,
        kvTensorPaging: false,
      }),
    });

    expect(makeBackendProfile("unlocked-browser-transformer")).toMatchObject({
      id: "unlocked-browser-transformer",
      label: "Custom WebGPU Kernel Lab",
      capabilities: expect.objectContaining({
        qkvAccess: true,
        layerSparseRouting: true,
        kvTensorPaging: true,
      }),
    });
  });
});

describe("compiled WebLLM adapter config", () => {
  it("keeps the compiled production candidate disabled until its adapter is explicitly enabled", () => {
    expect(resolveCompiledWebLlmEnabled({})).toBe(false);
    expect(resolveCompiledWebLlmEnabled({ VITE_COMPILED_WEBLLM_ENABLED: "false" })).toBe(false);
    expect(resolveCompiledWebLlmEnabled({ VITE_COMPILED_WEBLLM_ENABLED: "true" })).toBe(true);
  });

  it("defaults compiled production builds to the WebLLM artifact id while preserving the local Kernel Lab model id", () => {
    expect(resolveDefaultModel({}, "compiled-browser-webllm")).toBe("Qwen3-0.6B-q4f16_1-MLC");
    expect(resolveDefaultModel({}, "unlocked-browser-transformer")).toBe("Qwen/Qwen3-0.6B");
    expect(resolveDefaultModel({ VITE_DEFAULT_MODEL: "custom-model" }, "compiled-browser-webllm")).toBe("custom-model");
  });
});

describe("benchmark telemetry config", () => {
  it("requires both the enable flag and endpoint before browser benchmarks can submit telemetry", () => {
    expect(resolveBenchmarkTelemetryConfig({})).toEqual({
      enabled: false,
      url: "",
    });
    expect(resolveBenchmarkTelemetryConfig({
      VITE_BENCHMARK_TELEMETRY_ENABLED: "true",
    })).toEqual({
      enabled: false,
      url: "",
    });
    expect(resolveBenchmarkTelemetryConfig({
      VITE_BENCHMARK_TELEMETRY_ENABLED: "true",
      VITE_BENCHMARK_TELEMETRY_URL: " /api/benchmark-runs ",
      VITE_APP_VERSION: " 0.1.0 ",
      VITE_GIT_SHA: " abc123 ",
      VITE_DEPLOY_URL: " https://agent.example.com ",
    })).toEqual({
      enabled: true,
      url: "/api/benchmark-runs",
      appVersion: "0.1.0",
      gitSha: "abc123",
      deployUrl: "https://agent.example.com",
    });
  });
});

describe("unlocked runtime profile resolution", () => {
  it("defaults uncapped browser runtime execution to the full profile", () => {
    expect(resolveUnlockedRuntimeProfile({})).toMatchObject({
      profile: "full",
      caps: {
        maxRuntimePromptTokens: null,
        maxRuntimeLayers: null,
        logitCandidateLimit: null,
        maxGenerationTokens: null,
      },
      capStatus: {
        prompt: false,
        layers: false,
        generation: false,
        logits: false,
      },
    });
  });

  it("makes preview caps explicit while full removes artificial caps", () => {
    expect(resolveUnlockedRuntimeProfile({ VITE_UNLOCKED_RUNTIME_PROFILE: "preview" })).toMatchObject({
      profile: "preview",
      caps: {
        maxRuntimePromptTokens: 4,
        maxRuntimeLayers: 1,
        logitCandidateLimit: 256,
        maxGenerationTokens: 1,
      },
      capStatus: {
        prompt: true,
        layers: true,
        generation: true,
        logits: true,
      },
    });

    expect(resolveUnlockedRuntimeProfile({ VITE_UNLOCKED_RUNTIME_PROFILE: "full" })).toMatchObject({
      profile: "full",
      caps: {
        maxRuntimePromptTokens: null,
        maxRuntimeLayers: null,
        logitCandidateLimit: null,
        maxGenerationTokens: null,
      },
      capStatus: {
        prompt: false,
        layers: false,
        generation: false,
        logits: false,
      },
    });
  });

  it("preserves explicit cap env overrides and fails strict full-profile gates when capped", () => {
    const fullWithOverride = resolveUnlockedRuntimeProfile({
      VITE_UNLOCKED_RUNTIME_PROFILE: "full",
      VITE_UNLOCKED_MAX_RUNTIME_LAYERS: "2",
    });

    expect(fullWithOverride.caps.maxRuntimeLayers).toBe(2);
    expect(fullWithOverride.capStatus.layers).toBe(true);
    expect(() => assertUnlockedFullProfile(fullWithOverride)).toThrow(/requires VITE_UNLOCKED_RUNTIME_PROFILE=full without artificial caps/i);
    expect(() => assertUnlockedFullProfile(resolveUnlockedRuntimeProfile({ VITE_UNLOCKED_RUNTIME_PROFILE: "full" }))).not.toThrow();
  });

  it("fails loudly for malformed explicit cap overrides", () => {
    expect(() => resolveUnlockedRuntimeProfile({
      VITE_UNLOCKED_RUNTIME_PROFILE: "full",
      VITE_UNLOCKED_MAX_GENERATION_TOKENS: "abc",
    })).toThrow(/VITE_UNLOCKED_MAX_GENERATION_TOKENS must be a positive integer/i);
    expect(() => resolveUnlockedRuntimeProfile({
      VITE_UNLOCKED_MAX_RUNTIME_LAYERS: "0",
    })).toThrow(/VITE_UNLOCKED_MAX_RUNTIME_LAYERS must be a positive integer/i);
  });
});

describe("interactive unlocked runtime limits", () => {
  it("keeps the release/runtime profile full while using production chat budgets by default", () => {
    const full = resolveUnlockedRuntimeProfile({});

    expect(full.profile).toBe("full");
    expect(full.caps.maxRuntimeLayers).toBeNull();
    expect(resolveInteractiveRuntimeLimits({}, full.caps)).toEqual({
      maxRuntimePromptTokens: null,
      maxRuntimeLayers: null,
      logitCandidateLimit: null,
      logitTopK: 1,
      logitTileRows: 32768,
      maxGenerationTokens: 256,
    });
  });

  it("honors explicit interactive overrides and explicit uncapped chat controls", () => {
    const full = resolveUnlockedRuntimeProfile({});

    expect(resolveInteractiveRuntimeLimits({
      VITE_CHAT_MAX_RUNTIME_PROMPT_TOKENS: "128",
      VITE_CHAT_MAX_RUNTIME_LAYERS: "full",
      VITE_CHAT_LOGIT_CANDIDATE_LIMIT: "256",
      VITE_CHAT_LOGIT_TOP_K: "32",
      VITE_CHAT_LOGIT_TILE_ROWS: "2048",
      VITE_CHAT_MAX_GENERATION_TOKENS: "3",
    }, full.caps)).toEqual({
      maxRuntimePromptTokens: 128,
      maxRuntimeLayers: null,
      logitCandidateLimit: 256,
      logitTopK: 32,
      logitTileRows: 2048,
      maxGenerationTokens: 3,
    });

    expect(resolveInteractiveRuntimeLimits({
      VITE_CHAT_MAX_GENERATION_TOKENS: "full",
    }, full.caps).maxGenerationTokens).toBe(40_960);
  });
});

describe("Qwen thinking mode", () => {
  it("defaults to fast visible no-think mode while preserving an explicit thinking override", () => {
    expect(resolveQwenThinkingMode(undefined)).toBe("disabled");
    expect(resolveQwenThinkingMode("bad")).toBe("disabled");
    expect(resolveQwenThinkingMode("enabled")).toBe("enabled");
    expect(resolveQwenThinkingMode("disabled")).toBe("disabled");
  });
});

describe("agent prompt budget", () => {
  it("defaults prompt packing to the Qwen context window instead of the old 6k app budget", () => {
    expect(resolveAgentMaxPromptTokens(undefined)).toBe(40_960);
  });

  it("honors explicit app-level prompt packing budgets", () => {
    expect(resolveAgentMaxPromptTokens("8192")).toBe(8192);
    expect(resolveAgentMaxPromptTokens("bad")).toBe(40_960);
  });
});

describe("memory provider config", () => {
  it("defaults to browser-vector while preserving indexeddb as a compatibility alias", () => {
    expect(normalizeMemoryProviderMode(undefined)).toBe("browser-vector");
    expect(normalizeMemoryProviderMode("browser-vector")).toBe("browser-vector");
    expect(normalizeMemoryProviderMode("indexeddb")).toBe("indexeddb");
    expect(normalizeMemoryProviderMode("remote-http")).toBe("remote-http");
    expect(normalizeMemoryProviderMode("unknown-provider")).toBe("browser-vector");
  });

  it("makes memory fallback opt-out so remote and sidecar layers are optional by default", () => {
    expect(resolveMemoryFallback(undefined)).toBe(true);
    expect(resolveMemoryFallback("")).toBe(true);
    expect(resolveMemoryFallback("true")).toBe(true);
    expect(resolveMemoryFallback("false")).toBe(false);
    expect(resolveMemoryFallback("FALSE")).toBe(false);
  });
});

describe("MTP config wiring", () => {
  it("keeps browser production target-only unless MTP is explicitly lab-enabled", () => {
    expect(resolveMtpEnabled({})).toBe(false);
    expect(resolveMtpEnabled({ VITE_MTP_ENABLED: "false" })).toBe(false);
    expect(resolveMtpEnabled({ VITE_MTP_ENABLED: "TRUE" })).toBe(false);
    expect(resolveMtpEnabled({ VITE_MTP_ENABLED: "true" })).toBe(true);
  });

  it("passes enabled intent while clamping browser draft windows before worker setup", () => {
    expect(makeBrowserMtpClientOptions({
      enabled: true,
      draftModelId: "some/neural-drafter",
      numSpeculativeTokens: 4,
      minAcceptanceRate: 0.45,
      disableWhenLatencyWorse: true,
    })).toEqual({
      enabled: true,
      draftModelId: "some/neural-drafter",
      numSpeculativeTokens: 3,
      minAcceptanceRate: 0.45,
      disableWhenLatencyWorse: true,
    });
  });

  it("registers runtime draft profiles only for supported browser-local drafters", () => {
    expect(makeRuntimeMtpConfig({
      enabled: true,
      draftModelId: "some/neural-drafter",
      numSpeculativeTokens: 4,
      minAcceptanceRate: 0.45,
      disableWhenLatencyWorse: true,
      targetModelId: "Qwen/Qwen3-0.6B",
    })).toMatchObject({
      enabled: true,
      draftModelId: "some/neural-drafter",
      numSpeculativeTokens: 3,
      draftModelProfiles: [],
    });
    expect(makeRuntimeMtpConfig({
      enabled: true,
      draftModelId: BROWSER_NGRAM_MTP_DRAFT_MODEL_ID,
      numSpeculativeTokens: 4,
      minAcceptanceRate: 0.45,
      disableWhenLatencyWorse: true,
      targetModelId: "Qwen/Qwen3-0.6B",
    }).draftModelProfiles).toEqual([
      expect.objectContaining({
        modelId: BROWSER_NGRAM_MTP_DRAFT_MODEL_ID,
        maxSpeculativeTokens: 3,
        targetModelIds: ["Qwen/Qwen3-0.6B"],
      }),
    ]);
    expect(makeRuntimeMtpConfig({
      enabled: true,
      draftModelId: BROWSER_QWEN_PREFIX_MTP_DRAFT_MODEL_ID,
      numSpeculativeTokens: 4,
      minAcceptanceRate: 0.45,
      disableWhenLatencyWorse: true,
      targetModelId: "Qwen/Qwen3-0.6B",
    }).draftModelProfiles).toEqual([
      expect.objectContaining({
        modelId: BROWSER_QWEN_PREFIX_MTP_DRAFT_MODEL_ID,
        tokenizerId: "qwen3",
        maxSpeculativeTokens: 3,
        targetModelIds: ["Qwen/Qwen3-0.6B"],
      }),
    ]);
  });
});
