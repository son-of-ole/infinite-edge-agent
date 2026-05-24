import { describe, expect, it } from "vitest";
import { BROWSER_NGRAM_MTP_DRAFT_MODEL_ID } from "./config";
import { makeUnlockedBrowserWorkerOptions } from "./App";

describe("App unlocked worker wiring", () => {
  it("passes supported browser MTP options through to the unlocked worker client", () => {
    const options = makeUnlockedBrowserWorkerOptions({
      modelId: "Qwen/Qwen3-0.6B",
      manifestPath: "/models/qwen3-0.6b-unlocked/manifest.json",
      manifestSha256: "a".repeat(64),
      allowFixtureWeights: false,
      backendPreference: "webgpu",
      requireWebGpu: true,
      maxRuntimePromptTokens: 4,
      maxRuntimeLayers: 1,
      logitCandidateLimit: 256,
      logitTopK: 32,
      logitTileRows: 2048,
      maxGenerationTokens: 1,
      qwenThinkingMode: "enabled",
      mtpEnabled: true,
      mtpDraftModelId: BROWSER_NGRAM_MTP_DRAFT_MODEL_ID,
      mtpNumSpeculativeTokens: 4,
      mtpMinAcceptanceRate: 0.45,
      mtpDisableWhenLatencyWorse: true,
      kvPersistenceEnabled: true,
      kvPersistenceNamespace: "local:browser:session",
      kvPersistencePreferOpfs: true,
      kvPersistenceMaxBlocks: 16,
      kvPersistenceMaxBytes: 4096,
      kvPersistenceClearOnInit: false,
    });

    expect(options).toMatchObject({
      modelId: "Qwen/Qwen3-0.6B",
      backendPreference: "webgpu",
      requireWebGpu: true,
      warmModelResidency: true,
      warmModelResidencyMode: "pipeline_preload",
      maxRuntimePromptTokens: 4,
      maxRuntimeLayers: 1,
      logitCandidateLimit: 256,
      logitTopK: 32,
      logitTileRows: 2048,
      maxGenerationTokens: 1,
      qwenThinkingMode: "enabled",
      mtp: {
        enabled: true,
        draftModelId: BROWSER_NGRAM_MTP_DRAFT_MODEL_ID,
        numSpeculativeTokens: 3,
        minAcceptanceRate: 0.45,
        disableWhenLatencyWorse: true,
      },
      kvPersistence: {
        enabled: true,
        namespace: "local:browser:session",
        preferOpfs: true,
        maxBlocks: 16,
        maxBytes: 4096,
        clearOnInit: false,
      },
    });
  });

  it("passes arbitrary draft IDs to the worker so the browser client can classify unsupported fallback", () => {
    const options = makeUnlockedBrowserWorkerOptions({
      modelId: "Qwen/Qwen3-0.6B",
      manifestPath: "/models/qwen3-0.6b-unlocked/manifest.json",
      manifestSha256: "b".repeat(64),
      allowFixtureWeights: false,
      mtpEnabled: true,
      mtpDraftModelId: "some/neural-drafter",
      mtpNumSpeculativeTokens: 2,
      mtpMinAcceptanceRate: 0.6,
      mtpDisableWhenLatencyWorse: false,
    });

    expect(options.mtp).toEqual({
      enabled: true,
      draftModelId: "some/neural-drafter",
      numSpeculativeTokens: 2,
      minAcceptanceRate: 0.6,
      disableWhenLatencyWorse: false,
    });
  });

  it("passes KV persistence clear-on-init reset intent to the unlocked worker", () => {
    const options = makeUnlockedBrowserWorkerOptions({
      modelId: "Qwen/Qwen3-0.6B",
      manifestPath: "/models/qwen3-0.6b-unlocked/manifest.json",
      manifestSha256: "c".repeat(64),
      allowFixtureWeights: false,
      mtpEnabled: false,
      mtpDraftModelId: BROWSER_NGRAM_MTP_DRAFT_MODEL_ID,
      mtpNumSpeculativeTokens: 1,
      mtpMinAcceptanceRate: 0.45,
      mtpDisableWhenLatencyWorse: true,
      kvPersistenceEnabled: true,
      kvPersistenceNamespace: "tenant:cell:private-session",
      kvPersistenceClearOnInit: true,
    });

    expect(options.kvPersistence).toMatchObject({
      enabled: true,
      namespace: "tenant:cell:private-session",
      clearOnInit: true,
    });
  });

  it("uses a session-scoped KV namespace so reset clears blocks across model switches", () => {
    const baseInput = {
      modelId: "Qwen/Qwen3-0.6B",
      manifestPath: "/models/qwen3-0.6b-unlocked/manifest.json",
      manifestSha256: "d".repeat(64),
      allowFixtureWeights: false,
      mtpEnabled: false,
      mtpDraftModelId: BROWSER_NGRAM_MTP_DRAFT_MODEL_ID,
      mtpNumSpeculativeTokens: 1,
      mtpMinAcceptanceRate: 0.45,
      mtpDisableWhenLatencyWorse: true,
      kvPersistenceEnabled: true,
      kvPersistenceNamespace: "tenant:cell:session_1",
    };
    const qwenOptions = makeUnlockedBrowserWorkerOptions(baseInput);
    const otherModelOptions = makeUnlockedBrowserWorkerOptions({
      ...baseInput,
      modelId: "Other/Model",
    });

    expect(qwenOptions.kvPersistence?.namespace).toBe("tenant:cell:session_1");
    expect(otherModelOptions.kvPersistence?.namespace).toBe("tenant:cell:session_1");
  });
});
