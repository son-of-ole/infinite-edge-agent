import { describe, expect, it } from "vitest";
import {
  getBrowserBackendRegistryEntry,
  selectBrowserBackend,
} from "./backendBroker";

describe("backend broker", () => {
  it("classifies the custom WebGPU runtime as the Kernel Lab backend, not the production answer backend", () => {
    const entry = getBrowserBackendRegistryEntry("unlocked-browser-transformer");

    expect(entry).toMatchObject({
      backendId: "unlocked-browser-transformer",
      adapterKind: "custom-webgpu-kernel-lab",
      productionRole: "research_kernel_lab",
      deployDefault: false,
      capabilities: expect.objectContaining({
        supportsSparseAttentionRouting: true,
        supportsKvSwapPersistence: true,
        supportsCompiledGraph: false,
      }),
    });
  });

  it("selects the compiled browser backend for grounded production answers when available", () => {
    const selection = selectBrowserBackend({
      task: "grounded_answer",
      availableBackendIds: ["compiled-browser-webllm", "unlocked-browser-transformer"],
      preferredModelId: "Qwen3-0.6B-q4f16_1-MLC",
    });

    expect(selection).toMatchObject({
      backendId: "compiled-browser-webllm",
      productionRole: "production_candidate",
      modelId: "Qwen3-0.6B-q4f16_1-MLC",
      deployReadyCandidate: true,
      reason: "compiled_first_grounded_answer",
      fallbackChain: ["unlocked-browser-transformer", "wasm-small-core"],
      proofRequirements: expect.arrayContaining([
        "memory_grounding",
        "quality_canaries",
        "speed_floor",
      ]),
    });
  });

  it("routes explicit kernel research tasks to the custom WebGPU Kernel Lab", () => {
    const selection = selectBrowserBackend({
      task: "kernel_research",
      availableBackendIds: ["compiled-browser-webllm", "unlocked-browser-transformer"],
    });

    expect(selection).toMatchObject({
      backendId: "unlocked-browser-transformer",
      productionRole: "research_kernel_lab",
      deployReadyCandidate: false,
      reason: "kernel_lab_required",
    });
  });
});
