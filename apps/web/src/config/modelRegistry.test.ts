import { describe, expect, it } from "vitest";
import {
  getModelRegistryEntry,
  listLocalModelOptionsFromRegistry,
  MODEL_REGISTRY,
} from "./modelRegistry";

describe("v12 model registry", () => {
  it("records backend-specific model entries with an ETP-style tensor ABI contract", () => {
    const compiled = getModelRegistryEntry("compiled-browser-webllm", "Qwen3-0.6B-q4f16_1-MLC");
    const kernelLab = getModelRegistryEntry("unlocked-browser-transformer", "Qwen/Qwen3-0.6B");

    expect(MODEL_REGISTRY.schema).toBe("edge-model-registry/v1");
    expect(compiled).toMatchObject({
      backendId: "compiled-browser-webllm",
      artifactKind: "webllm-compiled",
      productionRole: "production_candidate",
      tensorAbi: {
        schema: "edge-tensor-protocol/v1",
        residency: "backend_native",
        proof: expect.objectContaining({
          requireBackendTrace: true,
          requireQualityGate: true,
        }),
      },
    });
    expect(kernelLab).toMatchObject({
      backendId: "unlocked-browser-transformer",
      artifactKind: "custom-sharded-f16",
      productionRole: "research_kernel_lab",
      tensorAbi: {
        schema: "edge-tensor-protocol/v1",
        residency: "stable_gpu",
        kernels: expect.objectContaining({
          supportsSparseAttentionRouting: true,
          supportsKvSwapPersistence: true,
        }),
        proof: expect.objectContaining({
          forbidCpuFallback: true,
          forbidHiddenReadback: true,
          forbidFullLogitReadback: true,
        }),
      },
    });
  });

  it("derives local model options from the registry without making the Kernel Lab the deploy default", () => {
    const options = listLocalModelOptionsFromRegistry();

    expect(options[0]).toMatchObject({
      backend: "compiled-browser-webllm",
      id: "Qwen3-0.6B-q4f16_1-MLC",
    });
    expect(options.find((option) => option.backend === "unlocked-browser-transformer")).toMatchObject({
      backend: "unlocked-browser-transformer",
      id: "Qwen/Qwen3-0.6B",
    });
  });
});
