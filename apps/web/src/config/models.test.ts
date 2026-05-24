import { describe, expect, it } from "vitest";
import { LOCAL_MODEL_OPTIONS } from "./models";

describe("local model options", () => {
  it("ships both the compiled production candidate and the custom WebGPU Kernel Lab lane", () => {
    expect(LOCAL_MODEL_OPTIONS).toEqual(expect.arrayContaining([
      expect.objectContaining({
        id: "Qwen3-0.6B-q4f16_1-MLC",
        backend: "compiled-browser-webllm",
      }),
      expect.objectContaining({
        id: "Qwen/Qwen3-0.6B",
        backend: "unlocked-browser-transformer",
      }),
    ]));
  });
});
