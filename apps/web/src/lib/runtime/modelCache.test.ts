import { describe, expect, it } from "vitest";
import { isLikelyModelCacheName } from "./modelCache";

describe("modelCache", () => {
  it("identifies browser model cache names without matching ordinary app memory", () => {
    expect(isLikelyModelCacheName("unlocked-qwen-cache")).toBe(true);
    expect(isLikelyModelCacheName("qwen3-0.6b-sharded")).toBe(true);
    expect(isLikelyModelCacheName("huggingface-transformers")).toBe(true);
    expect(isLikelyModelCacheName("infinite-edge-agent")).toBe(false);
  });
});
