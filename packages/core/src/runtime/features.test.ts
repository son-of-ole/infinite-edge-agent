import { describe, expect, it } from "vitest";
import { assertProductionRuntimeFeatures, createDefaultRuntimeFeatureRegistry } from "./features";

describe("runtime feature registry", () => {
  it("allows Tier-0 fallback modes in production when they are explicit", () => {
    const features = createDefaultRuntimeFeatureRegistry().list();

    expect(() => assertProductionRuntimeFeatures(features)).not.toThrow();
  });

  it("rejects test-disabled Tier-0 runtime features for production", () => {
    const registry = createDefaultRuntimeFeatureRegistry();
    registry.set({ name: "ssa", state: "disabled_for_test", mode: "disabled" });

    expect(() => registry.assertProductionReady()).toThrow("disabled for test in production: ssa");
  });
});
