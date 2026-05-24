import {
  STRICT_UNLOCKED_WEBGPU_GATES,
  type StrictUnlockedWebGpuGate,
} from "@infinite-edge-agent/core";

export {
  assertUnlockedWebGpuCoverageGates,
  evaluateUnlockedWebGpuCoverageGates,
  STRICT_UNLOCKED_WEBGPU_GATES,
  summarizeUnlockedWebGpuCoverage,
  type StrictUnlockedWebGpuGate,
  type UnlockedCoverageDecodeProof,
  type UnlockedKernelBackend,
  type UnlockedWebGpuCoverageGateResult,
  type UnlockedWebGpuCoverageSummary,
} from "@infinite-edge-agent/core";

export function readStrictUnlockedWebGpuGatesFromEnv(
  env: Record<string, string | undefined>,
): StrictUnlockedWebGpuGate[] {
  if (
    env.RELEASE_REQUIRE_UNLOCKED_STRICT_WEBGPU === "true"
    || env.VITE_REQUIRE_WEBGPU_KERNELS === "true"
    || (
      env.RELEASE_REQUIRE_UNLOCKED_MODEL === "true"
      && env.RELEASE_REQUIRE_UNLOCKED_STRICT_WEBGPU !== "false"
      && env.VITE_REQUIRE_WEBGPU_KERNELS !== "false"
    )
  ) {
    return STRICT_UNLOCKED_WEBGPU_GATES;
  }
  const gates: StrictUnlockedWebGpuGate[] = [];
  if (env.RELEASE_REQUIRE_UNLOCKED_WEBGPU_MLP === "true") gates.push("mlp");
  if (env.RELEASE_REQUIRE_UNLOCKED_WEBGPU_LOGITS === "true") gates.push("logits");
  if (env.RELEASE_REQUIRE_UNLOCKED_WEBGPU_ATTENTION === "true") gates.push("attention");
  if (env.RELEASE_REQUIRE_UNLOCKED_WEBGPU_PROJECTION === "true") gates.push("projection");
  return gates;
}
