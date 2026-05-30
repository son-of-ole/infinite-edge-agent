import { describe, expect, it } from "vitest";
import {
  makeReleaseGateDefaultEnvOverrides,
  parseManifestSha256Sidecar,
  makeProductionEvalEnvOverrides,
  makeQwenParityEnvOverrides,
  makeReleaseGateChildEnv,
  makeUnlockedBenchmarkEnvOverrides,
  makeUnlockedVerifyEnvOverrides,
  makeReleaseGateTestEnvOverrides,
} from "./releaseGateConfig";

describe("release gate unlocked verification config", () => {
  it("normalizes sha256sum sidecar lines to the raw manifest hash", () => {
    expect(parseManifestSha256Sidecar(`${"e".repeat(64)}  manifest.json\n`)).toBe("e".repeat(64));
    expect(parseManifestSha256Sidecar(`${"f".repeat(64)} *manifest.json\n`)).toBe("f".repeat(64));
  });

  it("promotes configured local Qwen release gates before child steps run", () => {
    expect(makeReleaseGateDefaultEnvOverrides({
      VITE_UNLOCKED_MODEL_MANIFEST_PATH: "/models/qwen3-0.6b-unlocked/manifest.json",
      VITE_UNLOCKED_MODEL_MANIFEST_SHA256: "f".repeat(64),
    })).toMatchObject({
      RELEASE_REQUIRE_UNLOCKED_MODEL: "true",
      RELEASE_REQUIRE_UNLOCKED_QWEN_MATH: "true",
      RELEASE_REQUIRE_UNLOCKED_QWEN_PARITY: "true",
      RELEASE_REQUIRE_UNLOCKED_FULL_PROFILE: "true",
      RELEASE_REQUIRE_UNLOCKED_STRICT_WEBGPU: "true",
      RELEASE_REQUIRE_UNLOCKED_PACKED_ASSETS: "true",
      RELEASE_REQUIRE_BROWSER_PREVIEW_PROOF: "true",
      VITE_REQUIRE_WEBGPU_KERNELS: "true",
      VITE_UNLOCKED_MODEL_MANIFEST_PATH: "/models/qwen3-0.6b-unlocked/manifest.json",
      VITE_UNLOCKED_MODEL_MANIFEST_SHA256: "f".repeat(64),
      VITE_UNLOCKED_ALLOW_FIXTURE: "false",
      VITE_UNLOCKED_WEIGHT_FORMAT: "f16-packed",
      VITE_UNLOCKED_BACKEND_PREFERENCE: "webgpu",
      VITE_MEMORY_PROVIDER: "browser-vector",
    });
  });

  it("does not auto-promote the Kernel Lab gate when v12 compiled production proof is required", () => {
    expect(makeReleaseGateDefaultEnvOverrides({
      RELEASE_REQUIRE_V12_PRODUCTION: "true",
      VITE_UNLOCKED_MODEL_MANIFEST_PATH: "/models/qwen3-0.6b-unlocked/manifest.json",
      VITE_UNLOCKED_MODEL_MANIFEST_SHA256: "f".repeat(64),
    })).toEqual({});
    expect(makeReleaseGateDefaultEnvOverrides({
      RELEASE_REQUIRE_V12_PRODUCTION: "true",
      RELEASE_REQUIRE_UNLOCKED_MODEL: "true",
      VITE_UNLOCKED_MODEL_MANIFEST_PATH: "/models/qwen3-0.6b-unlocked/manifest.json",
      VITE_UNLOCKED_MODEL_MANIFEST_SHA256: "f".repeat(64),
    })).toMatchObject({
      RELEASE_REQUIRE_UNLOCKED_MODEL: "true",
      VITE_LLM_BACKEND: "unlocked-browser-transformer",
    });
  });

  it("does not auto-promote installed Kernel Lab models unless explicitly requested", () => {
    expect(makeReleaseGateDefaultEnvOverrides({})).toEqual({});
    expect(makeReleaseGateDefaultEnvOverrides({
      RELEASE_AUTO_DETECT_UNLOCKED_MODEL: "false",
    })).toEqual({});
  });

  it("strips deploy runtime env from non-production child checks", () => {
    expect(makeReleaseGateTestEnvOverrides({
      VITE_LLM_BACKEND: "compiled-browser-webllm",
      VITE_BENCHMARK_TELEMETRY_ENABLED: "true",
      VITE_BENCHMARK_TELEMETRY_URL: "/api/benchmark-runs",
      VITE_REQUIRE_WEBGPU_KERNELS: "false",
      RELEASE_REQUIRE_V12_PRODUCTION: "true",
    })).toEqual({
      VITE_LLM_BACKEND: undefined,
      VITE_DEFAULT_MODEL: undefined,
      VITE_COMPILED_WEBLLM_ENABLED: undefined,
      VITE_REQUIRE_UNLOCKED_RUNTIME: undefined,
      VITE_REQUIRE_WEBGPU_KERNELS: undefined,
      VITE_BENCHMARK_TELEMETRY_ENABLED: undefined,
      VITE_BENCHMARK_TELEMETRY_URL: undefined,
      VITE_MEMORY_PROVIDER: undefined,
      VITE_QWEN_THINKING_MODE: undefined,
      VITE_MTP_ENABLED: undefined,
    });
  });

  it("defaults model-present local release gates to non-authoritative Node verification", () => {
    expect(makeUnlockedVerifyEnvOverrides({
      VITE_UNLOCKED_MODEL_MANIFEST_PATH: "/models/qwen3-0.6b-unlocked/manifest.json",
      VITE_UNLOCKED_MODEL_MANIFEST_SHA256: "a".repeat(64),
    })).toEqual({
      RELEASE_REQUIRE_UNLOCKED_MODEL: "true",
      RELEASE_REQUIRE_UNLOCKED_QWEN_MATH: "true",
      RELEASE_REQUIRE_UNLOCKED_QWEN_PARITY: "true",
      RELEASE_REQUIRE_UNLOCKED_FULL_PROFILE: "true",
      RELEASE_REQUIRE_BROWSER_PREVIEW_PROOF: "true",
      RELEASE_REQUIRE_UNLOCKED_STRICT_WEBGPU: "false",
      RELEASE_REQUIRE_UNLOCKED_PACKED_ASSETS: "true",
      RELEASE_REQUIRE_UNLOCKED_KV_REUSE: "true",
      VITE_DEFAULT_MODEL: "Qwen/Qwen3-0.6B",
      VITE_LLM_BACKEND: "unlocked-browser-transformer",
      VITE_REQUIRE_UNLOCKED_RUNTIME: "true",
      VITE_REQUIRE_WEBGPU_KERNELS: "false",
      VITE_UNLOCKED_MODEL_MANIFEST_PATH: "/models/qwen3-0.6b-unlocked/manifest.json",
      VITE_UNLOCKED_MODEL_MANIFEST_SHA256: "a".repeat(64),
      VITE_UNLOCKED_MANIFEST_FORMAT: "sharded",
      VITE_UNLOCKED_WEIGHT_FORMAT: "f16-packed",
      VITE_UNLOCKED_ALLOW_FIXTURE: "false",
      VITE_UNLOCKED_BACKEND_PREFERENCE: "cpu",
      VITE_MEMORY_PROVIDER: "browser-vector",
      VITE_QWEN_THINKING_MODE: "disabled",
      VITE_MTP_ENABLED: "false",
      VITE_MTP_DRAFT_MODEL_ID: "browser/qwen-prefix-drafter",
      VITE_MTP_NUM_SPECULATIVE_TOKENS: "2",
      VITE_MTP_DRAFT_LAYER_COUNT: "4",
      VITE_MTP_MIN_ACCEPTANCE_RATE: "0",
      VITE_UNLOCKED_RUNTIME_PROFILE: "full",
    });
  });

  it("defaults model-present local release gates to configured full-profile browser benchmarks", () => {
    expect(makeUnlockedBenchmarkEnvOverrides({
      VITE_UNLOCKED_MODEL_MANIFEST_PATH: "/models/qwen3-0.6b-unlocked/manifest.json",
      VITE_UNLOCKED_MODEL_MANIFEST_SHA256: "b".repeat(64),
    })).toMatchObject({
      RELEASE_REQUIRE_UNLOCKED_MODEL: "true",
      RELEASE_REQUIRE_UNLOCKED_QWEN_MATH: "true",
      RELEASE_REQUIRE_UNLOCKED_QWEN_PARITY: "true",
      RELEASE_REQUIRE_UNLOCKED_FULL_PROFILE: "true",
      RELEASE_REQUIRE_BROWSER_PREVIEW_PROOF: "true",
      RELEASE_REQUIRE_UNLOCKED_STRICT_WEBGPU: "true",
      RELEASE_REQUIRE_UNLOCKED_PACKED_ASSETS: "true",
      RELEASE_REQUIRE_UNLOCKED_KV_REUSE: "true",
      VITE_UNLOCKED_MODEL_MANIFEST_PATH: "/models/qwen3-0.6b-unlocked/manifest.json",
      VITE_UNLOCKED_MODEL_MANIFEST_SHA256: "b".repeat(64),
      VITE_REQUIRE_WEBGPU_KERNELS: "true",
      VITE_UNLOCKED_BACKEND_PREFERENCE: "webgpu",
      VITE_UNLOCKED_WEIGHT_FORMAT: "f16-packed",
      VITE_MEMORY_PROVIDER: "browser-vector",
      VITE_QWEN_THINKING_MODE: "disabled",
      VITE_MTP_ENABLED: "false",
      VITE_MTP_DRAFT_MODEL_ID: "browser/qwen-prefix-drafter",
      VITE_UNLOCKED_RUNTIME_PROFILE: "full",
      BROWSER_RUNTIME_BENCH_STRICT: "true",
      BROWSER_RUNTIME_BENCH_REQUIRE_BROWSER_PREVIEW: "true",
      BROWSER_RUNTIME_BENCH_PREVIEW_TIMEOUT_MS: "120000",
      BROWSER_RUNTIME_BENCH_PREVIEW_MIN_GENERATED_TOKENS: "1",
      BROWSER_RUNTIME_BENCH_PREVIEW_REQUIRE_KV_REUSE: "true",
      BROWSER_RUNTIME_BENCH_PREVIEW_REQUIRE_KV_PREDICTIVE_PREFETCH: "true",
      BROWSER_RUNTIME_BENCH_PREVIEW_REQUIRE_STRICT_WEBGPU: "true",
      BROWSER_RUNTIME_BENCH_PREVIEW_MEMORY_GROUNDING_CASE: "qa_corpus_v1",
      BROWSER_RUNTIME_BENCH_PREVIEW_MEMORY_CORPUS_SIZE: "64",
      BROWSER_RUNTIME_BENCH_REQUIRE_STRICT_WEBGPU: "true",
      BROWSER_RUNTIME_BENCH_MAX_GENERATION_TOKENS: "16",
      BROWSER_RUNTIME_BENCH_PROMPTS: "What is the capital of Utah?|Write two clear sentences about Earth.",
      BROWSER_RUNTIME_BENCH_EXPECTED_SUBSTRINGS: "Salt Lake|Earth",
    });
  });

  it("defaults model-present local release gates to real Qwen parity accuracy", () => {
    expect(makeQwenParityEnvOverrides({
      VITE_UNLOCKED_MODEL_MANIFEST_PATH: "/models/qwen3-0.6b-unlocked/manifest.json",
      VITE_UNLOCKED_MODEL_MANIFEST_SHA256: "c".repeat(64),
    })).toEqual({
      RELEASE_REQUIRE_QWEN_ACCURACY_REAL_MODEL: "true",
      QWEN_PARITY_MANIFEST_PATH: "apps/web/public/models/qwen3-0.6b-unlocked/manifest.json",
    });
  });

  it("keeps explicit fixture release gates available for dev and CI-without-model lanes", () => {
    expect(makeUnlockedVerifyEnvOverrides({
      RELEASE_ALLOW_FIXTURE_GATE: "true",
      VITE_UNLOCKED_MODEL_MANIFEST_PATH: "/models/qwen3-0.6b-unlocked/manifest.json",
      VITE_UNLOCKED_MODEL_MANIFEST_SHA256: "d".repeat(64),
    })).toEqual({});
    expect(makeUnlockedVerifyEnvOverrides({
      CI: "true",
    })).toEqual({});
  });

  it("requires full-profile verification when strict unlocked model mode is enabled", () => {
    expect(makeUnlockedVerifyEnvOverrides({
      RELEASE_REQUIRE_UNLOCKED_MODEL: "true",
    })).toEqual({
      RELEASE_REQUIRE_UNLOCKED_MODEL: "true",
      RELEASE_REQUIRE_UNLOCKED_QWEN_MATH: "true",
      RELEASE_REQUIRE_UNLOCKED_QWEN_PARITY: "true",
      RELEASE_REQUIRE_UNLOCKED_FULL_PROFILE: "true",
      RELEASE_REQUIRE_BROWSER_PREVIEW_PROOF: "true",
      RELEASE_REQUIRE_UNLOCKED_KV_REUSE: "true",
      RELEASE_REQUIRE_UNLOCKED_STRICT_WEBGPU: "false",
      RELEASE_REQUIRE_UNLOCKED_PACKED_ASSETS: "true",
      VITE_DEFAULT_MODEL: "Qwen/Qwen3-0.6B",
      VITE_LLM_BACKEND: "unlocked-browser-transformer",
      VITE_REQUIRE_UNLOCKED_RUNTIME: "true",
      VITE_REQUIRE_WEBGPU_KERNELS: "false",
      VITE_UNLOCKED_MANIFEST_FORMAT: "sharded",
      VITE_UNLOCKED_WEIGHT_FORMAT: "f16-packed",
      VITE_UNLOCKED_ALLOW_FIXTURE: "false",
      VITE_UNLOCKED_BACKEND_PREFERENCE: "cpu",
      VITE_MEMORY_PROVIDER: "browser-vector",
      VITE_QWEN_THINKING_MODE: "disabled",
      VITE_MTP_ENABLED: "false",
      VITE_MTP_DRAFT_MODEL_ID: "browser/qwen-prefix-drafter",
      VITE_MTP_NUM_SPECULATIVE_TOKENS: "2",
      VITE_MTP_DRAFT_LAYER_COUNT: "4",
      VITE_MTP_MIN_ACCEPTANCE_RATE: "0",
      VITE_UNLOCKED_RUNTIME_PROFILE: "full",
    });
  });

  it("keeps strict WebGPU out of Node verification unless the Node-specific flag is set", () => {
    expect(makeUnlockedVerifyEnvOverrides({
      RELEASE_REQUIRE_UNLOCKED_MODEL: "true",
      RELEASE_REQUIRE_UNLOCKED_STRICT_WEBGPU: "true",
    })).toMatchObject({
      RELEASE_REQUIRE_UNLOCKED_STRICT_WEBGPU: "false",
      VITE_REQUIRE_WEBGPU_KERNELS: "false",
      VITE_UNLOCKED_BACKEND_PREFERENCE: "cpu",
    });
    expect(makeUnlockedVerifyEnvOverrides({
      RELEASE_REQUIRE_UNLOCKED_MODEL: "true",
      RELEASE_REQUIRE_UNLOCKED_NODE_STRICT_WEBGPU: "true",
    })).toMatchObject({
      RELEASE_REQUIRE_UNLOCKED_STRICT_WEBGPU: "true",
      VITE_REQUIRE_WEBGPU_KERNELS: "true",
      VITE_UNLOCKED_BACKEND_PREFERENCE: "webgpu",
      RELEASE_REQUIRE_UNLOCKED_WEBGPU_MLP: "true",
      RELEASE_REQUIRE_UNLOCKED_WEBGPU_LOGITS: "true",
      RELEASE_REQUIRE_UNLOCKED_WEBGPU_ATTENTION: "true",
      RELEASE_REQUIRE_UNLOCKED_WEBGPU_PROJECTION: "true",
    });
  });

  it("requires full-profile browser benchmarks when strict unlocked model mode is enabled", () => {
    expect(makeUnlockedBenchmarkEnvOverrides({
      RELEASE_REQUIRE_UNLOCKED_MODEL: "true",
    })).toEqual({
      RELEASE_REQUIRE_UNLOCKED_MODEL: "true",
      RELEASE_REQUIRE_UNLOCKED_QWEN_MATH: "true",
      RELEASE_REQUIRE_UNLOCKED_QWEN_PARITY: "true",
      RELEASE_REQUIRE_UNLOCKED_FULL_PROFILE: "true",
      RELEASE_REQUIRE_BROWSER_PREVIEW_PROOF: "true",
      RELEASE_REQUIRE_UNLOCKED_KV_REUSE: "true",
      RELEASE_REQUIRE_UNLOCKED_STRICT_WEBGPU: "true",
      RELEASE_REQUIRE_UNLOCKED_PACKED_ASSETS: "true",
      VITE_DEFAULT_MODEL: "Qwen/Qwen3-0.6B",
      VITE_LLM_BACKEND: "unlocked-browser-transformer",
      VITE_REQUIRE_UNLOCKED_RUNTIME: "true",
      VITE_REQUIRE_WEBGPU_KERNELS: "true",
      VITE_UNLOCKED_MANIFEST_FORMAT: "sharded",
      VITE_UNLOCKED_WEIGHT_FORMAT: "f16-packed",
      VITE_UNLOCKED_ALLOW_FIXTURE: "false",
      VITE_UNLOCKED_BACKEND_PREFERENCE: "webgpu",
      VITE_MEMORY_PROVIDER: "browser-vector",
      VITE_QWEN_THINKING_MODE: "disabled",
      VITE_MTP_ENABLED: "false",
      VITE_MTP_DRAFT_MODEL_ID: "browser/qwen-prefix-drafter",
      VITE_MTP_NUM_SPECULATIVE_TOKENS: "2",
      VITE_MTP_DRAFT_LAYER_COUNT: "4",
      VITE_MTP_MIN_ACCEPTANCE_RATE: "0",
      VITE_UNLOCKED_RUNTIME_PROFILE: "full",
      BROWSER_RUNTIME_BENCH_STRICT: "true",
      BROWSER_RUNTIME_BENCH_REQUIRE_BROWSER_PREVIEW: "true",
      BROWSER_RUNTIME_BENCH_PREVIEW_TIMEOUT_MS: "120000",
      BROWSER_RUNTIME_BENCH_PREVIEW_MIN_GENERATED_TOKENS: "1",
      BROWSER_RUNTIME_BENCH_PREVIEW_REQUIRE_KV_REUSE: "true",
      BROWSER_RUNTIME_BENCH_PREVIEW_REQUIRE_KV_PREDICTIVE_PREFETCH: "true",
      BROWSER_RUNTIME_BENCH_PREVIEW_REQUIRE_STRICT_WEBGPU: "true",
      BROWSER_RUNTIME_BENCH_PREVIEW_MEMORY_GROUNDING_CASE: "qa_corpus_v1",
      BROWSER_RUNTIME_BENCH_PREVIEW_MEMORY_CORPUS_SIZE: "64",
      BROWSER_RUNTIME_BENCH_REQUIRE_STRICT_WEBGPU: "true",
      BROWSER_RUNTIME_BENCH_MAX_GENERATION_TOKENS: "16",
      BROWSER_RUNTIME_BENCH_PROMPTS: "What is the capital of Utah?|Write two clear sentences about Earth.",
      BROWSER_RUNTIME_BENCH_EXPECTED_SUBSTRINGS: "Salt Lake|Earth",
    });
  });

  it("passes strict WebGPU release intent through to the browser benchmark", () => {
    expect(makeUnlockedBenchmarkEnvOverrides({
      RELEASE_REQUIRE_UNLOCKED_MODEL: "true",
      RELEASE_REQUIRE_UNLOCKED_STRICT_WEBGPU: "true",
    })).toMatchObject({
      BROWSER_RUNTIME_BENCH_REQUIRE_STRICT_WEBGPU: "true",
    });
  });

  it("requires real Qwen parity when strict unlocked model mode is enabled", () => {
    expect(makeQwenParityEnvOverrides({
      RELEASE_REQUIRE_UNLOCKED_MODEL: "true",
      VITE_UNLOCKED_MODEL_MANIFEST_PATH: "/models/qwen/manifest.json",
    })).toEqual({
      RELEASE_REQUIRE_QWEN_ACCURACY_REAL_MODEL: "true",
      QWEN_PARITY_MANIFEST_PATH: "apps/web/public/models/qwen/manifest.json",
    });
  });

  it("preserves explicit filesystem Qwen parity paths", () => {
    expect(makeQwenParityEnvOverrides({
      RELEASE_REQUIRE_UNLOCKED_MODEL: "true",
      QWEN_PARITY_MANIFEST_PATH: ".artifacts/models/qwen/manifest.json",
      VITE_UNLOCKED_MODEL_MANIFEST_PATH: "/models/qwen/manifest.json",
    })).toMatchObject({
      QWEN_PARITY_MANIFEST_PATH: ".artifacts/models/qwen/manifest.json",
    });
  });

  it("keeps development preview verification available outside strict unlocked model mode", () => {
    expect(makeUnlockedVerifyEnvOverrides({
      RELEASE_REQUIRE_UNLOCKED_MODEL: "false",
    })).toEqual({});
    expect(makeUnlockedVerifyEnvOverrides({
      RELEASE_REQUIRE_UNLOCKED_MODEL: "false",
      VITE_UNLOCKED_MODEL_MANIFEST_PATH: "/models/qwen3-0.6b-unlocked/manifest.json",
      VITE_UNLOCKED_MODEL_MANIFEST_SHA256: "a".repeat(64),
    })).toEqual({});
    expect(makeUnlockedVerifyEnvOverrides({})).toEqual({});
  });

  it("isolates child eval artifacts under the current release-gate run directory", () => {
    expect(makeReleaseGateChildEnv(".artifacts/evals/release-gate/run-1", {
      VITE_DEFAULT_MODEL: "Qwen/Qwen3-0.6B",
    })).toEqual({
      VITE_DEFAULT_MODEL: "Qwen/Qwen3-0.6B",
      EVAL_ARTIFACT_DIR: ".artifacts/evals/release-gate/run-1/child-evals",
    });
  });

  it("keeps production eval browser-first when RELEASE_REQUIRE_SIDECAR is unset", () => {
    expect(makeProductionEvalEnvOverrides({})).toEqual({});
    expect(makeProductionEvalEnvOverrides({
      PRODUCTION_EVAL_SIDECAR_MODE: "skip",
    })).toEqual({
      PRODUCTION_EVAL_SIDECAR_MODE: "skip",
    });
  });

  it("makes production eval sidecar blocking only for sidecar-packaged release gates", () => {
    expect(makeProductionEvalEnvOverrides({
      RELEASE_REQUIRE_SIDECAR: "true",
      PRODUCTION_EVAL_SIDECAR_MODE: "skip",
    })).toEqual({
      PRODUCTION_EVAL_SIDECAR_MODE: "required",
    });
  });
});
