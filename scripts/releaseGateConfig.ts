import { existsSync, readFileSync } from "node:fs";
import { join } from "node:path";

const DEFAULT_QWEN_MODEL_ID = "Qwen/Qwen3-0.6B";
const DEFAULT_QWEN_MANIFEST_PATH = "/models/qwen3-0.6b-unlocked/manifest.json";
const DEFAULT_QWEN_MANIFEST_FORMAT = "sharded";
const DEFAULT_QWEN_WEIGHT_FORMAT = "f16-packed";
const DEFAULT_QWEN_BACKEND = "unlocked-browser-transformer";
const DEFAULT_QWEN_BACKEND_PREFERENCE = "webgpu";
const DEFAULT_MEMORY_PROVIDER = "browser-vector";
const DEFAULT_MTP_DRAFT_MODEL_ID = "browser/qwen-prefix-drafter";
const DEFAULT_BROWSER_MTP_SPECULATIVE_TOKENS = "2";
const MAX_BROWSER_MTP_SPECULATIVE_TOKENS = 3;
const DEFAULT_STRICT_BROWSER_BENCH_PROMPTS = "What is the capital of Utah?|Write two clear sentences about Earth.";
const DEFAULT_STRICT_BROWSER_BENCH_GENERATION_TOKENS = "16";

export function makeReleaseGateDefaultEnvOverrides(env: NodeJS.ProcessEnv): Record<string, string | undefined> {
  if (env.RELEASE_REQUIRE_V12_PRODUCTION === "true" && env.RELEASE_REQUIRE_UNLOCKED_MODEL !== "true") {
    return {};
  }
  return makeStrictRealQwenEnvOverrides(env, { detectInstalledDefaultModel: true });
}

export function makeUnlockedVerifyEnvOverrides(env: NodeJS.ProcessEnv): Record<string, string | undefined> {
  const strictEnv = makeStrictRealQwenEnvOverrides(env);
  if (strictEnv.RELEASE_REQUIRE_UNLOCKED_MODEL !== "true") return {};
  const strictNodeWebGpu = readBooleanEnv(env.RELEASE_REQUIRE_UNLOCKED_NODE_STRICT_WEBGPU, false);
  return {
    ...strictEnv,
    RELEASE_REQUIRE_UNLOCKED_FULL_PROFILE: "true",
    RELEASE_REQUIRE_UNLOCKED_KV_REUSE: "true",
    RELEASE_REQUIRE_UNLOCKED_STRICT_WEBGPU: strictNodeWebGpu ? "true" : "false",
    VITE_REQUIRE_WEBGPU_KERNELS: strictNodeWebGpu ? "true" : "false",
    VITE_UNLOCKED_BACKEND_PREFERENCE: env.VITE_UNLOCKED_VERIFY_BACKEND_PREFERENCE ?? (strictNodeWebGpu ? "webgpu" : "cpu"),
    VITE_MTP_ENABLED: env.VITE_UNLOCKED_VERIFY_MTP_ENABLED ?? env.UNLOCKED_VERIFY_MTP_ENABLED ?? "false",
    ...(strictNodeWebGpu ? {
      RELEASE_REQUIRE_UNLOCKED_WEBGPU_MLP: "true",
      RELEASE_REQUIRE_UNLOCKED_WEBGPU_LOGITS: "true",
      RELEASE_REQUIRE_UNLOCKED_WEBGPU_ATTENTION: "true",
      RELEASE_REQUIRE_UNLOCKED_WEBGPU_PROJECTION: "true",
    } : {}),
    VITE_MTP_DRAFT_MODEL_ID: env.VITE_MTP_DRAFT_MODEL_ID ?? strictEnv.VITE_MTP_DRAFT_MODEL_ID ?? DEFAULT_MTP_DRAFT_MODEL_ID,
    VITE_MTP_NUM_SPECULATIVE_TOKENS: resolveBrowserMtpSpeculativeTokens(env.VITE_MTP_NUM_SPECULATIVE_TOKENS ?? strictEnv.VITE_MTP_NUM_SPECULATIVE_TOKENS),
    VITE_MTP_MIN_ACCEPTANCE_RATE: env.VITE_MTP_MIN_ACCEPTANCE_RATE ?? "0",
    VITE_MTP_DRAFT_LAYER_COUNT: env.VITE_MTP_DRAFT_LAYER_COUNT ?? "4",
    VITE_UNLOCKED_RUNTIME_PROFILE: env.VITE_UNLOCKED_RUNTIME_PROFILE ?? strictEnv.VITE_UNLOCKED_RUNTIME_PROFILE ?? "full",
  };
}

export function makeUnlockedBenchmarkEnvOverrides(env: NodeJS.ProcessEnv): Record<string, string | undefined> {
  const strictEnv = makeStrictRealQwenEnvOverrides(env);
  if (strictEnv.RELEASE_REQUIRE_UNLOCKED_MODEL !== "true") return {};
  const strictWebGpu = readBooleanEnv(
    env.RELEASE_REQUIRE_UNLOCKED_STRICT_WEBGPU ?? strictEnv.RELEASE_REQUIRE_UNLOCKED_STRICT_WEBGPU,
    false,
  );
  return {
    ...strictEnv,
    RELEASE_REQUIRE_UNLOCKED_FULL_PROFILE: "true",
    RELEASE_REQUIRE_UNLOCKED_KV_REUSE: "true",
    VITE_MTP_ENABLED: env.RELEASE_REQUIRE_MTP_ACCELERATION === "true"
      ? "true"
      : (env.VITE_BROWSER_BENCH_MTP_ENABLED ?? env.BROWSER_RUNTIME_BENCH_MTP_ENABLED ?? "false"),
    BROWSER_RUNTIME_BENCH_STRICT: "true",
    BROWSER_RUNTIME_BENCH_REQUIRE_BROWSER_PREVIEW: env.BROWSER_RUNTIME_BENCH_REQUIRE_BROWSER_PREVIEW ?? "true",
    BROWSER_RUNTIME_BENCH_PREVIEW_TIMEOUT_MS: env.BROWSER_RUNTIME_BENCH_PREVIEW_TIMEOUT_MS ?? "120000",
    BROWSER_RUNTIME_BENCH_PREVIEW_MIN_GENERATED_TOKENS: env.BROWSER_RUNTIME_BENCH_PREVIEW_MIN_GENERATED_TOKENS ?? "1",
    BROWSER_RUNTIME_BENCH_PREVIEW_REQUIRE_KV_REUSE: env.BROWSER_RUNTIME_BENCH_PREVIEW_REQUIRE_KV_REUSE ?? "true",
    BROWSER_RUNTIME_BENCH_PREVIEW_REQUIRE_KV_PREDICTIVE_PREFETCH: env.BROWSER_RUNTIME_BENCH_PREVIEW_REQUIRE_KV_PREDICTIVE_PREFETCH ?? "true",
    BROWSER_RUNTIME_BENCH_PREVIEW_REQUIRE_STRICT_WEBGPU: env.BROWSER_RUNTIME_BENCH_PREVIEW_REQUIRE_STRICT_WEBGPU ?? "true",
    BROWSER_RUNTIME_BENCH_PREVIEW_MEMORY_GROUNDING_CASE: env.BROWSER_RUNTIME_BENCH_PREVIEW_MEMORY_GROUNDING_CASE ?? "qa_corpus_v1",
    BROWSER_RUNTIME_BENCH_PREVIEW_MEMORY_CORPUS_SIZE: env.BROWSER_RUNTIME_BENCH_PREVIEW_MEMORY_CORPUS_SIZE ?? "64",
    BROWSER_RUNTIME_BENCH_MAX_GENERATION_TOKENS: env.BROWSER_RUNTIME_BENCH_MAX_GENERATION_TOKENS ?? DEFAULT_STRICT_BROWSER_BENCH_GENERATION_TOKENS,
    BROWSER_RUNTIME_BENCH_PROMPTS: env.BROWSER_RUNTIME_BENCH_PROMPTS ?? DEFAULT_STRICT_BROWSER_BENCH_PROMPTS,
    BROWSER_RUNTIME_BENCH_EXPECTED_SUBSTRINGS: env.BROWSER_RUNTIME_BENCH_EXPECTED_SUBSTRINGS ?? "Salt Lake|Earth",
    ...(strictWebGpu ? {
      BROWSER_RUNTIME_BENCH_REQUIRE_STRICT_WEBGPU: "true",
    } : {}),
    ...(env.RELEASE_REQUIRE_MTP_ACCELERATION === "true" ? {
      VITE_MTP_ENABLED: "true",
      BROWSER_RUNTIME_BENCH_REQUIRE_MTP_ACCELERATION: "true",
      BROWSER_RUNTIME_BENCH_MIN_MTP_ACCEPTANCE_RATE: env.BROWSER_RUNTIME_BENCH_MIN_MTP_ACCEPTANCE_RATE ?? "0.25",
      BROWSER_RUNTIME_BENCH_MIN_MTP_NET_SPEEDUP: env.BROWSER_RUNTIME_BENCH_MIN_MTP_NET_SPEEDUP ?? "1.05",
    } : {}),
  };
}

export function makeQwenParityEnvOverrides(env: NodeJS.ProcessEnv): Record<string, string | undefined> {
  const strictEnv = makeStrictRealQwenEnvOverrides(env);
  if (strictEnv.RELEASE_REQUIRE_UNLOCKED_MODEL !== "true") return {};
  const manifestPath = env.QWEN_PARITY_MANIFEST_PATH
    ?? env.VITE_UNLOCKED_MODEL_MANIFEST_PATH
    ?? strictEnv.VITE_UNLOCKED_MODEL_MANIFEST_PATH
    ?? "apps/web/public/models/qwen3-0.6b-unlocked/manifest.json";
  return {
    RELEASE_REQUIRE_QWEN_ACCURACY_REAL_MODEL: "true",
    QWEN_PARITY_MANIFEST_PATH: toNodeModelManifestPath(manifestPath),
  };
}

export function makeProductionEvalEnvOverrides(env: NodeJS.ProcessEnv): Record<string, string | undefined> {
  if (readBooleanEnv(env.RELEASE_REQUIRE_SIDECAR, false)) {
    return { PRODUCTION_EVAL_SIDECAR_MODE: "required" };
  }
  return env.PRODUCTION_EVAL_SIDECAR_MODE
    ? { PRODUCTION_EVAL_SIDECAR_MODE: env.PRODUCTION_EVAL_SIDECAR_MODE }
    : {};
}

export function makeReleaseGateChildEnv(
  suiteDir: string,
  envOverrides: Record<string, string | undefined> = {},
): Record<string, string | undefined> {
  return {
    ...envOverrides,
    EVAL_ARTIFACT_DIR: join(suiteDir, "child-evals"),
  };
}

export function makeReleaseGateTestEnvOverrides(_env: NodeJS.ProcessEnv): Record<string, string | undefined> {
  return {
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
  };
}

function readBooleanEnv(value: string | undefined, fallback: boolean): boolean {
  if (value === undefined || value === "") return fallback;
  return value === "true";
}

function makeStrictRealQwenEnvOverrides(
  env: NodeJS.ProcessEnv,
  options: { detectInstalledDefaultModel?: boolean } = {},
): Record<string, string | undefined> {
  const explicitStrict = env.RELEASE_REQUIRE_UNLOCKED_MODEL === "true";
  const explicitOptOut = env.RELEASE_REQUIRE_UNLOCKED_MODEL === "false";
  const configured = resolveConfiguredQwenModel(env, options);
  const modelPresent = Boolean(configured.manifestPath && configured.manifestSha256);
  const fixtureEscape = env.RELEASE_ALLOW_FIXTURE_GATE === "true";
  const ciWithoutModel = env.CI === "true" && !modelPresent;

  if (explicitOptOut) return {};
  if (!explicitStrict && (fixtureEscape || ciWithoutModel || !modelPresent)) return {};

  return {
    RELEASE_REQUIRE_UNLOCKED_MODEL: "true",
    RELEASE_REQUIRE_UNLOCKED_QWEN_MATH: "true",
    RELEASE_REQUIRE_UNLOCKED_QWEN_PARITY: "true",
    RELEASE_REQUIRE_UNLOCKED_FULL_PROFILE: "true",
    RELEASE_REQUIRE_UNLOCKED_STRICT_WEBGPU: env.RELEASE_REQUIRE_UNLOCKED_STRICT_WEBGPU ?? "true",
    RELEASE_REQUIRE_UNLOCKED_PACKED_ASSETS: env.RELEASE_REQUIRE_UNLOCKED_PACKED_ASSETS ?? "true",
    RELEASE_REQUIRE_BROWSER_PREVIEW_PROOF: env.RELEASE_REQUIRE_BROWSER_PREVIEW_PROOF ?? "true",
    VITE_DEFAULT_MODEL: env.VITE_DEFAULT_MODEL ?? DEFAULT_QWEN_MODEL_ID,
    VITE_LLM_BACKEND: env.VITE_LLM_BACKEND ?? DEFAULT_QWEN_BACKEND,
    VITE_REQUIRE_UNLOCKED_RUNTIME: env.VITE_REQUIRE_UNLOCKED_RUNTIME ?? "true",
    VITE_REQUIRE_WEBGPU_KERNELS: env.VITE_REQUIRE_WEBGPU_KERNELS ?? "true",
    ...(configured.manifestPath ? { VITE_UNLOCKED_MODEL_MANIFEST_PATH: configured.manifestPath } : {}),
    ...(configured.manifestSha256 ? { VITE_UNLOCKED_MODEL_MANIFEST_SHA256: configured.manifestSha256 } : {}),
    VITE_UNLOCKED_MANIFEST_FORMAT: env.VITE_UNLOCKED_MANIFEST_FORMAT ?? DEFAULT_QWEN_MANIFEST_FORMAT,
    VITE_UNLOCKED_WEIGHT_FORMAT: env.VITE_UNLOCKED_WEIGHT_FORMAT ?? DEFAULT_QWEN_WEIGHT_FORMAT,
    VITE_UNLOCKED_ALLOW_FIXTURE: "false",
    VITE_UNLOCKED_BACKEND_PREFERENCE: env.VITE_UNLOCKED_BACKEND_PREFERENCE ?? DEFAULT_QWEN_BACKEND_PREFERENCE,
    VITE_MEMORY_PROVIDER: env.VITE_MEMORY_PROVIDER ?? DEFAULT_MEMORY_PROVIDER,
    VITE_QWEN_THINKING_MODE: env.VITE_QWEN_THINKING_MODE ?? "disabled",
    VITE_MTP_ENABLED: env.VITE_MTP_ENABLED ?? "false",
    VITE_MTP_DRAFT_MODEL_ID: env.VITE_MTP_DRAFT_MODEL_ID ?? DEFAULT_MTP_DRAFT_MODEL_ID,
    VITE_MTP_NUM_SPECULATIVE_TOKENS: resolveBrowserMtpSpeculativeTokens(env.VITE_MTP_NUM_SPECULATIVE_TOKENS),
    VITE_MTP_MIN_ACCEPTANCE_RATE: env.VITE_MTP_MIN_ACCEPTANCE_RATE ?? "0",
    VITE_MTP_DRAFT_LAYER_COUNT: env.VITE_MTP_DRAFT_LAYER_COUNT ?? "4",
    VITE_UNLOCKED_RUNTIME_PROFILE: env.VITE_UNLOCKED_RUNTIME_PROFILE ?? "full",
  };
}

function resolveBrowserMtpSpeculativeTokens(value: string | undefined): string {
  if (!value?.trim()) return DEFAULT_BROWSER_MTP_SPECULATIVE_TOKENS;
  const parsed = Number(value);
  if (!Number.isFinite(parsed)) return DEFAULT_BROWSER_MTP_SPECULATIVE_TOKENS;
  const integer = Math.floor(parsed);
  if (integer <= 0) return DEFAULT_BROWSER_MTP_SPECULATIVE_TOKENS;
  return String(Math.min(integer, MAX_BROWSER_MTP_SPECULATIVE_TOKENS));
}

function resolveConfiguredQwenModel(
  env: NodeJS.ProcessEnv,
  options: { detectInstalledDefaultModel?: boolean },
): { manifestPath?: string; manifestSha256?: string } {
  const envManifestPath = env.VITE_UNLOCKED_MODEL_MANIFEST_PATH;
  const envManifestSha256 = env.VITE_UNLOCKED_MODEL_MANIFEST_SHA256;
  if (envManifestPath || envManifestSha256) {
    const configured: { manifestPath?: string; manifestSha256?: string } = {};
    if (envManifestPath) configured.manifestPath = envManifestPath;
    const manifestSha256 = envManifestSha256 ?? readManifestSha256(envManifestPath);
    if (manifestSha256) configured.manifestSha256 = manifestSha256;
    return configured;
  }
  if (!options.detectInstalledDefaultModel) return {};

  const defaultNodePath = toNodeModelManifestPath(DEFAULT_QWEN_MANIFEST_PATH);
  if (!existsSync(defaultNodePath)) return {};
  const configured: { manifestPath?: string; manifestSha256?: string } = {
    manifestPath: DEFAULT_QWEN_MANIFEST_PATH,
  };
  const manifestSha256 = readManifestSha256(DEFAULT_QWEN_MANIFEST_PATH);
  if (manifestSha256) configured.manifestSha256 = manifestSha256;
  return configured;
}

function readManifestSha256(manifestPath: string | undefined): string | undefined {
  if (!manifestPath) return undefined;
  const shaPath = `${toNodeModelManifestPath(manifestPath)}.sha256`;
  if (!existsSync(shaPath)) return undefined;
  return parseManifestSha256Sidecar(readFileSync(shaPath, "utf8"));
}

function toNodeModelManifestPath(manifestPath: string): string {
  if (manifestPath.startsWith("/models/")) return join("apps/web/public", manifestPath.slice(1));
  return manifestPath;
}

export function parseManifestSha256Sidecar(value: string): string {
  const firstToken = value.trim().split(/\s+/)[0] ?? "";
  return firstToken.trim();
}
