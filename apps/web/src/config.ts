import {
  normalizeMemoryProviderMode,
  type AgentRuntimeMtpConfig,
  type AgentRuntimeConfig,
  type MemoryProviderMode
} from "@infinite-edge-agent/core";
import type { AdvancedRuntimeModelProfile, DeviceProfile, InferenceBackendProfile } from "@infinite-edge-agent/core";
import {
  resolveUnlockedRuntimeProfile,
  type UnlockedRuntimeCaps,
  type UnlockedRuntimeProfileResolution,
} from "./lib/runtime/unlockedRuntimeProfile";
import {
  LOCAL_BROWSER_MTP_DEFAULT_SPECULATIVE_TOKENS,
  LOCAL_BROWSER_MTP_MAX_SPECULATIVE_TOKENS,
} from "./lib/llm/unlockedBrowserTransformerClient";

export {
  assertUnlockedFullProfile,
  resolveUnlockedRuntimeProfile,
  type UnlockedRuntimeProfileName,
  type UnlockedRuntimeProfileResolution,
} from "./lib/runtime/unlockedRuntimeProfile";
export { normalizeMemoryProviderMode } from "@infinite-edge-agent/core";

export const PRODUCTION_MODE = import.meta.env.PROD || import.meta.env.VITE_PRODUCTION_MODE === "true";
export const QWEN3_UNLOCKED_MODEL_ID = "Qwen/Qwen3-0.6B";
export const QWEN3_COMPILED_WEBLLM_MODEL_ID = "Qwen3-0.6B-q4f16_1-MLC";
export const DEFAULT_EMBEDDING_MODEL = import.meta.env.VITE_EMBEDDING_MODEL ?? "Xenova/all-MiniLM-L6-v2";
export const EMBEDDING_PREFER_WEBGPU = import.meta.env.VITE_EMBEDDING_PREFER_WEBGPU === "true";
export const DEFAULT_LLM_BACKEND = import.meta.env.VITE_LLM_BACKEND ?? "unlocked-browser-transformer";
export const DEFAULT_MODEL = resolveDefaultModel(import.meta.env, DEFAULT_LLM_BACKEND);
export const QWEN3_UNLOCKED_LOCAL_MANIFEST_PATH = "/models/qwen3-0.6b-unlocked/manifest.json";
export const QWEN3_UNLOCKED_LOCAL_MANIFEST_SHA256 = "6f89087e755568187951956a39077a3d9e5538768399e982febf76364e4caebd";
export const QWEN3_CONTEXT_WINDOW_TOKENS = 40_960;
export const DEFAULT_CHAT_MAX_GENERATION_TOKENS = 256;
export type QwenThinkingMode = "disabled" | "enabled";

export interface UnlockedModelAssetConfig {
  manifestPath: string;
  manifestSha256: string;
  manifestFormat: string;
  allowFixture: boolean;
  backendPreference: string;
}

export interface UnlockedModelAssetEnv {
  VITE_UNLOCKED_MODEL_MANIFEST_PATH?: string | undefined;
  VITE_UNLOCKED_MODEL_MANIFEST_SHA256?: string | undefined;
  VITE_UNLOCKED_MANIFEST_FORMAT?: string | undefined;
  VITE_UNLOCKED_ALLOW_FIXTURE?: string | undefined;
  VITE_UNLOCKED_BACKEND_PREFERENCE?: string | undefined;
}

export interface RequireWebGpuKernelsEnv {
  VITE_REQUIRE_WEBGPU_KERNELS?: string | undefined;
}

export interface CompiledWebLlmEnv {
  VITE_COMPILED_WEBLLM_ENABLED?: string | undefined;
}

export interface DefaultModelEnv {
  VITE_DEFAULT_MODEL?: string | undefined;
}

export function resolveDefaultModel(env: DefaultModelEnv, backend: string): string {
  const explicitModel = env.VITE_DEFAULT_MODEL?.trim();
  if (explicitModel) return explicitModel;
  return backend === "compiled-browser-webllm"
    ? QWEN3_COMPILED_WEBLLM_MODEL_ID
    : QWEN3_UNLOCKED_MODEL_ID;
}

export function resolveCompiledWebLlmEnabled(env: CompiledWebLlmEnv): boolean {
  return env.VITE_COMPILED_WEBLLM_ENABLED === "true";
}

export function resolveUnlockedModelAssetConfig(env: UnlockedModelAssetEnv, _production = false): UnlockedModelAssetConfig {
  const allowFixture = env.VITE_UNLOCKED_ALLOW_FIXTURE === "true";
  const manifestPath = env.VITE_UNLOCKED_MODEL_MANIFEST_PATH?.trim() || (allowFixture ? "" : QWEN3_UNLOCKED_LOCAL_MANIFEST_PATH);
  const manifestSha256 = env.VITE_UNLOCKED_MODEL_MANIFEST_SHA256?.trim()
    || (manifestPath === QWEN3_UNLOCKED_LOCAL_MANIFEST_PATH ? QWEN3_UNLOCKED_LOCAL_MANIFEST_SHA256 : "");
  return {
    manifestPath,
    manifestSha256,
    manifestFormat: env.VITE_UNLOCKED_MANIFEST_FORMAT?.trim() || (manifestPath ? "sharded" : ""),
    allowFixture,
    backendPreference: env.VITE_UNLOCKED_BACKEND_PREFERENCE?.trim() || "webgpu",
  };
}

export const UNLOCKED_MODEL_ASSET_CONFIG = resolveUnlockedModelAssetConfig(import.meta.env, PRODUCTION_MODE);
export const UNLOCKED_MODEL_MANIFEST_PATH = UNLOCKED_MODEL_ASSET_CONFIG.manifestPath;
export const UNLOCKED_MODEL_MANIFEST_SHA256 = UNLOCKED_MODEL_ASSET_CONFIG.manifestSha256;
export const UNLOCKED_MANIFEST_FORMAT = UNLOCKED_MODEL_ASSET_CONFIG.manifestFormat;
export const UNLOCKED_ALLOW_FIXTURE = UNLOCKED_MODEL_ASSET_CONFIG.allowFixture;
export const UNLOCKED_BACKEND_PREFERENCE = UNLOCKED_MODEL_ASSET_CONFIG.backendPreference;
export const COMPILED_WEBLLM_ENABLED = resolveCompiledWebLlmEnabled(import.meta.env);
export const UNLOCKED_RUNTIME_PROFILE_RESOLUTION: UnlockedRuntimeProfileResolution = resolveUnlockedRuntimeProfile(import.meta.env);
export const UNLOCKED_RUNTIME_PROFILE = UNLOCKED_RUNTIME_PROFILE_RESOLUTION.profile;
export const UNLOCKED_RUNTIME_CAPS = UNLOCKED_RUNTIME_PROFILE_RESOLUTION.caps;
export const UNLOCKED_RUNTIME_CAP_STATUS = UNLOCKED_RUNTIME_PROFILE_RESOLUTION.capStatus;
export const UNLOCKED_MAX_RUNTIME_PROMPT_TOKENS = UNLOCKED_RUNTIME_CAPS.maxRuntimePromptTokens;
export const UNLOCKED_MAX_RUNTIME_LAYERS = UNLOCKED_RUNTIME_CAPS.maxRuntimeLayers;
export const UNLOCKED_LOGIT_CANDIDATE_LIMIT = UNLOCKED_RUNTIME_CAPS.logitCandidateLimit;
export const UNLOCKED_MAX_GENERATION_TOKENS = UNLOCKED_RUNTIME_CAPS.maxGenerationTokens;
export const REQUIRE_WEBGPU_KERNELS = resolveRequireWebGpuKernels(import.meta.env, DEFAULT_LLM_BACKEND);
export interface InteractiveRuntimeLimits {
  maxRuntimePromptTokens: number | null;
  maxRuntimeLayers: number | null;
  logitCandidateLimit: number | null;
  logitTopK: number;
  logitTileRows: number;
  maxGenerationTokens: number;
}

export type InteractiveRuntimeLimitsEnv = Partial<Record<
  | "VITE_CHAT_MAX_RUNTIME_PROMPT_TOKENS"
  | "VITE_CHAT_MAX_RUNTIME_LAYERS"
  | "VITE_CHAT_LOGIT_CANDIDATE_LIMIT"
  | "VITE_CHAT_LOGIT_TOP_K"
  | "VITE_CHAT_LOGIT_TILE_ROWS"
  | "VITE_CHAT_MAX_GENERATION_TOKENS"
  | "VITE_QWEN_THINKING_MODE",
  string | undefined
>>;

const DEFAULT_INTERACTIVE_RUNTIME_LIMITS = {
  maxRuntimePromptTokens: null,
  maxRuntimeLayers: null,
  logitCandidateLimit: null,
  logitTopK: 1,
  logitTileRows: 32768,
  maxGenerationTokens: DEFAULT_CHAT_MAX_GENERATION_TOKENS,
} satisfies InteractiveRuntimeLimits;

export function resolveInteractiveRuntimeLimits(
  env: InteractiveRuntimeLimitsEnv,
  runtimeCaps: UnlockedRuntimeCaps,
): InteractiveRuntimeLimits {
  return {
    maxRuntimePromptTokens: resolveOptionalInteractiveCap(
      env.VITE_CHAT_MAX_RUNTIME_PROMPT_TOKENS,
      "VITE_CHAT_MAX_RUNTIME_PROMPT_TOKENS",
      runtimeCaps.maxRuntimePromptTokens,
      DEFAULT_INTERACTIVE_RUNTIME_LIMITS.maxRuntimePromptTokens,
    ),
    maxRuntimeLayers: resolveOptionalInteractiveCap(
      env.VITE_CHAT_MAX_RUNTIME_LAYERS,
      "VITE_CHAT_MAX_RUNTIME_LAYERS",
      runtimeCaps.maxRuntimeLayers,
      DEFAULT_INTERACTIVE_RUNTIME_LIMITS.maxRuntimeLayers,
    ),
    logitCandidateLimit: resolveOptionalInteractiveCap(
      env.VITE_CHAT_LOGIT_CANDIDATE_LIMIT,
      "VITE_CHAT_LOGIT_CANDIDATE_LIMIT",
      null,
      DEFAULT_INTERACTIVE_RUNTIME_LIMITS.logitCandidateLimit,
    ),
    logitTopK: readPositiveIntegerEnv(env.VITE_CHAT_LOGIT_TOP_K)
      ?? DEFAULT_INTERACTIVE_RUNTIME_LIMITS.logitTopK,
    logitTileRows: readPositiveIntegerEnv(env.VITE_CHAT_LOGIT_TILE_ROWS)
      ?? DEFAULT_INTERACTIVE_RUNTIME_LIMITS.logitTileRows,
    maxGenerationTokens: resolveInteractiveGenerationTokens(
      env.VITE_CHAT_MAX_GENERATION_TOKENS,
      runtimeCaps.maxGenerationTokens,
    ),
  };
}

export const CHAT_RUNTIME_LIMITS = resolveInteractiveRuntimeLimits(import.meta.env, UNLOCKED_RUNTIME_CAPS);
export const CHAT_MAX_RUNTIME_PROMPT_TOKENS = CHAT_RUNTIME_LIMITS.maxRuntimePromptTokens;
export const CHAT_MAX_RUNTIME_LAYERS = CHAT_RUNTIME_LIMITS.maxRuntimeLayers;
export const CHAT_LOGIT_CANDIDATE_LIMIT = CHAT_RUNTIME_LIMITS.logitCandidateLimit;
export const CHAT_LOGIT_TOP_K = CHAT_RUNTIME_LIMITS.logitTopK;
export const CHAT_LOGIT_TILE_ROWS = CHAT_RUNTIME_LIMITS.logitTileRows;
export const QWEN_THINKING_MODE = resolveQwenThinkingMode(import.meta.env.VITE_QWEN_THINKING_MODE);
export interface MtpEnabledEnv {
  VITE_MTP_ENABLED?: string | undefined;
}

export function resolveMtpEnabled(env: MtpEnabledEnv): boolean {
  return env.VITE_MTP_ENABLED === "true";
}

export const MTP_ENABLED = resolveMtpEnabled(import.meta.env);
export const BROWSER_NGRAM_MTP_DRAFT_MODEL_ID = "browser/ngram-drafter";
export const BROWSER_QWEN_PREFIX_MTP_DRAFT_MODEL_ID = "browser/qwen-prefix-drafter";
export const MTP_DRAFT_MODEL_ID = import.meta.env.VITE_MTP_DRAFT_MODEL_ID?.trim() || BROWSER_QWEN_PREFIX_MTP_DRAFT_MODEL_ID;
export const MTP_LOCAL_BROWSER_DRAFTER_CONFIGURED = isSupportedBrowserMtpDrafter(MTP_DRAFT_MODEL_ID);
export const MTP_NUM_SPECULATIVE_TOKENS = normalizeBrowserMtpDraftWindow(
  readPositiveIntegerEnv(import.meta.env.VITE_MTP_NUM_SPECULATIVE_TOKENS) ?? LOCAL_BROWSER_MTP_DEFAULT_SPECULATIVE_TOKENS,
);
export const MTP_MIN_ACCEPTANCE_RATE = readRatioEnv(import.meta.env.VITE_MTP_MIN_ACCEPTANCE_RATE) ?? 0;
export const MTP_DISABLE_WHEN_LATENCY_WORSE = import.meta.env.VITE_MTP_DISABLE_WHEN_LATENCY_WORSE !== "false";
export const MTP_DRAFT_LAYER_COUNT = readPositiveIntegerEnv(import.meta.env.VITE_MTP_DRAFT_LAYER_COUNT) ?? 4;
export const CHAT_MAX_GENERATION_TOKENS = CHAT_RUNTIME_LIMITS.maxGenerationTokens;
export const KVSWAP_PERSISTENCE_ENABLED =
  import.meta.env.VITE_KVSWAP_PERSISTENCE_ENABLED === "true"
  || (import.meta.env.VITE_KVSWAP_PERSISTENCE_ENABLED !== "false" && DEFAULT_LLM_BACKEND === "unlocked-browser-transformer");
export const KVSWAP_PERSISTENCE_PREFER_OPFS = import.meta.env.VITE_KVSWAP_PERSISTENCE_PREFER_OPFS !== "false";
export const KVSWAP_PERSISTENCE_MAX_BLOCKS = readPositiveIntegerEnv(import.meta.env.VITE_KVSWAP_PERSISTENCE_MAX_BLOCKS) ?? 512;
export const KVSWAP_PERSISTENCE_MAX_BYTES = readPositiveIntegerEnv(import.meta.env.VITE_KVSWAP_PERSISTENCE_MAX_BYTES) ?? 256 * 1024 * 1024;
export const KVSWAP_PERSISTENCE_CLEAR_ON_INIT = import.meta.env.VITE_KVSWAP_PERSISTENCE_CLEAR_ON_INIT === "true";
export const REQUIRE_UNLOCKED_RUNTIME =
  import.meta.env.VITE_REQUIRE_UNLOCKED_RUNTIME === "true"
  || (import.meta.env.VITE_REQUIRE_UNLOCKED_RUNTIME !== "false" && DEFAULT_LLM_BACKEND === "unlocked-browser-transformer");
export const MEMORY_PROVIDER = normalizeMemoryProviderMode(import.meta.env.VITE_MEMORY_PROVIDER);
export const ALLOW_MEMORY_FALLBACK = resolveMemoryFallback(import.meta.env.VITE_ALLOW_MEMORY_FALLBACK);
export const REMOTE_MEMORY_URL = import.meta.env.VITE_REMOTE_MEMORY_URL ?? "";
export const REMOTE_MEMORY_TOKEN = import.meta.env.VITE_REMOTE_MEMORY_TOKEN ?? "";
export const HAS_PUBLIC_REMOTE_MEMORY_TOKEN = Boolean(import.meta.env.VITE_REMOTE_MEMORY_TOKEN);
export const REMOTE_MEMORY_CREDENTIALS = normalizeRemoteMemoryCredentials(import.meta.env.VITE_REMOTE_MEMORY_CREDENTIALS);
export const REMOTE_MEMORY_TENANT_ID = import.meta.env.VITE_REMOTE_MEMORY_TENANT_ID;
export const REMOTE_MEMORY_CELL_ID = import.meta.env.VITE_REMOTE_MEMORY_CELL_ID;
export const MEMORY_TENANT_ID = REMOTE_MEMORY_TENANT_ID ?? "local";
export const MEMORY_CELL_ID = REMOTE_MEMORY_CELL_ID ?? "browser";

function normalizeRemoteMemoryCredentials(value: string | undefined): RequestCredentials {
  if (value === "include" || value === "omit" || value === "same-origin") return value;
  return "same-origin";
}

export function resolveMemoryFallback(value: string | undefined): boolean {
  return value?.trim().toLowerCase() !== "false";
}

export function resolveQwenThinkingMode(value: string | undefined): QwenThinkingMode {
  const normalized = value?.trim().toLowerCase();
  if (normalized === "enabled") return "enabled";
  return "disabled";
}

export function resolveRequireWebGpuKernels(env: RequireWebGpuKernelsEnv, llmBackend: string): boolean {
  const explicit = env.VITE_REQUIRE_WEBGPU_KERNELS?.trim().toLowerCase();
  if (explicit === "true") return true;
  if (explicit === "false") return false;
  return llmBackend === "unlocked-browser-transformer";
}

export function resolveAgentMaxPromptTokens(value: string | undefined, fallback = QWEN3_CONTEXT_WINDOW_TOKENS): number {
  return readPositiveIntegerEnv(value) ?? fallback;
}

function readPositiveIntegerEnv(value: string | undefined): number | undefined {
  if (!value?.trim()) return undefined;
  const parsed = Number(value);
  if (!Number.isFinite(parsed)) return undefined;
  const integer = Math.floor(parsed);
  return integer > 0 ? integer : undefined;
}

function resolveOptionalInteractiveCap(
  value: string | undefined,
  envName: string,
  runtimeCap: number | null,
  fallback: number | null,
): number | null {
  const parsed = readOptionalInteractiveCap(value, envName);
  if (parsed !== undefined) return parsed;
  return runtimeCap ?? fallback;
}

function resolveInteractiveGenerationTokens(
  value: string | undefined,
  runtimeCap: number | null,
): number {
  const parsed = readOptionalInteractiveCap(value, "VITE_CHAT_MAX_GENERATION_TOKENS");
  if (parsed === null) return QWEN3_CONTEXT_WINDOW_TOKENS;
  if (parsed !== undefined) return parsed;
  return runtimeCap ?? DEFAULT_INTERACTIVE_RUNTIME_LIMITS.maxGenerationTokens;
}

function readOptionalInteractiveCap(value: string | undefined, envName: string): number | null | undefined {
  if (!value?.trim()) return undefined;
  const normalized = value.trim().toLowerCase();
  if (normalized === "full" || normalized === "none" || normalized === "uncapped") return null;
  const parsed = Number(value);
  if (!Number.isFinite(parsed) || !Number.isInteger(parsed) || parsed <= 0) {
    throw new Error(`${envName} must be a positive integer or "full", received "${value}".`);
  }
  return parsed;
}

function readRatioEnv(value: string | undefined): number | undefined {
  if (!value?.trim()) return undefined;
  const parsed = Number(value);
  if (!Number.isFinite(parsed)) return undefined;
  return Math.max(0, Math.min(1, parsed));
}

export const AGENT_MAX_PROMPT_TOKENS = resolveAgentMaxPromptTokens(import.meta.env.VITE_AGENT_MAX_PROMPT_TOKENS);

export const AGENT_CONFIG: AgentRuntimeConfig = {
  modelId: DEFAULT_MODEL,
  embeddingModelId: DEFAULT_EMBEDDING_MODEL,
  memoryTopK: 8,
  maxRetrievedMemoryTokens: 1800,
  maxRecentConversationTokens: 2000,
  maxPromptTokens: AGENT_MAX_PROMPT_TOKENS,
  maxGenerationTokens: CHAT_MAX_GENERATION_TOKENS,
  mtp: makeRuntimeMtpConfig({
    enabled: MTP_ENABLED,
    draftModelId: MTP_DRAFT_MODEL_ID,
    numSpeculativeTokens: MTP_NUM_SPECULATIVE_TOKENS,
    minAcceptanceRate: MTP_MIN_ACCEPTANCE_RATE,
    disableWhenLatencyWorse: MTP_DISABLE_WHEN_LATENCY_WORSE,
    targetModelId: DEFAULT_MODEL,
  }),
};

export interface BrowserMtpConfigInput {
  enabled: boolean;
  draftModelId: string;
  numSpeculativeTokens: number;
  minAcceptanceRate: number;
  disableWhenLatencyWorse: boolean;
  draftLayerCount?: number;
}

export interface RuntimeMtpConfigInput extends BrowserMtpConfigInput {
  targetModelId: string;
}

export function makeBrowserMtpClientOptions(input: BrowserMtpConfigInput) {
  return {
    enabled: input.enabled,
    draftModelId: input.draftModelId,
    numSpeculativeTokens: normalizeBrowserMtpDraftWindow(input.numSpeculativeTokens),
    minAcceptanceRate: input.minAcceptanceRate,
    disableWhenLatencyWorse: input.disableWhenLatencyWorse,
    ...(input.draftLayerCount !== undefined ? { draftLayerCount: input.draftLayerCount } : {}),
  };
}

export function makeRuntimeMtpConfig(input: RuntimeMtpConfigInput): AgentRuntimeMtpConfig {
  const localBrowserDrafterConfigured = isSupportedBrowserMtpDrafter(input.draftModelId);
  const numSpeculativeTokens = normalizeBrowserMtpDraftWindow(input.numSpeculativeTokens);
  return {
    enabled: input.enabled,
    draftModelId: input.draftModelId,
    numSpeculativeTokens,
    minAcceptanceRate: input.minAcceptanceRate,
    disableWhenLatencyWorse: input.disableWhenLatencyWorse,
    targetTokenizerId: "qwen3",
    draftModelProfiles: localBrowserDrafterConfigured ? [
      {
        modelId: input.draftModelId,
        role: "draft",
        tokenizerId: "qwen3",
        maxSpeculativeTokens: numSpeculativeTokens,
        targetModelIds: [input.targetModelId],
      },
    ] : [],
  };
}

export function normalizeBrowserMtpDraftWindow(value: number): number {
  return Math.max(1, Math.min(Math.floor(value), LOCAL_BROWSER_MTP_MAX_SPECULATIVE_TOKENS));
}

function isSupportedBrowserMtpDrafter(draftModelId: string): boolean {
  return draftModelId === BROWSER_NGRAM_MTP_DRAFT_MODEL_ID
    || draftModelId === BROWSER_QWEN_PREFIX_MTP_DRAFT_MODEL_ID;
}

export const QWEN3_0_6B_PROFILE: AdvancedRuntimeModelProfile = {
  modelId: QWEN3_UNLOCKED_MODEL_ID,
  parameterBytes: 1_200_000_000,
  effectiveParameterCount: 600_000_000,
  layers: 28,
  hiddenSize: 1024,
  kvHeads: 8,
  contextWindowTokens: QWEN3_CONTEXT_WINDOW_TOKENS
};

export const DEFAULT_DEVICE_PROFILE: DeviceProfile = {
  name: "browser-webgpu-edge",
  vramBudgetBytes: 4 * 1024 * 1024 * 1024,
  ramBudgetBytes: 8 * 1024 * 1024 * 1024
};

export function makeModelProfile(modelId: string): AdvancedRuntimeModelProfile {
  return { ...QWEN3_0_6B_PROFILE, modelId };
}

export function makeBackendProfile(backend: string): InferenceBackendProfile {
  if (backend === "compiled-browser-webllm") {
    return {
      id: "compiled-browser-webllm",
      label: "Compiled Browser WebLLM Backend",
      mode: "custom",
      capabilities: {
        qkvAccess: false,
        layerSparseRouting: false,
        pinnedKvBlocks: false,
        kvTensorPaging: false,
        tspScheduleExecution: false,
        speculativeVerifierBatching: false,
      }
    };
  }
  if (backend === "unlocked-browser-transformer") {
    return {
      id: "unlocked-browser-transformer",
      label: "Custom WebGPU Kernel Lab",
      mode: "custom",
      capabilities: {
        qkvAccess: true,
        layerSparseRouting: true,
        pinnedKvBlocks: true,
        kvTensorPaging: true,
        tspScheduleExecution: true,
        speculativeVerifierBatching: true,
      }
    };
  }
  return {
    id: backend,
    label: "Unsupported opaque inference backend",
    mode: "custom",
    capabilities: {
      qkvAccess: false,
      layerSparseRouting: false,
      pinnedKvBlocks: false,
      kvTensorPaging: false,
      tspScheduleExecution: false,
      speculativeVerifierBatching: false,
    },
  };
}

export const SYSTEM_PROMPT = `You are Infinite Edge Agent, a local-first AI assistant.

Core rules:
- Prefer accurate, grounded answers over broad claims.
- Use retrieved long-term memory when it is relevant, but do not pretend memory exists if it was not retrieved.
- When implementation details are uncertain, explain the assumption and provide a testable next step.
- Keep private user data local and avoid recommending cloud services unless the user asks for them.
- For code, produce runnable, minimal, maintainable changes.`;

export const MEMORY_SERVER_ENABLED = import.meta.env.VITE_ENABLE_MEMORY_SERVER === "true";
export const MEMORY_SERVER_URL = import.meta.env.VITE_MEMORY_SERVER_URL ?? "http://127.0.0.1:8787";
