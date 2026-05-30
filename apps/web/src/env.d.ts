/// <reference types="vite/client" />

interface ImportMetaEnv {
  readonly VITE_DEFAULT_MODEL?: string;
  readonly VITE_LLM_BACKEND?: string;
  readonly VITE_UNLOCKED_MODEL_MANIFEST_PATH?: string;
  readonly VITE_UNLOCKED_MODEL_MANIFEST_SHA256?: string;
  readonly VITE_UNLOCKED_MANIFEST_FORMAT?: "sharded" | "inline";
  readonly VITE_UNLOCKED_WEIGHT_FORMAT?: "f32-reference" | "f16-packed";
  readonly VITE_UNLOCKED_ALLOW_FIXTURE?: string;
  readonly VITE_UNLOCKED_BACKEND_PREFERENCE?: "cpu" | "webgpu";
  readonly VITE_REQUIRE_WEBGPU_KERNELS?: string;
  readonly VITE_COMPILED_WEBLLM_ENABLED?: string;
  readonly VITE_UNLOCKED_RUNTIME_PROFILE?: "preview" | "balanced" | "full" | "ci";
  readonly VITE_UNLOCKED_MAX_RUNTIME_PROMPT_TOKENS?: string;
  readonly VITE_UNLOCKED_MAX_RUNTIME_LAYERS?: string;
  readonly VITE_UNLOCKED_LOGIT_CANDIDATE_LIMIT?: string;
  readonly VITE_UNLOCKED_MAX_GENERATION_TOKENS?: string;
  readonly VITE_REQUIRE_UNLOCKED_RUNTIME?: string;
  readonly VITE_MTP_ENABLED?: string;
  readonly VITE_MTP_DRAFT_MODEL_ID?: string;
  readonly VITE_MTP_NUM_SPECULATIVE_TOKENS?: string;
  readonly VITE_MTP_MIN_ACCEPTANCE_RATE?: string;
  readonly VITE_MTP_DISABLE_WHEN_LATENCY_WORSE?: string;
  readonly VITE_MTP_DRAFT_LAYER_COUNT?: string;
  readonly VITE_CHAT_MAX_RUNTIME_PROMPT_TOKENS?: string;
  readonly VITE_CHAT_MAX_RUNTIME_LAYERS?: string;
  readonly VITE_CHAT_LOGIT_CANDIDATE_LIMIT?: string;
  readonly VITE_CHAT_LOGIT_TOP_K?: string;
  readonly VITE_CHAT_LOGIT_TILE_ROWS?: string;
  readonly VITE_CHAT_MAX_GENERATION_TOKENS?: string;
  readonly VITE_QWEN_THINKING_MODE?: "disabled" | "enabled";
  readonly VITE_AGENT_MAX_PROMPT_TOKENS?: string;
  readonly VITE_KVSWAP_PERSISTENCE_ENABLED?: string;
  readonly VITE_KVSWAP_PERSISTENCE_PREFER_OPFS?: string;
  readonly VITE_KVSWAP_PERSISTENCE_MAX_BLOCKS?: string;
  readonly VITE_KVSWAP_PERSISTENCE_MAX_BYTES?: string;
  readonly VITE_KVSWAP_PERSISTENCE_CLEAR_ON_INIT?: string;
  readonly VITE_BUNDLE_UNLOCKED_MODEL?: string;
  readonly VITE_ENABLE_MEMORY_SERVER?: string;
  readonly VITE_MEMORY_SERVER_URL?: string;
  readonly VITE_MEMORY_PROVIDER?: string;
  readonly VITE_ALLOW_MEMORY_FALLBACK?: string;
  readonly VITE_REMOTE_MEMORY_URL?: string;
  readonly VITE_REMOTE_MEMORY_TOKEN?: string;
  readonly VITE_REMOTE_MEMORY_CREDENTIALS?: "omit" | "same-origin" | "include";
  readonly VITE_REMOTE_MEMORY_TENANT_ID?: string;
  readonly VITE_REMOTE_MEMORY_CELL_ID?: string;
  readonly VITE_PRODUCTION_MODE?: string;
  readonly VITE_EMBEDDING_MODEL?: string;
  readonly VITE_EMBEDDING_PREFER_WEBGPU?: string;
  readonly VITE_PUBLIC_EMBED_ALLOWED_ORIGINS?: string;
  readonly VITE_BENCHMARK_TELEMETRY_ENABLED?: string;
  readonly VITE_BENCHMARK_TELEMETRY_URL?: string;
  readonly VITE_APP_VERSION?: string;
  readonly VITE_GIT_SHA?: string;
  readonly VITE_DEPLOY_URL?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
