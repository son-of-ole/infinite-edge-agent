export type UnlockedRuntimeProfileName = "preview" | "balanced" | "full" | "ci";

export interface UnlockedRuntimeCaps {
  maxRuntimePromptTokens: number | null;
  maxRuntimeLayers: number | null;
  logitCandidateLimit: number | null;
  maxGenerationTokens: number | null;
}

export interface UnlockedRuntimeCapStatus {
  prompt: boolean;
  layers: boolean;
  generation: boolean;
  logits: boolean;
}

export interface UnlockedRuntimeProfileResolution {
  profile: UnlockedRuntimeProfileName;
  caps: UnlockedRuntimeCaps;
  capStatus: UnlockedRuntimeCapStatus;
}

export type UnlockedRuntimeProfileEnv = Partial<Record<
  | "VITE_UNLOCKED_RUNTIME_PROFILE"
  | "VITE_UNLOCKED_MAX_RUNTIME_PROMPT_TOKENS"
  | "VITE_UNLOCKED_MAX_RUNTIME_LAYERS"
  | "VITE_UNLOCKED_LOGIT_CANDIDATE_LIMIT"
  | "VITE_UNLOCKED_MAX_GENERATION_TOKENS",
  string | undefined
>>;

const PROFILE_CAPS: Record<UnlockedRuntimeProfileName, UnlockedRuntimeCaps> = {
  preview: {
    maxRuntimePromptTokens: 4,
    maxRuntimeLayers: 1,
    logitCandidateLimit: 256,
    maxGenerationTokens: 1,
  },
  balanced: {
    maxRuntimePromptTokens: 1024,
    maxRuntimeLayers: 8,
    logitCandidateLimit: 1024,
    maxGenerationTokens: 32,
  },
  full: {
    maxRuntimePromptTokens: null,
    maxRuntimeLayers: null,
    logitCandidateLimit: null,
    maxGenerationTokens: null,
  },
  ci: {
    maxRuntimePromptTokens: 4,
    maxRuntimeLayers: 1,
    logitCandidateLimit: 64,
    maxGenerationTokens: 1,
  },
};

export function resolveUnlockedRuntimeProfile(env: UnlockedRuntimeProfileEnv): UnlockedRuntimeProfileResolution {
  const profile = normalizeRuntimeProfileName(env.VITE_UNLOCKED_RUNTIME_PROFILE);
  const caps = {
    ...PROFILE_CAPS[profile],
    ...readCapOverride("maxRuntimePromptTokens", "VITE_UNLOCKED_MAX_RUNTIME_PROMPT_TOKENS", env.VITE_UNLOCKED_MAX_RUNTIME_PROMPT_TOKENS),
    ...readCapOverride("maxRuntimeLayers", "VITE_UNLOCKED_MAX_RUNTIME_LAYERS", env.VITE_UNLOCKED_MAX_RUNTIME_LAYERS),
    ...readCapOverride("logitCandidateLimit", "VITE_UNLOCKED_LOGIT_CANDIDATE_LIMIT", env.VITE_UNLOCKED_LOGIT_CANDIDATE_LIMIT),
    ...readCapOverride("maxGenerationTokens", "VITE_UNLOCKED_MAX_GENERATION_TOKENS", env.VITE_UNLOCKED_MAX_GENERATION_TOKENS),
  };
  return {
    profile,
    caps,
    capStatus: {
      prompt: caps.maxRuntimePromptTokens !== null,
      layers: caps.maxRuntimeLayers !== null,
      generation: caps.maxGenerationTokens !== null,
      logits: caps.logitCandidateLimit !== null,
    },
  };
}

export function assertUnlockedFullProfile(resolution: UnlockedRuntimeProfileResolution): void {
  if (resolution.profile === "full" && !hasAnyActiveCap(resolution.capStatus)) return;
  throw new Error(
    `Unlocked release requires VITE_UNLOCKED_RUNTIME_PROFILE=full without artificial caps; `
    + `resolved profile=${resolution.profile}, caps=${JSON.stringify(resolution.caps)}.`,
  );
}

function normalizeRuntimeProfileName(value: string | undefined): UnlockedRuntimeProfileName {
  const normalized = value?.trim().toLowerCase();
  if (!normalized) return "full";
  if (normalized === "preview" || normalized === "balanced" || normalized === "full" || normalized === "ci") {
    return normalized;
  }
  throw new Error(`Unsupported VITE_UNLOCKED_RUNTIME_PROFILE="${value}". Expected preview, balanced, full, or ci.`);
}

function readCapOverride<K extends keyof UnlockedRuntimeCaps>(
  key: K,
  envName: string,
  value: string | undefined,
): Pick<UnlockedRuntimeCaps, K> | Record<string, never> {
  const parsed = readPositiveInteger(value, envName);
  return parsed === null ? {} : { [key]: parsed } as Pick<UnlockedRuntimeCaps, K>;
}

function readPositiveInteger(value: string | undefined, envName: string): number | null {
  if (!value?.trim()) return null;
  const parsed = Number(value);
  if (!Number.isFinite(parsed)) throw new Error(`${envName} must be a positive integer, received "${value}".`);
  if (!Number.isInteger(parsed) || parsed <= 0) throw new Error(`${envName} must be a positive integer, received "${value}".`);
  return parsed;
}

function hasAnyActiveCap(status: UnlockedRuntimeCapStatus): boolean {
  return status.prompt || status.layers || status.generation || status.logits;
}
