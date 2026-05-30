export type HostedDeploymentProfileEnv = Record<string, string | undefined>;

export interface HostedDeploymentProfileReport {
  passed: boolean;
  blockers: string[];
  warnings: string[];
  expectedBenchmarkUrl: string | null;
  profile: {
    llmBackend: string | null;
    defaultModel: string | null;
    compiledWebLlmEnabled: boolean;
    requireUnlockedRuntime: boolean;
    mtpProductionDisabled: boolean;
    telemetryEnabled: boolean;
    telemetryStorage: string | null;
    telemetryAdminProtected: boolean;
    telemetryRateLimited: boolean;
    benchmarkUrl: string | null;
    benchmarkBackend: string | null;
    benchmarkMemoryGrounding: string | null;
    benchmarkMemoryGroundingProfile: string | null;
    benchmarkExpectedExact: string | null;
    benchmarkRequiresSubmitTelemetry: boolean;
  };
}

const HOSTED_BACKEND_ID = "compiled-browser-webllm";
const HOSTED_DEFAULT_MODEL = "Qwen3-0.6B-q4f16_1-MLC";
const HOSTED_EXACT_ANSWER = "Helena";

export function evaluateHostedDeploymentProfile(env: HostedDeploymentProfileEnv): HostedDeploymentProfileReport {
  const blockers: string[] = [];
  const warnings: string[] = [];
  const llmBackend = readString(env.VITE_LLM_BACKEND);
  const defaultModel = readString(env.VITE_DEFAULT_MODEL);
  const compiledWebLlmEnabled = readBoolean(env.VITE_COMPILED_WEBLLM_ENABLED);
  const requireUnlockedRuntime = readBoolean(env.VITE_REQUIRE_UNLOCKED_RUNTIME);
  const mtpProductionDisabled = readString(env.VITE_MTP_ENABLED) !== "true";
  const browserTelemetryEnabled = readBoolean(env.VITE_BENCHMARK_TELEMETRY_ENABLED);
  const collectorTelemetryEnabled = readBoolean(env.BENCHMARK_TELEMETRY_ENABLED);
  const telemetryEnabled = browserTelemetryEnabled && collectorTelemetryEnabled;
  const telemetryStorage = readString(env.BENCHMARK_TELEMETRY_STORAGE)?.toLowerCase() ?? null;
  const telemetryAdminProtected = Boolean(readString(env.BENCHMARK_TELEMETRY_ADMIN_TOKEN));
  const telemetryRateLimited = readPositiveNumber(env.BENCHMARK_TELEMETRY_RATE_LIMIT_MAX)
    && readPositiveNumber(env.BENCHMARK_TELEMETRY_RATE_LIMIT_WINDOW_MS);
  const expectedBenchmarkUrl = resolveBenchmarkUrl(env);
  const parsedBenchmark = parseBenchmarkUrl(expectedBenchmarkUrl);

  if (llmBackend !== HOSTED_BACKEND_ID) {
    blockers.push("Hosted production requires VITE_LLM_BACKEND=compiled-browser-webllm.");
  }
  if (compiledWebLlmEnabled !== true) {
    blockers.push("Hosted production requires VITE_COMPILED_WEBLLM_ENABLED=true.");
  }
  if (requireUnlockedRuntime) {
    blockers.push("Hosted compiled production must set VITE_REQUIRE_UNLOCKED_RUNTIME=false so the Kernel Lab is not required for deploy readiness.");
  }
  if (!mtpProductionDisabled) {
    blockers.push("Hosted production must keep VITE_MTP_ENABLED=false; MTP remains lab-only until paired speed proof passes.");
  }
  if (!telemetryEnabled) {
    blockers.push("Hosted benchmark telemetry must be enabled on both browser and collector.");
  }
  if (telemetryStorage !== "postgres") {
    blockers.push("Hosted telemetry must use BENCHMARK_TELEMETRY_STORAGE=postgres.");
  }
  if (!readString(env.BENCHMARK_TELEMETRY_DATABASE_URL) && !readString(env.DATABASE_URL)) {
    blockers.push("Hosted telemetry requires BENCHMARK_TELEMETRY_DATABASE_URL or DATABASE_URL.");
  }
  if (!telemetryAdminProtected) {
    blockers.push("Hosted telemetry requires BENCHMARK_TELEMETRY_ADMIN_TOKEN for list, summary, dashboard, and CSV export routes.");
  }
  if (!telemetryRateLimited) {
    blockers.push("Hosted telemetry requires positive BENCHMARK_TELEMETRY_RATE_LIMIT_MAX and BENCHMARK_TELEMETRY_RATE_LIMIT_WINDOW_MS.");
  }
  if (!readString(env.VITE_BENCHMARK_TELEMETRY_URL)) {
    blockers.push("Hosted browser config requires VITE_BENCHMARK_TELEMETRY_URL.");
  }

  if (!expectedBenchmarkUrl) {
    blockers.push("Hosted production requires HOSTED_PRODUCTION_BENCHMARK_URL or BROWSER_RUNTIME_BENCH_PREVIEW_URL.");
  } else if (!parsedBenchmark) {
    blockers.push("Hosted production benchmark URL must be an absolute or root-relative URL.");
  } else {
    if (parsedBenchmark.pathname !== "/__bench/browser-runtime") {
      blockers.push("Hosted production benchmark URL must target /__bench/browser-runtime.");
    }
    if (parsedBenchmark.searchParams.get("backend") !== HOSTED_BACKEND_ID) {
      blockers.push("Hosted production benchmark URL must set backend=compiled-browser-webllm.");
    }
    if (!benchmarkProvesGrounding(parsedBenchmark.searchParams)) {
      blockers.push("Hosted production benchmark URL must require memoryGrounding=montana_capital or memoryGroundingProfile=qa_corpus_v1.");
    }
    if (parsedBenchmark.searchParams.get("expectedExact") !== HOSTED_EXACT_ANSWER) {
      blockers.push("Hosted production benchmark URL must set expectedExact=Helena.");
    }
    if (!benchmarkSubmitsTelemetry(parsedBenchmark.searchParams)) {
      blockers.push("Hosted production benchmark URL must opt in with submitTelemetry=true or benchmarkTelemetry=true.");
    }
    const benchmarkModelId = parsedBenchmark.searchParams.get("modelId");
    if (benchmarkModelId && benchmarkModelId !== HOSTED_DEFAULT_MODEL) {
      warnings.push(`Hosted benchmark modelId is ${benchmarkModelId}; expected ${HOSTED_DEFAULT_MODEL} for the public compiled proof profile.`);
    }
  }

  return {
    passed: blockers.length === 0,
    blockers,
    warnings,
    expectedBenchmarkUrl,
    profile: {
      llmBackend,
      defaultModel,
      compiledWebLlmEnabled,
      requireUnlockedRuntime,
      mtpProductionDisabled,
      telemetryEnabled,
      telemetryStorage,
      telemetryAdminProtected,
      telemetryRateLimited,
      benchmarkUrl: expectedBenchmarkUrl,
      benchmarkBackend: parsedBenchmark?.searchParams.get("backend") ?? null,
      benchmarkMemoryGrounding: parsedBenchmark?.searchParams.get("memoryGrounding") ?? null,
      benchmarkMemoryGroundingProfile: parsedBenchmark?.searchParams.get("memoryGroundingProfile") ?? null,
      benchmarkExpectedExact: parsedBenchmark?.searchParams.get("expectedExact") ?? null,
      benchmarkRequiresSubmitTelemetry: parsedBenchmark ? benchmarkSubmitsTelemetry(parsedBenchmark.searchParams) : false,
    },
  };
}

function resolveBenchmarkUrl(env: HostedDeploymentProfileEnv): string | null {
  const configured = readString(env.HOSTED_PRODUCTION_BENCHMARK_URL)
    ?? readString(env.BROWSER_RUNTIME_BENCH_PREVIEW_URL);
  if (configured) return configured;
  const deployUrl = readString(env.VITE_DEPLOY_URL);
  if (!deployUrl) return null;
  const base = deployUrl.endsWith("/") ? deployUrl : `${deployUrl}/`;
  try {
    const url = new URL("__bench/browser-runtime", base);
    url.searchParams.set("backend", HOSTED_BACKEND_ID);
    url.searchParams.set("modelId", HOSTED_DEFAULT_MODEL);
    url.searchParams.set("memoryGrounding", "montana_capital");
    url.searchParams.set("expectedExact", HOSTED_EXACT_ANSWER);
    url.searchParams.set("submitTelemetry", "true");
    url.searchParams.set("qwenThinkingMode", "disabled");
    return url.toString();
  } catch {
    return null;
  }
}

function parseBenchmarkUrl(value: string | null): URL | null {
  if (!value) return null;
  try {
    return new URL(value, value.startsWith("/") ? "https://hosted-profile.local" : undefined);
  } catch {
    return null;
  }
}

function benchmarkProvesGrounding(params: URLSearchParams): boolean {
  return params.get("memoryGrounding") === "montana_capital"
    || params.get("memoryGroundingProfile") === "qa_corpus_v1";
}

function benchmarkSubmitsTelemetry(params: URLSearchParams): boolean {
  return params.get("submitTelemetry") === "true" || params.get("benchmarkTelemetry") === "true";
}

function readString(value: string | undefined): string | null {
  const trimmed = value?.trim();
  return trimmed ? trimmed : null;
}

function readBoolean(value: string | undefined): boolean {
  return value?.trim().toLowerCase() === "true";
}

function readPositiveNumber(value: string | undefined): boolean {
  const parsed = Number(value);
  return Number.isFinite(parsed) && parsed > 0;
}

if (import.meta.url === `file://${process.argv[1]}`) {
  const report = evaluateHostedDeploymentProfile(process.env);
  console.log(JSON.stringify(report, null, 2));
  if (!report.passed) process.exitCode = 1;
}
