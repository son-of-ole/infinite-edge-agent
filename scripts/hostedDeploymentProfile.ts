import { mkdir, writeFile } from "node:fs/promises";
import { join } from "node:path";

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
    deployUrl: string | null;
    benchmarkDeployUrlBound: boolean | null;
    benchmarkUrl: string | null;
    benchmarkBackend: string | null;
    benchmarkModelId: string | null;
    benchmarkMemoryGrounding: string | null;
    benchmarkMemoryGroundingProfile: string | null;
    benchmarkExpectedExact: string | null;
    benchmarkRequiresSubmitTelemetry: boolean;
    benchmarkWarmResidentSpeedProof: boolean;
  };
}

export interface HostedDeploymentProfileArtifact {
  name: "hosted-deployment-profile";
  createdAt: string;
  passed: boolean;
  summary: Record<string, number | string | boolean | null>;
  report: HostedDeploymentProfileReport;
}

export interface HostedDeploymentProfileArtifactWriteResult {
  artifact: HostedDeploymentProfileArtifact;
  latestPath: string;
  resultPath: string;
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
  const rawDeployUrl = readString(env.VITE_DEPLOY_URL);
  const deployUrl = rawDeployUrl ? normalizePublicHttpsOrigin(rawDeployUrl) : null;
  const expectedBenchmarkUrl = resolveBenchmarkUrl(env);
  const parsedBenchmark = parseBenchmarkUrl(expectedBenchmarkUrl);
  const benchmarkDeployUrlBound = deployUrl && parsedBenchmark
    ? parsedBenchmark.origin === deployUrl
    : null;

  if (llmBackend !== HOSTED_BACKEND_ID) {
    blockers.push("Hosted production requires VITE_LLM_BACKEND=compiled-browser-webllm.");
  }
  if (defaultModel !== HOSTED_DEFAULT_MODEL) {
    blockers.push(`Hosted production requires VITE_DEFAULT_MODEL=${HOSTED_DEFAULT_MODEL}.`);
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
  if (!rawDeployUrl) {
    blockers.push("Hosted production requires VITE_DEPLOY_URL to identify the canonical public deploy origin.");
  } else if (!deployUrl) {
    blockers.push("Hosted production deploy URL must use a public HTTPS origin.");
  }

  if (!expectedBenchmarkUrl) {
    blockers.push("Hosted production requires HOSTED_PRODUCTION_BENCHMARK_URL or BROWSER_RUNTIME_BENCH_PREVIEW_URL.");
  } else if (!parsedBenchmark) {
    blockers.push("Hosted production benchmark URL must be an absolute URL.");
  } else {
    if (!isPublicHttpsUrl(parsedBenchmark)) {
      blockers.push("Hosted production benchmark URL must use a public HTTPS origin.");
    }
    if (deployUrl && benchmarkDeployUrlBound !== true) {
      blockers.push(`Hosted production benchmark URL origin ${parsedBenchmark.origin} must match deploy origin ${deployUrl}.`);
    }
    if (parsedBenchmark.pathname !== "/__bench/browser-runtime") {
      blockers.push("Hosted production benchmark URL must target /__bench/browser-runtime.");
    }
    if (parsedBenchmark.searchParams.get("backend") !== HOSTED_BACKEND_ID) {
      blockers.push("Hosted production benchmark URL must set backend=compiled-browser-webllm.");
    }
    if (parsedBenchmark.searchParams.get("modelId") !== HOSTED_DEFAULT_MODEL) {
      blockers.push(`Hosted production benchmark URL must set modelId=${HOSTED_DEFAULT_MODEL}.`);
    }
    if (!benchmarkProvesGrounding(parsedBenchmark.searchParams)) {
      blockers.push("Hosted production benchmark URL must require memoryGrounding=montana_capital or memoryGroundingProfile=qa_corpus_v1.");
    }
    if (parsedBenchmark.searchParams.get("expectedExact") !== HOSTED_EXACT_ANSWER) {
      blockers.push("Hosted production benchmark URL must set expectedExact=Helena.");
    }
    if (!benchmarkRequestsWarmResidentSpeedProof(parsedBenchmark.searchParams)) {
      blockers.push("Hosted production benchmark URL must request speedProof=warm_resident or warmResidentSpeedProof=true.");
    }
    if (!benchmarkSubmitsTelemetry(parsedBenchmark.searchParams)) {
      blockers.push("Hosted production benchmark URL must opt in with submitTelemetry=true or benchmarkTelemetry=true.");
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
      deployUrl,
      benchmarkDeployUrlBound,
      benchmarkUrl: expectedBenchmarkUrl,
      benchmarkBackend: parsedBenchmark?.searchParams.get("backend") ?? null,
      benchmarkModelId: parsedBenchmark?.searchParams.get("modelId") ?? null,
      benchmarkMemoryGrounding: parsedBenchmark?.searchParams.get("memoryGrounding") ?? null,
      benchmarkMemoryGroundingProfile: parsedBenchmark?.searchParams.get("memoryGroundingProfile") ?? null,
      benchmarkExpectedExact: parsedBenchmark?.searchParams.get("expectedExact") ?? null,
      benchmarkRequiresSubmitTelemetry: parsedBenchmark ? benchmarkSubmitsTelemetry(parsedBenchmark.searchParams) : false,
      benchmarkWarmResidentSpeedProof: parsedBenchmark ? benchmarkRequestsWarmResidentSpeedProof(parsedBenchmark.searchParams) : false,
    },
  };
}

export function buildHostedDeploymentProfileArtifact(
  report: HostedDeploymentProfileReport,
  createdAt = new Date().toISOString(),
): HostedDeploymentProfileArtifact {
  return {
    name: "hosted-deployment-profile",
    createdAt,
    passed: report.passed,
    summary: {
      hostedProfilePassed: report.passed,
      hostedProfileBlockerCount: report.blockers.length,
      hostedProfileWarningCount: report.warnings.length,
      hostedProfileBackend: report.profile.llmBackend,
      hostedProfileDefaultModel: report.profile.defaultModel,
      hostedProfileCompiledWebLlmEnabled: report.profile.compiledWebLlmEnabled,
      hostedProfileRequireUnlockedRuntime: report.profile.requireUnlockedRuntime,
      hostedProfileMtpProductionDisabled: report.profile.mtpProductionDisabled,
      hostedProfileTelemetryEnabled: report.profile.telemetryEnabled,
      hostedProfileTelemetryStorage: report.profile.telemetryStorage,
      hostedProfileTelemetryAdminProtected: report.profile.telemetryAdminProtected,
      hostedProfileTelemetryRateLimited: report.profile.telemetryRateLimited,
      hostedProfileDeployUrl: report.profile.deployUrl,
      hostedProfileBenchmarkDeployUrlBound: report.profile.benchmarkDeployUrlBound,
      hostedProfileBenchmarkBackend: report.profile.benchmarkBackend,
      hostedProfileBenchmarkModelId: report.profile.benchmarkModelId,
      hostedProfileBenchmarkMemoryGrounding: report.profile.benchmarkMemoryGrounding,
      hostedProfileBenchmarkMemoryGroundingProfile: report.profile.benchmarkMemoryGroundingProfile,
      hostedProfileBenchmarkExpectedExact: report.profile.benchmarkExpectedExact,
      hostedProfileBenchmarkRequiresSubmitTelemetry: report.profile.benchmarkRequiresSubmitTelemetry,
      hostedProfileBenchmarkWarmResidentSpeedProof: report.profile.benchmarkWarmResidentSpeedProof,
    },
    report,
  };
}

export async function writeHostedDeploymentProfileArtifact(
  report: HostedDeploymentProfileReport,
  options: { artifactDir?: string; createdAt?: string } = {},
): Promise<HostedDeploymentProfileArtifactWriteResult> {
  const artifactDir = options.artifactDir ?? process.env.EVAL_ARTIFACT_DIR ?? ".artifacts/evals";
  const artifact = buildHostedDeploymentProfileArtifact(report, options.createdAt);
  const runDir = join(artifactDir, "hosted-deployment-profile");
  const timestamp = artifact.createdAt.replace(/[:.]/g, "-");
  const latestPath = join(artifactDir, "hosted-deployment-profile-latest.json");
  const resultPath = join(runDir, `${timestamp}.json`);
  const json = `${JSON.stringify(artifact, null, 2)}\n`;

  await mkdir(runDir, { recursive: true });
  await writeFile(resultPath, json);
  await writeFile(latestPath, json);

  return {
    artifact,
    latestPath,
    resultPath,
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
    url.searchParams.set("speedProof", "warm_resident");
    return url.toString();
  } catch {
    return null;
  }
}

function parseBenchmarkUrl(value: string | null): URL | null {
  if (!value) return null;
  try {
    return new URL(value);
  } catch {
    return null;
  }
}

function normalizePublicHttpsOrigin(value: string): string | null {
  try {
    const url = new URL(value);
    return isPublicHttpsUrl(url) ? url.origin : null;
  } catch {
    return null;
  }
}

function isPublicHttpsUrl(url: URL): boolean {
  if (url.protocol !== "https:") return false;
  const hostname = normalizeUrlHostname(url.hostname);
  return !isLocalhost(hostname) && !isPrivateIpv4Host(hostname) && !isPrivateIpv6Host(hostname);
}

function normalizeUrlHostname(hostname: string): string {
  let normalized = hostname.trim().toLowerCase();
  if (normalized.startsWith("[") && normalized.endsWith("]")) {
    normalized = normalized.slice(1, -1);
  }
  while (normalized.endsWith(".")) {
    normalized = normalized.slice(0, -1);
  }
  return normalized;
}

function isLocalhost(hostname: string): boolean {
  return hostname === "localhost" || hostname.endsWith(".localhost") || hostname.endsWith(".local");
}

function isPrivateIpv4Host(hostname: string): boolean {
  const parts = hostname.split(".");
  if (parts.length !== 4) return false;
  const octets = parts.map((part) => {
    if (!/^\d{1,3}$/.test(part)) return Number.NaN;
    return Number(part);
  });
  if (octets.some((octet) => !Number.isInteger(octet) || octet < 0 || octet > 255)) {
    return false;
  }
  const [a = 0, b = 0] = octets;
  return a === 0
    || a === 10
    || a === 127
    || (a === 169 && b === 254)
    || (a === 172 && b >= 16 && b <= 31)
    || (a === 192 && b === 168);
}

function isPrivateIpv6Host(hostname: string): boolean {
  const host = hostname.split("%", 1)[0] ?? "";
  if (!host.includes(":")) return false;
  if (host === "::" || host === "::1" || host === "0:0:0:0:0:0:0:1") return true;
  if (host.startsWith("fc") || host.startsWith("fd") || host.startsWith("fe80:")) return true;
  const ipv4Mapped = host.match(/^::ffff:(\d{1,3}(?:\.\d{1,3}){3})$/);
  return Boolean(ipv4Mapped?.[1] && isPrivateIpv4Host(ipv4Mapped[1]));
}

function benchmarkProvesGrounding(params: URLSearchParams): boolean {
  return params.get("memoryGrounding") === "montana_capital"
    || params.get("memoryGroundingProfile") === "qa_corpus_v1";
}

function benchmarkSubmitsTelemetry(params: URLSearchParams): boolean {
  return params.get("submitTelemetry") === "true" || params.get("benchmarkTelemetry") === "true";
}

function benchmarkRequestsWarmResidentSpeedProof(params: URLSearchParams): boolean {
  return params.get("speedProof") === "warm_resident"
    || params.get("warmResidentSpeedProof") === "true"
    || params.get("requireWarmResidentSpeedProof") === "true";
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
  await writeHostedDeploymentProfileArtifact(report);
  console.log(JSON.stringify(report, null, 2));
  if (!report.passed) process.exitCode = 1;
}
