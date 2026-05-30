import type { BrowserPreviewBenchmarkPayload } from "./browserPreviewBenchmark";

export interface BenchmarkTelemetryConfig {
  enabled: boolean;
  url: string;
  appVersion?: string;
  gitSha?: string;
  deployUrl?: string;
}

export interface BenchmarkTelemetrySubmitInput {
  requested: boolean;
  config: BenchmarkTelemetryConfig;
  benchmarkPayload: BrowserPreviewBenchmarkPayload;
  fetcher?: typeof fetch;
  browserContext?: Partial<BenchmarkTelemetryBrowserContext>;
}

export interface BenchmarkTelemetrySubmitResult {
  requested: boolean;
  configured: boolean;
  submitted: boolean;
  status?: number;
  error?: string;
}

export interface BenchmarkTelemetryBrowserContext {
  userAgent: string;
  hardwareConcurrency: number | null;
  deviceMemoryGb: number | null;
  mobile: boolean;
  screenWidth: number | null;
  screenHeight: number | null;
  webgpuAvailable: boolean;
  deployUrl: string;
}

export interface BenchmarkTelemetryPayload {
  runId: string;
  createdAt: string;
  appVersion: string | null;
  gitSha: string | null;
  deployUrl: string | null;
  benchmarkProfile: string | null;
  backendId: string | null;
  modelId: string | null;
  device: {
    os: string;
    browserName: string;
    browserVersion: string | null;
    userAgentHash: string;
    mobile: boolean;
    hardwareConcurrency: number | null;
    deviceMemoryGb: number | null;
    screen: {
      width: number | null;
      height: number | null;
    };
    webgpuAvailable: boolean;
  };
  summary: {
    initLoadMs: number | null;
    timeToFirstTokenMs: number | null;
    tokensPerSecond: number | null;
    memoryGroundingPassed: boolean | null;
    expectedExactPassed: boolean | null;
    productionDeployReadyPassed: boolean | null;
    compiledBackendReadyPassed: boolean | null;
  };
  artifactJson: unknown;
}

export async function submitBenchmarkTelemetry(
  input: BenchmarkTelemetrySubmitInput,
): Promise<BenchmarkTelemetrySubmitResult> {
  const configured = input.config.enabled && input.config.url.trim().length > 0;
  if (!input.requested || !configured) {
    return {
      requested: input.requested,
      configured,
      submitted: false,
      ...(input.requested && !configured ? { error: "benchmark telemetry is not enabled or no endpoint is configured" } : {}),
    };
  }

  try {
    const payload = buildBenchmarkTelemetryPayload(input);
    const response = await (input.fetcher ?? fetch)(input.config.url, {
      method: "POST",
      headers: {
        "content-type": "application/json",
      },
      body: JSON.stringify(payload),
    });
    return {
      requested: true,
      configured: true,
      submitted: response.ok,
      status: response.status,
      ...(response.ok ? {} : { error: `telemetry endpoint returned HTTP ${response.status}` }),
    };
  } catch (error) {
    return {
      requested: true,
      configured: true,
      submitted: false,
      error: error instanceof Error ? error.message : String(error),
    };
  }
}

export function buildBenchmarkTelemetryPayload(
  input: Pick<BenchmarkTelemetrySubmitInput, "benchmarkPayload" | "config" | "browserContext">,
): BenchmarkTelemetryPayload {
  const browserContext = {
    ...readBrowserTelemetryContext(),
    ...(input.browserContext ?? {}),
  };
  const summary = input.benchmarkPayload.summary;
  const firstRun = input.benchmarkPayload.runs[0];
  const browser = detectBrowser(browserContext.userAgent);
  return {
    runId: makeTelemetryRunId(input.benchmarkPayload),
    createdAt: input.benchmarkPayload.createdAt,
    appVersion: input.config.appVersion ?? null,
    gitSha: input.config.gitSha ?? null,
    deployUrl: input.config.deployUrl ?? browserContext.deployUrl ?? null,
    benchmarkProfile: readSummaryString(summary.profile),
    backendId: readSummaryString(summary.runtimeBackendId ?? firstRun?.runtimeTrace.backend),
    modelId: readSummaryString(summary.modelId),
    device: {
      os: detectOs(browserContext.userAgent),
      browserName: browser.name,
      browserVersion: browser.version,
      userAgentHash: hashString(browserContext.userAgent),
      mobile: browserContext.mobile,
      hardwareConcurrency: browserContext.hardwareConcurrency,
      deviceMemoryGb: browserContext.deviceMemoryGb,
      screen: {
        width: browserContext.screenWidth,
        height: browserContext.screenHeight,
      },
      webgpuAvailable: browserContext.webgpuAvailable,
    },
    summary: {
      initLoadMs: readSummaryNumber(summary.meanInitLoadMs),
      timeToFirstTokenMs: readSummaryNumber(summary.meanTimeToFirstTokenMs),
      tokensPerSecond: readSummaryNumber(summary.meanTokensPerSecond),
      memoryGroundingPassed: readSummaryBoolean(summary.memoryGroundingPassed),
      expectedExactPassed: readSummaryBoolean(summary.expectedExactPassed),
      productionDeployReadyPassed: readSummaryBoolean(summary.productionDeployReadyPassed),
      compiledBackendReadyPassed: readSummaryBoolean(summary.compiledBackendReadyPassed),
    },
    artifactJson: sanitizeBenchmarkArtifact(input.benchmarkPayload),
  };
}

export function sanitizeBenchmarkArtifact(payload: BrowserPreviewBenchmarkPayload): unknown {
  return {
    ...payload,
    runs: payload.runs.map((run) => ({
      ...run,
      prompt: "[redacted]",
      response: "[redacted]",
      expectedSubstrings: run.expectedSubstrings.length > 0 ? ["[redacted]"] : [],
      expectedSubstringMatches: run.expectedSubstringMatches.length > 0 ? ["[redacted]"] : [],
      ...(run.expectedExact ? { expectedExact: ["[redacted]"] } : {}),
      ...(run.expectedExactMatches ? {
        expectedExactMatches: run.expectedExactMatches.map((match) => ({ matched: match.matched })),
      } : {}),
      tokenDiagnostics: undefined,
    })),
  };
}

function readBrowserTelemetryContext(): BenchmarkTelemetryBrowserContext {
  const navigatorLike = globalThis.navigator as (Navigator & { deviceMemory?: number; gpu?: unknown }) | undefined;
  const screenLike = globalThis.screen;
  return {
    userAgent: navigatorLike?.userAgent ?? "",
    hardwareConcurrency: navigatorLike?.hardwareConcurrency ?? null,
    deviceMemoryGb: typeof navigatorLike?.deviceMemory === "number" ? navigatorLike.deviceMemory : null,
    mobile: /mobile|iphone|ipad|android/i.test(navigatorLike?.userAgent ?? ""),
    screenWidth: screenLike?.width ?? null,
    screenHeight: screenLike?.height ?? null,
    webgpuAvailable: Boolean(navigatorLike?.gpu),
    deployUrl: globalThis.location?.origin ?? "",
  };
}

function makeTelemetryRunId(payload: BrowserPreviewBenchmarkPayload): string {
  const seed = `${payload.createdAt}:${payload.summary.runtimeBackendId ?? "unknown"}:${payload.summary.meanTokensPerSecond ?? "na"}`;
  return `bench_${hashString(seed)}`;
}

function detectBrowser(userAgent: string): { name: string; version: string | null } {
  const patterns: Array<[string, RegExp]> = [
    ["Edge", /Edg\/([0-9.]+)/],
    ["Chrome", /Chrome\/([0-9.]+)/],
    ["Safari", /Version\/([0-9.]+).*Safari/],
    ["Firefox", /Firefox\/([0-9.]+)/],
  ];
  for (const [name, pattern] of patterns) {
    const match = userAgent.match(pattern);
    if (match?.[1]) return { name, version: match[1] };
  }
  return { name: userAgent ? "Unknown" : "Unknown", version: null };
}

function detectOs(userAgent: string): string {
  if (/Windows/i.test(userAgent)) return "Windows";
  if (/iPhone|iPad|iPod/i.test(userAgent)) return "iOS";
  if (/Mac OS X|Macintosh/i.test(userAgent)) return "macOS";
  if (/Android/i.test(userAgent)) return "Android";
  if (/Linux/i.test(userAgent)) return "Linux";
  return "Unknown";
}

function hashString(value: string): string {
  let hash = 0x811c9dc5;
  for (let index = 0; index < value.length; index += 1) {
    hash ^= value.charCodeAt(index);
    hash = Math.imul(hash, 0x01000193);
  }
  return (hash >>> 0).toString(16).padStart(8, "0");
}

function readSummaryString(value: unknown): string | null {
  return typeof value === "string" && value.trim() ? value : null;
}

function readSummaryNumber(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function readSummaryBoolean(value: unknown): boolean | null {
  return typeof value === "boolean" ? value : null;
}
