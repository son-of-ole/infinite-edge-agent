import { mkdir, writeFile } from "node:fs/promises";
import { join, resolve } from "node:path";
import { pathToFileURL } from "node:url";

type HeadlessWebGpuCiParityLane = "browser-runner";
type HeadlessWebGpuCiParityStatus = "passed" | "failed" | "skipped";

export interface HeadlessWebGpuCiParityAdapterInfo {
  vendor?: string;
  architecture?: string;
  device?: string;
  description?: string;
  isFallbackAdapter?: boolean;
}

export interface HeadlessWebGpuCiParityProbeResult {
  webGpuAvailable: boolean;
  adapterFeatures: string[];
  adapterInfo?: HeadlessWebGpuCiParityAdapterInfo;
  limits?: Record<string, number>;
  expectedOutput: number[];
  actualOutput: number[];
  durationMs?: number;
  error?: string;
}

export interface HeadlessWebGpuCiParityArtifact {
  name: "headless-webgpu-ci-parity";
  createdAt: string;
  passed: boolean;
  status: HeadlessWebGpuCiParityStatus;
  lane: HeadlessWebGpuCiParityLane;
  required: boolean;
  webGpuAvailable: boolean;
  browser: {
    engine: "chromium";
    channel?: string;
    executablePath?: string;
    headless: true;
    flags: string[];
  };
  probe: HeadlessWebGpuCiParityProbeResult & {
    outputMatches: boolean;
  };
  failureReason?: string;
  summary: Record<string, number | string | boolean | null>;
}

interface HeadlessWebGpuCiParityArgs {
  lane: HeadlessWebGpuCiParityLane;
  required: boolean;
  artifactDir: string;
  browserChannel?: string;
  browserExecutable?: string;
  timeoutMs: number;
}

interface HeadlessWebGpuCiParityBuildInput {
  createdAt?: string;
  lane?: HeadlessWebGpuCiParityLane;
  required: boolean;
  browser?: {
    engine?: "chromium";
    channel?: string;
    executablePath?: string;
  };
  probe: HeadlessWebGpuCiParityProbeResult;
}

const repoRoot = resolve(new URL("..", import.meta.url).pathname);
const EXPECTED_COMPUTE_OUTPUT = [2, 4, 6, 8];
const DEFAULT_BROWSER_FLAGS = [
  "--enable-unsafe-webgpu",
  "--enable-features=WebGPUDeveloperFeatures,Vulkan",
  "--enable-vulkan",
  "--ignore-gpu-blocklist",
  "--disable-gpu-sandbox",
];
const BROWSER_WEBGPU_COMPUTE_PROBE = `(async () => {
  const startedAt = performance.now();
  const expectedOutput = [2, 4, 6, 8];
  const unavailable = (error) => ({
    webGpuAvailable: false,
    adapterFeatures: [],
    expectedOutput,
    actualOutput: [],
    error,
  });
  const gpu = navigator.gpu;
  if (!gpu || typeof gpu.requestAdapter !== "function") {
    return unavailable("navigator.gpu is unavailable in this headless browser.");
  }
  const adapter = await gpu.requestAdapter();
  if (!adapter) {
    return unavailable("navigator.gpu.requestAdapter() returned null.");
  }
  const device = await adapter.requestDevice();
  const rawAdapterInfo = adapter.info
    || (typeof adapter.requestAdapterInfo === "function" ? await adapter.requestAdapterInfo().catch(() => null) : null)
    || null;
  const readInfo = (key) => {
    const value = rawAdapterInfo && rawAdapterInfo[key];
    return typeof value === "string" && value.length > 0 ? value : undefined;
  };
  const adapterInfo = {
    vendor: readInfo("vendor"),
    architecture: readInfo("architecture"),
    device: readInfo("device"),
    description: readInfo("description"),
    isFallbackAdapter: Boolean(adapter.isFallbackAdapter),
  };
  const input = new Float32Array([1, 2, 3, 4]);
  const storage = device.createBuffer({
    size: input.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
  });
  const readback = device.createBuffer({
    size: input.byteLength,
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(storage, 0, input);
  const shader = device.createShaderModule({
    code: \`
      @group(0) @binding(0) var<storage, read_write> values: array<f32>;
      @compute @workgroup_size(4)
      fn main(@builtin(global_invocation_id) id: vec3<u32>) {
        let index = id.x;
        if (index < 4u) {
          values[index] = values[index] * 2.0;
        }
      }
    \`,
  });
  const pipeline = device.createComputePipeline({
    layout: "auto",
    compute: { module: shader, entryPoint: "main" },
  });
  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [{ binding: 0, resource: { buffer: storage } }],
  });
  const encoder = device.createCommandEncoder();
  const pass = encoder.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);
  pass.dispatchWorkgroups(1);
  pass.end();
  encoder.copyBufferToBuffer(storage, 0, readback, 0, input.byteLength);
  device.queue.submit([encoder.finish()]);
  await device.queue.onSubmittedWorkDone();
  await readback.mapAsync(GPUMapMode.READ);
  const actualOutput = Array.from(new Float32Array(readback.getMappedRange()).slice(0, 4))
    .map((value) => Math.round(value * 1000) / 1000);
  readback.unmap();
  return {
    webGpuAvailable: true,
    adapterFeatures: Array.from(adapter.features.values()).sort(),
    adapterInfo,
    limits: {
      maxComputeInvocationsPerWorkgroup: adapter.limits.maxComputeInvocationsPerWorkgroup,
      maxComputeWorkgroupSizeX: adapter.limits.maxComputeWorkgroupSizeX,
      maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
    },
    expectedOutput,
    actualOutput,
    durationMs: Math.round((performance.now() - startedAt) * 1000) / 1000,
  };
})()`;

if (isMainModule()) {
  const args = parseHeadlessWebGpuCiParityArgs(process.argv.slice(2));
  const artifact = await runHeadlessWebGpuCiParity(args);
  await writeHeadlessWebGpuCiParityArtifact(artifact, { artifactDir: args.artifactDir });
  printHeadlessWebGpuCiParityArtifact(artifact);
  if (artifact.required && artifact.status !== "passed") process.exitCode = 1;
}

export function parseHeadlessWebGpuCiParityArgs(
  argv: string[],
  env: Partial<Record<string, string | undefined>> = process.env,
): HeadlessWebGpuCiParityArgs {
  const parsed = new Map<string, string>();
  const flags = new Set<string>();
  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index];
    if (!arg.startsWith("--")) continue;
    const value = argv[index + 1];
    if (!value || value.startsWith("--")) {
      flags.add(arg.slice(2));
      continue;
    }
    parsed.set(arg.slice(2), value);
    index += 1;
  }
  return {
    lane: parseRealBrowserLane(parsed.get("lane") ?? env.HEADLESS_WEBGPU_CI_LANE),
    required: flags.has("required") || env.HEADLESS_WEBGPU_CI_REQUIRED === "true",
    artifactDir: parsed.get("artifact-dir") ?? env.EVAL_ARTIFACT_DIR ?? ".artifacts/evals",
    ...(readBrowserChannel(parsed, env) ? { browserChannel: readBrowserChannel(parsed, env) } : {}),
    ...(readBrowserExecutable(parsed, env) ? { browserExecutable: readBrowserExecutable(parsed, env) } : {}),
    timeoutMs: parsePositiveInteger(parsed.get("timeout-ms") ?? env.HEADLESS_WEBGPU_CI_TIMEOUT_MS, 45_000),
  };
}

export async function runHeadlessWebGpuCiParity(
  args: HeadlessWebGpuCiParityArgs,
): Promise<HeadlessWebGpuCiParityArtifact> {
  let playwright: typeof import("playwright-core");
  try {
    playwright = await import("playwright-core");
  } catch (error) {
    return buildHeadlessWebGpuCiParityArtifact({
      lane: args.lane,
      required: args.required,
      browser: {
        engine: "chromium",
        ...(args.browserChannel ? { channel: args.browserChannel } : {}),
        ...(args.browserExecutable ? { executablePath: args.browserExecutable } : {}),
      },
      probe: unavailableProbe(`playwright-core is unavailable: ${error instanceof Error ? error.message : String(error)}`),
    });
  }

  let browser: Awaited<ReturnType<typeof playwright.chromium.launch>> | null = null;
  try {
    browser = await playwright.chromium.launch({
      headless: true,
      ...(args.browserChannel ? { channel: args.browserChannel } : {}),
      ...(args.browserExecutable ? { executablePath: args.browserExecutable } : {}),
      args: DEFAULT_BROWSER_FLAGS,
      timeout: args.timeoutMs,
    });
    const page = await browser.newPage();
    const probe = await page.evaluate(BROWSER_WEBGPU_COMPUTE_PROBE) as HeadlessWebGpuCiParityProbeResult;
    return buildHeadlessWebGpuCiParityArtifact({
      lane: args.lane,
      required: args.required,
      browser: {
        engine: "chromium",
        ...(args.browserChannel ? { channel: args.browserChannel } : {}),
        ...(args.browserExecutable ? { executablePath: args.browserExecutable } : {}),
      },
      probe,
    });
  } catch (error) {
    return buildHeadlessWebGpuCiParityArtifact({
      lane: args.lane,
      required: args.required,
      browser: {
        engine: "chromium",
        ...(args.browserChannel ? { channel: args.browserChannel } : {}),
        ...(args.browserExecutable ? { executablePath: args.browserExecutable } : {}),
      },
      probe: unavailableProbe(error instanceof Error ? error.message : String(error)),
    });
  } finally {
    await browser?.close().catch(() => undefined);
  }
}

export function buildHeadlessWebGpuCiParityArtifact(
  input: HeadlessWebGpuCiParityBuildInput,
): HeadlessWebGpuCiParityArtifact {
  const outputMatches = computeOutputMatches(input.probe.actualOutput, input.probe.expectedOutput);
  const adapterAssessment = assessRealHardwareAdapter(input.probe);
  const passed = input.probe.webGpuAvailable && outputMatches && adapterAssessment.realHardware;
  const status: HeadlessWebGpuCiParityStatus = passed
    ? "passed"
    : input.required || input.probe.webGpuAvailable
      ? "failed"
      : "skipped";
  const failureReason = passed ? undefined : buildFailureReason(input.probe, outputMatches, adapterAssessment.reason);
  const lane = input.lane ?? "browser-runner";
  const browser = {
    engine: input.browser?.engine ?? "chromium",
    ...(input.browser?.channel ? { channel: input.browser.channel } : {}),
    ...(input.browser?.executablePath ? { executablePath: input.browser.executablePath } : {}),
    headless: true,
    flags: DEFAULT_BROWSER_FLAGS,
  } satisfies HeadlessWebGpuCiParityArtifact["browser"];

  return {
    name: "headless-webgpu-ci-parity",
    createdAt: input.createdAt ?? new Date().toISOString(),
    passed,
    status,
    lane,
    required: input.required,
    webGpuAvailable: input.probe.webGpuAvailable,
    browser,
    probe: {
      ...input.probe,
      outputMatches,
    },
    ...(failureReason ? { failureReason } : {}),
    summary: {
      lane,
      status,
      required: input.required,
      webGpuAvailable: input.probe.webGpuAvailable,
      outputMatches,
      realHardwareAdapter: adapterAssessment.realHardware,
      adapterIdentityKnown: adapterAssessment.identityKnown,
      softwareAdapterDetected: adapterAssessment.softwareDetected,
      adapterVendor: input.probe.adapterInfo?.vendor ?? null,
      adapterArchitecture: input.probe.adapterInfo?.architecture ?? null,
      adapterDevice: input.probe.adapterInfo?.device ?? null,
      adapterDescription: input.probe.adapterInfo?.description ?? null,
      adapterFallback: input.probe.adapterInfo?.isFallbackAdapter ?? null,
      browserRunner: true,
      expectedOutput: input.probe.expectedOutput.join(","),
      actualOutput: input.probe.actualOutput.length > 0 ? input.probe.actualOutput.join(",") : null,
      adapterFeatureCount: input.probe.adapterFeatures.length,
      durationMs: input.probe.durationMs ?? null,
      failureReason: failureReason ?? null,
    },
  };
}

export async function writeHeadlessWebGpuCiParityArtifact(
  artifact: HeadlessWebGpuCiParityArtifact,
  options: { artifactDir: string },
): Promise<{ outputDir: string; resultsPath: string; summaryPath: string; latestPath: string }> {
  const timestamp = artifact.createdAt.replace(/[:.]/g, "-");
  const root = resolve(repoRoot, options.artifactDir);
  const outputDir = join(root, "headless-webgpu-ci-parity", timestamp);
  const resultsPath = join(outputDir, "results.json");
  const summaryPath = join(outputDir, "summary.md");
  const latestPath = join(root, "headless-webgpu-ci-parity-latest.json");
  await mkdir(outputDir, { recursive: true });
  await mkdir(root, { recursive: true });
  await writeFile(resultsPath, `${JSON.stringify(artifact, null, 2)}\n`);
  await writeFile(summaryPath, buildMarkdownSummary(artifact));
  await writeFile(latestPath, `${JSON.stringify(artifact, null, 2)}\n`);
  return { outputDir, resultsPath, summaryPath, latestPath };
}

export function buildMarkdownSummary(artifact: HeadlessWebGpuCiParityArtifact): string {
  return `# Headless WebGPU CI Parity

- Created: ${artifact.createdAt}
- Status: ${artifact.status}
- Passed: ${artifact.passed}
- Required: ${artifact.required}
- Lane: ${artifact.lane}
- Browser: ${artifact.browser.engine}${artifact.browser.channel ? `/${artifact.browser.channel}` : ""}
- WebGPU available: ${artifact.webGpuAvailable}
- Output matches: ${artifact.probe.outputMatches}
- Real hardware adapter: ${artifact.summary.realHardwareAdapter}
- Adapter identity known: ${artifact.summary.adapterIdentityKnown}
- Software adapter detected: ${artifact.summary.softwareAdapterDetected}
- Adapter: ${formatAdapterInfo(artifact.probe.adapterInfo)}
- Expected output: ${artifact.probe.expectedOutput.join(",")}
- Actual output: ${artifact.probe.actualOutput.length > 0 ? artifact.probe.actualOutput.join(",") : "none"}
- Failure reason: ${artifact.failureReason ?? "none"}
`;
}

function printHeadlessWebGpuCiParityArtifact(artifact: HeadlessWebGpuCiParityArtifact): void {
  console.log(`Headless WebGPU CI parity: ${artifact.status.toUpperCase()}`);
  console.log(`Lane: ${artifact.lane}`);
  console.log(`Required: ${artifact.required}`);
  console.log(`WebGPU available: ${artifact.webGpuAvailable}`);
  console.log(`Output matches: ${artifact.probe.outputMatches}`);
  if (artifact.failureReason) console.log(`Reason: ${artifact.failureReason}`);
}

function unavailableProbe(error: string): HeadlessWebGpuCiParityProbeResult {
  return {
    webGpuAvailable: false,
    adapterFeatures: [],
    expectedOutput: EXPECTED_COMPUTE_OUTPUT,
    actualOutput: [],
    error,
  };
}

function buildFailureReason(
  probe: HeadlessWebGpuCiParityProbeResult,
  outputMatches: boolean,
  adapterReason: string,
): string {
  if (!probe.webGpuAvailable) return probe.error ?? "real browser WebGPU is unavailable.";
  if (!outputMatches) {
    return `compute shader output did not match expected ${probe.expectedOutput.join(",")}; received ${probe.actualOutput.join(",")}.`;
  }
  if (adapterReason) return adapterReason;
  return probe.error ?? "headless WebGPU parity probe failed.";
}

function assessRealHardwareAdapter(probe: HeadlessWebGpuCiParityProbeResult): {
  realHardware: boolean;
  identityKnown: boolean;
  softwareDetected: boolean;
  reason: string;
} {
  if (!probe.webGpuAvailable) {
    return {
      realHardware: false,
      identityKnown: false,
      softwareDetected: false,
      reason: "",
    };
  }
  const info = probe.adapterInfo;
  const identityText = [
    normalizeAdapterIdentityValue(info?.vendor),
    normalizeAdapterIdentityValue(info?.architecture),
    normalizeAdapterIdentityValue(info?.device),
    normalizeAdapterIdentityValue(info?.description),
  ].filter((value): value is string => Boolean(value)).join(" ").trim();
  const identityKnown = identityText.length > 0;
  if (!identityKnown) {
    return {
      realHardware: false,
      identityKnown: false,
      softwareDetected: false,
      reason: "browser WebGPU adapter identity is unavailable, so real hardware cannot be proven.",
    };
  }
  const lowerIdentity = identityText.toLowerCase();
  const softwareTokens = [
    "swiftshader",
    "lavapipe",
    "llvmpipe",
    "softpipe",
    "software",
    "cpu rasterizer",
    "fallback adapter",
  ];
  const softwareDetected = info?.isFallbackAdapter === true
    || softwareTokens.some((token) => lowerIdentity.includes(token));
  if (softwareDetected) {
    return {
      realHardware: false,
      identityKnown,
      softwareDetected: true,
      reason: `software WebGPU adapter rejected: ${identityText}.`,
    };
  }
  return {
    realHardware: true,
    identityKnown,
    softwareDetected: false,
    reason: "",
  };
}

function normalizeAdapterIdentityValue(value: string | undefined): string | undefined {
  const trimmed = value?.trim();
  if (!trimmed) return undefined;
  const normalized = trimmed.toLowerCase();
  if (
    normalized === "unknown"
    || normalized === "undefined"
    || normalized === "null"
    || normalized === "n/a"
    || normalized === "na"
    || normalized === "none"
    || normalized === "not available"
    || normalized === "unavailable"
  ) {
    return undefined;
  }
  return trimmed;
}

function formatAdapterInfo(info: HeadlessWebGpuCiParityAdapterInfo | undefined): string {
  if (!info) return "unknown";
  const parts = [
    info.vendor,
    info.architecture,
    info.device,
    info.description,
    info.isFallbackAdapter === true ? "fallback" : undefined,
  ].filter(Boolean);
  return parts.length > 0 ? parts.join(" / ") : "unknown";
}

function computeOutputMatches(actual: number[], expected: number[]): boolean {
  return expected.length > 0
    && actual.length === expected.length
    && expected.every((value, index) => Object.is(actual[index], value));
}

function readBrowserChannel(parsed: Map<string, string>, env: Partial<Record<string, string | undefined>>): string | undefined {
  return parsed.get("browser-channel") ?? env.HEADLESS_WEBGPU_CI_BROWSER_CHANNEL ?? env.BROWSER_RUNTIME_BENCH_BROWSER_CHANNEL;
}

function readBrowserExecutable(parsed: Map<string, string>, env: Partial<Record<string, string | undefined>>): string | undefined {
  return parsed.get("browser-executable") ?? env.HEADLESS_WEBGPU_CI_BROWSER_EXECUTABLE ?? env.BROWSER_RUNTIME_BENCH_BROWSER_EXECUTABLE;
}

function parseRealBrowserLane(value: string | undefined): HeadlessWebGpuCiParityLane {
  if (!value || value === "browser-runner") return "browser-runner";
  throw new Error(`Only the real browser-runner lane is supported for headless WebGPU CI parity; received "${value}".`);
}

function parsePositiveInteger(value: string | undefined, fallback: number): number {
  if (!value?.trim()) return fallback;
  const parsed = Number(value);
  if (!Number.isFinite(parsed)) return fallback;
  const integer = Math.floor(parsed);
  return integer > 0 ? integer : fallback;
}

function isMainModule(): boolean {
  return process.argv[1] ? import.meta.url === pathToFileURL(process.argv[1]).toString() : false;
}
