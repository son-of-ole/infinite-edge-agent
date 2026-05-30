import { mkdir, readFile, writeFile } from "node:fs/promises";
import { join } from "node:path";

export interface HostedBenchmarkProof {
  sourceName: string;
  runtimeBackendId: string | null;
  deployBackendId: string | null;
  response: string | null;
  productionDeployReadyPassed: boolean;
  compiledBackendReadyPassed: boolean;
  memoryGroundingRequired: boolean;
  memoryGroundingPassed: boolean;
  memoryExpectedHitPassed: boolean;
  memoryContextRebuildPassed: boolean;
  memoryAnswerOnlyPassed: boolean;
  expectedExactPassed: boolean;
  expectedExactCheckCount: number | null;
  productionQualityPassed: boolean;
  productionSpeedFloorPassed: boolean;
  productionSpeedTokensPerSecond: number | null;
  productionSpeedFloorTokensPerSecond: number | null;
  meanTokensPerSecond: number | null;
  directModelFactualProofUsed: boolean;
  technicalProofOnly: boolean;
  strictWebGpuPassed: boolean;
  cpuFallbackUsed: boolean;
  backendBrokerTraceCount: number;
  backendBrokerSelectionPassed: boolean;
  brokerSelectedBackendId: string | null;
  brokerSelectedModelId: string | null;
  brokerProductionRole: string | null;
  brokerDeployReadyCandidate: boolean;
  brokerReason: string | null;
  brokerProofRequirements: string[];
}

export interface HostedBenchmarkProofReport {
  passed: boolean;
  blockers: string[];
  artifactPath: string | null;
  proof: HostedBenchmarkProof;
}

export interface HostedBenchmarkProofArtifact {
  name: "hosted-benchmark-proof";
  createdAt: string;
  passed: boolean;
  summary: Record<string, number | string | boolean | null>;
  report: HostedBenchmarkProofReport;
}

export interface HostedBenchmarkProofArtifactWriteResult {
  artifact: HostedBenchmarkProofArtifact;
  latestPath: string;
  resultPath: string;
}

const DEFAULT_BACKEND_ID = "compiled-browser-webllm";
const DEFAULT_EXPECTED_RESPONSE = "Helena";
const DEFAULT_SPEED_FLOOR = 2;

export async function evaluateHostedBenchmarkProofFile(
  artifactPath: string,
  options: {
    expectedBackendId?: string;
    expectedResponse?: string | null;
    minTokensPerSecond?: number;
  } = {},
): Promise<HostedBenchmarkProofReport> {
  const artifact = JSON.parse(await readFile(artifactPath, "utf8")) as unknown;
  return evaluateHostedBenchmarkProof({
    artifact,
    artifactPath,
    ...options,
  });
}

export function evaluateHostedBenchmarkProof(input: {
  artifact: unknown;
  artifactPath?: string | null;
  expectedBackendId?: string;
  expectedResponse?: string | null;
  minTokensPerSecond?: number;
}): HostedBenchmarkProofReport {
  const expectedBackendId = input.expectedBackendId ?? DEFAULT_BACKEND_ID;
  const expectedResponse = input.expectedResponse === undefined ? DEFAULT_EXPECTED_RESPONSE : input.expectedResponse;
  const minTokensPerSecond = input.minTokensPerSecond ?? DEFAULT_SPEED_FLOOR;
  const source = extractBenchmarkSource(input.artifact);
  const proof = source
    ? buildProofFromSource(source)
    : buildEmptyProof();
  const blockers: string[] = [];

  if (!source) {
    blockers.push("Hosted benchmark proof artifact must be browser-preview-benchmark or browser-runtime-bench with completed browserPreview.");
  } else if (source.sourcePassed !== true) {
    blockers.push("Hosted benchmark proof artifact must have passed=true.");
  }
  if (proof.runtimeBackendId !== expectedBackendId) {
    blockers.push(`Hosted benchmark proof requires runtimeBackendId=${expectedBackendId}.`);
  }
  if (proof.deployBackendId !== expectedBackendId) {
    blockers.push(`Hosted benchmark proof requires deployBackendId=${expectedBackendId}.`);
  }
  if (proof.productionDeployReadyPassed !== true) {
    blockers.push("Hosted benchmark proof requires productionDeployReadyPassed=true.");
  }
  if (proof.compiledBackendReadyPassed !== true) {
    blockers.push("Hosted benchmark proof requires compiledBackendReadyPassed=true.");
  }
  if (proof.memoryGroundingRequired !== true) {
    blockers.push("Hosted benchmark proof requires memoryGroundingRequired=true.");
  }
  if (proof.memoryGroundingPassed !== true) {
    blockers.push("Hosted benchmark proof requires memoryGroundingPassed=true.");
  }
  if (proof.memoryExpectedHitPassed !== true) {
    blockers.push("Hosted benchmark proof requires memoryExpectedHitPassed=true.");
  }
  if (proof.memoryContextRebuildPassed !== true) {
    blockers.push("Hosted benchmark proof requires memoryContextRebuildPassed=true.");
  }
  if (proof.memoryAnswerOnlyPassed !== true) {
    blockers.push("Hosted benchmark proof requires memoryAnswerOnlyPassed=true.");
  }
  if (proof.expectedExactPassed !== true || (proof.expectedExactCheckCount ?? 0) <= 0) {
    blockers.push("Hosted benchmark proof requires a passing expectedExact check.");
  }
  if (proof.directModelFactualProofUsed === true) {
    blockers.push("Hosted benchmark proof cannot count direct model factual output as retrieval proof.");
  }
  if (proof.technicalProofOnly === true) {
    blockers.push("Hosted benchmark proof cannot be technicalProofOnly.");
  }
  if (proof.productionQualityPassed !== true) {
    blockers.push("Hosted benchmark proof requires productionQualityPassed=true.");
  }
  if (proof.productionSpeedFloorPassed !== true) {
    blockers.push("Hosted benchmark proof requires productionSpeedFloorPassed=true.");
  }
  if ((proof.productionSpeedTokensPerSecond ?? proof.meanTokensPerSecond ?? 0) < minTokensPerSecond) {
    blockers.push(`Hosted benchmark proof requires at least ${minTokensPerSecond} tokens/sec.`);
  }
  if (proof.strictWebGpuPassed !== true) {
    blockers.push("Hosted benchmark proof requires strictWebGpuPassed=true.");
  }
  if (proof.cpuFallbackUsed !== false) {
    blockers.push("Hosted benchmark proof requires cpuFallbackUsed=false.");
  }
  if (
    proof.backendBrokerSelectionPassed !== true
    || proof.backendBrokerTraceCount <= 0
    || proof.brokerSelectedBackendId !== expectedBackendId
    || proof.brokerProductionRole !== "production_candidate"
    || proof.brokerDeployReadyCandidate !== true
    || !proof.brokerProofRequirements.includes("backend_trace")
    || !proof.brokerProofRequirements.includes("memory_grounding")
  ) {
    blockers.push(`Hosted benchmark proof requires Backend Broker selection evidence for ${expectedBackendId}.`);
  }
  if (expectedResponse && proof.response !== null && proof.response.trim() !== expectedResponse) {
    blockers.push(`Hosted benchmark proof expected response ${expectedResponse}.`);
  }

  return {
    passed: blockers.length === 0,
    blockers,
    artifactPath: input.artifactPath ?? null,
    proof,
  };
}

export function buildHostedBenchmarkProofArtifact(
  report: HostedBenchmarkProofReport,
  createdAt = new Date().toISOString(),
): HostedBenchmarkProofArtifact {
  return {
    name: "hosted-benchmark-proof",
    createdAt,
    passed: report.passed,
    summary: {
      hostedBenchmarkProofPassed: report.passed,
      hostedBenchmarkProofBlockerCount: report.blockers.length,
      hostedBenchmarkArtifactPath: report.artifactPath,
      hostedBenchmarkRuntimeBackendId: report.proof.runtimeBackendId,
      hostedBenchmarkDeployBackendId: report.proof.deployBackendId,
      hostedBenchmarkCompiledBackendReadyPassed: report.proof.compiledBackendReadyPassed,
      hostedBenchmarkProductionDeployReadyPassed: report.proof.productionDeployReadyPassed,
      hostedBenchmarkMemoryGroundingPassed: report.proof.memoryGroundingPassed,
      hostedBenchmarkExpectedExactPassed: report.proof.expectedExactPassed,
      hostedBenchmarkProductionSpeedFloorPassed: report.proof.productionSpeedFloorPassed,
      hostedBenchmarkMeanTokensPerSecond: report.proof.meanTokensPerSecond,
      hostedBenchmarkDirectModelFactualProofUsed: report.proof.directModelFactualProofUsed,
      hostedBenchmarkTechnicalProofOnly: report.proof.technicalProofOnly,
      hostedBenchmarkCpuFallbackUsed: report.proof.cpuFallbackUsed,
      hostedBenchmarkStrictWebGpuPassed: report.proof.strictWebGpuPassed,
      hostedBenchmarkBackendBrokerSelectionPassed: report.proof.backendBrokerSelectionPassed,
      hostedBenchmarkBackendBrokerTraceCount: report.proof.backendBrokerTraceCount,
      hostedBenchmarkBrokerSelectedBackendId: report.proof.brokerSelectedBackendId,
      hostedBenchmarkBrokerSelectedModelId: report.proof.brokerSelectedModelId,
      hostedBenchmarkBrokerProductionRole: report.proof.brokerProductionRole,
      hostedBenchmarkBrokerDeployReadyCandidate: report.proof.brokerDeployReadyCandidate,
      hostedBenchmarkBrokerReason: report.proof.brokerReason,
    },
    report,
  };
}

export async function writeHostedBenchmarkProofArtifact(
  report: HostedBenchmarkProofReport,
  options: { artifactDir?: string; createdAt?: string } = {},
): Promise<HostedBenchmarkProofArtifactWriteResult> {
  const artifactDir = options.artifactDir ?? process.env.EVAL_ARTIFACT_DIR ?? ".artifacts/evals";
  const artifact = buildHostedBenchmarkProofArtifact(report, options.createdAt);
  const runDir = join(artifactDir, "hosted-benchmark-proof");
  const timestamp = artifact.createdAt.replace(/[:.]/g, "-");
  const latestPath = join(artifactDir, "hosted-benchmark-proof-latest.json");
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

interface BenchmarkSource {
  sourceName: string;
  sourcePassed: boolean;
  summary: Record<string, unknown>;
  runs: unknown[];
}

function extractBenchmarkSource(artifact: unknown): BenchmarkSource | null {
  if (!isRecord(artifact)) return null;
  const name = readString(artifact.name) ?? "unknown";
  if (name === "browser-runtime-bench") {
    const browserPreview = isRecord(artifact.browserPreview) ? artifact.browserPreview : null;
    if (browserPreview?.mode !== "completed" || !isRecord(browserPreview.summary)) return null;
    return {
      sourceName: "browser-runtime-bench.browserPreview",
      sourcePassed: artifact.passed === true && browserPreview.passed === true,
      summary: browserPreview.summary,
      runs: Array.isArray(browserPreview.runs) ? browserPreview.runs : [],
    };
  }
  if (!isRecord(artifact.summary)) return null;
  return {
    sourceName: name,
    sourcePassed: artifact.passed === true,
    summary: artifact.summary,
    runs: Array.isArray(artifact.runs) ? artifact.runs : [],
  };
}

function buildProofFromSource(source: BenchmarkSource): HostedBenchmarkProof {
  const firstRun = isRecord(source.runs[0]) ? source.runs[0] : {};
  const runtimeTrace = isRecord(firstRun.runtimeTrace) ? firstRun.runtimeTrace : {};
  const runtimeBackendId = readString(source.summary.runtimeBackendId) ?? readString(runtimeTrace.backend);
  const brokerSelection = readBrokerSelection(runtimeTrace.brokerSelection);
  const brokerSelectedBackendId = readString(source.summary.backendBrokerSelectedBackendId) ?? brokerSelection?.backendId ?? null;
  const brokerSelectedModelId = readString(source.summary.backendBrokerSelectedModelId) ?? brokerSelection?.modelId ?? null;
  const brokerProductionRole = readString(source.summary.backendBrokerProductionRole) ?? brokerSelection?.productionRole ?? null;
  const brokerDeployReadyCandidate = readBoolean(source.summary.backendBrokerDeployReadyCandidate)
    || brokerSelection?.deployReadyCandidate === true;
  const brokerReason = readString(source.summary.backendBrokerReason) ?? brokerSelection?.reason ?? null;
  const brokerProofRequirements = readStringList(source.summary.backendBrokerProofRequirements)
    ?? brokerSelection?.proofRequirements
    ?? [];
  const backendBrokerTraceCount = readNumber(source.summary.backendBrokerTraceCount)
    ?? (brokerSelection ? 1 : 0);
  const backendBrokerSelectionPassed = readBoolean(source.summary.backendBrokerSelectionPassed)
    || (
      brokerSelectedBackendId === runtimeBackendId
      && brokerProductionRole === "production_candidate"
      && brokerDeployReadyCandidate
      && brokerProofRequirements.includes("backend_trace")
      && brokerProofRequirements.includes("memory_grounding")
    );
  return {
    sourceName: source.sourceName,
    runtimeBackendId,
    deployBackendId: readString(source.summary.deployBackendId) ?? runtimeBackendId,
    response: readString(firstRun.response),
    productionDeployReadyPassed: readBoolean(source.summary.productionDeployReadyPassed),
    compiledBackendReadyPassed: readBoolean(source.summary.compiledBackendReadyPassed),
    memoryGroundingRequired: readBoolean(source.summary.memoryGroundingRequired),
    memoryGroundingPassed: readBoolean(source.summary.memoryGroundingPassed),
    memoryExpectedHitPassed: readBoolean(source.summary.memoryExpectedHitPassed),
    memoryContextRebuildPassed: readBoolean(source.summary.memoryContextRebuildPassed),
    memoryAnswerOnlyPassed: readBoolean(source.summary.memoryAnswerOnlyPassed),
    expectedExactPassed: readBoolean(source.summary.expectedExactPassed),
    expectedExactCheckCount: readNumber(source.summary.expectedExactCheckCount),
    productionQualityPassed: readBoolean(source.summary.productionQualityPassed),
    productionSpeedFloorPassed: readBoolean(source.summary.productionSpeedFloorPassed),
    productionSpeedTokensPerSecond: readNumber(source.summary.productionSpeedTokensPerSecond),
    productionSpeedFloorTokensPerSecond: readNumber(source.summary.productionSpeedFloorTokensPerSecond),
    meanTokensPerSecond: readNumber(source.summary.meanTokensPerSecond),
    directModelFactualProofUsed: readBoolean(source.summary.directModelFactualProofUsed),
    technicalProofOnly: readBoolean(source.summary.technicalProofOnly),
    strictWebGpuPassed: readBoolean(source.summary.strictWebGpuPassed),
    cpuFallbackUsed: readBoolean(source.summary.cpuFallbackUsed),
    backendBrokerTraceCount,
    backendBrokerSelectionPassed,
    brokerSelectedBackendId,
    brokerSelectedModelId,
    brokerProductionRole,
    brokerDeployReadyCandidate,
    brokerReason,
    brokerProofRequirements,
  };
}

function buildEmptyProof(): HostedBenchmarkProof {
  return {
    sourceName: "unknown",
    runtimeBackendId: null,
    deployBackendId: null,
    response: null,
    productionDeployReadyPassed: false,
    compiledBackendReadyPassed: false,
    memoryGroundingRequired: false,
    memoryGroundingPassed: false,
    memoryExpectedHitPassed: false,
    memoryContextRebuildPassed: false,
    memoryAnswerOnlyPassed: false,
    expectedExactPassed: false,
    expectedExactCheckCount: null,
    productionQualityPassed: false,
    productionSpeedFloorPassed: false,
    productionSpeedTokensPerSecond: null,
    productionSpeedFloorTokensPerSecond: null,
    meanTokensPerSecond: null,
    directModelFactualProofUsed: false,
    technicalProofOnly: true,
    strictWebGpuPassed: false,
    cpuFallbackUsed: true,
    backendBrokerTraceCount: 0,
    backendBrokerSelectionPassed: false,
    brokerSelectedBackendId: null,
    brokerSelectedModelId: null,
    brokerProductionRole: null,
    brokerDeployReadyCandidate: false,
    brokerReason: null,
    brokerProofRequirements: [],
  };
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function readString(value: unknown): string | null {
  return typeof value === "string" && value.trim() ? value : null;
}

function readBoolean(value: unknown): boolean {
  return value === true;
}

function readNumber(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function readStringList(value: unknown): string[] | null {
  if (!Array.isArray(value)) return null;
  return value.filter((item): item is string => typeof item === "string" && item.trim().length > 0);
}

function readBrokerSelection(value: unknown): {
  backendId: string;
  modelId: string;
  productionRole: string;
  deployReadyCandidate: boolean;
  reason: string;
  proofRequirements: string[];
} | null {
  if (!isRecord(value)) return null;
  const backendId = readString(value.backendId);
  const modelId = readString(value.modelId);
  const productionRole = readString(value.productionRole);
  const reason = readString(value.reason);
  const proofRequirements = readStringList(value.proofRequirements);
  if (!backendId || !modelId || !productionRole || !reason || !proofRequirements) return null;
  return {
    backendId,
    modelId,
    productionRole,
    deployReadyCandidate: value.deployReadyCandidate === true,
    reason,
    proofRequirements,
  };
}

if (import.meta.url === `file://${process.argv[1]}`) {
  const artifactPath = process.argv[2] ?? process.env.HOSTED_BENCHMARK_ARTIFACT_PATH;
  if (!artifactPath) {
    console.error("Usage: pnpm verify:hosted-benchmark-proof -- <browser-runtime-bench-latest.json>");
    console.error("Or set HOSTED_BENCHMARK_ARTIFACT_PATH.");
    process.exit(1);
  }
  const report = await evaluateHostedBenchmarkProofFile(artifactPath);
  await writeHostedBenchmarkProofArtifact(report);
  console.log(JSON.stringify(report, null, 2));
  if (!report.passed) process.exitCode = 1;
}
