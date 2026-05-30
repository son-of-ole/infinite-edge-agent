import { mkdir, readFile, writeFile } from "node:fs/promises";
import { join } from "node:path";

export interface HostedBenchmarkProof {
  sourceName: string;
  v12ProductionProofSchemaVersion: number | null;
  sourceGitSha: string | null;
  sourceCommitEvidencePassed: boolean;
  runtimeBackendId: string | null;
  deployBackendId: string | null;
  response: string | null;
  productionDeployReadyPassed: boolean;
  compiledBackendReadyPassed: boolean;
  memoryGroundingRequired: boolean;
  memoryGroundingPassed: boolean;
  concreteMemoryGroundingPassed: boolean;
  memoryGroundingRunCount: number;
  memoryGroundingCaseId: string | null;
  memorySeededCorpusCount: number | null;
  memoryRetrievedCount: number | null;
  memoryIncludedCount: number | null;
  memoryExpectedMemoryIdCount: number | null;
  memoryExpectedHitMeanRank: number | null;
  memoryExpectedHitMinTopScoreMargin: number | null;
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
  gpuLabelEvidencePassed: boolean;
  gpuVendor: string | null;
  gpuArchitecture: string | null;
  gpuDevice: string | null;
  gpuDescription: string | null;
  webglRenderer: string | null;
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
  brokerDeployBackendId: string | null;
  brokerKernelLabBackendId: string | null;
  brokerFallbackBackendId: string | null;
  brokerFallbackBackendCount: number | null;
  brokerFallbackDeployReadyCandidate: boolean;
  brokerRoleBoundaryPassed: boolean;
}

export interface HostedBenchmarkProofReport {
  passed: boolean;
  blockers: string[];
  artifactPath: string | null;
  expectedSourceGitSha: string | null;
  sourceBoundRequired: boolean;
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
export const REQUIRED_V12_PRODUCTION_PROOF_SCHEMA_VERSION = 2;

export async function evaluateHostedBenchmarkProofFile(
  artifactPath: string,
  options: {
    expectedBackendId?: string;
    expectedResponse?: string | null;
    minTokensPerSecond?: number;
    expectedSourceGitSha?: string | null;
    requireSourceBound?: boolean;
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
  expectedSourceGitSha?: string | null;
  requireSourceBound?: boolean;
}): HostedBenchmarkProofReport {
  const expectedBackendId = input.expectedBackendId ?? DEFAULT_BACKEND_ID;
  const expectedResponse = input.expectedResponse === undefined ? DEFAULT_EXPECTED_RESPONSE : input.expectedResponse;
  const minTokensPerSecond = input.minTokensPerSecond ?? DEFAULT_SPEED_FLOOR;
  const expectedSourceGitSha = normalizeString(input.expectedSourceGitSha);
  const sourceBoundRequired = input.requireSourceBound === true;
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
  if (expectedSourceGitSha && proof.sourceGitSha !== expectedSourceGitSha) {
    blockers.push(`Hosted benchmark proof source commit ${proof.sourceGitSha ?? "unknown"} does not match expected commit ${expectedSourceGitSha}.`);
  }
  if (sourceBoundRequired && !expectedSourceGitSha) {
    blockers.push("Hosted benchmark proof requires an expected source commit when source binding is required.");
  }
  if (proof.v12ProductionProofSchemaVersion !== REQUIRED_V12_PRODUCTION_PROOF_SCHEMA_VERSION) {
    blockers.push(`Hosted benchmark proof requires v12 production proof schema version ${REQUIRED_V12_PRODUCTION_PROOF_SCHEMA_VERSION}.`);
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
  if (proof.concreteMemoryGroundingPassed !== true) {
    blockers.push("Hosted benchmark proof requires concrete run-level memory grounding evidence.");
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
  if (proof.gpuLabelEvidencePassed !== true) {
    blockers.push("Hosted benchmark proof requires browser GPU label evidence.");
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
  if (
    proof.brokerDeployBackendId !== expectedBackendId
    || proof.brokerKernelLabBackendId !== "unlocked-browser-transformer"
    || proof.brokerFallbackBackendId !== "wasm-small-core"
    || proof.brokerFallbackBackendCount !== 1
    || proof.brokerFallbackDeployReadyCandidate !== false
    || proof.brokerRoleBoundaryPassed !== true
  ) {
    blockers.push("Hosted benchmark proof requires Backend Broker role-boundary evidence for compiled deploy, Kernel Lab, and fallback backends.");
  }
  if (expectedResponse && proof.response !== null && proof.response.trim() !== expectedResponse) {
    blockers.push(`Hosted benchmark proof expected response ${expectedResponse}.`);
  }

  return {
    passed: blockers.length === 0,
    blockers,
    artifactPath: input.artifactPath ?? null,
    expectedSourceGitSha,
    sourceBoundRequired,
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
      hostedBenchmarkV12ProductionProofSchemaVersion: report.proof.v12ProductionProofSchemaVersion,
      hostedBenchmarkProofSourceGitSha: report.proof.sourceGitSha,
      hostedBenchmarkProofSourceCommitEvidencePassed: report.proof.sourceCommitEvidencePassed,
      hostedBenchmarkExpectedSourceGitSha: report.expectedSourceGitSha,
      hostedBenchmarkProofSourceBoundRequired: report.sourceBoundRequired,
      hostedBenchmarkProofSourceBound: report.expectedSourceGitSha
        ? report.proof.sourceGitSha === report.expectedSourceGitSha
        : null,
      hostedBenchmarkRuntimeBackendId: report.proof.runtimeBackendId,
      hostedBenchmarkDeployBackendId: report.proof.deployBackendId,
      hostedBenchmarkCompiledBackendReadyPassed: report.proof.compiledBackendReadyPassed,
      hostedBenchmarkProductionDeployReadyPassed: report.proof.productionDeployReadyPassed,
      hostedBenchmarkMemoryGroundingPassed: report.proof.memoryGroundingPassed,
      hostedBenchmarkConcreteMemoryGroundingPassed: report.proof.concreteMemoryGroundingPassed,
      hostedBenchmarkMemoryGroundingRunCount: report.proof.memoryGroundingRunCount,
      hostedBenchmarkMemoryGroundingCaseId: report.proof.memoryGroundingCaseId,
      hostedBenchmarkMemorySeededCorpusCount: report.proof.memorySeededCorpusCount,
      hostedBenchmarkMemoryRetrievedCount: report.proof.memoryRetrievedCount,
      hostedBenchmarkMemoryIncludedCount: report.proof.memoryIncludedCount,
      hostedBenchmarkMemoryExpectedMemoryIdCount: report.proof.memoryExpectedMemoryIdCount,
      hostedBenchmarkMemoryExpectedHitMeanRank: report.proof.memoryExpectedHitMeanRank,
      hostedBenchmarkMemoryExpectedHitMinTopScoreMargin: report.proof.memoryExpectedHitMinTopScoreMargin,
      hostedBenchmarkExpectedExactPassed: report.proof.expectedExactPassed,
      hostedBenchmarkProductionSpeedFloorPassed: report.proof.productionSpeedFloorPassed,
      hostedBenchmarkMeanTokensPerSecond: report.proof.meanTokensPerSecond,
      hostedBenchmarkGpuLabelEvidencePassed: report.proof.gpuLabelEvidencePassed,
      hostedBenchmarkGpuVendor: report.proof.gpuVendor,
      hostedBenchmarkGpuArchitecture: report.proof.gpuArchitecture,
      hostedBenchmarkGpuDevice: report.proof.gpuDevice,
      hostedBenchmarkGpuDescription: report.proof.gpuDescription,
      hostedBenchmarkWebGlRenderer: report.proof.webglRenderer,
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
      hostedBenchmarkBrokerDeployBackendId: report.proof.brokerDeployBackendId,
      hostedBenchmarkBrokerKernelLabBackendId: report.proof.brokerKernelLabBackendId,
      hostedBenchmarkBrokerFallbackBackendId: report.proof.brokerFallbackBackendId,
      hostedBenchmarkBrokerFallbackBackendCount: report.proof.brokerFallbackBackendCount,
      hostedBenchmarkBrokerFallbackDeployReadyCandidate: report.proof.brokerFallbackDeployReadyCandidate,
      hostedBenchmarkBrokerRoleBoundaryPassed: report.proof.brokerRoleBoundaryPassed,
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
  schemaVersion: number | null;
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
      schemaVersion: readNumber(browserPreview.schemaVersion) ?? readNumber(artifact.schemaVersion),
      summary: browserPreview.summary,
      runs: Array.isArray(browserPreview.runs) ? browserPreview.runs : [],
    };
  }
  if (!isRecord(artifact.summary)) return null;
  return {
    sourceName: name,
    sourcePassed: artifact.passed === true,
    schemaVersion: readNumber(artifact.schemaVersion),
    summary: artifact.summary,
    runs: Array.isArray(artifact.runs) ? artifact.runs : [],
  };
}

function buildProofFromSource(source: BenchmarkSource): HostedBenchmarkProof {
  const firstRun = isRecord(source.runs[0]) ? source.runs[0] : {};
  const runtimeTrace = isRecord(firstRun.runtimeTrace) ? firstRun.runtimeTrace : {};
  const runtimeBackendId = readString(source.summary.runtimeBackendId) ?? readString(runtimeTrace.backend);
  const v12ProductionProofSchemaVersion = readNumber(source.summary.v12ProductionProofSchemaVersion) ?? source.schemaVersion;
  const sourceGitSha = readString(source.summary.v12ProductionProofSourceGitSha)
    ?? readString(source.summary.gitSha)
    ?? null;
  const sourceCommitEvidencePassed = readBoolean(source.summary.v12ProductionProofSourceCommitEvidencePassed)
    || Boolean(sourceGitSha);
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
  const brokerDeployBackendId = readString(source.summary.backendBrokerDeployBackendId);
  const brokerKernelLabBackendId = readString(source.summary.backendBrokerKernelLabBackendId);
  const brokerFallbackBackendId = readString(source.summary.backendBrokerFallbackBackendId);
  const brokerFallbackBackendCount = readNumber(source.summary.backendBrokerFallbackBackendCount);
  const brokerFallbackDeployReadyCandidate = readBoolean(source.summary.backendBrokerFallbackDeployReadyCandidate);
  const brokerRoleBoundaryPassed = readBoolean(source.summary.backendBrokerRoleBoundaryPassed);
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
  const memoryGroundingEvidence = readMemoryGroundingEvidence(source.runs);
  const gpuEvidence = readGpuEvidence(source.summary, source.runs);
  return {
    sourceName: source.sourceName,
    v12ProductionProofSchemaVersion,
    sourceGitSha,
    sourceCommitEvidencePassed,
    runtimeBackendId,
    deployBackendId: readString(source.summary.deployBackendId) ?? runtimeBackendId,
    response: readString(firstRun.response),
    productionDeployReadyPassed: readBoolean(source.summary.productionDeployReadyPassed),
    compiledBackendReadyPassed: readBoolean(source.summary.compiledBackendReadyPassed),
    memoryGroundingRequired: readBoolean(source.summary.memoryGroundingRequired),
    memoryGroundingPassed: readBoolean(source.summary.memoryGroundingPassed),
    concreteMemoryGroundingPassed: memoryGroundingEvidence.passed,
    memoryGroundingRunCount: memoryGroundingEvidence.runCount,
    memoryGroundingCaseId: memoryGroundingEvidence.caseId,
    memorySeededCorpusCount: memoryGroundingEvidence.seededCorpusCount,
    memoryRetrievedCount: memoryGroundingEvidence.retrievedCount,
    memoryIncludedCount: memoryGroundingEvidence.includedCount,
    memoryExpectedMemoryIdCount: memoryGroundingEvidence.expectedMemoryIdCount,
    memoryExpectedHitMeanRank: memoryGroundingEvidence.expectedHitMeanRank,
    memoryExpectedHitMinTopScoreMargin: memoryGroundingEvidence.expectedHitMinTopScoreMargin,
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
    gpuLabelEvidencePassed: gpuEvidence.passed,
    gpuVendor: gpuEvidence.vendor,
    gpuArchitecture: gpuEvidence.architecture,
    gpuDevice: gpuEvidence.device,
    gpuDescription: gpuEvidence.description,
    webglRenderer: gpuEvidence.webglRenderer,
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
    brokerDeployBackendId,
    brokerKernelLabBackendId,
    brokerFallbackBackendId,
    brokerFallbackBackendCount,
    brokerFallbackDeployReadyCandidate,
    brokerRoleBoundaryPassed,
  };
}

function buildEmptyProof(): HostedBenchmarkProof {
  return {
    sourceName: "unknown",
    v12ProductionProofSchemaVersion: null,
    sourceGitSha: null,
    sourceCommitEvidencePassed: false,
    runtimeBackendId: null,
    deployBackendId: null,
    response: null,
    productionDeployReadyPassed: false,
    compiledBackendReadyPassed: false,
    memoryGroundingRequired: false,
    memoryGroundingPassed: false,
    concreteMemoryGroundingPassed: false,
    memoryGroundingRunCount: 0,
    memoryGroundingCaseId: null,
    memorySeededCorpusCount: null,
    memoryRetrievedCount: null,
    memoryIncludedCount: null,
    memoryExpectedMemoryIdCount: null,
    memoryExpectedHitMeanRank: null,
    memoryExpectedHitMinTopScoreMargin: null,
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
    gpuLabelEvidencePassed: false,
    gpuVendor: null,
    gpuArchitecture: null,
    gpuDevice: null,
    gpuDescription: null,
    webglRenderer: null,
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
    brokerDeployBackendId: null,
    brokerKernelLabBackendId: null,
    brokerFallbackBackendId: null,
    brokerFallbackBackendCount: null,
    brokerFallbackDeployReadyCandidate: false,
    brokerRoleBoundaryPassed: false,
  };
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function readString(value: unknown): string | null {
  return normalizeString(value);
}

function readBoolean(value: unknown): boolean {
  return value === true;
}

function readNumber(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function normalizeString(value: unknown): string | null {
  return typeof value === "string" && value.trim() ? value.trim() : null;
}

function readStringList(value: unknown): string[] | null {
  if (!Array.isArray(value)) return null;
  return value.filter((item): item is string => typeof item === "string" && item.trim().length > 0);
}

function readMemoryGroundingEvidence(runs: unknown[]): {
  passed: boolean;
  runCount: number;
  caseId: string | null;
  seededCorpusCount: number | null;
  retrievedCount: number | null;
  includedCount: number | null;
  expectedMemoryIdCount: number | null;
  expectedHitMeanRank: number | null;
  expectedHitMinTopScoreMargin: number | null;
} {
  const records = runs
    .map((run) => isRecord(run) && isRecord(run.memoryGrounding)
      ? readMemoryGroundingRecord(run, run.memoryGrounding)
      : null)
    .filter((record): record is NonNullable<ReturnType<typeof readMemoryGroundingRecord>> => Boolean(record));
  const validRecords = records.filter((record) => record.passed);
  return {
    passed: records.length > 0 && records.every((record) => record.passed),
    runCount: records.length,
    caseId: summarizeString(records.map((record) => record.caseId)),
    seededCorpusCount: maxNumber(records.map((record) => record.corpusCount)),
    retrievedCount: sumNumber(records.map((record) => record.retrievedCount)),
    includedCount: sumNumber(records.map((record) => record.includedCount)),
    expectedMemoryIdCount: sumNumber(records.map((record) => record.expectedMemoryIdCount)),
    expectedHitMeanRank: meanNumber(validRecords.map((record) => record.retrievalRank)),
    expectedHitMinTopScoreMargin: minNumber(validRecords.map((record) => record.retrievalTopScoreMargin)),
  };
}

function readMemoryGroundingRecord(
  run: Record<string, unknown>,
  memoryGrounding: Record<string, unknown>,
): {
  passed: boolean;
  caseId: string | null;
  corpusCount: number | null;
  retrievedCount: number;
  includedCount: number;
  expectedMemoryIdCount: number;
  retrievalRank: number | null;
  retrievalTopScoreMargin: number | null;
} {
  const expectedMemoryIds = readStringList(memoryGrounding.expectedMemoryIds) ?? [];
  const retrievedMemoryIds = readStringList(memoryGrounding.retrievedMemoryIds) ?? [];
  const includedMemoryIds = readStringList(memoryGrounding.includedMemoryIds) ?? [];
  const retrieved = new Set(retrievedMemoryIds);
  const included = new Set(includedMemoryIds);
  const retrievalRank = readNumber(memoryGrounding.retrievalRank);
  const retrievalTopScoreMargin = readNumber(memoryGrounding.retrievalTopScoreMargin);
  const answerOnlyExpected = readBoolean(memoryGrounding.answerOnlyExpected);
  const answerOnlyPassed = readBoolean(memoryGrounding.answerOnlyPassed) || readBoolean(run.expectedAnswerOnlyPassed);
  const expectedIdsPresent = expectedMemoryIds.length > 0;
  const expectedIdsRetrieved = expectedMemoryIds.every((id) => retrieved.has(id));
  const expectedIdsIncluded = expectedMemoryIds.every((id) => included.has(id));
  return {
    passed: readString(memoryGrounding.mode) === "seeded_browser_vector_context_rebuild"
      && readNumber(memoryGrounding.corpusCount) !== null
      && (readNumber(memoryGrounding.corpusCount) ?? 0) > 0
      && expectedIdsPresent
      && expectedIdsRetrieved
      && expectedIdsIncluded
      && readBoolean(memoryGrounding.expectedMemoryHitPassed)
      && readBoolean(memoryGrounding.contextRebuildPassed)
      && (!answerOnlyExpected || answerOnlyPassed)
      && retrievalRank !== null
      && retrievalRank > 0,
    caseId: readString(memoryGrounding.caseId),
    corpusCount: readNumber(memoryGrounding.corpusCount),
    retrievedCount: retrievedMemoryIds.length,
    includedCount: includedMemoryIds.length,
    expectedMemoryIdCount: expectedMemoryIds.length,
    retrievalRank,
    retrievalTopScoreMargin,
  };
}

function readGpuEvidence(
  summary: Record<string, unknown>,
  runs: unknown[],
): {
  passed: boolean;
  vendor: string | null;
  architecture: string | null;
  device: string | null;
  description: string | null;
  webglRenderer: string | null;
} {
  const runDeviceRecords = runs
    .map((run) => isRecord(run) && isRecord(run.device) ? run.device : null)
    .filter((record): record is Record<string, unknown> => Boolean(record));
  const vendor = readString(summary.benchmarkGpuVendor)
    ?? readString(summary.gpuVendor)
    ?? summarizeString(runDeviceRecords.map((record) => readString(record.gpuVendor)));
  const architecture = readString(summary.benchmarkGpuArchitecture)
    ?? readString(summary.gpuArchitecture)
    ?? summarizeString(runDeviceRecords.map((record) => readString(record.gpuArchitecture)));
  const device = readString(summary.benchmarkGpuDevice)
    ?? readString(summary.gpuDevice)
    ?? summarizeString(runDeviceRecords.map((record) => readString(record.gpuDevice)));
  const description = readString(summary.benchmarkGpuDescription)
    ?? readString(summary.gpuDescription)
    ?? summarizeString(runDeviceRecords.map((record) => readString(record.gpuDescription)));
  const webglRenderer = readString(summary.benchmarkWebGlRenderer)
    ?? readString(summary.webglRenderer)
    ?? summarizeString(runDeviceRecords.map((record) => readString(record.webglRenderer)));
  const hasLabel = Boolean(vendor || device || description || webglRenderer);
  return {
    passed: hasLabel && readBoolean(summary.benchmarkGpuLabelEvidencePassed),
    vendor,
    architecture,
    device,
    description,
    webglRenderer,
  };
}

function summarizeString(values: Array<string | null>): string | null {
  const unique = [...new Set(values.filter((value): value is string => Boolean(value)))];
  return unique.length === 1 ? unique[0] ?? null : unique.length > 1 ? "mixed" : null;
}

function sumNumber(values: Array<number | null>): number | null {
  const finite = values.filter((value): value is number => typeof value === "number" && Number.isFinite(value));
  return finite.length > 0 ? finite.reduce((total, value) => total + value, 0) : null;
}

function maxNumber(values: Array<number | null>): number | null {
  const finite = values.filter((value): value is number => typeof value === "number" && Number.isFinite(value));
  return finite.length > 0 ? Math.max(...finite) : null;
}

function minNumber(values: Array<number | null>): number | null {
  const finite = values.filter((value): value is number => typeof value === "number" && Number.isFinite(value));
  return finite.length > 0 ? Math.min(...finite) : null;
}

function meanNumber(values: Array<number | null>): number | null {
  const finite = values.filter((value): value is number => typeof value === "number" && Number.isFinite(value));
  if (finite.length === 0) return null;
  return finite.reduce((total, value) => total + value, 0) / finite.length;
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
  const report = await evaluateHostedBenchmarkProofFile(artifactPath, {
    expectedSourceGitSha: process.env.HOSTED_BENCHMARK_EXPECTED_GIT_SHA ?? process.env.GITHUB_SHA,
    requireSourceBound: process.env.HOSTED_BENCHMARK_REQUIRE_SOURCE_BOUND === "true",
  });
  await writeHostedBenchmarkProofArtifact(report);
  console.log(JSON.stringify(report, null, 2));
  if (!report.passed) process.exitCode = 1;
}
