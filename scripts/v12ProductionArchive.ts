import { mkdir, writeFile } from "node:fs/promises";
import { join } from "node:path";
import {
  runV12ReadinessSuite,
  type V12ReadinessSuiteRunResult,
} from "./v12ReadinessSuite";
import {
  isBackendReadinessProofBoundToHostedBenchmark,
  summarizeBackendModelRegistryAlignment,
} from "./backendReadinessMatrix";
import type { HostedDeploymentProfileEnv } from "./hostedDeploymentProfile";

export interface V12ProductionArchive {
  passed: boolean;
  blockers: string[];
  suiteLatestPath: string;
  suiteResultPath: string;
  suiteResult: V12ReadinessSuiteRunResult;
}

export interface V12ProductionArchiveArtifact {
  name: "v12-production-archive";
  createdAt: string;
  passed: boolean;
  summary: Record<string, number | string | boolean | null>;
  archive: V12ProductionArchive;
}

export interface V12ProductionArchiveRunResult {
  archive: V12ProductionArchiveArtifact;
  latestPath: string;
  resultPath: string;
}

export async function runV12ProductionArchive(options: {
  env?: HostedDeploymentProfileEnv;
  artifactDir?: string;
  createdAt?: string;
  hostedBenchmarkArtifactPath?: string;
} = {}): Promise<V12ProductionArchiveRunResult> {
  const artifactDir = options.artifactDir ?? process.env.EVAL_ARTIFACT_DIR ?? ".artifacts/evals";
  const createdAt = options.createdAt ?? new Date().toISOString();
  const env = {
    ...(options.env ?? process.env),
    RELEASE_REQUIRE_HOSTED_BENCHMARK_PROOF: "true",
  };
  const suiteResult = await runV12ReadinessSuite({
    env,
    artifactDir,
    createdAt,
    ...(options.hostedBenchmarkArtifactPath ? { hostedBenchmarkArtifactPath: options.hostedBenchmarkArtifactPath } : {}),
  });
  const archive = buildV12ProductionArchiveArtifact({
    suiteResult,
    createdAt,
  });
  const written = await writeV12ProductionArchiveArtifact(archive, { artifactDir });

  return {
    archive,
    latestPath: written.latestPath,
    resultPath: written.resultPath,
  };
}

export function buildV12ProductionArchiveArtifact(input: {
  suiteResult: V12ReadinessSuiteRunResult;
  createdAt?: string;
}): V12ProductionArchiveArtifact {
  const suite = input.suiteResult.suite;
  const blockers = suite.blockers;
  const artifactCount = suite.totalArtifactCount + 1;
  const hostedProof = suite.hostedBenchmarkProof?.proof;
  const expectedSourceGitSha = suite.hostedBenchmarkProof?.expectedSourceGitSha ?? null;
  const proofSourceBoundRequired = suite.hostedBenchmarkProof?.sourceBoundRequired ?? false;
  const proofSourceGitSha = hostedProof?.sourceGitSha ?? null;
  const proofSourceBound = expectedSourceGitSha
    ? proofSourceGitSha === expectedSourceGitSha
    : false;
  const modelRegistry = summarizeBackendModelRegistryAlignment(suite.backendMatrix);

  return {
    name: "v12-production-archive",
    createdAt: input.createdAt ?? new Date().toISOString(),
    passed: suite.passed,
    summary: {
      v12ProductionArchivePassed: suite.passed,
      v12ProductionBlockerCount: blockers.length,
      v12ProductionSuitePassed: suite.passed,
      v12ProductionDeployBackendId: suite.deployBackendId,
      v12ProductionKernelLabBackendId: suite.kernelLabBackendId,
      v12ProductionFallbackBackendId: suite.v12Bundle.fallbackBackendId,
      v12ProductionBackendRoleBoundaryPassed: suite.v12Bundle.backendRoleBoundaryPassed,
      v12ProductionHostedBenchmarkProofRequired: suite.hostedBenchmarkProofRequired,
      v12ProductionHostedBenchmarkProofPassed: suite.hostedBenchmarkProofPassed,
      v12ProductionBackendReadinessProofBound: isBackendReadinessProofBoundToHostedBenchmark(suite.backendMatrix),
      v12ProductionModelRegistryAligned: modelRegistry.aligned,
      v12ProductionModelRegistryModelCount: modelRegistry.modelCount,
      v12ProductionPublicModelOptionCount: modelRegistry.publicOptionCount,
      v12ProductionPublicDeployOptionCount: modelRegistry.publicDeployOptionCount,
      v12ProductionPublicKernelLabOptionCount: modelRegistry.publicKernelLabOptionCount,
      v12ProductionProofSchemaVersion: hostedProof?.v12ProductionProofSchemaVersion ?? null,
      v12ProductionProofSourceGitSha: proofSourceGitSha,
      v12ProductionExpectedSourceGitSha: expectedSourceGitSha,
      v12ProductionProofSourceBoundRequired: proofSourceBoundRequired,
      v12ProductionProofSourceBound: proofSourceBound,
      v12ProductionArtifactCount: artifactCount,
      v12ProductionSuiteArtifactCount: suite.totalArtifactCount,
      v12ProductionChildArtifactCount: suite.childArtifactCount,
      v12ProductionHostedBenchmarkRuntimeBackendId: hostedProof?.runtimeBackendId ?? null,
      v12ProductionHostedBenchmarkDeployBackendId: hostedProof?.deployBackendId ?? null,
      v12ProductionCompiledBackendReadyPassed: hostedProof?.compiledBackendReadyPassed ?? false,
      v12ProductionDeployReadyPassed: hostedProof?.productionDeployReadyPassed ?? false,
      v12ProductionMemoryGroundingPassed: hostedProof?.memoryGroundingPassed ?? false,
      v12ProductionConcreteMemoryGroundingPassed: hostedProof?.concreteMemoryGroundingPassed ?? false,
      v12ProductionMemoryGroundingRunCount: hostedProof?.memoryGroundingRunCount ?? 0,
      v12ProductionMemoryGroundingCaseId: hostedProof?.memoryGroundingCaseId ?? null,
      v12ProductionMemorySeededCorpusCount: hostedProof?.memorySeededCorpusCount ?? null,
      v12ProductionMemoryRetrievedCount: hostedProof?.memoryRetrievedCount ?? null,
      v12ProductionMemoryIncludedCount: hostedProof?.memoryIncludedCount ?? null,
      v12ProductionMemoryExpectedMemoryIdCount: hostedProof?.memoryExpectedMemoryIdCount ?? null,
      v12ProductionMemoryExpectedHitMeanRank: hostedProof?.memoryExpectedHitMeanRank ?? null,
      v12ProductionMemoryExpectedHitMinTopScoreMargin: hostedProof?.memoryExpectedHitMinTopScoreMargin ?? null,
      v12ProductionExpectedExactPassed: hostedProof?.expectedExactPassed ?? false,
      v12ProductionSpeedFloorPassed: hostedProof?.productionSpeedFloorPassed ?? false,
      v12ProductionMeanTokensPerSecond: hostedProof?.meanTokensPerSecond ?? null,
      v12ProductionDirectModelFactualProofUsed: hostedProof?.directModelFactualProofUsed ?? null,
      v12ProductionTechnicalProofOnly: hostedProof?.technicalProofOnly ?? null,
      v12ProductionCpuFallbackUsed: hostedProof?.cpuFallbackUsed ?? null,
      v12ProductionStrictWebGpuPassed: hostedProof?.strictWebGpuPassed ?? false,
      v12ProductionBackendBrokerSelectionPassed: hostedProof?.backendBrokerSelectionPassed ?? false,
      v12ProductionBackendBrokerTraceCount: hostedProof?.backendBrokerTraceCount ?? 0,
      v12ProductionBrokerSelectedBackendId: hostedProof?.brokerSelectedBackendId ?? null,
      v12ProductionBrokerSelectedModelId: hostedProof?.brokerSelectedModelId ?? null,
      v12ProductionBrokerProductionRole: hostedProof?.brokerProductionRole ?? null,
      v12ProductionBrokerDeployReadyCandidate: hostedProof?.brokerDeployReadyCandidate ?? false,
      v12ProductionBrokerReason: hostedProof?.brokerReason ?? null,
      v12ProductionBrokerDeployBackendId: hostedProof?.brokerDeployBackendId ?? null,
      v12ProductionBrokerKernelLabBackendId: hostedProof?.brokerKernelLabBackendId ?? null,
      v12ProductionBrokerFallbackBackendId: hostedProof?.brokerFallbackBackendId ?? null,
      v12ProductionBrokerFallbackBackendCount: hostedProof?.brokerFallbackBackendCount ?? null,
      v12ProductionBrokerFallbackDeployReadyCandidate: hostedProof?.brokerFallbackDeployReadyCandidate ?? false,
      v12ProductionBrokerRoleBoundaryPassed: hostedProof?.brokerRoleBoundaryPassed ?? false,
    },
    archive: {
      passed: suite.passed,
      blockers,
      suiteLatestPath: input.suiteResult.latestPath,
      suiteResultPath: input.suiteResult.resultPath,
      suiteResult: input.suiteResult,
    },
  };
}

async function writeV12ProductionArchiveArtifact(
  artifact: V12ProductionArchiveArtifact,
  options: { artifactDir?: string } = {},
): Promise<{ latestPath: string; resultPath: string }> {
  const artifactDir = options.artifactDir ?? process.env.EVAL_ARTIFACT_DIR ?? ".artifacts/evals";
  const runDir = join(artifactDir, "v12-production-archive");
  const timestamp = artifact.createdAt.replace(/[:.]/g, "-");
  const latestPath = join(artifactDir, "v12-production-archive-latest.json");
  const resultPath = join(runDir, `${timestamp}.json`);
  const json = `${JSON.stringify(artifact, null, 2)}\n`;

  await mkdir(runDir, { recursive: true });
  await writeFile(resultPath, json);
  await writeFile(latestPath, json);

  return { latestPath, resultPath };
}

if (import.meta.url === `file://${process.argv[1]}`) {
  const result = await runV12ProductionArchive();
  console.log(JSON.stringify(result.archive.archive, null, 2));
  if (!result.archive.passed) process.exitCode = 1;
}
