import { mkdir, writeFile } from "node:fs/promises";
import { join } from "node:path";
import {
  runV12ReadinessSuite,
  type V12ReadinessSuiteRunResult,
} from "./v12ReadinessSuite";
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
      v12ProductionHostedBenchmarkProofRequired: suite.hostedBenchmarkProofRequired,
      v12ProductionHostedBenchmarkProofPassed: suite.hostedBenchmarkProofPassed,
      v12ProductionArtifactCount: artifactCount,
      v12ProductionSuiteArtifactCount: suite.totalArtifactCount,
      v12ProductionChildArtifactCount: suite.childArtifactCount,
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
