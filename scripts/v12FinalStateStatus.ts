import { existsSync } from "node:fs";
import { mkdir, readFile, writeFile } from "node:fs/promises";
import { join } from "node:path";
import {
  evaluateRepositoryPublicationStatus,
  type RepositoryPublicationStatusReport,
} from "./repositoryPublicationStatus";
import {
  evaluateV12ReadinessBundle,
  type V12ReadinessBundle,
} from "./v12ReadinessBundle";
import type { V12ProductionArchiveArtifact } from "./v12ProductionArchive";

export type V12FinalStateNextAction =
  | "ready"
  | "fix_v12_architecture_readiness"
  | "publish_source_history"
  | "run_hosted_production_proof"
  | "fix_backend_specific_production_evidence";

export interface V12FinalStateRequirement {
  id: string;
  label: string;
  passed: boolean;
  evidence: string;
  blockers: string[];
}

export interface V12FinalStateStatus {
  passed: boolean;
  blockers: string[];
  nextAction: V12FinalStateNextAction;
  deployBackendId: string | null;
  kernelLabBackendId: string | null;
  fallbackBackendId: string | null;
  requirements: V12FinalStateRequirement[];
  summary: Record<string, number | string | boolean | null>;
  readinessBundle: V12ReadinessBundle;
  repositoryPublication: RepositoryPublicationStatusReport;
  productionArchive: V12ProductionArchiveArtifact | null;
}

export interface V12FinalStateStatusArtifact {
  name: "v12-final-state-status";
  createdAt: string;
  passed: boolean;
  summary: Record<string, number | string | boolean | null>;
  status: V12FinalStateStatus;
}

export interface V12FinalStateStatusWriteResult {
  artifact: V12FinalStateStatusArtifact;
  latestPath: string;
  resultPath: string;
}

export function evaluateV12FinalStateStatus(input: {
  readinessBundle: V12ReadinessBundle;
  repositoryPublication: RepositoryPublicationStatusReport;
  productionArchive?: V12ProductionArchiveArtifact | null;
}): V12FinalStateStatus {
  const readiness = input.readinessBundle;
  const publication = input.repositoryPublication;
  const archive = input.productionArchive ?? null;
  const archiveSummary = archive?.summary ?? {};
  const deployBackendId = readiness.deployBackendId;
  const kernelLabBackendId = readiness.kernelLabBackendId;
  const fallbackBackendId = readiness.fallbackBackendId;

  const architectureReady = readiness.passed === true
    && readiness.backendRoleBoundaryPassed === true
    && deployBackendId === "compiled-browser-webllm"
    && kernelLabBackendId === "unlocked-browser-transformer"
    && fallbackBackendId === "wasm-small-core";
  const sourcePublished = publication.published === true;
  const exactHistoryHandoffReady = publication.published === true || publication.bundleHandoffReady === true;
  const sourceBoundHostedArchiveReady = archive?.passed === true
    && archiveSummary.v12ProductionArchivePassed === true
    && archiveSummary.v12ProductionHostedBenchmarkProofPassed === true
    && archiveSummary.v12ProductionProofSourceBoundRequired === true
    && archiveSummary.v12ProductionProofSourceBound === true;
  const backendSpecificProductionEvidenceReady = sourceBoundHostedArchiveReady
    && archiveSummary.v12ProductionDeployBackendId === "compiled-browser-webllm"
    && archiveSummary.v12ProductionKernelLabBackendId === "unlocked-browser-transformer"
    && archiveSummary.v12ProductionFallbackBackendId === "wasm-small-core"
    && archiveSummary.v12ProductionBackendRoleBoundaryPassed === true
    && archiveSummary.v12ProductionCompiledBackendReadyPassed === true
    && archiveSummary.v12ProductionDeployReadyPassed === true
    && archiveSummary.v12ProductionMemoryGroundingPassed === true
    && archiveSummary.v12ProductionExpectedExactPassed === true
    && archiveSummary.v12ProductionSpeedFloorPassed === true
    && archiveSummary.v12ProductionStrictWebGpuPassed === true
    && archiveSummary.v12ProductionCpuFallbackUsed === false
    && archiveSummary.v12ProductionBackendBrokerSelectionPassed === true
    && archiveSummary.v12ProductionBrokerRoleBoundaryPassed === true;

  const requirements: V12FinalStateRequirement[] = [
    {
      id: "v12_architecture_readiness",
      label: "Backend Broker, compiled deploy backend, Kernel Lab, fallback, and shared runtime boundaries are proven.",
      passed: architectureReady,
      evidence: "v12-readiness-bundle",
      blockers: architectureReady ? [] : ["V12 readiness bundle has not proven the deploy, Kernel Lab, fallback, and shared runtime boundary."],
    },
    {
      id: "source_published_to_github",
      label: "The exact current source history is published to origin/main.",
      passed: sourcePublished,
      evidence: "repository-publication-status",
      blockers: sourcePublished
        ? []
        : [exactHistoryHandoffReady
          ? "Source history is only bundle-handoff ready; publish main to origin/main before claiming final state."
          : "Source history is not published and no verified exact-history handoff is ready."],
    },
    {
      id: "source_bound_hosted_production_archive",
      label: "A real hosted Chrome/Edge production proof is archived and source-bound to the deployed commit.",
      passed: sourceBoundHostedArchiveReady,
      evidence: "v12-production-archive",
      blockers: sourceBoundHostedArchiveReady
        ? []
        : [archive
          ? "V12 production archive exists but is not passing, hosted-proof-backed, and source-bound."
          : "Missing v12 production archive with hosted benchmark proof."],
    },
    {
      id: "backend_specific_production_evidence",
      label: "The hosted proof shows compiled-browser-webllm as deploy backend and unlocked-browser-transformer as Kernel Lab.",
      passed: backendSpecificProductionEvidenceReady,
      evidence: "v12-production-archive",
      blockers: backendSpecificProductionEvidenceReady
        ? []
        : ["Hosted production evidence does not yet prove compiled deploy readiness, Kernel Lab separation, quality, memory grounding, speed floor, and no CPU fallback together."],
    },
  ];
  const blockers = requirements.flatMap((requirement) =>
    requirement.passed ? [] : requirement.blockers.map((blocker) => `${requirement.id}: ${blocker}`),
  );
  const passed = blockers.length === 0;
  const nextAction = chooseNextAction({
    architectureReady,
    sourcePublished,
    sourceBoundHostedArchiveReady,
    backendSpecificProductionEvidenceReady,
  });

  return {
    passed,
    blockers,
    nextAction,
    deployBackendId,
    kernelLabBackendId,
    fallbackBackendId,
    requirements,
    summary: {
      v12FinalStatePassed: passed,
      v12FinalStateBlockerCount: blockers.length,
      v12FinalStateNextAction: nextAction,
      v12FinalStateDeployBackendId: deployBackendId,
      v12FinalStateKernelLabBackendId: kernelLabBackendId,
      v12FinalStateFallbackBackendId: fallbackBackendId,
      v12FinalStateArchitectureReady: architectureReady,
      v12FinalStateSourcePublished: sourcePublished,
      v12FinalStateExactHistoryHandoffReady: exactHistoryHandoffReady,
      v12FinalStateHostedProductionArchivePassed: archive?.passed ?? false,
      v12FinalStateHostedBenchmarkProofPassed: archiveSummary.v12ProductionHostedBenchmarkProofPassed ?? false,
      v12FinalStateHostedProofSourceBound: archiveSummary.v12ProductionProofSourceBound ?? false,
      v12FinalStateBackendSpecificProductionEvidenceReady: backendSpecificProductionEvidenceReady,
      v12FinalStateRequirementCount: requirements.length,
      v12FinalStatePassedRequirementCount: requirements.filter((requirement) => requirement.passed).length,
      v12FinalStateRepositoryHeadSha: publication.snapshot.headSha,
      v12FinalStateRepositoryAheadCount: publication.snapshot.aheadCount,
      v12FinalStateRepositoryBehindCount: publication.snapshot.behindCount,
      v12FinalStateRepositoryDirty: publication.snapshot.dirty,
      v12FinalStateProductionProofSourceGitSha: archiveSummary.v12ProductionProofSourceGitSha ?? null,
      v12FinalStateProductionExpectedSourceGitSha: archiveSummary.v12ProductionExpectedSourceGitSha ?? null,
      v12FinalStateProductionDeployUrl: archiveSummary.v12ProductionHostedBenchmarkDeployUrl ?? null,
      v12FinalStateProductionMeanTokensPerSecond: archiveSummary.v12ProductionMeanTokensPerSecond ?? null,
    },
    readinessBundle: readiness,
    repositoryPublication: publication,
    productionArchive: archive,
  };
}

export function buildV12FinalStateStatusArtifact(
  status: V12FinalStateStatus,
  createdAt = new Date().toISOString(),
): V12FinalStateStatusArtifact {
  return {
    name: "v12-final-state-status",
    createdAt,
    passed: status.passed,
    summary: status.summary,
    status,
  };
}

export async function writeV12FinalStateStatusArtifact(
  status: V12FinalStateStatus,
  options: { artifactDir?: string; createdAt?: string } = {},
): Promise<V12FinalStateStatusWriteResult> {
  const artifactDir = options.artifactDir ?? process.env.EVAL_ARTIFACT_DIR ?? ".artifacts/evals";
  const artifact = buildV12FinalStateStatusArtifact(status, options.createdAt);
  const runDir = join(artifactDir, "v12-final-state-status");
  const timestamp = artifact.createdAt.replace(/[:.]/g, "-");
  const latestPath = join(artifactDir, "v12-final-state-status-latest.json");
  const resultPath = join(runDir, `${timestamp}.json`);
  const json = `${JSON.stringify(artifact, null, 2)}\n`;

  await mkdir(runDir, { recursive: true });
  await writeFile(resultPath, json);
  await writeFile(latestPath, json);

  return { artifact, latestPath, resultPath };
}

export async function evaluateCurrentV12FinalStateStatus(options: {
  artifactDir?: string;
  productionArchivePath?: string;
} = {}): Promise<V12FinalStateStatus> {
  const artifactDir = options.artifactDir ?? process.env.EVAL_ARTIFACT_DIR ?? ".artifacts/evals";
  const productionArchivePath = options.productionArchivePath
    ?? process.env.V12_FINAL_STATE_PRODUCTION_ARCHIVE_PATH
    ?? join(artifactDir, "v12-production-archive-latest.json");
  const productionArchive = await readOptionalProductionArchive(productionArchivePath);
  return evaluateV12FinalStateStatus({
    readinessBundle: evaluateV12ReadinessBundle(),
    repositoryPublication: await evaluateRepositoryPublicationStatus(),
    productionArchive,
  });
}

function chooseNextAction(input: {
  architectureReady: boolean;
  sourcePublished: boolean;
  sourceBoundHostedArchiveReady: boolean;
  backendSpecificProductionEvidenceReady: boolean;
}): V12FinalStateNextAction {
  if (!input.architectureReady) return "fix_v12_architecture_readiness";
  if (!input.sourcePublished) return "publish_source_history";
  if (!input.sourceBoundHostedArchiveReady) return "run_hosted_production_proof";
  if (!input.backendSpecificProductionEvidenceReady) return "fix_backend_specific_production_evidence";
  return "ready";
}

async function readOptionalProductionArchive(path: string): Promise<V12ProductionArchiveArtifact | null> {
  if (!existsSync(path)) return null;
  const parsed = JSON.parse(await readFile(path, "utf8")) as V12ProductionArchiveArtifact;
  return parsed.name === "v12-production-archive" ? parsed : null;
}

if (import.meta.url === `file://${process.argv[1]}`) {
  const artifactDir = process.env.EVAL_ARTIFACT_DIR ?? ".artifacts/evals";
  const status = await evaluateCurrentV12FinalStateStatus({ artifactDir });
  await writeV12FinalStateStatusArtifact(status, { artifactDir });
  console.log(JSON.stringify(status, null, 2));
  if (!status.passed) process.exitCode = 1;
}
