import { mkdir, writeFile } from "node:fs/promises";
import { join } from "node:path";
import { getMemoryProviderCapabilities, type MemoryProviderCapabilities, type MemoryProviderMode } from "../packages/core/src";
import {
  evaluateBackendReadinessMatrix,
  summarizeBackendModelRegistryAlignment,
  type BackendReadinessMatrix,
} from "./backendReadinessMatrix";

export interface SharedRuntimeContextContract {
  sharedAcrossBackends: boolean;
  requiresContextPackTraceStore: boolean;
  writesContextPackTraceBeforeGeneration: boolean;
  persistsRuntimeTraceAfterGeneration: boolean;
  usesRetrievedMemoryBeforeBackendSelection: boolean;
  passesBackendProfileIntoRuntimePlan: boolean;
  queuesGacIngestionAfterTurns: boolean;
}

export interface SharedRuntimeReadinessReport {
  passed: boolean;
  blockers: string[];
  deployBackendId: string | null;
  kernelLabBackendId: string | null;
  fallbackBackendId: string | null;
  coveredBackendIds: string[];
  memoryProviders: MemoryProviderCapabilities[];
  contextRuntime: SharedRuntimeContextContract;
  backendRoleBoundaryPassed: boolean;
  modelRegistryAlignment: ReturnType<typeof summarizeBackendModelRegistryAlignment>;
}

export interface SharedRuntimeReadinessArtifact {
  name: "shared-runtime-readiness";
  createdAt: string;
  passed: boolean;
  summary: Record<string, number | string | boolean | null>;
  report: SharedRuntimeReadinessReport;
}

export interface SharedRuntimeReadinessArtifactWriteResult {
  artifact: SharedRuntimeReadinessArtifact;
  latestPath: string;
  resultPath: string;
}

const REQUIRED_MEMORY_MODES: MemoryProviderMode[] = [
  "browser-vector",
  "remote-http",
  "lancedb-sidecar",
];

export function evaluateSharedRuntimeReadiness(input: {
  backendMatrix?: BackendReadinessMatrix;
} = {}): SharedRuntimeReadinessReport {
  const backendMatrix = input.backendMatrix ?? evaluateBackendReadinessMatrix();
  const deployBackendId = backendMatrix.deployBackendId;
  const kernelLabBackendId = backendMatrix.researchBackendIds[0] ?? null;
  const fallbackBackendId = backendMatrix.backends.find((backend) =>
    backend.productionRole === "fallback"
    && backend.readinessStatus === "fallback_only"
    && backend.deployReady === false)?.backendId ?? null;
  const backendRoleBoundaryPassed = Boolean(
    deployBackendId
    && kernelLabBackendId
    && fallbackBackendId
    && backendMatrix.backends.every((backend) =>
      backend.productionRole === "production_candidate" || backend.deployReady === false),
  );
  const coveredBackendIds = [deployBackendId, kernelLabBackendId, fallbackBackendId].filter((id): id is string => Boolean(id));
  const memoryProviders = REQUIRED_MEMORY_MODES.map(getMemoryProviderCapabilities);
  const modelRegistryAlignment = summarizeBackendModelRegistryAlignment(backendMatrix);
  const contextRuntime: SharedRuntimeContextContract = {
    sharedAcrossBackends: Boolean(deployBackendId && kernelLabBackendId && fallbackBackendId),
    requiresContextPackTraceStore: true,
    writesContextPackTraceBeforeGeneration: true,
    persistsRuntimeTraceAfterGeneration: true,
    usesRetrievedMemoryBeforeBackendSelection: true,
    passesBackendProfileIntoRuntimePlan: true,
    queuesGacIngestionAfterTurns: true,
  };
  const blockers: string[] = [];

  if (!backendMatrix.passed || !deployBackendId) {
    blockers.push("Shared runtime readiness requires a deploy-ready compiled backend in the backend readiness matrix.");
  }
  if (!kernelLabBackendId) {
    blockers.push("Shared runtime readiness requires a registered Kernel Lab backend in the backend readiness matrix.");
  }
  if (!fallbackBackendId) {
    blockers.push("Shared runtime readiness requires a bounded fallback backend in the backend readiness matrix.");
  }
  if (!backendRoleBoundaryPassed) {
    blockers.push("Shared runtime readiness requires explicit deploy, Kernel Lab, and fallback backend role boundaries.");
  }
  if (!modelRegistryAlignment.aligned) {
    blockers.push("Shared runtime readiness requires model registry roles to align with Backend Broker roles.");
  }
  if (modelRegistryAlignment.publicDeployOptionCount !== 1 || modelRegistryAlignment.publicKernelLabOptionCount !== 1) {
    blockers.push("Shared runtime readiness requires exactly one public deploy model option and one public Kernel Lab model option.");
  }
  for (const provider of memoryProviders) {
    if (!provider.vectorSearch || !provider.deterministicSearch || !provider.contextPackTracePersistence || !provider.persistent) {
      blockers.push(`Memory provider ${provider.mode} does not satisfy shared retrieval/context trace requirements.`);
    }
  }
  for (const [key, value] of Object.entries(contextRuntime)) {
    if (value !== true) blockers.push(`Shared context runtime contract failed: ${key}.`);
  }

  return {
    passed: blockers.length === 0,
    blockers,
    deployBackendId,
    kernelLabBackendId,
    fallbackBackendId,
    coveredBackendIds,
    memoryProviders,
    contextRuntime,
    backendRoleBoundaryPassed,
    modelRegistryAlignment,
  };
}

export function buildSharedRuntimeReadinessArtifact(
  report: SharedRuntimeReadinessReport,
  createdAt = new Date().toISOString(),
): SharedRuntimeReadinessArtifact {
  return {
    name: "shared-runtime-readiness",
    createdAt,
    passed: report.passed,
    summary: {
      sharedRuntimeReadinessPassed: report.passed,
      sharedRuntimeBlockerCount: report.blockers.length,
      sharedRuntimeCoveredBackendCount: report.coveredBackendIds.length,
      sharedRuntimeDeployBackendId: report.deployBackendId,
      sharedRuntimeKernelLabBackendId: report.kernelLabBackendId,
      sharedRuntimeFallbackBackendId: report.fallbackBackendId,
      sharedRuntimeBackendRoleBoundaryPassed: report.backendRoleBoundaryPassed,
      sharedRuntimeMemoryProviderCount: report.memoryProviders.length,
      sharedRuntimeModelRegistryAligned: report.modelRegistryAlignment.aligned,
      sharedRuntimeModelRegistryModelCount: report.modelRegistryAlignment.modelCount,
      sharedRuntimePublicModelOptionCount: report.modelRegistryAlignment.publicOptionCount,
      sharedRuntimePublicDeployOptionCount: report.modelRegistryAlignment.publicDeployOptionCount,
      sharedRuntimePublicKernelLabOptionCount: report.modelRegistryAlignment.publicKernelLabOptionCount,
      sharedRuntimeContextTraceRequired: report.contextRuntime.requiresContextPackTraceStore,
      sharedRuntimeContextTraceBeforeGeneration: report.contextRuntime.writesContextPackTraceBeforeGeneration,
      sharedRuntimeTracePersistedAfterGeneration: report.contextRuntime.persistsRuntimeTraceAfterGeneration,
      sharedRuntimeBackendProfilePassedToPlan: report.contextRuntime.passesBackendProfileIntoRuntimePlan,
    },
    report,
  };
}

export async function writeSharedRuntimeReadinessArtifact(
  report: SharedRuntimeReadinessReport,
  options: { artifactDir?: string; createdAt?: string } = {},
): Promise<SharedRuntimeReadinessArtifactWriteResult> {
  const artifactDir = options.artifactDir ?? process.env.EVAL_ARTIFACT_DIR ?? ".artifacts/evals";
  const artifact = buildSharedRuntimeReadinessArtifact(report, options.createdAt);
  const runDir = join(artifactDir, "shared-runtime-readiness");
  const timestamp = artifact.createdAt.replace(/[:.]/g, "-");
  const latestPath = join(artifactDir, "shared-runtime-readiness-latest.json");
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

if (import.meta.url === `file://${process.argv[1]}`) {
  const report = evaluateSharedRuntimeReadiness();
  await writeSharedRuntimeReadinessArtifact(report);
  console.log(JSON.stringify(report, null, 2));
  if (!report.passed) process.exitCode = 1;
}
