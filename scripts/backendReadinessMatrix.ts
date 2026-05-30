import { mkdir, writeFile } from "node:fs/promises";
import { join } from "node:path";
import {
  BROWSER_BACKEND_REGISTRY,
  type BrowserBackendProductionRole,
} from "../apps/web/src/lib/runtime/backendBroker";
import {
  evaluateHostedDeploymentProfile,
  type HostedDeploymentProfileReport,
} from "./hostedDeploymentProfile";

export type BackendReadinessStatus =
  | "deploy_ready"
  | "blocked"
  | "research_only"
  | "fallback_only";

export interface BackendReadinessEntry {
  backendId: string;
  label: string;
  productionRole: BrowserBackendProductionRole;
  deployDefault: boolean;
  deployReady: boolean;
  readinessStatus: BackendReadinessStatus;
  proofSource: string;
  blockers: string[];
  proofRequirements: string[];
}

export interface BackendReadinessMatrix {
  passed: boolean;
  blockers: string[];
  deployBackendId: string | null;
  researchBackendIds: string[];
  backends: BackendReadinessEntry[];
}

export interface BackendReadinessMatrixArtifact {
  name: "backend-readiness-matrix";
  createdAt: string;
  passed: boolean;
  summary: Record<string, number | string | boolean | null>;
  matrix: BackendReadinessMatrix;
}

export interface BackendReadinessMatrixArtifactWriteResult {
  artifact: BackendReadinessMatrixArtifact;
  latestPath: string;
  resultPath: string;
}

export function evaluateBackendReadinessMatrix(input: {
  hostedProfile?: HostedDeploymentProfileReport;
} = {}): BackendReadinessMatrix {
  const hostedProfile = input.hostedProfile ?? evaluateHostedDeploymentProfile(process.env);
  const backends = BROWSER_BACKEND_REGISTRY.map((entry): BackendReadinessEntry => {
    if (entry.productionRole === "production_candidate") {
      const deployReady = entry.backendId === hostedProfile.profile.llmBackend && hostedProfile.passed;
      return {
        backendId: entry.backendId,
        label: entry.label,
        productionRole: entry.productionRole,
        deployDefault: entry.deployDefault,
        deployReady,
        readinessStatus: deployReady ? "deploy_ready" : "blocked",
        proofSource: "hosted_deployment_profile",
        blockers: deployReady ? [] : [
          "Compiled production backend is not deploy-ready because hosted deployment profile failed.",
          ...hostedProfile.blockers,
        ],
        proofRequirements: [
          "compiled_backend_selected",
          "hosted_profile_passed",
          "grounded_exact_benchmark_url",
          "durable_benchmark_telemetry",
        ],
      };
    }
    if (entry.productionRole === "research_kernel_lab") {
      return {
        backendId: entry.backendId,
        label: entry.label,
        productionRole: entry.productionRole,
        deployDefault: entry.deployDefault,
        deployReady: false,
        readinessStatus: "research_only",
        proofSource: "kernel_lab_research_gates",
        blockers: [],
        proofRequirements: [
          "strict_webgpu",
          "decode_hot_path",
          "kernel_parity",
          "research_trace",
        ],
      };
    }
    return {
      backendId: entry.backendId,
      label: entry.label,
      productionRole: entry.productionRole,
      deployDefault: entry.deployDefault,
      deployReady: false,
      readinessStatus: "fallback_only",
      proofSource: "bounded_fallback_contract",
      blockers: [],
      proofRequirements: [
        "task_bounds",
        "fallback_trace",
      ],
    };
  });
  const deployBackend = backends.find((entry) => entry.productionRole === "production_candidate" && entry.deployReady);
  const blockers = deployBackend
    ? []
    : ["Compiled production backend is not deploy-ready because hosted deployment profile failed."];

  return {
    passed: blockers.length === 0,
    blockers,
    deployBackendId: deployBackend?.backendId ?? null,
    researchBackendIds: backends
      .filter((entry) => entry.productionRole === "research_kernel_lab")
      .map((entry) => entry.backendId),
    backends,
  };
}

export function buildBackendReadinessMatrixArtifact(
  matrix: BackendReadinessMatrix,
  createdAt = new Date().toISOString(),
): BackendReadinessMatrixArtifact {
  const productionCandidateCount = matrix.backends.filter((entry) => entry.productionRole === "production_candidate").length;
  const deployReadyCount = matrix.backends.filter((entry) => entry.deployReady).length;
  const researchBackendIds = matrix.backends
    .filter((entry) => entry.productionRole === "research_kernel_lab")
    .map((entry) => entry.backendId);
  const hostedCompiledBackend = matrix.backends.find((entry) => entry.backendId === "compiled-browser-webllm");

  return {
    name: "backend-readiness-matrix",
    createdAt,
    passed: matrix.passed,
    summary: {
      backendReadinessMatrixPassed: matrix.passed,
      backendReadinessBlockerCount: matrix.blockers.length,
      backendReadinessBackendCount: matrix.backends.length,
      backendReadinessDeployBackendId: matrix.deployBackendId,
      backendReadinessProductionCandidateCount: productionCandidateCount,
      backendReadinessDeployReadyCount: deployReadyCount,
      backendReadinessResearchBackendCount: researchBackendIds.length,
      backendReadinessKernelLabBackendId: researchBackendIds[0] ?? null,
      backendReadinessCompiledHostedProfilePassed: hostedCompiledBackend?.deployReady ?? false,
    },
    matrix,
  };
}

export async function writeBackendReadinessMatrixArtifact(
  matrix: BackendReadinessMatrix,
  options: { artifactDir?: string; createdAt?: string } = {},
): Promise<BackendReadinessMatrixArtifactWriteResult> {
  const artifactDir = options.artifactDir ?? process.env.EVAL_ARTIFACT_DIR ?? ".artifacts/evals";
  const artifact = buildBackendReadinessMatrixArtifact(matrix, options.createdAt);
  const runDir = join(artifactDir, "backend-readiness-matrix");
  const timestamp = artifact.createdAt.replace(/[:.]/g, "-");
  const latestPath = join(artifactDir, "backend-readiness-matrix-latest.json");
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
  const matrix = evaluateBackendReadinessMatrix();
  await writeBackendReadinessMatrixArtifact(matrix);
  console.log(JSON.stringify(matrix, null, 2));
  if (!matrix.passed) process.exitCode = 1;
}
