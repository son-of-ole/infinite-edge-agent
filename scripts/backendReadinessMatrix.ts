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
import type { HostedBenchmarkProofReport } from "./hostedBenchmarkProof";

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
  hostedProfilePassed: boolean | null;
  readinessStatus: BackendReadinessStatus;
  proofSource: string;
  blockers: string[];
  proofRequirements: string[];
  hostedBenchmarkProofSourceGitSha: string | null;
  hostedBenchmarkExpectedSourceGitSha: string | null;
  hostedBenchmarkProofSourceBound: boolean | null;
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
  hostedBenchmarkProof?: HostedBenchmarkProofReport | null;
  requireHostedBenchmarkProof?: boolean;
} = {}): BackendReadinessMatrix {
  const hostedProfile = input.hostedProfile ?? evaluateHostedDeploymentProfile(process.env);
  const requireHostedBenchmarkProof = input.requireHostedBenchmarkProof === true;
  const hostedBenchmarkProof = input.hostedBenchmarkProof ?? null;
  const backends = BROWSER_BACKEND_REGISTRY.map((entry): BackendReadinessEntry => {
    if (entry.productionRole === "production_candidate") {
      const hostedProfileReady = entry.backendId === hostedProfile.profile.llmBackend && hostedProfile.passed;
      const hostedBenchmarkProofSourceBound = requireHostedBenchmarkProof
        ? Boolean(
          hostedBenchmarkProof?.expectedSourceGitSha
          && hostedBenchmarkProof.proof.sourceGitSha === hostedBenchmarkProof.expectedSourceGitSha,
        )
        : null;
      const hostedBenchmarkPassed = hostedBenchmarkProof?.passed === true
        && hostedBenchmarkProof.proof.runtimeBackendId === entry.backendId
        && hostedBenchmarkProof.proof.deployBackendId === entry.backendId;
      const hostedBenchmarkReady = !requireHostedBenchmarkProof
        || (hostedBenchmarkPassed && hostedBenchmarkProofSourceBound === true);
      const deployReady = hostedProfileReady && hostedBenchmarkReady;
      const proofSource = requireHostedBenchmarkProof
        ? "hosted_deployment_profile+hosted_benchmark_proof"
        : "hosted_deployment_profile";
      const proofRequirements = [
        "compiled_backend_selected",
        "hosted_profile_passed",
        "grounded_exact_benchmark_url",
        "durable_benchmark_telemetry",
        ...(requireHostedBenchmarkProof ? [
          "hosted_benchmark_artifact_passed",
          "hosted_benchmark_artifact_source_bound",
        ] : []),
      ];
      const blockers = deployReady ? [] : [
        ...(!hostedProfileReady ? [
          "Compiled production backend is not deploy-ready because hosted deployment profile failed.",
          ...hostedProfile.blockers,
        ] : []),
        ...(requireHostedBenchmarkProof && !hostedBenchmarkPassed ? [
          "Hosted benchmark proof is required to mark compiled-browser-webllm deploy-ready.",
          ...(hostedBenchmarkProof?.blockers ?? []),
        ] : []),
        ...(requireHostedBenchmarkProof && hostedBenchmarkPassed && hostedBenchmarkProofSourceBound !== true ? [
          "Hosted benchmark proof must be source-bound to the expected deployment commit.",
        ] : []),
      ];
      return {
        backendId: entry.backendId,
        label: entry.label,
        productionRole: entry.productionRole,
        deployDefault: entry.deployDefault,
        deployReady,
        hostedProfilePassed: hostedProfileReady,
        readinessStatus: deployReady ? "deploy_ready" : "blocked",
        proofSource,
        blockers,
        proofRequirements,
        hostedBenchmarkProofSourceGitSha: hostedBenchmarkProof?.proof.sourceGitSha ?? null,
        hostedBenchmarkExpectedSourceGitSha: hostedBenchmarkProof?.expectedSourceGitSha ?? null,
        hostedBenchmarkProofSourceBound: hostedBenchmarkProofSourceBound ?? null,
      };
    }
    if (entry.productionRole === "research_kernel_lab") {
      return {
        backendId: entry.backendId,
        label: entry.label,
        productionRole: entry.productionRole,
        deployDefault: entry.deployDefault,
        deployReady: false,
        hostedProfilePassed: null,
        readinessStatus: "research_only",
        proofSource: "kernel_lab_research_gates",
        blockers: [],
        proofRequirements: [
          "strict_webgpu",
          "decode_hot_path",
          "kernel_parity",
          "research_trace",
        ],
        hostedBenchmarkProofSourceGitSha: null,
        hostedBenchmarkExpectedSourceGitSha: null,
        hostedBenchmarkProofSourceBound: null,
      };
    }
    return {
      backendId: entry.backendId,
      label: entry.label,
      productionRole: entry.productionRole,
      deployDefault: entry.deployDefault,
      deployReady: false,
      hostedProfilePassed: null,
      readinessStatus: "fallback_only",
      proofSource: "bounded_fallback_contract",
      blockers: [],
      proofRequirements: [
        "task_bounds",
        "fallback_trace",
      ],
      hostedBenchmarkProofSourceGitSha: null,
      hostedBenchmarkExpectedSourceGitSha: null,
      hostedBenchmarkProofSourceBound: null,
    };
  });
  const deployBackend = backends.find((entry) => entry.productionRole === "production_candidate" && entry.deployReady);
  const blockers = deployBackend
    ? []
    : [
      requireHostedBenchmarkProof && hostedBenchmarkProof?.passed === true
        ? "Compiled production backend is not deploy-ready because hosted benchmark proof is required and missing, failed, or not source-bound."
        : requireHostedBenchmarkProof && (!hostedBenchmarkProof || hostedBenchmarkProof.passed !== true)
          ? "Compiled production backend is not deploy-ready because hosted benchmark proof is required and missing or failed."
        : "Compiled production backend is not deploy-ready because hosted deployment profile failed.",
    ];

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
  const fallbackBackendIds = matrix.backends
    .filter((entry) => entry.productionRole === "fallback")
    .map((entry) => entry.backendId);
  const fallbackDeployReadyCount = matrix.backends
    .filter((entry) => entry.productionRole === "fallback" && entry.deployReady)
    .length;
  const hostedCompiledBackend = matrix.backends.find((entry) => entry.backendId === "compiled-browser-webllm");
  const roleBoundaryPassed = deployReadyCount === 1
    && matrix.deployBackendId === "compiled-browser-webllm"
    && fallbackDeployReadyCount === 0
    && matrix.backends.every((entry) =>
      entry.productionRole === "production_candidate" || entry.deployReady === false);

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
      backendReadinessFallbackBackendCount: fallbackBackendIds.length,
      backendReadinessFallbackBackendId: fallbackBackendIds[0] ?? null,
      backendReadinessFallbackDeployReadyCount: fallbackDeployReadyCount,
      backendReadinessRoleBoundaryPassed: roleBoundaryPassed,
      backendReadinessCompiledHostedProfilePassed: hostedCompiledBackend?.hostedProfilePassed ?? false,
      backendReadinessCompiledDeployReady: hostedCompiledBackend?.deployReady ?? false,
      backendReadinessProofBoundToHostedBenchmark: isBackendReadinessProofBoundToHostedBenchmark(matrix),
      backendReadinessHostedBenchmarkProofSourceGitSha: hostedCompiledBackend?.hostedBenchmarkProofSourceGitSha ?? null,
      backendReadinessHostedBenchmarkExpectedSourceGitSha: hostedCompiledBackend?.hostedBenchmarkExpectedSourceGitSha ?? null,
      backendReadinessHostedBenchmarkProofSourceBound: hostedCompiledBackend?.hostedBenchmarkProofSourceBound ?? null,
    },
    matrix,
  };
}

export function isBackendReadinessProofBoundToHostedBenchmark(matrix: BackendReadinessMatrix): boolean {
  const deployBackend = matrix.backends.find((entry) =>
    entry.backendId === matrix.deployBackendId
    && entry.productionRole === "production_candidate"
    && entry.deployReady);
  return Boolean(
    deployBackend
    && deployBackend.proofSource.includes("hosted_benchmark_proof")
    && deployBackend.proofRequirements.includes("hosted_benchmark_artifact_passed")
    && deployBackend.proofRequirements.includes("hosted_benchmark_artifact_source_bound")
    && deployBackend.hostedBenchmarkProofSourceBound === true
  );
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
