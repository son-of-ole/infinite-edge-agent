import { mkdir, writeFile } from "node:fs/promises";
import { join } from "node:path";
import {
  evaluateBackendReadinessMatrix,
  writeBackendReadinessMatrixArtifact,
  type BackendReadinessMatrix,
  type BackendReadinessMatrixArtifactWriteResult,
} from "./backendReadinessMatrix";
import {
  evaluateHostedBenchmarkProofFile,
  writeHostedBenchmarkProofArtifact,
  type HostedBenchmarkProofArtifactWriteResult,
  type HostedBenchmarkProofReport,
} from "./hostedBenchmarkProof";
import {
  evaluateHostedDeploymentProfile,
  writeHostedDeploymentProfileArtifact,
  type HostedDeploymentProfileEnv,
  type HostedDeploymentProfileReport,
  type HostedDeploymentProfileArtifactWriteResult,
} from "./hostedDeploymentProfile";
import {
  evaluateSharedRuntimeReadiness,
  writeSharedRuntimeReadinessArtifact,
  type SharedRuntimeReadinessArtifactWriteResult,
  type SharedRuntimeReadinessReport,
} from "./sharedRuntimeReadiness";
import {
  evaluateV12ReadinessBundle,
  writeV12ReadinessBundleArtifact,
  type V12ReadinessBundle,
  type V12ReadinessBundleArtifactWriteResult,
} from "./v12ReadinessBundle";
import {
  evaluateV12ProductionWorkflowPreflightSync,
  writeV12ProductionWorkflowPreflightArtifact,
  type V12ProductionWorkflowPreflightReport,
  type V12ProductionWorkflowPreflightWriteResult,
} from "./v12ProductionWorkflowPreflight";

export interface V12ReadinessSuite {
  passed: boolean;
  blockers: string[];
  deployBackendId: string | null;
  kernelLabBackendId: string | null;
  childArtifactCount: number;
  totalArtifactCount: number;
  hostedProfilePassed: boolean;
  backendReadinessPassed: boolean;
  sharedRuntimePassed: boolean;
  v12ReadinessBundlePassed: boolean;
  v12ProductionWorkflowPreflightPassed: boolean;
  hostedBenchmarkProofRequired: boolean;
  hostedBenchmarkProofSourceBoundRequired: boolean;
  hostedBenchmarkProofPassed: boolean | null;
  hostedProfile: HostedDeploymentProfileReport;
  backendMatrix: BackendReadinessMatrix;
  sharedRuntime: SharedRuntimeReadinessReport;
  v12Bundle: V12ReadinessBundle;
  v12ProductionWorkflowPreflight: V12ProductionWorkflowPreflightReport;
  hostedBenchmarkProof: HostedBenchmarkProofReport | null;
}

export interface V12ReadinessSuiteChildArtifactRef {
  name: string;
  passed: boolean;
  latestPath: string;
  resultPath: string;
}

export interface V12ReadinessSuiteChildArtifacts {
  hostedDeploymentProfile: V12ReadinessSuiteChildArtifactRef;
  backendReadinessMatrix: V12ReadinessSuiteChildArtifactRef;
  sharedRuntimeReadiness: V12ReadinessSuiteChildArtifactRef;
  v12ReadinessBundle: V12ReadinessSuiteChildArtifactRef;
  v12ProductionWorkflowPreflight: V12ReadinessSuiteChildArtifactRef;
  hostedBenchmarkProof?: V12ReadinessSuiteChildArtifactRef;
}

export interface V12ReadinessSuiteArtifact {
  name: "v12-readiness-suite";
  createdAt: string;
  passed: boolean;
  summary: Record<string, number | string | boolean | null>;
  suite: V12ReadinessSuite & {
    childArtifacts: V12ReadinessSuiteChildArtifacts;
  };
}

export interface V12ReadinessSuiteArtifactWriteResult {
  artifact: V12ReadinessSuiteArtifact;
  latestPath: string;
  resultPath: string;
}

export interface V12ReadinessSuiteRunResult extends V12ReadinessSuiteArtifactWriteResult {
  suite: V12ReadinessSuite;
  childArtifacts: V12ReadinessSuiteChildArtifacts;
}

export function evaluateV12ReadinessSuite(input: {
  env?: HostedDeploymentProfileEnv;
  hostedProfile?: HostedDeploymentProfileReport;
  backendMatrix?: BackendReadinessMatrix;
  sharedRuntime?: SharedRuntimeReadinessReport;
  v12Bundle?: V12ReadinessBundle;
  v12ProductionWorkflowPreflight?: V12ProductionWorkflowPreflightReport;
  hostedBenchmarkProof?: HostedBenchmarkProofReport | null;
} = {}): V12ReadinessSuite {
  const env = input.env ?? process.env;
  const hostedBenchmarkProofRequired = env.RELEASE_REQUIRE_HOSTED_BENCHMARK_PROOF === "true";
  const hostedBenchmarkProofSourceBoundRequired = readHostedBenchmarkSourceBoundRequired(
    env,
    hostedBenchmarkProofRequired,
  );
  const hostedBenchmarkProof = input.hostedBenchmarkProof ?? null;
  const hostedProfile = input.hostedProfile ?? evaluateHostedDeploymentProfile(input.env ?? process.env);
  const backendMatrix = input.backendMatrix ?? evaluateBackendReadinessMatrix({
    hostedProfile,
    hostedBenchmarkProof,
    requireHostedBenchmarkProof: hostedBenchmarkProofRequired,
  });
  const sharedRuntime = input.sharedRuntime ?? evaluateSharedRuntimeReadiness({ backendMatrix });
  const v12Bundle = input.v12Bundle ?? evaluateV12ReadinessBundle({
    hostedProfile,
    backendMatrix,
    sharedRuntime,
  });
  const v12ProductionWorkflowPreflight = input.v12ProductionWorkflowPreflight
    ?? evaluateV12ProductionWorkflowPreflightSync();
  const blockers = [
    ...hostedProfile.blockers.map((blocker) => `hosted_deployment_profile: ${blocker}`),
    ...backendMatrix.blockers.map((blocker) => `backend_readiness_matrix: ${blocker}`),
    ...sharedRuntime.blockers.map((blocker) => `shared_runtime_readiness: ${blocker}`),
    ...v12Bundle.blockers.map((blocker) => `v12_readiness_bundle: ${blocker}`),
    ...v12ProductionWorkflowPreflight.blockers.map((blocker) => `v12_production_workflow_preflight: ${blocker}`),
    ...(hostedBenchmarkProof
      ? hostedBenchmarkProof.blockers.map((blocker) => `hosted_benchmark_proof: ${blocker}`)
      : hostedBenchmarkProofRequired
        ? ["hosted_benchmark_proof: required but no HOSTED_BENCHMARK_ARTIFACT_PATH or report was provided."]
        : []),
    ...(hostedBenchmarkProof && hostedBenchmarkProofSourceBoundRequired && hostedBenchmarkProof.sourceBoundRequired !== true
      ? ["hosted_benchmark_proof: source-bound proof mode was required but the proof report was not generated with source binding required."]
      : []),
  ];
  const childArtifactCount = hostedBenchmarkProof ? 6 : 5;

  return {
    passed: blockers.length === 0,
    blockers,
    deployBackendId: v12Bundle.deployBackendId,
    kernelLabBackendId: v12Bundle.kernelLabBackendId,
    childArtifactCount,
    totalArtifactCount: childArtifactCount + 1,
    hostedProfilePassed: hostedProfile.passed,
    backendReadinessPassed: backendMatrix.passed,
    sharedRuntimePassed: sharedRuntime.passed,
    v12ReadinessBundlePassed: v12Bundle.passed,
    v12ProductionWorkflowPreflightPassed: v12ProductionWorkflowPreflight.passed,
    hostedBenchmarkProofRequired,
    hostedBenchmarkProofSourceBoundRequired,
    hostedBenchmarkProofPassed: hostedBenchmarkProof ? hostedBenchmarkProof.passed : hostedBenchmarkProofRequired ? false : null,
    hostedProfile,
    backendMatrix,
    sharedRuntime,
    v12Bundle,
    v12ProductionWorkflowPreflight,
    hostedBenchmarkProof,
  };
}

export function buildV12ReadinessSuiteArtifact(
  suite: V12ReadinessSuite,
  options: {
    createdAt?: string;
    childArtifacts?: V12ReadinessSuiteChildArtifacts;
  } = {},
): V12ReadinessSuiteArtifact {
  const childArtifacts = options.childArtifacts ?? emptyChildArtifacts();

  return {
    name: "v12-readiness-suite",
    createdAt: options.createdAt ?? new Date().toISOString(),
    passed: suite.passed,
    summary: {
      v12SuitePassed: suite.passed,
      v12SuiteBlockerCount: suite.blockers.length,
      v12SuiteArtifactCount: suite.totalArtifactCount,
      v12SuiteChildArtifactCount: suite.childArtifactCount,
      v12SuiteDeployBackendId: suite.deployBackendId,
      v12SuiteKernelLabBackendId: suite.kernelLabBackendId,
      v12SuiteFallbackBackendId: suite.v12Bundle.fallbackBackendId,
      v12SuiteBackendRoleBoundaryPassed: suite.v12Bundle.backendRoleBoundaryPassed,
      v12SuiteRequirementCount: suite.v12Bundle.requirements.length,
      v12SuitePassedRequirementCount: suite.v12Bundle.requirements.filter((requirement) => requirement.passed).length,
      v12SuiteHostedProfilePassed: suite.hostedProfilePassed,
      v12SuiteBackendReadinessPassed: suite.backendReadinessPassed,
      v12SuiteSharedRuntimePassed: suite.sharedRuntimePassed,
      v12SuiteReadinessBundlePassed: suite.v12ReadinessBundlePassed,
      v12SuiteProductionWorkflowPreflightPassed: suite.v12ProductionWorkflowPreflightPassed,
      v12SuiteHostedBenchmarkProofRequired: suite.hostedBenchmarkProofRequired,
      v12SuiteHostedBenchmarkProofSourceBoundRequired: suite.hostedBenchmarkProofSourceBoundRequired,
      v12SuiteHostedBenchmarkProofPassed: suite.hostedBenchmarkProofPassed,
    },
    suite: {
      ...suite,
      childArtifacts,
    },
  };
}

export async function writeV12ReadinessSuiteArtifact(
  suite: V12ReadinessSuite,
  options: {
    artifactDir?: string;
    createdAt?: string;
    childArtifacts?: V12ReadinessSuiteChildArtifacts;
  } = {},
): Promise<V12ReadinessSuiteArtifactWriteResult> {
  const artifactDir = options.artifactDir ?? process.env.EVAL_ARTIFACT_DIR ?? ".artifacts/evals";
  const artifact = buildV12ReadinessSuiteArtifact(suite, {
    createdAt: options.createdAt,
    childArtifacts: options.childArtifacts,
  });
  const runDir = join(artifactDir, "v12-readiness-suite");
  const timestamp = artifact.createdAt.replace(/[:.]/g, "-");
  const latestPath = join(artifactDir, "v12-readiness-suite-latest.json");
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

export async function runV12ReadinessSuite(options: {
  env?: HostedDeploymentProfileEnv;
  artifactDir?: string;
  createdAt?: string;
  hostedBenchmarkArtifactPath?: string;
} = {}): Promise<V12ReadinessSuiteRunResult> {
  const artifactDir = options.artifactDir ?? process.env.EVAL_ARTIFACT_DIR ?? ".artifacts/evals";
  const createdAt = options.createdAt ?? new Date().toISOString();
  const env = options.env ?? process.env;
  const hostedBenchmarkProofRequired = env.RELEASE_REQUIRE_HOSTED_BENCHMARK_PROOF === "true";
  const hostedBenchmarkProofSourceBoundRequired = readHostedBenchmarkSourceBoundRequired(
    env,
    hostedBenchmarkProofRequired,
  );
  const hostedBenchmarkArtifactPath = options.hostedBenchmarkArtifactPath ?? env.HOSTED_BENCHMARK_ARTIFACT_PATH;
  const hostedBenchmarkProof = hostedBenchmarkArtifactPath
    ? await evaluateHostedBenchmarkProofFile(hostedBenchmarkArtifactPath, {
      expectedSourceGitSha: readExpectedHostedBenchmarkGitSha(env),
      requireSourceBound: hostedBenchmarkProofSourceBoundRequired,
    })
    : null;
  const suite = evaluateV12ReadinessSuite({ env, hostedBenchmarkProof });
  const hosted = await writeHostedDeploymentProfileArtifact(suite.hostedProfile, { artifactDir, createdAt });
  const backend = await writeBackendReadinessMatrixArtifact(suite.backendMatrix, { artifactDir, createdAt });
  const shared = await writeSharedRuntimeReadinessArtifact(suite.sharedRuntime, { artifactDir, createdAt });
  const bundle = await writeV12ReadinessBundleArtifact(suite.v12Bundle, { artifactDir, createdAt });
  const workflowPreflight = await writeV12ProductionWorkflowPreflightArtifact(suite.v12ProductionWorkflowPreflight, {
    artifactDir,
    createdAt,
  });
  const hostedBenchmark = suite.hostedBenchmarkProof
    ? await writeHostedBenchmarkProofArtifact(suite.hostedBenchmarkProof, { artifactDir, createdAt })
    : null;
  const childArtifacts = toChildArtifacts({ hosted, backend, shared, bundle, workflowPreflight, hostedBenchmark });
  const written = await writeV12ReadinessSuiteArtifact(suite, {
    artifactDir,
    createdAt,
    childArtifacts,
  });

  return {
    ...written,
    suite,
    childArtifacts,
  };
}

function readExpectedHostedBenchmarkGitSha(env: HostedDeploymentProfileEnv): string | null {
  return env.HOSTED_BENCHMARK_EXPECTED_GIT_SHA?.trim()
    || env.GITHUB_SHA?.trim()
    || null;
}

function readHostedBenchmarkSourceBoundRequired(
  env: HostedDeploymentProfileEnv,
  hostedBenchmarkProofRequired: boolean,
): boolean {
  return hostedBenchmarkProofRequired || env.HOSTED_BENCHMARK_REQUIRE_SOURCE_BOUND === "true";
}

function toChildArtifacts(input: {
  hosted: HostedDeploymentProfileArtifactWriteResult;
  backend: BackendReadinessMatrixArtifactWriteResult;
  shared: SharedRuntimeReadinessArtifactWriteResult;
  bundle: V12ReadinessBundleArtifactWriteResult;
  workflowPreflight: V12ProductionWorkflowPreflightWriteResult;
  hostedBenchmark?: HostedBenchmarkProofArtifactWriteResult | null;
}): V12ReadinessSuiteChildArtifacts {
  const childArtifacts: V12ReadinessSuiteChildArtifacts = {
    hostedDeploymentProfile: {
      name: input.hosted.artifact.name,
      passed: input.hosted.artifact.passed,
      latestPath: input.hosted.latestPath,
      resultPath: input.hosted.resultPath,
    },
    backendReadinessMatrix: {
      name: input.backend.artifact.name,
      passed: input.backend.artifact.passed,
      latestPath: input.backend.latestPath,
      resultPath: input.backend.resultPath,
    },
    sharedRuntimeReadiness: {
      name: input.shared.artifact.name,
      passed: input.shared.artifact.passed,
      latestPath: input.shared.latestPath,
      resultPath: input.shared.resultPath,
    },
    v12ReadinessBundle: {
      name: input.bundle.artifact.name,
      passed: input.bundle.artifact.passed,
      latestPath: input.bundle.latestPath,
      resultPath: input.bundle.resultPath,
    },
    v12ProductionWorkflowPreflight: {
      name: input.workflowPreflight.artifact.name,
      passed: input.workflowPreflight.artifact.passed,
      latestPath: input.workflowPreflight.latestPath,
      resultPath: input.workflowPreflight.resultPath,
    },
  };
  if (input.hostedBenchmark) {
    childArtifacts.hostedBenchmarkProof = {
      name: input.hostedBenchmark.artifact.name,
      passed: input.hostedBenchmark.artifact.passed,
      latestPath: input.hostedBenchmark.latestPath,
      resultPath: input.hostedBenchmark.resultPath,
    };
  }
  return childArtifacts;
}

function emptyChildArtifacts(): V12ReadinessSuiteChildArtifacts {
  return {
    hostedDeploymentProfile: {
      name: "hosted-deployment-profile",
      passed: false,
      latestPath: "",
      resultPath: "",
    },
    backendReadinessMatrix: {
      name: "backend-readiness-matrix",
      passed: false,
      latestPath: "",
      resultPath: "",
    },
    sharedRuntimeReadiness: {
      name: "shared-runtime-readiness",
      passed: false,
      latestPath: "",
      resultPath: "",
    },
    v12ReadinessBundle: {
      name: "v12-readiness-bundle",
      passed: false,
      latestPath: "",
      resultPath: "",
    },
    v12ProductionWorkflowPreflight: {
      name: "v12-production-workflow-preflight",
      passed: false,
      latestPath: "",
      resultPath: "",
    },
  };
}

if (import.meta.url === `file://${process.argv[1]}`) {
  const result = await runV12ReadinessSuite();
  console.log(JSON.stringify(result.suite, null, 2));
  if (!result.suite.passed) process.exitCode = 1;
}
