import { mkdir, writeFile } from "node:fs/promises";
import { join } from "node:path";
import {
  evaluateBackendReadinessMatrix,
  type BackendReadinessMatrix,
} from "./backendReadinessMatrix";
import {
  evaluateHostedDeploymentProfile,
  type HostedDeploymentProfileReport,
} from "./hostedDeploymentProfile";
import {
  evaluateSharedRuntimeReadiness,
  type SharedRuntimeReadinessReport,
} from "./sharedRuntimeReadiness";

export interface V12ReadinessRequirement {
  id: string;
  label: string;
  passed: boolean;
  evidence: string;
  blockers: string[];
}

export interface V12ReadinessBundle {
  passed: boolean;
  blockers: string[];
  deployBackendId: string | null;
  kernelLabBackendId: string | null;
  fallbackBackendId: string | null;
  backendRoleBoundaryPassed: boolean;
  requirements: V12ReadinessRequirement[];
  hostedProfile: HostedDeploymentProfileReport;
  backendMatrix: BackendReadinessMatrix;
  sharedRuntime: SharedRuntimeReadinessReport;
}

export interface V12ReadinessBundleArtifact {
  name: "v12-readiness-bundle";
  createdAt: string;
  passed: boolean;
  summary: Record<string, number | string | boolean | null>;
  bundle: V12ReadinessBundle;
}

export interface V12ReadinessBundleArtifactWriteResult {
  artifact: V12ReadinessBundleArtifact;
  latestPath: string;
  resultPath: string;
}

export function evaluateV12ReadinessBundle(input: {
  hostedProfile?: HostedDeploymentProfileReport;
  backendMatrix?: BackendReadinessMatrix;
  sharedRuntime?: SharedRuntimeReadinessReport;
} = {}): V12ReadinessBundle {
  const hostedProfile = input.hostedProfile ?? evaluateHostedDeploymentProfile(process.env);
  const backendMatrix = input.backendMatrix ?? evaluateBackendReadinessMatrix({ hostedProfile });
  const sharedRuntime = input.sharedRuntime ?? evaluateSharedRuntimeReadiness({ backendMatrix });
  const deployBackendId = backendMatrix.deployBackendId;
  const kernelLabBackendId = backendMatrix.researchBackendIds[0] ?? sharedRuntime.kernelLabBackendId;
  const fallbackBackendId = sharedRuntime.fallbackBackendId
    ?? backendMatrix.backends.find((backend) => backend.productionRole === "fallback")?.backendId
    ?? null;
  const backendRoleBoundaryPassed = sharedRuntime.backendRoleBoundaryPassed
    && deployBackendId === "compiled-browser-webllm"
    && kernelLabBackendId === "unlocked-browser-transformer"
    && fallbackBackendId === "wasm-small-core";
  const modelRegistryAlignmentPassed = sharedRuntime.modelRegistryAlignment.aligned === true
    && sharedRuntime.modelRegistryAlignment.publicDeployOptionCount === 1
    && sharedRuntime.modelRegistryAlignment.publicKernelLabOptionCount === 1
    && sharedRuntime.modelRegistryAlignment.publicOptionCount >= 2;
  const requirements: V12ReadinessRequirement[] = [
    {
      id: "backend_broker",
      label: "Backend Broker separates deploy, research, and fallback backend roles.",
      passed: backendMatrix.backends.length >= 3
        && backendMatrix.backends.some((backend) => backend.productionRole === "production_candidate")
        && backendMatrix.backends.some((backend) => backend.productionRole === "research_kernel_lab")
        && backendMatrix.backends.some((backend) => backend.productionRole === "fallback")
        && backendRoleBoundaryPassed,
      evidence: "backend-readiness-matrix",
      blockers: backendRoleBoundaryPassed ? [] : ["Backend Broker role boundary did not prove deploy, Kernel Lab, and fallback separation."],
    },
    {
      id: "compiled_production_backend",
      label: "Compiled browser backend is the deploy-ready production answer path.",
      passed: hostedProfile.passed
        && hostedProfile.profile.llmBackend === "compiled-browser-webllm"
        && deployBackendId === "compiled-browser-webllm",
      evidence: "hosted-deployment-profile",
      blockers: hostedProfile.passed ? [] : ["Hosted deployment profile did not pass for compiled-browser-webllm."],
    },
    {
      id: "custom_webgpu_kernel_lab",
      label: "Custom WebGPU runtime is classified as Kernel Lab, not deploy answer backend.",
      passed: kernelLabBackendId === "unlocked-browser-transformer"
        && backendMatrix.backends.some((backend) =>
          backend.backendId === "unlocked-browser-transformer"
          && backend.readinessStatus === "research_only"
          && backend.deployReady === false
        ),
      evidence: "backend-readiness-matrix",
      blockers: kernelLabBackendId === "unlocked-browser-transformer" ? [] : ["Kernel Lab backend is not recorded as unlocked-browser-transformer."],
    },
    {
      id: "model_registry_alignment",
      label: "Model registry exposes one deploy model option and one Kernel Lab option aligned to Backend Broker roles.",
      passed: modelRegistryAlignmentPassed,
      evidence: "backend-readiness-matrix+shared-runtime-readiness",
      blockers: modelRegistryAlignmentPassed ? [] : ["Model registry role/options do not match Backend Broker production roles."],
    },
    {
      id: "shared_memory_context_runtime",
      label: "Memory retrieval and context rebuild are shared above backend execution.",
      passed: sharedRuntime.passed
        && sharedRuntime.deployBackendId === "compiled-browser-webllm"
        && sharedRuntime.kernelLabBackendId === "unlocked-browser-transformer"
        && sharedRuntime.fallbackBackendId === "wasm-small-core"
        && sharedRuntime.backendRoleBoundaryPassed === true,
      evidence: "shared-runtime-readiness",
      blockers: sharedRuntime.passed ? [] : ["Shared runtime readiness did not pass."],
    },
    {
      id: "backend_specific_readiness",
      label: "Production readiness is backend-specific and backed by artifacts.",
      passed: backendMatrix.passed
        && hostedProfile.passed
        && sharedRuntime.passed
        && deployBackendId === "compiled-browser-webllm",
      evidence: "v12-readiness-bundle",
      blockers: backendMatrix.passed ? [] : ["Backend readiness matrix did not pass."],
    },
  ].map((requirement) => ({
    ...requirement,
    blockers: requirement.passed ? [] : requirement.blockers,
  }));
  const blockers = requirements.flatMap((requirement) =>
    requirement.passed
      ? []
      : (requirement.blockers.length > 0
        ? requirement.blockers.map((blocker) => `${requirement.id}: ${blocker}`)
        : [`${requirement.id}: Requirement did not pass.`]),
  );

  return {
    passed: blockers.length === 0,
    blockers,
    deployBackendId,
    kernelLabBackendId,
    fallbackBackendId,
    backendRoleBoundaryPassed,
    requirements,
    hostedProfile,
    backendMatrix,
    sharedRuntime,
  };
}

export function buildV12ReadinessBundleArtifact(
  bundle: V12ReadinessBundle,
  createdAt = new Date().toISOString(),
): V12ReadinessBundleArtifact {
  return {
    name: "v12-readiness-bundle",
    createdAt,
    passed: bundle.passed,
    summary: {
      v12ReadinessPassed: bundle.passed,
      v12BlockerCount: bundle.blockers.length,
      v12DeployBackendId: bundle.deployBackendId,
      v12KernelLabBackendId: bundle.kernelLabBackendId,
      v12FallbackBackendId: bundle.fallbackBackendId,
      v12BackendRoleBoundaryPassed: bundle.backendRoleBoundaryPassed,
      v12ModelRegistryAligned: bundle.sharedRuntime.modelRegistryAlignment.aligned,
      v12ModelRegistryModelCount: bundle.sharedRuntime.modelRegistryAlignment.modelCount,
      v12PublicModelOptionCount: bundle.sharedRuntime.modelRegistryAlignment.publicOptionCount,
      v12PublicDeployOptionCount: bundle.sharedRuntime.modelRegistryAlignment.publicDeployOptionCount,
      v12PublicKernelLabOptionCount: bundle.sharedRuntime.modelRegistryAlignment.publicKernelLabOptionCount,
      v12RequirementCount: bundle.requirements.length,
      v12PassedRequirementCount: bundle.requirements.filter((requirement) => requirement.passed).length,
      v12HostedProfilePassed: bundle.hostedProfile.passed,
      v12BackendReadinessPassed: bundle.backendMatrix.passed,
      v12SharedRuntimePassed: bundle.sharedRuntime.passed,
    },
    bundle,
  };
}

export async function writeV12ReadinessBundleArtifact(
  bundle: V12ReadinessBundle,
  options: { artifactDir?: string; createdAt?: string } = {},
): Promise<V12ReadinessBundleArtifactWriteResult> {
  const artifactDir = options.artifactDir ?? process.env.EVAL_ARTIFACT_DIR ?? ".artifacts/evals";
  const artifact = buildV12ReadinessBundleArtifact(bundle, options.createdAt);
  const runDir = join(artifactDir, "v12-readiness-bundle");
  const timestamp = artifact.createdAt.replace(/[:.]/g, "-");
  const latestPath = join(artifactDir, "v12-readiness-bundle-latest.json");
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
  const bundle = evaluateV12ReadinessBundle();
  await writeV12ReadinessBundleArtifact(bundle);
  console.log(JSON.stringify(bundle, null, 2));
  if (!bundle.passed) process.exitCode = 1;
}
