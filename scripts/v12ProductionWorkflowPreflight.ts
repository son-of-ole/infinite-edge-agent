import { readFileSync } from "node:fs";
import { mkdir, readFile, writeFile } from "node:fs/promises";
import { join } from "node:path";

export interface V12ProductionWorkflowPreflightCheck {
  id: string;
  label: string;
  passed: boolean;
  evidence: string[];
  blockers: string[];
}

export interface V12ProductionWorkflowPreflightReport {
  passed: boolean;
  blockers: string[];
  checks: V12ProductionWorkflowPreflightCheck[];
}

export interface V12ProductionWorkflowPreflightArtifact {
  name: "v12-production-workflow-preflight";
  createdAt: string;
  passed: boolean;
  summary: Record<string, number | boolean>;
  report: V12ProductionWorkflowPreflightReport;
}

export interface V12ProductionWorkflowPreflightWriteResult {
  artifact: V12ProductionWorkflowPreflightArtifact;
  latestPath: string;
  resultPath: string;
}

interface PackageJson {
  scripts?: Record<string, unknown>;
}

const WORKFLOW_PATH = ".github/workflows/v12-production-proof.yml";
const PACKAGE_PATH = "package.json";

export async function evaluateV12ProductionWorkflowPreflight(
  input: { rootDir?: string } = {},
): Promise<V12ProductionWorkflowPreflightReport> {
  const rootDir = input.rootDir ?? process.cwd();
  const workflow = await readTextFile(join(rootDir, WORKFLOW_PATH));
  const packageJson = await readJsonFile<PackageJson>(join(rootDir, PACKAGE_PATH));
  return buildV12ProductionWorkflowPreflightReport(workflow, packageJson);
}

export function evaluateV12ProductionWorkflowPreflightSync(
  input: { rootDir?: string } = {},
): V12ProductionWorkflowPreflightReport {
  const rootDir = input.rootDir ?? process.cwd();
  const workflow = readTextFileSync(join(rootDir, WORKFLOW_PATH));
  const packageJson = readJsonFileSync<PackageJson>(join(rootDir, PACKAGE_PATH));
  return buildV12ProductionWorkflowPreflightReport(workflow, packageJson);
}

function buildV12ProductionWorkflowPreflightReport(
  workflow: string,
  packageJson: PackageJson | null,
): V12ProductionWorkflowPreflightReport {
  const checks = [
    checkWorkflowInputs(workflow),
    checkHostedOriginBinding(workflow),
    checkCompiledBackendProfile(workflow),
    checkSourceBoundProof(workflow),
    checkTelemetryProfile(workflow),
    checkProofSteps(workflow),
    checkArtifactRetention(workflow),
    checkPackageScript(packageJson),
  ];
  const blockers = checks.flatMap((check) =>
    check.passed ? [] : check.blockers.map((blocker) => `${check.id}: ${blocker}`));

  return {
    passed: blockers.length === 0,
    blockers,
    checks,
  };
}

export function buildV12ProductionWorkflowPreflightArtifact(
  report: V12ProductionWorkflowPreflightReport,
  createdAt = new Date().toISOString(),
): V12ProductionWorkflowPreflightArtifact {
  return {
    name: "v12-production-workflow-preflight",
    createdAt,
    passed: report.passed,
    summary: {
      v12ProductionWorkflowPreflightPassed: report.passed,
      v12ProductionWorkflowPreflightBlockerCount: report.blockers.length,
      v12ProductionWorkflowPreflightCheckCount: report.checks.length,
      v12ProductionWorkflowPreflightPassedCheckCount: report.checks.filter((check) => check.passed).length,
    },
    report,
  };
}

export async function writeV12ProductionWorkflowPreflightArtifact(
  report: V12ProductionWorkflowPreflightReport,
  options: { artifactDir?: string; createdAt?: string } = {},
): Promise<V12ProductionWorkflowPreflightWriteResult> {
  const artifactDir = options.artifactDir ?? process.env.EVAL_ARTIFACT_DIR ?? ".artifacts/evals";
  const artifact = buildV12ProductionWorkflowPreflightArtifact(report, options.createdAt);
  const runDir = join(artifactDir, "v12-production-workflow-preflight");
  const timestamp = artifact.createdAt.replace(/[:.]/g, "-");
  const latestPath = join(artifactDir, "v12-production-workflow-preflight-latest.json");
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

function checkWorkflowInputs(workflow: string): V12ProductionWorkflowPreflightCheck {
  const blockers = collectMissing(workflow, [
    ["deploy_url:", "production proof workflow must accept deploy_url input."],
    ["required: true", "production proof workflow must require at least one explicit input."],
    ["hosted_production_benchmark_url:", "production proof workflow must accept explicit hosted benchmark URL override."],
    ["hosted_benchmark_artifact_url:", "production proof workflow must accept HTTPS hosted artifact URL input."],
    ["hosted_benchmark_artifact_json:", "production proof workflow must accept pasted artifact JSON input."],
    ["hosted_benchmark_artifact_base64:", "production proof workflow must accept base64 artifact JSON input."],
  ]);

  return makeCheck({
    id: "workflow_inputs",
    label: "Workflow accepts deploy URL and all hosted benchmark artifact source inputs.",
    evidence: [WORKFLOW_PATH],
    blockers,
  });
}

function checkHostedOriginBinding(workflow: string): V12ProductionWorkflowPreflightCheck {
  const blockers = collectMissing(workflow, [
    ["VITE_DEPLOY_URL: ${{ inputs.deploy_url }}", "production proof workflow must pass deploy_url into VITE_DEPLOY_URL."],
    ["HOSTED_PRODUCTION_BENCHMARK_URL: ${{ inputs.hosted_production_benchmark_url }}", "production proof workflow must pass the explicit benchmark URL override into HOSTED_PRODUCTION_BENCHMARK_URL."],
  ]);

  return makeCheck({
    id: "hosted_origin_binding",
    label: "Workflow binds the requested public deploy origin into hosted profile and benchmark proof configuration.",
    evidence: [WORKFLOW_PATH],
    blockers,
  });
}

function checkCompiledBackendProfile(workflow: string): V12ProductionWorkflowPreflightCheck {
  const blockers = collectMissing(workflow, [
    ["VITE_LLM_BACKEND: compiled-browser-webllm", "production proof workflow must select compiled-browser-webllm."],
    ["VITE_DEFAULT_MODEL: Qwen3-0.6B-q4f16_1-MLC", "production proof workflow must use the public compiled Qwen model id."],
    ["VITE_COMPILED_WEBLLM_ENABLED: \"true\"", "production proof workflow must enable the compiled WebLLM backend."],
    ["VITE_REQUIRE_UNLOCKED_RUNTIME: \"false\"", "production proof workflow must not require the Kernel Lab runtime."],
    ["VITE_MTP_ENABLED: \"false\"", "production proof workflow must keep MTP disabled in production proof."],
    ["VITE_MEMORY_PROVIDER: browser-vector", "production proof workflow must use browser-vector memory."],
    ["VITE_QWEN_THINKING_MODE: disabled", "production proof workflow must disable thinking mode for exact canaries."],
    ["RELEASE_REQUIRE_V12_PRODUCTION: \"true\"", "production proof workflow must require the v12 production archive gate."],
    ["RELEASE_REQUIRE_UNLOCKED_MODEL: \"false\"", "production proof workflow must keep unlocked Kernel Lab release requirement off."],
  ]);

  return makeCheck({
    id: "compiled_backend_profile",
    label: "Workflow runs the compiled production backend profile, not the Kernel Lab path.",
    evidence: [WORKFLOW_PATH],
    blockers,
  });
}

function checkSourceBoundProof(workflow: string): V12ProductionWorkflowPreflightCheck {
  const blockers = collectMissing(workflow, [
    ["HOSTED_BENCHMARK_EXPECTED_GIT_SHA: ${{ github.sha }}", "production proof workflow must bind hosted proof to github.sha."],
    ["HOSTED_BENCHMARK_REQUIRE_SOURCE_BOUND: \"true\"", "production proof workflow must set HOSTED_BENCHMARK_REQUIRE_SOURCE_BOUND to true."],
    ["VITE_GIT_SHA: ${{ github.sha }}", "production proof workflow must inject github.sha into the hosted/runtime profile."],
  ]);

  return makeCheck({
    id: "source_bound_proof",
    label: "Workflow requires the hosted browser proof to be source-bound to the checked commit.",
    evidence: [WORKFLOW_PATH],
    blockers,
  });
}

function checkTelemetryProfile(workflow: string): V12ProductionWorkflowPreflightCheck {
  const blockers = collectMissing(workflow, [
    ["VITE_BENCHMARK_TELEMETRY_ENABLED: \"true\"", "production proof workflow must enable browser telemetry submission."],
    ["VITE_BENCHMARK_TELEMETRY_URL: /api/benchmark-runs", "production proof workflow must use the benchmark telemetry API route."],
    ["BENCHMARK_TELEMETRY_ENABLED: \"true\"", "production proof workflow must enable telemetry collection server-side."],
    ["BENCHMARK_TELEMETRY_STORAGE: postgres", "production proof workflow must use durable telemetry storage."],
    ["BENCHMARK_TELEMETRY_DATABASE_URL: ${{ secrets.BENCHMARK_TELEMETRY_DATABASE_URL }}", "production proof workflow must source telemetry database URL from secrets."],
    ["BENCHMARK_TELEMETRY_ADMIN_TOKEN: ${{ secrets.BENCHMARK_TELEMETRY_ADMIN_TOKEN }}", "production proof workflow must source telemetry admin token from secrets."],
  ]);

  return makeCheck({
    id: "telemetry_profile",
    label: "Workflow uses the hosted benchmark telemetry profile and repository secrets.",
    evidence: [WORKFLOW_PATH],
    blockers,
  });
}

function checkProofSteps(workflow: string): V12ProductionWorkflowPreflightCheck {
  const blockers = collectMissing(workflow, [
    ["pnpm materialize:hosted-benchmark", "production proof workflow must materialize the hosted benchmark artifact."],
    ["pnpm verify:hosted-profile", "production proof workflow must verify the hosted deploy profile."],
    ["pnpm verify:hosted-benchmark-proof", "production proof workflow must verify the saved hosted benchmark proof."],
    ["HOSTED_BENCHMARK_ARTIFACT_PATH: ${{ steps.hosted-artifact.outputs.artifact_path }}", "production proof workflow must pass the materialized artifact path into proof steps."],
    ["pnpm eval:v12-production", "production proof workflow must build the v12 production archive."],
    ["pnpm release:gate", "production proof workflow must run the release gate with v12 production env."],
  ]);

  return makeCheck({
    id: "proof_steps",
    label: "Workflow runs materialization, hosted proof verification, v12 archive, and release gate steps.",
    evidence: [WORKFLOW_PATH],
    blockers,
  });
}

function checkArtifactRetention(workflow: string): V12ProductionWorkflowPreflightCheck {
  const blockers = collectMissing(workflow, [
    ["EVAL_ARTIFACT_DIR: .artifacts/evals/v12-production-proof", "production proof workflow must write artifacts under the v12 proof directory."],
    ["HOSTED_BENCHMARK_ARTIFACT_OUTPUT_PATH: .artifacts/evals/v12-production-proof/hosted/browser-runtime-bench-latest.json", "production proof workflow must preserve the materialized browser artifact inside the uploaded bundle."],
    ["uses: actions/upload-artifact@v4", "production proof workflow must upload proof artifacts."],
    ["name: v12-production-proof-artifacts", "production proof workflow must use the stable v12 proof artifact name."],
    ["path: .artifacts/evals/v12-production-proof", "production proof workflow must upload the v12 production proof directory."],
    ["if-no-files-found: error", "production proof workflow must fail if no proof artifacts are produced."],
  ]);

  return makeCheck({
    id: "artifact_retention",
    label: "Workflow keeps the source browser artifact inside the uploaded proof bundle.",
    evidence: [WORKFLOW_PATH],
    blockers,
  });
}

function checkPackageScript(packageJson: PackageJson | null): V12ProductionWorkflowPreflightCheck {
  const script = packageJson?.scripts?.["verify:v12-production-workflow"];
  const blockers: string[] = [];
  if (script !== "node --import tsx scripts/v12ProductionWorkflowPreflight.ts") {
    blockers.push("package.json must expose verify:v12-production-workflow.");
  }

  return makeCheck({
    id: "package_script",
    label: "Package scripts expose the v12 production workflow preflight verifier.",
    evidence: [PACKAGE_PATH],
    blockers,
  });
}

function collectMissing(workflow: string, requirements: Array<[needle: string, blocker: string]>): string[] {
  return requirements
    .filter(([needle]) => !workflow.includes(needle))
    .map(([, blocker]) => blocker);
}

function makeCheck(input: {
  id: string;
  label: string;
  evidence: string[];
  blockers: string[];
}): V12ProductionWorkflowPreflightCheck {
  return {
    ...input,
    passed: input.blockers.length === 0,
  };
}

async function readJsonFile<T>(path: string): Promise<T | null> {
  try {
    return JSON.parse(await readFile(path, "utf8")) as T;
  } catch {
    return null;
  }
}

async function readTextFile(path: string): Promise<string> {
  try {
    return await readFile(path, "utf8");
  } catch {
    return "";
  }
}

function readJsonFileSync<T>(path: string): T | null {
  try {
    return JSON.parse(readFileSync(path, "utf8")) as T;
  } catch {
    return null;
  }
}

function readTextFileSync(path: string): string {
  try {
    return readFileSync(path, "utf8");
  } catch {
    return "";
  }
}

if (import.meta.url === `file://${process.argv[1]}`) {
  const report = await evaluateV12ProductionWorkflowPreflight();
  await writeV12ProductionWorkflowPreflightArtifact(report);
  console.log(JSON.stringify(report, null, 2));
  if (!report.passed) process.exitCode = 1;
}
