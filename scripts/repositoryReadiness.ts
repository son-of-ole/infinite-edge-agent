import { existsSync } from "node:fs";
import { mkdir, readFile, writeFile } from "node:fs/promises";
import { join } from "node:path";
import { evaluateV12ProductionWorkflowPreflight } from "./v12ProductionWorkflowPreflight";

export interface RepositoryReadinessCheck {
  id: string;
  label: string;
  passed: boolean;
  evidence: string[];
  blockers: string[];
}

export interface RepositoryReadinessReport {
  passed: boolean;
  blockers: string[];
  checks: RepositoryReadinessCheck[];
}

export interface RepositoryReadinessArtifact {
  name: "repository-readiness";
  createdAt: string;
  passed: boolean;
  summary: Record<string, number | string | boolean | null>;
  report: RepositoryReadinessReport;
}

export interface RepositoryReadinessArtifactWriteResult {
  artifact: RepositoryReadinessArtifact;
  latestPath: string;
  resultPath: string;
}

interface PackageJson {
  name?: unknown;
  description?: unknown;
  license?: unknown;
  repository?: unknown;
  homepage?: unknown;
  bugs?: unknown;
}

const EXPECTED_DESCRIPTION = "Browser-native persistent AI agent runtime with local memory, context reconstruction, compiled WebGPU inference, and a custom WebGPU Kernel Lab.";
const EXPECTED_REPOSITORY_URL = "https://github.com/son-of-ole/infinite-edge-agent.git";
const EXPECTED_HOMEPAGE = "https://github.com/son-of-ole/infinite-edge-agent#readme";
const EXPECTED_BUGS_URL = "https://github.com/son-of-ole/infinite-edge-agent/issues";
const REQUIRED_PUBLIC_FILES = [
  "README.md",
  "LICENSE",
  "SECURITY.md",
  "CONTRIBUTING.md",
  "CODE_OF_CONDUCT.md",
  "CITATION.cff",
  "docs/58_REPOSITORY_METADATA.md",
  "docs/assets/infinite-edge-agent-readme-photo.png",
  "docs/assets/infinite-edge-agent-social.png",
];

export async function evaluateRepositoryReadiness(input: { rootDir?: string } = {}): Promise<RepositoryReadinessReport> {
  const rootDir = input.rootDir ?? process.cwd();
  const checks: RepositoryReadinessCheck[] = [
    await checkPackageMetadata(rootDir),
    checkPublicReleaseFiles(rootDir),
    await checkReadmeV12Story(rootDir),
    await checkGithubWorkflows(rootDir),
    await checkV12ProductionWorkflowPreflight(rootDir),
  ];
  const blockers = checks.flatMap((check) =>
    check.passed ? [] : check.blockers.map((blocker) => `${check.id}: ${blocker}`));

  return {
    passed: blockers.length === 0,
    blockers,
    checks,
  };
}

export function buildRepositoryReadinessArtifact(
  report: RepositoryReadinessReport,
  createdAt = new Date().toISOString(),
): RepositoryReadinessArtifact {
  return {
    name: "repository-readiness",
    createdAt,
    passed: report.passed,
    summary: {
      repositoryReadinessPassed: report.passed,
      repositoryReadinessBlockerCount: report.blockers.length,
      repositoryReadinessCheckCount: report.checks.length,
      repositoryReadinessPassedCheckCount: report.checks.filter((check) => check.passed).length,
    },
    report,
  };
}

export async function writeRepositoryReadinessArtifact(
  report: RepositoryReadinessReport,
  options: { artifactDir?: string; createdAt?: string } = {},
): Promise<RepositoryReadinessArtifactWriteResult> {
  const artifactDir = options.artifactDir ?? process.env.EVAL_ARTIFACT_DIR ?? ".artifacts/evals";
  const artifact = buildRepositoryReadinessArtifact(report, options.createdAt);
  const runDir = join(artifactDir, "repository-readiness");
  const timestamp = artifact.createdAt.replace(/[:.]/g, "-");
  const latestPath = join(artifactDir, "repository-readiness-latest.json");
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

async function checkPackageMetadata(rootDir: string): Promise<RepositoryReadinessCheck> {
  const evidence = ["package.json"];
  const blockers: string[] = [];
  const packageJson = await readJsonFile<PackageJson>(join(rootDir, "package.json"));

  if (packageJson?.name !== "infinite-edge-agent") blockers.push("package.json must use name infinite-edge-agent.");
  if (packageJson?.description !== EXPECTED_DESCRIPTION) blockers.push("package.json description must match public repository metadata.");
  if (packageJson?.license !== "MIT") blockers.push("package.json must declare MIT license.");
  if (readNestedString(packageJson?.repository, "url") !== EXPECTED_REPOSITORY_URL) blockers.push("package.json repository URL must target son-of-ole/infinite-edge-agent.");
  if (packageJson?.homepage !== EXPECTED_HOMEPAGE) blockers.push("package.json homepage must target the GitHub README.");
  if (readNestedString(packageJson?.bugs, "url") !== EXPECTED_BUGS_URL) blockers.push("package.json bugs URL must target GitHub issues.");

  return makeCheck({
    id: "package_metadata",
    label: "Package metadata matches the public GitHub repository.",
    evidence,
    blockers,
  });
}

function checkPublicReleaseFiles(rootDir: string): RepositoryReadinessCheck {
  const blockers = REQUIRED_PUBLIC_FILES
    .filter((path) => !existsSync(join(rootDir, path)))
    .map((path) => `Required public release file is missing: ${path}.`);

  return makeCheck({
    id: "public_release_files",
    label: "Public release files, README hero photo, and social preview asset are present.",
    evidence: REQUIRED_PUBLIC_FILES,
    blockers,
  });
}

async function checkReadmeV12Story(rootDir: string): Promise<RepositoryReadinessCheck> {
  const readme = await readTextFile(join(rootDir, "README.md"));
  const required = [
    "docs/assets/infinite-edge-agent-readme-photo.png",
    "docs/assets/infinite-edge-agent-social.png",
    "compiled-browser-webllm",
    "unlocked-browser-transformer",
    "wasm-small-core",
    "V12 Production Proof",
  ];
  const blockers = required
    .filter((needle) => !readme.includes(needle))
    .map((needle) => `README must mention ${needle}.`);

  return makeCheck({
    id: "readme_v12_story",
    label: "README describes the v12 backend roles and production proof path.",
    evidence: ["README.md"],
    blockers,
  });
}

async function checkGithubWorkflows(rootDir: string): Promise<RepositoryReadinessCheck> {
  const ciWorkflowPath = ".github/workflows/ci.yml";
  const productionWorkflowPath = ".github/workflows/v12-production-proof.yml";
  const ciWorkflow = await readTextFile(join(rootDir, ciWorkflowPath));
  const productionWorkflow = await readTextFile(join(rootDir, productionWorkflowPath));
  const blockers: string[] = [];

  if (!ciWorkflow.includes("name: V12 readiness invariants")) {
    blockers.push("CI must run the V12 readiness invariants step.");
  }
  if (!ciWorkflow.includes("VITE_DEPLOY_URL:")) {
    blockers.push("CI V12 readiness invariants must set VITE_DEPLOY_URL.");
  }
  if (!ciWorkflow.includes("HOSTED_PRODUCTION_BENCHMARK_URL:")) {
    blockers.push("CI V12 readiness invariants must set HOSTED_PRODUCTION_BENCHMARK_URL.");
  }
  if (!productionWorkflow.includes("HOSTED_BENCHMARK_REQUIRE_SOURCE_BOUND: \"true\"")) {
    blockers.push("V12 production proof workflow must require source-bound hosted benchmark proof.");
  }
  if (!productionWorkflow.includes("HOSTED_BENCHMARK_ARTIFACT_OUTPUT_PATH: .artifacts/evals/v12-production-proof/hosted/browser-runtime-bench-latest.json")) {
    blockers.push("V12 production proof workflow must store the materialized browser artifact inside .artifacts/evals/v12-production-proof.");
  }
  if (!productionWorkflow.includes("path: .artifacts/evals/v12-production-proof")) {
    blockers.push("V12 production proof workflow must upload the v12 production proof artifact directory.");
  }

  return makeCheck({
    id: "github_workflows",
    label: "GitHub workflows enforce v12 invariant and hosted production proof gates.",
    evidence: [ciWorkflowPath, productionWorkflowPath],
    blockers,
  });
}

async function checkV12ProductionWorkflowPreflight(rootDir: string): Promise<RepositoryReadinessCheck> {
  const report = await evaluateV12ProductionWorkflowPreflight({ rootDir });

  return makeCheck({
    id: "v12_production_workflow_preflight",
    label: "V12 production proof workflow passes the dedicated hosted-proof preflight.",
    evidence: [".github/workflows/v12-production-proof.yml", "package.json"],
    blockers: report.blockers,
  });
}

function makeCheck(input: {
  id: string;
  label: string;
  evidence: string[];
  blockers: string[];
}): RepositoryReadinessCheck {
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

function readNestedString(value: unknown, key: string): string | null {
  if (typeof value !== "object" || value === null || Array.isArray(value)) return null;
  const nested = (value as Record<string, unknown>)[key];
  return typeof nested === "string" ? nested : null;
}

if (import.meta.url === `file://${process.argv[1]}`) {
  const report = await evaluateRepositoryReadiness();
  await writeRepositoryReadinessArtifact(report);
  console.log(JSON.stringify(report, null, 2));
  if (!report.passed) process.exitCode = 1;
}
