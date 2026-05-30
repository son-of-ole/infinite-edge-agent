import { spawn } from "node:child_process";
import { existsSync } from "node:fs";
import { mkdir, readFile, writeFile } from "node:fs/promises";
import { basename, join } from "node:path";
import { summarizeReleaseGateArtifact, type ReleaseGateArtifactInput } from "./releaseGateArtifactSummary";
import {
  makeReleaseGateDefaultEnvOverrides,
  makeProductionEvalEnvOverrides,
  makeQwenParityEnvOverrides,
  makeReleaseGateChildEnv,
  makeUnlockedBenchmarkEnvOverrides,
  makeUnlockedVerifyEnvOverrides,
} from "./releaseGateConfig";
import { classifyReleaseGateProof, computeReleaseGatePassed } from "./releaseGateStatus";

type StepStatus = "passed" | "failed" | "skipped";

interface GateStep {
  name: string;
  command?: string;
  status: StepStatus;
  durationMs: number;
  exitCode?: number | null;
  reason?: string;
}

interface LatestArtifact {
  name: string;
  path: string;
  passed: boolean | null;
  summary: Record<string, number | string | boolean | null>;
}

const createdAt = new Date().toISOString();
const timestamp = createdAt.replace(/[:.]/g, "-");
const artifactRoot = process.env.EVAL_ARTIFACT_DIR ?? ".artifacts/evals";
const suiteDir = join(artifactRoot, "release-gate", timestamp);
const childArtifactRoot = join(suiteDir, "child-evals");
const failFast = process.argv.includes("--fail-fast");
const releaseGateDefaultEnv = makeReleaseGateDefaultEnvOverrides(process.env);
const releaseGateEnv = { ...process.env, ...releaseGateDefaultEnv };
const requireHostedProfile = releaseGateEnv.RELEASE_REQUIRE_HOSTED_PROFILE === "true";
const requireBackendReadinessMatrix = requireHostedProfile || releaseGateEnv.RELEASE_REQUIRE_BACKEND_READINESS_MATRIX === "true";
const requireSharedRuntimeReadiness = requireHostedProfile || releaseGateEnv.RELEASE_REQUIRE_SHARED_RUNTIME_READINESS === "true";
const requireV12ReadinessBundle = requireHostedProfile || releaseGateEnv.RELEASE_REQUIRE_V12_READINESS === "true";
const requireV12ReadinessSuite = requireHostedProfile || releaseGateEnv.RELEASE_REQUIRE_V12_SUITE === "true";
const requireHostedBenchmarkProof = releaseGateEnv.RELEASE_REQUIRE_HOSTED_BENCHMARK_PROOF === "true";

const steps: GateStep[] = [];

await runGate("typecheck", ["run", "typecheck"]);
await runGate("tests", ["run", "test"]);
await runGate(
  "unlocked verify",
  ["run", "verify:unlocked"],
  `backendPreference=${releaseGateEnv.VITE_UNLOCKED_BACKEND_PREFERENCE ?? "auto"}`,
  makeUnlockedVerifyEnvOverrides(releaseGateEnv),
);
await runGate("core smoke", ["run", "smoke:core"]);
await runGate("sdk smoke", ["run", "smoke:sdk"]);
await runGate("core stress", ["run", "stress:core"]);
await runGate(
  "browser runtime benchmark",
  ["run", "bench:browser-runtime"],
  undefined,
  makeUnlockedBenchmarkEnvOverrides(releaseGateEnv),
);
await runGate("Qwen parity accuracy", ["run", "eval:qwen-parity"], undefined, makeQwenParityEnvOverrides(releaseGateEnv));
await runGate("production eval", ["run", "eval:production"], undefined, makeProductionEvalEnvOverrides(releaseGateEnv));
if (requireHostedProfile) {
  await runGate("hosted deployment profile", ["run", "verify:hosted-profile"]);
}
if (requireBackendReadinessMatrix) {
  await runGate("backend readiness matrix", ["run", "eval:backend-readiness"]);
}
if (requireSharedRuntimeReadiness) {
  await runGate("shared runtime readiness", ["run", "eval:shared-runtime"]);
}
if (requireV12ReadinessBundle) {
  await runGate("v12 readiness bundle", ["run", "eval:v12-readiness"]);
}
if (requireV12ReadinessSuite) {
  await runGate("v12 readiness suite", ["run", "eval:v12-suite"]);
}
if (requireHostedBenchmarkProof) {
  await runGate("hosted benchmark proof", ["run", "verify:hosted-benchmark-proof"]);
}
await runGate("build", ["run", "build"]);
await runGate("web dist size", ["run", "check:web-dist"]);

const latestArtifacts = await readLatestArtifacts();
const passed = computeReleaseGatePassed({
  steps,
  latestArtifacts,
  strictUnlockedModel: releaseGateEnv.RELEASE_REQUIRE_UNLOCKED_MODEL === "true",
  requireBrowserPreviewProof: releaseGateEnv.RELEASE_REQUIRE_BROWSER_PREVIEW_PROOF === "true",
  requireMtpAcceleration: releaseGateEnv.RELEASE_REQUIRE_MTP_ACCELERATION === "true",
});
const proofClassification = classifyReleaseGateProof({
  passed,
  strictUnlockedModel: releaseGateEnv.RELEASE_REQUIRE_UNLOCKED_MODEL === "true",
  requireBrowserPreviewProof: releaseGateEnv.RELEASE_REQUIRE_BROWSER_PREVIEW_PROOF === "true",
  releaseAllowFixtureGate: releaseGateEnv.RELEASE_ALLOW_FIXTURE_GATE === "true",
  unlockedAllowFixture: releaseGateEnv.VITE_UNLOCKED_ALLOW_FIXTURE,
  manifestPath: releaseGateEnv.VITE_UNLOCKED_MODEL_MANIFEST_PATH,
  manifestSha256: releaseGateEnv.VITE_UNLOCKED_MODEL_MANIFEST_SHA256,
  latestArtifacts,
});
const release = {
  name: "release-gate",
  createdAt,
  passed,
  ...proofClassification,
  packageManager: "pnpm",
  invocation: "npm exec pnpm -- run release:gate",
  steps,
  latestArtifacts,
};

await mkdir(suiteDir, { recursive: true });
await writeFile(join(suiteDir, "results.json"), `${JSON.stringify(release, null, 2)}\n`);
await writeFile(join(suiteDir, "trace.jsonl"), `${steps.map((step) => JSON.stringify({ event: "step", createdAt, step })).join("\n")}\n`);
await writeFile(join(suiteDir, "summary.md"), buildSummary(release));
await writeFile(join(artifactRoot, "release-gate-latest.json"), `${JSON.stringify(release, null, 2)}\n`);

console.log(`Release gate: ${release.passed ? "PASS" : "FAIL"}`);
console.log(`Results: ${join(suiteDir, "results.json")}`);
console.log(`Summary: ${join(suiteDir, "summary.md")}`);

if (!release.passed) {
  process.exitCode = 1;
}

async function runGate(name: string, args: string[], reason?: string, envOverrides: Record<string, string | undefined> = {}): Promise<void> {
  if (failFast && steps.some((step) => step.status === "failed")) return;

  const started = performance.now();
  const command = pnpmCommand();
  console.log(`\n==> ${name}: ${[command.cmd, ...command.args, ...args].join(" ")}`);
  const exitCode = await spawnAndWait(command.cmd, [...command.args, ...args], envOverrides);
  const durationMs = Math.round(performance.now() - started);
  const status: StepStatus = exitCode === 0 ? "passed" : "failed";
  steps.push({
    name,
    command: [command.display, ...args].join(" "),
    status,
    durationMs,
    exitCode,
    ...(reason ? { reason } : {}),
  });
  if (status === "failed" && failFast) await writeEarlyFailureAndExit();
}

function pnpmCommand(): { cmd: string; args: string[]; display: string } {
  const npmExecPath = process.env.npm_execpath;
  if (npmExecPath && basename(npmExecPath).includes("pnpm")) {
    return { cmd: process.execPath, args: [npmExecPath], display: "pnpm" };
  }
  return { cmd: "pnpm", args: [], display: "pnpm" };
}

function spawnAndWait(command: string, args: string[], envOverrides: Record<string, string | undefined> = {}): Promise<number | null> {
  return new Promise((resolve) => {
    const child = spawn(command, args, {
      stdio: "inherit",
      env: { ...process.env, ...makeReleaseGateChildEnv(suiteDir, envOverrides) },
    });
    child.on("error", (error) => {
      console.error(`Failed to start ${command}: ${error.message}`);
      resolve(1);
    });
    child.on("close", (code) => resolve(code));
  });
}

async function readLatestArtifacts(): Promise<LatestArtifact[]> {
  const files = [
    ["core-smoke", join(childArtifactRoot, "core-smoke-latest.json")],
    ["sdk-smoke", join(childArtifactRoot, "sdk-smoke-latest.json")],
    ["core-stress", join(childArtifactRoot, "core-stress-latest.json")],
    ["browser-runtime-bench", join(childArtifactRoot, "browser-runtime-bench-latest.json")],
    ["qwen-parity-accuracy", join(childArtifactRoot, "qwen-parity-accuracy-latest.json")],
    ["unlocked-verify", join(childArtifactRoot, "unlocked-verify-latest.json")],
    ["production-readiness", join(childArtifactRoot, "production-latest.json")],
    ...(requireHostedProfile
      ? [["hosted-deployment-profile", join(childArtifactRoot, "hosted-deployment-profile-latest.json")] as const]
      : []),
    ...(requireBackendReadinessMatrix
      ? [["backend-readiness-matrix", join(childArtifactRoot, "backend-readiness-matrix-latest.json")] as const]
      : []),
    ...(requireSharedRuntimeReadiness
      ? [["shared-runtime-readiness", join(childArtifactRoot, "shared-runtime-readiness-latest.json")] as const]
      : []),
    ...(requireV12ReadinessBundle
      ? [["v12-readiness-bundle", join(childArtifactRoot, "v12-readiness-bundle-latest.json")] as const]
      : []),
    ...(requireV12ReadinessSuite
      ? [["v12-readiness-suite", join(childArtifactRoot, "v12-readiness-suite-latest.json")] as const]
      : []),
    ...(requireHostedBenchmarkProof
      ? [["hosted-benchmark-proof", join(childArtifactRoot, "hosted-benchmark-proof-latest.json")] as const]
      : []),
  ] as const;
  const artifacts: LatestArtifact[] = [];
  for (const [name, path] of files) {
    if (!existsSync(path)) {
      artifacts.push({ name, path, passed: null, summary: { missing: true } });
      continue;
    }
    const parsed = JSON.parse(await readFile(path, "utf8")) as ReleaseGateArtifactInput & { passed?: boolean };
    artifacts.push({
      name,
      path,
      passed: typeof parsed.passed === "boolean" ? parsed.passed : null,
      summary: summarizeReleaseGateArtifact(parsed),
    });
  }
  return artifacts;
}

function buildSummary(results: typeof release): string {
  const stepRows = results.steps
    .map((step) => `| ${step.name} | ${step.status.toUpperCase()} | ${step.durationMs} | ${step.reason ?? ""} |`)
    .join("\n");
  const artifactRows = results.latestArtifacts
    .map((artifact) => `| ${artifact.name} | ${artifact.passed === null ? "UNKNOWN" : artifact.passed ? "PASS" : "FAIL"} | ${artifact.path} | ${formatSummary(artifact.summary)} |`)
    .join("\n");

  return `# Release Gate

- Created: ${results.createdAt}
- Passed: ${results.passed}
- Proof mode: ${results.proofMode}
- Production release proof: ${results.productionReleaseProof}
- Grounded answer quality browser proof: ${results.groundedAnswerQualityBrowserProof}
- Capped technical speed proof: ${results.cappedTechnicalSpeedProof}
- Deploy-ready speed+quality proof: ${results.deployReadySpeedQualityProof}
- Invocation: \`${results.invocation}\`

## Steps

| Step | Status | Duration ms | Notes |
| --- | --- | ---: | --- |
${stepRows}

## Local Eval Artifacts

| Artifact | Status | Path | Summary |
| --- | --- | --- | --- |
${artifactRows}
`;
}

function formatSummary(summary: Record<string, number | string | boolean | null>): string {
  return Object.entries(summary).map(([key, value]) => `${key}=${value}`).join("; ");
}

async function writeEarlyFailureAndExit(): Promise<void> {
  await mkdir(suiteDir, { recursive: true });
  const latestArtifacts = await readLatestArtifacts();
  const proofClassification = classifyReleaseGateProof({
    passed: false,
    strictUnlockedModel: releaseGateEnv.RELEASE_REQUIRE_UNLOCKED_MODEL === "true",
    requireBrowserPreviewProof: releaseGateEnv.RELEASE_REQUIRE_BROWSER_PREVIEW_PROOF === "true",
    releaseAllowFixtureGate: releaseGateEnv.RELEASE_ALLOW_FIXTURE_GATE === "true",
    unlockedAllowFixture: releaseGateEnv.VITE_UNLOCKED_ALLOW_FIXTURE,
    manifestPath: releaseGateEnv.VITE_UNLOCKED_MODEL_MANIFEST_PATH,
    manifestSha256: releaseGateEnv.VITE_UNLOCKED_MODEL_MANIFEST_SHA256,
    latestArtifacts,
  });
  const release = {
    name: "release-gate",
    createdAt,
    passed: false,
    ...proofClassification,
    packageManager: "pnpm",
    invocation: "npm exec pnpm -- run release:gate",
    steps,
    latestArtifacts,
  };
  await writeFile(join(suiteDir, "results.json"), `${JSON.stringify(release, null, 2)}\n`);
  await writeFile(join(suiteDir, "summary.md"), buildSummary(release));
  await writeFile(join(artifactRoot, "release-gate-latest.json"), `${JSON.stringify(release, null, 2)}\n`);
  process.exit(1);
}
