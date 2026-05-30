import { mkdir, mkdtemp, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import {
  evaluateV12ProductionWorkflowPreflight,
  writeV12ProductionWorkflowPreflightArtifact,
} from "./v12ProductionWorkflowPreflight";

describe("v12 production workflow preflight", () => {
  it("passes for the checked-in production proof workflow", async () => {
    const report = await evaluateV12ProductionWorkflowPreflight({ rootDir: process.cwd() });

    expect(report.passed).toBe(true);
    expect(report.blockers).toEqual([]);
    expect(report.checks).toEqual(expect.arrayContaining([
      expect.objectContaining({ id: "workflow_inputs", passed: true }),
      expect.objectContaining({ id: "hosted_origin_binding", passed: true }),
      expect.objectContaining({ id: "compiled_backend_profile", passed: true }),
      expect.objectContaining({ id: "source_bound_proof", passed: true }),
      expect.objectContaining({ id: "telemetry_profile", passed: true }),
      expect.objectContaining({ id: "proof_steps", passed: true }),
      expect.objectContaining({ id: "artifact_retention", passed: true }),
      expect.objectContaining({ id: "package_script", passed: true }),
    ]));
  });

  it("fails if the production proof workflow does not require the final-state gate", async () => {
    const rootDir = await makeFixtureRepository({
      workflow: [
        "name: V12 Production Proof",
        "on:",
        "  workflow_dispatch:",
        "    inputs:",
        "      deploy_url:",
        "        required: true",
        "      hosted_production_benchmark_url:",
        "        required: false",
        "      hosted_benchmark_artifact_url:",
        "        required: false",
        "      hosted_benchmark_artifact_json:",
        "        required: false",
        "      hosted_benchmark_artifact_base64:",
        "        required: false",
        "jobs:",
        "  v12-production-proof:",
        "    env:",
        "      EVAL_ARTIFACT_DIR: .artifacts/evals/v12-production-proof",
        "      HOSTED_BENCHMARK_ARTIFACT_OUTPUT_PATH: .artifacts/evals/v12-production-proof/hosted/browser-runtime-bench-latest.json",
        "      VITE_DEPLOY_URL: ${{ inputs.deploy_url }}",
        "      HOSTED_PRODUCTION_BENCHMARK_URL: ${{ inputs.hosted_production_benchmark_url }}",
        "      HOSTED_BENCHMARK_EXPECTED_GIT_SHA: ${{ github.sha }}",
        "      HOSTED_BENCHMARK_REQUIRE_SOURCE_BOUND: \"true\"",
        "      VITE_GIT_SHA: ${{ github.sha }}",
        "      VITE_LLM_BACKEND: compiled-browser-webllm",
        "      VITE_DEFAULT_MODEL: Qwen3-0.6B-q4f16_1-MLC",
        "      VITE_COMPILED_WEBLLM_ENABLED: \"true\"",
        "      VITE_REQUIRE_UNLOCKED_RUNTIME: \"false\"",
        "      VITE_MTP_ENABLED: \"false\"",
        "      VITE_MEMORY_PROVIDER: browser-vector",
        "      VITE_QWEN_THINKING_MODE: disabled",
        "      VITE_BENCHMARK_TELEMETRY_ENABLED: \"true\"",
        "      VITE_BENCHMARK_TELEMETRY_URL: /api/benchmark-runs",
        "      BENCHMARK_TELEMETRY_ENABLED: \"true\"",
        "      BENCHMARK_TELEMETRY_STORAGE: postgres",
        "      BENCHMARK_TELEMETRY_DATABASE_URL: ${{ secrets.BENCHMARK_TELEMETRY_DATABASE_URL }}",
        "      BENCHMARK_TELEMETRY_ADMIN_TOKEN: ${{ secrets.BENCHMARK_TELEMETRY_ADMIN_TOKEN }}",
        "      RELEASE_REQUIRE_V12_PRODUCTION: \"true\"",
        "      RELEASE_REQUIRE_UNLOCKED_MODEL: \"false\"",
        "    steps:",
        "      - run: pnpm materialize:hosted-benchmark",
        "      - run: pnpm verify:hosted-profile",
        "      - run: pnpm verify:hosted-benchmark-proof",
        "        env:",
        "          HOSTED_BENCHMARK_ARTIFACT_PATH: ${{ steps.hosted-artifact.outputs.artifact_path }}",
        "      - run: pnpm eval:v12-production",
        "      - run: pnpm release:gate",
        "      - uses: actions/upload-artifact@v4",
        "        with:",
        "          name: v12-production-proof-artifacts",
        "          path: .artifacts/evals/v12-production-proof",
        "          if-no-files-found: error",
      ].join("\n"),
      packageJson: {
        scripts: {
          "verify:v12-production-workflow": "node --import tsx scripts/v12ProductionWorkflowPreflight.ts",
          "eval:v12-final-state": "node --import tsx scripts/v12FinalStateStatus.ts",
        },
      },
    });

    const report = await evaluateV12ProductionWorkflowPreflight({ rootDir });

    expect(report.passed).toBe(false);
    expect(report.blockers).toContain("final_state_gate: production proof workflow must set RELEASE_REQUIRE_V12_FINAL_STATE to true.");
    expect(report.blockers).toContain("final_state_gate: production proof workflow must run pnpm eval:v12-final-state before artifact upload.");
  });

  it("fails if production proof can run without source-bound hosted benchmark proof", async () => {
    const rootDir = await makeFixtureRepository({
      workflow: [
        "name: V12 Production Proof",
        "on:",
        "  workflow_dispatch:",
        "    inputs:",
        "      deploy_url:",
        "        required: true",
        "      hosted_benchmark_artifact_json:",
        "        required: false",
        "jobs:",
        "  v12-production-proof:",
        "    env:",
        "      VITE_LLM_BACKEND: compiled-browser-webllm",
        "      VITE_COMPILED_WEBLLM_ENABLED: \"true\"",
        "      VITE_MTP_ENABLED: \"false\"",
        "      RELEASE_REQUIRE_V12_PRODUCTION: \"true\"",
        "    steps:",
        "      - run: pnpm materialize:hosted-benchmark",
        "      - run: pnpm verify:hosted-profile",
        "      - run: pnpm verify:hosted-benchmark-proof",
        "      - run: pnpm eval:v12-production",
        "      - run: pnpm release:gate",
        "      - uses: actions/upload-artifact@v4",
        "        with:",
        "          path: .artifacts/evals/v12-production-proof",
      ].join("\n"),
      packageJson: { scripts: { "verify:v12-production-workflow": "node --import tsx scripts/v12ProductionWorkflowPreflight.ts" } },
    });

    const report = await evaluateV12ProductionWorkflowPreflight({ rootDir });

    expect(report.passed).toBe(false);
    expect(report.blockers).toContain("source_bound_proof: production proof workflow must set HOSTED_BENCHMARK_REQUIRE_SOURCE_BOUND to true.");
    expect(report.blockers).toContain("source_bound_proof: production proof workflow must bind hosted proof to github.sha.");
  });

  it("fails if production proof does not bind the deploy URL input into the hosted profile environment", async () => {
    const rootDir = await makeFixtureRepository({
      workflow: [
        "name: V12 Production Proof",
        "on:",
        "  workflow_dispatch:",
        "    inputs:",
        "      deploy_url:",
        "        required: true",
        "      hosted_production_benchmark_url:",
        "        required: false",
        "      hosted_benchmark_artifact_url:",
        "        required: false",
        "      hosted_benchmark_artifact_json:",
        "        required: false",
        "      hosted_benchmark_artifact_base64:",
        "        required: false",
        "jobs:",
        "  v12-production-proof:",
        "    env:",
        "      EVAL_ARTIFACT_DIR: .artifacts/evals/v12-production-proof",
        "      HOSTED_BENCHMARK_ARTIFACT_OUTPUT_PATH: .artifacts/evals/v12-production-proof/hosted/browser-runtime-bench-latest.json",
        "      HOSTED_PRODUCTION_BENCHMARK_URL: ${{ inputs.hosted_production_benchmark_url }}",
        "      HOSTED_BENCHMARK_EXPECTED_GIT_SHA: ${{ github.sha }}",
        "      HOSTED_BENCHMARK_REQUIRE_SOURCE_BOUND: \"true\"",
        "      VITE_GIT_SHA: ${{ github.sha }}",
        "      VITE_LLM_BACKEND: compiled-browser-webllm",
        "      VITE_DEFAULT_MODEL: Qwen3-0.6B-q4f16_1-MLC",
        "      VITE_COMPILED_WEBLLM_ENABLED: \"true\"",
        "      VITE_REQUIRE_UNLOCKED_RUNTIME: \"false\"",
        "      VITE_MTP_ENABLED: \"false\"",
        "      VITE_MEMORY_PROVIDER: browser-vector",
        "      VITE_QWEN_THINKING_MODE: disabled",
        "      VITE_BENCHMARK_TELEMETRY_ENABLED: \"true\"",
        "      VITE_BENCHMARK_TELEMETRY_URL: /api/benchmark-runs",
        "      BENCHMARK_TELEMETRY_ENABLED: \"true\"",
        "      BENCHMARK_TELEMETRY_STORAGE: postgres",
        "      BENCHMARK_TELEMETRY_DATABASE_URL: ${{ secrets.BENCHMARK_TELEMETRY_DATABASE_URL }}",
        "      BENCHMARK_TELEMETRY_ADMIN_TOKEN: ${{ secrets.BENCHMARK_TELEMETRY_ADMIN_TOKEN }}",
        "    steps:",
        "      - run: pnpm materialize:hosted-benchmark",
        "      - run: pnpm verify:hosted-profile",
        "      - run: pnpm verify:hosted-benchmark-proof",
        "        env:",
        "          HOSTED_BENCHMARK_ARTIFACT_PATH: ${{ steps.hosted-artifact.outputs.artifact_path }}",
        "      - run: pnpm eval:v12-production",
        "      - run: pnpm release:gate",
        "      - uses: actions/upload-artifact@v4",
        "        with:",
        "          name: v12-production-proof-artifacts",
        "          path: .artifacts/evals/v12-production-proof",
        "          if-no-files-found: error",
      ].join("\n"),
      packageJson: { scripts: { "verify:v12-production-workflow": "node --import tsx scripts/v12ProductionWorkflowPreflight.ts" } },
    });

    const report = await evaluateV12ProductionWorkflowPreflight({ rootDir });

    expect(report.passed).toBe(false);
    expect(report.blockers).toContain("hosted_origin_binding: production proof workflow must pass deploy_url into VITE_DEPLOY_URL.");
  });

  it("writes latest and timestamped workflow preflight artifacts", async () => {
    const artifactDir = await mkdtemp(join(tmpdir(), "v12-production-workflow-preflight-"));
    const report = await evaluateV12ProductionWorkflowPreflight({ rootDir: process.cwd() });

    const written = await writeV12ProductionWorkflowPreflightArtifact(report, {
      artifactDir,
      createdAt: "2026-05-30T22:00:00.000Z",
    });

    expect(written.latestPath).toBe(join(artifactDir, "v12-production-workflow-preflight-latest.json"));
    expect(written.resultPath).toBe(join(artifactDir, "v12-production-workflow-preflight", "2026-05-30T22-00-00-000Z.json"));
  });
});

async function makeFixtureRepository(input: { workflow: string; packageJson: unknown }): Promise<string> {
  const rootDir = await mkdtemp(join(tmpdir(), "v12-production-workflow-preflight-fixture-"));
  await mkdir(join(rootDir, ".github", "workflows"), { recursive: true });
  await mkdir(join(rootDir, "scripts"), { recursive: true });
  await writeFile(join(rootDir, ".github", "workflows", "v12-production-proof.yml"), input.workflow);
  await writeFile(join(rootDir, "package.json"), `${JSON.stringify(input.packageJson, null, 2)}\n`);
  return rootDir;
}
