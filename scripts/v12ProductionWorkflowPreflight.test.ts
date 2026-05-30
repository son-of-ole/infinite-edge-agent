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
      expect.objectContaining({ id: "compiled_backend_profile", passed: true }),
      expect.objectContaining({ id: "source_bound_proof", passed: true }),
      expect.objectContaining({ id: "telemetry_profile", passed: true }),
      expect.objectContaining({ id: "proof_steps", passed: true }),
      expect.objectContaining({ id: "artifact_retention", passed: true }),
      expect.objectContaining({ id: "package_script", passed: true }),
    ]));
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
