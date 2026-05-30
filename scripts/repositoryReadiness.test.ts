import { mkdir, mkdtemp, readFile, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import {
  evaluateRepositoryReadiness,
  writeRepositoryReadinessArtifact,
} from "./repositoryReadiness";

describe("repository readiness", () => {
  it("passes for the public v12 repository surface", async () => {
    const report = await evaluateRepositoryReadiness({ rootDir: process.cwd() });

    expect(report.passed).toBe(true);
    expect(report.blockers).toEqual([]);
    expect(report.checks).toEqual(expect.arrayContaining([
      expect.objectContaining({ id: "package_metadata", passed: true }),
      expect.objectContaining({ id: "public_release_files", passed: true }),
      expect.objectContaining({ id: "readme_v12_story", passed: true }),
      expect.objectContaining({ id: "github_workflows", passed: true }),
      expect.objectContaining({ id: "v12_production_workflow_preflight", passed: true }),
    ]));
  });

  it("fails when the v12 production proof workflow does not preserve the source browser artifact", async () => {
    const rootDir = await makeMinimalRepositoryFixture({
      productionWorkflow: [
        "name: V12 Production Proof",
        "env:",
        "  EVAL_ARTIFACT_DIR: .artifacts/evals/v12-production-proof",
        "  HOSTED_BENCHMARK_ARTIFACT_OUTPUT_PATH: .artifacts/evals/hosted/browser-runtime-bench-latest.json",
        "steps:",
        "  - run: pnpm eval:v12-production",
        "  - uses: actions/upload-artifact@v4",
        "    with:",
        "      path: .artifacts/evals/v12-production-proof",
      ].join("\n"),
    });

    const report = await evaluateRepositoryReadiness({ rootDir });

    expect(report.passed).toBe(false);
    expect(report.blockers).toContain("github_workflows: V12 production proof workflow must store the materialized browser artifact inside .artifacts/evals/v12-production-proof.");
  });

  it("fails when CI v12 readiness does not declare a canonical deploy URL", async () => {
    const rootDir = await makeMinimalRepositoryFixture({
      ciWorkflow: [
        "name: CI",
        "steps:",
        "  - name: V12 readiness invariants",
        "    env:",
        "      HOSTED_PRODUCTION_BENCHMARK_URL: 'https://ci.example.com/__bench/browser-runtime?backend=compiled-browser-webllm&modelId=Qwen3-0.6B-q4f16_1-MLC&memoryGrounding=montana_capital&expectedExact=Helena&submitTelemetry=true&qwenThinkingMode=disabled'",
        "    run: pnpm eval:v12-readiness",
      ].join("\n"),
    });

    const report = await evaluateRepositoryReadiness({ rootDir });

    expect(report.passed).toBe(false);
    expect(report.blockers).toContain("github_workflows: CI V12 readiness invariants must set VITE_DEPLOY_URL.");
  });

  it("writes latest and timestamped repository readiness artifacts", async () => {
    const artifactDir = await mkdtemp(join(tmpdir(), "repository-readiness-"));
    const report = await evaluateRepositoryReadiness({ rootDir: process.cwd() });

    const written = await writeRepositoryReadinessArtifact(report, {
      artifactDir,
      createdAt: "2026-05-30T21:00:00.000Z",
    });

    expect(written.latestPath).toBe(join(artifactDir, "repository-readiness-latest.json"));
    expect(written.resultPath).toBe(join(artifactDir, "repository-readiness", "2026-05-30T21-00-00-000Z.json"));

    const latest = JSON.parse(await readFile(written.latestPath, "utf8")) as { passed: boolean; summary: Record<string, unknown> };
    expect(latest.passed).toBe(true);
    expect(latest.summary.repositoryReadinessPassed).toBe(true);
  });
});

async function makeMinimalRepositoryFixture(input: { productionWorkflow?: string; ciWorkflow?: string } = {}): Promise<string> {
  const rootDir = join(tmpdir(), `repository-readiness-fixture-${Date.now()}-${Math.random().toString(16).slice(2)}`);
  await mkdir(join(rootDir, ".github", "workflows"), { recursive: true });
  await mkdir(join(rootDir, "docs", "assets"), { recursive: true });
  await writeFile(join(rootDir, "package.json"), JSON.stringify({
    name: "infinite-edge-agent",
    description: "Browser-native persistent AI agent runtime with local memory, context reconstruction, compiled WebGPU inference, and a custom WebGPU Kernel Lab.",
    license: "MIT",
    repository: { url: "https://github.com/son-of-ole/infinite-edge-agent.git" },
    homepage: "https://github.com/son-of-ole/infinite-edge-agent#readme",
    bugs: { url: "https://github.com/son-of-ole/infinite-edge-agent/issues" },
  }, null, 2));
  await writeFile(join(rootDir, "README.md"), [
    "# Infinite Edge Agent",
    "docs/assets/infinite-edge-agent-readme-photo.png",
    "docs/assets/infinite-edge-agent-social.png",
    "compiled-browser-webllm",
    "unlocked-browser-transformer",
    "wasm-small-core",
    "V12 Production Proof",
  ].join("\n"));
  await writeFile(join(rootDir, "LICENSE"), "MIT\n");
  await writeFile(join(rootDir, "SECURITY.md"), "# Security\n");
  await writeFile(join(rootDir, "CONTRIBUTING.md"), "# Contributing\n");
  await writeFile(join(rootDir, "CODE_OF_CONDUCT.md"), "# Code of Conduct\n");
  await writeFile(join(rootDir, "CITATION.cff"), "cff-version: 1.2.0\n");
  await writeFile(join(rootDir, "docs", "58_REPOSITORY_METADATA.md"), "# Repository Metadata\n");
  await writeFile(join(rootDir, "docs", "assets", "infinite-edge-agent-readme-photo.png"), "fixture\n");
  await writeFile(join(rootDir, "docs", "assets", "infinite-edge-agent-social.png"), "fixture\n");
  await writeFile(join(rootDir, ".github", "workflows", "ci.yml"), input.ciWorkflow ?? [
    "name: CI",
    "steps:",
    "  - name: V12 readiness invariants",
    "    env:",
    "      VITE_DEPLOY_URL: https://ci.example.com",
    "    run: pnpm eval:v12-readiness",
  ].join("\n"));
  await writeFile(join(rootDir, ".github", "workflows", "v12-production-proof.yml"), input.productionWorkflow ?? [
    "name: V12 Production Proof",
    "env:",
    "  EVAL_ARTIFACT_DIR: .artifacts/evals/v12-production-proof",
    "  HOSTED_BENCHMARK_REQUIRE_SOURCE_BOUND: \"true\"",
    "  HOSTED_BENCHMARK_ARTIFACT_OUTPUT_PATH: .artifacts/evals/v12-production-proof/hosted/browser-runtime-bench-latest.json",
    "steps:",
    "  - run: pnpm eval:v12-production",
    "  - uses: actions/upload-artifact@v4",
    "    with:",
    "      path: .artifacts/evals/v12-production-proof",
  ].join("\n"));
  return rootDir;
}
