import { mkdtemp, readFile, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import {
  buildHostedBenchmarkProofArtifact,
  evaluateHostedBenchmarkProof,
  evaluateHostedBenchmarkProofFile,
  writeHostedBenchmarkProofArtifact,
} from "./hostedBenchmarkProof";

function makePassingBrowserPreviewArtifact() {
  return {
    name: "browser-preview-benchmark",
    createdAt: "2026-05-30T21:00:00.000Z",
    passed: true,
    summary: {
      runtimeBackendId: "compiled-browser-webllm",
      runtimeBackendRole: "production_candidate",
      deployBackendId: "compiled-browser-webllm",
      productionDeployReadyPassed: true,
      compiledBackendReadyPassed: true,
      memoryGroundingRequired: true,
      memoryGroundingPassed: true,
      memoryExpectedHitPassed: true,
      memoryContextRebuildPassed: true,
      memoryAnswerOnlyPassed: true,
      directModelFactualProofUsed: false,
      expectedExactCheckCount: 1,
      expectedExactPassCount: 1,
      expectedExactPassed: true,
      technicalProofOnly: false,
      productionQualityPassed: true,
      productionSpeedFloorPassed: true,
      productionSpeedTokensPerSecond: 2.7,
      productionSpeedFloorTokensPerSecond: 2,
      meanTokensPerSecond: 2.7,
      strictWebGpuPassed: true,
      cpuFallbackUsed: false,
    },
    runs: [
      {
        response: "Helena",
        runtimeTrace: {
          backend: "compiled-browser-webllm",
        },
      },
    ],
  };
}

describe("hosted benchmark proof verifier", () => {
  it("passes a real browser compiled-backend artifact with grounded exact output", () => {
    const report = evaluateHostedBenchmarkProof({
      artifact: makePassingBrowserPreviewArtifact(),
      artifactPath: "/tmp/browser-runtime-bench-latest.json",
    });

    expect(report).toMatchObject({
      passed: true,
      blockers: [],
      artifactPath: "/tmp/browser-runtime-bench-latest.json",
      proof: {
        sourceName: "browser-preview-benchmark",
        runtimeBackendId: "compiled-browser-webllm",
        deployBackendId: "compiled-browser-webllm",
        response: "Helena",
        productionDeployReadyPassed: true,
        compiledBackendReadyPassed: true,
        memoryGroundingPassed: true,
        expectedExactPassed: true,
        productionSpeedFloorPassed: true,
        meanTokensPerSecond: 2.7,
        directModelFactualProofUsed: false,
        technicalProofOnly: false,
        strictWebGpuPassed: true,
        cpuFallbackUsed: false,
      },
    });
  });

  it("fails when direct factual output is counted without memory grounding", () => {
    const report = evaluateHostedBenchmarkProof({
      artifact: {
        ...makePassingBrowserPreviewArtifact(),
        summary: {
          ...makePassingBrowserPreviewArtifact().summary,
          memoryGroundingPassed: false,
          directModelFactualProofUsed: true,
        },
      },
    });

    expect(report.passed).toBe(false);
    expect(report.blockers).toEqual(expect.arrayContaining([
      "Hosted benchmark proof requires memoryGroundingPassed=true.",
      "Hosted benchmark proof cannot count direct model factual output as retrieval proof.",
    ]));
  });

  it("extracts proof fields from a browser-runtime-bench wrapper artifact", () => {
    const report = evaluateHostedBenchmarkProof({
      artifact: {
        name: "browser-runtime-bench",
        passed: true,
        summary: {
          browserPreviewMode: "completed",
          browserPreviewPassed: true,
        },
        browserPreview: {
          mode: "completed",
          passed: true,
          summary: makePassingBrowserPreviewArtifact().summary,
          runs: makePassingBrowserPreviewArtifact().runs,
        },
      },
    });

    expect(report.passed).toBe(true);
    expect(report.proof.sourceName).toBe("browser-runtime-bench.browserPreview");
    expect(report.proof.response).toBe("Helena");
  });

  it("builds a release-gate friendly hosted benchmark proof artifact", () => {
    const artifact = buildHostedBenchmarkProofArtifact(
      evaluateHostedBenchmarkProof({
        artifact: makePassingBrowserPreviewArtifact(),
        artifactPath: "/tmp/browser-runtime-bench-latest.json",
      }),
      "2026-05-30T21:00:00.000Z",
    );

    expect(artifact).toMatchObject({
      name: "hosted-benchmark-proof",
      createdAt: "2026-05-30T21:00:00.000Z",
      passed: true,
      summary: {
        hostedBenchmarkProofPassed: true,
        hostedBenchmarkProofBlockerCount: 0,
        hostedBenchmarkArtifactPath: "/tmp/browser-runtime-bench-latest.json",
        hostedBenchmarkRuntimeBackendId: "compiled-browser-webllm",
        hostedBenchmarkDeployBackendId: "compiled-browser-webllm",
        hostedBenchmarkCompiledBackendReadyPassed: true,
        hostedBenchmarkProductionDeployReadyPassed: true,
        hostedBenchmarkMemoryGroundingPassed: true,
        hostedBenchmarkExpectedExactPassed: true,
        hostedBenchmarkProductionSpeedFloorPassed: true,
        hostedBenchmarkMeanTokensPerSecond: 2.7,
        hostedBenchmarkDirectModelFactualProofUsed: false,
        hostedBenchmarkTechnicalProofOnly: false,
        hostedBenchmarkCpuFallbackUsed: false,
        hostedBenchmarkStrictWebGpuPassed: true,
      },
    });
  });

  it("reads a saved artifact and writes latest plus timestamped proof artifacts", async () => {
    const artifactDir = await mkdtemp(join(tmpdir(), "hosted-benchmark-proof-"));
    const sourcePath = join(artifactDir, "browser-runtime-bench-latest.json");
    await writeFile(sourcePath, `${JSON.stringify(makePassingBrowserPreviewArtifact(), null, 2)}\n`);

    const report = await evaluateHostedBenchmarkProofFile(sourcePath);
    const written = await writeHostedBenchmarkProofArtifact(report, {
      artifactDir,
      createdAt: "2026-05-30T21:00:00.000Z",
    });

    expect(report.passed).toBe(true);
    expect(written.latestPath).toBe(join(artifactDir, "hosted-benchmark-proof-latest.json"));
    expect(written.resultPath).toBe(join(artifactDir, "hosted-benchmark-proof", "2026-05-30T21-00-00-000Z.json"));

    const latest = JSON.parse(await readFile(written.latestPath, "utf8")) as ReturnType<typeof buildHostedBenchmarkProofArtifact>;
    expect(latest.summary.hostedBenchmarkRuntimeBackendId).toBe("compiled-browser-webllm");
  });
});
