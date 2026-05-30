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
  const brokerSelection = {
    backendId: "compiled-browser-webllm",
    modelId: "Qwen3-0.6B-q4f16_1-MLC",
    productionRole: "production_candidate",
    deployReadyCandidate: true,
    reason: "compiled_first_grounded_answer",
    fallbackChain: ["unlocked-browser-transformer", "wasm-small-core"],
    proofRequirements: ["memory_grounding", "quality_canaries", "speed_floor", "backend_trace"],
  };

  return {
    name: "browser-preview-benchmark",
    createdAt: "2026-05-30T21:00:00.000Z",
    schemaVersion: 2,
    passed: true,
    summary: {
      v12ProductionProofSchemaVersion: 2,
      v12ProductionProofSourceGitSha: "abc123",
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
      benchmarkGpuLabelEvidencePassed: true,
      benchmarkGpuVendor: "apple",
      benchmarkGpuDescription: "Apple M3",
      benchmarkWebGlRenderer: "ANGLE Metal Renderer: Apple M3",
      strictWebGpuPassed: true,
      cpuFallbackUsed: false,
      backendBrokerTraceCount: 1,
      backendBrokerSelectionPassed: true,
      backendBrokerSelectedBackendId: brokerSelection.backendId,
      backendBrokerSelectedModelId: brokerSelection.modelId,
      backendBrokerProductionRole: brokerSelection.productionRole,
      backendBrokerDeployReadyCandidate: brokerSelection.deployReadyCandidate,
      backendBrokerReason: brokerSelection.reason,
      backendBrokerDeployBackendId: "compiled-browser-webllm",
      backendBrokerKernelLabBackendId: "unlocked-browser-transformer",
      backendBrokerFallbackBackendId: "wasm-small-core",
      backendBrokerFallbackBackendCount: 1,
      backendBrokerFallbackDeployReadyCandidate: false,
      backendBrokerRoleBoundaryPassed: true,
    },
    runs: [
      {
        response: "Helena",
        expectedAnswerOnlyPassed: true,
        memoryGrounding: {
          mode: "seeded_browser_vector_context_rebuild",
          caseId: "montana_capital",
          corpusCount: 16,
          retrievedMemoryIds: ["bench_memory_montana_capital"],
          includedMemoryIds: ["bench_memory_montana_capital"],
          expectedMemoryIds: ["bench_memory_montana_capital"],
          expectedMemoryHitPassed: true,
          contextRebuildPassed: true,
          answerOnlyExpected: true,
          answerOnlyPassed: true,
          contextEstimatedTokens: 42,
          retrievalMs: 2,
          contextRebuildMs: 1,
          retrievalRank: 1,
          retrievalScore: 0.99,
          retrievalTopScoreMargin: 0.4,
        },
        runtimeTrace: {
          backend: "compiled-browser-webllm",
          brokerSelection,
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
        gpuLabelEvidencePassed: true,
        gpuVendor: "apple",
        gpuDescription: "Apple M3",
        webglRenderer: "ANGLE Metal Renderer: Apple M3",
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

  it("fails when summary booleans claim grounding but run-level retrieval evidence is missing", () => {
    const artifact = makePassingBrowserPreviewArtifact();
    delete (artifact.runs[0] as Record<string, unknown>).memoryGrounding;

    const report = evaluateHostedBenchmarkProof({ artifact });

    expect(report.passed).toBe(false);
    expect(report.blockers).toContain("Hosted benchmark proof requires concrete run-level memory grounding evidence.");
  });

  it("fails production proof when the hosted artifact lacks Backend Broker selection evidence", () => {
    const artifact = makePassingBrowserPreviewArtifact();
    delete (artifact.summary as Record<string, unknown>).backendBrokerTraceCount;
    delete (artifact.summary as Record<string, unknown>).backendBrokerSelectionPassed;
    delete (artifact.runs[0]?.runtimeTrace as Record<string, unknown>).brokerSelection;

    const report = evaluateHostedBenchmarkProof({ artifact });

    expect(report.passed).toBe(false);
    expect(report.blockers).toContain("Hosted benchmark proof requires Backend Broker selection evidence for compiled-browser-webllm.");
  });

  it("fails production proof when the hosted artifact lacks Backend Broker role-boundary evidence", () => {
    const artifact = makePassingBrowserPreviewArtifact();
    delete (artifact.summary as Record<string, unknown>).backendBrokerRoleBoundaryPassed;
    delete (artifact.summary as Record<string, unknown>).backendBrokerFallbackBackendId;

    const report = evaluateHostedBenchmarkProof({ artifact });

    expect(report.passed).toBe(false);
    expect(report.blockers).toContain("Hosted benchmark proof requires Backend Broker role-boundary evidence for compiled deploy, Kernel Lab, and fallback backends.");
  });

  it("fails production proof when the hosted artifact uses a stale v12 proof schema", () => {
    const artifact = makePassingBrowserPreviewArtifact();
    artifact.schemaVersion = 1;
    (artifact.summary as Record<string, unknown>).v12ProductionProofSchemaVersion = 1;

    const report = evaluateHostedBenchmarkProof({ artifact });

    expect(report.passed).toBe(false);
    expect(report.blockers).toContain("Hosted benchmark proof requires v12 production proof schema version 2.");
  });

  it("fails production proof when the hosted artifact lacks GPU label evidence", () => {
    const artifact = makePassingBrowserPreviewArtifact();
    delete (artifact.summary as Record<string, unknown>).benchmarkGpuLabelEvidencePassed;
    delete (artifact.summary as Record<string, unknown>).benchmarkGpuVendor;
    delete (artifact.summary as Record<string, unknown>).benchmarkGpuDescription;
    delete (artifact.summary as Record<string, unknown>).benchmarkWebGlRenderer;

    const report = evaluateHostedBenchmarkProof({ artifact });

    expect(report.passed).toBe(false);
    expect(report.blockers).toContain("Hosted benchmark proof requires browser GPU label evidence.");
  });

  it("fails production proof when the hosted artifact does not match the expected source commit", () => {
    const report = evaluateHostedBenchmarkProof({
      artifact: makePassingBrowserPreviewArtifact(),
      expectedSourceGitSha: "def456",
    });

    expect(report.passed).toBe(false);
    expect(report.blockers).toContain("Hosted benchmark proof source commit abc123 does not match expected commit def456.");
  });

  it("fails production proof when source binding is required but no expected source commit is provided", () => {
    const report = evaluateHostedBenchmarkProof({
      artifact: makePassingBrowserPreviewArtifact(),
      requireSourceBound: true,
    });

    expect(report.passed).toBe(false);
    expect(report.sourceBoundRequired).toBe(true);
    expect(report.blockers).toContain("Hosted benchmark proof requires an expected source commit when source binding is required.");
  });

  it("passes source-bound production proof when source binding is required and commits match", () => {
    const report = evaluateHostedBenchmarkProof({
      artifact: makePassingBrowserPreviewArtifact(),
      expectedSourceGitSha: "abc123",
      requireSourceBound: true,
    });

    expect(report.passed).toBe(true);
    expect(report.sourceBoundRequired).toBe(true);
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
        hostedBenchmarkConcreteMemoryGroundingPassed: true,
        hostedBenchmarkMemoryGroundingRunCount: 1,
        hostedBenchmarkMemoryGroundingCaseId: "montana_capital",
        hostedBenchmarkMemorySeededCorpusCount: 16,
        hostedBenchmarkMemoryRetrievedCount: 1,
        hostedBenchmarkMemoryIncludedCount: 1,
        hostedBenchmarkMemoryExpectedMemoryIdCount: 1,
        hostedBenchmarkMemoryExpectedHitMeanRank: 1,
        hostedBenchmarkMemoryExpectedHitMinTopScoreMargin: 0.4,
        hostedBenchmarkExpectedExactPassed: true,
        hostedBenchmarkProductionSpeedFloorPassed: true,
        hostedBenchmarkMeanTokensPerSecond: 2.7,
        hostedBenchmarkGpuLabelEvidencePassed: true,
        hostedBenchmarkGpuVendor: "apple",
        hostedBenchmarkGpuDescription: "Apple M3",
        hostedBenchmarkWebGlRenderer: "ANGLE Metal Renderer: Apple M3",
        hostedBenchmarkDirectModelFactualProofUsed: false,
        hostedBenchmarkTechnicalProofOnly: false,
        hostedBenchmarkCpuFallbackUsed: false,
        hostedBenchmarkStrictWebGpuPassed: true,
        hostedBenchmarkV12ProductionProofSchemaVersion: 2,
        hostedBenchmarkProofSourceGitSha: "abc123",
        hostedBenchmarkProofSourceBoundRequired: false,
        hostedBenchmarkBackendBrokerSelectionPassed: true,
        hostedBenchmarkBrokerSelectedBackendId: "compiled-browser-webllm",
        hostedBenchmarkBrokerSelectedModelId: "Qwen3-0.6B-q4f16_1-MLC",
        hostedBenchmarkBrokerProductionRole: "production_candidate",
        hostedBenchmarkBrokerDeployReadyCandidate: true,
        hostedBenchmarkBrokerDeployBackendId: "compiled-browser-webllm",
        hostedBenchmarkBrokerKernelLabBackendId: "unlocked-browser-transformer",
        hostedBenchmarkBrokerFallbackBackendId: "wasm-small-core",
        hostedBenchmarkBrokerFallbackBackendCount: 1,
        hostedBenchmarkBrokerFallbackDeployReadyCandidate: false,
        hostedBenchmarkBrokerRoleBoundaryPassed: true,
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
