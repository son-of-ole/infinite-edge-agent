import { mkdtemp, readFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import type { RepositoryPublicationStatusReport } from "./repositoryPublicationStatus";
import type { V12ProductionArchiveArtifact } from "./v12ProductionArchive";
import type { V12ReadinessBundle } from "./v12ReadinessBundle";
import {
  buildV12FinalStateStatusArtifact,
  evaluateV12FinalStateStatus,
  writeV12FinalStateStatusArtifact,
} from "./v12FinalStateStatus";

function makeReadinessBundle(overrides: Partial<V12ReadinessBundle> = {}): V12ReadinessBundle {
  return {
    passed: true,
    blockers: [],
    deployBackendId: "compiled-browser-webllm",
    kernelLabBackendId: "unlocked-browser-transformer",
    fallbackBackendId: "wasm-small-core",
    backendRoleBoundaryPassed: true,
    requirements: [],
    hostedProfile: {} as V12ReadinessBundle["hostedProfile"],
    backendMatrix: {} as V12ReadinessBundle["backendMatrix"],
    sharedRuntime: {} as V12ReadinessBundle["sharedRuntime"],
    v12ProductionWorkflowPreflight: { passed: true, blockers: [], checks: [] },
    ...overrides,
  };
}

function makePublication(overrides: Partial<RepositoryPublicationStatusReport> = {}): RepositoryPublicationStatusReport {
  return {
    passed: true,
    published: true,
    bundleHandoffReady: false,
    blockers: [],
    summary: {
      repositoryPublicationPassed: true,
      repositoryPublicationPublished: true,
      repositoryPublicationBundleHandoffReady: false,
      repositoryPublicationHeadSha: "a".repeat(40),
      repositoryPublicationAheadCount: 0,
      repositoryPublicationBehindCount: 0,
      repositoryPublicationDirty: false,
    },
    snapshot: {
      branch: "main",
      headSha: "a".repeat(40),
      upstream: "origin/main",
      remoteUrl: "https://github.com/son-of-ole/infinite-edge-agent.git",
      aheadCount: 0,
      behindCount: 0,
      dirty: false,
      bundles: [],
    },
    ...overrides,
  };
}

function makeProductionArchive(overrides: Partial<V12ProductionArchiveArtifact> = {}): V12ProductionArchiveArtifact {
  return {
    name: "v12-production-archive",
    createdAt: "2026-05-30T23:40:00.000Z",
    passed: true,
    summary: {
      v12ProductionArchivePassed: true,
      v12ProductionDeployBackendId: "compiled-browser-webllm",
      v12ProductionKernelLabBackendId: "unlocked-browser-transformer",
      v12ProductionFallbackBackendId: "wasm-small-core",
      v12ProductionBackendRoleBoundaryPassed: true,
      v12ProductionHostedBenchmarkProofPassed: true,
      v12ProductionProofSourceBoundRequired: true,
      v12ProductionProofSourceBound: true,
      v12ProductionDeployReadyPassed: true,
      v12ProductionCompiledBackendReadyPassed: true,
      v12ProductionMemoryGroundingPassed: true,
      v12ProductionExpectedExactPassed: true,
      v12ProductionSpeedFloorPassed: true,
      v12ProductionStrictWebGpuPassed: true,
      v12ProductionCpuFallbackUsed: false,
      v12ProductionBackendBrokerSelectionPassed: true,
      v12ProductionBrokerRoleBoundaryPassed: true,
      v12ProductionProofSourceGitSha: "a".repeat(40),
    },
    archive: {
      passed: true,
      blockers: [],
      suiteLatestPath: ".artifacts/evals/v12-readiness-suite-latest.json",
      suiteResultPath: ".artifacts/evals/v12-readiness-suite/2026.json",
      suiteResult: {} as V12ProductionArchiveArtifact["archive"]["suiteResult"],
    },
    ...overrides,
  };
}

describe("v12 final state status", () => {
  it("is exposed as a package script for final-state release checks", async () => {
    const packageJson = JSON.parse(await readFile(join(process.cwd(), "package.json"), "utf8")) as {
      scripts?: Record<string, string>;
    };

    expect(packageJson.scripts?.["eval:v12-final-state"]).toBe("node --import tsx scripts/v12FinalStateStatus.ts");
  });

  it("passes only when architecture, GitHub publication, and hosted source-bound production proof all pass", () => {
    const status = evaluateV12FinalStateStatus({
      readinessBundle: makeReadinessBundle(),
      repositoryPublication: makePublication(),
      productionArchive: makeProductionArchive(),
    });

    expect(status).toMatchObject({
      passed: true,
      blockers: [],
      nextAction: "ready",
      deployBackendId: "compiled-browser-webllm",
      kernelLabBackendId: "unlocked-browser-transformer",
      fallbackBackendId: "wasm-small-core",
    });
    expect(status.requirements).toEqual(expect.arrayContaining([
      expect.objectContaining({ id: "v12_architecture_readiness", passed: true }),
      expect.objectContaining({ id: "source_published_to_github", passed: true }),
      expect.objectContaining({ id: "source_bound_hosted_production_archive", passed: true }),
      expect.objectContaining({ id: "backend_specific_production_evidence", passed: true }),
    ]));
  });

  it("does not treat exact-history bundle handoff as published production state", () => {
    const status = evaluateV12FinalStateStatus({
      readinessBundle: makeReadinessBundle(),
      repositoryPublication: makePublication({
        published: false,
        bundleHandoffReady: true,
        summary: {
          repositoryPublicationPassed: true,
          repositoryPublicationPublished: false,
          repositoryPublicationBundleHandoffReady: true,
          repositoryPublicationHeadSha: "b".repeat(40),
          repositoryPublicationAheadCount: 81,
          repositoryPublicationBehindCount: 0,
          repositoryPublicationDirty: false,
        },
        snapshot: {
          branch: "main",
          headSha: "b".repeat(40),
          upstream: "origin/main",
          remoteUrl: "https://github.com/son-of-ole/infinite-edge-agent.git",
          aheadCount: 81,
          behindCount: 0,
          dirty: false,
          bundles: [],
        },
      }),
      productionArchive: makeProductionArchive(),
    });

    expect(status.passed).toBe(false);
    expect(status.nextAction).toBe("publish_source_history");
    expect(status.blockers).toContain("source_published_to_github: Source history is only bundle-handoff ready; publish main to origin/main before claiming final state.");
    expect(status.summary).toMatchObject({
      v12FinalStatePassed: false,
      v12FinalStateSourcePublished: false,
      v12FinalStateExactHistoryHandoffReady: true,
      v12FinalStateRepositoryAheadCount: 81,
    });
  });

  it("prioritizes publishing source history when exact-history handoff is ready even if hosted env is not configured", () => {
    const status = evaluateV12FinalStateStatus({
      readinessBundle: makeReadinessBundle({
        passed: false,
        deployBackendId: null,
        backendRoleBoundaryPassed: false,
        blockers: ["Hosted deployment profile did not pass."],
      }),
      repositoryPublication: makePublication({
        published: false,
        bundleHandoffReady: true,
        snapshot: {
          branch: "main",
          headSha: "c".repeat(40),
          upstream: "origin/main",
          remoteUrl: "https://github.com/son-of-ole/infinite-edge-agent.git",
          aheadCount: 82,
          behindCount: 0,
          dirty: false,
          bundles: [],
        },
      }),
      productionArchive: null,
    });

    expect(status.passed).toBe(false);
    expect(status.nextAction).toBe("publish_source_history");
  });

  it("requires a source-bound hosted production archive after architecture and source publication are ready", () => {
    const status = evaluateV12FinalStateStatus({
      readinessBundle: makeReadinessBundle(),
      repositoryPublication: makePublication(),
      productionArchive: null,
    });

    expect(status.passed).toBe(false);
    expect(status.nextAction).toBe("run_hosted_production_proof");
    expect(status.blockers).toContain("source_bound_hosted_production_archive: Missing v12 production archive with hosted benchmark proof.");
  });

  it("builds and writes a release-friendly final-state artifact", async () => {
    const artifactDir = await mkdtemp(join(tmpdir(), "v12-final-state-status-"));
    const status = evaluateV12FinalStateStatus({
      readinessBundle: makeReadinessBundle(),
      repositoryPublication: makePublication(),
      productionArchive: makeProductionArchive(),
    });
    const artifact = buildV12FinalStateStatusArtifact(status, "2026-05-30T23:45:00.000Z");
    const written = await writeV12FinalStateStatusArtifact(status, {
      artifactDir,
      createdAt: "2026-05-30T23:45:00.000Z",
    });

    expect(artifact).toMatchObject({
      name: "v12-final-state-status",
      passed: true,
      summary: {
        v12FinalStatePassed: true,
        v12FinalStateDeployBackendId: "compiled-browser-webllm",
        v12FinalStateKernelLabBackendId: "unlocked-browser-transformer",
        v12FinalStateFallbackBackendId: "wasm-small-core",
        v12FinalStateNextAction: "ready",
      },
    });
    expect(written.latestPath).toBe(join(artifactDir, "v12-final-state-status-latest.json"));
    expect(written.resultPath).toBe(join(artifactDir, "v12-final-state-status", "2026-05-30T23-45-00-000Z.json"));

    const latest = JSON.parse(await readFile(written.latestPath, "utf8")) as ReturnType<typeof buildV12FinalStateStatusArtifact>;
    expect(latest.summary.v12FinalStatePassed).toBe(true);
  });
});
