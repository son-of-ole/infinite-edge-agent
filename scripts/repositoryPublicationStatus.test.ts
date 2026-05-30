import { mkdtemp, readFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import {
  buildRepositoryPublicationStatusArtifact,
  evaluateRepositoryPublicationSnapshot,
  writeRepositoryPublicationStatusArtifact,
} from "./repositoryPublicationStatus";

const expectedRemoteUrl = "https://github.com/son-of-ole/infinite-edge-agent.git";

describe("repository publication status", () => {
  it("is exposed as a package script for release handoff checks", async () => {
    const packageJson = JSON.parse(await readFile(join(process.cwd(), "package.json"), "utf8")) as {
      scripts?: Record<string, string>;
    };

    expect(packageJson.scripts?.["eval:repository-publication"]).toBe("node --import tsx scripts/repositoryPublicationStatus.ts");
  });

  it("passes as published only when clean local history matches the upstream remote", () => {
    const report = evaluateRepositoryPublicationSnapshot({
      expectedRemoteUrl,
      snapshot: {
        branch: "main",
        headSha: "a".repeat(40),
        upstream: "origin/main",
        remoteUrl: expectedRemoteUrl,
        aheadCount: 0,
        behindCount: 0,
        dirty: false,
        bundles: [],
      },
    });

    expect(report.passed).toBe(true);
    expect(report.published).toBe(true);
    expect(report.bundleHandoffReady).toBe(false);
    expect(report.blockers).toEqual([]);
    expect(report.summary).toMatchObject({
      repositoryPublicationPublished: true,
      repositoryPublicationAheadCount: 0,
      repositoryPublicationDirty: false,
    });
  });

  it("passes as exact-history handoff when local history is ahead but verified bundles contain the current head", () => {
    const headSha = "b".repeat(40);
    const report = evaluateRepositoryPublicationSnapshot({
      expectedRemoteUrl,
      snapshot: {
        branch: "main",
        headSha,
        upstream: "origin/main",
        remoteUrl: expectedRemoteUrl,
        aheadCount: 78,
        behindCount: 0,
        dirty: false,
        bundles: [
          {
            kind: "ahead",
            path: "/private/tmp/infinite-edge-agent-main-ahead78.bundle",
            verified: true,
            headSha,
            completeHistory: false,
          },
          {
            kind: "full",
            path: "/private/tmp/infinite-edge-agent-main-full.bundle",
            verified: true,
            headSha,
            completeHistory: true,
          },
        ],
      },
    });

    expect(report.passed).toBe(true);
    expect(report.published).toBe(false);
    expect(report.bundleHandoffReady).toBe(true);
    expect(report.blockers).toEqual([]);
    expect(report.summary).toMatchObject({
      repositoryPublicationPublished: false,
      repositoryPublicationBundleHandoffReady: true,
      repositoryPublicationAheadCount: 78,
      repositoryPublicationAheadBundleVerified: true,
      repositoryPublicationFullBundleVerified: true,
    });
  });

  it("fails when local history is ahead and no verified bundle matches the current head", () => {
    const report = evaluateRepositoryPublicationSnapshot({
      expectedRemoteUrl,
      snapshot: {
        branch: "main",
        headSha: "c".repeat(40),
        upstream: "origin/main",
        remoteUrl: expectedRemoteUrl,
        aheadCount: 3,
        behindCount: 0,
        dirty: false,
        bundles: [
          {
            kind: "ahead",
            path: "/private/tmp/stale.bundle",
            verified: true,
            headSha: "d".repeat(40),
            completeHistory: false,
          },
        ],
      },
    });

    expect(report.passed).toBe(false);
    expect(report.published).toBe(false);
    expect(report.bundleHandoffReady).toBe(false);
    expect(report.blockers).toContain("Repository has unpublished commits and no verified exact-history bundle contains the current head.");
  });

  it("builds and writes a release-friendly publication status artifact", async () => {
    const artifactDir = await mkdtemp(join(tmpdir(), "repository-publication-status-"));
    const report = evaluateRepositoryPublicationSnapshot({
      expectedRemoteUrl,
      snapshot: {
        branch: "main",
        headSha: "e".repeat(40),
        upstream: "origin/main",
        remoteUrl: expectedRemoteUrl,
        aheadCount: 0,
        behindCount: 0,
        dirty: false,
        bundles: [],
      },
    });
    const artifact = buildRepositoryPublicationStatusArtifact(report, "2026-05-30T23:20:00.000Z");
    const written = await writeRepositoryPublicationStatusArtifact(report, {
      artifactDir,
      createdAt: "2026-05-30T23:20:00.000Z",
    });

    expect(artifact).toMatchObject({
      name: "repository-publication-status",
      passed: true,
      summary: {
        repositoryPublicationPassed: true,
        repositoryPublicationPublished: true,
      },
    });
    expect(written.latestPath).toBe(join(artifactDir, "repository-publication-status-latest.json"));
    expect(written.resultPath).toBe(join(artifactDir, "repository-publication-status", "2026-05-30T23-20-00-000Z.json"));

    const latest = JSON.parse(await readFile(written.latestPath, "utf8")) as { name: string; passed: boolean };
    expect(latest.name).toBe("repository-publication-status");
    expect(latest.passed).toBe(true);
  });
});
