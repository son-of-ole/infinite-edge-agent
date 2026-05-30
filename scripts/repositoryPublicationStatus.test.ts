import { mkdtemp, readFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import {
  buildRepositoryPublicationStatusArtifact,
  buildRepositoryPublicationHandoffArtifact,
  evaluateRepositoryPublicationSnapshot,
  writeRepositoryPublicationHandoffArtifact,
  writeRepositoryPublicationStatusArtifact,
} from "./repositoryPublicationStatus";

const expectedRemoteUrl = "https://github.com/son-of-ole/infinite-edge-agent.git";

describe("repository publication status", () => {
  it("is exposed as a package script for release handoff checks", async () => {
    const packageJson = JSON.parse(await readFile(join(process.cwd(), "package.json"), "utf8")) as {
      scripts?: Record<string, string>;
    };

    expect(packageJson.scripts?.["eval:repository-publication"]).toBe("node --import tsx scripts/repositoryPublicationStatus.ts");
    expect(packageJson.scripts?.["handoff:repository-publication"]).toBe("node --import tsx scripts/repositoryPublicationStatus.ts");
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

  it("passes as published in GitHub Actions detached checkout when github.sha is running on main", () => {
    const headSha = "1".repeat(40);
    const report = evaluateRepositoryPublicationSnapshot({
      expectedRemoteUrl,
      githubActions: {
        isActions: true,
        refName: "main",
        sha: headSha,
      },
      snapshot: {
        branch: "HEAD",
        headSha,
        upstream: null,
        remoteUrl: expectedRemoteUrl,
        aheadCount: null,
        behindCount: null,
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
      repositoryPublicationGithubActionsPublished: true,
      repositoryPublicationBranch: "HEAD",
      repositoryPublicationHeadSha: headSha,
    });
  });

  it("does not treat a detached GitHub Actions checkout from a non-main ref as published", () => {
    const headSha = "2".repeat(40);
    const report = evaluateRepositoryPublicationSnapshot({
      expectedRemoteUrl,
      githubActions: {
        isActions: true,
        refName: "feature/test",
        sha: headSha,
      },
      snapshot: {
        branch: "HEAD",
        headSha,
        upstream: null,
        remoteUrl: expectedRemoteUrl,
        aheadCount: null,
        behindCount: null,
        dirty: false,
        bundles: [],
      },
    });

    expect(report.passed).toBe(false);
    expect(report.published).toBe(false);
    expect(report.blockers).toContain("Repository publication status must be evaluated from main.");
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

  it("writes an operator-ready exact-history bundle handoff when local main is ahead", async () => {
    const artifactDir = await mkdtemp(join(tmpdir(), "repository-publication-handoff-"));
    const headSha = "f".repeat(40);
    const report = evaluateRepositoryPublicationSnapshot({
      expectedRemoteUrl,
      snapshot: {
        branch: "main",
        headSha,
        upstream: "origin/main",
        remoteUrl: expectedRemoteUrl,
        aheadCount: 80,
        behindCount: 0,
        dirty: false,
        bundles: [
          {
            kind: "ahead",
            path: "/private/tmp/infinite-edge-agent-main-ahead80.bundle",
            verified: true,
            headSha,
            completeHistory: false,
            requiredBaseSha: "1".repeat(40),
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

    const artifact = buildRepositoryPublicationHandoffArtifact(report, "2026-05-30T23:30:00.000Z");
    const written = await writeRepositoryPublicationHandoffArtifact(report, {
      artifactDir,
      createdAt: "2026-05-30T23:30:00.000Z",
    });

    expect(artifact).toMatchObject({
      name: "repository-publication-handoff",
      passed: true,
      summary: {
        repositoryPublicationPublished: false,
        repositoryPublicationBundleHandoffReady: true,
        repositoryPublicationHandoffCommandCount: 2,
      },
    });
    expect(artifact.markdown).toContain("Head SHA: `ffffffffffffffffffffffffffffffffffffffff`");
    expect(artifact.markdown).toContain("git clone /private/tmp/infinite-edge-agent-main-full.bundle infinite-edge-agent");
    expect(artifact.markdown).toContain("git merge --ff-only refs/remotes/bundle/main");
    expect(artifact.markdown).toContain("git push origin main");
    expect(artifact.markdown).toContain("## After Publish: Source-Bound V12 Production Proof");
    expect(artifact.markdown).toContain("git fetch origin main");
    expect(artifact.markdown).toContain("test \"$(git rev-parse HEAD)\" = \"$(git rev-parse origin/main)\"");
    expect(artifact.markdown).toContain("gh workflow run v12-production-proof.yml");
    expect(artifact.markdown).toContain("--repo son-of-ole/infinite-edge-agent");
    expect(artifact.markdown).toContain("--ref main");
    expect(artifact.markdown).toContain("-f deploy_url=\"$DEPLOY_URL\"");
    expect(artifact.markdown).toContain("gh run watch");
    expect(artifact.markdown).toContain("gh run download");
    expect(artifact.markdown).toContain("--dir .artifacts/evals/v12-production-proof");
    expect(artifact.markdown).toContain("EVAL_ARTIFACT_DIR=.artifacts/evals/v12-production-proof pnpm eval:v12-final-state");
    expect(written.latestJsonPath).toBe(join(artifactDir, "repository-publication-handoff-latest.json"));
    expect(written.latestMarkdownPath).toBe(join(artifactDir, "repository-publication-handoff-latest.md"));

    const latestMarkdown = await readFile(written.latestMarkdownPath, "utf8");
    expect(latestMarkdown).toContain("Exact-History GitHub Publication Handoff");
  });
});
