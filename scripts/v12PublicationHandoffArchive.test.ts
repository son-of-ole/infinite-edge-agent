import { execFile } from "node:child_process";
import { existsSync } from "node:fs";
import { mkdir, mkdtemp, readFile, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { promisify } from "node:util";
import { describe, expect, it } from "vitest";
import type { RepositoryPublicationStatusReport } from "./repositoryPublicationStatus";
import {
  createV12PublicationHandoffArchive,
  timestampToPathSegment,
} from "./v12PublicationHandoffArchive";

const execFileAsync = promisify(execFile);

function makeReport(rootDir: string): RepositoryPublicationStatusReport {
  return {
    passed: true,
    published: false,
    bundleHandoffReady: true,
    blockers: [],
    summary: {
      repositoryPublicationPassed: true,
      repositoryPublicationPublished: false,
      repositoryPublicationBundleHandoffReady: true,
      repositoryPublicationHeadSha: "1234567890abcdef1234567890abcdef12345678",
      repositoryPublicationAheadCount: 86,
      repositoryPublicationDirty: false,
    },
    snapshot: {
      branch: "main",
      headSha: "1234567890abcdef1234567890abcdef12345678",
      upstream: "origin/main",
      remoteUrl: "https://github.com/son-of-ole/infinite-edge-agent.git",
      aheadCount: 86,
      behindCount: 0,
      dirty: false,
      bundles: [
        {
          kind: "ahead",
          path: join(rootDir, "infinite-edge-agent-main-ahead86.bundle"),
          verified: true,
          headSha: "1234567890abcdef1234567890abcdef12345678",
          completeHistory: false,
          requiredBaseSha: "0".repeat(40),
        },
        {
          kind: "full",
          path: join(rootDir, "infinite-edge-agent-main-full.bundle"),
          verified: true,
          headSha: "1234567890abcdef1234567890abcdef12345678",
          completeHistory: true,
        },
      ],
    },
  };
}

describe("v12 publication handoff archive", () => {
  it("is exposed as a package script", async () => {
    const packageJson = JSON.parse(await readFile(join(process.cwd(), "package.json"), "utf8")) as {
      scripts?: Record<string, string>;
    };

    expect(packageJson.scripts?.["handoff:v12-publication"]).toBe("node --import tsx scripts/v12PublicationHandoffArchive.ts");
  });

  it("normalizes artifact timestamps for release-gate summary paths", () => {
    expect(timestampToPathSegment("2026-05-30T23:52:27.891Z")).toBe("2026-05-30T23-52-27-891Z");
  });

  it("packages verified bundles and release evidence into a portable tarball", async () => {
    const rootDir = await mkdtemp(join(tmpdir(), "v12-publication-handoff-root-"));
    const artifactDir = join(rootDir, ".artifacts", "evals");
    const releaseGateRunDir = join(artifactDir, "release-gate", "2026-05-30T23-52-27-891Z");
    const outputRoot = await mkdtemp(join(tmpdir(), "v12-publication-handoff-output-"));

    await mkdir(releaseGateRunDir, { recursive: true });
    await writeFile(join(rootDir, "infinite-edge-agent-main-ahead86.bundle"), "ahead bundle");
    await writeFile(join(rootDir, "infinite-edge-agent-main-full.bundle"), "full bundle");
    await writeFile(join(artifactDir, "repository-publication-handoff-latest.md"), "# handoff\n");
    await writeFile(join(artifactDir, "repository-publication-handoff-latest.json"), "{}\n");
    await writeFile(join(artifactDir, "repository-publication-status-latest.json"), "{}\n");
    await writeFile(join(artifactDir, "v12-final-state-status-latest.json"), "{}\n");
    await writeFile(join(artifactDir, "release-gate-latest.json"), JSON.stringify({
      passed: true,
      createdAt: "2026-05-30T23:52:27.891Z",
    }));
    await writeFile(join(releaseGateRunDir, "summary.md"), "# release gate\n");

    const result = await createV12PublicationHandoffArchive({
      artifactDir,
      outputRoot,
      repositoryPublication: makeReport(rootDir),
      createdAt: "2026-05-31T22:00:00.000Z",
    });

    expect(result.headSha).toBe("1234567890abcdef1234567890abcdef12345678");
    expect(result.archivePath.endsWith(".tar.gz")).toBe(true);
    expect(existsSync(result.archivePath)).toBe(true);
    expect(result.copiedFiles.map((file) => file.relativePath)).toEqual(expect.arrayContaining([
      "bundles/infinite-edge-agent-main-ahead86.bundle",
      "bundles/infinite-edge-agent-main-full.bundle",
      "artifacts/repository-publication-handoff-latest.md",
      "artifacts/release-gate-summary.md",
    ]));

    const { stdout } = await execFileAsync("tar", ["-tzf", result.archivePath]);
    expect(stdout).toContain("bundles/infinite-edge-agent-main-ahead86.bundle");
    expect(stdout).toContain("artifacts/release-gate-summary.md");
  });
});
