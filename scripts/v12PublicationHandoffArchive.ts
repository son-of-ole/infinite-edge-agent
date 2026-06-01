import { existsSync } from "node:fs";
import { copyFile, mkdir, readFile, stat, writeFile } from "node:fs/promises";
import { basename, dirname, join } from "node:path";
import { execFile } from "node:child_process";
import { promisify } from "node:util";
import {
  evaluateRepositoryPublicationStatus,
  writeRepositoryPublicationHandoffArtifact,
  writeRepositoryPublicationStatusArtifact,
  type RepositoryPublicationBundleStatus,
  type RepositoryPublicationStatusReport,
} from "./repositoryPublicationStatus";

const execFileAsync = promisify(execFile);

export interface V12PublicationHandoffCopiedFile {
  sourcePath: string;
  relativePath: string;
  bytes: number;
}

export interface V12PublicationHandoffArchiveResult {
  archivePath: string;
  directoryPath: string;
  headSha: string;
  copiedFiles: V12PublicationHandoffCopiedFile[];
}

export async function createV12PublicationHandoffArchive(options: {
  rootDir?: string;
  artifactDir?: string;
  outputRoot?: string;
  createdAt?: string;
  repositoryPublication?: RepositoryPublicationStatusReport;
} = {}): Promise<V12PublicationHandoffArchiveResult> {
  const rootDir = options.rootDir ?? process.cwd();
  const artifactDir = options.artifactDir ?? process.env.EVAL_ARTIFACT_DIR ?? join(rootDir, ".artifacts", "evals");
  const outputRoot = options.outputRoot ?? process.env.V12_PUBLICATION_HANDOFF_OUTPUT_ROOT ?? "/private/tmp";
  const createdAt = options.createdAt ?? new Date().toISOString();
  const report = options.repositoryPublication ?? await evaluateRepositoryPublicationStatus({ rootDir });

  if (!report.passed || !report.bundleHandoffReady) {
    throw new Error("V12 publication handoff archive requires a passing exact-history bundle handoff.");
  }
  const headSha = report.snapshot.headSha;
  if (!headSha) {
    throw new Error("V12 publication handoff archive requires a repository head SHA.");
  }

  await writeRepositoryPublicationStatusArtifact(report, { artifactDir, createdAt });
  await writeRepositoryPublicationHandoffArtifact(report, { artifactDir, createdAt });

  const shortSha = headSha.slice(0, 7);
  const timestamp = timestampToPathSegment(createdAt);
  const archiveName = `infinite-edge-agent-publication-handoff-${shortSha}-${timestamp}`;
  const directoryPath = join(outputRoot, archiveName);
  const archivePath = `${directoryPath}.tar.gz`;
  const copiedFiles: V12PublicationHandoffCopiedFile[] = [];

  await mkdir(join(directoryPath, "bundles"), { recursive: true });
  await mkdir(join(directoryPath, "artifacts"), { recursive: true });

  for (const bundle of matchingVerifiedBundles(report)) {
    copiedFiles.push(await copyIntoArchive(bundle.path, join("bundles", basename(bundle.path)), directoryPath));
  }

  for (const artifact of await resolveArtifactSources(artifactDir)) {
    copiedFiles.push(await copyIntoArchive(artifact.sourcePath, join("artifacts", artifact.outputName), directoryPath));
  }
  copiedFiles.push(...await writeTopLevelHandoffFiles({
    directoryPath,
    report,
    archiveName,
    copiedFiles,
  }));

  await execFileAsync("tar", ["-czf", archivePath, "-C", outputRoot, archiveName], {
    maxBuffer: 1024 * 1024 * 4,
  });

  return {
    archivePath,
    directoryPath,
    headSha,
    copiedFiles,
  };
}

async function writeTopLevelHandoffFiles(input: {
  directoryPath: string;
  report: RepositoryPublicationStatusReport;
  archiveName: string;
  copiedFiles: V12PublicationHandoffCopiedFile[];
}): Promise<V12PublicationHandoffCopiedFile[]> {
  const headSha = input.report.snapshot.headSha ?? "unknown";
  const aheadCount = input.report.snapshot.aheadCount ?? null;
  const readmePath = join(input.directoryPath, "README.md");
  const manifestPath = join(input.directoryPath, "handoff-manifest.json");
  const files = input.copiedFiles.map((file) => ({
    relativePath: file.relativePath,
    bytes: file.bytes,
  }));
  const readme = [
    "# Exact-History V12 Publication Handoff",
    "",
    `Archive: \`${input.archiveName}\``,
    `Head SHA: \`${headSha}\``,
    `Ahead count: \`${aheadCount ?? "unknown"}\``,
    "",
    "This archive carries the verified Git bundles and release evidence needed to publish the exact local history from a machine that can reach GitHub.",
    "",
    "## Publish",
    "",
    "```bash",
    "git clone bundles/infinite-edge-agent-main-full.bundle infinite-edge-agent",
    "cd infinite-edge-agent",
    "git remote set-url origin https://github.com/son-of-ole/infinite-edge-agent.git",
    "git push origin main",
    "```",
    "",
    "## Source-Bound Production Proof",
    "",
    "After pushing, deploy this same commit and run the GitHub production proof workflow. The detailed commands are in `artifacts/repository-publication-handoff-latest.md`.",
    "",
    "```bash",
    "gh workflow run v12-production-proof.yml --repo son-of-ole/infinite-edge-agent --ref main -f deploy_url=\"$DEPLOY_URL\"",
    "gh run watch --repo son-of-ole/infinite-edge-agent --workflow v12-production-proof.yml --exit-status",
    "gh run download --repo son-of-ole/infinite-edge-agent --name v12-production-proof-artifacts --dir .artifacts/evals/v12-production-proof",
    "EVAL_ARTIFACT_DIR=.artifacts/evals/v12-production-proof pnpm eval:v12-final-state",
    "```",
    "",
  ].join("\n");
  const manifest = {
    name: "v12-publication-handoff",
    archiveName: input.archiveName,
    headSha,
    aheadCount,
    published: input.report.published,
    bundleHandoffReady: input.report.bundleHandoffReady,
    files,
  };

  await writeFile(readmePath, readme);
  await writeFile(manifestPath, `${JSON.stringify(manifest, null, 2)}\n`);

  return [
    await describeWrittenFile(readmePath, "README.md"),
    await describeWrittenFile(manifestPath, "handoff-manifest.json"),
  ];
}

async function describeWrittenFile(sourcePath: string, relativePath: string): Promise<V12PublicationHandoffCopiedFile> {
  const copied = await stat(sourcePath);
  return {
    sourcePath,
    relativePath,
    bytes: copied.size,
  };
}

export function timestampToPathSegment(value: string): string {
  return value.replace(/[:.]/g, "-");
}

function matchingVerifiedBundles(report: RepositoryPublicationStatusReport): RepositoryPublicationBundleStatus[] {
  const headSha = report.snapshot.headSha;
  return report.snapshot.bundles.filter((bundle) =>
    bundle.verified === true
    && Boolean(headSha)
    && bundle.headSha === headSha
    && existsSync(bundle.path));
}

async function resolveArtifactSources(artifactDir: string): Promise<Array<{ sourcePath: string; outputName: string }>> {
  const releaseGateLatestPath = join(artifactDir, "release-gate-latest.json");
  const sources = [
    { sourcePath: join(artifactDir, "repository-publication-handoff-latest.md"), outputName: "repository-publication-handoff-latest.md" },
    { sourcePath: join(artifactDir, "repository-publication-handoff-latest.json"), outputName: "repository-publication-handoff-latest.json" },
    { sourcePath: join(artifactDir, "repository-publication-status-latest.json"), outputName: "repository-publication-status-latest.json" },
    { sourcePath: join(artifactDir, "v12-final-state-status-latest.json"), outputName: "v12-final-state-status-latest.json" },
    { sourcePath: releaseGateLatestPath, outputName: "release-gate-latest.json" },
  ];
  const releaseGateSummaryPath = await resolveReleaseGateSummaryPath(artifactDir, releaseGateLatestPath);
  if (releaseGateSummaryPath) {
    sources.push({ sourcePath: releaseGateSummaryPath, outputName: "release-gate-summary.md" });
  }
  return sources.filter((source) => existsSync(source.sourcePath));
}

async function resolveReleaseGateSummaryPath(
  artifactDir: string,
  releaseGateLatestPath: string,
): Promise<string | null> {
  if (!existsSync(releaseGateLatestPath)) return null;
  const parsed = JSON.parse(await readFile(releaseGateLatestPath, "utf8")) as { createdAt?: string };
  if (!parsed.createdAt) return null;
  const summaryPath = join(artifactDir, "release-gate", timestampToPathSegment(parsed.createdAt), "summary.md");
  return existsSync(summaryPath) ? summaryPath : null;
}

async function copyIntoArchive(
  sourcePath: string,
  relativePath: string,
  directoryPath: string,
): Promise<V12PublicationHandoffCopiedFile> {
  const destination = join(directoryPath, relativePath);
  await mkdir(dirname(destination), { recursive: true });
  await copyFile(sourcePath, destination);
  const copied = await stat(destination);
  return {
    sourcePath,
    relativePath,
    bytes: copied.size,
  };
}

if (import.meta.url === `file://${process.argv[1]}`) {
  try {
    const result = await createV12PublicationHandoffArchive();
    console.log(JSON.stringify(result, null, 2));
  } catch (error) {
    console.error(error instanceof Error ? error.message : String(error));
    process.exitCode = 1;
  }
}
