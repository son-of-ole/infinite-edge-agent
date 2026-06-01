import { existsSync } from "node:fs";
import { copyFile, mkdir, readFile, stat } from "node:fs/promises";
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
  const timestamp = timestampToPathSegment(createdAt).slice(0, 15);
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
