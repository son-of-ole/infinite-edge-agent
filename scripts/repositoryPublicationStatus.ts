import { existsSync } from "node:fs";
import { mkdir, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { execFile } from "node:child_process";
import { promisify } from "node:util";

const execFileAsync = promisify(execFile);

const EXPECTED_REMOTE_URL = "https://github.com/son-of-ole/infinite-edge-agent.git";

export type RepositoryPublicationBundleKind = "ahead" | "full";

export interface RepositoryPublicationBundleStatus {
  kind: RepositoryPublicationBundleKind;
  path: string;
  verified: boolean;
  headSha: string | null;
  completeHistory: boolean | null;
  requiredBaseSha?: string | null;
  error?: string | null;
}

export interface RepositoryPublicationSnapshot {
  branch: string | null;
  headSha: string | null;
  upstream: string | null;
  remoteUrl: string | null;
  aheadCount: number | null;
  behindCount: number | null;
  dirty: boolean;
  bundles: RepositoryPublicationBundleStatus[];
}

export interface RepositoryPublicationStatusReport {
  passed: boolean;
  published: boolean;
  bundleHandoffReady: boolean;
  blockers: string[];
  summary: Record<string, number | string | boolean | null>;
  snapshot: RepositoryPublicationSnapshot;
}

export interface RepositoryPublicationStatusArtifact {
  name: "repository-publication-status";
  createdAt: string;
  passed: boolean;
  summary: Record<string, number | string | boolean | null>;
  report: RepositoryPublicationStatusReport;
}

export interface RepositoryPublicationStatusWriteResult {
  artifact: RepositoryPublicationStatusArtifact;
  latestPath: string;
  resultPath: string;
}

export function evaluateRepositoryPublicationSnapshot(input: {
  snapshot: RepositoryPublicationSnapshot;
  expectedRemoteUrl?: string;
}): RepositoryPublicationStatusReport {
  const expectedRemoteUrl = input.expectedRemoteUrl ?? EXPECTED_REMOTE_URL;
  const snapshot = input.snapshot;
  const remoteMatches = snapshot.remoteUrl === expectedRemoteUrl;
  const onMain = snapshot.branch === "main";
  const clean = snapshot.dirty === false;
  const hasUpstream = Boolean(snapshot.upstream);
  const behindCount = snapshot.behindCount ?? Number.POSITIVE_INFINITY;
  const aheadCount = snapshot.aheadCount ?? Number.POSITIVE_INFINITY;
  const published = onMain
    && clean
    && remoteMatches
    && hasUpstream
    && aheadCount === 0
    && behindCount === 0;
  const matchingBundles = snapshot.bundles.filter((bundle) =>
    bundle.verified === true
    && Boolean(snapshot.headSha)
    && bundle.headSha === snapshot.headSha);
  const aheadBundleVerified = matchingBundles.some((bundle) => bundle.kind === "ahead");
  const fullBundleVerified = matchingBundles.some((bundle) => bundle.kind === "full" && bundle.completeHistory === true);
  const bundleHandoffReady = onMain
    && clean
    && remoteMatches
    && hasUpstream
    && behindCount === 0
    && aheadCount > 0
    && (aheadBundleVerified || fullBundleVerified);

  const blockers: string[] = [];
  if (!onMain) blockers.push("Repository publication status must be evaluated from main.");
  if (!clean) blockers.push("Repository has uncommitted changes.");
  if (!remoteMatches) blockers.push(`Repository origin remote must be ${expectedRemoteUrl}.`);
  if (!hasUpstream) blockers.push("Repository main branch must track an upstream branch.");
  if (behindCount > 0) blockers.push("Repository is behind its upstream branch.");
  if (!published && !bundleHandoffReady && aheadCount > 0 && behindCount === 0) {
    blockers.push("Repository has unpublished commits and no verified exact-history bundle contains the current head.");
  }

  return {
    passed: published || bundleHandoffReady,
    published,
    bundleHandoffReady,
    blockers,
    summary: {
      repositoryPublicationPassed: published || bundleHandoffReady,
      repositoryPublicationPublished: published,
      repositoryPublicationBundleHandoffReady: bundleHandoffReady,
      repositoryPublicationBranch: snapshot.branch,
      repositoryPublicationHeadSha: snapshot.headSha,
      repositoryPublicationUpstream: snapshot.upstream,
      repositoryPublicationRemoteUrl: snapshot.remoteUrl,
      repositoryPublicationExpectedRemoteUrl: expectedRemoteUrl,
      repositoryPublicationRemoteMatches: remoteMatches,
      repositoryPublicationAheadCount: Number.isFinite(aheadCount) ? aheadCount : null,
      repositoryPublicationBehindCount: Number.isFinite(behindCount) ? behindCount : null,
      repositoryPublicationDirty: snapshot.dirty,
      repositoryPublicationBundleCount: snapshot.bundles.length,
      repositoryPublicationAheadBundleVerified: aheadBundleVerified,
      repositoryPublicationFullBundleVerified: fullBundleVerified,
      repositoryPublicationBlockerCount: blockers.length,
    },
    snapshot,
  };
}

export async function evaluateRepositoryPublicationStatus(input: {
  rootDir?: string;
  expectedRemoteUrl?: string;
} = {}): Promise<RepositoryPublicationStatusReport> {
  const rootDir = input.rootDir ?? process.cwd();
  const snapshot = await collectRepositoryPublicationSnapshot(rootDir);
  return evaluateRepositoryPublicationSnapshot({
    snapshot,
    expectedRemoteUrl: input.expectedRemoteUrl,
  });
}

export function buildRepositoryPublicationStatusArtifact(
  report: RepositoryPublicationStatusReport,
  createdAt = new Date().toISOString(),
): RepositoryPublicationStatusArtifact {
  return {
    name: "repository-publication-status",
    createdAt,
    passed: report.passed,
    summary: report.summary,
    report,
  };
}

export async function writeRepositoryPublicationStatusArtifact(
  report: RepositoryPublicationStatusReport,
  options: { artifactDir?: string; createdAt?: string } = {},
): Promise<RepositoryPublicationStatusWriteResult> {
  const artifactDir = options.artifactDir ?? process.env.EVAL_ARTIFACT_DIR ?? ".artifacts/evals";
  const artifact = buildRepositoryPublicationStatusArtifact(report, options.createdAt);
  const runDir = join(artifactDir, "repository-publication-status");
  const timestamp = artifact.createdAt.replace(/[:.]/g, "-");
  const latestPath = join(artifactDir, "repository-publication-status-latest.json");
  const resultPath = join(runDir, `${timestamp}.json`);
  const json = `${JSON.stringify(artifact, null, 2)}\n`;

  await mkdir(runDir, { recursive: true });
  await writeFile(resultPath, json);
  await writeFile(latestPath, json);

  return { artifact, latestPath, resultPath };
}

async function collectRepositoryPublicationSnapshot(rootDir: string): Promise<RepositoryPublicationSnapshot> {
  const branch = await runGitOrNull(rootDir, ["rev-parse", "--abbrev-ref", "HEAD"]);
  const headSha = await runGitOrNull(rootDir, ["rev-parse", "HEAD"]);
  const upstream = await runGitOrNull(rootDir, ["rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}"]);
  const remoteUrl = await runGitOrNull(rootDir, ["remote", "get-url", "origin"]);
  const status = await runGitOrNull(rootDir, ["status", "--porcelain"]);
  const aheadBehind = upstream ? await runGitOrNull(rootDir, ["rev-list", "--left-right", "--count", "@{u}...HEAD"]) : null;
  const [behindCount, aheadCount] = parseAheadBehind(aheadBehind);
  const bundles = await collectBundleStatuses(rootDir, headSha, aheadCount);

  return {
    branch,
    headSha,
    upstream,
    remoteUrl,
    aheadCount,
    behindCount,
    dirty: Boolean(status?.trim()),
    bundles,
  };
}

async function collectBundleStatuses(
  rootDir: string,
  _headSha: string | null,
  aheadCount: number | null,
): Promise<RepositoryPublicationBundleStatus[]> {
  const candidates = resolveBundleCandidates(aheadCount);
  const statuses: RepositoryPublicationBundleStatus[] = [];
  for (const candidate of candidates) {
    statuses.push(await verifyBundle(rootDir, candidate.kind, candidate.path));
  }
  return statuses;
}

function resolveBundleCandidates(aheadCount: number | null): Array<{ kind: RepositoryPublicationBundleKind; path: string }> {
  const candidates: Array<{ kind: RepositoryPublicationBundleKind; path: string }> = [];
  const explicitAhead = process.env.REPOSITORY_PUBLICATION_AHEAD_BUNDLE_PATH;
  const explicitFull = process.env.REPOSITORY_PUBLICATION_FULL_BUNDLE_PATH;
  if (explicitAhead) candidates.push({ kind: "ahead", path: explicitAhead });
  if (explicitFull) candidates.push({ kind: "full", path: explicitFull });
  const defaultAhead = aheadCount && aheadCount > 0
    ? `/private/tmp/infinite-edge-agent-main-ahead${aheadCount}.bundle`
    : null;
  const defaultFull = "/private/tmp/infinite-edge-agent-main-full.bundle";
  const portableAhead = aheadCount && aheadCount > 0
    ? join(tmpdir(), `infinite-edge-agent-main-ahead${aheadCount}.bundle`)
    : null;
  const portableFull = join(tmpdir(), "infinite-edge-agent-main-full.bundle");
  for (const candidate of [
    defaultAhead ? { kind: "ahead" as const, path: defaultAhead } : null,
    portableAhead ? { kind: "ahead" as const, path: portableAhead } : null,
    { kind: "full" as const, path: defaultFull },
    { kind: "full" as const, path: portableFull },
  ]) {
    if (candidate && existsSync(candidate.path) && !candidates.some((existing) => existing.path === candidate.path)) {
      candidates.push(candidate);
    }
  }
  return candidates;
}

async function verifyBundle(
  rootDir: string,
  kind: RepositoryPublicationBundleKind,
  path: string,
): Promise<RepositoryPublicationBundleStatus> {
  if (!existsSync(path)) {
    return { kind, path, verified: false, headSha: null, completeHistory: null, error: "Bundle file does not exist." };
  }
  try {
    const output = await runGit(rootDir, ["bundle", "verify", path]);
    return {
      kind,
      path,
      verified: true,
      headSha: parseBundleHeadSha(output),
      completeHistory: output.includes("records a complete history"),
      requiredBaseSha: parseBundleRequiredBaseSha(output),
    };
  } catch (error) {
    return {
      kind,
      path,
      verified: false,
      headSha: null,
      completeHistory: null,
      error: error instanceof Error ? error.message : String(error),
    };
  }
}

function parseAheadBehind(value: string | null): [behindCount: number | null, aheadCount: number | null] {
  const parts = value?.trim().split(/\s+/) ?? [];
  if (parts.length !== 2) return [null, null];
  const behind = Number(parts[0]);
  const ahead = Number(parts[1]);
  return [
    Number.isFinite(behind) ? behind : null,
    Number.isFinite(ahead) ? ahead : null,
  ];
}

function parseBundleHeadSha(output: string): string | null {
  return output.match(/\b([0-9a-f]{40})\s+refs\/heads\/[^\s]+/i)?.[1] ?? null;
}

function parseBundleRequiredBaseSha(output: string): string | null {
  return output.match(/requires this ref:\s*\n([0-9a-f]{40})/i)?.[1] ?? null;
}

async function runGitOrNull(rootDir: string, args: string[]): Promise<string | null> {
  try {
    return (await runGit(rootDir, args)).trim();
  } catch {
    return null;
  }
}

async function runGit(rootDir: string, args: string[]): Promise<string> {
  const { stdout, stderr } = await execFileAsync("git", args, {
    cwd: rootDir,
    maxBuffer: 1024 * 1024 * 4,
  });
  return `${stdout}${stderr}`;
}

if (import.meta.url === `file://${process.argv[1]}`) {
  const report = await evaluateRepositoryPublicationStatus();
  await writeRepositoryPublicationStatusArtifact(report);
  console.log(JSON.stringify(report, null, 2));
  if (!report.passed) process.exitCode = 1;
}
