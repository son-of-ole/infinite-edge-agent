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

export interface RepositoryPublicationGithubActionsContext {
  isActions: boolean;
  refName: string | null;
  sha: string | null;
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

export interface RepositoryPublicationHandoffArtifact {
  name: "repository-publication-handoff";
  createdAt: string;
  passed: boolean;
  summary: Record<string, number | string | boolean | null>;
  report: RepositoryPublicationStatusReport;
  markdown: string;
}

export interface RepositoryPublicationStatusWriteResult {
  artifact: RepositoryPublicationStatusArtifact;
  latestPath: string;
  resultPath: string;
}

export interface RepositoryPublicationHandoffWriteResult {
  artifact: RepositoryPublicationHandoffArtifact;
  latestJsonPath: string;
  resultJsonPath: string;
  latestMarkdownPath: string;
  resultMarkdownPath: string;
}

export function evaluateRepositoryPublicationSnapshot(input: {
  snapshot: RepositoryPublicationSnapshot;
  expectedRemoteUrl?: string;
  githubActions?: RepositoryPublicationGithubActionsContext;
}): RepositoryPublicationStatusReport {
  const expectedRemoteUrl = input.expectedRemoteUrl ?? EXPECTED_REMOTE_URL;
  const snapshot = input.snapshot;
  const githubActions = input.githubActions;
  const remoteMatches = snapshot.remoteUrl === expectedRemoteUrl;
  const githubActionsPublished = githubActions?.isActions === true
    && githubActions.refName === "main"
    && Boolean(snapshot.headSha)
    && githubActions.sha === snapshot.headSha
    && remoteMatches
    && snapshot.dirty === false;
  const onMain = snapshot.branch === "main" || githubActionsPublished;
  const clean = snapshot.dirty === false;
  const hasUpstream = Boolean(snapshot.upstream) || githubActionsPublished;
  const behindCount = snapshot.behindCount ?? Number.POSITIVE_INFINITY;
  const aheadCount = snapshot.aheadCount ?? Number.POSITIVE_INFINITY;
  const locallyPublished = snapshot.branch === "main"
    && clean
    && remoteMatches
    && Boolean(snapshot.upstream)
    && aheadCount === 0
    && behindCount === 0;
  const published = locallyPublished || githubActionsPublished;
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
  if (!githubActionsPublished && behindCount > 0) blockers.push("Repository is behind its upstream branch.");
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
      repositoryPublicationGithubActionsPublished: githubActionsPublished,
      repositoryPublicationGithubActionsRefName: githubActions?.refName ?? null,
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
  githubActions?: RepositoryPublicationGithubActionsContext;
} = {}): Promise<RepositoryPublicationStatusReport> {
  const rootDir = input.rootDir ?? process.cwd();
  const snapshot = await collectRepositoryPublicationSnapshot(rootDir);
  return evaluateRepositoryPublicationSnapshot({
    snapshot,
    expectedRemoteUrl: input.expectedRemoteUrl,
    githubActions: input.githubActions ?? readGithubActionsContext(process.env),
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

export function buildRepositoryPublicationHandoffArtifact(
  report: RepositoryPublicationStatusReport,
  createdAt = new Date().toISOString(),
): RepositoryPublicationHandoffArtifact {
  const handoffCommandCount = countHandoffCommandGroups(report);
  return {
    name: "repository-publication-handoff",
    createdAt,
    passed: report.passed,
    summary: {
      ...report.summary,
      repositoryPublicationHandoffCommandCount: handoffCommandCount,
    },
    report,
    markdown: buildRepositoryPublicationHandoffMarkdown(report, {
      createdAt,
      handoffCommandCount,
    }),
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

export async function writeRepositoryPublicationHandoffArtifact(
  report: RepositoryPublicationStatusReport,
  options: { artifactDir?: string; createdAt?: string } = {},
): Promise<RepositoryPublicationHandoffWriteResult> {
  const artifactDir = options.artifactDir ?? process.env.EVAL_ARTIFACT_DIR ?? ".artifacts/evals";
  const artifact = buildRepositoryPublicationHandoffArtifact(report, options.createdAt);
  const runDir = join(artifactDir, "repository-publication-handoff");
  const timestamp = artifact.createdAt.replace(/[:.]/g, "-");
  const latestJsonPath = join(artifactDir, "repository-publication-handoff-latest.json");
  const resultJsonPath = join(runDir, `${timestamp}.json`);
  const latestMarkdownPath = join(artifactDir, "repository-publication-handoff-latest.md");
  const resultMarkdownPath = join(runDir, `${timestamp}.md`);
  const json = `${JSON.stringify(artifact, null, 2)}\n`;

  await mkdir(runDir, { recursive: true });
  await writeFile(resultJsonPath, json);
  await writeFile(latestJsonPath, json);
  await writeFile(resultMarkdownPath, artifact.markdown);
  await writeFile(latestMarkdownPath, artifact.markdown);

  return { artifact, latestJsonPath, resultJsonPath, latestMarkdownPath, resultMarkdownPath };
}

function countHandoffCommandGroups(report: RepositoryPublicationStatusReport): number {
  if (!report.bundleHandoffReady) return 0;
  const matchingBundles = getMatchingVerifiedBundles(report);
  const hasFullBundle = matchingBundles.some((bundle) => bundle.kind === "full" && bundle.completeHistory === true);
  const hasAheadBundle = matchingBundles.some((bundle) => bundle.kind === "ahead");
  return Number(hasFullBundle) + Number(hasAheadBundle);
}

function buildRepositoryPublicationHandoffMarkdown(
  report: RepositoryPublicationStatusReport,
  options: { createdAt: string; handoffCommandCount: number },
): string {
  const snapshot = report.snapshot;
  const remoteUrl = String(report.summary.repositoryPublicationExpectedRemoteUrl ?? EXPECTED_REMOTE_URL);
  const matchingBundles = getMatchingVerifiedBundles(report);
  const fullBundle = matchingBundles.find((bundle) => bundle.kind === "full" && bundle.completeHistory === true);
  const aheadBundle = matchingBundles.find((bundle) => bundle.kind === "ahead");
  const lines = [
    "# Exact-History GitHub Publication Handoff",
    "",
    `Created: ${options.createdAt}`,
    `Repository: \`${remoteUrl}\``,
    `Branch: \`${snapshot.branch ?? "unknown"}\``,
    `Head SHA: \`${snapshot.headSha ?? "unknown"}\``,
    `Upstream: \`${snapshot.upstream ?? "unknown"}\``,
    `Ahead count: \`${snapshot.aheadCount ?? "unknown"}\``,
    `Behind count: \`${snapshot.behindCount ?? "unknown"}\``,
    `Published: \`${report.published}\``,
    `Bundle handoff ready: \`${report.bundleHandoffReady}\``,
    "",
  ];

  if (report.published) {
    lines.push("The local branch already matches the upstream remote. No bundle handoff is required.", "");
  } else if (!report.bundleHandoffReady) {
    lines.push("Publication handoff is not ready.", "");
    if (report.blockers.length > 0) {
      lines.push("## Blockers", "");
      for (const blocker of report.blockers) lines.push(`- ${blocker}`);
      lines.push("");
    }
  } else {
    lines.push("Use one of the bundle paths below to publish the exact local Git history. Do not recreate the repository with GitHub contents API commits; that would flatten the verified local history.", "");
    lines.push("## Direct Push", "", "Try this first from a network that can resolve GitHub:", "", "```bash", "git push origin main", "```", "");
    if (fullBundle) {
      lines.push("## Full Bundle Restore", "", "Use this on another machine when you want a complete standalone clone from the verified bundle:", "", "```bash");
      lines.push(`git clone ${fullBundle.path} infinite-edge-agent`);
      lines.push("cd infinite-edge-agent");
      lines.push(`git remote set-url origin ${remoteUrl}`);
      lines.push("git push origin main");
      lines.push("```", "");
    }
    if (aheadBundle) {
      lines.push("## Existing Clone Fast-Forward", "", "Use this inside an existing clone whose `origin/main` contains the required base commit:", "", "```bash");
      lines.push(`git fetch ${aheadBundle.path} main:refs/remotes/bundle/main`);
      lines.push("git merge --ff-only refs/remotes/bundle/main");
      lines.push("git push origin main");
      lines.push("```", "");
    }
  }

  if (report.published || report.bundleHandoffReady) {
    lines.push(
      "## After Publish: Source-Bound V12 Production Proof",
      "",
      "After the exact Git history is on GitHub, verify that `origin/main` resolves to this head and trigger the hosted production proof against the deployed public URL.",
      "",
      "Prerequisites:",
      "",
      "- GitHub Actions secrets `BENCHMARK_TELEMETRY_DATABASE_URL` and `BENCHMARK_TELEMETRY_ADMIN_TOKEN` are configured.",
      "- The hosted app was deployed from this same commit and emits `VITE_GIT_SHA` in benchmark artifacts.",
      "- `DEPLOY_URL` is the public HTTPS origin for the hosted app, not localhost.",
      "",
      "```bash",
      "git fetch origin main",
      "test \"$(git rev-parse HEAD)\" = \"$(git rev-parse origin/main)\"",
      "",
      "DEPLOY_URL=\"https://<your-hosted-app>\"",
      "gh workflow run v12-production-proof.yml \\",
      "  --repo son-of-ole/infinite-edge-agent \\",
      "  --ref main \\",
      "  -f deploy_url=\"$DEPLOY_URL\"",
      "",
      "gh run watch --repo son-of-ole/infinite-edge-agent --workflow v12-production-proof.yml --exit-status",
      "gh run download --repo son-of-ole/infinite-edge-agent --name v12-production-proof-artifacts --dir .artifacts/evals/v12-production-proof",
      "EVAL_ARTIFACT_DIR=.artifacts/evals/v12-production-proof pnpm eval:v12-final-state",
      "```",
      "",
      "The final-state artifact should only pass after the source is published and the hosted proof is source-bound to the same commit SHA.",
      "",
    );
  }

  lines.push("## Verified Bundles", "");
  for (const bundle of snapshot.bundles) {
    lines.push(`- \`${bundle.kind}\`: \`${bundle.path}\`, verified=\`${bundle.verified}\`, head=\`${bundle.headSha ?? "unknown"}\`, completeHistory=\`${bundle.completeHistory ?? "unknown"}\``);
  }
  lines.push("", `Handoff command groups: \`${options.handoffCommandCount}\``, "");
  return `${lines.join("\n")}\n`;
}

function getMatchingVerifiedBundles(report: RepositoryPublicationStatusReport): RepositoryPublicationBundleStatus[] {
  const headSha = report.snapshot.headSha;
  return report.snapshot.bundles.filter((bundle) =>
    bundle.verified === true
    && Boolean(headSha)
    && bundle.headSha === headSha);
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

function readGithubActionsContext(env: NodeJS.ProcessEnv): RepositoryPublicationGithubActionsContext {
  return {
    isActions: env.GITHUB_ACTIONS === "true",
    refName: env.GITHUB_REF_NAME ?? null,
    sha: env.GITHUB_SHA ?? null,
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
  await writeRepositoryPublicationHandoffArtifact(report);
  console.log(JSON.stringify(report, null, 2));
  if (!report.passed) process.exitCode = 1;
}
