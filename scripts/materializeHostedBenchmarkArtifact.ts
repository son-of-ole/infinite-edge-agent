import { appendFile, mkdir, writeFile } from "node:fs/promises";
import { dirname } from "node:path";

export type HostedBenchmarkArtifactSource = "inline_json" | "url";

export interface MaterializeHostedBenchmarkArtifactInput {
  inlineJson?: string | undefined;
  url?: string | undefined;
  outputPath?: string | undefined;
  fetchImpl?: typeof fetch | undefined;
  githubOutputPath?: string | undefined;
}

export interface MaterializeHostedBenchmarkArtifactResult {
  artifactPath: string;
  source: HostedBenchmarkArtifactSource;
  bytes: number;
}

const DEFAULT_OUTPUT_PATH = ".artifacts/evals/hosted/browser-runtime-bench-latest.json";

export async function materializeHostedBenchmarkArtifact(
  input: MaterializeHostedBenchmarkArtifactInput = {},
): Promise<MaterializeHostedBenchmarkArtifactResult> {
  const outputPath = input.outputPath
    ?? process.env.HOSTED_BENCHMARK_ARTIFACT_OUTPUT_PATH
    ?? process.env.HOSTED_BENCHMARK_ARTIFACT_PATH
    ?? DEFAULT_OUTPUT_PATH;
  const inlineJson = input.inlineJson ?? process.env.HOSTED_BENCHMARK_ARTIFACT_JSON;
  const artifactUrl = input.url ?? process.env.HOSTED_BENCHMARK_ARTIFACT_URL;

  const source: HostedBenchmarkArtifactSource = inlineJson?.trim() ? "inline_json" : "url";
  const raw = inlineJson?.trim()
    ? inlineJson
    : await fetchArtifactJson(artifactUrl, input.fetchImpl ?? fetch);
  const parsed = parseArtifactJson(raw);
  const json = `${JSON.stringify(parsed, null, 2)}\n`;

  await mkdir(dirname(outputPath), { recursive: true });
  await writeFile(outputPath, json);

  const githubOutputPath = input.githubOutputPath ?? process.env.GITHUB_OUTPUT;
  if (githubOutputPath) {
    await appendFile(githubOutputPath, `artifact_path=${outputPath}\n`);
  }

  return {
    artifactPath: outputPath,
    source,
    bytes: Buffer.byteLength(json),
  };
}

async function fetchArtifactJson(
  artifactUrl: string | undefined,
  fetchImpl: typeof fetch,
): Promise<string> {
  if (!artifactUrl?.trim()) {
    throw new Error("Provide HOSTED_BENCHMARK_ARTIFACT_JSON or HOSTED_BENCHMARK_ARTIFACT_URL.");
  }
  const url = new URL(artifactUrl);
  if (url.protocol !== "https:" && url.protocol !== "http:") {
    throw new Error("HOSTED_BENCHMARK_ARTIFACT_URL must be an absolute http(s) URL.");
  }
  const response = await fetchImpl(url);
  if (!response.ok) {
    throw new Error(`Failed to fetch hosted benchmark artifact: HTTP ${response.status}.`);
  }
  return response.text();
}

function parseArtifactJson(raw: string): unknown {
  try {
    const parsed = JSON.parse(raw) as unknown;
    if (typeof parsed !== "object" || parsed === null || Array.isArray(parsed)) {
      throw new Error("not an object");
    }
    return parsed;
  } catch {
    throw new Error("Hosted benchmark artifact JSON is invalid.");
  }
}

if (import.meta.url === `file://${process.argv[1]}`) {
  try {
    const result = await materializeHostedBenchmarkArtifact();
    console.log(JSON.stringify(result, null, 2));
  } catch (error) {
    console.error(error instanceof Error ? error.message : String(error));
    process.exitCode = 1;
  }
}
