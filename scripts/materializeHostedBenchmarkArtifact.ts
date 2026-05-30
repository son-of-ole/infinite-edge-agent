import { appendFile, mkdir, writeFile } from "node:fs/promises";
import { dirname } from "node:path";

export type HostedBenchmarkArtifactSource = "inline_json" | "base64_json" | "url";

export interface MaterializeHostedBenchmarkArtifactInput {
  inlineJson?: string | undefined;
  base64Json?: string | undefined;
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
  const base64Json = input.base64Json ?? process.env.HOSTED_BENCHMARK_ARTIFACT_BASE64;
  const artifactUrl = input.url ?? process.env.HOSTED_BENCHMARK_ARTIFACT_URL;

  const source = resolveSource({ inlineJson, base64Json });
  const raw = source === "inline_json"
    ? inlineJson?.trim() ?? ""
    : source === "base64_json"
      ? decodeBase64Json(base64Json)
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

function resolveSource(input: {
  inlineJson?: string | undefined;
  base64Json?: string | undefined;
}): HostedBenchmarkArtifactSource {
  if (input.inlineJson?.trim()) return "inline_json";
  if (input.base64Json?.trim()) return "base64_json";
  return "url";
}

async function fetchArtifactJson(
  artifactUrl: string | undefined,
  fetchImpl: typeof fetch,
): Promise<string> {
  if (!artifactUrl?.trim()) {
    throw new Error("Provide HOSTED_BENCHMARK_ARTIFACT_JSON, HOSTED_BENCHMARK_ARTIFACT_BASE64, or HOSTED_BENCHMARK_ARTIFACT_URL.");
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

function decodeBase64Json(base64Json: string | undefined): string {
  if (!base64Json?.trim()) {
    throw new Error("Provide HOSTED_BENCHMARK_ARTIFACT_JSON, HOSTED_BENCHMARK_ARTIFACT_BASE64, or HOSTED_BENCHMARK_ARTIFACT_URL.");
  }
  try {
    return Buffer.from(base64Json.trim(), "base64").toString("utf8");
  } catch {
    throw new Error("HOSTED_BENCHMARK_ARTIFACT_BASE64 is not valid base64.");
  }
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
