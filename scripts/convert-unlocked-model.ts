import { convertUnlockedModel } from "./unlockedModelConverter";

interface CliArgs {
  inputDir: string;
  outputDir: string;
  modelId?: string;
  maxLayers?: number;
  shardDir?: string;
  tensorFormat?: "f32" | "f16";
}

try {
  const result = await convertUnlockedModel(parseArgs(process.argv.slice(2)));
  console.log(`Manifest: ${result.manifestPath}`);
  console.log(`Manifest SHA-256: ${result.manifestSha256}`);
  console.log(`Manifest SHA-256 sidecar: ${result.manifestSha256Path}`);
  console.log(`Env example: ${result.envPath}`);
} catch (error) {
  console.error(error instanceof Error ? error.message : String(error));
  process.exitCode = 1;
}

function parseArgs(argv: string[]): CliArgs {
  const parsed = new Map<string, string>();
  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index];
    if (!arg.startsWith("--")) throw new Error(`Unexpected argument ${arg}`);
    const value = argv[index + 1];
    if (!value || value.startsWith("--")) throw new Error(`Missing value for ${arg}`);
    parsed.set(arg.slice(2), value);
    index += 1;
  }

  const inputDir = parsed.get("input");
  const outputDir = parsed.get("output");
  if (!inputDir) throw new Error("Missing required --input <hf-dir>.");
  if (!outputDir) throw new Error("Missing required --output <out-dir>.");

  return {
    inputDir,
    outputDir,
    ...(parsed.get("model-id") ? { modelId: parsed.get("model-id") } : {}),
    ...(parsed.get("max-layers") ? { maxLayers: parsePositiveInteger(parsed.get("max-layers") as string, "--max-layers") } : {}),
    ...(parsed.get("shard-dir") ? { shardDir: parsed.get("shard-dir") } : {}),
    ...(parsed.get("tensor-format") ? { tensorFormat: parseTensorFormat(parsed.get("tensor-format") as string) } : {}),
  };
}

function parsePositiveInteger(value: string, name: string): number {
  const parsed = Number(value);
  if (!Number.isInteger(parsed) || parsed <= 0) throw new Error(`${name} must be a positive integer.`);
  return parsed;
}

function parseTensorFormat(value: string): "f32" | "f16" {
  const normalized = value.trim().toLowerCase();
  if (normalized === "f32" || normalized === "f32-reference") return "f32";
  if (normalized === "f16" || normalized === "f16-packed" || normalized === "packed-f16") return "f16";
  throw new Error("--tensor-format must be f32 or f16.");
}
