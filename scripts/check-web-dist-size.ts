import { readdir, stat } from "node:fs/promises";
import { join, relative, resolve } from "node:path";

const distDir = resolve(process.env.WEB_DIST_DIR ?? "apps/web/dist");
const allowBundledUnlocked = process.env.RELEASE_ALLOW_BUNDLED_UNLOCKED_MODEL === "true";
const maxStaticBytes = readPositiveInt(process.env.RELEASE_MAX_STATIC_FILE_BYTES, 100 * 1024 * 1024);

const files = await walk(distDir);
const violations: string[] = [];

for (const file of files) {
  const stats = await stat(file);
  const rel = relative(distDir, file);
  if (file.endsWith(".litertlm") || file.endsWith(".litertlm.sha256")) {
    violations.push(`${rel}: opaque .litertlm artifacts are not part of the Qwen unlocked release lane.`);
  }
  if (!allowBundledUnlocked && rel.startsWith("models/") && /\.(bin|safetensors|onnx|npy|npz|gguf)$/i.test(rel)) {
    violations.push(`${rel}: unlocked model shards must be hosted outside the static app bundle unless RELEASE_ALLOW_BUNDLED_UNLOCKED_MODEL=true.`);
  }
  if (stats.size > maxStaticBytes) {
    violations.push(`${rel}: ${stats.size} bytes exceeds RELEASE_MAX_STATIC_FILE_BYTES=${maxStaticBytes}.`);
  }
}

if (violations.length > 0) {
  throw new Error(`Web dist is not deploy-safe:\n${violations.map((violation) => `- ${violation}`).join("\n")}`);
}

console.log(`Web dist size check: PASS (${files.length} files, max static file ${maxStaticBytes} bytes)`);

async function walk(dir: string): Promise<string[]> {
  const entries = await readdir(dir, { withFileTypes: true });
  const nested = await Promise.all(entries.map((entry) => {
    const path = join(dir, entry.name);
    return entry.isDirectory() ? walk(path) : Promise.resolve([path]);
  }));
  return nested.flat();
}

function readPositiveInt(value: string | undefined, fallback: number): number {
  if (!value) return fallback;
  const parsed = Number(value);
  if (!Number.isSafeInteger(parsed) || parsed <= 0) {
    throw new Error(`Invalid positive integer: ${value}`);
  }
  return parsed;
}
