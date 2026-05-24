import { createHash } from "node:crypto";
import { existsSync } from "node:fs";
import { mkdir, readFile, stat, writeFile } from "node:fs/promises";
import { basename, join, resolve } from "node:path";
import { pathToFileURL } from "node:url";
import {
  LOCAL_BROWSER_MTP_DEFAULT_SPECULATIVE_TOKENS,
  LOCAL_BROWSER_MTP_DRAFT_MODEL_ID,
  LOCAL_BROWSER_MTP_MAX_SPECULATIVE_TOKENS,
  UnlockedBrowserTransformerClient,
} from "../apps/web/src/lib/llm/unlockedBrowserTransformerClient";
import {
  assertUnlockedFullProfile,
  resolveUnlockedRuntimeProfile,
  type UnlockedRuntimeProfileResolution,
} from "../apps/web/src/lib/runtime/unlockedRuntimeProfile";
import {
  assertUnlockedWebGpuCoverageGates,
  readStrictUnlockedWebGpuGatesFromEnv,
  summarizeUnlockedWebGpuCoverage,
  type StrictUnlockedWebGpuGate,
  type UnlockedWebGpuCoverageSummary,
} from "./unlockedWebGpuCoverage";

interface VerifyArgs {
  manifestPath: string;
  expectedModelId: string;
  manifestSha256: string;
  backendPreference?: "cpu" | "webgpu";
  publicDir: string;
  artifactDir: string;
  requireConfigured: boolean;
  requireManifestSha256: boolean;
  requireSharded: boolean;
  requireQwenMath: boolean;
  requireQwenParity: boolean;
  requirePackedAssets: boolean;
  requireFullProfile: boolean;
  requireWebGpuGates: StrictUnlockedWebGpuGate[];
  requireKvReuse: boolean;
  runtimeProfile: UnlockedRuntimeProfileResolution;
  verificationMaxTokens: number;
  mtpEnabled: boolean;
  mtpDraftModelId: string;
  mtpNumSpeculativeTokens: number;
  mtpMinAcceptanceRate: number;
  mtpDisableWhenLatencyWorse: boolean;
  mtpDraftLayerCount: number;
}

const repoRoot = resolve(new URL("..", import.meta.url).pathname);
const args = parseArgs(process.argv.slice(2));
if (args.requireConfigured && !args.manifestPath) {
  throw new Error("Unlocked verification requires --manifest-path or VITE_UNLOCKED_MODEL_MANIFEST_PATH when strict configured mode is enabled.");
}
if (args.requireManifestSha256 && !args.manifestSha256) {
  throw new Error("Unlocked verification requires --manifest-sha256 or VITE_UNLOCKED_MODEL_MANIFEST_SHA256 when manifest SHA mode is enabled.");
}
if (args.requireFullProfile) assertUnlockedFullProfile(args.runtimeProfile);
const target = args.manifestPath
  ? await resolveManifestTarget(args)
  : await createFixtureTarget(args.artifactDir);
const manifestText = await readManifestText(target.manifestUrl);
const manifestSha256 = args.manifestSha256 || sha256Text(manifestText);
const manifestJson = JSON.parse(manifestText) as unknown;
if (args.requireSharded) assertShardedManifest(manifestJson);
if (args.requireQwenMath) assertQwenMathManifest(manifestJson);
if (args.requireQwenParity) assertQwenParityManifest(manifestJson);
if (args.requirePackedAssets) assertPackedProductionManifest(manifestJson);

await installFileFetch();
const client = new UnlockedBrowserTransformerClient({
  modelId: args.expectedModelId || target.modelId,
  manifestPath: target.manifestUrl,
  manifestSha256,
  allowFixtureWeights: false,
  ...(args.backendPreference ? { backendPreference: args.backendPreference } : {}),
  ...(args.requireWebGpuGates.length > 0 ? { requireWebGpu: true } : {}),
  maxRuntimePromptTokens: args.runtimeProfile.caps.maxRuntimePromptTokens,
  maxRuntimeLayers: args.runtimeProfile.caps.maxRuntimeLayers,
  logitCandidateLimit: args.runtimeProfile.caps.logitCandidateLimit,
  maxGenerationTokens: args.runtimeProfile.caps.maxGenerationTokens,
  mtp: {
    enabled: args.mtpEnabled,
    draftModelId: args.mtpDraftModelId,
    numSpeculativeTokens: args.mtpNumSpeculativeTokens,
    minAcceptanceRate: args.mtpMinAcceptanceRate,
    disableWhenLatencyWorse: args.mtpDisableWhenLatencyWorse,
    draftLayerCount: args.mtpDraftLayerCount,
  },
  kvPersistence: {
    enabled: true,
    namespace: `verify-${Date.now().toString(36)}`,
    preferOpfs: false,
    maxBlocks: 4096,
    maxBytes: 1024 * 1024 * 1024,
  },
});
await client.init();
const chunks: string[] = [];
for await (const chunk of client.streamChat([{ role: "user", content: "alpha beta" }], {
  maxTokens: args.verificationMaxTokens,
  includeProofMarker: true,
})) {
  chunks.push(chunk);
}
const text = chunks.join("");
if (!text.includes("[unlocked:ssa-kv-tsp]")) {
  throw new Error("Unlocked verification did not stream the SSA/KV/TSP proof marker.");
}
if (!client.lastDecodeProof?.tensorControl || !client.lastDecodeProof.tspSteps.includes("attention")) {
  throw new Error("Unlocked verification did not produce a tensor-control decode proof.");
}
const coverageSummary = summarizeUnlockedWebGpuCoverage(client.lastDecodeProof);
assertUnlockedWebGpuCoverageGates(coverageSummary, args.requireWebGpuGates);
if (
  args.mtpEnabled
  && args.mtpDraftModelId === LOCAL_BROWSER_MTP_DRAFT_MODEL_ID
  && client.lastDecodeProof.mtp?.mode !== "draft_verify"
) {
  throw new Error("Unlocked verification did not exercise the browser MTP verifier path.");
}
const primaryDecodeProof = client.lastDecodeProof;
let kvReuseHealth = primaryDecodeProof.kvPersistence ?? null;
if (args.requireKvReuse) {
  await client.flushKvPersistence();
  for await (const _chunk of client.streamChat([{ role: "user", content: "alpha beta" }], {
    maxTokens: 1,
  })) {
    // Drain the second identical prompt so KV reuse proof is populated.
  }
  kvReuseHealth = client.lastDecodeProof?.kvPersistence ?? null;
  if (kvReuseHealth?.decodeReuse !== true) {
    throw new Error("Unlocked verification did not exercise exact-match KV decode reuse.");
  }
}
await writeVerificationArtifact({
  args,
  target,
  manifestSha256,
  manifest: manifestJson,
  response: text,
  decodeProof: primaryDecodeProof,
  kvReuseHealth,
  coverageSummary,
  tspSteps: primaryDecodeProof.tspSteps,
  kvPagingEvents: primaryDecodeProof.kvPagingEvents,
});

console.log(`Unlocked asset OK: ${target.manifestUrl}`);
console.log(`Mode: ${target.mode}`);
console.log(`Model: ${target.modelId}`);
console.log(`Backend preference requested: ${args.backendPreference ?? "auto"}`);
console.log(`Runtime profile: ${args.runtimeProfile.profile}`);
console.log(`Runtime caps: ${JSON.stringify(args.runtimeProfile.caps)}`);
console.log(`WebGPU coverage: ${JSON.stringify(coverageSummary)}`);
console.log(`CPU fallback used: ${coverageSummary.cpuFallbackUsed}`);
console.log(`Logit projection backend: ${client.lastDecodeProof.logitProjectionBackend ?? "unknown"}`);
console.log(`MTP mode: ${client.lastDecodeProof.mtp?.mode ?? "unknown"}`);
console.log(`Manifest SHA-256: ${manifestSha256}`);
console.log(`Response: ${text}`);

async function resolveManifestTarget(args: VerifyArgs): Promise<{ manifestUrl: string; modelId: string; mode: "configured" }> {
  const manifestUrl = toManifestUrl(args.manifestPath, args.publicDir);
  const manifest = JSON.parse(await readManifestText(manifestUrl)) as { modelId?: string };
  if (typeof manifest.modelId !== "string" || !manifest.modelId.trim()) {
    throw new Error("Unlocked manifest must include modelId.");
  }
  if (args.expectedModelId && manifest.modelId !== args.expectedModelId) {
    throw new Error(`Unlocked manifest modelId mismatch: expected ${args.expectedModelId}, received ${manifest.modelId}.`);
  }
  return { manifestUrl, modelId: manifest.modelId, mode: "configured" };
}

async function createFixtureTarget(artifactDir: string): Promise<{ manifestUrl: string; modelId: string; mode: "generated-fixture" }> {
  const fixtureDir = resolve(repoRoot, artifactDir, "unlocked-shard-fixture");
  await mkdir(fixtureDir, { recursive: true });
  const weights = new Float32Array([
    1, 0,
    0, 1,
    1, 1,
    0.5, 0.5,
    0, 0,
    0, 0,
    10, 10,
    -1, -1,
    1, 0,
    0, 1,
  ]);
  const weightsPath = resolve(fixtureDir, "weights.bin");
  await writeFile(weightsPath, Buffer.from(weights.buffer));
  const weightSha256 = sha256Buffer(Buffer.from(weights.buffer));
  const manifest = {
    schemaVersion: 1,
    modelId: "Qwen/Qwen3-0.6B",
    architecture: "qwen3_decoder_control",
    vocabSize: 4,
    hiddenSize: 2,
    headDim: 2,
    tokenEmbedding: shardTensor(0, [4, 2], weightSha256),
    outputProjection: shardTensor(8, [4, 2], weightSha256),
    tokenizer: {
      kind: "vocab",
      tokens: ["alpha", "beta", "MANIFEST_TOKEN", "delta"],
      unknownTokenId: 0,
    },
    layers: [
      {
        qProj: shardTensor(16, [2, 2], weightSha256),
        kProj: shardTensor(16, [2, 2], weightSha256),
        vProj: shardTensor(16, [2, 2], weightSha256),
        oProj: shardTensor(16, [2, 2], weightSha256),
        mlpUpProj: shardTensor(16, [2, 2], weightSha256),
        mlpDownProj: shardTensor(16, [2, 2], weightSha256),
      },
    ],
  };
  const manifestPath = resolve(fixtureDir, "manifest.json");
  await writeFile(manifestPath, `${JSON.stringify(manifest, null, 2)}\n`);
  return {
    manifestUrl: pathToFileURL(manifestPath).toString(),
    modelId: manifest.modelId,
    mode: "generated-fixture",
  };
}

function shardTensor(floatOffset: number, shape: number[], sha256: string): Record<string, unknown> {
  return {
    kind: "f32-shard",
    uri: "weights.bin",
    byteOffset: floatOffset * Float32Array.BYTES_PER_ELEMENT,
    shape,
    sha256,
  };
}

function parseArgs(argv: string[]): VerifyArgs {
  const parsed = new Map<string, string>();
  const flags = new Set<string>();
  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index];
    if (arg === "--") continue;
    if (!arg.startsWith("--")) continue;
    const value = argv[index + 1];
    if (!value || value.startsWith("--")) {
      flags.add(arg.slice(2));
      continue;
    }
    parsed.set(arg.slice(2), value);
    index += 1;
  }
  const requireConfigured = flags.has("require-configured") || process.env.RELEASE_REQUIRE_UNLOCKED_MODEL === "true";
  const requireManifestSha256 = flags.has("require-manifest-sha256") || process.env.RELEASE_REQUIRE_UNLOCKED_MODEL === "true";
  const requireQwenMath = flags.has("require-qwen-math") || process.env.RELEASE_REQUIRE_UNLOCKED_QWEN_MATH === "true";
  const requireQwenParity = flags.has("require-qwen-parity") || process.env.RELEASE_REQUIRE_UNLOCKED_QWEN_PARITY === "true";
  const requirePackedAssets = flags.has("require-packed-assets") || process.env.RELEASE_REQUIRE_UNLOCKED_PACKED_ASSETS === "true";
  const requireFullProfile = flags.has("require-full-profile") || process.env.RELEASE_REQUIRE_UNLOCKED_FULL_PROFILE === "true";
  const requireKvReuse = flags.has("require-kv-reuse") || process.env.RELEASE_REQUIRE_UNLOCKED_KV_REUSE === "true";
  const requireWebGpuGates = [
    ...readStrictUnlockedWebGpuGatesFromEnv(process.env),
    ...(flags.has("require-webgpu-mlp") ? ["mlp" as const] : []),
    ...(flags.has("require-webgpu-logits") ? ["logits" as const] : []),
    ...(flags.has("require-webgpu-attention") ? ["attention" as const] : []),
    ...(flags.has("require-webgpu-projection") ? ["projection" as const] : []),
  ].filter((gate, index, gates) => gates.indexOf(gate) === index);
  const mtpEnabled = !flags.has("mtp-disabled") && (flags.has("mtp-enabled") || process.env.VITE_MTP_ENABLED === "true");
  const runtimeProfile = resolveUnlockedRuntimeProfile({
    VITE_UNLOCKED_RUNTIME_PROFILE: parsed.get("runtime-profile") ?? process.env.VITE_UNLOCKED_RUNTIME_PROFILE ?? (process.env.CI ? "ci" : "full"),
    VITE_UNLOCKED_MAX_RUNTIME_PROMPT_TOKENS: parsed.get("max-runtime-prompt-tokens") ?? process.env.VITE_UNLOCKED_MAX_RUNTIME_PROMPT_TOKENS,
    VITE_UNLOCKED_MAX_RUNTIME_LAYERS: parsed.get("max-runtime-layers") ?? process.env.VITE_UNLOCKED_MAX_RUNTIME_LAYERS,
    VITE_UNLOCKED_LOGIT_CANDIDATE_LIMIT: parsed.get("logit-candidate-limit") ?? process.env.VITE_UNLOCKED_LOGIT_CANDIDATE_LIMIT,
    VITE_UNLOCKED_MAX_GENERATION_TOKENS: parsed.get("max-generation-tokens") ?? process.env.VITE_UNLOCKED_MAX_GENERATION_TOKENS,
  });
  return {
    manifestPath: parsed.get("manifest-path") ?? process.env.VITE_UNLOCKED_MODEL_MANIFEST_PATH ?? "",
    expectedModelId: parsed.get("model-id") ?? process.env.VITE_DEFAULT_MODEL ?? "",
    manifestSha256: parsed.get("manifest-sha256") ?? process.env.VITE_UNLOCKED_MODEL_MANIFEST_SHA256 ?? "",
    ...(parseBackendPreference(parsed.get("backend-preference") ?? process.env.VITE_UNLOCKED_BACKEND_PREFERENCE) !== undefined
      ? { backendPreference: parseBackendPreference(parsed.get("backend-preference") ?? process.env.VITE_UNLOCKED_BACKEND_PREFERENCE) }
      : {}),
    publicDir: resolve(repoRoot, parsed.get("public-dir") ?? "apps/web/public"),
    artifactDir: parsed.get("artifact-dir") ?? process.env.EVAL_ARTIFACT_DIR ?? ".artifacts/evals",
    requireConfigured,
    requireManifestSha256,
    requireSharded: flags.has("require-sharded") || requireConfigured || requireQwenMath || requireQwenParity,
    requireQwenMath,
    requireQwenParity,
    requirePackedAssets,
    requireFullProfile,
    requireWebGpuGates,
    requireKvReuse,
    runtimeProfile,
    verificationMaxTokens: parsePositiveInteger(parsed.get("verification-max-tokens") ?? process.env.UNLOCKED_VERIFY_MAX_TOKENS, 1),
    mtpEnabled,
    mtpDraftModelId: parsed.get("mtp-draft-model-id") ?? process.env.VITE_MTP_DRAFT_MODEL_ID ?? LOCAL_BROWSER_MTP_DRAFT_MODEL_ID,
    mtpNumSpeculativeTokens: parseBrowserMtpSpeculativeTokens(parsed.get("mtp-num-speculative-tokens") ?? process.env.VITE_MTP_NUM_SPECULATIVE_TOKENS),
    mtpMinAcceptanceRate: parseRatio(parsed.get("mtp-min-acceptance-rate") ?? process.env.VITE_MTP_MIN_ACCEPTANCE_RATE, 0),
    mtpDisableWhenLatencyWorse: (parsed.get("mtp-disable-when-latency-worse") ?? process.env.VITE_MTP_DISABLE_WHEN_LATENCY_WORSE) !== "false",
    mtpDraftLayerCount: parsePositiveInteger(parsed.get("mtp-draft-layer-count") ?? process.env.VITE_MTP_DRAFT_LAYER_COUNT, 4),
  };
}

function parsePositiveInteger(value: string | undefined, fallback: number): number {
  if (!value?.trim()) return fallback;
  const parsed = Number(value);
  if (!Number.isFinite(parsed)) return fallback;
  const integer = Math.floor(parsed);
  return integer > 0 ? integer : fallback;
}

function parseBrowserMtpSpeculativeTokens(value: string | undefined): number {
  return Math.min(
    parsePositiveInteger(value, LOCAL_BROWSER_MTP_DEFAULT_SPECULATIVE_TOKENS),
    LOCAL_BROWSER_MTP_MAX_SPECULATIVE_TOKENS,
  );
}

function parseRatio(value: string | undefined, fallback: number): number {
  if (!value?.trim()) return fallback;
  const parsed = Number(value);
  if (!Number.isFinite(parsed)) return fallback;
  return Math.max(0, Math.min(1, parsed));
}

function parseBackendPreference(value: string | undefined): "cpu" | "webgpu" | undefined {
  if (!value) return undefined;
  if (value === "cpu" || value === "webgpu") return value;
  throw new Error(`Unlocked verification backend preference must be "cpu" or "webgpu", received "${value}".`);
}

function toManifestUrl(manifestPath: string, publicDir: string): string {
  if (manifestPath.startsWith("http://") || manifestPath.startsWith("https://") || manifestPath.startsWith("file://")) {
    return manifestPath;
  }
  if (manifestPath.startsWith("/") && existsSync(manifestPath)) return pathToFileURL(manifestPath).toString();
  if (manifestPath.startsWith("/")) return pathToFileURL(resolve(publicDir, manifestPath.slice(1))).toString();
  return pathToFileURL(resolve(manifestPath)).toString();
}

async function readManifestText(manifestUrl: string): Promise<string> {
  if (manifestUrl.startsWith("file://")) {
    return readFile(new URL(manifestUrl), "utf8");
  }
  const response = await fetch(manifestUrl);
  if (!response.ok) throw new Error(`Unlocked manifest failed to load: ${response.status}`);
  const contentType = response.headers.get("content-type") ?? "";
  if (contentType.toLowerCase().includes("text/html")) {
    throw new Error("Unlocked manifest request returned HTML; this usually means the app shell was served instead of manifest JSON.");
  }
  return response.text();
}

async function installFileFetch(): Promise<void> {
  const originalFetch = globalThis.fetch.bind(globalThis);
  globalThis.fetch = async (input: RequestInfo | URL, init?: RequestInit): Promise<Response> => {
    const url = String(input);
    if (!url.startsWith("file://")) return originalFetch(input, init);
    const path = new URL(url);
    const stats = await stat(path);
    if (!stats.isFile()) return new Response("not found", { status: 404 });
    const bytes = await readFile(path);
    const name = basename(path.pathname);
    const type = name.endsWith(".json")
      ? "application/json"
      : name.endsWith(".bin")
        ? "application/octet-stream"
        : "application/octet-stream";
    return new Response(bytes, {
      status: 200,
      headers: {
        "content-type": type,
        "content-length": String(bytes.byteLength),
      },
    });
  };
}

function sha256Text(value: string): string {
  return createHash("sha256").update(value).digest("hex");
}

function sha256Buffer(value: Buffer): string {
  return createHash("sha256").update(value).digest("hex");
}

function assertShardedManifest(value: unknown): void {
  if (!isRecord(value)) throw new Error("Unlocked manifest must be a JSON object.");
  if (value.schemaVersion !== 1) throw new Error("Unlocked manifest must use schemaVersion: 1.");
  assertShardDescriptor(value.tokenEmbedding, "tokenEmbedding", 2);
  assertShardDescriptor(value.outputProjection, "outputProjection", 2);
  if (!Array.isArray(value.layers) || value.layers.length === 0) {
    throw new Error("Unlocked sharded manifest must include at least one layer.");
  }
  value.layers.forEach((layer, index) => {
    if (!isRecord(layer)) throw new Error(`Unlocked manifest layers[${index}] must be an object.`);
    assertShardDescriptor(layer.qProj, `layers[${index}].qProj`, 2);
    assertShardDescriptor(layer.kProj, `layers[${index}].kProj`, 2);
    assertShardDescriptor(layer.vProj, `layers[${index}].vProj`, 2);
    assertShardDescriptor(layer.oProj, `layers[${index}].oProj`, 2);
    if (layer.inputLayerNorm !== undefined) assertShardDescriptor(layer.inputLayerNorm, `layers[${index}].inputLayerNorm`, 1);
    if (layer.qNorm !== undefined) assertShardDescriptor(layer.qNorm, `layers[${index}].qNorm`, 1);
    if (layer.kNorm !== undefined) assertShardDescriptor(layer.kNorm, `layers[${index}].kNorm`, 1);
    if (layer.postAttentionLayerNorm !== undefined) assertShardDescriptor(layer.postAttentionLayerNorm, `layers[${index}].postAttentionLayerNorm`, 1);
    if (layer.mlpGateProj !== undefined) assertShardDescriptor(layer.mlpGateProj, `layers[${index}].mlpGateProj`, 2);
    if (layer.mlpUpProj !== undefined) assertShardDescriptor(layer.mlpUpProj, `layers[${index}].mlpUpProj`, 2);
    if (layer.mlpDownProj !== undefined) assertShardDescriptor(layer.mlpDownProj, `layers[${index}].mlpDownProj`, 2);
  });
}

function assertQwenMathManifest(value: unknown): void {
  if (!isRecord(value)) throw new Error("Unlocked Qwen math manifest must be a JSON object.");
  assertShardDescriptor(value.finalNorm, "finalNorm", 1);
  if (typeof value.rmsNormEps !== "number" || value.rmsNormEps <= 0) {
    throw new Error("Unlocked Qwen math manifest requires positive numeric rmsNormEps.");
  }
  if (!Array.isArray(value.layers) || value.layers.length === 0) {
    throw new Error("Unlocked Qwen math manifest must include at least one layer.");
  }
  value.layers.forEach((layer, index) => {
    if (!isRecord(layer)) throw new Error(`Unlocked Qwen math manifest layers[${index}] must be an object.`);
    assertShardDescriptor(layer.inputLayerNorm, `layers[${index}].inputLayerNorm`, 1);
    assertShardDescriptor(layer.postAttentionLayerNorm, `layers[${index}].postAttentionLayerNorm`, 1);
    assertShardDescriptor(layer.qNorm, `layers[${index}].qNorm`, 1);
    assertShardDescriptor(layer.kNorm, `layers[${index}].kNorm`, 1);
    assertShardDescriptor(layer.mlpGateProj, `layers[${index}].mlpGateProj`, 2);
    assertShardDescriptor(layer.mlpUpProj, `layers[${index}].mlpUpProj`, 2);
    assertShardDescriptor(layer.mlpDownProj, `layers[${index}].mlpDownProj`, 2);
  });
}

function assertQwenParityManifest(value: unknown): void {
  if (!isRecord(value)) throw new Error("Unlocked Qwen parity manifest must be a JSON object.");
  const hiddenSize = readPositiveIntegerField(value.hiddenSize, "hiddenSize");
  const headDim = readPositiveIntegerField(value.headDim, "headDim");
  const numAttentionHeads = readPositiveIntegerField(value.numAttentionHeads, "numAttentionHeads");
  const numKeyValueHeads = readPositiveIntegerField(value.numKeyValueHeads, "numKeyValueHeads");
  readPositiveIntegerField(value.maxPositionEmbeddings, "maxPositionEmbeddings");
  readPositiveNumberField(value.ropeTheta, "ropeTheta");
  if (typeof value.tieWordEmbeddings !== "boolean") {
    throw new Error("Unlocked Qwen parity manifest requires boolean tieWordEmbeddings.");
  }
  if (!isRecord(value.conversion) || value.conversion.projectionMode !== "full-qwen-gqa-rope") {
    throw new Error("Unlocked Qwen parity manifest requires conversion.projectionMode=\"full-qwen-gqa-rope\".");
  }
  if (!Array.isArray(value.layers) || value.layers.length === 0) {
    throw new Error("Unlocked Qwen parity manifest must include at least one layer.");
  }
  const qRows = numAttentionHeads * headDim;
  const kvRows = numKeyValueHeads * headDim;
  value.layers.forEach((layer, index) => {
    if (!isRecord(layer)) throw new Error(`Unlocked Qwen parity manifest layers[${index}] must be an object.`);
    assertShardShape(layer.qProj, `layers[${index}].qProj`, [qRows, hiddenSize]);
    assertShardShape(layer.kProj, `layers[${index}].kProj`, [kvRows, hiddenSize]);
    assertShardShape(layer.vProj, `layers[${index}].vProj`, [kvRows, hiddenSize]);
    assertShardShape(layer.oProj, `layers[${index}].oProj`, [hiddenSize, qRows]);
  });
}

function assertPackedProductionManifest(value: unknown): void {
  if (!isRecord(value)) throw new Error("Unlocked packed production manifest must be a JSON object.");
  const source = isRecord(value.weights) ? value.weights : value;
  const tensorStorage = source.tensorStorage;
  if (!isRecord(tensorStorage)) {
    throw new Error("Unlocked packed production manifest requires tensorStorage metadata.");
  }
  if (tensorStorage.format !== "f16-packed" || tensorStorage.dtype !== "f16" || tensorStorage.shardKind !== "f16-shard") {
    throw new Error("Unlocked packed production manifest requires tensorStorage format=f16-packed, dtype=f16, shardKind=f16-shard.");
  }
  const shardKinds = collectShardKinds(source);
  if (shardKinds.length === 0) throw new Error("Unlocked packed production manifest must include shard descriptors.");
  const nonPacked = shardKinds.filter((kind) => kind !== "f16-shard");
  if (nonPacked.length > 0) {
    throw new Error(`Unlocked packed production manifest cannot include non-f16 shard descriptors: ${[...new Set(nonPacked)].join(", ")}.`);
  }
}

function assertShardShape(value: unknown, name: string, expectedShape: [number, number]): void {
  assertShardDescriptor(value, name, 2);
  if (!isRecord(value) || !Array.isArray(value.shape)) {
    throw new Error(`Unlocked manifest ${name} must include shape.`);
  }
  const [expectedRows, expectedCols] = expectedShape;
  if (value.shape[0] !== expectedRows || value.shape[1] !== expectedCols) {
    throw new Error(`Unlocked Qwen parity manifest ${name} shape must be [${expectedRows}, ${expectedCols}], received [${value.shape.join(", ")}].`);
  }
}

function readPositiveIntegerField(value: unknown, name: string): number {
  if (typeof value !== "number" || !Number.isInteger(value) || value <= 0) {
    throw new Error(`Unlocked Qwen parity manifest requires positive integer ${name}.`);
  }
  return value;
}

function readPositiveNumberField(value: unknown, name: string): number {
  if (typeof value !== "number" || !Number.isFinite(value) || value <= 0) {
    throw new Error(`Unlocked Qwen parity manifest requires positive numeric ${name}.`);
  }
  return value;
}

function assertShardDescriptor(value: unknown, name: string, dimensions: 1 | 2): void {
  if (!isRecord(value) || (value.kind !== "f32-shard" && value.kind !== "f16-shard") || typeof value.uri !== "string") {
    throw new Error(`Unlocked manifest ${name} must be an f32-shard or f16-shard descriptor.`);
  }
  const expectedDtype = value.kind === "f16-shard" ? "f16" : "f32";
  if (value.dtype !== undefined && value.dtype !== expectedDtype) {
    throw new Error(`Unlocked manifest ${name} dtype must match ${value.kind}.`);
  }
  if (value.byteOffset !== 0 && (typeof value.byteOffset !== "number" || !Number.isInteger(value.byteOffset) || value.byteOffset < 0)) {
    throw new Error(`Unlocked manifest ${name} has an invalid byteOffset.`);
  }
  if (!Array.isArray(value.shape) || value.shape.length !== dimensions || !value.shape.every((item) => Number.isInteger(item) && item > 0)) {
    throw new Error(`Unlocked manifest ${name} must include a positive ${dimensions}D shape.`);
  }
  if (typeof value.sha256 !== "string" || !/^[a-fA-F0-9]{64}$/.test(value.sha256)) {
    throw new Error(`Unlocked manifest ${name} must include a 64-character shard sha256.`);
  }
}

async function writeVerificationArtifact(input: {
  args: VerifyArgs;
  target: { manifestUrl: string; modelId: string; mode: string };
  manifestSha256: string;
  manifest: unknown;
  response: string;
  decodeProof: {
    mtp?: unknown;
    kvPersistence?: unknown;
    logitProjectionBackend?: "webgpu" | "cpu_reference";
    logitProjectionPurpose?: "candidate_logit_projection" | "full_vocab_logit_projection" | "full_vocab_topk_logit_projection";
    logitProjectionSelectedRows?: number;
    logitProjectionFullRows?: number;
    mlpKernelBackends?: Array<{ layerIndex: number; backend: "webgpu" | "cpu_reference"; activationKind: string }>;
  };
  kvReuseHealth: unknown;
  coverageSummary: UnlockedWebGpuCoverageSummary;
  tspSteps: string[];
  kvPagingEvents: number;
}): Promise<void> {
  const createdAt = new Date().toISOString();
  const timestamp = createdAt.replace(/[:.]/g, "-");
  const outputDir = resolve(repoRoot, input.args.artifactDir, "unlocked-verify", timestamp);
  const result = {
    name: "unlocked-verify",
    createdAt,
    passed: true,
    mode: input.target.mode,
    manifestUrl: input.target.manifestUrl,
    modelId: input.target.modelId,
    manifestSha256: input.manifestSha256,
    requestedBackendPreference: input.args.backendPreference ?? "auto",
    runtimeProfile: {
      activeProfile: input.args.runtimeProfile.profile,
      resolvedCaps: input.args.runtimeProfile.caps,
      capsActive: input.args.runtimeProfile.capStatus,
      strictFullProfileRequired: input.args.requireFullProfile,
      runIsCapped: isVerificationRunCapped(input.args.runtimeProfile),
    },
    runtimeBudget: {
      promptTokenBudgetUsed: input.args.runtimeProfile.caps.maxRuntimePromptTokens,
      manifestLayerCount: countManifestLayers(input.manifest),
      effectiveRuntimeLayerCount: resolveEffectiveRuntimeLayerCount(input.manifest, input.args.runtimeProfile.caps.maxRuntimeLayers),
      generationTokenBudgetUsed: input.args.verificationMaxTokens,
      logitCandidateBudgetUsed: input.args.runtimeProfile.caps.logitCandidateLimit,
    },
    effectiveBackends: {
      logitProjection: input.decodeProof.logitProjectionBackend ?? "unknown",
      logitProjectionPurpose: input.decodeProof.logitProjectionPurpose ?? "unknown",
      logitProjectionSelectedRows: input.decodeProof.logitProjectionSelectedRows ?? null,
      logitProjectionFullRows: input.decodeProof.logitProjectionFullRows ?? null,
      mlpLayers: input.decodeProof.mlpKernelBackends ?? [],
    },
    webGpuCoverage: input.coverageSummary,
    kvPersistence: input.kvReuseHealth ?? input.decodeProof.kvPersistence ?? null,
    tokenizerKind: isRecord(input.manifest) && isRecord(input.manifest.tokenizer) ? input.manifest.tokenizer.kind : "unknown",
    tensorStorage: summarizeTensorStorage(input.manifest),
    shardCount: countShardDescriptors(input.manifest),
    response: input.response,
    tensorControl: true,
    tspSteps: input.tspSteps,
    kvPagingEvents: input.kvPagingEvents,
    strict: {
      requireConfigured: input.args.requireConfigured,
      requireManifestSha256: input.args.requireManifestSha256,
      requireSharded: input.args.requireSharded,
      requireQwenMath: input.args.requireQwenMath,
      requireQwenParity: input.args.requireQwenParity,
      requirePackedAssets: input.args.requirePackedAssets,
      requireFullProfile: input.args.requireFullProfile,
      requireWebGpuGates: input.args.requireWebGpuGates,
      requireKvReuse: input.args.requireKvReuse,
      mtpEnabled: input.args.mtpEnabled,
      mtpDraftModelId: input.args.mtpDraftModelId,
    },
    mtp: input.decodeProof.mtp ?? null,
  };
  await mkdir(outputDir, { recursive: true });
  await writeFile(join(outputDir, "results.json"), `${JSON.stringify(result, null, 2)}\n`);
  await writeFile(join(outputDir, "summary.md"), buildArtifactSummary(result));
  await mkdir(resolve(repoRoot, input.args.artifactDir), { recursive: true });
  await writeFile(resolve(repoRoot, input.args.artifactDir, "unlocked-verify-latest.json"), `${JSON.stringify(result, null, 2)}\n`);
}

function buildArtifactSummary(result: {
  passed: boolean;
  mode: string;
  modelId: string;
  manifestUrl: string;
  manifestSha256: string;
  requestedBackendPreference: string;
  runtimeProfile: {
    activeProfile: string;
    resolvedCaps: Record<string, number | null>;
    capsActive: Record<string, boolean>;
    strictFullProfileRequired: boolean;
    runIsCapped: boolean;
  };
  runtimeBudget: {
    promptTokenBudgetUsed: number | null;
    manifestLayerCount: number;
    effectiveRuntimeLayerCount: number;
    generationTokenBudgetUsed: number;
    logitCandidateBudgetUsed: number | null;
  };
  effectiveBackends: { logitProjection: string };
  webGpuCoverage: UnlockedWebGpuCoverageSummary;
  kvPersistence?: unknown;
  mtp?: unknown;
  shardCount: number;
  response: string;
}): string {
  return [
    `# Unlocked Verify: ${result.passed ? "PASS" : "FAIL"}`,
    "",
    `- Mode: ${result.mode}`,
    `- Model: ${result.modelId}`,
    `- Backend preference requested: ${result.requestedBackendPreference}`,
    `- Runtime profile: ${result.runtimeProfile.activeProfile}`,
    `- Runtime caps: ${JSON.stringify(result.runtimeProfile.resolvedCaps)}`,
    `- Runtime caps active: ${JSON.stringify(result.runtimeProfile.capsActive)}`,
    `- Strict full profile required: ${result.runtimeProfile.strictFullProfileRequired}`,
    `- Run is capped: ${result.runtimeProfile.runIsCapped}`,
    `- Runtime budget: ${JSON.stringify(result.runtimeBudget)}`,
    `- Logit projection backend: ${result.effectiveBackends.logitProjection}`,
    `- WebGPU coverage: ${JSON.stringify(result.webGpuCoverage)}`,
    `- CPU fallback used: ${result.webGpuCoverage.cpuFallbackUsed}`,
    `- KV decode reuse: ${isRecord(result.kvPersistence) ? result.kvPersistence.decodeReuse ?? false : false}`,
    `- MTP: ${isRecord(result.mtp) ? result.mtp.mode ?? "unknown" : "none"}`,
    `- Manifest: ${result.manifestUrl}`,
    `- Manifest SHA-256: ${result.manifestSha256}`,
    `- Shards: ${result.shardCount}`,
    `- Response: ${result.response}`,
    "",
  ].join("\n");
}

function countShardDescriptors(value: unknown): number {
  if (!isRecord(value)) return 0;
  let count = 0;
  const visit = (item: unknown): void => {
    if (Array.isArray(item)) {
      item.forEach(visit);
      return;
    }
    if (!isRecord(item)) return;
    if (item.kind === "f32-shard" || item.kind === "f16-shard") count += 1;
    Object.values(item).forEach(visit);
  };
  visit(value);
  return count;
}

function collectShardKinds(value: unknown): string[] {
  const kinds: string[] = [];
  const visit = (item: unknown): void => {
    if (Array.isArray(item)) {
      item.forEach(visit);
      return;
    }
    if (!isRecord(item)) return;
    if (item.kind === "f32-shard" || item.kind === "f16-shard") kinds.push(item.kind);
    Object.values(item).forEach(visit);
  };
  visit(value);
  return kinds;
}

function summarizeTensorStorage(value: unknown): Record<string, string | number | boolean | null> {
  const source = isRecord(value) && isRecord(value.weights) ? value.weights : value;
  const tensorStorage = isRecord(source) && isRecord(source.tensorStorage) ? source.tensorStorage : null;
  const shardKinds = collectShardKinds(source);
  const uniqueShardKinds = [...new Set(shardKinds)].sort();
  const explicit = tensorStorage !== null;
  const format = explicit && typeof tensorStorage.format === "string" ? tensorStorage.format : "missing";
  const dtype = explicit && typeof tensorStorage.dtype === "string" ? tensorStorage.dtype : "missing";
  const shardKind = explicit && typeof tensorStorage.shardKind === "string" ? tensorStorage.shardKind : uniqueShardKinds.length === 1 ? uniqueShardKinds[0] as string : "mixed";
  const byteWidth = explicit && typeof tensorStorage.byteWidth === "number" ? tensorStorage.byteWidth : null;
  const productionTarget = explicit && typeof tensorStorage.productionTarget === "string" ? tensorStorage.productionTarget : "unknown";
  const runtimeRepresentation = explicit && typeof tensorStorage.runtimeRepresentation === "string" ? tensorStorage.runtimeRepresentation : "unknown";
  const packedRuntimeCompute = explicit && typeof tensorStorage.packedRuntimeCompute === "boolean" ? tensorStorage.packedRuntimeCompute : false;
  const descriptorsArePacked = uniqueShardKinds.length > 0 && uniqueShardKinds.every((kind) => kind === "f16-shard");
  return {
    explicit,
    format,
    dtype,
    shardKind,
    byteWidth,
    productionTarget,
    runtimeRepresentation,
    packedRuntimeCompute,
    packedProductionReady: explicit
      && format === "f16-packed"
      && dtype === "f16"
      && shardKind === "f16-shard"
      && descriptorsArePacked,
    shardKindCount: uniqueShardKinds.length,
  };
}

function isVerificationRunCapped(profile: UnlockedRuntimeProfileResolution): boolean {
  return profile.capStatus.prompt
    || profile.capStatus.layers
    || profile.capStatus.generation
    || profile.capStatus.logits;
}

function countManifestLayers(value: unknown): number {
  const source = isRecord(value) && isRecord(value.weights) ? value.weights : value;
  if (!isRecord(source) || !Array.isArray(source.layers)) return 0;
  return source.layers.length;
}

function resolveEffectiveRuntimeLayerCount(value: unknown, maxRuntimeLayers: number | null): number {
  const manifestLayerCount = countManifestLayers(value);
  return maxRuntimeLayers === null ? manifestLayerCount : Math.min(manifestLayerCount, maxRuntimeLayers);
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}
