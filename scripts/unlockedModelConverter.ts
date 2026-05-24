import { createHash } from "node:crypto";
import { mkdir, open, readdir, readFile, writeFile } from "node:fs/promises";
import { basename, resolve } from "node:path";
import posixPath from "node:path/posix";

export interface ConvertUnlockedModelOptions {
  inputDir: string;
  outputDir: string;
  modelId?: string;
  maxLayers?: number;
  shardDir?: string;
  tensorFormat?: UnlockedTensorFormat;
}

export interface ConvertUnlockedModelResult {
  manifestPath: string;
  manifestSha256: string;
  manifestSha256Path: string;
  envPath: string;
}

type SafetensorsDtype = "F32" | "F16" | "BF16";
export type UnlockedTensorFormat = "f32" | "f16";
type MatrixShape = [number, number];
type VectorShape = [number];
type DescriptorShape = MatrixShape | VectorShape;

interface TensorRef {
  name: string;
  fileName: string;
  dtype: SafetensorsDtype;
  shape: number[];
  bytes: Buffer;
}

interface SafetensorsTensorMetadata {
  name: string;
  dtype: SafetensorsDtype;
  shape: number[];
  absoluteStart: number;
  absoluteEnd: number;
  expectedBytes: number;
}

interface TensorDescriptor {
  kind: "f32-shard" | "f16-shard";
  uri: string;
  byteOffset: 0;
  shape: DescriptorShape;
  sha256: string;
  dtype: UnlockedTensorFormat;
  sourceTensor: string;
  sourceFile: string;
  sourceDtype: SafetensorsDtype;
  sourceShape: number[];
  slice: SliceMetadata;
}

interface SliceMetadata {
  mode: "full" | "rows" | "columns" | "rows-and-columns";
  rowStart: number;
  rowEnd: number;
  colStart: number;
  colEnd: number;
}

interface LoadedModel {
  config: Record<string, unknown>;
  tokenizer: UnlockedTokenizerManifest;
  tensors: Map<string, TensorRef>;
}

interface UnlockedTokenizerManifest {
  kind: "vocab" | "qwen-bpe";
  tokens: string[];
  unknownTokenId: number;
  merges?: Array<[string, string]>;
  specialTokens?: string[];
  chatTemplate?: string;
}

export async function convertUnlockedModel(options: ConvertUnlockedModelOptions): Promise<ConvertUnlockedModelResult> {
  const inputDir = resolve(options.inputDir);
  const outputDir = resolve(options.outputDir);
  const shardDir = sanitizeShardDir(options.shardDir ?? "shards");
  const tensorFormat = normalizeTensorFormat(options.tensorFormat);
  const shardOutputDir = resolve(outputDir, shardDir);
  const loaded = await loadModel(inputDir);
  const modelId = options.modelId?.trim() || readOptionalString(loaded.config.model_id) || basename(inputDir);
  const vocabSize = readPositiveInteger(loaded.config.vocab_size ?? loaded.config.vocabSize, "config.vocab_size");
  const tokenizer = padTokenizer(loaded.tokenizer, vocabSize);
  const hiddenSize = readPositiveInteger(loaded.config.hidden_size ?? loaded.config.hiddenSize, "config.hidden_size");
  const layerCount = readLayerCount(loaded.config, options.maxLayers);
  const headDim = readHeadDim(loaded.config, hiddenSize);
  const numAttentionHeads = readPositiveInteger(loaded.config.num_attention_heads ?? loaded.config.numAttentionHeads, "config.num_attention_heads");
  const numKeyValueHeads = readPositiveInteger(loaded.config.num_key_value_heads ?? loaded.config.numKeyValueHeads ?? numAttentionHeads, "config.num_key_value_heads");
  const maxPositionEmbeddings = readPositiveInteger(loaded.config.max_position_embeddings ?? loaded.config.maxPositionEmbeddings, "config.max_position_embeddings");
  const ropeTheta = readPositiveNumber(loaded.config.rope_theta ?? loaded.config.ropeTheta, "config.rope_theta");
  const tieWordEmbeddings = loaded.config.tie_word_embeddings === true || loaded.config.tieWordEmbeddings === true;
  const queryProjectionRows = numAttentionHeads * headDim;
  const keyValueProjectionRows = numKeyValueHeads * headDim;
  const rmsNormEps = typeof loaded.config.rms_norm_eps === "number"
    ? loaded.config.rms_norm_eps
    : typeof loaded.config.rmsNormEps === "number"
      ? loaded.config.rmsNormEps
      : undefined;
  const hiddenActivation = readOptionalString(loaded.config.hidden_act ?? loaded.config.hiddenActivation);
  const hasMlpTensors = hasOptionalLayerTensor(loaded.tensors, layerCount, "mlp.gate_proj.weight")
    || hasOptionalLayerTensor(loaded.tensors, layerCount, "mlp.up_proj.weight")
    || hasOptionalLayerTensor(loaded.tensors, layerCount, "mlp.down_proj.weight");
  const intermediateSize = hasMlpTensors
    ? readPositiveInteger(loaded.config.intermediate_size ?? loaded.config.intermediateSize, "config.intermediate_size")
    : undefined;

  await mkdir(shardOutputDir, { recursive: true });

  const tokenEmbedding = await writeMatrixShard({
    tensor: requireTensor(loaded.tensors, "model.embed_tokens.weight"),
    shardOutputDir,
    shardDir,
    fileStem: "token-embedding",
    shape: [vocabSize, hiddenSize],
    slice: fullSlice(vocabSize, hiddenSize),
    tensorFormat,
  });
  const outputProjectionSource = resolveOutputProjectionTensor(loaded.tensors, loaded.config);
  const outputProjection = await writeMatrixShard({
    tensor: outputProjectionSource,
    shardOutputDir,
    shardDir,
    fileStem: "output-projection",
    shape: [vocabSize, hiddenSize],
    slice: fullSlice(vocabSize, hiddenSize),
    tensorFormat,
  });
  const finalNorm = await writeOptionalVectorShard({
    tensor: loaded.tensors.get("model.norm.weight"),
    shardOutputDir,
    shardDir,
    fileStem: "final-norm",
    shape: [hiddenSize],
    slice: fullVectorSlice(hiddenSize),
    tensorFormat,
  });

  const layers = [];
  for (let layerIndex = 0; layerIndex < layerCount; layerIndex += 1) {
    const inputLayerNorm = await writeOptionalVectorShard({
      tensor: loaded.tensors.get(`model.layers.${layerIndex}.input_layernorm.weight`),
      shardOutputDir,
      shardDir,
      fileStem: `layer-${layerIndex}-input-layernorm`,
      shape: [hiddenSize],
      slice: fullVectorSlice(hiddenSize),
      tensorFormat,
    });
    const postAttentionLayerNorm = await writeOptionalVectorShard({
      tensor: loaded.tensors.get(`model.layers.${layerIndex}.post_attention_layernorm.weight`),
      shardOutputDir,
      shardDir,
      fileStem: `layer-${layerIndex}-post-attention-layernorm`,
      shape: [hiddenSize],
      slice: fullVectorSlice(hiddenSize),
      tensorFormat,
    });
    const qNorm = await writeOptionalVectorShard({
      tensor: loaded.tensors.get(`model.layers.${layerIndex}.self_attn.q_norm.weight`),
      shardOutputDir,
      shardDir,
      fileStem: `layer-${layerIndex}-q-norm`,
      shape: [headDim],
      slice: fullVectorSlice(headDim),
      tensorFormat,
    });
    const kNorm = await writeOptionalVectorShard({
      tensor: loaded.tensors.get(`model.layers.${layerIndex}.self_attn.k_norm.weight`),
      shardOutputDir,
      shardDir,
      fileStem: `layer-${layerIndex}-k-norm`,
      shape: [headDim],
      slice: fullVectorSlice(headDim),
      tensorFormat,
    });
    const mlpGateProj = intermediateSize === undefined ? undefined : await writeOptionalMatrixShard({
      tensor: loaded.tensors.get(`model.layers.${layerIndex}.mlp.gate_proj.weight`),
      shardOutputDir,
      shardDir,
      fileStem: `layer-${layerIndex}-mlp-gate-proj`,
      shape: [intermediateSize, hiddenSize],
      slice: fullSlice(intermediateSize, hiddenSize),
      tensorFormat,
    });
    const mlpUpProj = intermediateSize === undefined ? undefined : await writeOptionalMatrixShard({
      tensor: loaded.tensors.get(`model.layers.${layerIndex}.mlp.up_proj.weight`),
      shardOutputDir,
      shardDir,
      fileStem: `layer-${layerIndex}-mlp-up-proj`,
      shape: [intermediateSize, hiddenSize],
      slice: fullSlice(intermediateSize, hiddenSize),
      tensorFormat,
    });
    const mlpDownProj = intermediateSize === undefined ? undefined : await writeOptionalMatrixShard({
      tensor: loaded.tensors.get(`model.layers.${layerIndex}.mlp.down_proj.weight`),
      shardOutputDir,
      shardDir,
      fileStem: `layer-${layerIndex}-mlp-down-proj`,
      shape: [hiddenSize, intermediateSize],
      slice: fullSlice(hiddenSize, intermediateSize),
      tensorFormat,
    });
    layers.push({
      ...(inputLayerNorm ? { inputLayerNorm } : {}),
      ...(postAttentionLayerNorm ? { postAttentionLayerNorm } : {}),
      ...(qNorm ? { qNorm } : {}),
      ...(kNorm ? { kNorm } : {}),
      qProj: await writeMatrixShard({
        tensor: requireTensor(loaded.tensors, `model.layers.${layerIndex}.self_attn.q_proj.weight`),
        shardOutputDir,
        shardDir,
        fileStem: `layer-${layerIndex}-q-proj`,
        shape: [queryProjectionRows, hiddenSize],
        slice: fullSlice(queryProjectionRows, hiddenSize),
        tensorFormat,
      }),
      kProj: await writeMatrixShard({
        tensor: requireTensor(loaded.tensors, `model.layers.${layerIndex}.self_attn.k_proj.weight`),
        shardOutputDir,
        shardDir,
        fileStem: `layer-${layerIndex}-k-proj`,
        shape: [keyValueProjectionRows, hiddenSize],
        slice: fullSlice(keyValueProjectionRows, hiddenSize),
        tensorFormat,
      }),
      vProj: await writeMatrixShard({
        tensor: requireTensor(loaded.tensors, `model.layers.${layerIndex}.self_attn.v_proj.weight`),
        shardOutputDir,
        shardDir,
        fileStem: `layer-${layerIndex}-v-proj`,
        shape: [keyValueProjectionRows, hiddenSize],
        slice: fullSlice(keyValueProjectionRows, hiddenSize),
        tensorFormat,
      }),
      oProj: await writeMatrixShard({
        tensor: requireTensor(loaded.tensors, `model.layers.${layerIndex}.self_attn.o_proj.weight`),
        shardOutputDir,
        shardDir,
        fileStem: `layer-${layerIndex}-o-proj`,
        shape: [hiddenSize, queryProjectionRows],
        slice: fullSlice(hiddenSize, queryProjectionRows),
        tensorFormat,
      }),
      ...(mlpGateProj ? { mlpGateProj } : {}),
      ...(mlpUpProj ? { mlpUpProj } : {}),
      ...(mlpDownProj ? { mlpDownProj } : {}),
    });
  }

  const manifest = {
    schemaVersion: 1,
    modelId,
    architecture: "qwen3_decoder_control",
    vocabSize,
    hiddenSize,
    headDim,
    numAttentionHeads,
    numKeyValueHeads,
    maxPositionEmbeddings,
    ropeTheta,
    tieWordEmbeddings,
    ...(intermediateSize !== undefined ? { intermediateSize } : {}),
    ...(rmsNormEps !== undefined ? { rmsNormEps } : {}),
    ...(hiddenActivation ? { hiddenActivation } : {}),
    tokenizer,
    tokenEmbedding,
    outputProjection,
    ...(finalNorm ? { finalNorm } : {}),
    layers,
    tensorStorage: {
      format: tensorFormat === "f16" ? "f16-packed" : "f32-reference",
      dtype: tensorFormat,
      shardKind: `${tensorFormat}-shard`,
      byteWidth: tensorFormatByteLength(tensorFormat),
      productionTarget: tensorFormat === "f16" ? "webgpu-packed" : "reference",
      runtimeRepresentation: tensorFormat === "f16" ? "packed-f16-runtime-lazy-decode" : "f32-reference",
      packedRuntimeCompute: false,
    },
    conversion: {
      sourceFormat: "huggingface-safetensors",
      sourceFiles: [...new Set([...loaded.tensors.values()].map((tensor) => tensor.fileName))].sort(),
      projectionMode: "full-qwen-gqa-rope",
      layerCount,
      maxLayers: options.maxLayers ?? null,
      mlpTensors: hasMlpTensors ? "included" : "omitted",
      tensorFormat,
    },
  };

  await mkdir(outputDir, { recursive: true });
  const manifestPath = resolve(outputDir, "manifest.json");
  const manifestBytes = Buffer.from(`${JSON.stringify(manifest, null, 2)}\n`, "utf8");
  const manifestSha256 = sha256Hex(manifestBytes);
  await writeFile(manifestPath, manifestBytes);
  const manifestSha256Path = resolve(outputDir, "manifest.json.sha256");
  await writeFile(manifestSha256Path, `${manifestSha256}  manifest.json\n`);
  const envPath = resolve(outputDir, "unlocked.env.example");
  const publicManifestPath = `/models/${basename(outputDir)}/manifest.json`;
  await writeFile(envPath, [
    `VITE_LLM_BACKEND=unlocked-browser-transformer`,
    `VITE_DEFAULT_MODEL=${modelId}`,
    `VITE_REQUIRE_UNLOCKED_RUNTIME=true`,
    `VITE_UNLOCKED_MODEL_MANIFEST_PATH=${publicManifestPath}`,
    `VITE_UNLOCKED_MODEL_MANIFEST_SHA256=${manifestSha256}`,
    `VITE_UNLOCKED_MANIFEST_FORMAT=sharded`,
    `VITE_UNLOCKED_WEIGHT_FORMAT=${tensorFormat === "f16" ? "f16-packed" : "f32-reference"}`,
    `VITE_UNLOCKED_ALLOW_FIXTURE=false`,
    `VITE_UNLOCKED_BACKEND_PREFERENCE=webgpu`,
    "",
  ].join("\n"));
  return {
    manifestPath,
    manifestSha256,
    manifestSha256Path,
    envPath,
  };
}

async function loadModel(inputDir: string): Promise<LoadedModel> {
  const config = JSON.parse(await readFile(resolve(inputDir, "config.json"), "utf8")) as unknown;
  if (!isRecord(config)) throw new Error("config.json must contain a JSON object.");
  const tokenizerJson = JSON.parse(await readFile(resolve(inputDir, "tokenizer.json"), "utf8")) as unknown;
  const tokenizerConfig = await readOptionalJson(resolve(inputDir, "tokenizer_config.json"));
  const tokenizer = extractTokenizer(tokenizerJson, tokenizerConfig);
  const safetensorFiles = (await readdir(inputDir))
    .filter((fileName) => fileName.endsWith(".safetensors"))
    .sort();
  if (safetensorFiles.length === 0) {
    throw new Error("Input directory must contain at least one .safetensors file.");
  }

  const tensors = new Map<string, TensorRef>();
  for (const fileName of safetensorFiles) {
    for (const tensor of await parseSafetensorsFile(fileName, resolve(inputDir, fileName))) {
      if (tensors.has(tensor.name)) throw new Error(`Duplicate safetensors tensor ${tensor.name}.`);
      tensors.set(tensor.name, tensor);
    }
  }
  return { config, tokenizer, tensors };
}

async function parseSafetensorsFile(fileName: string, filePath: string): Promise<TensorRef[]> {
  const handle = await open(filePath, "r");
  try {
    const stats = await handle.stat();
    if (stats.size < 8) throw new Error(`${fileName} is too small to be a safetensors file.`);
    const lengthPrefix = Buffer.alloc(8);
    await readExactly(handle, lengthPrefix, 0);
    const headerLength = Number(lengthPrefix.readBigUInt64LE(0));
    if (!Number.isSafeInteger(headerLength) || headerLength <= 0) {
      throw new Error(`${fileName} has an invalid safetensors header length.`);
    }
    const dataStart = 8 + headerLength;
    if (dataStart > stats.size) {
      throw new Error(`${fileName} safetensors header extends past end of file.`);
    }
    const headerBytes = Buffer.alloc(headerLength);
    await readExactly(handle, headerBytes, 8);
    const header = JSON.parse(headerBytes.toString("utf8").trimEnd()) as unknown;
    const metadata = parseSafetensorsHeader(fileName, header, dataStart, stats.size);
    const tensors: TensorRef[] = [];
    for (const item of metadata) {
      const bytes = Buffer.alloc(item.expectedBytes);
      await readExactly(handle, bytes, item.absoluteStart);
      tensors.push({
        name: item.name,
        fileName,
        dtype: item.dtype,
        shape: item.shape,
        bytes,
      });
    }
    return tensors;
  } finally {
    await handle.close();
  }
}

async function readExactly(
  handle: Awaited<ReturnType<typeof open>>,
  buffer: Buffer,
  position: number,
): Promise<void> {
  let offset = 0;
  while (offset < buffer.byteLength) {
    const { bytesRead } = await handle.read(buffer, offset, buffer.byteLength - offset, position + offset);
    if (bytesRead === 0) throw new Error(`Unexpected end of file while reading safetensors data at byte ${position + offset}.`);
    offset += bytesRead;
  }
}

function parseSafetensorsHeader(
  fileName: string,
  header: unknown,
  dataStart: number,
  fileByteLength: number,
): SafetensorsTensorMetadata[] {
  if (!isRecord(header)) throw new Error(`${fileName} safetensors header must be a JSON object.`);

  const tensors: SafetensorsTensorMetadata[] = [];
  for (const [name, entry] of Object.entries(header)) {
    if (name === "__metadata__") continue;
    if (!isRecord(entry)) throw new Error(`${fileName}:${name} metadata must be an object.`);
    const dtype = entry.dtype;
    if (dtype !== "F32" && dtype !== "F16" && dtype !== "BF16") {
      throw new Error(`${fileName}:${name} dtype ${String(dtype)} is not supported.`);
    }
    if (!Array.isArray(entry.shape) || !entry.shape.every((item): item is number => Number.isInteger(item) && item > 0)) {
      throw new Error(`${fileName}:${name} must include a positive integer shape.`);
    }
    const shape = entry.shape;
    if (!Array.isArray(entry.data_offsets) || entry.data_offsets.length !== 2) {
      throw new Error(`${fileName}:${name} must include data_offsets.`);
    }
    const [offsetStart, offsetEnd] = entry.data_offsets;
    if (typeof offsetStart !== "number" || typeof offsetEnd !== "number" || !Number.isInteger(offsetStart) || !Number.isInteger(offsetEnd) || offsetStart < 0 || offsetEnd < offsetStart) {
      throw new Error(`${fileName}:${name} has invalid data_offsets.`);
    }
    const absoluteStart = dataStart + offsetStart;
    const absoluteEnd = dataStart + offsetEnd;
    if (absoluteEnd > fileByteLength) {
      throw new Error(`${fileName}:${name} data_offsets extend past end of file.`);
    }
    const expectedBytes = shape.reduce((product, value) => product * value, 1) * dtypeByteLength(dtype);
    if (offsetEnd - offsetStart !== expectedBytes) {
      throw new Error(`${fileName}:${name} byte length does not match dtype and shape.`);
    }
    tensors.push({
      name,
      dtype,
      shape,
      absoluteStart,
      absoluteEnd,
      expectedBytes,
    });
  }
  return tensors;
}

async function readOptionalJson(path: string): Promise<unknown | undefined> {
  try {
    return JSON.parse(await readFile(path, "utf8")) as unknown;
  } catch (error) {
    if (isNodeError(error) && error.code === "ENOENT") return undefined;
    throw error;
  }
}

function extractTokenizer(value: unknown, tokenizerConfig: unknown): UnlockedTokenizerManifest {
  if (!isRecord(value) || !isRecord(value.model) || !isRecord(value.model.vocab)) {
    throw new Error("tokenizer.json must include model.vocab.");
  }
  const vocabEntries = Object.entries(value.model.vocab);
  const tokens: string[] = [];
  for (const [token, id] of vocabEntries) {
    if (typeof id !== "number" || !Number.isInteger(id) || id < 0) {
      throw new Error(`tokenizer.json model.vocab token ${token} must map to a non-negative integer id.`);
    }
    tokens[id] = token;
  }
  if (Array.isArray(value.added_tokens)) {
    for (const entry of value.added_tokens) {
      if (!isRecord(entry)) continue;
      const id = entry.id;
      const content = entry.content;
      if (typeof id !== "number" || !Number.isInteger(id) || id < 0 || typeof content !== "string") continue;
      tokens[id] = content;
    }
  }
  if (tokens.length === 0) throw new Error("tokenizer.json model.vocab must not be empty.");
  for (let index = 0; index < tokens.length; index += 1) {
    tokens[index] ??= `<reserved_${index}>`;
  }
  const unknownTokenId = findUnknownTokenId(value.model, value, tokens);
  const merges = extractMerges(value.model);
  const specialTokens = extractSpecialTokens(value, tokenizerConfig, tokens);
  const chatTemplate = isRecord(tokenizerConfig) ? readOptionalString(tokenizerConfig.chat_template) : undefined;
  return {
    kind: merges.length > 0 ? "qwen-bpe" : "vocab",
    tokens,
    unknownTokenId,
    ...(merges.length > 0 ? { merges } : {}),
    ...(specialTokens.length > 0 ? { specialTokens } : {}),
    ...(chatTemplate ? { chatTemplate } : {}),
  };
}

function padTokenizer(tokenizer: UnlockedTokenizerManifest, vocabSize: number): UnlockedTokenizerManifest {
  if (tokenizer.tokens.length > vocabSize) {
    throw new Error(`tokenizer.json has ${tokenizer.tokens.length} token ids, which exceeds config.vocab_size ${vocabSize}.`);
  }
  const tokens = [...tokenizer.tokens];
  for (let index = tokens.length; index < vocabSize; index += 1) {
    tokens[index] = `<reserved_${index}>`;
  }
  return {
    ...tokenizer,
    tokens,
  };
}

function extractMerges(model: Record<string, unknown>): Array<[string, string]> {
  if (!Array.isArray(model.merges)) return [];
  return model.merges.map((merge, index) => {
    if (Array.isArray(merge) && merge.length === 2 && typeof merge[0] === "string" && typeof merge[1] === "string") {
      return [merge[0], merge[1]] as [string, string];
    }
    if (typeof merge === "string") {
      const parts = merge.split(" ");
      if (parts.length === 2 && parts[0] && parts[1]) return [parts[0], parts[1]] as [string, string];
    }
    throw new Error(`tokenizer.json model.merges[${index}] must be a merge pair.`);
  });
}

function extractSpecialTokens(tokenizerJson: Record<string, unknown>, tokenizerConfig: unknown, tokens: string[]): string[] {
  const specials = new Set<string>();
  const add = (value: unknown): void => {
    const token = typeof value === "string"
      ? value
      : isRecord(value)
        ? readOptionalString(value.content)
        : undefined;
    if (token && tokens.includes(token)) specials.add(token);
  };
  for (const entry of Array.isArray(tokenizerJson.added_tokens) ? tokenizerJson.added_tokens : []) {
    if (isRecord(entry) && entry.special === true) add(entry.content);
  }
  if (isRecord(tokenizerConfig)) {
    add(tokenizerConfig.bos_token);
    add(tokenizerConfig.eos_token);
    add(tokenizerConfig.pad_token);
    add(tokenizerConfig.unk_token);
    if (Array.isArray(tokenizerConfig.additional_special_tokens)) {
      tokenizerConfig.additional_special_tokens.forEach(add);
    }
    if (isRecord(tokenizerConfig.added_tokens_decoder)) {
      Object.values(tokenizerConfig.added_tokens_decoder).forEach(add);
    }
  }
  return [...specials].sort();
}

function findUnknownTokenId(model: Record<string, unknown>, tokenizer: Record<string, unknown>, tokens: string[]): number {
  const vocab = model.vocab as Record<string, unknown>;
  const candidates = [
    readOptionalString(model.unk_token),
    readOptionalString(tokenizer.unk_token),
    isRecord(tokenizer.unk_token) ? readOptionalString(tokenizer.unk_token.content) : undefined,
    "<unk>",
    "<|endoftext|>",
  ].filter((candidate): candidate is string => typeof candidate === "string" && candidate.length > 0);
  for (const candidate of candidates) {
    const id = vocab[candidate];
    if (typeof id === "number" && Number.isInteger(id) && id >= 0 && id < tokens.length) return id;
  }
  return 0;
}

function readLayerCount(config: Record<string, unknown>, maxLayers: number | undefined): number {
  const configured = readPositiveInteger(config.num_hidden_layers ?? config.numHiddenLayers, "config.num_hidden_layers");
  if (maxLayers === undefined) return configured;
  if (!Number.isInteger(maxLayers) || maxLayers <= 0) throw new Error("maxLayers must be a positive integer.");
  return Math.min(configured, maxLayers);
}

function readHeadDim(config: Record<string, unknown>, hiddenSize: number): number {
  const explicit = config.head_dim ?? config.headDim;
  if (explicit !== undefined) return readPositiveInteger(explicit, "config.head_dim");
  const heads = readPositiveInteger(config.num_attention_heads ?? config.numAttentionHeads, "config.num_attention_heads");
  if (hiddenSize % heads !== 0) {
    throw new Error("config.hidden_size must be divisible by config.num_attention_heads when config.head_dim is absent.");
  }
  return hiddenSize / heads;
}

function resolveOutputProjectionTensor(tensors: Map<string, TensorRef>, config: Record<string, unknown>): TensorRef {
  const lmHead = tensors.get("lm_head.weight");
  if (lmHead) return lmHead;
  if (config.tie_word_embeddings === true || config.tieWordEmbeddings === true) {
    return requireTensor(tensors, "model.embed_tokens.weight");
  }
  throw new Error("Missing lm_head.weight. Set config.tie_word_embeddings=true to reuse model.embed_tokens.weight.");
}

async function writeMatrixShard(options: {
  tensor: TensorRef;
  shardOutputDir: string;
  shardDir: string;
  fileStem: string;
  shape: MatrixShape;
  slice: SliceMetadata;
  tensorFormat?: UnlockedTensorFormat;
}): Promise<TensorDescriptor> {
  const [rows, cols] = options.shape;
  assertTensorCoversSlice(options.tensor, options.slice);
  const tensorFormat = normalizeTensorFormat(options.tensorFormat);
  const values = Buffer.alloc(rows * cols * tensorFormatByteLength(tensorFormat));
  let writeOffset = 0;
  for (let sourceRow = options.slice.rowStart; sourceRow < options.slice.rowEnd; sourceRow += 1) {
    for (let sourceCol = options.slice.colStart; sourceCol < options.slice.colEnd; sourceCol += 1) {
      writePackedTensorValue(values, writeOffset, tensorFormat, readTensorValue(options.tensor, sourceRow, sourceCol));
      writeOffset += tensorFormatByteLength(tensorFormat);
    }
  }
  const fileName = `${options.fileStem}.${tensorFormat}.bin`;
  const shardPath = resolve(options.shardOutputDir, fileName);
  await writeFile(shardPath, values);
  const uri = posixPath.join(options.shardDir.split(/[\\/]+/).filter(Boolean).join("/"), fileName);
  return {
    kind: `${tensorFormat}-shard`,
    uri,
    byteOffset: 0,
    shape: options.shape,
    sha256: sha256Hex(values),
    dtype: tensorFormat,
    sourceTensor: options.tensor.name,
    sourceFile: options.tensor.fileName,
    sourceDtype: options.tensor.dtype,
    sourceShape: options.tensor.shape,
    slice: options.slice,
  };
}

async function writeOptionalMatrixShard(options: {
  tensor: TensorRef | undefined;
  shardOutputDir: string;
  shardDir: string;
  fileStem: string;
  shape: MatrixShape;
  slice: SliceMetadata;
  tensorFormat?: UnlockedTensorFormat;
}): Promise<TensorDescriptor | undefined> {
  if (!options.tensor) return undefined;
  return writeMatrixShard({
    tensor: options.tensor,
    shardOutputDir: options.shardOutputDir,
    shardDir: options.shardDir,
    fileStem: options.fileStem,
    shape: options.shape,
    slice: options.slice,
    tensorFormat: options.tensorFormat,
  });
}

async function writeOptionalVectorShard(options: {
  tensor: TensorRef | undefined;
  shardOutputDir: string;
  shardDir: string;
  fileStem: string;
  shape: VectorShape;
  slice: SliceMetadata;
  tensorFormat?: UnlockedTensorFormat;
}): Promise<TensorDescriptor | undefined> {
  if (!options.tensor) return undefined;
  assertTensorCoversSlice(options.tensor, options.slice);
  const tensorFormat = normalizeTensorFormat(options.tensorFormat);
  const [length] = options.shape;
  const values = Buffer.alloc(length * tensorFormatByteLength(tensorFormat));
  for (let sourceRow = options.slice.rowStart; sourceRow < options.slice.rowEnd; sourceRow += 1) {
    writePackedTensorValue(
      values,
      (sourceRow - options.slice.rowStart) * tensorFormatByteLength(tensorFormat),
      tensorFormat,
      readTensorValue(options.tensor, sourceRow, 0),
    );
  }
  const fileName = `${options.fileStem}.${tensorFormat}.bin`;
  const shardPath = resolve(options.shardOutputDir, fileName);
  await writeFile(shardPath, values);
  const uri = posixPath.join(options.shardDir.split(/[\\/]+/).filter(Boolean).join("/"), fileName);
  return {
    kind: `${tensorFormat}-shard`,
    uri,
    byteOffset: 0,
    shape: options.shape,
    sha256: sha256Hex(values),
    dtype: tensorFormat,
    sourceTensor: options.tensor.name,
    sourceFile: options.tensor.fileName,
    sourceDtype: options.tensor.dtype,
    sourceShape: options.tensor.shape,
    slice: options.slice,
  };
}

function assertTensorCoversSlice(tensor: TensorRef, slice: SliceMetadata): void {
  if (tensor.shape.length === 1) {
    const length = tensor.shape[0];
    if (length === undefined) throw new Error(`${tensor.name} must include vector length.`);
    if (slice.mode === "full" && (slice.rowStart !== 0 || slice.rowEnd !== length || slice.colStart !== 0 || slice.colEnd !== 1)) {
      throw new Error(`${tensor.name} shape [${tensor.shape.join(", ")}] must exactly match requested full vector shape [${slice.rowEnd - slice.rowStart}].`);
    }
    if (slice.colStart !== 0 || slice.colEnd !== 1 || slice.rowEnd > length) {
      throw new Error(`${tensor.name} shape [${tensor.shape.join(", ")}] cannot satisfy requested vector slice ${slice.rowStart}:${slice.rowEnd}.`);
    }
    return;
  }
  if (tensor.shape.length !== 2) {
    throw new Error(`${tensor.name} must be rank-2 for the unlocked runtime manifest.`);
  }
  const rows = tensor.shape[0];
  const cols = tensor.shape[1];
  if (rows === undefined || cols === undefined) throw new Error(`${tensor.name} must include matrix rows and columns.`);
  if (slice.mode === "full" && (slice.rowStart !== 0 || slice.rowEnd !== rows || slice.colStart !== 0 || slice.colEnd !== cols)) {
    throw new Error(`${tensor.name} shape [${tensor.shape.join(", ")}] must exactly match requested full matrix shape [${slice.rowEnd - slice.rowStart}, ${slice.colEnd - slice.colStart}].`);
  }
  if (slice.rowEnd > rows || slice.colEnd > cols) {
    throw new Error(`${tensor.name} shape [${tensor.shape.join(", ")}] cannot satisfy requested slice rows ${slice.rowStart}:${slice.rowEnd}, columns ${slice.colStart}:${slice.colEnd}.`);
  }
}

function readTensorValue(tensor: TensorRef, row: number, col: number): number {
  const colCount = tensor.shape.length === 1 ? 1 : tensor.shape[1];
  if (colCount === undefined) throw new Error(`${tensor.name} must include a column count.`);
  const index = row * colCount + col;
  const byteOffset = index * dtypeByteLength(tensor.dtype);
  if (tensor.dtype === "F32") return tensor.bytes.readFloatLE(byteOffset);
  if (tensor.dtype === "F16") return float16BitsToFloat32(tensor.bytes.readUInt16LE(byteOffset));
  return bfloat16BitsToFloat32(tensor.bytes.readUInt16LE(byteOffset));
}

function writePackedTensorValue(buffer: Buffer, byteOffset: number, tensorFormat: UnlockedTensorFormat, value: number): void {
  if (tensorFormat === "f32") {
    buffer.writeFloatLE(value, byteOffset);
    return;
  }
  buffer.writeUInt16LE(float32ToFloat16Bits(value), byteOffset);
}

function normalizeTensorFormat(value: UnlockedTensorFormat | undefined): UnlockedTensorFormat {
  if (value === undefined || value === "f32") return "f32";
  if (value === "f16") return "f16";
  throw new Error(`tensorFormat must be "f32" or "f16", received ${String(value)}.`);
}

function tensorFormatByteLength(format: UnlockedTensorFormat): number {
  return format === "f32" ? Float32Array.BYTES_PER_ELEMENT : Uint16Array.BYTES_PER_ELEMENT;
}

function float16BitsToFloat32(bits: number): number {
  const sign = bits & 0x8000 ? -1 : 1;
  const exponent = (bits >> 10) & 0x1f;
  const mantissa = bits & 0x03ff;
  if (exponent === 0) return sign * (mantissa === 0 ? 0 : (mantissa / 1024) * 2 ** -14);
  if (exponent === 0x1f) return mantissa === 0 ? sign * Infinity : Number.NaN;
  return sign * (1 + mantissa / 1024) * 2 ** (exponent - 15);
}

function float32ToFloat16Bits(value: number): number {
  if (Number.isNaN(value)) return 0x7e00;
  if (value === Infinity) return 0x7c00;
  if (value === -Infinity) return 0xfc00;
  const sign = value < 0 || Object.is(value, -0) ? 0x8000 : 0;
  const abs = Math.abs(value);
  if (abs === 0) return sign;
  if (abs >= 65504) return sign | 0x7bff;
  if (abs < 2 ** -24) return sign;
  if (abs < 2 ** -14) {
    return sign | Math.round(abs / 2 ** -24);
  }
  const exponent = Math.floor(Math.log2(abs));
  const mantissa = Math.round((abs / 2 ** exponent - 1) * 1024);
  if (mantissa === 1024) {
    return sign | ((exponent + 16) << 10);
  }
  return sign | ((exponent + 15) << 10) | (mantissa & 0x03ff);
}

function bfloat16BitsToFloat32(bits: number): number {
  const buffer = Buffer.alloc(4);
  buffer.writeUInt32LE((bits << 16) >>> 0, 0);
  return buffer.readFloatLE(0);
}

function dtypeByteLength(dtype: SafetensorsDtype): number {
  return dtype === "F32" ? 4 : 2;
}

function requireTensor(tensors: Map<string, TensorRef>, name: string): TensorRef {
  const tensor = tensors.get(name);
  if (!tensor) throw new Error(`Missing required tensor ${name}.`);
  return tensor;
}

function fullSlice(rows: number, cols: number): SliceMetadata {
  return { mode: "full", rowStart: 0, rowEnd: rows, colStart: 0, colEnd: cols };
}

function fullVectorSlice(length: number): SliceMetadata {
  return { mode: "full", rowStart: 0, rowEnd: length, colStart: 0, colEnd: 1 };
}

function hasOptionalLayerTensor(tensors: Map<string, TensorRef>, layerCount: number, suffix: string): boolean {
  for (let layerIndex = 0; layerIndex < layerCount; layerIndex += 1) {
    if (tensors.has(`model.layers.${layerIndex}.${suffix}`)) return true;
  }
  return false;
}

function sanitizeShardDir(value: string): string {
  const cleaned = value.trim().replace(/^\/+/, "").replace(/\/+$/, "");
  if (!cleaned || cleaned.includes("..")) throw new Error("shardDir must be a relative output directory.");
  return cleaned;
}

function readPositiveInteger(value: unknown, name: string): number {
  if (typeof value !== "number" || !Number.isInteger(value) || value <= 0) throw new Error(`${name} must be a positive integer.`);
  return value;
}

function readPositiveNumber(value: unknown, name: string): number {
  if (typeof value !== "number" || !Number.isFinite(value) || value <= 0) throw new Error(`${name} must be a positive number.`);
  return value;
}

function readOptionalString(value: unknown): string | undefined {
  return typeof value === "string" && value.trim() ? value.trim() : undefined;
}

function sha256Hex(value: Buffer): string {
  return createHash("sha256").update(value).digest("hex");
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function isNodeError(value: unknown): value is NodeJS.ErrnoException {
  return value instanceof Error && "code" in value;
}
