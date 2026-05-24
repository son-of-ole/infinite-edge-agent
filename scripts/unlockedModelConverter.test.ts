import { spawn } from "node:child_process";
import { createHash } from "node:crypto";
import { mkdtempSync } from "node:fs";
import { mkdir, mkdtemp, readFile, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { afterEach, describe, expect, it, vi } from "vitest";
import { convertUnlockedModel } from "./unlockedModelConverter";

describe("unlockedModelConverter", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("converts local Qwen safetensors into full GQA/RoPE f32 shard manifests", async () => {
    const fixtureDir = await mkdtemp(resolve(tmpdir(), "unlocked-hf-"));
    const outputDir = await mkdtemp(resolve(tmpdir(), "unlocked-out-"));
    await writeQwenFixture(fixtureDir, { tieWordEmbeddings: true });

    const result = await convertUnlockedModel({
      inputDir: fixtureDir,
      outputDir,
      modelId: "Qwen/Test-0.01B",
      maxLayers: 1,
    });

    const manifest = JSON.parse(await readFile(result.manifestPath, "utf8"));
    expect(result.manifestSha256).toMatch(/^[a-f0-9]{64}$/);
    expect(await readFile(result.manifestSha256Path, "utf8")).toContain(result.manifestSha256);
    const envExample = await readFile(result.envPath, "utf8");
    expect(envExample).toContain(`VITE_UNLOCKED_MODEL_MANIFEST_PATH=/models/${outputDir.split("/").at(-1)}/manifest.json`);
    expect(envExample).toContain(`VITE_UNLOCKED_MODEL_MANIFEST_SHA256=${result.manifestSha256}`);
    expect(manifest).toMatchObject({
      schemaVersion: 1,
      modelId: "Qwen/Test-0.01B",
      architecture: "qwen3_decoder_control",
      vocabSize: 4,
      hiddenSize: 2,
      headDim: 2,
      numAttentionHeads: 2,
      numKeyValueHeads: 1,
      maxPositionEmbeddings: 128,
      ropeTheta: 1000000,
      tieWordEmbeddings: true,
      tokenizer: {
        kind: "qwen-bpe",
        tokens: ["<unk>", "alpha", "beta", "MANIFEST_TOKEN"],
        unknownTokenId: 0,
        merges: [["a", "l"], ["al", "pha"]],
        specialTokens: ["<unk>", "MANIFEST_TOKEN"],
        chatTemplate: "{{ qwen chat template }}",
      },
      conversion: {
        sourceFormat: "huggingface-safetensors",
        projectionMode: "full-qwen-gqa-rope",
        tensorFormat: "f32",
      },
      tensorStorage: {
        format: "f32-reference",
      dtype: "f32",
      shardKind: "f32-shard",
      byteWidth: 4,
      productionTarget: "reference",
      runtimeRepresentation: "f32-reference",
      packedRuntimeCompute: false,
    },
    });
    expect(manifest.layers).toHaveLength(1);
    expect(manifest.layers[0].qProj.shape).toEqual([4, 2]);
    expect(manifest.layers[0].kProj.shape).toEqual([2, 2]);
    expect(manifest.layers[0].vProj.shape).toEqual([2, 2]);
    expect(manifest.layers[0].oProj.shape).toEqual([2, 4]);
    expect(manifest.outputProjection.sourceTensor).toBe("model.embed_tokens.weight");
    expect(manifest.outputProjection.slice).toMatchObject({ mode: "full" });
    expect(manifest.layers[0].qProj.sourceTensor).toBe("model.layers.0.self_attn.q_proj.weight");
    expect(manifest.layers[0].qProj.slice).toMatchObject({
      rowStart: 0,
      rowEnd: 4,
      colStart: 0,
      colEnd: 2,
    });
    expect(manifest.layers[0].oProj.slice).toMatchObject({
      rowStart: 0,
      rowEnd: 2,
      colStart: 0,
      colEnd: 4,
    });

    const embedding = await readDescriptorMatrix(outputDir, manifest.tokenEmbedding);
    const outputProjection = await readDescriptorMatrix(outputDir, manifest.outputProjection);
    const qProj = await readDescriptorMatrix(outputDir, manifest.layers[0].qProj);
    const kProj = await readDescriptorMatrix(outputDir, manifest.layers[0].kProj);
    const vProj = await readDescriptorMatrix(outputDir, manifest.layers[0].vProj);
    const oProj = await readDescriptorMatrix(outputDir, manifest.layers[0].oProj);

    expect(embedding).toEqual([
      [1, 0],
      [0, 1],
      [1, 1],
      [0.5, 0.5],
    ]);
    expect(outputProjection).toEqual(embedding);
    expect(qProj).toEqual([
      [1, 2],
      [3, 4],
      [5, 6],
      [7, 8],
    ]);
    expect(kProj[0][0]).toBeCloseTo(1.5, 5);
    expect(vProj[0][0]).toBeCloseTo(-0.5, 5);
    expect(oProj).toEqual([
      [1, 2, 3, 4],
      [5, 6, 7, 8],
    ]);

    for (const descriptor of [
      manifest.tokenEmbedding,
      manifest.outputProjection,
      manifest.layers[0].qProj,
      manifest.layers[0].kProj,
      manifest.layers[0].vProj,
      manifest.layers[0].oProj,
    ]) {
      expect(descriptor).toMatchObject({
        kind: "f32-shard",
        byteOffset: 0,
        dtype: "f32",
      });
      expect(descriptor.sha256).toBe(await sha256File(resolve(outputDir, descriptor.uri)));
    }

    const verifyPath = resolve(fileURLToPath(new URL(".", import.meta.url)), "verify-unlocked-asset.ts");
    const paritySuccess = await runCli([
      resolve(fileURLToPath(new URL("..", import.meta.url)), "node_modules/.bin/tsx"),
      verifyPath,
      "--manifest-path",
      result.manifestPath,
      "--model-id",
      "Qwen/Test-0.01B",
      "--manifest-sha256",
      result.manifestSha256,
      "--require-configured",
      "--require-manifest-sha256",
      "--require-sharded",
      "--require-qwen-parity",
    ], { VITE_UNLOCKED_BACKEND_PREFERENCE: "webgpu" });
    if (paritySuccess.code !== 0) {
      throw new Error(`verify-unlocked strict qwen parity failed\nstdout:\n${paritySuccess.stdout}\nstderr:\n${paritySuccess.stderr}`);
    }
    expect(paritySuccess.stdout).toContain("Mode: configured");
    expect(paritySuccess.stdout).toContain("Backend preference requested: webgpu");
    expect(paritySuccess.stdout).toContain("Logit projection backend:");

    const parityVerification = await runCli([
      resolve(fileURLToPath(new URL("..", import.meta.url)), "node_modules/.bin/tsx"),
      verifyPath,
      "--require-qwen-parity",
    ]);
    expect(parityVerification.code).not.toBe(0);
    expect(`${parityVerification.stdout}\n${parityVerification.stderr}`).toContain("requires positive integer numAttentionHeads");
  }, 15_000);

  it("includes optional Qwen RMSNorm and gated MLP tensors when present", async () => {
    const fixtureDir = await mkdtemp(resolve(tmpdir(), "unlocked-hf-qwen-"));
    const outputDir = await mkdtemp(resolve(tmpdir(), "unlocked-out-qwen-"));
    await writeQwenFixture(fixtureDir, { tieWordEmbeddings: true, includeQwenOptionals: true });

    const result = await convertUnlockedModel({
      inputDir: fixtureDir,
      outputDir,
      modelId: "Qwen/Test-RMSNorm-MLP",
      maxLayers: 1,
    });

    const manifest = JSON.parse(await readFile(result.manifestPath, "utf8"));
    expect(manifest.finalNorm).toMatchObject({
      kind: "f32-shard",
      byteOffset: 0,
      shape: [2],
      dtype: "f32",
      sourceTensor: "model.norm.weight",
      sourceShape: [2],
      slice: { mode: "full", rowStart: 0, rowEnd: 2, colStart: 0, colEnd: 1 },
    });
    expect(manifest).toMatchObject({
      intermediateSize: 2,
      rmsNormEps: 0.000001,
      hiddenActivation: "silu",
    });
    expect(manifest.layers[0].inputLayerNorm.shape).toEqual([2]);
    expect(manifest.layers[0].postAttentionLayerNorm.shape).toEqual([2]);
    expect(manifest.layers[0].qNorm.shape).toEqual([2]);
    expect(manifest.layers[0].kNorm.shape).toEqual([2]);
    expect(manifest.layers[0].mlpGateProj.shape).toEqual([2, 2]);
    expect(manifest.layers[0].mlpUpProj.shape).toEqual([2, 2]);
    expect(manifest.layers[0].mlpDownProj.shape).toEqual([2, 2]);
    expect(manifest.conversion.mlpTensors).toBe("included");

    expect(await readDescriptorVector(outputDir, manifest.finalNorm)).toEqual([0.25, 0.5]);
    expect(await readDescriptorVector(outputDir, manifest.layers[0].inputLayerNorm)).toEqual([1, 1.5]);
    expect(await readDescriptorVector(outputDir, manifest.layers[0].postAttentionLayerNorm)).toEqual([2, 2.5]);
    expect(await readDescriptorVector(outputDir, manifest.layers[0].qNorm)).toEqual([3, 3.5]);
    expect(await readDescriptorVector(outputDir, manifest.layers[0].kNorm)).toEqual([4, 4.5]);
    expect(await readDescriptorMatrix(outputDir, manifest.layers[0].mlpGateProj)).toEqual([
      [2, 3],
      [4, 5],
    ]);
    expect(await readDescriptorMatrix(outputDir, manifest.layers[0].mlpUpProj)).toEqual([
      [1, 0],
      [0, 1],
    ]);
    expect(await readDescriptorMatrix(outputDir, manifest.layers[0].mlpDownProj)).toEqual([
      [1, 0],
      [0, 1],
    ]);

    for (const descriptor of [
      manifest.finalNorm,
      manifest.layers[0].inputLayerNorm,
      manifest.layers[0].postAttentionLayerNorm,
      manifest.layers[0].qNorm,
      manifest.layers[0].kNorm,
      manifest.layers[0].mlpGateProj,
      manifest.layers[0].mlpUpProj,
      manifest.layers[0].mlpDownProj,
    ]) {
      expect(descriptor).toMatchObject({
        kind: "f32-shard",
        byteOffset: 0,
        dtype: "f32",
        sourceFile: "model.safetensors",
      });
      expect(descriptor.sha256).toBe(await sha256File(resolve(outputDir, descriptor.uri)));
    }

    const verifyPath = resolve(fileURLToPath(new URL(".", import.meta.url)), "verify-unlocked-asset.ts");
    const mismatchVerification = await runCli([
      resolve(fileURLToPath(new URL("..", import.meta.url)), "node_modules/.bin/tsx"),
      verifyPath,
      "--manifest-path",
      result.manifestPath,
      "--model-id",
      "Qwen/Wrong-Model",
      "--manifest-sha256",
      result.manifestSha256,
      "--require-configured",
      "--require-manifest-sha256",
      "--require-sharded",
      "--require-qwen-math",
    ]);
    expect(mismatchVerification.code).not.toBe(0);
    expect(`${mismatchVerification.stdout}\n${mismatchVerification.stderr}`).toContain("modelId mismatch");
  });

  it("can emit f16-packed production shards and require them in verification", async () => {
    const fixtureDir = await mkdtemp(resolve(tmpdir(), "unlocked-hf-f16-"));
    const outputDir = await mkdtemp(resolve(tmpdir(), "unlocked-out-f16-"));
    await writeQwenFixture(fixtureDir, { tieWordEmbeddings: true, includeQwenOptionals: true });

    const result = await convertUnlockedModel({
      inputDir: fixtureDir,
      outputDir,
      modelId: "Qwen/Test-F16",
      maxLayers: 1,
      tensorFormat: "f16",
    });

    const manifest = JSON.parse(await readFile(result.manifestPath, "utf8"));
    expect(manifest.tensorStorage).toMatchObject({
      format: "f16-packed",
      dtype: "f16",
      shardKind: "f16-shard",
      byteWidth: 2,
      productionTarget: "webgpu-packed",
      runtimeRepresentation: "packed-f16-runtime-lazy-decode",
      packedRuntimeCompute: false,
    });
    expect(manifest.conversion.tensorFormat).toBe("f16");
    expect(await readFile(result.envPath, "utf8")).toContain("VITE_UNLOCKED_WEIGHT_FORMAT=f16-packed");
    expect(manifest.layers[0].qProj).toMatchObject({
      kind: "f16-shard",
      dtype: "f16",
    });
    expect(await readDescriptorMatrix(outputDir, manifest.layers[0].qProj)).toEqual([
      [1, 2],
      [3, 4],
      [5, 6],
      [7, 8],
    ]);

    const verifyPath = resolve(fileURLToPath(new URL(".", import.meta.url)), "verify-unlocked-asset.ts");
    const packedVerification = await runCli([
      resolve(fileURLToPath(new URL("..", import.meta.url)), "node_modules/.bin/tsx"),
      verifyPath,
      "--manifest-path",
      result.manifestPath,
      "--model-id",
      "Qwen/Test-F16",
      "--manifest-sha256",
      result.manifestSha256,
      "--require-configured",
      "--require-manifest-sha256",
      "--require-sharded",
      "--require-qwen-parity",
      "--require-packed-assets",
    ], { VITE_UNLOCKED_BACKEND_PREFERENCE: "webgpu" });
    if (packedVerification.code !== 0) {
      throw new Error(`verify-unlocked packed f16 failed\nstdout:\n${packedVerification.stdout}\nstderr:\n${packedVerification.stderr}`);
    }

    const f32OutputDir = await mkdtemp(resolve(tmpdir(), "unlocked-out-f32-strict-"));
    const f32Result = await convertUnlockedModel({
      inputDir: fixtureDir,
      outputDir: f32OutputDir,
      modelId: "Qwen/Test-F16",
      maxLayers: 1,
    });
    const unpackedVerification = await runCli([
      resolve(fileURLToPath(new URL("..", import.meta.url)), "node_modules/.bin/tsx"),
      verifyPath,
      "--manifest-path",
      f32Result.manifestPath,
      "--model-id",
      "Qwen/Test-F16",
      "--manifest-sha256",
      f32Result.manifestSha256,
      "--require-configured",
      "--require-manifest-sha256",
      "--require-sharded",
      "--require-packed-assets",
    ]);
    expect(unpackedVerification.code).not.toBe(0);
    expect(`${unpackedVerification.stdout}\n${unpackedVerification.stderr}`).toContain("format=f16-packed");
  }, 15_000);

  it("rejects source tensors that do not exactly match requested full Qwen projection shapes", async () => {
    const fixtureDir = await mkdtemp(resolve(tmpdir(), "unlocked-hf-bad-shape-"));
    const outputDir = await mkdtemp(resolve(tmpdir(), "unlocked-out-bad-shape-"));
    await writeQwenFixture(fixtureDir, { tieWordEmbeddings: true, oversizeKeyValueTensors: true });

    await expect(convertUnlockedModel({
      inputDir: fixtureDir,
      outputDir,
      modelId: "Qwen/Test-Bad-Shape",
      maxLayers: 1,
    })).rejects.toThrow("must exactly match requested full matrix shape");
  });

  it("exposes a no-network CLI that prints the manifest path and manifest hash", async () => {
    const fixtureDir = await mkdtemp(resolve(tmpdir(), "unlocked-hf-cli-"));
    const outputDir = await mkdtemp(resolve(tmpdir(), "unlocked-out-cli-"));
    await writeQwenFixture(fixtureDir, { tieWordEmbeddings: false });

    const cliPath = resolve(fileURLToPath(new URL(".", import.meta.url)), "convert-unlocked-model.ts");
    const result = await runCli([
      resolve(fileURLToPath(new URL("..", import.meta.url)), "node_modules/.bin/tsx"),
      cliPath,
      "--input",
      fixtureDir,
      "--output",
      outputDir,
      "--model-id",
      "Qwen/Test-CLI",
      "--max-layers",
      "1",
    ]);

    expect(result.code).toBe(0);
    expect(result.stdout).toContain("Manifest:");
    expect(result.stdout).toContain("Manifest SHA-256:");
    expect(result.stdout).toContain("Env example:");
    expect(result.stderr).toBe("");
    const manifest = JSON.parse(await readFile(resolve(outputDir, "manifest.json"), "utf8"));
    expect(manifest.modelId).toBe("Qwen/Test-CLI");
    expect(manifest.outputProjection.sourceTensor).toBe("lm_head.weight");
  });
});

async function writeQwenFixture(inputDir: string, options: { tieWordEmbeddings: boolean; includeQwenOptionals?: boolean; oversizeKeyValueTensors?: boolean }): Promise<void> {
  await mkdir(inputDir, { recursive: true });
  await writeFile(resolve(inputDir, "config.json"), `${JSON.stringify({
    architectures: ["Qwen3ForCausalLM"],
    vocab_size: 4,
    hidden_size: 2,
    intermediate_size: options.includeQwenOptionals ? 2 : undefined,
    rms_norm_eps: options.includeQwenOptionals ? 0.000001 : undefined,
    hidden_act: options.includeQwenOptionals ? "silu" : undefined,
    head_dim: 2,
    num_hidden_layers: 2,
    num_attention_heads: 2,
    num_key_value_heads: 1,
    max_position_embeddings: 128,
    rope_theta: 1000000,
    tie_word_embeddings: options.tieWordEmbeddings,
  }, null, 2)}\n`);
  await writeFile(resolve(inputDir, "tokenizer.json"), `${JSON.stringify({
    added_tokens: [
      { id: 0, content: "<unk>", special: true },
      { id: 3, content: "MANIFEST_TOKEN", special: true },
    ],
    model: {
      type: "BPE",
      vocab: {
        "<unk>": 0,
        alpha: 1,
        beta: 2,
        MANIFEST_TOKEN: 3,
      },
      merges: [["a", "l"], ["al", "pha"]],
      unk_token: "<unk>",
    },
  }, null, 2)}\n`);
  await writeFile(resolve(inputDir, "tokenizer_config.json"), `${JSON.stringify({
    chat_template: "{{ qwen chat template }}",
    unk_token: "<unk>",
    additional_special_tokens: ["MANIFEST_TOKEN"],
  }, null, 2)}\n`);

  const tensors: Record<string, FixtureTensor> = {
    "model.embed_tokens.weight": {
      dtype: "F32",
      shape: [4, 2],
      values: [1, 0, 0, 1, 1, 1, 0.5, 0.5],
    },
    "lm_head.weight": {
      dtype: "F32",
      shape: [4, 2],
      values: [0, 0, 0, 0, 10, 10, -1, -1],
    },
    "model.layers.0.self_attn.q_proj.weight": {
      dtype: "F32",
      shape: [4, 2],
      values: [1, 2, 3, 4, 5, 6, 7, 8],
    },
    "model.layers.0.self_attn.k_proj.weight": {
      dtype: "F16",
      shape: options.oversizeKeyValueTensors ? [4, 2] : [2, 2],
      values: options.oversizeKeyValueTensors
        ? [1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5]
        : [1.5, 2.5, 3.5, 4.5],
    },
    "model.layers.0.self_attn.v_proj.weight": {
      dtype: "BF16",
      shape: options.oversizeKeyValueTensors ? [4, 2] : [2, 2],
      values: options.oversizeKeyValueTensors
        ? [-0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5]
        : [-0.5, 1.5, 2.5, 3.5],
    },
    "model.layers.0.self_attn.o_proj.weight": {
      dtype: "F32",
      shape: [2, 4],
      values: [1, 2, 3, 4, 5, 6, 7, 8],
    },
    "model.layers.1.self_attn.q_proj.weight": {
      dtype: "F32",
      shape: [4, 2],
      values: [9, 9, 9, 9, 9, 9, 9, 9],
    },
    "model.layers.1.self_attn.k_proj.weight": {
      dtype: "F32",
      shape: [2, 2],
      values: [9, 9, 9, 9],
    },
    "model.layers.1.self_attn.v_proj.weight": {
      dtype: "F32",
      shape: [2, 2],
      values: [9, 9, 9, 9],
    },
    "model.layers.1.self_attn.o_proj.weight": {
      dtype: "F32",
      shape: [2, 4],
      values: [9, 9, 9, 9, 9, 9, 9, 9],
    },
  };
  if (options.includeQwenOptionals) {
    Object.assign(tensors, {
      "model.norm.weight": {
        dtype: "F32",
        shape: [2],
        values: [0.25, 0.5],
      },
      "model.layers.0.input_layernorm.weight": {
        dtype: "F32",
        shape: [2],
        values: [1, 1.5],
      },
      "model.layers.0.post_attention_layernorm.weight": {
        dtype: "F32",
        shape: [2],
        values: [2, 2.5],
      },
      "model.layers.0.self_attn.q_norm.weight": {
        dtype: "F32",
        shape: [2],
        values: [3, 3.5],
      },
      "model.layers.0.self_attn.k_norm.weight": {
        dtype: "F32",
        shape: [2],
        values: [4, 4.5],
      },
      "model.layers.0.mlp.gate_proj.weight": {
        dtype: "F32",
        shape: [2, 2],
        values: [2, 3, 4, 5],
      },
      "model.layers.0.mlp.up_proj.weight": {
        dtype: "F32",
        shape: [2, 2],
        values: [1, 0, 0, 1],
      },
      "model.layers.0.mlp.down_proj.weight": {
        dtype: "F32",
        shape: [2, 2],
        values: [1, 0, 0, 1],
      },
    } satisfies Record<string, FixtureTensor>);
  }
  if (options.tieWordEmbeddings) delete tensors["lm_head.weight"];
  await writeFile(resolve(inputDir, "model.safetensors"), makeSafetensors(tensors));
}

interface FixtureTensor {
  dtype: "F32" | "F16" | "BF16";
  shape: number[];
  values: number[];
}

function makeSafetensors(tensors: Record<string, FixtureTensor>): Buffer {
  const header: Record<string, { dtype: string; shape: number[]; data_offsets: [number, number] }> = {};
  const chunks: Buffer[] = [];
  let offset = 0;
  for (const [name, tensor] of Object.entries(tensors)) {
    const bytes = encodeTensor(tensor);
    header[name] = {
      dtype: tensor.dtype,
      shape: tensor.shape,
      data_offsets: [offset, offset + bytes.byteLength],
    };
    chunks.push(bytes);
    offset += bytes.byteLength;
  }
  const headerBytes = Buffer.from(JSON.stringify(header), "utf8");
  const lengthPrefix = Buffer.alloc(8);
  lengthPrefix.writeBigUInt64LE(BigInt(headerBytes.byteLength), 0);
  return Buffer.concat([lengthPrefix, headerBytes, ...chunks]);
}

function encodeTensor(tensor: FixtureTensor): Buffer {
  if (tensor.dtype === "F32") {
    const buffer = Buffer.alloc(tensor.values.length * 4);
    tensor.values.forEach((value, index) => buffer.writeFloatLE(value, index * 4));
    return buffer;
  }
  const buffer = Buffer.alloc(tensor.values.length * 2);
  tensor.values.forEach((value, index) => {
    const bits = tensor.dtype === "F16" ? float32ToFloat16Bits(value) : float32ToBfloat16Bits(value);
    buffer.writeUInt16LE(bits, index * 2);
  });
  return buffer;
}

function float32ToBfloat16Bits(value: number): number {
  const buffer = Buffer.alloc(4);
  buffer.writeFloatLE(value, 0);
  return buffer.readUInt32LE(0) >>> 16;
}

function float32ToFloat16Bits(value: number): number {
  if (Number.isNaN(value)) return 0x7e00;
  if (value === Infinity) return 0x7c00;
  if (value === -Infinity) return 0xfc00;
  const sign = value < 0 || Object.is(value, -0) ? 0x8000 : 0;
  const absolute = Math.abs(value);
  if (absolute === 0) return sign;
  if (absolute >= 65504) return sign | 0x7bff;
  if (absolute < 2 ** -14) return sign | Math.round(absolute / 2 ** -24);
  const exponent = Math.floor(Math.log2(absolute));
  const mantissa = Math.round((absolute / 2 ** exponent - 1) * 1024);
  return sign | ((exponent + 15) << 10) | (mantissa & 0x3ff);
}

async function readDescriptorMatrix(outputDir: string, descriptor: { uri: string; shape: [number, number]; sha256: string; kind?: string; dtype?: string }): Promise<number[][]> {
  const shardPath = resolve(outputDir, descriptor.uri);
  expect(await sha256File(shardPath)).toBe(descriptor.sha256);
  const bytes = await readFile(shardPath);
  const values = readDescriptorValues(bytes, descriptor);
  const rows: number[][] = [];
  for (let row = 0; row < descriptor.shape[0]; row += 1) {
    const start = row * descriptor.shape[1];
    rows.push(Array.from(values.slice(start, start + descriptor.shape[1])));
  }
  return rows;
}

async function readDescriptorVector(outputDir: string, descriptor: { uri: string; shape: [number]; sha256: string; kind?: string; dtype?: string }): Promise<number[]> {
  const shardPath = resolve(outputDir, descriptor.uri);
  expect(await sha256File(shardPath)).toBe(descriptor.sha256);
  const bytes = await readFile(shardPath);
  return Array.from(readDescriptorValues(bytes, descriptor));
}

function readDescriptorValues(bytes: Buffer, descriptor: { kind?: string; dtype?: string }): Float32Array {
  if (descriptor.kind === "f16-shard" || descriptor.dtype === "f16") {
    const packed = new Uint16Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / Uint16Array.BYTES_PER_ELEMENT);
    const values = new Float32Array(packed.length);
    for (let index = 0; index < packed.length; index += 1) {
      values[index] = float16BitsToFloat32(packed[index] ?? 0);
    }
    return values;
  }
  return new Float32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / Float32Array.BYTES_PER_ELEMENT);
}

function float16BitsToFloat32(bits: number): number {
  const sign = bits & 0x8000 ? -1 : 1;
  const exponent = (bits >> 10) & 0x1f;
  const mantissa = bits & 0x03ff;
  if (exponent === 0) return sign * (mantissa === 0 ? 0 : (mantissa / 1024) * 2 ** -14);
  if (exponent === 0x1f) return mantissa === 0 ? sign * Infinity : Number.NaN;
  return sign * (1 + mantissa / 1024) * 2 ** (exponent - 15);
}

async function sha256File(path: string): Promise<string> {
  return createHash("sha256").update(await readFile(path)).digest("hex");
}

function runCli(command: string[], extraEnv: Record<string, string> = {}): Promise<{ code: number | null; stdout: string; stderr: string }> {
  return new Promise((resolvePromise) => {
    const env = { ...process.env };
    delete env.RELEASE_REQUIRE_UNLOCKED_MODEL;
    delete env.RELEASE_REQUIRE_UNLOCKED_QWEN_MATH;
    delete env.RELEASE_REQUIRE_UNLOCKED_QWEN_PARITY;
    delete env.RELEASE_REQUIRE_UNLOCKED_PACKED_ASSETS;
    delete env.VITE_UNLOCKED_MODEL_MANIFEST_PATH;
    delete env.VITE_UNLOCKED_MODEL_MANIFEST_SHA256;
    delete env.VITE_UNLOCKED_BACKEND_PREFERENCE;
    delete env.VITE_UNLOCKED_MODEL_ALLOW_FIXTURE;
    env.EVAL_ARTIFACT_DIR = mkdtempSync(resolve(tmpdir(), "unlocked-verify-test-artifacts-"));
    Object.assign(env, extraEnv);
    const child = spawn(command[0], command.slice(1), {
      cwd: resolve(fileURLToPath(new URL("..", import.meta.url))),
      env,
      stdio: ["ignore", "pipe", "pipe"],
    });
    let stdout = "";
    let stderr = "";
    child.stdout.setEncoding("utf8");
    child.stderr.setEncoding("utf8");
    child.stdout.on("data", (chunk) => {
      stdout += chunk;
    });
    child.stderr.on("data", (chunk) => {
      stderr += chunk;
    });
    child.on("error", (error) => {
      resolvePromise({ code: 1, stdout, stderr: error.message });
    });
    child.on("close", (code) => {
      resolvePromise({ code, stdout, stderr });
    });
  });
}
