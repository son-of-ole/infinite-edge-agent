import { existsSync } from "node:fs";
import { mkdir, readFile, writeFile } from "node:fs/promises";
import { join, resolve } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import {
  UnlockedBrowserTransformerBackend,
  type SSALayerRoutingPolicy,
  type UnlockedBrowserTransformerWeights,
} from "@infinite-edge-agent/core";
import {
  UnlockedBrowserTransformerClient,
  __unlockedBrowserTransformerClientTestHooks,
  type UnlockedBrowserDecodeProof,
} from "../apps/web/src/lib/llm/unlockedBrowserTransformerClient";

export interface QwenParityThresholds {
  maxAbsError: number;
  deterministicTokenParity: true;
  retrievalGroundedResponse: true;
}

export interface QwenParityGate {
  name: string;
  passed: boolean;
  actual: number | string | boolean;
  expected: number | string | boolean;
}

export interface QwenParitySuite {
  name: string;
  passed: boolean;
  metrics: Record<string, number | string | boolean>;
  gates: QwenParityGate[];
}

export interface QwenParityFixtures {
  rmsnorm: { input: number[]; weights: number[]; actual: number[]; expected: number[] };
  rope: { position: number; input: number[]; actual: number[]; expected: number[] };
  gqa: { queryHeads: number[][]; keyValueHeadMap: number[]; actual: number[][]; expected: number[][] };
  gatedMlp: { input: number[]; actual: number[]; expected: number[] };
  tokenizerChatTemplate: { prompt: string; tokenIds: number[]; expectedTokenIds: number[]; rendered: string };
  logitProjection: { hidden: number[]; selectedLogits: Array<{ tokenId: number; logit: number }>; expectedSelectedLogits: Array<{ tokenId: number; logit: number }> };
  multiTokenDecode: {
    promptTokenIds: number[];
    seedInputTokenId: number;
    requestedTokenCount: number;
    generatedTokenIds: number[];
    expectedGeneratedTokenIds: number[];
    generatedTokenCount: number;
    perStep: Array<{ step: number; inputTokenId: number; generatedTokenId: number; logitTokenIds: number[] }>;
  };
  retrievalGroundedResponse: { prompt: string; retrievedSourceIds: string[]; response: string };
}

export interface QwenRealModelParityArtifact {
  mode: "skipped" | "installed";
  manifestPath?: string;
  prompt?: string;
  promptTokenIds?: number[];
  generatedTokenIds?: number[];
  response?: string;
  backendProof?: Pick<
    UnlockedBrowserDecodeProof,
    | "tensorControl"
    | "tspSteps"
    | "kvPagingEvents"
    | "tokenId"
    | "logitProjectionBackend"
    | "logitProjectionPurpose"
    | "logitProjectionSelectedRows"
    | "logitProjectionFullRows"
  >;
  reason?: string;
  error?: string;
}

export interface QwenParityAccuracyResult {
  name: "qwen-parity-accuracy";
  createdAt: string;
  passed: boolean;
  thresholds: QwenParityThresholds;
  summary: {
    maxAbsError: number;
    failedGateCount: number;
    tokenParityPassed: boolean;
    multiTokenDecodePassed: boolean;
    retrievalGroundedPassed: boolean;
    realModelParityMode: "skipped" | "installed";
  };
  suites: QwenParitySuite[];
  fixtures: QwenParityFixtures;
  realModelParity: QwenRealModelParityArtifact;
}

export interface QwenParityEvalOptions {
  manifestPath?: string;
  requireRealModelParity?: boolean;
  thresholds?: Partial<Pick<QwenParityThresholds, "maxAbsError">>;
}

export interface QwenParityArtifactPaths {
  resultsPath: string;
  latestPath: string;
  summaryPath: string;
}

const repoRoot = resolve(new URL("..", import.meta.url).pathname);
const defaultThresholds: QwenParityThresholds = {
  maxAbsError: 0.000001,
  deterministicTokenParity: true,
  retrievalGroundedResponse: true,
};

export async function evaluateQwenParityAccuracy(options: QwenParityEvalOptions = {}): Promise<QwenParityAccuracyResult> {
  const thresholds = { ...defaultThresholds, ...options.thresholds };
  const fixtures = await buildFixtures();
  const realModelParity = await realModelParitySuite(resolveManifestPath(options.manifestPath), Boolean(options.requireRealModelParity));
  const suites = [
    mathSuite("rmsnorm", fixtures.rmsnorm.actual, fixtures.rmsnorm.expected, thresholds.maxAbsError),
    mathSuite("rope", fixtures.rope.actual, fixtures.rope.expected, thresholds.maxAbsError),
    mathSuite("gqa", fixtures.gqa.actual.flat(), fixtures.gqa.expected.flat(), thresholds.maxAbsError),
    mathSuite("gated-mlp", fixtures.gatedMlp.actual, fixtures.gatedMlp.expected, thresholds.maxAbsError),
    tokenParitySuite(fixtures.tokenizerChatTemplate),
    logitProjectionSuite(fixtures.logitProjection, thresholds.maxAbsError),
    multiTokenDecodeSuite(fixtures.multiTokenDecode),
    retrievalGroundedSuite(fixtures.retrievalGroundedResponse),
    realModelParity.suite,
  ];
  const failedGateCount = suites.reduce((sum, suite) => sum + suite.gates.filter((gate) => !gate.passed).length, 0);
  const maxAbsError = Math.max(...suites.map((suite) => Number(suite.metrics.maxAbsError ?? 0)));
  const realModelSuite = suites.find((suite) => suite.name === "real-model-parity");
  return {
    name: "qwen-parity-accuracy",
    createdAt: new Date().toISOString(),
    passed: suites.every((suite) => suite.passed),
    thresholds,
    summary: {
      maxAbsError,
      failedGateCount,
      tokenParityPassed: suites.find((suite) => suite.name === "tokenizer-chat-template")?.passed ?? false,
      multiTokenDecodePassed: suites.find((suite) => suite.name === "multi-token-decode")?.passed ?? false,
      retrievalGroundedPassed: suites.find((suite) => suite.name === "retrieval-grounded-response")?.passed ?? false,
      realModelParityMode: realModelSuite?.metrics.mode === "installed" ? "installed" : "skipped",
    },
    suites,
    fixtures,
    realModelParity: realModelParity.artifact,
  };
}

export function buildQwenParityAccuracyArtifact(result: QwenParityAccuracyResult): QwenParityAccuracyResult {
  return result;
}

export async function writeQwenParityAccuracyArtifact(
  result: QwenParityAccuracyResult,
  options: { artifactDir?: string } = {},
): Promise<QwenParityArtifactPaths> {
  const artifactRoot = resolve(repoRoot, options.artifactDir ?? process.env.EVAL_ARTIFACT_DIR ?? ".artifacts/evals");
  const timestamp = result.createdAt.replace(/[:.]/g, "-");
  const outputDir = join(artifactRoot, "qwen-parity-accuracy", timestamp);
  await mkdir(outputDir, { recursive: true });
  await mkdir(artifactRoot, { recursive: true });
  const resultsPath = join(outputDir, "results.json");
  const summaryPath = join(outputDir, "summary.md");
  const latestPath = join(artifactRoot, "qwen-parity-accuracy-latest.json");
  const artifact = buildQwenParityAccuracyArtifact(result);
  await writeFile(resultsPath, `${JSON.stringify(artifact, null, 2)}\n`);
  await writeFile(summaryPath, buildSummary(artifact));
  await writeFile(latestPath, `${JSON.stringify(artifact, null, 2)}\n`);
  return { resultsPath, summaryPath, latestPath };
}

async function buildFixtures(): Promise<QwenParityFixtures> {
  const rmsInput = [1, -2, 3, -4];
  const rmsWeights = [0.5, 1, 1.5, 2];
  const ropeInput = [0.25, -0.5, 0.75, 1];
  const queryHeads = [
    [0.3, 0.7],
    [0.1, 0.4],
    [0.8, -0.2],
    [0.5, 0.5],
  ];
  const keys = [
    [[0.25, 0.75], [0.6, -0.1]],
    [[0.9, 0.1], [0.2, 0.8]],
  ];
  const values = [
    [[1, 0], [0.25, 0.75]],
    [[0, 1], [0.5, 0.5]],
  ];
  const mlpInput = [0.5, -1, 1.5];
  const tokenizerFixture = buildProductionTokenizerFixture();
  const hidden = [0.25, -0.5, 1];
  const selectedTokenIds = [0, 2, 4];
  const selectedLogits = projectSelectedLogits(hidden, selectedTokenIds);
  const multiTokenDecode = await runMultiTokenDecodeFixture();
  return {
    rmsnorm: {
      input: rmsInput,
      weights: rmsWeights,
      actual: rmsNorm(rmsInput, rmsWeights, 0.000001),
      expected: [0.18257417, -0.73029669, 1.64316756, -2.92118678],
    },
    rope: {
      position: 3,
      input: ropeInput,
      actual: applyRope(ropeInput, 3, 10000),
      expected: [-0.35333813, -0.52977052, -0.70721437, 0.98455228],
    },
    gqa: {
      queryHeads,
      keyValueHeadMap: [0, 0, 1, 1],
      actual: runGqa(queryHeads, keys, values),
      expected: [
        [0.68932322, 0.31067678],
        [0.66528166, 0.33471834],
        [0.18936117, 0.81063883],
        [0.25, 0.75],
      ],
    },
    gatedMlp: {
      input: mlpInput,
      actual: gatedMlp(mlpInput),
      expected: [-0.61103461, 0.2043288],
    },
    tokenizerChatTemplate: {
      prompt: "literal special-token escaping",
      tokenIds: tokenizerFixture.tokenIds,
      expectedTokenIds: tokenizerFixture.expectedTokenIds,
      rendered: tokenizerFixture.rendered,
    },
    logitProjection: {
      hidden,
      selectedLogits,
      expectedSelectedLogits: [
        { tokenId: 0, logit: 0.125 },
        { tokenId: 2, logit: 0.75 },
        { tokenId: 4, logit: 0.3125 },
      ],
    },
    multiTokenDecode,
    retrievalGroundedResponse: {
      prompt: "Which launch gate is blocking production?",
      retrievedSourceIds: ["doc:qwen-parity-thresholds", "doc:retrieval-grounding"],
      response: "The required production gate is Qwen parity thresholds, sourced from doc:qwen-parity-thresholds.",
    },
  };
}

function mathSuite(name: string, actual: number[], expected: number[], threshold: number): QwenParitySuite {
  const maxAbsError = maxError(actual, expected);
  return suite(name, { maxAbsError, valueCount: actual.length }, [
    gate("max abs error", maxAbsError <= threshold, `<= ${threshold}`, maxAbsError),
  ]);
}

function tokenParitySuite(fixture: QwenParityFixtures["tokenizerChatTemplate"]): QwenParitySuite {
  const parity = arraysEqual(fixture.tokenIds, fixture.expectedTokenIds);
  const specialTokenEscaped = fixture.rendered.includes("<|im_end| >") && !fixture.rendered.includes("literal <|im_end|> should");
  return suite("tokenizer-chat-template", {
    tokenCount: fixture.tokenIds.length,
    tokenParity: parity,
    specialTokenEscaped,
  }, [
    gate("deterministic token parity", parity, true, parity),
    gate("special-token escaping", specialTokenEscaped, true, specialTokenEscaped),
  ]);
}

function logitProjectionSuite(fixture: QwenParityFixtures["logitProjection"], threshold: number): QwenParitySuite {
  const actual = fixture.selectedLogits.map((item) => item.logit);
  const expected = fixture.expectedSelectedLogits.map((item) => item.logit);
  const tokenIdsMatch = JSON.stringify(fixture.selectedLogits.map((item) => item.tokenId)) === JSON.stringify(fixture.expectedSelectedLogits.map((item) => item.tokenId));
  const maxAbsError = maxError(actual, expected);
  return suite("logit-projection", { maxAbsError, selectedTokenCount: actual.length, tokenIdsMatch }, [
    gate("selected token ids", tokenIdsMatch, true, tokenIdsMatch),
    gate("selected logit max abs error", maxAbsError <= threshold, `<= ${threshold}`, maxAbsError),
  ]);
}

function multiTokenDecodeSuite(fixture: QwenParityFixtures["multiTokenDecode"]): QwenParitySuite {
  const generatedTokenParity = arraysEqual(fixture.generatedTokenIds, fixture.expectedGeneratedTokenIds);
  const generatedTokenCountMatches = fixture.generatedTokenCount === fixture.requestedTokenCount
    && fixture.generatedTokenCount === fixture.expectedGeneratedTokenIds.length;
  const perStepAccountingMatches = fixture.perStep.length === fixture.requestedTokenCount
    && fixture.perStep.every((step, index) => step.step === index && step.generatedTokenId === fixture.generatedTokenIds[index]);
  return suite("multi-token-decode", {
    promptTokenCount: fixture.promptTokenIds.length,
    requestedTokenCount: fixture.requestedTokenCount,
    generatedTokenCount: fixture.generatedTokenCount,
    expectedGeneratedTokenCount: fixture.expectedGeneratedTokenIds.length,
    generatedTokenParity,
    perStepAccountingMatches,
  }, [
    gate("generated token ids parity", generatedTokenParity, true, generatedTokenParity),
    gate("generated token count", generatedTokenCountMatches, fixture.requestedTokenCount, fixture.generatedTokenCount),
    gate("per-step decode accounting", perStepAccountingMatches, true, perStepAccountingMatches),
  ]);
}

function retrievalGroundedSuite(fixture: QwenParityFixtures["retrievalGroundedResponse"]): QwenParitySuite {
  const citesRetrievedSource = fixture.retrievedSourceIds.some((sourceId) => fixture.response.includes(sourceId));
  const avoidsUnsupportedAnswer = !fixture.response.toLowerCase().includes("opaque chat api satisfies unlocked");
  const grounded = citesRetrievedSource && avoidsUnsupportedAnswer;
  return suite("retrieval-grounded-response", {
    retrievedSourceCount: fixture.retrievedSourceIds.length,
    citesRetrievedSource,
    avoidsUnsupportedAnswer,
  }, [
    gate("retrieval grounded response", grounded, true, grounded),
  ]);
}

async function realModelParitySuite(
  manifestPath: string,
  required: boolean,
): Promise<{ suite: QwenParitySuite; artifact: QwenRealModelParityArtifact }> {
  if (!manifestPath) {
    if (required) throw new Error("Strict Qwen real model parity requires VITE_UNLOCKED_MODEL_MANIFEST_PATH or --manifest-path.");
    const artifact: QwenRealModelParityArtifact = { mode: "skipped", reason: "converted Qwen manifest not installed" };
    return {
      artifact,
      suite: suite("real-model-parity", { mode: "skipped", reason: artifact.reason as string }, [
        gate("optional unless installed", true, true, true),
      ]),
    };
  }
  if (!existsSync(manifestPath)) {
    if (required) throw new Error(`Strict Qwen real model parity manifest was not found: ${manifestPath}`);
    const artifact: QwenRealModelParityArtifact = { mode: "skipped", manifestPath, reason: `manifest not found: ${manifestPath}` };
    return {
      artifact,
      suite: suite("real-model-parity", { mode: "skipped", reason: artifact.reason as string }, [
        gate("optional unless installed", true, true, true),
      ]),
    };
  }
  try {
    await installFileFetch();
    const manifest = JSON.parse(await readFile(manifestPath, "utf8")) as Record<string, unknown>;
    const tokenizer = isRecord(manifest.tokenizer) ? manifest.tokenizer : {};
    const conversion = isRecord(manifest.conversion) ? manifest.conversion : {};
    const modelId = typeof manifest.modelId === "string" && manifest.modelId.trim() ? manifest.modelId : "Qwen/Qwen3-0.6B";
    const vocabSize = Number(manifest.vocabSize ?? 0);
    const client = new UnlockedBrowserTransformerClient({
      modelId,
      manifestPath: pathToFileUrl(manifestPath),
      allowFixtureWeights: false,
      backendPreference: "cpu",
      maxRuntimePromptTokens: required ? null : 16,
      maxRuntimeLayers: required ? null : 1,
      logitTopK: 8,
      logitTileRows: 4096,
      maxGenerationTokens: 2,
    });
    await client.init();
    const chunks: string[] = [];
    const prompt = "alpha beta";
    for await (const chunk of client.streamChat([{ role: "user", content: prompt }])) {
      chunks.push(chunk);
    }
    const proof = client.lastDecodeProof;
    const generatedTokenIds = client.lastGeneratedTokenIds;
    const artifact: QwenRealModelParityArtifact = {
      mode: "installed",
      manifestPath,
      prompt,
      promptTokenIds: client.lastPromptTokenIds,
      generatedTokenIds,
      response: chunks.join(""),
      ...(proof
        ? {
            backendProof: {
              tensorControl: proof.tensorControl,
              tspSteps: proof.tspSteps,
              kvPagingEvents: proof.kvPagingEvents,
              tokenId: proof.tokenId,
              logitProjectionBackend: proof.logitProjectionBackend,
              logitProjectionPurpose: proof.logitProjectionPurpose,
              logitProjectionSelectedRows: proof.logitProjectionSelectedRows,
              logitProjectionFullRows: proof.logitProjectionFullRows,
            },
          }
        : {}),
    };
    const installedChecks = [
      gate("qwen model id", modelId === "Qwen/Qwen3-0.6B", "Qwen/Qwen3-0.6B", modelId),
      gate("qwen vocab size", vocabSize === 151936, 151936, vocabSize),
      gate("qwen architecture", manifest.architecture === "qwen3_decoder_control", "qwen3_decoder_control", String(manifest.architecture ?? "")),
      gate("qwen tokenizer", tokenizer.kind === "qwen-bpe", "qwen-bpe", String(tokenizer.kind ?? "")),
      gate("qwen projection mode", conversion.projectionMode === "full-qwen-gqa-rope", "full-qwen-gqa-rope", String(conversion.projectionMode ?? "")),
      gate("qwen layers present", Array.isArray(manifest.layers) && manifest.layers.length > 0, true, Array.isArray(manifest.layers) && manifest.layers.length > 0),
      gate("manifest-backed decode generated tokens", generatedTokenIds.length > 0, "> 0", generatedTokenIds.length),
      gate("manifest-backed decode proof", Boolean(proof?.tensorControl), true, Boolean(proof?.tensorControl)),
      gate("full-vocab top-k logits", proof?.logitProjectionPurpose === "full_vocab_topk_logit_projection", "full_vocab_topk_logit_projection", proof?.logitProjectionPurpose ?? "missing"),
      gate("full-vocab rows covered", proof?.logitProjectionFullRows === vocabSize, vocabSize, Number(proof?.logitProjectionFullRows ?? 0)),
      gate("top-k rows selected", Number(proof?.logitProjectionSelectedRows ?? 0) > 0, "> 0", Number(proof?.logitProjectionSelectedRows ?? 0)),
    ];
    return {
      artifact,
      suite: suite("real-model-parity", {
        mode: "installed",
        manifestPath,
        modelId,
        vocabSize,
        layerCount: Array.isArray(manifest.layers) ? manifest.layers.length : 0,
        generatedTokenCount: generatedTokenIds.length,
        logitProjectionBackend: proof?.logitProjectionBackend ?? "unknown",
        logitProjectionPurpose: proof?.logitProjectionPurpose ?? "unknown",
        logitProjectionSelectedRows: proof?.logitProjectionSelectedRows ?? 0,
        logitProjectionFullRows: proof?.logitProjectionFullRows ?? 0,
      }, installedChecks),
    };
  } catch (error) {
    if (required) throw error;
    const message = error instanceof Error ? error.message : String(error);
    const artifact: QwenRealModelParityArtifact = { mode: "installed", manifestPath, error: message };
    return {
      artifact,
      suite: suite("real-model-parity", {
        mode: "installed",
        manifestPath,
        error: message,
      }, [
        gate("manifest-backed decode execution", false, true, message),
      ]),
    };
  }
}

function rmsNorm(input: number[], weight: number[], eps: number): number[] {
  const meanSquare = input.reduce((sum, value) => sum + value * value, 0) / input.length;
  const scale = 1 / Math.sqrt(meanSquare + eps);
  return input.map((value, index) => round(value * scale * (weight[index] ?? 1)));
}

function applyRope(input: number[], position: number, theta: number): number[] {
  const half = input.length / 2;
  const output = [...input];
  for (let index = 0; index < half; index += 1) {
    const inverseFrequency = 1 / Math.pow(theta, (2 * index) / input.length);
    const angle = position * inverseFrequency;
    const cos = Math.cos(angle);
    const sin = Math.sin(angle);
    const first = input[index] ?? 0;
    const second = input[index + half] ?? 0;
    output[index] = round(first * cos - second * sin);
    output[index + half] = round(first * sin + second * cos);
  }
  return output;
}

function runGqa(queryHeads: number[][], keys: number[][][], values: number[][][]): number[][] {
  const groupSize = queryHeads.length / keys.length;
  return queryHeads.map((query, headIndex) => {
    const kvIndex = Math.floor(headIndex / groupSize);
    const logits = keys[kvIndex].map((key) => dot(query, key) / Math.sqrt(query.length));
    const weights = softmax(logits);
    return values[kvIndex][0].map((_, dim) => round(weights.reduce((sum, weight, tokenIndex) => sum + weight * values[kvIndex][tokenIndex][dim], 0)));
  });
}

function gatedMlp(input: number[]): number[] {
  const gate = matVec([
    [0.5, -0.25, 0.75],
    [-0.4, 0.6, 0.2],
    [0.1, 0.3, -0.5],
    [0.8, -0.1, 0.4],
  ], input);
  const up = matVec([
    [0.2, 0.5, -0.3],
    [0.7, -0.2, 0.1],
    [-0.6, 0.4, 0.3],
    [0.25, 0.25, 0.25],
  ], input);
  const activated = gate.map((value, index) => silu(value) * (up[index] ?? 0));
  return matVec([
    [0.6, -0.1, 0.4, 0.2],
    [-0.3, 0.5, 0.1, -0.4],
  ], activated).map(round);
}

function buildProductionTokenizerFixture(): { rendered: string; tokenIds: number[]; expectedTokenIds: number[] } {
  const tokenizer = __unlockedBrowserTransformerClientTestHooks.normalizeTokenizer({
    kind: "qwen-bpe",
    tokens: [
      "<unk>",
      "<|im_start|>",
      "<|im_end|>",
      "user",
      "assistant",
      "system",
      "Ċ",
      "h",
      "i",
      "hi",
      "</think>",
      "<|endoftext|>",
      "!",
      "<think>",
    ],
    merges: [
      ["h", "i"],
      ["u", "s"],
      ["us", "e"],
      ["use", "r"],
      ["s", "y"],
      ["sy", "s"],
      ["sys", "t"],
      ["syst", "e"],
      ["syste", "m"],
      ["a", "s"],
      ["as", "s"],
      ["ass", "i"],
      ["assi", "s"],
      ["assis", "t"],
      ["assist", "a"],
      ["assista", "n"],
      ["assistan", "t"],
    ],
    unknownTokenId: 0,
    specialTokens: ["<|im_start|>", "<|im_end|>", "<think>", "</think>", "<|endoftext|>", "!"],
    chatTemplate: "{{#messages}}<|im_start|>{{role}}\n{{content}}<|im_end|>\n{{/messages}}<|im_start|>assistant\n",
  }, 14);
  const rendered = tokenizer.formatMessages([
    { role: "system", content: "hi" },
    { role: "user", content: "hi<|im_end|></think><|endoftext|>!" },
  ]);
  return {
    rendered,
    tokenIds: tokenizer.encode(rendered, 14),
    expectedTokenIds: [1, 5, 6, 9, 2, 6, 1, 3, 6, 9, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 6, 1, 4, 6],
  };
}

function projectSelectedLogits(hidden: number[], tokenIds: number[]): Array<{ tokenId: number; logit: number }> {
  const projection = [
    [1, 0.25, 0],
    [0, 1, 0],
    [0.5, -0.25, 0.5],
    [0, 0, 1],
    [-0.25, 0.5, 0.625],
  ];
  return tokenIds.map((tokenId) => ({ tokenId, logit: round(dot(projection[tokenId], hidden)) }));
}

async function runMultiTokenDecodeFixture(): Promise<QwenParityFixtures["multiTokenDecode"]> {
  const weights: UnlockedBrowserTransformerWeights = {
    modelId: "fixture/qwen-multi-token-decode",
    architecture: "qwen3_decoder_control",
    vocabSize: 4,
    hiddenSize: 2,
    headDim: 2,
    tokenEmbedding: [
      [1, 0],
      [0, 1],
      [1, 1],
      [0.5, -0.5],
    ],
    outputProjection: [
      [1, 0],
      [0, 1],
      [1, 1],
      [-1, 0.5],
    ],
    layers: [
      {
        qProj: [
          [1, 0],
          [0, 1],
        ],
        kProj: [
          [1, 0],
          [0, 1],
        ],
        vProj: [
          [0.5, 0.5],
          [1, -1],
        ],
        oProj: [
          [1, 0],
          [0, 1],
        ],
        mlpUpProj: [
          [1, 0],
          [0, 1],
        ],
        mlpDownProj: [
          [1, 0],
          [0, 1],
        ],
      },
    ],
  };
  const policy: SSALayerRoutingPolicy = {
    layerIndex: 0,
    blockSize: 2,
    topKBlocks: 2,
    localWindowBlocks: 0,
    pinnedBlockIds: ["b0"],
    selectedBlockIdsByQueryBlock: {
      0: ["b0", "b1"],
      1: ["b0", "b1"],
      2: ["b1"],
    },
    denseFallback: true,
  };
  const promptTokenIds = [0, 1];
  const requestedTokenCount = 3;
  const generatedTokenIds: number[] = [];
  const perStep: QwenParityFixtures["multiTokenDecode"]["perStep"] = [];
  const backend = new UnlockedBrowserTransformerBackend({ weights, backendPreference: "cpu" });
  await backend.initializeModel(weights.modelId);
  const prefill = await backend.prefill(new Int32Array(promptTokenIds), {
    requestId: "qwen_parity_multi_token_decode",
    layerPolicies: [policy],
  });
  let inputTokenId = promptTokenIds[promptTokenIds.length - 1] ?? 0;
  for (let step = 0; step < requestedTokenCount; step += 1) {
    const decoded = await backend.decode({
      requestId: "qwen_parity_multi_token_decode",
      inputTokenId,
      kvCacheHandle: prefill.kvCacheHandle,
      policy: [policy],
      logitCandidateTokenIds: [0, 1, 2, 3],
    });
    generatedTokenIds.push(decoded.tokenId);
    perStep.push({
      step,
      inputTokenId,
      generatedTokenId: decoded.tokenId,
      logitTokenIds: [0, 1, 2, 3],
    });
    inputTokenId = decoded.tokenId;
  }
  return {
    promptTokenIds,
    seedInputTokenId: promptTokenIds[promptTokenIds.length - 1] ?? 0,
    requestedTokenCount,
    generatedTokenIds,
    expectedGeneratedTokenIds: [0, 2, 0],
    generatedTokenCount: generatedTokenIds.length,
    perStep,
  };
}

function resolveManifestPath(path: string | undefined): string {
  const raw = path ?? process.env.QWEN_PARITY_MANIFEST_PATH ?? process.env.VITE_UNLOCKED_MODEL_MANIFEST_PATH ?? "";
  if (!raw) return "";
  if (raw.startsWith("file://")) return fileURLToPath(raw);
  if (raw.startsWith("/")) return raw;
  return resolve(repoRoot, raw);
}

function pathToFileUrl(path: string): string {
  return pathToFileURL(path.startsWith("/") ? path : resolve(path)).toString();
}

async function installFileFetch(): Promise<void> {
  if ((globalThis as { __qwenParityFileFetchInstalled?: boolean }).__qwenParityFileFetchInstalled) return;
  const originalFetch = globalThis.fetch.bind(globalThis);
  globalThis.fetch = async (input: RequestInfo | URL, init?: RequestInit): Promise<Response> => {
    const url = String(input);
    if (!url.startsWith("file://")) return originalFetch(input, init);
    const bytes = await readFile(fileURLToPath(url));
    return new Response(bytes, {
      status: 200,
      headers: {
        "content-type": url.endsWith(".json") ? "application/json" : "application/octet-stream",
        "content-length": String(bytes.byteLength),
      },
    });
  };
  (globalThis as { __qwenParityFileFetchInstalled?: boolean }).__qwenParityFileFetchInstalled = true;
}

function suite(name: string, metrics: Record<string, number | string | boolean>, gates: QwenParityGate[]): QwenParitySuite {
  return { name, passed: gates.every((gate) => gate.passed), metrics, gates };
}

function gate(name: string, passed: boolean, expected: number | string | boolean, actual: number | string | boolean): QwenParityGate {
  return { name, passed, expected, actual };
}

function maxError(actual: number[], expected: number[]): number {
  return round(Math.max(...actual.map((value, index) => Math.abs(value - (expected[index] ?? Number.NaN)))));
}

function matVec(matrix: number[][], vector: number[]): number[] {
  return matrix.map((row) => dot(row, vector));
}

function dot(left: number[], right: number[]): number {
  return left.reduce((sum, value, index) => sum + value * (right[index] ?? 0), 0);
}

function softmax(values: number[]): number[] {
  const max = Math.max(...values);
  const exps = values.map((value) => Math.exp(value - max));
  const total = exps.reduce((sum, value) => sum + value, 0);
  return exps.map((value) => value / total);
}

function silu(value: number): number {
  return value / (1 + Math.exp(-value));
}

function round(value: number): number {
  return Number(value.toFixed(8));
}

function arraysEqual(left: number[], right: number[]): boolean {
  return left.length === right.length && left.every((value, index) => value === right[index]);
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function buildSummary(result: QwenParityAccuracyResult): string {
  const rows = result.suites
    .map((suite) => `| ${suite.name} | ${suite.passed ? "PASS" : "FAIL"} | ${formatMetrics(suite.metrics)} | ${suite.gates.filter((gate) => !gate.passed).map((gate) => gate.name).join(", ")} |`)
    .join("\n");
  return `# Qwen Parity Accuracy: ${result.passed ? "PASS" : "FAIL"}

- Created: ${result.createdAt}
- Max abs error: ${result.summary.maxAbsError}
- Max abs error threshold: ${result.thresholds.maxAbsError}
- Token parity: ${result.summary.tokenParityPassed}
- Multi-token decode: ${result.summary.multiTokenDecodePassed}
- Retrieval grounded response: ${result.summary.retrievalGroundedPassed}
- Real model parity mode: ${result.summary.realModelParityMode}

| Suite | Status | Metrics | Failed Gates |
| --- | --- | --- | --- |
${rows}
`;
}

function formatMetrics(metrics: Record<string, number | string | boolean>): string {
  return Object.entries(metrics).map(([key, value]) => `${key}=${value}`).join("; ");
}

async function runCli(): Promise<void> {
  const flags = new Set(process.argv.slice(2).filter((arg) => arg.startsWith("--")).map((arg) => arg.slice(2)));
  const parsed = new Map<string, string>();
  for (let index = 2; index < process.argv.length; index += 1) {
    const arg = process.argv[index];
    if (!arg.startsWith("--")) continue;
    const value = process.argv[index + 1];
    if (value && !value.startsWith("--")) {
      parsed.set(arg.slice(2), value);
      index += 1;
    }
  }
  const result = await evaluateQwenParityAccuracy({
    manifestPath: parsed.get("manifest-path"),
    requireRealModelParity: flags.has("require-real-model") || process.env.RELEASE_REQUIRE_QWEN_ACCURACY_REAL_MODEL === "true",
  });
  const paths = await writeQwenParityAccuracyArtifact(result);
  console.log(`Qwen parity accuracy: ${result.passed ? "PASS" : "FAIL"}`);
  console.log(`Max abs error: ${result.summary.maxAbsError}`);
  console.log(`Real model parity: ${result.summary.realModelParityMode}`);
  console.log(`Results: ${paths.resultsPath}`);
  console.log(`Summary: ${paths.summaryPath}`);
  if (!result.passed) process.exitCode = 1;
}

if (process.argv[1] && fileURLToPath(import.meta.url) === resolve(process.argv[1])) {
  await runCli();
}
