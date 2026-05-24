import { mkdtemp, readFile, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { resolve } from "node:path";
import { describe, expect, it } from "vitest";
import {
  buildQwenParityAccuracyArtifact,
  evaluateQwenParityAccuracy,
  writeQwenParityAccuracyArtifact,
} from "./qwenParityAccuracy";

describe("Qwen parity accuracy eval", () => {
  it("passes deterministic toy fixtures for Qwen math, tokenizer, logits, and retrieval checks", async () => {
    const result = await evaluateQwenParityAccuracy();

    expect(result.passed).toBe(true);
    expect(result.summary).toMatchObject({
      maxAbsError: expect.any(Number),
      failedGateCount: 0,
      tokenParityPassed: true,
      retrievalGroundedPassed: true,
      realModelParityMode: "skipped",
    });
    expect(result.summary.maxAbsError).toBeLessThanOrEqual(result.thresholds.maxAbsError);
    expect(result.suites.map((suite) => suite.name)).toEqual([
      "rmsnorm",
      "rope",
      "gqa",
      "gated-mlp",
      "tokenizer-chat-template",
      "logit-projection",
      "multi-token-decode",
      "retrieval-grounded-response",
      "real-model-parity",
    ]);
    expect(result.suites.every((suite) => suite.passed)).toBe(true);
  });

  it("writes release artifacts with threshold details and selected logits", async () => {
    const artifactDir = await mkdtemp(resolve(tmpdir(), "qwen-parity-artifacts-"));
    const result = await evaluateQwenParityAccuracy();
    const paths = await writeQwenParityAccuracyArtifact(result, { artifactDir });
    const latest = JSON.parse(await readFile(paths.latestPath, "utf8")) as ReturnType<typeof buildQwenParityAccuracyArtifact>;

    expect(latest.name).toBe("qwen-parity-accuracy");
    expect(latest.passed).toBe(true);
    expect(latest.thresholds).toMatchObject({
      maxAbsError: 0.000001,
      deterministicTokenParity: true,
      retrievalGroundedResponse: true,
    });
    expect(latest.fixtures.logitProjection.selectedLogits).toEqual([
      { tokenId: 0, logit: 0.125 },
      { tokenId: 2, logit: 0.75 },
      { tokenId: 4, logit: 0.3125 },
    ]);
    expect(latest.fixtures.tokenizerChatTemplate.rendered).toBe(
      "<|im_start|>system\nhi<|im_end|>\n<|im_start|>user\nhi<|im_end| ></think ><|endoftext| ><|im_end|>\n<|im_start|>assistant\n",
    );
    expect(latest.fixtures.multiTokenDecode).toMatchObject({
      promptTokenIds: [0, 1],
      generatedTokenIds: [0, 2, 0],
      expectedGeneratedTokenIds: [0, 2, 0],
      requestedTokenCount: 3,
      generatedTokenCount: 3,
    });
    expect(latest.suites.find((suite) => suite.name === "multi-token-decode")).toMatchObject({
      passed: true,
      metrics: {
        generatedTokenCount: 3,
        expectedGeneratedTokenCount: 3,
        generatedTokenParity: true,
      },
    });
    expect(latest.suites.find((suite) => suite.name === "real-model-parity")?.metrics.mode).toBe("skipped");
  });

  it("fails real model parity only when the strict gate is requested without an installed manifest", async () => {
    await expect(evaluateQwenParityAccuracy({ requireRealModelParity: true, manifestPath: "" })).rejects.toThrow(
      /requires VITE_UNLOCKED_MODEL_MANIFEST_PATH/i,
    );
  });

  it("executes installed real-model parity through the production unlocked client path", async () => {
    const artifactDir = await mkdtemp(resolve(tmpdir(), "qwen-real-model-"));
    const manifestPath = resolve(artifactDir, "manifest.json");
    await writeFile(manifestPath, `${JSON.stringify(makeInlineQwenManifest(), null, 2)}\n`);

    const result = await evaluateQwenParityAccuracy({ manifestPath, requireRealModelParity: true });

    expect(result.passed).toBe(true);
    expect(result.summary.realModelParityMode).toBe("installed");
    expect(result.realModelParity).toMatchObject({
      mode: "installed",
      manifestPath,
      prompt: "alpha beta",
      generatedTokenIds: expect.arrayContaining([expect.any(Number)]),
      backendProof: expect.objectContaining({
        tensorControl: true,
        logitProjectionBackend: "cpu_reference",
        logitProjectionPurpose: "full_vocab_topk_logit_projection",
        logitProjectionSelectedRows: 8,
        logitProjectionFullRows: 151936,
      }),
    });
    expect(result.suites.find((suite) => suite.name === "real-model-parity")).toMatchObject({
      passed: true,
      metrics: expect.objectContaining({
        mode: "installed",
        modelId: "Qwen/Qwen3-0.6B",
        vocabSize: 151936,
        generatedTokenCount: expect.any(Number),
        logitProjectionPurpose: "full_vocab_topk_logit_projection",
        logitProjectionSelectedRows: 8,
        logitProjectionFullRows: 151936,
      }),
    });
  });
});

function makeInlineQwenManifest(): Record<string, unknown> {
  const vocabSize = 151936;
  const tokens = ["<unk>", "<|im_start|>", "<|im_end|>", "user", "assistant", "system", "Ċ", "a", "b", "alpha", "beta", "l", "p"];
  while (tokens.length < vocabSize) tokens.push(`tok_${tokens.length}`);
  return {
    schemaVersion: 1,
    modelId: "Qwen/Qwen3-0.6B",
    architecture: "qwen3_decoder_control",
    vocabSize,
    hiddenSize: 2,
    headDim: 2,
    numAttentionHeads: 1,
    numKeyValueHeads: 1,
    maxPositionEmbeddings: 128,
    ropeTheta: 10000,
    tieWordEmbeddings: false,
    tokenEmbedding: Array.from({ length: vocabSize }, (_value, index) => [index % 3 === 0 ? 1 : 0, index % 3 === 1 ? 1 : 0]),
    outputProjection: Array.from({ length: vocabSize }, (_value, index) => [index === 2 ? 2 : index % 2, index === 4 ? 2 : (index + 1) % 2]),
    conversion: { projectionMode: "full-qwen-gqa-rope" },
    tokenizer: {
      kind: "qwen-bpe",
      tokens,
      merges: [["a", "l"], ["al", "p"], ["alp", "h"], ["alph", "a"], ["b", "e"], ["be", "t"], ["bet", "a"]],
      unknownTokenId: 0,
      specialTokens: ["<|im_start|>", "<|im_end|>"],
      chatTemplate: "{{#messages}}<|im_start|>{{role}}\n{{content}}<|im_end|>\n{{/messages}}<|im_start|>assistant\n",
    },
    layers: [
      {
        qProj: [[1, 0], [0, 1]],
        kProj: [[1, 0], [0, 1]],
        vProj: [[0.5, 0.5], [1, -1]],
        oProj: [[1, 0], [0, 1]],
        mlpUpProj: [[1, 0], [0, 1]],
        mlpDownProj: [[1, 0], [0, 1]],
      },
    ],
  };
}
