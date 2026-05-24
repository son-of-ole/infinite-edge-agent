import { describe, expect, it } from "vitest";
import {
  bucketHeadDim,
  bucketPromptLength,
  bucketSelectedBlockCount,
  bucketTileRows,
  createStaticPipelineCacheKey,
  planPrefillChunks,
  resolveStaticShapeBucket,
} from "./shapeBuckets";

describe("SSA WebGPU shape buckets", () => {
  it("maps nearby prompt lengths to the same prompt bucket and pipeline key", () => {
    const first = resolveStaticShapeBucket({
      operation: "prefill_attention",
      promptLengthTokens: 1001,
      selectedBlockCount: 17,
      headDim: 127,
      tileRows: 3000,
    });
    const second = resolveStaticShapeBucket({
      operation: "prefill_attention",
      promptLengthTokens: 1023,
      selectedBlockCount: 18,
      headDim: 128,
      tileRows: 4095,
    });

    expect(first.promptLengthBucket).toBe(1024);
    expect(second.promptLengthBucket).toBe(1024);
    expect(first.pipelineCacheKey).toBe(second.pipelineCacheKey);
  });

  it("buckets selected-block counts, head dimensions, and tile rows into stable keys", () => {
    expect(bucketSelectedBlockCount(17)).toBe(32);
    expect(bucketSelectedBlockCount(31)).toBe(32);
    expect(bucketHeadDim(129)).toBe(160);
    expect(bucketHeadDim(159)).toBe(160);
    expect(bucketTileRows(3000)).toBe(4096);
    expect(bucketTileRows(4095)).toBe(4096);

    expect(createStaticPipelineCacheKey({
      operation: "decode_sparse_attention",
      promptLengthTokens: 4090,
      selectedBlockCount: 31,
      headDim: 159,
      tileRows: 4095,
      precision: "f16",
    })).toBe(createStaticPipelineCacheKey({
      operation: "decode_sparse_attention",
      promptLengthTokens: 4096,
      selectedBlockCount: 32,
      headDim: 160,
      tileRows: 4096,
      precision: "f16",
    }));
  });

  it("splits long prompts into bounded prefill chunks", () => {
    const plan = planPrefillChunks(5000, {
      maxChunkTokens: 1024,
      maxDispatchEstimatedMs: 8,
    });

    expect(plan.prefillChunkCount).toBeGreaterThan(1);
    expect(plan.prefillChunkSize).toBeLessThanOrEqual(1024);
    expect(plan.chunks.every((chunk) => chunk.tokenCount <= 1024)).toBe(true);
    expect(plan.chunks.at(0)).toMatchObject({
      index: 0,
      tokenStart: 0,
    });
    expect(plan.chunks.at(-1)?.tokenEnd).toBe(5000);
  });

  it("keeps estimated dispatch time within the configured budget", () => {
    const defaultPlan = planPrefillChunks(4096);
    expect(defaultPlan.maxDispatchEstimatedMs).toBeLessThanOrEqual(defaultPlan.dispatchBudgetMs);

    const tightPlan = planPrefillChunks(2048, {
      selectedBlockCount: 64,
      headDim: 256,
      maxChunkTokens: 1024,
      maxDispatchEstimatedMs: 4,
    });
    expect(tightPlan.maxDispatchEstimatedMs).toBeLessThanOrEqual(4);
    expect(tightPlan.chunks.every((chunk) => chunk.estimatedDispatchMs <= 4)).toBe(true);
  });

  it("uses a wider watchdog-safe default chunk for strict long-prompt WebGPU prefill", () => {
    const plan = planPrefillChunks(2048, {
      selectedBlockCount: 512,
      headDim: 128,
      maxChunkTokens: 1024,
    });

    expect(plan.prefillChunkSize).toBeGreaterThanOrEqual(1);
    expect(plan.prefillChunkCount).toBeGreaterThan(16);
    expect(plan.maxDispatchEstimatedMs).toBeLessThanOrEqual(plan.dispatchBudgetMs);
    expect(plan.shapeBucket).toContain("prompt<=16");
  });

  it("estimates block-16 full-context routing by selected tokens instead of undercounting selected blocks", () => {
    const block16Plan = planPrefillChunks(6000, {
      selectedBlockCount: 375,
      headDim: 128,
      blockSize: 16,
      maxChunkTokens: 1024,
    });
    const block2Plan = planPrefillChunks(6000, {
      selectedBlockCount: 3000,
      headDim: 128,
      blockSize: 2,
      maxChunkTokens: 1024,
    });

    expect(block16Plan.selectedBlockCountBucket).toBe(512);
    expect(block2Plan.selectedBlockCountBucket).toBe(4096);
    expect(block16Plan.prefillChunkCount).toBe(block2Plan.prefillChunkCount);
    expect(block16Plan.prefillChunkSize).toBe(block2Plan.prefillChunkSize);
    expect(block16Plan.maxDispatchEstimatedMs).toBeLessThanOrEqual(block16Plan.dispatchBudgetMs);
    expect(block2Plan.maxDispatchEstimatedMs).toBeLessThanOrEqual(block2Plan.dispatchBudgetMs);
  });

  it("normalizes invalid or zero inputs safely", () => {
    expect(bucketPromptLength(0)).toBe(1);
    expect(bucketPromptLength(Number.NaN)).toBe(1);
    expect(bucketSelectedBlockCount(-8)).toBe(1);
    expect(bucketHeadDim(Number.POSITIVE_INFINITY)).toBe(1);
    expect(bucketTileRows(0)).toBe(1);

    const emptyPlan = planPrefillChunks(0, {
      maxChunkTokens: 0,
      maxDispatchEstimatedMs: 0,
    });
    expect(emptyPlan.tokenCount).toBe(0);
    expect(emptyPlan.prefillChunkCount).toBe(0);
    expect(emptyPlan.prefillChunkSize).toBe(0);
    expect(emptyPlan.chunks).toEqual([]);
    expect(emptyPlan.maxDispatchEstimatedMs).toBe(0);
  });
});
