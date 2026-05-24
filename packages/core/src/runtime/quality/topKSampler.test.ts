import { describe, expect, it } from "vitest";
import { sampleFromCompactTopK } from "./topKSampler";

describe("sampleFromCompactTopK", () => {
  it("uses greedy selection when temperature is zero", () => {
    const result = sampleFromCompactTopK([
      { tokenId: 10, score: 1 },
      { tokenId: 11, score: 3 },
      { tokenId: 12, score: 2 },
    ], {
      temperature: 0,
      topP: 0.9,
      seed: 123,
    });

    expect(result).toMatchObject({
      tokenId: 11,
      selectedRank: 1,
      strategy: "greedy",
      effectiveCandidateCount: 3,
    });
  });

  it("applies repetition penalty before sampling compact candidates", () => {
    const result = sampleFromCompactTopK([
      { tokenId: 20, score: 10 },
      { tokenId: 21, score: 9.7 },
      { tokenId: 22, score: 5 },
    ], {
      temperature: 0,
      repetitionPenalty: 2,
      recentTokenIds: [20],
    });

    expect(result.tokenId).toBe(21);
  });

  it("samples reproducibly from a top-p filtered compact set", () => {
    const first = sampleFromCompactTopK([
      { tokenId: 30, score: 4 },
      { tokenId: 31, score: 3.5 },
      { tokenId: 32, score: 2 },
      { tokenId: 33, score: 0 },
    ], {
      temperature: 0.7,
      topP: 0.9,
      seed: 42,
    });
    const second = sampleFromCompactTopK([
      { tokenId: 30, score: 4 },
      { tokenId: 31, score: 3.5 },
      { tokenId: 32, score: 2 },
      { tokenId: 33, score: 0 },
    ], {
      temperature: 0.7,
      topP: 0.9,
      seed: 42,
    });

    expect(second).toEqual(first);
    expect(first.strategy).toBe("compact_topk_sample");
    expect(first.effectiveCandidateCount).toBeLessThan(4);
  });
});
