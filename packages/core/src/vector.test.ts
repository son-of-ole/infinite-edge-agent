import { describe, expect, it } from "vitest";
import { cosineSimilarity, normalizeVector } from "./vector";

it("computes cosine similarity", () => {
  expect(cosineSimilarity([1, 0], [1, 0])).toBeCloseTo(1);
  expect(cosineSimilarity([1, 0], [0, 1])).toBeCloseTo(0);
});

it("normalizes vectors", () => {
  const vector = normalizeVector([3, 4]);
  expect(vector[0]).toBeCloseTo(0.6);
  expect(vector[1]).toBeCloseTo(0.8);
});
