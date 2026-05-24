import { cosineSimilarity } from "@infinite-edge-agent/core";

function randomVector(dim: number): number[] {
  return Array.from({ length: dim }, () => Math.random() - 0.5);
}

const dim = Number(process.argv[2] ?? 384);
const count = Number(process.argv[3] ?? 10000);
const query = randomVector(dim);
const vectors = Array.from({ length: count }, () => randomVector(dim));

const start = performance.now();
const top = vectors
  .map((vector, index) => ({ index, score: cosineSimilarity(query, vector) }))
  .sort((a, b) => b.score - a.score)
  .slice(0, 10);
const elapsed = performance.now() - start;

console.log({ dim, count, elapsedMs: elapsed.toFixed(2), top });
