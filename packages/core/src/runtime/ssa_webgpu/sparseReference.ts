import { dot, softmax, validateAttentionShapes, weightedSum, type Matrix } from "./denseReference";

export interface SparseReferenceInput {
  q: Matrix;
  k: Matrix;
  v: Matrix;
  selectedKeyIndexesByQuery: number[][];
  causal?: boolean;
  scale?: number;
}

export function sparseReferenceAttention(input: SparseReferenceInput): Matrix {
  validateAttentionShapes(input.q, input.k, input.v);
  const scale = input.scale ?? 1 / Math.sqrt(input.q[0]?.length ?? 1);

  return input.q.map((query, qi) => {
    const selected = input.selectedKeyIndexesByQuery[qi] ?? [];
    const legal = selected.filter((ki) => ki >= 0 && ki < input.k.length && (input.causal === false || ki <= qi));
    if (legal.length === 0) return new Array(query.length).fill(0);
    const scores = legal.map((ki) => dot(query, input.k[ki] ?? []) * scale);
    const weights = softmax(scores);
    const values = legal.map((ki) => input.v[ki] ?? []);
    return weightedSum(weights, values);
  });
}

export function cosineSimilarity(a: number[], b: number[]): number {
  if (a.length !== b.length) throw new Error(`cosine shape mismatch: ${a.length} !== ${b.length}`);
  let dotSum = 0;
  let aNorm = 0;
  let bNorm = 0;
  for (let i = 0; i < a.length; i += 1) {
    const av = a[i] ?? 0;
    const bv = b[i] ?? 0;
    dotSum += av * bv;
    aNorm += av * av;
    bNorm += bv * bv;
  }
  const denom = Math.sqrt(aNorm) * Math.sqrt(bNorm);
  return denom === 0 ? 0 : dotSum / denom;
}
