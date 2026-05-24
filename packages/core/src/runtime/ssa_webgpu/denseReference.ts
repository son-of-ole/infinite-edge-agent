export type Vector = number[];
export type Matrix = Vector[];

export interface AttentionOptions {
  causal?: boolean;
  scale?: number;
}

export function denseReferenceAttention(q: Matrix, k: Matrix, v: Matrix, options: AttentionOptions = {}): Matrix {
  validateAttentionShapes(q, k, v);
  const scale = options.scale ?? 1 / Math.sqrt(q[0]?.length ?? 1);
  return q.map((query, qi) => {
    const scores = k.map((key, ki) => {
      if (options.causal !== false && ki > qi) return Number.NEGATIVE_INFINITY;
      return dot(query, key) * scale;
    });
    const weights = softmax(scores);
    return weightedSum(weights, v);
  });
}

export function dot(a: Vector, b: Vector): number {
  if (a.length !== b.length) throw new Error(`dot shape mismatch: ${a.length} !== ${b.length}`);
  let total = 0;
  for (let i = 0; i < a.length; i += 1) {
    total += (a[i] ?? 0) * (b[i] ?? 0);
  }
  return total;
}

export function softmax(values: Vector): Vector {
  const finite = values.filter(Number.isFinite);
  const max = finite.length > 0 ? Math.max(...finite) : 0;
  const exps = values.map((value) => (Number.isFinite(value) ? Math.exp(value - max) : 0));
  const denom = exps.reduce((sum, value) => sum + value, 0);
  return denom === 0 ? exps.map(() => 0) : exps.map((value) => value / denom);
}

export function weightedSum(weights: Vector, values: Matrix): Vector {
  const dim = values[0]?.length ?? 0;
  const out = new Array<number>(dim).fill(0);
  for (let i = 0; i < values.length; i += 1) {
    const weight = weights[i] ?? 0;
    const row = values[i] ?? [];
    for (let d = 0; d < dim; d += 1) {
      out[d] = (out[d] ?? 0) + weight * (row[d] ?? 0);
    }
  }
  return out;
}

export function validateAttentionShapes(q: Matrix, k: Matrix, v: Matrix): void {
  if (k.length !== v.length) throw new Error(`K/V length mismatch: ${k.length} !== ${v.length}`);
  const dim = q[0]?.length;
  if (dim === undefined) throw new Error("Q must not be empty");
  for (const [name, matrix] of [["Q", q], ["K", k], ["V", v]] as const) {
    for (const row of matrix) {
      if (row.length !== dim) throw new Error(`${name} dim mismatch: expected ${dim}, got ${row.length}`);
    }
  }
}
