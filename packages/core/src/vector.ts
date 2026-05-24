export function dot(a: readonly number[], b: readonly number[]): number {
  const n = Math.min(a.length, b.length);
  let sum = 0;
  for (let i = 0; i < n; i += 1) {
    sum += (a[i] ?? 0) * (b[i] ?? 0);
  }
  return sum;
}

export function magnitude(a: readonly number[]): number {
  return Math.sqrt(dot(a, a));
}

export function cosineSimilarity(a: readonly number[], b: readonly number[]): number {
  const denom = magnitude(a) * magnitude(b);
  if (denom === 0) return 0;
  return dot(a, b) / denom;
}

export function normalizeVector(vector: readonly number[]): number[] {
  const mag = magnitude(vector);
  if (mag === 0) return Array.from(vector);
  return vector.map((value) => value / mag);
}

export function assertSameDimension(a: readonly number[], b: readonly number[]): void {
  if (a.length !== b.length) {
    throw new Error(`Vector dimension mismatch: ${a.length} !== ${b.length}`);
  }
}
