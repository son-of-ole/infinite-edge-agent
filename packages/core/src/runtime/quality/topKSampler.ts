export interface CompactTopKCandidate {
  tokenId: number;
  score: number;
}

export interface CompactTopKSamplingOptions {
  temperature?: number;
  topP?: number;
  repetitionPenalty?: number;
  recentTokenIds?: readonly number[];
  suppressedTokenIds?: readonly number[];
  seed?: number;
}

export interface CompactTopKSamplingResult {
  tokenId: number;
  score: number;
  selectedRank: number;
  strategy: "greedy" | "compact_topk_sample";
  topK: number;
  effectiveCandidateCount: number;
}

const NEG_INF = Number.NEGATIVE_INFINITY;

export function sampleFromCompactTopK(
  candidates: readonly CompactTopKCandidate[],
  options: CompactTopKSamplingOptions = {},
): CompactTopKSamplingResult {
  if (candidates.length === 0) throw new Error("Cannot sample from an empty compact top-k set.");

  const suppressed = new Set(options.suppressedTokenIds ?? []);
  const recent = new Set(options.recentTokenIds ?? []);
  const repetitionPenalty = normalizePositive(options.repetitionPenalty, 1);
  const temperature = normalizeNonNegative(options.temperature, 1);
  const topP = clamp01(options.topP ?? 1);

  const adjusted = candidates
    .map((candidate, index) => {
      let score = finiteOrNegInf(candidate.score);
      if (suppressed.has(candidate.tokenId)) score = NEG_INF;
      if (recent.has(candidate.tokenId) && repetitionPenalty > 1 && Number.isFinite(score)) {
        score = score > 0 ? score / repetitionPenalty : score * repetitionPenalty;
      }
      return { ...candidate, score, originalRank: index };
    })
    .filter((candidate) => Number.isFinite(candidate.score))
    .sort((left, right) => right.score - left.score || left.tokenId - right.tokenId);

  if (adjusted.length === 0) {
    const fallback = candidates[0];
    if (!fallback) throw new Error("No compact top-k fallback candidate was available.");
    return {
      tokenId: fallback.tokenId,
      score: fallback.score,
      selectedRank: 0,
      strategy: "greedy",
      topK: candidates.length,
      effectiveCandidateCount: 0,
    };
  }

  if (temperature <= 1e-6 || adjusted.length === 1) {
    const best = adjusted[0] as typeof adjusted[number];
    return {
      tokenId: best.tokenId,
      score: best.score,
      selectedRank: best.originalRank,
      strategy: "greedy",
      topK: candidates.length,
      effectiveCandidateCount: adjusted.length,
    };
  }

  const probabilities = softmax(adjusted.map((candidate) => candidate.score / temperature));
  const nucleus = buildNucleus(adjusted, probabilities, topP);
  const random = seededUnitRandom(options.seed ?? 0x9e3779b9);
  let cumulative = 0;
  for (const item of nucleus) {
    cumulative += item.probability;
    if (random <= cumulative) {
      return {
        tokenId: item.candidate.tokenId,
        score: item.candidate.score,
        selectedRank: item.candidate.originalRank,
        strategy: "compact_topk_sample",
        topK: candidates.length,
        effectiveCandidateCount: nucleus.length,
      };
    }
  }

  const last = nucleus[nucleus.length - 1]?.candidate ?? adjusted[0] as typeof adjusted[number];
  return {
    tokenId: last.tokenId,
    score: last.score,
    selectedRank: last.originalRank,
    strategy: "compact_topk_sample",
    topK: candidates.length,
    effectiveCandidateCount: nucleus.length,
  };
}

function softmax(scores: readonly number[]): number[] {
  const maxScore = Math.max(...scores);
  const exps = scores.map((score) => Math.exp(score - maxScore));
  const total = exps.reduce((sum, value) => sum + value, 0);
  return exps.map((value) => value / total);
}

function buildNucleus<T extends { score: number }>(
  candidates: readonly T[],
  probabilities: readonly number[],
  topP: number,
): Array<{ candidate: T; probability: number }> {
  const paired = candidates.map((candidate, index) => ({ candidate, probability: probabilities[index] ?? 0 }));
  if (topP >= 1) return renormalize(paired);
  const selected: typeof paired = [];
  let cumulative = 0;
  for (const item of paired) {
    selected.push(item);
    cumulative += item.probability;
    if (cumulative >= topP) break;
  }
  return renormalize(selected.length > 0 ? selected : paired.slice(0, 1));
}

function renormalize<T>(items: Array<{ candidate: T; probability: number }>): Array<{ candidate: T; probability: number }> {
  const total = items.reduce((sum, item) => sum + item.probability, 0);
  if (total <= 0) return items.map((item, index) => ({ ...item, probability: index === 0 ? 1 : 0 }));
  return items.map((item) => ({ ...item, probability: item.probability / total }));
}

function seededUnitRandom(seed: number): number {
  let state = seed >>> 0;
  state ^= state << 13;
  state ^= state >>> 17;
  state ^= state << 5;
  return ((state >>> 0) % 1_000_000) / 1_000_000;
}

function normalizePositive(value: number | undefined, fallback: number): number {
  return typeof value === "number" && Number.isFinite(value) && value > 0 ? value : fallback;
}

function normalizeNonNegative(value: number | undefined, fallback: number): number {
  return typeof value === "number" && Number.isFinite(value) && value >= 0 ? value : fallback;
}

function clamp01(value: number): number {
  if (!Number.isFinite(value)) return 1;
  return Math.min(1, Math.max(0, value));
}

function finiteOrNegInf(value: number): number {
  return Number.isFinite(value) ? value : NEG_INF;
}
