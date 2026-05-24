import {
  cosineSimilarity,
  packContext,
  type ChatMessage,
  type MemorySearchHit
} from "@infinite-edge-agent/core";
import { mkdir, writeFile } from "node:fs/promises";
import { join } from "node:path";

interface StressConfig {
  vectorCount: number;
  dimension: number;
  topK: number;
  memoryTokenBudget: number;
  recentMessageCount: number;
  seed: number;
  maxSearchMs: number;
}

interface GateResult {
  name: string;
  passed: boolean;
  actual: number | string | boolean;
  expected: number | string | boolean;
}

interface SearchHit {
  index: number;
  score: number;
}

const DEFAULT_CONFIG: StressConfig = {
  vectorCount: 1500,
  dimension: 128,
  topK: 20,
  memoryTokenBudget: 700,
  recentMessageCount: 16,
  seed: 1337,
  maxSearchMs: 250
};

const REQUIRED_ANCHOR_INDEX = 3;
const REQUIRED_ANCHOR_ID = `memory-${REQUIRED_ANCHOR_INDEX.toString().padStart(5, "0")}`;

function optionValue(name: string): string | undefined {
  const prefix = `--${name}=`;
  const inline = process.argv.find((arg) => arg.startsWith(prefix));
  if (inline) return inline.slice(prefix.length);

  const index = process.argv.indexOf(`--${name}`);
  if (index >= 0) {
    const value = process.argv[index + 1];
    if (value === undefined || value.startsWith("--")) {
      throw new Error(`Missing value for --${name}`);
    }
    return value;
  }

  return undefined;
}

function readNumberOption(name: string, envName: string, fallback: number): number {
  const raw = optionValue(name) ?? process.env[envName];
  if (raw === undefined) return fallback;
  const parsed = Number(raw);
  if (!Number.isFinite(parsed) || parsed <= 0) {
    throw new Error(`Invalid numeric option ${name}/${envName}: ${raw}`);
  }
  return Math.floor(parsed);
}

function readConfig(): StressConfig {
  return {
    vectorCount: readNumberOption("vectors", "CORE_STRESS_VECTORS", DEFAULT_CONFIG.vectorCount),
    dimension: readNumberOption("dim", "CORE_STRESS_DIM", DEFAULT_CONFIG.dimension),
    topK: readNumberOption("top-k", "CORE_STRESS_TOP_K", DEFAULT_CONFIG.topK),
    memoryTokenBudget: readNumberOption(
      "memory-token-budget",
      "CORE_STRESS_MEMORY_TOKEN_BUDGET",
      DEFAULT_CONFIG.memoryTokenBudget
    ),
    recentMessageCount: readNumberOption(
      "recent-messages",
      "CORE_STRESS_RECENT_MESSAGES",
      DEFAULT_CONFIG.recentMessageCount
    ),
    seed: readNumberOption("seed", "CORE_STRESS_SEED", DEFAULT_CONFIG.seed),
    maxSearchMs: readNumberOption("max-search-ms", "CORE_STRESS_MAX_SEARCH_MS", DEFAULT_CONFIG.maxSearchMs)
  };
}

function seededNoise(seed: number, index: number): number {
  const value = Math.sin((seed + 1) * (index + 1) * 12.9898) * 43758.5453;
  return value - Math.floor(value);
}

function deterministicVector(dimension: number, seed: number, topic: number, variant: number): number[] {
  return Array.from({ length: dimension }, (_, dimIndex) => {
    const topicSignal = dimIndex % 11 === topic ? 1.75 : 0;
    const lowFrequency = Math.cos((topic + 1) * (dimIndex + 1) * 0.03125) * 0.25;
    const noise = seededNoise(seed + variant, dimIndex) * 0.18;
    return topicSignal + lowFrequency + noise;
  });
}

function buildMemoryCorpus(config: StressConfig): MemorySearchHit[] {
  const createdAt = "2026-01-01T00:00:00.000Z";
  return Array.from({ length: config.vectorCount }, (_, index) => {
    const topic = index % 11;
    const isRequiredAnchor = index === REQUIRED_ANCHOR_INDEX;
    const tokenCount = isRequiredAnchor ? 34 : 22 + (index % 9);

    return {
      id: `memory-${index.toString().padStart(5, "0")}`,
      text: `Core stress memory ${index} for topic ${topic}. ${
        isRequiredAnchor ? "Required anchor: browser-free production stress gate." : "Synthetic durable memory."
      } Deterministic payload for vector/context packing.`,
      embedding: deterministicVector(config.dimension, config.seed, topic, isRequiredAnchor ? 42 : index),
      sessionId: `session-${index % 5}`,
      source: index % 7 === 0 ? "summary" : "chat",
      role: index % 2 === 0 ? "user" : "assistant",
      createdAt,
      updatedAt: createdAt,
      tags: isRequiredAnchor ? ["stress", "required-anchor", "topic-3"] : ["stress", `topic-${topic}`],
      metadata: {
        deterministic: true,
        topic
      },
      tokenCount,
      score: 0
    };
  });
}

function searchMemory(
  corpus: MemorySearchHit[],
  queryEmbedding: number[],
  topK: number
): { hits: MemorySearchHit[]; elapsedMs: number } {
  const started = performance.now();
  const top: SearchHit[] = corpus
    .map((memory, index) => ({
      index,
      score: cosineSimilarity(queryEmbedding, memory.embedding)
    }))
    .sort((a, b) => b.score - a.score)
    .slice(0, topK);
  const elapsedMs = performance.now() - started;

  return {
    hits: top.map(({ index, score }) => {
      const memory = corpus[index];
      if (!memory) {
        throw new Error(`Search result referenced missing memory index ${index}`);
      }
      return {
        ...memory,
        score
      };
    }),
    elapsedMs
  };
}

function buildRecentMessages(config: StressConfig): ChatMessage[] {
  return Array.from({ length: config.recentMessageCount }, (_, index) => ({
    id: `recent-${index}`,
    role: index % 2 === 0 ? "user" : "assistant",
    content: `Recent deterministic turn ${index}. Keep enough context to test packing budgets.`,
    createdAt: "2026-01-01T00:00:00.000Z",
    sessionId: "stress-session"
  }));
}

function gate(name: string, actual: GateResult["actual"], expected: GateResult["expected"], passed: boolean): GateResult {
  return { name, actual, expected, passed };
}

function buildSummary(results: {
  name: string;
  passed: boolean;
  createdAt: string;
  config: StressConfig;
  metrics: Record<string, number | string | boolean>;
  gates: GateResult[];
}): string {
  const gateRows = results.gates
    .map((item) => `| ${item.name} | ${item.passed ? "PASS" : "FAIL"} | ${item.actual} | ${item.expected} |`)
    .join("\n");

  return `# Core Stress Eval

- Created: ${results.createdAt}
- Passed: ${results.passed}
- Vectors: ${results.config.vectorCount}
- Dimension: ${results.config.dimension}
- Search elapsed: ${results.metrics.elapsedSearchMs} ms
- Selected context tokens: ${results.metrics.selectedContextTokenCount}
- Packed prompt tokens: ${results.metrics.packedPromptTokenCount}

## Gates

| Gate | Status | Actual | Expected |
| --- | --- | --- | --- |
${gateRows}
`;
}

const config = readConfig();
if (config.vectorCount <= REQUIRED_ANCHOR_INDEX) {
  throw new Error(`vectors (${config.vectorCount}) must be greater than ${REQUIRED_ANCHOR_INDEX} for required anchor ${REQUIRED_ANCHOR_ID}`);
}
if (config.topK > config.vectorCount) {
  throw new Error(`top-k (${config.topK}) must be <= vector count (${config.vectorCount})`);
}

const createdAt = new Date().toISOString();
const timestamp = createdAt.replace(/[:.]/g, "-");
const artifactRoot = process.env.EVAL_ARTIFACT_DIR ?? ".artifacts/evals";
const suiteDir = join(artifactRoot, "core-stress", timestamp);

const corpus = buildMemoryCorpus(config);
const queryEmbedding = deterministicVector(config.dimension, config.seed, 3, 42);
const search = searchMemory(corpus, queryEmbedding, config.topK);
const recentMessages = buildRecentMessages(config);
const packed = packContext({
  systemPrompt: "You are a production stress evaluator for the local-first core runtime.",
  retrievedMemory: search.hits,
  recentMessages,
  userMessage: "Run the deterministic stress gate and preserve the required anchor.",
  config: {
    maxPromptTokens: 4000,
    maxRecentConversationTokens: 500,
    maxRetrievedMemoryTokens: config.memoryTokenBudget
  }
});

const selectedMemory = search.hits.filter((hit) => packed.includedMemoryIds.includes(hit.id));
const selectedContextTokenCount = selectedMemory.reduce((sum, hit) => sum + hit.tokenCount, 0);
const requiredAnchorIncluded = packed.includedMemoryIds.includes(REQUIRED_ANCHOR_ID);
const provenanceComplete = search.hits.every((hit) => hit.id && hit.sessionId && hit.source && hit.createdAt);
const dimensionsValid = corpus.every((memory) => memory.embedding.length === config.dimension);

const metrics = {
  memoryVectorCount: corpus.length,
  dimension: config.dimension,
  elapsedSearchMs: Number(search.elapsedMs.toFixed(3)),
  selectedContextTokenCount,
  packedPromptTokenCount: packed.estimatedTokens,
  retrievedHitCount: search.hits.length,
  includedMemoryCount: packed.includedMemoryIds.length,
  requiredAnchorId: REQUIRED_ANCHOR_ID,
  requiredAnchorIncluded,
  provenanceComplete,
  topScore: Number((search.hits[0]?.score ?? 0).toFixed(6))
};

const gates = [
  gate("memory vector count", metrics.memoryVectorCount, config.vectorCount, metrics.memoryVectorCount === config.vectorCount),
  gate("vector dimension", metrics.dimension, config.dimension, dimensionsValid),
  gate("search latency", metrics.elapsedSearchMs, `<= ${config.maxSearchMs} ms`, search.elapsedMs <= config.maxSearchMs),
  gate(
    "selected context token budget",
    selectedContextTokenCount,
    `<= ${config.memoryTokenBudget}`,
    selectedContextTokenCount <= config.memoryTokenBudget
  ),
  gate("required anchor included", requiredAnchorIncluded ? REQUIRED_ANCHOR_ID : "missing", REQUIRED_ANCHOR_ID, requiredAnchorIncluded),
  gate("retrieval provenance complete", provenanceComplete, true, provenanceComplete),
  gate("packed prompt token budget", packed.estimatedTokens, "<= 4000", packed.estimatedTokens <= 4000)
];

const artifact = {
  name: "core-stress",
  createdAt,
  passed: gates.every((item) => item.passed),
  config,
  metrics,
  gates,
  topHitIds: search.hits.slice(0, 10).map((hit) => hit.id),
  includedMemoryIds: packed.includedMemoryIds
};

const traceLines = [
  { event: "config", createdAt, config },
  {
    event: "search",
    elapsedMs: metrics.elapsedSearchMs,
    topHitIds: artifact.topHitIds,
    topScore: metrics.topScore
  },
  {
    event: "context-packed",
    selectedContextTokenCount,
    packedPromptTokenCount: packed.estimatedTokens,
    includedMemoryIds: packed.includedMemoryIds
  },
  { event: "gates", passed: artifact.passed, gates }
];

await mkdir(suiteDir, { recursive: true });
await writeFile(join(suiteDir, "results.json"), `${JSON.stringify(artifact, null, 2)}\n`);
await writeFile(join(suiteDir, "trace.jsonl"), `${traceLines.map((line) => JSON.stringify(line)).join("\n")}\n`);
await writeFile(join(suiteDir, "summary.md"), buildSummary(artifact));
await writeFile(join(artifactRoot, "core-stress-latest.json"), `${JSON.stringify(artifact, null, 2)}\n`);

console.log(`Core stress eval: ${artifact.passed ? "PASS" : "FAIL"}`);
console.log(`Results: ${join(suiteDir, "results.json")}`);
console.log(`Summary: ${join(suiteDir, "summary.md")}`);

if (!artifact.passed) {
  throw new Error("Core stress artifact failed acceptance checks.");
}
