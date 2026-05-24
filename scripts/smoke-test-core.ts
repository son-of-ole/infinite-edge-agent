import { chunkText, cosineSimilarity, packContext } from "@infinite-edge-agent/core";
import { mkdir, writeFile } from "node:fs/promises";
import { join } from "node:path";

const chunks = chunkText("This is a small smoke test. ".repeat(80), {
  chunkTokens: 40,
  overlapTokens: 8,
  minChunkTokens: 1
});

console.log(`Chunk count: ${chunks.length}`);
const cosine = cosineSimilarity([1, 0], [1, 0]);
console.log(`Cosine: ${cosine.toFixed(2)}`);

const packed = packContext({
  systemPrompt: "You are a test agent.",
  retrievedMemory: [],
  recentMessages: [],
  userMessage: "Hello",
  config: {
    maxPromptTokens: 1000,
    maxRecentConversationTokens: 100,
    maxRetrievedMemoryTokens: 100
  }
});

console.log(`Packed messages: ${packed.messages.length}`);

const artifactDir = process.env.EVAL_ARTIFACT_DIR ?? ".artifacts/evals";
await mkdir(artifactDir, { recursive: true });
const artifact = {
  name: "core-smoke",
  createdAt: new Date().toISOString(),
  passed: chunks.length > 0 && cosine === 1 && packed.messages.length === 2,
  metrics: {
    chunkCount: chunks.length,
    cosine,
    packedMessages: packed.messages.length,
    estimatedTokens: packed.estimatedTokens
  }
};
const artifactPath = join(artifactDir, "core-smoke-latest.json");
await writeFile(artifactPath, `${JSON.stringify(artifact, null, 2)}\n`);
console.log(`Eval artifact: ${artifactPath}`);

if (!artifact.passed) {
  throw new Error("Core smoke artifact failed acceptance checks.");
}
