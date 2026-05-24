import { approximateTokenCount } from "./tokenBudget";
import type { ChatRole, MemoryChunk, MemorySource } from "./types";

export interface ChunkTextOptions {
  chunkTokens: number;
  overlapTokens: number;
  minChunkTokens?: number;
}

export interface TextChunk {
  id: string;
  text: string;
  tokenCount: number;
  ordinal: number;
}

function makeId(prefix = "chunk"): string {
  if (typeof crypto !== "undefined" && "randomUUID" in crypto) {
    return `${prefix}_${crypto.randomUUID()}`;
  }
  return `${prefix}_${Date.now()}_${Math.random().toString(16).slice(2)}`;
}

export function chunkText(text: string, options: ChunkTextOptions): TextChunk[] {
  const clean = text.replace(/\r\n/g, "\n").trim();
  if (!clean) return [];

  const chunkChars = Math.max(1, options.chunkTokens * 4);
  const overlapChars = Math.max(0, Math.min(chunkChars - 1, options.overlapTokens * 4));
  const minTokens = options.minChunkTokens ?? 20;

  const chunks: TextChunk[] = [];
  let start = 0;
  let ordinal = 0;

  while (start < clean.length) {
    let end = Math.min(clean.length, start + chunkChars);

    if (end < clean.length) {
      const boundary = Math.max(
        clean.lastIndexOf("\n\n", end),
        clean.lastIndexOf(". ", end),
        clean.lastIndexOf(" ", end)
      );
      if (boundary > start + chunkChars * 0.55) {
        end = boundary + 1;
      }
    }

    const chunk = clean.slice(start, end).trim();
    const tokenCount = approximateTokenCount(chunk);
    if (tokenCount >= minTokens || chunks.length === 0) {
      chunks.push({ id: makeId(), text: chunk, tokenCount, ordinal });
      ordinal += 1;
    }

    if (end >= clean.length) break;
    start = Math.max(end - overlapChars, start + 1);
  }

  return chunks;
}

export interface MakeMemoryChunksInput {
  text: string;
  embeddings: number[][];
  sessionId: string;
  source: MemorySource;
  role?: ChatRole;
  tags?: string[];
  metadata?: Record<string, unknown>;
  chunkOptions: ChunkTextOptions;
}

export function makeMemoryChunks(input: MakeMemoryChunksInput): MemoryChunk[] {
  const chunks = chunkText(input.text, input.chunkOptions);
  if (chunks.length !== input.embeddings.length) {
    throw new Error(`Expected ${chunks.length} embeddings but received ${input.embeddings.length}.`);
  }
  const now = new Date().toISOString();
  return chunks.map((chunk, index) => ({
    id: chunk.id,
    text: chunk.text,
    embedding: input.embeddings[index] ?? [],
    sessionId: input.sessionId,
    source: input.source,
    ...(input.role ? { role: input.role } : {}),
    createdAt: now,
    updatedAt: now,
    tags: input.tags ?? [],
    metadata: {
      ...input.metadata,
      ordinal: chunk.ordinal
    },
    tokenCount: chunk.tokenCount
  }));
}
