import type { MemoryChunk } from "./types";

export const MEMORY_SCHEMA_VERSION = 1;

export type StoredMemoryChunk = MemoryChunk & {
  schemaVersion: typeof MEMORY_SCHEMA_VERSION;
};

export function toStoredMemoryChunk(chunk: MemoryChunk): StoredMemoryChunk {
  return {
    ...chunk,
    schemaVersion: MEMORY_SCHEMA_VERSION
  };
}

export function fromStoredMemoryChunk(chunk: StoredMemoryChunk): MemoryChunk {
  const { schemaVersion: _schemaVersion, ...rest } = chunk;
  return rest;
}
