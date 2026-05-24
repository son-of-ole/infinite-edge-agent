import type { ContextPackTraceRecord, MemoryChunk, RuntimeTrace } from "../types";

export interface StartupContextSnapshotInput {
  tenantId: string;
  cellId: string;
  sessionId?: string;
  memoryChunks: MemoryChunk[];
  runtimeTraces?: RuntimeTrace[];
  contextPackTraces?: ContextPackTraceRecord[];
  now?: Date;
}

export interface StartupContextSnapshot {
  tenantId: string;
  cellId: string;
  sessionId?: string;
  rebuiltAt: string;
  status: "empty" | "rebuilt";
  memoryCount: number;
  runtimeTraceCount: number;
  contextPackTraceCount: number;
  pinnedRawMemoryIds: string[];
  identityPinIds: string[];
  representativeIds: string[];
  lastRuntimeTraceId?: string;
  lastContextPackTraceId?: string;
  contextDigest: string;
}

export function rebuildStartupContextSnapshot(input: StartupContextSnapshotInput): StartupContextSnapshot {
  const memory = [...input.memoryChunks].sort((a, b) => b.createdAt.localeCompare(a.createdAt));
  const runtimeTraces = [...(input.runtimeTraces ?? [])].sort((a, b) => b.createdAt.localeCompare(a.createdAt));
  const contextPackTraces = [...(input.contextPackTraces ?? [])].sort((a, b) => b.createdAt.localeCompare(a.createdAt));
  const gacMetadata = memory.map(readGacMetadata);
  const pinnedRawMemoryIds = unique([
    ...gacMetadata.flatMap((metadata) => metadata.identityPinId ? readStringList(metadata, "rawMemoryIds", "rawMemoryId") : []),
    ...contextPackTraces.flatMap((trace) => trace.rawMemoryIds),
  ]);
  const identityPinIds = unique([
    ...gacMetadata.map((metadata) => readString(metadata, "identityPinId")).filter((id): id is string => Boolean(id)),
    ...contextPackTraces.flatMap((trace) => trace.identityPinIds),
  ]);
  const representativeIds = unique([
    ...gacMetadata.map((metadata) => readString(metadata, "representativeId")).filter((id): id is string => Boolean(id)),
    ...contextPackTraces.flatMap((trace) => trace.representativeIds),
  ]);
  const lastRuntimeTrace = runtimeTraces[0];
  const lastContextPackTrace = contextPackTraces[0];

  return {
    tenantId: input.tenantId,
    cellId: input.cellId,
    ...(input.sessionId ? { sessionId: input.sessionId } : {}),
    rebuiltAt: (input.now ?? new Date()).toISOString(),
    status: memory.length === 0 && runtimeTraces.length === 0 && contextPackTraces.length === 0 ? "empty" : "rebuilt",
    memoryCount: memory.length,
    runtimeTraceCount: runtimeTraces.length,
    contextPackTraceCount: contextPackTraces.length,
    pinnedRawMemoryIds,
    identityPinIds,
    representativeIds,
    ...(lastRuntimeTrace ? { lastRuntimeTraceId: lastRuntimeTrace.traceId } : {}),
    ...(lastContextPackTrace ? { lastContextPackTraceId: lastContextPackTrace.id } : {}),
    contextDigest: buildContextDigest(memory, contextPackTraces),
  };
}

function buildContextDigest(memory: MemoryChunk[], traces: ContextPackTraceRecord[]): string {
  const pinned = memory.filter((chunk) => readGacMetadata(chunk).identityPinId);
  const recent = [...pinned, ...memory.filter((chunk) => !readGacMetadata(chunk).identityPinId)]
    .slice(0, 5)
    .map((chunk) => `- ${chunk.text.slice(0, 180)}`);
  const traceLine = traces[0]
    ? `Last context pack: ${traces[0].id} (${traces[0].includedMemoryIds.length} memories).`
    : "No context-pack traces yet.";
  return [...recent, traceLine].join("\n");
}

function readGacMetadata(chunk: MemoryChunk): Record<string, unknown> {
  const gac = chunk.metadata.gac;
  return typeof gac === "object" && gac !== null && !Array.isArray(gac)
    ? { ...chunk.metadata, ...gac }
    : chunk.metadata;
}

function readString(metadata: Record<string, unknown>, key: string): string | undefined {
  const value = metadata[key];
  return typeof value === "string" && value.length > 0 ? value : undefined;
}

function readStringList(metadata: Record<string, unknown>, arrayKey: string, singleKey: string): string[] {
  const array = metadata[arrayKey];
  if (Array.isArray(array)) return array.filter((item): item is string => typeof item === "string" && item.length > 0);
  const single = readString(metadata, singleKey);
  return single ? [single] : [];
}

function unique(values: string[]): string[] {
  return [...new Set(values)];
}
