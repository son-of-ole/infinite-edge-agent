import { describe, expect, it } from "vitest";
import type { ContextPackTraceRecord, MemoryChunk, RuntimeTrace } from "../types";
import { rebuildStartupContextSnapshot } from "./startupRebuild";

describe("rebuildStartupContextSnapshot", () => {
  it("summarizes durable memory and latest traces for wake/startup without replaying every turn", () => {
    const snapshot = rebuildStartupContextSnapshot({
      tenantId: "tenant_1",
      cellId: "cell_1",
      sessionId: "session_1",
      memoryChunks: [
        makeChunk("mem_old", "Older memory", "2026-05-14T00:00:00.000Z"),
        makeChunk("mem_pin", "Remember: LanceDB is the production memory engine.", "2026-05-15T00:00:00.000Z", {
          gac: {
            rawMemoryId: "raw_pin",
            identityPinId: "pin_1",
            representativeId: "rep_1",
            memoryClass: "PINNED_EXACT",
          },
        }),
      ],
      runtimeTraces: [makeRuntimeTrace("trace_1", "2026-05-15T00:00:01.000Z")],
      contextPackTraces: [makeContextPackTrace("ctx_1", "trace_1", "2026-05-15T00:00:02.000Z")],
      now: new Date("2026-05-15T12:00:00.000Z"),
    });

    expect(snapshot).toMatchObject({
      tenantId: "tenant_1",
      cellId: "cell_1",
      sessionId: "session_1",
      memoryCount: 2,
      runtimeTraceCount: 1,
      contextPackTraceCount: 1,
      pinnedRawMemoryIds: ["raw_pin"],
      identityPinIds: ["pin_1"],
      representativeIds: ["rep_1"],
      lastRuntimeTraceId: "trace_1",
      lastContextPackTraceId: "ctx_1",
      status: "rebuilt",
    });
    expect(snapshot.contextDigest).toContain("Remember: LanceDB");
  });
});

function makeChunk(id: string, text: string, createdAt: string, metadata: Record<string, unknown> = {}): MemoryChunk {
  return {
    id,
    text,
    embedding: [0.1, 0.2],
    sessionId: "session_1",
    source: "chat",
    role: "user",
    createdAt,
    updatedAt: createdAt,
    tags: ["user"],
    metadata,
    tokenCount: 4,
  };
}

function makeRuntimeTrace(traceId: string, createdAt: string): RuntimeTrace {
  return {
    traceId,
    requestId: `req_${traceId}`,
    tenantId: "tenant_1",
    cellId: "cell_1",
    sessionId: "session_1",
    modelId: "Qwen/Qwen3-0.6B",
    backend: "unlocked-browser-transformer",
    createdAt,
    runtime: {},
  };
}

function makeContextPackTrace(id: string, traceId: string, createdAt: string): ContextPackTraceRecord {
  return {
    id,
    traceId,
    tenantId: "tenant_1",
    cellId: "cell_1",
    sessionId: "session_1",
    queryId: "req_1",
    contextPackId: "pack_1",
    rawMemoryIds: ["raw_pin"],
    representativeIds: ["rep_1"],
    identityPinIds: ["pin_1"],
    tokenBudget: 4096,
    estimatedTokens: 256,
    packingStrategy: "advanced-runtime",
    includedMemoryIds: ["mem_pin"],
    createdAt,
  };
}
