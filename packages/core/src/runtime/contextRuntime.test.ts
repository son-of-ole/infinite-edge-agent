import { describe, expect, it } from "vitest";
import { buildContextRuntimePlan } from "./contextRuntime";
import type { ContextPackTraceRecord, IdentityPinRecord, MemorySearchHit, RetrievalAuditRecord } from "../types";

const NOW = "2026-05-21T00:00:00.000Z";

describe("buildContextRuntimePlan adaptive GAC rebuild", () => {
  it("protects identity-pinned raw memory even when semantic score is low", () => {
    const result = buildContextRuntimePlan({
      requestId: "req_pin",
      systemPrompt: "You are local.",
      userMessage: "What constraints must I preserve?",
      recentMessages: [],
      retrievedMemory: [
        memory({
          id: "ordinary_high",
          score: 0.99,
          text: "Ordinary high semantic score memory.",
          tokenCount: 10,
        }),
        memory({
          id: "pin_low",
          score: 0.05,
          text: "Pinned exact memory: never delete raw identity facts during consolidation.",
          tokenCount: 10,
          metadata: {
            gac: {
              rawMemoryId: "raw_pin",
              identityPinId: "pin_1",
              memoryClass: "PINNED_EXACT",
              mustAttend: true,
              pinStrength: 1,
              identityRisk: 0.99,
            },
          },
        }),
      ],
      identityPins: [identityPin({ id: "pin_1", rawMemoryId: "raw_pin" })],
      maxRetrievedMemoryTokens: 10,
      maxRecentConversationTokens: 0,
      maxPromptTokens: 500,
    });

    expect(result.packed.includedMemoryIds).toEqual(["pin_low"]);
    expect(result.plan.memoryPriorityMap.pin_low).toMatchObject({
      protected: true,
      finalScore: 1,
      reasons: expect.arrayContaining([
        "identity_pin",
        "pinned_exact",
        "must_attend",
      ]),
      rawMemoryIds: ["raw_pin"],
      identityPinIds: ["pin_1"],
    });
    expect(result.plan.droppedFrameIds).toEqual(["ordinary_high"]);
  });

  it("learns from failed retrieval audits and boosts the raw memory that must be recovered next turn", () => {
    const result = buildContextRuntimePlan({
      requestId: "req_audit",
      systemPrompt: "You are local.",
      userMessage: "Recover the missed source detail.",
      recentMessages: [],
      retrievedMemory: [
        memory({
          id: "ordinary",
          score: 0.5,
          text: "Ordinary result that used to outrank the missed raw fact.",
          tokenCount: 10,
        }),
        memory({
          id: "missed_raw",
          score: 0.2,
          text: "Raw memory that failed a prior identity-preservation retrieval probe.",
          tokenCount: 10,
          metadata: {
            rawMemoryId: "raw_failed",
            memoryClass: "HIGH_RISK_RAW",
            identityRisk: 0.7,
          },
        }),
      ],
      retrievalAudits: [retrievalAudit({
        id: "audit_1",
        expectedRawMemoryId: "raw_failed",
        retrievedRawMemoryIds: ["ordinary"],
        identityPreserved: false,
      })],
      maxRetrievedMemoryTokens: 10,
      maxRecentConversationTokens: 0,
      maxPromptTokens: 500,
    });

    expect(result.packed.includedMemoryIds).toEqual(["missed_raw"]);
    expect(result.plan.memoryPriorityMap.missed_raw).toMatchObject({
      finalScore: expect.closeTo(0.69, 5),
      reasons: expect.arrayContaining([
        "high_risk_raw",
        "retrieval_audit_failure_repair",
      ]),
      rawMemoryIds: ["raw_failed"],
    });
    expect(result.plan.learningSignals.boostedMemoryIds).toContain("missed_raw");
  });

  it("builds a lineage map and records prior context-pack learning signals", () => {
    const result = buildContextRuntimePlan({
      requestId: "req_lineage",
      systemPrompt: "You are local.",
      userMessage: "Continue the task.",
      recentMessages: [],
      retrievedMemory: [
        memory({
          id: "rep_hit",
          score: 0.35,
          text: "Representative memory with lineage.",
          tokenCount: 8,
          metadata: {
            representativeId: "rep_1",
            rawMemoryIds: ["raw_a", "raw_b"],
            identityPinId: "pin_a",
            memoryClass: "LOW_RISK_REPRESENTATIVE",
          },
        }),
      ],
      contextPackTraces: [contextPackTrace({
        id: "ctx_prior",
        includedMemoryIds: ["old_consolidated_hit_id"],
        rawMemoryIds: ["raw_a"],
        representativeIds: ["rep_1"],
      })],
      maxRetrievedMemoryTokens: 8,
      maxRecentConversationTokens: 0,
      maxPromptTokens: 500,
    });

    expect(result.packed.includedMemoryIds).toEqual(["rep_hit"]);
    expect(result.plan.sourceLineageMap.rep_hit).toEqual({
      rawMemoryIds: ["raw_a", "raw_b"],
      representativeIds: ["rep_1"],
      identityPinIds: ["pin_a"],
    });
    expect(result.plan.memoryPriorityMap.rep_hit?.reasons).toEqual(expect.arrayContaining([
      "representative_lineage",
      "prior_context_inclusion",
    ]));
    expect(result.plan.learningSignals).toMatchObject({
      contextTraceCount: 1,
      retrievalAuditCount: 0,
      boostedMemoryIds: ["rep_hit"],
    });
  });
});

function memory(overrides: Partial<MemorySearchHit> & Pick<MemorySearchHit, "id" | "score" | "text">): MemorySearchHit {
  return {
    embedding: [1],
    sessionId: "session_1",
    source: "chat",
    createdAt: NOW,
    updatedAt: NOW,
    tags: [],
    metadata: {},
    tokenCount: Math.ceil(overrides.text.length / 4),
    ...overrides,
  };
}

function identityPin(overrides: Partial<IdentityPinRecord> & Pick<IdentityPinRecord, "id" | "rawMemoryId">): IdentityPinRecord {
  return {
    tenantId: "tenant_1",
    cellId: "cell_1",
    sessionId: "session_1",
    pinReason: "user_instruction",
    pinStrength: 1,
    createdBy: "policy",
    createdAt: NOW,
    ...overrides,
  };
}

function retrievalAudit(overrides: Partial<RetrievalAuditRecord> & Pick<RetrievalAuditRecord, "id" | "expectedRawMemoryId" | "retrievedRawMemoryIds" | "identityPreserved">): RetrievalAuditRecord {
  return {
    tenantId: "tenant_1",
    cellId: "cell_1",
    sessionId: "session_1",
    queryText: "Retrieve missed raw memory",
    retrievedRepresentativeIds: [],
    createdAt: NOW,
    ...overrides,
  };
}

function contextPackTrace(overrides: Partial<ContextPackTraceRecord> & Pick<ContextPackTraceRecord, "id" | "includedMemoryIds">): ContextPackTraceRecord {
  return {
    traceId: "trace_prior",
    tenantId: "tenant_1",
    cellId: "cell_1",
    sessionId: "session_1",
    queryId: "req_prior",
    contextPackId: "pack_prior",
    rawMemoryIds: [],
    representativeIds: [],
    identityPinIds: [],
    tokenBudget: 500,
    packingStrategy: "advanced-runtime",
    createdAt: NOW,
    ...overrides,
  };
}
