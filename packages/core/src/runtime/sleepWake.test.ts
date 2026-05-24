import { expect, it } from "vitest";
import type { BuildWakeContextInput } from "./sleepWake";
import { buildFailedSleepCycleRunRecord, buildWakeContext, rollbackSleepCycle, shouldRunSleepCycle } from "./sleepWake";
import type { ContextPackTraceRecord, IdentityPinRecord, RawMemoryRecord, RuntimeTrace } from "../types";

const NOW = new Date("2026-05-11T12:00:00.000Z");

function rawMemory(overrides: Partial<RawMemoryRecord> & Pick<RawMemoryRecord, "id" | "text" | "memoryKind">): RawMemoryRecord {
  return {
    tenantId: "tenant_1",
    cellId: "cell_1",
    sessionId: "session_1",
    sourceType: "chat",
    importance: 0.8,
    identityRiskSeed: 0.2,
    createdAt: NOW.toISOString(),
    updatedAt: NOW.toISOString(),
    retentionClass: "normal",
    hash: `hash_${overrides.id}`,
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
    createdBy: "user",
    createdAt: NOW.toISOString(),
    ...overrides,
  };
}

function contextPackTrace(): ContextPackTraceRecord {
  return {
    id: "ctx_trace_1",
    traceId: "trace_1",
    tenantId: "tenant_1",
    cellId: "cell_1",
    sessionId: "session_1",
    queryId: "request_1",
    contextPackId: "pack_1",
    rawMemoryIds: ["raw_pin", "raw_decision"],
    representativeIds: ["rep_existing"],
    identityPinIds: ["pin_1"],
    tokenBudget: 2048,
    packingStrategy: "advanced-runtime",
    includedMemoryIds: ["raw_pin", "raw_decision"],
    createdAt: NOW.toISOString(),
  };
}

function runtimeTrace(): RuntimeTrace {
  return {
    traceId: "runtime_trace_1",
    requestId: "request_1",
    sessionId: "session_1",
    modelId: "Qwen/Qwen3-0.6B",
    backend: "unlocked-browser-transformer",
    createdAt: NOW.toISOString(),
    runtime: {
      features: [],
    },
  };
}

function input(): BuildWakeContextInput {
  return {
    tenantId: "tenant_1",
    cellId: "cell_1",
    sessionId: "session_1",
    runId: "run_sleep_1",
    currentGoal: "Ship the first production sleep/wake consolidation layer.",
    rawMemories: [
      rawMemory({
        id: "raw_pin",
        text: "Keep raw memory as the ground truth and never overwrite it during sleep.",
        memoryKind: "instruction",
        retentionClass: "pinned",
      }),
      rawMemory({
        id: "raw_decision",
        text: "Decision: wake_context.md is derived and replayable from raw memory.",
        memoryKind: "decision",
      }),
    ],
    identityPins: [identityPin({ id: "pin_1", rawMemoryId: "raw_pin", pinReason: "source_of_truth" })],
    contextPackTraces: [contextPackTrace()],
    runtimeTraces: [runtimeTrace()],
    openTasks: ["Persist records into the LanceDB sidecar once storage tables are wired."],
    risks: ["Representative rollback still needs storage-level inactive marking."],
    now: NOW,
  };
}

it("builds wake_context.md with every required wake section", () => {
  const result = buildWakeContext(input());

  expect(result.wakeContext.fileName).toBe("wake_context.md");
  expect(result.wakeContext.status).toBe("complete");
  expect(Object.keys(result.wakeContext.sections)).toEqual([
    "Cell Identity",
    "Current Goal",
    "Pinned Constraints",
    "Decisions Since Last Wake",
    "Open Tasks",
    "Important Sources",
    "Memory Map",
    "Risks and Unknowns",
    "Next Suggested Actions",
  ]);
  expect(result.wakeContext.markdown).toContain("## Cell Identity");
  expect(result.wakeContext.markdown).toContain("## Next Suggested Actions");
});

it("keeps pinned constraints linked to raw memory and emits retrieval probes", () => {
  const result = buildWakeContext(input());

  expect(result.wakeContext.sections["Pinned Constraints"]).toContain("never overwrite");
  expect(result.wakeContext.sections["Pinned Constraints"]).toContain("[raw:raw_pin]");
  expect(result.wakeContext.identityPinIds).toEqual(["pin_1"]);
  expect(result.retrievalAudits).toHaveLength(1);
  expect(result.retrievalAudits[0]).toMatchObject({
    expectedRawMemoryId: "raw_pin",
    retrievedRawMemoryIds: ["raw_pin"],
    identityPreserved: true,
    hitAtK: 1,
  });
});

it("preserves open tasks in the wake context and next suggested actions", () => {
  const result = buildWakeContext(input());

  expect(result.wakeContext.sections["Open Tasks"]).toContain("Persist records into the LanceDB sidecar");
  expect(result.wakeContext.sections["Next Suggested Actions"]).toContain("Continue: Persist records into the LanceDB sidecar");
  expect(result.runRecord.openTaskCount).toBe(1);
});

it("uses deterministic ids for the same runId", () => {
  const first = buildWakeContext(input());
  const second = buildWakeContext(input());

  expect(second.runRecord.id).toBe(first.runRecord.id);
  expect(second.wakeContext.id).toBe(first.wakeContext.id);
  expect(second.representatives.map((record) => record.id)).toEqual(first.representatives.map((record) => record.id));
  expect(second.retrievalAudits.map((record) => record.id)).toEqual(first.retrievalAudits.map((record) => record.id));
});

it("does not mutate raw inputs while deriving records", () => {
  const buildInput = input();
  const before = JSON.stringify(buildInput);

  buildWakeContext(buildInput);

  expect(JSON.stringify(buildInput)).toBe(before);
});

it("rolls back at the representative layer without deleting raw source ids", () => {
  const result = buildWakeContext(input());
  const rollback = rollbackSleepCycle({
    runRecord: result.runRecord,
    reason: "Bad representative summary failed manual audit.",
    now: new Date("2026-05-11T12:10:00.000Z"),
  });

  expect(rollback.runRecord.status).toBe("rolled_back");
  expect(rollback.runRecord.sourceRawMemoryIds).toEqual(["raw_pin", "raw_decision"]);
  expect(rollback.rollbackAudit.sourceRawMemoryIds).toEqual(["raw_pin", "raw_decision"]);
  expect(rollback.rollbackAudit.affectedRepresentativeIds).toEqual(result.runRecord.representativeIds);
  expect(rollback.rollbackAudit.previousWakeContextId).toBe(result.wakeContext.id);
});

it("records failed sleep cycles without producing a replacement wake context", () => {
  const failed = buildFailedSleepCycleRunRecord(input(), "retrieval probes did not pass");

  expect(failed.status).toBe("failed");
  expect(failed.wakeContextId).toBeUndefined();
  expect(failed.representativeIds).toEqual([]);
  expect(failed.sourceRawMemoryIds).toEqual(["raw_pin", "raw_decision"]);
  expect(failed.error).toBe("retrieval probes did not pass");
});

it("lets operators disable the sleep cycle before derived records are built", () => {
  expect(shouldRunSleepCycle({
    enabled: false,
    pendingRawMemoryCount: 100,
  })).toEqual({
    run: false,
    reason: "disabled_by_policy",
  });
});
