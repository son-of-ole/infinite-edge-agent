import { describe, expect, it, vi } from "vitest";
import type { ProposeModelMemoryActionRequest } from "../types";
import {
  buildModelMemoryToolInterface,
  modelMemoryActionTraceToRecord,
  proposeModelMemoryAction,
} from "./modelMemoryActions";

const now = new Date("2026-05-11T18:00:00.000Z");

describe("proposeModelMemoryAction", () => {
  it("approves in shadow mode without executing", async () => {
    const execute = vi.fn();
    const traceSink = vi.fn();

    const response = await proposeModelMemoryAction(makeRequest(), {
      now,
      actionIdFactory: () => "action_shadow",
      execute,
      onTrace: traceSink,
    });

    expect(response).toMatchObject({
      actionId: "action_shadow",
      approved: true,
      executed: false,
      resultIds: [],
    });
    expect(execute).not.toHaveBeenCalled();
    expect(traceSink).toHaveBeenCalledWith(expect.objectContaining({
      actionId: "action_shadow",
      mode: "shadow",
      timestamp: "2026-05-11T18:00:00.000Z",
      policyViolations: [],
      request: expect.objectContaining({
        actionType: "pin_memory",
        targetIds: ["mem_1"],
      }),
      decision: expect.objectContaining({ approved: true }),
    }));
    expect(response.policyNotes).toContain("Shadow mode recorded the approved action without executing it.");
  });

  it("executes approved actions in enforced mode", async () => {
    const execute = vi.fn(async () => ({ resultIds: ["pin_1"] }));

    const response = await proposeModelMemoryAction(makeRequest(), {
      mode: "enforced",
      now,
      actionIdFactory: () => "action_enforced",
      execute,
    });

    expect(response).toMatchObject({
      actionId: "action_enforced",
      approved: true,
      executed: true,
      resultIds: ["pin_1"],
    });
    expect(execute).toHaveBeenCalledOnce();
    expect(execute).toHaveBeenCalledWith(expect.objectContaining({ actionType: "pin_memory" }), expect.objectContaining({
      actionId: "action_enforced",
      mode: "enforced",
    }));
  });

  it("rejects all actions when disabled", async () => {
    const execute = vi.fn();

    const response = await proposeModelMemoryAction(makeRequest(), {
      mode: "disabled",
      now,
      actionIdFactory: () => "action_disabled",
      execute,
    });

    expect(response.approved).toBe(false);
    expect(response.executed).toBe(false);
    expect(response.trace.policyViolations.map((item) => item.code)).toContain("memory_action_disabled");
    expect(execute).not.toHaveBeenCalled();
  });

  it("rejects unsafe destructive actions unless explicitly allowed", async () => {
    const response = await proposeModelMemoryAction({
      ...makeRequest(),
      actionType: "forget_memory",
      targetIds: ["mem_1"],
      arguments: undefined,
    }, {
      mode: "enforced",
      now,
      actionIdFactory: () => "action_forget",
      execute: vi.fn(),
    });

    expect(response.approved).toBe(false);
    expect(response.executed).toBe(false);
    expect(response.trace.policyViolations.map((item) => item.code)).toContain("destructive_action_not_allowed");
  });

  it("allows destructive actions only when policy explicitly grants them", async () => {
    const execute = vi.fn(() => ["tombstone_1"]);

    const response = await proposeModelMemoryAction({
      ...makeRequest(),
      actionType: "forget_memory",
      targetIds: ["mem_1"],
      arguments: undefined,
    }, {
      mode: "enforced",
      policy: { allowDestructiveActions: true },
      now,
      actionIdFactory: () => "action_forget_allowed",
      execute,
    });

    expect(response).toMatchObject({
      approved: true,
      executed: true,
      resultIds: ["tombstone_1"],
    });
  });

  it("rejects pin_memory without required pin arguments", async () => {
    const response = await proposeModelMemoryAction({
      ...makeRequest(),
      arguments: { pinReason: "architecture_decision" },
    }, {
      now,
      actionIdFactory: () => "action_bad_pin",
    });

    expect(response.approved).toBe(false);
    expect(response.trace.policyViolations.map((item) => item.code)).toContain("invalid_pin_arguments");
  });

  it("rejects low confidence proposals", async () => {
    const response = await proposeModelMemoryAction({
      ...makeRequest(),
      confidence: 0.74,
    }, {
      now,
      actionIdFactory: () => "action_low_confidence",
    });

    expect(response.approved).toBe(false);
    expect(response.trace.policyViolations.map((item) => item.code)).toContain("low_confidence");
  });

  it("rejects required-target actions with empty targetIds", async () => {
    const response = await proposeModelMemoryAction({
      ...makeRequest(),
      targetIds: [],
    }, {
      now,
      actionIdFactory: () => "action_missing_targets",
    });

    expect(response.approved).toBe(false);
    expect(response.trace.policyViolations.map((item) => item.code)).toContain("missing_target_ids");
  });

  it("rejects cross-tenant and cross-cell attempts outside policy scope", async () => {
    const response = await proposeModelMemoryAction({
      ...makeRequest(),
      tenantId: "tenant_other",
      cellId: "cell_other",
    }, {
      now,
      actionIdFactory: () => "action_scope_mismatch",
      policy: {
        scope: {
          tenantId: "tenant_allowed",
          cellId: "cell_allowed",
        },
      },
    });

    expect(response.approved).toBe(false);
    expect(response.trace.policyViolations.map((item) => item.code)).toEqual(expect.arrayContaining([
      "tenant_scope_mismatch",
      "cell_scope_mismatch",
    ]));
  });

  it("defines a prompt/tool interface and converts traces to auditable storage records", async () => {
    const toolInterface = buildModelMemoryToolInterface({
      mode: "shadow",
      tenantId: "tenant_allowed",
      cellId: "cell_allowed",
    });
    const response = await proposeModelMemoryAction(makeRequest(), {
      now,
      actionIdFactory: () => "action_logged",
    });
    const record = modelMemoryActionTraceToRecord(response.trace, {
      tenantId: "tenant_allowed",
      cellId: "cell_allowed",
    });

    expect(toolInterface.toolName).toBe("propose_model_memory_action");
    expect(toolInterface.systemInstruction).toContain("shadow mode");
    expect(toolInterface.jsonSchema.required).toEqual(expect.arrayContaining([
      "sessionId",
      "modelId",
      "actionType",
      "targetIds",
      "confidence",
    ]));
    expect(record).toMatchObject({
      id: "action_logged",
      tenantId: "tenant_allowed",
      cellId: "cell_allowed",
      sessionId: "session_1",
      actionType: "pin_memory",
      targetIds: ["mem_1"],
      approvedByPolicy: true,
      mode: "shadow",
    });
  });
});

function makeRequest(): ProposeModelMemoryActionRequest {
  return {
    tenantId: "tenant_allowed",
    cellId: "cell_allowed",
    sessionId: "session_1",
    modelId: "Qwen/Qwen3-0.6B",
    actionType: "pin_memory",
    targetIds: ["mem_1"],
    arguments: {
      pinReason: "architecture_decision",
      pinStrength: 0.95,
    },
    confidence: 0.91,
  };
}
