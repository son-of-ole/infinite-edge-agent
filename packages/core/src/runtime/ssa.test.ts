import { describe, expect, it } from "vitest";
import type { MemorySearchHit } from "../types";
import {
  buildSparseSSAPlan,
  FallbackSSARuntime,
  NativeSSABackendTestDouble,
  selectSSAAnchors,
  type ContextBlock,
  type SSAPlanInput,
} from "./ssa";

describe("SSA planner", () => {
  it("pins anchors, emits sparse per-query routes, and explains every block", async () => {
    const input = makeSSAInput();
    const plan = await new FallbackSSARuntime().plan(input);

    expect(plan.mode).toBe("fallback_sparse_planner");
    expect(plan.pinnedBlockIds).toEqual(["b_current", "b_system"]);
    expect(plan.selectedBlockIds).toContain("b_needle");
    expect(plan.droppedBlockIds).toContain("b_low");
    expect(Object.keys(plan.layerPolicies[0]?.selectedBlockIdsByQueryBlock ?? {})).toHaveLength(6);
    expect(plan.layerPolicies[0]?.selectedBlockIdsByQueryBlock[4]).toEqual(expect.arrayContaining(["b_system", "b_current"]));
    expect(plan.routingTrace).toHaveLength(input.activeBlocks.length);
    expect(plan.routingTrace.every((entry) => entry.reasons.length > 0)).toBe(true);
    expect(plan.kernelTraces[0]).toMatchObject({
      requestId: "req_ssa",
      layerIndex: 0,
      denseTokenCountEstimate: 96,
      sparseTokenCountEstimate: 48,
    });
  });

  it("selects implicit and explicit anchors above threshold only", () => {
    const anchors = selectSSAAnchors({
      ...makeSSAInput(),
      anchors: [
        { blockId: "b_low", reason: "too_weak", score: 0.2 },
        { blockId: "b_needle", reason: "explicit_constraint", score: 0.9 },
      ],
      minAnchorScore: 0.5,
    });

    expect(anchors.map((anchor) => anchor.blockId)).toEqual(["b_current", "b_system", "b_needle"]);
    expect(anchors.find((anchor) => anchor.blockId === "b_low")).toBeUndefined();
  });

  it("keeps pinned anchors even when sparse budget is smaller than the anchor set", () => {
    const plan = buildSparseSSAPlan({ ...makeSSAInput(), maxBlocks: 1 });

    expect(plan.selectedBlockIds).toEqual(expect.arrayContaining(["b_system", "b_current"]));
    expect(plan.droppedBlockIds).not.toContain("b_system");
    expect(plan.droppedBlockIds).not.toContain("b_current");
  });

  it("provides a native backend harness without claiming browser tensor execution", async () => {
    const backend = new NativeSSABackendTestDouble();
    const plan = await backend.planSparseAttention(makeSSAInput());
    const output = await backend.executeSparseForward({
      requestId: "req_ssa",
      layerIndex: 3,
      qHandle: { id: "q" },
      kHandle: { id: "k" },
      vHandle: { id: "v" },
      routingPolicy: plan.layerPolicies[0] as NonNullable<typeof plan.layerPolicies[0]>,
    });

    expect(backend.supportsNativeSSA()).toBe(true);
    expect(plan.mode).toBe("backend_native");
    expect(plan.layerPolicies[0]?.denseFallback).toBe(false);
    expect(output.trace).toMatchObject({ requestId: "req_ssa", layerIndex: 3 });
  });
});

function makeSSAInput(): SSAPlanInput {
  const blocks: ContextBlock[] = [
    block("b_system", 0, 16, 0.7, ["system"]),
    block("b_old", 16, 32, 0.2),
    block("b_needle", 32, 48, 0.6, ["document"]),
    block("b_low", 48, 64, 0.05),
    block("b_current", 64, 80, 0.9, ["current_user_request"]),
    block("b_misc", 80, 96, 0.1),
  ];
  return {
    requestId: "req_ssa",
    activeBlocks: blocks,
    anchors: [{ blockId: "b_system", reason: "system_prompt", score: 1 }],
    memoryHits: [memoryHit("b_needle", 0.95)],
    maxBlocks: 3,
    minAnchorScore: 0.5,
    blockSize: 16,
    topKBlocks: 1,
    localWindowBlocks: 1,
  };
}

function block(id: string, tokenStart: number, tokenEnd: number, priority: number, tags: string[] = []): ContextBlock {
  return {
    id,
    text: id,
    tokenStart,
    tokenEnd,
    priority,
    source: "test",
    tags,
  };
}

function memoryHit(id: string, score: number): MemorySearchHit {
  return {
    id,
    text: id,
    embedding: [1, 0],
    score,
    sessionId: "session",
    source: "document",
    createdAt: "2026-05-15T00:00:00.000Z",
    updatedAt: "2026-05-15T00:00:00.000Z",
    tags: [],
    metadata: {},
    tokenCount: 16,
  };
}
