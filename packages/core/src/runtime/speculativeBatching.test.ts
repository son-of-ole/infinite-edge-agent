import { describe, expect, it } from "vitest";
import {
  evaluateSpeculationAutoDisable,
  summarizeSpeculativeBatchMetrics,
  verifySpeculativeBatch,
  type SpeculativeVerifierBackend,
} from "./speculative";

describe("MTP speculative verifier batching", () => {
  it("verifies multiple draft branches deterministically and records token traces", async () => {
    const backend: SpeculativeVerifierBackend = async (batch) => ({
      requestId: batch.requestId,
      verifyLatencyMs: 40,
      branches: [
        {
          branchId: "branch-a",
          verification: [
            { token: "A", accepted: true },
            { token: "B", accepted: true },
          ],
        },
        {
          branchId: "branch-b",
          verification: [
            { token: "C", accepted: true },
            { token: "X", accepted: false, replacement: "X" },
          ],
        },
      ],
    });

    const result = await verifySpeculativeBatch({
      requestId: "req-123",
      modelPair: { draftModelId: "draft-small", targetModelId: "target-main" },
      taskType: "chat",
      draftLatencyMs: 15,
      targetOnlyLatencyMs: 80,
      minAcceptanceRate: 0.45,
      disableWhenLatencyWorse: true,
      branches: [
        { branchId: "branch-a", draft: [{ token: "A" }, { token: "B" }] },
        { branchId: "branch-b", draft: [{ token: "C" }, { token: "D" }, { token: "E" }] },
      ],
    }, backend);

    expect(result.branches).toEqual([
      {
        branchId: "branch-a",
        streamedTokens: ["A", "B"],
        acceptedTokens: 2,
        rejectedTokens: 0,
      },
      {
        branchId: "branch-b",
        streamedTokens: ["C", "X"],
        acceptedTokens: 1,
        rejectedTokens: 2,
        correctedToken: "X",
      },
    ]);

    expect(result.traces).toEqual([
      expect.objectContaining({
        requestId: "req-123",
        branchId: "branch-a",
        taskType: "chat",
        modelPair: { draftModelId: "draft-small", targetModelId: "target-main" },
        acceptedTokens: ["A", "B"],
        rejectedTokens: [],
      }),
      expect.objectContaining({
        requestId: "req-123",
        branchId: "branch-b",
        taskType: "chat",
        modelPair: { draftModelId: "draft-small", targetModelId: "target-main" },
        acceptedTokens: ["C"],
        rejectedTokens: ["D", "E"],
        correctedToken: "X",
      }),
    ]);
  });

  it("summarizes batch metrics by model pair and task type", async () => {
    const backend: SpeculativeVerifierBackend = async () => ({
      verifyLatencyMs: 25,
      branches: [
        {
          branchId: "branch-a",
          verification: [
            { token: "A", accepted: true },
            { token: "B", accepted: true },
          ],
        },
        {
          branchId: "branch-b",
          verification: [
            { token: "C", accepted: false, replacement: "Z" },
          ],
        },
      ],
    });

    const result = await verifySpeculativeBatch({
      requestId: "req-metrics",
      modelPair: { draftModelId: "draft-small", targetModelId: "target-main" },
      taskType: "summarization",
      draftLatencyMs: 10,
      targetOnlyLatencyMs: 70,
      minAcceptanceRate: 0.45,
      disableWhenLatencyWorse: true,
      branches: [
        { branchId: "branch-a", draft: [{ token: "A" }, { token: "B" }] },
        { branchId: "branch-b", draft: [{ token: "C" }, { token: "D" }] },
      ],
    }, backend);

    expect(result.metrics).toMatchObject({
      draftTokens: 4,
      acceptedTokens: 2,
      rejectedTokens: 2,
      acceptanceRate: 0.5,
      draftLatencyMs: 10,
      verifyLatencyMs: 25,
      netSpeedupRatio: 2,
      modelPair: { draftModelId: "draft-small", targetModelId: "target-main" },
      taskType: "summarization",
    });

    expect(summarizeSpeculativeBatchMetrics(result.traces, {
      draftLatencyMs: 10,
      verifyLatencyMs: 25,
      targetOnlyLatencyMs: 70,
      minAcceptanceRate: 0.45,
      disableWhenLatencyWorse: true,
    })).toEqual(result.metrics);
  });

  it("lets auto-disable consume slower low-acceptance batch metrics", async () => {
    const backend: SpeculativeVerifierBackend = async () => ({
      verifyLatencyMs: 90,
      branches: [
        {
          branchId: "branch-a",
          verification: [{ token: "wrong", accepted: false, replacement: "right" }],
        },
      ],
    });

    const result = await verifySpeculativeBatch({
      requestId: "req-slow",
      modelPair: { draftModelId: "draft-small", targetModelId: "target-main" },
      taskType: "chat",
      draftLatencyMs: 30,
      targetOnlyLatencyMs: 100,
      minAcceptanceRate: 0.45,
      disableWhenLatencyWorse: true,
      branches: [
        { branchId: "branch-a", draft: [{ token: "wrong" }, { token: "extra" }] },
      ],
    }, backend);

    expect(result.metrics).toMatchObject({
      acceptedTokens: 0,
      rejectedTokens: 2,
      acceptanceRate: 0,
      disabledReason: "acceptance_rate_below_threshold",
    });
    expect(evaluateSpeculationAutoDisable([result.metrics], {
      minAcceptanceRate: 0.45,
      disableWhenLatencyWorse: true,
    })).toMatchObject({
      disabled: true,
      disabledReason: "acceptance_rate_below_threshold",
    });
  });

  it("adds request and branch context to backend verifier errors", async () => {
    const backend: SpeculativeVerifierBackend = async () => {
      throw Object.assign(new Error("target verifier unavailable"), { branchId: "branch-b" });
    };

    let error: Error | undefined;
    try {
      await verifySpeculativeBatch({
        requestId: "req-error",
        modelPair: { draftModelId: "draft-small", targetModelId: "target-main" },
        taskType: "chat",
        draftLatencyMs: 1,
        targetOnlyLatencyMs: 1,
        minAcceptanceRate: 0.45,
        disableWhenLatencyWorse: true,
        branches: [
          { branchId: "branch-a", draft: [{ token: "A" }] },
          { branchId: "branch-b", draft: [{ token: "B" }] },
        ],
      }, backend);
    } catch (caught) {
      error = caught as Error;
    }

    expect(error).toBeDefined();
    expect(error?.message).toContain("Speculative verifier failed for request req-error branch branch-b");
    expect(error?.stack).toContain("Speculative verifier failed for request req-error branch branch-b");
    expect((error as Error & { cause?: unknown }).cause).toBeInstanceOf(Error);
  });

  it("rejects mismatched backend request ids", async () => {
    const backend: SpeculativeVerifierBackend = async () => ({
      requestId: "req-other",
      verifyLatencyMs: 1,
      branches: [
        {
          branchId: "branch-a",
          verification: [{ token: "A", accepted: true }],
        },
      ],
    });

    await expect(verifySpeculativeBatch({
      requestId: "req-expected",
      modelPair: { draftModelId: "draft-small", targetModelId: "target-main" },
      taskType: "chat",
      draftLatencyMs: 1,
      targetOnlyLatencyMs: 2,
      minAcceptanceRate: 0.45,
      disableWhenLatencyWorse: true,
      branches: [
        { branchId: "branch-a", draft: [{ token: "A" }] },
      ],
    }, backend)).rejects.toThrow(
      "Speculative verifier failed for request req-expected: backend returned request req-other",
    );
  });

  it("rejects duplicate input branch ids before calling the backend", async () => {
    let called = false;
    const backend: SpeculativeVerifierBackend = async () => {
      called = true;
      return { verifyLatencyMs: 1, branches: [] };
    };

    await expect(verifySpeculativeBatch({
      requestId: "req-dup-input",
      modelPair: { draftModelId: "draft-small", targetModelId: "target-main" },
      taskType: "chat",
      draftLatencyMs: 1,
      targetOnlyLatencyMs: 2,
      minAcceptanceRate: 0.45,
      disableWhenLatencyWorse: true,
      branches: [
        { branchId: "branch-a", draft: [{ token: "A" }] },
        { branchId: "branch-a", draft: [{ token: "B" }] },
      ],
    }, backend)).rejects.toThrow(
      "Speculative verifier failed for request req-dup-input branch branch-a: duplicate input branch id",
    );
    expect(called).toBe(false);
  });

  it("rejects duplicate backend branch ids", async () => {
    const backend: SpeculativeVerifierBackend = async () => ({
      verifyLatencyMs: 1,
      branches: [
        {
          branchId: "branch-a",
          verification: [{ token: "A", accepted: true }],
        },
        {
          branchId: "branch-a",
          verification: [{ token: "B", accepted: true }],
        },
      ],
    });

    await expect(verifySpeculativeBatch({
      requestId: "req-dup-backend",
      modelPair: { draftModelId: "draft-small", targetModelId: "target-main" },
      taskType: "chat",
      draftLatencyMs: 1,
      targetOnlyLatencyMs: 2,
      minAcceptanceRate: 0.45,
      disableWhenLatencyWorse: true,
      branches: [
        { branchId: "branch-a", draft: [{ token: "A" }] },
      ],
    }, backend)).rejects.toThrow(
      "Speculative verifier failed for request req-dup-backend branch branch-a: duplicate backend branch id",
    );
  });
});
