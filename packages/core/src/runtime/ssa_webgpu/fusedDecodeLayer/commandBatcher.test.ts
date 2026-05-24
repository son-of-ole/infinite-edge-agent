import { describe, expect, it } from "vitest";

import { WebGpuDecodeCommandBatch } from "./commandBatcher";

describe("WebGpuDecodeCommandBatch", () => {
  it("defers GPU cleanup until after the batched command buffer is submitted", async () => {
    const events: string[] = [];
    const batch = new WebGpuDecodeCommandBatch({
      queue: {
        submit: () => {
          events.push("submit");
        },
      },
      createCommandEncoder: () => ({
        beginComputePass: () => ({
          setPipeline: () => undefined,
          setBindGroup: () => undefined,
          dispatchWorkgroups: () => undefined,
          end: () => undefined,
        }),
        finish: () => {
          events.push("finish");
          return {};
        },
      }),
    }, {
      requestId: "req_batch_cleanup",
      tokenIndex: 1,
      layerIndex: 0,
    });

    batch.recordComputePass({
      label: "fixture-pass",
      dispatches: 1,
      record: (encoder) => {
        const pass = encoder.beginComputePass?.();
        pass?.end();
      },
    });
    batch.deferAfterSubmit(() => {
      events.push("cleanup");
    });

    expect(events).toEqual(["finish"]);
    await batch.submitOnce();

    expect(events).toEqual(["finish", "submit", "cleanup"]);
  });

  it("keeps deferred cleanup pending until real WebGPU reports submitted work done", async () => {
    const events: string[] = [];
    let completeSubmittedWork: () => void = () => undefined;
    const submittedWorkDone = new Promise<void>((resolve) => {
      completeSubmittedWork = resolve;
    });
    const batch = new WebGpuDecodeCommandBatch({
      queue: {
        submit: () => {
          events.push("submit");
        },
        onSubmittedWorkDone: async () => submittedWorkDone,
      },
      createCommandEncoder: () => ({
        beginComputePass: () => ({
          setPipeline: () => undefined,
          setBindGroup: () => undefined,
          dispatchWorkgroups: () => undefined,
          end: () => undefined,
        }),
        finish: () => {
          events.push("finish");
          return {};
        },
      }),
    }, {
      requestId: "req_batch_async_cleanup",
      tokenIndex: 1,
      layerIndex: 0,
    });

    batch.recordComputePass({
      label: "fixture-pass",
      dispatches: 1,
      record: (encoder) => {
        const pass = encoder.beginComputePass?.();
        pass?.end();
      },
    });
    batch.deferAfterSubmit(() => {
      events.push("cleanup");
    });

    await batch.submitOnce();
    expect(events).toEqual(["finish", "submit"]);

    completeSubmittedWork();
    await submittedWorkDone;
    await Promise.resolve();

    expect(events).toEqual(["finish", "submit", "cleanup"]);
  });
});
