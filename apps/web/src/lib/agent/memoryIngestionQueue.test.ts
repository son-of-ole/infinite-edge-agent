import { describe, expect, it, vi } from "vitest";
import { MemoryIngestionQueue } from "./memoryIngestionQueue";

describe("MemoryIngestionQueue", () => {
  it("runs queued memory jobs serially and reports drain status", async () => {
    const events: string[] = [];
    const queue = new MemoryIngestionQueue();

    const first = queue.enqueue("first", async () => {
      events.push("first:start");
      await Promise.resolve();
      events.push("first:end");
    });
    const second = queue.enqueue("second", async () => {
      events.push("second:start");
      events.push("second:end");
    });

    expect(queue.stats.pending).toBe(2);
    await queue.flush({ throwOnError: true });

    expect(events).toEqual(["first:start", "first:end", "second:start", "second:end"]);
    expect(await first.settled).toMatchObject({ ok: true, label: "first" });
    expect(await second.settled).toMatchObject({ ok: true, label: "second" });
    expect(queue.stats).toMatchObject({ pending: 0, completed: 2, failed: 0 });
  });

  it("keeps later memory jobs running after a background ingestion failure", async () => {
    const queue = new MemoryIngestionQueue();
    const successfulJob = vi.fn(async () => undefined);

    queue.enqueue("bad", async () => {
      throw new Error("sidecar unavailable");
    });
    queue.enqueue("good", successfulJob);

    await expect(queue.flush({ throwOnError: true })).rejects.toThrow("sidecar unavailable");
    await expect(queue.flush()).resolves.toMatchObject({ failed: 1, completed: 1 });
    expect(successfulJob).toHaveBeenCalledOnce();
    expect(queue.stats.lastError).toContain("sidecar unavailable");
  });
});
