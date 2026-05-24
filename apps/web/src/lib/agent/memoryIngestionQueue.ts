export interface MemoryIngestionJobResult {
  ok: boolean;
  label: string;
  error?: string;
}

export interface MemoryIngestionQueueStats {
  pending: number;
  completed: number;
  failed: number;
  lastError?: string;
}

export interface QueuedMemoryIngestionJob {
  id: string;
  label: string;
  settled: Promise<MemoryIngestionJobResult>;
}

export interface MemoryIngestionFlushOptions {
  throwOnError?: boolean;
}

export class MemoryIngestionQueue {
  private tail: Promise<void> = Promise.resolve();
  private pendingJobs = 0;
  private completedJobs = 0;
  private failedJobs = 0;
  private lastFailure: Error | null = null;
  private currentDrain: Promise<void> = this.tail;

  get stats(): MemoryIngestionQueueStats {
    return {
      pending: this.pendingJobs,
      completed: this.completedJobs,
      failed: this.failedJobs,
      ...(this.lastFailure ? { lastError: this.lastFailure.message } : {}),
    };
  }

  enqueue(label: string, run: () => Promise<void>): QueuedMemoryIngestionJob {
    const id = makeQueueId(label);
    this.pendingJobs += 1;
    const settled = this.tail.then(async () => {
      try {
        await run();
        this.completedJobs += 1;
        return { ok: true, label } satisfies MemoryIngestionJobResult;
      } catch (error) {
        this.failedJobs += 1;
        this.lastFailure = error instanceof Error ? error : new Error(String(error));
        return {
          ok: false,
          label,
          error: this.lastFailure.message,
        } satisfies MemoryIngestionJobResult;
      } finally {
        this.pendingJobs = Math.max(0, this.pendingJobs - 1);
      }
    });
    this.tail = settled.then(() => undefined, () => undefined);
    this.currentDrain = this.tail;
    return { id, label, settled };
  }

  async flush(options: MemoryIngestionFlushOptions = {}): Promise<MemoryIngestionQueueStats> {
    const drain = this.currentDrain;
    await drain;
    const stats = this.stats;
    if (options.throwOnError && this.lastFailure) {
      throw this.lastFailure;
    }
    return stats;
  }
}

function makeQueueId(label: string): string {
  const safeLabel = label.replace(/[^a-z0-9_-]/gi, "_");
  if (typeof crypto !== "undefined" && "randomUUID" in crypto) {
    return `memory_${safeLabel}_${crypto.randomUUID()}`;
  }
  return `memory_${safeLabel}_${Date.now()}_${Math.random().toString(16).slice(2)}`;
}
