import type { WebGpuDecodeCommandBatchTrace } from "./types";

export interface WebGpuCommandEncoderLike {
  beginComputePass?(): {
    setPipeline(pipeline: unknown): void;
    setBindGroup(index: number, bindGroup: unknown): void;
    dispatchWorkgroups(x: number, y?: number, z?: number): void;
    end(): void;
  };
  copyBufferToBuffer?(source: unknown, sourceOffset: number, destination: unknown, destinationOffset: number, size: number): void;
  finish(): unknown;
}

export interface WebGpuDeviceLike {
  queue: {
    submit(commandBuffers: unknown[]): void;
    onSubmittedWorkDone?(): Promise<unknown>;
  };
  createCommandEncoder(descriptor?: { label?: string }): WebGpuCommandEncoderLike;
}

export interface RecordComputePassInput {
  label: string;
  dispatches?: number;
  record: (encoder: WebGpuCommandEncoderLike) => void;
}

export interface RecordCopyCommandInput {
  label: string;
  record: (encoder: WebGpuCommandEncoderLike) => void;
}

export interface WebGpuDecodeCommandBatchOptions {
  requestId: string;
  tokenIndex: number;
  layerIndex?: number;
  label?: string;
}

export class WebGpuDecodeCommandBatch {
  private readonly commandBuffers: unknown[] = [];
  private passCount = 0;
  private dispatchCount = 0;
  private submitCount = 0;
  private submitted = false;
  private forbiddenSyncDetected = false;
  private readonly labels: string[] = [];
  private readonly afterSubmit: Array<() => void> = [];

  constructor(
    private readonly device: WebGpuDeviceLike,
    private readonly options: WebGpuDecodeCommandBatchOptions,
  ) {}

  recordComputePass(input: RecordComputePassInput): void {
    this.assertOpen();
    this.labels.push(input.label);
    this.passCount += 1;
    this.dispatchCount += Math.max(0, Math.floor(input.dispatches ?? 1));
    const encoder = this.createCommandEncoder(input.label);
    input.record(encoder);
    this.commandBuffers.push(encoder.finish());
  }

  recordCopy(input: RecordCopyCommandInput): void {
    this.assertOpen();
    this.labels.push(input.label);
    const encoder = this.createCommandEncoder(input.label);
    input.record(encoder);
    this.commandBuffers.push(encoder.finish());
  }

  markForbiddenSync(label: string): void {
    this.forbiddenSyncDetected = true;
    this.labels.push(`forbidden-sync:${label}`);
  }

  deferAfterSubmit(cleanup: () => void): void {
    this.assertOpen();
    this.afterSubmit.push(cleanup);
  }

  async submitOnce(): Promise<WebGpuDecodeCommandBatchTrace> {
    this.assertOpen();
    this.device.queue.submit(this.commandBuffers.splice(0));
    this.submitCount = 1;
    this.submitted = true;
    this.flushDeferredCleanup();
    return this.trace();
  }

  trace(): WebGpuDecodeCommandBatchTrace {
    return {
      requestId: this.options.requestId,
      tokenIndex: this.options.tokenIndex,
      ...(this.options.layerIndex !== undefined ? { layerIndex: this.options.layerIndex } : {}),
      passCount: this.passCount,
      commandBufferCount: this.passCount,
      dispatchCount: this.dispatchCount,
      submitCount: this.submitCount,
      submitted: this.submitted,
      labels: [...this.labels],
      forbiddenSyncDetected: this.forbiddenSyncDetected,
    };
  }

  private assertOpen(): void {
    if (this.submitted) throw new Error("WebGpuDecodeCommandBatch has already been submitted.");
  }

  private createCommandEncoder(label: string): WebGpuCommandEncoderLike {
    return this.device.createCommandEncoder({
      label: `${this.options.label ?? `decode-batch:${this.options.requestId}:${this.options.tokenIndex}:${this.options.layerIndex ?? "token"}`}:${this.labels.length}:${label}`,
    });
  }

  private flushDeferredCleanup(): void {
    const cleanups = this.afterSubmit.splice(0).reverse();
    if (cleanups.length === 0) return;
    const runCleanups = (): void => {
      for (const cleanup of cleanups) cleanup();
    };
    const onSubmittedWorkDone = this.device.queue.onSubmittedWorkDone;
    if (typeof onSubmittedWorkDone === "function") {
      void onSubmittedWorkDone.call(this.device.queue).then(runCleanups, runCleanups);
      return;
    }
    runCleanups();
  }
}

export function assertNoStrictDecodeSync(input: {
  phase: "prefill" | "decode";
  strict: boolean;
  operation: string;
}): void {
  if (input.phase === "decode" && input.strict) {
    throw new Error(`Strict WebGPU decode forbids synchronization operation: ${input.operation}`);
  }
}
