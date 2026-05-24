import type { TSPScheduleStep } from "./tsp";

export interface TSPExecutionContext {
  step: TSPScheduleStep;
  order: number;
  originalIndex: number;
  metadata: Record<string, unknown>;
}

export type TSPExecutionCallback = (context: TSPExecutionContext) => unknown | Promise<unknown>;

export type TSPExecutorCallbacks = Partial<Record<TSPScheduleStep["kind"], TSPExecutionCallback>>;

export interface TSPExecutorOptions {
  metadata?: Record<string, unknown>;
}

export interface TSPExecutionTraceRecord {
  stepId: string;
  kind: TSPScheduleStep["kind"];
  sequenceShard: number;
  tensorShard: number;
  tokenStart: number;
  tokenEnd: number;
  order: number;
  originalIndex: number;
  status: "ok";
  metadata: Record<string, unknown>;
}

interface IndexedTSPScheduleStep {
  step: TSPScheduleStep;
  originalIndex: number;
}

export async function executeTSPSchedule(
  schedule: TSPScheduleStep[],
  callbacks: TSPExecutorCallbacks,
  options: TSPExecutorOptions = {},
): Promise<TSPExecutionTraceRecord[]> {
  const ordered = schedule
    .map((step, originalIndex) => ({ step, originalIndex }))
    .sort(compareIndexedSteps);
  const trace: TSPExecutionTraceRecord[] = [];

  for (const { step } of ordered) {
    if (!callbacks[step.kind]) {
      throw new Error(`TSP executor missing required callback for step kind "${step.kind}" (step: ${step.id})`);
    }
  }

  for (let order = 0; order < ordered.length; order += 1) {
    const { step, originalIndex } = ordered[order] as IndexedTSPScheduleStep;
    const callback = callbacks[step.kind] as TSPExecutionCallback;
    const callbackMetadata = frozenMetadataSnapshot(options.metadata ?? {});

    await callback({ step, order, originalIndex, metadata: callbackMetadata });
    trace.push({
      stepId: step.id,
      kind: step.kind,
      sequenceShard: step.sequenceShard,
      tensorShard: step.tensorShard,
      tokenStart: step.tokenStart,
      tokenEnd: step.tokenEnd,
      order,
      originalIndex,
      status: "ok",
      metadata: frozenMetadataSnapshot(options.metadata ?? {}),
    });
  }

  return trace;
}

function compareIndexedSteps(a: IndexedTSPScheduleStep, b: IndexedTSPScheduleStep): number {
  return a.step.sequenceShard - b.step.sequenceShard
    || a.step.tensorShard - b.step.tensorShard
    || a.originalIndex - b.originalIndex;
}

function frozenMetadataSnapshot(metadata: Record<string, unknown>): Record<string, unknown> {
  return deepFreeze(cloneMetadataValue(metadata)) as Record<string, unknown>;
}

function cloneMetadataValue(value: unknown): unknown {
  if (Array.isArray(value)) return value.map(cloneMetadataValue);
  if (value && typeof value === "object") {
    const clone: Record<string, unknown> = {};
    for (const [key, child] of Object.entries(value)) {
      clone[key] = cloneMetadataValue(child);
    }
    return clone;
  }
  return value;
}

function deepFreeze<T>(value: T): T {
  if (!value || typeof value !== "object" || Object.isFrozen(value)) return value;
  Object.freeze(value);
  for (const child of Object.values(value)) {
    deepFreeze(child);
  }
  return value;
}
