export type UnlockedKernelBackend = "webgpu" | "cpu_reference" | "mixed" | "unknown";
export type StrictUnlockedWebGpuGate = "mlp" | "logits" | "attention" | "projection";

export const STRICT_UNLOCKED_WEBGPU_GATES: StrictUnlockedWebGpuGate[] = [
  "mlp",
  "logits",
  "attention",
  "projection",
];

export interface UnlockedWebGpuCoverageGateResult {
  gates: StrictUnlockedWebGpuGate[];
  passed: boolean;
  failedReasons: string[];
}

export interface UnlockedCoverageDecodeProof {
  expectedLayerCount?: number;
  executedLayerCount?: number;
  mlpKernelBackends?: Array<{
    layerIndex: number;
    backend?: UnlockedKernelBackend;
    activationKind?: string;
  }>;
  prefillMlpKernelBackends?: Array<{
    layerIndex: number;
    backend?: UnlockedKernelBackend;
    activationKind?: string;
    rowCount?: number;
  }>;
  logitProjectionBackend?: UnlockedKernelBackend;
  logitProjectionPurpose?: string;
  logitProjectionSelectedRows?: number;
  logitProjectionFullRows?: number;
  prefillProjectionBackends?: Array<{
    layerIndex: number;
    qProjection?: UnlockedKernelBackend;
    kProjection?: UnlockedKernelBackend;
    vProjection?: UnlockedKernelBackend;
    oProjection?: UnlockedKernelBackend;
  }>;
  decodeProjectionBackends?: Array<{
    layerIndex: number;
    qProjection?: UnlockedKernelBackend;
    kProjection?: UnlockedKernelBackend;
    vProjection?: UnlockedKernelBackend;
    oProjection?: UnlockedKernelBackend;
    projectionKind?: string;
    tokens?: number;
    selectedRows?: number;
  }>;
  prefillAttentionBackends?: Array<{
    layerIndex: number;
    attentionBackend?: UnlockedKernelBackend;
    packedHeadBackends?: UnlockedKernelBackend[];
    packedHeadCount?: number;
  }>;
  decodeAttentionBackends?: Array<{
    layerIndex: number;
    attentionBackend?: UnlockedKernelBackend;
    packedHeadBackends?: UnlockedKernelBackend[];
    packedHeadCount?: number;
  }>;
}

export interface UnlockedWebGpuCoverageSummary {
  expectedLayerCount?: number;
  executedLayerCount?: number;
  mlpLayersByBackend: Record<UnlockedKernelBackend, number>;
  prefillMlpLayers: Array<{ layerIndex: number; backend: UnlockedKernelBackend; activationKind: string }>;
  decodeMlpLayers: Array<{ layerIndex: number; backend: UnlockedKernelBackend; activationKind: string }>;
  mlpLayers: Array<{ layerIndex: number; backend: UnlockedKernelBackend; activationKind: string }>;
  logitProjection: {
    backend: UnlockedKernelBackend;
    purpose: string;
    selectedRows: number | null;
    fullRows: number | null;
  };
  prefillProjectionBackends: {
    qProjection: Record<UnlockedKernelBackend, number>;
    kProjection: Record<UnlockedKernelBackend, number>;
    vProjection: Record<UnlockedKernelBackend, number>;
    oProjection: Record<UnlockedKernelBackend, number>;
    layers: Array<{
      layerIndex: number;
      qProjection: UnlockedKernelBackend;
      kProjection: UnlockedKernelBackend;
      vProjection: UnlockedKernelBackend;
      oProjection: UnlockedKernelBackend;
    }>;
  };
  decodeProjectionBackends: {
    qProjection: Record<UnlockedKernelBackend, number>;
    kProjection: Record<UnlockedKernelBackend, number>;
    vProjection: Record<UnlockedKernelBackend, number>;
    oProjection: Record<UnlockedKernelBackend, number>;
    layers: Array<{
      layerIndex: number;
      qProjection: UnlockedKernelBackend;
      kProjection: UnlockedKernelBackend;
      vProjection: UnlockedKernelBackend;
      oProjection: UnlockedKernelBackend;
      projectionKind: string;
      tokens: number | null;
      selectedRows: number | null;
    }>;
  };
  attentionBackends: {
    prefill: Record<UnlockedKernelBackend, number>;
    decode: Record<UnlockedKernelBackend, number>;
    packedHeads: Record<UnlockedKernelBackend, number>;
    prefillLayers: Array<{
      layerIndex: number;
      attentionBackend: UnlockedKernelBackend;
      packedHeadBackends: UnlockedKernelBackend[];
      packedHeadCount: number | null;
      packedHeadComplete: boolean;
    }>;
    decodeLayers: Array<{
      layerIndex: number;
      attentionBackend: UnlockedKernelBackend;
      packedHeadBackends: UnlockedKernelBackend[];
      packedHeadCount: number | null;
      packedHeadComplete: boolean;
    }>;
    incompletePackedHeadProofs: number;
  };
  cpuFallbackUsed: boolean;
}

export function summarizeUnlockedWebGpuCoverage(proof: UnlockedCoverageDecodeProof): UnlockedWebGpuCoverageSummary {
  const prefillMlpLayers = (proof.prefillMlpKernelBackends ?? []).map((layer) => ({
    layerIndex: layer.layerIndex,
    backend: normalizeBackend(layer.backend),
    activationKind: layer.activationKind ?? "unknown",
  }));
  const decodeMlpLayers = (proof.mlpKernelBackends ?? []).map((layer) => ({
    layerIndex: layer.layerIndex,
    backend: normalizeBackend(layer.backend),
    activationKind: layer.activationKind ?? "unknown",
  }));
  const mlpLayers = [
    ...prefillMlpLayers,
    ...decodeMlpLayers,
  ];
  const prefillProjectionLayers = (proof.prefillProjectionBackends ?? []).map((layer) => ({
    layerIndex: layer.layerIndex,
    qProjection: normalizeBackend(layer.qProjection),
    kProjection: normalizeBackend(layer.kProjection),
    vProjection: normalizeBackend(layer.vProjection),
    oProjection: normalizeBackend(layer.oProjection),
  }));
  const decodeProjectionLayers = (proof.decodeProjectionBackends ?? []).map((layer) => ({
    layerIndex: layer.layerIndex,
    qProjection: normalizeBackend(layer.qProjection),
    kProjection: normalizeBackend(layer.kProjection),
    vProjection: normalizeBackend(layer.vProjection),
    oProjection: normalizeBackend(layer.oProjection),
    projectionKind: layer.projectionKind ?? "unknown",
    tokens: normalizePositiveCount(layer.tokens),
    selectedRows: normalizePositiveCount(layer.selectedRows),
  }));
  const prefillLayers = (proof.prefillAttentionBackends ?? []).map((layer) => ({
    layerIndex: layer.layerIndex,
    attentionBackend: normalizeBackend(layer.attentionBackend),
    packedHeadBackends: (layer.packedHeadBackends ?? []).map(normalizeBackend),
    packedHeadCount: normalizePackedHeadCount(layer.packedHeadCount),
    packedHeadComplete: isPackedHeadProofComplete(layer.packedHeadBackends ?? [], layer.packedHeadCount),
  }));
  const decodeLayers = (proof.decodeAttentionBackends ?? []).map((layer) => ({
    layerIndex: layer.layerIndex,
    attentionBackend: normalizeBackend(layer.attentionBackend),
    packedHeadBackends: (layer.packedHeadBackends ?? []).map(normalizeBackend),
    packedHeadCount: normalizePackedHeadCount(layer.packedHeadCount),
    packedHeadComplete: isPackedHeadProofComplete(layer.packedHeadBackends ?? [], layer.packedHeadCount),
  }));
  const expectedLayerCount = normalizePositiveInteger(proof.expectedLayerCount);
  const executedLayerCount = normalizeNonNegativeInteger(proof.executedLayerCount)
    ?? countUniqueLayerIndexes([
      ...prefillMlpLayers,
      ...decodeMlpLayers,
      ...prefillProjectionLayers,
      ...decodeProjectionLayers,
      ...prefillLayers,
      ...decodeLayers,
    ]);
  const summary: UnlockedWebGpuCoverageSummary = {
    ...(expectedLayerCount !== null ? { expectedLayerCount } : {}),
    executedLayerCount,
    mlpLayersByBackend: countBackends(mlpLayers.map((layer) => layer.backend)),
    prefillMlpLayers,
    decodeMlpLayers,
    mlpLayers,
    logitProjection: {
      backend: normalizeBackend(proof.logitProjectionBackend),
      purpose: proof.logitProjectionPurpose ?? "unknown",
      selectedRows: proof.logitProjectionSelectedRows ?? null,
      fullRows: proof.logitProjectionFullRows ?? null,
    },
    prefillProjectionBackends: {
      qProjection: countBackends(prefillProjectionLayers.map((layer) => layer.qProjection)),
      kProjection: countBackends(prefillProjectionLayers.map((layer) => layer.kProjection)),
      vProjection: countBackends(prefillProjectionLayers.map((layer) => layer.vProjection)),
      oProjection: countBackends(prefillProjectionLayers.map((layer) => layer.oProjection)),
      layers: prefillProjectionLayers,
    },
    decodeProjectionBackends: {
      qProjection: countBackends(decodeProjectionLayers.map((layer) => layer.qProjection)),
      kProjection: countBackends(decodeProjectionLayers.map((layer) => layer.kProjection)),
      vProjection: countBackends(decodeProjectionLayers.map((layer) => layer.vProjection)),
      oProjection: countBackends(decodeProjectionLayers.map((layer) => layer.oProjection)),
      layers: decodeProjectionLayers,
    },
    attentionBackends: {
      prefill: countBackends(prefillLayers.map((layer) => layer.attentionBackend)),
      decode: countBackends(decodeLayers.map((layer) => layer.attentionBackend)),
      packedHeads: countBackends([
        ...prefillLayers.flatMap((layer) => layer.packedHeadBackends),
        ...decodeLayers.flatMap((layer) => layer.packedHeadBackends),
      ]),
      prefillLayers,
      decodeLayers,
      incompletePackedHeadProofs: [...prefillLayers, ...decodeLayers].filter((layer) => !layer.packedHeadComplete).length,
    },
    cpuFallbackUsed: false,
  };
  summary.cpuFallbackUsed = coverageUsesCpuFallback(summary);
  return summary;
}

export function assertUnlockedWebGpuCoverageGates(
  summary: UnlockedWebGpuCoverageSummary,
  gates: Iterable<StrictUnlockedWebGpuGate>,
): void {
  const result = evaluateUnlockedWebGpuCoverageGates(summary, gates);
  if (result.failedReasons.length > 0) {
    throw new Error(`Unlocked WebGPU strict gate failed: ${result.failedReasons.join("; ")}.`);
  }
}

export function evaluateUnlockedWebGpuCoverageGates(
  summary: UnlockedWebGpuCoverageSummary,
  gates: Iterable<StrictUnlockedWebGpuGate>,
): UnlockedWebGpuCoverageGateResult {
  const requestedGates = [...new Set(gates)];
  const failures: string[] = [];
  for (const gate of requestedGates) {
    if (gate === "mlp") {
      const prefillMlpLayers = summary.prefillMlpLayers ?? [];
      const decodeMlpLayers = summary.decodeMlpLayers ?? [];
      failures.push(...evaluateExpectedLayerCoverage("MLP", "prefill", summary.expectedLayerCount, prefillMlpLayers));
      failures.push(...evaluateExpectedLayerCoverage("MLP", "decode", summary.expectedLayerCount, decodeMlpLayers));
      if (prefillMlpLayers.length === 0) {
        failures.push("prefill MLP proof is missing");
      }
      if (decodeMlpLayers.length === 0) {
        failures.push("decode MLP proof is missing");
      }
      if (!allPresentBackendsAreWebGpu([
        ...prefillMlpLayers.map((layer) => layer.backend),
        ...decodeMlpLayers.map((layer) => layer.backend),
      ])) {
        failures.push("MLP layers are not fully WebGPU-backed");
      }
    }
    if (gate === "logits") {
      failures.push(...evaluateStrictLogitProjection(summary.logitProjection));
    }
    if (gate === "projection") {
      const prefillProjectionLayers = summary.prefillProjectionBackends.layers;
      const decodeProjectionLayers = summary.decodeProjectionBackends.layers;
      const projectionBackends = [
        ...prefillProjectionLayers.flatMap((layer) => [
          layer.qProjection,
          layer.kProjection,
          layer.vProjection,
          layer.oProjection,
        ]),
        ...decodeProjectionLayers.flatMap((layer) => [
          layer.qProjection,
          layer.kProjection,
          layer.vProjection,
          layer.oProjection,
        ]),
      ];
      failures.push(...evaluateProjectionComponentCoverage("prefill Q/K/V/O", summary.expectedLayerCount, prefillProjectionLayers));
      failures.push(...evaluateProjectionComponentCoverage("decode Q/K/V/O", summary.expectedLayerCount, decodeProjectionLayers));
      if (prefillProjectionLayers.length === 0) {
        failures.push("prefill Q/K/V/O projection proof is missing");
      }
      if (decodeProjectionLayers.length === 0) {
        failures.push("decode Q/K/V/O projection proof is missing");
      }
      if (!allPresentBackendsAreWebGpu(projectionBackends)) {
        failures.push("dense projection kernels are not fully WebGPU-backed");
      }
    }
    if (gate === "attention") {
      const prefillAttentionLayers = summary.attentionBackends.prefillLayers;
      const decodeAttentionLayers = summary.attentionBackends.decodeLayers;
      const attentionBackends = [
        ...prefillAttentionLayers.map((layer) => layer.attentionBackend),
        ...decodeAttentionLayers.map((layer) => layer.attentionBackend),
        ...prefillAttentionLayers.flatMap((layer) => layer.packedHeadBackends),
        ...decodeAttentionLayers.flatMap((layer) => layer.packedHeadBackends),
      ];
      const incompletePackedHeads = summary.attentionBackends.incompletePackedHeadProofs > 0;
      failures.push(...evaluateExpectedLayerCoverage("attention", "prefill", summary.expectedLayerCount, prefillAttentionLayers));
      failures.push(...evaluateExpectedLayerCoverage("attention", "decode", summary.expectedLayerCount, decodeAttentionLayers));
      if (prefillAttentionLayers.length === 0) {
        failures.push("prefill attention proof is missing");
      }
      if (decodeAttentionLayers.length === 0) {
        failures.push("decode attention proof is missing");
      }
      if (!allPresentBackendsAreWebGpu(attentionBackends)) {
        failures.push("attention kernels are not fully WebGPU-backed");
      }
      if (incompletePackedHeads) {
        failures.push("attention packed-head proofs are incomplete");
      }
    }
  }
  return {
    gates: requestedGates,
    passed: failures.length === 0,
    failedReasons: failures,
  };
}

function evaluateExpectedLayerCoverage(
  family: "MLP" | "projection" | "attention",
  phase: string,
  expectedLayerCount: number | undefined,
  layers: Array<{ layerIndex: number }>,
): string[] {
  if (!isPositiveLayerCount(expectedLayerCount)) {
    return [`${family} expected layer count is missing`];
  }

  const covered = new Set<number>();
  const unexpected: number[] = [];
  for (const layer of layers) {
    if (!Number.isInteger(layer.layerIndex) || layer.layerIndex < 0 || layer.layerIndex >= expectedLayerCount) {
      unexpected.push(layer.layerIndex);
      continue;
    }
    covered.add(layer.layerIndex);
  }

  const missing: number[] = [];
  for (let layerIndex = 0; layerIndex < expectedLayerCount; layerIndex += 1) {
    if (!covered.has(layerIndex)) missing.push(layerIndex);
  }

  if (missing.length === 0 && unexpected.length === 0) return [];
  return [
    `${family} layer coverage is incomplete for ${phase} (expected ${expectedLayerCount} layers; covered ${covered.size}; missing ${formatLayerIndexes(missing)}; unexpected ${formatLayerIndexes(unexpected)})`,
  ];
}

function evaluateProjectionComponentCoverage(
  phase: string,
  expectedLayerCount: number | undefined,
  layers: Array<{
    layerIndex: number;
    qProjection: UnlockedKernelBackend;
    kProjection: UnlockedKernelBackend;
    vProjection: UnlockedKernelBackend;
    oProjection: UnlockedKernelBackend;
  }>,
): string[] {
  if (!isPositiveLayerCount(expectedLayerCount)) {
    return ["projection expected layer count is missing"];
  }

  const componentCoverage = {
    qProjection: new Set<number>(),
    kProjection: new Set<number>(),
    vProjection: new Set<number>(),
    oProjection: new Set<number>(),
  };
  const coveredLayers = new Set<number>();
  const unexpected: number[] = [];
  for (const layer of layers) {
    if (!Number.isInteger(layer.layerIndex) || layer.layerIndex < 0 || layer.layerIndex >= expectedLayerCount) {
      unexpected.push(layer.layerIndex);
      continue;
    }
    coveredLayers.add(layer.layerIndex);
    if (layer.qProjection !== "unknown") componentCoverage.qProjection.add(layer.layerIndex);
    if (layer.kProjection !== "unknown") componentCoverage.kProjection.add(layer.layerIndex);
    if (layer.vProjection !== "unknown") componentCoverage.vProjection.add(layer.layerIndex);
    if (layer.oProjection !== "unknown") componentCoverage.oProjection.add(layer.layerIndex);
  }

  const componentMissing = {
    qProjection: missingLayerIndexes(expectedLayerCount, componentCoverage.qProjection),
    kProjection: missingLayerIndexes(expectedLayerCount, componentCoverage.kProjection),
    vProjection: missingLayerIndexes(expectedLayerCount, componentCoverage.vProjection),
    oProjection: missingLayerIndexes(expectedLayerCount, componentCoverage.oProjection),
  };
  const hasMissing = Object.values(componentMissing).some((missing) => missing.length > 0);
  if (!hasMissing && unexpected.length === 0) return [];

  return [
    `projection layer coverage is incomplete for ${phase} (expected ${expectedLayerCount} layers; covered ${coveredLayers.size}; q missing ${formatLayerIndexes(componentMissing.qProjection)}; k missing ${formatLayerIndexes(componentMissing.kProjection)}; v missing ${formatLayerIndexes(componentMissing.vProjection)}; o missing ${formatLayerIndexes(componentMissing.oProjection)}; unexpected ${formatLayerIndexes(unexpected)})`,
  ];
}

function missingLayerIndexes(expectedLayerCount: number, covered: Set<number>): number[] {
  const missing: number[] = [];
  for (let layerIndex = 0; layerIndex < expectedLayerCount; layerIndex += 1) {
    if (!covered.has(layerIndex)) missing.push(layerIndex);
  }
  return missing;
}

function isPositiveLayerCount(value: number | undefined): value is number {
  return Number.isInteger(value) && value !== undefined && value > 0;
}

function allPresentBackendsAreWebGpu(backends: UnlockedKernelBackend[]): boolean {
  return backends.every((backend) => backend === "webgpu");
}

function evaluateStrictLogitProjection(logitProjection: UnlockedWebGpuCoverageSummary["logitProjection"]): string[] {
  if (logitProjection.backend !== "webgpu") {
    return [`logit projection backend is ${logitProjection.backend}`];
  }
  if (
    logitProjection.purpose !== "full_vocab_logit_projection"
    && logitProjection.purpose !== "full_vocab_topk_logit_projection"
    && logitProjection.purpose !== "greedy_argmax_logit_projection"
    && logitProjection.purpose !== "compact_topk_logit_projection"
  ) {
    return [
      `logit projection purpose is ${logitProjection.purpose}; strict logits require full-vocab production proof`,
    ];
  }
  if (!isPositiveInteger(logitProjection.selectedRows) || !isPositiveInteger(logitProjection.fullRows)) {
    return ["logit projection row coverage is missing or non-positive"];
  }
  if (logitProjection.selectedRows > logitProjection.fullRows) {
    return [
      `logit projection row coverage is invalid (selectedRows ${logitProjection.selectedRows}, fullRows ${logitProjection.fullRows})`,
    ];
  }
  if (
    logitProjection.purpose === "full_vocab_logit_projection"
    && logitProjection.selectedRows !== logitProjection.fullRows
  ) {
    return [
      `full-vocab logit projection rows are incomplete (selectedRows ${logitProjection.selectedRows}, fullRows ${logitProjection.fullRows})`,
    ];
  }
  return [];
}

function isPositiveInteger(value: number | null): value is number {
  return Number.isInteger(value) && value !== null && value > 0;
}

function coverageUsesCpuFallback(summary: UnlockedWebGpuCoverageSummary): boolean {
  const backends = [
    ...summary.mlpLayers.map((layer) => layer.backend),
    summary.logitProjection.backend,
    ...summary.prefillProjectionBackends.layers.flatMap((layer) => [
      layer.qProjection,
      layer.kProjection,
      layer.vProjection,
      layer.oProjection,
    ]),
    ...summary.decodeProjectionBackends.layers.flatMap((layer) => [
      layer.qProjection,
      layer.kProjection,
      layer.vProjection,
      layer.oProjection,
    ]),
    ...summary.attentionBackends.prefillLayers.map((layer) => layer.attentionBackend),
    ...summary.attentionBackends.decodeLayers.map((layer) => layer.attentionBackend),
    ...summary.attentionBackends.prefillLayers.flatMap((layer) => layer.packedHeadBackends),
    ...summary.attentionBackends.decodeLayers.flatMap((layer) => layer.packedHeadBackends),
  ];
  return summary.attentionBackends.incompletePackedHeadProofs > 0
    || backends.some((backend) => backend === "cpu_reference" || backend === "mixed" || backend === "unknown");
}

function countBackends(backends: UnlockedKernelBackend[]): Record<UnlockedKernelBackend, number> {
  return {
    webgpu: backends.filter((backend) => backend === "webgpu").length,
    cpu_reference: backends.filter((backend) => backend === "cpu_reference").length,
    mixed: backends.filter((backend) => backend === "mixed").length,
    unknown: backends.filter((backend) => backend === "unknown").length,
  };
}

function normalizeBackend(backend: UnlockedKernelBackend | string | undefined): UnlockedKernelBackend {
  if (backend === "webgpu" || backend === "cpu_reference" || backend === "mixed") return backend;
  return "unknown";
}

function normalizePackedHeadCount(value: number | undefined): number | null {
  if (value === undefined) return null;
  if (!Number.isInteger(value) || value <= 0) return null;
  return value;
}

function normalizePositiveInteger(value: number | undefined): number | null {
  if (value === undefined) return null;
  if (!Number.isInteger(value) || value <= 0) return null;
  return value;
}

function normalizeNonNegativeInteger(value: number | undefined): number | null {
  if (value === undefined) return null;
  if (!Number.isInteger(value) || value < 0) return null;
  return value;
}

function normalizePositiveCount(value: number | undefined): number | null {
  if (value === undefined) return null;
  if (!Number.isInteger(value) || value < 0) return null;
  return value;
}

function countUniqueLayerIndexes(layers: Array<{ layerIndex: number }>): number {
  return new Set(layers.filter((layer) => Number.isInteger(layer.layerIndex) && layer.layerIndex >= 0).map((layer) => layer.layerIndex)).size;
}

function isPackedHeadProofComplete(backends: unknown[] | undefined, expectedCount: number | undefined): boolean {
  if (expectedCount === undefined) return false;
  if (!Number.isInteger(expectedCount) || expectedCount <= 0) return false;
  const observedCount = (backends ?? []).length;
  return observedCount >= expectedCount && observedCount % expectedCount === 0;
}

function formatLayerIndexes(values: number[]): string {
  return values.length > 0 ? `[${values.join(",")}]` : "[]";
}
