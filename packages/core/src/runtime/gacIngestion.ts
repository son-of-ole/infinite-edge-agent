import { cosineSimilarity } from "../vector";
import type {
  ClusterMetricRecord,
  ConsolidationRunRecord,
  GacMemoryKind,
  GacMemoryStore,
  GacRawMemorySourceType,
  GacRetentionClass,
  IdentityPinReason,
  IdentityPinRecord,
  MemoryChunk,
  MemoryClusterRecord,
  MemoryContradictionRecord,
  MemoryLineageRecord,
  MemoryRepresentativeRecord,
  RawMemoryRecord,
  RetrievalAuditRecord,
  SourceDocumentTrustLevel,
  SourceDocumentMemoryWritePolicy,
  TrainingExampleRecord,
} from "../types";

export interface ImmediateGacIngestionInput {
  tenantId: string;
  cellId: string;
  sessionId: string;
  chunks: MemoryChunk[];
  sourceType?: GacRawMemorySourceType;
  sourceUri?: string;
  sourceTrust?: SourceDocumentTrustLevel;
  allowExternalPins?: boolean;
  now?: Date;
}

export interface ImmediateGacIngestionPlan {
  chunks: MemoryChunk[];
  rawMemory: RawMemoryRecord[];
  identityPins: IdentityPinRecord[];
  clusters: MemoryClusterRecord[];
  clusterMetrics: ClusterMetricRecord[];
  representatives: MemoryRepresentativeRecord[];
  lineage: MemoryLineageRecord[];
  consolidationRuns: ConsolidationRunRecord[];
  contradictions: MemoryContradictionRecord[];
  trainingExamples: TrainingExampleRecord[];
}

export interface AdaptiveConsolidationJobPlanInput {
  tenantId: string;
  cellId: string;
  sessionId?: string;
  rawMemory: RawMemoryRecord[];
  identityPins?: IdentityPinRecord[];
  retrievalAudits?: RetrievalAuditRecord[];
  minCandidateCount?: number;
  mode?: ConsolidationRunRecord["mode"];
  now?: Date;
}

export interface AdaptiveConsolidationJobPlan {
  shouldRun: boolean;
  consolidationRun: ConsolidationRunRecord | null;
  candidateRawMemoryIds: string[];
  protectedRawMemoryIds: string[];
  reasonCodes: string[];
}

export interface TrainingExampleBuildInput {
  tenantId: string;
  cellId: string;
  rawMemory: RawMemoryRecord[];
  exportMode?: "local_private" | "shared";
  now?: Date;
}

export function buildImmediateGacIngestionPlan(input: ImmediateGacIngestionInput): ImmediateGacIngestionPlan {
  const now = input.now ?? new Date();
  const timestamp = now.toISOString();
  const sourceType = input.sourceType ?? toRawSourceType(input.chunks[0]?.source);
  const trust = input.sourceTrust ?? (sourceType === "external" ? "untrusted" : "trusted");
  const memoryWritePolicy = getMemoryWritePolicy(sourceType, trust, input.allowExternalPins === true);
  const rawMemory = input.chunks.map((chunk) => {
    const policy = classifyMemoryPolicy(chunk.text, {
      sourceType,
      sourceTrust: trust,
      memoryWritePolicy,
    });
    return toRawMemoryRecord(chunk, input, sourceType, input.sourceUri, policy, timestamp);
  });

  const identityPins = rawMemory
    .map((record) => toIdentityPin(record, classifyMemoryPolicy(record.text, {
      sourceType,
      sourceTrust: trust,
      memoryWritePolicy,
    }), timestamp))
    .filter((record): record is IdentityPinRecord => Boolean(record));

  const pinByRawId = new Map(identityPins.map((pin) => [pin.rawMemoryId, pin]));
  const cluster = buildClusterRecord(input, rawMemory, timestamp);
  const contradictions = findMemoryContradictions({
    tenantId: input.tenantId,
    cellId: input.cellId,
    sessionId: input.sessionId,
    rawMemory,
    now,
  });
  const clusterMetric = buildClusterMetric(input, cluster, rawMemory, contradictions, timestamp);
  const runId = `run_${stableHash(`${cluster.id}:${timestamp}`)}`;
  const representatives = buildRepresentatives(rawMemory, input.chunks, pinByRawId, cluster, runId, timestamp);
  const lineage = buildLineage(representatives, rawMemory, input.chunks, timestamp);
  const consolidationRun: ConsolidationRunRecord = {
    id: runId,
    tenantId: input.tenantId,
    cellId: input.cellId,
    sessionId: input.sessionId,
    mode: "immediate",
    inputCount: rawMemory.length,
    clusterCount: rawMemory.length > 0 ? 1 : 0,
    representativeCount: representatives.length,
    pinCount: identityPins.length,
    status: "complete",
    startedAt: timestamp,
    completedAt: timestamp,
    configHash: `gac_ingestion_${stableHash("immediate:v1")}`,
  };
  const chunks = input.chunks.map((chunk) => withGacMetadata(chunk, {
    raw: rawMemory.find((record) => record.id === chunk.id),
    pin: pinByRawId.get(chunk.id),
    policy: classifyMemoryPolicy(chunk.text, {
      sourceType,
      sourceTrust: trust,
      memoryWritePolicy,
    }),
    memoryWritePolicy,
    trust,
  }));
  const trainingExamples = buildTrainingExamplesFromRawMemory({
    tenantId: input.tenantId,
    cellId: input.cellId,
    rawMemory,
    exportMode: "shared",
    now,
  });

  return {
    chunks,
    rawMemory,
    identityPins,
    clusters: rawMemory.length > 0 ? [cluster] : [],
    clusterMetrics: rawMemory.length > 0 ? [clusterMetric] : [],
    representatives,
    lineage,
    consolidationRuns: rawMemory.length > 0 ? [consolidationRun] : [],
    contradictions,
    trainingExamples,
  };
}

export async function writeImmediateGacIngestionPlan(
  store: Partial<GacMemoryStore>,
  plan: ImmediateGacIngestionPlan,
): Promise<void> {
  if (typeof store.writeRawMemory === "function") await store.writeRawMemory(plan.rawMemory);
  if (typeof store.writeIdentityPins === "function") await store.writeIdentityPins(plan.identityPins);
  if (typeof store.writeMemoryClusters === "function") await store.writeMemoryClusters(plan.clusters);
  if (typeof store.writeClusterMetrics === "function") await store.writeClusterMetrics(plan.clusterMetrics);
  if (typeof store.writeMemoryRepresentatives === "function") {
    await store.writeMemoryRepresentatives(plan.representatives, plan.lineage);
  } else if (typeof store.writeMemoryLineage === "function") {
    await store.writeMemoryLineage(plan.lineage);
  }
  if (typeof store.writeConsolidationRuns === "function") await store.writeConsolidationRuns(plan.consolidationRuns);
  if (typeof store.writeMemoryContradictions === "function") await store.writeMemoryContradictions(plan.contradictions);
  if (typeof store.writeTrainingExamples === "function") await store.writeTrainingExamples(plan.trainingExamples);
}

export function buildAdaptiveConsolidationJobPlan(
  input: AdaptiveConsolidationJobPlanInput,
): AdaptiveConsolidationJobPlan {
  const now = input.now ?? new Date();
  const timestamp = now.toISOString();
  const minCandidateCount = input.minCandidateCount ?? 2;
  const reasonCodes: string[] = [];
  const identityPinRawIds = new Set((input.identityPins ?? []).map((pin) => pin.rawMemoryId));
  const failedAuditRawIds = new Set((input.retrievalAudits ?? [])
    .filter((audit) => !audit.identityPreserved)
    .map((audit) => audit.expectedRawMemoryId));
  const protectedRawMemoryIds = unique([
    ...input.rawMemory
      .filter((record) => record.retentionClass === "pinned" || record.retentionClass === "legal" || record.retentionClass === "security")
      .map((record) => record.id),
    ...identityPinRawIds,
    ...failedAuditRawIds,
  ].filter((id) => input.rawMemory.some((record) => record.id === id)));

  if (identityPinRawIds.size > 0 || input.rawMemory.some((record) => record.retentionClass === "pinned")) {
    reasonCodes.push("identity_pin_protected");
  }
  if (failedAuditRawIds.size > 0) reasonCodes.push("retrieval_failure_protected");
  if (input.rawMemory.some((record) => record.retentionClass === "legal" || record.retentionClass === "security")) {
    reasonCodes.push("regulated_memory_protected");
  }

  const protectedSet = new Set(protectedRawMemoryIds);
  const candidateRawMemoryIds = input.rawMemory
    .filter((record) => !protectedSet.has(record.id))
    .filter((record) => record.deletedAt === undefined && record.retentionClass === "normal")
    .map((record) => record.id);
  const shouldRun = candidateRawMemoryIds.length >= minCandidateCount;
  reasonCodes.push(shouldRun ? "candidate_threshold_met" : "below_candidate_threshold");

  const consolidationRun: ConsolidationRunRecord | null = shouldRun
    ? {
        id: `run_${stableHash(`${input.tenantId}:${input.cellId}:${input.sessionId ?? ""}:adaptive:${candidateRawMemoryIds.join(":")}:${timestamp}`)}`,
        tenantId: input.tenantId,
        cellId: input.cellId,
        ...(input.sessionId ? { sessionId: input.sessionId } : {}),
        mode: input.mode ?? "sleep",
        inputCount: candidateRawMemoryIds.length,
        clusterCount: 1,
        representativeCount: 0,
        pinCount: input.identityPins?.length ?? 0,
        status: "running",
        startedAt: timestamp,
        configHash: `adaptive_consolidation_${stableHash(JSON.stringify({
          minCandidateCount,
          candidateRawMemoryIds,
          protectedRawMemoryIds,
          reasonCodes,
        }))}`,
      }
    : null;

  return {
    shouldRun,
    consolidationRun,
    candidateRawMemoryIds,
    protectedRawMemoryIds,
    reasonCodes: unique(reasonCodes),
  };
}

export function buildTrainingExamplesFromRawMemory(input: TrainingExampleBuildInput): TrainingExampleRecord[] {
  const now = input.now ?? new Date();
  return input.rawMemory.map((record) => {
    const privacyClass = classifyTrainingPrivacy(record);
    const exportAllowed = input.exportMode === "shared"
      ? privacyClass === "synthetic" || privacyClass === "consented"
      : privacyClass !== "private" || record.retentionClass === "normal";
    return {
      id: `training_${stableHash(`${record.id}:${record.updatedAt}:raw_memory_event`)}`,
      tenantId: input.tenantId,
      cellId: input.cellId,
      ...(record.sessionId ? { sessionId: record.sessionId } : {}),
      datasetType: "raw_memory_event",
      sourceRawMemoryIds: [record.id],
      inputJson: {
        text: record.text,
        sourceType: record.sourceType,
        sourceUri: record.sourceUri ?? "",
      },
      labelsJson: {
        memoryKind: record.memoryKind,
        importance: record.importance,
        identityRisk: record.identityRiskSeed,
        pinRecommended: record.retentionClass === "pinned",
      },
      privacyClass,
      exportAllowed,
      createdAt: now.toISOString(),
    };
  });
}

export function findMemoryContradictions(input: {
  tenantId: string;
  cellId: string;
  sessionId?: string;
  rawMemory: RawMemoryRecord[];
  now?: Date;
}): MemoryContradictionRecord[] {
  const contradictions: MemoryContradictionRecord[] = [];
  const timestamp = (input.now ?? new Date()).toISOString();
  for (let i = 0; i < input.rawMemory.length; i += 1) {
    for (let j = i + 1; j < input.rawMemory.length; j += 1) {
      const left = input.rawMemory[i]!;
      const right = input.rawMemory[j]!;
      if (!isNegationContradiction(left.text, right.text)) continue;
      contradictions.push({
        id: `contradiction_${stableHash(`${left.id}:${right.id}:negation`)}`,
        tenantId: input.tenantId,
        cellId: input.cellId,
        ...(input.sessionId ? { sessionId: input.sessionId } : {}),
        rawMemoryIds: [left.id, right.id],
        contradictionType: "negation",
        confidence: 0.82,
        status: "open",
        createdAt: timestamp,
      });
    }
  }
  return contradictions;
}

interface MemoryPolicyClassification {
  kind: GacMemoryKind;
  importance: number;
  identityRisk: number;
  retentionClass: GacRetentionClass;
  pinReason?: IdentityPinReason;
  pinStrength: number;
}

function classifyMemoryPolicy(text: string, options: {
  sourceType: GacRawMemorySourceType;
  sourceTrust: SourceDocumentTrustLevel;
  memoryWritePolicy: SourceDocumentMemoryWritePolicy;
}): MemoryPolicyClassification {
  const lower = text.toLocaleLowerCase();
  const explicitInstruction = /\b(remember|from now on|do not|don't|must|always|never|required)\b/i.test(text);
  const architectureDecision = /\b(architecture|adr|decision|runtime|lancedb|qwen|webgpu|production|deploy|schema|api contract)\b/i.test(text);
  const legalSecurity = /\b(legal|contract|attorney|security|credential|secret|token|api key|password)\b/i.test(text);
  const dateMoneyNameUrl = /https?:\/\/|\$[0-9]|(?:\b\d{4}-\d{2}-\d{2}\b)|(?:\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\b)|[A-Z][A-Za-z0-9_-]{2,}|\b[A-Za-z0-9_.-]+\/[A-Za-z0-9_.-]+\b/.test(text);
  const sourceOfTruth = options.sourceType === "file" || options.sourceType === "tool" || options.sourceTrust === "user_confirmed";
  const externalQuarantine = options.sourceType === "external" && options.memoryWritePolicy !== "allow_pins";

  let risk = 0.2;
  if (dateMoneyNameUrl) risk += 0.2;
  if (explicitInstruction) risk += 0.3;
  if (legalSecurity) risk += 0.3;
  if (architectureDecision) risk += 0.25;
  if (options.sourceType === "external") risk += 0.2;
  if (isLowInformation(text)) risk -= 0.25;
  risk = clamp01(risk);

  const kind: GacMemoryKind = externalQuarantine
    ? "observation"
    : explicitInstruction
      ? "instruction"
      : architectureDecision
        ? "decision"
        : sourceOfTruth
          ? "observation"
          : "fact";

  const pinReason = externalQuarantine
    ? undefined
    : explicitInstruction
      ? "user_instruction"
      : architectureDecision
        ? "architecture_decision"
        : legalSecurity
          ? lower.includes("credential") || lower.includes("token") || lower.includes("password") || lower.includes("api key")
            ? "credential_metadata"
            : lower.includes("security") || lower.includes("secret")
              ? "security"
              : "legal"
          : dateMoneyNameUrl
            ? "date_money_name_url"
            : sourceOfTruth
              ? "source_of_truth"
              : undefined;

  const pinStrength = pinReason ? (pinReason === "user_instruction" ? 1 : clamp01(Math.max(0.72, risk))) : 0;
  return {
    kind,
    importance: clamp01(Math.max(0.35, risk)),
    identityRisk: risk,
    retentionClass: pinReason ? "pinned" : "normal",
    ...(pinReason ? { pinReason } : {}),
    pinStrength,
  };
}

function toRawMemoryRecord(
  chunk: MemoryChunk,
  input: ImmediateGacIngestionInput,
  sourceType: GacRawMemorySourceType,
  sourceUri: string | undefined,
  policy: MemoryPolicyClassification,
  timestamp: string,
): RawMemoryRecord {
  return {
    id: chunk.id,
    tenantId: input.tenantId,
    cellId: input.cellId,
    sessionId: input.sessionId,
    sourceType,
    ...(sourceUri ? { sourceUri } : {}),
    text: chunk.text,
    canonicalText: canonicalizeText(chunk.text),
    memoryKind: policy.kind,
    importance: policy.importance,
    identityRiskSeed: policy.identityRisk,
    createdAt: chunk.createdAt || timestamp,
    updatedAt: timestamp,
    retentionClass: policy.retentionClass,
    hash: stableHash(`${input.tenantId}:${input.cellId}:${sourceType}:${sourceUri ?? ""}:${chunk.text}`),
  };
}

function toIdentityPin(
  record: RawMemoryRecord,
  policy: MemoryPolicyClassification,
  timestamp: string,
): IdentityPinRecord | null {
  if (!policy.pinReason) return null;
  return {
    id: `pin_${stableHash(`${record.id}:${policy.pinReason}`)}`,
    tenantId: record.tenantId,
    cellId: record.cellId,
    ...(record.sessionId ? { sessionId: record.sessionId } : {}),
    rawMemoryId: record.id,
    pinReason: policy.pinReason,
    pinStrength: policy.pinStrength,
    createdBy: "policy",
    createdAt: timestamp,
  };
}

function buildClusterRecord(input: ImmediateGacIngestionInput, rawMemory: RawMemoryRecord[], timestamp: string): MemoryClusterRecord {
  const rawMemoryIds = rawMemory.map((record) => record.id);
  return {
    id: `cluster_${stableHash(`${input.tenantId}:${input.cellId}:${input.sessionId}:${rawMemoryIds.join(":")}`)}`,
    tenantId: input.tenantId,
    cellId: input.cellId,
    sessionId: input.sessionId,
    clusterVersion: 1,
    algorithm: "local_radius",
    memberCount: rawMemory.length,
    status: "open",
    rawMemoryIds,
    createdAt: timestamp,
    updatedAt: timestamp,
  };
}

function buildClusterMetric(
  input: ImmediateGacIngestionInput,
  cluster: MemoryClusterRecord,
  rawMemory: RawMemoryRecord[],
  contradictions: MemoryContradictionRecord[],
  timestamp: string,
): ClusterMetricRecord {
  const distances = pairwiseDistances(input.chunks);
  const meanDistance = mean(distances);
  const maxDistance = distances.length > 0 ? Math.max(...distances) : 0;
  const medianDistance = median(distances);
  const riskMean = mean(rawMemory.map((record) => record.identityRiskSeed));
  const theta = 0.85;
  return {
    id: `metric_${stableHash(`${cluster.id}:${cluster.clusterVersion}:${timestamp}`)}`,
    tenantId: input.tenantId,
    cellId: input.cellId,
    sessionId: input.sessionId,
    clusterId: cluster.id,
    clusterVersion: cluster.clusterVersion,
    meanDistance,
    maxDistance,
    medianDistance,
    effectiveDimension: Math.max(1, Math.min(input.chunks.length, 1 + maxDistance * Math.max(1, input.chunks.length - 1))),
    rho: clamp01(1 - meanDistance),
    theta,
    thetaPrime: Number(Math.acos(theta).toFixed(6)),
    identityErrorBound: clamp01(Math.max(riskMean, maxDistance)),
    densityScore: clamp01(1 - meanDistance),
    contradictionScore: contradictions.length > 0 ? 1 : 0,
    computedAt: timestamp,
  };
}

function buildRepresentatives(
  rawMemory: RawMemoryRecord[],
  chunks: MemoryChunk[],
  pinByRawId: Map<string, IdentityPinRecord>,
  cluster: MemoryClusterRecord,
  runId: string,
  timestamp: string,
): MemoryRepresentativeRecord[] {
  if (rawMemory.length === 0) return [];
  const medoid = selectMedoid(rawMemory, chunks);
  const representatives: MemoryRepresentativeRecord[] = [{
    id: `rep_${stableHash(`${cluster.id}:medoid:${medoid.id}`)}`,
    tenantId: medoid.tenantId,
    cellId: medoid.cellId,
    ...(medoid.sessionId ? { sessionId: medoid.sessionId } : {}),
    clusterId: cluster.id,
    clusterVersion: cluster.clusterVersion,
    type: "medoid",
    embedding: chunks.find((chunk) => chunk.id === medoid.id)?.embedding ?? [],
    text: medoid.text,
    sourceRawMemoryId: medoid.id,
    riskScore: medoid.identityRiskSeed,
    coverageScore: 1,
    createdByRunId: runId,
    createdAt: timestamp,
    modelVisible: false,
    factual: false,
  }];

  for (const record of rawMemory) {
    const pin = pinByRawId.get(record.id);
    if (!pin) continue;
    representatives.push({
      id: `rep_${stableHash(`${cluster.id}:pin_shadow:${record.id}:${pin.id}`)}`,
      tenantId: record.tenantId,
      cellId: record.cellId,
      ...(record.sessionId ? { sessionId: record.sessionId } : {}),
      clusterId: cluster.id,
      clusterVersion: cluster.clusterVersion,
      type: "pin_shadow",
      embedding: chunks.find((chunk) => chunk.id === record.id)?.embedding ?? [],
      text: record.text,
      sourceRawMemoryId: record.id,
      riskScore: record.identityRiskSeed,
      coverageScore: 1,
      createdByRunId: runId,
      createdAt: timestamp,
      modelVisible: true,
      factual: true,
    });
  }

  return representatives;
}

function buildLineage(
  representatives: MemoryRepresentativeRecord[],
  rawMemory: RawMemoryRecord[],
  chunks: MemoryChunk[],
  timestamp: string,
): MemoryLineageRecord[] {
  const chunkById = new Map(chunks.map((chunk) => [chunk.id, chunk]));
  return representatives.flatMap((representative) => {
    if (representative.type === "pin_shadow" && representative.sourceRawMemoryId) {
      return [lineageRecord(representative, representative.sourceRawMemoryId, 1, 0, true, timestamp)];
    }
    const repChunk = representative.sourceRawMemoryId ? chunkById.get(representative.sourceRawMemoryId) : undefined;
    return rawMemory.map((record) => {
      const chunk = chunkById.get(record.id);
      const distance = repChunk && chunk ? cosineDistance(repChunk.embedding, chunk.embedding) : 0;
      return lineageRecord(
        representative,
        record.id,
        representative.sourceRawMemoryId === record.id ? 1 : clamp01(1 - distance),
        distance,
        representative.sourceRawMemoryId === record.id,
        timestamp,
      );
    });
  });
}

function lineageRecord(
  representative: MemoryRepresentativeRecord,
  rawMemoryId: string,
  membershipWeight: number,
  distanceToRep: number,
  isPrimary: boolean,
  timestamp: string,
): MemoryLineageRecord {
  return {
    representativeId: representative.id,
    rawMemoryId,
    tenantId: representative.tenantId,
    cellId: representative.cellId,
    ...(representative.sessionId ? { sessionId: representative.sessionId } : {}),
    membershipWeight,
    distanceToRep,
    isPrimary,
    createdAt: timestamp,
  };
}

function withGacMetadata(chunk: MemoryChunk, input: {
  raw: RawMemoryRecord | undefined;
  pin: IdentityPinRecord | undefined;
  policy: MemoryPolicyClassification;
  memoryWritePolicy: SourceDocumentMemoryWritePolicy;
  trust: SourceDocumentTrustLevel;
}): MemoryChunk {
  if (!input.raw) return chunk;
  const sourceTrust = input.trust === "trusted" ? 1 : input.trust === "user_confirmed" ? 0.85 : 0.2;
  return {
    ...chunk,
    metadata: {
      ...chunk.metadata,
      gac: {
        rawMemoryId: input.raw.id,
        rawMemoryIds: [input.raw.id],
        memoryClass: input.pin ? "PINNED_EXACT" : input.policy.identityRisk >= 0.55 ? "HIGH_RISK_RAW" : "RECENT_SESSION",
        identityRisk: input.policy.identityRisk,
        sourceTrust,
        memoryWritePolicy: input.memoryWritePolicy,
        mustAttend: Boolean(input.pin),
        ...(input.pin ? {
          identityPinId: input.pin.id,
          pinStrength: input.pin.pinStrength,
        } : {}),
      },
    },
  };
}

function toRawSourceType(source: MemoryChunk["source"] | undefined): GacRawMemorySourceType {
  if (source === "document") return "file";
  if (source === "summary") return "reflection";
  if (source === "tool") return "tool";
  if (source === "system") return "system";
  return "chat";
}

function getMemoryWritePolicy(
  sourceType: GacRawMemorySourceType,
  trust: SourceDocumentTrustLevel,
  allowExternalPins: boolean,
): SourceDocumentMemoryWritePolicy {
  if (sourceType !== "external") return "allow_pins";
  if (allowExternalPins || trust === "user_confirmed") return "allow_pins";
  if (trust === "trusted") return "allow_raw";
  return "quarantine";
}

function classifyTrainingPrivacy(record: RawMemoryRecord): TrainingExampleRecord["privacyClass"] {
  if (record.sourceUri?.startsWith("synthetic://")) return "synthetic";
  if (record.sourceUri?.startsWith("consented://")) return "consented";
  if (record.sourceUri?.startsWith("internal-eval://")) return "internal_eval";
  if (record.retentionClass === "security" || record.retentionClass === "legal" || record.retentionClass === "pinned") return "private";
  return "private";
}

function selectMedoid(rawMemory: RawMemoryRecord[], chunks: MemoryChunk[]): RawMemoryRecord {
  if (rawMemory.length === 1) return rawMemory[0]!;
  let best = rawMemory[0]!;
  let bestDistance = Number.POSITIVE_INFINITY;
  for (const candidate of rawMemory) {
    const candidateChunk = chunks.find((chunk) => chunk.id === candidate.id);
    const total = rawMemory.reduce((sum, record) => {
      const chunk = chunks.find((item) => item.id === record.id);
      return sum + (candidateChunk && chunk ? cosineDistance(candidateChunk.embedding, chunk.embedding) : 0);
    }, 0);
    if (total < bestDistance) {
      best = candidate;
      bestDistance = total;
    }
  }
  return best;
}

function pairwiseDistances(chunks: MemoryChunk[]): number[] {
  const distances: number[] = [];
  for (let i = 0; i < chunks.length; i += 1) {
    for (let j = i + 1; j < chunks.length; j += 1) {
      distances.push(cosineDistance(chunks[i]!.embedding, chunks[j]!.embedding));
    }
  }
  return distances;
}

function cosineDistance(a: readonly number[], b: readonly number[]): number {
  return clamp01(1 - cosineSimilarity(a, b));
}

function isNegationContradiction(left: string, right: string): boolean {
  const leftNegated = hasNegation(left);
  const rightNegated = hasNegation(right);
  if (leftNegated === rightNegated) return false;
  const leftTokens = contentTokens(stripNegation(left));
  const rightTokens = contentTokens(stripNegation(right));
  if (leftTokens.size === 0 || rightTokens.size === 0) return false;
  const overlap = [...leftTokens].filter((token) => rightTokens.has(token)).length;
  const union = new Set([...leftTokens, ...rightTokens]).size;
  return union > 0 && overlap / union >= 0.5;
}

function hasNegation(text: string): boolean {
  return /\b(do not|don't|never|not|no)\b/i.test(text);
}

function stripNegation(text: string): string {
  return text.replace(/\b(do not|don't|never|not|no)\b/gi, " ");
}

function contentTokens(text: string): Set<string> {
  const stop = new Set(["a", "an", "the", "in", "on", "for", "to", "of", "and", "or", "is", "as", "this", "that"]);
  return new Set(text
    .toLocaleLowerCase()
    .replace(/[^a-z0-9 _-]/g, " ")
    .split(/\s+/)
    .filter((token) => token.length > 2 && !stop.has(token)));
}

function isLowInformation(text: string): boolean {
  return contentTokens(text).size <= 2;
}

function canonicalizeText(text: string): string {
  return text.trim().replace(/\s+/g, " ");
}

function mean(values: number[]): number {
  if (values.length === 0) return 0;
  return Number((values.reduce((sum, value) => sum + value, 0) / values.length).toFixed(6));
}

function median(values: number[]): number {
  if (values.length === 0) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  const middle = Math.floor(sorted.length / 2);
  if (sorted.length % 2 === 1) return Number((sorted[middle] ?? 0).toFixed(6));
  return Number((((sorted[middle - 1] ?? 0) + (sorted[middle] ?? 0)) / 2).toFixed(6));
}

function clamp01(value: number): number {
  if (!Number.isFinite(value)) return 0;
  return Math.max(0, Math.min(1, Number(value.toFixed(6))));
}

function unique(values: string[]): string[] {
  return [...new Set(values)];
}

function stableHash(value: string): string {
  let hash = 2166136261;
  for (let i = 0; i < value.length; i += 1) {
    hash ^= value.charCodeAt(i);
    hash = Math.imul(hash, 16777619);
  }
  return (hash >>> 0).toString(16).padStart(8, "0");
}
