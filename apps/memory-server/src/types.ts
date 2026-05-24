import { z } from "zod";

const MAX_MEMORY_BATCH = 512;
const MAX_GAC_BATCH = 512;
const MAX_TEXT_CHARS = 64_000;
const MAX_TAGS = 64;
const MAX_LIMIT = 1000;
const finiteNumber = z.number().finite();
const boundedLimit = z.number().int().positive().max(MAX_LIMIT);
const vector = z.array(finiteNumber).min(1).max(16_384);
const tagList = z.array(z.string().min(1).max(128)).max(MAX_TAGS);
const textField = z.string().max(MAX_TEXT_CHARS);

export const memoryChunkSchema = z.object({
  id: z.string().min(1).max(256),
  text: textField,
  embedding: vector,
  sessionId: z.string().min(1).max(256),
  source: z.enum(["chat", "document", "summary", "tool", "system"]),
  role: z.enum(["system", "user", "assistant", "tool"]).optional(),
  createdAt: z.string(),
  updatedAt: z.string(),
  tags: tagList,
  metadata: z.record(z.unknown()),
  tokenCount: z.number().int().nonnegative()
});

export const upsertRequestSchema = z.object({
  chunks: z.array(memoryChunkSchema).min(1).max(MAX_MEMORY_BATCH)
});

export const importMemoryRequestSchema = z.object({
  chunks: z.array(memoryChunkSchema).max(MAX_MEMORY_BATCH)
});

export const searchRequestSchema = z.object({
  embedding: vector,
  options: z.object({
    limit: boundedLimit.optional(),
    minScore: finiteNumber.optional(),
    sessionId: z.string().max(256).optional(),
    tags: tagList.optional(),
    maxAgeMs: z.number().int().positive().optional()
  }).optional()
});

export const deleteMemoryRequestSchema = z.object({
  options: z.object({
    sessionId: z.string().max(256).optional(),
    tags: tagList.optional()
  }).refine((options) => Boolean(options.sessionId) || Boolean(options.tags?.length), {
    message: "Targeted deletion requires sessionId or at least one tag."
  })
});

export const runtimeTraceSchema = z.object({
  traceId: z.string(),
  requestId: z.string(),
  sessionId: z.string(),
  tenantId: z.string().optional(),
  cellId: z.string().optional(),
  modelId: z.string(),
  backend: z.string(),
  createdAt: z.string(),
  runtime: z.record(z.unknown())
});

export const writeRuntimeTraceRequestSchema = z.object({
  trace: runtimeTraceSchema
});

export const rawMemorySearchRequestSchema = z.object({
  options: z.object({
    tenantId: z.string().optional(),
    cellId: z.string().optional(),
    sessionId: z.string().optional(),
    rawMemoryId: z.string().optional(),
    limit: boundedLimit.optional(),
    queryText: textField.optional(),
    includeDeleted: z.boolean().optional()
  }).optional()
});

const isoString = z.string();
const tenantScoped = {
  tenantId: z.string(),
  cellId: z.string(),
  sessionId: z.string().optional()
};

export const rawMemoryRecordSchema = z.object({
  id: z.string(),
  ...tenantScoped,
  sourceType: z.enum(["chat", "file", "code", "tool", "system", "reflection", "external"]),
  sourceUri: z.string().optional(),
  text: textField,
  canonicalText: z.string().optional(),
  memoryKind: z.enum(["fact", "instruction", "decision", "preference", "event", "summary", "observation", "code", "trace"]),
  importance: z.number().min(0).max(1),
  identityRiskSeed: z.number().min(0).max(1),
  createdAt: isoString,
  updatedAt: isoString,
  deletedAt: isoString.optional(),
  retentionClass: z.enum(["normal", "pinned", "legal", "security", "ephemeral", "user_deleted"]),
  hash: z.string()
});

export const identityPinRecordSchema = z.object({
  id: z.string(),
  ...tenantScoped,
  rawMemoryId: z.string(),
  pinReason: z.enum([
    "user_instruction",
    "architecture_decision",
    "legal",
    "security",
    "credential_metadata",
    "date_money_name_url",
    "source_of_truth",
    "manual"
  ]),
  pinStrength: z.number().min(0).max(1),
  expiresAt: isoString.optional(),
  createdBy: z.enum(["user", "policy", "agent", "admin"]),
  createdAt: isoString
});

export const memoryClusterRecordSchema = z.object({
  id: z.string(),
  ...tenantScoped,
  clusterVersion: z.number().int(),
  algorithm: z.string(),
  memberCount: z.number().int().min(0),
  status: z.enum(["open", "stable", "split", "merged", "archived"]),
  rawMemoryIds: z.array(z.string()),
  createdAt: isoString,
  updatedAt: isoString
});

export const clusterMetricRecordSchema = z.object({
  id: z.string(),
  ...tenantScoped,
  clusterId: z.string(),
  clusterVersion: z.number().int(),
  meanDistance: z.number().min(0),
  maxDistance: z.number().min(0),
  medianDistance: z.number().min(0),
  effectiveDimension: z.number().min(0),
  rho: z.number().optional(),
  theta: z.number(),
  thetaPrime: z.number(),
  identityErrorBound: z.number().optional(),
  densityScore: z.number().optional(),
  contradictionScore: z.number().optional(),
  computedAt: isoString
});

export const memoryRepresentativeRecordSchema = z.object({
  id: z.string(),
  ...tenantScoped,
  clusterId: z.string(),
  clusterVersion: z.number().int(),
  type: z.enum(["centroid", "medoid", "residual", "summary", "pin_shadow"]),
  embedding: vector,
  text: z.string().optional(),
  sourceRawMemoryId: z.string().optional(),
  riskScore: z.number().min(0).max(1),
  coverageScore: z.number().min(0).max(1),
  createdByRunId: z.string(),
  createdAt: isoString,
  modelVisible: z.boolean().optional(),
  factual: z.boolean().optional()
});

export const memoryLineageRecordSchema = z.object({
  representativeId: z.string(),
  rawMemoryId: z.string(),
  ...tenantScoped,
  membershipWeight: z.number().min(0),
  distanceToRep: z.number().min(0),
  isPrimary: z.boolean(),
  createdAt: isoString
});

export const consolidationRunRecordSchema = z.object({
  id: z.string(),
  ...tenantScoped,
  mode: z.enum(["immediate", "hourly", "daily", "sleep", "migration", "manual"]),
  inputCount: z.number().int().min(0),
  clusterCount: z.number().int().min(0),
  representativeCount: z.number().int().min(0),
  pinCount: z.number().int().min(0),
  status: z.enum(["running", "complete", "failed", "rolled_back"]),
  startedAt: isoString,
  completedAt: isoString.optional(),
  configHash: z.string(),
  error: z.string().optional()
});

export const retrievalAuditRecordSchema = z.object({
  id: z.string(),
  ...tenantScoped,
  queryText: z.string(),
  expectedRawMemoryId: z.string(),
  retrievedRawMemoryIds: z.array(z.string()),
  retrievedRepresentativeIds: z.array(z.string()),
  hitAtK: z.number().int().optional(),
  identityPreserved: z.boolean(),
  failureMode: z.enum(["centroid_collapse", "over_pruned", "bad_cluster", "embedding_drift", "query_ambiguous", "policy_bug"]).optional(),
  createdAt: isoString
});

export const modelMemoryActionRecordSchema = z.object({
  id: z.string(),
  tenantId: z.string(),
  cellId: z.string(),
  sessionId: z.string(),
  modelId: z.string(),
  actionType: z.string(),
  targetIds: z.array(z.string()),
  argumentsJson: z.record(z.unknown()),
  confidence: z.number().min(0).max(1),
  approvedByPolicy: z.boolean(),
  executedAt: isoString.optional(),
  createdAt: isoString,
  mode: z.enum(["shadow", "enforced", "disabled"]).optional(),
  policyViolations: z.array(z.object({
    code: z.enum([
      "memory_action_disabled",
      "low_confidence",
      "missing_target_ids",
      "destructive_action_not_allowed",
      "invalid_pin_arguments",
      "tenant_scope_mismatch",
      "cell_scope_mismatch"
    ]),
    message: z.string()
  })).optional()
});

export const memoryContradictionRecordSchema = z.object({
  id: z.string(),
  ...tenantScoped,
  rawMemoryIds: z.array(z.string()),
  contradictionType: z.enum(["negation", "numeric", "entity", "manual"]),
  confidence: z.number().min(0).max(1),
  status: z.enum(["open", "resolved", "false_positive"]),
  createdAt: isoString
});

export const sourceDocumentRecordSchema = z.object({
  id: z.string(),
  ...tenantScoped,
  sourceUri: z.string(),
  sourceType: z.enum(["file", "external", "tool", "code", "system"]),
  trustLevel: z.enum(["trusted", "untrusted", "user_confirmed"]),
  memoryWritePolicy: z.enum(["disabled", "quarantine", "allow_raw", "allow_pins"]),
  createdAt: isoString
});

export const trainingExampleRecordSchema = z.object({
  id: z.string(),
  ...tenantScoped,
  datasetType: z.enum(["raw_memory_event", "cluster_consolidation", "identity_preservation", "source_grounding", "sleep_cycle"]),
  sourceRawMemoryIds: z.array(z.string()),
  inputJson: z.record(z.unknown()),
  labelsJson: z.record(z.unknown()),
  privacyClass: z.enum(["private", "consented", "synthetic", "internal_eval"]),
  exportAllowed: z.boolean(),
  createdAt: isoString
});

const gacRoutingBlockMetadataSchema = z.object({
  blockId: z.string(),
  memoryClass: z.enum([
    "PINNED_EXACT",
    "HIGH_RISK_RAW",
    "LOW_RISK_REPRESENTATIVE",
    "BACKGROUND_SUMMARY",
    "SOURCE_EVIDENCE",
    "RECENT_SESSION",
    "TASK_STATE"
  ]),
  rawMemoryId: z.string().optional(),
  representativeId: z.string().optional(),
  identityRisk: z.number().optional(),
  pinStrength: z.number().optional(),
  sourceTrust: z.number().optional(),
  mustAttend: z.boolean().optional()
});

const gacKvSwapPriorityMetadataSchema = z.object({
  blockId: z.string(),
  tier: z.enum(["PIN_HOT", "TASK_HOT", "SESSION_WARM", "BACKGROUND_WARM", "COLD"]),
  priorityScore: z.number(),
  reasonCodes: z.array(z.string())
});

export const contextPackTraceRecordSchema = z.object({
  id: z.string(),
  traceId: z.string(),
  tenantId: z.string(),
  cellId: z.string(),
  sessionId: z.string(),
  queryId: z.string(),
  contextPackId: z.string(),
  rawMemoryIds: z.array(z.string()),
  representativeIds: z.array(z.string()),
  identityPinIds: z.array(z.string()),
  tokenBudget: z.number().int(),
  estimatedTokens: z.number().int().optional(),
  packingStrategy: z.string(),
  includedMemoryIds: z.array(z.string()),
  omittedMemoryIds: z.array(z.string()).optional(),
  ssaRoutingBlocks: z.array(gacRoutingBlockMetadataSchema).optional(),
  kvSwapPriorities: z.array(gacKvSwapPriorityMetadataSchema).optional(),
  createdAt: isoString
});

export const writeRawMemoryRequestSchema = z.object({ records: z.array(rawMemoryRecordSchema).max(MAX_GAC_BATCH) });
export const writeIdentityPinsRequestSchema = z.object({ records: z.array(identityPinRecordSchema).max(MAX_GAC_BATCH) });
export const writeMemoryClustersRequestSchema = z.object({ records: z.array(memoryClusterRecordSchema).max(MAX_GAC_BATCH) });
export const writeClusterMetricsRequestSchema = z.object({ records: z.array(clusterMetricRecordSchema).max(MAX_GAC_BATCH) });
export const writeMemoryRepresentativesRequestSchema = z.object({
  records: z.array(memoryRepresentativeRecordSchema).max(MAX_GAC_BATCH),
  lineage: z.array(memoryLineageRecordSchema).max(MAX_GAC_BATCH).optional()
});
export const writeMemoryLineageRequestSchema = z.object({ records: z.array(memoryLineageRecordSchema).max(MAX_GAC_BATCH) });
export const writeConsolidationRunsRequestSchema = z.object({ records: z.array(consolidationRunRecordSchema).max(MAX_GAC_BATCH) });
export const writeRetrievalAuditsRequestSchema = z.object({ records: z.array(retrievalAuditRecordSchema).max(MAX_GAC_BATCH) });
export const writeContextPackTracesRequestSchema = z.object({ records: z.array(contextPackTraceRecordSchema).max(MAX_GAC_BATCH) });
export const writeModelMemoryActionsRequestSchema = z.object({ records: z.array(modelMemoryActionRecordSchema).max(MAX_GAC_BATCH) });
export const writeMemoryContradictionsRequestSchema = z.object({ records: z.array(memoryContradictionRecordSchema).max(MAX_GAC_BATCH) });
export const writeSourceDocumentsRequestSchema = z.object({ records: z.array(sourceDocumentRecordSchema).max(MAX_GAC_BATCH) });
export const writeTrainingExamplesRequestSchema = z.object({ records: z.array(trainingExampleRecordSchema).max(MAX_GAC_BATCH) });
