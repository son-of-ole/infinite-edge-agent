export type ChatRole = "system" | "user" | "assistant" | "tool";

export interface ChatMessage {
  id: string;
  role: ChatRole;
  content: string;
  createdAt: string;
  sessionId: string;
  metadata?: Record<string, unknown>;
}

export type MemorySource = "chat" | "document" | "summary" | "tool" | "system";

export interface MemoryChunk {
  id: string;
  text: string;
  embedding: number[];
  sessionId: string;
  source: MemorySource;
  role?: ChatRole;
  createdAt: string;
  updatedAt: string;
  tags: string[];
  metadata: Record<string, unknown>;
  tokenCount: number;
}

export interface MemorySearchHit extends MemoryChunk {
  score: number;
}

export interface MemorySearchOptions {
  limit?: number;
  minScore?: number;
  sessionId?: string;
  tags?: string[];
  metadata?: Record<string, unknown>;
  maxAgeMs?: number;
  tenantId?: string;
  cellId?: string;
}

export interface MemoryDeleteOptions {
  sessionId?: string;
  tags?: string[];
  tenantId?: string;
  cellId?: string;
}

export interface MemoryStore {
  upsert(chunks: MemoryChunk[]): Promise<void>;
  search(queryEmbedding: number[], options?: MemorySearchOptions): Promise<MemorySearchHit[]>;
  deleteMemory(options: MemoryDeleteOptions): Promise<number>;
  clear(): Promise<void>;
}

export interface MemorySnapshotStore {
  listMemoryChunks(options?: { sessionId?: string; limit?: number; tenantId?: string; cellId?: string }): Promise<MemoryChunk[]>;
  importMemoryChunks(chunks: MemoryChunk[]): Promise<void>;
}

export type MemoryProviderMode = "remote-http" | "lancedb-sidecar" | "browser-vector" | "indexeddb" | "unavailable";

export interface MemoryProviderCapabilities {
  mode: MemoryProviderMode;
  storage: "indexeddb" | "lancedb" | "remote-http" | "none";
  localOnly: boolean;
  vectorSearch: boolean;
  deterministicSearch: boolean;
  metadataFilters: boolean;
  vectorDimension?: number | undefined;
  persistent: boolean;
  importExport: boolean;
  contextPackTracePersistence: boolean;
  remoteSync: boolean;
}

export interface RuntimeTraceStore {
  writeRuntimeTrace(trace: RuntimeTrace): Promise<void>;
  listRuntimeTraces(options?: RuntimeTraceListOptions): Promise<RuntimeTrace[]>;
}

export interface RuntimeTraceListOptions {
  sessionId?: string;
  limit?: number;
  tenantId?: string;
  cellId?: string;
}

export interface RuntimeTraceSnapshotStore extends RuntimeTraceStore {
  importRuntimeTraces(traces: RuntimeTrace[]): Promise<void>;
}

export interface AgentRuntimeConfig {
  modelId: string;
  embeddingModelId: string;
  memoryTopK: number;
  maxRetrievedMemoryTokens: number;
  maxRecentConversationTokens: number;
  maxPromptTokens: number;
  maxGenerationTokens?: number;
  mtp?: AgentRuntimeMtpConfig;
}

export interface AgentRuntimeMtpConfig {
  enabled?: boolean;
  draftModelId?: string | null;
  mode?: "draft_verify" | "tree_draft_verify" | "backend_native";
  numSpeculativeTokens?: number;
  minAcceptanceRate?: number;
  disableWhenLatencyWorse?: boolean;
  targetTokenizerId?: string;
  draftModelProfiles?: AgentRuntimeMtpModelProfile[];
}

export interface AgentRuntimeMtpModelProfile {
  modelId: string;
  role: "draft" | "target" | "both";
  tokenizerId?: string;
  maxSpeculativeTokens?: number;
  targetModelIds?: string[];
}

export interface RuntimeTrace {
  traceId: string;
  requestId: string;
  sessionId: string;
  tenantId?: string;
  cellId?: string;
  modelId: string;
  backend: string;
  createdAt: string;
  runtime: Record<string, unknown>;
}

export type GacRawMemorySourceType = "chat" | "file" | "code" | "tool" | "system" | "reflection" | "external";
export type GacMemoryKind = "fact" | "instruction" | "decision" | "preference" | "event" | "summary" | "observation" | "code" | "trace";
export type GacRetentionClass = "normal" | "pinned" | "legal" | "security" | "ephemeral" | "user_deleted";

export interface RawMemoryRecord {
  id: string;
  tenantId: string;
  cellId: string;
  sessionId?: string | undefined;
  sourceType: GacRawMemorySourceType;
  sourceUri?: string | undefined;
  text: string;
  canonicalText?: string | undefined;
  memoryKind: GacMemoryKind;
  importance: number;
  identityRiskSeed: number;
  createdAt: string;
  updatedAt: string;
  deletedAt?: string | undefined;
  retentionClass: GacRetentionClass;
  hash: string;
}

export type IdentityPinReason =
  | "user_instruction"
  | "architecture_decision"
  | "legal"
  | "security"
  | "credential_metadata"
  | "date_money_name_url"
  | "source_of_truth"
  | "manual";
export type IdentityPinCreator = "user" | "policy" | "agent" | "admin";

export interface IdentityPinRecord {
  id: string;
  tenantId: string;
  cellId: string;
  sessionId?: string | undefined;
  rawMemoryId: string;
  pinReason: IdentityPinReason;
  pinStrength: number;
  expiresAt?: string | undefined;
  createdBy: IdentityPinCreator;
  createdAt: string;
}

export type MemoryRepresentativeType = "centroid" | "medoid" | "residual" | "summary" | "pin_shadow";

export interface MemoryRepresentativeRecord {
  id: string;
  tenantId: string;
  cellId: string;
  sessionId?: string | undefined;
  clusterId: string;
  clusterVersion: number;
  type: MemoryRepresentativeType;
  embedding: number[];
  text?: string | undefined;
  sourceRawMemoryId?: string | undefined;
  riskScore: number;
  coverageScore: number;
  createdByRunId: string;
  createdAt: string;
  modelVisible?: boolean | undefined;
  factual?: boolean | undefined;
}

export interface MemoryLineageRecord {
  representativeId: string;
  rawMemoryId: string;
  tenantId: string;
  cellId: string;
  sessionId?: string | undefined;
  membershipWeight: number;
  distanceToRep: number;
  isPrimary: boolean;
  createdAt: string;
}

export type MemoryClusterStatus = "open" | "stable" | "split" | "merged" | "archived";

export interface MemoryClusterRecord {
  id: string;
  tenantId: string;
  cellId: string;
  sessionId?: string | undefined;
  clusterVersion: number;
  algorithm: string;
  memberCount: number;
  status: MemoryClusterStatus;
  rawMemoryIds: string[];
  createdAt: string;
  updatedAt: string;
}

export interface ClusterMetricRecord {
  id: string;
  tenantId: string;
  cellId: string;
  sessionId?: string | undefined;
  clusterId: string;
  clusterVersion: number;
  meanDistance: number;
  maxDistance: number;
  medianDistance: number;
  effectiveDimension: number;
  rho?: number | undefined;
  theta: number;
  thetaPrime: number;
  identityErrorBound?: number | undefined;
  densityScore?: number | undefined;
  contradictionScore?: number | undefined;
  computedAt: string;
}

export type ConsolidationRunMode = "immediate" | "hourly" | "daily" | "sleep" | "migration" | "manual";
export type ConsolidationRunStatus = "running" | "complete" | "failed" | "rolled_back";

export interface ConsolidationRunRecord {
  id: string;
  tenantId: string;
  cellId: string;
  sessionId?: string | undefined;
  mode: ConsolidationRunMode;
  inputCount: number;
  clusterCount: number;
  representativeCount: number;
  pinCount: number;
  status: ConsolidationRunStatus;
  startedAt: string;
  completedAt?: string | undefined;
  configHash: string;
  error?: string | undefined;
}

export interface ModelMemoryActionRecord {
  id: string;
  tenantId: string;
  cellId: string;
  sessionId: string;
  modelId: string;
  actionType: string;
  targetIds: string[];
  argumentsJson: Record<string, unknown>;
  confidence: number;
  approvedByPolicy: boolean;
  executedAt?: string | undefined;
  createdAt: string;
  mode?: ModelMemoryActionMode | undefined;
  policyViolations?: ModelMemoryPolicyViolation[] | undefined;
}

export type MemoryContradictionType = "negation" | "numeric" | "entity" | "manual";
export type MemoryContradictionStatus = "open" | "resolved" | "false_positive";

export interface MemoryContradictionRecord {
  id: string;
  tenantId: string;
  cellId: string;
  sessionId?: string | undefined;
  rawMemoryIds: string[];
  contradictionType: MemoryContradictionType;
  confidence: number;
  status: MemoryContradictionStatus;
  createdAt: string;
}

export type SourceDocumentTrustLevel = "trusted" | "untrusted" | "user_confirmed";
export type SourceDocumentMemoryWritePolicy = "disabled" | "quarantine" | "allow_raw" | "allow_pins";

export interface SourceDocumentRecord {
  id: string;
  tenantId: string;
  cellId: string;
  sessionId?: string | undefined;
  sourceUri: string;
  sourceType: "file" | "external" | "tool" | "code" | "system";
  trustLevel: SourceDocumentTrustLevel;
  memoryWritePolicy: SourceDocumentMemoryWritePolicy;
  createdAt: string;
}

export type TrainingDatasetType =
  | "raw_memory_event"
  | "cluster_consolidation"
  | "identity_preservation"
  | "source_grounding"
  | "sleep_cycle";
export type TrainingPrivacyClass = "private" | "consented" | "synthetic" | "internal_eval";

export interface TrainingExampleRecord {
  id: string;
  tenantId: string;
  cellId: string;
  sessionId?: string | undefined;
  datasetType: TrainingDatasetType;
  sourceRawMemoryIds: string[];
  inputJson: Record<string, unknown>;
  labelsJson: Record<string, unknown>;
  privacyClass: TrainingPrivacyClass;
  exportAllowed: boolean;
  createdAt: string;
}

export type RetrievalFailureMode =
  | "centroid_collapse"
  | "over_pruned"
  | "bad_cluster"
  | "embedding_drift"
  | "query_ambiguous"
  | "policy_bug";

export interface RetrievalAuditRecord {
  id: string;
  tenantId: string;
  cellId: string;
  sessionId?: string | undefined;
  queryText: string;
  expectedRawMemoryId: string;
  retrievedRawMemoryIds: string[];
  retrievedRepresentativeIds: string[];
  hitAtK?: number | undefined;
  identityPreserved: boolean;
  failureMode?: RetrievalFailureMode | undefined;
  createdAt: string;
}

export type SleepCycleRunStatus = "complete" | "failed" | "rolled_back";

export type WakeContextSectionName =
  | "Cell Identity"
  | "Current Goal"
  | "Pinned Constraints"
  | "Decisions Since Last Wake"
  | "Open Tasks"
  | "Important Sources"
  | "Memory Map"
  | "Risks and Unknowns"
  | "Next Suggested Actions";

export interface SleepCycleRunRecord {
  id: string;
  tenantId: string;
  cellId: string;
  sessionId: string;
  runId: string;
  mode: "sleep";
  status: SleepCycleRunStatus;
  sourceRawMemoryIds: string[];
  identityPinIds: string[];
  contextPackTraceIds: string[];
  runtimeTraceIds: string[];
  representativeIds: string[];
  retrievalAuditIds: string[];
  wakeContextId?: string | undefined;
  inputCount: number;
  representativeCount: number;
  pinCount: number;
  openTaskCount: number;
  riskCount: number;
  configHash: string;
  startedAt: string;
  completedAt?: string | undefined;
  error?: string | undefined;
  rollbackAuditId?: string | undefined;
}

export interface WakeContextRecord {
  id: string;
  tenantId: string;
  cellId: string;
  sessionId: string;
  runId: string;
  status: "complete" | "failed";
  fileName: "wake_context.md";
  markdown: string;
  sections: Record<WakeContextSectionName, string>;
  sourceRawMemoryIds: string[];
  identityPinIds: string[];
  contextPackTraceIds: string[];
  runtimeTraceIds: string[];
  representativeIds: string[];
  retrievalAuditIds: string[];
  createdAt: string;
}

export interface SleepCycleRollbackAuditRecord {
  id: string;
  tenantId: string;
  cellId: string;
  sessionId: string;
  runId: string;
  status: "rolled_back";
  reason: string;
  sourceRawMemoryIds: string[];
  affectedRepresentativeIds: string[];
  previousWakeContextId?: string | undefined;
  createdAt: string;
}

export interface GacRoutingBlockMetadata {
  blockId: string;
  memoryClass: "PINNED_EXACT" | "HIGH_RISK_RAW" | "LOW_RISK_REPRESENTATIVE" | "BACKGROUND_SUMMARY" | "SOURCE_EVIDENCE" | "RECENT_SESSION" | "TASK_STATE";
  rawMemoryId?: string | undefined;
  representativeId?: string | undefined;
  identityRisk?: number | undefined;
  pinStrength?: number | undefined;
  sourceTrust?: number | undefined;
  mustAttend?: boolean | undefined;
}

export interface GacKvSwapPriorityMetadata {
  blockId: string;
  tier: "PIN_HOT" | "TASK_HOT" | "SESSION_WARM" | "BACKGROUND_WARM" | "COLD";
  priorityScore: number;
  reasonCodes: string[];
}

export interface ContextPackTraceRecord {
  id: string;
  traceId: string;
  tenantId: string;
  cellId: string;
  sessionId: string;
  queryId: string;
  contextPackId: string;
  rawMemoryIds: string[];
  representativeIds: string[];
  identityPinIds: string[];
  tokenBudget: number;
  estimatedTokens?: number | undefined;
  packingStrategy: string;
  includedMemoryIds: string[];
  omittedMemoryIds?: string[] | undefined;
  ssaRoutingBlocks?: GacRoutingBlockMetadata[] | undefined;
  kvSwapPriorities?: GacKvSwapPriorityMetadata[] | undefined;
  predictivePlanId?: string | undefined;
  predictedRetrievals?: unknown[] | undefined;
  contextBranches?: unknown[] | undefined;
  kvHotPages?: unknown[] | undefined;
  mtpBranches?: unknown[] | undefined;
  createdAt: string;
}

export interface GacListOptions {
  tenantId?: string;
  cellId?: string;
  sessionId?: string;
  rawMemoryId?: string;
  representativeId?: string;
  contextPackId?: string;
  limit?: number;
}

export interface RawMemorySearchOptions extends GacListOptions {
  queryText?: string;
  includeDeleted?: boolean;
}

export interface GacWriteResult {
  ok: true;
  count: number;
  traceId: string;
}

export interface GacMemoryStore {
  writeRawMemory(records: RawMemoryRecord[]): Promise<GacWriteResult>;
  listRawMemory(options?: GacListOptions): Promise<RawMemoryRecord[]>;
  writeIdentityPins(records: IdentityPinRecord[]): Promise<GacWriteResult>;
  listIdentityPins(options?: GacListOptions): Promise<IdentityPinRecord[]>;
  writeMemoryRepresentatives(records: MemoryRepresentativeRecord[], lineage?: MemoryLineageRecord[]): Promise<GacWriteResult>;
  listMemoryRepresentatives(options?: GacListOptions): Promise<MemoryRepresentativeRecord[]>;
  writeMemoryLineage(records: MemoryLineageRecord[]): Promise<GacWriteResult>;
  listMemoryLineage(options?: GacListOptions): Promise<MemoryLineageRecord[]>;
  writeRetrievalAudits(records: RetrievalAuditRecord[]): Promise<GacWriteResult>;
  listRetrievalAudits(options?: GacListOptions): Promise<RetrievalAuditRecord[]>;
  writeContextPackTraces(records: ContextPackTraceRecord[]): Promise<GacWriteResult>;
  listContextPackTraces(options?: GacListOptions): Promise<ContextPackTraceRecord[]>;
  writeMemoryClusters(records: MemoryClusterRecord[]): Promise<GacWriteResult>;
  listMemoryClusters(options?: GacListOptions): Promise<MemoryClusterRecord[]>;
  writeClusterMetrics(records: ClusterMetricRecord[]): Promise<GacWriteResult>;
  listClusterMetrics(options?: GacListOptions): Promise<ClusterMetricRecord[]>;
  writeConsolidationRuns(records: ConsolidationRunRecord[]): Promise<GacWriteResult>;
  listConsolidationRuns(options?: GacListOptions): Promise<ConsolidationRunRecord[]>;
  writeModelMemoryActions(records: ModelMemoryActionRecord[]): Promise<GacWriteResult>;
  listModelMemoryActions(options?: GacListOptions): Promise<ModelMemoryActionRecord[]>;
  writeMemoryContradictions(records: MemoryContradictionRecord[]): Promise<GacWriteResult>;
  listMemoryContradictions(options?: GacListOptions): Promise<MemoryContradictionRecord[]>;
  writeSourceDocuments(records: SourceDocumentRecord[]): Promise<GacWriteResult>;
  listSourceDocuments(options?: GacListOptions): Promise<SourceDocumentRecord[]>;
  writeTrainingExamples(records: TrainingExampleRecord[]): Promise<GacWriteResult>;
  listTrainingExamples(options?: GacListOptions): Promise<TrainingExampleRecord[]>;
}

export interface ContextPackTraceStore {
  writeContextPackTraces(records: ContextPackTraceRecord[]): Promise<GacWriteResult>;
  listContextPackTraces(options?: GacListOptions): Promise<ContextPackTraceRecord[]>;
}

export type ModelMemoryActionType =
  | "create_raw_memory"
  | "pin_memory"
  | "request_consolidation"
  | "request_retrieval_probe"
  | "forget_memory";

export type ModelMemoryActionMode = "shadow" | "enforced" | "disabled";

export type ModelMemoryPolicyViolationCode =
  | "memory_action_disabled"
  | "low_confidence"
  | "missing_target_ids"
  | "destructive_action_not_allowed"
  | "invalid_pin_arguments"
  | "tenant_scope_mismatch"
  | "cell_scope_mismatch";

export interface ProposeModelMemoryActionRequest {
  tenantId?: string | undefined;
  cellId?: string | undefined;
  sessionId: string;
  modelId: string;
  actionType: ModelMemoryActionType;
  targetIds: string[];
  arguments?: Record<string, unknown> | undefined;
  confidence: number;
}

export interface ModelMemoryPolicyScope {
  tenantId?: string | undefined;
  cellId?: string | undefined;
}

export interface ModelMemoryActionPolicy {
  minConfidence?: number | undefined;
  allowDestructiveActions?: boolean | undefined;
  scope?: ModelMemoryPolicyScope | undefined;
}

export interface ModelMemoryPolicyViolation {
  code: ModelMemoryPolicyViolationCode;
  message: string;
}

export interface ModelMemoryPolicyDecision {
  approved: boolean;
  policyNotes: string[];
  violations: ModelMemoryPolicyViolation[];
}

export interface ModelMemoryActionTrace {
  actionId: string;
  request: ProposeModelMemoryActionRequest;
  decision: ModelMemoryPolicyDecision;
  mode: ModelMemoryActionMode;
  timestamp: string;
  policyViolations: ModelMemoryPolicyViolation[];
}

export interface ModelMemoryActionExecutionResult {
  resultIds?: string[] | undefined;
}

export interface ProposeModelMemoryActionOptions {
  mode?: ModelMemoryActionMode | undefined;
  policy?: ModelMemoryActionPolicy | undefined;
  now?: Date | undefined;
  actionIdFactory?: ((request: ProposeModelMemoryActionRequest) => string) | undefined;
  execute?: ((request: ProposeModelMemoryActionRequest, trace: ModelMemoryActionTrace) => Promise<ModelMemoryActionExecutionResult | string[] | void> | ModelMemoryActionExecutionResult | string[] | void) | undefined;
  onTrace?: ((trace: ModelMemoryActionTrace) => Promise<void> | void) | undefined;
}

export interface ProposeModelMemoryActionResponse {
  actionId: string;
  approved: boolean;
  executed: boolean;
  resultIds: string[];
  policyNotes: string[];
  trace: ModelMemoryActionTrace;
}

export interface ContextPackInput {
  systemPrompt: string;
  retrievedMemory: MemorySearchHit[];
  recentMessages: ChatMessage[];
  userMessage: string;
  config: Pick<AgentRuntimeConfig, "maxRetrievedMemoryTokens" | "maxRecentConversationTokens" | "maxPromptTokens">;
}

export interface PackedContext {
  messages: Array<{ role: "system" | "user" | "assistant"; content: string }>;
  includedMemoryIds: string[];
  estimatedTokens: number;
}
