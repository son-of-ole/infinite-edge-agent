export const LANCEDB_TABLES = {
  memoryChunks: "memory_chunks",
  rawMemory: "raw_memory",
  identityPin: "identity_pin",
  memoryCluster: "memory_cluster",
  clusterMetric: "cluster_metric",
  memoryRepresentative: "memory_representative",
  memoryLineage: "memory_lineage",
  consolidationRun: "consolidation_run",
  retrievalAudit: "retrieval_audit",
  contextPackTrace: "context_pack_trace",
  modelMemoryAction: "model_memory_action",
  memoryContradiction: "memory_contradiction",
  sourceDocument: "source_document",
  trainingExample: "training_example",
  memorySummaries: "memory_summaries",
  contextLedgers: "context_ledgers",
  runtimeTraces: "runtime_traces",
  embeddingJobs: "embedding_jobs",
  documents: "documents",
} as const;

export interface LanceDBMemoryChunkRow {
  id: string;
  session_id: string;
  document_id?: string | null;
  source_type: "chat" | "document" | "summary" | "tool" | "system";
  role?: "system" | "user" | "assistant" | "tool" | null;
  text: string;
  embedding: number[];
  token_count: number;
  importance: number;
  recency_score: number;
  access_count: number;
  created_at: string;
  updated_at: string;
  tags: string[];
  metadata: Record<string, unknown>;
  provenance: Array<Record<string, unknown>>;
}

export interface RuntimeTraceRow {
  trace_id: string;
  session_id: string;
  request_id: string;
  tenant_id?: string | null;
  cell_id?: string | null;
  model_id?: string | null;
  backend?: string | null;
  runtime_json: Record<string, unknown>;
  latency_ms: number;
  created_at: string;
}
