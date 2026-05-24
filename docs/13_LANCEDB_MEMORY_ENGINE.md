# 13 — LanceDB Memory Engine

## Role

LanceDB is the primary durable memory engine for the system. IndexedDB remains a fallback and offline queue, but LanceDB is the target production memory store for real projects because it is disk-backed, vector-native, and designed around embeddings plus metadata.

## Deployment modes

| Mode | Description | Use case |
|---|---|---|
| Browser-only fallback | IndexedDB stores chunks and vectors. Search is linear or small-index only. | Pure web demo |
| Local sidecar | Node server exposes LanceDB over localhost. | Developer preview |
| Desktop bundle | Tauri/Electron/native host embeds LanceDB directly. | Production local-first app |
| Edge appliance | Local service on device or LAN. | Enterprise/private deployment |

## Tables

### `memory_chunks`

Primary semantic memory table.

| Column | Type | Required | Description |
|---|---|---:|---|
| `id` | string | yes | Stable chunk id |
| `session_id` | string | yes | Session that created the chunk |
| `document_id` | string/null | no | Source document id |
| `source_type` | string | yes | chat, document, tool, summary, system |
| `role` | string/null | no | user, assistant, tool, system |
| `text` | string | yes | Chunk content |
| `embedding` | vector<float> | yes | Embedding vector |
| `token_count` | int | yes | Estimated token count |
| `importance` | float | yes | 0..1 durable importance |
| `recency_score` | float | yes | 0..1 recency weight |
| `access_count` | int | yes | Read count |
| `created_at` | timestamp | yes | Created time |
| `updated_at` | timestamp | yes | Updated time |
| `tags` | list<string> | no | Topic and project tags |
| `metadata` | json | no | Arbitrary source metadata |
| `provenance` | json | yes | Source pointer, hash, offsets |

### `memory_summaries`

Stores compacted summaries for old ledger ranges.

| Column | Type | Description |
|---|---|---|
| `id` | string | Summary id |
| `scope` | string | session, project, document, user_profile |
| `source_ids` | list<string> | Chunk or ledger ids summarized |
| `summary_text` | string | Summary body |
| `embedding` | vector<float> | Summary embedding |
| `valid_from` | timestamp | Start time |
| `valid_to` | timestamp/null | End time |
| `metadata` | json | Compression settings |

### `runtime_traces`

Stores runtime behavior for eval and debugging.

| Column | Type | Description |
|---|---|---|
| `trace_id` | string | Trace id |
| `session_id` | string | Session id |
| `request_id` | string | Request id |
| `runtime_json` | json | SSA/TSP/MTP/KVSwap/Context trace |
| `latency_ms` | int | Total latency |
| `created_at` | timestamp | Created time |

### GAC trace tables

The sidecar now persists the production GAC tables as LanceDB-compatible tables with denormalized filter columns plus a JSON record payload. These tables are intentionally generic and tenant/cell scoped; no deployment-specific tenant names are baked into the schema.

| Table | Purpose |
|---|---|
| `raw_memory` | Immutable exact memory records with source, kind, retention, importance, and hash metadata. |
| `identity_pin` | Exact raw memories that policy or users require the packer to preserve. |
| `memory_cluster` | Tenant/cell-scoped cluster state for incremental and background consolidation. |
| `cluster_metric` | Geometry metrics used to decide whether a cluster can be compressed safely. |
| `memory_representative` | Compact cluster representatives with risk and coverage scores. |
| `memory_lineage` | Representative-to-raw-memory mappings used to prove provenance. |
| `consolidation_run` | Replayable audit rows for immediate, hourly, daily, sleep, migration, or manual consolidation jobs. |
| `retrieval_audit` | Probe queries and retrieval outcomes for identity-preservation evaluation. |
| `context_pack_trace` | Per-request context-pack provenance including included memory IDs, token budget, strategy, and SSA/KVSwap metadata hooks. |
| `model_memory_action` | Policy-gated model memory proposals and shadow/enforced action traces. |
| `memory_contradiction` | Open/resolved contradiction candidates that must not be silently merged. |
| `source_document` | External/file/tool source trust and memory-write policy records. |
| `training_example` | Local/private or export-allowed synthetic/consented examples for future GAC training. |

Model-visible or factual representatives must have lineage at write time. The store writes lineage before the representative row and rejects missing lineage with `MISSING_LINEAGE`.

At runtime, context packing also treats lineage as a model-visibility guard: representative, factual, or model-visible memory hits without raw lineage metadata are dropped from the prompt and recorded in the context-pack trace drop reasons. Context-pack trace writes are blocking for model calls so every generated response has pack provenance.

## Retrieval pipeline

```text
user query
  -> embed query in embedding worker
  -> LanceDB vector search with filters
  -> rerank by score + recency + importance + source authority
  -> return MemorySearchHit[]
  -> Context Runtime packs selected memory into context
  -> SSA planner marks memory blocks as attention candidates
```

## Write pipeline

```text
new event
  -> ledger write
  -> chunker
  -> background embedding queue
  -> upsert to memory_chunks
  -> write raw_memory, identity_pin, cluster_metric, representatives, lineage
  -> training/export policy and contradiction gates
  -> optional summary/compaction job
```

## Required API

```ts
interface DurableMemoryEngine {
  upsertChunks(chunks: MemoryChunk[]): Promise<void>;
  search(query: MemoryQuery): Promise<MemorySearchResult>;
  getByIds(ids: string[]): Promise<MemoryChunk[]>;
  writeRuntimeTrace(trace: RuntimeTrace): Promise<void>;
  compact(scope: CompactionScope): Promise<MemorySummary[]>;
}
```

## Acceptance gates

- 10k memory chunks searched under 150 ms on local sidecar.
- 100k memory chunks searched under 500 ms on developer laptop.
- Search returns provenance for every hit.
- Index rebuild is resumable.
- Browser-only fallback can still answer with degraded status.
