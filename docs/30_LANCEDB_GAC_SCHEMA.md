# 30 — LanceDB GAC Schema

## Purpose

This document defines the persistence schema for Geometry-Aware Memory Consolidation using LanceDB as the primary vector and columnar memory engine.

The schema must support exact memory, vector search, clustering, consolidation lineage, identity pins, auditability, and future training dataset generation.

## Storage principles

1. Raw memory is immutable.
2. Derived records must point back to raw memory.
3. Representatives accelerate retrieval but never replace ground truth.
4. Every consolidation decision must be replayable.
5. Every model-visible memory must have provenance.
6. The system must distinguish semantic similarity from identity equivalence.

## LanceDB table overview

### Required tables

- `raw_memory`
- `memory_embedding`
- `memory_cluster`
- `cluster_metric`
- `memory_representative`
- `memory_lineage`
- `identity_pin`
- `consolidation_run`
- `retrieval_audit`
- `context_pack_trace`
- `model_memory_action`

### Optional tables

- `memory_contradiction`
- `memory_revision`
- `source_document`
- `sleep_cycle_run`
- `training_example`

## Table: raw_memory

Stores the exact, immutable record.

| Field | Type | Required | Description |
|---|---:|---:|---|
| `id` | string | yes | Stable UUID or ULID. |
| `tenant_id` | string | yes | User, workspace, or cell owner. |
| `cell_id` | string | yes | Cell that created or owns the memory. |
| `session_id` | string | no | Session where memory was generated. |
| `source_type` | enum | yes | `chat`, `file`, `code`, `tool`, `system`, `reflection`, `external`. |
| `source_uri` | string | no | File path, URL, tool call ID, commit SHA, message ID. |
| `text` | string | yes | Exact text payload. |
| `canonical_text` | string | no | Normalized text for embedding. |
| `memory_kind` | enum | yes | `fact`, `instruction`, `decision`, `preference`, `event`, `summary`, `observation`, `code`, `trace`. |
| `importance` | number | yes | 0..1 score from user, agent, or policy. |
| `identity_risk_seed` | number | yes | Initial risk before cluster metrics. |
| `created_at` | timestamp | yes | Creation time. |
| `updated_at` | timestamp | yes | Updates only metadata, never raw text. |
| `deleted_at` | timestamp | no | Soft-delete marker. |
| `retention_class` | enum | yes | `normal`, `pinned`, `legal`, `security`, `ephemeral`, `user_deleted`. |
| `hash` | string | yes | Hash of exact text plus source metadata. |

### Invariants

- `text` is append-only and immutable.
- If memory is corrected, create a new `raw_memory` plus a `memory_revision` record.
- User deletion requests must be honored by tombstoning and deleting derived embeddings where required.

## Table: memory_embedding

Stores embedding vectors and model metadata.

| Field | Type | Required | Description |
|---|---:|---:|---|
| `id` | string | yes | Embedding record ID. |
| `raw_memory_id` | string | yes | Link to raw memory. |
| `embedding` | vector | yes | Dense vector. |
| `embedding_model` | string | yes | Model name and version. |
| `embedding_dim` | integer | yes | Vector dimension. |
| `normalization` | enum | yes | `l2`, `none`, `cosine_ready`. |
| `chunk_index` | integer | no | For chunked source documents. |
| `created_at` | timestamp | yes | Creation time. |

### Indexes

- Vector index on `embedding`.
- Filter index on `tenant_id`, `cell_id`, `memory_kind`, `created_at` via joined/raw metadata or denormalized fields.

## Table: memory_cluster

Stores a cluster assignment for a set of memory embeddings.

| Field | Type | Required | Description |
|---|---:|---:|---|
| `id` | string | yes | Cluster ID. |
| `tenant_id` | string | yes | Owner. |
| `cell_id` | string | yes | Cell scope. |
| `cluster_version` | integer | yes | Incremented when cluster membership changes. |
| `algorithm` | string | yes | `hdbscan`, `kmeans`, `incremental_hnsw`, `local_radius`, etc. |
| `member_count` | integer | yes | Number of raw members. |
| `status` | enum | yes | `open`, `stable`, `split`, `merged`, `archived`. |
| `created_at` | timestamp | yes | Creation time. |
| `updated_at` | timestamp | yes | Last update. |

## Table: cluster_metric

Stores geometry measurements used by the consolidation router.

| Field | Type | Required | Description |
|---|---:|---:|---|
| `id` | string | yes | Metric ID. |
| `cluster_id` | string | yes | Cluster ID. |
| `cluster_version` | integer | yes | Metrics match this version. |
| `mean_distance` | number | yes | Mean within-cluster cosine distance. |
| `max_distance` | number | yes | Worst internal distance. |
| `median_distance` | number | yes | Median internal distance. |
| `effective_dimension` | number | yes | Participation-ratio or local dimension estimate. |
| `rho` | number | no | Spectral concentration / anisotropy proxy. |
| `theta` | number | yes | Retrieval threshold. |
| `theta_prime` | number | yes | Retrieval cap half-angle proxy. |
| `identity_error_bound` | number | no | Estimated lower bound or risk proxy. |
| `density_score` | number | no | Local density. |
| `contradiction_score` | number | no | Mixed/contradictory concepts. |
| `computed_at` | timestamp | yes | Measurement time. |

## Table: memory_representative

Stores derived vectors used for compact retrieval.

| Field | Type | Required | Description |
|---|---:|---:|---|
| `id` | string | yes | Representative ID. |
| `cluster_id` | string | yes | Parent cluster. |
| `cluster_version` | integer | yes | Cluster version. |
| `type` | enum | yes | `centroid`, `medoid`, `residual`, `summary`, `pin_shadow`. |
| `embedding` | vector | yes | Representative vector. |
| `text` | string | no | Human-readable label or generated summary. |
| `source_raw_memory_id` | string | no | Required for medoid or pin shadow. |
| `risk_score` | number | yes | Identity risk estimate. |
| `coverage_score` | number | yes | How much of cluster it covers. |
| `created_by_run_id` | string | yes | Consolidation run. |
| `created_at` | timestamp | yes | Creation time. |

### Important

Centroid representatives may not be shown to the model as factual text unless accompanied by lineage or generated summary with provenance.

## Table: memory_lineage

Maps representatives to exact raw members.

| Field | Type | Required | Description |
|---|---:|---:|---|
| `representative_id` | string | yes | Representative. |
| `raw_memory_id` | string | yes | Exact source memory. |
| `membership_weight` | number | yes | Contribution/coverage weight. |
| `distance_to_rep` | number | yes | Distance from member to representative. |
| `is_primary` | boolean | yes | Whether this is a primary represented member. |
| `created_at` | timestamp | yes | Creation time. |

## Table: identity_pin

Stores exact memories that must not be compressed away.

| Field | Type | Required | Description |
|---|---:|---:|---|
| `id` | string | yes | Pin ID. |
| `raw_memory_id` | string | yes | Pinned memory. |
| `pin_reason` | enum | yes | `user_instruction`, `architecture_decision`, `legal`, `security`, `credential_metadata`, `date_money_name_url`, `source_of_truth`, `manual`. |
| `pin_strength` | number | yes | 0..1. |
| `expires_at` | timestamp | no | Optional expiration. |
| `created_by` | enum | yes | `user`, `policy`, `agent`, `admin`. |
| `created_at` | timestamp | yes | Creation time. |

## Table: consolidation_run

Audit table for every consolidation job.

| Field | Type | Required | Description |
|---|---:|---:|---|
| `id` | string | yes | Run ID. |
| `tenant_id` | string | yes | Owner. |
| `cell_id` | string | yes | Cell scope. |
| `mode` | enum | yes | `immediate`, `hourly`, `daily`, `sleep`, `migration`, `manual`. |
| `input_count` | integer | yes | Memories considered. |
| `cluster_count` | integer | yes | Clusters processed. |
| `representative_count` | integer | yes | Representatives written. |
| `pin_count` | integer | yes | Pins written. |
| `status` | enum | yes | `running`, `complete`, `failed`, `rolled_back`. |
| `started_at` | timestamp | yes | Start time. |
| `completed_at` | timestamp | no | Completion time. |
| `config_hash` | string | yes | Hash of config/policy. |
| `error` | string | no | Failure message. |

## Table: retrieval_audit

Stores validation probes and retrieval outcomes.

| Field | Type | Required | Description |
|---|---:|---:|---|
| `id` | string | yes | Audit ID. |
| `query_text` | string | yes | Probe query. |
| `expected_raw_memory_id` | string | yes | The exact memory that should be retrieved. |
| `retrieved_raw_memory_ids` | string[] | yes | Retrieved exact memories. |
| `retrieved_representative_ids` | string[] | yes | Retrieved representatives. |
| `hit_at_k` | integer | no | First rank hit. |
| `identity_preserved` | boolean | yes | Whether exact identity was retrieved. |
| `failure_mode` | enum | no | `centroid_collapse`, `over_pruned`, `bad_cluster`, `embedding_drift`, `query_ambiguous`, `policy_bug`. |
| `created_at` | timestamp | yes | Audit time. |

## Table: context_pack_trace

Stores what memory entered the active context.

| Field | Type | Required | Description |
|---|---:|---:|---|
| `id` | string | yes | Trace ID. |
| `session_id` | string | yes | Session. |
| `query_id` | string | yes | User query or agent step. |
| `raw_memory_ids` | string[] | yes | Raw exact memories included. |
| `representative_ids` | string[] | yes | Representatives included. |
| `identity_pin_ids` | string[] | yes | Pins included. |
| `token_budget` | integer | yes | Budget used. |
| `packing_strategy` | string | yes | Strategy name. |
| `created_at` | timestamp | yes | Creation time. |

## Table: model_memory_action

Stores model-native memory decisions.

| Field | Type | Required | Description |
|---|---:|---:|---|
| `id` | string | yes | Action ID. |
| `session_id` | string | yes | Session. |
| `model_id` | string | yes | Model that produced the action. |
| `action_type` | enum | yes | `write`, `pin`, `merge`, `split`, `summarize`, `retrieve_raw`, `demote`, `forget_request`. |
| `target_ids` | string[] | yes | Memory, cluster, or representative IDs. |
| `arguments_json` | object | yes | Structured payload. |
| `confidence` | number | yes | 0..1. |
| `approved_by_policy` | boolean | yes | Whether policy allowed it. |
| `executed_at` | timestamp | no | Execution time. |
| `created_at` | timestamp | yes | Creation time. |

## Migration strategy

### From v3 memory model

1. Keep existing raw memory/chunk tables.
2. Add `memory_embedding` if not already separated.
3. Add identity pins.
4. Add cluster metrics.
5. Add representatives and lineage.
6. Backfill GAC metrics in batches.

### Backfill priority

1. User instructions.
2. Architecture decisions.
3. Project decisions.
4. Recent sessions.
5. Uploaded documents.
6. Low-value chat history.

## Security and privacy

- Do not embed secrets directly when preventable.
- Secret-like raw memories should be encrypted and excluded from model-visible summaries.
- Identity pins may contain sensitive facts; retrieval must obey user, tenant, and cell policy.
- User deletion must remove derived representatives and embeddings when requested.

## Acceptance gates

- Every representative has lineage.
- Every model-visible memory has provenance.
- Every identity pin can be retrieved exactly.
- No consolidation run can hard-delete raw memory.
- Retrieval audit exposes identity collapse failures.
- A deterministic re-run with the same config produces the same cluster decisions or records the source of nondeterminism.
