# 49 — Engineering Handoff Checklist: Model-Native GAC

## Purpose

This checklist tells the engineering team what must exist before model-native GAC is considered build-ready.

## Documentation checklist

- [x] Architecture docs reviewed.
- [x] ADRs accepted.
- [x] Schema reviewed.
- [x] API contracts reviewed.
- [x] Security/privacy reviewed.
- [x] Eval gates accepted.
- [x] Rollout plan accepted.

## Storage checklist

- [x] Raw memory table exists.
- [x] Embeddings are persisted with `memory_chunks`.
- [x] Identity pin table exists.
- [x] Cluster tables exist.
- [x] Representative table exists.
- [x] Lineage table exists.
- [x] Retrieval audit table exists.
- [x] Deletion propagation implemented.

## Runtime checklist

- [x] Ingestion writes raw memory.
- [x] Embedding worker writes vectors.
- [x] Identity policy pins protected facts.
- [x] GAC metrics compute on clusters.
- [x] Consolidation router writes representatives.
- [x] Context packer consumes pins and reps.
- [x] SSA receives routing metadata.
- [x] KVSwap receives priority metadata.

## Model checklist

- [x] Memory action schema exists.
- [x] Prompt/tool interface exists.
- [x] Policy gate exists.
- [x] Shadow mode exists.
- [x] Execution disabled by default.
- [x] Actions logged.

## Eval checklist

- [x] Synthetic hard negative dataset exists.
- [x] Pinned recall eval exists.
- [x] Contradiction eval exists.
- [x] Sleep/wake eval exists.
- [x] SSA routing eval exists.
- [x] KVSwap priority eval exists.
- [x] Local production eval reports metrics.

## Security checklist

- [x] Sensitive labels exist.
- [x] Secret masking exists.
- [x] External document memory writes are gated.
- [x] Cross-cell access policy exists.
- [x] Training export policy exists.
- [x] User deletion tested.

## Release checklist

- [x] Tier-0 runtime coordinator exists before inference.
- [x] Runtime traces persist in browser fallback memory and sidecar API.
- [x] Runtime-only GAC passes gates.
- [x] Model action shadow mode passes gates.
- [x] Rollback tested.
- [x] Memory inspector available.
- [x] Sleep cycle can be disabled.
- [x] Raw retrieval fallback works.

## Done definition

The first production release is done when:

1. Raw memory and identity pins work.
2. GAC representatives have lineage.
3. Context packer preserves exact constraints.
4. Sleep/wake preserves project continuity.
5. Model memory actions are logged and policy-gated.
6. Evals prove lower bloat without identity loss.

## Explicit Native-Backend Boundary

The current release owns production contracts, traceability, storage, release gates, the WebGPU toy sparse-attention proof, and the repo-owned unlocked/native-edge runtime contracts. `NativeEdgeReferenceBackend`, `UnlockedBrowserTransformerBackend`, `KVTensorPagingRegistry`, `executeTSPSchedule`, and `verifySpeculativeBatch` are complete production infrastructure for custom/native backends.

The production win condition is the unlocked browser transformer route. That route owns model weights, Q/K/V handles, KV tensor paging, and TSP callback execution in the browser. It currently has the converted Qwen manifest path and fixture-backed release gates; a production deployment may claim full model quality only after the real browser shard set passes response-quality and speed gates.

Opaque inference runtimes are not a supported production branch for the unlocked system because they do not expose model-layer Q/K/V tensors, KV-cache handles, or native sparse dispatch hooks.
