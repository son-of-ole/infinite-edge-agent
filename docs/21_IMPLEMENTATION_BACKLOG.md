# 21 — Implementation Backlog

## Phase 0 — Repo and contracts

- [x] Web app scaffold.
- [x] Core package.
- [x] Memory interfaces.
- [x] Runtime feature contracts.
- [x] First-class advanced runtime docs.
- [x] Runtime status panel wired to feature registry.
- [x] Eval artifact writer.

## Phase 1 — Durable memory

- [x] LanceDB sidecar is available as the local durable-memory profile; open-source browser default remains IndexedDB.
- [x] IndexedDB fallback remains available.
- [x] `memory_chunks` schema migration.
- [x] `runtime_traces` table.
- [x] GAC persistence foundation tables and HTTP routes (`raw_memory`, `identity_pin`, `memory_representative`, `memory_lineage`, `retrieval_audit`, `context_pack_trace`).
- [x] background embedding queue.
- [x] memory recall eval.

## Phase 2 — Context Runtime

- [x] context ledger.
- [x] context frame model.
- [x] startup rebuild.
- [x] request-time rebuild.
- [x] provenance writer.
- [x] context rebuild eval.

## Phase 3 — SSA planner

- [x] sparse block planner.
- [x] anchor selector.
- [x] routing trace.
- [x] sparse planner UI diagnostics.
- [x] SSA eval suite.
- [x] native backend interface test harness.

## Phase 4 — TSP planner

- [x] device profile detector.
- [x] model profile registry.
- [x] memory estimator.
- [x] fold schedule builder.
- [x] budget handoff to context runtime.
- [x] TSP eval suite.

## Phase 5 — KVSwap manager

- [x] KV block metadata model.
- [x] pin policy.
- [x] eviction policy.
- [x] prefetch policy.
- [x] pressure telemetry.
- [x] KVSwap eval suite.
- [x] backend tensor handle integration point.

## Phase 6 — MTP/speculative decoding

- [x] draft model registry.
- [x] target model registry.
- [x] draft/verify loop.
- [x] target-only fallback.
- [x] acceptance metrics.
- [x] auto-disable when worse.
- [x] speculative decoding eval suite.

## Phase 7 — native/custom kernel path

- [x] define backend ABI.
- [x] WebGPU compute proof of concept.
- [x] native desktop host proof of concept.
- [x] SSA kernel integration.
- [x] TSP schedule execution.
- [x] KV tensor paging.
- [x] MTP verifier batching.

## Phase 8 — production hardening

- [x] memory encryption option.
- [x] import/export bundle.
- [x] database repair.
- [x] model cache management.
- [x] first-pass privacy controls.
- [x] stress tests.

## Explicit Native-Backend Boundary

Phase 7 is complete for the repo-owned production reference path. `NativeEdgeReferenceBackend` initializes a model boundary, owns deterministic Q/K/V and KV-cache handles, executes one layer through the shared SSA sparse-forward path, validates dense parity on tiny fixtures, and decodes deterministically from the backend-owned cache. `KVTensorPagingRegistry`, `executeTSPSchedule`, and `verifySpeculativeBatch` complete the KV paging, TSP schedule execution, and MTP verifier batching contracts.

The production lane is intentionally limited to the unlocked browser transformer route. Opaque browser chat APIs are not considered satisfying SSA/KV/TSP/MTP, even if they can generate text. The remaining backlog is real-model browser parity and speed: the converted Qwen shard set must produce accurate responses through the unlocked runtime, with full or top-k logit coverage that does not rely on fixture-only proof.
