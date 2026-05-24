# 44 — Deployment Strategy for Model-Native GAC

## Purpose

This document defines how to ship GAC safely from MVP to model-native production.

The system should not jump directly to custom model training. It should collect evidence and training data through a staged rollout.

## Deployment stages

### Stage 0 — Documentation and contracts

Deliverables:

- Architecture docs.
- Schema docs.
- API contracts.
- Eval plan.
- ADRs.

### Stage 1 — Runtime-only GAC

Implement:

- Raw memory tables.
- Identity pins.
- Basic cluster metrics.
- Context pack integration.
- Retrieval audits.

No model training yet.

### Stage 2 — GAC router

Implement:

- Centroid/medoid/residual strategies.
- Cluster splitting.
- Representative lineage.
- Sleep-cycle consolidation.

### Stage 3 — Model tool actions

Allow model to propose memory actions using structured tool calls.

All actions are policy-gated.

### Stage 4 — Controller model

Train a small model or classifier on memory action logs.

The controller recommends:

- write.
- pin.
- split.
- compress.
- fetch raw.

### Stage 5 — Adapter-tuned model

Fine-tune LoRA/adapters for memory action prediction and identity-risk.

### Stage 6 — Full model-native research

Modify transformer heads if justified.

## Rollout flags

- `GAC_ENABLED`
- `GAC_WRITE_RAW_ENABLED`
- `GAC_IDENTITY_PINS_ENABLED`
- `GAC_CONSOLIDATION_ENABLED`
- `GAC_SLEEP_CYCLE_ENABLED`
- `MODEL_MEMORY_ACTIONS_ENABLED`
- `MODEL_MEMORY_ACTIONS_EXECUTE_ENABLED`
- `GAC_SSA_ROUTING_ENABLED`
- `GAC_KVSWAP_PRIORITY_ENABLED`

Default production setting should allow observation before execution.

## Shadow mode

Before executing model-proposed memory actions, run them in shadow mode.

Shadow mode stores:

- Proposed action.
- Policy decision.
- What would have changed.
- Whether future retrieval would improve.

## Rollback strategy

Rollback by disabling:

1. Model memory action execution.
2. Consolidation writes.
3. Representative retrieval.
4. Sleep-cycle wake context.

Raw memory should remain intact.

## Compatibility

GAC must work with:

- Browser-only IndexedDB fallback.
- LanceDB sidecar.
- Unlocked browser transformer runtime.
- Future native model runtime.
- SSA fallback/dense reference runtime.

## Migration requirements

Embedding model changes require:

- Re-embedding.
- New cluster versions.
- Side-by-side representative sets.
- Retrieval audit before cutover.

Schema changes require:

- Forward-compatible fields.
- Migration scripts.
- Backup of raw memory.
- Rollback plan.

## Acceptance gates

- Runtime-only GAC improves retrieval duplicate rate without reducing pinned recall.
- Model action shadow mode demonstrates safety before execution.
- Rollback can disable representatives and return to raw retrieval.
- Sleep-cycle wake context can be regenerated from raw memory.
