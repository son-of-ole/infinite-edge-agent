# 45 — Engineering Backlog: Model-Native GAC

## Purpose

This document turns the model-native GAC architecture into implementation epics and tasks.

## Epic 1 — Schema and storage

### Tasks

- Add `raw_memory` table.
- Add `memory_embedding` table.
- Add `identity_pin` table.
- Add `memory_cluster` table.
- Add `cluster_metric` table.
- Add `memory_representative` table.
- Add `memory_lineage` table.
- Add `consolidation_run` table.
- Add `retrieval_audit` table.
- Add migration scripts.

### Acceptance

- Raw memory is immutable.
- Representatives have lineage.
- Identity pins can be queried by cell/project.

## Epic 2 — Identity pin policy

### Tasks

- Implement rule-based pin detector.
- Add protected category detection.
- Add user explicit memory handler.
- Add correction/supersession flow.
- Add deletion propagation.

### Acceptance

- Explicit user instructions are pinned.
- Corrections supersede old memory.
- Deletion invalidates derived records.

## Epic 3 — GAC metrics

### Tasks

- Implement mean within-cluster distance.
- Implement max/median distance.
- Implement effective dimension proxy.
- Implement density and spread scoring.
- Implement contradiction score placeholder.

### Acceptance

- Metrics are deterministic on fixtures.
- Metrics are stored per cluster version.

## Epic 4 — Consolidation router

### Tasks

- Implement centroid strategy.
- Implement medoid strategy.
- Implement residual medoid strategy.
- Implement split decision.
- Implement no-compression decision.
- Implement lineage writing.

### Acceptance

- Router chooses expected strategy on fixtures.
- High-risk clusters are not centroid-compressed.

## Epic 5 — Context packer integration

### Tasks

- Add GAC-aware scoring.
- Add lane-based token budget.
- Add raw lineage fetch for high-risk reps.
- Add context pack trace table.
- Add debug UI output.

### Acceptance

- Pinned facts appear in context pack.
- Representative-only packs are rejected for high-risk facts.

## Epic 6 — Sleep cycle

### Tasks

- Implement sleep trigger.
- Extract session decisions.
- Create wake context.
- Run retrieval probes.
- Store sleep report.

### Acceptance

- Cell wakes with pinned constraints and open tasks.

## Epic 7 — Model memory actions

### Tasks

- Define structured action schema.
- Add model prompt/tool interface.
- Add policy gate.
- Add shadow mode.
- Add action execution.

### Acceptance

- Model can propose actions.
- Unsafe actions are rejected.
- Actions are logged.

## Epic 8 — SSA integration

### Tasks

- Emit routing metadata from context packer.
- Add block classes.
- Add identity boost in SSA planner.
- Add selected/dropped block trace.

### Acceptance

- Pinned exact blocks are routed.
- Sparse routing beats semantic-only on identity recall.

## Epic 9 — KVSwap integration

### Tasks

- Emit cache priority metadata.
- Add hot/warm/cold tier policy.
- Add prefetch plan.
- Add cache trace.

### Acceptance

- Old critical pins are not evicted during active tasks.

## Epic 10 — Evals

### Tasks

- Build synthetic identity datasets.
- Build contradiction dataset.
- Build sleep/wake eval.
- Build regression harness.
- Add CI checks.

### Acceptance

- Metrics from `42_GAC_EVALS_AND_ACCEPTANCE_GATES.md` are reported.

## Suggested implementation order

1. Schema.
2. Identity pins.
3. Raw memory ingestion.
4. Retrieval audit.
5. GAC metrics.
6. Consolidation router.
7. Context packing.
8. Sleep cycle.
9. Model action shadow mode.
10. SSA and KVSwap integration.
11. Controller training.

## Team responsibilities

### Backend/storage engineer

Owns LanceDB schemas, jobs, migrations.

### Runtime engineer

Owns context packer, SSA metadata, KVSwap metadata.

### ML engineer

Owns metrics, evals, controller model, training dataset.

### Frontend engineer

Owns memory debug UI and user memory controls.

### Security engineer

Owns deletion, access control, memory poisoning defenses.
