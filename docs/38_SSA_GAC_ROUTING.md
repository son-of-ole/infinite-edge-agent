# 38 — SSA + GAC Routing

## Purpose

This document defines how Geometry-Aware Memory Consolidation improves Subquadratic Selective Attention routing.

SSA chooses sparse blocks/tokens to attend to. GAC provides identity-risk and consolidation metadata so SSA does not drop rare but important facts.

## Problem

Sparse attention can fail if it routes only by semantic relevance or learned query-key scores.

A memory may be rare but critical:

- "Do not use Sandbox in production."
- "The policy limit is $30K."
- "Use LanceDB as primary memory engine."
- "SSA is first-class from day one."

These may not be frequent, but losing them changes the answer.

## Routing metadata

Each memory/context block should have metadata:

- `semantic_relevance`
- `task_relevance`
- `identity_risk`
- `pin_strength`
- `source_trust`
- `cluster_spread`
- `representative_coverage`
- `recency`
- `user_importance`
- `legal_security_financial_flag`
- `code_symbol_flag`

## Block classes

| Class | Meaning | SSA Behavior |
|---|---|---|
| `SYSTEM` | System/developer instructions | Always attend. |
| `CELL_STATE` | Cell manifest/task state | Always attend or high priority. |
| `PINNED_EXACT` | Identity-pinned memory | High priority, protected. |
| `HIGH_RISK_RAW` | Fragile exact memory | High priority when relevant. |
| `REPRESENTATIVE` | Low-risk compressed memory | Normal sparse routing. |
| `SOURCE_EVIDENCE` | File/source excerpt | High priority for grounded claims. |
| `RECENT_SESSION` | Current conversation | Recency boosted. |
| `BACKGROUND` | Broad memory | Low/normal priority. |

## Routing score

SSA planner score:

`ssa_score = qk_score + semantic_score + task_score + identity_boost + pin_boost + source_boost + recency_boost - duplicate_penalty - stale_penalty`

The exact function is backend-dependent, but identity and pin boosts must be available to the planner.

## Hard constraints

- System instructions cannot be pruned.
- Required identity pins cannot be pruned.
- Required source excerpts cannot be pruned for cited factual answers.
- High-risk clusters must route to raw memory or lineage before using representative-only context.

## WebGPU SSA implications

The WebGPU SSA kernel plan should allow metadata-aware block selection before sparse attention computation.

Pipeline:

1. Compute or receive block summaries.
2. Score blocks against query.
3. Apply GAC metadata boosts.
4. Select top blocks under budget.
5. Gather K/V for selected blocks.
6. Run sparse attention.
7. Record selected block IDs for trace.

## Interaction with context packing

Context packing decides what enters the model input.

SSA routing decides what the model attends to inside the active input.

Both must use GAC signals, but they operate at different layers.

## Debug trace

Every SSA call should optionally record:

- Candidate blocks.
- Selected blocks.
- Dropped blocks.
- GAC scores.
- Identity pins included.
- High-risk blocks omitted.
- Attention outcome metrics.

## Failure modes

### Rare constraint dropped

Mitigation: identity pins and hard routing rules.

### Representative selected without raw lineage

Mitigation: context packer fetches raw lineage for high-risk reps.

### Duplicate blocks dominate sparse budget

Mitigation: duplicate penalty and representative coverage.

### Old critical memory demoted

Mitigation: pin strength and identity risk override recency decay.

## Acceptance gates

- Pinned exact memories route even when old.
- High-risk raw memories route for exact queries.
- Sparse routing maintains identity recall on needle tests.
- Debug trace explains why a memory block was selected or dropped.
- GAC-augmented routing beats semantic-only routing on exact decision recall.
