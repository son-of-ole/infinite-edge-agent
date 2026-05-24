# 42 — GAC Evals and Acceptance Gates

## Purpose

This document defines how to evaluate Geometry-Aware Memory Consolidation and model-native memory behavior.

The system must be evaluated on identity preservation, not just semantic retrieval.

## Core metrics

### Identity Recall@K

Measures whether the exact raw memory is retrieved in top K.

Required K values:

- @1
- @3
- @5
- @10

### Pinned Recall@K

Same as Identity Recall, but only for identity-pinned memories.

Production gate should be very high.

### Compression Ratio

`raw_memory_count / representative_count`

High compression is good only if identity recall does not drop.

### Identity Collapse Rate

Percent of queries where a representative is retrieved but the exact source identity is lost.

### Source Lineage Coverage

Percent of model-visible memory items with source lineage.

Required production target: 100% for factual memory.

### Decision Preservation Rate

Percent of architecture/project decisions preserved after sleep cycle.

### Constraint Preservation Rate

Percent of explicit constraints preserved after sleep cycle.

### False Merge Rate

Percent of contradictory or distinct memories merged into one representative.

### False Pin Rate

Percent of low-value memories incorrectly pinned.

### Missed Pin Rate

Percent of identity-critical memories not pinned.

## Baselines

Compare GAC against:

- Raw top-k retrieval.
- Centroid compression.
- Medoid-only compression.
- Summarization-only memory.
- Recency-only retrieval.
- Importance-only retrieval.
- HNSW prune.
- Product quantization where relevant.

## Eval suites

### Suite 1 — exact instruction recall

Tests whether user instructions survive consolidation.

Example:

- Instruction: "Do not use Vercel Sandbox in production."
- Query: "What is our production sandbox policy?"
- Expected: exact instruction or source-linked equivalent.

### Suite 2 — architecture decision recall

Tests decisions across long project histories.

### Suite 3 — numeric/date/name recall

Tests fragile exact facts.

### Suite 4 — contradiction separation

Tests that similar but contradictory memories are not merged.

### Suite 5 — sleep/wake continuity

Tests whether a cell wakes with correct state, decisions, and tasks.

### Suite 6 — SSA routing with GAC

Tests whether sparse routing preserves rare critical constraints.

### Suite 7 — KVSwap priority with GAC

Tests whether cache tiering keeps critical memories available.

## Production acceptance gates

Minimum recommended gates for v1:

| Metric | Gate |
|---|---:|
| Pinned Recall@5 | >= 0.98 |
| Source Lineage Coverage | 1.00 |
| False Merge Rate for contradictions | <= 0.01 |
| Missed Pin Rate for explicit user instructions | <= 0.02 |
| Decision Preservation after sleep | >= 0.95 |
| Identity Collapse Rate on protected facts | <= 0.02 |

Research gates can be lower, but production cannot silently degrade memory.

## Regression tests

Every bug involving memory loss or wrong retrieval must become a regression test.

Regression examples:

- User said one exact thing and model remembered a vague version.
- Summary replaced source fact.
- Pinned fact omitted from wake context.
- Similar but opposite decisions merged.
- Cache evicted old critical fact during long task.

## Eval artifact outputs

Each eval run produces:

- Summary metrics.
- Failed queries.
- Retrieved memory IDs.
- Expected memory IDs.
- Context pack traces.
- SSA selected blocks.
- KVSwap tier changes.
- Consolidation run ID.

## Human review queue

Failures should be queued for review when:

- The system is unsure whether identity was preserved.
- The query is ambiguous.
- The answer could be legally/financially/security significant.
- The memory conflict needs user resolution.

## Acceptance gates

- Evals run in CI for synthetic fixtures.
- Nightly evals run on larger memory sets.
- Production telemetry samples retrieval audits.
- Any schema or embedding model migration must pass identity recall gates before rollout.
