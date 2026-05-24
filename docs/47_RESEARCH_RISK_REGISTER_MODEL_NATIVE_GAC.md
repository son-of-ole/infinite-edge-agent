# 47 — Research Risk Register: Model-Native GAC

## Purpose

This document records risks and unknowns in the model-native GAC direction.

## Risk 1 — Geometry law may not transfer to all embedding models

The Geometry of Consolidation repo reports experiments across multiple encoders/corpora, but production memory may have different distributions.

Mitigation:

- Validate on our own data.
- Keep raw memory.
- Avoid irreversible compression.

## Risk 2 — Model actions may hallucinate memory operations

A model may propose pins, merges, or deletions for wrong reasons.

Mitigation:

- Policy gate.
- Shadow mode.
- Human review for high-impact actions.

## Risk 3 — Summaries may become false source of truth

Generated summaries can drift.

Mitigation:

- Summaries must have lineage.
- Raw memory remains source of truth.
- Source-grounding evals.

## Risk 4 — Over-pinning causes bloat

If everything is pinned, memory becomes noisy.

Mitigation:

- Pin strength.
- Expiration for some pins.
- Human controls.
- Low-value pin detection.

## Risk 5 — Under-pinning causes identity loss

If the policy is too weak, exact constraints are lost.

Mitigation:

- Conservative default for explicit user instructions.
- Retrieval audits.
- Regression tests.

## Risk 6 — Embedding migration breaks clusters

Changing embedding model can change geometry.

Mitigation:

- Version embeddings.
- Side-by-side clusters.
- Migration eval before cutover.

## Risk 7 — Cross-cell contamination

One cell's bad memory may affect another.

Mitigation:

- Cell-scoped memory.
- Trust boundaries.
- Global pin approval.

## Risk 8 — Privacy conflict with persistence

Memory-native systems can remember too much.

Mitigation:

- User-visible memory controls.
- Deletion propagation.
- Sensitivity labels.
- Training opt-in.

## Risk 9 — Runtime complexity

GAC + SSA + KVSwap + LanceDB can become too complex.

Mitigation:

- Strong contracts.
- Debug traces.
- Feature flags.
- Fallback to raw retrieval.

## Risk 10 — Evaluation misses real failures

Synthetic evals may not match real memory failures.

Mitigation:

- Use user correction telemetry.
- Add regressions for every observed failure.
- Periodically sample context pack traces.

## Go/no-go criteria

Do not enable model-executed memory actions until:

- Shadow mode is stable.
- Pinned recall gate passes.
- Deletion propagation works.
- Source lineage is complete.
- Rollback has been tested.
