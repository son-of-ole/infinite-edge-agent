# 36 — GAC Training Objectives

## Purpose

This document defines training objectives for a model-native or controller-model implementation of Geometry-Aware Memory Consolidation.

The goal is to train a model that understands not only what text means, but what memory operations are safe.

## Training targets

The model/controller must learn:

1. Which events should become durable memory.
2. Which memories are identity-critical.
3. Which clusters can be compressed.
4. Which clusters must be split.
5. Which representatives preserve retrieval.
6. When raw lineage is required.
7. When a generated answer needs source grounding.
8. Which memories should stay hot in context/cache.

## Objective 1 — next-token language modeling

Keep normal language ability.

Loss:

`L_lm = cross_entropy(next_token_logits, target_tokens)`

This prevents memory specialization from damaging normal response quality.

## Objective 2 — memory action prediction

Train model to emit structured memory actions.

Actions:

- `NOOP`
- `WRITE_RAW`
- `PIN_EXACT`
- `MERGE_SAFE`
- `SPLIT_CLUSTER`
- `FETCH_RAW_LINEAGE`
- `MARK_CONTRADICTION`
- `DEMOTE_LOW_VALUE`

Loss:

`L_action = cross_entropy(action_logits, action_label)`

Labels come from policy, GAC router outputs, human annotation, and retrieval audits.

## Objective 3 — identity-risk prediction

Train the model to predict whether a memory is unsafe to compress.

Loss:

`L_identity = binary_cross_entropy(predicted_identity_risk, identity_collapse_label)`

Labels come from:

- GAC cluster metrics.
- Exact retrieval tests.
- User corrections.
- Synthetic memory pairs.
- Contradiction datasets.

## Objective 4 — compression strategy prediction

Train model to select consolidation strategy.

Classes:

- centroid.
- medoid.
- medoid plus residuals.
- split.
- no compression.

Loss:

`L_strategy = cross_entropy(strategy_logits, strategy_label)`

## Objective 5 — reconstruction loss

Given representatives, train model/controller to recover source identities or decide that recovery is impossible.

This prevents over-compression.

Loss:

`L_reconstruct = contrastive_loss(query, correct_raw_memory, hard_negatives)`

## Objective 6 — retrieval preservation loss

Train against hit@k failure.

If exact memory should be retrievable but is not found after consolidation, penalize the consolidation decision.

Loss:

`L_retrieval = max(0, margin - score(correct_memory) + score(nearest_wrong_memory))`

## Objective 7 — source-grounding loss

Train the model to know when a claim needs source evidence.

Labels:

- source required.
- representative sufficient.
- exact raw required.
- no external memory needed.

Loss:

`L_source = cross_entropy(source_requirement_logits, source_label)`

## Objective 8 — contradiction preservation loss

Train model not to merge conflicting memories.

Examples:

- "Use Sandbox in production" vs "Do not use Sandbox in production."
- "The price is $30K" vs "The price is $300K."

Loss:

`L_contradiction = contrastive_loss(contradictory_pair, non_contradictory_pair)`

Contradictory memories should remain separable in memory actions.

## Objective 9 — cache priority loss

Train model/controller to predict KVSwap priority.

Labels:

- hot.
- warm.
- cold.
- pinned.
- prefetch.

Loss:

`L_cache = cross_entropy(cache_priority_logits, cache_label)`

Signals:

- Future usage.
- Identity pins.
- Task relevance.
- Retrieval audit failures.

## Objective 10 — sleep-cycle summary loss

Train sleep-cycle output to preserve decisions, constraints, and unresolved tasks.

Do not optimize only for ROUGE-style summary similarity. Optimize for exact recall.

Metrics:

- Decision recall.
- Constraint recall.
- Open task recall.
- Source pointer accuracy.
- Identity pin recall.

## Combined objective

Recommended combined loss:

`L_total = L_lm + aL_action + bL_identity + cL_strategy + dL_reconstruct + eL_retrieval + fL_source + gL_contradiction + hL_cache + iL_sleep`

Weights should be tuned by task. Identity and source losses should be heavily weighted for production agents.

## Dataset sources

### Automatically generated

- Chat transcripts.
- Context pack traces.
- Consolidation runs.
- Retrieval audits.
- User corrections.
- Sleep-cycle outputs.
- Tool call histories.

### Synthetic

- Near-duplicate memory clusters.
- Contradictory memory pairs.
- Similar but distinct architecture decisions.
- Numeric/date perturbations.
- URL/name/code-symbol perturbations.
- Long-session memory probes.

### Human labeled

- Which memories should be pinned.
- Whether compression preserved meaning.
- Whether response used correct source.
- Whether model forgot an important constraint.

## Hard negative generation

Hard negatives are essential.

Generate examples where memories look semantically similar but differ in identity.

Examples:

- "Use Vercel" vs "Use Vercel but not Sandbox."
- "LanceDB stores memory" vs "IndexedDB stores raw fallback memory."
- "SSA is a future adapter" vs "SSA is first-class from day one."
- "$30K policy limit" vs "$300K policy limit."

## Acceptance gates

- Identity-risk model detects exact-fact compression danger.
- Compression strategy beats centroid baseline on identity recall.
- Source-grounding head reduces unsupported claims.
- Sleep-cycle output preserves pinned decisions.
- Controller can be audited from logs.
- User corrections become training examples.
