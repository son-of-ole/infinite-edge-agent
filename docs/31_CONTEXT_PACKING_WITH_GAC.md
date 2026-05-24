# 31 — Context Packing With GAC

## Purpose

This document defines how the Context Runtime uses Geometry-Aware Memory Consolidation to build an active prompt or active SSA context pack.

The goal is not to retrieve the top-k most similar chunks. The goal is to construct a context pack that preserves task relevance, exact constraints, source grounding, and broad background coverage without drowning the model in duplicates.

## Why normal top-k retrieval is not enough

Top-k vector retrieval has several failure modes:

1. It retrieves many duplicates.
2. It misses rare but identity-critical facts.
3. It over-represents recently repeated terms.
4. It retrieves broad topic matches instead of exact constraints.
5. It cannot tell whether a centroid represents safe compression or identity collapse.
6. It has no memory of what was intentionally pinned.

GAC changes the retrieval unit from "similar chunks" to "risk-aware memory evidence."

## Context pack inputs

The packer receives:

- User query.
- Active task state.
- Current session transcript.
- Cell manifest.
- System/developer instructions.
- LanceDB retrieval candidates.
- GAC representatives.
- GAC identity pins.
- Recent unresolved decisions.
- Open file/code references.
- Safety and access policy.
- Token budget.
- SSA runtime profile.

## Context pack output

The packer returns a structured pack:

- `system_instructions`
- `developer_constraints`
- `cell_identity`
- `active_task_state`
- `identity_pins`
- `exact_memories`
- `representative_memories`
- `source_excerpts`
- `recent_transcript`
- `retrieval_trace`
- `omitted_but_available`

The final model prompt may be textual, but the internal representation must remain structured until the last possible stage.

## Packing order

### 1. Required fixed context

Always include:

- System instructions.
- Active safety policy.
- Cell identity and capability envelope.
- Current user message.
- Current task state.

### 2. Identity pins

Include exact pinned memories that match any of the following:

- Directly relevant to query.
- Relevant to current project/cell.
- Marked global user preference.
- Marked architecture invariant.
- Required by active policy.

Identity pins should be included as exact text, not summaries.

### 3. Recent local context

Include recent messages and actions from the same session. Recent session history is often more important than old semantic matches.

### 4. Exact high-risk memories

Retrieve exact raw memories from high identity-risk clusters.

Use when:

- Cluster spread is high.
- Effective dimension is high.
- Representative coverage is low.
- Query mentions names, dates, money, URLs, code symbols, legal terms, or explicit prior decisions.

### 5. Representatives for broad background

Use representative memories for low-risk, tight clusters.

These provide coverage without duplicating raw memory.

### 6. Source excerpts

When the answer must be grounded, include exact source excerpts or file pointers.

### 7. Omission trace

Record important memories that were not included due to budget.

The model should not see the full omission trace unless needed, but the system must store it for debugging.

## Scoring model

Each candidate gets a packing score:

`packing_score = semantic_relevance + task_relevance + recency + importance + identity_risk + pin_boost + source_trust - duplication_penalty - token_cost_penalty`

### Required dimensions

| Dimension | Meaning |
|---|---|
| `semantic_relevance` | Vector/text similarity to current query. |
| `task_relevance` | Match to current task and cell state. |
| `recency` | Temporal freshness. |
| `importance` | User/agent/policy importance. |
| `identity_risk` | Risk that compressed representation loses the exact fact. |
| `pin_boost` | Boost for identity pins. |
| `source_trust` | Preference for source-grounded records. |
| `duplication_penalty` | Penalizes near-duplicate content. |
| `token_cost_penalty` | Penalizes large items unless necessary. |

## Token budget lanes

The packer must allocate budget into lanes.

| Lane | Default Share | Notes |
|---|---:|---|
| Fixed system/cell context | 10% | Hard required. |
| Current task/session | 20% | High priority. |
| Identity pins | 15% | Can expand if needed. |
| Exact high-risk memories | 20% | Raw memories from fragile clusters. |
| Representatives | 15% | Broad background. |
| Source excerpts | 15% | Evidence. |
| Scratch/agent plan | 5% | Internal reasoning support. |

In SSA mode with very large context capacity, these percentages become priority bands rather than hard caps.

## SSA-specific behavior

For dense or normal models, context packing must aggressively fit a limited prompt.

For SSA-enabled models, context packing should still remove junk, but it can include more raw memory and source evidence.

GAC helps SSA by labeling blocks:

- `PINNED_EXACT`
- `HIGH_RISK_RAW`
- `LOW_RISK_REPRESENTATIVE`
- `BACKGROUND_SUMMARY`
- `SOURCE_EVIDENCE`
- `RECENT_SESSION`
- `TASK_STATE`

These labels become attention-routing metadata.

## Packer modes

### Mode: minimal

For small local models or fast responses.

- Include system, current query, recent context, top identity pins, top representatives.

### Mode: balanced

Default for normal usage.

- Include pins, exact high-risk memories, representatives, and source excerpts.

### Mode: exhaustive

For large SSA context windows.

- Include all relevant pins, many exact memories, many representatives, and source excerpts.
- Still avoid duplicate low-value memories.

### Mode: audit

For debugging and evaluation.

- Include trace metadata and retrieval provenance.
- Used for benchmarks and memory failure analysis.

## Prompt formatting rules

The model-visible prompt must clearly distinguish:

- Exact memory.
- Representative memory.
- Generated summary.
- Source excerpt.
- Current user message.
- System constraint.

Never mix these into one undifferentiated paragraph.

## Example pack item

```json
{
  "kind": "identity_pin",
  "raw_memory_id": "mem_01...",
  "text": "SSA is first-class from the start, not a future research adapter.",
  "reason": "architecture_decision",
  "source_uri": "chat://session/turn/123",
  "identity_risk": 0.96,
  "include_mode": "exact"
}
```

## Failure handling

If retrieval returns only representatives for a high-risk cluster, the packer must fetch raw lineage before finalizing the context pack.

If a representative has missing lineage, the packer must ignore it for factual answers.

If token budget is too small to include required pins, the packer must degrade by using fewer broad representatives, not by dropping pins.

## Acceptance gates

- Exact pins are never replaced by summaries.
- High-risk clusters trigger raw memory fetch.
- Representatives are used only with lineage.
- Context pack trace is stored for every model call.
- The model can distinguish representative from source-grounded fact.
- Benchmarks show lower duplicate rate without lower identity recall.
