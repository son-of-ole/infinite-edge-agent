# 39 — KVSwap + GAC Priority

## Purpose

This document defines how Geometry-Aware Memory Consolidation improves KV cache tiering and eviction.

KVSwap manages hot/warm/cold KV cache placement. GAC tells it which memory is identity-critical and should not be evicted merely because it is old.

## Problem

Long sessions create large KV caches. Simple cache policies such as recency or frequency are not enough.

A memory can be old and rare but critical.

If the cache evicts that memory, the agent may forget an important constraint during generation.

## Cache tiers

| Tier | Location | Use |
|---|---|---|
| Hot | VRAM/GPU memory | Immediate attention. |
| Warm | System RAM | Fast prefetch. |
| Cold | Disk/IndexedDB/LanceDB sidecar | Stored for later. |
| Rebuild | Raw memory/context pack | Reconstruct if needed. |

## GAC signals for cache policy

- `identity_risk`
- `pin_strength`
- `cluster_spread`
- `representative_coverage`
- `task_relevance`
- `recency`
- `source_trust`
- `future_use_prediction`
- `retrieval_failure_history`

## Cache priority score

`cache_priority = task_relevance + recency + future_use + pin_strength + identity_risk + source_trust - size_penalty - representative_substitutability`

If a memory has a high-quality representative and low identity risk, it can be colder.

If a memory is pinned or high-risk, it should be hot or quickly prefetchable.

## Cache labels

- `PIN_HOT`: keep in hot cache while task active.
- `KEEP_WARM`: keep in RAM.
- `PREFETCH`: likely needed soon.
- `SWAP_COLD`: safe to move to disk.
- `REBUILD_FROM_SOURCE`: can be rebuilt from raw memory/representative.

## Eviction rules

Do not evict:

- Active system instructions.
- Current task state.
- Required identity pins.
- Source excerpts used in current answer.

Prefer evicting:

- Duplicate low-risk representatives.
- Old low-importance background.
- Memories with strong low-risk representatives.
- Memories with low task relevance and low identity risk.

## Prefetch rules

Prefetch when:

- Query mentions a pinned topic.
- SSA routing selects a representative with high-risk lineage.
- Context Runtime schedules exact raw memories.
- Model retrieval intent head asks for raw lineage.
- User reopens a project/cell.

## Interaction with sleep cycle

When a cell sleeps, hot/warm cache should be cleared or persisted only as allowed by platform. But GAC should write a wake context and prefetch plan.

When waking, KVSwap should warm:

- System/cell state.
- Pinned constraints.
- Active task memories.
- Recent decisions.
- Source excerpts for open tasks.

## Observability

Track:

- Hot cache identity pin count.
- Evicted pinned memory count. Must be zero unless session ended.
- Prefetch hit rate.
- Cache miss causing retrieval failure.
- Cache pressure by memory class.
- Time spent rebuilding context.

## Acceptance gates

- Cache policy never evicts active required pins.
- High-risk old facts beat low-risk recent background when needed.
- Prefetch improves TTFT for project wake-up.
- Cache trace explains tier changes.
- GAC-aware policy beats recency-only policy in long-session tests.
