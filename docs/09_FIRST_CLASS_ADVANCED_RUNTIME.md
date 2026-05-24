# 09 — First-Class Advanced Runtime

## Purpose

This document replaces the old “research adapters” concept. LanceDB, SSA, TSP, MTP/speculative decoding, KVSwap, and Context Runtime are now Tier-0 components.

The runtime is the intelligence system boundary. The model is one subsystem inside it, not the whole agent. Persistence, identity, continuity, memory selection, cache residency, sparse routing, decode strategy, and consolidation belong to the runtime layer.

In short:

```text
model = reasoning engine
runtime = operating system for persistent cognition
memory = durable state and lineage
context runtime = active working-state assembly
SSA = selective cognition
KVSwap = working-memory paging
GAC = memory consolidation
MTP = predictive execution
TSP = memory-distributed compute orchestration
```

## Tier-0 definition

A Tier-0 component is:

1. Present in the architecture diagram.
2. Present in config.
3. Present in TypeScript contracts.
4. Present in runtime status and telemetry.
5. Present in the eval suite.
6. Present in the build roadmap.
7. Allowed to run in fallback mode only until the production backend is ready.

## Runtime component matrix

| Component | Required from day 1 | Fallback allowed | Production backend |
|---|---:|---:|---|
| LanceDB | Yes | IndexedDB fallback | Embedded LanceDB / local sidecar / desktop bundle |
| SSA | Yes | Sparse context planner | SSA-capable attention kernel/model backend |
| TSP | Yes | Budget planner | WebGPU/native folded TP+SP scheduler |
| MTP/speculative decoding | Yes | Target-only decoding | Draft/target verifier with acceptance metrics |
| KVSwap | Yes | Cache metadata simulator | Real KV tensor offload/prefetch manager |
| Context Runtime | Yes | No | Ledger + rebuild + provenance + packing runtime |

## Required runtime status payload

Every model response must produce a runtime trace:

```json
{
  "traceId": "trace_...",
  "sessionId": "session_...",
  "modelId": "...",
  "contextRuntime": {
    "ledgerEntriesRead": 0,
    "memoryChunksRetrieved": 0,
    "tokensPacked": 0,
    "pinnedAnchors": []
  },
  "lancedb": {
    "enabled": true,
    "table": "memory_chunks",
    "queryMs": 0
  },
  "ssa": {
    "mode": "fallback_sparse_planner",
    "selectedBlocks": 0,
    "sparsityRatio": 0
  },
  "tsp": {
    "mode": "fallback_budget_planner",
    "sequenceShards": 1,
    "tensorShards": 1
  },
  "mtp": {
    "mode": "target_only",
    "draftTokens": 0,
    "acceptedTokens": 0
  },
  "kvswap": {
    "mode": "metadata_only",
    "pinnedBlocks": 0,
    "evictedBlocks": 0,
    "prefetchedBlocks": 0
  },
  "predictive": {
    "planId": "pred_...",
    "predictedRetrievals": [],
    "contextBranches": [],
    "kvHotPages": [],
    "sparseBlocks": [],
    "mtpBranches": [],
    "cacheBudget": {
      "prefetchBlockIds": [],
      "evictableBlockIds": []
    }
  }
}
```

## Build implications

The app must never be wired as:

```text
UI -> opaque chat API -> response
```

The app must be wired as:

```text
UI
  -> Agent Session Controller
  -> Context Runtime
  -> Advanced Runtime Coordinator
  -> Inference Backend
  -> Runtime Trace Writer
```

This keeps SSA/TSP/MTP/KVSwap in the core path even before their production kernels exist.

The runtime must not stop at `retrieve chunks -> build prompt`. It must plan compute, memory, attention, cache movement, decode strategy, token allocation, and consolidation as part of every turn. That is the difference between a normal chatbot pipeline and this persistent runtime architecture.

## Predictive runtime layer

The advanced coordinator now emits a predictive runtime plan on every turn. This plan is the bridge between MTP, GAC, SSA, KVSwap, and context rebuilding:

- GAC metadata predicts follow-on raw/representative retrievals.
- SSA-selected blocks become predicted sparse blocks and KV hot pages.
- MTP verifier branches add future decode pressure before the target model commits tokens.
- KVSwap receives predictive prefetch hints with confidence and reasons.
- Context-pack traces persist the predictive plan id, retrieval predictions, context branches, hot KV pages, and MTP branches.

This is the first production step toward speculative cognition: the runtime can predict not only likely next tokens, but also likely memory accesses, cache residency needs, and context branches.
