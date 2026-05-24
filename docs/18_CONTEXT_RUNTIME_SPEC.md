# 18 — Context Runtime Spec

## Role

The Context Runtime is the persistent session brain. It owns the ledger, rebuilds context across sessions, selects memory, assigns provenance, and coordinates SSA/TSP/KVSwap/MTP.

It is not only a prompt assembler. It reconstructs the temporary working cognition state for each inference cycle from long-term memory, recent interaction state, pinned constraints, runtime state, current goals, retrieved knowledge, and system identity. The model receives only the assembled active context and live KV state; the rest of memory remains external unless the runtime selects it.

## Core objects

### Context ledger entry

```ts
interface ContextLedgerEntry {
  id: string;
  sessionId: string;
  type: "user" | "assistant" | "tool" | "document" | "memory" | "runtime_trace";
  content: string;
  sourceRef?: string;
  tokenCount: number;
  hash: string;
  createdAt: string;
  metadata: Record<string, unknown>;
}
```

### Context frame

```ts
interface ContextFrame {
  id: string;
  kind: "recent" | "retrieved" | "summary" | "anchor" | "tool" | "document";
  text: string;
  tokenCount: number;
  priority: number;
  provenance: ProvenanceRef[];
}
```

### Context rebuild plan

```ts
interface ContextRebuildPlan {
  requestId: string;
  frames: ContextFrame[];
  pinnedAnchorIds: string[];
  retrievedMemoryIds: string[];
  droppedFrameIds: string[];
  estimatedTokens: number;
  runtimeTraceId: string;
}
```

## Lifecycle

### Session startup

```text
load config
  -> detect device capabilities
  -> open memory engine
  -> load recent ledger entries
  -> query memory summaries
  -> build startup context summary
  -> warm embedding worker
  -> initialize inference backend
```

### Request handling

```text
user input
  -> enter runtime event path
  -> update transcript, active goals, task graph, and execution state
  -> write ledger entry
  -> embed query asynchronously when possible
  -> retrieve semantic, pinned, lineage, task-state, and identity memory
  -> evaluate GAC representatives, exact raw memories, identity risk, and cluster expansion
  -> build working memory set, token budget plan, priority map, and lineage map
  -> ask SSA for sparse routing, block priority, and attention allocation
  -> ask KVSwap for hot/warm/cold cache, prefetch, and eviction-protection plan
  -> ask MTP for speculative decode and draft/target verification strategy
  -> ask TSP for memory sharding and activation layout
  -> assemble final active context
  -> run generation through unlocked inference backend
  -> capture retrieval, routing, dropped-context, cache, token, and generation traces
  -> persist answer, memory chunks, runtime trace
  -> schedule sleep/consolidation jobs where enabled
```

## Context packing order

1. System prompt.
2. Policy and product constraints.
3. Current user request.
4. Pinned anchors.
5. Recent unresolved turns.
6. Top LanceDB memory hits.
7. Tool results.
8. Summaries.
9. Low-priority history only when budget permits.

## Provenance requirement

Every memory-derived statement should trace to one of:

- a ledger id,
- a memory chunk id,
- a document id and byte/character range,
- a tool result id,
- a summary id with source ids.

## Acceptance gates

- Context rebuild works after browser refresh.
- Context rebuild works after app restart when LanceDB sidecar is available.
- Every packed memory frame has provenance.
- Dropped context is logged with a reason.
- Runtime traces include all advanced subsystem statuses.
