# 01 — Architecture

## Executive summary

Infinite Edge Agent is a local-first AI runtime that turns an edge device into a persistent agent environment. The browser app owns the UI and orchestration. Workers isolate inference, embeddings, memory writes, context rebuilds, and runtime planning. LanceDB provides disk-backed semantic memory when a sidecar or desktop bundle is available; IndexedDB remains the browser-only fallback.

The architecture is designed around a first-class long-context stack:

```text
Application UI
  ↓
Agent Session Controller
  ↓
Context Runtime
  ├─ context ledger
  ├─ active context map
  ├─ memory retrieval planner
  ├─ context packing planner
  └─ provenance writer
  ↓
Advanced Runtime Coordinator
  ├─ SSA Runtime
  ├─ TSP Runtime
  ├─ MTP Speculative Decoder
  └─ KVSwap Manager
  ↓
Inference Backend
  └─ unlocked-browser-transformer
      ├─ browser-owned Qwen tensors
      ├─ WebGPU/CPU reference kernels
      └─ SSA/KV/TSP/MTP proof traces
  ↓
Memory + Storage
  ├─ LanceDB memory engine
  ├─ IndexedDB fallback store
  ├─ Cache API model cache
  └─ local file/object storage in desktop/edge bundle
```

## Layer 1 — Application layer

Responsibilities:

- Render chat, memory status, runtime status, and diagnostics.
- Capture user input, files, selected DOM/context elements, and tool events.
- Stream model output.
- Surface runtime degradation warnings such as “SSA fallback planner active” or “KVSwap disk tier disabled.”

Files:

- `apps/web/src/App.tsx`
- `apps/web/src/components/*`
- `apps/web/src/lib/agent/localAgent.ts`

## Layer 2 — Context Runtime

The context runtime is the active brainstem of the system. It decides what belongs in active context, what remains in semantic memory, what must be pinned, and what can be summarized or evicted.

Core objects:

- `ContextLedgerEntry`
- `ContextFrame`
- `ContextRebuildPlan`
- `ActiveContextMap`
- `PinnedAnchor`
- `MemoryRecallTrace`

Main lifecycle:

```text
new input
  -> write raw event to ledger
  -> chunk and embed in background
  -> query LanceDB for relevant memory
  -> choose anchors and recent turns
  -> build active prompt/context plan
  -> pass context plan to SSA/TSP/KVSwap/MTP coordinator
  -> stream response
  -> write response and runtime trace back to ledger
```

See `docs/18_CONTEXT_RUNTIME_SPEC.md`.

## Layer 3 — LanceDB Memory Engine

LanceDB is the primary semantic memory engine. It stores text chunks, embeddings, metadata, memory type, source references, session ids, importance, and provenance.

Tables:

- `memory_chunks`
- `memory_summaries`
- `context_ledgers`
- `runtime_traces`
- `embedding_jobs`
- `documents`

See `docs/13_LANCEDB_MEMORY_ENGINE.md` and `docs/24_SCHEMA_REGISTRY.md`.

## Layer 4 — SSA Runtime

SSA is treated as the attention routing layer. The application cannot simply “turn on SSA” unless the model/backend supports it. The architecture still includes SSA from the beginning through:

- sparse attention plans,
- anchor token selection,
- memory-to-token routing metadata,
- semantic block routing,
- kernel interface shape,
- eval gates for retrieval at long context.

Fallback mode uses sparse context planning instead of true sparse model attention. Production mode replaces model attention execution with SSA-capable kernels.

See `docs/14_SSA_RUNTIME_SPEC.md`.

## Layer 5 — TSP Runtime

TSP is the memory planner for long sequences and limited device memory. It computes how sequence shards, weight shards, activation windows, and KV blocks should be scheduled.

Fallback mode uses budget-aware chunking and batch scheduling. Production mode integrates with custom WebGPU/native kernels.

See `docs/15_TSP_RUNTIME_SPEC.md`.

## Layer 6 — MTP / Speculative Decoding Runtime

The generation runtime supports a draft/target model pipeline:

```text
draft model proposes k tokens
  -> target model verifies in batch
  -> accepted prefix streams immediately
  -> rejection falls back to target token
  -> metrics update acceptance controller
```

Fallback mode is normal single-target decoding. Production mode uses optimized draft verification.

See `docs/16_MTP_SPECULATIVE_DECODING_SPEC.md`.

## Layer 7 — KVSwap Runtime

KVSwap manages KV cache pressure across VRAM, RAM, and disk. It controls pinning, eviction, prefetching, low-rank key summaries, and cache metadata. Low-rank summaries carry rank, projection id, layer/head grouping, block id, checksum, quality score, and a bounded numeric summary vector. The planner can score those summaries against a query summary to choose predicted hot KV blocks before attention, then trace `lowRankSummaryRank`, `predictedHotBlocks`, `prefetchedBlocks`, `prefetchHitRate`, `prefetchBytes`, `prefetchLatencyMs`, and `attentionStallMs`.

Fallback mode stores metadata and estimated cache blocks. Production mode persists and reloads real KV tensors through backend hooks. Browser proof now distinguishes exact prompt reuse, low-rank predictive prefetch, and miss/stall behavior, while remaining honest that async browser loads are scheduled ahead of decode rather than guaranteed GPU/disk overlap.

See `docs/17_KVSWAP_RUNTIME_SPEC.md`.

## Layer 8 — Inference Backend

Current backend strategy:

- `compiled-browser-webllm` is the compiled-first production answer candidate. It is selected by the Backend Broker for grounded/final answer tasks once its adapter is enabled and backend-specific Chrome proof passes.
- `unlocked-browser-transformer` is the custom WebGPU Kernel Lab. It owns converted Qwen manifests and weight shards in the browser, creates Q/K/V handles during prefill, runs sparse layer execution through the SSA backend, executes TSP callbacks, persists KV blocks in browser storage, and emits target-only/MTP lab proof traces.
- `wasm-small-core` is reserved for bounded control/fallback work such as memory tagging and context triage.
- Browser-vector memory is the open-source default. LanceDB sidecar and remote HTTP memory are scale layers, not required desktop-only defaults.

## First-class runtime rule

No application code should assume one model runtime is the whole product architecture. Application code calls the shared runtime/memory/context layer and the Backend Broker chooses the backend for the turn. Kernel Lab features stay in the research lane unless a backend-specific production gate proves quality, speed, and hot-path behavior for that backend.
