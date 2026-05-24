# 00 — Scope and Reality Check

## Product goal

Build a local-first, persistent AI agent that runs primarily on edge devices and preserves useful context across sessions. The product combines browser inference, persistent local memory, long-context planning, and cache/runtime management.

## Architectural identity

This project is not a stateless chatbot and not just a RAG wrapper. It is a persistent runtime intelligence system.

The model is the reasoning engine, but it is not the whole agent. The agent is the runtime system around the model:

```text
Agent =
  Runtime
  + Memory System
  + Context Rebuilder
  + Attention Routing
  + Cache Management
  + Consolidation
  + Planning
  + Model
```

Therefore:

```text
Model != Memory
Model != Runtime
Model != Agent
```

The model only sees the current active tokens and live KV cache. Persistence, continuity, identity, memory lifecycle, retrieval strategy, context reconstruction, and long-horizon execution are runtime responsibilities.

## Dream-state goal

The north star is the fully unlocked browser runtime, not a chatbot wrapper and not an opaque hosted/model API. A completed system must:

- Load user-provided, licensed Qwen-class transformer artifacts in the browser through the unlocked manifest/shard format.
- Own tokenization, prompt packing, Q/K/V projection, attention routing, KV cache state, decode logits, and stop-token handling inside the browser runtime.
- Keep browser-vector IndexedDB/OPFS memory as the open-source default while allowing LanceDB or compatible remote HTTP memory as optional scale layers.
- Run every turn through context rebuild, SSA routing, KVSwap planning/persistence, TSP schedule execution, and MTP draft/target verification when enabled.
- Produce coherent arbitrary responses even when no memory has been stored yet.
- Support long outputs through explicit generation budgets and long prompts up to the model context window, with sparse/pinned behavior beyond dense budgets called out honestly.
- Emit runtime proof metadata showing which SSA/KVSwap/TSP/MTP paths actually executed and whether WebGPU or CPU fallback was used.
- Ship acceptance artifacts and browser-preview proof before any production claim.

Anything less than that is a staged milestone, not the final product.

## Memory hierarchy

Infinite memory does not mean putting everything into the prompt. It means managing memory across layers with explicit movement, compression, priority, and provenance.

The target memory hierarchy is:

1. Long-term semantic memory: LanceDB, browser-vector, raw memory, lineage, and retrieval audits. Persistent, large, mostly cold.
2. Consolidated memory: GAC representatives, summaries, medoids, residuals, identity pins, and lineage. Compact, curated, and structured.
3. Active retrieval set: recent chat, relevant artifacts, current goals, pinned constraints, task state, and selected memories for the current request.
4. Active model context: the actual tokens and sparse/pinned blocks exposed to the model for this inference cycle.
5. KV cache: the live neural working state used during generation and speculative verification.

The Context Runtime is responsible for assembling layer 4 from layers 1-3, while SSA, KVSwap, TSP, and MTP decide how attention, cache residency, tensor scheduling, and decode strategy should operate over that working set.

## Per-turn runtime flow

Every user turn must pass through the runtime, not directly into the model:

1. User message arrives.
2. Message enters the runtime event path.
3. Runtime updates session transcript, active goals, task graph, and execution state where available.
4. Message embedding is queued or computed.
5. Memory retrieval executes across semantic, pinned, lineage, task-state, and identity sources where configured.
6. GAC evaluates representative memories, raw exact memories, identity-risk memories, and cluster expansion rules.
7. Context Runtime builds a working memory set, token budget plan, memory priority map, and source lineage map.
8. SSA planner builds sparse routing, block priority, and attention allocation plans.
9. KVSwap planner builds hot/warm/cold cache, prefetch, and eviction-protection plans.
10. MTP planner builds speculative decode and draft/target verification strategy.
11. TSP planner builds memory sharding and activation layout plans.
12. Final active context is assembled.
13. Prompt, routing metadata, and runtime options are sent to the inference runtime.
14. Generation occurs.
15. Runtime captures traces for retrieval, routing, dropped context, token allocation, cache decisions, generation metrics, and backend coverage.
16. Memory ingestion and sleep/consolidation jobs are scheduled where enabled.

The rebuild step is temporary working cognition assembly. It does not rewrite model weights and does not replace durable memory.

## Corrected architectural stance

The advanced pieces are **not research notes** and are **not optional future plugins**. They are first-class runtime subsystems from the first commit:

- LanceDB memory engine.
- SSA sparse attention runtime boundary.
- TSP folding parallelism planner.
- MTP/speculative decoding runtime.
- KVSwap cache tiering runtime.
- Context runtime with ledger, provenance, session rebuild, and prompt/context packing.

The practical build still requires staged implementation. Some capabilities need custom kernels, model support, or low-level inference runtime control that opaque browser chat APIs do not expose. The repo therefore uses a strict boundary pattern:

```text
first-class subsystem contract
  -> fallback planner or compatibility mode
  -> measurable acceptance gate
  -> production backend integration
```

A fallback is allowed only when it lets the app run while preserving the final runtime contract.

## What can be implemented immediately

- Web app shell.
- Unlocked Qwen browser transformer inference through `unlocked-browser-transformer`.
- IndexedDB fallback memory.
- LanceDB local sidecar or remote-compatible memory where configured.
- Local embedding worker.
- Semantic memory schema.
- Context rebuild pipeline.
- Runtime feature registry.
- SSA/TSP/KVSwap/MTP planners and telemetry contracts.
- Evals for memory recall, context rebuild, latency, and cache policy behavior.

## What requires deeper engine work

- Optimized SSA attention inside the full production model beyond the current correctness-first sparse kernel boundary.
- TSP folded tensor/sequence execution beyond the current explicit schedule callback boundary.
- Native KV tensor paging from GPU memory to CPU/disk in browser runtime beyond exact-match OPFS/IndexedDB KV reconstruction.
- Tuned target/draft pairing beyond the current browser Qwen-prefix drafter plus batched target continuation verifier; default speedup still needs device-specific acceptance and net-throughput proof.
- Real-model browser generation parity and speed gates against the converted Qwen shard set, not only fixture-backed math checks.

These are not excluded. They are represented by stable interfaces and acceptance gates so the team can implement the product while kernel/runtime work proceeds in parallel.

## Scope boundaries

This repository does not claim that a normal browser TypeScript app can independently replace transformer attention kernels. Instead, it defines the complete product/runtime architecture needed to integrate such kernels as they become available or as the team builds them.

## Definition of done for v1

A v1 release is complete when:

1. The app runs locally.
2. The agent can store and retrieve memory across sessions.
3. Browser-vector memory works by default, with LanceDB or a compatible remote API available as scale layers.
4. The real unlocked browser model lane can answer arbitrary prompts coherently without relying on seeded memory or prompt-specific candidate hacks.
5. The chat path is not artificially capped at tiny prompt/output budgets; production defaults expose full prompt budget and long generation budget.
6. The runtime exposes SSA, TSP, MTP, KVSwap, and Context Runtime status.
7. Fallbacks are explicit and measured.
8. All advanced runtime decisions are recorded in the context ledger.
9. The eval suite reports recall, latency, cache pressure, speculative acceptance, real-model parity, browser output proof, and context rebuild quality.
