# Roadmap

## Phase 0 — Compile and run

- Install dependencies.
- Run browser app.
- Load local model.
- Send/receive chat messages.
- Emit a runtime trace for every response.

## Phase 1 — LanceDB-first durable memory

- Store chat chunks in IndexedDB for browser fallback.
- Store semantic memory in LanceDB sidecar for the primary local profile.
- Link vector rows to raw transcript, provenance, and context anchors.
- Retrieve by vector similarity and metadata filters.
- Clear/export/import memory.

## Phase 2 — Context Runtime

- Add active context graph.
- Add pinned anchors and unresolved task state.
- Add reflection and summary memory.
- Add context rebuild traces.
- Ensure no LLM call bypasses context rebuild.

## Phase 3 — SSA planner mandatory

- Block active context.
- Route every call through SSA planner.
- Preserve anchors.
- Emit selected/dropped block reasons.
- Run synthetic needle routing tests.

## Phase 4 — SSA reference implementation

- Add CPU dense attention reference.
- Add CPU sparse attention reference.
- Add sparse-vs-dense approximation metrics.
- Add WebGPU toy kernel dispatches for block summary, route scoring, KV gather, and sparse attention.
- Add CPU-vs-WebGPU sparse parity tests.

## Phase 5 — Repository/document ingestion

- Drag/drop files.
- Chunk code and markdown differently.
- Track file path, commit hash, language, and dependency edges.
- Add project namespaces.
- Feed document blocks into LanceDB and SSA context blocks.

## Phase 6 — Native advanced runtime bridge

- Add custom backend contract for Q/K/V tensor access.
- Wire SSA kernels into one transformer layer.
- Add KVSwap metadata simulation, then real tensor offload/prefetch.
- Add MTP speculative draft/verify path.
- Add TSP memory planner and backend schedule.
- Add runtime capability negotiation.

## Phase 7 — Production hardening

- Encrypted local storage.
- Signed releases.
- Model/license manager.
- Observability dashboard.
- CI/CD.
- Formal privacy review.
- Long-context eval suite.
