# 22 — Agent Handoff Build Order

## Instruction to build agents

Do not build a simple chat app first and “add memory later.” Build the runtime skeleton first.

## Order of operations

### 1. Runtime feature registry

Create a registry with these required features:

- `lancedb`
- `contextRuntime`
- `ssa`
- `tsp`
- `mtp`
- `kvswap`
- `inferenceBackend`

The app should not start without a registry report.

### 2. Memory engine

Implement the local memory interface with:

- IndexedDB fallback.
- LanceDB sidecar.
- common search/upsert API.

### 3. Context ledger

Before model generation, every user event is written to the ledger. After generation, assistant response and runtime trace are written.

### 4. Context rebuild

Build the context frame list from:

- current user input,
- recent turns,
- memory hits,
- summaries,
- pinned anchors.

### 5. TSP budget call

Before prompt packing, call TSP planner to get safe token budget.

### 6. SSA block selection

After frames are built, call SSA planner to select active blocks.

### 7. KVSwap policy call

Before generation, call KVSwap to pin/evict/prefetch cache blocks.

### 8. MTP decoding wrapper

Generation must go through speculative decoding wrapper, even in target-only fallback.

### 9. Inference backend

Only now call the unlocked inference backend.

### 10. Trace persistence

Write the full runtime trace after each response.

## Forbidden shortcuts

- Do not call an opaque model API directly from UI components.
- Do not store memory only in chat history.
- Do not skip runtime traces.
- Do not hide fallback status.
- Do not remove SSA/TSP/KVSwap/MTP from config because native kernels are pending.
