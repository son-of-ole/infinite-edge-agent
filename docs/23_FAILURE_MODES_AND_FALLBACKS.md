# 23 — Failure Modes and Fallbacks

## Principle

Fallbacks keep the product usable; they do not remove the target architecture.

## LanceDB unavailable

Fallback:

- Use IndexedDB memory store.
- Mark memory mode as degraded.
- Queue writes for later LanceDB sync if sidecar returns.

User impact:

- Smaller memory scale.
- Slower vector search at large chunk counts.

## WebGPU unavailable

Fallback:

- WASM or smaller model.
- Reduce context budget.
- Disable native backend features.

User impact:

- Slower generation.
- Less active context.

## SSA native unavailable

Fallback:

- Sparse context planner.
- Dense inference over packed selected context.

User impact:

- Cannot claim true subquadratic model attention.
- Still benefits from sparse block selection.

## TSP native unavailable

Fallback:

- Static memory estimator and safe token budget.

User impact:

- Less efficient memory use.
- Fewer tokens packed.

## MTP draft model unavailable

Fallback:

- Target-only decoding.

User impact:

- Slower token generation.

## KV tensor handles unavailable

Fallback:

- Metadata-only KVSwap.
- Aggressive prompt/context packing.

User impact:

- Cannot page real KV tensors.
- Long sessions rely more heavily on memory rebuild.

## Context ledger corruption

Fallback:

- Recover from latest valid ledger snapshot.
- Rebuild memory summaries from LanceDB chunks.
- Quarantine corrupt entries.

User impact:

- Some provenance may be unavailable until repair completes.
