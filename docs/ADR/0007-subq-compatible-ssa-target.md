# ADR 0007 — Adopt SubQ-Compatible SSA Target with Public SSA Implementation Path

## Status

Accepted

## Context

The product goal requires model-level sparse attention for massive active context. SubQ is the desired product-level target because its public positioning describes a fully subquadratic sparse-attention model class for long-context tasks. However, SubQ's exact technical report and model weights are not available for direct implementation.

A public implementation-complete reference exists in the SSA repository and paper, which define Sparse Sparse Attention using sparse/full attention streams, block sparse routing, and bidirectional alignment.

## Decision

The project will implement a **SubQ-compatible SSA target** rather than claiming to clone SubQ.

The implementation path is:

1. Use the public SSA repo/paper as the engineering foundation.
2. Define SSA as a Tier-0 runtime subsystem.
3. Start with fallback sparse planning for prompt/context packing.
4. Add CPU dense/sparse references.
5. Add WebGPU block-summary, routing, KV-gather, and sparse-attention kernels.
6. Integrate with a native/custom inference backend only when the backend exposes Q/K/V and KV-cache ownership.

## Consequences

- Product docs can talk about SubQ-compatible behavior without implying access to proprietary internals.
- The current browser scaffold remains runnable.
- The backend boundary is explicit and testable.
- Native SSA cannot be claimed until a model layer actually executes sparse attention through the kernel path.

## Non-goals

- Reimplementing proprietary SubQ internals.
- Pretending prompt-level retrieval equals model-level sparse attention.
- Depending permanently on a backend that cannot expose Q/K/V tensors.
