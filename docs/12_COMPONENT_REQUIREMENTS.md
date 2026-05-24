# 12 — Component Requirements

## LanceDB Memory Engine

Must:

- Store all durable semantic memory.
- Support vector search, metadata filters, session filters, document filters, and source filters.
- Store raw text pointer, chunk text, embedding, token count, importance, recency, source, and provenance.
- Return scored memory hits plus retrieval trace.
- Support background writes so generation is not blocked.

Should:

- Support hybrid vector + full text search.
- Support table migrations.
- Support local sidecar and desktop embedded modes.

## SSA Runtime

Must:

- Build sparse attention plans from active context blocks.
- Track anchors, selected blocks, dropped blocks, and routing reason.
- Expose a kernel boundary for true SSA execution.
- Provide fallback sparse planning if kernel support is unavailable.

Should:

- Prefer system prompt anchors, current task anchors, memory hits, recent turns, and unresolved constraints.
- Emit sparsity ratio and retrieval accuracy metrics.

## TSP Runtime

Must:

- Estimate memory pressure by sequence length, hidden size, layer count, KV precision, and batch size.
- Produce a fold plan for tensor shards and sequence shards.
- Tell the context runtime how many tokens can fit in current hardware mode.
- Expose a backend schedule boundary.

Should:

- Support WebGPU limit detection.
- Support native edge runtime mode for desktop bundles.

## MTP / Speculative Decoding Runtime

Must:

- Support draft model config.
- Support target model config.
- Track drafted, accepted, rejected, and corrected tokens.
- Fall back to target-only decoding.
- Emit acceptance rate, latency gain, and quality flags.

Should:

- Dynamically tune speculative token count.
- Disable speculation when acceptance rate collapses.

## KVSwap Runtime

Must:

- Model KV blocks by layer, sequence range, tier, pin status, importance, and last access.
- Pin system prompt, safety policy, active task, recent turns, and high-importance retrieved memory.
- Evict low-importance blocks when memory pressure exceeds thresholds.
- Prefetch predicted blocks before verification/generation.
- Expose hooks for real tensor paging.

Should:

- Maintain compressed key summaries.
- Group KV entries to match disk/page block behavior.

## Context Runtime

Must:

- Own the session ledger.
- Rebuild context at session start.
- Pack prompt/context for the inference backend.
- Persist runtime traces.
- Maintain provenance from user/document/memory source to generated answer.

Should:

- Support replay.
- Support compaction and summary generation.
- Support selected DOM/text/file elements as explicit context frames.
