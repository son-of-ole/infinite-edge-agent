# ADR 0006 — Speculative Decoding Uses a Draft/Target Model Pair

## Status

Accepted.

## Context

Autoregressive decoding is latency-sensitive. Speculative decoding can improve throughput when a draft model has sufficient acceptance rate.

## Decision

The runtime supports a draft/target model pair from day one. Target-only mode is treated as fallback.

## Consequences

- Config must include draft model settings.
- Evals must track acceptance rate and net speedup.
- Runtime can disable speculation when it hurts latency.
