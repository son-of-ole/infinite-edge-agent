# ADR 0014 — GAC Signals Feed KVSwap Priority

## Status

Accepted

## Context

Long-context KV caches require tiering. Recency-only eviction can drop old but critical facts.

## Decision

KVSwap priority must consume GAC metadata including identity risk, pin strength, task relevance, and representative substitutability.

## Consequences

- Cache policy becomes memory-aware.
- Pinned task-critical facts stay hot/warm.
- Cache traces must explain GAC-driven tier choices.
