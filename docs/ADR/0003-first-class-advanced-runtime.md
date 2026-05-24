# ADR 0003 — Advanced Runtime Features Are First-Class

## Status

Accepted.

## Context

The architecture depends on LanceDB, SSA, TSP, MTP/speculative decoding, KVSwap, and Context Runtime. Treating these as research adapters creates the wrong build path: the app would become a simple chat/RAG product with advanced features bolted on later.

## Decision

All advanced runtime features are Tier-0. They must exist in config, code contracts, telemetry, docs, and evals from the start.

## Consequences

- The app calls an Advanced Runtime Coordinator instead of directly calling an opaque chat API.
- Fallback modes are explicit and visible.
- Native kernel work can proceed without rewriting product code.
- Evals can track whether fallback modes are good enough for MVP.
