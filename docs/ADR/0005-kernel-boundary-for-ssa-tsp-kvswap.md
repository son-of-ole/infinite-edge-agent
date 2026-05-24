# ADR 0005 — Kernel Boundary for SSA, TSP, and KVSwap

## Status

Accepted.

## Context

SSA, TSP, and real KV tensor paging require inference backend support. Standard application TypeScript cannot safely modify transformer internals by itself.

## Decision

Define stable kernel/backend boundaries now. Implement fallback planners for MVP. Add native/WebGPU/edge backends behind the same contracts.

## Consequences

- The UI and agent logic remain stable while kernels evolve.
- Runtime traces distinguish planner mode from native mode.
- Native backends can be tested independently.
