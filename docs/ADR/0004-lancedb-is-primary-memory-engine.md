# ADR 0004 — LanceDB Is the Primary Memory Engine

## Status

Accepted.

## Context

The product requires durable cross-session memory, semantic retrieval, metadata filters, and scale beyond small browser-only demos.

## Decision

LanceDB is the primary durable vector memory engine. IndexedDB is a fallback and browser-only queue.

## Consequences

- Local development should run the memory sidecar by default.
- Desktop/edge production bundles should embed or launch LanceDB locally.
- Memory APIs must not depend on IndexedDB-only semantics.
