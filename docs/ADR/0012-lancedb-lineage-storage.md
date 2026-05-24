# ADR 0012 — LanceDB Stores Representatives and Lineage

## Status

Accepted

## Context

The existing architecture already uses LanceDB as the primary memory engine. GAC requires vector representatives and lineage tables.

## Decision

Use LanceDB for raw memory embeddings, representatives, cluster metrics, and lineage records where practical.

A browser-only IndexedDB fallback may exist, but LanceDB is the target memory engine.

## Consequences

- GAC can query representatives efficiently.
- Context Runtime can fetch lineage.
- Evals can audit retrieval.
- Schema migrations must be versioned.
