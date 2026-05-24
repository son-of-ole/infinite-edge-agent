# ADR 0008 — Adopt Geometry-Aware Memory Consolidation

## Status

Accepted

## Context

The agent needs persistent long-term memory. Naive summarization and centroid compression can destroy exact identity while preserving broad topic similarity.

The Geometry of Consolidation work provides a practical framing: cluster geometry determines whether compression can preserve retrieval identity.

## Decision

Adopt Geometry-Aware Memory Consolidation as a first-class memory subsystem.

GAC will decide whether clusters use centroid, medoid, medoid plus residuals, split, or no compression.

## Consequences

- Memory compaction becomes auditable.
- Raw memory remains immutable.
- Representatives require lineage.
- Context Runtime consumes GAC metadata.
- Evals must track identity recall, not only semantic similarity.

## Alternatives rejected

- Summarization-only memory.
- Centroid-only vector compression.
- Raw top-k retrieval only.
- Treating GAC as later research rather than initial architecture.
