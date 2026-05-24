# ADR 0010 — Raw Memory and Lineage Are Required

## Status

Accepted

## Context

If derived summaries or representatives replace source memory, the system eventually loses truth.

## Decision

Consolidation may create representatives, summaries, and wake contexts, but it may not hard-delete raw memory.

Every representative must link to raw memory through lineage.

User deletion is the exception and must propagate to derived records.

## Consequences

- Storage cost is higher.
- Auditability is preserved.
- Source-grounding is possible.
- Memory repair and re-consolidation are possible.
