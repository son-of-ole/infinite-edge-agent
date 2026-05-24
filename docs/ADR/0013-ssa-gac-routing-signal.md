# ADR 0013 — GAC Signals Feed SSA Routing

## Status

Accepted

## Context

Sparse attention can drop rare but important facts if routing only uses semantic relevance.

## Decision

SSA routing must consume GAC metadata including identity risk, pin strength, source trust, and cluster spread.

## Consequences

- Context blocks need metadata.
- SSA debug traces must record GAC scores.
- Identity pins can override normal sparse selection.
