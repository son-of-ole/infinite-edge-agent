# ADR 0011 — Consolidation Head Is a First-Class Research Target

## Status

Accepted for research roadmap

## Context

A model-native memory system requires the model or controller to predict memory operations.

## Decision

Define a consolidation head/controller as a first-class research component.

It predicts:

- write.
- pin.
- compress.
- split.
- fetch raw lineage.
- mark contradiction.
- demote.

## Consequences

- Training objectives are required.
- Model memory action logs must be collected.
- Policy gate remains mandatory.
