# ADR 0002: Split browser and LanceDB memory stores

## Status

Accepted.

## Context

The target architecture wants local vector memory. Browser storage and native embedded databases have different constraints.

## Decision

Implement two memory stores behind the same interface:

1. IndexedDB store for browser-only MVP.
2. LanceDB sidecar for local desktop/edge profile.

## Consequences

Positive:

- MVP works without installation.
- Scalable local vector search is available when sidecar runs.
- The app can fall back gracefully.

Negative:

- Two storage paths must be tested.
- Sidecar packaging requires operational/security hardening.
