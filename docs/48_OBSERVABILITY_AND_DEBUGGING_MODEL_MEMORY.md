# 48 — Observability and Debugging for Model Memory

## Purpose

This document defines observability for GAC, model-native memory actions, context packing, SSA routing, and KVSwap priority.

## Debugging question

For any answer, engineers must be able to answer:

**Why did the model remember this, retrieve this, compress this, route this, cache this, or forget this?**

## Required traces

### Memory ingestion trace

Records:

- Source event.
- Raw memory ID.
- Embedding ID.
- Pin decision.
- Policy version.

### Consolidation trace

Records:

- Cluster ID.
- Metrics.
- Strategy decision.
- Representatives written.
- Lineage.
- Validation probes.

### Context pack trace

Records:

- Query.
- Candidate memories.
- Included memories.
- Omitted memories.
- Token budget lanes.
- Source requirements.

### Model memory action trace

Records:

- Proposed action.
- Model confidence.
- Policy decision.
- Execution result.

### SSA routing trace

Records:

- Candidate blocks.
- Selected blocks.
- Dropped blocks.
- GAC boosts.
- Must-attend blocks.

### KVSwap trace

Records:

- Block tier changes.
- Evictions.
- Prefetches.
- Cache misses.
- GAC priority reasons.

## UI views

### Memory Inspector

Shows raw memory, representative, lineage, pins, and sensitivity labels.

### Consolidation Dashboard

Shows compression ratio, cluster risk, failed probes, and job status.

### Context Pack Viewer

Shows exactly what entered the model.

### Answer Provenance Panel

Shows which memories/sources supported the answer.

### Sleep Cycle Report

Shows what the cell remembered, pinned, summarized, and left open.

## Alerts

Alert on:

- Pinned retrieval failure.
- Representative missing lineage.
- User deletion stuck.
- High false merge rate.
- Sleep-cycle failure.
- Memory action policy rejection spike.
- Cache evicted required pin.

## Logs

Logs must include trace IDs but avoid exposing raw sensitive memory by default.

Use redaction for:

- Secrets.
- Sensitive user data.
- Legal/financial text.

## Acceptance gates

- Every answer has context pack trace.
- Every representative has lineage trace.
- Every model memory action has policy trace.
- Engineers can reproduce memory failures from traces.
- Sensitive logs are redacted by default.
