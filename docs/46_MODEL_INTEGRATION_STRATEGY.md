# 46 — Model Integration Strategy

## Purpose

This document defines how the GAC subsystem integrates with the unlocked browser model lane and future custom models.

## Model categories

### Category A — Opaque API model

Examples:

- Hosted or local chat APIs that return text but do not expose model tensors.

Integration:

- Prompt-based memory action extraction.
- External GAC tool.
- Runtime-only memory control.
- Not a production route for SSA/KV/TSP/MTP.

### Category B — Tool-calling model

Integration:

- Structured memory tools.
- Tool schema for write/pin/split/compress/retrieve.
- Policy-gated execution.

### Category C — Adapter-tuned model

Integration:

- Fine-tuned memory action behavior.
- Better identity-risk classification.
- Still external memory store.

### Category D — Unlocked consolidation-native model

Integration:

- Dedicated heads.
- Retrieval and consolidation training objective.
- SSA/KVSwap metadata outputs.
- Browser-owned or native-owned model tensors with the `NativeSSABackendContract`.

## Prompt interface for Category A/B

The model should receive clear memory rules:

- Do not claim memory without source.
- Prefer exact raw memory for decisions.
- Use representative memory only as background.
- Propose pins for durable constraints.
- Ask for raw lineage when unsure.

## Tool interface

Tools:

- `memory.writeRaw`
- `memory.pinExact`
- `memory.fetchRawLineage`
- `memory.markContradiction`
- `memory.proposeClusterSplit`
- `memory.proposeConsolidation`
- `memory.requestRetrievalAudit`

## Model output validation

Every model memory action must be validated for:

- Schema correctness.
- Policy compliance.
- Source availability.
- Access rights.
- Deletion conflicts.
- Compression safety.

## Fine-tuning path

1. Collect memory action traces.
2. Label good/bad actions.
3. Train controller model.
4. Compare controller against base model prompt actions.
5. Fine-tune adapters only after enough evidence.

## Local edge constraints

For browser/edge models:

- Keep memory controller small.
- Prefer rule+metric path for first release.
- Avoid heavy training in browser.
- Use WebGPU inference only for lightweight classifiers if possible.

## Acceptance gates

- Generic model works with tool/runtime GAC.
- Tool-calling model improves structured actions.
- Adapter/controller is not required for MVP.
- Custom model path is documented but isolated from production dependency.
