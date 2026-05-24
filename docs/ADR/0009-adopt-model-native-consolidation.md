# ADR 0009 — Adopt Model-Native Consolidation Direction

## Status

Accepted for architecture; staged for implementation

## Context

The user asked whether consolidation can be built directly into the language model, similar to earlier approaches that trained models to use tools such as calculators.

The system needs a memory-native model/runtime that can reason about writing, retrieving, compressing, pinning, and preserving memory.

## Decision

Adopt model-native consolidation as the long-term direction.

Implementation will be staged:

1. Tool-native GAC.
2. Runtime-native GAC.
3. Controller model.
4. Adapter-tuned model.
5. Full consolidation-native transformer research.

## Consequences

- Model memory actions become structured outputs.
- Actions are policy-gated.
- Logs become training data.
- Model weights are not the primary storage for user memory.
- External memory remains source-grounded and deleteable.

## Alternatives rejected

- Fine-tuning all user memories into model weights.
- Treating memory as plain RAG only.
- Letting model write/delete memory without policy gate.
