# 43 — Model Memory Security and Privacy

## Purpose

This document defines security and privacy requirements for model-native memory and GAC.

A memory-native agent has more power than a stateless chatbot. It must therefore have stricter controls over what is remembered, retrieved, compressed, exposed, deleted, and used for training.

## Security principles

1. Raw memory is sensitive by default.
2. Memory access must obey tenant, cell, and user policy.
3. Generated summaries are not source of truth.
4. User deletion must propagate to derived records.
5. Model-proposed memory actions require policy approval.
6. Training usage requires explicit policy.
7. Secrets must not be embedded or surfaced casually.

## Access control

Memory access should be scoped by:

- Tenant/user.
- Workspace.
- Cell.
- Project.
- Source type.
- Sensitivity label.
- Capability envelope.

A child cell should not automatically see all parent memory unless policy grants it.

## Sensitive memory labels

- `normal`
- `private`
- `legal`
- `financial`
- `security`
- `credential_metadata`
- `health`
- `minor_related`
- `business_confidential`
- `user_deleted`

These labels influence retrieval and model visibility.

## Secret handling

Secrets should not be stored as ordinary memory.

If a tool output contains secrets:

- Mask before embedding.
- Store encrypted if retention is required.
- Exclude from model-visible summaries.
- Store only metadata when possible.

## User deletion

When a user requests deletion:

1. Tombstone raw memory.
2. Delete or tombstone embeddings.
3. Invalidate representatives containing that memory.
4. Remove lineage entries.
5. Re-run consolidation for affected clusters.
6. Delete training examples if required.
7. Log deletion completion.

## Training privacy

Model-native GAC training data should be separated into:

- Local private training logs.
- User-consented workspace data.
- Synthetic/open data.
- Internal eval-only data.

Do not mix private user memory into shared models without explicit authorization.

## Prompt injection and memory poisoning

A malicious document may try to create false durable memory.

Mitigations:

- Source-type trust scores.
- Memory write policy.
- Require user confirmation for high-impact external instructions.
- Mark external content as untrusted.
- Do not let documents pin instructions unless user/system approves.

## Cross-cell contamination

If one cell goes rogue or ingests poisoned content, it must not contaminate global memory.

Use:

- Cell-scoped memory.
- Quarantine labels.
- Trust boundaries.
- Capability envelopes.
- Human review for global pins.

## Auditability

Every memory-visible answer should be traceable to:

- Raw memory ID.
- Source URI.
- Representative ID if used.
- Context pack trace.
- Model memory action if generated.

## Acceptance gates

- User deletion removes derived records.
- External documents cannot create global pins without approval.
- Sensitive memories are filtered by policy.
- Training exports exclude private memory by default.
- Every model memory action is auditable.
