# 33 — Identity Preservation Policy

## Purpose

This document defines what memory identity means and which memories must be protected from unsafe compression.

Identity preservation is the difference between remembering a topic and remembering the exact fact that mattered.

## Definition

A memory has preserved identity when the system can retrieve and present the exact original fact, decision, instruction, source, or constraint that the user or agent needs.

A memory has collapsed identity when the system retrieves only a vague summary, similar topic, or representative that no longer preserves the original fact.

## Always pin categories

The following categories must default to identity-pinned unless policy explicitly overrides them.

### User instructions

Examples:

- Preferred writing style.
- Explicit technical constraints.
- Permanent preferences.
- "From now on" statements.
- "Do not" statements.

### Architecture decisions

Examples:

- Stack choices.
- Security boundaries.
- Deployment constraints.
- Runtime requirements.
- Feature decisions.
- ADR outcomes.

### Legal and financial facts

Examples:

- Contract clauses.
- Settlement numbers.
- Attorney advice notes.
- Dates and deadlines.
- Prices and limits.

### Names, URLs, dates, money, IDs

These facts are fragile under summarization.

Examples:

- Domain names.
- GitHub repos.
- File names.
- Model names.
- Commit SHAs.
- Deadlines.
- Costs.

### Code and build facts

Examples:

- Error messages.
- Fixes.
- Commands that worked.
- Package versions.
- API contracts.
- Schema decisions.

### Source of truth records

Examples:

- Uploaded documents.
- User-provided text.
- Tool outputs.
- Calendar/email/file records.
- Repository files.

## Conditional pin categories

Pin these when relevant to active task or high importance:

- User goals.
- Project ideas.
- Long-running research insights.
- Product names.
- UX preferences.
- Past evaluation outcomes.

## Compression-safe categories

These can usually be compressed if geometry says the cluster is tight:

- Repeated greetings.
- Repeated confirmations.
- Near-duplicate chunks.
- Low-importance progress notes.
- Redundant summaries with lineage.
- Bulk low-risk background material.

## Identity risk scoring

Initial score:

| Signal | Risk |
|---|---:|
| Contains exact number/date/money | +0.20 |
| Contains URL/file path/code symbol | +0.20 |
| User says remember/from now on/do not | +0.30 |
| Legal/financial/security topic | +0.30 |
| Architecture decision | +0.25 |
| Source document excerpt | +0.20 |
| Repeated duplicate | -0.15 |
| Low informational content | -0.25 |

Cluster geometry then adjusts the score.

High spread increases risk. Tight clusters reduce risk only if there are no protected categories.

## Compression authority

The model may recommend compression, but policy must approve it.

The model may recommend pinning, and policy should usually allow pinning unless it violates privacy or retention rules.

User explicit memory deletion overrides identity preservation.

## Source lineage requirement

Any compressed or generated memory shown to the model must include one of:

- Raw source ID.
- Representative ID with lineage.
- Generated summary ID with source pointers.

No anonymous memory is allowed in production context packs.

## User correction handling

When the user corrects memory:

1. Create a new raw memory with the correction.
2. Mark old memory as superseded, not deleted unless requested.
3. Pin the correction.
4. Add contradiction link.
5. Re-run retrieval probes.

## Privacy handling

Identity preservation does not mean permanent retention.

If user asks to forget something, the system must delete or tombstone raw and derived records according to product policy.

If memory contains sensitive data, it may be pinned but encrypted, masked, or excluded from model-visible context unless required.

## Acceptance gates

- Pinned memory can be retrieved exactly.
- User instructions are never replaced by broad summaries.
- Generated summaries cannot become source of truth.
- User deletion is honored across raw, representative, and embedding layers.
- Corrections supersede old memory without hiding the audit trail from authorized debugging tools.
