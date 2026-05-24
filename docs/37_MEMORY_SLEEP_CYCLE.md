# 37 — Memory Sleep Cycle

## Purpose

This document defines the sleep-cycle architecture for persistent cells.

A cell should be able to work, accumulate memory, sleep, consolidate, and wake up with continuity. The sleep cycle is where the cell converts raw activity into durable, structured memory without losing exact identity.

## Why sleep cycle matters

A long-running agent cannot keep all working context hot forever. Without a sleep cycle, it either forgets or accumulates noise.

The sleep cycle gives the cell a controlled moment to:

- Extract decisions.
- Pin constraints.
- Compress safe clusters.
- Preserve exact sources.
- Summarize low-risk background.
- Write wake context.
- Create training examples.
- Reduce runtime bloat.

## Trigger conditions

Sleep cycle can trigger when:

- User closes session.
- Cell is idle for configured time.
- Task completes.
- Project milestone completes.
- Token/memory pressure crosses threshold.
- Manual user command.
- Scheduled maintenance.
- Before deployment/export.

## Sleep-cycle stages

### Stage 1 — freeze active state

Capture:

- Current transcript.
- Active files.
- Tool results.
- Open tasks.
- Runtime errors.
- Model actions.
- Context pack traces.

### Stage 2 — extract memory candidates

Classify memory candidates:

- Decisions.
- User instructions.
- Preferences.
- Facts.
- Errors and fixes.
- Open questions.
- Generated artifacts.
- External references.

### Stage 3 — apply identity policy

Pin exact memories based on `33_IDENTITY_PRESERVATION_POLICY.md`.

### Stage 4 — cluster and score

Run GAC clustering and metric computation.

### Stage 5 — consolidate safe memory

Route clusters:

- centroid.
- medoid.
- residual medoid.
- split.
- no compression.

### Stage 6 — build wake context

Create a compact wake context with:

- Cell identity.
- Current project state.
- Exact pinned constraints.
- Recent decisions.
- Open tasks.
- Source pointers.
- Representative background.
- Known risks.

### Stage 7 — run retrieval probes

Test whether pinned facts and recent decisions are retrievable.

### Stage 8 — write sleep report

Store sleep report for audit and UI.

## Wake context format

The wake context is a derived memory record, not source of truth.

It should include sections:

- `Cell Identity`
- `Current Goal`
- `Pinned Constraints`
- `Decisions Since Last Wake`
- `Open Tasks`
- `Important Sources`
- `Memory Map`
- `Risks and Unknowns`
- `Next Suggested Actions`

Every factual item should link to raw memory IDs or source records.

## Wake process

When the cell wakes:

1. Load cell manifest.
2. Load latest wake context.
3. Retrieve identity pins.
4. Retrieve active task memory.
5. Pull representative background.
6. Build SSA context pack.
7. Warm KV cache for pinned/task memories if supported.
8. Start session.

## Sleep-cycle failure modes

### Bad summary replaces truth

Mitigation: wake context is derived; raw memory remains ground truth.

### Pin loss

Mitigation: retrieval probes must test pinned facts.

### Over-compression

Mitigation: high-risk clusters are not compressed.

### Contradiction hidden

Mitigation: contradiction links must be preserved and surfaced.

### User deletion ignored

Mitigation: deletion requests have highest job priority.

## Sleep-cycle outputs

- `sleep_cycle_run` record.
- `wake_context` derived memory.
- New identity pins.
- New representatives.
- Retrieval audit records.
- Training examples.
- Operations metrics.

## UX requirements

The user should be able to inspect:

- What the cell remembered.
- What it pinned.
- What it summarized.
- What it compressed.
- What it decided to revisit.
- What it forgot/deleted.

The UI should avoid exposing complex math by default but provide debug details for engineers.

## Acceptance gates

- Cell wakes with correct current goal.
- Pinned constraints survive sleep/wake.
- Open tasks survive sleep/wake.
- Wake context links to raw memory.
- Retrieval probes pass before marking sleep cycle complete.
- Failed sleep cycle does not corrupt prior wake context.
