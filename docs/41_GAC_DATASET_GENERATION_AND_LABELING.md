# 41 — GAC Dataset Generation and Labeling

## Purpose

This document defines how to generate datasets for training and evaluating the model-native GAC controller.

The dataset must teach the system to preserve exact identity, not merely semantic similarity.

## Dataset types

### 1. Raw memory event dataset

Input:

- Message.
- Tool output.
- File chunk.
- Code diff.
- Decision log.

Labels:

- Should write memory.
- Memory kind.
- Importance.
- Identity risk.
- Pin recommendation.

### 2. Cluster consolidation dataset

Input:

- Cluster member texts.
- Embeddings.
- Cluster metrics.
- Retrieval threshold.
- Budget.

Labels:

- centroid.
- medoid.
- medoid plus residuals.
- split.
- no compression.

### 3. Identity preservation dataset

Input:

- Query.
- Original raw memory.
- Representative memory.
- Distractor memories.

Labels:

- Identity preserved.
- Correct memory ID.
- Failure mode.

### 4. Source-grounding dataset

Input:

- Model claim.
- Context pack.
- Available sources.

Labels:

- Source required.
- Exact source required.
- Representative sufficient.
- Unsupported claim.

### 5. Sleep-cycle dataset

Input:

- Full session transcript.
- Tool results.
- Final state.

Labels:

- Wake context.
- Pinned decisions.
- Open tasks.
- Safe summaries.
- Raw source links.

## Synthetic examples

Generate hard pairs where small changes matter.

### Numeric perturbations

- `$30K` vs `$300K`
- `May 11` vs `May 12`
- `1M tokens` vs `12M tokens`

### Negation perturbations

- `Use Sandbox` vs `Do not use Sandbox`
- `SSA is research-only` vs `SSA is first-class`

### Entity perturbations

- `LanceDB` vs `IndexedDB`
- `opaque chat API` vs `WebGPU`
- `Vercel` vs `Neon`

### Code perturbations

- Function name changes.
- Package version changes.
- Config flag changes.
- API parameter changes.

## Human labeling guidelines

Labelers should answer:

1. Would losing the exact wording change future behavior?
2. Is this a durable decision or temporary statement?
3. Could a summary safely replace it?
4. Does this memory contain numbers, dates, names, URLs, or code symbols?
5. Is there contradiction with another memory?
6. Should the system ask for clarification instead of merging?

## Automatic labels

Use policy rules for seed labels:

- Pin if contains explicit "remember", "from now on", "do not", or architecture decision.
- High risk if contains URLs, money, dates, code symbols.
- Low risk if near duplicate with high cosine similarity and no protected terms.

Use GAC metrics for cluster labels.

Use retrieval audits for outcome labels.

## Dataset split

Use project/cell-based split, not random row split, to avoid leakage.

Recommended:

- 70% train.
- 15% validation.
- 15% test.

Hold out entire projects and users for test.

## Evaluation labels

Track:

- Correct memory action.
- Correct pin decision.
- Correct compression strategy.
- Correct source requirement.
- Correct raw memory retrieval.
- Correct sleep-cycle decision extraction.

## Data privacy

Training data generated from user memory must obey privacy settings.

For open-source/shared datasets, use synthetic data or consented anonymized project data.

## Acceptance gates

- Dataset contains hard negatives.
- Dataset includes exact identity tests.
- Dataset separates semantic similarity from identity equivalence.
- Dataset records source lineage.
- Dataset has no hidden user-private leakage in shared mode.
