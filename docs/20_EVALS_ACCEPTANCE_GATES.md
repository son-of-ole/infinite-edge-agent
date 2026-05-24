# 20 — Evals and Acceptance Gates

## Why evals are core

This architecture can easily become hand-wavy unless every subsystem has measurable gates. Evals must run locally and produce JSON artifacts.

## Eval suites

### Memory recall eval

Tests LanceDB/IndexedDB retrieval.

Metrics:

- recall@k,
- precision@k,
- query latency,
- provenance completeness,
- stale memory rate.

Gate:

- recall@10 >= 0.90 on seeded local corpus.
- 100% retrieved chunks include provenance.

### Context rebuild eval

Tests session startup and request-time context packing.

Metrics:

- required anchor inclusion,
- dropped critical context count,
- token budget compliance,
- rebuild latency.

Gate:

- 100% required anchors included.
- token budget never exceeded.

### SSA routing eval

Tests sparse block selection.

Metrics:

- relevant block inclusion,
- sparsity ratio,
- anchor preservation,
- dropped relevant block count.

Gate:

- 95% relevant block inclusion in synthetic needle tests.

### TSP planning eval

Tests memory budget planning.

Metrics:

- estimated VRAM/RAM,
- max safe context,
- degradation correctness,
- schedule validity.

Gate:

- no plan exceeds configured memory budget.

### MTP speculative decoding eval

Tests draft/target acceptance behavior.

Metrics:

- acceptance rate,
- net speedup ratio,
- rejection recovery correctness,
- disabled-when-worse behavior.

Gate:

- if speedup < 1.0 for configured window, speculation disables automatically.
- first committed MTP token must match target-only decode for the same prompt/cache state.
- MTP must never degrade output into empty marker-only text or invalid-token noise.

### KVSwap policy eval

Tests cache pin/evict/prefetch decisions.

Metrics:

- pinned eviction violations,
- pressure reduction,
- prefetch hit rate,
- deterministic policy output.

Gate:

- 0 pinned eviction violations.
- deterministic output for same trace/config.

### Real browser output eval

Tests the integrated browser app with the real unlocked model lane, not fixture weights.

Metrics:

- initialization success,
- active chat budget,
- time to first visible token,
- generated token count,
- response text quality smoke result,
- runtime proof mode,
- WebGPU/CPU coverage,
- memory provider mode.

Gate:

- fixture weights disabled.
- chat budget reports full prompt budget and long generation budget.
- one arbitrary prompt with empty or irrelevant memory streams coherent assistant text.
- response must not be only proof markers, markdown punctuation, repeated whitespace, or invalid-token fragments.
- proof metadata records SSA/KVSwap/TSP and MTP mode when enabled.
- CPU fallback is allowed only when the gate is not a strict WebGPU release gate and must be reported.

## Artifact outputs

Each eval writes:

```text
.artifacts/evals/<suite>/<timestamp>/results.json
.artifacts/evals/<suite>/<timestamp>/trace.jsonl
.artifacts/evals/<suite>/<timestamp>/summary.md
```
