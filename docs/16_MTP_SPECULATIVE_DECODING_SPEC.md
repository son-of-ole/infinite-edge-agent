# 16 — MTP / Speculative Decoding Spec

## Role

MTP/speculative decoding reduces autoregressive latency by using a fast draft model to propose tokens and a target model to verify them.

## First-class requirement

Speculative decoding is part of the response runtime from day one. It may run in target-only fallback mode, but request traces, config, metrics, and model pairing must exist immediately.

## Modes

| Mode | Meaning |
|---|---|
| `target_only` | No draft model; normal decoding |
| `draft_verify` | Draft model proposes tokens; target verifies |
| `tree_draft_verify` | Draft model proposes token tree; target verifies multiple branches |
| `backend_native` | Inference backend provides optimized speculative API |

## Config

```json
{
  "mtp": {
    "enabled": true,
    "mode": "draft_verify",
    "draftModelId": "small-local-drafter",
    "targetModelId": "main-ssa-model",
    "numSpeculativeTokens": 2,
    "minAcceptanceRate": 0.45,
    "disableWhenLatencyWorse": true,
    "vocabCompression": {
      "enabled": true,
      "topTokenCount": 32000
    }
  }
}
```

Browser-native draft windows default to 2 tokens and clamp to 3 tokens because the normal client-side concurrency profile is 1. Larger windows belong to explicit native/server or future high-concurrency profiles, not the default browser proof lane.

## Runtime loop

```text
while generation not complete:
  draft = draftModel.propose(k tokens)
  verification = targetModel.verify(previous + draft)
  acceptedPrefix = sampler.accept(draft, verification)
  stream acceptedPrefix
  if rejection:
    stream target correction
  update acceptance metrics
  retune k if needed
```

## Required metrics

- `draftTokens`
- `acceptedTokens`
- `rejectedTokens`
- `acceptanceRate`
- `draftLatencyMs`
- `verifyLatencyMs`
- `netSpeedupRatio`
- `disabledReason`

## Acceptance gates

- Target-only fallback produces correct output.
- Speculative mode can be toggled without changing app code.
- If a real target-only baseline is available and net speedup ratio falls below 1.0 for N requests, runtime disables speculation and reports why. When a backend cannot measure that baseline without corrupting KV state, it must report the latency-disable policy as unsupported instead of fabricating a speedup/slower decision.
- Acceptance rate is recorded per model pair and per task type.

## Current implementation

`verifySpeculativeBatch` accepts multiple draft branches for a request, sends them through one verifier backend, applies target verification per branch, and emits branch traces plus scoped metrics for model pair and task type. It rejects duplicate branch IDs, backend request mismatches, and missing branch results before those failures can silently corrupt generation traces.

For the unlocked browser transformer, production is target-only by default. MTP is a lab feature enabled only with `VITE_MTP_ENABLED=true`; when enabled, the browser draft path defaults to `browser/qwen-prefix-drafter`, a tokenizer-compatible shallow Qwen draft backend created from the same manifest, weights, tokenizer, SSA/KV/TSP decode path, and WebGPU preference as the target. `VITE_MTP_DRAFT_LAYER_COUNT` controls how many leading target-family layers the draft backend uses. The older `browser/ngram-drafter` remains available as a deterministic local proof/fallback source built from recent prompt/generated token IDs. Arbitrary `VITE_MTP_DRAFT_MODEL_ID` values fall back to target-only until a real compatible draft backend is registered.

The unlocked target backend verifies a draft window through `verifySpeculativeDraft`: one browser-owned continuation pass over `[previousToken, ...draftPrefix]`, the same SSA/KV/TSP kernel boundaries as decode, and an accepted-prefix commit back into the live KV cache. Decode proofs report MTP mode, draft source (`qwen_prefix_draft` or `local_tokenizer_ngram`), draft token IDs/text, accepted/rejected/corrected counts, acceptance rate, verifier backend, `verifierStrategy: "batched_continuation"`, verified token count, target decode call count, committed input count, metrics, and any disabled reason. The Qwen-prefix browser path reports `latencyDisablePolicy: "paired_benchmark_required"` because a single speculative batch is not enough evidence to disable the feature; acceleration is claimed only when the paired target-only benchmark clears the configured acceptance and net-speedup floors.
