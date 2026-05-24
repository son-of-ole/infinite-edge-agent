# Browser MTP Verifier Runtime

## Goal

Close the next production blocker in the unlocked browser runtime: MTP/speculative decoding should move from target-only fallback to a configured browser verifier path when the unlocked backend is selected.

The implementation must stay production-honest. It may use a lightweight tokenizer-compatible browser drafter as the first draft source, but it must not claim a separate neural draft model unless one is actually configured. The unlocked backend must own target verification and KV-cache mutation.

## Current State

- SSA, KVSwap, and TSP can report enabled for the unlocked browser transformer.
- MTP reports enabled for the unlocked browser transformer when the configured browser draft profile is compatible and the backend advertises verifier batching.
- Core `verifySpeculativeBatch` already validates batched draft branch verification and metrics.
- The web runtime can stream through a browser MTP verifier loop or target-only fallback from `UnlockedBrowserTransformerClient.streamChat`.

## Scope

1. Runtime configuration
   - [x] Add MTP configuration fields to the shared runtime config.
   - [x] Add browser env parsing for MTP enablement, draft id, speculative token count, min acceptance rate, and latency-disable behavior.
   - [x] Register a tokenizer-compatible browser drafter profile for the unlocked Qwen target when configured.
   - [x] Mark unlocked backend capabilities with speculative verifier batching.

2. Unlocked browser client
   - [x] Add MTP options to `UnlockedBrowserTransformerClient` and its worker client messages.
   - [x] Implement a deterministic browser draft source that proposes token IDs from recent prompt/candidate vocabulary without adding network dependencies.
   - [x] Verify draft tokens through the unlocked target backend, committing accepted tokens through the real KV cache path and emitting one correction token on rejection.
   - [x] Preserve normal target-only generation when MTP is disabled or auto-disabled.
   - [x] Prevent unbounded work: respect max generation tokens, speculative token caps, runtime layer caps, and bounded logit candidate sets.

3. Proofs and traces
   - [x] Extend decode proof metadata with MTP mode, draft model id/source, draft tokens, accepted/rejected counts, acceptance rate, verifier backend, and disabled reason when applicable.
   - [x] Feed runtime plan MTP config into `RuntimeTrace.runtime.mtp` so StatusPanel can show MTP enabled when configured.
   - [x] Keep docs honest that this first browser drafter is a tokenizer-compatible local draft source, not a separate neural draft model.

4. Tests and gates
   - [x] Add core tests proving the advanced runtime reports MTP enabled only with a compatible configured draft and backend verifier capability.
   - [x] Add web client tests for speculative proof metadata, accepted/corrected token accounting, bounded max-token behavior, and target-only fallback.
   - [x] Update production readiness tests/docs if new production env is required.
   - Ensure focused tests, typechecks, release gate, and browser preview pass.

## Non-Claims

- This does not introduce a second neural draft model by default.
- This does not claim speedup unless acceptance metrics and latency prove it.
- The original browser-local n-gram drafter accepted only `browser/ngram-drafter`; the active follow-up adds `browser/qwen-prefix-drafter` as the default tokenizer-compatible shallow Qwen draft path, while arbitrary draft ids still fall back target-only until a compatible backend is registered.
- Acceptance-threshold fallback is supported. Latency-worse auto-disable is reported as unsupported for this browser path unless a future non-mutating target-only baseline is added.

## Done Criteria

- MTP status can show enabled for unlocked browser transformer when configured with the browser drafter and verifier batching capability.
- Browser decode proof includes speculative verification metrics when MTP is enabled.
- Target-only behavior remains available and tested.
- Full release gate passes with the configured Qwen manifest.
- Browser preview initializes and returns `[unlocked:ssa-kv-tsp]` without errors, with MTP visible as enabled when env config enables it.
