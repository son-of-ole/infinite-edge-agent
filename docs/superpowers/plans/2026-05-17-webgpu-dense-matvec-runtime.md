# WebGPU Dense MatVec Runtime Slice

## Goal

Move the next unlocked-browser production frontier from CPU-only projection/logit math toward WebGPU by adding a reusable dense matvec kernel boundary with deterministic CPU fallback, then wire it into candidate-logit decode.

## Scope

- Keep the full Qwen manifest path and existing SSA/KV/TSP tensor-control proof.
- Add a general WebGPU dense matvec runner that accepts runtime matrix rows and optional row selection.
- Preserve CPU fallback for CI and browsers without WebGPU.
- Use the runner for candidate-logit projection in `UnlockedBrowserTransformerBackend.decode`.
- Surface trace/proof metadata so tests and browser validation can prove when candidate logits used WebGPU/CPU matvec instead of the old untracked CPU loop.
- Do not claim full WebGPU MLP or full all-layer prefill acceleration yet.

## Tasks

1. Add focused kernel tests for dense matvec CPU fallback and selected-row projection parity.
2. Implement the reusable dense matvec runner alongside the existing SSA WebGPU runner.
3. Integrate selected-row logit projection in the unlocked transformer decode path and expose backend trace metadata.
4. Update web/client proof and docs so the runtime reports the new matvec acceleration slice honestly.
5. Run focused tests, strict release gate, and browser preview.

## Done

- Candidate-logit decode can use the WebGPU dense matvec boundary with CPU fallback.
- Existing full-vocab logits still work when no candidate set is supplied.
- Tests prove candidate-token ID mapping and backend metadata.
- Browser preview initializes, generates an `[unlocked:ssa-kv-tsp]` response, and returns to Ready.
