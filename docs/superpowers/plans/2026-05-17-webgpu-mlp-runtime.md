# WebGPU MLP Runtime Slice

## Goal

Move the next unlocked-browser production frontier from candidate-logit matvec acceleration into the decode MLP schedule step by adding a reusable WebGPU MLP runner with deterministic CPU fallback.

## Scope

- Keep the current unlocked Qwen manifest path, browser-owned Q/K/V tensors, SSA sparse attention boundary, KV paging, TSP schedule execution, and candidate-logit matvec proof.
- Add a reusable single-token MLP runner for Qwen-style gated MLP (`silu(gate(x)) * up(x) -> down`) and legacy GeLU MLP (`gelu(up(x)) -> down`).
- Preserve deterministic CPU fallback for CI and browsers without WebGPU.
- Route the decode TSP `mlp` callback through the runner when a layer has MLP weights.
- Surface proof metadata on decode handles and the web client so tests can prove the MLP step used the kernel boundary.
- Do not claim optimized persistent GPU buffers, full projection matmul acceleration, prefill attention acceleration, or full-vocab logit acceleration yet.

## Tasks

1. Add focused kernel tests proving CPU fallback MLP parity for gated and GeLU paths, plus WebGPU preparation/upload proof on a fake device.
2. Implement `runMlpWebGpu` next to the existing sparse-attention and dense-matvec runners.
3. Integrate the MLP runner in `UnlockedBrowserTransformerBackend.decode` while preserving the existing CPU prefill/reference path.
4. Extend decode proof metadata through `UnlockedBrowserDecodeHandle` and `UnlockedBrowserTransformerClient.lastDecodeProof`.
5. Update docs to describe the completed decode MLP kernel boundary and keep remaining production expansion honest.
6. Run focused core/web tests, typecheck, release gate, and browser preview.

## Done

- Decode layers with MLP weights execute the TSP `mlp` step through `runMlpWebGpu` with CPU fallback or browser WebGPU.
- Tests prove parity with the existing reference MLP math and prove WebGPU preparation uses the expected matrices/shader path.
- Decode proof includes MLP backend metadata.
- Browser preview initializes, generates an `[unlocked:ssa-kv-tsp]` response, and returns to Ready.
