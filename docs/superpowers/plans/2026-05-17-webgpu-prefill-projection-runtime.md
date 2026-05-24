# WebGPU Prefill/Projection Frontier

## Goal

Move the unlocked browser runtime beyond decode-only kernel proofs by adding reusable runtime-lifetime WebGPU buffer residency, full projection matmul coverage, prefill attention kernel boundaries, and broader Qwen logit parity evidence.

This is still the unlocked route: the browser runtime must own model tensors, Q/K/V/KV handles, SSA routing, KVSwap events, TSP callbacks, and decode proof traces. Do not route back to opaque MediaPipe/WebLLM generation APIs.

## Scope

1. Add a reusable WebGPU dense matmul kernel boundary.
   - Input: an activation matrix `[tokens, hidden]` and row-major projection `[out, hidden]`.
   - Output: `[tokens, out]`.
   - Include deterministic CPU fallback and trace metadata.
   - Use it for prefill Q/K/V/O projection paths where possible.

2. Add runtime-lifetime GPU buffer reuse for stable dense matrices.
   - This is not durable storage across browser reloads.
   - Cache only within the live WebGPU/device/runtime lifetime.
   - Avoid unsafe stale-cache behavior for mutable plain arrays. Prefer explicit cache keys or stable descriptor-backed matrices.
   - Preserve CPU fallback and existing fake-device testability.

3. Expand logit projection parity.
   - Full-vocab decode logits should route through the dense matvec kernel boundary instead of the old untraced CPU-only reference path.
   - Candidate-token logit projection should continue to return original token ids and selected-row proof metadata.
   - Proofs should report backend, selected-row/full-row count, and purpose metadata.

4. Add prefill attention kernel proof.
   - Prefill should execute causal attention through the sparse-attention kernel boundary per packed attention head, with dense causal selected-key indexes.
   - Preserve numerical behavior against the existing dense reference path.
   - Emit prefill proof metadata on the KV cache handle so tests can confirm projection and attention backends.

5. Update tests and docs.
   - Add focused core tests for dense matmul fallback/preparation, GPU buffer reuse, full-vocab logit projection proof, and prefill proof traces.
   - Update web client tests when proof shape changes.
   - Update `docs/53_UNLOCKED_BROWSER_RUNTIME.md` and `docs/26_WEBGPU_SSA_KERNEL_PLAN.md` so the docs describe the new boundary honestly.

## Done Criteria

- Focused core runtime tests pass. Done for `webgpuToyKernels.test.ts` and `unlockedBrowserTransformer.test.ts`.
- Focused web client tests pass. Done for `unlockedBrowserTransformerClient.test.ts`.
- Core and web typechecks pass. Done.
- Release gate passes with configured Qwen manifest env.
- Browser preview initializes `unlocked-browser-transformer` and returns `[unlocked:ssa-kv-tsp]` without runtime errors.

## Non-Claims

- Runtime-lifetime GPU buffer reuse is not a durable database or persistent GPU memory across refreshes.
- Runtime-lifetime GPU buffer reuse is only enabled for stable descriptor-backed dense matrices with an explicit stable-cache policy. Mutable plain arrays are uploaded per call.
- Prefill attention acceleration is a WebGPU sparse-attention boundary over dense causal routes; it is not yet an optimized FlashAttention-class kernel.
- Real manifest browser decode remains candidate-bounded by default until tiled/top-k full-vocab projection is implemented.

## Implementation Notes

- Added `runDenseMatMulWebGpu` with deterministic CPU fallback, trace metadata, and a correctness-first WGSL compute path.
- Added `WebGpuRuntimeBufferCache` for explicitly keyed dense projection buffers within one live WebGPU device/runtime lifetime.
- Routed unlocked Q/K/V projection and prefill O projection through dense matmul; routed full-vocab logits through the dense matvec boundary instead of the old untraced reference path.
- Routed prefill causal attention through sparse attention per packed head and attached prefill projection/attention proof metadata to the KV cache handle.
- Updated the web client decode proof to expose logit projection full-row count and purpose metadata.
- Fixed review issues by bounding real-manifest browser decode logits by default, disabling runtime buffer cache reuse for mutable plain arrays, and making sparse CPU fallback return head-width zero rows for empty/invalid selections.
