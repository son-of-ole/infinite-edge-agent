# WebGPU Packed SSA Decode

## Goal

Move the unlocked Qwen packed-head decode path onto the existing SSA kernel boundary so model-layer sparse attention no longer bypasses the WebGPU/CPU sparse attention runner.

## Scope

- Keep the existing correctness-first transformer math and real Qwen manifest path.
- Preserve CPU fallback for deterministic CI.
- Use the existing `runSparseAttentionWebGpu` kernel runner for each packed attention head.
- Surface enough trace metadata to prove the packed decode path used the SSA kernel backend.
- Do not claim optimized tiled WebGPU matmul/MLP yet.

## Tasks

1. Add regression coverage showing packed-head decode reports the kernel backend used for sparse attention and still matches the existing per-head sparse attention behavior.
2. Refactor `executePackedHeadSparseLayer` to call the SSA sparse attention runner per packed head, combining per-head outputs back into Qwen packed projection layout.
3. Update docs to describe this completed slice and keep the remaining frontier honest: projection matmul, MLP, and broader logit parity still need optimized GPU work.
4. Run focused tests, strict asset verification, release gate, and browser sanity validation.

## Done

- Packed-head decode uses `runSparseAttentionWebGpu` with CPU fallback or browser WebGPU.
- Tests pass for packed-head behavior and trace metadata.
- Release gate passes with `RELEASE_REQUIRE_UNLOCKED_QWEN_PARITY=true`.

## Completion Notes

- Packed-head sparse decode now runs the SSA sparse-attention kernel boundary once per attention head and reassembles the packed Qwen projection row layout.
- Direct multi-row `executeSparseLayer` calls preserve one sparse output row per query.
- Decode traces report the actual last query block route, route-specific selected block ids, sparse token count, and per-head backend proof.
