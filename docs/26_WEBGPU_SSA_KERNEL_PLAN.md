# 26 — WebGPU SSA Kernel Plan

## Purpose

This file defines the custom WebGPU kernel path for SSA. These kernels are included from the start so the project architecture is not blocked on opaque inference internals. They begin as reference kernels over toy tensors, then graduate into the unlocked browser model backend.

## Kernel pipeline

```text
Q, K, V tensors
  <- denseMatMul projection boundary
  -> blockSummary
  -> blockScoreTopK
  -> kvGather
  -> sparseAttention
  -> denseMatMul output projection / backend continuation
```

## Buffer model

| Buffer | Shape | Owner | Purpose |
|---|---:|---|---|
| `qBuffer` | `[queryTokens, heads, headDim]` | backend | Query vectors for current chunk |
| `kBuffer` | `[contextTokens, kvHeads, headDim]` | backend/KVSwap | Key cache |
| `vBuffer` | `[contextTokens, kvHeads, headDim]` | backend/KVSwap | Value cache |
| `blockSummaryBuffer` | `[blocks, kvHeads, headDim]` | SSA | Mean or learned summary per K block |
| `blockScoreBuffer` | `[queryBlocks, blocks]` | SSA | Routing score matrix |
| `selectedBlockBuffer` | `[queryBlocks, topK + pinned]` | SSA | Selected block ids |
| `gatheredKBuffer` | compact selected KV | SSA/KVSwap | Contiguous K blocks for attention |
| `gatheredVBuffer` | compact selected KV | SSA/KVSwap | Contiguous V blocks for attention |
| `outputBuffer` | `[queryTokens, heads, headDim]` | SSA/backend | Attention output |
| `projectionBuffer` | `[out, hidden]` | backend | Row-major dense projection weights for Q/K/V/O or other stable matrices |
| `activationBuffer` | `[tokens, hidden]` | backend | Dense projection input |

Stable dense projection buffers and compute pipelines may be reused by explicit cache keys inside one live WebGPU device/runtime lifetime. This cache is only an in-memory residency optimization; it is not durable storage and does not persist across reloads. The implementation only uses stable matrix buffers for descriptor-backed matrices with an explicit stable-cache policy; mutable plain arrays are uploaded per call even if a caller key is present. Shader modules and compute pipelines are cached per device for sparse attention, dense matvec, dense matmul, RMSNorm, residual add, and MLP entry points so hot decode paths stop recompiling the same kernels. Dense matmul, RMSNorm, sparse attention, residual add, and batched MLP can now return `webgpu_resident_tensor` handles and accept resident handles as inputs, so a toy transformer-layer chain can stay in GPU buffers until an explicit `readWebGpuResidentTensor()` or final top-k logit boundary. The real decode layer path now uses resident tensors for O projection, attention residual, post-attention RMSNorm, MLP, and MLP residual before materializing the layer hidden row for the existing KV/SSA state.

## Kernel 1: blockSummary

Goal: compress each K block into a routeable representation.

Default strategy:

```text
summary(block) = mean(K[tokenStart:tokenEnd])
```

Later strategies:

- learned block key projection,
- max/mean hybrid,
- anchor-aware summaries,
- topic/semantic block metadata fused from LanceDB.

## Kernel 2: blockScoreTopK

Goal: score query chunks against block summaries, select top-k previous blocks, then merge pinned anchors and local windows.

Score:

```text
score(queryBlock, keyBlock) = dot(mean(Q_queryBlock), summary(keyBlock))
```

Selection rules:

1. Causal mask: no attending to future blocks.
2. Pinned anchors are always included.
3. Local window blocks are included before global top-k if configured.
4. Remaining slots go to highest route scores.
5. Duplicate blocks are removed.

## Kernel 3: kvGather

Goal: transform sparse, non-contiguous K/V ranges into compact contiguous buffers before attention.

Why this matters:

- GPU attention kernels need contiguous memory for throughput.
- Fragmented per-head/per-token loads destroy effective bandwidth.
- KVSwap can prefetch just the selected blocks before gather.

## Kernel 4: sparseAttention

Goal: compute causal attention over only the gathered K/V blocks.

Pseudo formula:

```text
scores = Q @ K_selected.T / sqrt(headDim)
scores = applyCausalAndBlockMask(scores)
weights = softmax(scores)
out = weights @ V_selected
```

This kernel starts as a correctness-first toy implementation. Production optimization comes after parity.

## Dense projection boundary

Goal: provide one reusable kernel boundary for model projections that take an activation matrix and a row-major dense matrix:

```text
out[t, o] = dot(activation[t, 0:hidden], projection[o, 0:hidden])
```

Inputs:

- activations `[tokens, hidden]`
- projection `[out, hidden]`

Output:

- projected activations `[tokens, out]`

Current implementation status: Q/K/V prefill projection, decode-time Q/K/V projection, and prefill O projection use this boundary with deterministic CPU fallback. WebGPU execution uploads activations per call in the transformer integration and can reuse explicitly keyed stable projection buffers and compute pipelines for the live runtime/device lifetime. The lower-level dense matmul primitive also has a GPU-resident variant for projection chains: it returns a `webgpu_resident_tensor`, records `outputResident: true` and `readback: false`, and does not map GPU output unless the caller explicitly materializes it. The real decode O projection now takes a resident attention tensor and emits resident proof metadata before residual/MLP work. The unlocked decode logit path uploads the final hidden row as a resident tensor when WebGPU is active and feeds that resident buffer directly into tiled top-k projection. Trace metadata records backend, shape, purpose, projection-cache hits, pipeline-cache hits, `outputResident`, `readback`, and `vectorResident` where applicable.

## MLP boundary

Decode MLP uses a single-token WebGPU/CPU kernel boundary for Qwen-style SiLU-gated MLPs and non-gated GeLU MLPs. Prefill MLP now uses the same projection-cache contract but batches all prompt rows for one layer into a single two-stage WebGPU dispatch pair:

```text
intermediate[t, i] = silu(gate[i] dot hidden[t]) * (up[i] dot hidden[t])
out[t, h] = down[h] dot intermediate[t]
```

The batched path reduces prompt-load dispatch/readback churn from one MLP kernel call per prompt token to one MLP kernel call per layer. Proof traces include `tokens`, `inputSize`, `intermediateSize`, `outputSize`, activation kind, projection-cache hits, and pipeline-cache hits. The resident primitive path now proves RMSNorm -> projection -> sparse attention -> output projection -> residual add -> batched MLP -> residual add -> top-k logits with zero intermediate readbacks in the WebGPU test harness. The real decode path now uses the same resident O projection/residual/RMSNorm/MLP/residual segment before materializing the layer hidden row. The next deeper transformer-wide residency milestone is resident RoPE/head normalization and packed-head assembly for the real Q/K/V attention path.

## Logit projection boundary

Decode logits use the dense matvec boundary for both candidate rows and full-vocab tiled top-k rows. Candidate mode keeps the original token IDs alongside the compact logits and reports selected-row proof metadata, but it is now an explicit debug/budget override. The production default scans the full vocabulary in tiles, keeps only top-k rows, reports `full_vocab_topk_logit_projection`, and can reuse the dense-matvec compute pipeline plus stable output-projection tile buffers during a live runtime session.

## WebGPU constraints

- WGSL shaders cannot allocate dynamic memory.
- Top-k requires fixed maximum `topKBlocks` per dispatch or a multi-pass selection strategy.
- WebGPU buffer alignment must be explicit.
- Large context routing should be chunked by query block.
- Browser watchdogs can terminate long-running GPU work; dispatches must be bounded.
- Exact dense parity is only feasible on small fixtures.

Current implementation status: `packages/core/src/runtime/ssa_webgpu/shapeBuckets.ts` defines prompt-length, selected-block-count, head-dimension, and tile-row buckets so nearby dynamic prompt shapes share stable static pipeline keys. It also builds prefill chunk plans with bounded token windows and estimated per-dispatch cost metadata. Browser benchmark controls can request deterministic long prompts and strict long-prompt proof. Strict proof now requires the runtime to execute the chunk plan and report chunked dispatch proof instead of accepting planning-only metadata.

## Default constants

```ts
export const DEFAULT_SSA_WEBGPU_CONFIG = {
  blockSize: 16,
  topKBlocks: 16,
  localWindowBlocks: 2,
  pinnedAnchorBudget: 8,
  headDim: 128,
  maxQueryBlocksPerDispatch: 64,
  maxContextBlocksPerDispatch: 8192
};
```

## Development stages

### Stage A — CPU references

- `denseReferenceAttention`
- `sparseReferenceAttention`
- `selectSparseBlocksForQueryBlock`

### Stage B — WebGPU toy kernels

- Run kernels over generated tensors.
- Compare output against CPU sparse reference.
- No model integration yet.

Current implementation status: the sparse-attention toy kernel runs in browser WebGPU against tiny Q/K/V fixtures, accepts per-query selected key indexes, and has a CPU fallback path for deterministic CI tests. Browser validation on localhost confirmed `navigator.gpu` execution with max absolute error below `1e-6` against the CPU sparse reference. This is a compute proof of concept; production claims require real-model browser generation parity on the converted Qwen shard set.

### Stage C — backend bridge

- Backend exposes Q/K/V for one layer.
- SSA kernels execute that layer.
- Backend resumes layer stack.

### Stage D — prefill acceleration

- Use sparse routing during prefill over long input.
- Store selected-block traces for eval.

Current implementation status: unlocked prefill now executes causal attention through the sparse-attention kernel boundary per packed attention head, using dense causal selected-key indexes. The proof is correctness-first, not a FlashAttention-class optimization. KV cache handles include prefill projection and attention backend proof metadata so tests can confirm the path.

Sparse attention with an empty or fully invalid selected-key row returns a zero vector with the query/head dimension on the CPU fallback path, matching the WebGPU shader behavior and avoiding ragged hidden states. The WebGPU sparse-attention params carry the explicit `scale` value used by the CPU fallback; non-default scales should not silently revert to `1 / sqrt(headDim)`.

Prefill also now records chunk metadata (`prefillChunkCount`, `prefillChunkSize`, `shapeBucket`, `pipelineCacheKey`, `maxDispatchEstimatedMs`, and `prefillChunkDispatch`) on SSA traces, browser benchmark artifacts, browser preview payloads, and the prefill proof handle. For multi-chunk plans, causal prefill attention slices query-token windows, dispatches sparse attention per packed head per chunk, awaits between chunk windows, and reports `attentionDispatchCount` plus `awaitedDispatchBreaks`. This is still correctness-first sparse attention, not a FlashAttention-class optimization, but strict long-prompt proof now means chunk dispatch actually ran.

### Stage E — decode integration

- Use SSA routing for incremental decode with KVSwap prefetch.
- Keep anchors, recent window, and relevant memory blocks pinned.

## Required traces

Every native SSA pass must write a trace:

```ts
export interface SSAKernelTrace {
  requestId: string;
  layerIndex: number;
  queryBlockIndex: number;
  selectedBlockIds: string[];
  pinnedBlockIds: string[];
  denseTokenCountEstimate: number;
  sparseTokenCountEstimate: number;
  routingMs: number;
  gatherMs: number;
  attentionMs: number;
  prefillChunkCount?: number;
  prefillChunkSize?: number;
  shapeBucket?: string;
  pipelineCacheKey?: string;
  maxDispatchEstimatedMs?: number;
}
```

These traces are mandatory for debugging hallucinations, dropped-context failures, and performance regressions.
