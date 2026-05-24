# 25 — SubQ-Compatible SSA Target and Public Foundation

## Purpose

This document converts the SSA target into build requirements. SSA is not a vague future research feature. It is a first-class runtime layer that every long-context turn must pass through.

The project target is **SubQ-compatible behavior**, not a claim that this repo reimplements SubQ internals. SubQ's public material describes a subquadratic sparse-attention model for 12M-token reasoning, but the formal technical report is marked as coming soon and the exact routing/indexer implementation is not public. The build therefore uses the public SSA literature and implementation as the concrete engineering path.

## Product-level target

The desired runtime capability is:

```text
Given a very large active context, route each query/layer to a sparse subset of relevant key/value blocks,
keep anchor constraints always available,
execute attention over compact contiguous KV blocks,
and validate quality against a dense-attention reference path.
```

This is the central difference between the real target and ordinary RAG. RAG retrieves chunks before inference. SSA changes the attention computation itself.

## Public SSA foundation

The closest implementation-complete public reference is `zhenyi4/ssa`, the official repository for **SSA: Sparse Sparse Attention by Aligning Full and Sparse Attention Outputs in Feature Space**.

The public SSA foundation gives us these buildable ideas:

1. **Block sparse attention** — split context into blocks and select the top-k relevant blocks for each query region.
2. **Sparse and full streams** — train with both sparse and full attention pathways to reduce the mismatch between sparse inference and full-attention behavior.
3. **Bidirectional alignment** — sparse attention output is regularized toward full attention output, while full attention output is encouraged to become more sparse-compatible.
4. **Flexible inference budgets** — sparse inference can vary block size and selected-block count without changing application code.
5. **LLaMA-NSA style architecture** — the public repo uses a custom `llama-nsa` architecture with parameters such as `block_size`, `block_counts`, `window_size`, and `inference_mode`.

## What cannot be assumed

Do not assume:

- SubQ's actual lightning indexer is public.
- SubQ weights are available.
- Opaque browser chat APIs expose all tensor hooks required for native SSA.
- Standard browser inference engines can execute sparse SSA kernels without custom backend work.
- A prompt-level sparse packer is equivalent to model-level sparse attention.

## Required model/backend contract

A production SSA backend must expose the following capability boundary:

```ts
export interface NativeSSABackendContract {
  readonly backendName: string;
  readonly supportsQkvAccess: true;
  readonly supportsLayerSparseRouting: true;
  readonly supportsPinnedKvBlocks: boolean;
  readonly supportsDenseReferenceMode: boolean;

  initializeModel(modelId: string): Promise<void>;
  prefill(inputTokenIds: Int32Array, policy: SSAPrefillPolicy): Promise<SSAPrefillHandle>;
  executeSparseLayer(input: SparseLayerForwardInput): Promise<SparseLayerForwardOutput>;
  decode(input: SSADecodeInput): Promise<SSADecodeOutput>;
  dispose(): Promise<void>;
}
```

The app must not depend on inference-library-specific internals. It depends on this contract. The current production profile is `unlocked-browser-transformer`; additional backends must satisfy the same contract before they can claim SSA.

## Browser/WebGPU adaptation strategy

The public SSA implementation depends on Python, PyTorch, CUDA, FlashAttention, flash-linear-attention, and native-sparse-attention. The browser build cannot reuse those kernels directly. The port must implement the same logical stages as WebGPU compute passes:

1. **Block summary pass** — produce per-block key summaries.
2. **Routing/scoring pass** — score query chunks against previous block summaries.
3. **Top-k selection pass** — choose selected KV blocks plus pinned anchors.
4. **KV gather pass** — copy selected K/V blocks into compact contiguous GPU buffers.
5. **Sparse attention pass** — compute Q × selected-K, masked softmax, and weighted V.
6. **Dense-reference test path** — compare sparse output against dense output on small contexts.

## Layer-level sparsity rule

The build target is layer-level sparse routing, not ad-hoc head-level truncation. Head-level mixed lengths can fragment memory access and undermine GPU bandwidth. The SSA planner therefore emits a layer policy:

```ts
export interface SSALayerRoutingPolicy {
  layerIndex: number;
  blockSize: number;
  topKBlocks: number;
  localWindowBlocks: number;
  pinnedBlockIds: string[];
  selectedBlockIdsByQueryBlock: Record<number, string[]>;
  denseFallback: boolean;
}
```

The backend can use the same selected blocks across heads in a layer, while still allowing future specialization by head group if the kernel supports it.

## Minimum viable SSA path

The initial build sequence is:

1. Fallback SSA planner selects blocks before prompt packing.
2. Dense-reference module computes exact dense output for tiny fixtures.
3. Sparse-reference module computes block-sparse attention on CPU/TypeScript for tiny fixtures.
4. WebGPU kernels implement block scoring and KV gather.
5. WebGPU sparse attention is introduced for toy tensor fixtures.
6. Backend bridge wires sparse kernels to a model with Q/K/V access.
7. Full decode loop uses SSA for prefill and decode.

## Files implementing this spec

```text
packages/core/src/runtime/ssa.ts
packages/core/src/runtime/ssa_webgpu/types.ts
packages/core/src/runtime/ssa_webgpu/blockRouter.ts
packages/core/src/runtime/ssa_webgpu/denseReference.ts
packages/core/src/runtime/ssa_webgpu/sparseReference.ts
packages/core/src/runtime/ssa_webgpu/webgpuSsaBackend.ts
packages/core/src/runtime/ssa_webgpu/webgpuToyKernels.test.ts
packages/core/src/runtime/ssa_webgpu/wgsl/blockSummary.wgsl.ts
packages/core/src/runtime/ssa_webgpu/wgsl/blockScoreTopK.wgsl.ts
packages/core/src/runtime/ssa_webgpu/wgsl/kvGather.wgsl.ts
packages/core/src/runtime/ssa_webgpu/wgsl/sparseAttention.wgsl.ts
```

## Acceptance gates

| Gate | Requirement |
|---|---|
| Routing recall | Synthetic needle block included >= 99% at 100k-token simulation |
| Anchor safety | System/safety/user constraints are never dropped |
| Dense parity | Small fixture sparse output cosine similarity >= configured threshold vs dense oracle |
| Explanation | Every selected/dropped block has route reasons |
| Backend swap | Application code does not change when moving from fallback to native SSA backend |
| Kernel parity | WebGPU sparse-reference outputs match CPU sparse-reference within tolerance |

## Build warning

A browser app can implement SSA planning and WebGPU tensor kernels, but true SSA for an LLM requires a model/backend that exposes layer tensors and KV-cache ownership. Until that backend exists, the app runs the same contracts in fallback mode and records traces so the native backend can be integrated without rewriting the application.
