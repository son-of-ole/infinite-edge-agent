# 14 — SSA Runtime Spec

## Role

SSA is the sparse attention subsystem. It is responsible for deciding which key/value blocks remain visible to attention and for providing the kernel/backend boundary where true sparse attention executes.

SSA is first-class. It is not only a research note, a prompt-packing trick, or an optional optimization.

## Target profile

This repo targets **SubQ-compatible SSA behavior** while using the public SSA literature and implementation path as the concrete engineering foundation.

The product target is:

```text
O(n) or subquadratic long-context inference by routing queries to a sparse subset of selected key/value blocks,
with pinned anchors, dense-reference validation, and contiguous block execution.
```

The public build foundation is:

- block sparse attention,
- sparse/full dual-stream alignment as a training reference,
- layer-level sparse routing,
- block-size/top-k inference budgets,
- CPU and WebGPU sparse-reference kernels,
- backend contract implemented by the repo-owned unlocked browser runtime and extendable to additional custom runtimes.

## Modes

| Mode | Meaning |
|---|---|
| `disabled` | Not allowed in production paths; tests only |
| `fallback_sparse_planner` | Mandatory planner selects sparse prompt/context blocks, while underlying model may still use dense attention over packed context |
| `webgpu_reference` | WebGPU kernels execute sparse tensor fixtures or toy layers for parity validation |
| `backend_native` | Inference backend supports true model-level sparse attention execution |
| `hybrid` | Native sparse execution for supported layers/chunks and fallback planning for unsupported regions |

## Runtime input

```ts
interface SSAPlanInput {
  requestId: string;
  activeBlocks: ContextBlock[];
  anchors: ContextAnchor[];
  memoryHits: MemorySearchHit[];
  maxBlocks: number;
  minAnchorScore: number;
  blockSize?: number;
  topKBlocks?: number;
  localWindowBlocks?: number;
}
```

## Runtime output

```ts
interface SSAPlan {
  mode: SSAMode;
  targetProfile: SSATargetProfile;
  selectedBlockIds: string[];
  pinnedBlockIds: string[];
  droppedBlockIds: string[];
  routingReasons: Record<string, string[]>;
  layerPolicies: SSALayerRoutingPolicy[];
  estimatedDenseTokens: number;
  estimatedSparseTokens: number;
  sparsityRatio: number;
}
```

## Routing policy

Selection priority:

1. System prompt and safety anchors.
2. Current user request.
3. Explicit user constraints.
4. Recent unresolved assistant/tool state.
5. LanceDB memory hits above threshold.
6. Document chunks with high provenance confidence.
7. Long-term preference/profile memory.
8. Older low-confidence history only when semantically relevant.

Every selected or dropped block must have an explainable route reason.

## Layer-level routing policy

The SSA runtime emits layer policies. This keeps application code independent from the eventual backend implementation.

```ts
interface SSALayerRoutingPolicy {
  layerIndex: number;
  blockSize: number;
  topKBlocks: number;
  localWindowBlocks: number;
  pinnedBlockIds: string[];
  selectedBlockIdsByQueryBlock: Record<number, string[]>;
  denseFallback: boolean;
}
```

## Kernel/backend boundary

A native SSA backend must accept:

- input token ids,
- block metadata,
- Q/K/V tensors or handles,
- sparse routing matrix,
- layer-level policy,
- pinned anchor mask,
- fallback dense regions,
- KVSwap prefetch hints.

```ts
interface SSAKernelBackend {
  supportsNativeSSA(): boolean;
  planSparseAttention(input: SSAPlanInput): Promise<SSAPlan>;
  executeSparseForward(input: SparseForwardInput): Promise<SparseForwardOutput>;
}
```

## Current implementation

`NativeEdgeReferenceBackend` implements the native SSA boundary for the repo-owned reference path. It initializes a model boundary, creates backend-owned deterministic Q/K/V and KV-cache handles, executes one tiny layer through the shared sparse-forward path, validates dense parity when requested, and decodes deterministically from the backend-owned cache.

`UnlockedBrowserTransformerBackend` is the browser production target. It loads the converted Qwen manifest, owns model tensors, creates Q/K/V handles during prefill/decode, and executes sparse layer forward through the shared SSA backend. Production quality is not claimed by fixture math alone; real-model browser generation must pass the release/browser smoke gates for the converted shard set.

## WebGPU kernel phases

The WebGPU path is defined from the start:

1. `blockSummary` — compute per-block K summaries.
2. `blockScoreTopK` — route query blocks to selected key blocks.
3. `kvGather` — gather selected K/V blocks into contiguous buffers.
4. `sparseAttention` — run masked sparse attention over gathered blocks.
5. `traceWrite` — emit selected block IDs and timing metrics.

See `docs/26_WEBGPU_SSA_KERNEL_PLAN.md`.

## Fallback behavior

Fallback mode is not the final SSA architecture, but it is still mandatory because it exercises the same routing contracts before native kernels are available.

Fallback mode:

- reduces packed context to high-value blocks,
- emits the same SSA plan shape as native mode,
- records omitted blocks,
- pins anchors,
- allows evals to measure routing quality,
- keeps the app ready for native SSA without a rewrite.

Fallback mode does **not** claim to alter model internals.

## Acceptance gates

- Needle retrieval over synthetic 100k-token packed context: >= 99% relevant block inclusion.
- No system/safety/current-user anchor dropped.
- Every dropped block has a reason.
- Dense/sparse CPU tensor reference exists.
- WebGPU sparse kernel matches CPU sparse reference on toy tensors.
- Native reference backend executes one layer through the SSA sparse-forward path without changing application code.
