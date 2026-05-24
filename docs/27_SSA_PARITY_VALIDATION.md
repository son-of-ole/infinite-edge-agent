# 27 — SSA Parity and Validation

## Purpose

SSA must be proven, not assumed. This project needs an eval harness from the start because sparse attention can silently drop critical dependencies.

## Validation layers

```text
Prompt/block planner validation
  -> CPU dense vs CPU sparse tensor parity
  -> WebGPU sparse vs CPU sparse parity
  -> Backend layer parity
  -> End-to-end long-context task evals
```

## Test families

### 1. Routing tests

- Repeated motifs with one rare needle.
- Conflicting instructions where only one is current.
- Safety/system anchor preservation.
- User preference memory vs current explicit override.
- Codebase dependency across distant files.

Metrics:

```text
needle_block_recall >= 0.99
anchor_drop_count = 0
unsupported_selection_reason_count = 0
```

### 2. Dense/sparse tensor tests

Use tiny tensors so dense attention can be computed exactly.

Metrics:

```text
cosine(out_sparse, out_dense) >= threshold
max_abs_error <= tolerance
selected_attention_mass >= configured_minimum
```

Sparse attention is not expected to equal dense attention in all cases. The point is to measure approximation quality and fail when dropped attention mass is too large.

### 3. WebGPU kernel parity

For generated tensors:

```text
CPU sparse reference ~= WebGPU sparse kernel
```

Tolerance depends on f32/f16 path. Start with f32 for correctness.

Current gate: `packages/core/src/runtime/ssa_webgpu/webgpuToyKernels.test.ts` verifies the routed toy sparse-layer contract against the CPU sparse reference without requiring GPU in CI. Browser validation should additionally run the same sparse-attention fixture with `requireWebGpu: true` on a localhost or HTTPS page and require max absolute error below `1e-5`.

### 4. Backend integration tests

Once a native model backend exists:

- run one layer in dense mode,
- run same layer in sparse mode,
- compare output and logits,
- verify selected block trace,
- verify no pinned block is missing.

### 5. End-to-end agent tests

- Load a synthetic repository into LanceDB.
- Ask a question requiring distant files.
- Context Runtime pulls candidates.
- SSA planner/routes blocks.
- Agent answers with evidence IDs.

Acceptance:

```text
answer_supports_evidence = true
needle_dependency_recovered = true
no_pinned_constraints_dropped = true
```

## Failure classifications

| Failure | Meaning | Action |
|---|---|---|
| `ROUTING_MISS` | Relevant block was not selected | adjust scoring, anchors, or memory retrieval |
| `ANCHOR_DROP` | Required block was omitted | hard fail; pinning bug |
| `DENSE_PARITY_FAIL` | Sparse reference diverges too far | increase budget or fix kernel |
| `WEBGPU_PARITY_FAIL` | WebGPU differs from CPU sparse | shader bug or precision issue |
| `BACKEND_CONTRACT_FAIL` | backend lacks required tensor hook | cannot claim native SSA |
| `KV_PREFETCH_MISS` | selected block not available in active KV tier | KVSwap/prefetch bug |

## CI requirement

The repo should eventually add:

```text
scripts/eval-ssa-routing.ts
scripts/eval-ssa-tensors.ts
scripts/eval-ssa-webgpu.ts
scripts/eval-long-context-agent.ts
```

For now, `packages/core/src/runtime/ssa_webgpu` includes CPU references and route planners that can be unit-tested without GPU access.
