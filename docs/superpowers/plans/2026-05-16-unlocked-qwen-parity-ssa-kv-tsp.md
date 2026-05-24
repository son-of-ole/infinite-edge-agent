# Unlocked Qwen Parity + SSA/KV/TSP Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move the unlocked browser runtime from first-head control proof to full Qwen geometry with real browser-owned Q/K/V, RoPE, incremental KV state, and SSA/KVSwap/TSP execution over those tensors.

**Architecture:** Keep MediaPipe Gemma as a fallback only. The production full-control lane is `unlocked-browser-transformer`, with model geometry and tokenizer metadata in the sharded manifest, browser-owned Qwen tensors in the core backend, and strict verifier/release gates that prove the real path is configured. Use CPU reference math for correctness first and keep the existing WebGPU sparse-attention bridge active through `WebGpuSsaReferenceBackend`.

**Tech Stack:** TypeScript, Vite, Vitest, local Hugging Face safetensors converter, browser `fetch`, f32 shards, WebGPU sparse-attention reference bridge.

---

### Task 1: Full Qwen Manifest Geometry

**Files:**
- Modify: `scripts/unlockedModelConverter.ts`
- Modify: `scripts/unlockedModelConverter.test.ts`
- Modify: `scripts/verify-unlocked-asset.ts`
- Modify: `docs/53_UNLOCKED_BROWSER_RUNTIME.md`

- [ ] **Step 1: Write failing converter tests**

Add assertions that converted Qwen manifests include:

```ts
expect(manifest).toMatchObject({
  numAttentionHeads: 2,
  numKeyValueHeads: 1,
  maxPositionEmbeddings: 128,
  ropeTheta: 1000000,
  tieWordEmbeddings: true,
});
expect(manifest.conversion.projectionMode).toBe("full-qwen-gqa-rope");
expect(manifest.layers[0].qProj.shape).toEqual([4, 2]);
expect(manifest.layers[0].kProj.shape).toEqual([2, 2]);
expect(manifest.layers[0].vProj.shape).toEqual([2, 2]);
expect(manifest.layers[0].oProj.shape).toEqual([2, 4]);
```

Run:

```bash
npm exec --yes pnpm@9.15.0 -- test:converter
```

Expected: FAIL because metadata is missing and projections are first-head slices.

- [ ] **Step 2: Convert full Qwen projection shapes**

Read `num_attention_heads`, `num_key_value_heads`, `max_position_embeddings`, `rope_theta`, and `tie_word_embeddings` from `config.json`. Emit full Q/K/V/O shard shapes:

```ts
qProj: [numAttentionHeads * headDim, hiddenSize]
kProj: [numKeyValueHeads * headDim, hiddenSize]
vProj: [numKeyValueHeads * headDim, hiddenSize]
oProj: [hiddenSize, numAttentionHeads * headDim]
```

Set `conversion.projectionMode` to `"full-qwen-gqa-rope"`.

- [ ] **Step 3: Update strict verifier**

Add `--require-qwen-parity` / `RELEASE_REQUIRE_UNLOCKED_QWEN_PARITY=true` to require full attention metadata, RoPE metadata, and full Q/K/V/O shapes.

- [ ] **Step 4: Update docs**

Replace first-head caveats with the new full-geometry status, while keeping tokenizer/chat-template and performance caveats honest if a later task has not landed yet.

- [ ] **Step 5: Run converter tests**

Run:

```bash
npm exec --yes pnpm@9.15.0 -- test:converter
```

Expected: PASS.

### Task 2: Full Qwen Runtime Geometry

**Files:**
- Modify: `packages/core/src/runtime/unlockedBrowserTransformer.ts`
- Modify: `packages/core/src/runtime/unlockedBrowserTransformer.test.ts`
- Modify: `apps/web/src/lib/llm/unlockedBrowserTransformerClient.ts`
- Modify: `apps/web/src/lib/llm/unlockedBrowserTransformerClient.test.ts`

- [ ] **Step 1: Write failing runtime tests**

Add tests proving:

```ts
// Full GQA shapes are accepted.
expect(qHandle.matrix[0]).toHaveLength(4);
expect(kHandle.matrix[0]).toHaveLength(4); // expanded from one KV head to two Q heads.

// RoPE changes non-zero odd/even pairs by position.
expect(layer0KAtPosition1).not.toEqual(layer0KAtPosition0);

// Full O projection maps concatenated attention heads back to hidden size.
expect(readUnlockedBrowserDecodeHandle(decoded.logitsHandle).logits.every(Number.isFinite)).toBe(true);
```

Run:

```bash
npm exec --yes pnpm@9.15.0 -- --filter @infinite-edge-agent/core test -- src/runtime/unlockedBrowserTransformer.test.ts
```

Expected: FAIL because the backend validates first-head shapes.

- [ ] **Step 2: Add attention metadata to weights**

Add optional `numAttentionHeads`, `numKeyValueHeads`, `ropeTheta`, `maxPositionEmbeddings`, and `tieWordEmbeddings` to `UnlockedBrowserTransformerWeights`. Defaults preserve fixture behavior.

- [ ] **Step 3: Implement Qwen GQA + RoPE**

Project full Q/K/V, apply q/k RMSNorm per head, apply split-half RoPE per token position, expand K/V grouped-query heads into query-head space for SSA compatibility, run causal attention on expanded packed heads, and apply the full O projection.

- [ ] **Step 4: Load metadata in the browser client**

Normalize the new manifest fields into `UnlockedBrowserTransformerWeights`, and update sharded-manifest test fixtures to prove metadata flows to the backend.

- [ ] **Step 5: Run focused core and web client tests**

Run:

```bash
npm exec --yes pnpm@9.15.0 -- --filter @infinite-edge-agent/core test -- src/runtime/unlockedBrowserTransformer.test.ts
npm exec --yes pnpm@9.15.0 -- --filter @infinite-edge-agent/web test -- src/lib/llm/unlockedBrowserTransformerClient.test.ts
```

Expected: PASS.

### Task 3: Incremental KV Decode Over Real Projected Tensors

**Files:**
- Modify: `packages/core/src/runtime/unlockedBrowserTransformer.ts`
- Modify: `packages/core/src/runtime/unlockedBrowserTransformer.test.ts`

- [ ] **Step 1: Write failing KV tests**

Add tests proving decode appends one token per layer without reprojecting the entire previous prefix:

```ts
const cache = readUnlockedBrowserKvCacheHandle(prefill.kvCacheHandle);
const before = cache.layers[0].kHandle;
await backend.decode(...);
expect(readSsaToyTensorHandle(cache.layers[0].kHandle).matrix).toHaveLength(prefillLength + 1);
expect(cache.layers[0].kHandle).not.toBe(before);
expect(cache.layerStates[0].projectedTokenCount).toBe(prefillLength + 1);
```

Run:

```bash
npm exec --yes pnpm@9.15.0 -- --filter @infinite-edge-agent/core test -- src/runtime/unlockedBrowserTransformer.test.ts
```

Expected: FAIL because current decode rebuilds the whole prefix from token embeddings.

- [ ] **Step 2: Persist per-layer KV state**

Extend `UnlockedBrowserKvCacheHandle` with per-layer state storing post-layer hidden rows and expanded Q/K/V rows. During decode, compute projections only for the new row at each layer and append it.

- [ ] **Step 3: Keep SSA/KVSwap/TSP callbacks over the appended tensors**

Refresh layer tensor handles from the appended Q/K/V rows, register KV blocks for the current token count, prefetch selected blocks, run sparse attention, and use only the latest row to advance the next layer.

- [ ] **Step 4: Run focused core tests**

Run:

```bash
npm exec --yes pnpm@9.15.0 -- --filter @infinite-edge-agent/core test -- src/runtime/unlockedBrowserTransformer.test.ts
```

Expected: PASS.

### Task 4: Production Verification and Regenerated Model Asset

**Files:**
- Modify: `apps/web/public/models/qwen3-0.6b-unlocked/manifest.json`
- Modify: `apps/web/public/models/qwen3-0.6b-unlocked/manifest.json.sha256`
- Modify: `apps/web/.env.local`

- [ ] **Step 1: Regenerate the Qwen manifest**

Run:

```bash
npm exec --yes pnpm@9.15.0 -- convert:unlocked -- \
  --input .cache/models/qwen3-0.6b-hf \
  --output apps/web/public/models/qwen3-0.6b-unlocked \
  --model-id Qwen/Qwen3-0.6B
```

Expected: a manifest with full Qwen shapes and a new SHA.

- [ ] **Step 2: Update `.env.local`**

Set `VITE_UNLOCKED_MODEL_MANIFEST_SHA256` to the new manifest digest.

- [ ] **Step 3: Run strict milestone verification**

Run:

```bash
npm exec --yes pnpm@9.15.0 -- exec tsx scripts/verify-unlocked-asset.ts \
  --manifest-path /models/qwen3-0.6b-unlocked/manifest.json \
  --model-id Qwen/Qwen3-0.6B \
  --manifest-sha256 <new-sha> \
  --require-configured \
  --require-manifest-sha256 \
  --require-sharded \
  --require-qwen-math \
  --require-qwen-parity
```

Expected: PASS with `[unlocked:ssa-kv-tsp]` response marker.

- [ ] **Step 4: Run release gate**

Run:

```bash
env VITE_DEFAULT_MODEL=Qwen/Qwen3-0.6B \
  VITE_UNLOCKED_MODEL_MANIFEST_PATH=/models/qwen3-0.6b-unlocked/manifest.json \
  VITE_UNLOCKED_MODEL_MANIFEST_SHA256=<new-sha> \
  RELEASE_REQUIRE_UNLOCKED_MODEL=true \
  RELEASE_REQUIRE_UNLOCKED_QWEN_MATH=true \
  RELEASE_REQUIRE_UNLOCKED_QWEN_PARITY=true \
  npm exec --yes pnpm@9.15.0 -- release:gate
```

Expected: PASS.
