# Dream State Runtime Completion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Complete the browser-native persistent runtime intelligence dream state across browser reality proof, strict WebGPU execution, long-prompt safety, OPFS/KVSwap/MTP performance, headless parity, and GAC consolidation.

**Architecture:** The final system must stay on the real `unlocked-browser-transformer` lane. The model remains the reasoning engine; the runtime owns context rebuild, memory routing, SSA, KVSwap, TSP, MTP, cache/persistence policy, proof metadata, and acceptance gates. Each task below must leave behind code, tests, artifacts, and docs that separate runtime truth, release-gate truth, and product-quality truth.

**Tech Stack:** TypeScript, Vite, React, Web Workers, WebGPU/WGSL, OPFS, IndexedDB, Web Locks, Playwright, pnpm, Qwen unlocked manifest shards, browser-vector memory, optional LanceDB/remote memory.

---

## Current Baseline

This plan supersedes `docs/superpowers/plans/2026-05-17-full-production-completion.md` where newer artifacts have moved the truth forward.

- Current local project: `/Users/olson/Software/edge_AI/infinite-edge-agent-v4`.
- This checkout currently has no `.git` directory visible under `/Users/olson/Software/edge_AI`; workers should not rely on git worktree or commit commands unless a git root is restored.
- Current primary model target: `apps/web/public/models/qwen3-0.6b-unlocked/manifest.json`.
- Current 0.6B manifest facts: `Qwen/Qwen3-0.6B`, 28 layers, 151936 vocab, 1024 hidden size, 16 attention heads, 8 KV heads, 40960 max positions, `qwen-bpe`, f16-packed shards.
- Current 1.7B candidate: `apps/web/public/models/qwen3-1.7b-unlocked/manifest.json`; not production-packed yet because it is f32 and much larger.
- Latest `verify:unlocked` artifact passes but direct Node proof still reports CPU-reference coverage for major kernel families.
- Latest `release:gate` artifact fails at `browser runtime benchmark`.
- Latest browser preview proof reaches strict browser WebGPU with no CPU fallback, but the response is marker-only (`[unlocked:ssa-kv-tsp]`), not coherent text.
- MTP is wired as `draft_verify`, but latest measured acceptance is `0`; acceleration is not proven.
- KV persistence has OPFS/IndexedDB/memory adapters and exact-match prompt KV reuse tests, but does not yet implement worker-owned binary sync OPFS or low-rank predictive KV prefetch.

## Non-Negotiable Completion Rules

- Do not claim production readiness while browser preview output is marker-only, punctuation-only, whitespace-only, or one-token degenerate.
- Do not claim strict WebGPU readiness while production proof contains `cpu_reference`, `mixed`, missing, or `unknown` for MLP, logits, projection, attention, or packed heads.
- Do not claim MTP acceleration unless paired target-only comparison proves acceptance and net speedup on that profile/device.
- Do not claim KVSwap beyond exact-match hydration until low-rank summaries select and prefetch cache blocks ahead of attention.
- Do not solve by returning to an opaque hosted backend, MediaPipe-only lane, fixture manifest, prompt-specific candidate hack, or seeded-memory answer.
- Every task must produce at least one focused test and one machine-readable artifact or proof field.
- Browser-visible proof is part of done, not optional polish.

## Long-Running Execution Protocol

- Codex automation id: `edge-ai-dream-state-runtime-completion`.
- Work task-by-task in order unless a task is blocked by missing infrastructure from a later task.
- For each task, dispatch an implementation subagent, then a spec-compliance reviewer, then a code-quality reviewer.
- If subagents are not available in the execution environment, keep the same checkpoints manually: implementation, spec review, code quality review, verification.
- Keep the task checkbox state in this file current.
- For every run, append short evidence to the task's Evidence section: command, artifact path, pass/fail status, and the relevant summary fields.
- Use project-local commands:

```bash
pnpm test
pnpm typecheck
pnpm verify:unlocked
pnpm bench:browser-runtime
pnpm release:gate
```

## Task 1: Browser Reality Gate

**Goal:** Make the integrated browser route produce multi-token coherent output with expected substrings, KV reuse proof, strict WebGPU proof, and no marker-only pass.

**Files:**
- Modify: `apps/web/src/bench/browserPreviewBenchmark.ts`
- Modify: `apps/web/src/bench/browserPreviewBenchmarkRoute.tsx`
- Modify: `apps/web/src/bench/browserPreviewBenchmark.test.ts`
- Modify: `scripts/browserRuntimeBenchmark.ts`
- Modify: `scripts/browserRuntimeBenchmark.test.ts`
- Modify: `docs/53_UNLOCKED_BROWSER_RUNTIME.md`
- Read evidence: `.artifacts/evals/browser-runtime-bench-latest.json`

- [x] Step 1: Add or tighten tests that fail marker-only browser responses even when runtime trace and WebGPU proof exist.
- [x] Step 2: Make `browserPreviewBenchmark` require visible text that is not just `[unlocked:ssa-kv-tsp]`, proof markers, whitespace, punctuation, or a single stop fragment.
- [x] Step 3: Make strict real-Qwen benchmark defaults use meaningful prompts, expected substrings, `minGeneratedTokens >= 8`, and `requireKvReuse=true`.
- [x] Step 4: Ensure the route can run target-only mode for diagnosis when MTP is suspected of damaging output.
- [x] Step 5: Run focused tests:

```bash
pnpm test -- apps/web/src/bench/browserPreviewBenchmark.test.ts scripts/browserRuntimeBenchmark.test.ts
```

- [ ] Step 6: Run a real browser preview proof with full profile, strict WebGPU, expected substring, KV reuse, and enough timeout for current slow path.

```bash
pnpm dev:web
pnpm bench:browser-runtime -- --browser-preview-url http://127.0.0.1:5173/__bench/browser-runtime
```

**Acceptance:**
- Browser preview artifact has `passed: true`.
- `coherentResponseCount >= 1`.
- `generatedTokenCount >= 8`.
- `expectedSubstringsPassed: true`.
- `kvReusePassed: true` when requested.
- `strictWebGpuPassed: true`.
- Response is not marker-only.

**Evidence:**
- [x] Implementation: marker-only/proof-only visible responses now fail browser-preview quality gates; hidden `<think>...</think>` content is removed before visible-quality and expected-substring checks; strict configured preview defaults preserve meaningful prompts, expected substrings, `generationTokens=16`, `minGeneratedTokens=8`, strict WebGPU gates, and `requireKvReuse=true`; browser-preview URL generation uses repeated `prompt` plus per-prompt `expectedJson` arrays to avoid delimiter coupling while keeping old `prompts`, `expected`, and `expectedSubstrings` compatibility.
- [x] Spec review: `SPEC_APPROVED` from subagent `019e4355-6d81-7ef1-bd44-180d11b8e678`.
- [x] Code quality review: `QUALITY_APPROVED` from subagent `019e43d2-d07b-75a0-aa28-5a4b507f760e`.
- [x] Focused verification: `./node_modules/.bin/vitest run apps/web/src/bench/browserPreviewBenchmark.test.ts scripts/browserRuntimeBenchmark.test.ts` passed 2 files and 41 tests on 2026-05-19.
- [ ] Live browser proof remains pending because it requires a running dev server and slow real-model browser execution.

## Task 2: Production WebGPU-Only Kernel Coverage

**Goal:** Remove CPU-reference paths from production proof for MLP, Q/K/V/O projections, attention, packed heads, and logits in strict lanes.

**Files:**
- Modify: `packages/core/src/runtime/ssa_webgpu/index.ts`
- Modify: `packages/core/src/runtime/ssa_webgpu/webgpuSsaBackend.ts`
- Modify: `packages/core/src/runtime/ssa_webgpu/wgsl/*.wgsl.ts`
- Modify: `packages/core/src/runtime/ssa_webgpu/webgpuToyKernels.test.ts`
- Modify: `apps/web/src/lib/llm/unlockedBrowserTransformerClient.ts`
- Modify: `apps/web/src/lib/llm/unlockedBrowserTransformerClient.test.ts`
- Modify: `scripts/unlockedWebGpuCoverage.ts`
- Modify: `scripts/unlockedWebGpuCoverage.test.ts`
- Modify: `scripts/verify-unlocked-asset.ts`
- Modify: `docs/53_UNLOCKED_BROWSER_RUNTIME.md`

- [x] Step 1: Add failing coverage tests proving strict mode rejects CPU-reference proof for each kernel family.
- [x] Step 2: Route strict decode/prefill calls through WebGPU primitives for dense projection, MLP, sparse attention, packed heads, and tiled logits.
- [x] Step 3: Ensure `requireWebGpu` fails before fallback when a WebGPU device or kernel cannot be used.
- [x] Step 4: Add proof metadata that identifies exact family, layer count, backend count, and fallback reason.
- [x] Step 5: Run focused tests:

```bash
pnpm test -- packages/core/src/runtime/ssa_webgpu/webgpuToyKernels.test.ts apps/web/src/lib/llm/unlockedBrowserTransformerClient.test.ts scripts/unlockedWebGpuCoverage.test.ts
```

- [ ] Step 6: Run strict verification:

```bash
RELEASE_REQUIRE_UNLOCKED_STRICT_WEBGPU=true VITE_REQUIRE_WEBGPU_KERNELS=true pnpm verify:unlocked
```

**Acceptance:**
- Strict lane fails if any required family reports CPU fallback.
- On WebGPU-capable browser proof, required families report `webgpu`.
- Node-only lanes either prove WebGPU or clearly fail/mark as non-production.

**Evidence:**
- [x] Implementation: strict WebGPU coverage accounting now lives in browser-safe core runtime code and is shared by Node/script and browser-preview gates. Strict lanes fail closed for missing `expectedLayerCount`, partial expected-layer proof, CPU/mixed/unknown backends, candidate-only logits, invalid logit row metadata, empty packed-head proof, and old decode-O-only projection proof.
- [x] Runtime proof wiring: browser decode proof now includes `expectedLayerCount`, `executedLayerCount`, and per-layer decode Q/K/V/O projection backends. Core backend proof carries `projectionLayers` with decode Q/K/V/O traces; browser client serializes those into `decodeProjectionBackends`; browser preview uses the shared evaluator for requested gates and `positiveKernelProof`.
- [x] Spec review: `SPEC_APPROVED` from subagent `019e47eb-8f8a-7410-8754-02b11a7ba82f` after rejecting and fixing partial-layer and decode-Q/K/V proof gaps.
- [x] Code quality review: `QUALITY_APPROVED` from subagent `019e47ef-ae11-71c1-b105-69e21067b254`.
- [x] Focused verification: `./node_modules/.bin/vitest run scripts/unlockedWebGpuCoverage.test.ts scripts/browserRuntimeBenchmark.test.ts apps/web/src/bench/browserPreviewBenchmark.test.ts apps/web/src/lib/llm/unlockedBrowserTransformerClient.test.ts packages/core/src/runtime/unlockedBrowserTransformer.test.ts` passed 5 files and 132 tests on 2026-05-20.
- [x] Type verification: `./node_modules/.bin/tsc -p packages/core/tsconfig.json --noEmit` and `./node_modules/.bin/tsc -p apps/web/tsconfig.json --noEmit` passed on 2026-05-20.
- [ ] Strict real-model `verify:unlocked` remains pending because Task 1 browser live proof is still timeout-bound and the local environment has no `pnpm` binary on PATH.

## Task 3: Chunked Prefill And Static Shape Buckets

**Goal:** Make long prompts safe for browser GPU watchdogs and avoid dynamic-shape pipeline churn.

**Files:**
- Modify: `apps/web/src/lib/llm/unlockedBrowserTransformerClient.ts`
- Modify: `packages/core/src/runtime/ssa_webgpu/index.ts`
- Modify: `packages/core/src/runtime/ssa_webgpu/types.ts`
- Modify: `packages/core/src/runtime/ssa_webgpu/webgpuToyKernels.test.ts`
- Create or modify: `packages/core/src/runtime/ssa_webgpu/shapeBuckets.ts`
- Create or modify: `packages/core/src/runtime/ssa_webgpu/shapeBuckets.test.ts`
- Modify: `docs/26_WEBGPU_SSA_KERNEL_PLAN.md`
- Modify: `docs/53_UNLOCKED_BROWSER_RUNTIME.md`

- [x] Step 1: Add shape-bucket utilities for prompt lengths, selected-block counts, head dimensions, and tile rows.
- [x] Step 2: Add tests proving nearby dynamic lengths reuse the same bucket/pipeline key.
- [x] Step 3: Add prefill chunk planning that breaks long prompts into bounded dispatch windows.
- [x] Step 4: Add trace fields: `prefillChunkCount`, `prefillChunkSize`, `shapeBucket`, `pipelineCacheKey`, and `maxDispatchEstimatedMs` when available.
- [x] Step 5: Add browser benchmark controls for long-prompt proof and timeout-safe failure.
- [x] Step 6: Run focused tests:

```bash
pnpm test -- packages/core/src/runtime/ssa_webgpu/shapeBuckets.test.ts packages/core/src/runtime/ssa_webgpu/webgpuToyKernels.test.ts apps/web/src/lib/llm/unlockedBrowserTransformerClient.test.ts
```

**Acceptance:**
- Long prompts have deterministic benchmark controls and strict proof now requires executed chunked dispatch metadata rather than planning-only claims.
- Pipeline keys are stable across bucketed lengths.
- Browser preview can fail gracefully with chunk-plan metadata rather than looking like a generic hang.

**Evidence:**
- [x] First slice implementation: `packages/core/src/runtime/ssa_webgpu/shapeBuckets.ts` now provides prompt-length, selected-block-count, head-dimension, and tile-row buckets; stable static pipeline keys; bounded prefill chunk plans; and dispatch-time estimates. `SSAKernelTrace`, `SSAPrefillHandle`, core prefill proof, and browser decode proof can surface `prefillChunkCount`, `prefillChunkSize`, `shapeBucket`, `pipelineCacheKey`, and `maxDispatchEstimatedMs`.
- [x] Honesty boundary: docs now state these fields are planning/proof metadata and do not yet mean the transformer loop executes every long prefill as separate awaited WebGPU chunk windows.
- [x] Spec review: `SPEC_APPROVED` from subagent `019e47fd-6a2d-73a2-8e10-6f1d41f6af5f`.
- [x] Code quality review: `QUALITY_APPROVED` from subagent `019e47ff-5a3e-7f10-8f7f-3ed3b0a4b8b4`.
- [x] Focused verification: `./node_modules/.bin/vitest run packages/core/src/runtime/ssa_webgpu/shapeBuckets.test.ts packages/core/src/runtime/ssa_webgpu/webgpuToyKernels.test.ts apps/web/src/lib/llm/unlockedBrowserTransformerClient.test.ts` passed 3 files and 76 tests on 2026-05-20.
- [x] Type verification: `./node_modules/.bin/tsc -p packages/core/tsconfig.json --noEmit` and `./node_modules/.bin/tsc -p apps/web/tsconfig.json --noEmit` passed on 2026-05-20.
- [x] Second slice implementation: `scripts/browserRuntimeBenchmark.ts`, `/__bench/browser-runtime`, and browser preview payload builders now support deterministic long-prompt controls, propagate chunk-plan fields, and report executed chunked dispatch proof for strict multi-chunk prefill.
- [x] Second slice spec review: `SPEC_APPROVED` from subagent `019e4818-3b5a-79e3-81d6-ea2d2faf5fbb`.
- [x] Second slice code quality review: `QUALITY_APPROVED` from subagent `019e4833-7ed3-7a21-ae0c-0fcd4d78e75d` after bounding long-prompt seed scanning, preserving compact `promptSeed` determinism, and avoiding expanded prompt query forwarding.
- [x] Second slice focused verification: `./node_modules/.bin/vitest run packages/core/src/runtime/ssa_webgpu/shapeBuckets.test.ts scripts/browserRuntimeBenchmark.test.ts apps/web/src/bench/browserPreviewBenchmark.test.ts apps/web/src/lib/llm/unlockedBrowserTransformerClient.test.ts` passed 4 files and 108 tests on 2026-05-20.
- [x] Second slice type verification: `./node_modules/.bin/tsc -p packages/core/tsconfig.json --noEmit` and `./node_modules/.bin/tsc -p apps/web/tsconfig.json --noEmit` passed on 2026-05-20.
- [ ] Remaining Task 3 work: actual kernel-level time-sliced/chunked prefill dispatch.

## Task 4: Worker-Owned Binary OPFS With Cross-Tab Coordination

**Goal:** Upgrade KV persistence from async JSON OPFS/IndexedDB to worker-owned binary OPFS using `createSyncAccessHandle()` where available, guarded by Web Locks or a single shared routing worker.

**Files:**
- Modify: `apps/web/src/lib/runtime/kvSwapPersistence.ts`
- Modify: `apps/web/src/lib/runtime/kvSwapPersistence.test.ts`
- Create: `apps/web/src/workers/kvSwapPersistence.worker.ts`
- Create or modify: `apps/web/src/lib/runtime/kvSwapBinaryCodec.ts`
- Create or modify: `apps/web/src/lib/runtime/kvSwapBinaryCodec.test.ts`
- Modify: `apps/web/src/workers/unlockedTransformer.worker.ts`
- Modify: `docs/53_UNLOCKED_BROWSER_RUNTIME.md`

- [x] Step 1: Add a binary codec for serialized KV blocks, including version, namespace, ids, token ranges, Q/K/V/hidden rows, checksums, and summary metadata.
- [x] Step 2: Add tests for binary round-trip, corrupt record quarantine, and version mismatch.
- [x] Step 3: Implement worker-side OPFS sync-handle storage when `createSyncAccessHandle()` exists.
- [x] Step 4: Use Web Locks API when available to serialize access by namespace; fall back to single worker ownership or IndexedDB.
- [x] Step 5: Add health/proof fields: `mode`, `binary`, `syncAccessHandle`, `webLocks`, `lockWaitMs`, `bytesRead`, `bytesWritten`, `tabCoordination`.
- [x] Step 6: Run focused tests:

```bash
pnpm test -- apps/web/src/lib/runtime/kvSwapPersistence.test.ts apps/web/src/lib/runtime/kvSwapBinaryCodec.test.ts
```

**Acceptance:**
- Existing exact-match reuse still works.
- Binary OPFS is selected before async OPFS/IndexedDB when browser support exists.
- Multi-tab conflicts are coordinated instead of corrupting or crashing persistence.

**Evidence:**
- [x] Implementation: added `apps/web/src/lib/runtime/kvSwapBinaryCodec.ts`, `apps/web/src/workers/kvSwapPersistence.worker.ts`, binary sync-handle OPFS selection in `kvSwapPersistence.ts`, and docs for binary OPFS/Web Locks/single-worker routing behavior.
- [x] Coordination boundary: direct binary OPFS now requires Web Locks or explicit worker-owned routing; otherwise it falls back to async JSON OPFS/IndexedDB instead of claiming uncoordinated cross-tab safety.
- [x] Hardening: binary reads reject oversized files before allocation, binary decode enforces string/array/matrix budgets, worker stores are namespace-isolated, worker operations are serialized, and exact-reuse metadata is validated before hydration.
- [x] Spec review: `SPEC_APPROVED` from subagent `019e4849-e658-7901-8626-faf5aeb4ed3c`.
- [x] Code quality review: `QUALITY_APPROVED` from subagent `019e4855-9de2-74f1-869b-30c746cc7864`.
- [x] Focused verification: `./node_modules/.bin/vitest run apps/web/src/lib/runtime/kvSwapPersistence.test.ts apps/web/src/lib/runtime/kvSwapBinaryCodec.test.ts apps/web/src/lib/llm/unlockedBrowserTransformerClient.test.ts` passed 3 files and 75 tests on 2026-05-20.
- [x] Type verification: `./node_modules/.bin/tsc -p apps/web/tsconfig.json --noEmit` passed on 2026-05-20.

## Task 5: Real Low-Rank Predictive KVSwap Prefetch

**Goal:** Turn KVSwap from exact-match prompt hydration into predictive low-rank cache selection and asynchronous prefetch.

**Files:**
- Modify: `packages/core/src/runtime/kvswap.ts`
- Modify: `packages/core/src/runtime/kvswap.test.ts`
- Modify: `packages/core/src/runtime/kvTensorPaging.ts`
- Modify: `packages/core/src/runtime/advancedRuntimeCoordinator.ts`
- Modify: `apps/web/src/lib/llm/unlockedBrowserTransformerClient.ts`
- Modify: `apps/web/src/lib/runtime/kvSwapPersistence.ts`
- Modify: `docs/01_ARCHITECTURE.md`
- Modify: `docs/53_UNLOCKED_BROWSER_RUNTIME.md`

- [ ] Step 1: Define low-rank key summary metadata with rank, projection id, layer/head grouping, block id, checksum, and quality score.
- [ ] Step 2: Add tests for approximate attention block scoring from compressed key summaries.
- [ ] Step 3: Add a predictive prefetch planner that chooses hot KV blocks before attention.
- [ ] Step 4: Wire asynchronous persistence loads so I/O can overlap with compute where the browser permits it.
- [ ] Step 5: Add trace fields: `lowRankSummaryRank`, `predictedHotBlocks`, `prefetchedBlocks`, `prefetchHitRate`, `prefetchBytes`, `prefetchLatencyMs`, `attentionStallMs`.
- [ ] Step 6: Run focused tests:

```bash
pnpm test -- packages/core/src/runtime/kvswap.test.ts packages/core/src/runtime/advancedRuntimeCoordinator.test.ts apps/web/src/lib/llm/unlockedBrowserTransformerClient.test.ts
```

**Acceptance:**
- KVSwap can select blocks by predicted attention relevance, not only exact prompt identity.
- Prefetch decisions are visible in runtime traces.
- Benchmarks distinguish exact reuse, predictive prefetch, and miss/stall behavior.

**Evidence:**
- [ ] Pending.

## Task 6: Browser-Optimized MTP Draft Windows And Speed Proof

**Goal:** Rework speculative decoding for browser concurrency=1 with small draft windows and paired speed gates.

**Files:**
- Modify: `apps/web/src/lib/llm/unlockedBrowserTransformerClient.ts`
- Modify: `apps/web/src/lib/llm/unlockedBrowserTransformerClient.test.ts`
- Modify: `packages/core/src/runtime/speculative.ts`
- Modify: `packages/core/src/runtime/speculative.test.ts`
- Modify: `packages/core/src/runtime/speculativeBatching.test.ts`
- Modify: `scripts/browserRuntimeBenchmark.ts`
- Modify: `scripts/browserRuntimeBenchmark.test.ts`
- Modify: `docs/53_UNLOCKED_BROWSER_RUNTIME.md`

- [ ] Step 1: Change browser default speculative token count to a small window, initially 2 or 3, unless explicitly overridden.
- [ ] Step 2: Add tests proving draft verify keeps output equal to target-only for the same prompt/cache.
- [ ] Step 3: Add KV rewind/commit proof for accepted, rejected, and correction tokens.
- [ ] Step 4: Make paired target-only benchmark required before setting `mtpAccelerationPassed=true`.
- [ ] Step 5: Add device/profile-tuned fields: `draftWindow`, `draftLatencyMs`, `verifyLatencyMs`, `targetOnlyLatencyMs`, `acceptedTokens`, `rejectedTokens`, `netSpeedupRatio`.
- [ ] Step 6: Run focused tests:

```bash
pnpm test -- packages/core/src/runtime/speculative.test.ts packages/core/src/runtime/speculativeBatching.test.ts apps/web/src/lib/llm/unlockedBrowserTransformerClient.test.ts scripts/browserRuntimeBenchmark.test.ts
```

**Acceptance:**
- MTP never degrades into marker-only or invalid output.
- Acceleration remains false/skipped unless paired benchmark proves it.
- Browser default is tuned for low-concurrency local inference, not server batch throughput.

**Evidence:**
- [ ] Pending.

## Task 7: Headless WebGPU CI Parity

**Goal:** Add a CI lane that can validate WebGPU behavior through Dawn/Lavapipe or a clearly separated browser-runner route.

**Files:**
- Modify: `package.json`
- Modify: `.github/workflows/ci.yml`
- Create or modify: `scripts/headlessWebGpuProbe.ts`
- Create or modify: `scripts/headlessWebGpuProbe.test.ts`
- Modify: `scripts/browserRuntimeBenchmark.ts`
- Modify: `docs/07_TEST_PLAN.md`
- Modify: `docs/53_UNLOCKED_BROWSER_RUNTIME.md`

- [ ] Step 1: Add a capability probe that reports browser WebGPU, Node WebGPU, Dawn/Lavapipe, or unavailable.
- [ ] Step 2: Add tests for capability classification and artifact shape.
- [ ] Step 3: Add a CI-safe script that either runs headless WebGPU proof or records a non-production skip with reason.
- [ ] Step 4: Ensure release docs distinguish Node CPU reference, headless WebGPU parity, and live browser preview proof.
- [ ] Step 5: Run focused tests:

```bash
pnpm test -- scripts/headlessWebGpuProbe.test.ts scripts/browserRuntimeBenchmark.test.ts
```

**Acceptance:**
- CI artifacts cannot silently treat CPU reference as WebGPU parity.
- Local developers can see exactly which WebGPU proof mode ran.
- Browser preview remains the production-quality proof until headless parity is fully equivalent.

**Evidence:**
- [ ] Pending.

## Task 8: GAC Consolidation And Runtime-Learned Context Rebuild

**Goal:** Move beyond static retrieval into adaptive memory selection, identity pins, consolidation jobs, and runtime-learned context reconstruction.

**Files:**
- Modify: `packages/core/src/runtime/contextRuntime.ts`
- Modify: `packages/core/src/runtime/contextRuntime.test.ts`
- Modify: `packages/core/src/runtime/advancedRuntimeCoordinator.ts`
- Modify: `packages/core/src/runtime/gacIngestion.ts`
- Modify: `packages/core/src/runtime/gacIngestion.test.ts`
- Modify: `docs/18_CONTEXT_RUNTIME_SPEC.md`
- Modify: `docs/29_GEOMETRY_AWARE_MEMORY_CONSOLIDATION.md`
- Modify: `docs/31_CONTEXT_PACKING_WITH_GAC.md`
- Modify: `docs/34_MODEL_NATIVE_GAC.md`
- Modify: `docs/37_MEMORY_SLEEP_CYCLE.md`

- [ ] Step 1: Add context rebuild inputs for active goals, task graph, identity pins, consolidation state, and runtime feedback metrics.
- [ ] Step 2: Add tests proving identity-risk memories and pinned constraints survive ordinary similarity-based trimming.
- [ ] Step 3: Add consolidation job records for representatives, raw lineage, residuals, and sleep-cycle refresh.
- [ ] Step 4: Feed retrieval success/failure, dropped-context causes, hallucination-risk markers, and user correction signals back into the context plan.
- [ ] Step 5: Add trace fields: `identityPinIds`, `activeGoalIds`, `consolidationJobIds`, `retrievalFeedback`, `droppedContextReasons`, `runtimePolicyVersion`.
- [ ] Step 6: Run focused tests:

```bash
pnpm test -- packages/core/src/runtime/contextRuntime.test.ts packages/core/src/runtime/gacIngestion.test.ts packages/core/src/runtime/advancedRuntimeCoordinator.test.ts
```

**Acceptance:**
- Context rebuild is no longer just prompt packing plus retrieved chunks.
- GAC representatives and exact lineage both influence active context.
- The runtime records feedback needed for future trainable/adaptive context rebuilding.

**Evidence:**
- [ ] Pending.

## Final Dream-State Verification

Run the full suite after all tasks:

```bash
pnpm typecheck
pnpm test
pnpm verify:unlocked
pnpm bench:browser-runtime -- --browser-preview-url http://127.0.0.1:5173/__bench/browser-runtime
pnpm release:gate
```

Then capture or link artifacts proving:

- [ ] Real Qwen manifest, no fixture.
- [ ] Full runtime profile, no artificial tiny prompt/output caps.
- [ ] Browser preview coherent output.
- [ ] Expected substrings pass.
- [ ] Strict WebGPU proof passes.
- [ ] No CPU fallback in strict production browser proof.
- [ ] KV exact reuse passes.
- [ ] KV predictive prefetch metrics exist.
- [ ] MTP speed proof is either passed with paired benchmark or explicitly disabled/skipped.
- [ ] Long prompt prefill chunking and shape bucket proof exists.
- [ ] OPFS binary worker/Web Locks proof exists where supported.
- [ ] Headless parity or explicit non-production skip proof exists.
- [ ] GAC/context rebuild traces include identity pins, active goals, lineage, and feedback.

## Completion Definition

This goal is complete only when the integrated browser app can answer arbitrary prompts coherently through the real unlocked runtime while emitting proof that context rebuild, SSA, KVSwap, TSP, MTP policy, WebGPU coverage, memory persistence, and GAC/consolidation are actually participating. Anything less is a staged milestone and must remain labeled that way.
