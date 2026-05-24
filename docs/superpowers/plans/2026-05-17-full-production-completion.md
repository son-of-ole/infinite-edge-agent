# Full Production Completion Plan

## Goal

Finish the remaining production-hardening work for the unlocked browser runtime without drifting back to an opaque Gemma/MediaPipe lane. The target is an open-source, browser-runnable system where the default path is honest about capability, fast enough to use, accurate enough to validate, and deployable through the SDK with user-provided data and memory endpoints.

The dream-state target is stricter than "the app initializes." The final system must be a fully unlocked browser-owned transformer runtime: real licensed Qwen-class shards, browser-owned tokenization and Q/K/V/KV-cache state, real SSA/KVSwap/TSP/MTP execution proofs, browser-local memory by default, optional LanceDB/remote memory for scale, coherent arbitrary responses without seeded memory, long prompt support up to the model context window, long output budgets, and browser-preview proof that the integrated app streams usable text.

## Current Verified Baseline

- `pnpm release:gate` passes with strict unlocked Qwen manifest settings.
- Browser preview initializes `unlocked-browser-transformer`, uses `lancedb-sidecar`, and reports `ssa`, `kvswap`, `tsp`, and `mtp` enabled.
- `pnpm verify:unlocked` records `MTP mode: draft_verify`.
- Older evidence from `unlocked-verify-latest.json` showed CPU-reference MLP/logit projection and local preview caps; the latest delta below adds strict gates so those cannot be mistaken for a final production claim.

## Latest Completion Delta

- Strict WebGPU gates now propagate `requireWebGpu` into the unlocked client and kernel calls, so strict lanes fail before proof collection if WebGPU cannot be used or a kernel attempts CPU fallback.
- Browser KVSwap persistence now stores prompt identity plus Q/K/V/hidden rows and supports exact-match prefill KV reuse with a trace-visible `reuse` operation.
- MTP now verifies draft windows through a browser-owned batched continuation pass, records `verifierStrategy: "batched_continuation"`, `verifiedTokenCount`, `targetDecodeCalls`, and commits only accepted input rows to live KV.
- Prefill MLP rows now execute through a batched WebGPU/CPU kernel boundary per layer, with token-count, projection-cache, and pipeline-cache proof metadata instead of one MLP dispatch per prompt row.
- Browser-runtime benchmarks can now require paired MTP acceleration proof by comparing draft-verify runs against target-only runs and enforcing acceptance plus net speedup floors.
- Strict release status now requires real Qwen parity mode, non-fixture browser-runtime benchmark mode, MTP proof, KV reuse proof, and uncapped full-profile unlocked verification when `RELEASE_REQUIRE_UNLOCKED_MODEL=true`; browser UI initialization/response proof remains a separate browser automation checklist item for static SPA deployments.
- Interactive browser defaults now use `prompt full`, full layers, full-vocab top-k logits, and the Qwen 40,960-token generation ceiling instead of the old 512-token prompt and tiny output budget.
- The MTP route builder keeps explicit SSA routes for the trailing speculative continuation window, so batched verification rows use the same context visibility as target-only decode instead of falling back to pinned block-only attention.
- Prompt-specific state-capital candidate text was removed from the unlocked client; debug candidate-logit mode no longer carries a hidden Utah/geography bias.

## Non-Negotiable Rules

- Do not claim performance acceleration unless a benchmark measures it.
- Do not treat MediaPipe Gemma as satisfying unlocked SSA/KV/TSP/MTP.
- Do not hard-code Sonofol or any private endpoint into the open-source default.
- Keep browser IndexedDB/OPFS local-first defaults and make server memory optional.
- Every production claim must have an artifact, test, or browser proof.
- Do not accept marker-only, punctuation-only, whitespace-only, or invalid-token-fragment output as a working model response.
- Do not tune for a single prompt; arbitrary user prompts must work without requiring seeded memory.

## Task 1 - Runtime Profiles And Staged Uncapping Gates

Implement production runtime profiles for the unlocked browser runtime:

- Add named profiles such as `preview`, `balanced`, `full`, and `ci`.
- Centralize cap parsing for prompt tokens, layers, generation tokens, and logit candidate limits.
- Make the current local caps explicit as the `preview` profile, not the final production profile.
- Add verification artifacts that record the active profile, caps, layer count, token budget, and whether the run is capped.
- Add a strict/full profile gate that can fail when production env still uses preview caps.
- Update README/docs/env examples so users understand how to run preview vs full.

Acceptance:

- Unit tests cover env/profile parsing and strict full-profile failure.
- `verify:unlocked` artifacts include profile/cap metadata.
- Existing local preview remains runnable.

## Task 2 - WebGPU Coverage Accounting And Strict Gates

Make WebGPU/CPU backend coverage measurable and enforceable:

- Add a normalized coverage summary to unlocked verification artifacts: MLP layers by backend, logit projection backend, prefill projection backends, attention backends, and whether CPU fallback was used.
- Add env/CLI gates such as `RELEASE_REQUIRE_UNLOCKED_WEBGPU_MLP`, `RELEASE_REQUIRE_UNLOCKED_WEBGPU_LOGITS`, `RELEASE_REQUIRE_UNLOCKED_WEBGPU_ATTENTION`, and `RELEASE_REQUIRE_UNLOCKED_WEBGPU_PROJECTION`.
- In environments without WebGPU, the default gate may pass but must report CPU fallback honestly.
- Add docs describing which kernels remain CPU-reference and what strict WebGPU gates mean.

Acceptance:

- Tests cover coverage summarization and gate failure on CPU-reference proofs.
- Release artifacts make remaining CPU paths impossible to miss.

## Task 3 - Qwen Parity And Accuracy Suite Expansion

Broaden correctness validation beyond loader shape checks:

- Add deterministic Qwen fixture parity cases for tokenizer, chat formatting, special-token escaping, selected logits, and multi-token decode accounting.
- Add artifact output for parity checks with prompt, token ids, selected logits, generated token ids, and pass/fail gates.
- Add a strict flag/env for broader Qwen parity.
- Keep tests browser-free and deterministic unless explicitly marked live.

Acceptance:

- New parity tests pass in CI.
- `eval:production` or `verify:unlocked` can include parity artifact references.

## Task 4 - Browser Performance Benchmark Suite

Add a repeatable local benchmark for the browser runtime:

- Measure init/load time, prefill time, time to first token, decode latency, tokens/sec, memory mode, caps/profile, backend proofs, MTP acceptance, and optional paired target-only MTP acceleration.
- Write artifacts under `.artifacts/evals/browser-runtime-bench/<timestamp>`.
- Make benchmark runnable in Node against the unlocked client and optionally by browser preview.
- Do not fail normal releases on speed yet; emit thresholds only when env vars request them.

Acceptance:

- `pnpm bench:browser-runtime` exists and writes JSON/summary artifacts.
- Benchmarks include MTP proof, optional acceleration proof, and CPU/WebGPU coverage metadata.

## Task 5 - Browser KVSwap Persistence

Add a browser-native persistence adapter for serialized KV tensor blocks:

- Define a storage interface for KV block serialization.
- Implement IndexedDB-backed storage in the web app and a memory implementation for tests.
- Add tests for save/load/delete/list, version validation, and compressed key summary round-trip.
- Wire optional persistence metadata into runtime traces without changing model math unless explicitly enabled.

Acceptance:

- KV persistence tests pass.
- Runtime docs explain OPFS/IndexedDB KV storage and current limits.

## Task 6 - Browser-Native Vector Memory Provider

Strengthen the fully in-browser memory option:

- Add a named `browser-vector` provider alias backed by IndexedDB.
- Record provider capability metadata: local-only, vector dimension, persistence, import/export support, remote sync disabled.
- Ensure production readiness treats `browser-vector` as a valid browser-only production mode.
- Keep `indexeddb` as a compatibility alias.

Acceptance:

- Hybrid memory client tests cover `browser-vector`.
- UI/status and artifacts distinguish browser-native memory from sidecar/remote memory.

## Task 7 - SDK Deployment Hardening

Make the SDK deploy story production-complete:

- Add an SDK deployment example or fixture that mounts the hosted app with safe options.
- Add tests for URL construction, no secret query params, tenant/cell/session routing, launcher mode, and destroy behavior.
- Add docs for iframe headers, CSP, memory endpoint auth, and deployment presets.
- Add release artifact fields proving SDK smoke ran.

Acceptance:

- SDK smoke remains passing and covers deployment preset behavior.
- Docs let an open-source user embed the agent without private Sonofol assumptions.

## Task 8 - Open-Source Readiness And CI

Prepare the repo for external users:

- Add GitHub Actions workflow for install, typecheck, tests, verify unlocked fixture, smoke/eval/build, and dist-size check.
- Add `.env.example` presets for browser-only, sidecar, remote memory, and unlocked preview/full.
- Add release checklist covering model licenses, weights not committed, security headers, memory endpoint auth, and strict Qwen/WebGPU gates.
- Add README status table separating implemented, preview-capped, and frontier work.

Acceptance:

- CI workflow is syntactically valid.
- Docs are explicit that users bring their own data and memory endpoint.
- Release checklist maps to existing scripts/artifacts.

## Final Verification

Run:

```bash
npm exec --yes pnpm@9.15.0 -- typecheck
npm exec --yes pnpm@9.15.0 -- test
npm exec --yes pnpm@9.15.0 -- release:gate
```

Then use the browser preview to verify:

- app initializes,
- active runtime profile is visible,
- `memoryProvider`, `ssa`, `kvswap`, `tsp`, and `mtp` are enabled in the expected mode,
- one arbitrary message produces coherent assistant text and trace/proof metadata from the unlocked runtime,
- no browser console errors.
