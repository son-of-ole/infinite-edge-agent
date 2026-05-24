# Test Plan

## Unit tests

Current tests cover:

- Chunking.
- Vector similarity.
- Context packing.

Run:

```bash
pnpm test
```

## Browser smoke tests

Manual checklist:

1. App loads in Chrome/Edge.
2. Initialize local agent.
3. Model progress appears.
4. Send a message.
5. Response streams token-by-token.
6. Refresh page.
7. Send a related message.
8. Retrieved memory IDs appear.
9. Clear memory.
10. Retrieved memory IDs disappear on future turns.

## Sidecar tests

Manual checklist:

1. Start sidecar.
2. `GET /health` returns `{ ok: true }`.
3. Web app shows `browser-vector` memory mode for browser-only runs, or `lancedb-sidecar` when the optional sidecar is explicitly selected and healthy.
4. Send message.
5. `.data/lancedb` is created.
6. Search returns hits.
7. Stop sidecar and refresh web app.
8. App falls back to IndexedDB.

## Performance tests

Track:

- Model first-load time.
- Time to first token.
- Tokens/sec.
- Embedding latency per chunk.
- Memory search latency.
- Prompt estimated tokens.
- Browser memory usage.

## Core stress eval

Run the deterministic browser-free production hardening eval:

```bash
pnpm stress:core
```

It exercises core memory vector search and context packing without model weights or external services. The default run is intentionally local-fast. Increase coverage with CLI flags or matching env vars:

```bash
pnpm stress:core --vectors 10000 --dim 384 --top-k 40 --memory-token-budget 1200
```

Artifacts are written to `.artifacts/evals/core-stress/<timestamp>/results.json`, `trace.jsonl`, and `summary.md`. A quick gate copy is also written to `.artifacts/evals/core-stress-latest.json`.

## SDK smoke eval

Run the browser-free SDK embed gate:

```bash
pnpm smoke:sdk
```

It verifies that the SDK builds an `embed=1` iframe URL, rejects secret-shaped query parameters, mounts into a supplied DOM container, sets the browser runtime permissions, and destroys cleanly. Artifacts are written to `.artifacts/evals/sdk-smoke/<timestamp>/results.json`, `trace.jsonl`, and `summary.md`, plus `.artifacts/evals/sdk-smoke-latest.json`.

## Qwen parity accuracy eval

Run the browser-free deterministic Qwen parity gate:

```bash
pnpm eval:qwen-parity
```

It exercises toy/reference fixtures for RMSNorm, RoPE, GQA head grouping, gated MLP, tokenizer/chat template escaping, selected logit projection, multi-token decode accounting through the unlocked browser transformer reference path, and a retrieval-grounded response check. Artifacts are written to `.artifacts/evals/qwen-parity-accuracy/<timestamp>/results.json` and `summary.md`, plus `.artifacts/evals/qwen-parity-accuracy-latest.json`. The multi-token fixture records prompt token IDs, generated token IDs, expected generated token IDs, and per-step decode accounting so token parity failures are release-visible.

The default gate does not require private or licensed Qwen weights. If a converted model manifest is installed, pass `--manifest-path .artifacts/models/qwen3-0.6b-unlocked/manifest.json` or set `VITE_UNLOCKED_MODEL_MANIFEST_PATH` to add installed-manifest parity metadata. Set `RELEASE_REQUIRE_QWEN_ACCURACY_REAL_MODEL=true` only for release lanes where the converted model is required.

Broader eval datasets should be stored outside the repo or in gitignored artifact directories, then referenced by path or generated into local artifacts. Do not commit private prompts, retrieved documents, model outputs, or licensed model weights.

## Production eval sidecar mode

Production eval runs core production suites without network by default, then checks the LanceDB sidecar in `auto` mode:

```bash
pnpm eval:production
```

If the sidecar is not reachable, `auto` mode records an actionable skipped sidecar suite instead of failing with a raw connection error. Make the sidecar blocking for packaged/local sidecar releases:

```bash
PRODUCTION_EVAL_SIDECAR_MODE=required pnpm eval:production
```

Or let the eval start a sidecar command and wait for `/health`:

```bash
PRODUCTION_EVAL_SIDECAR_START_COMMAND="pnpm --filter @infinite-edge-agent/memory-server start" pnpm eval:production
```

Use `PRODUCTION_EVAL_SIDECAR_MODE=skip` only for browser-only gates where the sidecar is intentionally out of scope.

## Release gate

Run the local production release gate:

```bash
pnpm release:gate
```

The command is also compatible with environments that invoke pnpm through npm:

```bash
npm exec pnpm -- run release:gate
```

The gate runs typecheck, converter tests, package tests, unlocked manifest verification, core smoke, SDK smoke, core stress, browser runtime benchmark, Qwen parity accuracy, production eval, build, and web-dist size validation. It writes `.artifacts/evals/release-gate/<timestamp>/results.json`, `trace.jsonl`, and `summary.md`, plus `.artifacts/evals/release-gate-latest.json`. Child eval artifacts for that exact invocation are isolated under `.artifacts/evals/release-gate/<timestamp>/child-evals`, so release summaries do not read mutable top-level `*-latest.json` files from another concurrent run. When a configured or locally installed Qwen manifest/SHA is present, `release:gate` defaults to the real-Qwen lane instead of the generated fixture: `verify:unlocked` requires the configured manifest, manifest SHA, full profile, Qwen math/parity, KV decode reuse, packed assets, and `browser-vector` memory without treating Node as authoritative browser GPU proof. The browser benchmark requires WebGPU backend preference, direct expected-answer checks, and a completed browser-preview proof with the same expected-answer checks. `VITE_QWEN_THINKING_MODE=disabled` is the default visible-answer setting and is forwarded into browser-preview proof URLs unless explicitly overridden. Set `BROWSER_RUNTIME_BENCH_PREVIEW_URL=http://localhost:5173/__bench/browser-runtime` for local strict real-Qwen release gates; without it, the benchmark fails early instead of silently recording a preview skip. Set `RELEASE_ALLOW_FIXTURE_GATE=true` for explicit fixture/dev gates; CI without a model also stays fixture-capable. Set `RELEASE_REQUIRE_UNLOCKED_MODEL=true` when a release candidate must fail if no converted unlocked manifest is configured. Strict browser-preview WebGPU is the production proof by default; set `RELEASE_REQUIRE_UNLOCKED_NODE_STRICT_WEBGPU=true` only for a Node environment with a real WebGPU device.

By default, `release:gate` follows the browser-first open-source lane and leaves the LanceDB sidecar non-blocking. For sidecar-packaged releases, set `RELEASE_REQUIRE_SIDECAR=true`; the production eval step then requires a healthy sidecar profile matching `MEMORY_DB_URI` / `MEMORY_TABLE` or `PRODUCTION_EVAL_EXPECTED_DB_URI` / `PRODUCTION_EVAL_EXPECTED_TABLE_NAME`.

## Regression gates

Before shipping:

- `pnpm release:gate`
- Or run the equivalent manual sequence:
- `pnpm typecheck`
- `pnpm test`
- `pnpm test:converter`
- `pnpm verify:unlocked` for sharded manifest loading, shard integrity, backend init, tensor-control decode proof, runtime-profile metadata, and normalized WebGPU/CPU coverage in `.artifacts/evals/unlocked-verify-latest.json`
- For model-backed unlocked releases: `pnpm verify:unlocked -- --require-configured --require-manifest-sha256 --require-sharded --runtime-profile full --require-full-profile`
- For Qwen-math unlocked releases: add `--require-qwen-math`
- For strict WebGPU release lanes: use the browser-preview route with `BROWSER_RUNTIME_BENCH_PREVIEW_REQUIRE_STRICT_WEBGPU=true`; strict real-Qwen release mode sets this automatically. Node `verify:unlocked` is a manifest/parity/tensor-control gate by default and may run without WebGPU. Add `--require-webgpu-mlp`, `--require-webgpu-logits`, `--require-webgpu-attention`, `--require-webgpu-projection`, or `RELEASE_REQUIRE_UNLOCKED_NODE_STRICT_WEBGPU=true` only when that Node host has a real WebGPU device and should fail closed. For browser-runtime Node performance runs, add `--require-strict-webgpu` or `BROWSER_RUNTIME_BENCH_REQUIRE_STRICT_WEBGPU=true` when CPU fallback must fail that benchmark.
- `pnpm smoke:core` and confirm `.artifacts/evals/core-smoke-latest.json` reports `passed: true`
- `pnpm smoke:sdk` and confirm `.artifacts/evals/sdk-smoke-latest.json` reports `passed: true`
- `pnpm stress:core` and confirm `.artifacts/evals/core-stress-latest.json` reports `passed: true`
- `pnpm eval:qwen-parity` and confirm `.artifacts/evals/qwen-parity-accuracy-latest.json` reports `passed: true`
- `pnpm eval:production` and confirm `.artifacts/evals/production-latest.json` reports `passed: true`
- `pnpm build`
- `pnpm check:web-dist` and confirm local model weights are not bundled into the static deploy artifact unless explicitly intended.
- Browser smoke test on target OS/browser: verify the unlocked backend initializes, answers, retrieves memory, and persists a runtime trace.
- SDK smoke test: embed the hosted app with `@infinite-edge-agent/browser-sdk` and confirm the iframe loads with `embed=1` and no secret query params.
- Sidecar smoke test if packaged, including `/memory/status`, `/memory/repair`, and `/runtime/traces`.
- Model license review.
- Security checklist review.
