# Open-Source Release Checklist

Use this checklist before publishing a public branch, release archive, hosted demo, or SDK embed that external users may copy. The default release posture is browser-first and fixture-compatible: no private endpoint, user memory data, licensed model weights, or hosted-service-specific assumption is required.

## 1. Scope and data ownership

- Confirm the README status table still separates implemented, preview-capped, and frontier work.
- Confirm `.env.example` defaults to `VITE_MEMORY_PROVIDER=browser-vector` and `VITE_ALLOW_MEMORY_FALLBACK=true`.
- Confirm remote memory docs say users bring their own endpoint, data store, auth layer, retention policy, and deletion process.
- Do not commit private prompts, retrieved documents, user exports, traces, eval datasets, hosted endpoint URLs, or bearer tokens.

## 2. Model licenses and weights

- Review the license and use terms for every model named in the release notes or configured deployment.
- Qwen/unlocked manifests and shards must stay outside git unless a private deployment deliberately opts into `VITE_BUNDLE_UNLOCKED_MODEL=true` and `RELEASE_ALLOW_BUNDLED_UNLOCKED_MODEL=true`.
- Confirm `.gitignore` still blocks local weights, opaque `.litertlm` artifacts, converted unlocked artifacts under `apps/web/public/models`, plus `.artifacts`, `.data`, `.env`, and local env files.
- For model-backed unlocked releases, record the exact manifest SHA-256 from `manifest.json.sha256` and set `VITE_UNLOCKED_MODEL_MANIFEST_SHA256`.
- For production model-backed unlocked releases, convert with `pnpm convert:unlocked -- --tensor-format f16` and confirm the manifest reports `tensorStorage.format="f16-packed"`. f32 manifests are reference/parity assets only unless the release explicitly opts out of `RELEASE_REQUIRE_UNLOCKED_PACKED_ASSETS`.

## 3. Security headers and SDK embedding

- Keep cross-origin isolation headers for the web app so WebGPU, ONNX/WASM workers, and model assets do not silently degrade.
- Hosted model routes or CDNs must serve compatible COOP/COEP/CORP behavior for browser runtime use.
- SDK embeds must not put API tokens, memory bearer tokens, tenant secrets, or user data in iframe URLs.
- Hosted memory should use same-origin secure cookies/session state or a trusted proxy that injects server-side credentials.
- Review `docs/52_BROWSER_SDK_AND_DEPLOYMENT.md` and `docs/06_SECURITY_PRIVACY.md` for the deployment target.

## 4. Deterministic open-source CI lane

The GitHub Actions workflow uses fixture/browser-only settings and must not require real model weights, a sidecar service, or a private memory endpoint.

| Gate | Command | Expected artifact or evidence |
|---|---|---|
| Install | `pnpm install --frozen-lockfile` | Lockfile-compatible install |
| Typecheck | `pnpm typecheck` | Exit code `0` |
| Tests | `pnpm test` | Converter, core, web, memory-server, and SDK tests pass |
| Unlocked fixture verify | `VITE_UNLOCKED_RUNTIME_PROFILE=ci VITE_UNLOCKED_BACKEND_PREFERENCE=cpu pnpm verify:unlocked` | `.artifacts/evals/ci/unlocked-verify-latest.json` |
| Core smoke | `pnpm smoke:core` | `.artifacts/evals/ci/core-smoke-latest.json` |
| SDK smoke | `pnpm smoke:sdk` | `.artifacts/evals/ci/sdk-smoke-latest.json` |
| Qwen parity fixtures | `pnpm eval:qwen-parity` | `.artifacts/evals/ci/qwen-parity-accuracy-latest.json` |
| Browser-only production eval | `PRODUCTION_EVAL_SIDECAR_MODE=skip pnpm eval:production` | `.artifacts/evals/ci/production-latest.json` |
| Browser runtime benchmark | `VITE_UNLOCKED_RUNTIME_PROFILE=ci pnpm bench:browser-runtime` | `.artifacts/evals/ci/browser-runtime-bench-latest.json` |
| Build | `pnpm build` | Workspace build outputs |
| Dist size | `pnpm check:web-dist` | Passes without bundled local weights |

## 5. Strict release gates for model-backed builds

Run these only when the operator has installed licensed local model artifacts and intends to ship a model-backed release candidate:

```bash
VITE_UNLOCKED_ALLOW_FIXTURE=false \
VITE_UNLOCKED_RUNTIME_PROFILE=full \
RELEASE_REQUIRE_UNLOCKED_MODEL=true \
RELEASE_REQUIRE_UNLOCKED_QWEN_MATH=true \
RELEASE_REQUIRE_UNLOCKED_FULL_PROFILE=true \
RELEASE_REQUIRE_UNLOCKED_PACKED_ASSETS=true \
pnpm verify:unlocked -- \
  --manifest-path .artifacts/models/qwen3-0.6b-unlocked/manifest.json \
  --manifest-sha256 <64-character-sha256-hex-digest> \
  --require-configured \
  --require-manifest-sha256 \
  --require-sharded \
  --require-qwen-math \
  --require-packed-assets \
  --runtime-profile full \
  --require-full-profile
```

- Add `--require-qwen-parity` or `RELEASE_REQUIRE_UNLOCKED_QWEN_PARITY=true` only when the converted manifest is expected to satisfy the stricter parity schema.
- Keep `VITE_REQUIRE_WEBGPU_KERNELS=true` for production browser unlocked builds; use `VITE_REQUIRE_WEBGPU_KERNELS=false` only for explicit CPU-reference development, fixture, or non-authoritative Node verifier lanes.
- Keep `RELEASE_REQUIRE_UNLOCKED_PACKED_ASSETS=true` for production unlocked builds; use `false` only for f32 reference/parity development lanes.
- Add `RELEASE_REQUIRE_UNLOCKED_WEBGPU_MLP=true`, `RELEASE_REQUIRE_UNLOCKED_WEBGPU_LOGITS=true`, `RELEASE_REQUIRE_UNLOCKED_WEBGPU_ATTENTION=true`, and/or `RELEASE_REQUIRE_UNLOCKED_WEBGPU_PROJECTION=true` only for Node verifier hosts that expose a real WebGPU device and must reject CPU-reference fallback.
- Or set `RELEASE_REQUIRE_UNLOCKED_NODE_STRICT_WEBGPU=true` to request all strict WebGPU kernel families together in the Node verifier. The production WebGPU proof remains the browser-preview route.
- Strict model-backed `release:gate` runs require browser proof by default. Start the app, then set `BROWSER_RUNTIME_BENCH_PREVIEW_URL=http://localhost:5173/__bench/browser-runtime`; the gate forwards arbitrary prompts, direct expected-answer checks, `VITE_QWEN_THINKING_MODE`, minimum visible-output checks, KV reuse, browser strict WebGPU proof, and browser expected-answer checks to that route. Set `RELEASE_REQUIRE_MTP_ACCELERATION=true` only when MTP must beat target-only timing on the same device/profile.
- Confirm `.artifacts/evals/unlocked-verify-latest.json` records `runIsCapped: false` for full-profile releases.

## 6. Full local release gate

For a full local release candidate, run:

```bash
pnpm release:gate
```

`pnpm release:gate` writes `.artifacts/evals/release-gate/<timestamp>/results.json`, `trace.jsonl`, and `summary.md`, plus `.artifacts/evals/release-gate-latest.json`. Leave `RELEASE_REQUIRE_SIDECAR=false` for the browser-only/default open-source release lane. Set `RELEASE_REQUIRE_SIDECAR=true` only for sidecar-packaged releases where LanceDB availability is intentionally blocking.

## 7. Hosted compiled deployment profile

For the hosted compiled backend, verify the environment and production benchmark URL before claiming deploy readiness:

```bash
pnpm verify:hosted-profile
```

This check must pass before the real Chrome hosted benchmark is treated as authoritative. It requires the compiled backend, MTP-off production policy, Postgres-backed benchmark telemetry, admin-protected telemetry review routes, rate limiting, and the grounded exact Montana canary URL with telemetry opt-in. It writes `.artifacts/evals/hosted-deployment-profile-latest.json`.

To make the full release gate require and summarize this hosted profile artifact, run:

```bash
RELEASE_REQUIRE_HOSTED_PROFILE=true pnpm release:gate
```

The hosted profile flag also runs `pnpm eval:backend-readiness` and includes `.artifacts/evals/backend-readiness-matrix-latest.json`, which records `compiled-browser-webllm` as the deploy-ready backend only when the hosted proof passes and records `unlocked-browser-transformer` as Kernel Lab/research-only. Use `RELEASE_REQUIRE_BACKEND_READINESS_MATRIX=true` to require that matrix independently.

It also runs `pnpm eval:shared-runtime` and includes `.artifacts/evals/shared-runtime-readiness-latest.json`, which records that memory retrieval, context rebuild, context-pack trace persistence, runtime trace persistence, and backend profile routing are shared above the model backend. Use `RELEASE_REQUIRE_SHARED_RUNTIME_READINESS=true` to require that proof independently.

## 8. Final manual checks

- Browser-smoke the target deployment in Chrome or Edge: initialize one configured backend, send a prompt, retrieve memory, export memory, clear memory, confirm runtime traces persist, and confirm the browser-preview benchmark route returns `passed: true` for non-degenerate visible output.
- Confirm missing model assets fail with actionable errors rather than app-shell HTML fallbacks.
- Confirm hosted memory auth blocks unauthenticated writes and reads.
- Confirm public docs do not contain private endpoint names, private paths, tokens, or user data.
