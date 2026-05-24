# Build and Run

## 1. Install dependencies

```bash
corepack enable
pnpm install
```

## 2. Run browser-only MVP

The default open-source browser path is the unlocked browser transformer using `Qwen/Qwen3-0.6B` as the model target. A fresh checkout uses deterministic fixture weights so the browser-owned Q/K/V, KV-cache, SSA, TSP, and runtime-control path can boot without committing or downloading licensed model weights.

```bash
cp .env.example .env
```

Confirm the browser-first defaults:

```bash
VITE_LLM_BACKEND=unlocked-browser-transformer
VITE_DEFAULT_MODEL=Qwen/Qwen3-0.6B
VITE_REQUIRE_UNLOCKED_RUNTIME=true
VITE_UNLOCKED_ALLOW_FIXTURE=true
VITE_UNLOCKED_RUNTIME_PROFILE=preview
VITE_MEMORY_PROVIDER=browser-vector
VITE_ALLOW_MEMORY_FALLBACK=true
```

Run the browser app:

```bash
pnpm dev:web
```

Open the Vite URL in a WebGPU-capable browser. The unlocked browser transformer now defaults to `VITE_REQUIRE_WEBGPU_KERNELS=true`, so the production app fails closed instead of silently running CPU-reference kernels. Set `VITE_REQUIRE_WEBGPU_KERNELS=false` only for explicit local debugging or CPU-reference test lanes.

For production model-backed unlocked runs, convert/package a licensed local Qwen-compatible artifact outside git, host the manifest and shards from the app or a COOP/COEP-compatible CDN, and set:

```bash
pnpm convert:unlocked -- \
  --input /path/to/local/Qwen3-0.6B \
  --output .artifacts/models/qwen3-0.6b-unlocked \
  --model-id Qwen/Qwen3-0.6B \
  --tensor-format f16

VITE_UNLOCKED_ALLOW_FIXTURE=false
VITE_UNLOCKED_MODEL_MANIFEST_PATH=/models/qwen3-0.6b-unlocked/manifest.json
VITE_UNLOCKED_MODEL_MANIFEST_SHA256=<64-character-sha256-hex-digest>
VITE_UNLOCKED_MANIFEST_FORMAT=sharded
VITE_UNLOCKED_WEIGHT_FORMAT=f16-packed
VITE_UNLOCKED_RUNTIME_PROFILE=full
VITE_QWEN_THINKING_MODE=disabled
```

The browser app defaults to `VITE_QWEN_THINKING_MODE=disabled` so assistant replies are visible immediately. Set it to `enabled` only for explicit hidden-reasoning evals; the client hides generated `<think>...</think>` text, so visible output may arrive much later. A converted `Qwen/Qwen3-1.7B` candidate can be generated the same way and is useful for stronger-model testing, but its packed artifact is still much larger and slower than the default 0.6B browser target.

Then verify the installed unlocked asset:

```bash
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

For shared or hosted memory, set `VITE_MEMORY_PROVIDER=remote-http` and point `VITE_REMOTE_MEMORY_URL` at a same-origin authenticated proxy or another implementation of `docs/51_REMOTE_MEMORY_API_CONTRACT.md`.

## 2a. Qwen unlocked asset hosting

The production model lane is a repo-owned unlocked Qwen runtime. It loads a verified sharded manifest plus tensor shards, not an opaque browser chat API. Keep converted weights out of git and either host them from `/models/...` in the web app's static public directory or from a COOP/COEP-compatible HTTPS origin.

For a hosted model-backed release:

```bash
VITE_LLM_BACKEND=unlocked-browser-transformer
VITE_DEFAULT_MODEL=Qwen/Qwen3-0.6B
VITE_REQUIRE_UNLOCKED_RUNTIME=true
VITE_UNLOCKED_ALLOW_FIXTURE=false
VITE_UNLOCKED_MODEL_MANIFEST_PATH=/models/qwen3-0.6b-unlocked/manifest.json
VITE_UNLOCKED_MODEL_MANIFEST_SHA256=<64-character-sha256-hex-digest>
VITE_UNLOCKED_MANIFEST_FORMAT=sharded
VITE_UNLOCKED_RUNTIME_PROFILE=full
```

Use `VITE_BUNDLE_UNLOCKED_MODEL=true` only for a deliberate private bundle. The open-source default is to serve model assets externally or use deterministic fixtures for runtime development.

## 2b. Embed on any website

Build the browser SDK:

```bash
pnpm --filter @infinite-edge-agent/browser-sdk build
```

Use it from a site that should host the agent:

```ts
import { mountInfiniteEdgeAgent } from "@infinite-edge-agent/browser-sdk";

mountInfiniteEdgeAgent({
  agentUrl: "https://agent.example.com",
  container: "#edge-agent",
  mode: "launcher"
});
```

Do not put API tokens or memory bearer tokens in SDK URLs. Use a same-origin proxy or secure cookie/session layer for hosted memory.

## 3. Run the remote-compatible memory service

The included memory server exposes both the local sidecar routes and the deployable remote namespace:

```text
GET/POST http://127.0.0.1:8787/api/edge-ai/*
```

For a remote memory deployment, set:

```bash
MEMORY_API_PREFIX=/api/edge-ai
MEMORY_SERVER_TOKEN=<server-side token>
MEMORY_TENANT_ID=<tenant-id>
MEMORY_CELL_ID=<cell-id>
```

Then point the browser app at the deployed base URL with matching `VITE_REMOTE_MEMORY_*` values, such as `VITE_REMOTE_MEMORY_URL=https://your-memory.example.com/api/edge-ai`. First-party internal deployments can be used for testing, but open-source users should provide their own endpoint and data store.

Do not configure a memory bearer token in a `VITE_*` browser variable for production. `VITE_*` values are public in the compiled browser bundle. Use secure cookies/session auth at a same-origin proxy, or have a trusted server inject `MEMORY_SERVER_TOKEN` when forwarding to the included memory server.

## 4. Run with LanceDB sidecar

```bash
cp .env.example .env
```

Edit `.env`:

```bash
VITE_ENABLE_MEMORY_SERVER=true
MEMORY_DB_URI=.data/lancedb
```

Run two terminals:

```bash
pnpm dev:memory
```

```bash
pnpm dev:web
```

## 5. Typecheck and test

```bash
pnpm typecheck
pnpm test
pnpm verify:unlocked
pnpm bench:browser-runtime
pnpm smoke:core
pnpm stress:core
pnpm eval:production
pnpm check:web-dist
```

`pnpm verify:unlocked` runs in generated-fixture mode unless a converted manifest is configured, so it is safe for open-source CI and fresh clones. `pnpm bench:browser-runtime` records profile/caps, CPU/WebGPU coverage, MTP acceptance, optional paired MTP acceleration metrics, expected-answer substring checks, browser-preview proof, and strict WebGPU gates. Set `BROWSER_RUNTIME_BENCH_EXPECTED_SUBSTRINGS` alongside `BROWSER_RUNTIME_BENCH_PROMPTS` for model-backed quality sentinels; direct Node runs and browser-preview runs both fail when a required substring is absent. Set `VITE_QWEN_THINKING_MODE=enabled` when the target Qwen manifest needs hidden thinking for factual quality; the client strips reasoning tags from visible output and forwards the mode to browser-preview checks. Set `BROWSER_RUNTIME_BENCH_REQUIRE_MTP_ACCELERATION=true` only on a device/profile where speculative decoding must prove faster than target-only. Set `BROWSER_RUNTIME_BENCH_REQUIRE_STRICT_WEBGPU=true` or pass `--require-strict-webgpu` only on a WebGPU-capable Node release lane where CPU fallback must fail the Node benchmark; this strict benchmark mode passes `requireWebGpu` into the unlocked client instead of measuring a CPU fallback decode. Strict real-Qwen release mode also requires `BROWSER_RUNTIME_BENCH_PREVIEW_URL` and sends `BROWSER_RUNTIME_BENCH_PREVIEW_REQUIRE_STRICT_WEBGPU=true`, `BROWSER_RUNTIME_BENCH_PREVIEW_REQUIRE_KV_REUSE=true`, and `BROWSER_RUNTIME_BENCH_PREVIEW_MIN_GENERATED_TOKENS=8` to the browser preview. `pnpm smoke:core` writes `.artifacts/evals/core-smoke-latest.json`, `pnpm stress:core` writes `.artifacts/evals/core-stress-latest.json`, and `pnpm eval:production` writes `.artifacts/evals/production-latest.json`. These artifacts should report `"passed": true` before release. `pnpm check:web-dist` verifies the static app bundle does not accidentally include local model weights.

## 6. Build

```bash
pnpm build
```

Before release, also browser-smoke the target deployment: verify the unlocked fixture or configured Qwen manifest path can initialize, answer a prompt, retrieve memory, and persist a runtime trace. In strict real-Qwen release lanes, `RELEASE_REQUIRE_BROWSER_PREVIEW_PROOF=true` is automatic; provide `BROWSER_RUNTIME_BENCH_PREVIEW_URL` so `browser-runtime-bench` can capture the completed browser-preview proof instead of failing early.

## Troubleshooting

### WebGPU not available

Use a recent Chrome or Edge build, run over `localhost` or HTTPS, and verify that `navigator.gpu` exists in DevTools.

### Unlocked model manifest missing

If `VITE_UNLOCKED_ALLOW_FIXTURE=false`, startup requires `VITE_UNLOCKED_MODEL_MANIFEST_PATH` to point at a served unlocked Qwen manifest. A hosted route that returns `200 text/html` is treated as a bad model route, because that usually means the SPA fallback served the app shell instead of the manifest.

### Unlocked model manifest hash mismatch

If `VITE_UNLOCKED_MODEL_MANIFEST_SHA256` is set, the release gate and browser readiness check require the served manifest digest to match exactly. Recompute the digest only from the exact manifest being deployed, and verify that any CDN or release step did not rewrite, compress, redirect, or replace the file unexpectedly.

### ONNX/embedding worker fails to load WASM files

Keep the Vite cross-origin headers in `apps/web/vite.config.ts`. Some ONNX Runtime Web paths and threading modes require cross-origin isolation.

### Memory sidecar unavailable

The web app automatically falls back to `browser-vector` IndexedDB memory if `VITE_ENABLE_MEMORY_SERVER=true` or `VITE_MEMORY_PROVIDER=sidecar` but the sidecar health check fails. Set `VITE_ALLOW_MEMORY_FALLBACK=false` only when you want startup to hard-fail instead.

### Remote memory unavailable

When `VITE_MEMORY_PROVIDER=remote-http`, the app tries `VITE_REMOTE_MEMORY_URL` first and falls back to `browser-vector` if the endpoint is unavailable. Set `VITE_ALLOW_MEMORY_FALLBACK=false` only when a deployment should block startup instead of using local browser memory. The default `browser-vector` provider keeps memory, deterministic vector search, import/export bundles, and context-pack traces local to the current browser profile through IndexedDB. `indexeddb` remains a compatibility alias for older configs.

### Memory provider presets

Use `VITE_MEMORY_PROVIDER=browser-vector` for the zero-config open-source browser mode. It needs no sidecar or remote endpoint and persists/searches memory in IndexedDB.

Use `VITE_MEMORY_PROVIDER=sidecar` plus `VITE_ENABLE_MEMORY_SERVER=true` when you want the optional local LanceDB sidecar.

Use `VITE_MEMORY_PROVIDER=remote-http` with `VITE_REMOTE_MEMORY_URL=https://your-memory.example.com/api/edge-ai` for a hosted provider that implements [51_REMOTE_MEMORY_API_CONTRACT.md](./51_REMOTE_MEMORY_API_CONTRACT.md). The open-source default does not assume any private hosted endpoint.

### Memory export/import

Use the runtime controls to export or import a JSON memory bundle. Export/import is supported for browser-vector/IndexedDB, the local sidecar, and remote HTTP providers that implement `GET /memory/export`, `POST /memory/import`, runtime trace export/import, and context-pack trace persistence.

### Browser tab crashes

Use the smallest model first, close other GPU-heavy tabs, and lower retrieved memory budgets in `apps/web/src/config.ts`.
