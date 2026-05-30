# 52 — Browser SDK and Unlocked Qwen Deployment

## Production shape

The deployable system has three separable pieces:

- Hosted browser app: `apps/web`, built as static assets.
- Browser SDK: `@infinite-edge-agent/browser-sdk`, which embeds the hosted app in any website.
- Memory API: browser-vector IndexedDB by default, or any server implementing `docs/51_REMOTE_MEMORY_API_CONTRACT.md`.

The hosted app ships registered Backend Broker lanes, not opaque ad hoc chat backends:

- `compiled-browser-webllm` is the compiled browser production candidate and uses the WebLLM/MLC artifact id `Qwen3-0.6B-q4f16_1-MLC`.
- `unlocked-browser-transformer` is the custom WebGPU Kernel Lab for SSA, KVSwap, TSP, direct tensor residency, and fusion experiments.
- `wasm-small-core` is reserved for bounded control/fallback tasks.

Deploy readiness is backend-specific. A strict Kernel Lab proof is valuable research evidence, but the hosted production answer path should be claimed only after the compiled backend passes its own Chrome quality, memory-grounding, speed, and trace gates.

## Qwen unlocked assets

For a model-backed unlocked release, convert local licensed Qwen artifacts and run strict verification:

```bash
pnpm convert:unlocked -- --input /path/to/Qwen3-0.6B --output .artifacts/models/qwen3-0.6b-unlocked --model-id Qwen/Qwen3-0.6B --tensor-format f16
pnpm verify:unlocked -- --manifest-path .artifacts/models/qwen3-0.6b-unlocked/manifest.json --manifest-sha256 <sha> --require-configured --require-manifest-sha256 --require-sharded --require-qwen-math --require-packed-assets --runtime-profile full --require-full-profile
```

`pnpm verify:unlocked` validates the versioned sharded manifest, explicit f16-packed tensor-storage metadata, shard SHA-256 values, tokenizer metadata, backend initialization, and tensor-control decode proof. If no production manifest is configured, it generates a tiny sharded fixture so the open-source release gate still covers the loader path without bundling real Qwen weights.

Set `RELEASE_REQUIRE_UNLOCKED_MODEL=true` for model-backed release gates. Keep `RELEASE_REQUIRE_UNLOCKED_PACKED_ASSETS=true` for production model-backed releases. Add `RELEASE_REQUIRE_UNLOCKED_STRICT_WEBGPU=true` when the release must reject all MLP/logit/attention/projection CPU fallback.

## Static hosting headers and CSP

Browser ML runtimes need WebGPU and often SharedArrayBuffer-compatible isolation. The included `vercel.json` sets:

```http
Cross-Origin-Opener-Policy: same-origin
Cross-Origin-Embedder-Policy: require-corp
Cross-Origin-Resource-Policy: cross-origin
```

For other hosts, configure equivalent headers on HTML, JS, workers, WASM, tokenizer, manifest, and shard responses. Serve model files with long-lived immutable caching after filenames and SHA values are pinned.

Recommended iframe and asset headers:

- App HTML and JS workers: `Cross-Origin-Opener-Policy: same-origin`, `Cross-Origin-Embedder-Policy: require-corp`, and `Cross-Origin-Resource-Policy: same-origin` when app and assets are same-origin, or `cross-origin` for a public CDN asset host.
- Model, tokenizer, and shard files: include `Cross-Origin-Resource-Policy: cross-origin` when served from a separate asset origin, plus `Access-Control-Allow-Origin` for the hosted app origin.
- Iframe embed host: set `frame-src https://agent.example.com` and `child-src https://agent.example.com` in the parent site CSP.
- Hosted agent app: set `frame-ancestors` to the allowed embed origins. Use a concrete allowlist in production.
- Workers and WASM: CSP should allow `worker-src 'self' blob:` and `script-src 'self' 'wasm-unsafe-eval'` when required by the target browser/runtime bundle.
- Remote memory API: include the remote API origin in `connect-src`; prefer a same-origin `/api/edge-ai` proxy so browser CSP can remain tight.

The SDK iframe uses `allow="webgpu; cross-origin-isolated"` by default. If a host overrides `allow`, it must preserve WebGPU/cross-origin-isolated permissions.

## SDK usage

```ts
import { mountInfiniteEdgeAgent } from "@infinite-edge-agent/browser-sdk";

const agent = mountInfiniteEdgeAgent({
  agentUrl: "https://agent.example.com",
  container: "#edge-agent",
  mode: "launcher",
  tenantId: "public-site",
  cellId: "support",
  sessionId: "visitor-session",
  deployment: {
    preset: "remote-http",
    runtimeProfile: "unlocked-browser",
    modelProfile: "qwen3-0.6b-sharded",
  },
});
```

The SDK refuses non-local HTTP URLs by default and blocks URL-carried credentials such as URL userinfo, token-shaped params, OAuth codes, signed URL params, bearer values, OpenAI-style `sk-...` keys, raw JWT-looking values, and URL fragments. Do not pass memory bearer tokens through SDK options or URLs. Use a same-origin proxy, secure cookies, or server-side token injection for hosted memory.

## Deployment presets

SDK deployment presets are public requested routing hints. They are not credentials, they do not replace server-side memory auth, and they do not prove the hosted app observed that runtime mode.

| Preset | Requested query proof | Hosted app requirement | Intended use |
| --- | --- | --- | --- |
| `browser-only` | `deploymentPreset=browser-only&memoryMode=browser-vector&sidecar=disabled` | App build uses IndexedDB/browser vectors only | Open-source static demo, no server memory |
| `sidecar-disabled` | `deploymentPreset=sidecar-disabled&memoryMode=browser-vector&sidecar=disabled` | App build must not depend on local LanceDB sidecar | Hosted/static embed where LanceDB sidecar is unavailable |
| `remote-http` | `deploymentPreset=remote-http&memoryMode=remote-http&sidecar=disabled` | App deployment must already be configured for a same-origin authenticated remote API | Production app with authenticated memory endpoint |

The app only honors SDK `remote-http` hints when `VITE_REMOTE_MEMORY_URL` is already a same-origin path such as `/api/edge-ai`. An arbitrary iframe query string cannot force the app to call a third-party memory endpoint.

`tenantId`, `cellId`, and `sessionId` are public routing identifiers. They scope app state and remote memory calls, but they are not authorization. A production memory endpoint must authenticate and authorize every request.

## Memory modes

- `browser-vector`: default open-source mode; all memory remains in the visitor browser profile.
- `indexeddb`: compatibility alias for browser-local memory.
- `remote-http`: deployable memory API; use a same-origin authenticated proxy or an API compatible with `docs/51_REMOTE_MEMORY_API_CONTRACT.md`.

For production remote memory, prefer:

```bash
VITE_MEMORY_PROVIDER=remote-http
VITE_REMOTE_MEMORY_URL=/api/edge-ai
VITE_REMOTE_MEMORY_CREDENTIALS=include
```

The bundled memory server's `/api/edge-ai` namespace is private and single-scope unless wrapped by your own auth layer. It requires `MEMORY_SERVER_TOKEN`, `MEMORY_TENANT_ID`, and `MEMORY_CELL_ID`. Root local sidecar routes are exposed only on loopback by default; set `MEMORY_EXPOSE_LOCAL_ROUTES=true` only behind a trusted private wrapper.

## Hosted preflight

For the compiled production profile, the repo includes a machine-checkable environment and benchmark URL verifier:

```bash
VITE_LLM_BACKEND=compiled-browser-webllm
VITE_DEFAULT_MODEL=Qwen3-0.6B-q4f16_1-MLC
VITE_COMPILED_WEBLLM_ENABLED=true
VITE_REQUIRE_UNLOCKED_RUNTIME=false
VITE_MTP_ENABLED=false
VITE_BENCHMARK_TELEMETRY_ENABLED=true
VITE_BENCHMARK_TELEMETRY_URL=/api/benchmark-runs
BENCHMARK_TELEMETRY_ENABLED=true
BENCHMARK_TELEMETRY_STORAGE=postgres
BENCHMARK_TELEMETRY_DATABASE_URL=<postgres-connection-string>
BENCHMARK_TELEMETRY_ADMIN_TOKEN=<dashboard-export-token>
HOSTED_PRODUCTION_BENCHMARK_URL='https://agent.example.com/__bench/browser-runtime?backend=compiled-browser-webllm&modelId=Qwen3-0.6B-q4f16_1-MLC&memoryGrounding=montana_capital&expectedExact=Helena&submitTelemetry=true&qwenThinkingMode=disabled'
pnpm verify:hosted-profile
```

The verifier fails if the hosted profile points at the Kernel Lab, uses local JSONL telemetry for hosted production, leaves dashboard/export routes unprotected, or omits the grounded exact `Helena` benchmark with telemetry opt-in.

The verifier writes `.artifacts/evals/hosted-deployment-profile-latest.json` plus a timestamped artifact under `.artifacts/evals/hosted-deployment-profile/`. Set `RELEASE_REQUIRE_HOSTED_PROFILE=true` to make `pnpm release:gate` run this verifier and include the hosted profile artifact in the release summary.

Backend-specific readiness is captured separately:

```bash
pnpm eval:backend-readiness
```

That writes `.artifacts/evals/backend-readiness-matrix-latest.json`, marking `compiled-browser-webllm` as the deploy candidate only when hosted profile proof passes and keeping `unlocked-browser-transformer` recorded as the research Kernel Lab. `RELEASE_REQUIRE_HOSTED_PROFILE=true` includes this matrix automatically; `RELEASE_REQUIRE_BACKEND_READINESS_MATRIX=true` can require it without the full hosted profile flag.

Shared memory/context runtime proof is captured by:

```bash
pnpm eval:shared-runtime
```

That writes `.artifacts/evals/shared-runtime-readiness-latest.json`, recording that retrieval, context rebuild, context-pack traces, runtime traces, and backend profile routing are above the backend boundary. `RELEASE_REQUIRE_HOSTED_PROFILE=true` includes this proof automatically; `RELEASE_REQUIRE_SHARED_RUNTIME_READINESS=true` can require it independently.

The combined v12 final-state bundle is:

```bash
pnpm eval:v12-readiness
```

That writes `.artifacts/evals/v12-readiness-bundle-latest.json`, requiring hosted compiled profile proof, backend-specific deploy/research roles, and the shared memory/context runtime proof to pass together.

Before shipping a hosted embed, verify app and model URLs behave like assets instead of falling back to HTML:

```bash
curl -I https://agent.example.com/
curl -I https://agent.example.com/assets/index.js
curl -I https://agent.example.com/models/qwen3-0.6b-unlocked/manifest.json
curl -I https://agent.example.com/models/qwen3-0.6b-unlocked/shards/<shard>.bin
curl -i https://agent.example.com/api/edge-ai/health
```

Expected results:

- HTML, JS, workers, manifest, tokenizer, and shard responses are `200`, not SPA fallback HTML.
- Model shard responses are binary/static asset responses, never `text/html`.
- `/api/edge-ai/health` returns JSON such as `{ "ok": true, "mode": "remote-http" }`; app-shell HTML is rejected by the browser remote client.
- COOP, COEP, CORP, CORS, and cache headers are present on the right origins.
- `pnpm smoke:sdk` builds the SDK package, verifies package import, checks iframe URL safety, and records `.artifacts/evals/sdk-smoke-latest.json`.
