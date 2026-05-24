# Operations

## Local release checklist

- Run `pnpm release:gate` from the repository root. In npm-only shells, use `npm exec pnpm -- run release:gate`.
- Confirm `.artifacts/evals/release-gate-latest.json` reports `passed: true`.
- If running manually, use the same sequence: `pnpm typecheck`, `pnpm test`, `pnpm verify:unlocked`, `pnpm smoke:core`, `pnpm smoke:sdk`, `pnpm stress:core`, `pnpm eval:qwen-parity`, `pnpm eval:production`, `pnpm build`, and `pnpm check:web-dist`.
- For model-backed unlocked releases, first run `pnpm convert:unlocked -- --input /path/to/Qwen3-0.6B --output .artifacts/models/qwen3-0.6b-unlocked --model-id Qwen/Qwen3-0.6B --tensor-format f16`, then verification with `pnpm verify:unlocked -- --manifest-path .artifacts/models/qwen3-0.6b-unlocked/manifest.json --manifest-sha256 <sha> --require-configured --require-manifest-sha256 --require-sharded --require-qwen-math --require-packed-assets --runtime-profile full --require-full-profile`. Local `release:gate` promotes a configured or installed Qwen manifest/SHA into this real lane by default, including packed f16 model assets, Qwen math/parity, KV decode reuse, `browser-vector` memory, and a strict real-browser WebGPU proof. MTP remains target-only/off unless `RELEASE_REQUIRE_MTP_ACCELERATION=true` proves paired speedup; use `RELEASE_ALLOW_FIXTURE_GATE=true` only for explicit fixture/dev gates.
- Browser smoke the target OS/browser. Confirm the unlocked Qwen backend initializes, answers, retrieves memory, and persists a runtime trace.
- Sidecar smoke if packaged: `/memory/status`, `/memory/repair`, and `/runtime/traces`.
- Package static files. Local Qwen shards are omitted from `apps/web/dist` by default unless `VITE_BUNDLE_UNLOCKED_MODEL=true`; hosted deployments should serve shards from a large-file-capable asset host/CDN or intentionally bundle them in a private build.
- Package memory sidecar if used.
- Confirm sidecar binds to `127.0.0.1`.
- Confirm no API keys are bundled.
- Confirm SDK embed URLs contain no memory tokens, API keys, or bearer credentials.
- Confirm model licenses.
- Confirm memory deletion works.
- Confirm memory export/import works for the selected provider.

## Observability

Add counters for:

- `model_load_ms`
- `embedding_ms`
- `memory_search_ms`
- `prompt_tokens_estimated`
- `generation_ttft_ms`
- `tokens_per_second`
- `memory_chunks_total`
- `memory_store_mode`

Keep telemetry local unless the user opts in.

## Data retention

Recommended defaults:

- Raw chat chunks: keep until user deletes.
- Ephemeral scratchpad: 24 hours.
- Summaries: keep until user deletes.
- Tool outputs: configurable per tool.

## Backup and restore

The browser app now exposes memory export/import as a JSON bundle. The same bundle works with IndexedDB, the sidecar, or any remote API implementing `docs/51_REMOTE_MEMORY_API_CONTRACT.md`.

For sidecar deployments, also back up `.data/lancedb` as the provider-native database snapshot.

## Eval artifacts

Core smoke writes `.artifacts/evals/core-smoke-latest.json`.

SDK smoke writes `.artifacts/evals/sdk-smoke/<timestamp>/results.json`, `trace.jsonl`, and `summary.md`, plus `.artifacts/evals/sdk-smoke-latest.json`.

Production eval writes `.artifacts/evals/production/<timestamp>/results.json`, `trace.jsonl`, and `summary.md`, plus `.artifacts/evals/production-latest.json` for the quick release gate.

Qwen parity accuracy writes `.artifacts/evals/qwen-parity-accuracy/<timestamp>/results.json` and `summary.md`, plus `.artifacts/evals/qwen-parity-accuracy-latest.json`. The suite is deterministic and browser-free by default; broader local eval datasets should live outside source control or under gitignored artifacts, with only dataset schema/docs committed.

Core stress writes durable timestamped artifacts:

```text
.artifacts/evals/core-stress/<timestamp>/results.json
.artifacts/evals/core-stress/<timestamp>/trace.jsonl
.artifacts/evals/core-stress/<timestamp>/summary.md
```

Use `.artifacts/evals/core-stress-latest.json` for the quick release gate. The stress eval accepts local scale overrides with `--vectors`, `--dim`, `--top-k`, `--memory-token-budget`, `--recent-messages`, `--seed`, and `--max-search-ms`, or the equivalent `CORE_STRESS_*` environment variables.

Release gate writes `.artifacts/evals/release-gate/<timestamp>/results.json`, `trace.jsonl`, and `summary.md`, plus `.artifacts/evals/release-gate-latest.json`. The release summary pulls the latest local core smoke, SDK smoke, stress, Qwen parity accuracy, unlocked verification, and production eval artifacts into one performance/accuracy table.

## Sidecar eval operation

`pnpm eval:production` uses `PRODUCTION_EVAL_SIDECAR_MODE=auto` by default. In auto mode, it checks `MEMORY_SERVER_URL` or `VITE_MEMORY_SERVER_URL`, verifies the sidecar reports the expected `dbUri` and `tableName`, records a clear skipped sidecar suite if nothing matching is listening, and still runs the browser-free production gates.

For packaged sidecar release candidates, make the sidecar blocking:

```bash
PRODUCTION_EVAL_SIDECAR_MODE=required pnpm eval:production
```

To let the eval start the sidecar itself, provide a long-running start command:

```bash
PRODUCTION_EVAL_SIDECAR_START_COMMAND="pnpm --filter @infinite-edge-agent/memory-server start" pnpm eval:production
```

The eval waits up to `PRODUCTION_EVAL_SIDECAR_TIMEOUT_MS` for `/health`, then shuts down a sidecar process it started. Use `PRODUCTION_EVAL_SIDECAR_MODE=skip` only when a browser-only release intentionally excludes the local sidecar.

`pnpm release:gate` follows the browser-first open-source lane by default, so the sidecar remains non-blocking unless `RELEASE_REQUIRE_SIDECAR=true` is set. For sidecar-packaged release candidates, run the durable sidecar profile and make it blocking:

```bash
MEMORY_DB_URI=.data/lancedb MEMORY_TABLE=memory_chunks MEMORY_VECTOR_DIMENSION=384 pnpm dev:memory
RELEASE_REQUIRE_SIDECAR=true pnpm release:gate
```

For encrypted local sidecar storage, also set a strong non-public key:

```bash
MEMORY_ENCRYPTION_KEY="replace-with-a-long-random-secret" \
MEMORY_DB_URI=.data/lancedb \
MEMORY_TABLE=memory_chunks \
MEMORY_VECTOR_DIMENSION=384 \
pnpm dev:memory
```

The sidecar reports `encryptionEnabled: true` from `/health`, `/memory/status`, and `/memory/repair` when the key is active.
