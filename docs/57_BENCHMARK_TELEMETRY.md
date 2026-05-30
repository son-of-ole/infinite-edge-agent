# Benchmark Telemetry Plan

Infinite Edge Agent should be able to collect opt-in benchmark results from hosted users and store them in a database. This turns scattered device notes into a real browser/GPU performance matrix.

## Goal

When someone visits the hosted app, they should be able to run a benchmark and contribute an anonymous result:

```text
Hosted app
  -> benchmark route
  -> device/browser/GPU capability probe
  -> grounded exact-answer canary
  -> POST artifact to server
  -> database row
  -> public dashboard/export
```

## Current Implementation

The repo now includes an optional telemetry collector in `apps/memory-server`:

- `POST /api/benchmark-runs` validates and stores one benchmark run.
- `GET /api/benchmark-runs?limit=100` lists recent runs.
- `GET /api/benchmark-runs/summary` returns small aggregate counts.
- `GET /api/benchmark-runs/dashboard` renders a minimal sanitized dashboard with backend, OS/browser, GPU label, speed, and readiness fields.
- `GET /api/benchmark-runs/export.csv` exports sanitized run metadata, including GPU adapter/WebGL renderer fields, for spreadsheet analysis.

Enable it with:

```bash
BENCHMARK_TELEMETRY_ENABLED=true
BENCHMARK_TELEMETRY_PREFIX=/api/benchmark-runs
BENCHMARK_TELEMETRY_STORAGE=jsonl
BENCHMARK_TELEMETRY_DIR=.data/benchmark-runs
BENCHMARK_TELEMETRY_MAX_ARTIFACT_BYTES=1048576
BENCHMARK_TELEMETRY_RATE_LIMIT_MAX=60
BENCHMARK_TELEMETRY_RATE_LIMIT_WINDOW_MS=600000
BENCHMARK_TELEMETRY_ADMIN_TOKEN=<dashboard-export-token>
```

The included collector defaults to JSONL so it works in local development and simple hosted environments without adding another service. For durable hosted storage, set:

```bash
BENCHMARK_TELEMETRY_STORAGE=postgres
BENCHMARK_TELEMETRY_DATABASE_URL=<postgres-connection-string>
```

The Postgres adapter uses the same store contract as JSONL, creates the `benchmark_runs` table if needed, and writes the sanitized artifact as `jsonb`. It loads the optional `pg` package at runtime only when Postgres storage is selected, so local JSONL development does not require a database client dependency.

The server validates the submitted payload and sanitizes the artifact again before writing it, so it does not rely only on the browser-side redaction path. The dashboard and CSV export intentionally render only run metadata, GPU/device class labels, and pass/fail/speed fields, not prompt or response text.

The submit endpoint is rate-limited by client identity. Configure `BENCHMARK_TELEMETRY_RATE_LIMIT_MAX` and `BENCHMARK_TELEMETRY_RATE_LIMIT_WINDOW_MS` for the hosted traffic profile. Set `BENCHMARK_TELEMETRY_SUBMIT_TOKEN` only for private benchmark collectors; do not expose a submit secret through `VITE_*` browser variables. Set `BENCHMARK_TELEMETRY_ADMIN_TOKEN` to protect list, summary, dashboard, and CSV export routes.

## Recommended Durable Storage

For Replit-hosted deployments, use hosted SQL/Postgres rather than writing JSON files to the app filesystem. Deployment filesystems are not a durable benchmark database. The JSONL store remains useful for local proof and static/simple demos; production telemetry should use `BENCHMARK_TELEMETRY_STORAGE=postgres`.

The database should store one row per benchmark run and keep the sanitized artifact JSON for later analysis.

## Suggested Table

```sql
create table benchmark_runs (
  id text primary key,
  created_at timestamptz not null default now(),
  app_version text,
  git_sha text,
  deploy_url text,
  benchmark_profile text not null,
  backend_id text not null,
  model_id text,
  os text,
  browser_name text,
  browser_version text,
  user_agent_hash text,
  mobile boolean,
  hardware_concurrency integer,
  device_memory_gb numeric,
  screen_width integer,
  screen_height integer,
  webgpu_available boolean,
  gpu_vendor text,
  gpu_architecture text,
  gpu_device text,
  gpu_description text,
  webgl_renderer text,
  init_load_ms numeric,
  time_to_first_token_ms numeric,
  tokens_per_second numeric,
  generated_tokens integer,
  retrieval_ms numeric,
  context_rebuild_ms numeric,
  corpus_count integer,
  production_deploy_ready_passed boolean,
  compiled_backend_ready_passed boolean,
  memory_grounding_passed boolean,
  expected_exact_passed boolean,
  production_speed_floor_passed boolean,
  cpu_fallback_used boolean,
  artifact_json jsonb not null
);
```

## Client Payload

The browser benchmark route now includes an opt-in telemetry submitter. It only submits when both are true:

- deployment config enables and points telemetry at an endpoint,
- the benchmark URL includes `submitTelemetry=true` or `benchmarkTelemetry=true`.

Example:

```bash
VITE_BENCHMARK_TELEMETRY_ENABLED=true
VITE_BENCHMARK_TELEMETRY_URL=/api/benchmark-runs
VITE_APP_VERSION=0.1.0
VITE_GIT_SHA=<deployment-sha>
VITE_DEPLOY_URL=https://your-hosted-agent.example
```

`pnpm verify:hosted-profile` requires `VITE_DEPLOY_URL` as the canonical hosted origin. If `HOSTED_PRODUCTION_BENCHMARK_URL` is provided, that benchmark URL must still point at the same public HTTPS origin; the verifier reports this as `hostedProfileBenchmarkDeployUrlBound`.

For v12 production proof, the browser benchmark artifact carries this deployment SHA as `v12ProductionProofSourceGitSha` and carries the hosted origin as `deployUrl`. Release verification should set `HOSTED_BENCHMARK_EXPECTED_GIT_SHA=<deployment-sha>` so stale hosted artifacts from another build fail instead of being reused. `deployUrl` must be a public HTTPS origin and must match `HOSTED_BENCHMARK_EXPECTED_DEPLOY_URL` or `VITE_DEPLOY_URL`; local, loopback, link-local, private-network, non-HTTPS, and wrong-origin artifacts are rejected by hosted proof gates.

```text
/__bench/browser-runtime?backend=compiled-browser-webllm&memoryGrounding=montana_capital&expectedExact=Helena&submitTelemetry=true
```

The current implementation submits a sanitized artifact. Raw prompts, raw responses, expected strings, and token diagnostics are redacted before upload. When available, the browser also includes WebGPU adapter info (`vendor`, `architecture`, `device`, `description`) and a WebGL renderer string so cross-device benchmarks can distinguish Apple, NVIDIA, AMD, Intel, and mobile GPU classes without storing the raw user agent.

The client submits:

```ts
interface BenchmarkTelemetryPayload {
  runId: string;
  appVersion: string;
  gitSha: string;
  deployUrl: string;
  benchmarkProfile: string;
  backendId: string;
  modelId: string;
  device: {
    os: string;
    browserName: string;
    browserVersion: string;
    mobile: boolean;
    hardwareConcurrency: number | null;
    deviceMemoryGb: number | null;
    screen: { width: number; height: number };
    webgpuAvailable: boolean;
    gpuVendor: string | null;
    gpuArchitecture: string | null;
    gpuDevice: string | null;
    gpuDescription: string | null;
    webglRenderer: string | null;
  };
  summary: {
    initLoadMs: number | null;
    timeToFirstTokenMs: number | null;
    tokensPerSecond: number | null;
    memoryGroundingPassed: boolean;
    expectedExactPassed: boolean;
    productionDeployReadyPassed: boolean;
  };
  artifactJson: unknown; // sanitized benchmark artifact
}
```

## Hosted Profile Verification

Use the hosted profile verifier before calling a deployment production-ready:

```bash
pnpm verify:hosted-profile
```

For the hosted compiled profile, it expects:

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
HOSTED_BENCHMARK_EXPECTED_GIT_SHA=<deployment-sha>
HOSTED_BENCHMARK_REQUIRE_SOURCE_BOUND=true
```

If `HOSTED_PRODUCTION_BENCHMARK_URL` is omitted, the verifier can generate the canonical URL from `VITE_DEPLOY_URL`. The benchmark URL must resolve to a public HTTPS origin and must include the canonical compiled model id, `modelId=Qwen3-0.6B-q4f16_1-MLC`; localhost, loopback, link-local, private-network, and non-HTTPS origins are rejected. The generated URL still has to be run in real Chrome or Edge for authoritative proof.

The verifier writes `.artifacts/evals/hosted-deployment-profile-latest.json`, and `RELEASE_REQUIRE_HOSTED_PROFILE=true pnpm release:gate` includes that artifact in the release summary. This is the deploy-profile proof; the benchmark route plus telemetry database remain the runtime proof.

`pnpm eval:backend-readiness` writes `.artifacts/evals/backend-readiness-matrix-latest.json`. The matrix links the hosted profile proof to the compiled backend, keeps the custom WebGPU runtime recorded as Kernel Lab instead of production answer proof, and keeps `wasm-small-core` bounded to fallback-only. It also reports `backendReadinessModelRegistryAligned`, `backendReadinessPublicDeployOptionCount`, and `backendReadinessPublicKernelLabOptionCount` so public model options cannot drift away from Backend Broker roles. In strict production archive runs, the matrix also requires the saved hosted benchmark proof before it marks the compiled backend deploy-ready.

`pnpm eval:shared-runtime` writes `.artifacts/evals/shared-runtime-readiness-latest.json`. That proof keeps benchmark telemetry tied to the shared memory/context runtime across the compiled deploy backend, Kernel Lab, and bounded fallback instead of treating each backend as a separate product lane. It also carries `sharedRuntimeModelRegistryAligned`, `sharedRuntimePublicDeployOptionCount`, and `sharedRuntimePublicKernelLabOptionCount`.

`pnpm eval:v12-readiness` writes `.artifacts/evals/v12-readiness-bundle-latest.json`, which is the combined artifact to archive next to hosted Chrome benchmark and telemetry database evidence. The bundle requires deploy/Kernel Lab/fallback role-boundary proof plus explicit `model_registry_alignment` and `production_proof_workflow` requirements.

`pnpm eval:v12-suite` writes the complete final-state artifact set with one timestamp, including `.artifacts/evals/v12-readiness-suite-latest.json`. The suite includes the hosted profile, backend readiness matrix, shared runtime readiness, v12 readiness bundle, and v12 production workflow preflight. When `HOSTED_BENCHMARK_ARTIFACT_PATH` is set, it also includes `hosted-benchmark-proof-latest.json`. This is the preferred archive command when publishing hosted benchmark telemetry results across devices.

`pnpm eval:v12-production` is the strict production archive command for telemetry-backed release claims. It requires the saved hosted benchmark artifact, includes hosted benchmark proof, requires proof schema version `2`, requires Backend Broker selection and role-boundary evidence, requires `modelId=Qwen3-0.6B-q4f16_1-MLC` both in the hosted benchmark artifact and the Broker selected model, requires a public HTTPS `deployUrl` bound to the expected deploy URL, requires model-registry alignment, binds backend readiness to that proof, requires successful benchmark telemetry submission, requires the fallback backend to remain fallback-only, requires `v12ProductionWorkflowPreflightPassed`, and writes `.artifacts/evals/v12-production-archive-latest.json`. With `RELEASE_REQUIRE_V12_PRODUCTION=true`, `pnpm release:gate` requires that archive and validates its backend-specific proof fields before reporting production release proof.

`pnpm verify:hosted-benchmark-proof` validates the saved real Chrome/Edge benchmark artifact itself. Set `HOSTED_BENCHMARK_ARTIFACT_PATH` to the saved `browser-runtime-bench-latest.json` or pass the path after `--`. For release proof, also set `HOSTED_BENCHMARK_EXPECTED_GIT_SHA` and `HOSTED_BENCHMARK_REQUIRE_SOURCE_BOUND=true`. This is the runtime proof that should sit beside telemetry database exports. It requires concrete run-level memory evidence: expected memory ids, retrieved ids, context-included ids, and retrieval rank for the grounded answer. It also requires successful telemetry submission evidence from the benchmark summary, GPU label evidence from WebGPU adapter info or WebGL renderer data, plus browser artifact proof that `compiled-browser-webllm` is the deploy backend, `Qwen3-0.6B-q4f16_1-MLC` is the deployed model, `unlocked-browser-transformer` is the Kernel Lab, and `wasm-small-core` is fallback-only.

For GitHub-hosted release verification, the manual **V12 Production Proof** workflow accepts exactly one of `HOSTED_BENCHMARK_ARTIFACT_JSON`, `HOSTED_BENCHMARK_ARTIFACT_BASE64`, or `HOSTED_BENCHMARK_ARTIFACT_URL` through workflow inputs. URL-sourced artifacts must use public HTTPS URLs and cannot target localhost, loopback, link-local, or private-network hosts; for local or private artifacts, use pasted JSON or base64 instead. Internally it runs:

```bash
pnpm materialize:hosted-benchmark
pnpm verify:hosted-profile
pnpm verify:hosted-benchmark-proof
pnpm eval:v12-production
RELEASE_REQUIRE_V12_PRODUCTION=true pnpm release:gate
```

This keeps the normal open-source CI fixture-safe while still giving release operators a remote, artifact-producing v12 production proof lane for real hosted browser benchmark results.

The v12 workflow preflight verifies that the manual workflow passes `deploy_url` into `VITE_DEPLOY_URL` and passes the optional benchmark URL override into `HOSTED_PRODUCTION_BENCHMARK_URL`. The hosted benchmark verifier then compares the saved artifact `deployUrl` to that expected deploy origin. This keeps the hosted profile, browser artifact, telemetry row, and production archive tied to the same public hosted origin.

## Privacy Rules

Benchmark telemetry can become browser fingerprinting if it is collected silently. Use these rules:

- make full benchmark submission opt-in,
- do not store raw IP addresses,
- hash user/session ids if a stable id is needed,
- store only fixed public canary prompts and outputs,
- never upload private chat content,
- show the user what will be submitted, including GPU adapter/WebGL renderer fields when available,
- allow deletion/export if account-based telemetry is added later.

## Public Dashboard

A useful first dashboard should show:

- pass rate by browser and OS,
- median tokens/sec by GPU class,
- cold init time by device class,
- mobile vs desktop comparison,
- integrated GPU vs discrete GPU comparison,
- memory-grounding pass/fail history,
- latest production-ready backend version.

## Acceptance Gate

Telemetry is production-ready when:

- the database write path is authenticated or abuse-limited,
- invalid artifacts are rejected,
- private prompt text is not accepted,
- the dashboard aggregates without exposing user-specific fingerprints,
- and at least one hosted run from each target device class is saved.
