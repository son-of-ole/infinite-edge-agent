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

## Recommended Storage

For Replit-hosted deployments, use hosted SQL/Postgres rather than writing JSON files to the app filesystem. Deployment filesystems are not a durable benchmark database.

The database should store one row per benchmark run and keep the raw artifact JSON for later analysis.

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

```text
/__bench/browser-runtime?backend=compiled-browser-webllm&memoryGrounding=montana_capital&expectedExact=Helena&submitTelemetry=true
```

The current implementation submits a sanitized artifact. Raw prompts, raw responses, expected strings, and token diagnostics are redacted before upload.

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
    gpuVendor?: string;
    gpuArchitecture?: string;
    gpuDevice?: string;
    gpuDescription?: string;
    webglRenderer?: string;
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

## Privacy Rules

Benchmark telemetry can become browser fingerprinting if it is collected silently. Use these rules:

- make full benchmark submission opt-in,
- do not store raw IP addresses,
- hash user/session ids if a stable id is needed,
- store only fixed public canary prompts and outputs,
- never upload private chat content,
- show the user what will be submitted,
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
