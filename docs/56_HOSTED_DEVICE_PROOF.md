# Hosted Device Proof

This document records the current real-device validation state for Infinite Edge Agent.

The goal is not to claim native desktop throughput from a browser. The goal is to prove that the hosted browser runtime works across realistic devices, reports its backend honestly, and can answer from grounded memory when retrieval is required.

## Current Hosted Surface

The current hosted validation was run from a Replit-hosted browser deployment.

The production-candidate backend is:

```text
compiled-browser-webllm
```

The current default compiled model target is:

```text
Qwen3-0.6B-q4f16_1-MLC
```

The custom unlocked WebGPU transformer remains a Kernel Lab and research lane. Its benchmark results should not be mixed with the compiled backend deploy claim.

## Device Matrix

| Device class | Browser/runtime | Observed result | Notes |
|---|---|---|---|
| macOS laptop/desktop | Chrome, WebGPU | Passed and fast | Local browser proof and hosted app proof both looked strong. |
| Windows machine with discrete GPU | Chrome or Edge, WebGPU | Passed and fast | Expected best browser performance class. |
| Windows machine with Intel integrated GPU | Chrome or Edge, WebGPU | Passed, slower | Slowness is expected for this GPU class; functional behavior matters most. |
| iPhone 17 | Mobile browser | Passed and quick | Important mobile proof because phone-class browser runtimes are constrained. |

## Grounded Canary

The current production canary asks the hosted app to answer from seeded memory:

```text
Using retrieved memory only, in the Cedar Ridge operations dossier, which city is listed as the Montana field office hub? Answer with only the city.
```

Expected exact output:

```text
Helena
```

The local production artifact for the compiled backend reported:

```json
{
  "runtimeBackendId": "compiled-browser-webllm",
  "backendBrokerSelectionPassed": true,
  "backendBrokerSelectedBackendId": "compiled-browser-webllm",
  "productionDeployReadyPassed": true,
  "compiledBackendReadyPassed": true,
  "memoryGroundingPassed": true,
  "expectedExactPassed": true,
  "productionSpeedFloorPassed": true,
  "response": "Helena"
}
```

## What Counts As Production Proof

A hosted run counts only when it proves all of this:

- the app loads from HTTPS or an equivalent hosted secure context,
- WebGPU or the compiled browser runtime initializes,
- memory grounding is required for the factual canary,
- the expected memory id is retrieved and packed into context with run-level retrieved/included id evidence,
- the expected memory hit has a concrete retrieval rank,
- the exact answer check passes,
- the backend id is `compiled-browser-webllm`,
- `productionDeployReadyPassed` is true,
- and direct model factual output is not counted as retrieval proof.

## Hosted Profile Verifier

Before a hosted deploy-ready claim, run the environment and benchmark URL verifier:

```bash
pnpm verify:hosted-profile
```

The verifier checks the deploy profile, not the model output. It requires:

- `compiled-browser-webllm` as the hosted production backend,
- `VITE_COMPILED_WEBLLM_ENABLED=true`,
- MTP disabled in production,
- benchmark telemetry enabled on both browser and collector,
- `BENCHMARK_TELEMETRY_STORAGE=postgres`,
- admin protection for list, summary, dashboard, and CSV export routes,
- rate limiting configured,
- and a public HTTPS benchmark URL that runs `memoryGrounding=montana_capital`, `expectedExact=Helena`, and `submitTelemetry=true`.

The verifier does not replace the real Chrome benchmark. It prevents a hosted environment from being called production-ready before the authoritative browser proof can even produce the right artifact.

The verifier writes `.artifacts/evals/hosted-deployment-profile-latest.json` so the deploy profile can be archived alongside the real browser benchmark artifact. Set `RELEASE_REQUIRE_HOSTED_PROFILE=true` when the full release gate should include this proof.

The backend-specific matrix is written by:

```bash
pnpm eval:backend-readiness
```

It records `compiled-browser-webllm` as the deploy backend only when the hosted profile passes, records `unlocked-browser-transformer` as `research_only`, and records `wasm-small-core` as `fallback_only`. In strict production archive mode, the matrix also requires the saved hosted browser benchmark proof before `compiled-browser-webllm` can be marked deploy-ready. That proof must be source-bound to the expected deployment commit. The summary distinguishes `backendReadinessCompiledHostedProfilePassed` from `backendReadinessCompiledDeployReady`, and exposes `backendReadinessRoleBoundaryPassed`. This prevents Kernel Lab proof, fallback capability, or stale hosted proof from being mixed into the compiled backend deploy claim.

The shared runtime proof is written by:

```bash
pnpm eval:shared-runtime
```

It records that memory providers, retrieval, context rebuild, context-pack trace persistence, runtime trace persistence, and backend profile routing are shared above the backend boundary for the compiled deploy backend, Kernel Lab, and bounded fallback. This is the artifact that prevents the compiled backend, Kernel Lab, and fallback from becoming separate product runtimes.

The combined v12 readiness bundle is written by:

```bash
pnpm eval:v12-readiness
```

It writes `.artifacts/evals/v12-readiness-bundle-latest.json`, combining hosted profile proof, backend-specific deploy/Kernel Lab/fallback role proof, and shared runtime proof into one final-state deploy readiness artifact.

The full v12 readiness suite is written by:

```bash
pnpm eval:v12-suite
```

It writes the hosted profile, backend readiness matrix, shared runtime readiness, v12 readiness bundle, and `.artifacts/evals/v12-readiness-suite-latest.json` with the same timestamp. If `HOSTED_BENCHMARK_ARTIFACT_PATH` is set, it also validates and writes `hosted-benchmark-proof-latest.json` as part of the same suite. Set `HOSTED_BENCHMARK_REQUIRE_SOURCE_BOUND=true` when the suite should require source-bound verification against `HOSTED_BENCHMARK_EXPECTED_GIT_SHA` or `GITHUB_SHA`; the suite summary exposes this as `v12SuiteHostedBenchmarkProofSourceBoundRequired`. Archive this suite next to hosted Chrome/Edge benchmark evidence when making a backend-specific deploy-readiness claim.

The strict production archive is written by:

```bash
pnpm eval:v12-production
```

It requires `HOSTED_BENCHMARK_ARTIFACT_PATH`, forces hosted benchmark proof, writes the v12 suite, and writes `.artifacts/evals/v12-production-archive-latest.json`. This is the preferred final archive for a production-ready claim. When `RELEASE_REQUIRE_V12_PRODUCTION=true`, `pnpm release:gate` validates the archive's backend-specific proof fields: proof schema version `2`, `compiled-browser-webllm` deploy backend, `unlocked-browser-transformer` Kernel Lab backend, `wasm-small-core` fallback backend, backend role-boundary proof, hosted benchmark proof required and passed, source-bound-required proof mode, Backend Broker selection evidence, backend readiness bound to that hosted benchmark proof, sufficient child artifact counts, and zero blockers. The release summary exposes the binding as `v12ProductionBackendReadinessProofBound`.

The saved real-browser benchmark artifact is validated by:

```bash
HOSTED_BENCHMARK_ARTIFACT_PATH=.artifacts/evals/hosted/browser-runtime-bench-latest.json \
HOSTED_BENCHMARK_EXPECTED_GIT_SHA=<deployment-commit-sha> \
HOSTED_BENCHMARK_REQUIRE_SOURCE_BOUND=true \
pnpm verify:hosted-benchmark-proof
```

It writes `.artifacts/evals/hosted-benchmark-proof-latest.json` and fails if the artifact is not the compiled production backend, does not prove grounded memory with concrete run-level retrieved/included/expected memory ids and retrieval rank, does not pass exact output, falls below the speed floor, uses direct model factual output as proof, or shows CPU fallback.

For release claims, the saved artifact must also be source-bound. The hosted app should set `VITE_GIT_SHA` at build time so the benchmark summary emits `v12ProductionProofSourceGitSha`, and the verifier should compare that value with `HOSTED_BENCHMARK_EXPECTED_GIT_SHA` while `HOSTED_BENCHMARK_REQUIRE_SOURCE_BOUND=true`. The production archive preserves that strict verifier mode as `v12ProductionProofSourceBoundRequired`; the release gate requires it to be true.

## GitHub Production Proof Workflow

After a hosted benchmark is saved, run the manual GitHub Actions workflow:

```text
V12 Production Proof
```

Inputs:

- `deploy_url`: public hosted app URL,
- `hosted_production_benchmark_url`: optional explicit public HTTPS benchmark URL,
- exactly one of these saved-artifact inputs:
  - `hosted_benchmark_artifact_url`: public HTTPS URL to the saved benchmark JSON,
  - `hosted_benchmark_artifact_json`: pasted saved benchmark JSON,
  - `hosted_benchmark_artifact_base64`: base64-encoded saved benchmark JSON.

The artifact URL path is for public hosted artifacts only. The workflow rejects localhost, loopback, link-local, and private-network hosts; use pasted JSON or base64 for local/private proof files.

Generate base64 from a saved local artifact with:

```bash
base64 -i .artifacts/evals/hosted/browser-runtime-bench-latest.json | tr -d '\n'
```

Required repository secrets:

- `BENCHMARK_TELEMETRY_DATABASE_URL`
- `BENCHMARK_TELEMETRY_ADMIN_TOKEN`

The workflow materializes the saved browser artifact, verifies the hosted deploy profile, verifies the benchmark proof, writes the v12 production archive, runs the v12 production release gate, and uploads the proof artifact directory.

## What Still Needs Automation

Manual device testing has been encouraging. The next release-hardening step is to connect the hosted deployment to durable Postgres telemetry and save benchmark results from each target device class automatically.

See [Benchmark Telemetry Plan](57_BENCHMARK_TELEMETRY.md).
