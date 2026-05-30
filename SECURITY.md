# Security Policy

## Supported Branch

Security fixes should target `main` unless a maintained release branch is introduced.

## Reporting a Vulnerability

Open a private GitHub security advisory if available. If private advisories are not enabled, contact the repository owner directly before publishing exploit details.

Please include:

- affected commit or release,
- reproduction steps,
- browser, OS, and device details for WebGPU issues,
- whether private memory, telemetry, model assets, or hosted endpoints are involved,
- expected impact and any known workaround.

## Data Handling Expectations

Infinite Edge Agent is designed for local-first browser memory by default, but deployments can opt into hosted telemetry or remote memory. Treat the following as sensitive:

- user prompts and responses,
- memory exports and retrieved documents,
- benchmark artifacts containing raw prompt or response text,
- remote memory bearer tokens and hosted endpoint URLs,
- model conversion artifacts that carry third-party license obligations.

Hosted telemetry must be opt-in, sanitized in the browser and server, rate-limited, and protected for dashboard/export access.

## Model And Backend Safety

Model weights, third-party compiled artifacts, and browser model caches may carry separate licenses and privacy implications. Verify redistribution rights before publishing packaged model assets.

The custom WebGPU Kernel Lab is a research backend. Production deployments should use the backend-specific readiness gates instead of relying on broad WebGPU availability claims.

