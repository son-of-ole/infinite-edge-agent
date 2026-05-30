# Contributing

Infinite Edge Agent is a browser-native AI runtime project. Contributions should keep the production path, research path, and benchmark claims clearly separated.

## Development Setup

```bash
pnpm install
pnpm typecheck
pnpm test
pnpm build
```

Use Chrome or Edge for browser WebGPU proof. The Node and CI lanes are useful for deterministic checks, but real browser/GPU readiness must be proven in a browser artifact.

## Contribution Rules

- Keep private prompts, user memory, hosted endpoint URLs, tokens, and local model weights out of Git.
- Do not claim production readiness from the custom Kernel Lab unless the backend-specific readiness artifact explicitly supports it.
- Keep `compiled-browser-webllm` as the production candidate unless a new backend clears the same readiness gates.
- Keep MTP and math-changing WebGPU fusion behind explicit lab flags until they beat the target-only production path with quality gates passing.
- Add or update tests when changing runtime behavior, readiness gates, benchmark artifacts, memory retrieval, or deployment checks.

## Pull Request Checklist

- Describe what changed and why.
- Include the exact commands run.
- Include browser artifact paths or hosted benchmark evidence when touching runtime, memory, or benchmark behavior.
- State whether the change affects the compiled production lane, Kernel Lab lane, or both.

