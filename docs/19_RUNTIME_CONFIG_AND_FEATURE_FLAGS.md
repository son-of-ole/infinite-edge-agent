# 19 — Runtime Config and Feature Flags

## Goal

All advanced components must be visible in config from day one. A feature may be in fallback mode, but it must not be missing.

## Config file

See `configs/runtime.firstclass.example.json`.

## Required config sections

```json
{
  "memory": {},
  "contextRuntime": {},
  "ssa": {},
  "tsp": {},
  "mtp": {},
  "kvswap": {},
  "inferenceBackend": {},
  "telemetry": {},
  "evals": {}
}
```

## Feature state enum

```ts
type RuntimeFeatureState =
  | "required"
  | "enabled"
  | "fallback"
  | "disabled_for_test"
  | "unavailable";
```

Production builds should reject `disabled_for_test` for Tier-0 features.

## Degradation reporting

The UI must show:

- which backend is active,
- which Tier-0 features are native vs fallback,
- why fallback is active,
- expected user impact.

Example:

```json
{
  "feature": "kvswap",
  "state": "fallback",
  "mode": "metadata_only",
  "reason": "Selected backend does not expose KV tensor handles.",
  "impact": "Long active contexts use prompt packing and memory retrieval but cannot page true KV tensors yet."
}
```
