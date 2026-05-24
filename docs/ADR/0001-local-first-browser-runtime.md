# ADR 0001: Local-first browser runtime

## Status

Accepted.

## Context

The project aims to provide private, low-latency AI assistance without depending on cloud inference.

## Decision

Use a browser-first runtime with `unlocked-browser-transformer` for local Qwen inference and Transformers.js/ONNX Runtime Web for embeddings. Heavy computation must run in workers.

## Consequences

Positive:

- No server required for MVP.
- Privacy-preserving default.
- Easy distribution through a URL or local static app.

Negative:

- Browser storage and GPU memory limits apply.
- Model choice is constrained by the unlocked manifest and shard contract.
- Advanced attention/KV-cache features require backend-owned tensor handles and browser-validated generation quality.
