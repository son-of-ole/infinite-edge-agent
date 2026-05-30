# Repository Metadata

Use this file as the source of truth for the public GitHub repository settings and release copy.

## GitHub About

**Repository title**

Infinite Edge Agent

**Description**

Browser-native persistent AI agent runtime with local memory, context reconstruction, compiled WebGPU inference, and a custom WebGPU Kernel Lab.

**Website**

Set this to the public hosted demo URL once the Replit, Hugging Face Space, or production deployment is intended for public traffic.

**Topics**

```text
ai-agent
browser-ai
webgpu
webllm
local-first
persistent-memory
rag
edge-inference
typescript
react
indexeddb
benchmarking
```

## Social Preview

Use `docs/assets/infinite-edge-agent-hero.jpg` as the GitHub social preview image.

The generated hero is intentionally text-free so it remains usable across GitHub, Hugging Face, project pages, and launch posts.

## Short Project Summary

Infinite Edge Agent is a browser-native persistent AI runtime. It separates the agent from the model by making memory, context reconstruction, backend selection, model execution, and readiness proof independent layers. The compiled WebLLM lane is the production candidate; the custom WebGPU runtime remains a Kernel Lab for SSA, KVSwap, TSP, fusion, and backend research.

## Longer Public Abstract

Infinite Edge Agent explores what a persistent browser-native AI agent can look like when memory, context rebuild, backend routing, and production proof are treated as first-class runtime systems. The project supports local browser memory, grounded retrieval canaries, compiled WebGPU model execution through WebLLM, optional hosted benchmark telemetry, and a research-grade custom WebGPU Kernel Lab for low-level inference work.

The repository is not claiming a new base model. The contribution is the runtime architecture, browser deployment path, benchmark discipline, and separation between deploy-ready compiled backends and experimental kernel research.

## Release Framing

- Production lane: `compiled-browser-webllm`
- Research lane: `unlocked-browser-transformer`
- Memory default: browser-local IndexedDB/vector memory
- Hosted proof: real Chrome/Edge browser benchmark artifacts
- Public benchmark direction: opt-in, sanitized, device/GPU-aware telemetry
- Final-state archive: `pnpm eval:v12-suite`
- Hosted runtime proof: `pnpm verify:hosted-benchmark-proof`

## Suggested Hugging Face Space Blurb

Run a local-first browser AI agent with persistent memory, grounded retrieval proof, and WebGPU-backed inference directly in the browser. The demo uses the compiled production lane and reports benchmark/readiness artifacts instead of hiding backend fallbacks.
