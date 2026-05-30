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

Use `docs/assets/infinite-edge-agent-social.png` as the GitHub social preview image.

The generated social preview is intentionally text-free so it remains usable across GitHub, Hugging Face, project pages, and launch posts.

## README Hero Photo

Use `docs/assets/infinite-edge-agent-readme-photo.png` as the top README image. It is a generated, text-free project photo that emphasizes the core runtime idea: browser-resident intelligence, local memory routing, and on-device acceleration.

**README photo generation prompt**

```text
A polished open-source project hero image for "Infinite Edge Agent": a browser-native AI runtime visualized as a luminous web browser window on a laptop, local memory nodes and context-routing lines flowing into a compact on-device model core, subtle WebGPU-style compute grid, realistic modern workstation scene, clean technical aesthetic, high contrast, no text, no logos, suitable for a GitHub README social preview, 16:9 landscape.
```

**Social preview generation prompt**

```text
Create a polished text-free hero image for an open source project called Infinite Edge Agent. Visual concept: a browser-native AI runtime running across edge devices, with a laptop browser window, phone, and small workstation implied by silhouettes, connected by luminous memory graph nodes and subtle WebGPU-like compute grids. Style: realistic cinematic product/technology photography blended with clean technical visualization, high-end open source systems project, modern but not dark or cluttered. No readable text, no logos, no watermarks. Wide landscape composition suitable for a GitHub README social preview, generous safe margins, crisp details, professional lighting.
```

## Short Project Summary

Infinite Edge Agent is a browser-native persistent AI runtime. It separates the agent from the model by making memory, context reconstruction, backend selection, model execution, and readiness proof independent layers. The compiled WebLLM lane is the production candidate, the custom WebGPU runtime remains a Kernel Lab for SSA, KVSwap, TSP, fusion, and backend research, and the small WASM lane is bounded fallback only.

## Longer Public Abstract

Infinite Edge Agent explores what a persistent browser-native AI agent can look like when memory, context rebuild, backend routing, and production proof are treated as first-class runtime systems. The project supports local browser memory, grounded retrieval canaries, compiled WebGPU model execution through WebLLM, optional hosted benchmark telemetry, and a research-grade custom WebGPU Kernel Lab for low-level inference work.

The repository is not claiming a new base model. The contribution is the runtime architecture, browser deployment path, benchmark discipline, and separation between deploy-ready compiled backends and experimental kernel research.

## Release Framing

- Production lane: `compiled-browser-webllm`
- Research lane: `unlocked-browser-transformer`
- Fallback lane: `wasm-small-core`
- Memory default: browser-local IndexedDB/vector memory
- Hosted proof: real Chrome/Edge browser benchmark artifacts
- Public benchmark direction: opt-in, sanitized, device/GPU-aware telemetry
- Final-state archive: `pnpm eval:v12-suite`
- Production archive: `pnpm eval:v12-production`
- Final ship/no-ship status: `pnpm eval:v12-final-state`
- Hosted runtime proof: `pnpm verify:hosted-benchmark-proof`
- Source publication status: `pnpm eval:repository-publication`
- Exact-history publication handoff: `pnpm handoff:repository-publication`

## Suggested Hugging Face Space Blurb

Run a local-first browser AI agent with persistent memory, grounded retrieval proof, and WebGPU-backed inference directly in the browser. The demo uses the compiled production lane and reports benchmark/readiness artifacts instead of hiding backend fallbacks.
