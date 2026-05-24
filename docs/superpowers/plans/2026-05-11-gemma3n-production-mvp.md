# Gemma 3n Production MVP Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Harden Infinite Edge Agent into a runnable local-first production MVP centered on Google Gemma 3n E2B.

**Architecture:** Keep the documented Tier-0 runtime path: memory retrieval, Context Runtime, TSP, SSA, KVSwap, target-only MTP fallback, then inference. Use Gemma 3n E2B through Google AI Edge MediaPipe LLM Inference with a web-converted `.litertlm` asset as the primary browser path; keep local OpenAI-compatible and WebLLM backends as adapter paths.

**Tech Stack:** TypeScript, React/Vite, MediaPipe Tasks GenAI, WebLLM, local OpenAI-compatible chat endpoints, Transformers.js embeddings, IndexedDB, optional LanceDB sidecar, Vitest.

---

### Task 1: Restore Build Baseline

**Files:**
- Modify: `package.json`
- Modify: `apps/web/src/lib/llm/webllmClient.ts`
- Modify: `apps/web/src/workers/embedding.worker.ts`
- Modify: `apps/memory-server/tsconfig.json`
- Test: `pnpm typecheck`
- Test: `pnpm smoke:core`

- [x] Add `@infinite-edge-agent/core` as a root workspace dependency so root scripts can resolve it.
- [x] Fix WebLLM streaming types by using `ChatCompletionRequestStreaming` and only passing `initProgressCallback` when defined.
- [x] Narrow the Transformers.js pipeline call in the embedding worker to avoid the current too-complex union type.
- [x] Align the memory server TypeScript resolver with the rest of the workspace so it typechecks against source exports.
- [x] Run `npx -y pnpm@9.15.0 typecheck` and expect no TypeScript errors.
- [x] Run `npx -y pnpm@9.15.0 smoke:core` and expect chunk/cosine/context output.

### Task 2: Add Gemma 3n E2B Inference Boundary

**Files:**
- Create: `apps/web/src/lib/llm/types.ts`
- Create: `apps/web/src/lib/llm/mediapipeGemmaClient.ts`
- Create: `apps/web/src/lib/llm/openAICompatibleClient.ts`
- Modify: `apps/web/src/lib/llm/webllmClient.ts`
- Modify: `apps/web/src/config.ts`
- Modify: `apps/web/src/config/models.ts`
- Modify: `.env.example`
- Test: `pnpm typecheck`

- [x] Create a shared `ChatClient` interface for streaming chat clients.
- [x] Add a MediaPipe Gemma client that loads a web-converted `.litertlm` Gemma 3n E2B asset.
- [x] Add a local OpenAI-compatible client that streams from `VITE_LOCAL_LLM_BASE_URL` using model `google/gemma-3n-E2B`.
- [x] Keep WebLLM as a browser backend for prebuilt or custom MLC artifacts.
- [x] Make Gemma 3n E2B the default model target and expose the backend choice in config.
- [x] Document MediaPipe asset and local sidecar env vars in `.env.example`.
- [x] Run typecheck.

### Task 3: Route Generation Through Advanced Runtime Coordinator

**Files:**
- Create: `packages/core/src/runtime/advancedRuntimeCoordinator.ts`
- Modify: `packages/core/src/runtime/index.ts`
- Modify: `packages/core/src/types.ts`
- Modify: `apps/web/src/lib/agent/localAgent.ts`
- Modify: `apps/web/src/components/StatusPanel.tsx`
- Modify: `apps/web/src/App.tsx`
- Test: `packages/core/src/runtime/advancedRuntimeCoordinator.test.ts`

- [x] Add a coordinator that builds a Context Runtime plan, calls fallback TSP, fallback SSA, metadata-only KVSwap, and target-only MTP config before inference.
- [x] Return a serializable runtime trace with selected/dropped memory, fallback reasons, estimated token budgets, and model/backend metadata.
- [x] Store trace metadata on assistant messages.
- [x] Show Tier-0 runtime states and last trace summary in the UI.
- [x] Add unit tests for pinned/current-user retention and explicit fallback trace fields.
- [x] Run core tests and typecheck.

### Task 4: Persist Runtime Traces Locally

**Files:**
- Modify: `packages/core/src/types.ts`
- Modify: `apps/web/src/lib/storage/indexedDbMemoryStore.ts`
- Modify: `apps/web/src/lib/storage/sidecarMemoryStore.ts`
- Modify: `apps/memory-server/src/index.ts`
- Modify: `apps/memory-server/src/lancedbStore.ts`
- Modify: `apps/memory-server/src/types.ts`
- Test: `pnpm typecheck`

- [x] Add trace-store methods without breaking `MemoryStore` consumers.
- [x] Persist traces in IndexedDB for browser-only mode.
- [x] Add sidecar endpoints for trace writes and recent trace reads.
- [x] Add memory-server validation for trace payloads.
- [x] Run typecheck.

### Task 5: Verify Production MVP Surface

**Files:**
- Modify: `README.md`
- Modify: `docs/02_BUILD_AND_RUN.md`
- Modify: `docs/49_ENGINEERING_HANDOFF_CHECKLIST_MODEL_NATIVE_GAC.md`

- [x] Update docs with the Gemma 3n E2B MediaPipe asset path and sidecar path.
- [x] Explain that WebLLM browser-native Gemma 3n requires a compiled MLC model plus `model_lib`.
- [x] Run `npx -y pnpm@9.15.0 test`, `typecheck`, `smoke:core`, and `build`.
- [x] Start the web app and verify the first screen renders without overlap or blank panels.

### Remaining Production Tasks

- [x] Add real GAC raw-memory, identity-pin, representative, lineage, retrieval-audit, and context-pack trace tables.
- [x] Add model action shadow mode and policy gate.
- [x] Add sleep/wake jobs and rollback.
- [x] Add eval artifacts for memory recall, pinned recall, context rebuild, SSA routing, KVSwap priority, sidecar behavior, model actions, and sleep/wake continuity.
- [x] Add code splitting for large MediaPipe/WebLLM/ONNX chunks.
