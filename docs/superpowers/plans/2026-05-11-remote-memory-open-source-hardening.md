# Remote Memory Open-Source Hardening Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Finish the production path so Gemma 3n E2B can use a configurable remote memory database, with first-party internal deployments reserved for testing rather than open-source defaults.

**Architecture:** Keep Gemma 3n E2B as the primary model target through MediaPipe `.litertlm` browser inference or local OpenAI-compatible fallback. Use IndexedDB as the zero-config open-source default, and offer a generic `remote-http` memory provider for any compatible HTTPS database API.

**Tech Stack:** TypeScript, React/Vite, MediaPipe Tasks GenAI, HTTP memory API, IndexedDB fallback, optional LanceDB sidecar, Vitest.

---

### Task 1: Remote Memory Provider Contract

**Files:**
- Create: `apps/web/src/lib/storage/remoteMemoryStore.ts`
- Modify: `apps/web/src/lib/storage/hybridMemoryClient.ts`
- Modify: `apps/memory-server/src/index.ts`
- Modify: `apps/web/src/config.ts`
- Modify: `.env.example`
- Test: `apps/web/src/lib/storage/remoteMemoryStore.test.ts`

- [x] Add memory provider config: `remote-http`, `sidecar`, `indexeddb`.
- [x] Implement `RemoteMemoryStore` against a caller-provided base URL.
- [x] Use endpoints: `GET /health`, `POST /memory/upsert`, `POST /memory/search`, `DELETE /memory`, `POST /runtime/traces`, `GET /runtime/traces`.
- [x] Add optional bearer token and tenant/cell headers.
- [x] Expose the same API namespace from the memory server at `/api/edge-ai`.
- [x] Add tests with mocked fetch for auth headers, payload shape, search parsing, trace persistence, and health failure.

### Task 2: Production Readiness Gate

**Files:**
- Create: `apps/web/src/lib/runtime/productionReadiness.ts`
- Modify: `apps/web/src/App.tsx`
- Modify: `apps/web/src/components/StatusPanel.tsx`
- Modify: `apps/web/src/styles.css`
- Test: `apps/web/src/lib/runtime/productionReadiness.test.ts`

- [x] Validate Gemma production config before initialization.
- [x] Validate remote memory URL when `remote-http` is selected.
- [x] Validate MediaPipe `.litertlm` asset path when `mediapipe-gemma` is selected.
- [x] Surface readiness blockers in the UI.
- [x] Add tests for ready and blocked states.

### Task 3: Runtime Trace Names and Memory Mode

**Files:**
- Modify: `packages/core/src/runtime/advancedRuntimeCoordinator.ts`
- Modify: `packages/core/src/runtime/advancedRuntimeCoordinator.test.ts`
- Modify: `packages/core/src/types.ts`
- Modify: `apps/web/src/lib/agent/localAgent.ts`

- [x] Add `remote-http` as a first-class memory mode.
- [x] Change runtime feature reporting from LanceDB-specific language to production memory provider language.
- [x] Include remote provider metadata in runtime traces.
- [x] Keep compatibility with `lancedb-sidecar` and `indexeddb` modes.

### Task 4: Docs And Verification

**Files:**
- Modify: `README.md`
- Modify: `docs/02_BUILD_AND_RUN.md`
- Create: `docs/51_REMOTE_MEMORY_API_CONTRACT.md`
- Modify: `docs/superpowers/plans/2026-05-11-remote-memory-open-source-hardening.md`

- [x] Document remote endpoint contract and deployment expectations.
- [x] Document production env values.
- [x] Smoke-test the local remote API namespace with auth, memory search, and runtime traces.
- [x] Run `npx -y pnpm@9.15.0 typecheck`, `test`, `smoke:core`, and `build`.
- [x] Run browser smoke for the uninitialized app and verify no layout regressions.
