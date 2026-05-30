# V12 Readiness Suite Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add one auditable suite command that writes hosted profile, backend matrix, shared runtime, and v12 final-state readiness artifacts together.

**Architecture:** Keep the existing verifier modules authoritative. Add a thin orchestration script that evaluates each verifier once, writes every underlying artifact with one timestamp, then writes a suite manifest containing artifact paths and final pass/fail state.

**Tech Stack:** TypeScript, Vitest, Node `fs/promises`, existing pnpm script/eval pattern.

---

### Task 1: Suite Artifact Contract

**Files:**
- Create: `scripts/v12ReadinessSuite.test.ts`
- Create: `scripts/v12ReadinessSuite.ts`

- [ ] **Step 1: Write the failing test**

Create tests that require `runV12ReadinessSuite()` to return a passing suite with four child artifacts, consistent deploy/kernel backend ids, and latest/timestamped JSON paths.

- [ ] **Step 2: Run the test to verify it fails**

Run:

```bash
/Users/olson/.local/bin/pnpm exec vitest run scripts/v12ReadinessSuite.test.ts
```

Expected: fail because `scripts/v12ReadinessSuite.ts` does not exist.

- [ ] **Step 3: Implement the suite wrapper**

Create `scripts/v12ReadinessSuite.ts` with:

- `evaluateV12ReadinessSuite()`
- `buildV12ReadinessSuiteArtifact()`
- `writeV12ReadinessSuiteArtifact()`
- `runV12ReadinessSuite()`
- CLI entrypoint that exits non-zero if the suite fails.

- [ ] **Step 4: Verify green**

Run:

```bash
/Users/olson/.local/bin/pnpm exec vitest run scripts/v12ReadinessSuite.test.ts
```

Expected: pass.

### Task 2: Release Summary And Gate Wiring

**Files:**
- Modify: `scripts/releaseGateArtifactSummary.ts`
- Modify: `scripts/releaseGateArtifactSummary.test.ts`
- Modify: `scripts/release-gate.ts`
- Modify: `package.json`

- [ ] **Step 1: Write failing tests for summary fields**

Add a test requiring suite summary fields to remain visible in release summaries.

- [ ] **Step 2: Run the focused test**

Run:

```bash
/Users/olson/.local/bin/pnpm exec vitest run scripts/releaseGateArtifactSummary.test.ts
```

Expected: fail until summary fields are added.

- [ ] **Step 3: Add fields and release gate flag**

Add `pnpm eval:v12-suite`, summarize `v12Suite*` fields, and support `RELEASE_REQUIRE_V12_SUITE=true` in `release-gate.ts`.

- [ ] **Step 4: Verify focused tests**

Run:

```bash
/Users/olson/.local/bin/pnpm exec vitest run scripts/v12ReadinessSuite.test.ts scripts/releaseGateArtifactSummary.test.ts
```

Expected: pass.

### Task 3: Docs And Verification

**Files:**
- Modify: `README.md`
- Modify: `.env.example`
- Modify: `docs/54_OPEN_SOURCE_RELEASE_CHECKLIST.md`
- Modify: `docs/58_REPOSITORY_METADATA.md`

- [ ] **Step 1: Document the command**

Add `pnpm eval:v12-suite` to public commands and release checklist language.

- [ ] **Step 2: Run suite CLI with passing hosted env**

Run:

```bash
EVAL_ARTIFACT_DIR="$(mktemp -d)" \
VITE_LLM_BACKEND=compiled-browser-webllm \
VITE_DEFAULT_MODEL=Qwen3-0.6B-q4f16_1-MLC \
VITE_COMPILED_WEBLLM_ENABLED=true \
VITE_REQUIRE_UNLOCKED_RUNTIME=false \
VITE_MTP_ENABLED=false \
VITE_BENCHMARK_TELEMETRY_ENABLED=true \
VITE_BENCHMARK_TELEMETRY_URL=/api/benchmark-runs \
BENCHMARK_TELEMETRY_ENABLED=true \
BENCHMARK_TELEMETRY_STORAGE=postgres \
BENCHMARK_TELEMETRY_DATABASE_URL=postgres://example.test/infinite_edge_agent \
BENCHMARK_TELEMETRY_ADMIN_TOKEN=admin-token \
BENCHMARK_TELEMETRY_RATE_LIMIT_MAX=60 \
BENCHMARK_TELEMETRY_RATE_LIMIT_WINDOW_MS=600000 \
HOSTED_PRODUCTION_BENCHMARK_URL='https://agent.example.com/__bench/browser-runtime?backend=compiled-browser-webllm&modelId=Qwen3-0.6B-q4f16_1-MLC&memoryGrounding=montana_capital&expectedExact=Helena&submitTelemetry=true&qwenThinkingMode=disabled' \
/Users/olson/.local/bin/pnpm eval:v12-suite
```

Expected: pass and write suite plus child artifacts.

- [ ] **Step 3: Run final checks**

Run:

```bash
/Users/olson/.local/bin/pnpm exec vitest run scripts/v12ReadinessSuite.test.ts scripts/releaseGateArtifactSummary.test.ts
/Users/olson/.local/bin/pnpm typecheck
/Users/olson/.local/bin/pnpm build
git diff --check
```

Expected: all pass.

