import { readFileSync } from "node:fs";
import { join } from "node:path";
import { describe, expect, it } from "vitest";

const ciWorkflowPath = join(process.cwd(), ".github", "workflows", "ci.yml");

describe("CI v12 readiness gates", () => {
  it("runs the v12 invariant checks with the compiled backend profile", () => {
    const workflow = readFileSync(ciWorkflowPath, "utf8");

    expect(workflow).toContain("name: V12 readiness invariants");
    expect(workflow).toContain("VITE_LLM_BACKEND: compiled-browser-webllm");
    expect(workflow).toContain('VITE_COMPILED_WEBLLM_ENABLED: "true"');
    expect(workflow).toContain('VITE_REQUIRE_UNLOCKED_RUNTIME: "false"');
    expect(workflow).toContain('VITE_MTP_ENABLED: "false"');
    expect(workflow).toContain("BENCHMARK_TELEMETRY_STORAGE: postgres");
    expect(workflow).toContain("HOSTED_PRODUCTION_BENCHMARK_URL:");
    expect(workflow).toContain("pnpm verify:hosted-profile");
    expect(workflow).toContain("pnpm eval:backend-readiness");
    expect(workflow).toContain("pnpm eval:shared-runtime");
    expect(workflow).toContain("pnpm eval:v12-readiness");
  });
});
