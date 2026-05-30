import { readFileSync } from "node:fs";
import { join } from "node:path";
import { describe, expect, it } from "vitest";

const releaseGateSource = join(process.cwd(), "scripts", "release-gate.ts");

describe("release gate repository readiness wiring", () => {
  it("runs repository readiness and includes its latest artifact", () => {
    const source = readFileSync(releaseGateSource, "utf8");

    expect(source).toContain('await runGate("repository readiness", ["run", "verify:repository"])');
    expect(source).toContain('["repository-readiness", join(childArtifactRoot, "repository-readiness-latest.json")]');
  });

  it("can require repository publication status as explicit release evidence", () => {
    const source = readFileSync(releaseGateSource, "utf8");

    expect(source).toContain('RELEASE_REQUIRE_REPOSITORY_PUBLICATION === "true"');
    expect(source).toContain('await runGate("repository publication status", ["run", "eval:repository-publication"])');
    expect(source).toContain('["repository-publication-status", join(childArtifactRoot, "repository-publication-status-latest.json")]');
  });

  it("can require the v12 final-state status as explicit release evidence", () => {
    const source = readFileSync(releaseGateSource, "utf8");

    expect(source).toContain('RELEASE_REQUIRE_V12_FINAL_STATE === "true"');
    expect(source).toContain('await runGate("v12 final state status", ["run", "eval:v12-final-state"])');
    expect(source).toContain('["v12-final-state-status", join(childArtifactRoot, "v12-final-state-status-latest.json")]');
  });
});
