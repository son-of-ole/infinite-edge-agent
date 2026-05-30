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
});
