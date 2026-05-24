import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";

describe("sdk smoke script entrypoint", () => {
  it("imports the built package entrypoint instead of SDK source", () => {
    const source = readFileSync(new URL("./smoke-test-sdk.ts", import.meta.url), "utf8");

    expect(source).toContain("../packages/sdk/dist/index.js");
    expect(source).not.toContain("../packages/sdk/src/embed");
  });

  it("has a plain Node package-entrypoint smoke in the root smoke command", () => {
    const smokeSource = readFileSync(new URL("./smoke-test-sdk-package.mjs", import.meta.url), "utf8");
    const packageJson = readFileSync(new URL("../package.json", import.meta.url), "utf8");

    expect(smokeSource).toContain("@infinite-edge-agent/browser-sdk");
    expect(smokeSource).toContain("npm");
    expect(smokeSource).toContain("pack");
    expect(packageJson).toContain("node scripts/smoke-test-sdk-package.mjs");
  });
});
