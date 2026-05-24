import { describe, expect, it } from "vitest";
import { redactSensitiveMemoryText } from "./memoryPrivacy";

describe("redactSensitiveMemoryText", () => {
  it("redacts common API tokens before memory embedding", () => {
    const result = redactSensitiveMemoryText("Use OPENAI_API_KEY=sk-abcdefghijklmnopqrstuvwxyz1234567890 for tests.");

    expect(result.text).not.toContain("sk-abcdefghijklmnopqrstuvwxyz1234567890");
    expect(result.text).toContain("[redacted:openai_api_key]");
    expect(result.findings).toEqual([
      expect.objectContaining({ kind: "openai_api_key", replacement: "[redacted:openai_api_key]" }),
    ]);
  });

  it("leaves ordinary memory unchanged", () => {
    const result = redactSensitiveMemoryText("Remember that local memory defaults to IndexedDB.");

    expect(result.text).toBe("Remember that local memory defaults to IndexedDB.");
    expect(result.findings).toEqual([]);
  });
});
