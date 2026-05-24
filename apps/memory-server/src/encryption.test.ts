import { describe, expect, it } from "vitest";
import { createAesGcmStringCodec } from "./encryption";

describe("createAesGcmStringCodec", () => {
  it("round-trips encrypted strings without leaving plaintext in the stored value", () => {
    const codec = createAesGcmStringCodec("test-secret");
    const encrypted = codec.encodeString("private LanceDB memory");

    expect(encrypted).toMatch(/^enc:v1:/);
    expect(encrypted).not.toContain("private LanceDB memory");
    expect(codec.decodeString(encrypted)).toBe("private LanceDB memory");
  });

  it("keeps legacy plaintext readable for migration compatibility", () => {
    const codec = createAesGcmStringCodec("test-secret");

    expect(codec.decodeString("legacy plaintext")).toBe("legacy plaintext");
  });
});
