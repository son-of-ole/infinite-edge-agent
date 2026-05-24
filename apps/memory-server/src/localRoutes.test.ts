import { describe, expect, it } from "vitest";
import { shouldExposeLocalRoutes } from "./localRoutes";

describe("local memory route exposure", () => {
  it("exposes root sidecar routes by default only on loopback hosts", () => {
    expect(shouldExposeLocalRoutes({ host: "127.0.0.1" })).toBe(true);
    expect(shouldExposeLocalRoutes({ host: "localhost" })).toBe(true);
    expect(shouldExposeLocalRoutes({ host: "0.0.0.0" })).toBe(false);
    expect(shouldExposeLocalRoutes({ host: "::" })).toBe(false);
  });

  it("supports explicit opt-in and opt-out for deployment wrappers", () => {
    expect(shouldExposeLocalRoutes({ host: "0.0.0.0", value: "true" })).toBe(true);
    expect(shouldExposeLocalRoutes({ host: "127.0.0.1", value: "false" })).toBe(false);
  });
});
