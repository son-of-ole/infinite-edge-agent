import { describe, expect, it } from "vitest";
import { readBrowserEmbedConfig, resolveEmbedMemoryProvider } from "./embedConfig";

describe("readBrowserEmbedConfig", () => {
  it("detects SDK embed mode and public scope params", () => {
    expect(readBrowserEmbedConfig("?sdk=browser&sessionId=s1&tenantId=t1&cellId=c1")).toEqual({
      enabled: true,
      compact: true,
      sessionId: "s1",
      tenantId: "t1",
      cellId: "c1",
      deploymentPreset: undefined,
      requestedMemoryMode: undefined,
    });
  });

  it("stays disabled without embed query params", () => {
    expect(readBrowserEmbedConfig("?theme=dark")).toEqual({
      enabled: false,
      compact: false,
      sessionId: undefined,
      tenantId: undefined,
      cellId: undefined,
      deploymentPreset: undefined,
      requestedMemoryMode: undefined,
    });
  });

  it("can keep full chrome for explicit embed debugging", () => {
    expect(readBrowserEmbedConfig("?embed=1&compact=0")).toMatchObject({
      enabled: true,
      compact: false,
    });
  });

  it("reads public deployment hints without treating them as secrets or authority", () => {
    expect(readBrowserEmbedConfig("?sdk=browser&deploymentPreset=remote-http&memoryMode=remote-http")).toMatchObject({
      enabled: true,
      deploymentPreset: "remote-http",
      requestedMemoryMode: "remote-http",
    });
  });

  it("only allows SDK remote memory hints when the app is already configured for a same-origin proxy", () => {
    const embed = readBrowserEmbedConfig("?sdk=browser&deploymentPreset=remote-http&memoryMode=remote-http");

    expect(resolveEmbedMemoryProvider({
      embed,
      configuredProvider: "browser-vector",
      remoteMemoryUrl: "/api/edge-ai",
    })).toBe("remote-http");
    expect(resolveEmbedMemoryProvider({
      embed,
      configuredProvider: "browser-vector",
      remoteMemoryUrl: "https://memory.example.com/api/edge-ai",
    })).toBe("browser-vector");
    expect(resolveEmbedMemoryProvider({
      embed: readBrowserEmbedConfig("?sdk=browser&deploymentPreset=sidecar-disabled"),
      configuredProvider: "remote-http",
      remoteMemoryUrl: "/api/edge-ai",
    })).toBe("browser-vector");
  });
});
