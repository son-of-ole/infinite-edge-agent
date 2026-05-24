import { describe, expect, it } from "vitest";
import { buildInfiniteEdgeAgentUrl, mountInfiniteEdgeAgent } from "./embed";

describe("browser SDK embed", () => {
  it("builds an embed URL without leaking secrets into query parameters", () => {
    const url = buildInfiniteEdgeAgentUrl({
      agentUrl: "https://agent.example.com/app?theme=dark",
      sessionId: "session_123",
      tenantId: "tenant_a",
      cellId: "cell_main",
      deployment: {
        preset: "browser-only",
        runtimeProfile: "unlocked-browser",
        modelProfile: "qwen3-0.6b-sharded",
      },
    });

    expect(url.toString()).toBe(
      "https://agent.example.com/app?theme=dark&embed=1&sdk=browser&sdkVersion=0.1.0&sdkMode=inline&compact=1&sessionId=session_123&tenantId=tenant_a&cellId=cell_main&deploymentPreset=browser-only&memoryMode=browser-vector&sidecar=disabled&runtimeProfile=unlocked-browser&modelProfile=qwen3-0.6b-sharded",
    );
  });

  it("constructs remote HTTP deployment URLs with public routing only", () => {
    const url = buildInfiniteEdgeAgentUrl({
      agentUrl: "https://agent.example.com/app",
      mode: "launcher",
      sessionId: "session_remote",
      tenantId: "tenant_public",
      cellId: "cell_support",
      deployment: {
        preset: "remote-http",
        runtimeProfile: "unlocked-browser",
      },
    });

    expect(Object.fromEntries(url.searchParams)).toMatchObject({
      sdkMode: "launcher",
      sessionId: "session_remote",
      tenantId: "tenant_public",
      cellId: "cell_support",
      deploymentPreset: "remote-http",
      memoryMode: "remote-http",
      sidecar: "disabled",
      runtimeProfile: "unlocked-browser",
    });
    expect([...url.searchParams.keys()].some((key) => /token|secret|password|api[_-]?key|bearer|credential/i.test(key))).toBe(false);
  });

  it("can express sidecar-disabled browser vector mode without a private endpoint", () => {
    const url = buildInfiniteEdgeAgentUrl({
      agentUrl: "https://agent.example.com/app",
      deployment: "sidecar-disabled",
    });

    expect(url.searchParams.get("deploymentPreset")).toBe("sidecar-disabled");
    expect(url.searchParams.get("memoryMode")).toBe("browser-vector");
    expect(url.searchParams.get("sidecar")).toBe("disabled");
  });

  it("allows hosts to opt out of compact embed chrome", () => {
    const url = buildInfiniteEdgeAgentUrl({
      agentUrl: "https://agent.example.com/app",
      compact: false,
    });

    expect(url.searchParams.get("compact")).toBeNull();
    expect(url.searchParams.get("embed")).toBe("1");
  });

  it("rejects secret-shaped query params in the hosted agent URL", () => {
    expect(() =>
      buildInfiniteEdgeAgentUrl({
        agentUrl: "https://agent.example.com/?memoryToken=abc",
      }),
    ).toThrow(/Do not place credentials in the Infinite Edge Agent embed URL/);
  });

  it("rejects common URL-carried credential keys and signed URL params", () => {
    for (const agentUrl of [
      "https://agent.example.com/?jwt=abc",
      "https://agent.example.com/?auth=abc",
      "https://agent.example.com/?code=oauth-code",
      "https://agent.example.com/?X-Amz-Signature=abc",
      "https://agent.example.com/?signedUrl=https%3A%2F%2Fassets.example.com%2Fmodel%3FX-Amz-Signature%3Dabc",
    ]) {
      expect(() => buildInfiniteEdgeAgentUrl({ agentUrl })).toThrow(/Do not place credentials in the Infinite Edge Agent embed URL/);
    }
  });

  it("allows public query keys and nested public URLs containing code substrings", () => {
    const url = buildInfiniteEdgeAgentUrl({
      agentUrl:
        "https://agent.example.com/app?themeCode=dark&zipCode=84043&campaign_code=spring&next=https%3A%2F%2Fpublic.example.com%2Fsignup%3Fzipcode%3D84043",
    });

    expect(url.searchParams.get("themeCode")).toBe("dark");
    expect(url.searchParams.get("zipCode")).toBe("84043");
    expect(url.searchParams.get("campaign_code")).toBe("spring");
    expect(url.searchParams.get("next")).toBe("https://public.example.com/signup?zipcode=84043");
  });

  it("rejects URL userinfo credentials", () => {
    expect(() =>
      buildInfiniteEdgeAgentUrl({
        agentUrl: "https://user:pass@agent.example.com/app",
      }),
    ).toThrow(/URL userinfo is not allowed/);
  });

  it("rejects raw JWT-looking values and URL fragments", () => {
    const jwt = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjMifQ.signature";

    expect(() =>
      buildInfiniteEdgeAgentUrl({
        agentUrl: "https://agent.example.com/",
        sessionId: jwt,
      }),
    ).toThrow(/Do not place credentials in the Infinite Edge Agent embed URL/);

    expect(() =>
      buildInfiniteEdgeAgentUrl({
        agentUrl: "https://agent.example.com/#access_token=abc",
      }),
    ).toThrow(/URL fragments are not allowed/);
  });

  it("rejects secret-shaped public option values before they reach the iframe URL", () => {
    expect(() =>
      buildInfiniteEdgeAgentUrl({
        agentUrl: "https://agent.example.com/",
        sessionId: "Bearer abc123",
      }),
    ).toThrow(/Do not place credentials in the Infinite Edge Agent embed URL/);
  });

  it("allows public identifiers that contain sk hyphen substrings", () => {
    const url = buildInfiniteEdgeAgentUrl({
      agentUrl: "https://agent.example.com/",
      sessionId: "task-123",
      tenantId: "desk-1",
      cellId: "mask-a",
    });

    expect(url.searchParams.get("sessionId")).toBe("task-123");
    expect(url.searchParams.get("tenantId")).toBe("desk-1");
    expect(url.searchParams.get("cellId")).toBe("mask-a");
  });

  it("still rejects OpenAI-style sk keys with enough key material", () => {
    expect(() =>
      buildInfiniteEdgeAgentUrl({
        agentUrl: "https://agent.example.com/",
        sessionId: `sk-${"a".repeat(32)}`,
      }),
    ).toThrow(/Do not place credentials in the Infinite Edge Agent embed URL/);
  });

  it("requires HTTPS or localhost by default", () => {
    expect(() =>
      buildInfiniteEdgeAgentUrl({
        agentUrl: "http://example.com",
      }),
    ).toThrow(/must be HTTPS or localhost/);

    expect(
      buildInfiniteEdgeAgentUrl({
        agentUrl: "http://127.0.0.1:5173",
      }).origin,
    ).toBe("http://127.0.0.1:5173");
  });

  it("mounts and destroys an iframe in a supplied container", () => {
    const documentRef = new FakeDocument();
    const container = documentRef.createElement("div");

    const handle = mountInfiniteEdgeAgent({
      agentUrl: "https://agent.example.com",
      container: container as unknown as HTMLElement,
      document: documentRef as unknown as Document,
      height: "640px",
    });

    expect(container.children).toHaveLength(1);
    expect(handle.iframe.tagName).toBe("IFRAME");
    expect(handle.iframe.getAttribute("allow")).toBe("webgpu; cross-origin-isolated");
    expect(handle.iframe.style.height).toBe("640px");
    expect(handle.url.searchParams.get("compact")).toBe("1");

    handle.destroy();
    expect(container.children).toHaveLength(0);
  });

  it("mounts launcher mode in a removable wrapper and toggles iframe visibility", () => {
    const documentRef = new FakeDocument();
    const container = documentRef.createElement("div");

    const handle = mountInfiniteEdgeAgent({
      agentUrl: "https://agent.example.com",
      container: container as unknown as HTMLElement,
      document: documentRef as unknown as Document,
      mode: "launcher",
    });

    expect(container.children).toHaveLength(1);
    expect(container.children[0].className).toBe("infinite-edge-agent-launcher");
    expect(handle.iframe.hidden).toBe(true);
    expect(handle.url.searchParams.get("sdkMode")).toBe("launcher");

    handle.setOpen(true);
    expect(handle.iframe.hidden).toBe(false);
    handle.setOpen(false);
    expect(handle.iframe.hidden).toBe(true);

    handle.destroy();
    expect(container.children).toHaveLength(0);
  });
});

class FakeElement {
  readonly children: FakeElement[] = [];
  readonly style: Record<string, string> = {};
  readonly attributes = new Map<string, string>();
  parentElement: FakeElement | null = null;
  textContent = "";
  hidden = false;
  className = "";
  title = "";
  src = "";
  loading = "";
  referrerPolicy = "";

  constructor(readonly tagName: string) {}

  appendChild(child: FakeElement): FakeElement {
    child.parentElement = this;
    this.children.push(child);
    return child;
  }

  remove(): void {
    if (!this.parentElement) return;
    const index = this.parentElement.children.indexOf(this);
    if (index >= 0) this.parentElement.children.splice(index, 1);
    this.parentElement = null;
  }

  setAttribute(name: string, value: string): void {
    this.attributes.set(name, value);
  }

  getAttribute(name: string): string | null {
    return this.attributes.get(name) ?? null;
  }

  addEventListener(): void {
    // Fake DOM hook for SDK tests.
  }

  focus(): void {
    // Fake DOM hook for SDK tests.
  }
}

class FakeDocument {
  readonly body = new FakeElement("BODY");

  createElement(tagName: string): FakeElement {
    return new FakeElement(tagName.toUpperCase());
  }

  querySelector(): FakeElement | null {
    return null;
  }
}
