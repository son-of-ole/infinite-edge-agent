import { buildInfiniteEdgeAgentUrl, mountInfiniteEdgeAgent } from "../packages/sdk/dist/index.js";
import { mkdir, writeFile } from "node:fs/promises";
import { join } from "node:path";

interface GateResult {
  name: string;
  passed: boolean;
  actual: number | string | boolean;
  expected: number | string | boolean;
}

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
    // Minimal DOM hook for SDK smoke testing.
  }

  focus(): void {
    // Minimal DOM hook for SDK smoke testing.
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

const createdAt = new Date().toISOString();
const timestamp = createdAt.replace(/[:.]/g, "-");
const artifactRoot = process.env.EVAL_ARTIFACT_DIR ?? ".artifacts/evals";
const suiteDir = join(artifactRoot, "sdk-smoke", timestamp);

const documentRef = new FakeDocument();
const container = documentRef.createElement("div");
const url = buildInfiniteEdgeAgentUrl({
  agentUrl: "https://agent.example.com/app?theme=dark",
  sessionId: "session_smoke",
  tenantId: "tenant_smoke",
  cellId: "cell_smoke",
  mode: "launcher",
  deployment: {
    preset: "remote-http",
    runtimeProfile: "unlocked-browser",
    modelProfile: "fixture",
  },
});

let rejectedSecretUrl = false;
try {
  buildInfiniteEdgeAgentUrl({ agentUrl: "https://agent.example.com/?memoryToken=abc" });
} catch {
  rejectedSecretUrl = true;
}

let rejectedCredentialUrl = false;
try {
  buildInfiniteEdgeAgentUrl({ agentUrl: "https://agent.example.com/?jwt=abc" });
} catch {
  rejectedCredentialUrl = true;
}

let rejectedUrlFragment = false;
try {
  buildInfiniteEdgeAgentUrl({ agentUrl: "https://agent.example.com/#access_token=abc" });
} catch {
  rejectedUrlFragment = true;
}

let rejectedSecretValue = false;
try {
  buildInfiniteEdgeAgentUrl({ agentUrl: "https://agent.example.com/", sessionId: "Bearer abc123" });
} catch {
  rejectedSecretValue = true;
}

const handle = mountInfiniteEdgeAgent({
  agentUrl: "http://127.0.0.1:5173",
  container: container as unknown as HTMLElement,
  document: documentRef as unknown as Document,
  mode: "launcher",
  height: "640px",
  deployment: "sidecar-disabled",
});

const mounted = container.children.length === 1 && container.children[0].className === "infinite-edge-agent-launcher" && handle.iframe.tagName === "IFRAME";
const iframeUrl = new URL(handle.iframe.src);
const iframeHasEmbed = iframeUrl.searchParams.get("embed") === "1";
const iframeHasSdk = iframeUrl.searchParams.get("sdk") === "browser";
const iframeMode = iframeUrl.searchParams.get("sdkMode");
const iframeDeploymentPreset = iframeUrl.searchParams.get("deploymentPreset");
const iframeMemoryMode = iframeUrl.searchParams.get("memoryMode");
const iframeSidecar = iframeUrl.searchParams.get("sidecar");
const allow = handle.iframe.getAttribute("allow") ?? "";
const hasNoSecretParams = [...iframeUrl.searchParams, ...url.searchParams].every(
  ([key, value]) =>
    !/(token|secret|password|passwd|pwd|api[_-]?key|apikey|bearer|credential|jwt|auth|authorization|code|signature|signed|x-amz-|x-goog-)/i.test(key)
    && !/(bearer\s+|token=|secret=|password=|passwd=|pwd=|api[_-]?key=|apikey=|credential=|jwt=|auth=|authorization=|code=|signature=|x-amz-|x-goog-)/i.test(value)
    && !/(^|[^A-Za-z0-9_-])sk-(?:proj-|svcacct-)?[A-Za-z0-9_-]{20,}($|[^A-Za-z0-9_-])/i.test(value)
    && !/^[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]+$/.test(value),
);

handle.destroy();
const destroyed = container.children.length === 0;

const gates: GateResult[] = [
  gate("embed url flag", url.searchParams.get("embed"), "1", url.searchParams.get("embed") === "1"),
  gate("sdk url flag", url.searchParams.get("sdk"), "browser", url.searchParams.get("sdk") === "browser"),
  gate("requested deployment preset routed", url.searchParams.get("deploymentPreset") ?? "", "remote-http", url.searchParams.get("deploymentPreset") === "remote-http"),
  gate("requested memory mode routed", url.searchParams.get("memoryMode") ?? "", "remote-http", url.searchParams.get("memoryMode") === "remote-http"),
  gate("requested sidecar disabled", url.searchParams.get("sidecar") ?? "", "disabled", url.searchParams.get("sidecar") === "disabled"),
  gate("requested launcher mode routed", url.searchParams.get("sdkMode") ?? "", "launcher", url.searchParams.get("sdkMode") === "launcher"),
  gate("secret-shaped params rejected", rejectedSecretUrl, true, rejectedSecretUrl),
  gate("credential-shaped params rejected", rejectedCredentialUrl, true, rejectedCredentialUrl),
  gate("URL fragments rejected", rejectedUrlFragment, true, rejectedUrlFragment),
  gate("secret-shaped values rejected", rejectedSecretValue, true, rejectedSecretValue),
  gate("iframe mounted", mounted, true, mounted),
  gate("iframe embed flag", iframeHasEmbed, true, iframeHasEmbed),
  gate("iframe sdk flag", iframeHasSdk, true, iframeHasSdk),
  gate("iframe launcher mode", iframeMode ?? "", "launcher", iframeMode === "launcher"),
  gate("iframe sidecar-disabled preset", iframeDeploymentPreset ?? "", "sidecar-disabled", iframeDeploymentPreset === "sidecar-disabled"),
  gate("iframe browser vector memory mode", iframeMemoryMode ?? "", "browser-vector", iframeMemoryMode === "browser-vector"),
  gate("iframe sidecar disabled", iframeSidecar ?? "", "disabled", iframeSidecar === "disabled"),
  gate("iframe allow policy", allow, "webgpu; cross-origin-isolated", allow === "webgpu; cross-origin-isolated"),
  gate("no secret query params", hasNoSecretParams, true, hasNoSecretParams),
  gate("destroy removes iframe", destroyed, true, destroyed),
];

const artifact = {
  name: "sdk-smoke",
  createdAt,
  passed: gates.every((item) => item.passed),
  metrics: {
    mountedIframeCount: mounted ? 1 : 0,
    iframeHeight: handle.iframe.style.height,
    publicQueryParams: [...iframeUrl.searchParams.keys()].sort().join(","),
    requestedDeploymentPreset: url.searchParams.get("deploymentPreset") ?? "",
    requestedSdkMode: url.searchParams.get("sdkMode") ?? "",
    requestedMemoryMode: url.searchParams.get("memoryMode") ?? "",
    requestedSidecarDisabled: url.searchParams.get("sidecar") === "disabled" && iframeSidecar === "disabled",
    noSecretQueryParams: hasNoSecretParams,
  },
  gates,
};

await mkdir(suiteDir, { recursive: true });
await writeFile(join(suiteDir, "results.json"), `${JSON.stringify(artifact, null, 2)}\n`);
await writeFile(join(suiteDir, "trace.jsonl"), `${gates.map((item) => JSON.stringify({ event: "gate", createdAt, gate: item })).join("\n")}\n`);
await writeFile(join(suiteDir, "summary.md"), buildSummary(artifact));
await writeFile(join(artifactRoot, "sdk-smoke-latest.json"), `${JSON.stringify(artifact, null, 2)}\n`);

console.log(`SDK smoke: ${artifact.passed ? "PASS" : "FAIL"}`);
console.log(`Results: ${join(suiteDir, "results.json")}`);
console.log(`Summary: ${join(suiteDir, "summary.md")}`);

if (!artifact.passed) {
  throw new Error("SDK smoke artifact failed acceptance checks.");
}

function gate(name: string, actual: GateResult["actual"], expected: GateResult["expected"], passed: boolean): GateResult {
  return { name, actual, expected, passed };
}

function buildSummary(results: typeof artifact): string {
  const rows = results.gates
    .map((item) => `| ${item.name} | ${item.passed ? "PASS" : "FAIL"} | ${item.actual} | ${item.expected} |`)
    .join("\n");

  return `# SDK Smoke Eval

- Created: ${results.createdAt}
- Passed: ${results.passed}
- Public query params: ${results.metrics.publicQueryParams}
- Requested deployment preset: ${results.metrics.requestedDeploymentPreset}
- Requested memory mode: ${results.metrics.requestedMemoryMode}

| Gate | Status | Actual | Expected |
| --- | --- | --- | --- |
${rows}
`;
}
