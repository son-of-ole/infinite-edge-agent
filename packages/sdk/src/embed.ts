export const INFINITE_EDGE_BROWSER_SDK_VERSION = "0.1.0";

export type InfiniteEdgeAgentEmbedMode = "inline" | "launcher";
export type InfiniteEdgeAgentDeploymentPreset = "browser-only" | "remote-http" | "sidecar-disabled";
export type InfiniteEdgeAgentMemoryMode = "browser-vector" | "remote-http";

export interface InfiniteEdgeAgentDeploymentConfig {
  preset: InfiniteEdgeAgentDeploymentPreset;
  runtimeProfile?: string | undefined;
  modelProfile?: string | undefined;
}

export interface InfiniteEdgeAgentEmbedOptions {
  agentUrl: string | URL;
  container?: HTMLElement | string | undefined;
  mode?: InfiniteEdgeAgentEmbedMode | undefined;
  compact?: boolean | undefined;
  title?: string | undefined;
  sessionId?: string | undefined;
  tenantId?: string | undefined;
  cellId?: string | undefined;
  width?: string | undefined;
  height?: string | undefined;
  className?: string | undefined;
  allow?: string | undefined;
  requireSecureOrigin?: boolean | undefined;
  document?: Document | undefined;
  deployment?: InfiniteEdgeAgentDeploymentPreset | InfiniteEdgeAgentDeploymentConfig | undefined;
}

export interface InfiniteEdgeAgentEmbedHandle {
  iframe: HTMLIFrameElement;
  url: URL;
  destroy(): void;
  focus(): void;
  setOpen(open: boolean): void;
}

const OBVIOUS_CREDENTIAL_QUERY_PARAM =
  /(token|secret|password|passwd|pwd|api[_-]?key|apikey|bearer|credential|jwt|authorization|signature|x-amz-|x-goog-)/i;
const EXACT_SHORT_CREDENTIAL_QUERY_PARAM = /^(code|auth|signed|signedurl|signed_url|signed-url)$/i;
const OBVIOUS_CREDENTIAL_QUERY_VALUE =
  /(^|[?&#;\s])(?:token|secret|password|passwd|pwd|api[_-]?key|apikey|credential|jwt|authorization|signature|x-amz-[^=&#\s]*|x-goog-[^=&#\s]*)=/i;
const SHORT_CREDENTIAL_QUERY_VALUE = /(^|[?&#;\s])(?:code|auth|signed|signedurl|signed_url|signed-url)=/i;
const BEARER_CREDENTIAL_VALUE = /bearer\s+/i;
const JWT_LIKE_VALUE = /^[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]+$/;
const OPENAI_KEY_LIKE_VALUE = /(^|[^A-Za-z0-9_-])sk-(?:proj-|svcacct-)?[A-Za-z0-9_-]{20,}($|[^A-Za-z0-9_-])/i;
const DEFAULT_IFRAME_ALLOW = "webgpu; cross-origin-isolated";

export function buildInfiniteEdgeAgentUrl(options: InfiniteEdgeAgentEmbedOptions): URL {
  const url = new URL(options.agentUrl.toString(), globalThis.location?.href ?? "http://127.0.0.1");
  assertSafeAgentUrl(url, options.requireSecureOrigin ?? true);

  url.searchParams.set("embed", "1");
  url.searchParams.set("sdk", "browser");
  url.searchParams.set("sdkVersion", INFINITE_EDGE_BROWSER_SDK_VERSION);
  url.searchParams.set("sdkMode", options.mode ?? "inline");
  if (options.compact ?? true) url.searchParams.set("compact", "1");
  setPublicParam(url, "sessionId", options.sessionId);
  setPublicParam(url, "tenantId", options.tenantId);
  setPublicParam(url, "cellId", options.cellId);
  applyDeploymentConfig(url, options.deployment);

  return url;
}

export function mountInfiniteEdgeAgent(options: InfiniteEdgeAgentEmbedOptions): InfiniteEdgeAgentEmbedHandle {
  const documentRef = options.document ?? globalThis.document;
  if (!documentRef) {
    throw new Error("Infinite Edge Agent browser SDK requires a DOM document to mount into.");
  }

  const container = resolveContainer(documentRef, options.container);
  const url = buildInfiniteEdgeAgentUrl(options);
  const iframe = documentRef.createElement("iframe") as HTMLIFrameElement;
  iframe.src = url.toString();
  iframe.title = options.title ?? "Infinite Edge Agent";
  iframe.loading = "eager";
  iframe.referrerPolicy = "strict-origin-when-cross-origin";
  iframe.setAttribute("allow", options.allow ?? DEFAULT_IFRAME_ALLOW);
  iframe.className = options.className ?? "infinite-edge-agent-frame";
  iframe.style.border = "0";
  iframe.style.width = options.width ?? "100%";
  iframe.style.height = options.height ?? "720px";
  iframe.style.colorScheme = "dark";

  let wrapper: HTMLElement | null = null;
  if ((options.mode ?? "inline") === "launcher") {
    wrapper = mountLauncher(documentRef, container, iframe);
  } else {
    container.appendChild(iframe);
  }

  return {
    iframe,
    url,
    destroy() {
      wrapper?.remove();
      if (!wrapper) iframe.remove();
    },
    focus() {
      iframe.focus();
    },
    setOpen(open) {
      if (wrapper) {
        iframe.hidden = !open;
      }
    },
  };
}

function assertSafeAgentUrl(url: URL, requireSecureOrigin: boolean): void {
  if (url.hash) {
    throw new Error("Infinite Edge Agent embed URL fragments are not allowed because fragments commonly carry credentials.");
  }
  if (url.username || url.password) {
    throw new Error("Infinite Edge Agent embed URL userinfo is not allowed because userinfo commonly carries credentials.");
  }

  for (const [key, value] of url.searchParams) {
    if (isCredentialQueryKey(key)) {
      throwCredentialQueryError();
    }
    assertSafePublicValue(value);
  }

  if (!requireSecureOrigin) return;
  if (url.protocol === "https:" || isLocalhost(url.hostname)) return;
  throw new Error("Infinite Edge Agent embed URL must be HTTPS or localhost by default.");
}

function resolveContainer(documentRef: Document, container: HTMLElement | string | undefined): HTMLElement {
  if (!container) return documentRef.body;
  if (typeof container !== "string") return container;
  const resolved = documentRef.querySelector(container);
  if (!resolved) throw new Error(`Infinite Edge Agent embed container was not found: ${container}`);
  return resolved as HTMLElement;
}

function mountLauncher(documentRef: Document, container: HTMLElement, iframe: HTMLIFrameElement): HTMLElement {
  const wrapper = documentRef.createElement("div");
  wrapper.className = "infinite-edge-agent-launcher";
  wrapper.style.position = "fixed";
  wrapper.style.right = "24px";
  wrapper.style.bottom = "24px";
  wrapper.style.zIndex = "2147483647";

  const button = documentRef.createElement("button");
  button.type = "button";
  button.textContent = "Agent";
  button.setAttribute("aria-expanded", "false");
  button.setAttribute("aria-label", "Open Infinite Edge Agent");
  iframe.hidden = true;
  iframe.style.width = "min(420px, calc(100vw - 48px))";
  iframe.style.height = "min(720px, calc(100vh - 112px))";
  iframe.style.marginTop = "12px";
  iframe.style.borderRadius = "8px";

  button.addEventListener("click", () => {
    iframe.hidden = !iframe.hidden;
    button.setAttribute("aria-expanded", String(!iframe.hidden));
    if (!iframe.hidden) iframe.focus();
  });

  wrapper.appendChild(button);
  wrapper.appendChild(iframe);
  container.appendChild(wrapper);
  return wrapper;
}

function applyDeploymentConfig(url: URL, deployment: InfiniteEdgeAgentEmbedOptions["deployment"]): void {
  if (!deployment) return;
  const config = typeof deployment === "string" ? { preset: deployment } : deployment;
  const memoryMode = memoryModeForPreset(config.preset);

  setPublicParam(url, "deploymentPreset", config.preset);
  setPublicParam(url, "memoryMode", memoryMode);
  setPublicParam(url, "sidecar", "disabled");
  setPublicParam(url, "runtimeProfile", config.runtimeProfile);
  setPublicParam(url, "modelProfile", config.modelProfile);
}

function memoryModeForPreset(preset: InfiniteEdgeAgentDeploymentPreset): InfiniteEdgeAgentMemoryMode {
  return preset === "remote-http" ? "remote-http" : "browser-vector";
}

function setPublicParam(url: URL, key: string, value: string | undefined): void {
  const normalized = value?.trim();
  assertSafePublicValue(normalized);
  if (normalized) url.searchParams.set(key, normalized);
}

function assertSafePublicValue(value: string | undefined): void {
  if (
    value
    && (
      OBVIOUS_CREDENTIAL_QUERY_VALUE.test(value)
      || SHORT_CREDENTIAL_QUERY_VALUE.test(value)
      || BEARER_CREDENTIAL_VALUE.test(value)
      || JWT_LIKE_VALUE.test(value)
      || OPENAI_KEY_LIKE_VALUE.test(value)
    )
  ) {
    throwCredentialQueryError();
  }
}

function isCredentialQueryKey(key: string): boolean {
  return OBVIOUS_CREDENTIAL_QUERY_PARAM.test(key) || EXACT_SHORT_CREDENTIAL_QUERY_PARAM.test(key);
}

function throwCredentialQueryError(): never {
  throw new Error(
    "Do not place credentials in the Infinite Edge Agent embed URL. Use a same-origin memory proxy or secure cookie/session layer.",
  );
}

function isLocalhost(hostname: string): boolean {
  return hostname === "localhost" || hostname === "127.0.0.1" || hostname === "::1" || hostname.endsWith(".localhost");
}
