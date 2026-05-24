import type { MemoryProviderMode } from "@infinite-edge-agent/core";

export interface BrowserEmbedConfig {
  enabled: boolean;
  compact: boolean;
  sessionId?: string | undefined;
  tenantId?: string | undefined;
  cellId?: string | undefined;
  deploymentPreset?: "browser-only" | "remote-http" | "sidecar-disabled" | undefined;
  requestedMemoryMode?: "browser-vector" | "remote-http" | undefined;
}

export interface EmbedMemoryProviderInput {
  embed: BrowserEmbedConfig;
  configuredProvider: MemoryProviderMode | "sidecar";
  remoteMemoryUrl: string;
}

export function readBrowserEmbedConfig(search: string = globalThis.location?.search ?? ""): BrowserEmbedConfig {
  const params = new URLSearchParams(search);
  const enabled = params.get("embed") === "1" || params.get("sdk") === "browser";
  return {
    enabled,
    compact: enabled && params.get("compact") !== "0",
    sessionId: publicParam(params, "sessionId"),
    tenantId: publicParam(params, "tenantId"),
    cellId: publicParam(params, "cellId"),
    deploymentPreset: deploymentPreset(params.get("deploymentPreset")),
    requestedMemoryMode: memoryMode(params.get("memoryMode")),
  };
}

export function resolveEmbedMemoryProvider(input: EmbedMemoryProviderInput): MemoryProviderMode | "sidecar" {
  if (!input.embed.enabled) return input.configuredProvider;
  if (input.embed.deploymentPreset === "browser-only" || input.embed.deploymentPreset === "sidecar-disabled") {
    return "browser-vector";
  }
  if (input.embed.deploymentPreset === "remote-http" || input.embed.requestedMemoryMode === "remote-http") {
    return canUseEmbedRemoteMemory(input.remoteMemoryUrl) ? "remote-http" : "browser-vector";
  }
  return input.configuredProvider;
}

function publicParam(params: URLSearchParams, key: string): string | undefined {
  const value = params.get(key)?.trim();
  return value || undefined;
}

function deploymentPreset(value: string | null): BrowserEmbedConfig["deploymentPreset"] {
  if (value === "browser-only" || value === "remote-http" || value === "sidecar-disabled") return value;
  return undefined;
}

function memoryMode(value: string | null): BrowserEmbedConfig["requestedMemoryMode"] {
  if (value === "browser-vector" || value === "remote-http") return value;
  return undefined;
}

function canUseEmbedRemoteMemory(remoteMemoryUrl: string): boolean {
  const trimmed = remoteMemoryUrl.trim();
  return trimmed.startsWith("/") && !trimmed.startsWith("//");
}
