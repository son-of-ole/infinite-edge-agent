import { mountInfiniteEdgeAgent } from "@infinite-edge-agent/browser-sdk";

const agent = mountInfiniteEdgeAgent({
  agentUrl: "https://agent.example.com",
  container: "#infinite-edge-agent",
  mode: "launcher",
  tenantId: "public-site",
  cellId: "support",
  sessionId: "visitor-session",
  deployment: {
    preset: "remote-http",
    runtimeProfile: "unlocked-browser",
    modelProfile: "qwen3-0.6b-sharded",
  },
});

window.addEventListener("pagehide", () => {
  agent.destroy();
});
