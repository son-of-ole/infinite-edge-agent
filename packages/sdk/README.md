# Infinite Edge Agent Browser SDK

Embed a hosted Infinite Edge Agent browser app into any website.

```ts
import { mountInfiniteEdgeAgent } from "@infinite-edge-agent/browser-sdk";

const handle = mountInfiniteEdgeAgent({
  agentUrl: "https://agent.example.com",
  container: "#edge-agent",
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
```

The SDK creates an iframe with `embed=1`, `sdk=browser`, `sdkVersion`, and `sdkMode` routing. Compact chrome is enabled by default, launcher mode is supported, and HTTPS is required unless the agent is served from localhost.

## Deployment presets

Use `deployment` for public runtime shape only:

```ts
deployment: "browser-only"
deployment: "sidecar-disabled"
deployment: { preset: "remote-http", runtimeProfile: "unlocked-browser" }
```

- `browser-only`: browser vector/memory mode, no sidecar.
- `sidecar-disabled`: explicit browser-vector mode for hosted/static embeds.
- `remote-http`: requests remote-memory routing from the hosted app; the hosted app deployment must still be configured to use an authenticated remote memory API.

The SDK writes only public query params such as `deploymentPreset`, `memoryMode`, `sidecar=disabled`, `runtimeProfile`, and `modelProfile`. These are requested embed options, not proof of the hosted app's observed runtime memory mode. The SDK blocks URL-carried credentials such as tokens, JWTs, auth params, OAuth codes, signed URL params, bearer values, and URL fragments. Keep credentials out of browser URLs; use a same-origin proxy, secure cookies, or server-side session auth for hosted memory.

Run `pnpm smoke:sdk` after dependencies are installed. The root script builds `@infinite-edge-agent/browser-sdk`, packs and installs it into a temporary consumer, verifies a plain Node import of `@infinite-edge-agent/browser-sdk`, and then runs the dist-based iframe smoke.

See `examples/sdk-embed/hosted-launcher.ts` and `docs/52_BROWSER_SDK_AND_DEPLOYMENT.md` for the deploy header, CSP, and model-asset requirements.
