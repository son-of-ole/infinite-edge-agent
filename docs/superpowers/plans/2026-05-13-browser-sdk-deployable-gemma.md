# Browser SDK + Deployable Gemma Production Slice

## Goal

Make Infinite Edge Agent runnable as a hosted browser app with Google Gemma 3n E2B through MediaPipe, and embeddable on any website through a small browser SDK.

## Implementation Steps

1. Add core Gemma artifact helpers for safe filenames, served model paths, SHA sidecar names, and generated browser env values.
2. Add CLI tooling to install or verify a local/remote `gemma-3n-E2B-it-int4-Web.litertlm` artifact into `apps/web/public/models`.
3. Update browser model integrity checks so large production Gemma files can use a tiny `.sha256` sidecar instead of forcing the browser to buffer a multi-GB model.
4. Add `@infinite-edge-agent/browser-sdk` for iframe embedding on any website, with secret-query rejection and localhost/HTTPS validation.
5. Add embed mode to the web app so SDK-hosted iframe usage has a compact UI.
6. Add deployment headers and docs for Vercel/static hosting, model asset placement, LanceDB memory modes, and open-source configuration.
7. Run typecheck, tests, build, smoke, stress, production eval, and browser preview verification.

## Acceptance Gates

- `pnpm typecheck`
- `pnpm test`
- `pnpm smoke:core`
- `pnpm stress:core`
- `pnpm eval:production`
- `pnpm build`
- Browser preview initializes the app and reports the Gemma asset state clearly.
