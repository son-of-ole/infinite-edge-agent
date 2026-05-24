# Memory Layer

This document is the production-memory entrypoint. The detailed design is split across:

- [03_MEMORY_MODEL.md](./03_MEMORY_MODEL.md) for chunking, retrieval, privacy, and lifecycle.
- [05_STORAGE.md](./05_STORAGE.md) for IndexedDB, LanceDB sidecar, persistence, backup, and deployment storage.

## Production MVP Contract

- Browser-only deployments can run with IndexedDB for local profile memory.
- Desktop, local-dev, and serious edge deployments should run the LanceDB sidecar.
- Every memory chunk must carry tenant/cell scope in `metadata.edgeTenantId` and `metadata.edgeCellId`.
- Search, list, export, trace list, and targeted delete operations must pass tenant/cell scope when a tenant/cell is configured.
- Remote memory APIs must enforce tenant/cell scope server-side; browser-bundled bearer secrets are not production-safe.

## Local Dev Profile

Use the durable sidecar profile for production-like local testing:

```bash
MEMORY_DB_URI=.data/lancedb \
MEMORY_TABLE=memory_chunks \
MEMORY_VECTOR_DIMENSION=384 \
pnpm dev:memory
```

Add `MEMORY_ENCRYPTION_KEY` to that command when the LanceDB sidecar should encrypt stored memory payload fields at rest.

Then set the web app to:

```bash
VITE_MEMORY_PROVIDER=sidecar
VITE_ENABLE_MEMORY_SERVER=true
VITE_MEMORY_SERVER_URL=http://127.0.0.1:8787
```

The production eval and release gate now verify the sidecar profile so an eval-only database cannot accidentally satisfy the durable-memory gate.
