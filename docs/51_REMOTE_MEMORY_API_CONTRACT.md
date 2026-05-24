# 51 — Remote Memory API Contract

## Purpose

Infinite Edge Agent is open-source and memory-provider neutral. The default build uses `VITE_MEMORY_PROVIDER=browser-vector`, an IndexedDB-backed browser-native vector store with deterministic local search, metadata filters, import/export bundles, context-pack trace persistence, and `vectorDimension=384` for the default `Xenova/all-MiniLM-L6-v2` embedding model. The older `indexeddb` provider name remains a compatibility alias. Teams that want shared, server-backed memory can set:

```text
VITE_MEMORY_PROVIDER=remote-http
VITE_REMOTE_MEMORY_URL=https://your-memory.example.com/api/edge-ai
VITE_REMOTE_MEMORY_CREDENTIALS=same-origin
```

First-party teams may use their own internal deployment for testing, but no hosted service is part of the open-source default. Any user can deploy the included memory server or implement this contract with their own database.

## Authentication and routing headers

Browser applications must not use a shared bearer token from a `VITE_*` variable as a production secret. `VITE_*` values are bundled into public JavaScript. Production browser deployments should point `VITE_REMOTE_MEMORY_URL` at a same-origin authenticated proxy or an API that authenticates the user with secure cookies/session state and enforces tenant/cell scope server-side. The browser client sends fetch credentials according to `VITE_REMOTE_MEMORY_CREDENTIALS`, which defaults to `same-origin`.

Browser-originated requests may include:

```http
X-Edge-Agent-Tenant: <VITE_REMOTE_MEMORY_TENANT_ID>
X-Edge-Agent-Cell: <VITE_REMOTE_MEMORY_CELL_ID>
Content-Type: application/json
```

The included memory server's remote `/api/edge-ai` namespace is intended for private sidecar, server-side, or trusted proxy use. It must be configured with `MEMORY_SERVER_TOKEN`, `MEMORY_TENANT_ID`, and `MEMORY_CELL_ID`; remote requests to that server must provide the matching bearer token and tenant/cell headers. Do not expose that token directly to a public browser bundle. GAC payload or query `tenantId`/`cellId` values that do not match the authenticated scope are rejected with `GAC_SCOPE_MISMATCH`; runtime trace payload or query `tenantId`/`cellId` values that do not match the authenticated scope are rejected with `RUNTIME_TRACE_SCOPE_MISMATCH`. Missing remote security configuration is rejected before scoped data is served. Root sidecar routes remain explicit local routes.

## Required endpoints

### `GET /health`

Returns `200 OK` when the memory database is ready.

### `POST /memory/upsert`

Request:

```json
{
  "chunks": [
    {
      "id": "chunk_1",
      "text": "Memory text",
      "embedding": [0.1, 0.2],
      "sessionId": "session_1",
      "source": "chat",
      "role": "user",
      "createdAt": "2026-05-11T00:00:00.000Z",
      "updatedAt": "2026-05-11T00:00:00.000Z",
      "tags": ["user"],
      "metadata": {},
      "tokenCount": 8
    }
  ]
}
```

### `POST /memory/search`

Request:

```json
{
  "embedding": [0.1, 0.2],
  "options": {
    "limit": 8,
    "minScore": 0.15,
    "sessionId": "session_1",
    "tags": ["user"]
  }
}
```

Response:

```json
{
  "hits": [
    {
      "id": "chunk_1",
      "text": "Memory text",
      "embedding": [0.1, 0.2],
      "sessionId": "session_1",
      "source": "chat",
      "role": "user",
      "createdAt": "2026-05-11T00:00:00.000Z",
      "updatedAt": "2026-05-11T00:00:00.000Z",
      "tags": ["user"],
      "metadata": {},
      "tokenCount": 8,
      "score": 0.92
    }
  ]
}
```

### `DELETE /memory`

Clears memory in the scoped tenant/cell. Production deployments should restrict this endpoint by policy.

### `DELETE /memory/query`

Deletes memory in the scoped tenant/cell by session, by tag/project, or by both. At least one target is required.

Request:

```json
{
  "options": {
    "sessionId": "session_1",
    "tags": ["project:edge-ai"]
  }
}
```

Response:

```json
{
  "ok": true,
  "count": 3
}
```

### `GET /memory/export?sessionId=session_1&limit=100`

Response:

```json
{
  "chunks": []
}
```

### `POST /memory/import`

Request:

```json
{
  "chunks": []
}
```

Response:

```json
{
  "ok": true,
  "count": 0
}
```

### `POST /runtime/traces`

Request:

```json
{
  "trace": {
    "traceId": "trace_1",
    "requestId": "request_1",
    "sessionId": "session_1",
    "tenantId": "tenant_1",
    "cellId": "cell_1",
    "modelId": "Qwen/Qwen3-0.6B",
    "backend": "unlocked-browser-transformer",
    "createdAt": "2026-05-11T00:00:00.000Z",
    "runtime": {}
  }
}
```

Remote implementations must scope runtime trace writes and reads by authenticated tenant/cell. The included memory server stamps missing trace `tenantId`/`cellId` from the trusted request scope and rejects forged mismatches; unprefixed local sidecar routes remain local/admin routes.

### `GET /runtime/traces?sessionId=session_1&limit=20`

Response:

```json
{
  "traces": []
}
```

## GAC persistence endpoints

The same base URL also exposes versionable Geometry-Aware Consolidation persistence routes under `/gac/*`. The included memory server registers these routes both at the root sidecar namespace and under `MEMORY_API_PREFIX` such as `/api/edge-ai`; the prefixed namespace uses the same bearer, tenant, and cell auth pre-handler as the other remote endpoints.

All write endpoints accept:

```json
{
  "records": []
}
```

and return:

```json
{
  "ok": true,
  "count": 1,
  "traceId": "gac_context_pack_trace_..."
}
```

Available endpoints:

| Endpoint | Write | List response | Common filters |
|---|---|---|---|
| `/gac/raw-memory` | `POST` | `{ "records": [] }` | `tenantId`, `cellId`, `sessionId`, `rawMemoryId`, `limit` |
| `/gac/identity-pins` | `POST` | `{ "records": [] }` | `tenantId`, `cellId`, `sessionId`, `rawMemoryId`, `limit` |
| `/gac/clusters` | `POST` | `{ "records": [] }` | `tenantId`, `cellId`, `sessionId`, `rawMemoryId`, `limit` |
| `/gac/cluster-metrics` | `POST` | `{ "records": [] }` | `tenantId`, `cellId`, `sessionId`, `limit` |
| `/gac/representatives` | `POST` | `{ "records": [] }` | `tenantId`, `cellId`, `sessionId`, `representativeId`, `limit` |
| `/gac/lineage` | `POST` | `{ "records": [] }` | `tenantId`, `cellId`, `sessionId`, `rawMemoryId`, `representativeId`, `limit` |
| `/gac/consolidation-runs` | `POST` | `{ "records": [] }` | `tenantId`, `cellId`, `sessionId`, `limit` |
| `/gac/retrieval-audits` | `POST` | `{ "records": [] }` | `tenantId`, `cellId`, `sessionId`, `rawMemoryId`, `representativeId`, `limit` |
| `/gac/context-pack-traces` | `POST` | `{ "traces": [] }` | `tenantId`, `cellId`, `sessionId`, `contextPackId`, `limit` |
| `/gac/model-memory-actions` | `POST` | `{ "records": [] }` | `tenantId`, `cellId`, `sessionId`, `limit` |
| `/gac/contradictions` | `POST` | `{ "records": [] }` | `tenantId`, `cellId`, `sessionId`, `rawMemoryId`, `limit` |
| `/gac/source-documents` | `POST` | `{ "records": [] }` | `tenantId`, `cellId`, `sessionId`, `limit` |
| `/gac/training-examples` | `POST` | `{ "records": [] }` | `tenantId`, `cellId`, `sessionId`, `rawMemoryId`, `limit` |

Context-pack trace persistence is a hard runtime requirement for model calls in production-supported memory modes. The zero-config IndexedDB store, local sidecar store, and remote HTTP store all implement `writeContextPackTraces`; generation should block when a configured store cannot write the trace.

Representatives that are model-visible or factual must be written with lineage in the same request:

```json
{
  "records": [
    {
      "id": "rep_1",
      "tenantId": "tenant_1",
      "cellId": "cell_1",
      "clusterId": "cluster_1",
      "clusterVersion": 1,
      "type": "summary",
      "embedding": [0.1, 0.2],
      "riskScore": 0.2,
      "coverageScore": 0.9,
      "createdByRunId": "run_1",
      "createdAt": "2026-05-11T00:00:00.000Z",
      "modelVisible": true,
      "factual": true
    }
  ],
  "lineage": [
    {
      "representativeId": "rep_1",
      "rawMemoryId": "raw_1",
      "tenantId": "tenant_1",
      "cellId": "cell_1",
      "membershipWeight": 1,
      "distanceToRep": 0.1,
      "isPrimary": true,
      "createdAt": "2026-05-11T00:00:00.000Z"
    }
  ]
}
```

If required lineage is missing, the server returns `400` with `errorCode: "MISSING_LINEAGE"` and a trace id.

## Deployment rules

- `VITE_MEMORY_PROVIDER=indexeddb` is the zero-config open-source default.
- `VITE_MEMORY_PROVIDER=remote-http` requires a healthy `VITE_REMOTE_MEMORY_URL`.
- Production browser remote memory must use a same-origin authenticated proxy or secure cookie/session layer. `VITE_REMOTE_MEMORY_TOKEN` is intentionally not part of the browser configuration contract and production readiness blocks builds that configure it.
- The included `@infinite-edge-agent/memory-server` implements this namespace when `MEMORY_API_PREFIX=/api/edge-ai`.
- Runtime traces include `memoryProvider` status with `mode=remote-http` when this API is active.
- Set `MEMORY_SERVER_TOKEN`, `MEMORY_TENANT_ID`, and `MEMORY_CELL_ID` on the deployed memory server when it is called by a trusted server, private sidecar, or proxy. Remote `/api/edge-ai` routes reject requests until all three are configured.
- Set `MEMORY_CORS_ORIGIN` to the allowed web app origin list. Use `*` only for throwaway local testing.
