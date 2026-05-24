# Storage

## Browser storage

The browser-only MVP uses IndexedDB.

Pros:

- Works offline.
- No install.
- Private to the browser profile.
- Good for prototypes and smaller stores.

Cons:

- Exact vector scan becomes slow as memory grows.
- Browser quota and eviction policies vary.
- No native ANN index in this scaffold.

## LanceDB sidecar

For serious local memory, run `apps/memory-server`. It provides:

- Local HTTP API.
- Embedded LanceDB on disk.
- Vector search over a persistent table.
- Clean path to desktop packaging.

API:

```http
GET /health
POST /memory/upsert
POST /memory/search
DELETE /memory
```

## Why sidecar instead of pure browser LanceDB

LanceDB’s TypeScript package is a native/Node-oriented embedded database. A normal web page cannot directly open arbitrary local filesystem paths or load native Node modules. The clean architecture is therefore:

- Browser-only: IndexedDB fallback.
- Desktop/edge: local sidecar with LanceDB.
- Future: browser OPFS/WASM vector index if a production-ready option is selected.

## Data directory

Default sidecar data lives under:

```text
.data/lancedb
```

Do not commit `.data`.

## Backup

Production backup options:

1. Export chunks as JSONL.
2. Snapshot LanceDB directory.
3. Encrypt backup file with user-controlled key.
4. Provide import validation before restore.
