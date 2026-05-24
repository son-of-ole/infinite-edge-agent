# 17 — KVSwap Runtime Spec

## Role

KVSwap manages Key/Value cache growth during long-context inference. It tiers cache blocks across VRAM, RAM, and disk, with pinning, eviction, prediction, and prefetch.

## Why it is first-class

Even if attention becomes subquadratic, KV state grows with context. A persistent agent needs a cache policy that treats memory as a hierarchy rather than a single VRAM bucket.

## Modes

| Mode | Meaning |
|---|---|
| `metadata_only` | Track estimated KV blocks; no real tensor paging |
| `ram_tier` | Backend can move KV blocks between GPU and CPU memory |
| `disk_tier` | Backend can persist KV blocks to local disk |
| `predictive` | Runtime predicts needed blocks and prefetches them |

## KV block model

```ts
interface KVBlock {
  id: string;
  layer: number;
  startToken: number;
  endToken: number;
  tier: "vram" | "ram" | "disk";
  pinned: boolean;
  importance: number;
  lastAccessAt: number;
  sourceBlockId?: string;
  checksum?: string;
}
```

## Pinning rules

Always pin:

- system prompt,
- safety policy,
- current user request,
- current task constraints,
- tool call results needed for the current answer,
- top memory hits selected by SSA,
- recent generated tokens.

## Eviction rules

Evict candidates when:

- estimated VRAM usage exceeds threshold,
- block is not pinned,
- importance is below active threshold,
- recency is low,
- source can be reconstructed from LanceDB/context ledger.

## Prefetch rules

Prefetch candidates when:

- SSA selected corresponding context block,
- user query embedding matches block summary,
- decoder is about to verify speculative tokens that depend on a block,
- a tool result references the block.

Predictive prefetch hints carry a confidence score and reason list into the KVSwap decision proof. Today those hints are produced from the predictive runtime plan: GAC pins and representative memories, SSA-selected sparse blocks, and MTP branch pressure all become candidate hot pages before the target decode step needs them.

## Low-rank key summaries

Production KVSwap should store compressed key summaries for block utility prediction. The fallback manager stores metadata only, but the schema reserves:

```ts
compressedKeySummary?: Float32Array | string;
summaryRank?: number;
```

## Acceptance gates

- No pinned block is evicted.
- Cache pressure produces deterministic eviction decisions.
- Every eviction has a reason.
- In metadata mode, the runtime still reports pressure and projected memory savings.
- Production backend can attach real tensor handles to `KVBlock` without changing context code.

## Current implementation

`KVTensorPagingRegistry` owns registered `KVBlock` records with optional backend key/value tensor handles. It pages blocks across `vram`, `ram`, and serialized `disk` tiers, protects pinned blocks from eviction, promotes selected blocks before sparse attention, and round-trips compressed key summaries through a typed disk serialization format.

The browser app adds a KVSwap persistence adapter for unlocked local sessions. It prefers OPFS, falls back to IndexedDB, and uses an in-memory implementation for tests or unsupported environments. Persisted records are schema-versioned serialized KV blocks containing block metadata, prompt token identity, runtime block id, Q/K/V rows, hidden rows, compressed key summaries, byte estimates, and timestamps. Decode proofs and runtime traces include hydrate, persist, reuse, evict, and clear events with storage mode and quota/usage metadata. Exact-match decode reuse can skip fresh prefill and reconstruct browser-owned KV state only when namespace, model id/fingerprint, prompt token IDs, prompt hash, runtime layer count, policy hash, and every serialized row range match. Quota eviction skips pinned blocks, corrupt or malformed records are quarantined during load/list, and private/session reset clears the tenant/cell/session namespace across model switches.

The browser client persists the prompt/prefill KV state once per matching prompt by default. Decode steps append to the live in-memory KV cache but no longer rewrite the full serialized KV cache after every generated token. This keeps OPFS/IndexedDB from becoming the hot decode loop while preserving exact-match prefill reuse for later turns.

Decode reuse proof is per generation, not sticky client state. A fresh browser session using the same namespace may hydrate persisted prefill blocks and report `decodeReuse=true` for an exact prompt/runtime match, but a later non-matching prompt must show no reuse event and `decodeReuse=false`.
