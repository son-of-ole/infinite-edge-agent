# 24 — Schema Registry

## Runtime feature status

```ts
interface RuntimeFeatureStatus {
  name: "lancedb" | "contextRuntime" | "ssa" | "tsp" | "mtp" | "kvswap" | "inferenceBackend";
  state: "required" | "enabled" | "fallback" | "disabled_for_test" | "unavailable";
  mode: string;
  reason?: string;
  impact?: string;
  metrics?: Record<string, number | string | boolean>;
}
```

## Runtime trace

```ts
interface RuntimeTrace {
  traceId: string;
  requestId: string;
  sessionId: string;
  createdAt: string;
  features: RuntimeFeatureStatus[];
  context: ContextRuntimeTrace;
  ssa: SSATrace;
  tsp: TSPTrace;
  mtp: MTPTrace;
  kvswap: KVSwapTrace;
  memory: MemoryTrace;
  latency: LatencyTrace;
}
```

## Memory chunk

```ts
interface MemoryChunk {
  id: string;
  text: string;
  embedding: number[];
  sessionId: string;
  source: "chat" | "document" | "summary" | "tool" | "system";
  role?: "system" | "user" | "assistant" | "tool";
  createdAt: string;
  updatedAt: string;
  tags: string[];
  metadata: Record<string, unknown>;
  tokenCount: number;
  importance?: number;
  provenance?: ProvenanceRef[];
}
```

## Provenance ref

```ts
interface ProvenanceRef {
  sourceType: "ledger" | "memory_chunk" | "document" | "tool" | "summary";
  sourceId: string;
  startOffset?: number;
  endOffset?: number;
  hash?: string;
}
```

## Context block

```ts
interface ContextBlock {
  id: string;
  text: string;
  tokenStart: number;
  tokenEnd: number;
  priority: number;
  source: string;
  provenance: ProvenanceRef[];
}
```

## KV block

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
}
```
