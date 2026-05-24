# Inference Runtime

## MVP runtime

The MVP runtime is the repo-owned `unlocked-browser-transformer` lane. It loads a converted Qwen manifest and browser-served weight shards, then runs generation through the core backend instead of a hosted or opaque browser chat API.

The main thread initializes the client with the selected manifest:

```ts
const client = new UnlockedBrowserTransformerClient({
  modelId: "Qwen/Qwen3-0.6B",
  manifestPath: "/models/qwen3-0.6b-unlocked/manifest.json",
  backendPreference: "webgpu"
});
await client.init();
```

The runtime owns model tensors, Q/K/V handles, SSA routing, KV paging traces, TSP callback execution, and MTP/target-only decode proof metadata.

## Model selection

The default model target is:

- `Qwen/Qwen3-0.6B`

Additional models must use the same unlocked manifest contract before they can be production lanes.

## Prompt assembly

The app does not feed the full database into the model. It builds a prompt from:

1. Stable system prompt.
2. Retrieved long-term memory.
3. Recent conversation tail.
4. Latest user message.

This is the production-safe version of “infinite context”: persistent local memory plus active-context packing.

## Runtime capabilities

### SSA

Expose as a model/runtime capability flag and proof trace:

```ts
interface AttentionRuntimeCapabilities {
  supportsSparseAttention: boolean;
  maxContextTokens: number;
  attentionPattern?: "dense" | "sliding" | "ssa" | "hybrid";
}
```

### Speculative decoding

Expose as a generation option only when the runtime supports tokenizer-compatible drafter/target verification:

```ts
interface SpeculativeOptions {
  enabled: boolean;
  drafterModelId?: string;
  numSpeculativeTokens: number;
}
```

### KV cache offload

Expose as runtime telemetry and policy:

```ts
interface KvCachePolicy {
  pinnedTokenRanges: Array<{ start: number; end: number; reason: string }>;
  maxVramBytes: number;
  spillPath?: string;
}
```
