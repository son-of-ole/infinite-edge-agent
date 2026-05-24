# 53 - Unlocked Browser Runtime

## Decision

The production target for real SSA/KV/TSP is a custom browser-owned transformer runtime, not an opaque browser chat API or a hosted OpenAI-compatible endpoint.

`Qwen/Qwen3-0.6B` is the primary model target because it is small enough for browser work, Apache-2.0 licensed, and has a modern decoder architecture with grouped-query attention. Diffusion models are rejected for this runtime because the project needs autoregressive KV-cache ownership, verifier batching, and sequence/tensor scheduling.

## Runtime Boundary

The unlocked backend must own:

- token embeddings and layer weights,
- Q/K/V projection tensors,
- KV block tensor handles,
- sparse attention routing policy,
- KV prefetch/paging decisions,
- TSP schedule callbacks,
- decode proof traces.

Opaque text-generation APIs may remain as compatibility fallbacks, but they cannot satisfy the unlocked claim unless they expose equivalent tensor and schedule controls.

The shipped web app is locked to the unlocked Qwen lane. `VITE_REQUIRE_UNLOCKED_RUNTIME=true` fails closed for any runtime backend other than `unlocked-browser-transformer`; opaque chat clients and OpenAI-compatible proxy clients are not part of the production unlocked system.

## Implemented Surface

- `UnlockedBrowserTransformerBackend` implements the native SSA backend contract.
- Prefill creates browser-owned Q/K/V tensor handles from model weights.
- KV blocks are registered with `KVTensorPagingRegistry`.
- Decode executes a TSP schedule with `kv_prefetch`, `attention`, and `mlp` callbacks.
- Q/K/V and prefill output projections route through a reusable dense matmul kernel boundary; strict real-Qwen lanes require WebGPU and fail closed on CPU fallback.
- Stable descriptor-backed projection matrices and compute pipelines can be reused in runtime-lifetime WebGPU resources when the live runtime provides explicit cache keys and opts into the stable-cache policy where needed; mutable plain arrays are uploaded per call.
- SSA WebGPU shape-bucket planning is implemented for prompt lengths, selected-block counts, head dimensions, and tile rows. Prefill plans now expose bounded chunk metadata and stable static pipeline keys (`prefillChunkCount`, `prefillChunkSize`, `shapeBucket`, `pipelineCacheKey`, and `maxDispatchEstimatedMs`) through prefill/decode proof surfaces.
- Dense matmul, RMSNorm, sparse attention, residual add, and batched MLP now have GPU-resident primitive paths that return `webgpu_resident_tensor` handles, accept resident handles as chained inputs, and avoid readback until `readWebGpuResidentTensor()` or final top-k materialization is called.
- The real decode layer path uses resident tensors for O projection, attention residual, post-attention RMSNorm, MLP, and MLP residual before materializing the layer hidden row for the existing KV/SSA state.
- Prefill causal attention routes through the sparse-attention kernel boundary per packed attention head, using dense causal selected-key indexes and recording backend proof metadata on the KV cache handle.
- Long-prompt prefill now executes causal attention through the static chunk plan instead of only reporting planning metadata. Multi-chunk prefill slices query-token windows, dispatches sparse attention per packed head per chunk, awaits between chunk windows, and reports `prefillChunkDispatch`, `attentionDispatchCount`, and `awaitedDispatchBreaks` on proof surfaces.
- Browser runtime benchmarks now have deterministic long-prompt controls (`--long-prompt-target-tokens`, `--long-prompt-repeat`, `BROWSER_RUNTIME_BENCH_LONG_PROMPT_TARGET_TOKENS`, and the matching `/__bench/browser-runtime?longPromptTargetTokens=...` / `longPromptRepeat=...` URL controls). Benchmark artifacts and browser preview payloads carry `prefillChunkCount`, `prefillChunkSize`, `shapeBucket`, `pipelineCacheKey`, `maxDispatchEstimatedMs`, and `prefillChunkDispatch` when proof metadata exists. `--require-long-prompt-proof` and `strictLongPrompt=true` now require the proof to execute and report chunked dispatch rather than accepting planning-only claims.
- Strict browser WebGPU benchmark routes force full-vocab top-k logits whenever the logits gate is requested, even if a faster balanced/profile candidate-logit cap is present. Persisted KV prefill blocks also carry compact prefill proof metadata, so exact prompt KV reuse can prove the reused cache was originally produced through the same strict prefill projection, attention, packed-head, and MLP lanes.
- Full-vocab decode logits route through the tiled dense-matvec top-k boundary with backend, selected/full row count, scanned row count, tile count, and purpose proof metadata. When WebGPU is active, the decode hidden row is uploaded as a resident tensor and the top-k projection trace records `vectorResident: true`. Candidate-token projection remains available only as an explicit debug/budget override; strict Qwen gates reject bounded candidate-logit evidence.
- MTP can run through the unlocked browser verifier path when configured. The default browser drafter id, `browser/qwen-prefix-drafter`, is a tokenizer-compatible shallow Qwen draft path that uses the same manifest/tokenizer and browser-owned decode machinery as the target, with `VITE_MTP_DRAFT_LAYER_COUNT` controlling how many leading Qwen layers it uses. The deterministic `browser/ngram-drafter` remains available as a proof/fallback source. Target verification runs as a batched continuation pass owned by `UnlockedBrowserTransformerBackend.verifySpeculativeDraft`, and only the accepted-prefix/correction input rows are committed back into the live KV cache.
- Browser KVSwap persistence serializes local, non-sensitive KV blocks into binary OPFS records with `createSyncAccessHandle()` when the worker-owned file handle supports it, then falls back to async JSON OPFS, IndexedDB, and memory. Trace metadata records hydrate, persist, reuse, evict, and clear operations with mode/detail, quota/usage, block ids, binary/sync-handle proof, Web Locks or single-worker routing metadata, lock wait, bytes read, bytes written, and health. Exact-match reuse can reconstruct prompt Q/K/V/hidden rows and skip fresh prefill when namespace, model id/fingerprint, prompt token IDs, prompt hash, runtime layer count, and policy hash all match.
- The Advanced Runtime Coordinator emits a predictive runtime plan for every turn. GAC raw/representative lineage predicts future retrievals, SSA-selected blocks become sparse/KV hot-page predictions, MTP branches declare future verifier pressure, and KVSwap receives confidence-scored predictive prefetch hints. The same predictive plan is written to context-pack traces and shown in the browser runtime panel.
- Context rebuild now performs adaptive GAC-aware memory selection. Prior context-pack traces, retrieval audits, identity pins, and raw recovery records are read before each generation plan; exact pinned raw memory and failed-audit raw memory are reintroduced even when vector search misses them. Prior trace learning matches stable raw/representative/pin lineage instead of only transient hit IDs. Runtime traces expose `memoryPriorityMap`, `sourceLineageMap`, `contextRebuildLearning`, and `rawMemoryRecovery`.
- SSA, KVSwap, and predictive runtime planning now consume the active packed context, not dropped retrieval frames, so omitted memories cannot silently affect sparse routing or cache hot-page claims.
- Browser/local ingestion schedules adaptive consolidation jobs when the memory store exposes raw-memory, identity-pin, retrieval-audit, and consolidation-run methods. Candidate jobs protect identity pins, legal/security/pinned records, and raw facts that failed retrieval audits; only unprotected normal raw memories enter the background consolidation candidate set.
- Browser vector memory is a first-class production mode via `VITE_MEMORY_PROVIDER=browser-vector`. It is backed by IndexedDB, records local-only/import-export/context-pack-trace capability metadata, performs deterministic vector search with metadata filters, and does not require the LanceDB sidecar or a remote HTTP endpoint. `indexeddb` remains a compatibility alias.
- The web client exposes an `unlocked-browser-transformer` backend and emits a decode proof.
- Production readiness blocks fixture weights and requires `VITE_UNLOCKED_MODEL_MANIFEST_PATH`.
- Runtime feature status reports SSA, KVSwap, TSP, and MTP as enabled only when backend capabilities expose the needed controls.

## Browser MTP Configuration

## Current Real-Model Gate

Fixture-backed release gates prove the runtime contracts, tensor shapes, trace surfaces, and fallback behavior. They do not by themselves prove final response quality for the installed 2.8 GB Qwen shard set. A production claim must include a direct real-model benchmark and a real-browser smoke/eval run where fixture mode is disabled, the converted manifest is active, and required expected-answer checks pass within the configured latency budget.

The current local finding is precise: `Qwen/Qwen3-0.6B` with thinking disabled now passes the direct unlocked Utah sentinel (`The capital of Utah is **Salt Lake City**.`) while keeping visible output immediate. `VITE_QWEN_THINKING_MODE=enabled` remains available for explicit hidden-reasoning evals, but it can make the chat look blank or tiny while `<think>...</think>` tokens are filtered. The stronger `Qwen/Qwen3-1.7B` source artifact has been converted into `apps/web/public/models/qwen3-1.7b-unlocked` as a candidate quality target. Treat f32 manifests as reference/parity artifacts; production conversion should use `--tensor-format f16`, which produces packed distribution shards that today's runtime verifies and expands into f32 tensors until direct packed GPU upload/compute lands.

The active local development path now requires `full_vocab_topk_logit_projection` proof for real Qwen production claims. That path scans all `vocabSize` rows in tiles, keeps only bounded top-k rows for decode, records the full row count, selected top-k row count, tile size/count, backend, and purpose, and prevents bounded candidate logits from being mistaken for true model-distribution generation.

The production web app stays target-only unless MTP is explicitly enabled for a paired acceleration run:

```text
VITE_MTP_ENABLED=false
VITE_MTP_DRAFT_MODEL_ID=browser/qwen-prefix-drafter
VITE_MTP_NUM_SPECULATIVE_TOKENS=2
VITE_MTP_DRAFT_LAYER_COUNT=4
VITE_MTP_MIN_ACCEPTANCE_RATE=0
VITE_MTP_DISABLE_WHEN_LATENCY_WORSE=true
```

Set `VITE_MTP_ENABLED=true` only when measuring MTP. With the unlocked backend selected, this registers a Qwen-tokenizer-compatible browser draft profile and advertises speculative verifier batching when `VITE_MTP_DRAFT_MODEL_ID` is `browser/qwen-prefix-drafter` or `browser/ngram-drafter`. Other draft ids fall back target-only because there is no loaded tokenizer-compatible draft backend for them. The prefix drafter is a real browser-owned shallow target-family model path, not an opaque API call; it is still gated by acceptance and paired speed measurements before acceleration is claimed.

The verifier path batches the target side as a single continuation over `[previousToken, ...draftPrefix]`, computes target tokens for the whole draft window, and commits only the accepted input rows to the live KV cache. Browser-native MTP defaults to a 2-token draft window and clamps explicit browser windows to 3 tokens because local concurrency is normally 1. Decode proofs report `verifierStrategy: "batched_continuation"`, `numSpeculativeTokens`, `verifiedTokenCount`, `targetDecodeCalls`, and `committedInputTokens` so benchmarks can distinguish real batched verification from per-token target-only loops. If MTP is disabled, incompatible, or below the configured acceptance threshold, decode proof metadata reports target-only fallback or the disabled reason instead of claiming acceleration. Acceleration is now a separate benchmark claim: set `BROWSER_RUNTIME_BENCH_REQUIRE_MTP_ACCELERATION=true` or `RELEASE_REQUIRE_MTP_ACCELERATION=true` to run a paired target-only comparison and fail unless `BROWSER_RUNTIME_BENCH_MIN_MTP_ACCEPTANCE_RATE` and `BROWSER_RUNTIME_BENCH_MIN_MTP_NET_SPEEDUP` are met.

## Browser KVSwap Persistence

The web worker creates a KVSwap persistence store when the unlocked backend is selected and persistence is enabled:

```text
VITE_KVSWAP_PERSISTENCE_ENABLED=true
VITE_KVSWAP_PERSISTENCE_PREFER_OPFS=true
VITE_KVSWAP_PERSISTENCE_MAX_BLOCKS=512
VITE_KVSWAP_PERSISTENCE_MAX_BYTES=268435456
VITE_KVSWAP_PERSISTENCE_CLEAR_ON_INIT=false
```

The namespace is scoped to tenant, cell, and session so a private/session reset clears persisted KV blocks across model switches. OPFS is preferred for browser-local disk semantics. The first OPFS choice is now the binary sync-handle adapter: it writes `.bin` records with a magic/version header, structured UTF-8/DataView payload fields, payload checksum validation, namespace/id/model/request identity, token ranges, layer, Q/K/V/hidden rows, token ids, compressed key summaries, and summary metadata. When `navigator.locks` is available, binary OPFS serializes namespace access through Web Locks and reports `tabCoordination="web_locks"` plus `lockWaitMs`. When Web Locks are unavailable, the default factory does not claim cross-tab safety; it falls back to async JSON OPFS/IndexedDB unless the caller is the dedicated KV persistence routing worker, which creates the store with `singleWorkerRoute=true` and reports `tabCoordination="single_worker_route"`. If sync access handles are unavailable or denied, the adapter falls back to async JSON OPFS, then IndexedDB, and finally to an in-memory test adapter when neither browser API exists. Hydrate/list validates shape and types, quarantines malformed/corrupt binary or JSON records, and reports unhealthy trace events instead of treating bad persisted data as usable cache. Hydrated blocks are used for decode reuse only when every prompt/runtime identity field and every serialized row range matches exactly; otherwise the runtime falls back to fresh prefill and records normal persist events. Clearing memory calls the KV persistence clear path for the active session, and `VITE_KVSWAP_PERSISTENCE_CLEAR_ON_INIT=true` forces a private-session reset during initialization.

What is real now: binary sync-handle OPFS selection, checksum-backed quarantine, Web Locks coordination when exposed, a dedicated KV persistence routing worker entrypoint for single-owner OPFS access, explicit proof metadata for binary/sync-handle/locks/lock-wait/read/write/coordination, and the existing async JSON OPFS/IndexedDB/memory fallbacks. What remains: wiring every production KV persistence caller through the routing worker when Web Locks are absent; until that lands, the default factory falls back instead of labeling uncoordinated direct access as `single_worker_route`.

For production decode stability, the client persists prefill KV once and keeps token-by-token decode mutations in live browser-owned memory. It does not serialize the full KV cache after every generated token. This preserves prefill reuse while avoiding OPFS/IndexedDB write amplification during long generations.

Predictive KVSwap now has a real low-rank selection path in addition to exact prompt hydration. Persisted blocks carry or reconstruct a low-rank key summary with rank, projection id, layer/head grouping, block id, checksum, quality score, and compressed values derived from key rows. Core KVSwap scores those summaries against the current query summary with bounded deterministic math, selects predicted hot blocks by approximate attention relevance, and emits `lowRankSummaryRank`, `predictedHotBlocks`, `prefetchedBlocks`, `prefetchHitRate`, `prefetchBytes`, `prefetchLatencyMs`, `attentionStallMs`, and `prefetchStrategy`.

In the browser client, exact prompt reuse still wins first and reports `prefetchStrategy="exact_reuse"` when it can skip prefill. When exact reuse misses but persisted summaries exist, the client starts async persistence `load()` calls for predicted hot blocks while fresh prefill runs. The decode proof and KV persistence load events expose the predicted/prefetched blocks and hit or stall fields. This is honest async scheduling: it proves the browser has chosen and loaded likely KV blocks ahead of attention, but it does not claim full GPU/disk overlap or kernel-level partial KV injection until those backend hooks consume the prefetched rows directly.

Focused client tests now cover the browser-reload shape directly: a fresh `UnlockedBrowserTransformerClient` using the same persistence namespace hydrates exact-match prefill KV from the shared browser store, skips fresh prefill, and reports `decodeReuse=true`. KV persistence proof state is reset at the start of each generation so a later non-matching prompt cannot inherit stale reuse events from an earlier exact-match decode.

## Production Manifest Contract

For production, host a JSON manifest at `VITE_UNLOCKED_MODEL_MANIFEST_PATH`. The production v1 manifest is versioned, sharded, integrity-checked, and now carries explicit tensor-storage metadata. `f32-shard` remains the reference/parity lane; model-backed production gates require `f16-shard` plus `tensorStorage.format="f16-packed"` unless an operator explicitly opts out.

```json
{
  "schemaVersion": 1,
  "modelId": "Qwen/Qwen3-0.6B",
  "architecture": "qwen3_decoder_control",
  "vocabSize": 16,
  "hiddenSize": 4,
  "headDim": 4,
  "numAttentionHeads": 2,
  "numKeyValueHeads": 1,
  "maxPositionEmbeddings": 4096,
  "ropeTheta": 1000000,
  "tieWordEmbeddings": true,
  "intermediateSize": 12,
  "rmsNormEps": 0.000001,
  "hiddenActivation": "silu",
  "tensorStorage": {
    "format": "f16-packed",
    "dtype": "f16",
    "shardKind": "f16-shard",
    "byteWidth": 2,
    "productionTarget": "webgpu-packed",
    "runtimeRepresentation": "packed-f16-runtime-lazy-decode",
    "packedRuntimeCompute": false
  },
  "tokenEmbedding": {
    "kind": "f16-shard",
    "uri": "weights-00001.bin",
    "byteOffset": 0,
    "shape": [16, 4],
    "sha256": "<64-character-sha256-hex-digest>"
  },
  "outputProjection": {
    "kind": "f16-shard",
    "uri": "weights-00001.bin",
    "byteOffset": 128,
    "shape": [16, 4],
    "sha256": "<64-character-sha256-hex-digest>"
  },
  "tokenizer": {
    "kind": "vocab",
    "tokens": ["<unk>", "hello", "world", "..."],
    "unknownTokenId": 0
  },
  "finalNorm": {
    "kind": "f16-shard",
    "uri": "weights-00001.bin",
    "byteOffset": 224,
    "shape": [4],
    "sha256": "<64-character-sha256-hex-digest>"
  },
  "layers": [
    {
      "inputLayerNorm": { "kind": "f16-shard", "uri": "weights-00001.bin", "byteOffset": 232, "shape": [4], "sha256": "<64-character-sha256-hex-digest>" },
      "qProj": { "kind": "f16-shard", "uri": "weights-00001.bin", "byteOffset": 256, "shape": [8, 4], "sha256": "<64-character-sha256-hex-digest>" },
      "kProj": { "kind": "f16-shard", "uri": "weights-00001.bin", "byteOffset": 320, "shape": [4, 4], "sha256": "<64-character-sha256-hex-digest>" },
      "vProj": { "kind": "f16-shard", "uri": "weights-00001.bin", "byteOffset": 352, "shape": [4, 4], "sha256": "<64-character-sha256-hex-digest>" },
      "oProj": { "kind": "f16-shard", "uri": "weights-00001.bin", "byteOffset": 384, "shape": [4, 8], "sha256": "<64-character-sha256-hex-digest>" },
      "qNorm": { "kind": "f16-shard", "uri": "weights-00001.bin", "byteOffset": 448, "shape": [4], "sha256": "<64-character-sha256-hex-digest>" },
      "kNorm": { "kind": "f16-shard", "uri": "weights-00001.bin", "byteOffset": 456, "shape": [4], "sha256": "<64-character-sha256-hex-digest>" },
      "postAttentionLayerNorm": { "kind": "f16-shard", "uri": "weights-00001.bin", "byteOffset": 464, "shape": [4], "sha256": "<64-character-sha256-hex-digest>" },
      "mlpGateProj": { "kind": "f16-shard", "uri": "weights-00001.bin", "byteOffset": 472, "shape": [12, 4], "sha256": "<64-character-sha256-hex-digest>" },
      "mlpUpProj": { "kind": "f16-shard", "uri": "weights-00001.bin", "byteOffset": 568, "shape": [12, 4], "sha256": "<64-character-sha256-hex-digest>" },
      "mlpDownProj": { "kind": "f16-shard", "uri": "weights-00001.bin", "byteOffset": 664, "shape": [4, 12], "sha256": "<64-character-sha256-hex-digest>" }
    }
  ]
}
```

Shard `uri` values resolve relative to the manifest URL. `byteOffset` is measured in bytes and must align to the descriptor width: four-byte `f32` for reference shards, two-byte `f16` for packed production shards. Inline numeric matrices are retained only for tests and local experimentation; production readiness requires `VITE_UNLOCKED_MANIFEST_FORMAT=sharded`, and strict model-backed release verification also requires `--require-packed-assets` or `RELEASE_REQUIRE_UNLOCKED_PACKED_ASSETS=true`.

## Offline Converter

Use the local converter when you have accepted/licensed model artifacts already on disk:

```bash
pnpm convert:unlocked -- \
  --input /path/to/local/Qwen3-0.6B \
  --output .artifacts/models/qwen3-0.6b-unlocked \
  --model-id Qwen/Qwen3-0.6B \
  --tensor-format f16
```

The converter reads `config.json`, `tokenizer.json`, and local `.safetensors` files only. It does not contact Hugging Face or any network service. Output is:

```text
manifest.json
manifest.json.sha256
unlocked.env.example
shards/token-embedding.f16.bin
shards/output-projection.f16.bin
shards/layer-0-q-proj.f16.bin
...
```

The current converter maps HF/Qwen tensor names into full Qwen GQA/RoPE manifest geometry:

| Manifest tensor | HF source | Runtime slice |
|---|---|---|
| `tokenEmbedding` | `model.embed_tokens.weight` | full `[vocabSize, hiddenSize]` |
| `outputProjection` | `lm_head.weight`, or tied embeddings when `tie_word_embeddings=true` | full `[vocabSize, hiddenSize]` |
| `layers[i].qProj` | `model.layers.i.self_attn.q_proj.weight` | full `[numAttentionHeads * headDim, hiddenSize]` |
| `layers[i].kProj` | `model.layers.i.self_attn.k_proj.weight` | full `[numKeyValueHeads * headDim, hiddenSize]` |
| `layers[i].vProj` | `model.layers.i.self_attn.v_proj.weight` | full `[numKeyValueHeads * headDim, hiddenSize]` |
| `layers[i].oProj` | `model.layers.i.self_attn.o_proj.weight` | full `[hiddenSize, numAttentionHeads * headDim]` |
| `finalNorm` | `model.norm.weight` | full `[hiddenSize]` |
| `layers[i].inputLayerNorm` | `model.layers.i.input_layernorm.weight` | full `[hiddenSize]` |
| `layers[i].postAttentionLayerNorm` | `model.layers.i.post_attention_layernorm.weight` | full `[hiddenSize]` |
| `layers[i].qNorm` | `model.layers.i.self_attn.q_norm.weight` | full `[headDim]` |
| `layers[i].kNorm` | `model.layers.i.self_attn.k_norm.weight` | full `[headDim]` |
| `layers[i].mlpGateProj` | `model.layers.i.mlp.gate_proj.weight` | full `[intermediateSize, hiddenSize]` |
| `layers[i].mlpUpProj` | `model.layers.i.mlp.up_proj.weight` | full `[intermediateSize, hiddenSize]` |
| `layers[i].mlpDownProj` | `model.layers.i.mlp.down_proj.weight` | full `[hiddenSize, intermediateSize]` |

Input dtypes `F32`, `F16`, and `BF16` are converted into row-major little-endian shards in the requested output format: f32 reference shards by default, or f16 packed distribution shards with `--tensor-format f16`. Every descriptor includes the source tensor, source dtype, source shape, slice metadata, output dtype, and shard SHA-256 for auditability. The manifest also records `numAttentionHeads`, `numKeyValueHeads`, `maxPositionEmbeddings`, `ropeTheta`, `tieWordEmbeddings`, and `conversion.projectionMode="full-qwen-gqa-rope"` so release gates can distinguish full Qwen geometry from earlier first-head conversion artifacts.

This is an unlocked control-runtime conversion with the full Qwen attention geometry represented in the manifest. RMSNorm, Q/K head norms, final norm, residual block flow, SiLU-gated MLP, GQA head counts, RoPE settings, byte-level BPE tokenizer metadata, and chat-template metadata are represented when present. The browser runtime consumes the full Q/K/V/O projections, applies RoPE, expands grouped KV heads for SSA, appends decode KV rows through browser-owned state, escapes all manifest special tokens inside user-supplied content, and executes decode through KVSwap/TSP callbacks. Dense Q/K/V/O projection has a reusable WebGPU/CPU matmul boundary for `[tokens, hidden] x [out, hidden] -> [tokens, out]`, and live WebGPU runtimes can reuse explicitly keyed stable descriptor-backed projection buffers plus per-device compute pipelines for the lifetime of the device/runtime. The primitive layer supports a resident GPU chain across RMSNorm, dense projection, sparse attention, output projection, residual add, batched MLP, residual add, and final top-k logit projection with `outputResident: true`, `readback: false`, and `vectorResident: true` proof metadata where applicable. The real decode path now uses resident O projection, residual, post-attention RMSNorm, MLP, and residual before materializing the layer hidden row for the existing KV/SSA state. That reuse is not persistent storage, does not survive reloads, and stable matrix-buffer reuse is disabled for mutable plain arrays. Packed-head sparse decode attention and prefill causal attention now route through the SSA sparse-attention kernel boundary per attention head, using browser WebGPU in strict production lanes; CPU-reference execution is available only as explicit fixture/dev opt-out. Decode traces include the actual query-block route plus per-head backend proof, while prefill proof metadata is attached to the KV cache handle. Empty or fully invalid sparse selections produce zero vectors with the query/head dimension on CPU fallback only in those explicit dev lanes, matching the WebGPU path. Prefill MLP now uses one batched WebGPU/CPU boundary per layer over all prompt rows, and decode MLP uses the single-token boundary for Qwen SiLU-gated MLPs and non-gated GeLU MLPs; proofs include token/row count, backend, projection-cache hits, and pipeline-cache hits. Full-vocab decode logit selection uses tiled dense-matvec top-k with WebGPU dispatch in strict lanes; proofs preserve the winning token IDs, full vocabulary row count, selected top-k rows, scanned rows, tile rows, tile count, backend, pipeline-cache hit metadata, and `full_vocab_topk_logit_projection` purpose. MTP target verification uses a batched continuation over the draft window, preserving the same SSA/KV/TSP kernel boundaries, recording decode O-projection backend proof for the batched matmul, and committing only accepted input rows to the live KV cache. `VITE_UNLOCKED_RUNTIME_PROFILE` now owns artificial cap defaults: `preview` is the explicit local browser-proof profile, `ci` is the smallest automated profile, `balanced` raises preview budgets, and `full` removes prompt/layer/generation/logit caps unless an explicit `VITE_UNLOCKED_MAX_*` override is set. `VITE_CHAT_MAX_RUNTIME_PROMPT_TOKENS`, `VITE_CHAT_MAX_RUNTIME_LAYERS`, `VITE_CHAT_LOGIT_TOP_K`, `VITE_CHAT_LOGIT_TILE_ROWS`, `VITE_CHAT_LOGIT_CANDIDATE_LIMIT`, and `VITE_CHAT_MAX_GENERATION_TOKENS` are separate app-level interactive budgets used only by the chat UI. The quality-preserving browser default keeps full layer execution, full-vocab top-k logits, no 512-token prompt trim, a manifest-max 40,960-token generation budget, byte-level BPE streaming UTF-8 decode across token boundaries, and full prompt-block visibility up to the Qwen 40,960-token context window; beyond that window, the SSA sparse window keeps pinned and recent blocks explicit instead of silently claiming infinite context. Lower layer budgets, explicit candidate logits, or forced sparse decode windows are faster but can materially degrade live Qwen response quality and cannot satisfy strict production gates. The live browser path surfaces the active chat budget in the runtime panel while strict verification can still require an uncapped `full` runtime profile. The unlocked client does not apply fallback generation or real-manifest candidate caps on its own; callers must pass resolved profile caps or explicit request options. Verification artifacts record the active profile, resolved caps, active cap booleans, prompt token budget, manifest and effective layer counts, generation token budget used, logit projection purpose/rows, strict full-profile requirement, a normalized run-is-capped boolean, and `webGpuCoverage` with MLP layer counts by backend, logit projection backend, prefill and decode Q/K/V/O projection backend counts, prefill/decode attention backend counts, packed-head backend counts, and `cpuFallbackUsed`. The client renders explicit `{{#messages}}...{{/messages}}` mini-templates and known Qwen HF chat templates through a Qwen-compatible formatter; it preserves arbitrary HF/Jinja templates in the manifest for auditability, but does not evaluate arbitrary Jinja in the browser. The remaining production expansion is performance-focused: resident RoPE/head-normalization/packed-attention assembly for the real Q/K/V path, fused packed-head attention, scratch/readback-buffer reuse, and broader Qwen parity evidence beyond fixtures.

## WebGPU Coverage Gates

Default real-Qwen verification is split by host. `pnpm verify:unlocked` is the Node manifest/parity/tensor-control gate and does not claim authoritative browser GPU proof by default. The browser-preview benchmark is the production WebGPU gate: strict model lanes set `BROWSER_RUNTIME_BENCH_PREVIEW_REQUIRE_STRICT_WEBGPU=true` and pass `VITE_REQUIRE_WEBGPU_KERNELS=true` to the browser route, so Chrome/Edge proof fails closed if any MLP, dense matvec/matmul projection, sparse attention, packed-head attention, or logit projection proof reports `cpu_reference`, `mixed`, missing, or `unknown`. CPU-reference execution is still available for explicit fixture/dev or Node verifier work; those artifacts remain marked with `cpuFallbackUsed=true` and cannot satisfy production browser WebGPU gates.

Strict lanes can fail on fallback with:

```bash
pnpm verify:unlocked -- --require-webgpu-mlp --require-webgpu-logits --require-webgpu-attention
```

Add `--require-webgpu-projection` when dense prefill/decode projection fallback must fail too. The equivalent Node env vars are `RELEASE_REQUIRE_UNLOCKED_WEBGPU_MLP=true`, `RELEASE_REQUIRE_UNLOCKED_WEBGPU_LOGITS=true`, `RELEASE_REQUIRE_UNLOCKED_WEBGPU_ATTENTION=true`, and `RELEASE_REQUIRE_UNLOCKED_WEBGPU_PROJECTION=true`; `RELEASE_REQUIRE_UNLOCKED_NODE_STRICT_WEBGPU=true` expands to all four gates for hosts that expose a real Node WebGPU device. These gates also construct the unlocked client with `requireWebGpu: true`, so a strict lane fails before proof collection if the current runtime cannot create/use WebGPU or if a kernel tries to fall back to CPU. Every recorded proof in that family must still be `webgpu`; `cpu_reference`, `mixed`, missing, or unknown proofs fail the strict gate. For production decisions, prefer the browser-preview strict route over Node proof.

## Browser Runtime Benchmarks

Run the repeatable browser-runtime benchmark with:

```bash
pnpm bench:browser-runtime
```

Without `VITE_UNLOCKED_MODEL_MANIFEST_PATH`, the command generates a deterministic sharded fixture manifest so CI can exercise the same unlocked client, tensor-control decode path, MTP verifier path, and CPU/WebGPU proof accounting without shipping real weights. For local model-backed runs, point it at the real manifest:

```bash
pnpm bench:browser-runtime -- \
  --manifest-path apps/web/public/models/qwen3-0.6b-unlocked/manifest.json \
  --manifest-sha256 <64-character-sha256-hex-digest> \
  --runtime-profile balanced \
  --backend-preference webgpu
```

Artifacts are written to `.artifacts/evals/browser-runtime-bench/<timestamp>/results.json`, `.artifacts/evals/browser-runtime-bench/<timestamp>/summary.md`, and `.artifacts/evals/browser-runtime-bench-latest.json`. Each result records init/load time, prefill time, time to first generated token, decode latency, tokens/sec, memory mode, active profile/caps, backend preference, TSP/KV proof metadata, normalized CPU/WebGPU coverage, MTP mode/acceptance, and optional paired MTP acceleration metrics.

The default command runs the Node-hosted unlocked client path and records browser preview as explicitly skipped:

```json
{ "browserPreview": { "mode": "skipped", "requested": false, "reason": "not_requested" } }
```

Real browser WebGPU CI parity is a separate lane, not a CPU or Dawn/Lavapipe substitute:

```bash
pnpm ci:headless-webgpu
```

This launches headless Chromium with WebGPU flags and runs a tiny WGSL compute shader that doubles `[1, 2, 3, 4]` into `[2, 4, 6, 8]`. A passing artifact requires `navigator.gpu`, an identifiable non-software adapter/device, command submission, readback, and exact output match. SwiftShader, Lavapipe, llvmpipe, fallback adapters, unknown adapter identity, or mismatched compute output produce `status: "failed"` and `passed: false`; an unavailable browser WebGPU surface produces `status: "skipped"` and `passed: false` when `HEADLESS_WEBGPU_CI_REQUIRED` is unset. Use `HEADLESS_WEBGPU_CI_REQUIRED=true` plus `HEADLESS_WEBGPU_CI_BROWSER_CHANNEL=chrome` or `HEADLESS_WEBGPU_CI_BROWSER_EXECUTABLE=/path/to/chrome` on a GPU-backed runner when this lane must block. Dawn/Lavapipe is intentionally not accepted in this real-only lane.

The web app also exposes a browser-executed preview proof route:

```bash
pnpm dev:web
```

Then open:

```text
http://127.0.0.1:5173/__bench/browser-runtime?profile=full&prompt=Explain%20persistent%20runtime%20intelligence.&generationTokens=24&minGeneratedTokens=16
```

For multiple prompts, pass repeated `prompt` params or a pipe-delimited `prompts` value:

```text
http://127.0.0.1:5173/__bench/browser-runtime?profile=preview&prompt=What%20is%20Earth%3F&prompt=Explain%20runtime%20memory.
```

The route initializes the unlocked browser client inside the page and renders JSON in both `pre[data-browser-preview-benchmark-json]` and `script#browser-preview-benchmark-payload`. The payload includes the response, coherence status, expected-substring matches, runtime trace proof, predictive counts, WebGPU availability plus CPU-fallback evidence, MTP mode/acceptance, draft model/source, latency-disable policy, speculative window size, verified token count, target decode call count, verifier strategy, KV persistence event counts, KV prefetch strategy (`none`, `exact_reuse`, `predictive_prefetch`, or `miss_stall`), low-rank summary rank, predicted/prefetched block counts, prefetch hit rate, prefetch bytes, prefetch latency, attention stall time, and timing summary fields. Timing uses total generation wall time for throughput so a speculative batch that streams all accepted tokens in its first visible chunk cannot report impossible speed. Production config with `VITE_REQUIRE_WEBGPU_KERNELS=true` makes the browser route request all strict WebGPU gates by default; `backendPreference=webgpu`, `webGpuGates=mlp,logits,attention,projection`, and `strictWebGpu=true` remain useful explicit proof controls. Query params `generationTokens`, `minGeneratedTokens`, `expected=<substring>`, `expectedSubstrings=<prompt1-substrings|prompt2-substrings>`, `maxRuntimePromptTokens`, `maxRuntimeLayers`, `qwenThinkingMode=enabled`, and `logitCandidateLimit` override the selected profile for proof runs, so production-style arbitrary-output checks can run with `profile=full&generationTokens=24&minGeneratedTokens=16&expected=Salt%20Lake&qwenThinkingMode=enabled` instead of the intentionally one-token `preview` profile. The payload gate recomputes visible-response quality from the rendered text, so `[unlocked:ssa-kv-tsp]`, proof markers, whitespace, punctuation, stop fragments, or one-word fragments cannot pass even if runtime trace, strict WebGPU proof, and KV events exist. Add `mtp=false` when diagnosing whether draft verification is damaging output; the resulting proof reports `target_only` instead of claiming acceleration. Add `timeoutMs=<milliseconds>` so strict browser lanes return a failed payload instead of spinning indefinitely when the full path is too slow. Add `requireKvReuse=true` to make the route spin up a fresh client in the same browser page and require a same-namespace persisted KV reuse event; set `kvNamespace=<stable-id>` when proving reuse across reloads or browser sessions.

The Node benchmark can still consume a JSON preview endpoint when one is available:

```bash
pnpm bench:browser-runtime -- \
  --browser-preview-url http://localhost:5173/__bench/browser-runtime
```

or set `BROWSER_RUNTIME_BENCH_PREVIEW_URL`. That consumer calls the URL with query params `profile`, repeated `prompt=<text>` values, repeated `expectedJson=<json-string-array>` values aligned by prompt index, optional `backendPreference`, optional `webGpuGates`, `generationTokens`, `timeoutMs`, `minGeneratedTokens`, and `requireKvReuse` when those benchmark controls are configured. This avoids delimiter coupling for prompts or expected substrings containing `|` or `,`; the route still accepts older `prompts`, `expected`, and `expectedSubstrings` query params for manual/backward-compatible runs. It accepts raw JSON, a `file://` JSON artifact captured from a real browser run, HTML that already contains `script#browser-preview-benchmark-payload`, or a live SPA route that needs client execution. When static fetch sees the SPA shell, the benchmark launches `playwright-core` against the same URL, waits for `script#browser-preview-benchmark-payload` or `pre[data-browser-preview-benchmark-json]`, and validates that browser-produced payload. Root app URLs such as `http://127.0.0.1:5173/` are normalized to `/__bench/browser-runtime`; set `BROWSER_RUNTIME_BENCH_BROWSER_CHANNEL` or `BROWSER_RUNTIME_BENCH_BROWSER_EXECUTABLE` if Chrome is installed in a non-default place. A script-friendly JSON endpoint should return the same shape, at minimum:

```json
{
  "passed": true,
  "summary": {
    "meanInitLoadMs": 20,
    "meanPrefillMs": 8,
    "meanTimeToFirstTokenMs": 11,
    "meanDecodeLatencyMs": 6,
    "meanTokensPerSecond": 166,
    "mtpMode": "target_only",
    "mtpMaxSpeculativeTokens": 2,
    "mtpMeanSpeculativeTokens": 2,
    "mtpVerifiedTokenCount": 2,
    "mtpTargetDecodeCalls": 1,
    "mtpVerifierStrategy": "batched_continuation",
    "coherentResponseCount": 1,
    "runtimeTraceCount": 1,
    "predictiveSelectedBlockCount": 2,
    "predictiveKvPagingEventCount": 1,
    "webGpuAvailable": true,
    "cpuFallbackUsed": false,
    "noCpuFallback": true,
    "kvPersistenceEventCount": 2,
    "kvPrefetchStrategy": "predictive_prefetch",
    "kvExactReuseRunCount": 0,
    "kvPredictivePrefetchRunCount": 1,
    "kvMissStallRunCount": 0,
    "kvLowRankSummaryRank": 4,
    "kvPredictedHotBlockCount": 2,
    "kvPrefetchedBlockCount": 1,
    "kvPrefetchHitRate": 0.5,
    "kvPrefetchBytes": 2048,
    "kvPrefetchLatencyMs": 3.5,
    "kvAttentionStallMs": 0
  },
  "runs": []
}
```

Strict real-Qwen release mode now sets `RELEASE_REQUIRE_BROWSER_PREVIEW_PROOF=true` and `BROWSER_RUNTIME_BENCH_REQUIRE_BROWSER_PREVIEW=true` by default when a configured or locally installed manifest/SHA is present. It also defaults the benchmark to arbitrary Qwen prompts, expected substrings (`Salt Lake` for the Utah prompt and `Earth` for the Earth prompt), `VITE_QWEN_THINKING_MODE=disabled`, a bounded `BROWSER_RUNTIME_BENCH_MAX_GENERATION_TOKENS=16` proof budget, `BROWSER_RUNTIME_BENCH_PREVIEW_MIN_GENERATED_TOKENS=8`, `BROWSER_RUNTIME_BENCH_PREVIEW_REQUIRE_KV_REUSE=true`, and `BROWSER_RUNTIME_BENCH_PREVIEW_REQUIRE_STRICT_WEBGPU=true`. The benchmark script applies the same defaults when a configured manifest, required browser preview, and strict preview WebGPU gates are requested directly, so `--browser-preview-url` does not fall back to `alpha beta`, one generated token, or KV reuse as an optional report. Expected substrings are enforced in the direct Node benchmark artifact and again in browser preview, so a fluent false answer fails even when the tensor/proof path is otherwise real. The output filter strips `<think>...</think>` from visible assistant text while preserving the real unlocked decode path when thinking mode is explicitly enabled. If no `BROWSER_RUNTIME_BENCH_PREVIEW_URL` or `--browser-preview-url` is supplied in that lane, the benchmark fails early and records the missing browser proof in the artifact instead of silently skipping it.

Requested preview failures or malformed preview responses are recorded as `browserPreview.mode="failed"` and fail that benchmark invocation. Unrequested preview remains a non-blocking explicit skip for CI. If decode latency is not measurable, `tokensPerSecond` is recorded as `null` instead of inventing a throughput value; strict throughput thresholds fail on unavailable throughput.

Profiles are explicit. `ci` and `preview` are intentionally capped proof profiles, `balanced` raises budgets for realistic local checks, and `full` removes artificial runtime caps unless a `VITE_UNLOCKED_MAX_*` override is set. The benchmark does not pretend `full` is fast on every device; it records the actual numbers and leaves speed thresholds non-blocking by default.

Optional threshold env vars:

```text
BROWSER_RUNTIME_BENCH_MAX_INIT_MS=5000
BROWSER_RUNTIME_BENCH_MAX_TTFT_MS=2000
BROWSER_RUNTIME_BENCH_MIN_TOKENS_PER_SEC=1
BROWSER_RUNTIME_BENCH_STRICT=true
```

When `BROWSER_RUNTIME_BENCH_STRICT` is unset, threshold misses are emitted as artifact fields but do not fail the command or normal release gates. Set `BROWSER_RUNTIME_BENCH_STRICT=true` for a device-specific lane that should fail on performance regressions. Set `BROWSER_RUNTIME_BENCH_REQUIRE_MTP_ACCELERATION=true` to run each prompt twice, once with draft verification and once target-only, then fail unless `BROWSER_RUNTIME_BENCH_MIN_MTP_ACCEPTANCE_RATE` and `BROWSER_RUNTIME_BENCH_MIN_MTP_NET_SPEEDUP` are cleared. Local `release:gate` now promotes a configured or installed Qwen manifest/SHA into real-Qwen mode by default: configured manifest, manifest SHA, full runtime profile, Qwen math/parity, KV decode reuse, `browser-vector` memory, WebGPU backend preference, and a completed strict browser-preview proof are all expected. Set `RELEASE_ALLOW_FIXTURE_GATE=true` for an explicit fixture/dev gate; CI without an installed model still stays on the fixture lane. The release gate requires the configured non-fixture benchmark. Add `RELEASE_REQUIRE_MTP_ACCELERATION=true` when speculative decoding must prove a real speedup; otherwise MTP remains target-only/off for the production speed claim.

Strict WebGPU can be made blocking in the Node benchmark, but the authoritative production proof is the real browser-preview route. Production browser config with `VITE_REQUIRE_WEBGPU_KERNELS=true` and `BROWSER_RUNTIME_BENCH_PREVIEW_REQUIRE_STRICT_WEBGPU=true` requires WebGPU proofs for MLP, logits, attention, and projection together in the Chrome/Edge client path. Use `--require-webgpu-mlp`, `--require-webgpu-logits`, `--require-webgpu-attention`, `--require-webgpu-projection`, or `RELEASE_REQUIRE_UNLOCKED_NODE_STRICT_WEBGPU=true` only for explicit Node hosts that expose a real WebGPU device. In strict benchmark mode the client receives `requireWebGpu: true`, so a machine without WebGPU fails before CPU-reference decode instead of producing misleading throughput. Browser preview strict WebGPU also inherits `VITE_REQUIRE_WEBGPU_KERNELS=true`; `BROWSER_RUNTIME_BENCH_PREVIEW_REQUIRE_STRICT_WEBGPU=true` remains available for standalone preview proof lanes. Release lanes can keep using `RELEASE_REQUIRE_UNLOCKED_STRICT_WEBGPU=true`; the release gate passes that intent through to `bench:browser-runtime` and the benchmark artifact records `webGpuGate`, `strictWebGpuRequired`, `strictWebGpuPassed`, and failure counts. If these gates fail, the run is not production-WebGPU ready even if correctness and ordinary performance metrics were recorded.

## Environment

Local tensor-control proof:

```bash
VITE_LLM_BACKEND=unlocked-browser-transformer
VITE_DEFAULT_MODEL=Qwen/Qwen3-0.6B
VITE_REQUIRE_UNLOCKED_RUNTIME=true
VITE_UNLOCKED_ALLOW_FIXTURE=true
```

Production:

```bash
VITE_LLM_BACKEND=unlocked-browser-transformer
VITE_DEFAULT_MODEL=Qwen/Qwen3-0.6B
VITE_REQUIRE_UNLOCKED_RUNTIME=true
VITE_UNLOCKED_MODEL_MANIFEST_PATH=/models/qwen3-0.6b-unlocked/manifest.json
VITE_UNLOCKED_MODEL_MANIFEST_SHA256=<64-character-sha256-hex-digest>
VITE_UNLOCKED_MANIFEST_FORMAT=sharded
VITE_UNLOCKED_ALLOW_FIXTURE=false
VITE_UNLOCKED_BACKEND_PREFERENCE=webgpu
VITE_REQUIRE_WEBGPU_KERNELS=true
VITE_UNLOCKED_RUNTIME_PROFILE=full
VITE_MTP_ENABLED=false
VITE_MTP_DRAFT_MODEL_ID=browser/qwen-prefix-drafter
VITE_MTP_NUM_SPECULATIVE_TOKENS=2
VITE_MTP_DRAFT_LAYER_COUNT=4
VITE_MTP_MIN_ACCEPTANCE_RATE=0
VITE_KVSWAP_PERSISTENCE_ENABLED=true
VITE_KVSWAP_PERSISTENCE_PREFER_OPFS=true
VITE_CHAT_MAX_RUNTIME_PROMPT_TOKENS=full
VITE_CHAT_MAX_RUNTIME_LAYERS=full
VITE_CHAT_LOGIT_CANDIDATE_LIMIT=full
VITE_CHAT_LOGIT_TOP_K=64
VITE_CHAT_LOGIT_TILE_ROWS=4096
VITE_CHAT_MAX_GENERATION_TOKENS=full
VITE_QWEN_THINKING_MODE=disabled
VITE_AGENT_MAX_PROMPT_TOKENS=40960
```

If that manifest path points at local `/models/...` files and you want `vite preview` or another static app bundle to carry the converted shards, set `VITE_BUNDLE_UNLOCKED_MODEL=true` for that build. Keep it false for open-source and hosted releases that serve the large shard directory from external object storage/CDN.

## Acceptance Gates

- Core tests prove prefill creates Q/K/V handles and decode uses KV paging plus TSP callbacks.
- Web tests prove the browser client streams through `unlocked-browser-transformer`.
- Production readiness blocks production without a converted manifest and without a completed strict browser-preview WebGPU proof for the unlocked full-control lane.
- `pnpm verify:unlocked` validates a configured manifest or a generated sharded fixture, initializes the unlocked backend, honors `VITE_UNLOCKED_BACKEND_PREFERENCE` or `--backend-preference`, applies the full local runtime profile by default (`ci` in CI unless overridden), keeps MTP disabled unless explicitly enabled, and requires a tensor-control decode proof. Verification artifacts record the requested backend preference, active runtime profile, resolved caps, active cap booleans, strict full-profile requirement state, and normalized WebGPU coverage, so CI CPU fallback is visible when WebGPU is unavailable.
- `pnpm verify:unlocked -- --require-configured --require-manifest-sha256 --require-sharded` validates the model-backed production lane without falling back to the generated fixture.
- Add `--require-full-profile` or `RELEASE_REQUIRE_UNLOCKED_FULL_PROFILE=true` when the release must fail if preview/balanced/ci caps, or explicit cap overrides, are still active.
- Add `--require-qwen-math` or `RELEASE_REQUIRE_UNLOCKED_QWEN_MATH=true` when the release must include final norm, layer RMSNorms, q/k norms, and gated MLP tensors.
- Add `--require-qwen-parity` or `RELEASE_REQUIRE_UNLOCKED_QWEN_PARITY=true` when the release must include full Qwen attention metadata, RoPE metadata, and Q/K/V/O shard shapes consistent with `numAttentionHeads`, `numKeyValueHeads`, `headDim`, and `hiddenSize`.
- Add `--require-webgpu-mlp`, `--require-webgpu-logits`, `--require-webgpu-attention`, or `--require-webgpu-projection`, or the matching `RELEASE_REQUIRE_UNLOCKED_WEBGPU_*` env vars, when a release must fail instead of accepting CPU-reference fallback for those kernel families.
- `pnpm bench:browser-runtime` records browser-runtime performance, MTP acceptance, optional paired MTP acceleration, profile/caps, memory mode, and CPU/WebGPU coverage in release-gate summaries. Benchmark thresholds are advisory unless `BROWSER_RUNTIME_BENCH_STRICT=true`; paired acceleration is required only when `BROWSER_RUNTIME_BENCH_REQUIRE_MTP_ACCELERATION=true` or `RELEASE_REQUIRE_MTP_ACCELERATION=true`.
- `pnpm ci:headless-webgpu` records a real headless Chromium WebGPU compute parity artifact at `.artifacts/evals/headless-webgpu-ci-parity-latest.json`; only a real WebGPU adapter/device and matching compute output can pass.
- Browser preview must initialize the unlocked backend, produce non-degenerate visible output, preserve the unlocked runtime trace proof, and pass requested WebGPU/KV reuse gates.
