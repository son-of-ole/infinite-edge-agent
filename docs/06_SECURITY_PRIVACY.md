# Security and Privacy

## Privacy model

Default product stance:

- LLM inference is local.
- Embeddings are local.
- Memory is local.
- No analytics by default.
- No cloud sync by default.

## Threats

| Threat | Mitigation |
|---|---|
| Prompt injection stored in memory | Mark memory as untrusted context; system prompt must not obey instructions from memory chunks. |
| Secrets stored accidentally | Redact common API tokens before memory embedding and tag redacted chunks. |
| Browser storage exfiltration | Use normal web security controls, CSP, dependency review, and optional local-only deployment. |
| Malicious model artifacts | Use integrity verification and pinned model sources. |
| Sidecar exposed on LAN | Bind to `127.0.0.1` by default and avoid wildcard hosts. |
| Cross-origin requests to sidecar | Restrict CORS in production. |

## Production hardening checklist

- Add Content Security Policy.
- Pin dependency versions before release.
- [x] Add model artifact integrity checks.
- [x] Add user-visible memory delete/export/import.
- [x] Add secret detector before memory write.
- Add broader PII review before memory write.
- Bind sidecar to loopback only.
- Add sidecar auth token for desktop builds.
- [x] Add encrypted-at-rest option for LanceDB sidecar payload fields.
- Add dependency vulnerability scans in CI.

## Model artifact integrity

Unlocked Qwen browser deployments should pin the exact converted manifest and shard set:

```bash
VITE_UNLOCKED_MODEL_MANIFEST_PATH=/models/qwen3-0.6b-unlocked/manifest.json
VITE_UNLOCKED_MODEL_MANIFEST_SHA256=<64-character-sha256-hex-digest>
VITE_UNLOCKED_MANIFEST_FORMAT=sharded
VITE_UNLOCKED_ALLOW_FIXTURE=false
VITE_UNLOCKED_BACKEND_PREFERENCE=webgpu
```

Run `pnpm verify:unlocked -- --require-configured --require-manifest-sha256 --require-sharded` in CI/operator workflows for model-backed releases. Static open-source builds must not publish licensed weights by default. Hosted production should serve the manifest and shards with stable cache headers and update the manifest SHA only when intentionally replacing the model.

## Memory encryption

Set `MEMORY_ENCRYPTION_KEY` on the memory sidecar to encrypt stored memory text, runtime JSON, metadata JSON, and GAC record payloads with AES-256-GCM before they are written to LanceDB tables. Existing plaintext rows remain readable for migration compatibility, but newly written rows use the `enc:v1:` payload envelope.

Keep the key outside `VITE_*` browser variables. Rotating the key requires exporting/decrypting memory with the old key and re-importing with the new key.
