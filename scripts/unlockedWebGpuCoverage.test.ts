import { describe, expect, it } from "vitest";
import {
  assertUnlockedWebGpuCoverageGates,
  readStrictUnlockedWebGpuGatesFromEnv,
  summarizeUnlockedWebGpuCoverage,
} from "./unlockedWebGpuCoverage";

describe("unlocked WebGPU coverage accounting", () => {
  it("normalizes CPU-reference coverage across MLP, logits, prefill projections, and attention", () => {
    const summary = summarizeUnlockedWebGpuCoverage({
      mlpKernelBackends: [
        { layerIndex: 0, backend: "cpu_reference", activationKind: "silu_gated" },
        { layerIndex: 1, backend: "webgpu", activationKind: "silu_gated" },
      ],
      prefillMlpKernelBackends: [
        { layerIndex: 2, backend: "mixed", activationKind: "silu_gated", rowCount: 4 },
      ],
      logitProjectionBackend: "cpu_reference",
      logitProjectionPurpose: "candidate_logit_projection",
      logitProjectionSelectedRows: 32,
      logitProjectionFullRows: 151936,
      prefillProjectionBackends: [
        {
          layerIndex: 0,
          qProjection: "webgpu",
          kProjection: "cpu_reference",
          vProjection: "webgpu",
          oProjection: "cpu_reference",
        },
      ],
      decodeProjectionBackends: [
        { layerIndex: 0, oProjection: "cpu_reference", projectionKind: "matmul", tokens: 2 },
      ],
      prefillAttentionBackends: [
        { layerIndex: 0, attentionBackend: "mixed", packedHeadBackends: ["webgpu", "cpu_reference"], packedHeadCount: 2 },
      ],
      decodeAttentionBackends: [
        { layerIndex: 0, attentionBackend: "cpu_reference", packedHeadBackends: ["cpu_reference"], packedHeadCount: 1 },
      ],
    });

    expect(summary).toMatchObject({
      mlpLayersByBackend: { webgpu: 1, cpu_reference: 1, mixed: 1, unknown: 0 },
      logitProjection: {
        backend: "cpu_reference",
        purpose: "candidate_logit_projection",
        selectedRows: 32,
        fullRows: 151936,
      },
      prefillProjectionBackends: {
        qProjection: { webgpu: 1, cpu_reference: 0, mixed: 0, unknown: 0 },
        kProjection: { webgpu: 0, cpu_reference: 1, mixed: 0, unknown: 0 },
        vProjection: { webgpu: 1, cpu_reference: 0, mixed: 0, unknown: 0 },
        oProjection: { webgpu: 0, cpu_reference: 1, mixed: 0, unknown: 0 },
      },
      decodeProjectionBackends: {
        oProjection: { webgpu: 0, cpu_reference: 1, mixed: 0, unknown: 0 },
      },
      attentionBackends: {
        prefill: { webgpu: 0, cpu_reference: 0, mixed: 1, unknown: 0 },
        decode: { webgpu: 0, cpu_reference: 1, mixed: 0, unknown: 0 },
        packedHeads: { webgpu: 1, cpu_reference: 2, mixed: 0, unknown: 0 },
        incompletePackedHeadProofs: 0,
      },
      cpuFallbackUsed: true,
    });
  });

  it("fails strict WebGPU gates when proofs still use CPU-reference kernels", () => {
    const summary = summarizeUnlockedWebGpuCoverage({
      mlpKernelBackends: [{ layerIndex: 0, backend: "cpu_reference", activationKind: "gelu" }],
      logitProjectionBackend: "cpu_reference",
      prefillProjectionBackends: [
        { layerIndex: 0, qProjection: "cpu_reference", kProjection: "cpu_reference", vProjection: "cpu_reference", oProjection: "cpu_reference" },
      ],
      decodeProjectionBackends: [
        { layerIndex: 0, qProjection: "cpu_reference", kProjection: "cpu_reference", vProjection: "cpu_reference", oProjection: "cpu_reference", projectionKind: "matvec", selectedRows: 4 },
      ],
      prefillAttentionBackends: [
        { layerIndex: 0, attentionBackend: "cpu_reference", packedHeadBackends: ["cpu_reference"], packedHeadCount: 1 },
      ],
      decodeAttentionBackends: [
        { layerIndex: 0, attentionBackend: "cpu_reference", packedHeadBackends: ["cpu_reference"], packedHeadCount: 1 },
      ],
    });

    expect(() => assertUnlockedWebGpuCoverageGates(summary, ["mlp"])).toThrow(/MLP layers are not fully WebGPU-backed/);
    expect(() => assertUnlockedWebGpuCoverageGates(summary, ["logits"])).toThrow(/logit projection backend is cpu_reference/);
    expect(() => assertUnlockedWebGpuCoverageGates(summary, ["projection"])).toThrow(/dense projection kernels are not fully WebGPU-backed/);
    expect(() => assertUnlockedWebGpuCoverageGates(summary, ["attention"])).toThrow(/attention kernels are not fully WebGPU-backed/);
  });

  it("fails strict projection when prefill Q/K/V/O proof is absent even with decode O on WebGPU", () => {
    const summary = summarizeUnlockedWebGpuCoverage({
      decodeProjectionBackends: [
        { layerIndex: 0, oProjection: "webgpu", projectionKind: "matmul", tokens: 1 },
      ],
    });

    expect(() => assertUnlockedWebGpuCoverageGates(summary, ["projection"])).toThrow(/prefill Q\/K\/V\/O projection proof is missing/);
  });

  it("fails strict projection when decode Q/K/V proof is absent even with decode O on WebGPU", () => {
    const summary = summarizeUnlockedWebGpuCoverage({
      expectedLayerCount: 1,
      prefillProjectionBackends: [
        { layerIndex: 0, qProjection: "webgpu", kProjection: "webgpu", vProjection: "webgpu", oProjection: "webgpu" },
      ],
      decodeProjectionBackends: [
        { layerIndex: 0, oProjection: "webgpu", projectionKind: "matmul", tokens: 1 },
      ],
    });

    expect(() => assertUnlockedWebGpuCoverageGates(summary, ["projection"])).toThrow(/decode Q\/K\/V\/O/);
  });

  it("fails strict MLP when either prefill or decode proof is absent", () => {
    const decodeOnly = summarizeUnlockedWebGpuCoverage({
      mlpKernelBackends: [{ layerIndex: 0, backend: "webgpu", activationKind: "silu_gated" }],
    });
    const prefillOnly = summarizeUnlockedWebGpuCoverage({
      prefillMlpKernelBackends: [{ layerIndex: 0, backend: "webgpu", activationKind: "silu_gated", rowCount: 4 }],
    });

    expect(() => assertUnlockedWebGpuCoverageGates(decodeOnly, ["mlp"])).toThrow(/prefill MLP proof is missing/);
    expect(() => assertUnlockedWebGpuCoverageGates(prefillOnly, ["mlp"])).toThrow(/decode MLP proof is missing/);
  });

  it("fails strict MLP when expected layer coverage is partial even if reported layers are WebGPU", () => {
    const summary = summarizeUnlockedWebGpuCoverage({
      expectedLayerCount: 2,
      mlpKernelBackends: [{ layerIndex: 0, backend: "webgpu", activationKind: "silu_gated" }],
      prefillMlpKernelBackends: [{ layerIndex: 0, backend: "webgpu", activationKind: "silu_gated", rowCount: 4 }],
    });

    expect(() => assertUnlockedWebGpuCoverageGates(summary, ["mlp"])).toThrow(/MLP layer coverage is incomplete/);
  });

  it("fails strict logits when WebGPU proof is only candidate projection coverage", () => {
    const summary = summarizeUnlockedWebGpuCoverage({
      logitProjectionBackend: "webgpu",
      logitProjectionPurpose: "candidate_logit_projection",
      logitProjectionSelectedRows: 32,
      logitProjectionFullRows: 151936,
    });

    expect(() => assertUnlockedWebGpuCoverageGates(summary, ["logits"])).toThrow(/candidate_logit_projection/);
  });

  it("fails strict logits when WebGPU full-vocab row coverage is missing or invalid", () => {
    const missingRows = summarizeUnlockedWebGpuCoverage({
      logitProjectionBackend: "webgpu",
      logitProjectionPurpose: "full_vocab_topk_logit_projection",
    });
    const zeroSelectedRows = summarizeUnlockedWebGpuCoverage({
      logitProjectionBackend: "webgpu",
      logitProjectionPurpose: "full_vocab_topk_logit_projection",
      logitProjectionSelectedRows: 0,
      logitProjectionFullRows: 151936,
    });
    const zeroFullRows = summarizeUnlockedWebGpuCoverage({
      logitProjectionBackend: "webgpu",
      logitProjectionPurpose: "full_vocab_topk_logit_projection",
      logitProjectionSelectedRows: 32,
      logitProjectionFullRows: 0,
    });

    expect(() => assertUnlockedWebGpuCoverageGates(missingRows, ["logits"])).toThrow(/row coverage/);
    expect(() => assertUnlockedWebGpuCoverageGates(zeroSelectedRows, ["logits"])).toThrow(/row coverage/);
    expect(() => assertUnlockedWebGpuCoverageGates(zeroFullRows, ["logits"])).toThrow(/row coverage/);
  });

  it("fails strict logits when full-vocab logits cover only selected rows", () => {
    const summary = summarizeUnlockedWebGpuCoverage({
      logitProjectionBackend: "webgpu",
      logitProjectionPurpose: "full_vocab_logit_projection",
      logitProjectionSelectedRows: 32,
      logitProjectionFullRows: 151936,
    });

    expect(() => assertUnlockedWebGpuCoverageGates(summary, ["logits"])).toThrow(/rows are incomplete/);
  });

  it("fails strict attention when prefill or decode attention proof is absent", () => {
    const decodeOnly = summarizeUnlockedWebGpuCoverage({
      decodeAttentionBackends: [
        { layerIndex: 0, attentionBackend: "webgpu", packedHeadBackends: ["webgpu"], packedHeadCount: 1 },
      ],
    });
    const prefillOnly = summarizeUnlockedWebGpuCoverage({
      prefillAttentionBackends: [
        { layerIndex: 0, attentionBackend: "webgpu", packedHeadBackends: ["webgpu"], packedHeadCount: 1 },
      ],
    });

    expect(() => assertUnlockedWebGpuCoverageGates(decodeOnly, ["attention"])).toThrow(/prefill attention proof is missing/);
    expect(() => assertUnlockedWebGpuCoverageGates(prefillOnly, ["attention"])).toThrow(/decode attention proof is missing/);
  });

  it("fails strict projection when expected layer coverage is partial even if reported layers are WebGPU", () => {
    const summary = summarizeUnlockedWebGpuCoverage({
      expectedLayerCount: 2,
      prefillProjectionBackends: [
        { layerIndex: 0, qProjection: "webgpu", kProjection: "webgpu", vProjection: "webgpu", oProjection: "webgpu" },
      ],
      decodeProjectionBackends: [
        { layerIndex: 0, qProjection: "webgpu", kProjection: "webgpu", vProjection: "webgpu", oProjection: "webgpu", projectionKind: "matmul", tokens: 1 },
      ],
    });

    expect(() => assertUnlockedWebGpuCoverageGates(summary, ["projection"])).toThrow(/projection layer coverage is incomplete/);
  });

  it("fails strict attention when expected layer coverage is partial even if reported layers are WebGPU", () => {
    const summary = summarizeUnlockedWebGpuCoverage({
      expectedLayerCount: 2,
      prefillAttentionBackends: [
        { layerIndex: 0, attentionBackend: "webgpu", packedHeadBackends: ["webgpu"], packedHeadCount: 1 },
      ],
      decodeAttentionBackends: [
        { layerIndex: 0, attentionBackend: "webgpu", packedHeadBackends: ["webgpu"], packedHeadCount: 1 },
      ],
    });

    expect(() => assertUnlockedWebGpuCoverageGates(summary, ["attention"])).toThrow(/attention layer coverage is incomplete/);
  });

  it("passes strict gates only when every required proof is WebGPU-backed", () => {
    const summary = summarizeUnlockedWebGpuCoverage({
      expectedLayerCount: 1,
      mlpKernelBackends: [{ layerIndex: 0, backend: "webgpu", activationKind: "silu_gated" }],
      prefillMlpKernelBackends: [{ layerIndex: 0, backend: "webgpu", activationKind: "silu_gated", rowCount: 4 }],
      logitProjectionBackend: "webgpu",
      logitProjectionPurpose: "full_vocab_topk_logit_projection",
      logitProjectionSelectedRows: 4,
      logitProjectionFullRows: 151936,
      prefillProjectionBackends: [
        { layerIndex: 0, qProjection: "webgpu", kProjection: "webgpu", vProjection: "webgpu", oProjection: "webgpu" },
      ],
      decodeProjectionBackends: [
        { layerIndex: 0, qProjection: "webgpu", kProjection: "webgpu", vProjection: "webgpu", oProjection: "webgpu", projectionKind: "matmul", tokens: 2 },
      ],
      prefillAttentionBackends: [
        { layerIndex: 0, attentionBackend: "webgpu", packedHeadBackends: ["webgpu"], packedHeadCount: 1 },
      ],
      decodeAttentionBackends: [
        { layerIndex: 0, attentionBackend: "webgpu", packedHeadBackends: ["webgpu"], packedHeadCount: 1 },
      ],
    });

    expect(summary.cpuFallbackUsed).toBe(false);
    expect(() => assertUnlockedWebGpuCoverageGates(summary, ["mlp", "logits", "projection", "attention"])).not.toThrow();
  });

  it("reads release strict gate env vars", () => {
    expect(readStrictUnlockedWebGpuGatesFromEnv({
      RELEASE_REQUIRE_UNLOCKED_STRICT_WEBGPU: "true",
    } as NodeJS.ProcessEnv)).toEqual(["mlp", "logits", "attention", "projection"]);
    expect(readStrictUnlockedWebGpuGatesFromEnv({
      RELEASE_REQUIRE_UNLOCKED_MODEL: "true",
    } as NodeJS.ProcessEnv)).toEqual(["mlp", "logits", "attention", "projection"]);
    expect(readStrictUnlockedWebGpuGatesFromEnv({
      VITE_REQUIRE_WEBGPU_KERNELS: "true",
    } as NodeJS.ProcessEnv)).toEqual(["mlp", "logits", "attention", "projection"]);
    expect(readStrictUnlockedWebGpuGatesFromEnv({
      RELEASE_REQUIRE_UNLOCKED_MODEL: "true",
      RELEASE_REQUIRE_UNLOCKED_STRICT_WEBGPU: "false",
      VITE_REQUIRE_WEBGPU_KERNELS: "false",
    } as NodeJS.ProcessEnv)).toEqual([]);
    expect(readStrictUnlockedWebGpuGatesFromEnv({
      RELEASE_REQUIRE_UNLOCKED_WEBGPU_MLP: "true",
      RELEASE_REQUIRE_UNLOCKED_WEBGPU_LOGITS: "true",
      RELEASE_REQUIRE_UNLOCKED_WEBGPU_ATTENTION: "false",
      RELEASE_REQUIRE_UNLOCKED_WEBGPU_PROJECTION: "true",
    } as NodeJS.ProcessEnv)).toEqual(["mlp", "logits", "projection"]);
  });

  it("fails strict attention when packed-head proof count is incomplete", () => {
    const summary = summarizeUnlockedWebGpuCoverage({
      prefillAttentionBackends: [
        { layerIndex: 0, attentionBackend: "webgpu", packedHeadBackends: ["webgpu"], packedHeadCount: 2 },
      ],
      decodeAttentionBackends: [
        { layerIndex: 0, attentionBackend: "webgpu", packedHeadBackends: ["webgpu"], packedHeadCount: 1 },
      ],
    });

    expect(summary.attentionBackends.incompletePackedHeadProofs).toBe(1);
    expect(() => assertUnlockedWebGpuCoverageGates(summary, ["attention"])).toThrow(/packed-head proofs are incomplete/);
  });

  it("fails strict attention when packed-head proof is empty even if attention backends are WebGPU", () => {
    const summary = summarizeUnlockedWebGpuCoverage({
      prefillAttentionBackends: [
        { layerIndex: 0, attentionBackend: "webgpu", packedHeadBackends: [], packedHeadCount: 0 },
      ],
      decodeAttentionBackends: [
        { layerIndex: 0, attentionBackend: "webgpu", packedHeadBackends: [], packedHeadCount: 0 },
      ],
    });

    expect(summary.attentionBackends.incompletePackedHeadProofs).toBe(2);
    expect(() => assertUnlockedWebGpuCoverageGates(summary, ["attention"])).toThrow(/packed-head proofs are incomplete/);
  });
});
