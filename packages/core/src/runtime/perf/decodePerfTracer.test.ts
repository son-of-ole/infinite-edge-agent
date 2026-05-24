import { describe, expect, it } from "vitest";
import {
  DecodePerfTracer,
  evaluateDecodeSpeedGate,
  summarizeDecodeHotPath,
} from "./index";

describe("decode hot path performance tracing", () => {
  it("fails the browser production speed gate when decode leaves the GPU-resident hot path", () => {
    const tracer = new DecodePerfTracer({ requestId: "req_slow_path" });
    tracer.recordGeneratedToken();
    tracer.recordFullLogitsReadback({ rows: 151_936, bytes: 607_744 });
    tracer.recordWeightUpload({ bytes: 8_388_608, reason: "decode_cache_miss" });
    tracer.recordActivationUpload({ bytes: 4_096, reason: "decode_hidden_resident_upload" });
    tracer.recordHiddenReadback();
    tracer.recordF32Expansion({ bytes: 16_777_216, reason: "f16_matrix_full_decode" });
    tracer.recordPrefillDuringDecode();

    const summary = tracer.snapshot({ tokensPerSecond: 0.021 });
    const gate = evaluateDecodeSpeedGate(summary);

    expect(gate.passed).toBe(false);
    expect(gate.reasons).toEqual(expect.arrayContaining([
      "tokens_per_second_below_1",
      "full_logits_readback",
      "decode_weight_upload",
      "decode_activation_upload",
      "hidden_readback",
      "decode_f32_expansion",
      "prefill_during_decode",
      "resident_final_hidden_not_used",
    ]));
  });

  it("passes the browser production speed gate for compact argmax readback with resident weights and KV decode", () => {
    const tracer = new DecodePerfTracer({ requestId: "req_hot_path" });
    tracer.recordGeneratedToken();
    tracer.recordDecodeCall();
    tracer.recordDispatches(18);
    tracer.recordCompactLogitReadback({ rows: 1, bytes: 8 });
    tracer.recordKvDecodeReuse();
    tracer.recordResidentDecodeLayers({ residentLayerCount: 1, totalLayerCount: 1 });
    tracer.recordResidentFinalHiddenUsedForLogits(true);

    const summary = tracer.snapshot({ tokensPerSecond: 1.25 });
    const gate = evaluateDecodeSpeedGate(summary);

    expect(gate).toMatchObject({
      passed: true,
      reasons: [],
    });
    expect(summary).toMatchObject({
      cpuFallbackUsed: false,
      cpuValidationUsed: false,
      weightUploadBytesDuringDecode: 0,
      activationUploadBytesDuringDecode: 0,
      hiddenReadbackCountDuringDecode: 0,
      f32ExpansionCountDuringDecode: 0,
      fullLogitsReadbackCount: 0,
      prefillCountPerGeneratedToken: 0,
      residentDecodeLayerCoverage: 1,
      residentFinalHiddenUsedForLogits: true,
    });
  });

  it("summarizes proof metadata from the existing WebGPU logit trace shape", () => {
    const summary = summarizeDecodeHotPath({
      tokensPerSecond: 2.5,
      generatedTokens: 2,
      logitProjectionBackend: "webgpu",
      logitProjectionReadbackStrategy: "gpu_top1_candidates",
      logitProjectionReadbackRows: 1,
      logitProjectionReadbackBytes: 8,
      logitProjectionDispatchCount: 19,
      decodeSubmitCount: 9,
      prefillExecutionsDuringDecode: 0,
      activationUploadBytesDuringDecode: 0,
      activationUploadCountDuringDecode: 0,
      hiddenReadbackCountDuringDecode: 0,
      residentDecodeLayerCount: 4,
      totalDecodeLayerCount: 4,
      residentFinalHiddenUsedForLogits: true,
      fusedPackedQkvLayerCount: 4,
      fusedQkvNormRopeKvAppendLayerCount: 4,
      fusedOneTokenAttentionLayerCount: 2,
      fusedResidualRmsNormLayerCount: 2,
      fusedMlpLayerCount: 4,
      fusedFullLayerCount: 2,
    });

    expect(summary).toMatchObject({
      tokensPerSecond: 2.5,
      generatedTokenCount: 2,
      compactLogitReadbackCount: 1,
      fullLogitsReadbackCount: 0,
      totalReadbackBytes: 8,
      dispatchCount: 19,
      decodeSubmitCount: 9,
      decodeSubmitCountPerToken: 4.5,
      decodeDispatchCountPerToken: 9.5,
      decodeDispatchCountPerLayerPerToken: 4.75,
      prefillCountPerGeneratedToken: 0,
      residentDecodeLayerCoverage: 1,
      residentFinalHiddenUsedForLogits: true,
      fusedPackedQkvLayerCount: 4,
      fusedQkvNormRopeKvAppendLayerCount: 4,
      fusedOneTokenAttentionLayerCount: 2,
      fusedResidualRmsNormLayerCount: 2,
      fusedMlpLayerCount: 4,
      fusedFullLayerCount: 2,
      fusedLayerCoverage: 0.75,
    });
  });
});
