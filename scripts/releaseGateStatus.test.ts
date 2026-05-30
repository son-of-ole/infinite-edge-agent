import { describe, expect, it } from "vitest";
import { classifyReleaseGateProof, computeReleaseGatePassed } from "./releaseGateStatus";

describe("release gate status", () => {
  it("fails when a required latest artifact is missing or unknown", () => {
    expect(computeReleaseGatePassed({
      steps: [{ status: "passed" }],
      latestArtifacts: [{ name: "qwen-parity-accuracy", passed: null }],
    })).toBe(false);
  });

  it("allows only explicitly optional missing artifacts", () => {
    expect(computeReleaseGatePassed({
      steps: [{ status: "passed" }],
      latestArtifacts: [{ name: "optional-report", passed: null }],
      optionalArtifactNames: ["optional-report"],
    })).toBe(true);
  });

  it("fails when any latest artifact reports failure", () => {
    expect(computeReleaseGatePassed({
      steps: [{ status: "passed" }],
      latestArtifacts: [{ name: "production-readiness", passed: false }],
      optionalArtifactNames: ["production-readiness"],
    })).toBe(false);
  });

  it("fails when hosted benchmark proof is present but source binding was not required", () => {
    expect(computeReleaseGatePassed({
      steps: [{ status: "passed" }],
      latestArtifacts: [
        {
          name: "hosted-benchmark-proof",
          passed: true,
          summary: {
            hostedBenchmarkProofPassed: true,
            hostedBenchmarkProofSourceGitSha: "abc123",
            hostedBenchmarkExpectedSourceGitSha: "abc123",
            hostedBenchmarkProofSourceBound: true,
            hostedBenchmarkProofSourceBoundRequired: false,
          },
        },
      ],
    })).toBe(false);
  });

  it("fails when hosted benchmark proof is present but not source-bound", () => {
    expect(computeReleaseGatePassed({
      steps: [{ status: "passed" }],
      latestArtifacts: [
        {
          name: "hosted-benchmark-proof",
          passed: true,
          summary: {
            hostedBenchmarkProofPassed: true,
            hostedBenchmarkProofSourceGitSha: "abc123",
            hostedBenchmarkExpectedSourceGitSha: "def456",
            hostedBenchmarkProofSourceBound: false,
            hostedBenchmarkProofSourceBoundRequired: true,
          },
        },
      ],
    })).toBe(false);
  });

  it("accepts a standalone hosted benchmark proof only when it is source-bound", () => {
    expect(computeReleaseGatePassed({
      steps: [{ status: "passed" }],
      latestArtifacts: [
        {
          name: "hosted-benchmark-proof",
          passed: true,
          summary: {
            hostedBenchmarkProofPassed: true,
            hostedBenchmarkProofSourceGitSha: "abc123",
            hostedBenchmarkExpectedSourceGitSha: "abc123",
            hostedBenchmarkProofSourceBound: true,
            hostedBenchmarkProofSourceBoundRequired: true,
            hostedBenchmarkConcreteMemoryGroundingPassed: true,
            hostedBenchmarkMemoryGroundingRunCount: 1,
            hostedBenchmarkMemorySeededCorpusCount: 16,
            hostedBenchmarkMemoryRetrievedCount: 1,
            hostedBenchmarkMemoryIncludedCount: 1,
            hostedBenchmarkMemoryExpectedMemoryIdCount: 1,
            hostedBenchmarkMemoryExpectedHitMeanRank: 1,
            hostedBenchmarkBrokerDeployBackendId: "compiled-browser-webllm",
            hostedBenchmarkBrokerKernelLabBackendId: "unlocked-browser-transformer",
            hostedBenchmarkBrokerFallbackBackendId: "wasm-small-core",
            hostedBenchmarkBrokerFallbackBackendCount: 1,
            hostedBenchmarkBrokerFallbackDeployReadyCandidate: false,
            hostedBenchmarkBrokerRoleBoundaryPassed: true,
          },
        },
      ],
    })).toBe(true);
  });

  it("fails standalone hosted benchmark proof without Backend Broker role-boundary evidence", () => {
    expect(computeReleaseGatePassed({
      steps: [{ status: "passed" }],
      latestArtifacts: [
        {
          name: "hosted-benchmark-proof",
          passed: true,
          summary: {
            hostedBenchmarkProofPassed: true,
            hostedBenchmarkProofSourceGitSha: "abc123",
            hostedBenchmarkExpectedSourceGitSha: "abc123",
            hostedBenchmarkProofSourceBound: true,
            hostedBenchmarkProofSourceBoundRequired: true,
            hostedBenchmarkConcreteMemoryGroundingPassed: true,
            hostedBenchmarkMemoryGroundingRunCount: 1,
            hostedBenchmarkMemorySeededCorpusCount: 16,
            hostedBenchmarkMemoryRetrievedCount: 1,
            hostedBenchmarkMemoryIncludedCount: 1,
            hostedBenchmarkMemoryExpectedMemoryIdCount: 1,
            hostedBenchmarkMemoryExpectedHitMeanRank: 1,
            hostedBenchmarkBrokerRoleBoundaryPassed: false,
          },
        },
      ],
    })).toBe(false);
  });

  it("fails standalone hosted benchmark proof without concrete memory grounding evidence", () => {
    expect(computeReleaseGatePassed({
      steps: [{ status: "passed" }],
      latestArtifacts: [
        {
          name: "hosted-benchmark-proof",
          passed: true,
          summary: {
            hostedBenchmarkProofPassed: true,
            hostedBenchmarkProofSourceGitSha: "abc123",
            hostedBenchmarkExpectedSourceGitSha: "abc123",
            hostedBenchmarkProofSourceBound: true,
            hostedBenchmarkProofSourceBoundRequired: true,
            hostedBenchmarkConcreteMemoryGroundingPassed: false,
            hostedBenchmarkMemoryGroundingRunCount: 0,
          },
        },
      ],
    })).toBe(false);
  });

  it("fails strict unlocked model releases when real parity or benchmark mode is still fixture", () => {
    expect(computeReleaseGatePassed({
      steps: [{ status: "passed" }],
      strictUnlockedModel: true,
      latestArtifacts: [
        { name: "qwen-parity-accuracy", passed: true, summary: { realModelParityMode: "skipped" } },
        { name: "browser-runtime-bench", passed: true, summary: { browserBenchMemoryMode: "browser-local-fixture", browserBenchPreviewMode: "skipped" } },
        { name: "unlocked-verify", passed: true, summary: { profile: "full", runIsCapped: false } },
      ],
    })).toBe(false);
  });

  it("fails strict unlocked model releases when browser memory mode is not configured", () => {
    expect(computeReleaseGatePassed({
      steps: [{ status: "passed" }],
      strictUnlockedModel: true,
      latestArtifacts: [
        {
          name: "qwen-parity-accuracy",
          passed: true,
          summary: {
            realModelParityMode: "installed",
            realModelId: "Qwen/Qwen3-0.6B",
            realModelVocabSize: 151936,
            realModelLogitProjectionPurpose: "full_vocab_topk_logit_projection",
            realModelLogitProjectionSelectedRows: 8,
            realModelLogitProjectionFullRows: 151936,
          },
        },
        {
          name: "browser-runtime-bench",
          passed: true,
          summary: {
            browserBenchMemoryMode: "dev",
            browserBenchPreviewMode: "completed",
            browserBenchPreviewRequested: true,
            browserBenchPreviewPassed: true,
            browserBenchStrictWebGpuPassed: true,
            cpuFallbackUsed: false,
            browserBenchExpectedSubstringsPassed: true,
            browserBenchExpectedExactPassed: true,
            browserBenchExpectedExactCheckCount: 1,
            browserBenchVisibleResponseQualityPassed: true,
            browserBenchStopQualityPassed: true,
            browserBenchRunawayRepetitionPassed: true,
            browserBenchMarkerOnlyResponsePassed: true,
            browserBenchTechnicalProofOnly: false,
            browserBenchProductionQualityPassed: true,
            browserBenchProductionDeployReadyPassed: true,
            browserBenchGroundedProductionReadyPassed: true,
            browserBenchDirectModelFactualProofUsed: false,
            browserBenchMemoryGroundingRequired: true,
            browserBenchMemoryGroundingPassed: true,
            browserBenchMemoryExpectedHitPassed: true,
            browserBenchMemoryContextRebuildPassed: true,
            browserBenchMemoryAnswerOnlyPassed: true,
            browserBenchMemorySeededCorpusCount: 1024,
            browserBenchMemoryRetrievalAuditRequired: true,
            browserBenchMemoryRetrievalAuditPassed: true,
            browserBenchMemoryRetrievalAuditQueryCount: 64,
            browserBenchMemoryRecallAt1: 1,
            browserBenchRequireKvReuse: true,
            browserBenchKvReusePassed: true,
            browserBenchKvExactReuseRunCount: 1,
            browserBenchRequireKvPredictivePrefetch: true,
            browserBenchKvPredictivePrefetchPassed: true,
            browserBenchKvLowRankQuerySource: "persisted_q_rows",
            browserBenchProductionSpeedFloorPassed: true,
            browserBenchV11CommandBatchingPassed: true,
            browserBenchMtpMode: "target_only",
          },
        },
        {
          name: "unlocked-verify",
          passed: true,
          summary: {
            profile: "full",
            runIsCapped: false,
            logitProjectionPurpose: "full_vocab_topk_logit_projection",
            kvDecodeReuse: true,
            tensorStorageExplicit: true,
            packedProductionReady: true,
            tensorStorageFormat: "f16-packed",
            tensorStorageDtype: "f16",
          },
        },
      ],
    })).toBe(false);
  });

  it("classifies fixture/unconfigured release passes as non-production proof", () => {
    expect(classifyReleaseGateProof({
      passed: true,
      strictUnlockedModel: false,
      requireBrowserPreviewProof: false,
      releaseAllowFixtureGate: true,
      unlockedAllowFixture: "true",
      manifestPath: null,
      manifestSha256: null,
    })).toMatchObject({
      proofMode: "development-fixture-or-unconfigured",
      productionReleaseProof: false,
      strictEnv: {
        RELEASE_REQUIRE_UNLOCKED_MODEL: false,
        RELEASE_REQUIRE_BROWSER_PREVIEW_PROOF: false,
        RELEASE_ALLOW_FIXTURE_GATE: true,
        VITE_UNLOCKED_ALLOW_FIXTURE: "true",
        VITE_UNLOCKED_MODEL_MANIFEST_PATH: null,
        VITE_UNLOCKED_MODEL_MANIFEST_SHA256: null,
      },
    });
  });

  it("classifies strict browser releases without manifest identity as non-production proof", () => {
    expect(classifyReleaseGateProof({
      passed: true,
      strictUnlockedModel: true,
      requireBrowserPreviewProof: true,
      releaseAllowFixtureGate: false,
      unlockedAllowFixture: "false",
      manifestPath: "",
      manifestSha256: "",
    })).toMatchObject({
      proofMode: "development-fixture-or-unconfigured",
      productionReleaseProof: false,
      strictEnv: {
        RELEASE_REQUIRE_UNLOCKED_MODEL: true,
        RELEASE_REQUIRE_BROWSER_PREVIEW_PROOF: true,
        RELEASE_ALLOW_FIXTURE_GATE: false,
        VITE_UNLOCKED_ALLOW_FIXTURE: "false",
        VITE_UNLOCKED_MODEL_MANIFEST_PATH: null,
        VITE_UNLOCKED_MODEL_MANIFEST_SHA256: null,
      },
    });
  });

  it("classifies strict browser releases as production proof only after the gate passes", () => {
    expect(classifyReleaseGateProof({
      passed: true,
      strictUnlockedModel: true,
      requireBrowserPreviewProof: true,
      releaseAllowFixtureGate: false,
      unlockedAllowFixture: "false",
      manifestPath: "/models/qwen3/manifest.json",
      manifestSha256: "abc123",
    })).toMatchObject({
      proofMode: "production-strict-browser-runtime",
      productionReleaseProof: true,
      strictEnv: {
        RELEASE_REQUIRE_UNLOCKED_MODEL: true,
        RELEASE_REQUIRE_BROWSER_PREVIEW_PROOF: true,
        RELEASE_ALLOW_FIXTURE_GATE: false,
        VITE_UNLOCKED_ALLOW_FIXTURE: "false",
        VITE_UNLOCKED_MODEL_MANIFEST_PATH: "/models/qwen3/manifest.json",
        VITE_UNLOCKED_MODEL_MANIFEST_SHA256: "present",
      },
    });
  });

  it("classifies strict v12 production archives as backend-specific production proof", () => {
    expect(classifyReleaseGateProof({
      passed: true,
      strictUnlockedModel: false,
      requireBrowserPreviewProof: false,
      requireV12ProductionArchive: true,
      releaseAllowFixtureGate: false,
      unlockedAllowFixture: "false",
      manifestPath: null,
      manifestSha256: null,
      latestArtifacts: [
        {
          name: "v12-production-archive",
          passed: true,
          summary: makeV12ProductionArchiveSummary(),
        },
      ],
    })).toMatchObject({
      proofMode: "production-v12-compiled-browser-runtime",
      productionReleaseProof: true,
      backendSpecificProductionProof: true,
      v12ProductionArchiveProof: true,
      deployReadySpeedQualityProof: false,
      strictEnv: {
        RELEASE_REQUIRE_V12_PRODUCTION: true,
      },
    });
  });

  it("does not classify v12 production archives without concrete hosted runtime proof fields", () => {
    expect(classifyReleaseGateProof({
      passed: true,
      strictUnlockedModel: false,
      requireBrowserPreviewProof: false,
      requireV12ProductionArchive: true,
      releaseAllowFixtureGate: false,
      unlockedAllowFixture: "false",
      manifestPath: null,
      manifestSha256: null,
      latestArtifacts: [
        {
          name: "v12-production-archive",
          passed: true,
          summary: makeV12ProductionArchiveSummary({}, { includeHostedRuntimeProof: false }),
        },
      ],
    })).toMatchObject({
      proofMode: "development-fixture-or-unconfigured",
      productionReleaseProof: false,
      backendSpecificProductionProof: false,
      v12ProductionArchiveProof: false,
    });
  });

  it("does not classify v12 production archives without model registry alignment proof", () => {
    expect(classifyReleaseGateProof({
      passed: true,
      strictUnlockedModel: false,
      requireBrowserPreviewProof: false,
      requireV12ProductionArchive: true,
      releaseAllowFixtureGate: false,
      unlockedAllowFixture: "false",
      manifestPath: null,
      manifestSha256: null,
      latestArtifacts: [
        {
          name: "v12-production-archive",
          passed: true,
          summary: makeV12ProductionArchiveSummary({
            v12ProductionModelRegistryAligned: false,
          }),
        },
      ],
    })).toMatchObject({
      proofMode: "development-fixture-or-unconfigured",
      productionReleaseProof: false,
      backendSpecificProductionProof: false,
      v12ProductionArchiveProof: false,
    });
  });

  it("does not classify incomplete v12 production archives as production proof", () => {
    expect(classifyReleaseGateProof({
      passed: true,
      strictUnlockedModel: false,
      requireBrowserPreviewProof: false,
      releaseAllowFixtureGate: false,
      unlockedAllowFixture: "false",
      manifestPath: null,
      manifestSha256: null,
      latestArtifacts: [
        {
          name: "v12-production-archive",
          passed: true,
          summary: makeV12ProductionArchiveSummary({
            v12ProductionHostedBenchmarkProofRequired: false,
          }),
        },
      ],
    })).toMatchObject({
      proofMode: "development-fixture-or-unconfigured",
      productionReleaseProof: false,
      backendSpecificProductionProof: false,
      v12ProductionArchiveProof: false,
    });
  });

  it("fails strict unlocked model releases when installed parity still uses candidate logits", () => {
    expect(computeReleaseGatePassed({
      steps: [{ status: "passed" }],
      strictUnlockedModel: true,
      latestArtifacts: [
        {
          name: "qwen-parity-accuracy",
          passed: true,
          summary: {
            realModelParityMode: "installed",
            realModelId: "Qwen/Qwen3-0.6B",
            realModelVocabSize: 151936,
            realModelLogitProjectionPurpose: "candidate_logit_projection",
            realModelLogitProjectionSelectedRows: 4096,
            realModelLogitProjectionFullRows: 151936,
          },
        },
        { name: "browser-runtime-bench", passed: true, summary: { browserBenchMemoryMode: "browser-vector", browserBenchMtpMode: "draft_verify" } },
        {
          name: "unlocked-verify",
          passed: true,
          summary: {
            profile: "full",
            runIsCapped: false,
            logitProjectionPurpose: "full_vocab_topk_logit_projection",
            mtpMode: "draft_verify",
            mtpVerifierStrategy: "batched_continuation",
            mtpVerifiedTokenCount: 2,
            mtpTargetDecodeCalls: 1,
            kvDecodeReuse: true,
            tensorStorageExplicit: true,
            packedProductionReady: true,
            tensorStorageFormat: "f16-packed",
            tensorStorageDtype: "f16",
          },
        },
      ],
    })).toBe(false);
  });

  it("fails strict unlocked model releases while verify still reports f32 reference assets", () => {
    expect(computeReleaseGatePassed({
      steps: [{ status: "passed" }],
      strictUnlockedModel: true,
      latestArtifacts: [
        {
          name: "qwen-parity-accuracy",
          passed: true,
          summary: {
            realModelParityMode: "installed",
            realModelId: "Qwen/Qwen3-0.6B",
            realModelVocabSize: 151936,
            realModelLogitProjectionPurpose: "full_vocab_topk_logit_projection",
            realModelLogitProjectionSelectedRows: 8,
            realModelLogitProjectionFullRows: 151936,
          },
        },
        {
          name: "browser-runtime-bench",
          passed: true,
          summary: {
            browserBenchMemoryMode: "browser-vector",
            browserBenchPreviewMode: "completed",
            browserBenchPreviewRequested: true,
            browserBenchPreviewPassed: true,
            browserBenchStrictWebGpuPassed: true,
            cpuFallbackUsed: false,
            browserBenchExpectedSubstringsPassed: true,
            browserBenchExpectedExactPassed: true,
            browserBenchExpectedExactCheckCount: 1,
            browserBenchVisibleResponseQualityPassed: true,
            browserBenchStopQualityPassed: true,
            browserBenchRunawayRepetitionPassed: true,
            browserBenchMarkerOnlyResponsePassed: true,
            browserBenchMemoryGroundingPassed: true,
            browserBenchMemoryExpectedHitPassed: true,
            browserBenchMemoryContextRebuildPassed: true,
            browserBenchMemoryAnswerOnlyPassed: true,
            browserBenchMemorySeededCorpusCount: 1024,
            browserBenchMemoryRetrievalAuditRequired: true,
            browserBenchMemoryRetrievalAuditPassed: true,
            browserBenchMemoryRetrievalAuditQueryCount: 64,
            browserBenchMemoryRecallAt1: 1,
            browserBenchRequireKvReuse: true,
            browserBenchKvReusePassed: true,
            browserBenchKvExactReuseRunCount: 1,
            browserBenchRequireKvPredictivePrefetch: true,
            browserBenchKvPredictivePrefetchPassed: true,
            browserBenchKvLowRankQuerySource: "persisted_q_rows",
            browserBenchProductionSpeedFloorPassed: true,
            browserBenchMtpMode: "target_only",
          },
        },
        {
          name: "unlocked-verify",
          passed: true,
          summary: {
            profile: "full",
            runIsCapped: false,
            logitProjectionPurpose: "full_vocab_topk_logit_projection",
            mtpMode: "draft_verify",
            mtpVerifierStrategy: "batched_continuation",
            mtpVerifiedTokenCount: 2,
            mtpTargetDecodeCalls: 1,
            kvDecodeReuse: true,
            tensorStorageExplicit: true,
            packedProductionReady: false,
            tensorStorageFormat: "f32-reference",
            tensorStorageDtype: "f32",
          },
        },
      ],
    })).toBe(false);
  });

  it("fails strict unlocked model releases when packed tensor storage metadata was inferred instead of explicit", () => {
    expect(computeReleaseGatePassed({
      steps: [{ status: "passed" }],
      strictUnlockedModel: true,
      latestArtifacts: [
        {
          name: "qwen-parity-accuracy",
          passed: true,
          summary: {
            realModelParityMode: "installed",
            realModelId: "Qwen/Qwen3-0.6B",
            realModelVocabSize: 151936,
            realModelLogitProjectionPurpose: "full_vocab_topk_logit_projection",
            realModelLogitProjectionSelectedRows: 8,
            realModelLogitProjectionFullRows: 151936,
          },
        },
        {
          name: "browser-runtime-bench",
          passed: true,
          summary: {
            browserBenchMemoryMode: "browser-vector",
            browserBenchPreviewMode: "completed",
            browserBenchPreviewRequested: true,
            browserBenchPreviewPassed: true,
            browserBenchStrictWebGpuPassed: true,
            cpuFallbackUsed: false,
            browserBenchExpectedSubstringsPassed: true,
            browserBenchExpectedExactPassed: true,
            browserBenchExpectedExactCheckCount: 1,
            browserBenchVisibleResponseQualityPassed: true,
            browserBenchStopQualityPassed: true,
            browserBenchRunawayRepetitionPassed: true,
            browserBenchMarkerOnlyResponsePassed: true,
            browserBenchMemoryGroundingPassed: true,
            browserBenchMemoryExpectedHitPassed: true,
            browserBenchMemoryContextRebuildPassed: true,
            browserBenchMemoryAnswerOnlyPassed: true,
            browserBenchMemorySeededCorpusCount: 1024,
            browserBenchMemoryRetrievalAuditRequired: true,
            browserBenchMemoryRetrievalAuditPassed: true,
            browserBenchMemoryRetrievalAuditQueryCount: 64,
            browserBenchMemoryRecallAt1: 1,
            browserBenchRequireKvReuse: true,
            browserBenchKvReusePassed: true,
            browserBenchKvExactReuseRunCount: 1,
            browserBenchRequireKvPredictivePrefetch: true,
            browserBenchKvPredictivePrefetchPassed: true,
            browserBenchKvLowRankQuerySource: "persisted_q_rows",
            browserBenchProductionSpeedFloorPassed: true,
            browserBenchMtpMode: "target_only",
          },
        },
        {
          name: "unlocked-verify",
          passed: true,
          summary: {
            profile: "full",
            runIsCapped: false,
            logitProjectionPurpose: "full_vocab_topk_logit_projection",
            mtpMode: "draft_verify",
            mtpVerifierStrategy: "batched_continuation",
            mtpVerifiedTokenCount: 2,
            mtpTargetDecodeCalls: 1,
            kvDecodeReuse: true,
            tensorStorageExplicit: false,
            packedProductionReady: true,
            tensorStorageFormat: "f16-packed",
            tensorStorageDtype: "f16",
          },
        },
      ],
    })).toBe(false);
  });

  it("fails strict unlocked model releases when browser preview proof is skipped", () => {
    expect(computeReleaseGatePassed({
      steps: [{ status: "passed" }],
      strictUnlockedModel: true,
      latestArtifacts: [
        {
          name: "qwen-parity-accuracy",
          passed: true,
          summary: {
            realModelParityMode: "installed",
            realModelId: "Qwen/Qwen3-0.6B",
            realModelVocabSize: 151936,
            realModelLogitProjectionPurpose: "full_vocab_topk_logit_projection",
            realModelLogitProjectionSelectedRows: 8,
            realModelLogitProjectionFullRows: 151936,
          },
        },
        { name: "browser-runtime-bench", passed: true, summary: { browserBenchMemoryMode: "browser-vector", browserBenchPreviewMode: "skipped", browserBenchPreviewRequested: false, browserBenchPreviewPassed: false, browserBenchMtpMode: "draft_verify" } },
        {
          name: "unlocked-verify",
          passed: true,
          summary: {
            profile: "full",
            runIsCapped: false,
            logitProjectionPurpose: "full_vocab_topk_logit_projection",
            mtpMode: "draft_verify",
            mtpVerifierStrategy: "batched_continuation",
            mtpVerifiedTokenCount: 2,
            mtpTargetDecodeCalls: 1,
            kvDecodeReuse: true,
            tensorStorageExplicit: true,
            packedProductionReady: true,
            tensorStorageFormat: "f16-packed",
            tensorStorageDtype: "f16",
          },
        },
      ],
    })).toBe(false);
  });

  it("fails strict unlocked model releases when exact KV reuse was not proven", () => {
    expect(computeReleaseGatePassed({
      steps: [{ status: "passed" }],
      strictUnlockedModel: true,
      latestArtifacts: [
        {
          name: "qwen-parity-accuracy",
          passed: true,
          summary: {
            realModelParityMode: "installed",
            realModelId: "Qwen/Qwen3-0.6B",
            realModelVocabSize: 151936,
            realModelLogitProjectionPurpose: "full_vocab_topk_logit_projection",
            realModelLogitProjectionSelectedRows: 8,
            realModelLogitProjectionFullRows: 151936,
          },
        },
        {
          name: "browser-runtime-bench",
          passed: true,
          summary: {
            browserBenchMemoryMode: "browser-vector",
            browserBenchPreviewMode: "completed",
            browserBenchPreviewRequested: true,
            browserBenchPreviewPassed: true,
            browserBenchStrictWebGpuPassed: true,
            cpuFallbackUsed: false,
            browserBenchExpectedSubstringsPassed: true,
            browserBenchExpectedExactPassed: true,
            browserBenchExpectedExactCheckCount: 1,
            browserBenchVisibleResponseQualityPassed: true,
            browserBenchStopQualityPassed: true,
            browserBenchRunawayRepetitionPassed: true,
            browserBenchMarkerOnlyResponsePassed: true,
            browserBenchMemoryGroundingPassed: true,
            browserBenchMemoryExpectedHitPassed: true,
            browserBenchMemoryContextRebuildPassed: true,
            browserBenchMemoryAnswerOnlyPassed: true,
            browserBenchMemorySeededCorpusCount: 1024,
            browserBenchMemoryRetrievalAuditRequired: true,
            browserBenchMemoryRetrievalAuditPassed: true,
            browserBenchMemoryRetrievalAuditQueryCount: 64,
            browserBenchMemoryRecallAt1: 1,
            browserBenchRequireKvReuse: true,
            browserBenchKvReusePassed: true,
            browserBenchKvExactReuseRunCount: 0,
            browserBenchRequireKvPredictivePrefetch: true,
            browserBenchKvPredictivePrefetchPassed: true,
            browserBenchKvLowRankQuerySource: "persisted_q_rows",
            browserBenchProductionSpeedFloorPassed: true,
            browserBenchMtpMode: "target_only",
          },
        },
        {
          name: "unlocked-verify",
          passed: true,
          summary: {
            profile: "full",
            runIsCapped: false,
            logitProjectionPurpose: "full_vocab_topk_logit_projection",
            kvDecodeReuse: true,
            tensorStorageExplicit: true,
            packedProductionReady: true,
            tensorStorageFormat: "f16-packed",
            tensorStorageDtype: "f16",
          },
        },
      ],
    })).toBe(false);
  });

  it("passes strict unlocked model releases with installed parity, browser-vector benchmark, browser proof, MTP, KV reuse, and uncapped verify", () => {
    expect(computeReleaseGatePassed({
      steps: [{ status: "passed" }],
      strictUnlockedModel: true,
      latestArtifacts: [
        {
          name: "qwen-parity-accuracy",
          passed: true,
          summary: {
            realModelParityMode: "installed",
            realModelId: "Qwen/Qwen3-0.6B",
            realModelVocabSize: 151936,
            realModelLogitProjectionPurpose: "full_vocab_topk_logit_projection",
            realModelLogitProjectionSelectedRows: 8,
            realModelLogitProjectionFullRows: 151936,
          },
        },
        {
          name: "browser-runtime-bench",
          passed: true,
          summary: {
            browserBenchMemoryMode: "browser-vector",
            browserBenchPreviewMode: "completed",
            browserBenchPreviewRequested: true,
            browserBenchPreviewPassed: true,
            browserBenchStrictWebGpuPassed: true,
            cpuFallbackUsed: false,
            browserBenchExpectedSubstringsPassed: true,
            browserBenchExpectedExactPassed: true,
            browserBenchExpectedExactCheckCount: 1,
            browserBenchVisibleResponseQualityPassed: true,
            browserBenchStopQualityPassed: true,
            browserBenchRunawayRepetitionPassed: true,
            browserBenchMarkerOnlyResponsePassed: true,
            browserBenchTechnicalProofOnly: false,
            browserBenchProductionQualityPassed: true,
            browserBenchProductionDeployReadyPassed: true,
            browserBenchGroundedProductionReadyPassed: true,
            browserBenchDirectModelFactualProofUsed: false,
            browserBenchMemoryGroundingRequired: true,
            browserBenchMemoryGroundingPassed: true,
            browserBenchMemoryExpectedHitPassed: true,
            browserBenchMemoryContextRebuildPassed: true,
            browserBenchMemoryAnswerOnlyPassed: true,
            browserBenchMemorySeededCorpusCount: 1024,
            browserBenchMemoryRetrievalAuditRequired: true,
            browserBenchMemoryRetrievalAuditPassed: true,
            browserBenchMemoryRetrievalAuditQueryCount: 64,
            browserBenchMemoryRecallAt1: 1,
            browserBenchRequireKvReuse: true,
            browserBenchKvReusePassed: true,
            browserBenchKvExactReuseRunCount: 1,
            browserBenchRequireKvPredictivePrefetch: true,
            browserBenchKvPredictivePrefetchPassed: true,
            browserBenchKvLowRankQuerySource: "persisted_q_rows",
            browserBenchProductionSpeedFloorPassed: true,
            browserBenchV11CommandBatchingPassed: true,
            browserBenchMtpMode: "target_only",
          },
        },
        {
          name: "unlocked-verify",
          passed: true,
          summary: {
            profile: "full",
            runIsCapped: false,
            logitProjectionPurpose: "full_vocab_topk_logit_projection",
            mtpMode: "draft_verify",
            mtpVerifierStrategy: "batched_continuation",
            mtpVerifiedTokenCount: 2,
            mtpTargetDecodeCalls: 1,
            kvDecodeReuse: true,
            tensorStorageExplicit: true,
            packedProductionReady: true,
            tensorStorageFormat: "f16-packed",
            tensorStorageDtype: "f16",
          },
        },
      ],
    })).toBe(true);
  });

  it("fails strict unlocked model releases when a fast browser artifact is only technical proof", () => {
    expect(computeReleaseGatePassed({
      steps: [{ status: "passed" }],
      strictUnlockedModel: true,
      latestArtifacts: [
        {
          name: "qwen-parity-accuracy",
          passed: true,
          summary: {
            realModelParityMode: "installed",
            realModelId: "Qwen/Qwen3-0.6B",
            realModelVocabSize: 151936,
            realModelLogitProjectionPurpose: "full_vocab_topk_logit_projection",
            realModelLogitProjectionSelectedRows: 8,
            realModelLogitProjectionFullRows: 151936,
          },
        },
        {
          name: "browser-runtime-bench",
          passed: true,
          summary: {
            browserBenchMemoryMode: "browser-vector",
            browserBenchPreviewMode: "completed",
            browserBenchPreviewRequested: true,
            browserBenchPreviewPassed: true,
            browserBenchStrictWebGpuPassed: true,
            cpuFallbackUsed: false,
            browserBenchExpectedSubstringsPassed: true,
            browserBenchVisibleResponseQualityPassed: true,
            browserBenchStopQualityPassed: true,
            browserBenchRunawayRepetitionPassed: true,
            browserBenchMarkerOnlyResponsePassed: true,
            browserBenchTechnicalProofOnly: true,
            browserBenchProductionQualityPassed: false,
            browserBenchProductionDeployReadyPassed: false,
            browserBenchMemoryGroundingPassed: true,
            browserBenchMemoryExpectedHitPassed: true,
            browserBenchMemoryContextRebuildPassed: true,
            browserBenchMemoryAnswerOnlyPassed: true,
            browserBenchMemorySeededCorpusCount: 64,
            browserBenchRequireKvReuse: true,
            browserBenchKvReusePassed: true,
            browserBenchKvExactReuseRunCount: 1,
            browserBenchRequireKvPredictivePrefetch: true,
            browserBenchKvPredictivePrefetchPassed: true,
            browserBenchKvLowRankQuerySource: "persisted_q_rows",
            browserBenchProductionSpeedFloorPassed: true,
            browserBenchMtpMode: "target_only",
          },
        },
        {
          name: "unlocked-verify",
          passed: true,
          summary: {
            profile: "full",
            runIsCapped: false,
            logitProjectionPurpose: "full_vocab_topk_logit_projection",
            kvDecodeReuse: true,
            tensorStorageExplicit: true,
            packedProductionReady: true,
            tensorStorageFormat: "f16-packed",
            tensorStorageDtype: "f16",
          },
        },
      ],
    })).toBe(false);
  });

  it("fails browser-preview-required releases when the benchmark skipped the browser proof", () => {
    expect(computeReleaseGatePassed({
      steps: [{ status: "passed" }],
      requireBrowserPreviewProof: true,
      latestArtifacts: [
        {
          name: "browser-runtime-bench",
          passed: true,
          summary: {
            browserBenchPreviewMode: "skipped",
            browserBenchPreviewRequested: false,
            browserBenchPreviewPassed: false,
          },
        },
      ],
    })).toBe(false);
  });

  it("passes browser-preview-required releases when the benchmark completed the browser proof", () => {
    expect(computeReleaseGatePassed({
      steps: [{ status: "passed" }],
      requireBrowserPreviewProof: true,
      latestArtifacts: [
        {
          name: "browser-runtime-bench",
          passed: true,
          summary: {
            browserBenchPreviewMode: "completed",
            browserBenchPreviewRequested: true,
            browserBenchPreviewPassed: true,
          },
        },
      ],
    })).toBe(true);
  });

  it("fails strict unlocked model releases when browser proof lacks exact-output and v11 batching gates", () => {
    expect(computeReleaseGatePassed({
      steps: [{ status: "passed" }],
      strictUnlockedModel: true,
      latestArtifacts: makeStrictUnlockedArtifacts({
        browserBenchExpectedExactPassed: false,
        browserBenchExpectedExactCheckCount: 0,
        browserBenchV11CommandBatchingPassed: false,
        browserBenchGroundedProductionReadyPassed: false,
      }),
    })).toBe(false);
  });

  it("passes strict unlocked model releases with grounded exact browser production proof", () => {
    expect(computeReleaseGatePassed({
      steps: [{ status: "passed" }],
      strictUnlockedModel: true,
      latestArtifacts: makeStrictUnlockedArtifacts(),
    })).toBe(true);
  });

  it("fails v12-production-required releases when archive proof fields are incomplete", () => {
    expect(computeReleaseGatePassed({
      steps: [{ status: "passed" }],
      requireV12ProductionArchive: true,
      latestArtifacts: [
        {
          name: "v12-production-archive",
          passed: true,
          summary: {
            v12ProductionArchivePassed: true,
            v12ProductionBlockerCount: 0,
            v12ProductionSuitePassed: true,
            v12ProductionDeployBackendId: "compiled-browser-webllm",
            v12ProductionKernelLabBackendId: "unlocked-browser-transformer",
            v12ProductionHostedBenchmarkProofRequired: false,
            v12ProductionHostedBenchmarkProofPassed: true,
            v12ProductionArtifactCount: 7,
            v12ProductionSuiteArtifactCount: 6,
            v12ProductionChildArtifactCount: 5,
            v12ProductionHostedBenchmarkRuntimeBackendId: "compiled-browser-webllm",
            v12ProductionHostedBenchmarkDeployBackendId: "compiled-browser-webllm",
            v12ProductionCompiledBackendReadyPassed: true,
            v12ProductionDeployReadyPassed: true,
            v12ProductionMemoryGroundingPassed: true,
            v12ProductionExpectedExactPassed: true,
            v12ProductionSpeedFloorPassed: true,
            v12ProductionMeanTokensPerSecond: 2.7,
            v12ProductionDirectModelFactualProofUsed: false,
            v12ProductionTechnicalProofOnly: false,
            v12ProductionCpuFallbackUsed: false,
            v12ProductionStrictWebGpuPassed: true,
          },
        },
      ],
    })).toBe(false);
  });

  it("passes v12-production-required releases only with backend-specific archive proof", () => {
    expect(computeReleaseGatePassed({
      steps: [{ status: "passed" }],
      requireV12ProductionArchive: true,
      latestArtifacts: [
        {
          name: "v12-production-archive",
          passed: true,
          summary: makeV12ProductionArchiveSummary(),
        },
      ],
    })).toBe(true);
  });

  it("fails v12-production-required releases when Backend Broker proof is missing from the archive", () => {
    expect(computeReleaseGatePassed({
      steps: [{ status: "passed" }],
      requireV12ProductionArchive: true,
      latestArtifacts: [
        {
          name: "v12-production-archive",
          passed: true,
          summary: makeV12ProductionArchiveSummary({
            v12ProductionBackendBrokerSelectionPassed: false,
            v12ProductionBackendBrokerTraceCount: 0,
            v12ProductionBrokerSelectedBackendId: null,
          }),
        },
      ],
    })).toBe(false);
  });

  it("fails v12-production-required releases when hosted broker role-boundary proof is missing from the archive", () => {
    expect(computeReleaseGatePassed({
      steps: [{ status: "passed" }],
      requireV12ProductionArchive: true,
      latestArtifacts: [
        {
          name: "v12-production-archive",
          passed: true,
          summary: makeV12ProductionArchiveSummary({
            v12ProductionBrokerRoleBoundaryPassed: false,
            v12ProductionBrokerFallbackBackendId: null,
          }),
        },
      ],
    })).toBe(false);
  });

  it("fails v12-production-required releases when the production proof schema is stale", () => {
    expect(computeReleaseGatePassed({
      steps: [{ status: "passed" }],
      requireV12ProductionArchive: true,
      latestArtifacts: [
        {
          name: "v12-production-archive",
          passed: true,
          summary: makeV12ProductionArchiveSummary({
            v12ProductionProofSchemaVersion: 1,
          }),
        },
      ],
    })).toBe(false);
  });

  it("fails v12-production-required releases when hosted proof is not source-bound", () => {
    expect(computeReleaseGatePassed({
      steps: [{ status: "passed" }],
      requireV12ProductionArchive: true,
      latestArtifacts: [
        {
          name: "v12-production-archive",
          passed: true,
          summary: makeV12ProductionArchiveSummary({
            v12ProductionProofSourceBound: false,
          }),
        },
      ],
    })).toBe(false);
  });

  it("fails v12-production-required releases when hosted proof was not checked in source-bound-required mode", () => {
    expect(computeReleaseGatePassed({
      steps: [{ status: "passed" }],
      requireV12ProductionArchive: true,
      latestArtifacts: [
        {
          name: "v12-production-archive",
          passed: true,
          summary: makeV12ProductionArchiveSummary({
            v12ProductionProofSourceBoundRequired: false,
          }),
        },
      ],
    })).toBe(false);
  });

  it("fails v12-production-required releases when concrete memory grounding evidence is missing", () => {
    expect(computeReleaseGatePassed({
      steps: [{ status: "passed" }],
      requireV12ProductionArchive: true,
      latestArtifacts: [
        {
          name: "v12-production-archive",
          passed: true,
          summary: makeV12ProductionArchiveSummary({
            v12ProductionConcreteMemoryGroundingPassed: false,
            v12ProductionMemoryGroundingRunCount: 0,
          }),
        },
      ],
    })).toBe(false);
  });

  it("fails v12-production-required releases when backend role boundary proof is missing", () => {
    expect(computeReleaseGatePassed({
      steps: [{ status: "passed" }],
      requireV12ProductionArchive: true,
      latestArtifacts: [
        {
          name: "v12-production-archive",
          passed: true,
          summary: makeV12ProductionArchiveSummary({
            v12ProductionBackendRoleBoundaryPassed: false,
            v12ProductionFallbackBackendId: null,
          }),
        },
      ],
    })).toBe(false);
  });

  it("fails MTP-acceleration-required releases when the paired benchmark did not pass", () => {
    expect(computeReleaseGatePassed({
      steps: [{ status: "passed" }],
      requireMtpAcceleration: true,
      latestArtifacts: [
        {
          name: "browser-runtime-bench",
          passed: false,
          summary: {
            browserBenchMtpAccelerationMode: "completed",
            browserBenchMtpAccelerationRequested: true,
            browserBenchMtpAccelerationPassed: false,
            browserBenchMtpNetSpeedupRatio: 0.9,
            browserBenchMtpAcceptanceRate: 0,
          },
        },
      ],
    })).toBe(false);
  });

  it("passes MTP-acceleration-required releases when paired benchmark proves speedup and acceptance", () => {
    expect(computeReleaseGatePassed({
      steps: [{ status: "passed" }],
      requireMtpAcceleration: true,
      latestArtifacts: [
        {
          name: "browser-runtime-bench",
          passed: true,
          summary: {
            browserBenchMtpAccelerationMode: "completed",
            browserBenchMtpAccelerationRequested: true,
            browserBenchMtpAccelerationPassed: true,
            browserBenchMtpNetSpeedupRatio: 1.08,
            browserBenchMtpAcceptanceRate: 0.5,
          },
        },
      ],
    })).toBe(true);
  });
});

function makeStrictUnlockedArtifacts(
  browserSummaryOverrides: Record<string, number | string | boolean | null> = {},
) {
  return [
    {
      name: "qwen-parity-accuracy",
      passed: true,
      summary: {
        realModelParityMode: "installed",
        realModelId: "Qwen/Qwen3-0.6B",
        realModelVocabSize: 151936,
        realModelLogitProjectionPurpose: "full_vocab_topk_logit_projection",
        realModelLogitProjectionSelectedRows: 8,
        realModelLogitProjectionFullRows: 151936,
      },
    },
    {
      name: "browser-runtime-bench",
      passed: true,
      summary: {
        browserBenchMemoryMode: "browser-vector",
        browserBenchPreviewMode: "completed",
        browserBenchPreviewRequested: true,
        browserBenchPreviewPassed: true,
        browserBenchStrictWebGpuPassed: true,
        cpuFallbackUsed: false,
        browserBenchExpectedSubstringsPassed: true,
        browserBenchExpectedExactPassed: true,
        browserBenchExpectedExactCheckCount: 1,
        browserBenchVisibleResponseQualityPassed: true,
        browserBenchStopQualityPassed: true,
        browserBenchRunawayRepetitionPassed: true,
        browserBenchMarkerOnlyResponsePassed: true,
        browserBenchTechnicalProofOnly: false,
        browserBenchProductionQualityPassed: true,
        browserBenchProductionDeployReadyPassed: true,
        browserBenchGroundedProductionReadyPassed: true,
        browserBenchMemoryGroundingRequired: true,
        browserBenchMemoryGroundingPassed: true,
        browserBenchMemoryExpectedHitPassed: true,
        browserBenchMemoryContextRebuildPassed: true,
        browserBenchMemoryAnswerOnlyPassed: true,
        browserBenchMemorySeededCorpusCount: 1024,
        browserBenchMemoryRetrievalAuditRequired: true,
        browserBenchMemoryRetrievalAuditPassed: true,
        browserBenchMemoryRetrievalAuditQueryCount: 64,
        browserBenchMemoryRecallAt1: 1,
        browserBenchRequireKvReuse: true,
        browserBenchKvReusePassed: true,
        browserBenchKvExactReuseRunCount: 1,
        browserBenchRequireKvPredictivePrefetch: true,
        browserBenchKvPredictivePrefetchPassed: true,
        browserBenchKvLowRankQuerySource: "persisted_q_rows",
        browserBenchProductionSpeedFloorPassed: true,
        browserBenchV11CommandBatchingPassed: true,
        browserBenchMtpMode: "target_only",
        ...browserSummaryOverrides,
      },
    },
    {
      name: "unlocked-verify",
      passed: true,
      summary: {
        profile: "full",
        runIsCapped: false,
        logitProjectionPurpose: "full_vocab_topk_logit_projection",
        kvDecodeReuse: true,
        tensorStorageExplicit: true,
        packedProductionReady: true,
        tensorStorageFormat: "f16-packed",
        tensorStorageDtype: "f16",
      },
    },
  ];
}

function makeV12ProductionArchiveSummary(
  overrides: Record<string, number | string | boolean | null> = {},
  options: { includeHostedRuntimeProof?: boolean } = {},
): Record<string, number | string | boolean | null> {
  const summary: Record<string, number | string | boolean | null> = {
    v12ProductionArchivePassed: true,
    v12ProductionBlockerCount: 0,
    v12ProductionSuitePassed: true,
    v12ProductionDeployBackendId: "compiled-browser-webllm",
    v12ProductionKernelLabBackendId: "unlocked-browser-transformer",
    v12ProductionFallbackBackendId: "wasm-small-core",
    v12ProductionBackendRoleBoundaryPassed: true,
    v12ProductionHostedBenchmarkProofRequired: true,
    v12ProductionHostedBenchmarkProofPassed: true,
    v12ProductionModelRegistryAligned: true,
    v12ProductionModelRegistryModelCount: 3,
    v12ProductionPublicModelOptionCount: 2,
    v12ProductionPublicDeployOptionCount: 1,
    v12ProductionPublicKernelLabOptionCount: 1,
    v12ProductionArtifactCount: 7,
    v12ProductionSuiteArtifactCount: 6,
    v12ProductionChildArtifactCount: 5,
  };

  if (options.includeHostedRuntimeProof !== false) {
    Object.assign(summary, {
      v12ProductionHostedBenchmarkRuntimeBackendId: "compiled-browser-webllm",
      v12ProductionHostedBenchmarkDeployBackendId: "compiled-browser-webllm",
      v12ProductionCompiledBackendReadyPassed: true,
      v12ProductionDeployReadyPassed: true,
      v12ProductionMemoryGroundingPassed: true,
      v12ProductionConcreteMemoryGroundingPassed: true,
      v12ProductionMemoryGroundingRunCount: 1,
      v12ProductionMemoryGroundingCaseId: "montana_capital",
      v12ProductionMemorySeededCorpusCount: 16,
      v12ProductionMemoryRetrievedCount: 1,
      v12ProductionMemoryIncludedCount: 1,
      v12ProductionMemoryExpectedMemoryIdCount: 1,
      v12ProductionMemoryExpectedHitMeanRank: 1,
      v12ProductionMemoryExpectedHitMinTopScoreMargin: 0.4,
      v12ProductionExpectedExactPassed: true,
      v12ProductionSpeedFloorPassed: true,
      v12ProductionMeanTokensPerSecond: 2.7,
      v12ProductionDirectModelFactualProofUsed: false,
      v12ProductionTechnicalProofOnly: false,
      v12ProductionCpuFallbackUsed: false,
      v12ProductionStrictWebGpuPassed: true,
      v12ProductionProofSchemaVersion: 2,
      v12ProductionProofSourceGitSha: "abc123",
      v12ProductionExpectedSourceGitSha: "abc123",
      v12ProductionProofSourceBoundRequired: true,
      v12ProductionProofSourceBound: true,
      v12ProductionBackendBrokerSelectionPassed: true,
      v12ProductionBackendBrokerTraceCount: 1,
      v12ProductionBrokerSelectedBackendId: "compiled-browser-webllm",
      v12ProductionBrokerSelectedModelId: "Qwen3-0.6B-q4f16_1-MLC",
      v12ProductionBrokerProductionRole: "production_candidate",
      v12ProductionBrokerDeployReadyCandidate: true,
      v12ProductionBrokerDeployBackendId: "compiled-browser-webllm",
      v12ProductionBrokerKernelLabBackendId: "unlocked-browser-transformer",
      v12ProductionBrokerFallbackBackendId: "wasm-small-core",
      v12ProductionBrokerFallbackBackendCount: 1,
      v12ProductionBrokerFallbackDeployReadyCandidate: false,
      v12ProductionBrokerRoleBoundaryPassed: true,
    });
  }

  return {
    ...summary,
    ...overrides,
  };
}
