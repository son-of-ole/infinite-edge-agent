export type ReleaseGateArtifactSummary = Record<string, number | string | boolean | null>;

export interface ReleaseGateArtifactInput {
  summary?: Record<string, number | string | boolean | null>;
  metrics?: Record<string, number | string | boolean | null>;
  thresholds?: Record<string, number | string | boolean | null>;
  requestedBackendPreference?: string;
  runtimeProfile?: {
    activeProfile?: string;
    runIsCapped?: boolean;
  };
  webGpuCoverage?: {
    cpuFallbackUsed?: boolean;
    mlpLayersByBackend?: Partial<Record<"webgpu" | "cpu_reference" | "mixed" | "unknown", number>>;
    logitProjection?: { backend?: string; purpose?: string; selectedRows?: number; fullRows?: number };
    decodeProjectionBackends?: {
      oProjection?: Partial<Record<"webgpu" | "cpu_reference" | "mixed" | "unknown", number>>;
    };
    attentionBackends?: {
      prefill?: Partial<Record<"webgpu" | "cpu_reference" | "mixed" | "unknown", number>>;
      decode?: Partial<Record<"webgpu" | "cpu_reference" | "mixed" | "unknown", number>>;
      packedHeads?: Partial<Record<"webgpu" | "cpu_reference" | "mixed" | "unknown", number>>;
      incompletePackedHeadProofs?: number;
    };
  };
  kvPersistence?: {
    decodeReuse?: boolean;
    mode?: string;
  } | null;
  tensorStorage?: {
    explicit?: boolean;
    format?: string;
    dtype?: string;
    shardKind?: string;
    byteWidth?: number | null;
    productionTarget?: string;
    runtimeRepresentation?: string;
    packedRuntimeCompute?: boolean;
    packedProductionReady?: boolean;
  } | null;
  mtp?: {
    mode?: string;
    verifierStrategy?: string;
    verifiedTokenCount?: number;
    targetDecodeCalls?: number;
    acceptanceRate?: number;
  } | null;
  suites?: Array<{ name: string; metrics?: Record<string, number | string | boolean | null>; gates?: Array<{ name: string; passed: boolean }> }>;
  gates?: Array<{ name: string; passed: boolean }>;
  browserPreview?: {
    mode?: string;
    requested?: boolean;
    passed?: boolean;
    reason?: string;
    summary?: Record<string, number | string | boolean | null>;
  };
}

export function summarizeReleaseGateArtifact(parsed: ReleaseGateArtifactInput): ReleaseGateArtifactSummary {
  const summary: ReleaseGateArtifactSummary = {};
  if (parsed.requestedBackendPreference) summary.requestedBackendPreference = parsed.requestedBackendPreference;
  if (parsed.runtimeProfile) {
    summary.profile = parsed.runtimeProfile.activeProfile ?? "unknown";
    summary.runIsCapped = parsed.runtimeProfile.runIsCapped ?? true;
  }
  if (parsed.webGpuCoverage) {
    summary.cpuFallbackUsed = parsed.webGpuCoverage.cpuFallbackUsed ?? false;
    summary.logitProjectionBackend = parsed.webGpuCoverage.logitProjection?.backend ?? "unknown";
    summary.logitProjectionPurpose = parsed.webGpuCoverage.logitProjection?.purpose ?? "unknown";
    summary.logitProjectionSelectedRows = parsed.webGpuCoverage.logitProjection?.selectedRows ?? 0;
    summary.logitProjectionFullRows = parsed.webGpuCoverage.logitProjection?.fullRows ?? 0;
    summary.mlpWebGpuLayers = parsed.webGpuCoverage.mlpLayersByBackend?.webgpu ?? 0;
    summary.mlpCpuReferenceLayers = parsed.webGpuCoverage.mlpLayersByBackend?.cpu_reference ?? 0;
    summary.decodeOProjectionWebGpuLayers = parsed.webGpuCoverage.decodeProjectionBackends?.oProjection?.webgpu ?? 0;
    summary.decodeOProjectionCpuReferenceLayers = parsed.webGpuCoverage.decodeProjectionBackends?.oProjection?.cpu_reference ?? 0;
    summary.prefillAttentionWebGpuLayers = parsed.webGpuCoverage.attentionBackends?.prefill?.webgpu ?? 0;
    summary.prefillAttentionCpuReferenceLayers = parsed.webGpuCoverage.attentionBackends?.prefill?.cpu_reference ?? 0;
    summary.decodeAttentionWebGpuLayers = parsed.webGpuCoverage.attentionBackends?.decode?.webgpu ?? 0;
    summary.decodeAttentionCpuReferenceLayers = parsed.webGpuCoverage.attentionBackends?.decode?.cpu_reference ?? 0;
    summary.packedHeadWebGpu = parsed.webGpuCoverage.attentionBackends?.packedHeads?.webgpu ?? 0;
    summary.packedHeadCpuReference = parsed.webGpuCoverage.attentionBackends?.packedHeads?.cpu_reference ?? 0;
    summary.incompletePackedHeadProofs = parsed.webGpuCoverage.attentionBackends?.incompletePackedHeadProofs ?? 0;
  }
  if (parsed.kvPersistence) {
    summary.kvDecodeReuse = parsed.kvPersistence.decodeReuse ?? false;
    summary.kvPersistenceMode = parsed.kvPersistence.mode ?? "unknown";
  }
  if (parsed.tensorStorage) {
    summary.tensorStorageExplicit = parsed.tensorStorage.explicit ?? false;
    summary.tensorStorageFormat = parsed.tensorStorage.format ?? "unknown";
    summary.tensorStorageDtype = parsed.tensorStorage.dtype ?? "unknown";
    summary.tensorStorageShardKind = parsed.tensorStorage.shardKind ?? "unknown";
    summary.tensorStorageByteWidth = parsed.tensorStorage.byteWidth ?? null;
    summary.tensorStorageProductionTarget = parsed.tensorStorage.productionTarget ?? "unknown";
    summary.tensorStorageRuntimeRepresentation = parsed.tensorStorage.runtimeRepresentation ?? "unknown";
    summary.packedRuntimeCompute = parsed.tensorStorage.packedRuntimeCompute ?? false;
    summary.packedProductionReady = parsed.tensorStorage.packedProductionReady ?? false;
  }
  if (parsed.mtp) {
    summary.mtpMode = parsed.mtp.mode ?? "none";
    summary.mtpVerifierStrategy = parsed.mtp.verifierStrategy ?? "unknown";
    summary.mtpVerifiedTokenCount = parsed.mtp.verifiedTokenCount ?? 0;
    summary.mtpTargetDecodeCalls = parsed.mtp.targetDecodeCalls ?? 0;
    summary.mtpAcceptanceRate = parsed.mtp.acceptanceRate ?? 0;
  }
  if (parsed.metrics) {
    for (const key of [
      "elapsedSearchMs",
      "packedPromptTokenCount",
      "requiredAnchorIncluded",
      "provenanceComplete",
      "chunkCount",
      "estimatedTokens",
      "mountedIframeCount",
      "requestedDeploymentPreset",
      "requestedSdkMode",
      "requestedMemoryMode",
      "requestedSidecarDisabled",
      "noSecretQueryParams",
    ]) {
      if (key in parsed.metrics) summary[key] = parsed.metrics[key];
    }
    const sdkFields = {
      requestedDeploymentPreset: "sdkRequestedDeploymentPreset",
      requestedSdkMode: "sdkRequestedMode",
      requestedMemoryMode: "sdkRequestedMemoryMode",
      requestedSidecarDisabled: "sdkRequestedSidecarDisabled",
      noSecretQueryParams: "sdkNoSecretQueryParams",
    } as const;
    for (const [sourceKey, targetKey] of Object.entries(sdkFields)) {
      if (sourceKey in parsed.metrics) summary[targetKey] = parsed.metrics[sourceKey];
    }
  }
  if (parsed.summary) {
    const sharedRuntimeFields = [
      "sharedRuntimeReadinessPassed",
      "sharedRuntimeBlockerCount",
      "sharedRuntimeCoveredBackendCount",
      "sharedRuntimeDeployBackendId",
      "sharedRuntimeKernelLabBackendId",
      "sharedRuntimeMemoryProviderCount",
      "sharedRuntimeContextTraceRequired",
      "sharedRuntimeContextTraceBeforeGeneration",
      "sharedRuntimeTracePersistedAfterGeneration",
      "sharedRuntimeBackendProfilePassedToPlan",
    ] as const;
    for (const key of sharedRuntimeFields) {
      if (key in parsed.summary) summary[key] = parsed.summary[key];
    }
    const backendReadinessFields = [
      "backendReadinessMatrixPassed",
      "backendReadinessBlockerCount",
      "backendReadinessBackendCount",
      "backendReadinessDeployBackendId",
      "backendReadinessProductionCandidateCount",
      "backendReadinessDeployReadyCount",
      "backendReadinessResearchBackendCount",
      "backendReadinessKernelLabBackendId",
      "backendReadinessCompiledHostedProfilePassed",
    ] as const;
    for (const key of backendReadinessFields) {
      if (key in parsed.summary) summary[key] = parsed.summary[key];
    }
    const hostedProfileFields = [
      "hostedProfilePassed",
      "hostedProfileBlockerCount",
      "hostedProfileWarningCount",
      "hostedProfileBackend",
      "hostedProfileDefaultModel",
      "hostedProfileCompiledWebLlmEnabled",
      "hostedProfileRequireUnlockedRuntime",
      "hostedProfileMtpProductionDisabled",
      "hostedProfileTelemetryEnabled",
      "hostedProfileTelemetryStorage",
      "hostedProfileTelemetryAdminProtected",
      "hostedProfileTelemetryRateLimited",
      "hostedProfileBenchmarkBackend",
      "hostedProfileBenchmarkMemoryGrounding",
      "hostedProfileBenchmarkMemoryGroundingProfile",
      "hostedProfileBenchmarkExpectedExact",
      "hostedProfileBenchmarkRequiresSubmitTelemetry",
    ] as const;
    for (const key of hostedProfileFields) {
      if (key in parsed.summary) summary[key] = parsed.summary[key];
    }
    for (const key of [
      "maxAbsError",
      "tokenParityPassed",
      "multiTokenDecodePassed",
      "retrievalGroundedPassed",
      "realModelParityMode",
    ]) {
      if (key in parsed.summary) summary[key] = parsed.summary[key];
    }
    const browserBenchmarkFields = {
      profile: "browserBenchProfile",
      memoryMode: "browserBenchMemoryMode",
      promptCount: "browserBenchPromptCount",
      meanInitLoadMs: "browserBenchMeanInitLoadMs",
      meanPrefillMs: "browserBenchMeanPrefillMs",
      meanTimeToFirstTokenMs: "browserBenchMeanTtftMs",
      meanDecodeLatencyMs: "browserBenchMeanDecodeLatencyMs",
      meanTokensPerSecond: "browserBenchMeanTokensPerSecond",
      generationTokenBudgetUsed: "browserBenchGenerationTokenBudgetUsed",
      browserPreviewMode: "browserBenchPreviewMode",
      browserPreviewRequested: "browserBenchPreviewRequested",
      browserPreviewPassed: "browserBenchPreviewPassed",
      browserPreviewRequired: "browserBenchPreviewRequired",
      browserPreviewProductionDeployReadyPassed: "browserBenchPreviewProductionDeployReadyPassed",
      browserPreviewGroundedProductionReadyPassed: "browserBenchPreviewGroundedProductionReadyPassed",
      browserPreviewMemoryGroundingPassed: "browserBenchPreviewMemoryGroundingPassed",
      browserPreviewMemoryGeneratedParaphrasePassed: "browserBenchPreviewMemoryGeneratedParaphrasePassed",
      browserPreviewReason: "browserBenchPreviewReason",
      memoryQueryMode: "browserBenchMemoryQueryMode",
      memoryGroundingRequired: "browserBenchMemoryGroundingRequired",
      memoryGroundingPassed: "browserBenchMemoryGroundingPassed",
      memoryGroundingCoveragePassed: "browserBenchMemoryGroundingCoveragePassed",
      memoryExpectedHitPassed: "browserBenchMemoryExpectedHitPassed",
      memoryContextRebuildPassed: "browserBenchMemoryContextRebuildPassed",
      memoryAnswerOnlyPassed: "browserBenchMemoryAnswerOnlyPassed",
      memorySeededCorpusCount: "browserBenchMemorySeededCorpusCount",
      memoryRetrievedCount: "browserBenchMemoryRetrievedCount",
      memoryIncludedCount: "browserBenchMemoryIncludedCount",
      memoryRetrievalAuditRequired: "browserBenchMemoryRetrievalAuditRequired",
      memoryRetrievalAuditPassed: "browserBenchMemoryRetrievalAuditPassed",
      memoryRetrievalAuditQueryCount: "browserBenchMemoryRetrievalAuditQueryCount",
      memoryRetrievalAuditTop1CorrectCount: "browserBenchMemoryRetrievalAuditTop1CorrectCount",
      memoryRecallAt1: "browserBenchMemoryRecallAt1",
      memoryMrr: "browserBenchMemoryMrr",
      memoryGeneratedParaphraseRequired: "browserBenchMemoryGeneratedParaphraseRequired",
      memoryGeneratedParaphrasePassed: "browserBenchMemoryGeneratedParaphrasePassed",
      memoryGeneratedParaphraseQueryCount: "browserBenchMemoryGeneratedParaphraseQueryCount",
      memoryGeneratedParaphraseTop1CorrectCount: "browserBenchMemoryGeneratedParaphraseTop1CorrectCount",
      memoryGeneratedParaphraseRecallAt1: "browserBenchMemoryGeneratedParaphraseRecallAt1",
      memoryGeneratedParaphraseMrr: "browserBenchMemoryGeneratedParaphraseMrr",
      memoryMinTopScoreMargin: "browserBenchMemoryMinTopScoreMargin",
      memoryMeanExpectedHitRank: "browserBenchMemoryMeanExpectedHitRank",
      stopQualityPassed: "browserBenchStopQualityPassed",
      stopQualityFailureCount: "browserBenchStopQualityFailureCount",
      runawayRepetitionPassed: "browserBenchRunawayRepetitionPassed",
      runawayRepetitionFailureCount: "browserBenchRunawayRepetitionFailureCount",
      markerOnlyResponsePassed: "browserBenchMarkerOnlyResponsePassed",
      markerOnlyResponseFailureCount: "browserBenchMarkerOnlyResponseFailureCount",
      generationStopReason: "browserBenchGenerationStopReason",
      visibleResponseQualityPassed: "browserBenchVisibleResponseQualityPassed",
      expectedSubstringsPassed: "browserBenchExpectedSubstringsPassed",
      expectedExactCheckCount: "browserBenchExpectedExactCheckCount",
      expectedExactPassCount: "browserBenchExpectedExactPassCount",
      expectedExactPassed: "browserBenchExpectedExactPassed",
      technicalProofOnly: "browserBenchTechnicalProofOnly",
      productionSpeedFloorPassed: "browserBenchProductionSpeedFloorPassed",
      productionSpeedFloorTokensPerSecond: "browserBenchProductionSpeedFloorTokensPerSecond",
      productionLayerCoverageRequired: "browserBenchProductionLayerCoverageRequired",
      productionLayerCoveragePassed: "browserBenchProductionLayerCoveragePassed",
      productionLayerVisitsPerToken: "browserBenchProductionLayerVisitsPerToken",
      productionMinLayerVisitsPerToken: "browserBenchProductionMinLayerVisitsPerToken",
      diagnosticCappedLayerRun: "browserBenchDiagnosticCappedLayerRun",
      primarySpeedBottleneck: "browserBenchPrimarySpeedBottleneck",
      decodeSubmitCount: "browserBenchDecodeSubmitCount",
      decodeSubmitCountPerToken: "browserBenchDecodeSubmitCountPerToken",
      v11CommandBatchingPassed: "browserBenchV11CommandBatchingPassed",
      decodeDispatchCount: "browserBenchDecodeDispatchCount",
      decodeDispatchCountPerToken: "browserBenchDecodeDispatchCountPerToken",
      decodeDispatchCountPerLayerPerToken: "browserBenchDecodeDispatchCountPerLayerPerToken",
      fusedPackedQkvLayerCount: "browserBenchFusedPackedQkvLayerCount",
      fusedQkvNormRopeKvAppendLayerCount: "browserBenchFusedQkvNormRopeKvAppendLayerCount",
      fusedOneTokenAttentionLayerCount: "browserBenchFusedOneTokenAttentionLayerCount",
      fusedResidualRmsNormLayerCount: "browserBenchFusedResidualRmsNormLayerCount",
      fusedMlpLayerCount: "browserBenchFusedMlpLayerCount",
      fusedFullLayerCount: "browserBenchFusedFullLayerCount",
      fusedLayerCoverage: "browserBenchFusedLayerCoverage",
      parityRecordCount: "browserBenchParityRecordCount",
      parityPassedCount: "browserBenchParityPassedCount",
      parityFailedCount: "browserBenchParityFailedCount",
      productionQualityPassed: "browserBenchProductionQualityPassed",
      productionDeployReadyPassed: "browserBenchProductionDeployReadyPassed",
      groundedProductionReadyPassed: "browserBenchGroundedProductionReadyPassed",
      directModelFactualProofUsed: "browserBenchDirectModelFactualProofUsed",
      requireKvReuse: "browserBenchRequireKvReuse",
      kvReusePassed: "browserBenchKvReusePassed",
      kvExactReuseRunCount: "browserBenchKvExactReuseRunCount",
      requireKvPredictivePrefetch: "browserBenchRequireKvPredictivePrefetch",
      kvPredictivePrefetchPassed: "browserBenchKvPredictivePrefetchPassed",
      kvPredictivePrefetchRunCount: "browserBenchKvPredictivePrefetchRunCount",
      kvLowRankQuerySource: "browserBenchKvLowRankQuerySource",
      mtpMode: "browserBenchMtpMode",
      mtpAcceptanceRate: "browserBenchMtpAcceptanceRate",
      mtpAccelerationMode: "browserBenchMtpAccelerationMode",
      mtpAccelerationRequested: "browserBenchMtpAccelerationRequested",
      mtpAccelerationPassed: "browserBenchMtpAccelerationPassed",
      mtpNetSpeedupRatio: "browserBenchMtpNetSpeedupRatio",
      mtpTargetOnlyMeanTokensPerSecond: "browserBenchMtpTargetOnlyMeanTokensPerSecond",
      mtpDraftVerifyMeanTokensPerSecond: "browserBenchMtpDraftVerifyMeanTokensPerSecond",
      mtpAccelerationMinAcceptanceRate: "browserBenchMtpAccelerationMinAcceptanceRate",
      mtpAccelerationMinNetSpeedup: "browserBenchMtpAccelerationMinNetSpeedup",
      mtpAccelerationFailureCount: "browserBenchMtpAccelerationFailureCount",
      strictWebGpuRequired: "browserBenchStrictWebGpuRequired",
      strictWebGpuPassed: "browserBenchStrictWebGpuPassed",
      strictWebGpuGateCount: "browserBenchStrictWebGpuGateCount",
      strictWebGpuFailureCount: "browserBenchStrictWebGpuFailureCount",
      blockingThresholdFailures: "browserBenchBlockingThresholdFailures",
    } as const;
    for (const [sourceKey, targetKey] of Object.entries(browserBenchmarkFields)) {
      if (sourceKey in parsed.summary) summary[targetKey] = parsed.summary[sourceKey];
    }
  }
  if (parsed.browserPreview) {
    summary.browserBenchPreviewMode = parsed.browserPreview.mode ?? summary.browserBenchPreviewMode ?? "unknown";
    summary.browserBenchPreviewRequested = parsed.browserPreview.requested ?? summary.browserBenchPreviewRequested ?? false;
    summary.browserBenchPreviewPassed = parsed.browserPreview.passed ?? summary.browserBenchPreviewPassed ?? false;
    summary.browserBenchPreviewReason = parsed.browserPreview.reason ?? summary.browserBenchPreviewReason ?? "";
    if (parsed.browserPreview.mode === "completed" && parsed.browserPreview.summary) {
      if ("productionDeployReadyPassed" in parsed.browserPreview.summary) {
        summary.browserBenchPreviewProductionDeployReadyPassed = parsed.browserPreview.summary.productionDeployReadyPassed;
      }
      if ("groundedProductionReadyPassed" in parsed.browserPreview.summary) {
        summary.browserBenchPreviewGroundedProductionReadyPassed = parsed.browserPreview.summary.groundedProductionReadyPassed;
      }
      if ("memoryGroundingPassed" in parsed.browserPreview.summary) {
        summary.browserBenchPreviewMemoryGroundingPassed = parsed.browserPreview.summary.memoryGroundingPassed;
      }
      if ("memoryGeneratedParaphrasePassed" in parsed.browserPreview.summary) {
        summary.browserBenchPreviewMemoryGeneratedParaphrasePassed = parsed.browserPreview.summary.memoryGeneratedParaphrasePassed;
      }
      const browserPreviewProofFields = {
        memoryQueryMode: "browserBenchMemoryQueryMode",
        memoryGroundingRequired: "browserBenchMemoryGroundingRequired",
        memoryGroundingPassed: "browserBenchMemoryGroundingPassed",
        memoryGroundingCoveragePassed: "browserBenchMemoryGroundingCoveragePassed",
        memoryExpectedHitPassed: "browserBenchMemoryExpectedHitPassed",
        memoryContextRebuildPassed: "browserBenchMemoryContextRebuildPassed",
        memoryAnswerOnlyPassed: "browserBenchMemoryAnswerOnlyPassed",
        memorySeededCorpusCount: "browserBenchMemorySeededCorpusCount",
        memoryRetrievedCount: "browserBenchMemoryRetrievedCount",
        memoryIncludedCount: "browserBenchMemoryIncludedCount",
        memoryRetrievalAuditRequired: "browserBenchMemoryRetrievalAuditRequired",
        memoryRetrievalAuditPassed: "browserBenchMemoryRetrievalAuditPassed",
        memoryRetrievalAuditQueryCount: "browserBenchMemoryRetrievalAuditQueryCount",
        memoryRetrievalAuditTop1CorrectCount: "browserBenchMemoryRetrievalAuditTop1CorrectCount",
        memoryRecallAt1: "browserBenchMemoryRecallAt1",
        memoryMrr: "browserBenchMemoryMrr",
        memoryGeneratedParaphraseRequired: "browserBenchMemoryGeneratedParaphraseRequired",
        memoryGeneratedParaphrasePassed: "browserBenchMemoryGeneratedParaphrasePassed",
        memoryGeneratedParaphraseQueryCount: "browserBenchMemoryGeneratedParaphraseQueryCount",
        memoryGeneratedParaphraseTop1CorrectCount: "browserBenchMemoryGeneratedParaphraseTop1CorrectCount",
        memoryGeneratedParaphraseRecallAt1: "browserBenchMemoryGeneratedParaphraseRecallAt1",
        memoryGeneratedParaphraseMrr: "browserBenchMemoryGeneratedParaphraseMrr",
        memoryMinTopScoreMargin: "browserBenchMemoryMinTopScoreMargin",
        memoryMeanExpectedHitRank: "browserBenchMemoryMeanExpectedHitRank",
        stopQualityPassed: "browserBenchStopQualityPassed",
        stopQualityFailureCount: "browserBenchStopQualityFailureCount",
        runawayRepetitionPassed: "browserBenchRunawayRepetitionPassed",
        runawayRepetitionFailureCount: "browserBenchRunawayRepetitionFailureCount",
        markerOnlyResponsePassed: "browserBenchMarkerOnlyResponsePassed",
        markerOnlyResponseFailureCount: "browserBenchMarkerOnlyResponseFailureCount",
        generationStopReason: "browserBenchGenerationStopReason",
        visibleResponseQualityPassed: "browserBenchVisibleResponseQualityPassed",
        expectedSubstringsPassed: "browserBenchExpectedSubstringsPassed",
        expectedExactCheckCount: "browserBenchExpectedExactCheckCount",
        expectedExactPassCount: "browserBenchExpectedExactPassCount",
        expectedExactPassed: "browserBenchExpectedExactPassed",
        technicalProofOnly: "browserBenchTechnicalProofOnly",
        productionLayerCoverageRequired: "browserBenchProductionLayerCoverageRequired",
        productionLayerCoveragePassed: "browserBenchProductionLayerCoveragePassed",
        productionLayerVisitsPerToken: "browserBenchProductionLayerVisitsPerToken",
        productionMinLayerVisitsPerToken: "browserBenchProductionMinLayerVisitsPerToken",
        diagnosticCappedLayerRun: "browserBenchDiagnosticCappedLayerRun",
        productionQualityPassed: "browserBenchProductionQualityPassed",
        productionDeployReadyPassed: "browserBenchProductionDeployReadyPassed",
        groundedProductionReadyPassed: "browserBenchGroundedProductionReadyPassed",
        directModelFactualProofUsed: "browserBenchDirectModelFactualProofUsed",
        primarySpeedBottleneck: "browserBenchPrimarySpeedBottleneck",
        decodeSubmitCount: "browserBenchDecodeSubmitCount",
        decodeSubmitCountPerToken: "browserBenchDecodeSubmitCountPerToken",
        v11CommandBatchingPassed: "browserBenchV11CommandBatchingPassed",
        decodeDispatchCount: "browserBenchDecodeDispatchCount",
        decodeDispatchCountPerToken: "browserBenchDecodeDispatchCountPerToken",
        decodeDispatchCountPerLayerPerToken: "browserBenchDecodeDispatchCountPerLayerPerToken",
        fusedPackedQkvLayerCount: "browserBenchFusedPackedQkvLayerCount",
        fusedQkvNormRopeKvAppendLayerCount: "browserBenchFusedQkvNormRopeKvAppendLayerCount",
        fusedOneTokenAttentionLayerCount: "browserBenchFusedOneTokenAttentionLayerCount",
        fusedResidualRmsNormLayerCount: "browserBenchFusedResidualRmsNormLayerCount",
        fusedMlpLayerCount: "browserBenchFusedMlpLayerCount",
        fusedFullLayerCount: "browserBenchFusedFullLayerCount",
        fusedLayerCoverage: "browserBenchFusedLayerCoverage",
        parityRecordCount: "browserBenchParityRecordCount",
        parityPassedCount: "browserBenchParityPassedCount",
        parityFailedCount: "browserBenchParityFailedCount",
        strictWebGpuPassed: "browserBenchStrictWebGpuPassed",
        cpuFallbackUsed: "cpuFallbackUsed",
        requireKvReuse: "browserBenchRequireKvReuse",
        kvReusePassed: "browserBenchKvReusePassed",
        kvExactReuseRunCount: "browserBenchKvExactReuseRunCount",
        requireKvPredictivePrefetch: "browserBenchRequireKvPredictivePrefetch",
        kvPredictivePrefetchPassed: "browserBenchKvPredictivePrefetchPassed",
        kvPredictivePrefetchRunCount: "browserBenchKvPredictivePrefetchRunCount",
        kvLowRankQuerySource: "browserBenchKvLowRankQuerySource",
        mtpMode: "browserBenchMtpMode",
      } as const;
      for (const [sourceKey, targetKey] of Object.entries(browserPreviewProofFields)) {
        if (sourceKey in parsed.browserPreview.summary) summary[targetKey] = parsed.browserPreview.summary[sourceKey];
      }
    }
  }
  if (parsed.thresholds?.maxAbsError !== undefined) {
    summary.maxAbsErrorThreshold = parsed.thresholds.maxAbsError;
  }
  for (const suite of parsed.suites ?? []) {
    if (suite.name === "memory-recall" && suite.metrics) {
      summary.recallAt10 = suite.metrics.recallAt10;
      summary.pinnedRecallAt5 = suite.metrics.pinnedRecallAt5;
      summary.provenanceComplete = suite.metrics.provenanceComplete;
    }
    if (suite.name === "sidecar-behavior" && suite.metrics) {
      summary.sidecarSkipped = suite.metrics.skipped ?? false;
      summary.sidecarReason = suite.metrics.reason ?? "";
    }
    if (suite.name === "context-rebuild" && suite.metrics) {
      summary.contextEstimatedTokens = suite.metrics.estimatedTokens;
    }
    if (suite.name === "real-model-parity" && suite.metrics) {
      summary.realModelId = suite.metrics.modelId ?? "unknown";
      summary.realModelVocabSize = suite.metrics.vocabSize ?? 0;
      summary.realModelLogitProjectionPurpose = suite.metrics.logitProjectionPurpose ?? "unknown";
      summary.realModelLogitProjectionSelectedRows = suite.metrics.logitProjectionSelectedRows ?? 0;
      summary.realModelLogitProjectionFullRows = suite.metrics.logitProjectionFullRows ?? 0;
    }
  }
  const gateCount = (parsed.gates?.length ?? 0) + (parsed.suites ?? []).reduce((sum, suite) => sum + (suite.gates?.length ?? 0), 0);
  const failedGateCount = (parsed.gates ?? []).filter((gate) => !gate.passed).length
    + (parsed.suites ?? []).reduce((sum, suite) => sum + (suite.gates ?? []).filter((gate) => !gate.passed).length, 0);
  summary.gateCount = gateCount;
  summary.failedGateCount = failedGateCount;
  return summary;
}
