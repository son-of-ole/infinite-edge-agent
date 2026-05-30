export type ReleaseGateStepStatus = "passed" | "failed" | "skipped";

export interface ReleaseGateStepStatusInput {
  status: ReleaseGateStepStatus;
}

export interface ReleaseGateLatestArtifactStatusInput {
  name: string;
  passed: boolean | null;
  summary?: Record<string, number | string | boolean | null>;
}

export type ReleaseGateProofMode =
  | "production-strict-browser-runtime"
  | "production-v12-compiled-browser-runtime"
  | "development-fixture-or-unconfigured";

export function classifyReleaseGateProof(input: {
  passed: boolean;
  strictUnlockedModel?: boolean;
  requireBrowserPreviewProof?: boolean;
  requireV12ProductionArchive?: boolean;
  releaseAllowFixtureGate?: boolean;
  unlockedAllowFixture?: string | null;
  manifestPath?: string | null;
  manifestSha256?: string | null;
  latestArtifacts?: ReleaseGateLatestArtifactStatusInput[];
}): {
  proofMode: ReleaseGateProofMode;
  productionReleaseProof: boolean;
  backendSpecificProductionProof: boolean;
  groundedAnswerQualityBrowserProof: boolean;
  cappedTechnicalSpeedProof: boolean;
  deployReadySpeedQualityProof: boolean;
  v12ProductionArchiveProof: boolean;
  strictEnv: Record<string, boolean | string | null>;
} {
  const strictUnlockedModel = input.strictUnlockedModel === true;
  const requireBrowserPreviewProof = input.requireBrowserPreviewProof === true;
  const fixtureGateAllowed = input.releaseAllowFixtureGate === true || input.unlockedAllowFixture === "true";
  const modelManifestConfigured = Boolean(input.manifestPath?.trim()) && Boolean(input.manifestSha256?.trim());
  const productionConfigured = strictUnlockedModel
    && requireBrowserPreviewProof
    && !fixtureGateAllowed
    && modelManifestConfigured;
  const hasArtifactContext = (input.latestArtifacts?.length ?? 0) > 0;
  const browserBench = input.latestArtifacts?.find((artifact) => artifact.name === "browser-runtime-bench");
  const groundedAnswerQualityBrowserProof = hasArtifactContext
    ? browserAnswerQualityProofPassed(browserBench)
    : false;
  const technicalSpeedProof = hasArtifactContext
    ? browserTechnicalSpeedProofPassed(browserBench)
    : false;
  const deployReadySpeedQualityProof = productionConfigured
    && input.passed
    && (!hasArtifactContext || strictUnlockedArtifactsPassed(input.latestArtifacts ?? []));
  const v12ProductionArchive = input.latestArtifacts?.find((artifact) => artifact.name === "v12-production-archive");
  const v12ProductionArchiveProof = input.passed && v12ProductionArchiveProofPassed(v12ProductionArchive);
  const backendSpecificProductionProof = deployReadySpeedQualityProof || v12ProductionArchiveProof;
  const proofMode: ReleaseGateProofMode = v12ProductionArchiveProof
    ? "production-v12-compiled-browser-runtime"
    : productionConfigured
      ? "production-strict-browser-runtime"
      : "development-fixture-or-unconfigured";
  return {
    proofMode,
    productionReleaseProof: backendSpecificProductionProof,
    backendSpecificProductionProof,
    groundedAnswerQualityBrowserProof,
    cappedTechnicalSpeedProof: technicalSpeedProof && !backendSpecificProductionProof,
    deployReadySpeedQualityProof,
    v12ProductionArchiveProof,
    strictEnv: {
      RELEASE_REQUIRE_UNLOCKED_MODEL: strictUnlockedModel,
      RELEASE_REQUIRE_BROWSER_PREVIEW_PROOF: requireBrowserPreviewProof,
      RELEASE_REQUIRE_V12_PRODUCTION: input.requireV12ProductionArchive === true,
      RELEASE_ALLOW_FIXTURE_GATE: input.releaseAllowFixtureGate === true,
      VITE_UNLOCKED_ALLOW_FIXTURE: input.unlockedAllowFixture ?? null,
      VITE_UNLOCKED_MODEL_MANIFEST_PATH: input.manifestPath?.trim() || null,
      VITE_UNLOCKED_MODEL_MANIFEST_SHA256: input.manifestSha256 ? "present" : null,
    },
  };
}

function v12ProductionArchiveProofPassed(artifact: ReleaseGateLatestArtifactStatusInput | undefined): boolean {
  return artifact?.passed === true
    && artifact.summary?.v12ProductionArchivePassed === true
    && Number(artifact.summary?.v12ProductionBlockerCount ?? 1) === 0
    && artifact.summary?.v12ProductionSuitePassed === true
    && artifact.summary?.v12ProductionDeployBackendId === "compiled-browser-webllm"
    && artifact.summary?.v12ProductionKernelLabBackendId === "unlocked-browser-transformer"
    && artifact.summary?.v12ProductionFallbackBackendId === "wasm-small-core"
    && artifact.summary?.v12ProductionBackendRoleBoundaryPassed === true
    && artifact.summary?.v12ProductionHostedBenchmarkProofRequired === true
    && artifact.summary?.v12ProductionHostedBenchmarkProofPassed === true
    && artifact.summary?.v12ProductionWorkflowPreflightPassed === true
    && artifact.summary?.v12ProductionModelRegistryAligned === true
    && Number(artifact.summary?.v12ProductionModelRegistryModelCount ?? 0) >= 3
    && Number(artifact.summary?.v12ProductionPublicModelOptionCount ?? 0) >= 2
    && Number(artifact.summary?.v12ProductionPublicDeployOptionCount ?? 0) === 1
    && Number(artifact.summary?.v12ProductionPublicKernelLabOptionCount ?? 0) === 1
    && Number(artifact.summary?.v12ProductionArtifactCount ?? 0) >= 8
    && artifact.summary?.v12ProductionProofSchemaVersion === 2
    && typeof artifact.summary?.v12ProductionProofSourceGitSha === "string"
    && artifact.summary.v12ProductionProofSourceGitSha.trim().length > 0
    && artifact.summary?.v12ProductionProofSourceCommitEvidencePassed === true
    && hasGpuLabelEvidence(artifact.summary, "v12Production")
    && typeof artifact.summary?.v12ProductionExpectedSourceGitSha === "string"
    && artifact.summary.v12ProductionExpectedSourceGitSha.trim().length > 0
    && artifact.summary?.v12ProductionProofSourceBoundRequired === true
    && artifact.summary?.v12ProductionProofSourceBound === true
    && Number(artifact.summary?.v12ProductionSuiteArtifactCount ?? 0) >= 7
    && Number(artifact.summary?.v12ProductionChildArtifactCount ?? 0) >= 6
    && artifact.summary?.v12ProductionHostedBenchmarkRuntimeBackendId === "compiled-browser-webllm"
    && artifact.summary?.v12ProductionHostedBenchmarkDeployBackendId === "compiled-browser-webllm"
    && artifact.summary?.v12ProductionCompiledBackendReadyPassed === true
    && artifact.summary?.v12ProductionDeployReadyPassed === true
    && artifact.summary?.v12ProductionMemoryGroundingPassed === true
    && artifact.summary?.v12ProductionConcreteMemoryGroundingPassed === true
    && Number(artifact.summary?.v12ProductionMemoryGroundingRunCount ?? 0) > 0
    && Number(artifact.summary?.v12ProductionMemorySeededCorpusCount ?? 0) > 0
    && Number(artifact.summary?.v12ProductionMemoryRetrievedCount ?? 0) > 0
    && Number(artifact.summary?.v12ProductionMemoryIncludedCount ?? 0) > 0
    && Number(artifact.summary?.v12ProductionMemoryExpectedMemoryIdCount ?? 0) > 0
    && Number(artifact.summary?.v12ProductionMemoryExpectedHitMeanRank ?? Number.POSITIVE_INFINITY) <= 1
    && artifact.summary?.v12ProductionExpectedExactPassed === true
    && artifact.summary?.v12ProductionSpeedFloorPassed === true
    && Number(artifact.summary?.v12ProductionMeanTokensPerSecond ?? 0) >= 2
    && artifact.summary?.v12ProductionDirectModelFactualProofUsed === false
    && artifact.summary?.v12ProductionTechnicalProofOnly === false
    && artifact.summary?.v12ProductionCpuFallbackUsed === false
    && artifact.summary?.v12ProductionStrictWebGpuPassed === true
    && artifact.summary?.v12ProductionBackendBrokerSelectionPassed === true
    && Number(artifact.summary?.v12ProductionBackendBrokerTraceCount ?? 0) > 0
    && artifact.summary?.v12ProductionBrokerSelectedBackendId === "compiled-browser-webllm"
    && artifact.summary?.v12ProductionBrokerProductionRole === "production_candidate"
    && artifact.summary?.v12ProductionBrokerDeployReadyCandidate === true
    && artifact.summary?.v12ProductionBrokerDeployBackendId === "compiled-browser-webllm"
    && artifact.summary?.v12ProductionBrokerKernelLabBackendId === "unlocked-browser-transformer"
    && artifact.summary?.v12ProductionBrokerFallbackBackendId === "wasm-small-core"
    && Number(artifact.summary?.v12ProductionBrokerFallbackBackendCount ?? 0) === 1
    && artifact.summary?.v12ProductionBrokerFallbackDeployReadyCandidate === false
    && artifact.summary?.v12ProductionBrokerRoleBoundaryPassed === true;
}

export function computeReleaseGatePassed(input: {
  steps: ReleaseGateStepStatusInput[];
  latestArtifacts: ReleaseGateLatestArtifactStatusInput[];
  optionalArtifactNames?: string[];
  strictUnlockedModel?: boolean;
  requireBrowserPreviewProof?: boolean;
  requireMtpAcceleration?: boolean;
  requireV12ProductionArchive?: boolean;
}): boolean {
  const optionalArtifacts = new Set(input.optionalArtifactNames ?? []);
  return input.steps.every((step) => step.status !== "failed")
    && input.latestArtifacts.every((artifact) => {
      if (artifact.passed === true) return true;
      if (artifact.passed === null && optionalArtifacts.has(artifact.name)) return true;
      return false;
    })
    && hostedBenchmarkProofArtifactsPassed(input.latestArtifacts)
    && (!input.strictUnlockedModel || strictUnlockedArtifactsPassed(input.latestArtifacts))
    && (!input.requireBrowserPreviewProof || browserPreviewProofPassed(input.latestArtifacts))
    && (!input.requireMtpAcceleration || mtpAccelerationProofPassed(input.latestArtifacts))
    && (!input.requireV12ProductionArchive || v12ProductionArchiveProofPassed(
      input.latestArtifacts.find((artifact) => artifact.name === "v12-production-archive"),
    ));
}

function hostedBenchmarkProofArtifactsPassed(artifacts: ReleaseGateLatestArtifactStatusInput[]): boolean {
  const hostedProofs = artifacts.filter((artifact) => artifact.name === "hosted-benchmark-proof");
  if (hostedProofs.length === 0) return true;
  return hostedProofs.every((artifact) =>
    artifact.passed === true
    && artifact.summary?.hostedBenchmarkProofPassed === true
    && artifact.summary?.hostedBenchmarkProofSourceBoundRequired === true
    && artifact.summary?.hostedBenchmarkProofSourceBound === true
    && artifact.summary?.hostedBenchmarkProofSourceCommitEvidencePassed === true
    && hasGpuLabelEvidence(artifact.summary, "hostedBenchmark")
    && artifact.summary?.hostedBenchmarkConcreteMemoryGroundingPassed === true
    && Number(artifact.summary?.hostedBenchmarkMemoryGroundingRunCount ?? 0) > 0
    && Number(artifact.summary?.hostedBenchmarkMemorySeededCorpusCount ?? 0) > 0
    && Number(artifact.summary?.hostedBenchmarkMemoryRetrievedCount ?? 0) > 0
    && Number(artifact.summary?.hostedBenchmarkMemoryIncludedCount ?? 0) > 0
    && Number(artifact.summary?.hostedBenchmarkMemoryExpectedMemoryIdCount ?? 0) > 0
    && Number(artifact.summary?.hostedBenchmarkMemoryExpectedHitMeanRank ?? Number.POSITIVE_INFINITY) <= 1
    && artifact.summary?.hostedBenchmarkBrokerDeployBackendId === "compiled-browser-webllm"
    && artifact.summary?.hostedBenchmarkBrokerKernelLabBackendId === "unlocked-browser-transformer"
    && artifact.summary?.hostedBenchmarkBrokerFallbackBackendId === "wasm-small-core"
    && Number(artifact.summary?.hostedBenchmarkBrokerFallbackBackendCount ?? 0) === 1
    && artifact.summary?.hostedBenchmarkBrokerFallbackDeployReadyCandidate === false
    && artifact.summary?.hostedBenchmarkBrokerRoleBoundaryPassed === true
    && typeof artifact.summary?.hostedBenchmarkProofSourceGitSha === "string"
    && artifact.summary.hostedBenchmarkProofSourceGitSha.trim().length > 0
    && typeof artifact.summary?.hostedBenchmarkExpectedSourceGitSha === "string"
    && artifact.summary.hostedBenchmarkExpectedSourceGitSha.trim().length > 0
  );
}

function hasGpuLabelEvidence(
  summary: Record<string, number | string | boolean | null> | undefined,
  prefix: "hostedBenchmark" | "v12Production",
): boolean {
  if (!summary || summary[`${prefix}GpuLabelEvidencePassed`] !== true) return false;
  return [
    `${prefix}GpuVendor`,
    `${prefix}GpuArchitecture`,
    `${prefix}GpuDevice`,
    `${prefix}GpuDescription`,
    `${prefix}WebGlRenderer`,
  ].some((key) => typeof summary[key] === "string" && summary[key].trim().length > 0);
}

function strictUnlockedArtifactsPassed(artifacts: ReleaseGateLatestArtifactStatusInput[]): boolean {
  const qwenParity = artifacts.find((artifact) => artifact.name === "qwen-parity-accuracy");
  const browserBench = artifacts.find((artifact) => artifact.name === "browser-runtime-bench");
  const unlockedVerify = artifacts.find((artifact) => artifact.name === "unlocked-verify");
  return qwenParity?.summary?.realModelParityMode === "installed"
    && qwenParity?.summary?.realModelId === "Qwen/Qwen3-0.6B"
    && Number(qwenParity?.summary?.realModelVocabSize ?? 0) === 151936
    && qwenParity?.summary?.realModelLogitProjectionPurpose === "full_vocab_topk_logit_projection"
    && Number(qwenParity?.summary?.realModelLogitProjectionFullRows ?? 0) === 151936
    && Number(qwenParity?.summary?.realModelLogitProjectionSelectedRows ?? 0) > 0
    && browserBench?.summary?.browserBenchMemoryMode === "browser-vector"
    && strictBrowserRuntimeProofPassed(browserBench)
    && unlockedVerify?.summary?.profile === "full"
    && unlockedVerify?.summary?.runIsCapped === false
    && unlockedVerify?.summary?.logitProjectionPurpose === "full_vocab_topk_logit_projection"
    && unlockedVerify?.summary?.kvDecodeReuse === true
    && unlockedVerify?.summary?.tensorStorageExplicit === true
    && unlockedVerify?.summary?.packedProductionReady === true
    && unlockedVerify?.summary?.tensorStorageFormat === "f16-packed"
    && unlockedVerify?.summary?.tensorStorageDtype === "f16";
}

function strictBrowserRuntimeProofPassed(browserBench: ReleaseGateLatestArtifactStatusInput | undefined): boolean {
  return browserAnswerQualityProofPassed(browserBench)
    && browserTechnicalSpeedProofPassed(browserBench)
    && browserBench?.summary?.browserBenchTechnicalProofOnly === false
    && browserBench?.summary?.browserBenchProductionQualityPassed === true
    && browserBench?.summary?.browserBenchProductionDeployReadyPassed === true;
}

function browserAnswerQualityProofPassed(browserBench: ReleaseGateLatestArtifactStatusInput | undefined): boolean {
  return browserPreviewProofPassed(browserBench ? [browserBench] : [])
    && browserBench?.summary?.browserBenchExpectedSubstringsPassed === true
    && browserBench?.summary?.browserBenchExpectedExactPassed === true
    && Number(browserBench?.summary?.browserBenchExpectedExactCheckCount ?? 0) > 0
    && browserBench?.summary?.browserBenchVisibleResponseQualityPassed === true
    && browserBench?.summary?.browserBenchStopQualityPassed === true
    && browserBench?.summary?.browserBenchRunawayRepetitionPassed === true
    && browserBench?.summary?.browserBenchMarkerOnlyResponsePassed === true
    && browserBench?.summary?.browserBenchGroundedProductionReadyPassed === true
    && browserBench?.summary?.browserBenchDirectModelFactualProofUsed !== true
    && browserBench?.summary?.browserBenchMemoryGroundingRequired === true
    && browserBench?.summary?.browserBenchMemoryGroundingPassed === true
    && browserBench?.summary?.browserBenchMemoryExpectedHitPassed === true
    && browserBench?.summary?.browserBenchMemoryContextRebuildPassed === true
    && browserBench?.summary?.browserBenchMemoryAnswerOnlyPassed === true
    && browserBench?.summary?.browserBenchMemoryRetrievalAuditRequired === true
    && browserBench?.summary?.browserBenchMemoryRetrievalAuditPassed === true
    && Number(browserBench?.summary?.browserBenchMemoryRetrievalAuditQueryCount ?? 0) >= 64
    && Number(browserBench?.summary?.browserBenchMemoryRecallAt1 ?? 0) >= 1
    && Number(browserBench?.summary?.browserBenchMemorySeededCorpusCount ?? 0) >= 1024;
}

function browserTechnicalSpeedProofPassed(browserBench: ReleaseGateLatestArtifactStatusInput | undefined): boolean {
  return browserPreviewProofPassed(browserBench ? [browserBench] : [])
    && browserBench?.summary?.browserBenchStrictWebGpuPassed === true
    && browserBench?.summary?.cpuFallbackUsed === false
    && browserBench?.summary?.browserBenchV11CommandBatchingPassed === true
    && browserBench?.summary?.browserBenchRequireKvReuse === true
    && browserBench?.summary?.browserBenchKvReusePassed === true
    && Number(browserBench?.summary?.browserBenchKvExactReuseRunCount ?? 0) > 0
    && browserBench?.summary?.browserBenchRequireKvPredictivePrefetch === true
    && browserBench?.summary?.browserBenchKvPredictivePrefetchPassed === true
    && browserBench?.summary?.browserBenchKvLowRankQuerySource === "persisted_q_rows"
    && browserBench?.summary?.browserBenchProductionSpeedFloorPassed === true;
}

function browserPreviewProofPassed(artifacts: ReleaseGateLatestArtifactStatusInput[]): boolean {
  const browserBench = artifacts.find((artifact) => artifact.name === "browser-runtime-bench");
  return browserBench?.passed === true
    && browserBench.summary?.browserBenchPreviewMode === "completed"
    && browserBench.summary?.browserBenchPreviewRequested === true
    && browserBench.summary?.browserBenchPreviewPassed === true;
}

function mtpAccelerationProofPassed(artifacts: ReleaseGateLatestArtifactStatusInput[]): boolean {
  const browserBench = artifacts.find((artifact) => artifact.name === "browser-runtime-bench");
  return browserBench?.passed === true
    && browserBench.summary?.browserBenchMtpAccelerationMode === "completed"
    && browserBench.summary?.browserBenchMtpAccelerationRequested === true
    && browserBench.summary?.browserBenchMtpAccelerationPassed === true
    && Number(browserBench.summary?.browserBenchMtpNetSpeedupRatio ?? 0) > 1
    && Number(browserBench.summary?.browserBenchMtpAcceptanceRate ?? 0) > 0;
}
