export type ReleaseGateStepStatus = "passed" | "failed" | "skipped";

export interface ReleaseGateStepStatusInput {
  status: ReleaseGateStepStatus;
}

export interface ReleaseGateLatestArtifactStatusInput {
  name: string;
  passed: boolean | null;
  summary?: Record<string, number | string | boolean | null>;
}

export type ReleaseGateProofMode = "production-strict-browser-runtime" | "development-fixture-or-unconfigured";

export function classifyReleaseGateProof(input: {
  passed: boolean;
  strictUnlockedModel?: boolean;
  requireBrowserPreviewProof?: boolean;
  releaseAllowFixtureGate?: boolean;
  unlockedAllowFixture?: string | null;
  manifestPath?: string | null;
  manifestSha256?: string | null;
  latestArtifacts?: ReleaseGateLatestArtifactStatusInput[];
}): {
  proofMode: ReleaseGateProofMode;
  productionReleaseProof: boolean;
  groundedAnswerQualityBrowserProof: boolean;
  cappedTechnicalSpeedProof: boolean;
  deployReadySpeedQualityProof: boolean;
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
  return {
    proofMode: productionConfigured ? "production-strict-browser-runtime" : "development-fixture-or-unconfigured",
    productionReleaseProof: deployReadySpeedQualityProof,
    groundedAnswerQualityBrowserProof,
    cappedTechnicalSpeedProof: technicalSpeedProof && !deployReadySpeedQualityProof,
    deployReadySpeedQualityProof,
    strictEnv: {
      RELEASE_REQUIRE_UNLOCKED_MODEL: strictUnlockedModel,
      RELEASE_REQUIRE_BROWSER_PREVIEW_PROOF: requireBrowserPreviewProof,
      RELEASE_ALLOW_FIXTURE_GATE: input.releaseAllowFixtureGate === true,
      VITE_UNLOCKED_ALLOW_FIXTURE: input.unlockedAllowFixture ?? null,
      VITE_UNLOCKED_MODEL_MANIFEST_PATH: input.manifestPath?.trim() || null,
      VITE_UNLOCKED_MODEL_MANIFEST_SHA256: input.manifestSha256 ? "present" : null,
    },
  };
}

export function computeReleaseGatePassed(input: {
  steps: ReleaseGateStepStatusInput[];
  latestArtifacts: ReleaseGateLatestArtifactStatusInput[];
  optionalArtifactNames?: string[];
  strictUnlockedModel?: boolean;
  requireBrowserPreviewProof?: boolean;
  requireMtpAcceleration?: boolean;
}): boolean {
  const optionalArtifacts = new Set(input.optionalArtifactNames ?? []);
  return input.steps.every((step) => step.status !== "failed")
    && input.latestArtifacts.every((artifact) => {
      if (artifact.passed === true) return true;
      if (artifact.passed === null && optionalArtifacts.has(artifact.name)) return true;
      return false;
    })
    && (!input.strictUnlockedModel || strictUnlockedArtifactsPassed(input.latestArtifacts))
    && (!input.requireBrowserPreviewProof || browserPreviewProofPassed(input.latestArtifacts))
    && (!input.requireMtpAcceleration || mtpAccelerationProofPassed(input.latestArtifacts));
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
