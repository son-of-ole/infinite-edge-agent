import {
  type FusedDecodeFeatureFlags,
  type FusedDecodePlan,
  type FusedDecodeRollbackFlags,
  type FusedDecodeStage,
  type FusedDecodeStagePlan,
  mergeFusedDecodeFlags,
} from "./types";

const STAGES: Array<{
  stage: FusedDecodeStage;
  flag: keyof FusedDecodeFeatureFlags;
  rollbackFlag: keyof FusedDecodeRollbackFlags;
}> = [
  { stage: "command_batching", flag: "commandBatching", rollbackFlag: "disableCommandBatching" },
  { stage: "packed_qkv_projection", flag: "packedQkvProjection", rollbackFlag: "disablePackedQkvProjection" },
  { stage: "qkv_norm_rope_kv_append", flag: "qkvNormRopeKvAppend", rollbackFlag: "disableQkvNormRopeKvAppend" },
  { stage: "one_token_attention", flag: "oneTokenAttention", rollbackFlag: "disableOneTokenAttention" },
  { stage: "residual_rmsnorm", flag: "residualRmsNorm", rollbackFlag: "disableResidualRmsNorm" },
  { stage: "swiglu_mlp", flag: "swigluMlp", rollbackFlag: "disableSwiGluMlp" },
  { stage: "full_layer_plan", flag: "fullLayerPlan", rollbackFlag: "disableFullLayerPlan" },
];

export function buildFusedDecodePlan(input: {
  flags?: Partial<FusedDecodeFeatureFlags>;
  rollback?: FusedDecodeRollbackFlags;
  parityPassedStages?: FusedDecodeStage[];
} = {}): FusedDecodePlan {
  const flags = mergeFusedDecodeFlags(input.flags);
  const parity = new Set(input.parityPassedStages ?? []);
  const rollback = input.rollback ?? {};

  const stages: FusedDecodeStagePlan[] = STAGES.map((item) => {
    const requested = flags[item.flag] === true;
    const disabledByRollback = rollback[item.rollbackFlag] === true;
    const blockedByParity = flags.requireParityBeforeEnable === true
      && item.stage !== "command_batching"
      && requested
      && !parity.has(item.stage);
    const enabled = requested && !disabledByRollback && !blockedByParity;
    return {
      stage: item.stage,
      enabled,
      rollbackFlag: item.rollbackFlag,
      reason: !requested
        ? "feature_flag_disabled"
        : disabledByRollback
          ? "rollback_flag_disabled"
          : blockedByParity
            ? "parity_required_before_enable"
            : "enabled",
    };
  });

  return {
    flags,
    stages,
    enabledStages: stages.filter((stage) => stage.enabled).map((stage) => stage.stage),
    disabledStages: stages.filter((stage) => !stage.enabled).map((stage) => stage.stage),
  };
}

export function fusedStageEnabled(plan: FusedDecodePlan, stage: FusedDecodeStage): boolean {
  return plan.enabledStages.includes(stage);
}
