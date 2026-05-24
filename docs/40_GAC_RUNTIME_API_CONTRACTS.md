# 40 — GAC Runtime API Contracts

## Purpose

This document defines API contracts between the model, Context Runtime, LanceDB Memory Engine, GAC Runtime, SSA planner, and KVSwap planner.

These contracts are implementation targets for TypeScript interfaces and service endpoints.

## Principles

- All memory actions are structured.
- All writes are auditable.
- Raw memory is immutable.
- Derived memory has lineage.
- Model-proposed actions require policy validation.
- Contracts must be deterministic where possible.

## API: IngestMemory

### Request

```json
{
  "tenantId": "tenant_01",
  "cellId": "cell_01",
  "sessionId": "session_01",
  "sourceType": "chat",
  "sourceUri": "chat://session_01/turn_42",
  "text": "SSA is first-class from the start, not a research adapter.",
  "memoryKind": "decision",
  "importance": 0.95,
  "identityRiskHint": 0.9
}
```

### Response

```json
{
  "rawMemoryId": "mem_01",
  "embeddingId": "emb_01",
  "pinCreated": true,
  "pinId": "pin_01",
  "queuedForConsolidation": true
}
```

## API: ComputeClusterMetrics

### Request

```json
{
  "clusterId": "cluster_01",
  "clusterVersion": 3,
  "theta": 0.85
}
```

### Response

```json
{
  "clusterMetricId": "metric_01",
  "meanDistance": 0.22,
  "maxDistance": 0.48,
  "effectiveDimension": 7.4,
  "rho": 0.63,
  "identityErrorBound": 0.31,
  "riskClass": "spread"
}
```

## API: RouteConsolidation

### Request

```json
{
  "clusterId": "cluster_01",
  "clusterMetricId": "metric_01",
  "budget": 4,
  "policyId": "policy_default"
}
```

### Response

```json
{
  "decision": "medoid_plus_residuals",
  "representativeBudget": 4,
  "requiresRawLineage": true,
  "pinMemberIds": ["mem_02"],
  "reasonCodes": ["spread_cluster", "identity_risk_high"]
}
```

## API: BuildContextPack

### Request

```json
{
  "tenantId": "tenant_01",
  "cellId": "cell_01",
  "sessionId": "session_01",
  "query": "Update the architecture docs so SSA is first-class.",
  "mode": "balanced",
  "tokenBudget": 64000,
  "requireSourceGrounding": true
}
```

### Response

```json
{
  "contextPackId": "pack_01",
  "items": [
    {
      "kind": "identity_pin",
      "rawMemoryId": "mem_01",
      "text": "SSA is first-class from the start, not a research adapter.",
      "sourceUri": "chat://session_01/turn_42"
    }
  ],
  "traceId": "trace_01"
}
```

## API: ProposeModelMemoryAction

### Request

```json
{
  "sessionId": "session_01",
  "modelId": "local-model",
  "actionType": "pin_memory",
  "targetIds": ["mem_01"],
  "arguments": {
    "pinReason": "architecture_decision",
    "pinStrength": 0.95
  },
  "confidence": 0.91
}
```

### Response

```json
{
  "actionId": "action_01",
  "approved": true,
  "executed": true,
  "resultIds": ["pin_01"],
  "policyNotes": []
}
```

## API: GetSsaRoutingMetadata

### Request

```json
{
  "contextPackId": "pack_01",
  "queryId": "query_01"
}
```

### Response

```json
{
  "blocks": [
    {
      "blockId": "block_01",
      "memoryClass": "PINNED_EXACT",
      "identityRisk": 0.96,
      "pinStrength": 1.0,
      "sourceTrust": 0.9,
      "mustAttend": true
    }
  ]
}
```

## API: GetKvSwapPriority

### Request

```json
{
  "contextPackId": "pack_01",
  "activeTaskId": "task_01"
}
```

### Response

```json
{
  "priorities": [
    {
      "blockId": "block_01",
      "tier": "PIN_HOT",
      "priorityScore": 0.98,
      "reasonCodes": ["identity_pin", "active_task"]
    }
  ]
}
```

## Error model

All APIs return:

- `errorCode`
- `message`
- `retryable`
- `policyViolation`
- `traceId`

Common error codes:

- `RAW_MEMORY_NOT_FOUND`
- `MISSING_LINEAGE`
- `PIN_POLICY_REJECTED`
- `CONSOLIDATION_LOCKED`
- `EMBEDDING_MODEL_MISMATCH`
- `CONTEXT_BUDGET_EXCEEDED`
- `USER_DELETION_IN_PROGRESS`

## Acceptance gates

- Every API has structured request/response types.
- Every write returns trace IDs.
- Every representative can be traced to raw memory.
- Model actions require policy approval.
- SSA and KVSwap can consume GAC metadata without direct database coupling.
