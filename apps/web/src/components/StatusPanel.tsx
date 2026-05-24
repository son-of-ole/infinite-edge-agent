import type { MemoryChunk, RuntimeFeatureStatus, RuntimeTrace } from "@infinite-edge-agent/core";
import type { RetrievedMemoryDetail } from "../lib/agent/localAgent";
import type { BrowserMetric } from "../lib/runtime/browserMetrics";
import type { BrowserModelCacheSnapshot } from "../lib/runtime/modelCache";
import type { ProductionReadinessReport } from "../lib/runtime/productionReadiness";
import type { KVSwapPersistenceHealth } from "../lib/runtime/kvSwapPersistence";

interface StatusPanelProps {
  status: string;
  memoryMode: string;
  modelId: string;
  backend: string;
  unlockedRuntime: UnlockedRuntimeStatus;
  embeddingDevice: string;
  memoryIds: string[];
  features: RuntimeFeatureStatus[];
  lastTrace: RuntimeTrace | null;
  readiness: ProductionReadinessReport;
  retrievedMemory: RetrievedMemoryDetail[];
  memoryRows: MemoryChunk[];
  traceRows: RuntimeTrace[];
  inspectorStatus: string;
  onRefreshInspector: () => void;
  metrics: BrowserMetric[];
  cacheSnapshot: BrowserModelCacheSnapshot | null;
  kvPersistenceHealth: KVSwapPersistenceHealth;
  onRefreshCache: () => void;
  onClearModelCaches: () => void;
}

interface UnlockedRuntimeStatus {
  manifestPath: string;
  manifestSha256: string;
  manifestFormat: string;
  allowFixture: boolean;
  backendPreference: string;
  requireWebGpu: boolean;
  runtimeProfile: string;
  qwenThinkingMode: "disabled" | "enabled";
  interactiveLimits: {
    maxRuntimePromptTokens: number | null;
    maxRuntimeLayers: number | null;
    logitCandidateLimit: number | null;
    logitTopK: number;
    logitTileRows: number;
    maxGenerationTokens: number;
  };
}

interface PredictiveRuntimeStatus {
  planId: string;
  confidence: number | null;
  predictedRetrievals: number;
  contextBranches: number;
  kvHotPages: number;
  sparseBlocks: number;
  mtpBranches: number;
  prefetchBlockIds: number;
  reasons: string[];
}

export function StatusPanel({
  status,
  memoryMode,
  modelId,
  backend,
  unlockedRuntime,
  embeddingDevice,
  memoryIds,
  features,
  lastTrace,
  readiness,
  retrievedMemory,
  memoryRows,
  traceRows,
  inspectorStatus,
  onRefreshInspector,
  metrics,
  cacheSnapshot,
  kvPersistenceHealth,
  onRefreshCache,
  onClearModelCaches
}: StatusPanelProps) {
  const predictiveRuntime = getPredictiveRuntimeStatus(lastTrace);

  return (
    <aside className="status-panel">
      <h2>Runtime</h2>
      <dl>
        <dt>Status</dt>
        <dd>{status}</dd>
        <dt>Model</dt>
        <dd>{modelId}</dd>
        <dt>Backend</dt>
        <dd>{backend}</dd>
        {backend === "unlocked-browser-transformer" && (
          <>
            <dt>Qwen asset</dt>
            <dd>{unlockedRuntime.manifestPath || "fixture weights only"}</dd>
            <dt>Manifest SHA</dt>
            <dd>{formatSha(unlockedRuntime.manifestSha256)}</dd>
            <dt>Manifest</dt>
            <dd>{unlockedRuntime.manifestFormat || "not configured"}</dd>
            <dt>Fixture</dt>
            <dd>{unlockedRuntime.allowFixture ? "enabled" : "disabled"}</dd>
            <dt>Runtime profile</dt>
            <dd>{unlockedRuntime.runtimeProfile}</dd>
            <dt>Qwen thinking</dt>
            <dd>{unlockedRuntime.qwenThinkingMode}</dd>
            <dt>Chat budget</dt>
            <dd>{formatInteractiveLimits(unlockedRuntime.interactiveLimits)}</dd>
            <dt>Kernel target</dt>
            <dd>{unlockedRuntime.requireWebGpu ? "webgpu required" : `${unlockedRuntime.backendPreference || "auto"} with fallback`}</dd>
          </>
        )}
        <dt>Memory</dt>
        <dd>{memoryMode || "not initialized"}</dd>
        <dt>Embeddings</dt>
        <dd>{embeddingDevice || "not initialized"}</dd>
      </dl>
      <h3>Production readiness</h3>
      {readiness.ready ? (
        <p className="ready-state">Ready</p>
      ) : (
        <ul className="readiness-list">
          {readiness.blockers.map((blocker) => (
            <li key={blocker}>{blocker}</li>
          ))}
        </ul>
      )}
      {readiness.warnings.length > 0 && (
        <ul className="readiness-list warning">
          {readiness.warnings.map((warning) => (
            <li key={warning}>{warning}</li>
          ))}
        </ul>
      )}
      <h3>Tier-0 features</h3>
      {features.length === 0 ? (
        <p className="muted">Initialized after first turn.</p>
      ) : (
        <ul className="feature-list">
          {features.map((feature) => (
            <li key={feature.name}>
              <strong>{feature.name}</strong>
              <span>{feature.state}</span>
              <em>{feature.mode}</em>
              {feature.reason && <em>{feature.reason}</em>}
              {feature.impact && <em>{feature.impact}</em>}
            </li>
          ))}
        </ul>
      )}
      <h3>KV persistence</h3>
      <dl>
        <dt>Mode</dt>
        <dd>{kvPersistenceHealth.enabled ? kvPersistenceHealth.mode : "disabled"}</dd>
        <dt>Namespace</dt>
        <dd>{kvPersistenceHealth.namespace}</dd>
        <dt>Storage</dt>
        <dd>{formatBytes(kvPersistenceHealth.usageBytes)} used / {formatBytes(kvPersistenceHealth.quotaBytes)} quota</dd>
        <dt>Decode reuse</dt>
        <dd>{kvPersistenceHealth.decodeReuse ? "enabled" : "not enabled"}</dd>
        <dt>Last op</dt>
        <dd>
          {kvPersistenceHealth.lastOperation
            ? `${kvPersistenceHealth.lastOperation.operation} ${kvPersistenceHealth.lastOperation.ok ? "ok" : "failed"}`
            : "none"}
        </dd>
        <dt>Blocks</dt>
        <dd>{kvPersistenceHealth.lastOperation?.blockIds.length ?? 0}</dd>
        {kvPersistenceHealth.lastOperation?.reason && (
          <>
            <dt>Health</dt>
            <dd>{kvPersistenceHealth.lastOperation.reason}</dd>
          </>
        )}
      </dl>
      <h3>Last trace</h3>
      {lastTrace ? (
        <dl>
          <dt>Trace</dt>
          <dd>{lastTrace.traceId}</dd>
          <dt>Created</dt>
          <dd>{new Date(lastTrace.createdAt).toLocaleTimeString()}</dd>
        </dl>
      ) : (
        <p className="muted">No trace yet.</p>
      )}
      <h3>Predictive runtime</h3>
      {predictiveRuntime ? (
        <>
          <dl>
            <dt>Plan</dt>
            <dd>{predictiveRuntime.planId}</dd>
            <dt>Confidence</dt>
            <dd>{formatConfidence(predictiveRuntime.confidence)}</dd>
            <dt>Predictions</dt>
            <dd>
              retrieval {predictiveRuntime.predictedRetrievals} / kv {predictiveRuntime.kvHotPages} / sparse {predictiveRuntime.sparseBlocks} / mtp {predictiveRuntime.mtpBranches}
            </dd>
            <dt>Branches</dt>
            <dd>{predictiveRuntime.contextBranches}</dd>
            <dt>Prefetch</dt>
            <dd>{predictiveRuntime.prefetchBlockIds}</dd>
          </dl>
          {predictiveRuntime.reasons.length > 0 && (
            <ul className="inspector-list">
              {predictiveRuntime.reasons.map((reason) => (
                <li key={reason}>{reason}</li>
              ))}
            </ul>
          )}
        </>
      ) : (
        <p className="muted">Initialized after first advanced-runtime turn.</p>
      )}
      <h3>Retrieved memory IDs</h3>
      {memoryIds.length === 0 ? (
        <p className="muted">None yet.</p>
      ) : (
        <ul className="memory-list">
          {memoryIds.map((id) => (
            <li key={id}>{id}</li>
          ))}
        </ul>
      )}
      <h3>Retrieved details</h3>
      {retrievedMemory.length === 0 ? (
        <p className="muted">No retrieval details yet.</p>
      ) : (
        <ul className="inspector-list">
          {retrievedMemory.map((memory) => (
            <li key={memory.id}>
              <strong>{memory.id}</strong>
              <span>score {memory.score}</span>
              <p>{memory.textPreview}</p>
              <em>
                {memory.source}
                {memory.role ? ` / ${memory.role}` : ""} / {memory.sessionId}
              </em>
              {(memory.rawMemoryIds.length > 0 || memory.representativeId || memory.identityPinId) && (
                <em>
                  raw {memory.rawMemoryIds.join(", ") || "none"}
                  {memory.representativeId ? ` / rep ${memory.representativeId}` : ""}
                  {memory.identityPinId ? ` / pin ${memory.identityPinId}` : ""}
                </em>
              )}
            </li>
          ))}
        </ul>
      )}
      <div className="panel-heading-row">
        <h3>Memory inspector</h3>
        <button className="secondary small-button" onClick={onRefreshInspector}>Refresh</button>
      </div>
      {inspectorStatus && <p className="muted">{inspectorStatus}</p>}
      <h4>Rows</h4>
      {memoryRows.length === 0 ? (
        <p className="muted">No rows loaded.</p>
      ) : (
        <ul className="inspector-list">
          {memoryRows.map((row) => (
            <li key={row.id}>
              <strong>{row.id}</strong>
              <span>{row.tags.join(", ") || "untagged"}</span>
              <p>{row.text.slice(0, 280)}</p>
              <em>{row.sessionId} / {new Date(row.createdAt).toLocaleString()}</em>
            </li>
          ))}
        </ul>
      )}
      <h4>Traces</h4>
      {traceRows.length === 0 ? (
        <p className="muted">No traces loaded.</p>
      ) : (
        <ul className="inspector-list">
          {traceRows.map((trace) => (
            <li key={trace.traceId}>
              <strong>{trace.traceId}</strong>
              <span>{trace.backend} / {trace.modelId}</span>
              <em>{trace.sessionId} / {new Date(trace.createdAt).toLocaleString()}</em>
            </li>
          ))}
        </ul>
      )}
      <h3>Local metrics</h3>
      {metrics.length === 0 ? (
        <p className="muted">No timing metrics yet.</p>
      ) : (
        <ul className="metric-list">
          {metrics.map((metric) => (
            <li key={`${metric.name}-${metric.at}`}>
              <span>{metric.name}</span>
              <strong>{metric.valueMs} ms</strong>
            </li>
          ))}
        </ul>
      )}
      <div className="panel-heading-row">
        <h3>Model cache</h3>
        <button className="secondary small-button" onClick={onRefreshCache}>Refresh</button>
      </div>
      <button className="secondary cache-clear-button" onClick={onClearModelCaches}>Clear model caches</button>
      {cacheSnapshot ? (
        <>
          <dl>
            <dt>Storage</dt>
            <dd>{formatBytes(cacheSnapshot.usageBytes)} used / {formatBytes(cacheSnapshot.quotaBytes)} quota</dd>
            <dt>Checked</dt>
            <dd>{new Date(cacheSnapshot.checkedAt).toLocaleTimeString()}</dd>
          </dl>
          {cacheSnapshot.entries.length === 0 ? (
            <p className="muted">No browser cache entries reported.</p>
          ) : (
            <ul className="inspector-list">
              {cacheSnapshot.entries.map((entry) => (
                <li key={`${entry.kind}-${entry.name}`}>
                  <strong>{entry.name}</strong>
                  <span>{entry.kind}</span>
                  <em>{entry.modelRelated ? "model/runtime cache candidate" : "not selected for model-cache clear"}</em>
                </li>
              ))}
            </ul>
          )}
        </>
      ) : (
        <p className="muted">Cache not inspected yet.</p>
      )}
    </aside>
  );
}

function formatBytes(value: number | undefined): string {
  if (value === undefined) return "unknown";
  if (value < 1024) return `${value} B`;
  const mib = value / (1024 * 1024);
  if (mib < 1024) return `${mib.toFixed(1)} MiB`;
  return `${(mib / 1024).toFixed(2)} GiB`;
}

function formatSha(value: string): string {
  if (!value) return "not configured";
  if (value.length <= 16) return value;
  return `${value.slice(0, 12)}...${value.slice(-8)}`;
}

function formatInteractiveLimits(limits: UnlockedRuntimeStatus["interactiveLimits"]): string {
  return [
    `prompt ${formatLimit(limits.maxRuntimePromptTokens)}`,
    `layers ${formatLimit(limits.maxRuntimeLayers)}`,
    limits.logitCandidateLimit === null
      ? `logits top-${limits.logitTopK} / tile ${limits.logitTileRows}`
      : `logits candidates ${limits.logitCandidateLimit}`,
    `tokens ${limits.maxGenerationTokens}`,
  ].join(" / ");
}

function formatLimit(value: number | null): string {
  return value === null ? "full" : String(value);
}

function getPredictiveRuntimeStatus(trace: RuntimeTrace | null): PredictiveRuntimeStatus | null {
  const predictive = trace?.runtime.predictive;
  if (!isRecord(predictive)) return null;
  const cacheBudget = isRecord(predictive.cacheBudget) ? predictive.cacheBudget : {};
  return {
    planId: typeof predictive.planId === "string" ? predictive.planId : "unknown",
    confidence: typeof predictive.confidence === "number" ? predictive.confidence : null,
    predictedRetrievals: arrayLength(predictive.predictedRetrievals),
    contextBranches: arrayLength(predictive.contextBranches),
    kvHotPages: arrayLength(predictive.kvHotPages),
    sparseBlocks: arrayLength(predictive.sparseBlocks),
    mtpBranches: arrayLength(predictive.mtpBranches),
    prefetchBlockIds: arrayLength(cacheBudget.prefetchBlockIds),
    reasons: Array.isArray(predictive.reasons)
      ? predictive.reasons.filter((reason): reason is string => typeof reason === "string")
      : [],
  };
}

function arrayLength(value: unknown): number {
  return Array.isArray(value) ? value.length : 0;
}

function formatConfidence(value: number | null): string {
  return value === null ? "unknown" : `${Math.round(value * 100)}%`;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}
