import {
  buildRuntimeFeatureCapabilitySnapshot,
  createDefaultRuntimeFeatureRegistry,
  rebuildStartupContextSnapshot,
  type ChatMessage,
  type ContextPackTraceRecord,
  type InferenceBackendBrokerSelection,
  type MemoryDeleteOptions,
  type MemoryChunk,
  type MemoryProviderMode,
  type MemoryStore,
  type RuntimeFeatureStatus,
  type RuntimeTrace
} from "@infinite-edge-agent/core";
import { useEffect, useMemo, useRef, useState } from "react";
import {
  AGENT_CONFIG,
  CHAT_LOGIT_CANDIDATE_LIMIT,
  CHAT_LOGIT_TILE_ROWS,
  CHAT_LOGIT_TOP_K,
  CHAT_MAX_GENERATION_TOKENS,
  CHAT_MAX_RUNTIME_LAYERS,
  CHAT_MAX_RUNTIME_PROMPT_TOKENS,
  COMPILED_WEBLLM_ENABLED,
  DEFAULT_DEVICE_PROFILE,
  EMBEDDING_PREFER_WEBGPU,
  DEFAULT_LLM_BACKEND,
  DEFAULT_MODEL,
  KVSWAP_PERSISTENCE_CLEAR_ON_INIT,
  KVSWAP_PERSISTENCE_ENABLED,
  KVSWAP_PERSISTENCE_MAX_BLOCKS,
  KVSWAP_PERSISTENCE_MAX_BYTES,
  KVSWAP_PERSISTENCE_PREFER_OPFS,
  ALLOW_MEMORY_FALLBACK,
  HAS_PUBLIC_REMOTE_MEMORY_TOKEN,
  MEMORY_PROVIDER,
  MEMORY_SERVER_ENABLED,
  MEMORY_SERVER_URL,
  MEMORY_CELL_ID,
  MEMORY_TENANT_ID,
  MTP_DISABLE_WHEN_LATENCY_WORSE,
  MTP_DRAFT_LAYER_COUNT,
  MTP_DRAFT_MODEL_ID,
  MTP_ENABLED,
  MTP_MIN_ACCEPTANCE_RATE,
  MTP_NUM_SPECULATIVE_TOKENS,
  PRODUCTION_MODE,
  QWEN_THINKING_MODE,
  REMOTE_MEMORY_CELL_ID,
  REMOTE_MEMORY_CREDENTIALS,
  REMOTE_MEMORY_TENANT_ID,
  REMOTE_MEMORY_TOKEN,
  REMOTE_MEMORY_URL,
  REQUIRE_UNLOCKED_RUNTIME,
  REQUIRE_WEBGPU_KERNELS,
  SYSTEM_PROMPT,
  UNLOCKED_ALLOW_FIXTURE,
  UNLOCKED_BACKEND_PREFERENCE,
  UNLOCKED_MANIFEST_FORMAT,
  UNLOCKED_MODEL_MANIFEST_PATH,
  UNLOCKED_MODEL_MANIFEST_SHA256,
  UNLOCKED_RUNTIME_PROFILE,
  makeBrowserMtpClientOptions,
  makeBackendProfile,
  makeModelProfile
} from "./config";
import { LOCAL_MODEL_OPTIONS } from "./config/models";
import { readBrowserEmbedConfig, resolveEmbedMemoryProvider } from "./embedConfig";
import { ChatMessageView } from "./components/ChatMessageView";
import { StatusPanel } from "./components/StatusPanel";
import { LocalAgent, type AgentDocumentInput, type RetrievedMemoryDetail } from "./lib/agent/localAgent";
import { getOrCreateSessionId, makeMessageId } from "./lib/agent/session";
import { EmbeddingClient } from "./lib/embedding/embeddingClient";
import { CompiledWebLlmClient } from "./lib/llm/compiledWebLlmClient";
import type { ChatClient } from "./lib/llm/types";
import type { UnlockedBrowserTransformerClientOptions } from "./lib/llm/unlockedBrowserTransformerClient";
import {
  getBrowserBackendRegistryEntry,
  selectBrowserBackend,
  type BrowserBackendSelection,
  type BrowserBackendTask,
} from "./lib/runtime/backendBroker";
import { makeBrowserMetric, timed, type BrowserMetric } from "./lib/runtime/browserMetrics";
import type { KVSwapPersistenceHealth } from "./lib/runtime/kvSwapPersistence";
import { clearBrowserModelCaches, inspectBrowserModelCache, type BrowserModelCacheSnapshot } from "./lib/runtime/modelCache";
import { evaluateProductionReadiness, markInitializationFailure, type ProductionReadinessReport } from "./lib/runtime/productionReadiness";
import { createMemoryStore } from "./lib/storage/hybridMemoryClient";
import { exportMemoryBundle, importMemoryBundle, type MemoryExportBundle } from "./lib/storage/memoryBundle";

export function App() {
  const embedConfig = useMemo(() => readBrowserEmbedConfig(), []);
  const effectiveMemoryProvider = useMemo(() => resolveEmbedMemoryProvider({
    embed: embedConfig,
    configuredProvider: MEMORY_PROVIDER,
    remoteMemoryUrl: REMOTE_MEMORY_URL,
  }), [embedConfig]);
  const [status, setStatus] = useState("Not initialized");
  const [input, setInput] = useState("");
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [isGenerating, setIsGenerating] = useState(false);
  const [isInitializing, setIsInitializing] = useState(false);
  const [isUploadingDocuments, setIsUploadingDocuments] = useState(false);
  const [modelId, setModelId] = useState(DEFAULT_MODEL);
  const [llmBackend, setLlmBackend] = useState(DEFAULT_LLM_BACKEND);
  const [memoryMode, setMemoryMode] = useState("");
  const [embeddingDevice, setEmbeddingDevice] = useState("");
  const [memoryIds, setMemoryIds] = useState<string[]>([]);
  const [loadProgress, setLoadProgress] = useState("");
  const [runtimeFeatures, setRuntimeFeatures] = useState<RuntimeFeatureStatus[]>(() => createDefaultRuntimeFeatureRegistry().list());
  const [lastTrace, setLastTrace] = useState<RuntimeTrace | null>(null);
  const [retrievedMemory, setRetrievedMemory] = useState<RetrievedMemoryDetail[]>([]);
  const [memoryRows, setMemoryRows] = useState<MemoryChunk[]>([]);
  const [traceRows, setTraceRows] = useState<RuntimeTrace[]>([]);
  const [inspectorStatus, setInspectorStatus] = useState("");
  const [metrics, setMetrics] = useState<BrowserMetric[]>([]);
  const [cacheSnapshot, setCacheSnapshot] = useState<BrowserModelCacheSnapshot | null>(null);
  const [kvPersistenceHealth, setKvPersistenceHealth] = useState<KVSwapPersistenceHealth>({
    enabled: KVSWAP_PERSISTENCE_ENABLED,
    mode: KVSWAP_PERSISTENCE_ENABLED ? "memory" : "disabled",
    namespace: `${MEMORY_TENANT_ID}:${MEMORY_CELL_ID}:pending`,
    decodeReuse: false,
  });
  const [deleteSessionId, setDeleteSessionId] = useState("");
  const [deleteTags, setDeleteTags] = useState("");
  const [readiness, setReadiness] = useState<ProductionReadinessReport>(() =>
    evaluateProductionReadiness({
      memoryProvider: effectiveMemoryProvider,
      allowMemoryFallback: ALLOW_MEMORY_FALLBACK,
      llmBackend: DEFAULT_LLM_BACKEND,
      unlockedModelManifestPath: UNLOCKED_MODEL_MANIFEST_PATH,
      unlockedModelManifestSha256: UNLOCKED_MODEL_MANIFEST_SHA256,
      unlockedManifestFormat: UNLOCKED_MANIFEST_FORMAT,
      unlockedAllowFixture: UNLOCKED_ALLOW_FIXTURE,
      unlockedBackendPreference: UNLOCKED_BACKEND_PREFERENCE,
      requireWebGpuKernels: REQUIRE_WEBGPU_KERNELS,
      requireUnlockedRuntime: REQUIRE_UNLOCKED_RUNTIME,
      compiledBackendAdapterAvailable: COMPILED_WEBLLM_ENABLED,
      remoteMemoryUrl: REMOTE_MEMORY_URL,
      remoteMemoryTenantId: REMOTE_MEMORY_TENANT_ID,
      remoteMemoryCellId: REMOTE_MEMORY_CELL_ID,
      hasPublicRemoteMemoryToken: HAS_PUBLIC_REMOTE_MEMORY_TOKEN,
      production: PRODUCTION_MODE,
    })
  );

  const sessionId = useMemo(() => getOrCreateSessionId(embedConfig.sessionId), [embedConfig.sessionId]);
  const agentRef = useRef<LocalAgent | null>(null);
  const llmRef = useRef<ChatClient | null>(null);
  const initializingRef = useRef(false);
  const memoryRef = useRef<MemoryStore | null>(null);
  const memoryModeRef = useRef<MemoryProviderMode | "">("");
  const importInputRef = useRef<HTMLInputElement | null>(null);
  const documentInputRef = useRef<HTMLInputElement | null>(null);
  const tenantId = embedConfig.tenantId ?? MEMORY_TENANT_ID;
  const cellId = embedConfig.cellId ?? MEMORY_CELL_ID;

  useEffect(() => {
    const flushPendingKvPersistence = () => {
      void llmRef.current?.flushKvPersistence?.();
    };
    const flushWhenHidden = () => {
      if (document.visibilityState === "hidden") flushPendingKvPersistence();
    };
    window.addEventListener("pagehide", flushPendingKvPersistence);
    document.addEventListener("visibilitychange", flushWhenHidden);
    return () => {
      window.removeEventListener("pagehide", flushPendingKvPersistence);
      document.removeEventListener("visibilitychange", flushWhenHidden);
      const llm = llmRef.current;
      llmRef.current = null;
      void llm?.dispose?.();
    };
  }, []);

  function addMetric(name: string, valueMs: number) {
    setMetrics((existing) => [makeBrowserMetric(name, valueMs), ...existing].slice(0, 18));
  }

  async function initialize() {
    if (initializingRef.current || agentRef.current) return;
    initializingRef.current = true;
    setIsInitializing(true);

    try {
      const readinessReport = evaluateProductionReadiness({
        memoryProvider: effectiveMemoryProvider,
        allowMemoryFallback: ALLOW_MEMORY_FALLBACK,
        llmBackend,
        unlockedModelManifestPath: UNLOCKED_MODEL_MANIFEST_PATH,
        unlockedModelManifestSha256: UNLOCKED_MODEL_MANIFEST_SHA256,
        unlockedManifestFormat: UNLOCKED_MANIFEST_FORMAT,
        unlockedAllowFixture: UNLOCKED_ALLOW_FIXTURE,
        unlockedBackendPreference: UNLOCKED_BACKEND_PREFERENCE,
        requireWebGpuKernels: REQUIRE_WEBGPU_KERNELS,
        requireUnlockedRuntime: REQUIRE_UNLOCKED_RUNTIME,
        compiledBackendAdapterAvailable: COMPILED_WEBLLM_ENABLED,
        remoteMemoryUrl: REMOTE_MEMORY_URL,
        remoteMemoryTenantId: REMOTE_MEMORY_TENANT_ID,
        remoteMemoryCellId: REMOTE_MEMORY_CELL_ID,
        hasPublicRemoteMemoryToken: HAS_PUBLIC_REMOTE_MEMORY_TOKEN,
        production: PRODUCTION_MODE,
      });
      setReadiness(readinessReport);
      if (!readinessReport.ready) {
        setStatus(`Blocked: ${readinessReport.blockers.join(" ")}`);
        return;
      }

      try {
        setStatus("Initializing embedding worker...");
        const embeddings = new EmbeddingClient();
        const embeddingInfo = await timed("embedding_init_ms", { addMetric }, () => embeddings.init({
          modelId: AGENT_CONFIG.embeddingModelId,
          preferWebGPU: EMBEDDING_PREFER_WEBGPU
        }));
        setEmbeddingDevice(`${embeddingInfo.modelId} on ${embeddingInfo.device}`);

        const { store, mode } = await timed("memory_store_init_ms", { addMetric }, () => ensureMemoryStore());
        await llmRef.current?.dispose?.();
        llmRef.current = null;
        agentRef.current = null;
        setStatus("Rebuilding startup context...");
        const startupSnapshot = await timed("startup_rebuild_ms", { addMetric }, async () => {
          const chunks = hasMemoryChunkList(store)
            ? await store.listMemoryChunks({ limit: 50, tenantId, cellId })
            : [];
          const runtimeTraces = hasRuntimeTraceList(store)
            ? await store.listRuntimeTraces({ limit: 20, tenantId, cellId })
            : [];
          const contextPackTraces = hasContextPackTraceList(store)
            ? await store.listContextPackTraces({ limit: 20, tenantId, cellId })
            : [];
          return rebuildStartupContextSnapshot({
            tenantId,
            cellId,
            sessionId,
            memoryChunks: chunks,
            runtimeTraces,
            contextPackTraces,
          });
        });
        setInspectorStatus(
          `Startup rebuilt ${startupSnapshot.memoryCount} memory row${startupSnapshot.memoryCount === 1 ? "" : "s"}, `
          + `${startupSnapshot.identityPinIds.length} pin${startupSnapshot.identityPinIds.length === 1 ? "" : "s"}, `
          + `${startupSnapshot.contextPackTraceCount} context trace${startupSnapshot.contextPackTraceCount === 1 ? "" : "s"}.`
        );

        const brokerSelection = resolveBrowserAnswerBackendSelection({ backend: llmBackend, modelId });
        setStatus("Loading local language model...");
        const llm = await timed("model_load_ms", { addMetric }, () => createChatClient(brokerSelection.modelId, brokerSelection.backendId, (progress) => setLoadProgress(progress), {
          tenantId,
          cellId,
          sessionId,
        }));
        llmRef.current = llm;
        setKvPersistenceHealth(readKvPersistenceHealth(llm) ?? kvPersistenceHealth);
        const backendProfile = makeBackendProfile(llm.backendId, toInferenceBackendBrokerSelection(brokerSelection));
        const modelProfile = makeModelProfile(brokerSelection.modelId);
        setRuntimeFeatures(buildRuntimeFeatureCapabilitySnapshot({
          backend: backendProfile,
          model: modelProfile,
          memoryMode: mode,
          ...(AGENT_CONFIG.mtp ? { mtp: AGENT_CONFIG.mtp } : {}),
        }));

        agentRef.current = new LocalAgent({
          llm,
          embeddings,
          memory: store,
          tenantId,
          cellId,
          sessionId,
          systemPrompt: SYSTEM_PROMPT,
          config: { ...AGENT_CONFIG, modelId: brokerSelection.modelId },
          modelProfile,
          deviceProfile: DEFAULT_DEVICE_PROFILE,
          backendProfile,
          memoryMode: mode
        });
        setStatus("Ready");
        void refreshInspector();
        void refreshModelCache();
      } catch (error) {
        const message = error instanceof Error ? error.message : String(error);
        agentRef.current = null;
        void llmRef.current?.dispose?.();
        llmRef.current = null;
        setLoadProgress("");
        setIsGenerating(false);
        setReadiness((current) => markInitializationFailure(current, message));
        setStatus(`Initialization failed: ${message}`);
      }
    } finally {
      initializingRef.current = false;
      setIsInitializing(false);
    }
  }

  async function ensureMemoryStore(): Promise<{ store: MemoryStore; mode: MemoryProviderMode }> {
    if (memoryRef.current && memoryModeRef.current) {
      return { store: memoryRef.current, mode: memoryModeRef.current };
    }

    setStatus("Initializing memory store...");
    const result = await createMemoryStore({
      provider: effectiveMemoryProvider,
      allowFallback: ALLOW_MEMORY_FALLBACK,
      remoteUrl: REMOTE_MEMORY_URL,
      remoteToken: REMOTE_MEMORY_TOKEN,
      remoteCredentials: REMOTE_MEMORY_CREDENTIALS,
      useSidecar: MEMORY_SERVER_ENABLED,
      sidecarUrl: MEMORY_SERVER_URL,
      ...(REMOTE_MEMORY_TENANT_ID ? { remoteTenantId: REMOTE_MEMORY_TENANT_ID } : {}),
      ...(REMOTE_MEMORY_CELL_ID ? { remoteCellId: REMOTE_MEMORY_CELL_ID } : {})
    });
    memoryRef.current = result.store;
    memoryModeRef.current = result.mode;
    setMemoryMode(result.mode);
    return result;
  }

  async function sendMessage() {
    const content = input.trim();
    if (!content || !agentRef.current || isGenerating) return;
    setInput("");
    setIsGenerating(true);
    setMemoryIds([]);
    setRetrievedMemory([]);

    const userMessage: ChatMessage = {
      id: makeMessageId(),
      role: "user",
      content,
      createdAt: new Date().toISOString(),
      sessionId
    };
    const streamingAssistant: ChatMessage = {
      id: makeMessageId(),
      role: "assistant",
      content: "",
      createdAt: new Date().toISOString(),
      sessionId
    };
    setMessages((existing) => [...existing, userMessage, streamingAssistant]);

    let assistantText = "";
    try {
      const finalMessage = await agentRef.current.submitUserMessage(content, {
        onStatus: setStatus,
        onMemoryIds: setMemoryIds,
        onRetrievedMemory: setRetrievedMemory,
        onRuntimeTrace: (trace, features) => {
          setLastTrace(trace);
          setRuntimeFeatures(features);
          const health = readKvPersistenceHealthFromTrace(trace);
          if (health) setKvPersistenceHealth(health);
        },
        onMetric: addMetric,
        onToken: (token) => {
          assistantText += token;
          setMessages((existing) =>
            existing.map((message) =>
              message.id === streamingAssistant.id ? { ...message, content: assistantText } : message
            )
          );
        }
      });
      setMessages((existing) =>
        existing.map((message) => (message.id === streamingAssistant.id ? finalMessage : message))
      );
    } catch (error) {
      setStatus(error instanceof Error ? error.message : String(error));
      setMessages((existing) =>
        existing.map((message) =>
          message.id === streamingAssistant.id
            ? { ...message, content: `Error: ${error instanceof Error ? error.message : String(error)}` }
            : message
        )
      );
    } finally {
      setIsGenerating(false);
    }
  }

  async function clearMemory() {
    if (agentRef.current) {
      await agentRef.current.clearMemory();
    } else if (memoryRef.current) {
      await memoryRef.current.clear();
    }
    setMessages([]);
    setMemoryIds([]);
    setRetrievedMemory([]);
    setStatus("Memory cleared");
    await refreshInspector();
  }

  async function deleteScopedMemory(options: MemoryDeleteOptions) {
    try {
      const { store } = await ensureMemoryStore();
      const scopedOptions = withActiveScope(options);
      const count = await store.deleteMemory(scopedOptions);
      const deletesCurrentSessionTranscript = options.sessionId === sessionId && !options.tags?.length;
      if (deletesCurrentSessionTranscript) {
        agentRef.current?.deleteTranscript(scopedOptions);
        setMessages([]);
        setMemoryIds([]);
        setRetrievedMemory([]);
      }
      setStatus(`Deleted ${count} scoped memory chunk${count === 1 ? "" : "s"}.`);
      await refreshInspector();
    } catch (error) {
      setStatus(error instanceof Error ? error.message : String(error));
    }
  }

  function deleteBySession() {
    const targetSessionId = deleteSessionId.trim() || sessionId;
    void deleteScopedMemory({ sessionId: targetSessionId });
  }

  function deleteByTags() {
    const tags = deleteTags.split(",").map((tag) => tag.trim()).filter(Boolean);
    if (tags.length === 0) {
      setStatus("Enter at least one tag or project tag to delete.");
      return;
    }
    void deleteScopedMemory({ tags });
  }

  async function exportMemory() {
    try {
      const { store, mode } = await ensureMemoryStore();
      const bundle = await exportMemoryBundle(store, { providerMode: mode, tenantId, cellId });
      const blob = new Blob([JSON.stringify(bundle, null, 2)], { type: "application/json" });
      const url = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url;
      link.download = `infinite-edge-memory-${new Date().toISOString().slice(0, 10)}.json`;
      link.click();
      URL.revokeObjectURL(url);
      setStatus(`Exported ${bundle.chunks.length} memory chunks, ${bundle.runtimeTraces.length} runtime traces, and ${bundle.contextPackTraces?.length ?? 0} context traces.`);
      await refreshInspector();
    } catch (error) {
      setStatus(error instanceof Error ? error.message : String(error));
    }
  }

  async function importMemoryFile(file: File | undefined) {
    if (!file) return;
    try {
      const bundle = JSON.parse(await file.text()) as MemoryExportBundle;
      const { store } = await ensureMemoryStore();
      await importMemoryBundle(store, bundle);
      setStatus(`Imported ${bundle.chunks.length} memory chunks, ${bundle.runtimeTraces.length} runtime traces, and ${bundle.contextPackTraces?.length ?? 0} context traces.`);
      await refreshInspector();
    } catch (error) {
      setStatus(error instanceof Error ? error.message : String(error));
    } finally {
      if (importInputRef.current) importInputRef.current.value = "";
    }
  }

  async function uploadDocumentFiles(files: FileList | null) {
    const selectedFiles = Array.from(files ?? []);
    if (selectedFiles.length === 0) return;
    if (!agentRef.current) {
      setStatus("Initialize the local agent before uploading documents.");
      if (documentInputRef.current) documentInputRef.current.value = "";
      return;
    }

    setIsUploadingDocuments(true);
    try {
      validateDocumentUploadSelection(selectedFiles);
      setStatus(`Reading ${selectedFiles.length} document${selectedFiles.length === 1 ? "" : "s"}...`);
      const documents: AgentDocumentInput[] = await Promise.all(selectedFiles.map(async (file) => ({
        name: file.name,
        type: file.type,
        size: file.size,
        lastModified: file.lastModified,
        text: await file.text(),
      })));
      const result = await agentRef.current.ingestDocuments(documents, {
        onStatus: setStatus,
        onMetric: addMetric,
      });
      setStatus(`Uploaded ${result.documentCount} document${result.documentCount === 1 ? "" : "s"} into ${result.chunkCount} memory chunk${result.chunkCount === 1 ? "" : "s"}.`);
      await refreshInspector();
    } catch (error) {
      setStatus(error instanceof Error ? error.message : String(error));
    } finally {
      setIsUploadingDocuments(false);
      if (documentInputRef.current) documentInputRef.current.value = "";
    }
  }

  async function refreshInspector() {
    try {
      const { store } = await ensureMemoryStore();
      const chunks = hasMemoryChunkList(store)
        ? await store.listMemoryChunks({ limit: 20, tenantId, cellId })
        : [];
      const traces = hasRuntimeTraceList(store)
        ? await store.listRuntimeTraces({ limit: 10, tenantId, cellId })
        : [];
      setMemoryRows(chunks);
      setTraceRows(traces);
      setInspectorStatus(`Loaded ${chunks.length} memory row${chunks.length === 1 ? "" : "s"} and ${traces.length} trace${traces.length === 1 ? "" : "s"}.`);
    } catch (error) {
      setInspectorStatus(error instanceof Error ? error.message : String(error));
    }
  }

  async function refreshModelCache() {
    try {
      setCacheSnapshot(await inspectBrowserModelCache());
    } catch (error) {
      setStatus(error instanceof Error ? error.message : String(error));
    }
  }

  async function clearModelCaches() {
    try {
      const cleared = await clearBrowserModelCaches();
      setStatus(`Cleared ${cleared.length} model/runtime cache entr${cleared.length === 1 ? "y" : "ies"}.`);
      await refreshModelCache();
    } catch (error) {
      setStatus(error instanceof Error ? error.message : String(error));
    }
  }

  const initialized = agentRef.current !== null;

  function withActiveScope(options: MemoryDeleteOptions): MemoryDeleteOptions {
    return {
      ...options,
      tenantId,
      cellId
    };
  }

  return (
    <main className={embedConfig.enabled && embedConfig.compact ? "app-shell embed-shell" : "app-shell"}>
      <section className="chat-shell">
        <header className="hero">
          <div>
            <p className="eyebrow">Local-first persistent agent</p>
            <h1>Infinite Edge Agent</h1>
            <p>
              Browser LLM inference, persistent browser-vector memory, worker isolation, and optional sidecar or remote memory scale layers.
            </p>
          </div>
          <div className="controls">
            <label>
              Model
              <select
                value={`${llmBackend}:${modelId}`}
                onChange={(event) => {
                  const [backend, id] = event.target.value.split(":");
                  const nextBackend = backend ?? DEFAULT_LLM_BACKEND;
                  setLlmBackend(nextBackend);
                  setModelId(id ?? DEFAULT_MODEL);
                  setReadiness(evaluateProductionReadiness({
                    memoryProvider: effectiveMemoryProvider,
                    allowMemoryFallback: ALLOW_MEMORY_FALLBACK,
                    llmBackend: nextBackend,
                    unlockedModelManifestPath: UNLOCKED_MODEL_MANIFEST_PATH,
                    unlockedModelManifestSha256: UNLOCKED_MODEL_MANIFEST_SHA256,
                    unlockedManifestFormat: UNLOCKED_MANIFEST_FORMAT,
                    unlockedAllowFixture: UNLOCKED_ALLOW_FIXTURE,
                    unlockedBackendPreference: UNLOCKED_BACKEND_PREFERENCE,
                    requireUnlockedRuntime: REQUIRE_UNLOCKED_RUNTIME,
                    requireWebGpuKernels: REQUIRE_WEBGPU_KERNELS,
                    compiledBackendAdapterAvailable: COMPILED_WEBLLM_ENABLED,
                    remoteMemoryUrl: REMOTE_MEMORY_URL,
                    remoteMemoryTenantId: REMOTE_MEMORY_TENANT_ID,
                    remoteMemoryCellId: REMOTE_MEMORY_CELL_ID,
                    hasPublicRemoteMemoryToken: HAS_PUBLIC_REMOTE_MEMORY_TOKEN,
                    production: PRODUCTION_MODE,
                  }));
                }}
                disabled={initialized || isInitializing}
              >
                {LOCAL_MODEL_OPTIONS.map((model) => (
                  <option key={`${model.backend}:${model.id}`} value={`${model.backend}:${model.id}`}>
                    {model.label}
                  </option>
                ))}
              </select>
            </label>
            <button onClick={initialize} disabled={initialized || isInitializing || isGenerating}>
              {initialized ? "Initialized" : isInitializing ? "Initializing..." : "Initialize local agent"}
            </button>
            <button className="secondary" onClick={clearMemory} disabled={isGenerating}>
              Clear memory
            </button>
            <div className="memory-actions">
              <button className="secondary" onClick={() => void exportMemory()} disabled={isGenerating}>
                Export memory
              </button>
              <button className="secondary" onClick={() => importInputRef.current?.click()} disabled={isGenerating}>
                Import memory
              </button>
              <input
                ref={importInputRef}
                className="hidden-file-input"
                type="file"
                accept="application/json,.json"
                onChange={(event) => void importMemoryFile(event.currentTarget.files?.[0])}
              />
            </div>
            <div className="document-actions">
              <button
                className="secondary"
                onClick={() => {
                  if (!initialized) {
                    setStatus("Initialize the local agent before uploading documents.");
                    return;
                  }
                  documentInputRef.current?.click();
                }}
                disabled={isGenerating || isUploadingDocuments}
              >
                {isUploadingDocuments ? "Uploading docs..." : "Upload docs"}
              </button>
              <input
                ref={documentInputRef}
                className="hidden-file-input"
                type="file"
                multiple
                accept={DOCUMENT_UPLOAD_ACCEPT}
                onChange={(event) => void uploadDocumentFiles(event.currentTarget.files)}
              />
            </div>
            <div className="targeted-memory-actions">
              <label>
                Session
                <input
                  value={deleteSessionId}
                  onChange={(event) => setDeleteSessionId(event.target.value)}
                  placeholder={sessionId}
                  disabled={isGenerating}
                />
              </label>
              <button className="secondary" onClick={deleteBySession} disabled={isGenerating}>
                Delete session
              </button>
              <label>
                Tags or project
                <input
                  value={deleteTags}
                  onChange={(event) => setDeleteTags(event.target.value)}
                  placeholder="project:edge-ai, user"
                  disabled={isGenerating}
                />
              </label>
              <button className="secondary" onClick={deleteByTags} disabled={isGenerating || !deleteTags.trim()}>
                Delete tags
              </button>
            </div>
          </div>
        </header>

        {loadProgress && <p className="load-progress">Model load: {loadProgress}</p>}

        <section className="messages" aria-live="polite">
          {messages.length === 0 ? (
            <div className="empty-state">
              <h2>Start with a project brief</h2>
              <p>
                Example: “Remember that this project is a local-first agent. Draft the next implementation step.”
              </p>
            </div>
          ) : (
            messages.map((message) => <ChatMessageView key={message.id} message={message} />)
          )}
        </section>

        <form
          className="composer"
          onSubmit={(event) => {
            event.preventDefault();
            void sendMessage();
          }}
        >
          <textarea
            value={input}
            onChange={(event) => setInput(event.target.value)}
            placeholder={initialized ? "Message the local agent..." : "Initialize the agent first..."}
            disabled={!initialized || isGenerating}
          />
          <button type="submit" disabled={!initialized || isGenerating || !input.trim()}>
            {isGenerating ? "Generating..." : "Send"}
          </button>
        </form>
      </section>

      <StatusPanel
        status={status}
        memoryMode={memoryMode}
        modelId={modelId}
        backend={llmBackend}
        unlockedRuntime={{
          manifestPath: UNLOCKED_MODEL_MANIFEST_PATH,
          manifestSha256: UNLOCKED_MODEL_MANIFEST_SHA256,
          manifestFormat: UNLOCKED_MANIFEST_FORMAT,
          allowFixture: UNLOCKED_ALLOW_FIXTURE,
          backendPreference: UNLOCKED_BACKEND_PREFERENCE,
          requireWebGpu: REQUIRE_WEBGPU_KERNELS,
          runtimeProfile: UNLOCKED_RUNTIME_PROFILE,
          qwenThinkingMode: QWEN_THINKING_MODE,
          interactiveLimits: {
            maxRuntimePromptTokens: CHAT_MAX_RUNTIME_PROMPT_TOKENS,
            maxRuntimeLayers: CHAT_MAX_RUNTIME_LAYERS,
            logitCandidateLimit: CHAT_LOGIT_CANDIDATE_LIMIT,
            logitTopK: CHAT_LOGIT_TOP_K,
            logitTileRows: CHAT_LOGIT_TILE_ROWS,
            maxGenerationTokens: CHAT_MAX_GENERATION_TOKENS,
          },
        }}
        embeddingDevice={embeddingDevice}
        memoryIds={memoryIds}
        features={runtimeFeatures}
        lastTrace={lastTrace}
        readiness={readiness}
        retrievedMemory={retrievedMemory}
        memoryRows={memoryRows}
        traceRows={traceRows}
        inspectorStatus={inspectorStatus}
        onRefreshInspector={() => void refreshInspector()}
        metrics={metrics}
        cacheSnapshot={cacheSnapshot}
        kvPersistenceHealth={kvPersistenceHealth}
        onRefreshCache={() => void refreshModelCache()}
        onClearModelCaches={() => void clearModelCaches()}
      />
    </main>
  );
}

const DOCUMENT_UPLOAD_ACCEPT = [
  ".md",
  ".markdown",
  ".txt",
  ".text",
  ".csv",
  ".json",
  "text/markdown",
  "text/plain",
  "text/csv",
  "application/json",
].join(",");
const DOCUMENT_UPLOAD_MAX_FILES = 12;
const DOCUMENT_UPLOAD_MAX_BYTES = 2 * 1024 * 1024;
const DOCUMENT_UPLOAD_TOTAL_MAX_BYTES = 8 * 1024 * 1024;

function validateDocumentUploadSelection(files: File[]): void {
  if (files.length > DOCUMENT_UPLOAD_MAX_FILES) {
    throw new Error(`Upload at most ${DOCUMENT_UPLOAD_MAX_FILES} documents at a time.`);
  }
  const totalBytes = files.reduce((total, file) => total + file.size, 0);
  if (totalBytes > DOCUMENT_UPLOAD_TOTAL_MAX_BYTES) {
    throw new Error("Document upload batch is too large. Keep each batch under 8 MB.");
  }
  const unsupported = files.find((file) => !isSupportedDocumentFile(file));
  if (unsupported) {
    throw new Error(`Unsupported document type for ${unsupported.name}. Upload Markdown, plain text, CSV, or JSON text files.`);
  }
  const oversized = files.find((file) => file.size > DOCUMENT_UPLOAD_MAX_BYTES);
  if (oversized) {
    throw new Error(`${oversized.name} is too large. Upload files up to 2 MB each.`);
  }
}

function isSupportedDocumentFile(file: File): boolean {
  const name = file.name.toLowerCase();
  const type = file.type.toLowerCase();
  return [".md", ".markdown", ".txt", ".text", ".csv", ".json"].some((extension) => name.endsWith(extension))
    || ["text/markdown", "text/plain", "text/csv", "application/json"].includes(type);
}

function hasMemoryChunkList(store: MemoryStore): store is MemoryStore & {
  listMemoryChunks(options?: { sessionId?: string; limit?: number; tenantId?: string; cellId?: string }): Promise<MemoryChunk[]>;
} {
  return "listMemoryChunks" in store && typeof store.listMemoryChunks === "function";
}

function hasRuntimeTraceList(store: MemoryStore): store is MemoryStore & {
  listRuntimeTraces(options?: { sessionId?: string; limit?: number; tenantId?: string; cellId?: string }): Promise<RuntimeTrace[]>;
} {
  return "listRuntimeTraces" in store && typeof store.listRuntimeTraces === "function";
}

function hasContextPackTraceList(store: MemoryStore): store is MemoryStore & {
  listContextPackTraces(options?: { sessionId?: string; limit?: number; tenantId?: string; cellId?: string }): Promise<ContextPackTraceRecord[]>;
} {
  return "listContextPackTraces" in store && typeof store.listContextPackTraces === "function";
}

async function createChatClient(
  modelId: string,
  backend: string,
  onProgress: (progress: string) => void,
  scope: { tenantId: string; cellId: string; sessionId: string }
): Promise<ChatClient> {
  if (REQUIRE_UNLOCKED_RUNTIME && backend !== "unlocked-browser-transformer") {
    throw new Error(`Full-control runtime requires unlocked-browser-transformer, received ${backend}.`);
  }

  const backendEntry = getBrowserBackendRegistryEntry(backend);
  if (!backendEntry) {
    throw new Error(`Unsupported backend ${backend}. Backend Broker only ships registered browser backends.`);
  }
  const selection = resolveBrowserAnswerBackendSelection({
    backend,
    modelId,
    task: "grounded_answer",
  });
  onProgress(`Backend Broker selected ${selection.backendId} (${selection.reason})`);

  if (selection.productionRole === "production_candidate") {
    if (!COMPILED_WEBLLM_ENABLED) {
      throw new Error(`Compiled production backend ${backend} is registered by Backend Broker, but its runtime adapter is not available in this build yet.`);
    }
    onProgress("Loading compiled WebLLM backend");
    const client = new CompiledWebLlmClient({ modelId: selection.modelId, onProgress });
    await client.init();
    return client;
  }

  if (selection.backendId !== "unlocked-browser-transformer") {
    throw new Error(`Backend ${backend} is registered as ${backendEntry.productionRole}, not an answer-generation runtime for this build.`);
  }

  onProgress(UNLOCKED_MODEL_MANIFEST_PATH ? "Loading unlocked model manifest" : "Using unlocked tensor-control fixture");
  const options = makeUnlockedBrowserWorkerOptions({
    modelId: selection.modelId,
    manifestPath: UNLOCKED_MODEL_MANIFEST_PATH,
    manifestSha256: UNLOCKED_MODEL_MANIFEST_SHA256,
    allowFixtureWeights: UNLOCKED_ALLOW_FIXTURE,
    backendPreference: UNLOCKED_BACKEND_PREFERENCE,
    requireWebGpu: REQUIRE_WEBGPU_KERNELS,
    maxRuntimePromptTokens: CHAT_MAX_RUNTIME_PROMPT_TOKENS,
    maxRuntimeLayers: CHAT_MAX_RUNTIME_LAYERS,
    logitCandidateLimit: CHAT_LOGIT_CANDIDATE_LIMIT,
    logitTopK: CHAT_LOGIT_TOP_K,
    logitTileRows: CHAT_LOGIT_TILE_ROWS,
    maxGenerationTokens: CHAT_MAX_GENERATION_TOKENS,
    qwenThinkingMode: QWEN_THINKING_MODE,
    mtpEnabled: MTP_ENABLED,
    mtpDraftModelId: MTP_DRAFT_MODEL_ID,
    mtpNumSpeculativeTokens: MTP_NUM_SPECULATIVE_TOKENS,
    mtpMinAcceptanceRate: MTP_MIN_ACCEPTANCE_RATE,
    mtpDisableWhenLatencyWorse: MTP_DISABLE_WHEN_LATENCY_WORSE,
    mtpDraftLayerCount: MTP_DRAFT_LAYER_COUNT,
    kvPersistenceEnabled: KVSWAP_PERSISTENCE_ENABLED,
    kvPersistenceNamespace: `${scope.tenantId}:${scope.cellId}:${scope.sessionId}`,
    kvPersistencePreferOpfs: KVSWAP_PERSISTENCE_PREFER_OPFS,
    kvPersistenceMaxBlocks: KVSWAP_PERSISTENCE_MAX_BLOCKS,
    kvPersistenceMaxBytes: KVSWAP_PERSISTENCE_MAX_BYTES,
    kvPersistenceClearOnInit: KVSWAP_PERSISTENCE_CLEAR_ON_INIT,
  });
  const { UnlockedBrowserTransformerWorkerClient } = await import("./lib/llm/unlockedBrowserTransformerWorkerClient");
  const client = new UnlockedBrowserTransformerWorkerClient(options, { onProgress });
  try {
    await client.init();
    return client;
  } catch (error) {
    await client.dispose();
    throw error;
  }
}

function toInferenceBackendBrokerSelection(
  selection: BrowserBackendSelection,
): InferenceBackendBrokerSelection {
  return {
    backendId: selection.backendId,
    modelId: selection.modelId,
    productionRole: selection.productionRole,
    deployReadyCandidate: selection.deployReadyCandidate,
    reason: selection.reason,
    fallbackChain: selection.fallbackChain,
    proofRequirements: selection.proofRequirements,
  };
}

export interface BrowserAnswerBackendSelectionInput {
  backend: string;
  modelId: string;
  task?: BrowserBackendTask;
  availableBackendIds?: string[];
}

export function resolveBrowserAnswerBackendSelection(
  input: BrowserAnswerBackendSelectionInput,
): BrowserBackendSelection {
  const backendEntry = getBrowserBackendRegistryEntry(input.backend);
  if (!backendEntry) {
    throw new Error(`Unsupported backend ${input.backend}. Backend Broker only ships registered browser backends.`);
  }
  if (backendEntry.productionRole === "fallback" && input.task === undefined) {
    throw new Error(`Bounded fallback-only backend ${input.backend} cannot run browser answer generation.`);
  }
  const task = input.task
    ?? (backendEntry.productionRole === "research_kernel_lab" ? "kernel_research" : "grounded_answer");
  return selectBrowserBackend({
    task,
    preferredBackendId: input.backend,
    preferredModelId: input.modelId,
    ...(input.availableBackendIds ? { availableBackendIds: input.availableBackendIds } : {}),
  });
}

export interface UnlockedBrowserWorkerOptionsInput {
  modelId: string;
  manifestPath: string;
  manifestSha256: string;
  allowFixtureWeights: boolean;
  backendPreference?: string | undefined;
  requireWebGpu?: boolean | undefined;
  maxRuntimePromptTokens?: number | null | undefined;
  maxRuntimeLayers?: number | null | undefined;
  logitCandidateLimit?: number | null | undefined;
  logitTopK?: number | null | undefined;
  logitTileRows?: number | null | undefined;
  maxGenerationTokens?: number | null | undefined;
  qwenThinkingMode?: UnlockedBrowserTransformerClientOptions["qwenThinkingMode"];
  mtpEnabled: boolean;
  mtpDraftModelId: string;
  mtpNumSpeculativeTokens: number;
  mtpMinAcceptanceRate: number;
  mtpDisableWhenLatencyWorse: boolean;
  mtpDraftLayerCount?: number;
  kvPersistenceEnabled?: boolean;
  kvPersistenceNamespace?: string;
  kvPersistencePreferOpfs?: boolean;
  kvPersistenceMaxBlocks?: number;
  kvPersistenceMaxBytes?: number;
  kvPersistenceClearOnInit?: boolean;
}

export function makeUnlockedBrowserWorkerOptions(
  input: UnlockedBrowserWorkerOptionsInput,
): UnlockedBrowserTransformerClientOptions {
  const warmModelResidency = input.requireWebGpu === true || input.backendPreference === "webgpu";
  return {
    modelId: input.modelId,
    manifestPath: input.manifestPath,
    manifestSha256: input.manifestSha256,
    allowFixtureWeights: input.allowFixtureWeights,
    ...(input.backendPreference ? { backendPreference: input.backendPreference as UnlockedBrowserTransformerClientOptions["backendPreference"] } : {}),
    ...(input.requireWebGpu !== undefined ? { requireWebGpu: input.requireWebGpu } : {}),
    ...(input.maxRuntimePromptTokens !== undefined ? { maxRuntimePromptTokens: input.maxRuntimePromptTokens } : {}),
    ...(input.maxRuntimeLayers !== undefined ? { maxRuntimeLayers: input.maxRuntimeLayers } : {}),
    ...(input.logitCandidateLimit !== undefined ? { logitCandidateLimit: input.logitCandidateLimit } : {}),
    ...(input.logitTopK !== undefined ? { logitTopK: input.logitTopK } : {}),
    ...(input.logitTileRows !== undefined ? { logitTileRows: input.logitTileRows } : {}),
    ...(input.maxGenerationTokens !== undefined ? { maxGenerationTokens: input.maxGenerationTokens } : {}),
    ...(input.qwenThinkingMode !== undefined ? { qwenThinkingMode: input.qwenThinkingMode } : {}),
    ...(warmModelResidency
      ? {
          warmModelResidency: true,
          warmModelResidencyMode: "pipeline_preload" as const,
        }
      : {}),
    mtp: makeBrowserMtpClientOptions({
      enabled: input.mtpEnabled,
      draftModelId: input.mtpDraftModelId,
      numSpeculativeTokens: input.mtpNumSpeculativeTokens,
      minAcceptanceRate: input.mtpMinAcceptanceRate,
      disableWhenLatencyWorse: input.mtpDisableWhenLatencyWorse,
      ...(input.mtpDraftLayerCount !== undefined ? { draftLayerCount: input.mtpDraftLayerCount } : {}),
    }),
    kvPersistence: {
      enabled: input.kvPersistenceEnabled === true,
      ...(input.kvPersistenceNamespace !== undefined ? { namespace: input.kvPersistenceNamespace } : {}),
      ...(input.kvPersistencePreferOpfs !== undefined ? { preferOpfs: input.kvPersistencePreferOpfs } : {}),
      ...(input.kvPersistenceMaxBlocks !== undefined ? { maxBlocks: input.kvPersistenceMaxBlocks } : {}),
      ...(input.kvPersistenceMaxBytes !== undefined ? { maxBytes: input.kvPersistenceMaxBytes } : {}),
      ...(input.kvPersistenceClearOnInit !== undefined ? { clearOnInit: input.kvPersistenceClearOnInit } : {}),
    },
  };
}

function readKvPersistenceHealth(client: ChatClient): KVSwapPersistenceHealth | null {
  const candidate = client as ChatClient & { kvPersistenceHealth?: KVSwapPersistenceHealth };
  return candidate.kvPersistenceHealth ?? null;
}

function readKvPersistenceHealthFromTrace(trace: RuntimeTrace): KVSwapPersistenceHealth | null {
  const generation = trace.runtime.generation;
  if (!generation || typeof generation !== "object" || Array.isArray(generation)) return null;
  const decodeProof = (generation as Record<string, unknown>).decodeProof;
  if (!decodeProof || typeof decodeProof !== "object" || Array.isArray(decodeProof)) return null;
  const kvPersistence = (decodeProof as Record<string, unknown>).kvPersistence;
  if (!kvPersistence || typeof kvPersistence !== "object" || Array.isArray(kvPersistence)) return null;
  return kvPersistence as KVSwapPersistenceHealth;
}
