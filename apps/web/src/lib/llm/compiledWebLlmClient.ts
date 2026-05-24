import type { ChatClient, ChatClientMessage, ChatStreamOptions } from "./types";

const THINK_START_MARKER = "<think>";
const THINK_END_MARKER = "</think>";
const THINK_FILTER_MARKERS = [THINK_START_MARKER, THINK_END_MARKER];

export interface CompiledWebLlmProof {
  backendId: "compiled-browser-webllm";
  adapterKind: "compiled-browser";
  modelId: string;
  streaming: true;
  generatedText: string;
  generatedTokenEstimate: number;
}

export interface WebLlmChatCompletionChunk {
  choices?: Array<{
    delta?: {
      content?: string | null;
    };
    message?: {
      content?: string | null;
    };
  }>;
}

export interface WebLlmChatCompletions {
  create(input: {
    messages: ChatClientMessage[];
    stream: true;
    max_tokens?: number;
    temperature?: number;
    top_p?: number;
    stop?: string[];
    extra_body?: {
      enable_thinking?: boolean | null;
    };
  }): Promise<AsyncIterable<WebLlmChatCompletionChunk>> | AsyncIterable<WebLlmChatCompletionChunk>;
}

export interface WebLlmEngine {
  chat: {
    completions: WebLlmChatCompletions;
  };
  unload?: () => void | Promise<void>;
}

export interface WebLlmModule {
  CreateMLCEngine: (
    modelId: string,
    options?: { initProgressCallback?: (progress: unknown) => void },
  ) => Promise<WebLlmEngine>;
}

export type WebLlmModuleLoader = () => Promise<WebLlmModule>;
export type QwenThinkingMode = "disabled" | "enabled";

export interface CompiledWebLlmClientOptions {
  modelId: string;
  moduleLoader?: WebLlmModuleLoader;
  onProgress?: (progress: string) => void;
  qwenThinkingMode?: QwenThinkingMode;
}

export class CompiledWebLlmClient implements ChatClient {
  readonly backendId = "compiled-browser-webllm";
  readonly modelId: string;
  lastDecodeProof: CompiledWebLlmProof | null = null;
  lastPromptTokenIds: number[] = [];
  lastGeneratedTokenIds: number[] = [];
  lastGeneratedTokenTexts: string[] = [];
  lastGenerationStopReason: string | null = null;
  lastWarmupMs = 0;
  lastWarmupMode: null = null;
  lastWarmupProof: null = null;
  lastWarmupUploadedEntries: null = null;
  lastWarmupCacheHits: null = null;
  lastResidentReadbackCount: null = null;

  private readonly moduleLoader: WebLlmModuleLoader;
  private readonly onProgress: ((progress: string) => void) | undefined;
  private readonly qwenThinkingMode: QwenThinkingMode;
  private engine: WebLlmEngine | null = null;

  constructor(options: CompiledWebLlmClientOptions) {
    this.modelId = options.modelId;
    this.moduleLoader = options.moduleLoader ?? createDefaultWebLlmModuleLoader();
    this.onProgress = options.onProgress;
    this.qwenThinkingMode = options.qwenThinkingMode ?? "disabled";
  }

  async init(): Promise<void> {
    const module = await this.moduleLoader();
    this.engine = await module.CreateMLCEngine(this.modelId, {
      initProgressCallback: (progress) => {
        this.onProgress?.(typeof progress === "string" ? progress : JSON.stringify(progress));
      },
    });
  }

  async *streamChat(
    messages: ChatClientMessage[],
    options: ChatStreamOptions = {},
  ): AsyncGenerator<string, string, void> {
    if (!this.engine) throw new Error("CompiledWebLlmClient.init() must complete before streamChat().");
    const qwenThinkingDisabled = shouldDisableQwenThinking(this.modelId, this.qwenThinkingMode);
    const requestMessages = withQwenThinkingDirective(messages, qwenThinkingDisabled);
    const outputFilter = qwenThinkingDisabled ? new QwenThinkingOutputFilter() : null;
    this.lastDecodeProof = null;
    this.lastPromptTokenIds = estimateTokenIds(requestMessages.map((message) => message.content).join("\n"));
    this.lastGeneratedTokenIds = [];
    this.lastGeneratedTokenTexts = [];
    this.lastGenerationStopReason = null;
    const request = buildWebLlmChatRequest(requestMessages, options, qwenThinkingDisabled);
    const stream = await this.engine.chat.completions.create(request);
    let generatedText = "";
    for await (const chunk of stream) {
      const rawText = readChunkText(chunk);
      const text = outputFilter ? outputFilter.push(rawText) : rawText;
      if (!text) continue;
      generatedText += text;
      yield text;
    }
    const finalText = outputFilter?.flush() ?? "";
    if (finalText) {
      generatedText += finalText;
      yield finalText;
    }
    const generatedTokenTexts = estimateTokenTexts(generatedText);
    this.lastGeneratedTokenTexts = generatedTokenTexts;
    this.lastGeneratedTokenIds = generatedTokenTexts.map((_token, index) => index);
    this.lastGenerationStopReason = "stream_complete";
    this.lastDecodeProof = {
      backendId: this.backendId,
      adapterKind: "compiled-browser",
      modelId: this.modelId,
      streaming: true,
      generatedText,
      generatedTokenEstimate: generatedTokenTexts.length,
    };
    return generatedText;
  }

  async flushKvPersistence(): Promise<void> {
    // Compiled backends own KV state internally; there is no browser KVSwap flush path.
  }

  async dispose(_options?: unknown): Promise<void> {
    const engine = this.engine;
    this.engine = null;
    await engine?.unload?.();
  }
}

export function createDefaultWebLlmModuleLoader(
  importer?: (specifier: string) => Promise<unknown>,
): WebLlmModuleLoader {
  return async () => {
    try {
      const loaded = importer
        ? await importer("@mlc-ai/web-llm")
        : await import("@mlc-ai/web-llm");
      if (!isWebLlmModule(loaded)) {
        throw new Error("loaded module does not expose CreateMLCEngine");
      }
      return loaded;
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      throw new Error(`Compiled WebLLM backend is not available. Install @mlc-ai/web-llm or provide a bundled WebLLM module loader. Cause: ${message}`);
    }
  };
}

function buildWebLlmChatRequest(
  messages: ChatClientMessage[],
  options: ChatStreamOptions,
  qwenThinkingDisabled: boolean,
): Parameters<WebLlmChatCompletions["create"]>[0] {
  return {
    messages,
    stream: true,
    ...(options.maxTokens !== undefined ? { max_tokens: options.maxTokens } : {}),
    ...(options.temperature !== undefined ? { temperature: options.temperature } : {}),
    ...(options.topP !== undefined ? { top_p: options.topP } : {}),
    ...(options.stopAfterSequences && options.stopAfterSequences.length > 0 ? { stop: options.stopAfterSequences } : {}),
    ...(qwenThinkingDisabled ? { extra_body: { enable_thinking: false } } : {}),
  };
}

function readChunkText(chunk: WebLlmChatCompletionChunk): string {
  const choice = chunk.choices?.[0];
  return choice?.delta?.content ?? choice?.message?.content ?? "";
}

function isWebLlmModule(value: unknown): value is WebLlmModule {
  return typeof value === "object"
    && value !== null
    && typeof (value as { CreateMLCEngine?: unknown }).CreateMLCEngine === "function";
}

function estimateTokenTexts(text: string): string[] {
  const matches = text.trim().match(/\S+/g);
  if (matches && matches.length > 0) return matches;
  return text.length > 0 ? [text] : [];
}

function estimateTokenIds(text: string): number[] {
  return estimateTokenTexts(text).map((_token, index) => index);
}

function withQwenThinkingDirective(
  messages: ChatClientMessage[],
  qwenThinkingDisabled: boolean,
): ChatClientMessage[] {
  if (!qwenThinkingDisabled) return messages;
  const targetIndex = findLastUserMessageIndex(messages);
  if (targetIndex < 0) return messages;
  return messages.map((message, index) => (
    index === targetIndex
      ? { ...message, content: appendQwenNoThinkDirective(message.content) }
      : message
  ));
}

function findLastUserMessageIndex(messages: ChatClientMessage[]): number {
  for (let index = messages.length - 1; index >= 0; index -= 1) {
    if (messages[index]?.role === "user") return index;
  }
  return -1;
}

function appendQwenNoThinkDirective(content: string): string {
  if (/\/(?:no_)?think\b/i.test(content)) return content;
  return `${content.trimEnd()}\n/no_think`;
}

function shouldDisableQwenThinking(modelId: string, qwenThinkingMode: QwenThinkingMode): boolean {
  return qwenThinkingMode === "disabled" && /qwen/i.test(modelId);
}

class QwenThinkingOutputFilter {
  private buffer = "";
  private insideThinking = false;

  push(text: string): string {
    if (!text) return "";
    this.buffer += text;
    return this.drain(false);
  }

  flush(): string {
    return this.drain(true);
  }

  private drain(flush: boolean): string {
    let output = "";
    while (this.buffer.length > 0) {
      if (this.insideThinking) {
        const endIndex = this.buffer.indexOf(THINK_END_MARKER);
        if (endIndex === -1) {
          const holdback = partialMarkerHoldback(this.buffer, [THINK_END_MARKER]);
          this.buffer = holdback > 0 ? this.buffer.slice(-holdback) : "";
          if (flush) this.buffer = "";
          break;
        }
        this.buffer = this.buffer.slice(endIndex + THINK_END_MARKER.length).replace(/^\s+/, "");
        this.insideThinking = false;
        continue;
      }

      const startIndex = this.buffer.indexOf(THINK_START_MARKER);
      const endIndex = this.buffer.indexOf(THINK_END_MARKER);
      const nextIndex = [startIndex, endIndex].filter((index) => index >= 0).sort((a, b) => a - b)[0];
      if (nextIndex === undefined) {
        const holdback = flush ? 0 : partialMarkerHoldback(this.buffer, THINK_FILTER_MARKERS);
        output += this.buffer.slice(0, this.buffer.length - holdback);
        this.buffer = holdback > 0 ? this.buffer.slice(-holdback) : "";
        break;
      }

      output += this.buffer.slice(0, nextIndex);
      if (nextIndex === startIndex) {
        this.buffer = this.buffer.slice(nextIndex + THINK_START_MARKER.length);
        this.insideThinking = true;
      } else {
        this.buffer = this.buffer.slice(nextIndex + THINK_END_MARKER.length).replace(/^\s+/, "");
      }
    }
    return output;
  }
}

function partialMarkerHoldback(text: string, markers: readonly string[]): number {
  const maxMarkerLength = Math.max(...markers.map((marker) => marker.length));
  const max = Math.min(text.length, maxMarkerLength - 1);
  for (let length = max; length > 0; length -= 1) {
    const suffix = text.slice(-length);
    if (markers.some((marker) => marker.startsWith(suffix))) return length;
  }
  return 0;
}
