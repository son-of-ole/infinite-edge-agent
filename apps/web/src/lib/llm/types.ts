import type { ChatMessage } from "@infinite-edge-agent/core";

export type ChatClientMessage =
  | Pick<ChatMessage, "role" | "content">
  | { role: "system" | "user" | "assistant"; content: string };

export interface ChatStreamOptions {
  temperature?: number;
  topP?: number;
  repetitionPenalty?: number;
  samplingSeed?: number;
  maxTokens?: number;
  includeProofMarker?: boolean;
  stopAfterSequences?: string[];
  awaitKvPredictivePrefetchProof?: boolean;
}

export interface ChatClient {
  readonly backendId: string;
  readonly modelId: string;
  readonly lastDecodeProof?: unknown;
  streamChat(messages: ChatClientMessage[], options?: ChatStreamOptions): AsyncGenerator<string, string, void>;
  flushKvPersistence?(): Promise<void>;
  clearKvPersistence?(): Promise<unknown>;
  dispose?(): Promise<void>;
}
