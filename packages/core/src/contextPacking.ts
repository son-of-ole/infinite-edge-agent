import { approximateTokenCount } from "./tokenBudget";
import type { ChatMessage, ContextPackInput, MemorySearchHit, PackedContext } from "./types";

const TRUNCATION_MARKER = "\n...[truncated]";

function normalizeTokenBudget(value: number): number {
  return Number.isFinite(value) ? Math.max(0, Math.floor(value)) : 0;
}

export function formatMemoryForPrompt(memory: MemorySearchHit[]): string {
  if (memory.length === 0) return "No long-term memory retrieved for this turn.";
  return memory
    .map((hit, index) => {
      const source = [hit.source, hit.role, hit.sessionId].filter(Boolean).join("/");
      return `Memory ${index + 1} | score=${hit.score.toFixed(3)} | source=${source} | created=${hit.createdAt}\n${hit.text}`;
    })
    .join("\n\n---\n\n");
}

export function selectMemoryWithinBudget(
  memory: MemorySearchHit[],
  maxTokens: number
): { selected: MemorySearchHit[]; tokenCount: number } {
  const sorted = [...memory].sort((a, b) => b.score - a.score);
  const selected: MemorySearchHit[] = [];
  let total = 0;
  for (const hit of sorted) {
    const next = total + hit.tokenCount;
    if (next > maxTokens) continue;
    selected.push(hit);
    total = next;
  }
  return { selected, tokenCount: total };
}

export function selectRecentMessagesWithinBudget(
  messages: ChatMessage[],
  maxTokens: number
): Array<{ role: "user" | "assistant"; content: string }> {
  const selected: Array<{ role: "user" | "assistant"; content: string }> = [];
  let total = 0;
  for (const message of [...messages].reverse()) {
    if (message.role !== "user" && message.role !== "assistant") continue;
    const count = approximateTokenCount(message.content);
    if (total + count > maxTokens) break;
    selected.push({ role: message.role, content: message.content });
    total += count;
  }
  return selected.reverse();
}

function clampTextWithinBudget(text: string, maxTokens: number): string {
  const budget = normalizeTokenBudget(maxTokens);
  if (budget <= 0) return "";
  if (approximateTokenCount(text) <= budget) return text;

  const markerTokens = approximateTokenCount(TRUNCATION_MARKER);
  const maxChars = budget * 4;
  if (budget <= markerTokens) {
    return text.trim().slice(0, maxChars);
  }

  const contentBudgetChars = (budget - markerTokens) * 4;
  const clamped = `${text.trim().slice(0, contentBudgetChars).trimEnd()}${TRUNCATION_MARKER}`;
  if (approximateTokenCount(clamped) <= budget) return clamped;
  return text.trim().slice(0, maxChars);
}

function buildSystemWithMemory(systemPrompt: string, memory: MemorySearchHit[]): string {
  return `${systemPrompt.trim()}\n\n## Retrieved long-term memory\n${formatMemoryForPrompt(memory)}`;
}

function selectMemoryForSystemBudget(
  memory: MemorySearchHit[],
  maxMemoryTokens: number,
  systemPrompt: string,
  maxSystemTokens: number
): MemorySearchHit[] {
  const memoryBudget = normalizeTokenBudget(maxMemoryTokens);
  const systemBudget = normalizeTokenBudget(maxSystemTokens);
  if (memoryBudget <= 0 || systemBudget <= 0) return [];

  const sorted = [...memory].sort((a, b) => b.score - a.score);
  const selected: MemorySearchHit[] = [];
  let selectedMemoryTokens = 0;

  for (const hit of sorted) {
    const hitTokens = normalizeTokenBudget(hit.tokenCount);
    if (selectedMemoryTokens + hitTokens > memoryBudget) continue;

    const candidate = [...selected, hit];
    if (approximateTokenCount(buildSystemWithMemory(systemPrompt, candidate)) > systemBudget) continue;

    selected.push(hit);
    selectedMemoryTokens += hitTokens;
  }

  return selected;
}

function selectRecentMessagesForFinalBudget(
  messages: ChatMessage[],
  maxRecentTokens: number,
  maxTokens: number
): Array<{ role: "user" | "assistant"; content: string }> {
  const recentBudget = Math.min(normalizeTokenBudget(maxRecentTokens), normalizeTokenBudget(maxTokens));
  if (recentBudget <= 0) return [];

  const selected: Array<{ role: "user" | "assistant"; content: string }> = [];
  let total = 0;

  for (const message of [...messages].reverse()) {
    if (message.role !== "user" && message.role !== "assistant") continue;

    const remaining = recentBudget - total;
    if (remaining <= 0) break;

    const count = approximateTokenCount(message.content);
    const content = count <= remaining ? message.content : clampTextWithinBudget(message.content, remaining);
    if (approximateTokenCount(content) <= 0) break;

    selected.push({ role: message.role, content });
    total += approximateTokenCount(content);

    if (count > remaining) break;
  }

  return selected.reverse();
}

export function packContext(input: ContextPackInput): PackedContext {
  const maxPromptTokens = normalizeTokenBudget(input.config.maxPromptTokens);
  const minUserTokens = maxPromptTokens > 0 && approximateTokenCount(input.userMessage) > 0 ? 1 : 0;

  const systemBudget = Math.max(0, maxPromptTokens - minUserTokens);
  const selected = selectMemoryForSystemBudget(
    input.retrievedMemory,
    input.config.maxRetrievedMemoryTokens,
    input.systemPrompt,
    systemBudget
  );
  const systemContent = clampTextWithinBudget(buildSystemWithMemory(input.systemPrompt, selected), systemBudget);
  const usedSystemTokens = approximateTokenCount(systemContent);

  const recent = selectRecentMessagesForFinalBudget(
    input.recentMessages,
    input.config.maxRecentConversationTokens,
    Math.max(0, maxPromptTokens - usedSystemTokens - minUserTokens)
  );
  const usedRecentTokens = recent.reduce((sum, message) => sum + approximateTokenCount(message.content), 0);
  const userContent = clampTextWithinBudget(input.userMessage, Math.max(0, maxPromptTokens - usedSystemTokens - usedRecentTokens));

  const messages: PackedContext["messages"] = [
    {
      role: "system",
      content: systemContent
    },
    ...recent,
    {
      role: "user",
      content: userContent
    }
  ];

  const estimatedTokens = messages.reduce((sum, message) => sum + approximateTokenCount(message.content), 0);
  return {
    messages,
    includedMemoryIds: selected.map((hit) => hit.id),
    estimatedTokens
  };
}
