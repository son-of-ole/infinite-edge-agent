import { expect, it } from "vitest";
import { packContext } from "./contextPacking";
import type { MemorySearchHit } from "./types";

function memory(overrides: Partial<MemorySearchHit> & Pick<MemorySearchHit, "id" | "score" | "text">): MemorySearchHit {
  return {
    embedding: [1],
    sessionId: "s1",
    source: "chat",
    createdAt: "2026-05-11T00:00:00.000Z",
    updatedAt: "2026-05-11T00:00:00.000Z",
    tags: [],
    metadata: {},
    tokenCount: Math.ceil(overrides.text.length / 4),
    ...overrides
  };
}

it("packs context with memory and recent messages", () => {
  const packed = packContext({
    systemPrompt: "You are local.",
    retrievedMemory: [
      {
        id: "m1",
        score: 0.9,
        text: "The user prefers concise plans.",
        embedding: [1],
        sessionId: "s1",
        source: "chat",
        createdAt: new Date().toISOString(),
        updatedAt: new Date().toISOString(),
        tags: [],
        metadata: {},
        tokenCount: 8
      }
    ],
    recentMessages: [],
    userMessage: "What should we build?",
    config: {
      maxPromptTokens: 4000,
      maxRecentConversationTokens: 500,
      maxRetrievedMemoryTokens: 500
    }
  });
  expect(packed.messages[0]?.content).toContain("Retrieved long-term memory");
  expect(packed.includedMemoryIds).toEqual(["m1"]);
});

it("keeps the final pack within maxPromptTokens with huge current user, memory, and recent history", () => {
  const packed = packContext({
    systemPrompt: "You are local. Keep answers grounded in retrieved memory.",
    retrievedMemory: [
      memory({
        id: "m1",
        score: 0.98,
        text: "The user wants production readiness with tests and exact changed file paths."
      }),
      memory({
        id: "m2",
        score: 0.82,
        text: "The core package uses Vitest and approximate prompt token accounting."
      })
    ],
    recentMessages: [
      {
        id: "r1",
        role: "user",
        content: "Earlier question about context packing regressions and prompt budgets.",
        createdAt: "2026-05-11T00:00:00.000Z",
        sessionId: "s1"
      },
      {
        id: "r2",
        role: "assistant",
        content: "Earlier answer describing why final message totals must be checked.",
        createdAt: "2026-05-11T00:01:00.000Z",
        sessionId: "s1"
      }
    ],
    userMessage: "Current user message ".repeat(400),
    config: {
      maxPromptTokens: 180,
      maxRecentConversationTokens: 40,
      maxRetrievedMemoryTokens: 80
    }
  });

  expect(packed.estimatedTokens).toBeLessThanOrEqual(180);
  expect(packed.messages.at(-1)?.role).toBe("user");
  expect(packed.messages.at(-1)?.content).toContain("Current user message");
  expect(packed.messages.some((message) => message.role === "assistant")).toBe(true);
  expect(packed.includedMemoryIds.length).toBeGreaterThan(0);
});

it("reports only memory that actually fits in the final system memory block", () => {
  const packed = packContext({
    systemPrompt: "System.",
    retrievedMemory: [
      memory({
        id: "large",
        score: 0.99,
        text: "oversized memory ".repeat(160)
      }),
      memory({
        id: "compact",
        score: 0.5,
        text: "compact memory"
      })
    ],
    recentMessages: [],
    userMessage: "Hi",
    config: {
      maxPromptTokens: 48,
      maxRecentConversationTokens: 10,
      maxRetrievedMemoryTokens: 500
    }
  });

  expect(packed.estimatedTokens).toBeLessThanOrEqual(48);
  expect(packed.includedMemoryIds).toEqual(["compact"]);
  expect(packed.messages[0]?.content).toContain("compact memory");
  expect(packed.messages[0]?.content).not.toContain("oversized memory");
});

it("handles zero and negative prompt budgets without empty-budget pathologies", () => {
  for (const maxPromptTokens of [0, -20]) {
    const packed = packContext({
      systemPrompt: "System prompt.",
      retrievedMemory: [memory({ id: "m1", score: 1, text: "Important memory." })],
      recentMessages: [
        {
          id: "r1",
          role: "assistant",
          content: "Recent answer.",
          createdAt: "2026-05-11T00:00:00.000Z",
          sessionId: "s1"
        }
      ],
      userMessage: "Current question.",
      config: {
        maxPromptTokens,
        maxRecentConversationTokens: 50,
        maxRetrievedMemoryTokens: 50
      }
    });

    expect(packed.estimatedTokens).toBe(0);
    expect(packed.messages.map((message) => message.role)).toEqual(["system", "user"]);
    expect(packed.messages.at(-1)?.role).toBe("user");
    expect(packed.includedMemoryIds).toEqual([]);
  }
});
