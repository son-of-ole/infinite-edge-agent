import { describe, expect, it, vi } from "vitest";
import { CompiledWebLlmClient, createDefaultWebLlmModuleLoader } from "./compiledWebLlmClient";

describe("CompiledWebLlmClient", () => {
  it("streams OpenAI-compatible WebLLM chat chunks through the shared ChatClient interface", async () => {
    const create = vi.fn(async () => ({
      chat: {
        completions: {
          create: vi.fn(async function* () {
            yield { choices: [{ delta: { content: "Hel" } }] };
            yield { choices: [{ delta: { content: "ena" } }] };
          }),
        },
      },
      unload: vi.fn(),
    }));
    const client = new CompiledWebLlmClient({
      modelId: "Qwen3-0.6B-q4f16_1-MLC",
      moduleLoader: async () => ({ CreateMLCEngine: create }),
    });

    await client.init();
    const chunks: string[] = [];
    const result = await collectStream(client.streamChat([
      { role: "system", content: "Use retrieved memory." },
      { role: "user", content: "Answer only: Helena" },
    ], {
      maxTokens: 4,
      temperature: 0.2,
      topP: 0.9,
      stopAfterSequences: ["\n"],
    }), chunks);

    expect(result).toBe("Helena");
    expect(chunks).toEqual(["Hel", "ena"]);
    expect(create).toHaveBeenCalledWith("Qwen3-0.6B-q4f16_1-MLC", expect.objectContaining({
      initProgressCallback: expect.any(Function),
    }));
    expect(client.lastDecodeProof).toMatchObject({
      backendId: "compiled-browser-webllm",
      adapterKind: "compiled-browser",
      modelId: "Qwen3-0.6B-q4f16_1-MLC",
      streaming: true,
      generatedTokenEstimate: 1,
    });
    expect(client.lastGeneratedTokenTexts).toEqual(["Helena"]);
    expect(client.lastGenerationStopReason).toBe("stream_complete");
  });

  it("maps stop sequences and sampling options to the WebLLM request", async () => {
    const completionCreate = vi.fn(async function* () {
      yield { choices: [{ delta: { content: "ok" } }] };
    });
    const client = new CompiledWebLlmClient({
      modelId: "Qwen3-0.6B-q4f16_1-MLC",
      qwenThinkingMode: "enabled",
      moduleLoader: async () => ({
        CreateMLCEngine: async () => ({
          chat: { completions: { create: completionCreate } },
        }),
      }),
    });

    await client.init();
    await collectStream(client.streamChat([{ role: "user", content: "Return ok" }], {
      maxTokens: 8,
      temperature: 0.7,
      topP: 0.95,
      stopAfterSequences: ["</s>", "\n"],
    }));

    expect(completionCreate).toHaveBeenCalledWith({
      messages: [{ role: "user", content: "Return ok" }],
      stream: true,
      max_tokens: 8,
      temperature: 0.7,
      top_p: 0.95,
      stop: ["</s>", "\n"],
    });
  });

  it("appends Qwen no-think directive to the final user turn by default", async () => {
    const completionCreate = vi.fn(async function* () {
      yield { choices: [{ delta: { content: "Helena" } }] };
    });
    const client = new CompiledWebLlmClient({
      modelId: "Qwen3-0.6B-q4f16_1-MLC",
      moduleLoader: async () => ({
        CreateMLCEngine: async () => ({
          chat: { completions: { create: completionCreate } },
        }),
      }),
    });

    await client.init();
    await collectStream(client.streamChat([
      { role: "system", content: "Use retrieved memory." },
      { role: "user", content: "What is the capital of Montana? Answer only." },
    ]));

    expect(completionCreate).toHaveBeenCalledWith(expect.objectContaining({
      messages: [
        { role: "system", content: "Use retrieved memory." },
        { role: "user", content: "What is the capital of Montana? Answer only.\n/no_think" },
      ],
      extra_body: { enable_thinking: false },
    }));
  });

  it("does not append no-think when Qwen thinking mode is explicitly enabled", async () => {
    const completionCreate = vi.fn(async function* () {
      yield { choices: [{ delta: { content: "<think>" } }] };
    });
    const client = new CompiledWebLlmClient({
      modelId: "Qwen3-0.6B-q4f16_1-MLC",
      qwenThinkingMode: "enabled",
      moduleLoader: async () => ({
        CreateMLCEngine: async () => ({
          chat: { completions: { create: completionCreate } },
        }),
      }),
    });

    await client.init();
    await collectStream(client.streamChat([
      { role: "user", content: "Think through this." },
    ]));

    expect(completionCreate).toHaveBeenCalledWith(expect.objectContaining({
      messages: [{ role: "user", content: "Think through this." }],
    }));
  });

  it("filters Qwen empty thinking scaffold from streamed visible output", async () => {
    const client = new CompiledWebLlmClient({
      modelId: "Qwen3-0.6B-q4f16_1-MLC",
      moduleLoader: async () => ({
        CreateMLCEngine: async () => ({
          chat: {
            completions: {
              create: vi.fn(async function* () {
                yield { choices: [{ delta: { content: "<think>\n" } }] };
                yield { choices: [{ delta: { content: "\n</think>\n\nHel" } }] };
                yield { choices: [{ delta: { content: "ena" } }] };
              }),
            },
          },
        }),
      }),
    });

    await client.init();
    const chunks: string[] = [];
    const result = await collectStream(client.streamChat([
      { role: "user", content: "Answer only: Helena" },
    ]), chunks);

    expect(result).toBe("Helena");
    expect(chunks).toEqual(["Hel", "ena"]);
    expect(client.lastDecodeProof?.generatedText).toBe("Helena");
  });

  it("disposes the compiled engine when WebLLM exposes unload", async () => {
    const unload = vi.fn();
    const client = new CompiledWebLlmClient({
      modelId: "Qwen3-0.6B-q4f16_1-MLC",
      moduleLoader: async () => ({
        CreateMLCEngine: async () => ({
          chat: { completions: { create: vi.fn() } },
          unload,
        }),
      }),
    });

    await client.init();
    await client.dispose();

    expect(unload).toHaveBeenCalledTimes(1);
  });

  it("fails with an install-oriented message when the default WebLLM loader cannot resolve the package", async () => {
    await expect(createDefaultWebLlmModuleLoader(async () => {
      throw new Error("Cannot find package");
    })()).rejects.toThrow("Install @mlc-ai/web-llm or provide a bundled WebLLM module loader");
  });
});

async function collectStream(
  stream: AsyncGenerator<string, string, void>,
  chunks: string[] = [],
): Promise<string> {
  let next = await stream.next();
  while (!next.done) {
    chunks.push(next.value);
    next = await stream.next();
  }
  return next.value;
}
