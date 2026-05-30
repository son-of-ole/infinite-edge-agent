import { describe, expect, it, vi } from "vitest";
import {
  buildBenchmarkTelemetryPayload,
  sanitizeBenchmarkArtifact,
  submitBenchmarkTelemetry,
} from "./benchmarkTelemetry";
import type { BrowserPreviewBenchmarkPayload } from "./browserPreviewBenchmark";

describe("benchmark telemetry", () => {
  it("builds a compact device and readiness payload without raw user-agent storage", () => {
    const payload = buildBenchmarkTelemetryPayload({
      benchmarkPayload: makePayload(),
      config: {
        enabled: true,
        url: "/api/benchmark-runs",
        appVersion: "0.1.0",
        gitSha: "abc123",
        deployUrl: "https://agent.example.com",
      },
      browserContext: {
        userAgent: "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Chrome/125.0.0.0 Safari/537.36",
        hardwareConcurrency: 10,
        deviceMemoryGb: 8,
        mobile: false,
        screenWidth: 1440,
        screenHeight: 900,
        webgpuAvailable: true,
      },
    });

    expect(payload).toMatchObject({
      appVersion: "0.1.0",
      gitSha: "abc123",
      deployUrl: "https://agent.example.com",
      benchmarkProfile: "full",
      backendId: "compiled-browser-webllm",
      device: {
        os: "macOS",
        browserName: "Chrome",
        browserVersion: "125.0.0.0",
        hardwareConcurrency: 10,
        deviceMemoryGb: 8,
        webgpuAvailable: true,
      },
      summary: {
        tokensPerSecond: 9.1,
        memoryGroundingPassed: true,
        expectedExactPassed: true,
        productionDeployReadyPassed: true,
        compiledBackendReadyPassed: true,
      },
    });
    expect(payload.device.userAgentHash).toMatch(/^[a-f0-9]{8}$/);
    expect(JSON.stringify(payload)).not.toContain("Mozilla/5.0");
  });

  it("sanitizes raw prompts, responses, expected strings, and token diagnostics before upload", () => {
    const sanitized = sanitizeBenchmarkArtifact(makePayload());
    const serialized = JSON.stringify(sanitized);

    expect(serialized).not.toContain("private question");
    expect(serialized).not.toContain("Helena");
    expect(serialized).not.toContain("raw-token");
    expect(serialized).toContain("[redacted]");
    expect(serialized).toContain("\"matched\":true");
  });

  it("requires both explicit user request and configured endpoint before submitting", async () => {
    const fetcher = vi.fn();
    await expect(submitBenchmarkTelemetry({
      requested: false,
      config: { enabled: true, url: "/api/benchmark-runs" },
      benchmarkPayload: makePayload(),
      fetcher,
    })).resolves.toMatchObject({
      requested: false,
      configured: true,
      submitted: false,
    });
    await expect(submitBenchmarkTelemetry({
      requested: true,
      config: { enabled: false, url: "/api/benchmark-runs" },
      benchmarkPayload: makePayload(),
      fetcher,
    })).resolves.toMatchObject({
      requested: true,
      configured: false,
      submitted: false,
    });
    expect(fetcher).not.toHaveBeenCalled();
  });

  it("posts sanitized telemetry to the configured endpoint when opted in", async () => {
    const fetcher = vi.fn(async (_input: RequestInfo | URL, _init?: RequestInit): Promise<Response> =>
      new Response(null, { status: 202 })
    );
    await expect(submitBenchmarkTelemetry({
      requested: true,
      config: { enabled: true, url: "/api/benchmark-runs" },
      benchmarkPayload: makePayload(),
      fetcher,
      browserContext: { userAgent: "Mozilla/5.0 Windows NT Chrome/125.0.0.0" },
    })).resolves.toMatchObject({
      requested: true,
      configured: true,
      submitted: true,
      status: 202,
    });

    expect(fetcher).toHaveBeenCalledWith("/api/benchmark-runs", expect.objectContaining({
      method: "POST",
      headers: { "content-type": "application/json" },
    }));
    const call = fetcher.mock.calls[0];
    expect(call).toBeDefined();
    const init = call?.[1];
    expect(init).toBeDefined();
    const body = JSON.parse(String(init!.body));
    expect(JSON.stringify(body)).not.toContain("private question");
    expect(JSON.stringify(body)).not.toContain("Helena");
  });
});

function makePayload(): BrowserPreviewBenchmarkPayload {
  return {
    name: "browser-preview-benchmark",
    createdAt: "2026-05-30T00:00:00.000Z",
    passed: true,
    summary: {
      profile: "full",
      runtimeBackendId: "compiled-browser-webllm",
      meanInitLoadMs: 100,
      meanTimeToFirstTokenMs: 10,
      meanTokensPerSecond: 9.1,
      memoryGroundingPassed: true,
      expectedExactPassed: true,
      productionDeployReadyPassed: true,
      compiledBackendReadyPassed: true,
    },
    runs: [{
      promptId: "prompt-1",
      prompt: "private question about a benchmark",
      response: "Helena",
      coherent: true,
      expectedSubstrings: ["Helena"],
      expectedSubstringMatches: ["Helena"],
      expectedExact: ["Helena"],
      expectedExactMatches: [{ expected: "Helena", matched: true }],
      metrics: {
        initLoadMs: 100,
        prefillMs: 0,
        timeToFirstTokenMs: 10,
        decodeLatencyMs: 20,
        tokensPerSecond: 9.1,
        generatedTokens: 1,
      },
      tokenDiagnostics: {
        promptTokenHeadIds: [1],
        promptTokenTailIds: [2],
        generatedTokenIds: [3],
        generatedTokenTexts: ["raw-token"],
      },
      runtimeTrace: {
        backend: "compiled-browser-webllm",
        tensorControl: false,
        tspSteps: [],
        kvPagingEvents: 0,
        selectedBlockIds: [],
      },
      predictive: {
        promptTokenCount: 1,
        generatedTokenCount: 1,
        selectedBlockCount: 0,
        kvPagingEventCount: 0,
        tspStepCount: 0,
      },
      webGpu: {
        available: true,
        requestedBackendPreference: "compiled-browser",
        logitProjectionBackend: "backend_native",
        cpuFallbackUsed: false,
        noCpuFallback: true,
        requestedGates: [],
        passedGates: [],
        failedGates: [],
        positiveKernelProof: true,
      },
      mtp: {
        mode: "target_only",
        acceptedTokens: 0,
        rejectedTokens: 0,
        acceptanceRate: 0,
        numSpeculativeTokens: 0,
        verifiedTokenCount: 0,
        targetDecodeCalls: 1,
        verifierStrategy: "none",
      },
      kvPersistence: {
        enabled: false,
        mode: "backend_native",
        eventCount: 0,
        persistEvents: 0,
        hydrateEvents: 0,
        reuseEvents: 0,
        predictedHotBlocks: [],
        prefetchedBlocks: [],
      },
      memoryGrounding: {
        mode: "seeded_browser_vector_context_rebuild",
        caseId: "montana_capital",
        corpusCount: 16,
        retrievedMemoryIds: ["bench_memory_montana_capital"],
        includedMemoryIds: ["bench_memory_montana_capital"],
        expectedMemoryIds: ["bench_memory_montana_capital"],
        expectedMemoryHitPassed: true,
        contextRebuildPassed: true,
        answerOnlyExpected: true,
        answerOnlyPassed: true,
        contextEstimatedTokens: 149,
        retrievalMs: 1,
        contextRebuildMs: 1,
      },
      expectedAnswerOnlyPassed: true,
      generationStopReason: "stream_complete",
    }],
  };
}
