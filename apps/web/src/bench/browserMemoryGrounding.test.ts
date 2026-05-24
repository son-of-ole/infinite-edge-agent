import type { ContextPackTraceRecord, RuntimeTrace, StoredMemoryChunk } from "@infinite-edge-agent/core";
import { beforeEach, describe, expect, it, vi } from "vitest";

const records = new Map<string, StoredMemoryChunk>();
const runtimeTraces = new Map<string, RuntimeTrace>();
const contextPackTraces = new Map<string, ContextPackTraceRecord>();

function installIndexedDbMock(): void {
  vi.doMock("idb", () => ({
    openDB: vi.fn(async () => ({
      transaction: vi.fn((storeName: string | string[]) => ({
        store: makeObjectStore(Array.isArray(storeName) ? storeName[0] ?? "chunks" : storeName),
        objectStore: vi.fn((name: string) => makeObjectStore(name)),
        done: Promise.resolve(),
      })),
      put: vi.fn(async (storeName: string, value: StoredMemoryChunk | RuntimeTrace | ContextPackTraceRecord) => {
        putRecord(storeName, value);
      }),
      getAll: vi.fn(async (storeName: string) => getRecords(storeName)),
      getAllFromIndex: vi.fn(async (storeName: string, indexName: string, value: string) =>
        getRecords(storeName).filter((record) => matchesIndex(record, indexName, value))
      ),
    })),
  }));
}

describe("browser memory grounding benchmark", () => {
  beforeEach(() => {
    records.clear();
    runtimeTraces.clear();
    contextPackTraces.clear();
    vi.resetModules();
    vi.doUnmock("idb");
    installIndexedDbMock();
  });

  it("seeds browser-vector memory, retrieves the Montana fact, and packs it into context", async () => {
    const {
      buildMemoryGroundedMessages,
      createMemoryGroundingHarness,
      readBrowserPreviewBenchmarkRequest,
    } = await import("./browserPreviewBenchmarkRoute");
    const request = readBrowserPreviewBenchmarkRequest(new URL(
      "http://localhost:5173/__bench/browser-runtime?strictWebGpu=true&memoryGrounding=montana_capital",
    ));
    const harness = await createMemoryGroundingHarness(request);

    const grounded = await buildMemoryGroundedMessages({
      harness,
      prompt: request.prompts[0]?.text ?? "",
    });

    expect(harness.corpus).toHaveLength(16);
    expect(grounded.proof).toMatchObject({
      corpusCount: 16,
      retrievedMemoryIds: ["bench_memory_montana_capital"],
      includedMemoryIds: ["bench_memory_montana_capital"],
      expectedMemoryIds: ["bench_memory_montana_capital"],
      expectedMemoryHitPassed: true,
      contextRebuildPassed: true,
      answerOnlyExpected: true,
    });
    expect(grounded.messages[0]?.role).toBe("system");
    expect(grounded.messages[0]?.content).toContain("Cedar Ridge operations dossier");
    expect(grounded.messages[0]?.content).toContain("Answer token: Helena");
  });

  it("seeds a larger QA corpus and retrieves the expected pinned fact for each QA prompt", async () => {
    const {
      buildMemoryGroundedMessages,
      createMemoryGroundingHarness,
      readBrowserPreviewBenchmarkRequest,
    } = await import("./browserPreviewBenchmarkRoute");
    const request = readBrowserPreviewBenchmarkRequest(new URL(
      "http://localhost:5173/__bench/browser-runtime?strictWebGpu=true&memoryGrounding=qa_corpus_v1&memoryCorpusSize=64",
    ));
    const harness = await createMemoryGroundingHarness(request);

    expect(harness.corpus).toHaveLength(64);
    expect(request.prompts).toHaveLength(6);
    expect(request.prompts.every((prompt) => !prompt.text.includes("MEMORY_FACT_"))).toBe(true);
    expect(harness.corpus
      .filter((chunk) => chunk.tags.includes("pinned"))
      .every((chunk) => !chunk.text.includes("MEMORY_FACT_"))
    ).toBe(true);

    const grounded = await Promise.all(request.prompts.map((prompt) =>
      buildMemoryGroundedMessages({ harness, prompt: prompt.text })
    ));

    expect(grounded.map((result) => result.proof.expectedMemoryIds[0])).toEqual([
      "bench_memory_montana_capital",
      "bench_memory_edge_runtime_sentinel",
      "bench_memory_desert_lantern_archive_color",
      "bench_memory_orbital_pier_calibration_code",
      "bench_memory_maple_lock_recovery_contact",
      "bench_memory_tidal_forge_retention_window",
    ]);
    expect(grounded.every((result) => result.proof.expectedMemoryHitPassed)).toBe(true);
    expect(grounded.every((result) => result.proof.contextRebuildPassed)).toBe(true);
    expect(grounded.map((result) => result.proof.corpusCount)).toEqual([64, 64, 64, 64, 64, 64]);
  });

  it("audits natural-language aliases in the QA corpus retrieval proof", async () => {
    const {
      createMemoryGroundingHarness,
      readBrowserPreviewBenchmarkRequest,
    } = await import("./browserPreviewBenchmarkRoute");
    const request = readBrowserPreviewBenchmarkRequest(new URL(
      "http://localhost:5173/__bench/browser-runtime?strictWebGpu=true&memoryGrounding=qa_corpus_v1&memoryCorpusSize=64",
    ));
    const harness = await createMemoryGroundingHarness(request);

    expect(harness.retrievalAudit).toMatchObject({
      corpusCount: 64,
      queryCount: 23,
      top1CorrectCount: 23,
      recallAt1: 1,
      canonicalQueryCount: 6,
      canonicalRecallAt1: 1,
      aliasQueryCount: 5,
      aliasRecallAt1: 1,
      generatedParaphraseQueryCount: 12,
      generatedParaphraseRecallAt1: 1,
      passed: true,
    });
    expect(harness.retrievalAudit?.queryClassBreakdown).toEqual([
      { queryClass: "canonical", queryCount: 6, top1CorrectCount: 6, recallAt1: 1, mrr: 1 },
      { queryClass: "alias", queryCount: 5, top1CorrectCount: 5, recallAt1: 1, mrr: 1 },
      { queryClass: "generated_paraphrase", queryCount: 12, top1CorrectCount: 12, recallAt1: 1, mrr: 1 },
    ]);
  });

  it("keeps the full QA corpus and audit when model-generation prompts are bounded", async () => {
    const {
      createMemoryGroundingHarness,
      readBrowserPreviewBenchmarkRequest,
    } = await import("./browserPreviewBenchmarkRoute");
    const request = readBrowserPreviewBenchmarkRequest(new URL(
      "http://localhost:5173/__bench/browser-runtime"
      + "?strictWebGpu=true"
      + "&memoryGrounding=qa_corpus_v1"
      + "&memoryCorpusSize=64"
      + "&memoryPromptLimit=2",
    ));
    const harness = await createMemoryGroundingHarness(request);

    expect(request.prompts.map((prompt) => prompt.expectedSubstrings[0])).toEqual([
      "Helena",
      "edge-runtime-ok",
    ]);
    expect(harness.corpus).toHaveLength(64);
    expect(harness.retrievalAudit).toMatchObject({
      corpusCount: 64,
      queryCount: 23,
      top1CorrectCount: 23,
      generatedParaphraseQueryCount: 12,
      generatedParaphraseRecallAt1: 1,
      passed: true,
    });
  });

  it("exposes generated-paraphrase QA proof in the browser audit-only payload", async () => {
    const {
      readBrowserPreviewBenchmarkRequest,
      runMemoryGroundingAuditOnlyBenchmark,
    } = await import("./browserPreviewBenchmarkRoute");
    const request = readBrowserPreviewBenchmarkRequest(new URL(
      "http://localhost:5173/__bench/browser-runtime?memoryGroundingAuditOnly=true&memoryGrounding=qa_corpus_v1&memoryCorpusSize=64",
    ));

    const payload = await runMemoryGroundingAuditOnlyBenchmark(request);

    expect(payload.passed).toBe(true);
    expect(payload.summary).toMatchObject({
      memoryGroundingAuditOnly: true,
      memoryGroundingPassed: true,
      memoryRetrievalAuditPassed: true,
      memoryRetrievalAuditQueryCount: 23,
      memoryGeneratedParaphraseRequired: true,
      memoryGeneratedParaphrasePassed: true,
      memoryGeneratedParaphraseQueryCount: 12,
      memoryGeneratedParaphraseTop1CorrectCount: 12,
      memoryGeneratedParaphraseRecallAt1: 1,
      memoryGeneratedParaphraseMrr: 1,
    });
  });

  it("passes Montana audit-only payloads on retrieval and context proof without requiring a corpus audit", async () => {
    const {
      readBrowserPreviewBenchmarkRequest,
      runMemoryGroundingAuditOnlyBenchmark,
    } = await import("./browserPreviewBenchmarkRoute");
    const request = readBrowserPreviewBenchmarkRequest(new URL(
      "http://localhost:5173/__bench/browser-runtime?memoryGroundingAuditOnly=true&memoryGrounding=montana_capital&memoryCorpusSize=64",
    ));

    const payload = await runMemoryGroundingAuditOnlyBenchmark(request);

    expect(payload.passed).toBe(true);
    expect(payload.summary).toMatchObject({
      memoryGroundingAuditOnly: true,
      memoryGroundingPassed: true,
      memoryExpectedHitPassed: true,
      memoryContextRebuildPassed: true,
      memoryRetrievalAuditRequired: false,
      memoryRetrievalAuditPassed: true,
      memoryRetrievalAuditQueryCount: 0,
    });
  });

  it("audits rank-1 retrieval across a large synthetic browser-vector corpus", async () => {
    const {
      buildMemoryGroundedMessages,
      createMemoryGroundingHarness,
      readBrowserPreviewBenchmarkRequest,
    } = await import("./browserPreviewBenchmarkRoute");
    const request = readBrowserPreviewBenchmarkRequest(new URL(
      "http://localhost:5173/__bench/browser-runtime"
      + "?strictWebGpu=true"
      + "&memoryGrounding=large_synthetic_v1"
      + "&memoryCorpusSize=1024",
    ));
    const harness = await createMemoryGroundingHarness(request);

    expect(harness.corpus).toHaveLength(1024);
    expect(request.prompts).toHaveLength(5);
    expect(request.prompts[0]?.text).toContain("Helix Ledger synthetic corpus");
    expect(harness.retrievalAudit).toMatchObject({
      corpusCount: 1024,
      queryCount: 64,
      top1CorrectCount: 64,
      recallAt1: 1,
      passed: true,
    });

    const grounded = await buildMemoryGroundedMessages({
      harness,
      prompt: request.prompts[3]?.text ?? "",
    });

    expect(grounded.proof.expectedMemoryHitPassed).toBe(true);
    expect(grounded.proof.contextRebuildPassed).toBe(true);
    expect(grounded.proof.retrievalRank).toBe(1);
    expect(grounded.proof.retrievalAudit).toMatchObject({
      queryCount: 64,
      recallAt1: 1,
      passed: true,
    });
  });

  it("does not pass memory proof vacuously for unmatched explicit QA prompts", async () => {
    const {
      buildMemoryGroundedMessages,
      createMemoryGroundingHarness,
      readBrowserPreviewBenchmarkRequest,
    } = await import("./browserPreviewBenchmarkRoute");
    const request = readBrowserPreviewBenchmarkRequest(new URL(
      "http://localhost:5173/__bench/browser-runtime"
      + "?strictWebGpu=true"
      + "&memoryGrounding=qa_corpus_v1"
      + "&memoryCorpusSize=64"
      + "&prompt=Using%20retrieved%20memory%20only%2C%20what%20is%20the%20unlisted%20answer%3F"
      + "&expected=unknown",
    ));
    const harness = await createMemoryGroundingHarness(request);

    const grounded = await buildMemoryGroundedMessages({
      harness,
      prompt: request.prompts[0]?.text ?? "",
    });

    expect(grounded.proof.expectedMemoryIds).toEqual([]);
    expect(grounded.proof.expectedMemoryHitPassed).toBe(false);
    expect(grounded.proof.contextRebuildPassed).toBe(false);
  });

  it("resolves a natural Montana capital paraphrase to the seeded grounded fact", async () => {
    const {
      buildMemoryGroundedMessages,
      createMemoryGroundingHarness,
      readBrowserPreviewBenchmarkRequest,
    } = await import("./browserPreviewBenchmarkRoute");
    const request = readBrowserPreviewBenchmarkRequest(new URL(
      "http://localhost:5173/__bench/browser-runtime"
      + "?strictWebGpu=true"
      + "&memoryGrounding=qa_corpus_v1"
      + "&memoryCorpusSize=64"
      + "&prompt=Using%20retrieved%20memory%20only%2C%20what%20is%20the%20capital%20of%20Montana%3F%20Answer%20only.",
    ));
    const harness = await createMemoryGroundingHarness(request);

    const grounded = await buildMemoryGroundedMessages({
      harness,
      prompt: request.prompts[0]?.text ?? "",
    });

    expect(request.prompts[0]).toMatchObject({
      expectedSubstrings: ["Helena"],
      expectedExact: ["Helena"],
    });
    expect(grounded.proof).toMatchObject({
      retrievedMemoryIds: ["bench_memory_montana_capital"],
      includedMemoryIds: ["bench_memory_montana_capital"],
      expectedMemoryIds: ["bench_memory_montana_capital"],
      expectedMemoryHitPassed: true,
      contextRebuildPassed: true,
      retrievalRank: 1,
    });
  });

  it("resolves free-order Montana capital wording without the canonical dossier phrase", async () => {
    const {
      buildMemoryGroundedMessages,
      createMemoryGroundingHarness,
      readBrowserPreviewBenchmarkRequest,
    } = await import("./browserPreviewBenchmarkRoute");
    const request = readBrowserPreviewBenchmarkRequest(new URL(
      "http://localhost:5173/__bench/browser-runtime"
      + "?strictWebGpu=true"
      + "&memoryGrounding=qa_corpus_v1"
      + "&memoryCorpusSize=64"
      + "&prompt=Using%20retrieved%20memory%20only%2C%20which%20city%20is%20Montana%27s%20state%20capital%3F%20Answer%20only.",
    ));
    const harness = await createMemoryGroundingHarness(request);

    const grounded = await buildMemoryGroundedMessages({
      harness,
      prompt: request.prompts[0]?.text ?? "",
    });

    expect(request.prompts[0]).toMatchObject({
      expectedSubstrings: ["Helena"],
      expectedExact: ["Helena"],
    });
    expect(grounded.proof).toMatchObject({
      expectedMemoryIds: ["bench_memory_montana_capital"],
      expectedMemoryHitPassed: true,
      contextRebuildPassed: true,
      retrievalRank: 1,
    });
  });

  it("resolves a natural sentinel paraphrase through hybrid fact metadata", async () => {
    const {
      buildMemoryGroundedMessages,
      createMemoryGroundingHarness,
      readBrowserPreviewBenchmarkRequest,
    } = await import("./browserPreviewBenchmarkRoute");
    const request = readBrowserPreviewBenchmarkRequest(new URL(
      "http://localhost:5173/__bench/browser-runtime"
      + "?strictWebGpu=true"
      + "&memoryGrounding=qa_corpus_v1"
      + "&memoryCorpusSize=64"
      + "&prompt=Using%20retrieved%20memory%20only%2C%20which%20token%20does%20the%20browser%20production%20deployment%20check%20need%20to%20return%3F%20Answer%20only.",
    ));
    const harness = await createMemoryGroundingHarness(request);

    const grounded = await buildMemoryGroundedMessages({
      harness,
      prompt: request.prompts[0]?.text ?? "",
    });

    expect(request.prompts[0]).toMatchObject({
      expectedSubstrings: ["edge-runtime-ok"],
      expectedExact: ["edge-runtime-ok"],
    });
    expect(grounded.proof).toMatchObject({
      retrievedMemoryIds: ["bench_memory_edge_runtime_sentinel"],
      includedMemoryIds: ["bench_memory_edge_runtime_sentinel"],
      expectedMemoryIds: ["bench_memory_edge_runtime_sentinel"],
      expectedMemoryHitPassed: true,
      contextRebuildPassed: true,
      retrievalRank: 1,
    });
  });
});

function makeObjectStore(storeName: string) {
  return {
    put: vi.fn(async (value: StoredMemoryChunk | RuntimeTrace | ContextPackTraceRecord) => {
      putRecord(storeName, value);
    }),
    clear: vi.fn(async () => {
      getStore(storeName).clear();
    }),
    delete: vi.fn(async (id: string) => {
      getStore(storeName).delete(id);
    }),
    getAll: vi.fn(async () => getRecords(storeName)),
  };
}

function putRecord(storeName: string, value: StoredMemoryChunk | RuntimeTrace | ContextPackTraceRecord): void {
  if (storeName === "chunks") records.set((value as StoredMemoryChunk).id, value as StoredMemoryChunk);
  if (storeName === "runtimeTraces") runtimeTraces.set((value as RuntimeTrace).traceId, value as RuntimeTrace);
  if (storeName === "contextPackTraces") contextPackTraces.set((value as ContextPackTraceRecord).id, value as ContextPackTraceRecord);
}

function getRecords(storeName: string): Array<StoredMemoryChunk | RuntimeTrace | ContextPackTraceRecord> {
  return [...getStore(storeName).values()];
}

function getStore(storeName: string): Map<string, StoredMemoryChunk | RuntimeTrace | ContextPackTraceRecord> {
  if (storeName === "chunks") return records as Map<string, StoredMemoryChunk | RuntimeTrace | ContextPackTraceRecord>;
  if (storeName === "runtimeTraces") return runtimeTraces as Map<string, StoredMemoryChunk | RuntimeTrace | ContextPackTraceRecord>;
  if (storeName === "contextPackTraces") return contextPackTraces as Map<string, StoredMemoryChunk | RuntimeTrace | ContextPackTraceRecord>;
  return records as Map<string, StoredMemoryChunk | RuntimeTrace | ContextPackTraceRecord>;
}

function matchesIndex(
  record: StoredMemoryChunk | RuntimeTrace | ContextPackTraceRecord,
  indexName: string,
  value: string,
): boolean {
  if (indexName === "by_session") return "sessionId" in record && record.sessionId === value;
  if (indexName === "by_created") return "createdAt" in record && record.createdAt === value;
  if (indexName === "by_context_pack") return "contextPackId" in record && record.contextPackId === value;
  return false;
}
