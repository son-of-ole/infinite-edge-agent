import { describe, expect, it } from "vitest";
import { DEFAULT_BROWSER_VECTOR_DIMENSION, getMemoryProviderCapabilities } from "./memoryProviders";

describe("memory provider capabilities", () => {
  it("reports deterministic browser-vector capability metadata including vector dimension", () => {
    expect(DEFAULT_BROWSER_VECTOR_DIMENSION).toBe(384);
    expect(getMemoryProviderCapabilities("browser-vector")).toMatchObject({
      mode: "browser-vector",
      storage: "indexeddb",
      localOnly: true,
      vectorSearch: true,
      deterministicSearch: true,
      metadataFilters: true,
      vectorDimension: 384,
      persistent: true,
      importExport: true,
      contextPackTracePersistence: true,
      remoteSync: false,
    });
  });
});
