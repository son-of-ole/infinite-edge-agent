import { expect, it } from "vitest";
import { chunkText } from "./chunking";

it("chunks text with overlap", () => {
  const text = Array.from({ length: 200 }, (_, index) => `sentence-${index}`).join(". ");
  const chunks = chunkText(text, { chunkTokens: 30, overlapTokens: 5, minChunkTokens: 1 });
  expect(chunks.length).toBeGreaterThan(1);
  expect(chunks[0]?.text.length).toBeGreaterThan(0);
});
