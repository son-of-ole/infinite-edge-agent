import type { SSARoutingBlock, SSAWebGpuConfig } from "./types";

export interface SparseBlockSelection {
  queryBlockIndex: number;
  selectedBlockIds: string[];
  pinnedBlockIds: string[];
  localWindowBlockIds: string[];
  globalTopKBlockIds: string[];
}

export function selectSparseBlocksForQueryBlock(
  blocks: SSARoutingBlock[],
  queryBlockIndex: number,
  config: Pick<SSAWebGpuConfig, "topKBlocks" | "localWindowBlocks" | "pinnedAnchorBudget">,
): SparseBlockSelection {
  const causalCandidates = blocks.filter((block) => block.blockIndex <= queryBlockIndex);
  const pinned = causalCandidates
    .filter((block) => block.pinned)
    .sort((a, b) => a.blockIndex - b.blockIndex)
    .slice(0, config.pinnedAnchorBudget);

  const localWindowStart = Math.max(0, queryBlockIndex - config.localWindowBlocks);
  const localWindow = causalCandidates.filter(
    (block) => block.blockIndex >= localWindowStart && block.blockIndex <= queryBlockIndex,
  );

  const reserved = new Set([...pinned, ...localWindow].map((block) => block.id));
  const globalBudget = Math.max(0, config.topKBlocks - reserved.size);
  const globalTopK = causalCandidates
    .filter((block) => !reserved.has(block.id))
    .sort((a, b) => (b.score ?? 0) - (a.score ?? 0))
    .slice(0, globalBudget);

  const selected = uniqueById([...pinned, ...localWindow, ...globalTopK]);

  return {
    queryBlockIndex,
    selectedBlockIds: selected.map((block) => block.id),
    pinnedBlockIds: pinned.map((block) => block.id),
    localWindowBlockIds: localWindow.map((block) => block.id),
    globalTopKBlockIds: globalTopK.map((block) => block.id),
  };
}

function uniqueById(blocks: SSARoutingBlock[]): SSARoutingBlock[] {
  const seen = new Set<string>();
  const out: SSARoutingBlock[] = [];
  for (const block of blocks) {
    if (seen.has(block.id)) continue;
    seen.add(block.id);
    out.push(block);
  }
  return out.sort((a, b) => a.blockIndex - b.blockIndex);
}
