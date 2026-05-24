export const blockSummaryWgsl = /* wgsl */ `
// Correctness-first placeholder for SSA block summary generation.
// Production version computes mean-pooled K summaries per block/head.
@group(0) @binding(0) var<storage, read> k_cache: array<f32>;
@group(0) @binding(1) var<storage, read_write> block_summary: array<f32>;

struct Params {
  block_size: u32,
  head_dim: u32,
  blocks: u32,
  reserved: u32,
};
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  let total = params.blocks * params.head_dim;
  if (idx >= total) { return; }

  let block_id = idx / params.head_dim;
  let dim = idx % params.head_dim;
  var acc = 0.0;
  for (var t: u32 = 0u; t < params.block_size; t = t + 1u) {
    let token_index = block_id * params.block_size + t;
    acc = acc + k_cache[token_index * params.head_dim + dim];
  }
  block_summary[idx] = acc / f32(params.block_size);
}
`;
