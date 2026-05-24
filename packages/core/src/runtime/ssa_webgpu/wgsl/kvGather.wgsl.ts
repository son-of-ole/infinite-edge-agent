export const kvGatherWgsl = /* wgsl */ `
// Gather selected KV blocks into contiguous buffers for sparse attention.
@group(0) @binding(0) var<storage, read> source: array<f32>;
@group(0) @binding(1) var<storage, read> selected_blocks: array<u32>;
@group(0) @binding(2) var<storage, read_write> gathered: array<f32>;

struct Params {
  selected_block_count: u32,
  block_size: u32,
  head_dim: u32,
  reserved: u32,
};
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  let total = params.selected_block_count * params.block_size * params.head_dim;
  if (idx >= total) { return; }

  let values_per_block = params.block_size * params.head_dim;
  let selected_slot = idx / values_per_block;
  let within_block = idx % values_per_block;
  let source_block = selected_blocks[selected_slot];
  let source_idx = source_block * values_per_block + within_block;
  gathered[idx] = source[source_idx];
}
`;
