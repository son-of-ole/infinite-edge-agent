export const blockScoreTopKWgsl = /* wgsl */ `
// Correctness-first placeholder for route scoring.
// Production top-k should be split into bounded passes for large contexts.
@group(0) @binding(0) var<storage, read> q_summary: array<f32>;
@group(0) @binding(1) var<storage, read> k_block_summary: array<f32>;
@group(0) @binding(2) var<storage, read_write> scores: array<f32>;

struct Params {
  query_blocks: u32,
  key_blocks: u32,
  head_dim: u32,
  reserved: u32,
};
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let qb = gid.x;
  let kb = gid.y;
  if (qb >= params.query_blocks || kb >= params.key_blocks) { return; }

  // Causal block mask.
  if (kb > qb) {
    scores[qb * params.key_blocks + kb] = -3.402823e38;
    return;
  }

  var acc = 0.0;
  for (var d: u32 = 0u; d < params.head_dim; d = d + 1u) {
    acc = acc + q_summary[qb * params.head_dim + d] * k_block_summary[kb * params.head_dim + d];
  }
  scores[qb * params.key_blocks + kb] = acc;
}
`;
