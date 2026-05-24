export const sparseAttentionWgsl = /* wgsl */ `
// Correctness-first sparse attention kernel for tiny SSA parity fixtures.
// This intentionally prioritizes clarity over throughput: each output cell
// recomputes the selected-token softmax so the kernel can be validated against
// the CPU sparse reference before a tiled production path is introduced.
@group(0) @binding(0) var<storage, read> q: array<f32>;
@group(0) @binding(1) var<storage, read> k: array<f32>;
@group(0) @binding(2) var<storage, read> v: array<f32>;
@group(0) @binding(3) var<storage, read> selected_indices: array<i32>;
@group(0) @binding(4) var<storage, read_write> out: array<f32>;

struct Params {
  query_tokens: u32,
  key_tokens: u32,
  head_dim: u32,
  max_selected: u32,
  causal: u32,
  scale: f32,
  reserved1: u32,
  reserved2: u32,
};
@group(0) @binding(5) var<uniform> params: Params;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let qt = gid.x;
  let dim = gid.y;
  if (qt >= params.query_tokens || dim >= params.head_dim) { return; }

  var max_score = -3.402823e38;
  let scale = params.scale;

  for (var slot: u32 = 0u; slot < params.max_selected; slot = slot + 1u) {
    let key_index_i = selected_indices[qt * params.max_selected + slot];
    if (key_index_i < 0) { continue; }
    let key_index = u32(key_index_i);
    if (key_index >= params.key_tokens) { continue; }
    if (params.causal == 1u && key_index > qt) { continue; }

    var score = 0.0;
    for (var d: u32 = 0u; d < params.head_dim; d = d + 1u) {
      score = score + q[qt * params.head_dim + d] * k[key_index * params.head_dim + d];
    }
    max_score = max(max_score, score * scale);
  }

  var denom = 0.0;
  var weighted = 0.0;
  for (var slot2: u32 = 0u; slot2 < params.max_selected; slot2 = slot2 + 1u) {
    let key_index_i = selected_indices[qt * params.max_selected + slot2];
    if (key_index_i < 0) { continue; }
    let key_index = u32(key_index_i);
    if (key_index >= params.key_tokens) { continue; }
    if (params.causal == 1u && key_index > qt) { continue; }

    var score = 0.0;
    for (var d2: u32 = 0u; d2 < params.head_dim; d2 = d2 + 1u) {
      score = score + q[qt * params.head_dim + d2] * k[key_index * params.head_dim + d2];
    }
    let weight = exp(score * scale - max_score);
    denom = denom + weight;
    weighted = weighted + weight * v[key_index * params.head_dim + dim];
  }

  if (denom == 0.0) {
    out[qt * params.head_dim + dim] = 0.0;
  } else {
    out[qt * params.head_dim + dim] = weighted / denom;
  }
}
`;

export const packedSparseAttentionWgsl = /* wgsl */ `
// Fused packed-head sparse attention. One dispatch computes every query/head
// output column, avoiding a separate browser/GPU submission and readback per head.
@group(0) @binding(0) var<storage, read> q: array<f32>;
@group(0) @binding(1) var<storage, read> k: array<f32>;
@group(0) @binding(2) var<storage, read> v: array<f32>;
@group(0) @binding(3) var<storage, read> selected_indices: array<i32>;
@group(0) @binding(4) var<storage, read_write> out: array<f32>;

struct Params {
  query_tokens: u32,
  key_tokens: u32,
  head_dim: u32,
  max_selected: u32,
  causal: u32,
  scale: f32,
  head_count: u32,
  kv_head_count: u32,
};
@group(0) @binding(5) var<uniform> params: Params;

@compute @workgroup_size(8, 8)
fn packed_sparse_attention(@builtin(global_invocation_id) gid: vec3<u32>) {
  let qt = gid.x;
  let packed_dim = gid.y;
  let output_width = params.head_count * params.head_dim;
  if (qt >= params.query_tokens || packed_dim >= output_width) { return; }

  let head = packed_dim / params.head_dim;
  let dim = packed_dim - head * params.head_dim;
  let kv_head = min(params.kv_head_count - 1u, (head * params.kv_head_count) / params.head_count);
  let q_width = params.head_count * params.head_dim;
  let kv_width = params.kv_head_count * params.head_dim;
  let scale = params.scale;
  var max_score = -3.402823e38;

  for (var slot: u32 = 0u; slot < params.max_selected; slot = slot + 1u) {
    let key_index_i = selected_indices[qt * params.max_selected + slot];
    if (key_index_i < 0) { continue; }
    let key_index = u32(key_index_i);
    if (key_index >= params.key_tokens) { continue; }
    if (params.causal == 1u && key_index > qt) { continue; }

    var score = 0.0;
    for (var d: u32 = 0u; d < params.head_dim; d = d + 1u) {
      score = score + q[qt * q_width + head * params.head_dim + d] * k[key_index * kv_width + kv_head * params.head_dim + d];
    }
    max_score = max(max_score, score * scale);
  }

  var denom = 0.0;
  var weighted = 0.0;
  for (var slot2: u32 = 0u; slot2 < params.max_selected; slot2 = slot2 + 1u) {
    let key_index_i = selected_indices[qt * params.max_selected + slot2];
    if (key_index_i < 0) { continue; }
    let key_index = u32(key_index_i);
    if (key_index >= params.key_tokens) { continue; }
    if (params.causal == 1u && key_index > qt) { continue; }

    var score = 0.0;
    for (var d2: u32 = 0u; d2 < params.head_dim; d2 = d2 + 1u) {
      score = score + q[qt * q_width + head * params.head_dim + d2] * k[key_index * kv_width + kv_head * params.head_dim + d2];
    }
    let weight = exp(score * scale - max_score);
    denom = denom + weight;
    weighted = weighted + weight * v[key_index * kv_width + kv_head * params.head_dim + dim];
  }

  if (denom == 0.0) {
    out[qt * output_width + packed_dim] = 0.0;
  } else {
    out[qt * output_width + packed_dim] = weighted / denom;
  }
}
`;

export const packedSparseAttentionDecodeWgsl = /* wgsl */ `
// Decode-specialized packed-head attention for large single-query contexts.
// Stage 1 computes Q.K once per selected key/head; stage 2 reuses those scores
// across every output dimension instead of recomputing Q.K per column.
@group(0) @binding(0) var<storage, read> q: array<f32>;
@group(0) @binding(1) var<storage, read> k: array<f32>;
@group(0) @binding(2) var<storage, read> selected_indices: array<i32>;
@group(0) @binding(3) var<storage, read_write> scores: array<f32>;

struct Params {
  query_tokens: u32,
  key_tokens: u32,
  head_dim: u32,
  max_selected: u32,
  causal: u32,
  scale: f32,
  head_count: u32,
  kv_head_count: u32,
};
@group(0) @binding(4) var<uniform> score_params: Params;

@compute @workgroup_size(1, 4, 64)
fn packed_sparse_attention_decode_scores(@builtin(global_invocation_id) gid: vec3<u32>) {
  let qt = gid.x;
  let head = gid.y;
  let slot = gid.z;
  if (qt >= score_params.query_tokens || head >= score_params.head_count || slot >= score_params.max_selected) { return; }

  let score_index = (qt * score_params.head_count + head) * score_params.max_selected + slot;
  let key_index_i = selected_indices[qt * score_params.max_selected + slot];
  if (key_index_i < 0) {
    scores[score_index] = -3.402823e38;
    return;
  }
  let key_index = u32(key_index_i);
  if (key_index >= score_params.key_tokens) {
    scores[score_index] = -3.402823e38;
    return;
  }
  if (score_params.causal == 1u && key_index > qt) {
    scores[score_index] = -3.402823e38;
    return;
  }

  let kv_head = min(score_params.kv_head_count - 1u, (head * score_params.kv_head_count) / score_params.head_count);
  let q_width = score_params.head_count * score_params.head_dim;
  let kv_width = score_params.kv_head_count * score_params.head_dim;
  var score = 0.0;
  for (var d: u32 = 0u; d < score_params.head_dim; d = d + 1u) {
    score = score + q[qt * q_width + head * score_params.head_dim + d] * k[key_index * kv_width + kv_head * score_params.head_dim + d];
  }
  scores[score_index] = score * score_params.scale;
}

@group(1) @binding(0) var<storage, read> scored_values: array<f32>;
@group(1) @binding(1) var<storage, read> v: array<f32>;
@group(1) @binding(2) var<storage, read> output_selected_indices: array<i32>;
@group(1) @binding(3) var<storage, read_write> out: array<f32>;
@group(1) @binding(4) var<uniform> output_params: Params;

@compute @workgroup_size(1, 8)
fn packed_sparse_attention_decode_output(@builtin(global_invocation_id) gid: vec3<u32>) {
  let qt = gid.x;
  let packed_dim = gid.y;
  let output_width = output_params.head_count * output_params.head_dim;
  if (qt >= output_params.query_tokens || packed_dim >= output_width) { return; }

  let head = packed_dim / output_params.head_dim;
  let dim = packed_dim - head * output_params.head_dim;
  let kv_head = min(output_params.kv_head_count - 1u, (head * output_params.kv_head_count) / output_params.head_count);
  let kv_width = output_params.kv_head_count * output_params.head_dim;
  let score_base = (qt * output_params.head_count + head) * output_params.max_selected;
  var max_score = -3.402823e38;

  for (var slot: u32 = 0u; slot < output_params.max_selected; slot = slot + 1u) {
    let key_index_i = output_selected_indices[qt * output_params.max_selected + slot];
    if (key_index_i < 0) { continue; }
    let key_index = u32(key_index_i);
    if (key_index >= output_params.key_tokens) { continue; }
    if (output_params.causal == 1u && key_index > qt) { continue; }
    max_score = max(max_score, scored_values[score_base + slot]);
  }

  var denom = 0.0;
  var weighted = 0.0;
  for (var slot2: u32 = 0u; slot2 < output_params.max_selected; slot2 = slot2 + 1u) {
    let key_index_i = output_selected_indices[qt * output_params.max_selected + slot2];
    if (key_index_i < 0) { continue; }
    let key_index = u32(key_index_i);
    if (key_index >= output_params.key_tokens) { continue; }
    if (output_params.causal == 1u && key_index > qt) { continue; }
    let weight = exp(scored_values[score_base + slot2] - max_score);
    denom = denom + weight;
    weighted = weighted + weight * v[key_index * kv_width + kv_head * output_params.head_dim + dim];
  }

  if (denom == 0.0) {
    out[qt * output_width + packed_dim] = 0.0;
  } else {
    out[qt * output_width + packed_dim] = weighted / denom;
  }
}
`;
