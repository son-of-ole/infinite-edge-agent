export const QWEN_QKV_NORM_ROPE_APPEND_KV_WGSL = /* wgsl */ `
struct Params {
  num_q_heads: u32,
  num_kv_heads: u32,
  head_dim: u32,
  position: u32,
  seq_stride: u32,
  eps: f32,
};

@group(0) @binding(0) var<storage, read_write> q: array<f32>;
@group(0) @binding(1) var<storage, read_write> k: array<f32>;
@group(0) @binding(2) var<storage, read> v: array<f32>;
@group(0) @binding(3) var<storage, read> q_norm_weight: array<f32>;
@group(0) @binding(4) var<storage, read> k_norm_weight: array<f32>;
@group(0) @binding(5) var<storage, read> cos_table: array<f32>;
@group(0) @binding(6) var<storage, read> sin_table: array<f32>;
@group(0) @binding(7) var<storage, read_write> k_cache: array<f32>;
@group(0) @binding(8) var<storage, read_write> v_cache: array<f32>;
@group(0) @binding(9) var<uniform> params: Params;

fn rms_for_head(values: ptr<storage, array<f32>, read_write>, head: u32, weight: ptr<storage, array<f32>, read>) {
  var ss = 0.0;
  var i = 0u;
  let base = head * params.head_dim;
  loop {
    if (i >= params.head_dim) { break; }
    let x = (*values)[base + i];
    ss += x * x;
    i += 1u;
  }
  let inv = inverseSqrt(ss / f32(params.head_dim) + params.eps);
  i = 0u;
  loop {
    if (i >= params.head_dim) { break; }
    (*values)[base + i] = (*values)[base + i] * inv * (*weight)[i];
    i += 1u;
  }
}

fn apply_rope(values: ptr<storage, array<f32>, read_write>, head: u32) {
  let half = params.head_dim / 2u;
  let base = head * params.head_dim;
  var i = 0u;
  loop {
    if (i >= half) { break; }
    let a_idx = base + i;
    let b_idx = base + i + half;
    let a = (*values)[a_idx];
    let b = (*values)[b_idx];
    let table_idx = params.position * params.head_dim + i;
    let c = cos_table[table_idx];
    let s = sin_table[table_idx];
    (*values)[a_idx] = a * c - b * s;
    (*values)[b_idx] = b * c + a * s;
    i += 1u;
  }
}

@compute @workgroup_size(1)
fn qwen_qkv_norm_rope_append_kv(@builtin(global_invocation_id) gid: vec3<u32>) {
  let head = gid.x;

  if (head < params.num_q_heads) {
    rms_for_head(&q, head, &q_norm_weight);
    apply_rope(&q, head);
  }

  if (head < params.num_kv_heads) {
    rms_for_head(&k, head, &k_norm_weight);
    apply_rope(&k, head);

    var i = 0u;
    let src_base = head * params.head_dim;
    let dst_base = params.position * params.seq_stride + head * params.head_dim;
    loop {
      if (i >= params.head_dim) { break; }
      k_cache[dst_base + i] = k[src_base + i];
      v_cache[dst_base + i] = v[src_base + i];
      i += 1u;
    }
  }
}
`;

export const QWEN_QKV_NORM_ROPE_PAIR_WGSL = /* wgsl */ `
struct Params {
  tokens: u32,
  q_head_count: u32,
  k_head_count: u32,
  head_dim: u32,
  q_hidden: u32,
  k_hidden: u32,
  eps: f32,
  rope_theta: f32,
  q_norm_enabled: u32,
  k_norm_enabled: u32,
  rope_enabled: u32,
  _pad: u32,
};

@group(0) @binding(0) var<storage, read> q_projected: array<f32>;
@group(0) @binding(1) var<storage, read> k_projected: array<f32>;
@group(0) @binding(2) var<storage, read> q_norm_weight: array<f32>;
@group(0) @binding(3) var<storage, read> k_norm_weight: array<f32>;
@group(0) @binding(4) var<storage, read> positions: array<u32>;
@group(0) @binding(5) var<storage, read_write> q_output: array<f32>;
@group(0) @binding(6) var<storage, read_write> k_output: array<f32>;
@group(0) @binding(7) var<uniform> params: Params;

fn normalized_q(token: u32, head: u32, dim: u32) -> f32 {
  let head_start = token * params.q_hidden + head * params.head_dim;
  let offset = head_start + dim;
  var value = q_projected[offset];
  if (params.q_norm_enabled != 0u) {
    var mean_square = 0.0;
    for (var head_dim_index = 0u; head_dim_index < params.head_dim; head_dim_index = head_dim_index + 1u) {
      let sample = q_projected[head_start + head_dim_index];
      mean_square = mean_square + sample * sample;
    }
    mean_square = mean_square / f32(params.head_dim);
    value = value * inverseSqrt(mean_square + params.eps) * q_norm_weight[dim];
  }
  return value;
}

fn normalized_k(token: u32, head: u32, dim: u32) -> f32 {
  let head_start = token * params.k_hidden + head * params.head_dim;
  let offset = head_start + dim;
  var value = k_projected[offset];
  if (params.k_norm_enabled != 0u) {
    var mean_square = 0.0;
    for (var head_dim_index = 0u; head_dim_index < params.head_dim; head_dim_index = head_dim_index + 1u) {
      let sample = k_projected[head_start + head_dim_index];
      mean_square = mean_square + sample * sample;
    }
    mean_square = mean_square / f32(params.head_dim);
    value = value * inverseSqrt(mean_square + params.eps) * k_norm_weight[dim];
  }
  return value;
}

fn apply_rope(value: f32, pair_value: f32, token: u32, dim: u32) -> f32 {
  if (params.rope_enabled == 0u) {
    return value;
  }
  let half_dim = params.head_dim / 2u;
  let rotary_dim = dim % half_dim;
  let frequency = pow(params.rope_theta, -f32(rotary_dim) / f32(half_dim));
  let angle = f32(positions[token]) * frequency;
  let c = cos(angle);
  let s = sin(angle);
  if (dim < half_dim) {
    return value * c - pair_value * s;
  }
  return pair_value * s + value * c;
}

@compute @workgroup_size(16, 16)
fn qwen_qkv_norm_rope_pair(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let col = global_id.x;
  let token = global_id.y;
  if (token >= params.tokens) {
    return;
  }

  if (col < params.q_hidden) {
    let head = col / params.head_dim;
    let dim = col % params.head_dim;
    var pair_dim = dim + params.head_dim / 2u;
    if (dim >= params.head_dim / 2u) {
      pair_dim = dim - params.head_dim / 2u;
    }
    let value = normalized_q(token, head, dim);
    let pair_value = normalized_q(token, head, pair_dim);
    q_output[token * params.q_hidden + col] = apply_rope(value, pair_value, token, dim);
  }

  if (col < params.k_hidden) {
    let head = col / params.head_dim;
    let dim = col % params.head_dim;
    var pair_dim = dim + params.head_dim / 2u;
    if (dim >= params.head_dim / 2u) {
      pair_dim = dim - params.head_dim / 2u;
    }
    let value = normalized_k(token, head, dim);
    let pair_value = normalized_k(token, head, pair_dim);
    k_output[token * params.k_hidden + col] = apply_rope(value, pair_value, token, dim);
  }
}
`;
