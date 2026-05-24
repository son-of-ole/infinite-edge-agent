export const QWEN_ONE_TOKEN_ATTENTION_WGSL = /* wgsl */ `
struct Params {
  num_q_heads: u32,
  num_kv_heads: u32,
  head_dim: u32,
  sequence_length: u32,
  seq_stride: u32,
  scale: f32,
};

@group(0) @binding(0) var<storage, read> q: array<f32>;
@group(0) @binding(1) var<storage, read> k_cache: array<f32>;
@group(0) @binding(2) var<storage, read> v_cache: array<f32>;
@group(0) @binding(3) var<storage, read_write> attention_out: array<f32>;
@group(0) @binding(4) var<uniform> params: Params;

const WORKGROUP_SIZE: u32 = 128u;
const MAX_SEQUENCE_LENGTH: u32 = 1024u;

var<workgroup> scores: array<f32, 1024>;
var<workgroup> reductions: array<f32, 128>;

fn qk_dot(q_head: u32, kv_head: u32, pos: u32) -> f32 {
  var sum = 0.0;
  var i = 0u;
  let q_base = q_head * params.head_dim;
  let k_base = pos * params.seq_stride + kv_head * params.head_dim;
  loop {
    if (i >= params.head_dim) { break; }
    sum += q[q_base + i] * k_cache[k_base + i];
    i += 1u;
  }
  return sum * params.scale;
}

@compute @workgroup_size(128)
fn qwen_one_token_attention(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
) {
  let q_head = wid.x;
  let lane = lid.x;
  if (q_head >= params.num_q_heads) { return; }
  if (params.sequence_length > MAX_SEQUENCE_LENGTH) { return; }

  let repeat = params.num_q_heads / params.num_kv_heads;
  let kv_head = q_head / repeat;

  var local_max = -3.402823e38;
  var pos = lane;
  loop {
    if (pos >= params.sequence_length) { break; }
    let s = qk_dot(q_head, kv_head, pos);
    scores[pos] = s;
    local_max = max(local_max, s);
    pos += WORKGROUP_SIZE;
  }
  reductions[lane] = local_max;
  workgroupBarrier();

  var stride = WORKGROUP_SIZE / 2u;
  loop {
    if (stride == 0u) { break; }
    if (lane < stride) {
      reductions[lane] = max(reductions[lane], reductions[lane + stride]);
    }
    workgroupBarrier();
    stride = stride / 2u;
  }
  let max_score = reductions[0];

  var local_denom = 0.0;
  pos = lane;
  loop {
    if (pos >= params.sequence_length) { break; }
    local_denom += exp(scores[pos] - max_score);
    pos += WORKGROUP_SIZE;
  }
  reductions[lane] = local_denom;
  workgroupBarrier();

  stride = WORKGROUP_SIZE / 2u;
  loop {
    if (stride == 0u) { break; }
    if (lane < stride) {
      reductions[lane] = reductions[lane] + reductions[lane + stride];
    }
    workgroupBarrier();
    stride = stride / 2u;
  }
  let denom = max(reductions[0], 1e-20);

  var dim = lane;
  loop {
    if (dim >= params.head_dim) { break; }
    var acc = 0.0;
    pos = 0u;
    loop {
      if (pos >= params.sequence_length) { break; }
      let weight = exp(scores[pos] - max_score) / denom;
      let v_base = pos * params.seq_stride + kv_head * params.head_dim;
      acc += weight * v_cache[v_base + dim];
      pos += 1u;
    }
    attention_out[q_head * params.head_dim + dim] = acc;
    dim += WORKGROUP_SIZE;
  }
}
`;
