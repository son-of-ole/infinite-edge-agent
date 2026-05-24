export const PACKED_QKV_PROJECTION_WGSL = /* wgsl */ `
struct Params {
  hidden_size: u32,
  q_rows: u32,
  k_rows: u32,
  v_rows: u32,
  total_rows: u32,
  tokens: u32,
  _pad0: u32,
  _pad1: u32,
};

@group(0) @binding(0) var<storage, read> hidden: array<f32>;
@group(0) @binding(1) var<storage, read> q_weight: array<f32>;
@group(0) @binding(2) var<storage, read> k_weight: array<f32>;
@group(0) @binding(3) var<storage, read> v_weight: array<f32>;
@group(0) @binding(4) var<storage, read_write> q_out: array<f32>;
@group(0) @binding(5) var<storage, read_write> k_out: array<f32>;
@group(0) @binding(6) var<storage, read_write> v_out: array<f32>;
@group(0) @binding(7) var<uniform> params: Params;

fn dot_row(weight: ptr<storage, array<f32>, read>, token: u32, row: u32) -> f32 {
  var sum = 0.0;
  var col = 0u;
  loop {
    if (col >= params.hidden_size) { break; }
    sum += (*weight)[row * params.hidden_size + col] * hidden[token * params.hidden_size + col];
    col += 1u;
  }
  return sum;
}

@compute @workgroup_size(128)
fn packed_qkv_projection(@builtin(global_invocation_id) gid: vec3<u32>) {
  let row = gid.x;
  let token = gid.y;
  if (row >= params.total_rows || token >= params.tokens) { return; }

  if (row < params.q_rows) {
    q_out[token * params.q_rows + row] = dot_row(&q_weight, token, row);
    return;
  }

  let k_local = row - params.q_rows;
  if (k_local < params.k_rows) {
    k_out[token * params.k_rows + k_local] = dot_row(&k_weight, token, k_local);
    return;
  }

  let v_local = row - params.q_rows - params.k_rows;
  if (v_local < params.v_rows) {
    v_out[token * params.v_rows + v_local] = dot_row(&v_weight, token, v_local);
  }
}
`;
