export const RESIDUAL_RMSNORM_WGSL = /* wgsl */ `
struct Params {
  hidden_size: u32,
  eps: f32,
};

@group(0) @binding(0) var<storage, read> residual: array<f32>;
@group(0) @binding(1) var<storage, read> update: array<f32>;
@group(0) @binding(2) var<storage, read> norm_weight: array<f32>;
@group(0) @binding(3) var<storage, read_write> summed: array<f32>;
@group(0) @binding(4) var<storage, read_write> normed: array<f32>;
@group(0) @binding(5) var<storage, read_write> scratch: array<f32>;
@group(0) @binding(6) var<uniform> params: Params;

@compute @workgroup_size(256)
fn residual_sum_square(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= params.hidden_size) { return; }
  let x = residual[i] + update[i];
  summed[i] = x;
  scratch[i] = x * x;
}

@compute @workgroup_size(1)
fn residual_rmsnorm_apply(@builtin(global_invocation_id) gid: vec3<u32>) {
  var ss = 0.0;
  var i = 0u;
  loop {
    if (i >= params.hidden_size) { break; }
    ss += scratch[i];
    i += 1u;
  }
  let inv = inverseSqrt(ss / f32(params.hidden_size) + params.eps);
  i = 0u;
  loop {
    if (i >= params.hidden_size) { break; }
    normed[i] = summed[i] * inv * norm_weight[i];
    i += 1u;
  }
}
`;

export const RESIDUAL_RMSNORM_ONE_TOKEN_WGSL = /* wgsl */ `
struct Params {
  hidden_size: u32,
  eps: f32,
  _pad0: u32,
  _pad1: u32,
};

@group(0) @binding(0) var<storage, read> residual: array<f32>;
@group(0) @binding(1) var<storage, read> update: array<f32>;
@group(0) @binding(2) var<storage, read> norm_weight: array<f32>;
@group(0) @binding(3) var<storage, read_write> summed: array<f32>;
@group(0) @binding(4) var<storage, read_write> normed: array<f32>;
@group(0) @binding(5) var<uniform> params: Params;

@compute @workgroup_size(1)
fn residual_rmsnorm_one_token(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x > 0u) {
    return;
  }

  var ss = 0.0;
  var i = 0u;
  loop {
    if (i >= params.hidden_size) { break; }
    let value = residual[i] + update[i];
    summed[i] = value;
    ss = ss + value * value;
    i = i + 1u;
  }

  let inv = inverseSqrt(ss / f32(params.hidden_size) + params.eps);
  i = 0u;
  loop {
    if (i >= params.hidden_size) { break; }
    normed[i] = summed[i] * inv * norm_weight[i];
    i = i + 1u;
  }
}
`;
