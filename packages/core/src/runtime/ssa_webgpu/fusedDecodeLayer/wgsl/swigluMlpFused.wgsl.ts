export const SWIGLU_MLP_FUSED_WGSL = /* wgsl */ `
struct Params {
  hidden_size: u32,
  intermediate_size: u32,
};

@group(0) @binding(0) var<storage, read> hidden: array<f32>;
@group(0) @binding(1) var<storage, read> gate_weight: array<f32>;
@group(0) @binding(2) var<storage, read> up_weight: array<f32>;
@group(0) @binding(3) var<storage, read> down_weight: array<f32>;
@group(0) @binding(4) var<storage, read_write> intermediate: array<f32>;
@group(0) @binding(5) var<storage, read_write> down_out: array<f32>;
@group(0) @binding(6) var<uniform> params: Params;

fn silu(x: f32) -> f32 {
  return x / (1.0 + exp(-x));
}

@compute @workgroup_size(128)
fn swiglu_gate_up(@builtin(global_invocation_id) gid: vec3<u32>) {
  let row = gid.x;
  if (row >= params.intermediate_size) { return; }
  var gate = 0.0;
  var up = 0.0;
  var col = 0u;
  loop {
    if (col >= params.hidden_size) { break; }
    let x = hidden[col];
    gate += gate_weight[row * params.hidden_size + col] * x;
    up += up_weight[row * params.hidden_size + col] * x;
    col += 1u;
  }
  intermediate[row] = silu(gate) * up;
}

@compute @workgroup_size(128)
fn swiglu_down(@builtin(global_invocation_id) gid: vec3<u32>) {
  let row = gid.x;
  if (row >= params.hidden_size) { return; }
  var sum = 0.0;
  var col = 0u;
  loop {
    if (col >= params.intermediate_size) { break; }
    sum += down_weight[row * params.intermediate_size + col] * intermediate[col];
    col += 1u;
  }
  down_out[row] = sum;
}
`;
