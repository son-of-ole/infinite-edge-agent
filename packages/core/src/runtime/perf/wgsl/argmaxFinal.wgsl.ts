export const argmaxFinalWgsl = /* wgsl */ `
struct Params {
  partial_count: u32,
  _pad0: u32,
  _pad1: u32,
  _pad2: u32,
};

@group(0) @binding(0) var<storage, read> partial_pairs: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read_write> final_pair: array<vec2<f32>>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(1)
fn argmax_final(@builtin(global_invocation_id) gid: vec3<u32>) {
  var best_value = -3.402823e38;
  var best_row = -1.0;
  for (var index: u32 = 0u; index < params.partial_count; index = index + 1u) {
    let pair = partial_pairs[index];
    if (pair.x > best_value) {
      best_value = pair.x;
      best_row = pair.y;
    }
  }
  final_pair[0] = vec2<f32>(best_value, best_row);
}
`;
