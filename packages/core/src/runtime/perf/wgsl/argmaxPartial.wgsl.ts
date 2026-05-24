export const argmaxPartialWgsl = /* wgsl */ `
struct Params {
  row_count: u32,
  values_per_group: u32,
  suppressed_count: u32,
  _pad: u32,
};

@group(0) @binding(0) var<storage, read> logits: array<f32>;
@group(0) @binding(1) var<storage, read> suppressed: array<u32>;
@group(0) @binding(2) var<storage, read_write> partial_pairs: array<vec2<f32>>;
@group(0) @binding(3) var<uniform> params: Params;

fn is_suppressed(row: u32) -> bool {
  for (var index: u32 = 0u; index < params.suppressed_count; index = index + 1u) {
    if (suppressed[index] == row) { return true; }
  }
  return false;
}

@compute @workgroup_size(1)
fn argmax_partial(@builtin(workgroup_id) wid: vec3<u32>) {
  let group = wid.x;
  let start = group * params.values_per_group;
  let end = min(params.row_count, start + params.values_per_group);
  var best_value = -3.402823e38;
  var best_row = -1.0;
  for (var row = start; row < end; row = row + 1u) {
    if (!is_suppressed(row)) {
      let value = logits[row];
      if (value > best_value) {
        best_value = value;
        best_row = f32(row);
      }
    }
  }
  partial_pairs[group] = vec2<f32>(best_value, best_row);
}
`;
