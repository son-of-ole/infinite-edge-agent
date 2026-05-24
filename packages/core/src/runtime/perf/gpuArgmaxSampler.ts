export const GPU_ARGMAX_PAIR_FLOATS = 2;
export const GPU_ARGMAX_READBACK_BYTES = GPU_ARGMAX_PAIR_FLOATS * Float32Array.BYTES_PER_ELEMENT;

export interface GpuArgmaxSamplerProof {
  strategy: "gpu_argmax_compact_readback";
  readbackRows: 1;
  readbackBytes: number;
}

export function buildGpuArgmaxSamplerProof(): GpuArgmaxSamplerProof {
  return {
    strategy: "gpu_argmax_compact_readback",
    readbackRows: 1,
    readbackBytes: GPU_ARGMAX_READBACK_BYTES,
  };
}
