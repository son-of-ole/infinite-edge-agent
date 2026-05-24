#!/usr/bin/env node
console.log(`Fused layer parity audit scaffold

Wire this script to your existing v9 resident reference runner and the v10 fused runner.

Required comparisons per enabled stage:
- RMSNorm output
- Q projection
- K projection
- V projection
- Q/K norm output
- RoPE Q/K output
- KV cache row append
- attention context
- O projection output
- residual + post-attention RMSNorm
- SwiGLU intermediate
- MLP output
- final layer hidden
- greedy next token

Exit non-zero if any stage exceeds threshold or changes the greedy token unexpectedly.
`);
