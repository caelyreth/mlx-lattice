Numerical Conformance
=====================

Lattice separates semantic correctness from floating-point reproducibility.
Coordinates, relation edges, kernel positions, and artifact tensor layouts are
exact contracts. Floating-point reductions are evaluated against a common
binary64 reference because valid CUDA and Metal executions can accumulate terms
in different orders.

Reference evaluator
-------------------

``lattice_contract.submanifold_conv3d_f32_to_f64`` is a scalar, dependency-free
reference evaluator for conformance tests. It rounds feature, weight, and bias
leaves to binary32, multiplies them in binary64, and uses ``math.fsum`` for the
final sum. It also owns the canonical ``(batch, x, y, z)`` coordinates and
z-fastest ``(x, y, z)`` kernel-row interpretation.

The evaluator is deliberately unsuitable for model execution. Production
backends use their optimized relation and reduction kernels, then tests compare
their result to the evaluator with dtype-appropriate tolerances.

Floating-point policy
---------------------

For unquantized FP32 pointwise projections, the Metal backend uses a native
FP32 projection rather than MLX's accelerated matrix path. The latter can choose
reduced-precision hardware, which makes the result depend on matrix size. The
native path uses TensorOps for supported 32-channel tiles and a scalar FP32
Metal kernel otherwise. They are both checked against the same binary64 oracle;
their reduction order need not be bitwise identical. Lattice does not expose a
threshold or compatibility option: identical FP32 inputs receive the same
accuracy policy at every size.

Sparse relation kernels and CUDA dataflows may still differ by normal FP32
reduction order. They must preserve coordinate support and stay within the
documented oracle tolerance; bitwise equality across devices is not a contract.

Quantized weights are a separate numerical surface. Their affine packed values
are exact at storage boundaries, while their runtime output is compared to the
corresponding dequantized execution with the quantization tolerance declared by
the artifact test.

Migration and validation
------------------------

Historical TorchSparse checkpoints are not a numerical reference. Convert them
once with explicit kernel metadata, inspect the recorded permutation manifest,
and validate the converted block against its known input/output fixtures. The
runtime never infers legacy row order or silently adapts a checkpoint.

The permanent checks cover the contract evaluator, canonical non-cubic CUDA map
builders, Metal convolution, and CUDA-to-MLX artifact replay. This keeps a
backend optimization from becoming an undocumented semantic change.
