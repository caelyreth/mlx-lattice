Cross-runtime artifact semantics
================================

``mlx-lattice`` treats MLIR artifacts as the stable exchange contract between
training-side runtimes and MLX deployment. The artifact directory contains two
semantic files:

.. code-block:: text

   artifact/
     graph.mlir
     weights.safetensors

``graph.mlir`` owns graph semantics. ``weights.safetensors`` stores tensor
payloads referenced by ``lattice.weight`` operations.

Sparse convolution support
--------------------------

Convolution support policy is explicit in the operation name. It is not inferred
from stride, padding, or historical framework conventions.

.. list-table::
   :header-rows: 1
   :widths: 28 36 36

   * - MLIR op
     - Support policy
     - MLX lowering
   * - ``lattice.conv3d``
     - Forward/support-generating sparse convolution.
     - :func:`mlx_lattice.ops.conv3d`
   * - ``lattice.subm_conv3d``
     - Support-preserving submanifold convolution.
     - :func:`mlx_lattice.ops.subm_conv3d`
   * - ``lattice.target_conv3d``
     - Convolution over an explicit target coordinate set.
     - :func:`mlx_lattice.ops.conv3d` with ``coordinates=target``.
   * - ``lattice.conv_transpose3d``
     - Transposed convolution over derived support.
     - :func:`mlx_lattice.ops.conv_transpose3d`
   * - ``lattice.generative_conv_transpose3d``
     - Generative transposed support expansion.
     - :func:`mlx_lattice.ops.generative_conv_transpose3d`

Original TorchSparse compatibility is handled before artifact creation. In the
covered migration subset, original ``Conv3d(kernel_size > 1, stride = 1)`` maps
to the explicit submanifold operation. New artifacts should emit
``lattice.subm_conv3d`` directly for support-preserving convolution.

Generated and target transpose support
--------------------------------------

``lattice.generative_conv_transpose3d`` enumerates every canonical kernel
position for every active source coordinate. For a source coordinate ``s`` and
kernel position ``k``, generated output support is ``s * stride + k``. The
flattened kernel row is ``(x, y, z)`` with z varying fastest, so producers must
not reuse a historical framework's implicit flattening convention.

Target-conditioned transpose operations instead execute only at the coordinate
set carried by their target sparse value. That target must have a sparse stride
equal to ``source.stride / stride``. The exchange format deliberately models
this target as a graph input or earlier graph value; it never guesses an output
support from a weight tensor.

Coordinate and weight ABI
-------------------------

Sparse artifact inputs use physical ``batch_x_y_z`` coordinates and row-channel
feature storage. The coordinates passed through the ABI are never pre-divided
by sparse stride: an input declared with stride ``(2, 2, 2)`` must carry even
spatial positions. Importers validate that divisibility and convert to a local
logical representation only internally. Coordinate results use the same
physical representation and must match exactly across runtimes. Feature
results are checked with absolute and relative tolerances because CUDA and
Metal use different floating-point kernels and reduction orders. Convolution
artifact ops declare ``accumulation = \"canonical_f32\"``; FP16 storage does not
change the portable accumulation contract.

Each logical sparse input is represented by three consecutive MLIR arguments
tagged ``sparse_coords``, ``sparse_features``, and ``sparse_active``. Their ABI
names share an ``<input>_coords/features/active`` prefix. The MLX program binder
therefore accepts either low-level component arrays or one ``SparseTensor`` per
logical input. Multi-input artifacts can be called positionally as
``program(x, target)`` or by logical name as ``program(x=x, target=target)``.
Multiple graph returns are preserved as an ordered Python tuple; a single return
is unwrapped for convenience.

Weights are bound through ``lattice.weight``:

* convolution weights use ``conv3d_o_xyz_i``;
* linear weights use ``linear_o_i``;
* bias/channel vectors use ``bias_c`` or ``channel_c``;
* int4/int8 weights must declare packed ``lattice.packing`` metadata.

Quantized conformance fixtures compare against artifact-packed and dequantized
weights, not the pre-quantized training weights.

Compatibility and rejection policy
----------------------------------

Artifact loading is intentionally strict. Every MLIR version before IR v2,
legacy JSON manifests, missing schema digests, old ``conv3d_o_zyx_i``
convolution layouts, and undeclared weight files are rejected before execution.
A caller must
perform an explicit one-time migration rather than relying on a runtime
fallback. This keeps a portable artifact self-describing and prevents a legacy
kernel permutation from silently changing a model result.

Coordinate support is exact across runtimes: duplicate sparse coordinate rows
are invalid, and replay first aligns both sparse results in canonical Morton
order before requiring coordinate equality. Physical sparse-row order is not an
exchange contract. Feature values are floating-point results and are compared
with the fixture's declared absolute and relative tolerance, not with
cross-device bitwise equality.

Conformance replay
------------------

CUDA-side fixture archives should include the artifact, exact inputs, expected
outputs, and per-case tolerances. MLX-side replay reports average, median, p95,
p99, and max values for both absolute and relative error:

.. code-block:: bash

   uv run conformance replay fixtures.tar.gz \
     --report report.json

For sparse outputs, coordinate equality is mandatory before feature tolerances
are evaluated.

Application partitioning
------------------------

Export the tensor-compute graph, not an entire codec process. Dataset traversal,
entropy stream orchestration, adaptive Python control flow, and rendering remain
host responsibilities. A Gameleon-style artifact should contain the sparse
encoder/decoder block and its learned weights; the CUDA conformance gate trains
that composed block before exporting it and MLX replays the same graph with exact
coordinate equality.

Fixed-Profile Geometry Replay
-----------------------------

The Level-8 geometry conformance runner covers a fixed, active Gameleon
forward-BPP profile before it is representable as a single MLIR artifact. It
uses generic occupancy downsampling/expansion and Morton ordering from
``mlx-lattice``, plus MLX Core for the dense predictor heads. It includes the
FOG hierarchy and neural BPP estimate, but excludes PLY parsing, sparse input
construction, checkpoint loading, and arithmetic coding from its timed region.

Convert a trusted TorchSparse checkpoint once, then replay the prepared
Level-8 PLY on Metal:

.. code-block:: bash

   # Run from the Torch Lattice CUDA workspace.
   uv run convert-checkpoint checkpoint.pt weights.safetensors \
     --kernel-spec legacy-kernels.json
   uv run conformance geometry-profile \
     --weights weights.safetensors \
     --input level8.ply \
     --device metal

The CUDA-side conversion is strict: it canonicalizes each explicitly declared
historical ``.kernel`` state key and records its actual source layout and row
permutation in a JSON manifest. Historical TorchSparse uses x-fastest rows for
odd-volume kernels and z-fastest rows for even-volume kernels. MLX replay never
performs that conversion at load time.
