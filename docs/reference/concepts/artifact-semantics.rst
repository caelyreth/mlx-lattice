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

Coordinate and weight ABI
-------------------------

Sparse artifact inputs use ``batch_x_y_z`` coordinates and row-channel feature
storage. Coordinate results must match exactly across runtimes. Feature results
are checked with absolute and relative tolerances because CUDA and Metal use
different floating-point kernels and reduction orders.

Weights are bound through ``lattice.weight``:

* convolution weights use ``conv3d_o_zyx_i``;
* linear weights use ``linear_o_i``;
* bias/channel vectors use ``bias_c`` or ``channel_c``;
* int4/int8 weights must declare packed ``lattice.packing`` metadata.

Quantized conformance fixtures compare against artifact-packed and dequantized
weights, not the pre-quantized training weights.

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
