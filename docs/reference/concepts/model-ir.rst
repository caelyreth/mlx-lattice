Sparse model IR
===============

The current JSON artifact graph is a legacy deployment bridge. It remains
loadable for mlx-lattice-authored artifacts, but it is no longer the direction
for future cross-framework model exchange. New IR work should target the
MLIR-based lattice dialect direction described in
``references/mlir_lattice_dialect_direction.md``.

Current artifact layout
-----------------------

The legacy artifact directory is:

.. code-block:: text

   model.lattice/
     manifest.json
     weights.safetensors

``manifest.json`` contains a small ordered graph for the legacy MLX artifact
runner. ``weights.safetensors`` contains dense and packed quantized tensor
payloads keyed by manifest parameter names.

This graph format should be treated as an implementation bridge, not as the
future semantic center. In the MLIR direction, ``manifest.json`` becomes shallow
package metadata and graph semantics move to ``graph.mlir`` or ``graph.mlirbc``.

Legacy manifest scope
---------------------

The supported legacy graph is intentionally narrow:

* sparse convolution and transpose-convolution semantic ops;
* submanifold convolution;
* sparse addition;
* feature transforms such as linear, normalization, and activation;
* local and global pooling semantic ops;
* packed quantized weights carried through the same convolution/linear ops as
  dense weights.

The removed generic route
-------------------------

Earlier development builds registered every public ``mlx_lattice.ops`` function
as ``ops.<function_name>`` inside the JSON artifact runtime. That route has been
removed. Persisted artifacts should not be Python API call traces, and future
MLIR dialect work should not inherit the broad ``ops.*`` namespace.

Use explicit semantic names instead:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Legacy op
     - Meaning
   * - ``sparse.conv3d``
     - Sparse convolution; optional target support is represented as a value
       attribute.
   * - ``sparse.subm_conv3d``
     - Submanifold convolution preserving coordinate support.
   * - ``sparse.conv_transpose3d``
     - Sparse transpose convolution.
   * - ``sparse.generative_conv_transpose3d``
     - Coordinate-generating sparse transpose convolution.
   * - ``sparse.add``
     - Coordinate-aligned sparse addition.
   * - ``feature.linear``
     - Dense or packed-quantized feature projection.
   * - ``feature.*``
     - Sparse feature-only transforms preserving coordinate identity.
   * - ``pool.*``
     - Local and global pooling semantics.

Quantization model
------------------

Quantized graph op variants such as ``feature.quantized_linear`` and
``sparse.quantized_conv3d`` are no longer part of the semantic graph surface.
Quantization is represented by the parameter payload:

.. code-block:: text

   feature.linear(input, weight)
   sparse.conv3d(input, weight)

where ``weight`` may be dense or a packed ``QuantizedWeight`` payload stored in
``weights.safetensors``.

Runtime model
-------------

``load_lattice_model()`` validates runtime compatibility, graph wiring,
operation ports, output value types, and referenced weights before execution.
It reconstructs runtime ``SparseTensor`` objects and dispatches through the
approved MLX implementation bindings.

The runtime may rebuild coordinate managers, relations, CSR views,
implicit-GEMM maps, and TensorOps execution views internally. Those structures
are not portable graph semantics.

Sparse value ABI
----------------

Future artifact import/export work should decompose sparse values through the
explicit sparse component ABI:

.. code-block:: python

   components = x.export_components()
   x = SparseTensor.from_components(components)

The ABI contains coordinates, features, active row count, stride, and optional
batch row counts. It deliberately excludes ``CoordinateManager`` and
``CoordinateMapKey`` identity.

Forward direction
-----------------

The intended future package shape is:

.. code-block:: text

   model.lattice/
     graph.mlir
     weights.safetensors
     manifest.json

The MLIR lattice dialect should own sparse value types, operation semantics,
weight layout contracts, verifier rules, and cross-framework portability.
The JSON graph should not grow into a parallel IR.
