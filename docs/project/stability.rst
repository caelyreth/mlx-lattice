Stability contract
==================

This page defines which surfaces are intended for application use and which
surfaces are exposed for diagnostics or backend maintenance.

Stable public surface
---------------------

The following surfaces are intended to remain source-compatible across normal
minor releases:

* top-level package imports: ``SparseTensor``, ``SparseTensorComponents``,
  ``QuantizedWeight``, ``quantize_weight``, ``dequantize_weight``,
  ``backend_info``, and the ``core``, ``ops``, and ``nn`` namespaces;
* sparse tensor construction and metadata: coordinate rows ordered as
  ``(batch, x, y, z)``, feature rows ordered as ``(N, C)``, sparse stride, and
  coordinate identity via manager/key ownership;
* functional sparse operations in ``mlx_lattice.ops`` except where explicitly
  marked provisional;
* ``mlx_lattice.nn`` module wrappers over convolution, pooling, feature, and
  quantized inference operations;
* packed affine weight quantization through ``QuantizedWeight``,
  ``quantize_weight``, and ``dequantize_weight``;
* MLIR artifact package media: ``graph.mlir`` plus
  ``weights.safetensors``;
* MLX artifact bundle IO and capability discovery:
  ``save_lattice_artifact``, ``load_lattice_artifact``,
  ``validate_lattice_artifact``, and
  ``native_artifact_execution_available``.

Provisional or diagnostic surface
---------------------------------

The following surfaces are available for inspection, tests, or controlled
experiments, but applications should avoid treating their exact storage layout
as stable:

* relation execution views such as CSR views, implicit-GEMM views, and sorted
  implicit-GEMM views;
* coordinate manager/key identity objects under ``mlx_lattice.core``;
* exact MLIR dialect v0 operation/attribute details before the artifact ABI is
  declared stable;
* MLIR artifact runtime plan and lowering internals before the importer is
  stabilized;
* backend route names, Metal kernel names, TensorOps kernel variants, and
  diagnostic reference routes;
* encoded byte streams from entropy helpers;
* underscored modules such as ``mlx_lattice._native`` and
  ``mlx_lattice.ops._relation_exec``.

Coordinate order
----------------

The canonical coordinate row order is:

.. code-block:: text

   (batch, x, y, z)

This order is part of the stable public contract. Batch is the leading column
because batching, global pooling, relation caching, Morton ordering, and native
coordinate hashing all treat batch as the first grouping key. Use conversion
helpers at ingestion boundaries if your source data uses a different order;
do not store internal tensors as ``(x, y, z, batch)``.

Compatibility policy
--------------------

Minor releases should preserve stable public semantics. Performance route
selection may change when the result is semantically identical. Provisional
diagnostic storage may change when backend kernels are refactored.

Breaking changes to stable surfaces should be reserved for major releases and
called out in release notes.
