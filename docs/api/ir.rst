IR API
======

``lattice_contract`` is now MLIR-first. Its public surface contains the
annotation-backed lattice dialect schema plus a generated textual MLIR builder.
It does not import MLX, Torch, native kernels, or sparse runtime objects.

For the conceptual model, read :doc:`../reference/concepts/model-ir`.

Annotated dialect schema
------------------------

The dialect schema is executable metadata: decorators register types,
attributes, operations, operands, results, and operation attributes once. The
builder consumes this schema directly, so exporter code can call the generated
operation surface instead of carrying a parallel op-name registry.

The current dialect surface is intentionally small but semantic: sparse ABI
construction/decomposition, symbolic weight binding, the sparse convolution
family (forward, submanifold, target, transpose, and generative transpose),
local/global sparse pooling, point/voxel conversion, dense feature projection,
dense feature activation, dense feature normalization, sparse feature
replacement, and coordinate-aligned sparse binary algebra. Backend execution
routes such as TensorOps, CSR views, point-to-voxel cache maps, or rulebook
layouts are not encoded in the dialect.

.. automodule:: lattice_contract.schema
   :members:

.. automodule:: lattice_contract.dialect
   :members:

MLIR builder
------------

The builder emits textual lattice MLIR from the annotated schema. It is intended
for exporters and tests; semantic verification remains the job of
``lattice-opt`` and the MLIR dialect verifier.

Generated modules include the required artifact ABI metadata:
``lattice.ir_version = 2`` and
``lattice.weight_file = "weights.safetensors"``. These attributes are part of
the stable exchange contract, not optional documentation. The corresponding
Python constants are exported as ``CURRENT_DIALECT_VERSION``,
``ARTIFACT_GRAPH_FILE``, and ``ARTIFACT_WEIGHT_FILE``.

.. automodule:: lattice_contract.artifact
   :members:

.. automodule:: lattice_contract.mlir.builder
   :members:
