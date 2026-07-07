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

.. automodule:: lattice_contract.schema
   :members:

.. automodule:: lattice_contract.dialect
   :members:

MLIR builder
------------

The builder emits textual lattice MLIR from the annotated schema. It is intended
for exporters and tests; semantic verification remains the job of
``lattice-opt`` and the MLIR dialect verifier.

.. automodule:: lattice_contract.mlir
   :members:

.. automodule:: lattice_contract.mlir.builder
   :members:
