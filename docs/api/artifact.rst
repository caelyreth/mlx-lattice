Artifact API
============

``mlx_lattice.artifact`` handles MLIR-first exchange bundles. A bundle contains
textual ``graph.mlir`` and ``weights.safetensors``. It does not execute the
graph.

Execution import is intentionally a separate future lowering step from lattice
MLIR into an MLX runtime plan.

Bundle IO
---------

.. automodule:: mlx_lattice.artifact
   :members:

.. automodule:: mlx_lattice.artifact.io
   :members:
