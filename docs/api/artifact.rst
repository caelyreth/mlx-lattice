Artifact API
============

``mlx_lattice.artifact`` handles MLIR-first exchange bundles. A bundle contains
textual ``graph.mlir`` and ``weights.safetensors``. It does not execute the
graph.

Graph validation uses real MLIR infrastructure. MLIR-enabled native builds use
an in-process parser/verifier registered with the ``lattice`` dialect.
Lightweight builds can validate through the repository ``lattice-opt`` tool.
The Python layer does not parse lattice MLIR text itself.

Executable lowering is available when the MLIR-enabled native extension is
built. The native extension parses and verifies the module, returns a structured
runtime plan, and Python lowers that plan through existing ``mlx_lattice.ops``
routes.

Bundle IO
---------

.. automodule:: mlx_lattice.artifact
   :members:

.. automodule:: mlx_lattice.artifact.io
   :members:

MLIR validation
---------------

.. automodule:: mlx_lattice.artifact.validation
   :members:
