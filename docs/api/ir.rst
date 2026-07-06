IR API
======

``lattice_contract`` contains the backend-neutral dataclasses and semantic
operation contracts used by the legacy JSON artifact bridge. It does not import
MLX, Torch, native kernels, or sparse runtime objects.

For the conceptual model, read :doc:`../reference/concepts/model-ir`.

Manifest model
--------------

.. automodule:: lattice_contract.manifest
   :members:

Operation registry
------------------

The registry exposes explicit semantic operation contracts. It no longer mirrors
the full ``mlx_lattice.ops`` Python surface, and new cross-framework graph work
should target the MLIR lattice dialect rather than adding broad JSON operation
coverage.

.. automodule:: lattice_contract.ops
   :members:
