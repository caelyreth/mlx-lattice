Relations
=========

Relation objects connect coordinate semantics to backend execution. Start with
:doc:`../../reference/concepts/coordinates-relations` for the conceptual model,
then use this page for exact class members and builders. Convolution and
pooling route details are documented in
:doc:`../../reference/backend/convolution` and
:doc:`../../reference/backend/pooling`.

Stable application code should normally depend on ``KernelSpec``,
``KernelRelation``, ``NeighborRelation``, and the functional relation builders.
Low-level CSR views, implicit-GEMM views, sorted implicit-GEMM views, and
relation contracts are exposed for inspection and debugging, but their exact
layout is provisional backend metadata.

Relation specifications
-----------------------

.. automodule:: mlx_lattice.core.relations.specs
   :members:

Relation views
--------------

The view classes below are advanced/provisional. They are useful when
diagnosing relation construction or backend route selection; they should not be
used as a long-term application storage format.

.. automodule:: mlx_lattice.core.relations.views
   :members:

Relation builders
-----------------

.. automodule:: mlx_lattice.core.coords.builders
   :members:
