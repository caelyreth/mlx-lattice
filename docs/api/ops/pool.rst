Pooling operations
==================

Pooling operations reduce sparse feature rows through a kernel relation or
through batch metadata.

Local pooling computes:

.. math::

   y_{o,c} = \operatorname{reduce}_{e:o_e=o} x_{i_e,c}.

``sum`` and ``avg`` accept empty output rows as zero-valued reductions. ``max``
requires at least one contributing row for every output row. Global pooling
uses ``batch_counts`` metadata from the input sparse tensor and returns a dense
``(B, C)`` MLX array.

``pool_transpose3d`` reverses sparse resolution by averaging each coarse row
over a transposed kernel relation. Without a target it generates fine support;
with a target tensor it preserves that tensor's coordinates and emits zeros for
target rows with no contributors.

Related pages
-------------

* Backend reduction routes: :doc:`../../reference/backend/pooling`
* Relation model: :doc:`../../reference/concepts/coordinates-relations`
* Batch metadata: :doc:`../../reference/concepts/sparse-tensor`
* Module wrappers: :doc:`../nn/pooling`

.. automodule:: mlx_lattice.ops.pool
   :members:
