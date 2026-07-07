Feature operations
==================

Sparse feature operations preserve sparse coordinate identity. They replace
only the ``feats`` matrix and keep the coordinate manager, coordinate key,
stride, active rows, and batch metadata unchanged.

Use these functions when the operation is row-local or channel-local:
activations, normalization, dropout, and linear projections. A quantized linear
projection is selected by passing :class:`mlx_lattice.core.QuantizedWeight`.

The MLIR artifact ABI separates dense feature math from sparse identity.
``linear_features``, ``activation``, ``batch_norm_features``,
``layer_norm_features``, and ``rms_norm_features`` are dense rank-2 feature
tensor operations. Sparse wrappers such as ``relu``, ``linear``,
``batch_norm``, ``layer_norm``, and ``rms_norm`` apply those dense ops to
``x.feats`` and return ``x.replace(feats=...)``. Serialized MLIR graphs must
represent the same pattern with ``lattice.sparse.decompose`` and
``lattice.sparse.with_features``.

Related pages
-------------

* Sparse tensor identity: :doc:`../../reference/concepts/sparse-tensor`
* Feature module wrappers: :doc:`../nn/feature`
* Quantized weight storage: :doc:`../core/quantized-weights`
* Quantized route details: :doc:`../../reference/backend/quantization`

.. automodule:: mlx_lattice.ops.feature
   :members:
