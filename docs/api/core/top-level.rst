Top-level package
=================

``mlx_lattice`` re-exports the most common entry points for convenience:
``SparseTensor``, ``SparseTensorComponents``, quantized weight helpers,
``backend_info``, and the ``core``, ``ops``, and ``nn`` namespaces.

The canonical class and function documentation lives on the feature-specific
pages in this section. This page uses links instead of a full ``automodule``
member listing so objects such as ``SparseTensor`` are not rendered both here
and on their canonical page.

.. currentmodule:: mlx_lattice

Canonical pages
---------------

Use these pages for the actual object documentation:

* :doc:`sparse-tensor` for ``SparseTensor`` and ``SparseTensorComponents``.
* :doc:`coordinate-management` for runtime ``CoordinateManager`` and
  ``CoordinateMapKey`` internals under ``mlx_lattice.core``.
* :doc:`quantized-weights` for ``QuantizedWeight``, ``quantize_weight``, and
  ``dequantize_weight``.
* :doc:`../native` for ``backend_info``.
