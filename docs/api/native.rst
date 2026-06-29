Diagnostics API
===============

The stable diagnostics API is :func:`mlx_lattice.backend_info`. It reports
compiled extension metadata, version information, and backend capability
strings. Use it to verify that the native module imported successfully or to
attach environment details to bug reports.

``backend_info()`` does not select routes. Public operations still dispatch
from the active MLX device, input dtype, relation metadata, shape predicates,
and available backend kernels.

The underscored ``mlx_lattice._native`` module is an implementation module.
Do not import it from application code.

Related pages
-------------

* Backend path selection: :doc:`../reference/backend/path-selection`
* Convolution routes: :doc:`../reference/backend/convolution`
* Quantized routes: :doc:`../reference/backend/quantization`

.. autofunction:: mlx_lattice.backend_info
