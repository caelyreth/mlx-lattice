Pooling routes
==============

Pooling is relation reduction. Local pooling builds a ``KernelRelation`` and
reduces each output row's input neighbors. Global pooling ignores kernel
geometry and reduces rows by batch metadata.

Local pooling contract
----------------------

For a relation edge set :math:`\mathcal{E}`, local pooling computes:

.. math::

   Y_{o,c}^{sum} = \sum_{e_o=o} X_{e_i,c}

.. math::

   Y_{o,c}^{avg} =
   \frac{1}{|\{e : e_o=o\}|}
   \sum_{e_o=o} X_{e_i,c}

.. math::

   Y_{o,c}^{max} = \max_{e_o=o} X_{e_i,c}.

The denominator in average pooling is the sparse neighbor count for the output
row. It is not the dense kernel volume unless every dense kernel position is
active.

Pooling transpose
-----------------

Pooling transpose applies the inverse coordinate relation and averages the
coarse rows that reach each fine output row. For source coordinate :math:`s`
and kernel offset :math:`k`, generated coordinates satisfy

.. math::

   t = s \odot \operatorname{stride}
       + k \odot \operatorname{dilation} - \operatorname{padding}.

The generated route deduplicates this support. The explicit-target route keeps
the target's row order and support exactly, using a cached native implicit-GEMM
``(N_{out}, K)`` relation view. Both routes divide by the number of valid source
contributors; an unmatched target row receives zero. The target sparse stride
must equal the source stride divided component-wise by the operation stride.

This is the MLX equivalent of MinkowskiEngine
``MinkowskiPoolingTranspose``. ``expand_coordinates=True`` maps to the generated
route; passing ``coordinates=target`` maps to the explicit-target route.

Trilinear upsampling
--------------------

``trilinear_upsample3d`` derives kernel extent and padding from sparse stride.
For stride two, its separable one-dimensional weights are
``[0.5, 1.0, 0.5]``. It divides the weighted feature sum by the sum of weights
present on sparse support, so missing neighbors do not attenuate boundary rows.
It can generate fine support or consume exact target support.

Backend routes
--------------

.. list-table::
   :header-rows: 1
   :widths: 22 34 44

   * - Route
     - Predicate
     - Implementation
   * - CPU local pooling
     - Valid ``float32`` features and kernel relation
     - CPU relation reduction over edge arrays.
   * - Metal local pooling
     - Valid ``float32`` or inference-only ``float16`` features, ``int32``
       coordinates, Metal device
     - ``sparse_pool_relation_f32_i32`` or ``sparse_pool_relation_f16_i32``
       over output rows and channels. The FP16 kernel accumulates in FP32 and
       converts only its final result to FP16.
   * - Local pooling VJP
     - Differentiating through local pooling
     - Float32 sum/avg use direct gradient scatter; float32 max uses max-tie
       policy. FP16 local pooling rejects VJP/JVP; use float32 for training.
   * - Local pooling JVP
     - Forward-mode transform
     - ``sparse_pool_relation_jvp_f32_i32``.
   * - Generated pooling transpose
     - No target support supplied
     - Native transposed kernel relation and average reduction.
   * - Target pooling transpose
     - Explicit fine support supplied
     - Cached native target-transposed implicit-GEMM view and averaged gathers.
   * - Trilinear upsampling
     - Generated or explicit fine support
     - Cached target-transposed view and normalized separable linear weights.
   * - Global pooling
     - ``batch_counts`` metadata present
     - MLX dense reductions or scatter reductions over batch ids.

Input-exclusive gradient path
-----------------------------

The pooling backend carries an ``input_exclusive`` flag derived from kernel
geometry. When each input row contributes to at most one output row, the
gradient path can use an exclusive input-gradient kernel. Otherwise it uses the
sum/avg or max relation-gradient route.

Validation boundaries
---------------------

Local pooling currently validates:

* CPU feature dtype is ``float32``; Metal also accepts inference-only
  ``float16``;
* Metal coordinates are ``int32``;
* mode is ``sum``, ``max``, or ``avg``;
* relation metadata includes output coordinates, counts, kernel count, and
  output capacity.

Global pooling validates:

* ``batch_counts`` is present;
* empty batches are allowed for sum and average;
* empty batches are rejected for max pooling.

Global pooling formulas
-----------------------

For batch :math:`b` with row set :math:`R_b`:

.. math::

   G^{sum}_{b,c} = \sum_{i \in R_b} X_{i,c},
   \qquad
   G^{avg}_{b,c} =
   \frac{G^{sum}_{b,c}}{\max(|R_b|, 1)}.

``global_max_pool`` requires :math:`|R_b| > 0` for every batch because there is
no finite feature value that represents the maximum of an empty sparse set.

Related API and references
--------------------------

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * - Need
     - Page
     - Notes
   * - Functional pooling signatures
     - :doc:`../../api/ops/pool`
     - Local and global pooling functions.
   * - Module wrappers
     - :doc:`../../api/nn/pooling`
     - ``Pool3d`` variants and global pooling modules.
   * - Relation model
     - :doc:`../concepts/coordinates-relations`
     - Output CSR view used by relation reduction.
   * - Batch metadata
     - :doc:`../concepts/sparse-tensor`
     - ``batch_counts`` requirement for global pooling.
