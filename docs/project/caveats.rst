Caveats and stability notes
===========================

This page records user-visible constraints and stability boundaries.

For API stability levels, read :doc:`stability`.

Coordinate order is stable
--------------------------

Coordinate rows are ordered as ``(batch, x, y, z)``. This is the canonical
internal and public order. Inputs using another convention should be converted
before constructing ``SparseTensor`` objects.

Coordinate value equality is not identity
-----------------------------------------

Two tensors can contain equal coordinate rows but have different
``CoordinateMapKey`` values. Cached relations are keyed by coordinate identity,
not by a late comparison of array contents. Use sparse alignment when combining
independently constructed tensors.

Coordinate uniqueness is required
---------------------------------

Each active coordinate row must be unique. Lattice relations define one feature
row per ``(batch, x, y, z)`` location, so duplicate-coordinate inputs are not a
supported accumulation convention. Normalize duplicate coordinates explicitly
at the import boundary before constructing a ``SparseTensor``. In particular,
MinkowskiEngine checkpoints and tensors must not be treated as directly
interchangeable until their coordinate and kernel conventions have been
converted.

Transpose support is explicit
-----------------------------

``conv_transpose3d`` uses a relation derived from a compatible earlier forward
operation unless explicit target coordinates are supplied. By contrast,
``generative_conv_transpose3d`` expands support according to
``target = source * stride + offset``. Supplying ``coordinates`` to either
operation makes that set the complete output support: coordinates outside it
are never synthesized.

Transpose stride must divide the input sparse stride. A target sparse tensor
must use the resulting output stride. This is a validation error rather than a
best-effort coordinate conversion because silently accepting a mismatched
target changes the lattice represented by the result.

Active rows and capacity differ
-------------------------------

Native coordinate builders may allocate a buffer larger than the active set.
Use ``active_rows`` to determine how many rows are valid. Treating every
allocated row as active can include uninitialized or padded coordinate rows in
later operations.

Dtype boundaries
----------------

.. list-table::
   :header-rows: 1
   :widths: 28 34 38

   * - Surface
     - Supported dtype boundary
     - Notes
   * - Metal coordinates
     - ``int32``
     - Sparse convolution and pooling validate this before launch.
   * - Floating convolution
     - ``float16`` or ``float32`` features/weights
     - Specialized Metal routes depend on dtype and channel count.
   * - Local pooling
     - ``float32`` features
     - Sum, max, and average local pooling share this boundary.
   * - Point/voxel
     - ``float32`` points/features, ``int32`` maps
     - Native quantization and interpolation routes use this contract.
   * - Quantized weights
     - int4/int8 packed in ``uint32``
     - Scales/biases match feature dtype at execution.

Global pooling requires batch metadata
--------------------------------------

Global pooling reduces to a dense ``(B, C)`` array. It requires
``batch_counts`` because a sparse coordinate buffer alone does not encode empty
batches. Sum and average define empty-batch behavior; max rejects empty batches.

Internal routes are not public APIs
-----------------------------------

Kernel names, CSR view names, sorted implicit-GEMM views, TensorOps variants,
and diagnostic reference routes are backend implementation details. They can be
useful for debugging a failing run, but application code calls public
operations and modules.

Entropy byte streams are provisional
------------------------------------

Entropy helpers are public callables, but the exact encoded byte stream is not
yet a cross-version persistence format. If you store streams externally, record
the ``mlx-lattice`` version alongside them.

Quantization is storage-real, not fake quantization
---------------------------------------------------

``QuantizedWeight`` stores packed int4/int8 data plus affine metadata. Supported
native routes consume packed storage. If you want the floating contract, call
``dequantize_weight`` explicitly and use the floating operation.

Artifacts and checkpoints are versioned boundaries
--------------------------------------------------

The current artifact format is MLIR IR version 1 with canonical
``conv3d_o_xyz_i`` convolution weights. Legacy JSON manifests, MLIR IR version
0, and ``conv3d_o_zyx_i`` weights are rejected at load time. Runtime loaders do
not infer historical TorchSparse or MinkowskiEngine kernel-row order. Convert a
trusted legacy checkpoint once with the CUDA-side checkpoint converter, retain
its permutation manifest with the converted weights, and validate a known
input/output fixture before deployment.

Artifacts also do not serialize coordinate managers, relation caches, native
backend handles, or selected kernel routes. Those are reconstructed locally and
can differ by backend without changing the public graph semantics.

Sparse performance depends on geometry
--------------------------------------

Sparse runtime is not determined by active row count alone. Important factors
include edge count, kernel volume, channel count, coordinate pattern, sorted
view availability, and whether the operation is relation-bound or
arithmetic-bound. Report those fields with benchmark results.

Point/voxel maps are geometry-specific
--------------------------------------

A point-to-voxel map is tied to points, batch indices, voxel size, origin,
voxel coordinates, and interpolation mode. Reusing a map after changing any of
those inputs is a semantic error.
