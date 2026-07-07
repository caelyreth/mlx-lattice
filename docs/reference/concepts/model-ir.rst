Sparse model IR
===============

The sparse model exchange contract is MLIR-first. Graph semantics live in the
``lattice`` MLIR dialect; weights live in ``weights.safetensors``. The Python
contract package provides an annotation-backed dialect schema and textual MLIR
builder, but it does not define a second graph IR.

Artifact layout
---------------

The current exchange bundle is:

.. code-block:: text

   model.lattice/
     graph.mlir
     weights.safetensors

``graph.mlir`` owns sparse value construction, weight binding, convolution,
submanifold convolution, target convolution, feature projection, and sparse
algebra semantics. ``weights.safetensors`` stores dense or packed quantized
payloads referenced by ``lattice.weight`` operations.

Package metadata can be added later as shallow manifest data, but graph
semantics belong in MLIR.

Annotation-backed contract
--------------------------

``lattice-contract`` declares the dialect surface through annotations in one
schema. The MLIR builder consumes that schema directly:

.. code-block:: python

   builder = MLIRModuleBuilder()
   x = builder.sparse_make(...)
   w = builder.weight(...)
   y = builder.conv3d(input=x, weight=w, ...)

This keeps op names, result names, and attribute names centralized. Backend
exporters should call the generated builder surface rather than manually
formatting dialect strings or maintaining their own operation registries.

Quantization model
------------------

Quantized graph op variants such as ``lattice.quantized_conv3d`` are not part
of the semantic graph surface. Quantization is represented by weight packing:

.. code-block:: mlir

   %w = lattice.weight @stem.qweight
     {storage_key = "stem.qweight",
      layout = #lattice.weight_layout<conv3d_o_zyx_i>,
      packing = #lattice.packing<int4,
                                 group_size = 32,
                                 scale_dtype = f16,
                                 mode = affine>}
     : !lattice.weight<conv3d, i4>

The consuming operation remains ``lattice.conv3d``.

Runtime boundary
----------------

``mlx_lattice.artifact`` only loads and saves the exchange bundle. It does not
execute MLIR and does not carry a Python graph runtime. A future importer should
lower verified lattice MLIR into an MLX execution plan.

Runtime-only structures such as coordinate managers, relation caches, CSR
views, implicit-GEMM maps, TensorOps routes, and Metal kernel dispatch choices
must stay outside the portable IR.

Sparse value ABI
----------------

Sparse values should cross framework boundaries through explicit ABI tensors:
coordinates, features, active row count, and sparse stride. The dialect models
this with ``lattice.sparse.make`` and ``lattice.sparse.decompose``. It
deliberately excludes runtime identity such as ``CoordinateManager`` and
``CoordinateMapKey``.
