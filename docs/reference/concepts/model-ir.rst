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
submanifold convolution, target convolution, dense feature projection,
dense feature activation, and sparse algebra semantics.
``weights.safetensors`` stores dense or packed quantized payloads referenced
by ``lattice.weight`` operations.

Package metadata can be added later as shallow manifest data, but graph
semantics belong in MLIR.

Executable entry
----------------

An executable artifact contains one graph entry:

.. code-block:: text

   func.func @forward(...) -> (...) {
     ...
     return ... : ...
   }

The verifier rejects modules with no function, more than one function, an
entry not named ``forward``, or a ``forward`` function that returns no values.
The entry body is a single block containing only ``lattice`` dialect operations
and ``func.return``. SSA argument names in textual MLIR are not semantic API
names; stable named inputs should be added through explicit ABI metadata rather
than inferred from parser-preserved text.

The native MLX importer exposes a structured execution plan after verification.
Plan argument records include generated runtime value names, canonical MLIR type
text, and an ABI role. Sparse tensor shorthand is accepted only when the first
three entry arguments are tagged as ``sparse_coords``, ``sparse_features``, and
``sparse_active`` by the importer. This keeps the ergonomic Python call surface
while making the contract type/role based instead of name based.

Python freezes the native payload into ``RuntimePlan`` before execution.
Lowering annotations receive typed plan operations rather than raw dictionaries;
the dictionary shape is only the native binding transport, not a second runtime
IR. This freeze step resolves each operation through the annotated
``lattice-contract`` schema and validates operation arity plus required
attributes before any framework lowering runs. It also checks runtime
use/definition order, rejects duplicate value labels, and rejects returns that
do not refer to defined values.

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

.. code-block:: text

   %w = lattice.weight @stem.qweight
     {storage_key = "stem.qweight",
      layout = #lattice.weight_layout<conv3d_o_zyx_i>,
      packing = #lattice.packing<int4,
                                 group_size = 32,
                                 scale_dtype = f16,
                                 mode = affine>}
     : !lattice.weight<conv3d, i4>

The consuming operation remains ``lattice.conv3d``.

In ``weights.safetensors``, dense weights use the declared ``storage_key``
directly. Packed int4/int8 weights use ``<storage_key>.weight``,
``<storage_key>.scales``, and ``<storage_key>.biases``. The graph still owns
bit width, group size, scale dtype, mode, and logical layout through the
``lattice.weight`` attributes.

Bias is explicit dataflow. A convolution or linear operation may carry an
optional third operand produced by ``lattice.weight`` with family ``bias`` and
layout ``bias_c``. Omitting that operand means no bias is applied.

Feature-only updates
--------------------

Feature-only MLIR operations are value-aligned, not sparse-object mutations.
``lattice.linear`` and ``lattice.activation`` consume rank-2 dense feature
tensors. ``lattice.batch_norm``, ``lattice.layer_norm``, and
``lattice.rms_norm`` follow the same rule. A graph that needs to preserve
sparse support must spell that out:

.. code-block:: text

   %coords, %features, %active = lattice.sparse.decompose %input
     : !lattice.sparse_tensor<rank = 3,
                              coord = batch_x_y_z,
                              feature = row_channel,
                              dtype = f16>
       -> (tensor<?x4xi32>, tensor<?x32xf16>, tensor<1xi32>)

   %features2 = lattice.activation %features
     {kind = #lattice.activation<relu>,
      approximate = #lattice.gelu_approx<none>,
      alpha = 0.01 : f32,
      beta = 1.0 : f32,
      threshold = 20.0 : f32}
     : (tensor<?x32xf16>) -> tensor<?x32xf16>

   %out = lattice.sparse.with_features %input, %features2
     : (!lattice.sparse_tensor<rank = 3,
                               coord = batch_x_y_z,
                               feature = row_channel,
                               dtype = f16>,
        tensor<?x32xf16>)
       -> !lattice.sparse_tensor<rank = 3,
                                 coord = batch_x_y_z,
                                 feature = row_channel,
                                 dtype = f16>

This keeps coordinate identity visible to the verifier and prevents importers
from hiding sparse mutation semantics inside framework-specific helpers.

Normalization parameters are symbolic weights, not attributes. BatchNorm uses
explicit frozen mean and variance operands; training-mode running-stat updates
are outside the deployment artifact ABI. Scale, mean, and variance use the
``channel`` weight family with ``channel_c`` layout. Bias uses the ``bias``
family with ``bias_c`` layout.

Runtime boundary
----------------

``mlx_lattice.artifact`` loads and saves the exchange bundle. In MLIR-enabled
builds it can also ask the native parser/verifier for a structured execution
plan and lower that verified plan through ``mlx_lattice.ops``. The Python layer
does not parse MLIR text itself and should not carry a second graph runtime.

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
