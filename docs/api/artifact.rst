Artifact API
============

``mlx_lattice.artifact`` handles MLIR-first exchange bundles. A bundle contains
textual ``graph.mlir`` and ``weights.safetensors``. It does not execute the
graph.

Graph validation uses real MLIR infrastructure. MLIR-enabled native builds use
an in-process parser/verifier registered with the ``lattice`` dialect.
Lightweight builds can validate through the repository ``lattice-opt`` tool.
The Python layer does not parse lattice MLIR text itself.

Artifact validation also enforces the module-level ABI contract:
``lattice.ir_version`` must be ``0`` and ``lattice.weight_file`` must be
``"weights.safetensors"``. Unsupported versions or payload names fail before
the graph reaches runtime lowering. The module must also contain exactly one
``func.func`` entry named ``forward`` and that function must return at least
one value. The entry body is intentionally simple: one block containing only
``lattice`` dialect operations and ``func.return``. This matches the v0 MLX
importer contract and prevents silently ignored MLIR operations.

Executable lowering is available when the MLIR-enabled native extension is
built. The native extension parses and verifies the module, returns a structured
runtime plan, and Python lowers that plan through existing ``mlx_lattice.ops``
routes.

Runtime plan inputs carry importer-generated value names plus MLIR-derived
metadata: type text and ABI role. ``SparseTensor`` shorthand binding uses the
roles ``sparse_coords``, ``sparse_features``, and ``sparse_active`` on the first
three entry arguments. It does not depend on textual MLIR SSA argument names,
which are not a stable API contract.

The raw native payload is frozen into ``RuntimePlan`` before execution.
``LatticeProgram`` consumes that typed plan object, not arbitrary dictionaries,
so malformed importer payloads fail before any MLX operation lowering runs.
Plan freezing is schema-aware and ABI-aware: the plan must carry
``ir_version`` and ``weight_file`` values matching the verified module-level
contract, operation names must resolve through ``lattice-contract``, SSA
operand/result counts must match the annotated op definition, required
attributes must be present, enum attributes and structured packing metadata
must use valid v0 values, triple/numeric/string attributes must have the
expected payload shape, runtime value labels must be defined before use, and
returns must reference defined values. Frozen plans also deep-freeze nested
attribute payloads such as packing maps and stride triples so runtime lowerings
cannot accidentally mutate importer state.

The MLX importer lowers the sparse ABI operations, symbolic dense and packed
affine int4/int8 weights, forward/submanifold/target/transpose/generative
sparse convolution, local/global sparse pooling, point/voxel conversion,
feature projection, feature activation, sparse feature replacement, sparse
feature normalization, and sparse binary algebra. Unsupported dialect
operations should fail during lowering instead of being interpreted from
ad-hoc JSON or Python-side string switches.

Runtime lowerings are annotation-bound. A lowering declares each parameter with
``typing.Annotated`` metadata such as ``sparse_operand(0)``,
``array_operand(1)``, ``conv_weight_operand(1, input="x")``, or
``triple_attribute("kernel_size")``. The decorator compiles those annotations
into the registry entry, so operation modules do not manually inspect
``PlanOperation.operands`` or ``PlanOperation.attrs``. Binding metadata is
validated against the annotated dialect schema when the lowering is registered:
operand indexes must fit the declared SSA operands, attribute names must be
declared by the operation, and dependent bindings such as packed-weight
resolution must reference existing lowering parameters. The intended lowering
shape is a small semantic bridge:

.. code-block:: python

   @artifact_lowering(op=conv3d)
   def conv3d_from_artifact(
       x: Annotated[SparseTensor, sparse_operand(0)],
       weight: Annotated[mx.array | QuantizedWeight, conv_weight_operand(1, input="x")],
       bias: Annotated[mx.array | None, optional_array_operand(2)],
       *,
       kernel_size: Annotated[Triple, triple_attribute()],
       stride: Annotated[Triple, triple_attribute()],
       padding: Annotated[Triple, triple_attribute()],
       dilation: Annotated[Triple, triple_attribute()],
   ) -> SparseTensor:
       return conv3d(
           x,
           weight,
           bias,
           kernel_size=kernel_size,
           stride=stride,
           padding=padding,
           dilation=dilation,
       )

Dense weights are read from ``weights.safetensors`` using the exact
``storage_key`` declared by ``lattice.weight``. Packed weights use
``<storage_key>.weight``, ``<storage_key>.scales``, and
``<storage_key>.biases``. Bias parameters are regular dense
``lattice.weight`` values with family ``bias`` and layout ``bias_c``; they are
passed as optional operands to convolution-family and linear operations.

Feature-only dialect operations operate on dense feature tensors. This includes
``lattice.linear``, ``lattice.activation``, ``lattice.batch_norm``,
``lattice.layer_norm``, and ``lattice.rms_norm``. Sparse support preservation
must be explicit in MLIR through ``lattice.sparse.decompose`` followed by
``lattice.sparse.with_features``.

Bundle IO
---------

.. automodule:: mlx_lattice.artifact
   :members:

.. automodule:: mlx_lattice.artifact.io
   :members:

Runtime plan
------------

.. automodule:: mlx_lattice.artifact.plan
   :members:

MLIR validation
---------------

.. automodule:: mlx_lattice.artifact.validation
   :members:
