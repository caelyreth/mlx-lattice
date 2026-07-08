Artifact API
============

``mlx_lattice.artifact`` handles the MLX-side consumer half of the portable
model-artifact boundary. A bundle contains textual ``graph.mlir`` and
``weights.safetensors``. The bundle itself is exchange media; execution starts
only after ``load_lattice_program`` or ``compile_lattice_artifact`` validates
and lowers the graph.

The producer half belongs to training frameworks. For example, the Torch/CUDA
side should expose model-to-artifact APIs such as
``save_lattice_model_artifact``. Those APIs lower a framework model into
``graph.mlir`` plus weights. They should not be confused with MLX raw artifact
IO, which only saves, loads, validates, and compiles an already-formed bundle.

MLIR is the only artifact graph contract. Published macOS wheels include native
MLIR execution support. A source build may still lack that support when it was
compiled without ``MLX_LATTICE_ENABLE_MLIR``. In that case bundle IO remains
available, validation can still run through ``lattice-opt`` when provided, and
``native_artifact_execution_available()`` returns ``False``.
``compile_lattice_artifact`` and ``load_lattice_program`` then fail with a
direct capability error instead of selecting another graph runtime.

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

Programs can be invoked with one ``SparseTensor`` shorthand argument, positional
ABI tensors, or keyword ABI tensors:

.. code-block:: python

   program(x)
   program(x.coords, x.feats, x.active_rows)
   program(coords=x.coords, features=x.feats, active=x.active_rows)

These forms are intentionally strict. Sparse shorthand cannot be mixed with
keyword inputs; missing, duplicate, unexpected, or non-MLX input values fail
before any operation lowering runs. This makes producer/consumer drift visible
at the artifact boundary instead of inside a Metal kernel or sparse operator.

The raw native payload is frozen into ``RuntimePlan`` before execution.
``LatticeProgram`` consumes that typed plan object, not arbitrary dictionaries,
so malformed importer payloads fail before any MLX operation lowering runs.
Plan freezing is schema-aware and ABI-aware: the plan must carry
``ir_version`` and ``weight_file`` values matching the verified module-level
contract, operation names must resolve through ``lattice-contract``, SSA
operand/result counts must match the contract op definition, required
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

Runtime lowerings are schema-bound. A lowering names parameters after lattice
operands and attributes, and ``artifact_lowering`` binds those parameters from
the dialect schema. Operation modules do not manually inspect
``PlanOperation.operands`` or ``PlanOperation.attrs``. Extra metadata is only
needed for semantic cases that the raw MLIR type cannot express by itself, such
as resolving a packed convolution weight with the input channel count. The
intended lowering shape is a small semantic bridge:

.. code-block:: python

   @artifact_lowering(op=conv3d, weights={"weight": conv_weight(input="input")})
   def conv3d_from_artifact(
       input: SparseTensor,
       weight: mx.array | QuantizedWeight,
       bias: mx.array | None = None,
       *,
       kernel_size: Triple,
       stride: Triple,
       padding: Triple,
       dilation: Triple,
   ) -> SparseTensor:
       return conv3d(
           input,
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
Missing dense weights, incomplete packed triplets, or packed scale dtype drift
are runtime contract errors.

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

Executable import
-----------------

.. automodule:: mlx_lattice.artifact.runtime
   :members:

Runtime plan
------------

.. automodule:: mlx_lattice.artifact.plan
   :members:

MLIR validation
---------------

.. automodule:: mlx_lattice.artifact.validation
   :members:
