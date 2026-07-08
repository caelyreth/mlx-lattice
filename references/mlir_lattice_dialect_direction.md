# Locked Direction: MLIR Lattice Dialect and Sparse Artifact ABI

**Audience:** engineers working on `mlx-lattice`, future CUDA/Torch-side
training support, artifact conversion, and sparse runtime portability.

**Status:** accepted design direction for the next IR architecture pass. This
document locks the conceptual shape; exact operation spelling, TableGen/ODS
definitions, parser implementation, and packaging details still need an
implementation design.

**Related notes:**

- `references/mlir_ir_handoff_design.md`
- `references/mlir_sparse_api_export_handoff.md`
- `references/cuda_training_metal_inference.md`
- `references/lattice-contract-cuda-porting.md`

---

## 1. Decision summary

The project should move from the current JSON graph manifest toward an
MLIR-based sparse model artifact.

MLIR should replace the **computation graph contract**, not the whole artifact
package. The desired artifact shape is:

```text
model.lattice/
  graph.mlir             # computation semantics during early development
  graph.mlirbc           # optional compact bytecode after dialect stability
  weights.safetensors    # tensor payloads
  manifest.json          # shallow package metadata only
```

The core direction is:

```text
SparseTensor user/runtime object
  -> deterministic sparse value ABI
  -> lattice MLIR dialect operations
  -> target runtime reconstruction
```

The dialect must be portable and direction-neutral. The main current use case
is CUDA/Torch training to MLX/Metal inference, but the dialect should not be a
Torch trace, an MLX call trace, or a serialization of either library's Python
classes.

The most important architectural rule is:

> `SparseTensor` is a user/runtime API. The artifact boundary is a sparse value
> ABI plus explicit sparse operations. MLIR owns portable sparse semantics.

### 1.1 SHOULD DO: active project goals

The MLIR work should target a deployment contract and importer, not a custom
compiler stack. The accepted flow is:

```text
source exporter
  -> graph.mlir + weights.safetensors
  -> native MLIR parser/verifier
  -> typed RuntimePlan
  -> MLX/Metal execution through mlx_lattice.ops and native kernels
```

The project should do these things:

1. Treat `graph.mlir` as the only portable computation graph contract.
2. Treat `weights.safetensors` as the tensor payload store referenced by
   symbolic `lattice.weight` operations.
3. Keep any package manifest shallow: names, versions, file names, checksums,
   producer metadata, and optional compatibility notes only.
4. Verify artifacts with real MLIR parser/verifier infrastructure before any
   framework-specific lowering.
5. Freeze native importer output into typed plan objects (`RuntimePlan`,
   `PlanOperation`, typed arguments, typed attributes) before execution.
6. Lower verified plan operations to public `mlx_lattice.ops` or equivalent
   internal runtime entry points using schema-bound lowering signatures.
7. Make every accepted v0 dialect op either executable on MLX or rejected
   during verification/import with a clear diagnostic.
8. Keep backend execution choices—TensorOps routes, CSR views, coordinate
   caches, rulebooks, Metal/CUDA scheduling, and path-selection thresholds—out
   of the portable artifact.
9. Let future Torch/CUDA or other source packages depend on the dialect
   contract, not on `mlx-lattice` runtime internals.

### 1.2 DO NOT DO: rejected shapes

These are explicitly out of scope for the stable artifact direction:

1. Do not reintroduce JSON as a semantic graph contract.
2. Do not embed opaque JSON payloads inside MLIR operations as the main
   semantics.
3. Do not implement a Python MLIR parser in `mlx-lattice`; use MLIR tooling.
4. Do not serialize PyTorch modules, MLX modules, Python sparse objects,
   `CoordinateManager`, or `CoordinateMapKey` as correctness state.
5. Do not make a custom MLIR-to-Metal compiler the near-term target. The
   project already owns optimized runtime kernels; MLIR should import into that
   runtime first.
6. Do not make ahead-of-time device-specific executables the portable model
   artifact. They reduce portability and can be considered later only as an
   optional cache layer.
7. Do not persist backend route names, TensorOps tile masks, CUDA rulebook
   layouts, Metal dispatch choices, or benchmark strategy names as model
   semantics.
8. Do not maintain parallel hand-written op-name registries across Torch, MLX,
   docs, and tests once the dialect definitions can generate or validate those
   surfaces.

---

## 2. Why MLIR is useful here

MLIR is useful because it provides infrastructure the current JSON layer cannot
provide cleanly:

- typed SSA graph values;
- dialect-defined operations;
- dialect-defined attributes and types;
- parser and verifier infrastructure;
- human-readable text form;
- compact bytecode form later;
- canonicalization and conversion passes later;
- clear separation between high-level semantics and lower-level execution
  representations.

MLIR is not valuable merely because it has a stronger-looking file extension.
If the project stores opaque JSON blobs inside MLIR attributes or represents all
sparse computation as generic custom calls, it recreates the same ad-hoc
contract problem in a different syntax.

The useful move is to define a small project dialect with real sparse operation
semantics, real verifier rules, and clear value/weight/layout contracts.

---

## 3. What should stay outside MLIR

MLIR should not store these as portable model semantics:

- Python `SparseTensor` object identity;
- Python module class structure;
- `CoordinateManager` identity;
- `CoordinateMapKey` identity;
- runtime cache maps;
- Metal buffer handles;
- CUDA pointers or rulebook handles;
- TensorOps route names;
- implicit-GEMM tile masks as the main graph contract;
- backend scheduling decisions;
- benchmark or path-selection hints that do not affect correctness.

These can exist in runtime implementations and lowering passes, but they should
not define portable model behavior.

---

## 4. Layered model

The project should treat the sparse stack as three separate levels.

### 4.1 Level 0: portable model semantics

These are the operations users and artifacts care about:

- sparse tensor construction/decomposition;
- sparse convolution;
- submanifold convolution;
- convolution on target coordinates;
- transposed and generative sparse convolution;
- sparse pooling;
- feature projection and feature-only math;
- sparse coordinate-aligned algebra;
- point/voxel conversion where used by deployed models.

Level 0 belongs in portable MLIR artifacts.

### 4.2 Level 1: explicit reusable sparse structures

These are semantic auxiliary values that may be shared between operations:

- relation;
- coordinate alignment;
- sparse quantization;
- point-to-voxel interpolation map;
- coordinate set/order values;
- occupancy/expansion values if required by public semantics.

Level 1 can appear in portable MLIR artifacts when the structure is meaningful,
shared, expensive, or needed for validation.

### 4.3 Level 2: backend execution views

These are implementation details:

- CSR execution views;
- input/kernel CSR permutations;
- implicit-GEMM maps;
- sorted implicit-GEMM maps;
- TensorOps tile masks;
- CUDA rulebook buffer layout;
- Metal kernel dispatch routes;
- path-selection thresholds.

Level 2 should be rebuilt by the target runtime or generated by lowering
passes. It should not be the stable cross-framework model contract.

---

## 5. Sparse value ABI

The dialect must define a sparse value by visible tensor components and
explicit metadata. It should not depend on the target runtime's wrapper object.

Minimum sparse value components:

```text
coords        integer tensor, shape (capacity, 4)
features      dense tensor, shape (capacity, channels)
active_rows   integer scalar/vector shape (1,)
stride        static spatial triple
```

Minimum sparse value metadata:

```text
rank                  3 for current library surface
coordinate_order      batch_x_y_z
feature_layout        row_channel
coordinate_dtype      int32 preferred for portable GPU paths
feature_dtype         dtype of feature tensor
duplicate_rule        whether duplicate coordinates are rejected/coalesced
row_ordering          explicit if an op depends on row order
```

Current `mlx-lattice` maps well to this:

```text
SparseTensor.coords       -> ABI coords
SparseTensor.feats        -> ABI features
SparseTensor.active_rows  -> ABI active_rows
SparseTensor.stride       -> ABI stride
```

Runtime importers may generate internal value labels, but those labels are not
semantic ABI names. Entry argument binding should use verified type and role
metadata. For the current MLX importer, arguments consumed by
`lattice.sparse.make` are tagged as `sparse_coords`, `sparse_features`, and
`sparse_active`; ergonomic framework shorthand can be layered on those roles
without depending on textual SSA names.

The Python MLX runtime should freeze the native binding payload into typed plan
records before execution. Lowerings should consume typed `PlanOperation`
objects so malformed importer payloads fail at the boundary instead of leaking
as ad-hoc dictionary errors inside operation code. Plan freezing should be
schema-aware: qualified op names resolve through `lattice-contract`, SSA
operand/result arity is checked against the annotated op definition, and symbol
operands are handled as declared symbol operands even if the native transport
encodes them as attributes. The same boundary should validate SSA-like
dataflow: no duplicate runtime value labels, no use-before-definition, and no
returns of undefined values.

`CoordinateManager`, `CoordinateMapKey`, and relation caches are reconstruction
and performance state, not ABI state.

### 5.1 Batch metadata

`batch_counts` should not be a required sparse value component. It is useful
for optimized or row-partitioned execution, but semantically the batch
dimension already lives in `coords[:, 0]`.

For operations such as global pooling, the dialect should prefer explicit
semantics:

```text
batch_size attribute or tensor
batch coordinate column
pooling reduction mode
empty-batch behavior
```

`batch_counts` can remain an optional layout hint or runtime-derived structure
when row ordering guarantees make it safe.

---

## 6. Component ABI and sparse value type

The first implementation should support component-level interoperability because
source exporters, especially Torch-side exporters, can more easily expose tensor
leaves than custom Python objects.

However, the dialect should still have a first-class sparse value concept. A
good shape is:

```mlir
%sp = lattice.sparse.make %coords, %features, %active
  {stride = [1, 1, 1],
   coord_order = #lattice.coord<batch_x_y_z>}
  : (tensor<?x4xi32>, tensor<?x?xf16>, tensor<1xi32>)
    -> !lattice.sparse_tensor<rank = 3,
                              coord = batch_x_y_z,
                              feature = row_channel,
                              dtype = f16>

%coords2, %features2, %active2 = lattice.sparse.decompose %sp
  : !lattice.sparse_tensor<...>
    -> (tensor<?x4xi32>, tensor<?x?xf16>, tensor<1xi32>)
```

This gives both sides what they need:

- exporters can work with tensor components;
- the IR can still reason about sparse values;
- importers can reconstruct their preferred runtime object;
- future passes can avoid manually tracking loose tuples everywhere.

---

## 7. Primitive operation families

The dialect should group operations by semantic feature set, not by backend.

### 7.1 Sparse convolution family

First-class operations:

```text
lattice.conv3d
lattice.subm_conv3d
lattice.target_conv3d or lattice.conv3d with explicit target support
lattice.conv_transpose3d
lattice.generative_conv_transpose3d
```

Required attributes:

```text
kernel_size
stride, where semantically meaningful
padding, where semantically meaningful
dilation, where semantically meaningful
```

Submanifold convolution should not be represented as generic forward
convolution plus discarded coordinates. It has distinct output-support
semantics and should remain first-class.

Target-coordinate convolution is also first-class. The public API does not need
a separate `TargetConv` module, but the IR must be able to express:

```text
input sparse support
explicit target coordinate support
kernel geometry
output features on target support
```

Two acceptable MLIR shapes:

```mlir
%out = lattice.conv3d %input, %weight, target %target
  {kernel_size = [3, 3, 3], stride = [1, 1, 1], padding = [1, 1, 1]}
```

or:

```mlir
%rel = lattice.relation.target %input, %target
  {kernel_size = [3, 3, 3], stride = [1, 1, 1], padding = [1, 1, 1]}
%out = lattice.conv_with_relation %input, %weight, %rel
```

The high-level operation is better for simple model artifacts. The explicit
relation form is useful when relations are shared or inspected.

### 7.2 Relation family

Relations should exist as semantic values when explicit relation reuse or target
support matters.

Potential operations:

```text
lattice.relation.forward
lattice.relation.submanifold
lattice.relation.target
lattice.relation.transpose
lattice.relation.generative
lattice.conv_with_relation
lattice.pool_with_relation
```

Relation value semantics:

```text
kind
source support
optional target support
output support
kernel offsets / kernel geometry
active counts
edge semantics if materialized
```

Do not expose TensorOps or CUDA/Metal execution views as relation semantics.
Those are lowerings from a relation, not the relation itself.

### 7.3 Sparse algebra family

Coordinate-aligned sparse algebra should be part of the dialect rather than a
side convention.

Operations/concepts:

```text
lattice.align
lattice.sparse.binary
lattice.sparse.cat
lattice.coordinate.union
lattice.coordinate.intersection
lattice.coordinate.lookup
lattice.coordinate.order
lattice.sparse.crop
lattice.sparse.prune
```

Required semantic fields:

```text
binary op: add / sub / mul / maximum / minimum
join mode: inner / left / right / outer
fill values where relevant
stride compatibility
coordinate dtype compatibility
row ordering behavior
```

This prevents each backend from independently redefining sparse binary algebra,
alignment, and missing-row handling.

### 7.4 Feature-only family

Feature-only operations preserve sparse coordinate identity and transform only
the feature matrix.

Examples:

```text
linear
batch_norm
layer_norm
rms_norm
relu
sigmoid
silu
tanh
gelu
dropout, if training/export requires it
```

When a feature-only operation maps losslessly to standard tensor or StableHLO
ops, the dialect can decompose:

```mlir
%coords, %features, %active = lattice.sparse.decompose %input
%features2 = stablehlo.maximum %features, %zero
%output = lattice.sparse.with_features %input, %features2
```

But the sparse identity preservation must remain explicit. Do not make sparse
feature operations silently depend on object identity from the source runtime.

### 7.5 Pooling family

Pooling should distinguish:

```text
local sparse pooling over a kernel relation
global pooling over batch coordinates
```

Global pooling should define empty-batch behavior, reduction mode, and batch
cardinality semantics explicitly.

### 7.6 Point/voxel family

Point/voxel utilities are not merely helpers if deployed models use them. They
produce reusable semantic structures.

Reusable value types remain useful future work when a graph needs to share
point/voxel planning across multiple feature tensors:

```text
!lattice.quantization
!lattice.point_voxel_map
```

The first portable ABI should expose high-level point/voxel semantics:

```text
lattice.voxelize
lattice.devoxelize
```

Lower-level reusable operations can be added after the high-level contract is
stable:

```text
lattice.sparse_quantize
lattice.voxelize_features
lattice.point_voxel_map
lattice.interpolate_point_features
```

Required semantics:

```text
voxel_size
origin
batch index convention
active row handling
reduction: sum / mean
interpolation: nearest / linear
point and voxel dtype constraints
```

---

## 8. Weight and parameter contract

Weights should remain in `weights.safetensors`. MLIR should reference weights
symbolically and describe their logical contract.

Each parameter binding should include:

```text
symbol name
storage key
logical dtype
logical shape
logical layout
optional physical packing
packing/conversion version
```

Example direction:

```mlir
%w = lattice.weight @stem.weight
  {storage_key = "stem.weight",
   layout = #lattice.weight_layout<conv3d_o_zyx_i>,
   packing = #lattice.packing<dense>}
  : !lattice.weight<conv3d, f16>
```

Correctness-critical layout must not live only in string naming conventions.

---

## 9. Quantization contract

Quantization should be represented as parameter/storage semantics, not as a
duplicated operation family unless quantization changes mathematical semantics.

Prefer:

```text
lattice.conv3d input, quantized_weight
lattice.linear input, quantized_weight
```

over:

```text
lattice.quantized_conv3d
lattice.quantized_linear
```

The weight value/type should describe:

```text
bit width: int4 / int8
group size
scale dtype
zero point policy, if any
packing layout
logical dequantized layout
storage tensor keys
packing version
```

This keeps operation semantics stable and avoids op explosion.

Runtime importers remain free to select specialized quantized kernels when the
weight contract permits it.

---

## 10. StableHLO usage

StableHLO should be used only where it preserves semantics losslessly.

Good candidates:

```text
dense elementwise feature math
dense reductions
reshapes/broadcasts over ordinary dense tensors
parts of feature-only modules after sparse decomposition
```

Weak candidates:

```text
sparse convolution
submanifold convolution
target-coordinate convolution
coordinate alignment
relation construction
point/voxel sparse mapping
```

Avoid using StableHLO `custom_call` as the primary sparse representation. That
would make sparse semantics opaque and defeat the reason for adopting MLIR.

---

## 11. Verifier requirements

The dialect verifier should reject invalid artifacts before MLX, CUDA, or any
runtime-specific execution begins.

Minimum verifier coverage:

```text
coords rank is 2
coords second dimension is 4
coords dtype is supported
features rank is 2
coords/features capacities match
active_rows shape and dtype are valid
sparse rank is supported
coordinate order is supported
feature layout is supported
stride triples are positive
kernel/padding/dilation attrs are valid triples
submanifold attrs do not contain stride/padding
target conv has valid target support
weight symbol exists
weight logical layout matches operation expectation
quantized weight metadata is complete
join mode is valid
point/voxel dtypes and shapes match
```

Runtime importers can add backend-specific validation, but the dialect verifier
should catch semantic invalidity independent of backend.

---

## 12. Package manifest role

`manifest.json` should become shallow package metadata.

Allowed:

```text
package format version
graph file name
weight file name
producer name/version
optional target compatibility note
checksums
debug labels
```

Not allowed as primary graph semantics:

```text
op list
node inputs/outputs
kernel sizes
weight layouts
sparse layout rules
dtype policy for graph values
```

Those belong in MLIR.

---

## 13. Source-side requirements

A source runtime such as a future Torch/CUDA package should keep an ergonomic
user API:

```python
x = SparseTensor(coords, features, metadata=...)
y = model(x)
```

Internally, exportable sparse modules should lower to graph-visible functional
operators over tensor components and explicit metadata.

Required source-side properties:

```text
SparseTensor can decompose deterministically into ABI components
SparseTensor can reconstruct from ABI components
semantic metadata is separate from cache metadata
portable sparse ops have stable symbolic names
portable sparse ops have explicit schemas
portable sparse ops have fake/meta implementations
export mode observes sparse ops as graph nodes
unsupported ops fail clearly
```

The dialect should not require the source-side class and MLX-side class to be
identical. They only need to agree on sparse value ABI and operation semantics.

---

## 14. MLX-side importer requirements

The MLX importer should:

```text
parse MLIR
verify the lattice dialect
load safetensors weights
validate weight contracts
reconstruct MLX SparseTensor values from ABI components
rebuild CoordinateManager and CoordinateMapKey state locally
dispatch to public mlx_lattice operations or lower-level runtime calls
apply backend-specific path selection internally
```

The importer should not need to know Torch internals. It should only understand
the lattice dialect and weight/package contract.

---

## 15. JSON contract status

The old JSON graph contract is no longer present in the active
`lattice-contract` or `mlx_lattice.artifact` code path. The active exchange
media are `graph.mlir` plus `weights.safetensors`; any future package manifest
must remain shallow metadata and must not duplicate graph semantics.

The current implementation state is:

1. JSON graph authoring/loading is not an active compatibility surface.
2. MLIR dialect declarations live in the annotation-backed Python schema and
   the C++ ODS dialect/verifier.
3. Golden MLIR fixtures cover valid and invalid dialect shapes.
4. The native importer parses/verifies MLIR, transports ABI metadata, and
   freezes into `RuntimePlan`.
5. MLX execution lowerings are schema-bound semantic bridges to public
   `mlx_lattice.ops` calls.

Do not reintroduce a second graph contract in JSON form.

---

## 16. Source of truth and code generation

The long-term source of truth should be declarative dialect definitions, not
manually duplicated string registries in Torch, MLX, docs, and tests.

Preferred direction:

```text
lattice dialect op/type/attr definitions
  -> generated/shared Python contract constants
  -> generated docs/schema tables
  -> importer binding checks
  -> exporter validation tables
```

This directly addresses the current risk where op names, parameter names,
attribute names, and value types can drift across packages.

During the first spike, hand-written bindings are acceptable, but the design
should avoid creating new manual registries that become hard to replace.

---

## 17. Initial milestone

The first milestone should be small but structurally correct.

Suggested scope:

```text
lattice.sparse.make
lattice.sparse.decompose
lattice.conv3d
lattice.subm_conv3d
lattice.conv3d with explicit target support or lattice.relation.target
lattice.sparse.with_features
lattice.linear
lattice.sparse.binary with operation and join semantics
weight symbol binding
```

Success criteria:

```text
graph.mlir is human-readable
dialect verification catches malformed sparse values
weights stay in safetensors
MLX importer reconstructs SparseTensor values
MLX output matches source/golden output
no source Python object state is required
no backend execution view is persisted as model semantics
```

---

## 18. Anti-patterns to reject

Reject these during implementation review:

- MLIR ops with opaque JSON payloads as the main semantic representation.
- StableHLO custom calls for most sparse behavior.
- Serialized Python module/class names as correctness semantics.
- Serialized `CoordinateManager` or `CoordinateMapKey` state.
- Backend route names in the portable graph.
- Separate quantized op families when weight typing is enough.
- Hidden reliance on `batch_counts` when batch coordinate semantics are enough.
- Graph behavior determined only by weight key naming conventions.
- Silent fallback for unsupported sparse operations.

---

## 19. Open design questions

These still need implementation-level decisions:

1. Exact dialect namespace: `lattice`, `lattice_sparse`, or another stable
   spelling.
2. Whether the first implementation builds a C++ MLIR dialect, a Python-facing
   wrapper around generated dialect bindings, or a staged textual parser spike.
3. Exact sparse type syntax.
4. Exact weight layout attribute names.
5. Whether explicit relation ops are included in milestone one or introduced
   immediately after high-level conv import works.
6. How much of the Python `lattice_contract` package should be generated from
   dialect definitions.
7. How long JSON artifact loading remains supported after MLIR lands.

These are implementation choices. They do not change the locked direction that
portable sparse semantics belong in an MLIR lattice dialect and runtime objects
are reconstructed after import.

---

## 20. Final locked shape

The accepted architecture is:

```text
Normal modeling API
  SparseTensor and sparse modules remain ergonomic.

Export/runtime boundary
  Sparse values decompose into tensor components plus explicit metadata.

Portable graph
  MLIR lattice dialect owns sparse semantics, types, attributes, and verifier.

Weights
  safetensors stores payloads; MLIR binds symbols to logical layout/packing.

Package manifest
  JSON remains shallow and boring.

MLX runtime
  Reconstructs SparseTensor/CoordinateManager state and selects optimized Metal
  paths internally.

CUDA/Torch runtime
  Emits the same dialect from graph-visible functional sparse ops.
```

The direction is deliberately universal: the dialect is not "Torch export to
MLX" encoded as a one-way adapter. It is a portable sparse lattice computation
contract that the current one-way deployment path can use first.
