# Lattice MLIR Dialect v0 Contract

**Status:** implemented v0 contract for parser/verifier coverage and MLX
artifact import. This document defines the stable semantic ABI that exporters,
fixtures, and importers should follow.

**Scope:** sparse value ABI, weight binding, sparse convolution families,
point/voxel conversion, dense feature updates, coordinate-aligned sparse
algebra, artifact module metadata, and MLX importer-visible semantics.

**Non-scope:** MLIR bytecode compatibility, StableHLO integration, Torch
exporter implementation, and TensorOps/CSR/lower-level backend scheduling.

Related direction note:

- `references/mlir_lattice_dialect_direction.md`

Examples:

- `references/mlir_examples/README.md`
- `references/mlir_examples/conv3d_basic.mlir`
- `references/mlir_examples/subm_conv3d_basic.mlir`
- `references/mlir_examples/target_conv3d_basic.mlir`
- `references/mlir_examples/feature_linear_with_sparse_identity.mlir`
- `references/mlir_examples/feature_activation_with_sparse_identity.mlir`
- `references/mlir_examples/sparse_binary_outer.mlir`
- `references/mlir_examples/sparse_cat_outer.mlir`
- `references/mlir_examples/quantized_weight_conv3d.mlir`
- `references/mlir_examples/invalid/*.mlir`

---

## 1. v0 contract goal

The v0 MLIR layer is contract-first. The goal is to define what a valid
portable sparse lattice graph looks like and to make invalid graph shapes fail
before framework-specific lowering.

The v0 contract answers:

- What is a sparse value?
- What is a weight?
- How are weights bound to `safetensors` storage?
- How is target-coordinate convolution represented?
- How is submanifold convolution kept distinct from forward convolution?
- How is feature-only sparse identity preservation represented?
- How are coordinate-aligned sparse binary algebra and sparse feature
  concatenation represented?
- Which invalid graph shapes must be rejected by a future verifier?

---

## 2. Dialect namespace

Use:

```text
lattice
```

Do not use backend-specific names such as `mlx_lattice`, `torch_lattice`,
`metal`, or `cuda` in operation/type names.

The dialect is a portable sparse lattice computation contract. It is not a
Torch export trace and not an MLX runtime trace.

---

## 3. Artifact package shape

The package-level artifact shape remains:

```text
model.lattice/
  graph.mlir
  weights.safetensors
  manifest.json
```

`graph.mlir` contains computation semantics.

`weights.safetensors` contains tensor payloads.

Dense weight payloads use the exact `storage_key` declared by
`lattice.weight`. Packed affine int4/int8 payloads use deterministic suffixed
keys:

| Tensor | Key |
|---|---|
| packed integer codes | `<storage_key>.weight` |
| per-group scales | `<storage_key>.scales` |
| per-group affine biases | `<storage_key>.biases` |

The MLIR `packing` attribute remains the semantic source for bit width,
group size, scale dtype, and mode. The suffixed safetensors keys are only the
payload binding convention.

`manifest.json` is shallow package metadata only. It must not contain the graph
operation list, kernel attributes, sparse layout rules, or weight layout
semantics.

### 3.1 Executable entry function

The executable graph is a single MLIR function:

```mlir
func.func @forward(...) -> (...) {
  ...
  return ... : ...
}
```

Artifact verifier requirements:

- The module must contain exactly one `func.func`.
- The function must be named `forward`.
- The function must have a body.
- The function body must contain exactly one block.
- The body may contain only `lattice.*` operations followed by `func.return`.
- The function must return at least one value.

Function argument SSA names are textual conveniences and are not part of the
semantic ABI. Importers may expose generated runtime value names, but binding
decisions must come from MLIR-derived type and ABI-role metadata. The MLX
runtime plan currently tags entry arguments consumed by `lattice.sparse.make`
as `sparse_coords`, `sparse_features`, and `sparse_active`; Python
`SparseTensor` shorthand is allowed only for that verified entry shape. Future
named-input support should also be expressed through explicit ABI metadata, not
inferred from MLIR SSA names.

In the MLX runtime, the native binding transport is immediately frozen into a
typed `RuntimePlan`/`PlanOperation` view before lowering. Artifact lowerings
consume that typed view, not arbitrary dictionaries; dictionary-shaped payloads
are an FFI detail rather than a second IR. The freeze step resolves each
qualified operation name through the annotated `lattice-contract` schema and
validates module ABI metadata, SSA operand/result counts, and required
attributes. The native plan must carry `ir_version` and `weight_file` copied
from the verified MLIR module; Python rejects mismatched metadata before any
runtime lowering runs. Symbol operands such as `lattice.weight @name` are
represented in the native transport as attributes, but they remain
schema-declared symbol operands rather than ad-hoc runtime strings. Runtime
value labels are validated as SSA-like dataflow: arguments and operation
results define labels, operands must reference already defined labels, labels
cannot be redefined, and function returns must reference defined labels.

MLX artifact execution is schema-bound. Runtime lowering functions name their
parameters after dialect operands and attributes; `@artifact_lowering` binds
those parameters from the `lattice-contract` op definition. Lowering bodies
should remain small semantic bridges to public `mlx_lattice.ops` functions and
must not manually unpack `operation.operands` or `operation.attrs`. Extra
lowering metadata is reserved for semantic payload resolution that the raw
MLIR type cannot express by itself, such as packed convolution weights that
need the sparse input channel count.

### 3.2 v0 portability boundary

The core v0 dialect is the portable contract shared by Torch/CUDA export and
MLX/Metal import. If an operation is in this boundary, a source framework must
either export it faithfully or fail with a clear unsupported-export error; a
runtime importer must either execute it faithfully or fail with a clear
unsupported-import error.

Current boundary status:

| Semantic family | Contract status | MLX import/runtime | Torch export/runtime |
|---|---|---|---|
| Sparse ABI and dense/packed weights | Core v0 | Implemented | Dense export implemented; packed export pending |
| Convolution, submanifold, transpose, generative | Core v0 | Implemented | Export implemented by module identity |
| Target-coordinate convolution | Core v0 | Implemented | Pending Torch API/export |
| Feature linear, activation, frozen normalization | Core v0 | Implemented | Linear, activation, BatchNorm export implemented; InstanceNorm/GroupNorm rejected |
| Coordinate-aligned sparse algebra | Core v0 | Implemented | Add/sub/mul/min/max and cat export implemented |
| Local sparse pooling | Core v0 | Implemented | Pending Torch runtime/export |
| Global sparse pooling | Core v0 | Implemented | Export implemented |
| Point/voxel conversion | Core v0 | Implemented | Low-level kernels exist; high-level MLIR export pending |
| BEV/crop | Outside core v0 | Framework-local utility | Framework-local utility |

Do not expand the core boundary to BEV/crop until the current v0 families have
cross-side parity fixtures. If BEV/crop becomes required for common deployed
models, add it as a deliberate dialect extension or a v1 core addition rather
than encoding it in shallow package metadata.

---

## 4. Core value model

### 4.1 Sparse value ABI

A sparse value is represented by visible tensor components plus explicit
metadata:

```text
coords        tensor<?x4xi32>
features      tensor<?x?xf16> or tensor<?x?xf32>
active_rows   tensor<1xi32>
stride        static 3D integer attribute
```

The sparse ABI intentionally excludes:

- Python object identity;
- `CoordinateManager`;
- `CoordinateMapKey`;
- relation caches;
- CSR execution views;
- implicit-GEMM maps;
- backend handles.

### 4.2 Sparse tensor type

Draft type:

```mlir
!lattice.sparse_tensor<
  rank = 3,
  coord = batch_x_y_z,
  feature = row_channel,
  dtype = f16
>
```

Fields:

| Field | Meaning |
|---|---|
| `rank` | Spatial rank. First slice supports only `3`. |
| `coord` | Coordinate convention. First slice supports `batch_x_y_z`. |
| `feature` | Feature matrix layout. First slice supports `row_channel`. |
| `dtype` | Feature dtype. First slice supports `f16` and `f32`. |

The sparse type describes portable semantics. It does not encode backend
execution route or memory scheduling.

### 4.3 Sparse construction and decomposition

Sparse values cross the artifact boundary through component ops:

```mlir
%sp = lattice.sparse.make %coords, %features, %active
  {stride = array<i64: 1, 1, 1>,
   coord_order = #lattice.coord<batch_x_y_z>}
  : (tensor<?x4xi32>, tensor<?x?xf16>, tensor<1xi32>)
    -> !lattice.sparse_tensor<rank = 3,
                              coord = batch_x_y_z,
                              feature = row_channel,
                              dtype = f16>
```

```mlir
%coords, %features, %active = lattice.sparse.decompose %sp
  : !lattice.sparse_tensor<rank = 3,
                           coord = batch_x_y_z,
                           feature = row_channel,
                           dtype = f16>
    -> (tensor<?x4xi32>, tensor<?x?xf16>, tensor<1xi32>)
```

Verifier requirements:

- `coords` rank must be 2.
- `coords` second dimension must be 4.
- `coords` dtype must be integer; portable GPU path should prefer `i32`.
- `features` rank must be 2.
- `coords` and `features` capacities must match when statically known.
- `active_rows` must have shape `tensor<1xi32>`.
- `stride` must contain exactly three positive integers.
- `coord_order` must match the sparse type coordinate convention.
- Feature dtype in the result type must match the feature tensor dtype.

---

## 5. Weight model

Weights stay outside MLIR tensor payloads. MLIR references them symbolically and
records their logical contract.

### 5.1 Weight type

Draft type:

```mlir
!lattice.weight<conv3d, f16>
!lattice.weight<linear, f16>
!lattice.weight<bias, f16>
!lattice.weight<channel, f16>
!lattice.weight<conv3d, i4>
!lattice.weight<linear, i8>
```

Fields:

| Field | Meaning |
|---|---|
| first parameter | Logical consumer family, e.g. `conv3d`, `linear`, `channel`, or `bias`. |
| second parameter | Logical storage/compute dtype, including packed `i4`/`i8`. |

Bias parameters use the dedicated `bias` family because they are shared by
multiple consumers but have distinct shape and packing rules.

Channel-vector normalization parameters use the `channel` family with
`#lattice.weight_layout<channel_c>`.

### 5.2 Weight symbol operation

Dense example:

```mlir
%w = lattice.weight @stem.weight
  {storage_key = "stem.weight",
   layout = #lattice.weight_layout<conv3d_o_zyx_i>,
   packing = #lattice.packing<dense>}
  : !lattice.weight<conv3d, f16>
```

Quantized example:

```mlir
%w = lattice.weight @stem.weight
  {storage_key = "stem.weight",
   layout = #lattice.weight_layout<conv3d_o_zyx_i>,
   packing = #lattice.packing<int4,
                              group_size = 32,
                              scale_dtype = f16,
                              mode = affine>}
  : !lattice.weight<conv3d, i4>
```

Verifier requirements:

- `storage_key` must be present.
- `layout` must match the logical consumer family.
- `packing` must describe dense or quantized storage.
- Quantized packing must declare bit width, group size, scale dtype, and mode.
- Weight layout must not be inferred from the symbol name.
- Bias weights must use `layout = #lattice.weight_layout<bias_c>` and dense
  packing.
- Channel weights must use `layout = #lattice.weight_layout<channel_c>` and
  dense packing.
- Quantized payload tensors must be stored under `<storage_key>.weight`,
  `<storage_key>.scales`, and `<storage_key>.biases` in `weights.safetensors`.

### 5.3 Optional bias operands

Bias is represented as an explicit optional operand, not an attribute:

```mlir
%bias = lattice.weight @stem.bias
  {storage_key = "stem.bias",
   layout = #lattice.weight_layout<bias_c>,
   packing = #lattice.packing<dense>}
  : !lattice.weight<bias, f16>

%out = lattice.conv3d %input, %weight, %bias
  {...}
  : (..., !lattice.weight<conv3d, f16>, !lattice.weight<bias, f16>)
    -> !lattice.sparse_tensor<...>
```

The same bias operand convention applies to `lattice.linear` and the sparse
convolution family. Omitting the operand means no bias is applied.

### 5.4 No separate quantized operation family

The dialect should not define:

```text
lattice.quantized_conv3d
lattice.quantized_linear
```

Use the same operation with a quantized weight type:

```text
lattice.conv3d input, !lattice.weight<conv3d, i4>
lattice.linear features, !lattice.weight<linear, i8>
```

Runtime importers can still select specialized quantized kernels.

---

## 6. First operation set

The v0 first slice contains:

```text
lattice.sparse.make
lattice.sparse.decompose
lattice.sparse.with_features
lattice.weight
lattice.conv3d
lattice.subm_conv3d
lattice.target_conv3d
lattice.conv_transpose3d
lattice.generative_conv_transpose3d
lattice.pool3d
lattice.global_pool
lattice.voxelize
lattice.devoxelize
lattice.linear
lattice.activation
lattice.batch_norm
lattice.layer_norm
lattice.rms_norm
lattice.sparse.binary
```

These operations are enough to validate the core architecture:

- sparse component ABI;
- weight binding;
- normal convolution;
- submanifold output-support semantics;
- target-coordinate convolution;
- transposed convolution;
- generative transposed convolution;
- local sparse pooling;
- batch-wise global sparse pooling;
- point-to-sparse-voxel conversion;
- sparse-voxel-to-point interpolation;
- feature-only linear projection, activation, and normalization;
- explicit sparse identity preservation for feature-only updates;
- sparse coordinate alignment.

---

## 7. Convolution operations

Producer rule: framework front-ends must lower explicit source semantics to
explicit lattice operations. A source module named `Conv3d` lowers to
`lattice.conv3d` even when `stride = [1, 1, 1]`; a source module named
`SubmConv3d` lowers to `lattice.subm_conv3d`. Exporters must not infer
submanifold semantics from stride, padding, indice keys, or historical backend
defaults.

### 7.1 `lattice.conv3d`

Forward sparse convolution creates an output sparse support from the input
support and convolution geometry.

Draft syntax:

```mlir
%out = lattice.conv3d %input, %weight
  {kernel_size = array<i64: 3, 3, 3>,
   stride = array<i64: 1, 1, 1>,
   padding = array<i64: 1, 1, 1>,
   dilation = array<i64: 1, 1, 1>}
  : (!lattice.sparse_tensor<...>, !lattice.weight<conv3d, f16>)
    -> !lattice.sparse_tensor<...>
```

Verifier requirements:

- `kernel_size`, `stride`, `padding`, and `dilation` must be integer triples.
- `kernel_size`, `stride`, and `dilation` values must be positive.
- `padding` values must be non-negative.
- Weight consumer family must be `conv3d`.
- Input/output sparse rank must be `3`.

### 7.2 `lattice.subm_conv3d`

Submanifold convolution preserves input sparse support. It is not encoded as
forward convolution plus discarded coordinates.

Draft syntax:

```mlir
%out = lattice.subm_conv3d %input, %weight
  {kernel_size = array<i64: 3, 3, 3>,
   dilation = array<i64: 1, 1, 1>}
  : (!lattice.sparse_tensor<...>, !lattice.weight<conv3d, f16>)
    -> !lattice.sparse_tensor<...>
```

Verifier requirements:

- `kernel_size` and `dilation` must be integer triples.
- `kernel_size` values must be positive and odd.
- `dilation` values must be positive.
- `stride` and `padding` attributes are invalid.
- Output coordinate support must be semantically identical to input support.

### 7.3 `lattice.target_conv3d`

Target-coordinate convolution produces output features on an explicit target
sparse support.

Draft syntax:

```mlir
%out = lattice.target_conv3d %input, %target, %weight
  {kernel_size = array<i64: 3, 3, 3>,
   stride = array<i64: 1, 1, 1>,
   padding = array<i64: 1, 1, 1>,
   dilation = array<i64: 1, 1, 1>}
  : (!lattice.sparse_tensor<...>,
     !lattice.sparse_tensor<...>,
     !lattice.weight<conv3d, f16>)
    -> !lattice.sparse_tensor<...>
```

Verifier requirements:

- Input and target sparse tensors must have matching coordinate convention.
- Target sparse tensor defines output coordinate support.
- Output sparse support is semantically identical to target support.
- Weight consumer family must be `conv3d`.
- Geometry attributes follow `lattice.conv3d` rules.

### 7.4 `lattice.conv_transpose3d`

Sparse transpose convolution uses the same sparse value and weight contract as
forward convolution, but its support rule is the transpose relation.

Draft syntax:

```mlir
%out = lattice.conv_transpose3d %input, %weight
  {kernel_size = [2, 2, 2],
   stride = [2, 2, 2],
   padding = [0, 0, 0],
   dilation = [1, 1, 1]}
  : (!lattice.sparse_tensor<...>, !lattice.weight<conv3d, f16>)
    -> !lattice.sparse_tensor<...>
```

Verifier requirements:

- Weight consumer family must be `conv3d`.
- Input/output sparse rank must be `3`.
- Geometry attributes follow `lattice.conv3d` rules.

### 7.5 `lattice.generative_conv_transpose3d`

Generative transpose convolution creates output support directly from input
rows and transpose stride. It is distinct from regular transpose convolution
because padding and dilation are not part of its semantic contract.

Draft syntax:

```mlir
%out = lattice.generative_conv_transpose3d %input, %weight
  {kernel_size = [2, 2, 2],
   stride = [2, 2, 2]}
  : (!lattice.sparse_tensor<...>, !lattice.weight<conv3d, f16>)
    -> !lattice.sparse_tensor<...>
```

Verifier requirements:

- `kernel_size` and `stride` must be integer triples with positive values.
- `padding` and `dilation` attributes are invalid.
- Weight consumer family must be `conv3d`.
- Input/output sparse rank must be `3`.

### 7.6 `lattice.pool3d`

Local sparse pooling builds output sparse support from input support and kernel
geometry, then reduces contributing feature rows by mode.

Draft syntax:

```mlir
%out = lattice.pool3d %input
  {mode = #lattice.pool_mode<avg>,
   kernel_size = [2, 2, 2],
   stride = [2, 2, 2],
   padding = [0, 0, 0],
   dilation = [1, 1, 1]}
  : (!lattice.sparse_tensor<...>) -> !lattice.sparse_tensor<...>
```

Verifier requirements:

- `mode` must be `sum`, `max`, or `avg`.
- Geometry attributes follow `lattice.conv3d` rules.
- Input/output sparse rank must be `3`.

### 7.7 `lattice.global_pool`

Global sparse pooling reduces active feature rows independently for each batch
and returns a dense `(B, C)` tensor.

Draft syntax:

```mlir
%out = lattice.global_pool %input
  {mode = #lattice.pool_mode<sum>,
   batch_size = -1}
  : (!lattice.sparse_tensor<...>) -> tensor<?x?xf32>
```

Verifier requirements:

- `mode` must be `sum`, `max`, or `avg`.
- `batch_size = -1` means infer from sparse metadata/coordinates.
- Non-negative `batch_size` values request an explicit dense batch dimension.
- Result type must be rank-2 and match the sparse feature dtype.

### 7.8 `lattice.voxelize`

Voxelization converts continuous point rows and point features into a sparse
voxel tensor. Optional Python runtime defaults are not part of the ABI:
exporters must pass explicit `batch_indices` and `active_rows` tensors.

Draft syntax:

```mlir
%voxels = lattice.voxelize %points, %features, %batch_indices, %active_rows
  {voxel_size = array<f64: 0.1, 0.1, 0.1>,
   origin = array<f64: 0.0, 0.0, 0.0>,
   reduction = #lattice.voxel_reduction<mean>,
   stride = array<i64: 1, 1, 1>}
  : (tensor<?x3xf32>, tensor<?x?xf32>, tensor<?xi32>, tensor<1xi32>)
    -> !lattice.sparse_tensor<rank = 3,
                              coord = batch_x_y_z,
                              feature = row_channel,
                              dtype = f32>
```

Verifier requirements:

- `points` must be rank-2 `f32` with trailing dimension `3`.
- `features` must be rank-2 `f32`.
- Static point/feature row counts must match when known.
- `batch_indices` must be a rank-1 `i32` tensor.
- `active_rows` must be `tensor<1xi32>`.
- `voxel_size` must contain exactly three positive values.
- `origin` must contain exactly three values.
- `reduction` must be `sum` or `mean`.
- `stride` must contain exactly three positive integers.
- Result sparse dtype must match feature dtype.

### 7.9 `lattice.devoxelize`

Devoxelization samples sparse voxel features back to dense point rows. The
artifact passes explicit point batch and active-row tensors.

Draft syntax:

```mlir
%features = lattice.devoxelize
  %points, %voxels, %batch_indices, %point_active_rows
  {voxel_size = [0.1, 0.1, 0.1],
   origin = [0.0, 0.0, 0.0],
   interpolation = #lattice.point_interpolation<linear>}
  : (tensor<?x3xf32>, !lattice.sparse_tensor<...>, tensor<?xi32>,
     tensor<1xi32>) -> tensor<?x?xf32>
```

Verifier requirements:

- `points` must be rank-2 `f32` with trailing dimension `3`.
- `batch_indices` must be a rank-1 `i32` tensor.
- `point_active_rows` must be `tensor<1xi32>`.
- `voxel_size` must contain exactly three positive values.
- `origin` must contain exactly three values.
- `interpolation` must be `nearest` or `linear`.
- Result type must be rank-2 and match the sparse voxel feature dtype.

---

## 8. Feature-only operations

Feature-only operations operate on dense rank-2 feature tensors. Sparse
identity preservation is represented explicitly by `lattice.sparse.decompose`
and `lattice.sparse.with_features`; `lattice.linear` and
`lattice.activation`, `lattice.batch_norm`, `lattice.layer_norm`, and
`lattice.rms_norm` do not directly consume sparse values.

### 8.1 `lattice.linear`

`lattice.linear` operates on a dense rank-2 feature matrix. It does not
directly consume a sparse tensor in v0.

Draft syntax:

```mlir
%features2 = lattice.linear %features, %weight
  : (tensor<?x32xf16>, !lattice.weight<linear, f16>)
    -> tensor<?x64xf16>
```

Verifier requirements:

- Weight consumer family must be `linear`.
- Input and result tensors must have rank 2.
- Result dtype must match input dtype.
- Input trailing channel dimension must match weight input channels when
  statically known.
- Result trailing channel dimension must match weight output channels when
  statically known.

### 8.2 `lattice.activation`

`lattice.activation` applies a dense feature activation to a rank-2 feature
matrix.

Syntax:

```mlir
%features2 = lattice.activation %features
  {kind = #lattice.activation<gelu>,
   approximate = #lattice.gelu_approx<tanh>,
   alpha = 0.01 : f32,
   beta = 1.0 : f32,
   threshold = 20.0 : f32}
  : (tensor<?x32xf16>) -> tensor<?x32xf16>
```

Attribute semantics:

| Attribute | Meaning |
|---|---|
| `kind` | Activation function: `relu`, `sigmoid`, `gelu`, `silu`, `leaky_relu`, `tanh`, or `softplus`. |
| `approximate` | GELU approximation: `none`, `precise`, `tanh`, or `fast`; ignored by non-GELU activations. |
| `alpha` | Leaky-ReLU negative slope; ignored by other activations. |
| `beta` | Softplus beta; ignored by other activations. |
| `threshold` | Softplus threshold; ignored by other activations. |

Verifier requirements:

- Input and result tensors must have rank 2.
- Result dtype must match input dtype.
- `kind` and `approximate` must be one of the stable v0 enum values.
- `alpha` must be non-negative.
- `beta` must be positive.

### 8.3 Normalization operations

Normalization ops are dense rank-2 feature operations. BatchNorm uses explicit
frozen statistics; training-state updates and running-stat mutation are not
part of the artifact ABI.

```mlir
%features2 = lattice.batch_norm %features, %scale, %bias, %mean, %var
  {eps = 0.00001 : f32}
  : (tensor<?x32xf16>,
     !lattice.weight<channel, f16>,
     !lattice.weight<bias, f16>,
     !lattice.weight<channel, f16>,
     !lattice.weight<channel, f16>)
    -> tensor<?x32xf16>

%features3 = lattice.layer_norm %features2, %scale, %bias
  {eps = 0.00001 : f32}
  : (tensor<?x32xf16>,
     !lattice.weight<channel, f16>,
     !lattice.weight<bias, f16>)
    -> tensor<?x32xf16>

%features4 = lattice.rms_norm %features3, %scale
  {eps = 0.00001 : f32}
  : (tensor<?x32xf16>, !lattice.weight<channel, f16>)
    -> tensor<?x32xf16>
```

Verifier requirements:

- Input and result tensors must have rank 2.
- Result dtype must match input dtype.
- `eps` must be positive.
- Scale, mean, and variance operands must use `!lattice.weight<channel, *>`
  with `#lattice.weight_layout<channel_c>`.
- Bias operands must use `!lattice.weight<bias, *>` with
  `#lattice.weight_layout<bias_c>`.
- Channel and bias normalization weights must use dense packing.

Exporters should materialize identity scale and zero bias vectors when the
source module omits affine parameters. This keeps the artifact ABI positional,
unambiguous, and easy to verify.

### 8.4 `lattice.sparse.with_features`

This operation preserves sparse coordinate identity and replaces only the
feature matrix.

Draft syntax:

```mlir
%out = lattice.sparse.with_features %input, %features2
  : (!lattice.sparse_tensor<rank = 3,
                            coord = batch_x_y_z,
                            feature = row_channel,
                            dtype = f16>,
     tensor<?x64xf16>)
    -> !lattice.sparse_tensor<rank = 3,
                              coord = batch_x_y_z,
                              feature = row_channel,
                              dtype = f16>
```

Verifier requirements:

- New feature tensor rank must be 2.
- New feature capacity must match sparse capacity when statically known.
- Output sparse coordinate support is semantically identical to input support.
- Output feature dtype must match the replacement feature tensor dtype.

---

## 9. Sparse algebra operation

### 9.1 `lattice.sparse.binary`

Sparse binary algebra aligns coordinates and applies an elementwise feature
operation.

Draft syntax:

```mlir
%out = lattice.sparse.binary %lhs, %rhs
  {op = #lattice.binary_op<add>,
   join = #lattice.join<outer>,
   lhs_fill = 0.0 : f32,
   rhs_fill = 0.0 : f32}
  : (!lattice.sparse_tensor<...>, !lattice.sparse_tensor<...>)
    -> !lattice.sparse_tensor<...>
```

Verifier requirements:

- `op` must be one of `add`, `sub`, `mul`, `maximum`, or `minimum`.
- `join` must be one of `inner`, `left`, `right`, or `outer`.
- Input sparse tensors must have matching rank and coordinate convention.
- Input sparse tensors must have matching stride.
- Feature channel counts must match when statically known.
- Fill values must be representable in the feature dtype.
- Output sparse support follows the join mode.

### 9.2 `lattice.sparse.cat`

Sparse concatenation aligns coordinate support and concatenates feature
channels. It is the portable branch-merge form for skip connections that
preserve sparse values but increase channel count.

Draft syntax:

```mlir
%out = lattice.sparse.cat %lhs, %rhs
  {join = #lattice.join<inner>}
  : (!lattice.sparse_tensor<...>, !lattice.sparse_tensor<...>)
    -> !lattice.sparse_tensor<...>
```

Verifier requirements:

- `join` must be one of `inner`, `left`, `right`, or `outer`.
- Input sparse tensors must have matching rank, coordinate convention, feature
  layout, and feature dtype.
- Output sparse conventions and dtype must match the operands.
- Output sparse support follows the join mode.
- Output feature channel count is the sum of operand channel counts when
  statically known.

---

## 10. Attribute vocabulary

Initial attributes:

```text
#lattice.coord<batch_x_y_z>
#lattice.feature_layout<row_channel>
#lattice.weight_layout<conv3d_o_zyx_i>
#lattice.weight_layout<linear_o_i>
#lattice.packing<dense>
#lattice.packing<int4, group_size = 32, scale_dtype = f16, mode = affine>
#lattice.packing<int8, group_size = 64, scale_dtype = f16, mode = affine>
#lattice.binary_op<add>
#lattice.binary_op<sub>
#lattice.binary_op<mul>
#lattice.binary_op<maximum>
#lattice.binary_op<minimum>
#lattice.join<inner>
#lattice.join<left>
#lattice.join<right>
#lattice.join<outer>
```

The first slice should keep the enum set intentionally small.

---

## 11. Module-level metadata

Required module attributes:

```mlir
module attributes {
  lattice.ir_version = 0,
  lattice.weight_file = "weights.safetensors"
} {
  ...
}
```

The artifact verifier rejects modules that omit these attributes, use an
unsupported `lattice.ir_version`, or point at any weight payload other than
`weights.safetensors`. This check runs before runtime import and before
operation lowering. It also rejects modules with no executable entry function,
more than one function, an entry function not named `forward`, a multi-block
entry body, non-lattice executable operations, or a `forward` function that
returns no values.

Allowed:

- dialect IR version;
- weight file name;
- optional producer/debug metadata later, if it does not affect semantics.

Not allowed:

- backend route names;
- Metal kernel names;
- CUDA kernel names;
- operation semantics duplicated in JSON-style metadata.

---

## 12. Valid example expectations

The valid examples under `references/mlir_examples/` mirror the
parser-validated fixtures under `mlir/test/Dialect/Lattice/valid/` and should
be treated as golden contract examples.

Each valid example should satisfy:

- no runtime object identity is serialized;
- sparse values are built from ABI components;
- weights are symbolic and point to `safetensors` storage;
- operations are semantic, not backend-specific;
- quantization, when present, lives in the weight contract;
- target support, when present, is explicit.

---

## 13. Invalid example expectations

The invalid examples under `references/mlir_examples/invalid/` define verifier
behavior that should eventually become tests.

Initial invalid cases:

```text
subm_with_stride.mlir
  Submanifold convolution cannot carry stride/padding attributes.

sparse_make_bad_coords.mlir
  Sparse coordinates must have shape (?, 4), not (?, 3).

conv_bad_weight_layout.mlir
  Conv3d cannot consume a linear-layout weight.

quantized_conv_op_rejected.mlir
  Quantized convolution must be represented through a quantized weight, not a
  duplicated quantized op.
```

---

## 14. Implementation rule

The active implementation starts from this dialect contract rather than a JSON
manifest builder:

```text
dialect definitions
  -> verifier
  -> golden MLIR fixtures
  -> generated/shared Python metadata if useful
  -> MLX importer
  -> Torch/CUDA exporter
```

This ordering keeps the MLIR dialect and verifier as the source of truth. JSON
package metadata, if added later, must remain shallow and must not carry an
operation graph, kernel attributes, weight layout rules, or sparse semantics.
