# lattice-contract

`lattice-contract` is the backend-neutral artifact contract package used by
`mlx-lattice`.

It contains the sparse model manifest dataclasses, validation helpers, value
type names, operation-contract annotations, and the canonical built-in semantic
op registry. Built-ins are exported as contract objects such as
`SPARSE_CONV3D`, `FEATURE_LINEAR`, and `POOL_GLOBAL_AVG`; backend packages bind
implementations to those objects instead of independently maintaining op-name
and schema strings.

It intentionally does not import MLX, Torch, native extensions, or backend
runtime objects.

This package exists so training-side and deployment-side tools can share the
same sparse graph contract while keeping their execution stacks independent.
