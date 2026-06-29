# lattice-contract

`lattice-contract` is the backend-neutral artifact contract package used by
`mlx-lattice`.

It contains the sparse model manifest dataclasses, validation helpers, value
type names, and operation-contract annotations. It intentionally does not
import MLX, Torch, native extensions, or backend runtime objects.

This package exists so training-side and deployment-side tools can share the
same sparse graph contract while keeping their execution stacks independent.
