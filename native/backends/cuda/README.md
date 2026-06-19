# CUDA backend notes

This directory contains CUDA kernel algorithm sources. They are compiled to PTX
artifacts when `MLX_LATTICE_BUILD_CUDA=ON`.

The supported MLX CUDA extension surface used by mlx-lattice is
`mx.fast.precompiled_cuda_kernel`, which launches PTX/cubin through MLX's public
runtime. The old host bridge that reached into MLX CUDA C++ internals was
removed because PyPI wheels do not export that private ABI and the resulting
`_ext` module could link but fail to import on Linux.

Keep launch, stream, and output ownership inside MLX's public precompiled-kernel
API. Do not reintroduce `mlx/backend/cuda/*` includes.
