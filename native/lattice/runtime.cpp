#include "lattice/runtime.h"

namespace mlx_lattice {

std::string_view version() noexcept {
    return MLX_LATTICE_VERSION;
}

Capabilities capabilities() noexcept {
    return Capabilities{
        MLX_LATTICE_HAS_CPU,
        MLX_LATTICE_HAS_METAL,
        MLX_LATTICE_HAS_CUDA,
        MLX_LATTICE_HAS_ROCM,
    };
}

} // namespace mlx_lattice
