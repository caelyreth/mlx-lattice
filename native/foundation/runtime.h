#pragma once

#include <string_view>

namespace mlx_lattice {

struct Capabilities {
    bool cpu;
    bool metal;
    bool mlir;
};

std::string_view version() noexcept;
Capabilities capabilities() noexcept;

} // namespace mlx_lattice
