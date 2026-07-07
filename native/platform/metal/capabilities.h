#pragma once

#include <cstdint>

#include "mlx/stream.h"

namespace mlx_lattice::backend::metal::tensor_ops {

enum class CapabilityTier : std::uint8_t {
    unavailable,
    gpu,
    neural_accelerator,
};

CapabilityTier capability_tier(const mlx::core::Stream& stream);
bool has_neural_acceleration(const mlx::core::Stream& stream);

} // namespace mlx_lattice::backend::metal::tensor_ops
