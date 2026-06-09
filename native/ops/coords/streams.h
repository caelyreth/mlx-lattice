#pragma once

#include "mlx/device.h"
#include "mlx/stream.h"
#include "ops/coords/types.h"

namespace mlx_lattice {

mx::Device coord_device();
mx::Stream coord_stream(const mx::Device& device);

} // namespace mlx_lattice
