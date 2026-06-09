#pragma once

#include "mlx/primitives.h"
#include "ops/exec/types.h"

namespace mlx_lattice {

class SparsePrimitive : public mx::Primitive {
  public:
    using mx::Primitive::Primitive;
};

} // namespace mlx_lattice
