#pragma once

#include "ops/coords.h"

namespace mlx_lattice {

void validate_coords(const mx::array& coords);
void validate_coord_pair(const mx::array& lhs, const mx::array& rhs);
void validate_positive(Triple values, const char* name);
void validate_nonnegative(Triple values, const char* name);

} // namespace mlx_lattice
