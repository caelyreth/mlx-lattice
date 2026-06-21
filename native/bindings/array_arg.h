#pragma once

#include <nanobind/nanobind.h>

#include <stdexcept>
#include <string>

#include "mlx/array.h"

namespace mlx_lattice::bindings {

namespace nb = nanobind;
namespace mx = mlx::core;

inline const mx::array& array_arg(nb::handle value, const char* name) {
    auto array_type = nb::module_::import_("mlx.core").attr("array");
    if (PyObject_IsInstance(value.ptr(), array_type.ptr()) != 1) {
        throw std::invalid_argument(
            std::string(name) + " must be an mlx.core.array."
        );
    }
    auto* array = nb::inst_ptr<mx::array>(value);
    if (array == nullptr) {
        throw std::invalid_argument(
            std::string(name) + " must be an mlx.core.array."
        );
    }
    return *array;
}

} // namespace mlx_lattice::bindings
