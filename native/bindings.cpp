#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/variant.h>

#include "lattice/runtime.h"
#include "ops/conv3d.h"

namespace nb = nanobind;
using namespace nb::literals;

NB_MODULE(_ext, m) {
    m.doc() = "Native extension for mlx-lattice.";

    m.def("version", &mlx_lattice::version);
    m.def("capabilities", []() {
        auto caps = mlx_lattice::capabilities();
        nb::dict out;
        out["cpu"] = caps.cpu;
        out["metal"] = caps.metal;
        out["cuda"] = caps.cuda;
        out["rocm"] = caps.rocm;
        return out;
    });
    m.def(
        "conv3d_feats",
        &mlx_lattice::conv3d_feats,
        "feats"_a,
        "weight"_a,
        "maps"_a,
        "kernels"_a,
        "out_rows"_a,
        nb::kw_only(),
        "stream"_a = nb::none()
    );
}
