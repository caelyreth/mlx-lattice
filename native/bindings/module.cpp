#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>

#include <string>

#include "lattice/runtime.h"

namespace nb = nanobind;

namespace {

nb::dict capabilities() {
    auto caps = mlx_lattice::capabilities();
    nb::dict out;
    out["cpu"] = caps.cpu;
    out["metal"] = caps.metal;
    out["cuda"] = caps.cuda;
    out["rocm"] = caps.rocm;
    return out;
}

std::string version() { return std::string(mlx_lattice::version()); }

} // namespace

NB_MODULE(_ext, m) {
    m.doc() = "Native extension for mlx-lattice.";

    m.def("version", &version, "Return the native mlx-lattice version.");
    m.def(
        "capabilities",
        &capabilities,
        "Return compiled native backend capabilities."
    );
}
