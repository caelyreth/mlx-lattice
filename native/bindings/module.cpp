#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "lattice/runtime.h"
#include "mlx/ops.h"
#include "ops/coords.h"

namespace nb = nanobind;
using namespace nb::literals;

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

mlx_lattice::mx::array
array_from_rows(const std::vector<std::vector<int64_t>>& rows, int dtype_code) {
    if (dtype_code == 32) {
        std::vector<int32_t> data;
        data.reserve(rows.size() * 4);
        for (const auto& row : rows) {
            for (auto item : row) {
                data.push_back(static_cast<int32_t>(item));
            }
        }
        return mlx_lattice::mx::array(
            data.begin(),
            mlx_lattice::mx::Shape{int(rows.size()), 4},
            mlx_lattice::mx::int32
        );
    }

    std::vector<int64_t> data;
    data.reserve(rows.size() * 4);
    for (const auto& row : rows) {
        data.insert(data.end(), row.begin(), row.end());
    }
    return mlx_lattice::mx::array(
        data.begin(),
        mlx_lattice::mx::Shape{int(rows.size()), 4},
        mlx_lattice::mx::int64
    );
}

nb::list coord_list(const mlx_lattice::mx::array& values) {
    auto cpu_values = mlx_lattice::mx::contiguous(
        values, false, mlx_lattice::mx::Device::cpu
    );
    cpu_values.eval();
    cpu_values.wait();

    nb::list out;
    if (cpu_values.dtype() == mlx_lattice::mx::int32) {
        auto data = cpu_values.data<int32_t>();
        for (int row = 0; row < cpu_values.shape(0); ++row) {
            nb::list coord;
            auto base = static_cast<ptrdiff_t>(row) * 4;
            coord.append(data[base]);
            coord.append(data[base + 1]);
            coord.append(data[base + 2]);
            coord.append(data[base + 3]);
            out.append(coord);
        }
        return out;
    }

    auto data = cpu_values.data<int64_t>();
    for (int row = 0; row < cpu_values.shape(0); ++row) {
        nb::list coord;
        auto base = static_cast<ptrdiff_t>(row) * 4;
        coord.append(data[base]);
        coord.append(data[base + 1]);
        coord.append(data[base + 2]);
        coord.append(data[base + 3]);
        out.append(coord);
    }
    return out;
}

nb::list i32_list(const mlx_lattice::mx::array& values) {
    auto cpu_values = mlx_lattice::mx::contiguous(
        values, false, mlx_lattice::mx::Device::cpu
    );
    cpu_values.eval();
    cpu_values.wait();

    nb::list out;
    auto data = cpu_values.data<int32_t>();
    for (int row = 0; row < cpu_values.shape(0); ++row) {
        out.append(data[row]);
    }
    return out;
}

nb::list offset_list(const mlx_lattice::mx::array& values) {
    auto cpu_values = mlx_lattice::mx::contiguous(
        values, false, mlx_lattice::mx::Device::cpu
    );
    cpu_values.eval();
    cpu_values.wait();

    nb::list out;
    auto data = cpu_values.data<int32_t>();
    for (int row = 0; row < cpu_values.shape(0); ++row) {
        nb::list offset;
        auto base = static_cast<ptrdiff_t>(row) * 3;
        offset.append(data[base]);
        offset.append(data[base + 1]);
        offset.append(data[base + 2]);
        out.append(offset);
    }
    return out;
}

nb::tuple map_tuple(const mlx_lattice::NativeKernelMap& map) {
    return nb::make_tuple(
        i32_list(map.in_rows),
        i32_list(map.out_rows),
        i32_list(map.kernel_ids),
        coord_list(map.out_coords),
        offset_list(map.kernel_offsets)
    );
}

} // namespace

NB_MODULE(_ext, m) {
    m.doc() = "Native extension for mlx-lattice.";

    m.def("version", &version, "Return the native mlx-lattice version.");
    m.def(
        "capabilities",
        &capabilities,
        "Return compiled native backend capabilities."
    );
    m.def(
        "downsample_coords",
        [](const std::vector<std::vector<int64_t>>& coords,
           int dtype_code,
           int sx,
           int sy,
           int sz) {
            return coord_list(
                mlx_lattice::downsample_coords(
                    array_from_rows(coords, dtype_code), {sx, sy, sz}
                )
            );
        },
        "coords"_a,
        "dtype_code"_a,
        "sx"_a,
        "sy"_a,
        "sz"_a
    );
    m.def(
        "union_coords",
        [](const std::vector<std::vector<int64_t>>& lhs,
           const std::vector<std::vector<int64_t>>& rhs,
           int dtype_code) {
            return coord_list(
                mlx_lattice::union_coords(
                    array_from_rows(lhs, dtype_code),
                    array_from_rows(rhs, dtype_code)
                )
            );
        },
        "lhs"_a,
        "rhs"_a,
        "dtype_code"_a
    );
    m.def(
        "intersection_coords",
        [](const std::vector<std::vector<int64_t>>& lhs,
           const std::vector<std::vector<int64_t>>& rhs,
           int dtype_code) {
            return coord_list(
                mlx_lattice::intersection_coords(
                    array_from_rows(lhs, dtype_code),
                    array_from_rows(rhs, dtype_code)
                )
            );
        },
        "lhs"_a,
        "rhs"_a,
        "dtype_code"_a
    );
    m.def(
        "lookup_coords",
        [](const std::vector<std::vector<int64_t>>& coords,
           const std::vector<std::vector<int64_t>>& queries,
           int dtype_code) {
            return i32_list(
                mlx_lattice::lookup_coords(
                    array_from_rows(coords, dtype_code),
                    array_from_rows(queries, dtype_code)
                )
            );
        },
        "coords"_a,
        "queries"_a,
        "dtype_code"_a
    );
    m.def(
        "build_kernel_map",
        [](const std::vector<std::vector<int64_t>>& coords,
           int dtype_code,
           int kx,
           int ky,
           int kz,
           int sx,
           int sy,
           int sz,
           int px,
           int py,
           int pz,
           int dx,
           int dy,
           int dz) {
            return map_tuple(
                mlx_lattice::build_kernel_map(
                    array_from_rows(coords, dtype_code),
                    {kx, ky, kz},
                    {sx, sy, sz},
                    {px, py, pz},
                    {dx, dy, dz}
                )
            );
        },
        "coords"_a,
        "dtype_code"_a,
        "kx"_a,
        "ky"_a,
        "kz"_a,
        "sx"_a,
        "sy"_a,
        "sz"_a,
        "px"_a,
        "py"_a,
        "pz"_a,
        "dx"_a,
        "dy"_a,
        "dz"_a
    );
    m.def(
        "build_generative_map",
        [](const std::vector<std::vector<int64_t>>& coords,
           int dtype_code,
           int kx,
           int ky,
           int kz,
           int sx,
           int sy,
           int sz) {
            return map_tuple(
                mlx_lattice::build_generative_map(
                    array_from_rows(coords, dtype_code),
                    {kx, ky, kz},
                    {sx, sy, sz}
                )
            );
        },
        "coords"_a,
        "dtype_code"_a,
        "kx"_a,
        "ky"_a,
        "kz"_a,
        "sx"_a,
        "sy"_a,
        "sz"_a
    );
    m.def(
        "build_transposed_kernel_map",
        [](const std::vector<std::vector<int64_t>>& coords,
           int dtype_code,
           int kx,
           int ky,
           int kz,
           int sx,
           int sy,
           int sz,
           int px,
           int py,
           int pz,
           int dx,
           int dy,
           int dz) {
            return map_tuple(
                mlx_lattice::build_transposed_kernel_map(
                    array_from_rows(coords, dtype_code),
                    {kx, ky, kz},
                    {sx, sy, sz},
                    {px, py, pz},
                    {dx, dy, dz}
                )
            );
        },
        "coords"_a,
        "dtype_code"_a,
        "kx"_a,
        "ky"_a,
        "kz"_a,
        "sx"_a,
        "sy"_a,
        "sz"_a,
        "px"_a,
        "py"_a,
        "pz"_a,
        "dx"_a,
        "dy"_a,
        "dz"_a
    );
}
