#include "backends/cpu/pool/planning.h"

#include <algorithm>
#include <cstddef>
#include <unordered_map>
#include <unordered_set>

namespace mlx_lattice::backend::cpu::pool {
namespace {

struct CoordHash {
    size_t operator()(const Coord& coord) const {
        size_t seed = 0;
        for (auto value : coord) {
            auto part = std::hash<int64_t>{}(value);
            seed ^= part + 0x9e3779b97f4a7c15ULL + (seed << 6) + (seed >> 2);
        }
        return seed;
    }
};

int64_t floor_div(int64_t value, int64_t divisor) {
    auto quotient = value / divisor;
    auto remainder = value % divisor;
    if (remainder != 0 && ((remainder < 0) != (divisor < 0))) {
        --quotient;
    }
    return quotient;
}

std::vector<Coord> read_coords(const mx::array& coords, int active_rows) {
    std::vector<Coord> out;
    auto rows = std::min(active_rows, int(coords.shape(0)));
    out.reserve(rows);
    if (coords.dtype() == mx::int32) {
        const auto* data = coords.data<int32_t>();
        for (int row = 0; row < rows; ++row) {
            auto base = static_cast<std::ptrdiff_t>(row) * coords.strides(0);
            out.push_back({
                data[base],
                data[base + coords.strides(1)],
                data[base + 2 * coords.strides(1)],
                data[base + 3 * coords.strides(1)],
            });
        }
        return out;
    }

    const auto* data = coords.data<int64_t>();
    for (int row = 0; row < rows; ++row) {
        auto base = static_cast<std::ptrdiff_t>(row) * coords.strides(0);
        out.push_back({
            data[base],
            data[base + coords.strides(1)],
            data[base + 2 * coords.strides(1)],
            data[base + 3 * coords.strides(1)],
        });
    }
    return out;
}

std::vector<Triple> read_offsets(const mx::array& offsets) {
    std::vector<Triple> out;
    out.reserve(offsets.shape(0));
    const auto* data = offsets.data<int32_t>();
    for (int row = 0; row < offsets.shape(0); ++row) {
        auto base = static_cast<std::ptrdiff_t>(row) * offsets.strides(0);
        out.push_back({
            data[base],
            data[base + offsets.strides(1)],
            data[base + 2 * offsets.strides(1)],
        });
    }
    return out;
}

std::vector<Coord>
output_coords(const std::vector<Coord>& coords, Triple stride, Triple padding) {
    if (stride == Triple{1, 1, 1} && padding == Triple{0, 0, 0}) {
        return coords;
    }

    std::vector<Coord> out;
    out.reserve(coords.size());
    std::unordered_set<Coord, CoordHash> seen;
    seen.reserve(coords.size());
    for (auto coord : coords) {
        Coord quantized = {
            coord[0],
            floor_div(coord[1], stride[0]),
            floor_div(coord[2], stride[1]),
            floor_div(coord[3], stride[2]),
        };
        if (seen.insert(quantized).second) {
            out.push_back(quantized);
        }
    }
    return out;
}

} // namespace

// NOLINTBEGIN(bugprone-easily-swappable-parameters)
Plan build_plan(
    const mx::array& coords,
    const mx::array& active_rows,
    const mx::array& offsets,
    Triple stride,
    Triple padding
) {
    auto values = read_coords(coords, active_rows.data<int32_t>()[0]);
    auto kernel_offsets = read_offsets(offsets);
    std::unordered_map<Coord, int32_t, CoordHash> input_rows;
    input_rows.reserve(values.size());
    for (int row = 0; row < int(values.size()); ++row) {
        input_rows.emplace(values[row], static_cast<int32_t>(row));
    }

    Plan plan;
    plan.out_coords = output_coords(values, stride, padding);
    plan.edges.reserve(plan.out_coords.size() * kernel_offsets.size());
    for (int kernel = 0; kernel < int(kernel_offsets.size()); ++kernel) {
        auto offset = kernel_offsets[kernel];
        for (int out_row = 0; out_row < int(plan.out_coords.size());
             ++out_row) {
            auto coord = plan.out_coords[out_row];
            Coord candidate = {
                coord[0],
                coord[1] * stride[0] + offset[0] - padding[0],
                coord[2] * stride[1] + offset[1] - padding[1],
                coord[3] * stride[2] + offset[2] - padding[2],
            };
            auto match = input_rows.find(candidate);
            if (match != input_rows.end()) {
                plan.edges.push_back({
                    match->second,
                    static_cast<int32_t>(out_row),
                    static_cast<int32_t>(kernel),
                });
            }
        }
    }
    return plan;
}
// NOLINTEND(bugprone-easily-swappable-parameters)

void write_coords(mx::array& out, const std::vector<Coord>& coords) {
    if (out.dtype() == mx::int32) {
        auto* data = out.data<int32_t>();
        std::fill(data, data + out.size(), 0);
        for (int row = 0; row < int(coords.size()); ++row) {
            for (int axis = 0; axis < 4; ++axis) {
                data[static_cast<std::ptrdiff_t>(row) * 4 + axis] =
                    static_cast<int32_t>(coords[row][axis]);
            }
        }
        return;
    }

    auto* data = out.data<int64_t>();
    std::fill(data, data + out.size(), 0);
    for (int row = 0; row < int(coords.size()); ++row) {
        for (int axis = 0; axis < 4; ++axis) {
            data[static_cast<std::ptrdiff_t>(row) * 4 + axis] =
                coords[row][axis];
        }
    }
}

void write_counts(mx::array& out, const Plan& plan) {
    auto* data = out.data<int32_t>();
    std::fill(data, data + out.size(), 0);
    data[0] = int(plan.edges.size());
    data[1] = int(plan.out_coords.size());
}

std::vector<int32_t> degrees(const Plan& plan, int out_capacity) {
    std::vector<int32_t> out(out_capacity, 0);
    for (auto edge : plan.edges) {
        ++out[edge[1]];
    }
    return out;
}

} // namespace mlx_lattice::backend::cpu::pool
