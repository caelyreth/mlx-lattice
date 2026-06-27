// MARK: - coords

int64_t floor_div(int64_t value, int64_t divisor) {
    auto quotient = value / divisor;
    auto remainder = value % divisor;
    if (remainder != 0 && ((remainder < 0) != (divisor < 0))) {
        --quotient;
    }
    return quotient;
}

std::vector<Coord>
downsample_values(const std::vector<Coord>& coords, Triple stride) {
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

uint64_t split_morton_3(uint64_t value) {
    value &= 0x1fffffULL;
    value = (value | (value << 32)) & 0x1f00000000ffffULL;
    value = (value | (value << 16)) & 0x1f0000ff0000ffULL;
    value = (value | (value << 8)) & 0x100f00f00f00f00fULL;
    value = (value | (value << 4)) & 0x10c30c30c30c30c3ULL;
    value = (value | (value << 2)) & 0x1249249249249249ULL;
    return value;
}

int64_t morton_code(Coord coord) {
    auto code = split_morton_3(static_cast<uint64_t>(coord[1])) |
                (split_morton_3(static_cast<uint64_t>(coord[2])) << 1) |
                (split_morton_3(static_cast<uint64_t>(coord[3])) << 2);
    code += static_cast<uint64_t>(coord[0]) << 60;
    return static_cast<int64_t>(code);
}

std::vector<int64_t> morton_code_values(const mx::array& coords) {
    auto values = read_coords(coords);
    std::vector<int64_t> out;
    out.reserve(values.size());
    for (auto coord : values) {
        out.push_back(morton_code(coord));
    }
    return out;
}

int32_t child_index_for_coord(const Coord& coord) {
    auto index = int32_t(coord[1] & 1) + int32_t((coord[2] & 1) << 1) +
                 int32_t((coord[3] & 1) << 2);
    return int32_t(1 << index);
}

Coord voxel_coord_for_point(
    const float* points,
    const int32_t* batch_indices,
    int row,
    FloatTriple voxel_size,
    FloatTriple origin
) {
    auto base = static_cast<ptrdiff_t>(row) * 3;
    return {
        batch_indices[row],
        static_cast<int64_t>(
            std::floor((points[base] - origin[0]) / voxel_size[0])
        ),
        static_cast<int64_t>(
            std::floor((points[base + 1] - origin[1]) / voxel_size[1])
        ),
        static_cast<int64_t>(
            std::floor((points[base + 2] - origin[2]) / voxel_size[2])
        ),
    };
}

void write_sparse_quantization(
    std::vector<mx::array>& outputs,
    QuantizationInputs inputs,
    QuantizationSpec spec
) {
    auto point_data = inputs.points.data<float>();
    auto batch_data = inputs.batch_indices.data<int32_t>();
    auto point_count = std::min(inputs.active_rows, inputs.points.shape(0));

    std::vector<Coord> out_coords;
    std::vector<int32_t> inverse_rows(inputs.points.shape(0), -1);
    std::vector<int32_t> counts;
    std::unordered_map<Coord, int32_t, CoordHash> out_rows;
    out_coords.reserve(point_count);
    counts.reserve(point_count);
    out_rows.reserve(point_count);

    for (int point_row = 0; point_row < point_count; ++point_row) {
        auto candidate = voxel_coord_for_point(
            point_data, batch_data, point_row, spec.voxel_size, spec.origin
        );
        auto [match, inserted] = out_rows.emplace(
            candidate, static_cast<int32_t>(out_coords.size())
        );
        if (inserted) {
            out_coords.push_back(candidate);
            counts.push_back(0);
        }
        inverse_rows[point_row] = match->second;
        counts[match->second] += 1;
    }

    write_coords(outputs[0], out_coords, mx::int32);
    write_count(outputs[1], int(out_coords.size()));
    write_i32(outputs[2], inverse_rows, -1);
    write_i32(outputs[3], counts);
}

float voxel_reduce_scale(
    VoxelReduceOp reduce,
    const int32_t* voxel_counts,
    int voxel_row
) {
    if (reduce == VoxelReduceOp::Mean) {
        return 1.0F / static_cast<float>(std::max(voxel_counts[voxel_row], 1));
    }
    return 1.0F;
}

void write_voxel_features(
    mx::array& out,
    VoxelReduceOp reduce,
    VoxelFeatureInputs inputs,
    VoxelFeatureShape shape
) {
    auto point_count =
        std::min(read_scalar_i32(inputs.active_rows), shape.point_rows);
    auto feat_data = inputs.values.data<float>();
    auto inverse_data = inputs.inverse_rows.data<int32_t>();
    auto count_data = inputs.voxel_counts.data<int32_t>();
    auto out_data = out.data<float>();
    std::fill(out_data, out_data + out.size(), 0.0F);

    for (int point_row = 0; point_row < point_count; ++point_row) {
        auto voxel_row = inverse_data[point_row];
        if (voxel_row < 0 || voxel_row >= shape.voxel_rows) {
            continue;
        }
        auto scale = voxel_reduce_scale(reduce, count_data, voxel_row);
        for (int channel = 0; channel < shape.channels; ++channel) {
            auto point_index =
                static_cast<ptrdiff_t>(point_row) * shape.channels + channel;
            auto voxel_index =
                static_cast<ptrdiff_t>(voxel_row) * shape.channels + channel;
            out_data[voxel_index] += feat_data[point_index] * scale;
        }
    }
}

void write_voxel_feature_grad(
    mx::array& out,
    VoxelReduceOp reduce,
    VoxelFeatureInputs inputs,
    VoxelFeatureShape shape
) {
    auto point_count =
        std::min(read_scalar_i32(inputs.active_rows), shape.point_rows);
    auto cotangent_data = inputs.values.data<float>();
    auto inverse_data = inputs.inverse_rows.data<int32_t>();
    auto count_data = inputs.voxel_counts.data<int32_t>();
    auto out_data = out.data<float>();
    std::fill(out_data, out_data + out.size(), 0.0F);

    for (int point_row = 0; point_row < point_count; ++point_row) {
        auto voxel_row = inverse_data[point_row];
        if (voxel_row < 0 || voxel_row >= shape.voxel_rows) {
            continue;
        }
        auto scale = voxel_reduce_scale(reduce, count_data, voxel_row);
        for (int channel = 0; channel < shape.channels; ++channel) {
            auto point_index =
                static_cast<ptrdiff_t>(point_row) * shape.channels + channel;
            auto voxel_index =
                static_cast<ptrdiff_t>(voxel_row) * shape.channels + channel;
            out_data[point_index] = cotangent_data[voxel_index] * scale;
        }
    }
}

float point_axis_position(
    const float* points,
    int point_row,
    int axis,
    QuantizationSpec spec
) {
    auto value = points[static_cast<ptrdiff_t>(point_row) * 3 + axis];
    return (value - spec.origin[axis]) / spec.voxel_size[axis];
}

void write_nearest_point_voxel_map(
    mx::array& rows,
    mx::array& weights,
    PointVoxelMapInputs inputs,
    QuantizationSpec spec,
    PointVoxelMapShape shape
) {
    auto point_count =
        std::min(read_scalar_i32(inputs.point_active_rows), shape.point_rows);
    auto voxel_count =
        std::min(read_scalar_i32(inputs.voxel_active_rows), shape.voxel_rows);
    auto point_data = inputs.points.data<float>();
    auto batch_data = inputs.batch_indices.data<int32_t>();
    auto row_data = rows.data<int32_t>();
    auto weight_data = weights.data<float>();
    std::fill(row_data, row_data + rows.size(), -1);
    std::fill(weight_data, weight_data + weights.size(), 0.0F);

    auto voxel_values = read_coords(inputs.voxel_coords);
    voxel_values.resize(voxel_count);
    auto voxel_rows = first_row_map(voxel_values);

    for (int point_row = 0; point_row < point_count; ++point_row) {
        Coord query = {
            batch_data[point_row],
            static_cast<int64_t>(std::floor(
                point_axis_position(point_data, point_row, 0, spec) + 0.5F
            )),
            static_cast<int64_t>(std::floor(
                point_axis_position(point_data, point_row, 1, spec) + 0.5F
            )),
            static_cast<int64_t>(std::floor(
                point_axis_position(point_data, point_row, 2, spec) + 0.5F
            )),
        };
        auto match = voxel_rows.find(query);
        if (match == voxel_rows.end()) {
            continue;
        }
        row_data[static_cast<ptrdiff_t>(point_row) * 8] = match->second;
        weight_data[static_cast<ptrdiff_t>(point_row) * 8] = 1.0F;
    }
}

void write_linear_point_voxel_map(
    mx::array& rows,
    mx::array& weights,
    PointVoxelMapInputs inputs,
    QuantizationSpec spec,
    PointVoxelMapShape shape
) {
    auto point_count =
        std::min(read_scalar_i32(inputs.point_active_rows), shape.point_rows);
    auto voxel_count =
        std::min(read_scalar_i32(inputs.voxel_active_rows), shape.voxel_rows);
    auto point_data = inputs.points.data<float>();
    auto batch_data = inputs.batch_indices.data<int32_t>();
    auto row_data = rows.data<int32_t>();
    auto weight_data = weights.data<float>();
    std::fill(row_data, row_data + rows.size(), -1);
    std::fill(weight_data, weight_data + weights.size(), 0.0F);

    auto voxel_values = read_coords(inputs.voxel_coords);
    voxel_values.resize(voxel_count);
    auto voxel_rows = first_row_map(voxel_values);

    for (int point_row = 0; point_row < point_count; ++point_row) {
        float positions[3] = {
            point_axis_position(point_data, point_row, 0, spec),
            point_axis_position(point_data, point_row, 1, spec),
            point_axis_position(point_data, point_row, 2, spec),
        };
        int64_t lower[3] = {
            static_cast<int64_t>(std::floor(positions[0])),
            static_cast<int64_t>(std::floor(positions[1])),
            static_cast<int64_t>(std::floor(positions[2])),
        };
        float frac[3] = {
            positions[0] - static_cast<float>(lower[0]),
            positions[1] - static_cast<float>(lower[1]),
            positions[2] - static_cast<float>(lower[2]),
        };
        auto base = static_cast<ptrdiff_t>(point_row) * 8;
        for (int corner = 0; corner < 8; ++corner) {
            auto dx = corner & 1;
            auto dy = (corner >> 1) & 1;
            auto dz = (corner >> 2) & 1;
            Coord query = {
                batch_data[point_row],
                lower[0] + dx,
                lower[1] + dy,
                lower[2] + dz,
            };
            auto match = voxel_rows.find(query);
            if (match == voxel_rows.end()) {
                continue;
            }
            auto wx = dx == 0 ? 1.0F - frac[0] : frac[0];
            auto wy = dy == 0 ? 1.0F - frac[1] : frac[1];
            auto wz = dz == 0 ? 1.0F - frac[2] : frac[2];
            row_data[base + corner] = match->second;
            weight_data[base + corner] = wx * wy * wz;
        }
    }
}

void write_point_voxel_map(
    std::vector<mx::array>& outputs,
    PointVoxelMapInputs inputs,
    QuantizationSpec spec,
    PointVoxelInterpolationOp interpolation,
    PointVoxelMapShape shape
) {
    if (interpolation == PointVoxelInterpolationOp::Nearest) {
        write_nearest_point_voxel_map(
            outputs[0], outputs[1], inputs, spec, shape
        );
        return;
    }
    write_linear_point_voxel_map(outputs[0], outputs[1], inputs, spec, shape);
}

void write_point_features(
    mx::array& out,
    PointFeatureInputs inputs,
    VoxelFeatureShape shape
) {
    auto value_data = inputs.values.data<float>();
    auto row_data = inputs.rows.data<int32_t>();
    auto weight_data = inputs.weights.data<float>();
    auto out_data = out.data<float>();
    std::fill(out_data, out_data + out.size(), 0.0F);
    for (int point_row = 0; point_row < shape.point_rows; ++point_row) {
        for (int corner = 0; corner < 8; ++corner) {
            auto voxel_row =
                row_data[static_cast<ptrdiff_t>(point_row) * 8 + corner];
            if (voxel_row < 0 || voxel_row >= shape.voxel_rows) {
                continue;
            }
            auto weight =
                weight_data[static_cast<ptrdiff_t>(point_row) * 8 + corner];
            for (int channel = 0; channel < shape.channels; ++channel) {
                auto point_index =
                    static_cast<ptrdiff_t>(point_row) * shape.channels +
                    channel;
                auto voxel_index =
                    static_cast<ptrdiff_t>(voxel_row) * shape.channels +
                    channel;
                out_data[point_index] += value_data[voxel_index] * weight;
            }
        }
    }
}

void write_point_feature_grad(
    mx::array& out,
    PointFeatureInputs inputs,
    VoxelFeatureShape shape
) {
    auto point_grad_data = inputs.values.data<float>();
    auto row_data = inputs.rows.data<int32_t>();
    auto weight_data = inputs.weights.data<float>();
    auto out_data = out.data<float>();
    std::fill(out_data, out_data + out.size(), 0.0F);
    for (int point_row = 0; point_row < shape.point_rows; ++point_row) {
        for (int corner = 0; corner < 8; ++corner) {
            auto voxel_row =
                row_data[static_cast<ptrdiff_t>(point_row) * 8 + corner];
            if (voxel_row < 0 || voxel_row >= shape.voxel_rows) {
                continue;
            }
            auto weight =
                weight_data[static_cast<ptrdiff_t>(point_row) * 8 + corner];
            for (int channel = 0; channel < shape.channels; ++channel) {
                auto point_index =
                    static_cast<ptrdiff_t>(point_row) * shape.channels +
                    channel;
                auto voxel_index =
                    static_cast<ptrdiff_t>(voxel_row) * shape.channels +
                    channel;
                out_data[voxel_index] += point_grad_data[point_index] * weight;
            }
        }
    }
}

std::unordered_map<Coord, int32_t, CoordHash>
first_row_map(const std::vector<Coord>& coords) {
    std::unordered_map<Coord, int32_t, CoordHash> rows;
    rows.reserve(coords.size());
    for (int row = 0; row < int(coords.size()); ++row) {
        rows.emplace(coords[row], static_cast<int32_t>(row));
    }
    return rows;
}
