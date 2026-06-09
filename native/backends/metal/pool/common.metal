#include "native/backends/metal/coords/common.metal"

inline bool pool_coord_equal(thread const int* lhs, thread const int* rhs) {
    return lhs[0] == rhs[0] && lhs[1] == rhs[1] && lhs[2] == rhs[2] &&
           lhs[3] == rhs[3];
}

inline void pool_downsample_coord(
    device const int* coords,
    int row,
    int stride_x,
    int stride_y,
    int stride_z,
    thread int* out
) {
    int base = row * 4;
    out[0] = coords[base];
    out[1] = floor_div_int(coords[base + 1], stride_x);
    out[2] = floor_div_int(coords[base + 2], stride_y);
    out[3] = floor_div_int(coords[base + 3], stride_z);
}

inline bool pool_coord_seen(
    device const int* coords,
    int row,
    int stride_x,
    int stride_y,
    int stride_z,
    thread const int* candidate
) {
    for (int previous_row = 0; previous_row < row; ++previous_row) {
        int previous[4];
        pool_downsample_coord(
            coords, previous_row, stride_x, stride_y, stride_z, previous
        );
        if (pool_coord_equal(previous, candidate)) {
            return true;
        }
    }
    return false;
}

inline bool pool_find_input_row(
    device const int* coords,
    int rows,
    thread const int* target,
    thread int& out_row
) {
    for (int row = 0; row < rows; ++row) {
        if (coord4_equal(target, coords, row)) {
            out_row = row;
            return true;
        }
    }
    return false;
}

inline int pool_coord_hash(thread const int* coord) {
    uint hash = 2166136261u;
    hash = (hash ^ uint(coord[0])) * 16777619u;
    hash = (hash ^ uint(coord[1])) * 16777619u;
    hash = (hash ^ uint(coord[2])) * 16777619u;
    hash = (hash ^ uint(coord[3])) * 16777619u;
    int out = int(hash & 0x7fffffffu);
    return out == int(0x7fffffff) ? out - 1 : out;
}
