#pragma once

#include <cstddef>
#include <string>
#include <utility>

namespace mlx_lattice::metal {

std::string binary_dir();

} // namespace mlx_lattice::metal

#ifdef _METAL_
#include <algorithm>
#include <cstdint>
#include <initializer_list>
#include <type_traits>
#include <vector>

#include "mlx/allocator.h"
#include "mlx/array.h"
#include "mlx/backend/metal/device.h"
#include "mlx/dtype.h"

namespace mlx_lattice::backend::metal {

namespace mx = mlx::core;

inline auto lattice_library(const mx::Stream& stream) {
    auto& device = mx::metal::device(stream.device);
    return device.get_library("mlx_lattice", mlx_lattice::metal::binary_dir());
}

template <typename Library>
inline auto
lattice_kernel(const mx::Stream& stream, const char* name, Library& library) {
    auto& device = mx::metal::device(stream.device);
    return device.get_kernel(name, library);
}

inline auto& command_encoder(const mx::Stream& stream) {
    return mx::metal::get_command_encoder(stream);
}

inline int stride_i32(const mx::array& array, int dim) {
    return static_cast<int>(array.strides(dim));
}

template <typename Encoder>
void bind_input_arrays(
    Encoder& encoder,
    const std::vector<mx::array>& inputs,
    int first = 0
) {
    for (int index = 0; index < static_cast<int>(inputs.size()); ++index) {
        encoder.set_input_array(inputs[index], first + index);
    }
}

template <typename Encoder>
void bind_input_arrays(
    Encoder& encoder,
    const std::vector<mx::array>& inputs,
    int first,
    int count
) {
    for (int index = 0; index < count; ++index) {
        encoder.set_input_array(inputs[index], first + index);
    }
}

template <typename Encoder>
void bind_input_arrays(
    Encoder& encoder,
    const std::vector<mx::array>& inputs,
    std::initializer_list<int> source_indices,
    int first = 0
) {
    int target = first;
    for (auto source : source_indices) {
        encoder.set_input_array(inputs[source], target++);
    }
}

template <typename T> constexpr mx::Dtype dtype_for();

template <> constexpr mx::Dtype dtype_for<float>() { return mx::float32; }

template <> constexpr mx::Dtype dtype_for<mx::float16_t>() {
    return mx::float16;
}

template <> constexpr mx::Dtype dtype_for<std::int32_t>() { return mx::int32; }

template <typename T> mx::array make_temp(std::size_t elements) {
    static_assert(
        std::is_same_v<T, float> || std::is_same_v<T, mx::float16_t> ||
            std::is_same_v<T, std::int32_t>,
        "Unsupported Metal temporary type."
    );
    auto count = std::max<std::size_t>(elements, 1);
    return mx::array(
        mx::allocator::malloc(count * sizeof(T)),
        mx::Shape{static_cast<int>(count)},
        dtype_for<T>()
    );
}

} // namespace mlx_lattice::backend::metal

template <typename Encoder, typename Kernel>
void dispatch_1d(Encoder& encoder, Kernel* kernel, std::size_t elements) {
    auto threads = std::max<std::size_t>(elements, 1);
    auto group = std::min(threads, kernel->maxTotalThreadsPerThreadgroup());
    encoder.dispatch_threads(MTL::Size(threads, 1, 1), MTL::Size(group, 1, 1));
}

template <typename Encoder> void dispatch_single(Encoder& encoder) {
    encoder.dispatch_threads(MTL::Size(1, 1, 1), MTL::Size(1, 1, 1));
}

template <typename Encoder, typename... Values>
void set_bytes_range(Encoder& encoder, int first, Values&&... values) {
    int index = first;
    (encoder.set_bytes(std::forward<Values>(values), index++), ...);
}
#endif
