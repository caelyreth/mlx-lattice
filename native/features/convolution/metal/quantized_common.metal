#pragma once

#include <metal_stdlib>

using namespace metal;

template <int bits> constexpr int quantized_values_per_word() {
    return 32 / bits;
}

template <int bits> constexpr uint quantized_mask() {
    return (1u << bits) - 1u;
}

template <int bits>
inline uint load_quantized(
    device const uint* weights,
    int kernel_id,
    int ci,
    int co,
    int packed_words,
    int out_channels
) {
    constexpr int values_per_word = quantized_values_per_word<bits>();
    int word = ci / values_per_word;
    int shift = (ci - word * values_per_word) * bits;
    uint packed =
        weights[(kernel_id * packed_words + word) * out_channels + co];
    return (packed >> shift) & quantized_mask<bits>();
}

template <int bits>
inline half load_dequantized(
    device const uint* weights,
    device const half* scales,
    device const half* biases,
    int kernel_id,
    int ci,
    int co,
    int packed_words,
    int group_size,
    int groups,
    int out_channels
) {
    uint quantized = load_quantized<bits>(
        weights, kernel_id, ci, co, packed_words, out_channels
    );
    int group = ci / group_size;
    int quant_index = (kernel_id * groups + group) * out_channels + co;
    return half(
        float(quantized) * float(scales[quant_index]) +
        float(biases[quant_index])
    );
}
