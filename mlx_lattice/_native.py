from __future__ import annotations

from typing import Any

from mlx_lattice import _ext as ext
from mlx_lattice.backends import cuda


class CompiledBackend:
    name = 'compiled'

    def preload(self) -> None:
        ext.version()

    def downsample_coords(self, coords, stride):
        return ext.downsample_coords(coords, stride)

    def union_coords(self, lhs, rhs):
        return ext.union_coords(lhs, rhs)

    def intersection_coords(self, lhs, rhs):
        return ext.intersection_coords(lhs, rhs)

    def lookup_coords(self, coords, queries):
        return ext.lookup_coords(coords, queries)

    def morton_codes(self, coords):
        return ext.morton_codes(coords)

    def occupancy_downsample(self, coords, active_rows):
        return ext.occupancy_downsample(coords, active_rows)

    def occupancy_expand(self, coords, active_rows, occupancy):
        return ext.occupancy_expand(coords, active_rows, occupancy)

    def child_coords_from_indices(self, parent_coords, child_indices):
        return ext.child_coords_from_indices(parent_coords, child_indices)

    def build_target_kernel_relation(
        self,
        coords,
        active_rows,
        target_coords,
        target_active_rows,
        kernel_size,
        stride,
        padding,
        dilation,
    ):
        return ext.build_target_kernel_relation(
            coords,
            active_rows,
            target_coords,
            target_active_rows,
            kernel_size,
            stride,
            padding,
            dilation,
        )

    def build_kernel_relation(
        self,
        coords,
        active_rows,
        kernel_size,
        stride,
        padding,
        dilation,
    ):
        return ext.build_kernel_relation(
            coords,
            active_rows,
            kernel_size,
            stride,
            padding,
            dilation,
        )

    def build_generative_relation(
        self,
        coords,
        active_rows,
        kernel_size,
        stride,
    ):
        return ext.build_generative_relation(
            coords,
            active_rows,
            kernel_size,
            stride,
        )

    def build_transposed_kernel_relation(
        self,
        coords,
        active_rows,
        kernel_size,
        stride,
        padding,
        dilation,
    ):
        return ext.build_transposed_kernel_relation(
            coords,
            active_rows,
            kernel_size,
            stride,
            padding,
            dilation,
        )

    def build_knn_relation(
        self,
        source_coords,
        source_active_rows,
        query_coords,
        query_active_rows,
        k,
    ):
        return ext.build_knn_relation(
            source_coords,
            source_active_rows,
            query_coords,
            query_active_rows,
            k,
        )

    def build_radius_relation(
        self,
        source_coords,
        source_active_rows,
        query_coords,
        query_active_rows,
        radius,
        max_neighbors,
    ):
        return ext.build_radius_relation(
            source_coords,
            source_active_rows,
            query_coords,
            query_active_rows,
            radius,
            max_neighbors,
        )

    def sparse_quantize(
        self,
        points,
        batch_indices,
        active_rows,
        voxel_size,
        origin,
    ):
        return ext.sparse_quantize(
            points,
            batch_indices,
            active_rows,
            voxel_size,
            origin,
        )

    def voxelize_features(
        self,
        feats,
        inverse_rows,
        voxel_counts,
        active_rows,
        reduction,
    ):
        return ext.voxelize_features(
            feats,
            inverse_rows,
            voxel_counts,
            active_rows,
            reduction,
        )

    def sparse_conv_features(
        self,
        feats,
        weights,
        in_rows,
        out_rows,
        kernel_ids,
        counts,
        row_offsets,
        in_row_offsets,
        in_edge_ids,
        kernel_row_offsets,
        kernel_edge_ids,
        out_capacity,
        n_kernels,
    ):
        return ext.sparse_conv_features(
            feats,
            weights,
            in_rows,
            out_rows,
            kernel_ids,
            counts,
            row_offsets,
            in_row_offsets,
            in_edge_ids,
            kernel_row_offsets,
            kernel_edge_ids,
            out_capacity,
            n_kernels,
        )

    def sparse_pool_features(
        self,
        feats,
        in_rows,
        out_rows,
        kernel_ids,
        row_offsets,
        counts,
        in_row_offsets,
        in_edge_ids,
        input_exclusive,
        reduce,
        out_capacity,
        n_kernels,
    ):
        del in_row_offsets, in_edge_ids
        return ext.sparse_pool_features(
            feats,
            in_rows,
            out_rows,
            kernel_ids,
            row_offsets,
            counts,
            input_exclusive,
            reduce,
            out_capacity,
            n_kernels,
        )

    def normalized_cdf(self, prob):
        return ext.normalized_cdf(prob)

    def range_encode(self, cdf, symbols):
        return ext.range_encode(cdf, symbols)

    def range_decode(self, cdf, stream):
        return ext.range_decode(cdf, stream)

    def range_encode_from_prob(self, prob, symbols):
        return ext.range_encode_from_prob(prob, symbols)

    def range_decode_from_prob(self, prob, stream):
        return ext.range_decode_from_prob(prob, stream)

    def rans_encode_from_prob(self, prob, symbols):
        return ext.rans_encode_from_prob(prob, symbols)

    def rans_decode_from_prob(self, prob, stream):
        return ext.rans_decode_from_prob(prob, stream)


class CudaBackend:
    name = 'cuda'

    def __init__(self, compiled: CompiledBackend) -> None:
        self._compiled = compiled

    def preload(self) -> None:
        self._compiled.preload()

    def downsample_coords(self, coords, stride):
        return cuda.downsample_coords(coords, stride)

    def union_coords(self, lhs, rhs):
        return cuda.union_coords(lhs, rhs)

    def intersection_coords(self, lhs, rhs):
        return cuda.intersection_coords(lhs, rhs)

    def lookup_coords(self, coords, queries):
        return cuda.lookup_coords(coords, queries)

    def morton_codes(self, coords):
        return cuda.morton_codes(coords)

    def occupancy_downsample(self, coords, active_rows):
        return cuda.occupancy_downsample(coords, active_rows)

    def occupancy_expand(self, coords, active_rows, occupancy):
        return cuda.occupancy_expand(coords, active_rows, occupancy)

    def child_coords_from_indices(self, parent_coords, child_indices):
        return cuda.child_coords_from_indices(parent_coords, child_indices)

    def build_target_kernel_relation(self, *args):
        return cuda.build_target_kernel_relation(*args)

    def build_kernel_relation(self, *args):
        return cuda.build_kernel_relation(*args)

    def build_generative_relation(self, *args):
        return cuda.build_generative_relation(*args)

    def build_transposed_kernel_relation(self, *args):
        return cuda.build_transposed_kernel_relation(*args)

    def build_knn_relation(self, *args):
        return cuda.build_knn_relation(*args)

    def build_radius_relation(self, *args):
        return cuda.build_radius_relation(*args)

    def sparse_quantize(
        self,
        points,
        batch_indices,
        active_rows,
        voxel_size,
        origin,
    ):
        return cuda.sparse_quantize(
            points,
            batch_indices,
            active_rows,
            voxel_size,
            origin,
        )

    def voxelize_features(
        self,
        feats,
        inverse_rows,
        voxel_counts,
        active_rows,
        reduction,
    ):
        return cuda.voxelize_features(
            feats,
            inverse_rows,
            voxel_counts,
            active_rows,
            reduction,
        )

    def sparse_conv_features(
        self,
        feats,
        weights,
        in_rows,
        out_rows,
        kernel_ids,
        counts,
        row_offsets,
        in_row_offsets,
        in_edge_ids,
        kernel_row_offsets,
        kernel_edge_ids,
        out_capacity,
        n_kernels,
    ):
        return cuda.sparse_conv_features(
            feats,
            weights,
            in_rows,
            out_rows,
            kernel_ids,
            counts,
            row_offsets,
            in_row_offsets,
            in_edge_ids,
            kernel_row_offsets,
            kernel_edge_ids,
            out_capacity,
            n_kernels,
        )

    def sparse_pool_features(
        self,
        feats,
        in_rows,
        out_rows,
        kernel_ids,
        row_offsets,
        counts,
        in_row_offsets,
        in_edge_ids,
        input_exclusive,
        reduce,
        out_capacity,
        n_kernels,
    ):
        return cuda.sparse_pool_features(
            feats,
            in_rows,
            out_rows,
            kernel_ids,
            row_offsets,
            counts,
            in_row_offsets,
            in_edge_ids,
            input_exclusive,
            reduce,
            out_capacity,
            n_kernels,
        )

    def normalized_cdf(self, *args):
        return self._compiled.normalized_cdf(*args)

    def range_encode(self, *args):
        return self._compiled.range_encode(*args)

    def range_decode(self, *args):
        return self._compiled.range_decode(*args)

    def range_encode_from_prob(self, *args):
        return self._compiled.range_encode_from_prob(*args)

    def range_decode_from_prob(self, *args):
        return self._compiled.range_decode_from_prob(*args)

    def rans_encode_from_prob(self, *args):
        return self._compiled.rans_encode_from_prob(*args)

    def rans_decode_from_prob(self, *args):
        return self._compiled.rans_decode_from_prob(*args)


class NativeFacade:
    def __init__(self) -> None:
        self.compiled = CompiledBackend()
        self.cuda = CudaBackend(self.compiled)

    def current(self) -> CompiledBackend | CudaBackend:
        if cuda.selected():
            return self.cuda
        return self.compiled

    def preload(self) -> None:
        self.compiled.preload()

    def __getattr__(self, name: str) -> Any:
        return getattr(self.current(), name)


native = NativeFacade()


def backend_info() -> dict[str, object]:
    native_capabilities = ext.capabilities()
    capabilities = dict(native_capabilities)
    capabilities['cuda'] = cuda.runtime_available()
    return {
        'version': ext.version(),
        'backend': native.current().name,
        'capabilities': capabilities,
        'native_capabilities': native_capabilities,
        'cuda': cuda.info(),
    }


__all__ = [
    'CompiledBackend',
    'CudaBackend',
    'NativeFacade',
    'backend_info',
    'ext',
    'native',
]
