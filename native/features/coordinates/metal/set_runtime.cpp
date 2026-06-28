#include "features/coordinates/metal/runtime_detail.h"

namespace mlx_lattice::coords::metal {
// MARK: - set ops

void eval_set_coords(
    CoordSetOp op,
    Triple stride,
    CoordSetShape shape,
    const mx::Stream& stream,
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs
) {
    require_i32_inputs(inputs, {"coords"});
    if (op != CoordSetOp::Downsample) {
        require_i32_inputs(inputs, {"coords", "rhs coords"});
    }

#ifdef _METAL_
    auto& out_coords = outputs[0];
    auto& count = outputs[1];
    backend::allocate(out_coords);
    backend::allocate(count);

    auto library = backend::metal::lattice_library(stream);
    auto& encoder = backend::metal::command_encoder(stream);

    if (op == CoordSetOp::Downsample) {
        auto table_capacity = coord_hash_capacity(shape.lhs_rows);
        auto table = make_int32_temp(table_capacity);
        auto selected = make_int32_temp(shape.lhs_rows);
        encoder.add_temporaries({table, selected});
        clear_coord_hash(stream, library, encoder, table, table_capacity);
        auto build = backend::metal::lattice_kernel(
            stream, "build_downsample_coord_hash_i32", library
        );
        encoder.set_compute_pipeline_state(build);
        encoder.set_input_array(inputs[0], 0);
        encoder.set_output_array(table, 1);
        encoder.set_bytes(shape.lhs_rows, 2);
        encoder.set_bytes(table_capacity, 3);
        bind_triple_bytes(encoder, stride, 4);
        dispatch_1d(encoder, build, static_cast<size_t>(shape.lhs_rows));

        auto plan = backend::metal::lattice_kernel(
            stream, "plan_downsample_coord_set_i32", library
        );
        encoder.set_compute_pipeline_state(plan);
        encoder.set_input_array(inputs[0], 0);
        encoder.set_input_array(table, 1);
        encoder.set_output_array(selected, 2);
        encoder.set_bytes(shape.lhs_rows, 3);
        encoder.set_bytes(table_capacity, 4);
        bind_triple_bytes(encoder, stride, 5);
        dispatch_1d(encoder, plan, static_cast<size_t>(shape.lhs_rows));

        auto compact = backend::metal::lattice_kernel(
            stream,
            shape.lhs_rows >= kParallelCompactThreshold
                ? "scatter_downsample_coord_set_i32"
                : "compact_downsample_coord_set_i32",
            library
        );
        if (shape.lhs_rows >= kParallelCompactThreshold) {
            auto buffers = make_stable_compact_buffers(shape.lhs_rows);
            encode_stable_compact_offsets(
                stream,
                library,
                encoder,
                selected,
                count,
                buffers,
                shape.lhs_rows
            );
            encoder.set_compute_pipeline_state(compact);
            encoder.set_input_array(inputs[0], 0);
            encoder.set_input_array(selected, 1);
            encoder.set_input_array(buffers.local_offsets, 2);
            encoder.set_input_array(buffers.block_offsets, 3);
            encoder.set_output_array(out_coords, 4);
            encoder.set_bytes(shape.lhs_rows, 5);
            bind_triple_bytes(encoder, stride, 6);
            dispatch_1d(encoder, compact, static_cast<size_t>(shape.lhs_rows));
            return;
        }
        encoder.set_compute_pipeline_state(compact);
        encoder.set_input_array(inputs[0], 0);
        encoder.set_input_array(selected, 1);
        encoder.set_output_array(out_coords, 2);
        encoder.set_output_array(count, 3);
        encoder.set_bytes(shape.lhs_rows, 4);
        bind_triple_bytes(encoder, stride, 5);
        dispatch_single(encoder);
    } else if (op == CoordSetOp::Union) {
        auto total_rows = shape.lhs_rows + shape.rhs_rows;
        auto lhs_table_capacity = coord_hash_capacity(shape.lhs_rows);
        auto rhs_table_capacity = coord_hash_capacity(shape.rhs_rows);
        auto lhs_table = make_int32_temp(lhs_table_capacity);
        auto rhs_table = make_int32_temp(rhs_table_capacity);
        auto selected = make_int32_temp(total_rows);
        encoder.add_temporaries({lhs_table, rhs_table, selected});
        clear_coord_hash(
            stream, library, encoder, lhs_table, lhs_table_capacity
        );
        clear_coord_hash(
            stream, library, encoder, rhs_table, rhs_table_capacity
        );
        insert_coord_hash(
            stream,
            library,
            encoder,
            inputs[0],
            lhs_table,
            CoordHashShape{shape.lhs_rows, lhs_table_capacity}
        );
        insert_coord_hash(
            stream,
            library,
            encoder,
            inputs[1],
            rhs_table,
            CoordHashShape{shape.rhs_rows, rhs_table_capacity}
        );

        auto plan = backend::metal::lattice_kernel(
            stream, "plan_union_coord_set_i32", library
        );
        encoder.set_compute_pipeline_state(plan);
        encoder.set_input_array(inputs[0], 0);
        encoder.set_input_array(inputs[1], 1);
        encoder.set_input_array(lhs_table, 2);
        encoder.set_input_array(rhs_table, 3);
        encoder.set_output_array(selected, 4);
        encoder.set_bytes(shape.lhs_rows, 5);
        encoder.set_bytes(shape.rhs_rows, 6);
        encoder.set_bytes(lhs_table_capacity, 7);
        encoder.set_bytes(rhs_table_capacity, 8);
        dispatch_1d(encoder, plan, static_cast<size_t>(total_rows));

        auto compact = backend::metal::lattice_kernel(
            stream,
            total_rows >= kParallelCompactThreshold
                ? "scatter_union_coord_set_i32"
                : "compact_union_coord_set_i32",
            library
        );
        if (total_rows >= kParallelCompactThreshold) {
            auto buffers = make_stable_compact_buffers(total_rows);
            encode_stable_compact_offsets(
                stream, library, encoder, selected, count, buffers, total_rows
            );
            encoder.set_compute_pipeline_state(compact);
            encoder.set_input_array(inputs[0], 0);
            encoder.set_input_array(inputs[1], 1);
            encoder.set_input_array(selected, 2);
            encoder.set_input_array(buffers.local_offsets, 3);
            encoder.set_input_array(buffers.block_offsets, 4);
            encoder.set_output_array(out_coords, 5);
            encoder.set_bytes(shape.lhs_rows, 6);
            encoder.set_bytes(shape.rhs_rows, 7);
            dispatch_1d(encoder, compact, static_cast<size_t>(total_rows));
            return;
        }
        encoder.set_compute_pipeline_state(compact);
        encoder.set_input_array(inputs[0], 0);
        encoder.set_input_array(inputs[1], 1);
        encoder.set_input_array(selected, 2);
        encoder.set_output_array(out_coords, 3);
        encoder.set_output_array(count, 4);
        encoder.set_bytes(shape.lhs_rows, 5);
        encoder.set_bytes(shape.rhs_rows, 6);
        dispatch_single(encoder);
    } else {
        auto rhs_table_capacity = coord_hash_capacity(shape.rhs_rows);
        auto lhs_table_capacity = coord_hash_capacity(shape.lhs_rows);
        auto rhs_table = make_int32_temp(rhs_table_capacity);
        auto lhs_table = make_int32_temp(lhs_table_capacity);
        auto selected = make_int32_temp(shape.lhs_rows);
        encoder.add_temporaries({rhs_table, lhs_table, selected});
        clear_coord_hash(
            stream, library, encoder, rhs_table, rhs_table_capacity
        );
        clear_coord_hash(
            stream, library, encoder, lhs_table, lhs_table_capacity
        );
        insert_coord_hash(
            stream,
            library,
            encoder,
            inputs[1],
            rhs_table,
            CoordHashShape{shape.rhs_rows, rhs_table_capacity}
        );
        insert_coord_hash(
            stream,
            library,
            encoder,
            inputs[0],
            lhs_table,
            CoordHashShape{shape.lhs_rows, lhs_table_capacity}
        );
        auto plan = backend::metal::lattice_kernel(
            stream, "plan_intersection_coord_set_i32", library
        );
        encoder.set_compute_pipeline_state(plan);
        encoder.set_input_array(inputs[0], 0);
        encoder.set_input_array(inputs[1], 1);
        encoder.set_input_array(rhs_table, 2);
        encoder.set_input_array(lhs_table, 3);
        encoder.set_output_array(selected, 4);
        encoder.set_bytes(shape.lhs_rows, 5);
        encoder.set_bytes(rhs_table_capacity, 6);
        encoder.set_bytes(lhs_table_capacity, 7);
        dispatch_1d(encoder, plan, static_cast<size_t>(shape.lhs_rows));

        auto compact = backend::metal::lattice_kernel(
            stream,
            shape.lhs_rows >= kParallelCompactThreshold
                ? "scatter_intersection_coord_set_i32"
                : "compact_intersection_coord_set_i32",
            library
        );
        if (shape.lhs_rows >= kParallelCompactThreshold) {
            auto buffers = make_stable_compact_buffers(shape.lhs_rows);
            encode_stable_compact_offsets(
                stream,
                library,
                encoder,
                selected,
                count,
                buffers,
                shape.lhs_rows
            );
            encoder.set_compute_pipeline_state(compact);
            encoder.set_input_array(inputs[0], 0);
            encoder.set_input_array(selected, 1);
            encoder.set_input_array(buffers.local_offsets, 2);
            encoder.set_input_array(buffers.block_offsets, 3);
            encoder.set_output_array(out_coords, 4);
            encoder.set_bytes(shape.lhs_rows, 5);
            dispatch_1d(encoder, compact, static_cast<size_t>(shape.lhs_rows));
            return;
        }
        encoder.set_compute_pipeline_state(compact);
        encoder.set_input_array(inputs[0], 0);
        encoder.set_input_array(selected, 1);
        encoder.set_output_array(out_coords, 2);
        encoder.set_output_array(count, 3);
        encoder.set_bytes(shape.lhs_rows, 4);
        dispatch_single(encoder);
    }
#else
    (void)op;
    (void)stride;
    (void)shape;
    (void)stream;
    (void)inputs;
    (void)outputs;
    throw std::runtime_error("Metal support is not available.");
#endif
}

void eval_lookup_coords(
    CoordLookupShape shape,
    const mx::Stream& stream,
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs
) {
    require_i32_inputs(inputs, {"coords", "queries"});

#ifdef _METAL_
    auto& out = outputs[0];
    backend::allocate(out);

    auto library = backend::metal::lattice_library(stream);
    auto& encoder = backend::metal::command_encoder(stream);
    auto table_capacity = coord_hash_capacity(shape.rows);
    auto table = make_int32_temp(table_capacity);
    encoder.add_temporary(table);
    clear_coord_hash(stream, library, encoder, table, table_capacity);
    insert_coord_hash(
        stream,
        library,
        encoder,
        inputs[0],
        table,
        CoordHashShape{shape.rows, table_capacity}
    );

    auto kernel = backend::metal::lattice_kernel(
        stream, "lookup_coords_hash_i32", library
    );
    encoder.set_compute_pipeline_state(kernel);
    encoder.set_input_array(inputs[0], 0);
    encoder.set_input_array(inputs[1], 1);
    encoder.set_input_array(table, 2);
    encoder.set_output_array(out, 3);
    encoder.set_bytes(shape.query_rows, 4);
    encoder.set_bytes(table_capacity, 5);
    dispatch_1d(encoder, kernel, static_cast<size_t>(shape.query_rows));
#else
    (void)shape;
    (void)stream;
    (void)inputs;
    (void)outputs;
    throw std::runtime_error("Metal support is not available.");
#endif
}

void eval_sparse_alignment(
    SparseJoinOp join,
    SparseAlignmentShape shape,
    const mx::Stream& stream,
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs
) {
    require_i32_inputs(
        inputs,
        {
            "lhs coords",
            "lhs active rows",
            "rhs coords",
            "rhs active rows",
        }
    );

#ifdef _METAL_
    backend::allocate_all(outputs);

    auto library = backend::metal::lattice_library(stream);
    auto& encoder = backend::metal::command_encoder(stream);
    auto lhs_table_capacity = coord_hash_capacity(shape.lhs_rows);
    auto rhs_table_capacity = coord_hash_capacity(shape.rhs_rows);
    auto lhs_table = make_int32_temp(lhs_table_capacity);
    auto rhs_table = make_int32_temp(rhs_table_capacity);
    auto total = shape.output_rows;
    auto selected = make_int32_temp(total);
    auto plan_lhs_rows = make_int32_temp(total);
    auto plan_rhs_rows = make_int32_temp(total);
    encoder.add_temporaries(
        {lhs_table, rhs_table, selected, plan_lhs_rows, plan_rhs_rows}
    );

    clear_coord_hash(stream, library, encoder, lhs_table, lhs_table_capacity);
    clear_coord_hash(stream, library, encoder, rhs_table, rhs_table_capacity);
    auto insert_active = backend::metal::lattice_kernel(
        stream, "coord_hash_insert_active_rows_i32", library
    );
    encoder.set_compute_pipeline_state(insert_active);
    encoder.set_input_array(inputs[0], 0);
    encoder.set_input_array(inputs[1], 1);
    encoder.set_output_array(lhs_table, 2);
    encoder.set_bytes(shape.lhs_rows, 3);
    encoder.set_bytes(lhs_table_capacity, 4);
    dispatch_1d(encoder, insert_active, static_cast<size_t>(shape.lhs_rows));
    encoder.set_compute_pipeline_state(insert_active);
    encoder.set_input_array(inputs[2], 0);
    encoder.set_input_array(inputs[3], 1);
    encoder.set_output_array(rhs_table, 2);
    encoder.set_bytes(shape.rhs_rows, 3);
    encoder.set_bytes(rhs_table_capacity, 4);
    dispatch_1d(encoder, insert_active, static_cast<size_t>(shape.rhs_rows));

    auto plan = backend::metal::lattice_kernel(
        stream, "plan_sparse_alignment_i32", library
    );
    encoder.set_compute_pipeline_state(plan);
    bind_input_arrays(encoder, inputs);
    encoder.set_input_array(lhs_table, 4);
    encoder.set_input_array(rhs_table, 5);
    encoder.set_output_array(selected, 6);
    encoder.set_output_array(plan_lhs_rows, 7);
    encoder.set_output_array(plan_rhs_rows, 8);
    encoder.set_bytes(sparse_join_op_id(join), 9);
    encoder.set_bytes(shape.lhs_rows, 10);
    encoder.set_bytes(shape.rhs_rows, 11);
    encoder.set_bytes(lhs_table_capacity, 12);
    encoder.set_bytes(rhs_table_capacity, 13);
    dispatch_1d(encoder, plan, static_cast<size_t>(total));

    auto buffers = make_stable_compact_buffers(total);
    encode_stable_compact_offsets(
        stream, library, encoder, selected, outputs[1], buffers, total
    );

    auto scatter = backend::metal::lattice_kernel(
        stream, "scatter_sparse_alignment_i32", library
    );
    encoder.set_compute_pipeline_state(scatter);
    encoder.set_input_array(inputs[0], 0);
    encoder.set_input_array(inputs[2], 1);
    encoder.set_input_array(selected, 2);
    encoder.set_input_array(plan_lhs_rows, 3);
    encoder.set_input_array(plan_rhs_rows, 4);
    encoder.set_input_array(buffers.local_offsets, 5);
    encoder.set_input_array(buffers.block_offsets, 6);
    encoder.set_output_array(outputs[0], 7);
    encoder.set_output_array(outputs[2], 8);
    encoder.set_output_array(outputs[3], 9);
    encoder.set_bytes(total, 10);
    dispatch_1d(encoder, scatter, static_cast<size_t>(total));
#else
    (void)join;
    (void)shape;
    (void)stream;
    (void)inputs;
    (void)outputs;
    throw std::runtime_error("Metal support is not available.");
#endif
}

} // namespace mlx_lattice::coords::metal
