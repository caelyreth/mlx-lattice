from __future__ import annotations

from pathlib import Path

import pytest
from lattice_conformance.gameleon_geometry import (
    GeometryProfileWeights,
    Level8GeometryProfile,
    read_ascii_ply_xyz,
)

from tests.support import mx


def test_level8_profile_runs_the_full_teacher_forced_stage() -> None:
    profile = Level8GeometryProfile(GeometryProfileWeights(_zero_weights()))
    xyz = mx.array(
        [[x, y, z] for x in range(8) for y in range(8) for z in range(8)],
        dtype=mx.int32,
    )

    result = profile(profile.from_xyz(xyz))
    mx.eval(result.bits, result.bpp)

    assert result.input_points == 512
    assert result.levels == (8, 64)
    assert len(result.stages) == 1
    stage = result.stages[0]
    assert stage.parent_points == 8
    assert stage.target_points == 64
    assert stage.group1_points + stage.group2_points == 64
    assert float(result.bpp.item()) == pytest.approx(1.0, abs=1e-6)


def test_geometry_profile_rejects_legacy_kernel_state_at_load_time(
    tmp_path: Path,
) -> None:
    path = tmp_path / 'legacy.safetensors'
    mx.save_safetensors(
        str(path),
        {'encoder.kernel': mx.zeros((27, 1, 1), dtype=mx.float32)},
    )

    with pytest.raises(ValueError, match='convert-checkpoint'):
        GeometryProfileWeights.from_safetensors(path)


def test_ascii_ply_reader_uses_only_vertex_properties(
    tmp_path: Path,
) -> None:
    path = tmp_path / 'support.ply'
    path.write_text(
        '\n'.join(
            (
                'ply',
                'format ascii 1.0',
                'element vertex 2',
                'property float x',
                'property float y',
                'property float z',
                'property uchar quality',
                'element face 1',
                'property list uchar int vertex_indices',
                'end_header',
                '1 2 3 7',
                '4 5 6 8',
                '3 0 1 0',
                '',
            )
        ),
        encoding='ascii',
    )

    xyz = read_ascii_ply_xyz(path)
    mx.eval(xyz)

    assert xyz.tolist() == [[1, 2, 3], [4, 5, 6]]


def _zero_weights() -> dict[str, mx.array]:
    values: dict[str, mx.array] = {
        'prior_embedding.weight': mx.zeros((256, 32), dtype=mx.float32),
        'target_embedding.target_res_embedding.weight': mx.zeros(
            (8, 32), dtype=mx.float32
        ),
        'feature_fusion.0.weight': mx.zeros((32, 64), dtype=mx.float32),
        'feature_fusion.0.bias': mx.zeros((32,), dtype=mx.float32),
    }
    for prefix in ('prior_resnet', 'target_resnet'):
        values[f'{prefix}.0.weight'] = mx.zeros(
            (27, 32, 32), dtype=mx.float32
        )
        for block in (2, 3):
            for conv in ('conv0', 'conv1'):
                values[f'{prefix}.{block}.{conv}.weight'] = mx.zeros(
                    (27, 32, 32), dtype=mx.float32
                )
    for group in ('group1', 'group2'):
        for stage in ('s0', 's1'):
            prefix = f'{group}_spatial_conv_{stage}'
            for layer in (0, 2):
                values[f'{prefix}.{layer}.weight'] = mx.zeros(
                    (27, 32, 32), dtype=mx.float32
                )
        values[f'{group}_pred_head_s1_emb.weight'] = mx.zeros(
            (16, 32), dtype=mx.float32
        )
        for stage in ('s0', 's1'):
            prefix = f'{group}_pred_head_{stage}'
            for layer, shape in (
                (0, (32, 32)),
                (2, (32, 32)),
                (4, (16, 32)),
            ):
                values[f'{prefix}.{layer}.weight'] = mx.zeros(
                    shape, dtype=mx.float32
                )
                values[f'{prefix}.{layer}.bias'] = mx.zeros(
                    (shape[0],), dtype=mx.float32
                )
    values['neighbor_conv.weight'] = mx.zeros(
        (27, 32, 32), dtype=mx.float32
    )
    return values
