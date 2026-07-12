from __future__ import annotations

import pytest

from lattice_contract.reference import submanifold_conv3d_f32_to_f64


def test_submanifold_reference_uses_canonical_non_cubic_kernel_rows() -> (
    None
):
    coordinates = ((0, 0, 0, 0), (0, 1, 0, 0), (0, 0, 0, 1))
    features = ((1.0,), (2.0,), (4.0,))
    weight = tuple(((float(index + 1),),) for index in range(15))

    actual = submanifold_conv3d_f32_to_f64(
        coordinates,
        features,
        weight,
        kernel_size=(3, 1, 5),
    )

    assert actual == ((70.0,), (35.0,), (63.0,))


def test_submanifold_reference_rejects_noncanonical_inputs() -> None:
    with pytest.raises(ValueError, match='unique'):
        submanifold_conv3d_f32_to_f64(
            ((0, 0, 0, 0), (0, 0, 0, 0)),
            ((1.0,), (2.0,)),
            (((1.0,),),),
            kernel_size=1,
        )
