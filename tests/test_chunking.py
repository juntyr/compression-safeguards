from itertools import product

import numpy as np
import xarray as xr
from xarray_safeguards import apply_data_array_correction, produce_data_array_correction

from compression_safeguards.api import Safeguards
from compression_safeguards.safeguards._qois.expr.hashing import (
    HashingExpr,
    _patch_for_hashing_qoi_dev_only,
)
from compression_safeguards.safeguards.stencil import BoundaryCondition
from compression_safeguards.safeguards.stencil.qoi.eb import (
    StencilQuantityOfInterestErrorBoundSafeguard,
)


def check_all_boundaries(data: np.ndarray, chunks: int, constant_boundary=4.2):
    da = xr.DataArray(data, name="da").chunk(chunks)
    da_prediction = xr.DataArray(np.ones_like(data), name="da").chunk(chunks)

    for before, after, boundary in product(
        [0, 1, 2],
        [0, 1, 2],
        BoundaryCondition,
    ):
        try:
            safeguard = StencilQuantityOfInterestErrorBoundSafeguard(
                qoi="x",
                neighbourhood=[
                    dict(
                        axis=0,
                        before=before,
                        after=after,
                        boundary=boundary,
                        constant_boundary=constant_boundary
                        if boundary == BoundaryCondition.constant
                        else None,
                    )
                ],
                type="abs",
                eb=0,
            )
            safeguard._qoi_expr._expr = HashingExpr.from_data_shape(
                data_shape=safeguard._qoi_expr._stencil_shape,
                late_bound_constants=frozenset(),
            )
            safeguard._qoi_expr._late_bound_constants = (
                safeguard._qoi_expr._expr.late_bound_constants
            )

            with _patch_for_hashing_qoi_dev_only():
                global_hash = Safeguards(safeguards=[safeguard]).compute_correction(
                    data=da.values, prediction=da_prediction.values
                )
                chunked_hash = produce_data_array_correction(
                    data=da,
                    prediction=da_prediction,
                    safeguards=[safeguard],
                )
                np.testing.assert_array_equal(chunked_hash.values, global_hash)
        except Exception as err:
            print(before, after, boundary)  # noqa: T201
            raise err


def test_empty():
    check_all_boundaries(np.empty(0), 1)


def test_dimensions():
    check_all_boundaries(np.array([42.0]), 1)
    check_all_boundaries(np.array([42], dtype=np.int64), 1, constant_boundary=24)
    check_all_boundaries(np.array([[42.0]]), 1)
    check_all_boundaries(np.array([[[42.0]]]), 1)


def test_unit():
    data = np.linspace(-1.0, 1.0, 100, dtype=np.float16)
    check_all_boundaries(data[::10], 1, constant_boundary=2.5)
    check_all_boundaries(data, 10, constant_boundary=2.5)
    check_all_boundaries(data, 17, constant_boundary=2.5)


def test_circle():
    data = np.linspace(-np.pi * 2, np.pi * 2, 100, dtype=np.int64)
    check_all_boundaries(data[::10], 1, constant_boundary=42)
    check_all_boundaries(data, 10, constant_boundary=42)
    check_all_boundaries(data, 17, constant_boundary=42)


def test_arange():
    data = np.arange(100, dtype=float)
    check_all_boundaries(data[::10], 1)
    check_all_boundaries(data, 10)
    check_all_boundaries(data, 17)


def test_linspace():
    data = np.linspace(-1024, 1024, 2831, dtype=np.float32)
    check_all_boundaries(data[::283], 1, constant_boundary=2.5)
    check_all_boundaries(data, 100, constant_boundary=2.5)
    check_all_boundaries(data, 1738, constant_boundary=2.5)


def test_edge_cases():
    data = np.array(
        [
            np.inf,
            np.nan,
            -np.inf,
            -np.nan,
            np.finfo(float).min,
            np.finfo(float).max,
            np.finfo(float).smallest_normal,
            -np.finfo(float).smallest_normal,
            np.finfo(float).smallest_subnormal,
            -np.finfo(float).smallest_subnormal,
            0.0,
            -0.0,
        ]
    )
    check_all_boundaries(data, 1)
    check_all_boundaries(data, 5)


def test_xarray_accessors():
    da = xr.DataArray(np.linspace(0, 1), name="da").chunk(10)
    da_prediction = xr.DataArray(np.zeros_like(da.values), name="da").chunk(10)

    da_correction = produce_data_array_correction(
        da,
        da_prediction,
        safeguards=[dict(kind="eb", type="abs", eb=0.1)],
        late_bound=dict(),
    )
    assert tuple(s.get_config() for s in da_correction.safeguards) == (
        dict(kind="eb", type="abs", eb=0.1, equal_nan=False),
    )

    ds = xr.Dataset(dict(da=da_prediction, da_correction=da_correction))

    ds_safeguarded: xr.Dataset = ds.safeguarded

    np.testing.assert_allclose(ds_safeguarded["da"].values, da.values, rtol=0, atol=0.1)


def test_fuzzer_found_chunked_check_invalid_block_info():
    chunks = dict(a=1, b=2)
    da = xr.DataArray(
        np.array([[0, 0], [0, 0]], dtype=np.uint8), name="da", dims=["a", "b"]
    ).chunk(chunks)
    da_prediction = xr.DataArray(
        np.zeros_like(da.values), name="da", dims=["a", "b"]
    ).chunk(chunks)

    produce_data_array_correction(
        da,
        da_prediction,
        safeguards=[
            dict(
                kind="qoi_eb_stencil",
                qoi="x",
                neighbourhood=[dict(axis=-1, before=26, after=0, boundary="valid")],
                type="abs",
                eb="$x_min",
                qoi_dtype="float32",
            )
        ],
        late_bound=dict(),
    ).compute()


def test_fuzzer_found_correction_shape_mismatch():
    chunks = dict(a=2, b=1)
    da = xr.DataArray(
        np.array([[-547], [7421], [1015], [514], [0], [2], [-15868]], dtype=np.int16),
        name="da",
        dims=["a", "b"],
    ).chunk(chunks)
    da_prediction = xr.DataArray(
        np.zeros_like(da.values), name="da", dims=["a", "b"]
    ).chunk(chunks)

    produce_data_array_correction(
        da,
        da_prediction,
        safeguards=[
            dict(
                kind="qoi_eb_stencil",
                qoi="x",
                neighbourhood=[dict(axis=0, before=1, after=0, boundary="edge")],
                type="abs",
                eb=125,
                qoi_dtype="lossless",
            )
        ],
        late_bound=dict(),
    ).compute()


def test_fuzzer_found_hash():
    chunks = dict(a=14, b=1)
    da = xr.DataArray(
        np.array(
            [
                [194],
                [253],
                [0],
                [2],
                [221],
                [11],
                [62],
                [0],
                [62],
                [62],
                [62],
                [62],
                [42],
                [255],
                [255],
                [255],
                [255],
            ],
            dtype=np.uint8,
        ),
        name="da",
        dims=["a", "b"],
    ).chunk(chunks)
    da_prediction = xr.DataArray(
        np.ones_like(da.values), name="da", dims=["a", "b"]
    ).chunk(chunks)

    safeguard = StencilQuantityOfInterestErrorBoundSafeguard(
        qoi="x",
        neighbourhood=[dict(axis=0, before=1, after=0, boundary="edge")],
        type="abs",
        eb=0,
        qoi_dtype="float64",
    )
    safeguard._qoi_expr._expr = HashingExpr.from_data_shape(
        data_shape=safeguard._qoi_expr._stencil_shape, late_bound_constants=frozenset()
    )
    safeguard._qoi_expr._late_bound_constants = (
        safeguard._qoi_expr._expr.late_bound_constants
    )

    with _patch_for_hashing_qoi_dev_only():
        global_hash = Safeguards(safeguards=[safeguard]).compute_correction(
            data=da.values, prediction=da_prediction.values
        )
        chunked_hash = produce_data_array_correction(
            data=da,
            prediction=da_prediction,
            safeguards=[safeguard],
        )
        np.testing.assert_array_equal(chunked_hash.values, global_hash)


def test_fuzzer_found_hash_with_late_bound():
    chunks = dict(a=1, b=1)
    da = xr.DataArray(
        np.array(
            [[2.3694278e-38], [-1.0717227e37]],
            dtype=np.float32,
        ),
        name="da",
        dims=["a", "b"],
    ).chunk(chunks)
    da_prediction = xr.DataArray(
        np.ones_like(da.values), name="da", dims=["a", "b"]
    ).chunk(chunks)

    safeguard = StencilQuantityOfInterestErrorBoundSafeguard(
        qoi="x",
        neighbourhood=[
            dict(axis=1, before=1, after=64, boundary="constant", constant_boundary=4)
        ],
        type="ratio",
        eb=10,
        qoi_dtype="float32",
    )
    safeguard._qoi_expr._expr = HashingExpr.from_data_shape(
        data_shape=safeguard._qoi_expr._stencil_shape,
        late_bound_constants=frozenset(["foo"]),
    )
    safeguard._qoi_expr._late_bound_constants = (
        safeguard._qoi_expr._expr.late_bound_constants
    )

    late_bound = dict(foo=0)

    with _patch_for_hashing_qoi_dev_only():
        global_hash = Safeguards(safeguards=[safeguard]).compute_correction(
            data=da.values,
            prediction=da_prediction.values,
            late_bound=late_bound,
        )
        chunked_hash = produce_data_array_correction(
            data=da,
            prediction=da_prediction,
            safeguards=[safeguard],
            late_bound=late_bound,
        )
        np.testing.assert_array_equal(chunked_hash.values, global_hash)


def test_fuzzer_found_hash_x_max():
    chunks = dict(a=4, b=1)
    da = xr.DataArray(
        np.array(
            [
                [0],
                [0],
                [0],
                [-15],
                [-58],
                [64],
                [-15],
                [-15],
                [-15],
                [-15],
                [-38],
                [-15],
                [-7],
                [-15],
                [-15],
                [-15],
                [-15],
                [-15],
            ],
            dtype=np.int8,
        ),
        name="da",
        dims=["a", "b"],
    ).chunk(chunks)
    da_prediction = xr.DataArray(
        np.ones_like(da.values), name="da", dims=["a", "b"]
    ).chunk(chunks)

    safeguard = StencilQuantityOfInterestErrorBoundSafeguard(
        qoi="x",
        neighbourhood=[
            dict(
                axis=0,
                before=1,
                after=0,
                boundary="constant",
                constant_boundary="$x_max",
            )
        ],
        type="abs",
        eb=0,
        qoi_dtype="float128",
    )
    safeguard._qoi_expr._expr = HashingExpr.from_data_shape(
        data_shape=safeguard._qoi_expr._stencil_shape,
        late_bound_constants=frozenset(["foo"]),
    )
    safeguard._qoi_expr._late_bound_constants = (
        safeguard._qoi_expr._expr.late_bound_constants
    )

    late_bound = dict(foo=-15)

    with _patch_for_hashing_qoi_dev_only():
        global_hash = Safeguards(safeguards=[safeguard]).compute_correction(
            data=da.values,
            prediction=da_prediction.values,
            late_bound={**late_bound, "$x_max": np.amax(da.values)},
        )
        chunked_hash = produce_data_array_correction(
            data=da,
            prediction=da_prediction,
            safeguards=[safeguard],
            late_bound=late_bound,
        )
        np.testing.assert_array_equal(chunked_hash.values, global_hash)


def test_fuzzer_found_hash_reflect_boundary():
    chunks = dict(a=1, b=2)
    da = xr.DataArray(
        np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.int16),
        name="da",
        dims=["a", "b"],
    ).chunk(chunks)
    da_prediction = xr.DataArray(
        np.ones_like(da.values), name="da", dims=["a", "b"]
    ).chunk(chunks)

    # in this case, we have the following data in the rightmost column
    #  9, 6, |3, 6, 9|
    # based on the chunking, we currently only work with the following data
    #  6, 9|
    # which is extended by the safeguards to
    #  9, |6, 9|
    # so even though 9 is not supposed to influence 6, it now does
    # therefore, we need a buffer

    safeguard = StencilQuantityOfInterestErrorBoundSafeguard(
        qoi="x",
        neighbourhood=[
            dict(
                axis=0,
                before=1,
                after=0,
                boundary="reflect",
            )
        ],
        type="abs",
        eb=0,
        qoi_dtype="float128",
    )
    safeguard._qoi_expr._expr = HashingExpr.from_data_shape(
        data_shape=safeguard._qoi_expr._stencil_shape,
        late_bound_constants=frozenset(["foo"]),
    )
    safeguard._qoi_expr._late_bound_constants = (
        safeguard._qoi_expr._expr.late_bound_constants
    )

    late_bound = dict(foo=0)

    with _patch_for_hashing_qoi_dev_only():
        global_hash = Safeguards(safeguards=[safeguard]).compute_correction(
            data=da.values,
            prediction=da_prediction.values,
            late_bound=late_bound,
        )
        chunked_hash = produce_data_array_correction(
            data=da,
            prediction=da_prediction,
            safeguards=[safeguard],
            late_bound=late_bound,
        )
        np.testing.assert_array_equal(chunked_hash.values, global_hash)


def test_fuzzer_found_all_nan_xmin():
    chunks = dict(a=1, b=1)
    da = xr.DataArray(
        np.array([[np.nan], [np.nan]], dtype=np.float16),
        name="da",
        dims=["a", "b"],
    ).chunk(chunks)
    da_prediction = xr.DataArray(
        np.zeros_like(da.values), name="da", dims=["a", "b"]
    ).chunk(chunks)

    produce_data_array_correction(
        da,
        da_prediction,
        safeguards=[
            dict(
                kind="qoi_eb_stencil",
                qoi="x",
                neighbourhood=[dict(axis=0, before=0, after=0, boundary="reflect")],
                type="abs",
                eb="$x_min",
                qoi_dtype="float128",
            )
        ],
        late_bound=dict(),
    ).compute()


def test_stencil_qoi_valid_boundary_late_bound_eb():
    chunks = dict(a=9)
    da = xr.DataArray(np.linspace(0, 10), name="da", dims=["a"]).chunk(chunks)
    da_prediction = xr.DataArray(np.zeros_like(da.values), name="da", dims=["a"]).chunk(
        chunks
    )

    produce_data_array_correction(
        da,
        da_prediction,
        safeguards=[
            dict(
                kind="qoi_eb_stencil",
                qoi="finite_difference(x, order=3, accuracy=4, type=0, axis=0, grid_spacing=1)",
                neighbourhood=[dict(axis=0, before=3, after=3, boundary="valid")],
                type="rel",
                eb="eb",
            )
        ],
        late_bound=dict(eb=xr.DataArray(np.full(da.shape, 0.1), name="eb", dims=["a"])),
    ).compute()


def test_stencil_qoi_valid_boundary_late_bound_dimension():
    for chunks in [None, dict(), dict(a=9)]:
        da = xr.DataArray(
            np.linspace(0, 10, num=50), name="da", coords=dict(a=np.arange(50))
        )
        da_prediction = xr.DataArray(
            np.zeros_like(da.values), name="da", coords=dict(a=np.arange(50))
        )

        if chunks is not None:
            da = da.chunk(chunks)
            da_prediction = da_prediction.chunk(chunks)

        produce_data_array_correction(
            da,
            da_prediction,
            safeguards=[
                dict(
                    kind="qoi_eb_stencil",
                    qoi='x + sum(C["$d_a"])',
                    neighbourhood=[dict(axis=0, before=1, after=1, boundary="valid")],
                    type="rel",
                    eb="$d_a",
                )
            ],
            late_bound=dict(),
        ).compute()


def test_rechunk_correction():
    chunks = dict(a=9)
    da = xr.DataArray(np.linspace(0, 10), name="da", dims=["a"]).chunk(chunks)
    da_prediction = xr.DataArray(np.zeros_like(da.values), name="da", dims=["a"]).chunk(
        chunks
    )

    da_correction = produce_data_array_correction(
        da,
        da_prediction,
        safeguards=[
            dict(
                kind="qoi_eb_stencil",
                qoi="finite_difference(x, order=3, accuracy=4, type=0, axis=0, grid_spacing=1)",
                neighbourhood=[dict(axis=0, before=3, after=3, boundary="valid")],
                type="rel",
                eb="eb",
            )
        ],
        late_bound=dict(eb=xr.DataArray(np.full(da.shape, 0.1), name="eb", dims=["a"])),
    )

    rechunks = dict(a=7)

    da_corrected = apply_data_array_correction(da_prediction, da_correction)
    da_corrected_rechunked = apply_data_array_correction(
        da_prediction.chunk(rechunks), da_correction.chunk(rechunks)
    )

    np.testing.assert_array_equal(da_corrected_rechunked.values, da_corrected.values)


def test_different_chunks_and_safeguards():
    for chunks in [None, dict(), dict(a=9)]:
        da = xr.DataArray(np.linspace(0, 10), name="da", dims=["a"])
        da_prediction = xr.DataArray(np.zeros_like(da.values), name="da", dims=["a"])

        if chunks is not None:
            da = da.chunk(chunks)
            da_prediction = da_prediction.chunk(chunks)

        for safeguards in [
            [],
            [dict(kind="same", value=10)],
            [
                dict(kind="same", value=10),
                dict(
                    kind="monotonicity",
                    monotonicity="strict",
                    window=2,
                    boundary="valid",
                ),
            ],
            [
                dict(kind="same", value=10),
                dict(
                    kind="monotonicity",
                    monotonicity="strict",
                    window=2,
                    boundary="valid",
                ),
                dict(
                    kind="monotonicity", monotonicity="weak", window=1, boundary="wrap"
                ),
            ],
        ]:
            produce_data_array_correction(
                da,
                da_prediction,
                safeguards=safeguards,
                late_bound=dict(),
            ).compute()


def test_fuzzer_found_global_late_bound_max():
    chunks = dict(a=3, b=3)
    da = xr.DataArray(
        np.array(
            [
                [-1.7014118e38, -7.7297928e-37, -3.7617555e-37],
                [-7.7296816e-37, -4.2360169e09, 2.8973772e-14],
                [np.nan, np.nan, 8.6081765e-42],
            ],
            dtype=np.float32,
        ),
        name="da",
        dims=["a", "b"],
    ).chunk(chunks)
    da_prediction = xr.DataArray(
        np.ones_like(da.values), name="da", dims=["a", "b"]
    ).chunk(chunks)

    safeguard = StencilQuantityOfInterestErrorBoundSafeguard(
        qoi="x",
        neighbourhood=[
            dict(
                axis=1,
                before=0,
                after=61,
                boundary="constant",
                constant_boundary="$x_max",
            )
        ],
        type="rel",
        eb="$x_max",
        qoi_dtype="lossless",
    )
    safeguard._qoi_expr._expr = HashingExpr.from_data_shape(
        data_shape=safeguard._qoi_expr._stencil_shape,
        late_bound_constants=frozenset(["foo"]),
    )
    safeguard._qoi_expr._late_bound_constants = (
        safeguard._qoi_expr._expr.late_bound_constants
    )

    late_bound = dict(foo=0)

    with _patch_for_hashing_qoi_dev_only():
        global_hash = Safeguards(safeguards=[safeguard]).compute_correction(
            data=da.values,
            prediction=da_prediction.values,
            late_bound={**late_bound, "$x_max": np.nanmax(da.values)},
        )
        chunked_hash = produce_data_array_correction(
            data=da,
            prediction=da_prediction,
            safeguards=[safeguard],
            late_bound=late_bound,
        )
        np.testing.assert_array_equal(chunked_hash.values, global_hash)


def test_fuzzer_found_wrapping_constant_boundary_clash():
    chunks = dict(a=1, b=3)
    da = xr.DataArray(
        np.array([[0.0, 0.0, 0.0664, -0.05594, -0.0599]], dtype=np.float16),
        name="da",
        dims=["a", "b"],
    ).chunk(chunks)
    da_prediction = xr.DataArray(
        np.ones_like(da.values), name="da", dims=["a", "b"]
    ).chunk(chunks)

    # FIXME: wrapping boundary in case (b), only one overlap with the global
    #        boundary, needs to roll the wrapped values around to the other
    #        side so that other boundary conditions still see a global boundary

    safeguard = StencilQuantityOfInterestErrorBoundSafeguard(
        qoi="x",
        neighbourhood=[
            dict(
                axis=1,
                before=1,
                after=0,
                boundary="constant",
                constant_boundary="$x_max",
            ),
            dict(
                axis=0,
                before=0,
                after=0,
                boundary="wrap",
            ),
        ],
        type="rel",
        eb=0,
        qoi_dtype="lossless",
    )
    safeguard._qoi_expr._expr = HashingExpr.from_data_shape(
        data_shape=safeguard._qoi_expr._stencil_shape,
        late_bound_constants=frozenset(["foo"]),
    )
    safeguard._qoi_expr._late_bound_constants = (
        safeguard._qoi_expr._expr.late_bound_constants
    )

    late_bound = dict(foo=0)

    with _patch_for_hashing_qoi_dev_only():
        global_hash = Safeguards(safeguards=[safeguard]).compute_correction(
            data=da.values,
            prediction=da_prediction.values,
            late_bound={**late_bound, "$x_max": np.nanmax(da.values)},
        )
        chunked_hash = produce_data_array_correction(
            data=da,
            prediction=da_prediction,
            safeguards=[safeguard],
            late_bound=late_bound,
        )
        np.testing.assert_array_equal(chunked_hash.values, global_hash)
