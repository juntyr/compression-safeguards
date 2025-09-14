import numpy as np
import xarray as xr
from xarray_safeguards import produce_data_array_correction

from compression_safeguards.api import Safeguards
from compression_safeguards.safeguards._qois.expr.hashing import (
    HashingExpr,
    _patch_for_hashing_qoi_dev_only,
)
from compression_safeguards.safeguards.stencil.qoi.eb import (
    StencilQuantityOfInterestErrorBoundSafeguard,
)


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
        np.zeros_like(da.values), name="da", dims=["a", "b"]
    ).chunk(chunks)

    safeguard = StencilQuantityOfInterestErrorBoundSafeguard(
        qoi="x",
        neighbourhood=[dict(axis=0, before=1, after=0, boundary="edge")],
        type="abs",
        eb=0,
        qoi_dtype="float64",
    )
    safeguard._qoi_expr._expr = HashingExpr(
        data_indices=frozenset([(1,)]), late_bound_constants=frozenset()
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
        np.testing.assert_array_equal(global_hash, chunked_hash.values)


def test_fuzzer_found_hash_v2():
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
        np.zeros_like(da.values), name="da", dims=["a", "b"]
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
    safeguard._qoi_expr._expr = HashingExpr(
        data_indices=frozenset([(1,)]), late_bound_constants=frozenset()
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
        np.testing.assert_array_equal(global_hash, chunked_hash.values)
