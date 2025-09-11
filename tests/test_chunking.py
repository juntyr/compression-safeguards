import numpy as np
import xarray as xr
from xarray_safeguards import produce_data_array_correction


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


def test_fuzzer_found_negative_slice():
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
    )
