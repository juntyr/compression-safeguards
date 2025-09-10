import numpy as np
import xarray as xr
from xarray_safeguards import produce_data_array_correction


def test_xarray_accessors():
    da = xr.DataArray(np.linspace(0, 1), name="da")
    da_prediction = xr.DataArray(np.zeros_like(da.values), name="da")

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
