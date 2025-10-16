import numpy as np
import pytest
import xarray as xr
from numcodecs_safeguards import SafeguardsCodec
from xarray_safeguards import apply_data_array_correction, produce_data_array_correction


def test_direct_wrap():
    SafeguardsCodec(codec=dict(id="zero"), safeguards=[])

    with pytest.raises(ValueError, match="printer problem"):
        SafeguardsCodec(
            codec=SafeguardsCodec(codec=dict(id="zero"), safeguards=[]), safeguards=[]
        )


def test_codec_stack_wrap():
    with pytest.raises(ValueError, match="printer problem"):
        SafeguardsCodec(
            codec=dict(
                id="combinators.stack",
                codecs=[
                    dict(id="bitround", keepbits=10),
                    SafeguardsCodec(codec=dict(id="zero"), safeguards=[]),
                ],
            ),
            safeguards=[],
        )


def test_xarray_repeated_safeguarding():
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

    da_corrected = apply_data_array_correction(da_prediction, da_correction)
    np.testing.assert_allclose(da_corrected.values, da.values, rtol=0, atol=0.1)

    with pytest.raises(ValueError, match="printer problem"):
        produce_data_array_correction(
            da_corrected,
            da_prediction,
            safeguards=[dict(kind="eb", type="abs", eb=0.1)],
            late_bound=dict(),
        )

    with pytest.raises(ValueError, match="printer problem"):
        produce_data_array_correction(
            da,
            da_corrected,
            safeguards=[dict(kind="eb", type="abs", eb=0.1)],
            late_bound=dict(),
        )

    with pytest.warns(UserWarning, match="allow_unsafe_safeguards_override"):
        produce_data_array_correction(
            da,
            da_corrected,
            safeguards=[dict(kind="eb", type="abs", eb=0.1)],
            late_bound=dict(),
            allow_unsafe_safeguards_override=True,
        )
