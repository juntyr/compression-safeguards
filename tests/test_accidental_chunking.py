import sys

import numpy as np
import pytest
import xarray as xr
from numcodecs_combinators.stack import CodecStack


def test_codec_stack_pointwise():
    stack = CodecStack(
        dict(id="fixedscaleoffset", offset=0.0, scale=1.0, dtype=float),
        dict(
            id="safeguards",
            codec=dict(id="zero"),
            safeguards=[
                dict(
                    kind="eb",
                    type="abs",
                    eb=0.5,
                )
            ],
        ),
    )

    data = np.arange(100, dtype=float)

    encoded = stack.encode(data)
    decoded = stack.decode(encoded)
    assert np.all(np.abs(decoded - data) <= 0.5)

    encoded_decoded = stack.encode_decode(data)
    assert np.all(encoded_decoded == decoded)

    encoded_decoded_da = stack.encode_decode_data_array(xr.DataArray(data))
    assert np.all(encoded_decoded_da.values == decoded)

    encoded_decoded_da = stack.encode_decode_data_array(
        xr.DataArray(data).chunk(10)
    ).compute()
    assert np.all(encoded_decoded_da.values == decoded)


def test_codec_stack_stencil():
    stack = CodecStack(
        dict(id="fixedscaleoffset", offset=0.0, scale=1.0, dtype=float),
        dict(
            id="safeguards",
            codec=dict(id="zero"),
            safeguards=[
                dict(
                    kind="monotonicity",
                    monotonicity="strict",
                    window=1,
                    boundary="valid",
                )
            ],
        ),
    )

    data = np.arange(100, dtype=float)

    encoded = stack.encode(data)
    decoded = stack.decode(encoded)
    assert np.all(decoded[2:] > decoded[:-2])

    encoded_decoded = stack.encode_decode(data)
    assert np.all(encoded_decoded == decoded)

    encoded_decoded_da = stack.encode_decode_data_array(xr.DataArray(data))
    assert np.all(encoded_decoded_da.values == decoded)

    with pytest.raises(AssertionError, match="chunked array"):
        encoded_decoded_da = stack.encode_decode_data_array(
            xr.DataArray(data).chunk(10)
        ).compute()


@pytest.mark.skipif(
    sys.version_info < (3, 11), reason="zarr v3 requires >= Python 3.11"
)
def test_zarr_pointwise():
    import zarr

    data = np.arange(100, dtype=float)

    store = zarr.storage.MemoryStore()

    zarr.save_array(
        store,
        data,
        codecs=[
            dict(
                name="any-numcodecs.array-bytes",
                configuration=dict(
                    id="safeguards",
                    codec=dict(id="zero"),
                    safeguards=[
                        dict(
                            kind="eb",
                            type="abs",
                            eb=0.5,
                        )
                    ],
                ),
            ),
        ],
    )

    a = zarr.open_array(store)

    assert np.all(np.abs(np.asarray(a) - data) <= 0.5)


@pytest.mark.skipif(
    sys.version_info < (3, 11), reason="zarr v3 requires >= Python 3.11"
)
def test_zarr_stencil():
    import zarr

    data = np.arange(100, dtype=float)

    store = zarr.storage.MemoryStore()

    with pytest.raises(AssertionError, match="chunked array"):
        zarr.save_array(
            store,
            data,
            codecs=[
                dict(
                    name="any-numcodecs.array-bytes",
                    configuration=dict(
                        id="safeguards",
                        codec=dict(id="zero"),
                        safeguards=[
                            dict(
                                kind="monotonicity",
                                monotonicity="strict",
                                window=1,
                                boundary="valid",
                            )
                        ],
                    ),
                ),
            ],
        )
