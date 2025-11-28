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

    encoded_decoded_da = stack.encode_decode_data_array(xr.DataArray(data, name="da"))
    assert np.all(encoded_decoded_da.values == decoded)

    encoded_decoded_da = stack.encode_decode_data_array(
        xr.DataArray(data, name="da").chunk(10)
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
                    kind="qoi_eb_stencil",
                    qoi="""
                        all([
                            # strictly decreasing sequences stay strictly decreasing
                            any([all(X[1:] < X[:-1]), not(all(C["$X"][1:] < C["$X"][:-1]))]),
                            # strictly increasing sequences stay strictly increasing
                            any([all(X[1:] > X[:-1]), not(all(C["$X"][1:] > C["$X"][:-1]))]),
                        ])
                    """,
                    neighbourhood=[dict(axis=0, before=1, after=1, boundary="valid")],
                    type="abs",
                    eb=0,
                )
            ],
        ),
    )

    data = np.arange(100, dtype=float)

    encoded = stack.encode(data)
    decoded = stack.decode(encoded)
    assert np.all(decoded[1:] > decoded[:-1])

    encoded_decoded = stack.encode_decode(data)
    assert np.all(encoded_decoded == decoded)

    encoded_decoded_da = stack.encode_decode_data_array(xr.DataArray(data, name="da"))
    assert np.all(encoded_decoded_da.values == decoded)

    with pytest.raises(RuntimeError, match="chunked array"):
        encoded_decoded_da = stack.encode_decode_data_array(
            xr.DataArray(data, name="da").chunk(10)
        ).compute()


@pytest.mark.skipif(
    sys.version_info < (3, 11), reason="zarr v3 requires >= Python 3.11"
)
def test_zarr_pointwise():
    import zarr  # noqa: PLC0415

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
    import zarr  # noqa: PLC0415

    data = np.arange(100, dtype=float)

    store = zarr.storage.MemoryStore()

    with pytest.raises(RuntimeError, match="chunked array"):
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
                                kind="qoi_eb_stencil",
                                qoi="""
                                    all([
                                        # strictly decreasing sequences stay strictly decreasing
                                        any([all(X[1:] < X[:-1]), not(all(C["$X"][1:] < C["$X"][:-1]))]),
                                        # strictly increasing sequences stay strictly increasing
                                        any([all(X[1:] > X[:-1]), not(all(C["$X"][1:] > C["$X"][:-1]))]),
                                    ])
                                """,
                                neighbourhood=[
                                    dict(axis=0, before=1, after=1, boundary="valid")
                                ],
                                type="abs",
                                eb=0,
                            )
                        ],
                    ),
                ),
            ],
        )
