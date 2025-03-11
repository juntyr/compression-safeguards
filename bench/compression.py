from collections.abc import Generator
from pathlib import Path
from typing import Callable

import numpy as np
import xarray as xr
from numcodecs.abc import Codec
from numcodecs_safeguards import SafeguardsCodec
from numcodecs_wasm_sz3 import Sz3
from numcodecs_wasm_zfp import Zfp
from numcodecs_wasm_zlib import Zlib
from numcodecs_wasm_zstd import Zstd


def gen_data() -> Generator[tuple[str, np.ndarray], None, None]:
    ds: xr.Dataset = xr.open_dataset(
        Path(__file__) / ".." / "era5_t2m_2012_12_01_14:00.nc", engine="netcdf4"
    )
    yield "t2m", ds.t2m.values

    yield (
        "N(0,10)",
        np.random.default_rng(seed=42).normal(loc=0.0, scale=10.0, size=ds.t2m.shape),
    )


def gen_codecs_with_eb_abs(
    make_codec: Callable[[float], Codec],
) -> Generator[Codec, None, None]:
    for eb_abs in [1.0, 0.1, 0.01, 0.001]:
        yield make_codec(eb_abs)


def gen_single_codec(codec: Codec) -> Generator[Codec, None, None]:
    yield codec


def gen_codec(
    data: Generator[tuple[str, np.ndarray], None, None],
    codecs: Generator[Codec, None, None],
) -> Generator[tuple[str, Codec, float | Exception], None, None]:
    codec_list = list(codecs)

    for d, datum in data:
        for codec in codec_list:
            try:
                compressed = codec.encode(datum)
            except Exception as err:
                yield d, codec, err
            else:
                yield d, codec, datum.nbytes / np.asarray(compressed).nbytes


def gen_concat(*args: Generator) -> Generator:
    for a in args:
        yield from a


if __name__ == "__main__":
    for d, codec, result in gen_codec(
        gen_data(),
        gen_concat(
            gen_codecs_with_eb_abs(lambda eb_abs: Sz3(eb_mode="abs", eb_abs=eb_abs)),
            gen_codecs_with_eb_abs(
                lambda eb_abs: Sz3(
                    eb_mode="abs", eb_abs=eb_abs, predictor="linear-interpolation"
                )
            ),
            gen_codecs_with_eb_abs(
                lambda eb_abs: Sz3(eb_mode="abs", eb_abs=eb_abs, predictor=None)
            ),
            # FIXME: https://github.com/szcompressor/SZ3/issues/78
            # gen_codecs_with_eb_abs(
            #     lambda eb_abs: Sz3(eb_mode="abs", eb_abs=eb_abs, encoder=None)
            # ),
            # FIXME: https://github.com/szcompressor/SZ3/issues/78
            # gen_codecs_with_eb_abs(
            #     lambda eb_abs: Sz3(eb_mode="abs", eb_abs=eb_abs, lossless=None)
            # ),
            gen_codecs_with_eb_abs(
                lambda eb_abs: Zfp(mode="fixed-accuracy", tolerance=eb_abs)
            ),
            gen_single_codec(Zlib(level=9)),
            gen_single_codec(Zstd(level=20)),
            gen_codecs_with_eb_abs(
                lambda eb_abs: SafeguardsCodec(
                    codec=None, safeguards=[dict(kind="abs", eb_abs=eb_abs)]
                )
            ),
        ),
    ):
        print(f"- {d} {codec}: {result}")
