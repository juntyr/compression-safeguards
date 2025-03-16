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
    yield "+t2m", ds.t2m.values
    yield "-t2m", -ds.t2m.values

    yield (
        "N(0,10)",
        np.random.default_rng(seed=42).normal(loc=0.0, scale=10.0, size=ds.t2m.shape),
    )

    yield "+t2mi", np.round(ds.t2m.values * 1000).astype(int)
    yield "-t2mi", np.round(-ds.t2m.values * 1000).astype(int)

    yield (
        "N(0,10)i",
        np.round(
            np.random.default_rng(seed=42).normal(
                loc=0.0, scale=10.0, size=ds.t2m.shape
            )
            * 1000
        ).astype(int),
    )


def gen_codecs_with_eb_abs(
    make_codec: Callable[[float], Codec],
) -> Callable[[Generator[float]], Generator[Codec]]:
    def gen_codecs_with_eb_abs_inner(eb_abs: Generator[float]) -> Generator[Codec]:
        for eb_abs in eb_abs:
            yield make_codec(eb_abs)

    return gen_codecs_with_eb_abs_inner


def gen_single_codec(codec: Codec) -> Callable[[Generator[float]], Generator[Codec]]:
    def gen_single_codec_inner(_eb_abs: Generator[float]) -> Generator[Codec]:
        yield codec

    return gen_single_codec_inner


def gen_benchmark(
    data: Generator[tuple[str, np.ndarray], None, None],
    codec_gens: list[Callable[[Generator[float]], Generator[Codec]], None, None],
) -> Generator[tuple[str, Codec, float | Exception], None, None]:
    for d, datum in data:
        for codec_gen in codec_gens:
            if np.issubdtype(datum.dtype, np.floating):
                eb_abs = [1.0, 0.1, 0.01, 0.001]
            else:
                eb_abs = [1000, 100, 10, 1]

            for codec in codec_gen(iter(eb_abs)):
                try:
                    compressed = codec.encode(datum)
                except Exception as err:
                    yield d, codec, err
                else:
                    yield d, codec, datum.nbytes / np.asarray(compressed).nbytes


if __name__ == "__main__":
    for d, codec, result in gen_benchmark(
        gen_data(),
        [
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
        ],
    ):
        print(f"- {d} {codec}: {result}")
