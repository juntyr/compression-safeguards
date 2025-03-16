from abc import ABC, abstractmethod
from collections.abc import Generator
from pathlib import Path
from typing import Callable

import numcodecs
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


class Quantizer(ABC):
    @abstractmethod
    def encoded_dtype(self, dtype: np.dtype) -> np.dtype:
        pass

    @abstractmethod
    def encode(self, x, predict):
        pass

    @abstractmethod
    def decode(self, e, predict):
        pass


class RoundQuantizer(Quantizer):
    def __init__(self, eb_abs):
        self._eb_abs = eb_abs

    def encoded_dtype(self, dtype: np.dtype) -> np.dtype:
        return np.dtype(int)

    @abstractmethod
    def encode(self, x, predict):
        return np.round((x - predict) / self._eb_abs)

    @abstractmethod
    def decode(self, e, predict):
        return predict + e * self._eb_abs


class SafeguardQuantizer(Quantizer):
    def __init__(self, eb_abs):
        from numcodecs_safeguards.safeguards.elementwise.abs import (
            AbsoluteErrorBoundSafeguard,
        )

        self._safeguard = AbsoluteErrorBoundSafeguard(eb_abs=eb_abs)

    def encoded_dtype(self, dtype: np.dtype) -> np.dtype:
        return np.dtype(dtype.str.replace("f", "u").replace("i", "u"))

    @abstractmethod
    def encode(self, x, predict):
        from numcodecs_safeguards.intervals import _as_bits

        encoded = self._safeguard.compute_safe_intervals(np.array(x)).encode(
            np.array(predict)
        )[()]
        return _as_bits(np.array(predict)) - _as_bits(np.array(encoded))

    @abstractmethod
    def decode(self, e, predict):
        from numcodecs_safeguards.intervals import _as_bits

        return (_as_bits(np.array(predict)) - _as_bits(np.array(e))).view(
            np.array(predict).dtype
        )[()]


class Lorenzo2dPredictor(Codec):
    codec_id = "lorenzo"

    def __init__(self, quantizer: Quantizer):
        self._quantizer = quantizer
        self._lossless = numcodecs.zstd.Zstd(level=3)

    def encode(self, buf):
        from tqdm import tqdm

        from numcodecs_safeguards.safeguards.elementwise import _runlength_encode

        data = np.asarray(buf).squeeze()
        assert len(data.shape) == 2
        M, N = data.shape

        encoded = np.zeros(
            shape=data.shape, dtype=self._quantizer.encoded_dtype(data.dtype)
        )
        decoded = np.zeros_like(data)

        if data.size > 0:
            encoded[0, 0] = self._encode(data[0, 0], 0.0)
            decoded[0, 0] = self._decode(encoded[0, 0], 0.0)

            for i in range(1, N):
                predict = decoded[0, i - 1]
                encoded[0, i] = self._encode(data[0, i], predict)
                decoded[0, i] = self._decode(encoded[0, i], predict)

            for j in tqdm(range(1, M)):
                predict = decoded[j - 1, 0]
                encoded[j, 0] = self._encode(data[j, 0], predict)
                decoded[j, 0] = self._decode(encoded[j, 0], predict)

                for i in range(1, N):
                    predict = (
                        decoded[j, i - 1] + decoded[j - 1, i] - decoded[j - 1, i - 1]
                    )
                    encoded[j, i] = self._encode(data[j, i], predict)
                    decoded[j, i] = self._decode(encoded[j, i], predict)

        print(
            len(np.unique(encoded)),
            np.unique(encoded),
            np.count_nonzero(encoded),
            encoded.size,
        )

        encoded = _runlength_encode(encoded)

        return self._lossless.encode(encoded)

    def decode(self, buf, out=None):
        from tqdm import tqdm

        from numcodecs_safeguards.safeguards.elementwise import _runlength_decode

        assert out is not None

        decoded = np.asarray(out).squeeze()
        assert len(decoded.shape) == 2
        M, N = decoded.shape

        encoded = self._lossless.decode(buf)
        encoded = _runlength_decode(encoded, like=decoded)

        if decoded.size > 0:
            decoded[0, 0] = encoded[0, 0] * self.eb_abs

            for i in range(1, N):
                predict = decoded[0, i - 1]
                decoded[0, i] = predict + encoded[0, i] * self.eb_abs

            for j in tqdm(range(1, M)):
                predict = decoded[j - 1, 0]
                decoded[j, 0] = predict + encoded[j, 0] * self.eb_abs

                for i in range(1, N):
                    predict = (
                        decoded[j, i - 1] + decoded[j - 1, i] - decoded[j - 1, i - 1]
                    )
                    decoded[j, i] = predict + encoded[j, i] * self.eb_abs

        return numcodecs.compat.ndarray_copy(decoded.reshape(out.shape), out)


if __name__ == "__main__":
    for d, codec, result in gen_benchmark(
        gen_data(),
        [
            gen_codecs_with_eb_abs(lambda eb_abs: Sz3(eb_mode="abs", eb_abs=eb_abs)),
            # gen_codecs_with_eb_abs(
            #     lambda eb_abs: Sz3(
            #         eb_mode="abs", eb_abs=eb_abs, predictor="linear-interpolation"
            #     )
            # ),
            # gen_codecs_with_eb_abs(
            #     lambda eb_abs: Sz3(eb_mode="abs", eb_abs=eb_abs, predictor=None)
            # ),
            # FIXME: https://github.com/szcompressor/SZ3/issues/78
            # gen_codecs_with_eb_abs(
            #     lambda eb_abs: Sz3(eb_mode="abs", eb_abs=eb_abs, encoder=None)
            # ),
            # FIXME: https://github.com/szcompressor/SZ3/issues/78
            # gen_codecs_with_eb_abs(
            #     lambda eb_abs: Sz3(eb_mode="abs", eb_abs=eb_abs, lossless=None)
            # ),
            # gen_codecs_with_eb_abs(
            #     lambda eb_abs: Zfp(mode="fixed-accuracy", tolerance=eb_abs)
            # ),
            # gen_single_codec(Zlib(level=9)),
            # gen_single_codec(Zstd(level=20)),
            gen_codecs_with_eb_abs(
                lambda eb_abs: SafeguardsCodec(
                    codec=None, safeguards=[dict(kind="abs", eb_abs=eb_abs)]
                )
            ),
            gen_codecs_with_eb_abs(
                lambda eb_abs: Lorenzo2dPredictor(quantizer=RoundQuantizer(eb_abs=eb_abs))
            ),
            gen_codecs_with_eb_abs(
                lambda eb_abs: Lorenzo2dPredictor(
                    quantizer=SafeguardQuantizer(eb_abs=eb_abs)
                )
            ),
        ],
    ):
        print(f"- {d} {codec}: {result}")
