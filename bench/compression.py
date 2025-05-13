from abc import ABC, abstractmethod
from collections.abc import Generator
from pathlib import Path
from types import SimpleNamespace
from typing import Callable

import numcodecs
import numcodecs.compat
import numpy as np
import pandas as pd
import xarray as xr
from numcodecs.abc import Codec
from numcodecs_wasm_sz3 import Sz3
from numcodecs_wasm_zfp import Zfp
from numcodecs_wasm_zstd import Zstd
from tqdm import tqdm

from numcodecs_safeguards.cast import as_bits
from numcodecs_safeguards.codec import SafeguardsCodec
from numcodecs_safeguards.lossless import Lossless
from numcodecs_safeguards.quantizer import SafeguardsQuantizer
from numcodecs_safeguards.safeguards.pointwise.abs import AbsoluteErrorBoundSafeguard


def gen_data() -> Generator[tuple[str, np.ndarray], None, None]:
    t2m: xr.DataArray = xr.open_dataset(
        Path(__file__) / ".." / "era5_t2m_2012_12_01_14:00.nc", engine="netcdf4"
    ).t2m

    tp: xr.DataArray = xr.open_dataset(
        Path(__file__) / ".." / "era5_tp_2024_08_02_10:00.nc", engine="netcdf4"
    ).tp

    o3: xr.DataArray = xr.open_dataset(
        Path(__file__) / ".." / "era5_o3_pv_2024_08_02_12:00.nc", engine="netcdf4"
    ).o3

    yield "t2m-1d", t2m.values.flatten()
    yield "tp-1d", tp.values.flatten() * 100  # [cm]

    yield "+t2m", t2m.values
    yield "-t2m", -t2m.values

    yield "+tp", tp.values * 100  # [cm]
    yield "-tp", -tp.values * 100  # [cm]

    yield "+o3", o3.values * 1e6  # mg/mg
    yield "-o3", -o3.values * 1e6  # mg/mg

    yield (
        "N(0,10)",
        np.random.default_rng(seed=42)
        .normal(loc=0.0, scale=10.0, size=t2m.shape)
        .astype(t2m.dtype),
    )

    yield "+t2mi", np.round(t2m.values * 1000).astype(np.int32)
    yield "-t2mi", np.round(-t2m.values * 1000).astype(np.int32)

    yield "+tpi", np.round(tp.values * 100 * 1000).astype(np.int32)
    yield "-tpi", np.round(-tp.values * 100 * 1000).astype(np.int32)

    yield "+o3i", np.round(o3.values * 1e6 * 1000).astype(np.int32)  # mg/mg
    yield "-o3i", np.round(-o3.values * 1e6 * 1000).astype(np.int32)  # mg/mg

    yield (
        "N(0,10)i",
        np.round(
            np.random.default_rng(seed=42).normal(loc=0.0, scale=10.0, size=t2m.shape)
            * 1000
        ).astype(np.int32),
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
    # from matplotlib import pyplot as plt

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
                    # decompressed = codec.decode(compressed, out=np.empty_like(datum))
                    # fig, ax = plt.subplots()
                    # ax.hist((decompressed - datum).flatten())
                    # plt.show()
                    yield d, codec, datum.nbytes / np.asarray(compressed).nbytes


class MyQuantizer(ABC):
    @abstractmethod
    def encoded_dtype(self, dtype: np.dtype) -> np.dtype:
        pass

    @abstractmethod
    def encode(self, x, predict):
        pass

    @abstractmethod
    def decode(self, e, predict):
        pass


class MyLinearQuantizer(MyQuantizer):
    def __init__(self, eb_abs):
        self._eb_abs = eb_abs

    def encoded_dtype(self, dtype: np.dtype) -> np.dtype:
        return np.dtype(int)

    def encode(self, x, predict):
        return np.round((x - predict) / (self._eb_abs * 2))

    def decode(self, e, predict):
        return predict + e * self._eb_abs * 2

    def __repr__(self) -> str:
        return f"{type(self).__name__}(eb_abs={self._eb_abs})"


class MySafeguardsQuantizer(MyQuantizer):
    def __init__(self, eb_abs):
        self._quantizer = SafeguardsQuantizer(
            safeguards=[AbsoluteErrorBoundSafeguard(eb_abs=eb_abs)]
        )

    def encoded_dtype(self, dtype: np.dtype) -> np.dtype:
        return np.dtype(dtype.str.replace("f", "u").replace("i", "u"))

    def encode(self, x, predict):
        return self._quantizer.quantize(np.array(x), np.array(predict))[()]

    def decode(self, e, predict):
        return self._quantizer.recover(np.array(predict), np.array(e))[()]

    def __repr__(self) -> str:
        return f"{type(self).__name__}(safeguards={list(self._quantizer.safeguards)!r})"


class LorenzoPredictor(Codec):
    codec_id = "lorenzo"

    def __init__(self, quantizer: MyQuantizer):
        self._quantizer = quantizer
        self._lossless = Lossless().for_safeguards

    def encode(self, buf):
        data = np.asarray(buf).squeeze()

        if len(data.shape) == 1:
            encoded = self.encode_1d(data)
        elif len(data.shape) == 2:
            encoded = self.encode_2d(data)
        elif len(data.shape) == 3:
            encoded = self.encode_3d(data)
        else:
            raise TypeError("LorenzoPredictor currently only supports 1D-3D data")

        with np.printoptions(threshold=50):
            print(
                len(np.unique(encoded)),
                np.unique(as_bits(encoded, kind="i")),
                np.count_nonzero(encoded),
                encoded.size,
            )

        return self._lossless.encode(encoded)

    def decode(self, buf, out=None):
        assert out is not None
        dout = np.asarray(out).squeeze()

        encoded = (
            numcodecs.compat.ensure_ndarray(self._lossless.decode(buf))
            .view(self._quantizer.encoded_dtype(dout.dtype))
            .reshape(dout.shape)
        )

        if len(dout.shape) == 1:
            decoded = self.decode_1d(encoded, dout.dtype)
        elif len(dout.shape) == 2:
            decoded = self.decode_2d(encoded, dout.dtype)
        elif len(dout.shape) == 3:
            decoded = self.decode_3d(encoded, dout.dtype)
        else:
            raise TypeError("LorenzoPredictor currently only supports 1D-3D data")

        return numcodecs.compat.ndarray_copy(decoded.reshape(out.shape), out)

    def encode_1d(self, data: np.ndarray) -> np.ndarray:
        assert len(data.shape) == 1
        (N,) = data.shape

        encoded = np.zeros(
            shape=data.shape, dtype=self._quantizer.encoded_dtype(data.dtype)
        )
        decoded = np.zeros_like(data)

        if data.size > 0:
            encoded[0] = self._quantizer.encode(data[0], np.array(0, dtype=data.dtype))
            decoded[0] = self._quantizer.decode(
                encoded[0], np.array(0, dtype=data.dtype)
            )

            for i in tqdm(range(1, N)):
                predict = decoded[i - 1]
                encoded[i] = self._quantizer.encode(data[i], predict)
                decoded[i] = self._quantizer.decode(encoded[i], predict)

        return encoded

    def decode_1d(self, encoded: np.ndarray, dtype: np.dtype) -> np.ndarray:
        assert len(encoded.shape) == 1
        (N,) = encoded.shape

        decoded = np.zeros_like(encoded, dtype=dtype)

        if decoded.size > 0:
            decoded[0] = self._quantizer.decode(encoded[0], np.array(0, decoded.dtype))

            for i in tqdm(range(1, N)):
                predict = decoded[i - 1]
                decoded[i] = self._quantizer.decode(encoded[i], predict)

        return decoded

    def encode_2d(self, data: np.ndarray) -> np.ndarray:
        assert len(data.shape) == 2
        M, N = data.shape

        encoded = np.zeros(
            shape=data.shape, dtype=self._quantizer.encoded_dtype(data.dtype)
        )
        decoded = np.zeros_like(data)

        if data.size > 0:
            encoded[0, 0] = self._quantizer.encode(
                data[0, 0], np.array(0, dtype=data.dtype)
            )
            decoded[0, 0] = self._quantizer.decode(
                encoded[0, 0], np.array(0, dtype=data.dtype)
            )

            for i in range(1, N):
                predict = decoded[0, i - 1]
                encoded[0, i] = self._quantizer.encode(data[0, i], predict)
                decoded[0, i] = self._quantizer.decode(encoded[0, i], predict)

            for j in range(1, M):
                predict = decoded[j - 1, 0]
                encoded[j, 0] = self._quantizer.encode(data[j, 0], predict)
                decoded[j, 0] = self._quantizer.decode(encoded[j, 0], predict)

            for j in tqdm(range(1, M)):
                for i in range(1, N):
                    predict = (
                        decoded[j, i - 1] + decoded[j - 1, i] - decoded[j - 1, i - 1]
                    )
                    encoded[j, i] = self._quantizer.encode(data[j, i], predict)
                    decoded[j, i] = self._quantizer.decode(encoded[j, i], predict)

        return encoded

    def decode_2d(self, encoded: np.ndarray, dtype: np.dtype) -> np.ndarray:
        assert len(encoded.shape) == 2
        M, N = encoded.shape

        decoded = np.zeros_like(encoded, dtype=dtype)

        if decoded.size > 0:
            decoded[0, 0] = self._quantizer.decode(
                encoded[0, 0], np.array(0, decoded.dtype)
            )

            for i in range(1, N):
                predict = decoded[0, i - 1]
                decoded[0, i] = self._quantizer.decode(encoded[0, i], predict)

            for j in range(1, M):
                predict = decoded[j - 1, 0]
                decoded[j, 0] = self._quantizer.decode(encoded[j, 0], predict)

            for j in tqdm(range(1, M)):
                for i in range(1, N):
                    predict = (
                        decoded[j, i - 1] + decoded[j - 1, i] - decoded[j - 1, i - 1]
                    )
                    decoded[j, i] = self._quantizer.decode(encoded[j, i], predict)

        return decoded

    def encode_3d(self, data: np.ndarray) -> np.ndarray:
        assert len(data.shape) == 3
        L, M, N = data.shape

        encoded = np.zeros(
            shape=data.shape, dtype=self._quantizer.encoded_dtype(data.dtype)
        )
        decoded = np.zeros_like(data)

        if data.size > 0:
            encoded[0, 0, 0] = self._quantizer.encode(
                data[0, 0, 0], np.array(0, dtype=data.dtype)
            )
            decoded[0, 0, 0] = self._quantizer.decode(
                encoded[0, 0, 0], np.array(0, dtype=data.dtype)
            )

            for i in range(1, N):
                predict = decoded[0, 0, i - 1]
                encoded[0, 0, i] = self._quantizer.encode(data[0, 0, i], predict)
                decoded[0, 0, i] = self._quantizer.decode(encoded[0, 0, i], predict)

            for j in range(1, M):
                predict = decoded[0, j - 1, 0]
                encoded[0, j, 0] = self._quantizer.encode(data[0, j, 0], predict)
                decoded[0, j, 0] = self._quantizer.decode(encoded[0, j, 0], predict)

            for k in range(1, L):
                predict = decoded[k - 1, 0, 0]
                encoded[k, 0, 0] = self._quantizer.encode(data[k, 0, 0], predict)
                decoded[k, 0, 0] = self._quantizer.decode(encoded[k, 0, 0], predict)

            for j in tqdm(range(1, M)):
                for i in range(1, N):
                    predict = (
                        decoded[0, j, i - 1]
                        + decoded[0, j - 1, i]
                        - decoded[0, j - 1, i - 1]
                    )
                    encoded[0, j, i] = self._quantizer.encode(data[0, j, i], predict)
                    decoded[0, j, i] = self._quantizer.decode(encoded[0, j, i], predict)

            for k in tqdm(range(1, L)):
                for i in range(1, N):
                    predict = (
                        decoded[k, 0, i - 1]
                        + decoded[k - 1, 0, i]
                        - decoded[k - 1, 0, i - 1]
                    )
                    encoded[k, 0, i] = self._quantizer.encode(data[k, 0, i], predict)
                    decoded[k, 0, i] = self._quantizer.decode(encoded[k, 0, i], predict)

            for k in tqdm(range(1, L)):
                for j in range(1, M):
                    predict = (
                        decoded[k, j - 1, 0]
                        + decoded[k - 1, j, 0]
                        - decoded[k - 1, j - 1, 0]
                    )
                    encoded[k, j, 0] = self._quantizer.encode(data[k, j, 0], predict)
                    decoded[k, j, 0] = self._quantizer.decode(encoded[k, j, 0], predict)

            for k in tqdm(range(1, L), position=0):
                for j in tqdm(range(1, M), position=1):
                    for i in range(1, N):
                        predict = (
                            decoded[k, j, i - 1]
                            + decoded[k, j - 1, i]
                            + decoded[k - 1, j, i]
                            - decoded[k, j - 1, i - 1]
                            - decoded[k - 1, j, i - 1]
                            - decoded[k - 1, j - 1, i]
                            + decoded[k - 1, j - 1, i - 1]
                        )
                        encoded[k, j, i] = self._quantizer.encode(
                            data[k, j, i], predict
                        )
                        decoded[k, j, i] = self._quantizer.decode(
                            encoded[k, j, i], predict
                        )

        return encoded

    def decode_3d(self, encoded: np.ndarray, dtype: np.dtype) -> np.ndarray:
        assert len(encoded.shape) == 3
        L, M, N = encoded.shape

        decoded = np.zeros_like(encoded, dtype=dtype)

        if decoded.size > 0:
            decoded[0, 0, 0] = self._quantizer.decode(
                encoded[0, 0, 0], np.array(0, dtype=decoded.dtype)
            )

            for i in range(1, N):
                predict = decoded[0, 0, i - 1]
                decoded[0, 0, i] = self._quantizer.decode(encoded[0, 0, i], predict)

            for j in range(1, M):
                predict = decoded[0, j - 1, 0]
                decoded[0, j, 0] = self._quantizer.decode(encoded[0, j, 0], predict)

            for k in range(1, L):
                predict = decoded[k - 1, 0, 0]
                decoded[k, 0, 0] = self._quantizer.decode(encoded[k, 0, 0], predict)

            for j in tqdm(range(1, M)):
                for i in range(1, N):
                    predict = (
                        decoded[0, j, i - 1]
                        + decoded[0, j - 1, i]
                        - decoded[0, j - 1, i - 1]
                    )
                    decoded[0, j, i] = self._quantizer.decode(encoded[0, j, i], predict)

            for k in tqdm(range(1, L)):
                for i in range(1, N):
                    predict = (
                        decoded[k, 0, i - 1]
                        + decoded[k - 1, 0, i]
                        - decoded[k - 1, 0, i - 1]
                    )
                    decoded[k, 0, i] = self._quantizer.decode(encoded[k, 0, i], predict)

            for k in tqdm(range(1, L)):
                for j in range(1, M):
                    predict = (
                        decoded[k, j - 1, 0]
                        + decoded[k - 1, j, 0]
                        - decoded[k - 1, j - 1, 0]
                    )
                    decoded[k, j, 0] = self._quantizer.decode(encoded[k, j, 0], predict)

            for k in tqdm(range(1, L), position=0):
                for j in tqdm(range(1, M), position=1):
                    for i in range(1, N):
                        predict = (
                            decoded[k, j, i - 1]
                            + decoded[k, j - 1, i]
                            + decoded[k - 1, j, i]
                            - decoded[k, j - 1, i - 1]
                            - decoded[k - 1, j, i - 1]
                            - decoded[k - 1, j - 1, i]
                            + decoded[k - 1, j - 1, i - 1]
                        )
                        decoded[k, j, i] = self._quantizer.decode(
                            encoded[k, j, i], predict
                        )

        return decoded

    def __repr__(self) -> str:
        return f"{type(self).__name__}(quantizer={self._quantizer})"


if __name__ == "__main__":
    import matplotlib

    matplotlib.use("module://mpl_ascii")

    results = SimpleNamespace(dataset=[], compressor=[], ratio=[])

    for d, codec, result in gen_benchmark(
        gen_data(),
        [
            gen_codecs_with_eb_abs(lambda eb_abs: Sz3(eb_mode="abs", eb_abs=eb_abs)),
            gen_codecs_with_eb_abs(
                lambda eb_abs: Sz3(eb_mode="abs", eb_abs=eb_abs, predictor="lorenzo")
            ),
            gen_codecs_with_eb_abs(
                lambda eb_abs: Sz3(eb_mode="abs", eb_abs=eb_abs, predictor=None)
            ),
            gen_codecs_with_eb_abs(
                lambda eb_abs: Zfp(mode="fixed-accuracy", tolerance=eb_abs)
            ),
            gen_single_codec(Zstd(level=3)),
            gen_codecs_with_eb_abs(
                lambda eb_abs: SafeguardsCodec(
                    codec=None, safeguards=[dict(kind="abs", eb_abs=eb_abs)]
                )
            ),
            # gen_codecs_with_eb_abs(
            #     lambda eb_abs: LorenzoPredictor(
            #         quantizer=MyLinearQuantizer(eb_abs=eb_abs)
            #     )
            # ),
            # gen_codecs_with_eb_abs(
            #     lambda eb_abs: LorenzoPredictor(
            #         quantizer=MySafeguardsQuantizer(eb_abs=eb_abs)
            #     )
            # ),
        ],
    ):
        results.dataset.append(d)
        results.compressor.append(repr(codec))
        results.ratio.append(result)

        print(f"- {d} {codec}: {result}", flush=True)

    pd.DataFrame(results.__dict__).to_csv(
        Path(__file__).parent / "compression.csv",
        index=False,
    )
