import numcodecs.compat
import numpy as np
from numcodecs.abc import Codec
from numcodecs_combinators.framed import FramedCodecStack
from numcodecs_safeguards import SafeguardsCodec
from numcodecs_zero import ZeroCodec


class NegCodec(Codec):
    __slots__ = ()

    codec_id = "neg"  # type: ignore

    def encode(self, buf):
        return np.negative(buf)

    def decode(self, buf, out=None):
        return numcodecs.compat.ndarray_copy(buf, out)


class IdentityCodec(Codec):
    __slots__ = ()

    codec_id = "identity"  # type: ignore

    def encode(self, buf):
        return buf

    def decode(self, buf, out=None):
        return numcodecs.compat.ndarray_copy(buf, out)


class NoiseCodec(Codec):
    __slots__ = ("_seed",)
    _seed: int

    codec_id = "noise"  # type: ignore

    def __init__(self, *, seed: int):
        self._seed = seed

    def encode(self, buf):
        rng = np.random.Generator(np.random.PCG64(seed=self._seed))
        return buf + rng.normal(scale=0.1, size=buf.shape).astype(buf.dtype)

    def decode(self, buf, out=None):
        return numcodecs.compat.ndarray_copy(buf, out)


class MockCodec(Codec):
    __slots__ = ("data", "decoded")

    codec_id = "mock"  # type: ignore

    def __init__(self, data, decoded):
        self.data = data
        self.decoded = decoded

    def encode(self, buf):
        return b""

    def decode(self, buf, out=None):
        assert len(buf) == 0
        return numcodecs.compat.ndarray_copy(np.copy(self.decoded), out)


def encode_decode_zero(data: np.ndarray, **kwargs) -> np.ndarray:
    codec = SafeguardsCodec(codec=ZeroCodec(), **kwargs)

    encoded = codec.encode(data)
    decoded = codec.decode(encoded)

    return decoded


def encode_decode_neg(data: np.ndarray, **kwargs) -> np.ndarray:
    codec = SafeguardsCodec(codec=FramedCodecStack(NegCodec()), **kwargs)

    encoded = codec.encode(data)
    decoded = codec.decode(encoded)

    return decoded


def encode_decode_identity(data: np.ndarray, **kwargs) -> np.ndarray:
    codec = SafeguardsCodec(codec=FramedCodecStack(IdentityCodec()), **kwargs)

    encoded = codec.encode(data)
    assert isinstance(encoded, bytes)

    just_data = FramedCodecStack(IdentityCodec()).encode(data)
    assert isinstance(just_data, bytes)

    # Ensure that codecs that already satisfy the properties only have a
    #  single-byte overhead
    assert len(encoded) == (1 + len(just_data))

    decoded = codec.decode(encoded)

    return decoded


def encode_decode_noise(data: np.ndarray, **kwargs) -> np.ndarray:
    seed = np.random.randint(2**31)  # noqa: NPY002
    print(f"noise seed = {seed}")  # noqa: T201
    codec = SafeguardsCodec(codec=FramedCodecStack(NoiseCodec(seed=seed)), **kwargs)

    encoded = codec.encode(data)
    decoded = codec.decode(encoded)

    return decoded


def encode_decode_mock(data: np.ndarray, decoded: np.ndarray, **kwargs) -> np.ndarray:
    codec = SafeguardsCodec(codec=MockCodec(data, decoded), **kwargs)

    encoded = codec.encode(data)
    decoded = codec.decode(encoded)

    return decoded
