import numpy as np
from numcodecs.abc import Codec

from numcodecs_safeguards import SafeguardsCodec


class ZeroCodec(Codec):
    codec_id = "zero"

    def __init__(self):
        pass

    def encode(self, buf):
        return b""

    def decode(self, buf, out=None):
        assert out is not None
        np.copyto(out, 0)
        return out


class NegCodec(Codec):
    codec_id = "neg"

    def encode(self, buf):
        return np.array(buf).tobytes()

    def decode(self, buf, out=None):
        assert out is not None
        if out.size > 0:
            np.copyto(
                out,
                np.frombuffer(buf, dtype=out.dtype, count=np.prod(out.shape)).reshape(
                    out.shape
                ),
            )
            out[:] = np.negative(out)
        return out


class IdentityCodec(Codec):
    codec_id = "identity"

    def encode(self, buf):
        return np.array(buf).tobytes()

    def decode(self, buf, out=None):
        assert out is not None
        if out.size > 0:
            np.copyto(
                out,
                np.frombuffer(buf, dtype=out.dtype, count=np.prod(out.shape)).reshape(
                    out.shape
                ),
            )
        return out


class NoiseCodec(Codec):
    codec_id = "noise"

    def encode(self, buf):
        return (np.array(buf) + np.random.normal(scale=0.1, size=buf.shape)).tobytes()

    def decode(self, buf, out=None):
        assert out is not None
        if out.size > 0:
            np.copyto(
                out,
                np.frombuffer(buf, dtype=out.dtype, count=np.prod(out.shape)).reshape(
                    out.shape
                ),
            )
        return out


class MockCodec(Codec):
    codec_id = "mock"

    def __init__(self, data, decoded):
        self.data = data
        self.decoded = decoded

    def encode(self, buf):
        return b""

    def decode(self, buf, out=None):
        assert len(buf) == 0
        assert out is not None
        out[:] = self.decoded
        return out


def encode_decode_zero(data: np.ndarray, **kwargs) -> np.ndarray:
    codec = SafeguardsCodec(codec=ZeroCodec(), **kwargs)

    encoded = codec.encode(data)
    decoded = codec.decode(encoded, out=np.empty_like(data))

    return decoded


def encode_decode_neg(data: np.ndarray, **kwargs) -> np.ndarray:
    codec = SafeguardsCodec(codec=NegCodec(), **kwargs)

    encoded = codec.encode(data)
    decoded = codec.decode(encoded, out=np.empty_like(data))

    return decoded


def encode_decode_identity(data: np.ndarray, **kwargs) -> np.ndarray:
    codec = SafeguardsCodec(codec=IdentityCodec(), **kwargs)

    encoded = codec.encode(data)

    assert isinstance(encoded, bytes)
    # Ensure that codecs that already satisfy the properties only have a
    #  single-byte overhead
    assert len(encoded) == (1 + data.size * data.dtype.itemsize)

    decoded = codec.decode(encoded, out=np.empty_like(data))

    return decoded


def encode_decode_noise(data: np.ndarray, **kwargs) -> np.ndarray:
    codec = SafeguardsCodec(codec=NoiseCodec(), **kwargs)

    encoded = codec.encode(data)
    decoded = codec.decode(encoded, out=np.empty_like(data))

    return decoded


def encode_decode_mock(data: np.ndarray, decoded: np.ndarray, **kwargs) -> np.ndarray:
    codec = SafeguardsCodec(codec=MockCodec(data, decoded), **kwargs)

    encoded = codec.encode(data)
    decoded = codec.decode(encoded, out=np.empty_like(data))

    return decoded


def encode_decode_none(data: np.ndarray, **kwargs) -> np.ndarray:
    codec = SafeguardsCodec(codec=None, **kwargs)

    encoded = codec.encode(data)
    decoded = codec.decode(encoded, out=np.empty_like(data))

    return decoded
