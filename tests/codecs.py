import numpy as np

from numcodecs_guardrails import GuardrailsCodec
from numcodecs.abc import Codec


class ZeroCodec(Codec):
    codec_id = "zero"

    def __init__(self):
        pass

    def encode(self, buf):
        return b""

    def decode(self, buf, out=None):
        assert out is not None
        out[:] = 0
        return out


class NegCodec(Codec):
    codec_id = "neg"

    def encode(self, buf):
        return np.array(buf).tobytes()

    def decode(self, buf, out=None):
        assert out is not None
        if out.size > 0:
            out[:] = -np.frombuffer(buf, like=out)
        return out


class IdentityCodec(Codec):
    codec_id = "identity"

    def encode(self, buf):
        return np.array(buf).tobytes()

    def decode(self, buf, out=None):
        assert out is not None
        if out.size > 0:
            out[:] = np.frombuffer(buf, like=out)
        return out


def encode_decode_zero(data: np.ndarray, **kwargs) -> np.ndarray:
    codec = GuardrailsCodec(codec=ZeroCodec(), **kwargs)

    encoded = codec.encode(data)
    decoded = codec.decode(encoded, out=np.empty_like(data))

    return decoded


def encode_decode_neg(data: np.ndarray, **kwargs) -> np.ndarray:
    codec = GuardrailsCodec(codec=NegCodec(), **kwargs)

    encoded = codec.encode(data)
    decoded = codec.decode(encoded, out=np.empty_like(data))

    return decoded


def encode_decode_identity(data: np.ndarray, **kwargs) -> np.ndarray:
    codec = GuardrailsCodec(codec=IdentityCodec(), **kwargs)

    encoded = codec.encode(data)

    assert isinstance(encoded, bytes)
    # Ensure that codecs that already satisfy the properties only have a
    #  single-byte overhead
    assert len(encoded) == (1 + data.size * data.dtype.itemsize)

    decoded = codec.decode(encoded, out=np.empty_like(data))

    return decoded
