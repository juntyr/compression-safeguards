import numpy as np

from numcodecs_guardrail import GuardrailCodec, GuardrailKind
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
    codec = GuardrailCodec(ZeroCodec(), **kwargs)

    encoded = codec.encode(data)
    decoded = codec.decode(encoded, out=np.empty_like(data))

    return decoded


def encode_decode_neg(data: np.ndarray, **kwargs) -> np.ndarray:
    codec = GuardrailCodec(NegCodec(), **kwargs)

    encoded = codec.encode(data)
    decoded = codec.decode(encoded, out=np.empty_like(data))

    return decoded


def encode_decode_identity(data: np.ndarray, **kwargs) -> np.ndarray:
    codec = GuardrailCodec(IdentityCodec(), **kwargs)

    encoded = codec.encode(data)

    assert isinstance(encoded, bytes)
    assert len(encoded) == (1 + data.size * data.dtype.itemsize)

    decoded = codec.decode(encoded, out=np.empty_like(data))

    return decoded


def check_all_codecs(data: np.ndarray):
    decoded = encode_decode_zero(
        data, guardrail=GuardrailKind.rel_or_abs, eb_rel=0.1, eb_abs=0.1
    )
    np.testing.assert_allclose(decoded, data, rtol=0.1, atol=0.1)

    decoded = encode_decode_neg(
        data, guardrail=GuardrailKind.rel_or_abs, eb_rel=0.1, eb_abs=0.1
    )
    np.testing.assert_allclose(decoded, data, rtol=0.1, atol=0.1)

    decoded = encode_decode_identity(
        data, guardrail=GuardrailKind.rel_or_abs, eb_rel=0.1, eb_abs=0.1
    )
    np.testing.assert_allclose(decoded, data, rtol=0.0, atol=0.0)


def test_empty():
    check_all_codecs(np.empty(0))


def test_arange():
    check_all_codecs(np.arange(100, dtype=float))


def test_linspace():
    check_all_codecs(np.linspace(-1024, 1024, 2831))


def test_edge_cases():
    check_all_codecs(
        np.array(
            [np.inf, np.nan, -np.inf, -np.nan, np.finfo(float).min, np.finfo(float).max]
        )
    )
