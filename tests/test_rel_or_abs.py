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
        out[:] = -np.frombuffer(buf, like=out)
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


def test_arange():
    data = np.arange(100, dtype=float)

    decoded = encode_decode_neg(data, guardrail=GuardrailKind.rel_or_abs, eb_rel=0.1, eb_abs=0.1)
    np.testing.assert_allclose(decoded, data, rtol=0.1, atol=0.1)

    decoded = encode_decode_zero(data, guardrail=GuardrailKind.rel_or_abs, eb_rel=0.1, eb_abs=0.1)
    np.testing.assert_allclose(decoded, data, rtol=0.1, atol=0.1)


def test_linspace():
    data = np.linspace(-1024, 1024, 2831)

    decoded = encode_decode_neg(data, guardrail=GuardrailKind.rel_or_abs, eb_rel=0.1, eb_abs=0.1)
    np.testing.assert_allclose(decoded, data, rtol=0.1, atol=0.1)

    decoded = encode_decode_zero(data, guardrail=GuardrailKind.rel_or_abs, eb_rel=0.1, eb_abs=0.1)
    np.testing.assert_allclose(decoded, data, rtol=0.1, atol=0.1)


def test_edge_cases():
    data = np.array(
        [np.inf, np.nan, -np.inf, -np.nan, np.finfo(float).min, np.finfo(float).max]
    )

    decoded = encode_decode_neg(data, guardrail=GuardrailKind.rel_or_abs, eb_rel=0.1, eb_abs=0.1)
    np.testing.assert_allclose(decoded, data, rtol=0.1, atol=0.1)

    decoded = encode_decode_zero(data, guardrail=GuardrailKind.rel_or_abs, eb_rel=0.1, eb_abs=0.1)
    np.testing.assert_allclose(decoded, data, rtol=0.1, atol=0.1)
