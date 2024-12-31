import numpy as np

from numcodecs_guardrail import GuardrailKind


from .codecs import encode_decode_zero, encode_decode_neg, encode_decode_identity


def check_all_codecs(data: np.ndarray):
    decoded = encode_decode_zero(data, guardrail=GuardrailKind.abs, eb_abs=0.1)
    np.testing.assert_allclose(decoded, data, rtol=0.0, atol=0.1)

    decoded = encode_decode_neg(data, guardrail=GuardrailKind.abs, eb_abs=0.1)
    np.testing.assert_allclose(decoded, data, rtol=0.0, atol=0.1)

    decoded = encode_decode_identity(data, guardrail=GuardrailKind.abs, eb_abs=0.1)
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
