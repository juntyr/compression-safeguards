import numpy as np

from .codecs import (
    encode_decode_zero,
    encode_decode_neg,
    encode_decode_identity,
    encode_decode_noise,
)


def check_all_codecs(data: np.ndarray):
    for encode_decode in [
        encode_decode_zero,
        encode_decode_neg,
        encode_decode_identity,
        encode_decode_noise,
    ]:
        for qoi in [
            # polynomial
            "x",
            "x**2",
            "x**3",
            "x**2 + x + 1",
            # # exponential
            # "0.5**x",
            # "2**x",
            # "3**x",
            # "exp(x)",
            # "2 ** (x + 1)",
            # logarithm
            "log(x, 2)",
            "ln(x)",
            "ln(x + 1)",
            "log(2, x)",
            # power function
            "1 / (x + 3)",
            # # inverse
            "1 / x",
            "1 / x**2",
            "1 / x**3",
            # sqrt
            "sqrt(x)",
            "1 / sqrt(x)",
            # # sigmoid
            # "1 / (1 + exp(-x))",
            # # tanh
            # "(exp(x) - exp(-x)) / (exp(x) + exp(-x))",
            # composed
            "2 / (ln(x) + sqrt(x))",
        ]:
            for eb_abs in [10.0, 1.0, 0.1, 0.01]:
                encode_decode(
                    data,
                    safeguards=[dict(kind="qoi_abs", qoi=qoi, eb_abs=eb_abs)],
                )
                # np.testing.assert_allclose(
                #     qoi(decoded), qoi(data), rtol=0.0, atol=eb_abs
                # )


def test_empty():
    check_all_codecs(np.empty(0))


def test_arange():
    check_all_codecs(np.arange(100, dtype=float))


def test_linspace():
    check_all_codecs(np.linspace(-1024, 1024, 2831))


def test_edge_cases():
    check_all_codecs(
        np.array(
            [
                np.inf,
                np.nan,
                -np.inf,
                -np.nan,
                np.finfo(float).min,
                np.finfo(float).max,
                np.finfo(float).tiny,
                -np.finfo(float).tiny,
                -0.0,
                +0.0,
            ]
        )
    )
