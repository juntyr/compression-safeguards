import numpy as np
import pytest

from .codecs import (
    encode_decode_identity,
    encode_decode_neg,
    encode_decode_noise,
    encode_decode_zero,
)


def check_all_codecs(data: np.ndarray, qoi: str, shape: tuple[tuple[int, int], ...]):
    for encode_decode in [
        encode_decode_zero,
        encode_decode_neg,
        encode_decode_identity,
        encode_decode_noise,
    ]:
        for eb_abs in [10.0, 1.0, 0.1, 0.01, 0.0]:
            encode_decode(
                data,
                safeguards=[
                    dict(
                        kind="qoi_abs_stencil",
                        qoi=qoi,
                        shape=shape,
                        axes=tuple(range(len(shape))),  # FIXME
                        boundary="wrap",  # FIXME
                        eb_abs=eb_abs,
                    )
                ],
            )


def check_empty(qoi: str):
    check_all_codecs(np.empty(0), qoi, ((0, 0),))


def check_arange(qoi: str):
    check_all_codecs(np.arange(100, dtype=float), qoi, ((0, 0),))


def check_linspace(qoi: str):
    check_all_codecs(np.linspace(-1024, 1024, 2831), qoi, ((0, 0),))


def check_edge_cases(qoi: str):
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
        ),
        qoi,
        ((0, 0),),
    )


CHECKS = [
    check_empty,
    check_arange,
    check_linspace,
    check_edge_cases,
]


def test_sandbox():
    with pytest.raises(AssertionError, match="invalid qoi expression"):
        # sandbox escape based on https://stackoverflow.com/q/35804961 and
        #  https://stackoverflow.com/a/35806044
        check_all_codecs(
            np.empty(0),
            "f\"{[c for c in ().__class__.__base__.__subclasses__() if c.__name__ == 'catch_warnings'][0]()._module.__builtins__['quit']()}\"",
            ((0, 0),),
        )


@pytest.mark.parametrize("check", CHECKS)
def test_empty(check):
    with pytest.raises(AssertionError, match="empty"):
        check("")
    with pytest.raises(AssertionError, match="empty"):
        check("  \t   \n   ")


def test_non_expression():
    with pytest.raises(AssertionError, match="numeric expression"):
        check_all_codecs(np.empty(0), "exp", ((0, 0),))


@pytest.mark.parametrize("check", CHECKS)
def test_constant(check):
    with pytest.raises(AssertionError, match="constant"):
        check("0")
    with pytest.raises(AssertionError, match="constant"):
        check("pi")
    with pytest.raises(AssertionError, match="constant"):
        check("e")
    with pytest.raises(AssertionError, match="constant"):
        check("-(-(-e))")


@pytest.mark.parametrize("check", CHECKS)
def test_imaginary(check):
    with pytest.raises(AssertionError, match="imaginary"):
        check_all_codecs(
            np.array([2], dtype=np.uint64), "(-log(-20417,ln(x)))", ((0, 0),)
        )
    with pytest.raises(AssertionError, match="imaginary"):
        check("(-log(-20417,ln(x)))")


def test_invalid_array():
    with pytest.raises(AssertionError, match="numeric expression"):
        check_all_codecs(np.empty(0), "A", ((0, 0),))
    with pytest.raises(AssertionError, match="array constructor"):
        check_all_codecs(np.empty(0), "A(1)", ((0, 0),))


def test_mean():
    # check_all_codecs(
    #     np.arange(64, dtype=float).reshape(4, 4, 4),
    #     "(X[I+A[-1,0]]+X[I+A[+1,0]]+X[I+A[0,-1]]+X[I+A[0,+1]])/4",
    #     ((-1, 1), (-1, 1)),
    # )

    check_all_codecs(
        np.arange(64, dtype=float).reshape(4, 4, 4),
        "asum(X * A[[0.25, 0.5, 0.25], [0.5, 1.0, 0.5], [0.25, 0.5, 0.25]])",
        ((-1, 1), (-1, 1)),
    )


def test_relative_indexing():
    from numcodecs_safeguards.safeguards.stencil.qoi.abs import (
        QuantityOfInterestAbsoluteErrorBoundSafeguard,
    )

    safeguard = QuantityOfInterestAbsoluteErrorBoundSafeguard(
        "X[I-1] + X[I+2]",
        ((-1, 4),),
        (0,),
        "valid",
        0,
    )
    assert f"{safeguard._qoi_expr}" == "X[0] + X[3]"

    safeguard = QuantityOfInterestAbsoluteErrorBoundSafeguard(
        "X[I[0]][I[1]]",
        ((-1, 1), (-1, 1)),
        (0, 1),
        "valid",
        0,
    )
    assert f"{safeguard._qoi_expr}" == "X[1, 1]"

    safeguard = QuantityOfInterestAbsoluteErrorBoundSafeguard(
        "X[0][0]",
        ((-1, 1), (-1, 1)),
        (0, 1),
        "valid",
        0,
    )
    assert f"{safeguard._qoi_expr}" == "X[0, 0]"
