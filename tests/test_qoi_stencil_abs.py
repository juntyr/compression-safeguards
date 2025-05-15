import numpy as np

from .codecs import (
    encode_decode_identity,
    encode_decode_neg,
    encode_decode_noise,
    encode_decode_zero,
)


def check_all_codecs(
    data: np.ndarray, qoi: str, shape: tuple[tuple[int, str, int], ...]
):
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


def test_mean():
    # check_all_codecs(
    #     np.arange(64, dtype=float).reshape(4, 4, 4),
    #     "(X[i-1,j]+X[i+1,j]+X[i,j-1]+X[i,j+1])/4",
    #     ((-1, "i", 1), (-1, "j", 1)),
    # )

    check_all_codecs(
        np.arange(64, dtype=float).reshape(4, 4, 4),
        "sum(X * Array([[0.25, 0.5, 0.25], [0.5, 1.0, 0.5], [0.25, 0.5, 0.25]]))",
        ((-1, "i", 1), (-1, "j", 1)),
    )


def test_relative_indexing():
    from numcodecs_safeguards.safeguards.stencil.qoi.abs import (
        QuantityOfInterestAbsoluteErrorBoundSafeguard,
    )

    safeguard = QuantityOfInterestAbsoluteErrorBoundSafeguard(
        "X[i-1] + X[i+2]",
        ((-1, "i", 4),),
        (0,),
        "valid",
        0,
    )
    assert f"{safeguard._qoi_expr}" == "X[0] + X[3]"

    safeguard = QuantityOfInterestAbsoluteErrorBoundSafeguard(
        "X[i][j]",
        ((-1, "i", 1), (-1, "j", 1)),
        (0, 1),
        "valid",
        0,
    )
    assert f"{safeguard._qoi_expr}" == "X[1, 1]"

    safeguard = QuantityOfInterestAbsoluteErrorBoundSafeguard(
        "X[0][0]",
        ((-1, "i", 1), (-1, "j", 1)),
        (0, 1),
        "valid",
        0,
    )
    assert f"{safeguard._qoi_expr}" == "X[0, 0]"
