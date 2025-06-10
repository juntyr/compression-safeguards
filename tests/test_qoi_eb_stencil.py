from itertools import cycle, permutations, product

import numpy as np
import pytest

from compression_safeguards import Safeguards
from compression_safeguards.safeguards.stencil import BoundaryCondition
from compression_safeguards.safeguards.stencil.qoi.eb import (
    StencilQuantityOfInterestErrorBoundSafeguard,
)
from compression_safeguards.utils.bindings import Bindings

from .codecs import (
    encode_decode_identity,
    encode_decode_mock,
    encode_decode_neg,
    encode_decode_noise,
    encode_decode_zero,
)


def check_all_codecs(data: np.ndarray, qoi: str, shape: list[tuple[int, int]]):
    for encode_decode in [
        encode_decode_zero,
        encode_decode_neg,
        encode_decode_identity,
        encode_decode_noise,
    ]:
        for axes, boundaries, (type, eb) in zip(
            cycle(permutations(range(data.ndim), len(shape))),
            product(*[BoundaryCondition for _ in range(data.ndim)]),
            cycle(
                [
                    ("abs", 10.0),
                    ("abs", 1.0),
                    ("abs", 0.1),
                    ("abs", 0.01),
                    ("abs", 0.0),
                    ("rel", 10.0),
                    ("rel", 1.0),
                    ("rel", 0.1),
                    ("rel", 0.01),
                    ("rel", 0.0),
                    ("ratio", 10.0),
                    ("ratio", 2.0),
                    ("ratio", 1.1),
                    ("ratio", 1.01),
                    ("ratio", 1.0),
                ]
            ),
        ):
            try:
                encode_decode(
                    data,
                    safeguards=[
                        dict(
                            kind="qoi_eb_stencil",
                            qoi=qoi,
                            neighbourhood=[
                                dict(
                                    axis=axis,
                                    before=before,
                                    after=after,
                                    boundary=boundary,
                                    constant_boundary=4.2
                                    if boundary == BoundaryCondition.constant
                                    else None,
                                )
                                for axis, boundary, (before, after) in zip(
                                    axes, boundaries, shape
                                )
                            ],
                            type=type,
                            eb=eb,
                        )
                    ],
                )
            except Exception as err:
                print(qoi, shape, axes, boundaries, type, eb)
                raise err


def check_empty(qoi: str):
    data = np.empty(0)
    check_all_codecs(data, qoi, [(0, 0)])
    check_all_codecs(data, qoi, [(1, 1)])


def check_unit(qoi: str):
    check_all_codecs(np.linspace(-1.0, 1.0, 100), qoi, [(0, 0)])
    check_all_codecs(np.linspace(-1.0, 1.0, 100), qoi, [(1, 1)])


def check_circle(qoi: str):
    check_all_codecs(
        np.linspace(-np.pi * 2, np.pi * 2, 100, dtype=np.int64), qoi, [(0, 0)]
    )
    check_all_codecs(
        np.linspace(-np.pi * 2, np.pi * 2, 100, dtype=np.int64), qoi, [(1, 1)]
    )


def check_arange(qoi: str):
    data = np.arange(100, dtype=float)
    check_all_codecs(data, qoi, [(0, 0)])
    check_all_codecs(data, qoi, [(1, 1)])


def check_linspace(qoi: str):
    data = np.linspace(-1024, 1024, 2831)
    check_all_codecs(data, qoi, [(0, 0)])
    check_all_codecs(data, qoi, [(1, 1)])


def check_edge_cases(qoi: str):
    data = np.array(
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
    check_all_codecs(
        data,
        qoi,
        [(0, 0)],
    )
    check_all_codecs(
        data,
        qoi,
        [(1, 1)],
    )


CHECKS = [
    check_empty,
    check_unit,
    check_circle,
    check_arange,
    check_linspace,
    check_edge_cases,
]


def test_sandbox():
    with pytest.raises(AssertionError, match="invalid QoI expression"):
        # sandbox escape based on https://stackoverflow.com/q/35804961 and
        #  https://stackoverflow.com/a/35806044
        check_all_codecs(
            np.empty(0),
            "f\"{[c for c in ().__class__.__base__.__subclasses__() if c.__name__ == 'catch_warnings'][0]()._module.__builtins__['quit']()}\"",
            [(0, 0)],
        )


@pytest.mark.parametrize("check", CHECKS)
def test_empty(check):
    with pytest.raises(AssertionError, match="empty"):
        check("")
    with pytest.raises(AssertionError, match="empty"):
        check("  \t   \n   ")
    with pytest.raises(AssertionError, match="empty"):
        check(" # just a comment ")


def test_non_expression():
    with pytest.raises(AssertionError, match="numeric expression"):
        check_all_codecs(np.empty(0), "exp", [(0, 0)])


def test_whitespace():
    check_all_codecs(np.array([]), "  \n \t x   \t\n  ", [(0, 0)])
    check_all_codecs(np.array([]), "  \n \t x \t \n  - \t \n 3  \t\n  ", [(0, 0)])
    check_all_codecs(np.array([]), "x    -    3", [(0, 0)])
    check_all_codecs(np.array([]), "sqrt   \n (x)", [(0, 0)])
    check_all_codecs(np.array([]), "log ( x , base \t = \n 2 )", [(0, 0)])


def test_comment():
    check_all_codecs(np.array([]), "x # great variable", [(0, 0)])
    check_all_codecs(np.array([]), "# great variable\nx", [(0, 0)])
    check_all_codecs(np.array([]), "x # nothing 3+4 really matters 1/0", [(0, 0)])
    check_all_codecs(
        np.array([]),
        "log #1\n ( #2\n x #3\n , #4\n base #5\n = #6\n 2 #7\n )",
        [(0, 0)],
    )


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
            np.array([2], dtype=np.uint64), "(-log(-20417, base=ln(x)))", [(0, 0)]
        )
    with pytest.raises(AssertionError, match="imaginary"):
        check("(-log(-20417, base=ln(x)))")


def test_invalid_array():
    with pytest.raises(AssertionError, match="numeric expression"):
        check_all_codecs(np.empty(0), "A", [(0, 0)])
    with pytest.raises(AssertionError, match="array constructor"):
        check_all_codecs(np.empty(0), "A(1)", [(0, 0)])


def test_mean():
    # arithmetic mean
    check_all_codecs(
        np.arange(64, dtype=float).reshape(4, 4, 4),
        "(X[I+A[-1,0]]+X[I+A[+1,0]]+X[I+A[0,-1]]+X[I+A[0,+1]])/4",
        [(1, 1), (1, 1)],
    )

    # arithmetic mean using a convolution
    check_all_codecs(
        np.arange(64, dtype=float).reshape(4, 4, 4),
        "asum(X * A[[0.25, 0.5, 0.25], [0.5, 1.0, 0.5], [0.25, 0.5, 0.25]])",
        [(1, 1), (1, 1)],
    )

    # geometric mean
    check_all_codecs(
        np.arange(64, dtype=float).reshape(4, 4, 4),
        "(X[I+A[-1,0]]*X[I+A[+1,0]]*X[I+A[0,-1]]*X[I+A[0,+1]])**(1/4)",
        [(1, 1), (1, 1)],
    )


def test_findiff():
    from compression_safeguards.safeguards.stencil.qoi.eb import (
        StencilQuantityOfInterestErrorBoundSafeguard,
    )

    data = np.arange(81, dtype=float).reshape(9, 9)
    valid_5x5_neighbourhood = [
        dict(axis=0, before=4, after=4, boundary="valid"),
        dict(axis=1, before=4, after=4, boundary="valid"),
    ]

    safeguard = StencilQuantityOfInterestErrorBoundSafeguard(
        "findiff(x,order=0,accuracy=2,type=0,dx=1,axis=0)",
        valid_5x5_neighbourhood,
        0,
    )
    assert f"{safeguard._qoi_expr}" == "X[4, 4]"
    check_all_codecs(
        data,
        "findiff(x,order=0,accuracy=2,type=0,dx=1,axis=0)",
        [(4, 4), (4, 4)],
    )

    safeguard = StencilQuantityOfInterestErrorBoundSafeguard(
        "findiff(x,order=1,accuracy=1,type=1,dx=1,axis=0)",
        valid_5x5_neighbourhood,
        0,
    )
    assert f"{safeguard._qoi_expr}" == "-X[4, 4] + X[5, 4]"
    check_all_codecs(
        data,
        "findiff(x,order=1,accuracy=1,type=1,dx=1,axis=0)",
        [(4, 4), (4, 4)],
    )

    safeguard = StencilQuantityOfInterestErrorBoundSafeguard(
        "findiff(x,order=1,accuracy=1,type=-1,dx=1,axis=0)",
        valid_5x5_neighbourhood,
        0,
    )
    assert f"{safeguard._qoi_expr}" == "-X[3, 4] + X[4, 4]"
    check_all_codecs(
        data,
        "findiff(x,order=1,accuracy=1,type=-1,dx=1,axis=0)",
        [(4, 4), (4, 4)],
    )

    safeguard = StencilQuantityOfInterestErrorBoundSafeguard(
        "findiff(x,order=1,accuracy=2,type=0,dx=1,axis=0)",
        valid_5x5_neighbourhood,
        0,
    )
    assert f"{safeguard._qoi_expr}" == "-X[3, 4]/2 + X[5, 4]/2"
    check_all_codecs(
        data,
        "findiff(x,order=1,accuracy=2,type=0,dx=1,axis=0)",
        [(4, 4), (4, 4)],
    )

    safeguard = StencilQuantityOfInterestErrorBoundSafeguard(
        "findiff(x,order=1,accuracy=2,type=0,dx=1,axis=1)",
        valid_5x5_neighbourhood,
        0,
    )
    assert f"{safeguard._qoi_expr}" == "-X[4, 3]/2 + X[4, 5]/2"
    check_all_codecs(
        data,
        "findiff(x,order=1,accuracy=2,type=0,dx=1,axis=1)",
        [(4, 4), (4, 4)],
    )

    safeguard = StencilQuantityOfInterestErrorBoundSafeguard(
        "findiff(x,order=2,accuracy=2,type=0,dx=1,axis=0)",
        valid_5x5_neighbourhood,
        0,
    )
    assert f"{safeguard._qoi_expr}" == "X[3, 4] - 2*X[4, 4] + X[5, 4]"
    check_all_codecs(
        data,
        "findiff(x,order=2,accuracy=2,type=0,dx=1,axis=0)",
        [(4, 4), (4, 4)],
    )

    safeguard = StencilQuantityOfInterestErrorBoundSafeguard(
        "findiff(findiff(x,order=1,accuracy=2,type=0,dx=1,axis=0),order=1,accuracy=2,type=0,dx=1,axis=0)",
        valid_5x5_neighbourhood,
        0,
    )
    assert f"{safeguard._qoi_expr}" == "X[2, 4]/4 - X[4, 4]/2 + X[6, 4]/4"
    check_all_codecs(
        data,
        "findiff(findiff(x,order=1,accuracy=2,type=0,dx=1,axis=0),order=1,accuracy=2,type=0,dx=1,axis=0)",
        [(4, 4), (4, 4)],
    )

    safeguard = StencilQuantityOfInterestErrorBoundSafeguard(
        "findiff(findiff(x,order=1,accuracy=2,type=0,dx=1,axis=0),order=1,accuracy=2,type=0,dx=1,axis=1)",
        valid_5x5_neighbourhood,
        0,
    )
    assert f"{safeguard._qoi_expr}" == "X[3, 3]/4 - X[3, 5]/4 - X[5, 3]/4 + X[5, 5]/4"
    check_all_codecs(
        data,
        "findiff(findiff(x,order=1,accuracy=2,type=0,dx=1,axis=0),order=1,accuracy=2,type=0,dx=1,axis=1)",
        [(4, 4), (4, 4)],
    )


def test_matmul():
    from compression_safeguards.safeguards.stencil.qoi.eb import (
        StencilQuantityOfInterestErrorBoundSafeguard,
    )

    data = np.arange(16, dtype=float).reshape(4, 4)
    valid_3x3_neighbourhood = [
        dict(axis=0, before=1, after=1, boundary="valid"),
        dict(axis=1, before=1, after=1, boundary="valid"),
    ]

    safeguard = StencilQuantityOfInterestErrorBoundSafeguard(
        "matmul(A[[-1, -2, -3]], matmul(X, tr(A[[1, 2, 3]])))[0,0]",
        valid_3x3_neighbourhood,
        0,
    )
    assert (
        f"{safeguard._qoi_expr}"
        == "-X[0, 0] - 2*X[0, 1] - 3*X[0, 2] - 2*X[1, 0] - 4*X[1, 1] - 6*X[1, 2] - 3*X[2, 0] - 6*X[2, 1] - 9*X[2, 2]"
    )
    check_all_codecs(
        data,
        "matmul(A[[-1, -2, -3]], matmul(X, tr(A[[1, 2, 3]])))[0,0]",
        [(1, 1), (1, 1)],
    )


def test_indexing():
    from compression_safeguards.safeguards.stencil.qoi.eb import (
        StencilQuantityOfInterestErrorBoundSafeguard,
    )

    safeguard = StencilQuantityOfInterestErrorBoundSafeguard(
        "X[I-1] + X[I+2]",
        [dict(axis=0, before=1, after=4, boundary="valid")],
        0,
    )
    assert f"{safeguard._qoi_expr}" == "X[0] + X[3]"

    safeguard = StencilQuantityOfInterestErrorBoundSafeguard(
        "X[I[0]][I[1]]",
        [
            dict(axis=0, before=1, after=1, boundary="valid"),
            dict(axis=1, before=1, after=1, boundary="valid"),
        ],
        0,
    )
    assert f"{safeguard._qoi_expr}" == "X[1, 1]"

    safeguard = StencilQuantityOfInterestErrorBoundSafeguard(
        "X[0][0]",
        [
            dict(axis=0, before=1, after=1, boundary="valid"),
            dict(axis=1, before=1, after=1, boundary="valid"),
        ],
        0,
    )
    assert f"{safeguard._qoi_expr}" == "X[0, 0]"

    safeguard = StencilQuantityOfInterestErrorBoundSafeguard(
        "(X[I+A[-1,0]]+X[I+A[+1,0]]+X[I+A[0,-1]]+X[I+A[0,+1]])/4",
        [
            dict(axis=0, before=1, after=1, boundary="valid"),
            dict(axis=1, before=1, after=1, boundary="valid"),
        ],
        0,
    )
    assert f"{safeguard._qoi_expr}" == "X[0, 1]/4 + X[1, 0]/4 + X[1, 2]/4 + X[2, 1]/4"

    safeguard = StencilQuantityOfInterestErrorBoundSafeguard(
        "asum(X * A[[0.25, 0.5, 0.25], [0.5, 1.0, 0.5], [0.25, 0.5, 0.25]])",
        [
            dict(axis=0, before=1, after=1, boundary="valid"),
            dict(axis=1, before=1, after=1, boundary="valid"),
        ],
        0,
    )
    assert (
        f"{safeguard._qoi_expr}"
        == "0.25*X[0, 0] + 0.5*X[0, 1] + 0.25*X[0, 2] + 0.5*X[1, 0] + 1.0*X[1, 1] + 0.5*X[1, 2] + 0.25*X[2, 0] + 0.5*X[2, 1] + 0.25*X[2, 2]"
    )


def test_lambdify_indexing():
    import inspect

    from compression_safeguards.safeguards._qois.compile import sympy_expr_to_numpy
    from compression_safeguards.safeguards.stencil.qoi.eb import (
        StencilQuantityOfInterestErrorBoundSafeguard,
    )

    safeguard = StencilQuantityOfInterestErrorBoundSafeguard(
        "asum(X * A[[0.25, 0.5, 0.25], [0.5, 1.0, 0.5], [0.25, 0.5, 0.25]])",
        [
            dict(axis=0, before=1, after=1, boundary="valid"),
            dict(axis=1, before=1, after=1, boundary="valid"),
        ],
        0,
    )

    fn = sympy_expr_to_numpy([safeguard._X], safeguard._qoi_expr, np.dtype(np.float16))

    assert (
        inspect.getsource(fn)
        == "def _lambdifygenerated(X):\n    return float16('0.25')*X[..., 0, 0] + float16('0.5')*X[..., 0, 1] + float16('0.25')*X[..., 0, 2] + float16('0.5')*X[..., 1, 0] + float16('1.0')*X[..., 1, 1] + float16('0.5')*X[..., 1, 2] + float16('0.25')*X[..., 2, 0] + float16('0.5')*X[..., 2, 1] + float16('0.25')*X[..., 2, 2]\n"
    )


@pytest.mark.parametrize("dtype", sorted(d.name for d in Safeguards.supported_dtypes()))
def test_dtypes(dtype):
    check_all_codecs(np.array([[1]], dtype=dtype), "x/sqrt(pi)", [(0, 0)])


def test_fuzzer_window():
    encode_decode_mock(
        np.array([[6584]], dtype=np.int16),
        np.array([[2049]], dtype=np.int16),
        safeguards=[
            dict(
                kind="qoi_eb_stencil",
                qoi="ln((x**(x**pi)))",
                neighbourhood=[dict(axis=1, before=0, after=10, boundary="valid")],
                type="abs",
                eb=1,
            )
        ],
    )

    encode_decode_mock(
        np.array([], dtype=np.int16),
        np.array([], dtype=np.int16),
        safeguards=[
            dict(
                kind="qoi_eb_stencil",
                qoi="ln((x**(x**pi)))",
                neighbourhood=[dict(axis=0, before=0, after=10, boundary="valid")],
                type="abs",
                eb=1,
            )
        ],
    )


def test_fuzzer_findiff_int_iter():
    data = np.array([65373], dtype=np.uint16)
    decoded = np.array([42246], dtype=np.uint16)

    for boundary in BoundaryCondition:
        encode_decode_mock(
            data,
            decoded,
            safeguards=[
                dict(
                    kind="qoi_eb_stencil",
                    qoi="findiff(x, order=0, accuracy=1, type=-1, dx=2.2250738585072014e-308, axis=0)",
                    neighbourhood=[
                        dict(
                            axis=0,
                            before=0,
                            after=0,
                            boundary=boundary,
                            constant_boundary=4.2
                            if boundary == BoundaryCondition.constant
                            else None,
                        )
                    ],
                    type="abs",
                    eb=2.2250738585072014e-308,
                ),
            ],
        )


def test_fuzzer_findiff_fraction_overflow():
    data = np.array([7], dtype=np.int8)
    decoded = np.array([0], dtype=np.int8)

    for boundary in BoundaryCondition:
        encode_decode_mock(
            data,
            decoded,
            safeguards=[
                dict(
                    kind="qoi_eb_stencil",
                    qoi="findiff(x, order=7, accuracy=6, type=-1, dx=7.215110354450764e305, axis=0)",
                    neighbourhood=[
                        dict(
                            axis=0,
                            before=12,
                            after=0,
                            boundary=boundary,
                            constant_boundary=4.2
                            if boundary == BoundaryCondition.constant
                            else None,
                        )
                    ],
                    type="abs",
                    eb=2.2250738585072014e-308,
                ),
            ],
        )


def test_fuzzer_findiff_fraction_compare():
    data = np.array([1978047305655558])

    for boundary in BoundaryCondition:
        encode_decode_zero(
            data,
            safeguards=[
                dict(kind="zero", zero=7),
                dict(
                    kind="qoi_eb_stencil",
                    qoi="findiff(x, order=7, accuracy=7, type=1, dx=2.2250738585072014e-308, axis=0)",
                    neighbourhood=[
                        dict(
                            axis=0,
                            before=0,
                            after=13,
                            boundary=boundary,
                            constant_boundary=4.2
                            if boundary == BoundaryCondition.constant
                            else None,
                        )
                    ],
                    type="abs",
                    eb=2.2250738585072014e-308,
                ),
                dict(kind="sign"),
            ],
        )


def test_fuzzer_findiff_eb_abs():
    data = np.array([[-27, 8, 8], [8, 8, 8], [8, 8, 8]], dtype=np.int8)
    decoded = np.array([[8, 8, 8], [8, 8, 8], [8, 8, 8]], dtype=np.int8)

    for boundary in BoundaryCondition:
        encode_decode_mock(
            data,
            decoded,
            safeguards=[
                dict(
                    kind="qoi_eb_stencil",
                    qoi="findiff(x, order=4, accuracy=4, type=1, dx=4, axis=0)",
                    neighbourhood=[
                        dict(
                            axis=0,
                            before=0,
                            after=7,
                            boundary=boundary,
                            constant_boundary=4.2
                            if boundary == BoundaryCondition.constant
                            else None,
                        )
                    ],
                    type="abs",
                    eb=1,
                ),
                dict(kind="sign"),
            ],
        )


def test_fuzzer_findiff_fraction_float_overflow():
    data = np.array([[0], [0], [7], [0], [4], [0], [59], [199]], dtype=np.uint16)
    decoded = np.array(
        [[1], [1], [0], [30720], [124], [32768], [16427], [3797]], dtype=np.uint16
    )

    for boundary in BoundaryCondition:
        encode_decode_mock(
            data,
            decoded,
            safeguards=[
                dict(
                    kind="qoi_eb_stencil",
                    qoi="findiff(x, order=1, accuracy=3, type=1, dx=59, axis=0)",
                    neighbourhood=[
                        dict(
                            axis=0,
                            before=0,
                            after=3,
                            boundary=boundary,
                            constant_boundary=4.2
                            if boundary == BoundaryCondition.constant
                            else None,
                        )
                    ],
                    type="abs",
                    eb=8.812221249325077e307,
                ),
                dict(kind="sign"),
            ],
        )


def test_fuzzer_tuple_index_out_of_range():
    data = np.array([], dtype=np.int32)
    decoded = np.array([], dtype=np.int32)

    with pytest.raises(
        IndexError, match=r"axis index -65 is out of bounds for array of shape \(0,\)"
    ):
        encode_decode_mock(
            data,
            decoded,
            safeguards=[
                dict(
                    kind="qoi_eb_stencil",
                    qoi="(-(-x))",
                    neighbourhood=[
                        dict(
                            axis=-65,
                            before=0,
                            after=0,
                            boundary="valid",
                        )
                    ],
                    type="abs",
                    eb=0,
                ),
            ],
        )


def test_fuzzer_list_assignment_out_of_range():
    data = np.array([], dtype=np.uint8)
    decoded = np.array([], dtype=np.uint8)

    encode_decode_mock(
        data,
        decoded,
        safeguards=[
            dict(
                kind="qoi_eb_stencil",
                qoi="acsc((x-e))",
                neighbourhood=[
                    dict(
                        axis=-1,
                        before=93,
                        after=0,
                        boundary="valid",
                    )
                ],
                type="abs",
                eb=0,
            ),
        ],
    )


def test_late_bound_eb_abs():
    safeguard = StencilQuantityOfInterestErrorBoundSafeguard(
        qoi="x",
        neighbourhood=[
            dict(axis=0, before=1, after=0, boundary="valid"),
            dict(axis=1, before=1, after=1, boundary="valid"),
        ],
        type="abs",
        eb="eb",
    )

    data = np.arange(6).reshape(2, 3)

    late_bound = Bindings(eb=np.array([[2]]))

    valid = safeguard.compute_safe_intervals(data, late_bound=late_bound)
    # these bounds could be laxer but we're currently distributing across the
    #  full neighbourhood
    assert np.all(valid._lower == (data.flatten() - np.array([2, 2, 2, 2, 2, 2])))
    assert np.all(valid._upper == (data.flatten() + np.array([2, 2, 2, 2, 2, 2])))

    ok = safeguard.check_pointwise(data, -data, late_bound=late_bound)
    assert np.all(ok == np.array([True, True, True, True, False, True]).reshape(2, 3))


def test_late_bound_eb_rel():
    safeguard = StencilQuantityOfInterestErrorBoundSafeguard(
        qoi="x",
        neighbourhood=[
            dict(axis=0, before=1, after=0, boundary="valid"),
            dict(axis=1, before=1, after=1, boundary="valid"),
        ],
        type="rel",
        eb="eb",
    )

    data = np.arange(6).reshape(2, 3)

    late_bound = Bindings(eb=np.array([[2]]))

    valid = safeguard.compute_safe_intervals(data, late_bound=late_bound)
    # these bounds could be laxer but we're currently distributing across the
    #  full neighbourhood
    assert np.all(valid._lower == (data.flatten() - np.array([8, 8, 8, 8, 8, 8])))
    assert np.all(valid._upper == (data.flatten() + np.array([8, 8, 8, 8, 8, 8])))

    ok = safeguard.check_pointwise(data, -data, late_bound=late_bound)
    assert np.all(ok == np.array([True, True, True, True, True, True]).reshape(2, 3))


def test_late_bound_eb_ratio():
    safeguard = StencilQuantityOfInterestErrorBoundSafeguard(
        qoi="x",
        neighbourhood=[
            dict(axis=0, before=1, after=0, boundary="valid"),
            dict(axis=1, before=1, after=1, boundary="valid"),
        ],
        type="ratio",
        eb="eb",
    )

    data = np.arange(6).reshape(2, 3)

    late_bound = Bindings(eb=np.array([[2]]))

    valid = safeguard.compute_safe_intervals(data, late_bound=late_bound)
    # these bounds could be laxer but we're currently distributing across the
    #  full neighbourhood
    assert np.all(valid._lower == (data.flatten() - np.array([2, 2, 2, 2, 2, 2])))
    assert np.all(valid._upper == (data.flatten() + np.array([4, 4, 4, 4, 4, 4])))

    ok = safeguard.check_pointwise(data, -data, late_bound=late_bound)
    assert np.all(ok == np.array([True, True, True, True, False, True]).reshape(2, 3))
