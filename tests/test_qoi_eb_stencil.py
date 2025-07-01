from itertools import cycle, permutations, product

import numpy as np
import pytest

from compression_safeguards import Safeguards
from compression_safeguards.safeguards.stencil import BoundaryCondition
from compression_safeguards.safeguards.stencil.qoi.eb import (
    StencilQuantityOfInterestErrorBoundSafeguard,
)
from compression_safeguards.utils.bindings import Bindings
from compression_safeguards.utils.cast import to_float

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
                    ("rel", 1.0),
                    ("ratio", 1.1),
                    ("abs", 0.01),
                    ("rel", 0.0),
                    ("ratio", 10.0),
                    ("abs", 1.0),
                    ("rel", 0.1),
                    ("ratio", 1.01),
                    ("abs", 0.1),
                    ("rel", 0.01),
                    ("ratio", 1.0),
                    ("abs", 0.0),
                    ("rel", 10.0),
                    ("ratio", 2.0),
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
                                    constant_boundary=42
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
                print(encode_decode, qoi, shape, axes, boundaries, type, eb)
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
    with pytest.raises(AssertionError, match="invalid QoI expression"):
        check_all_codecs(np.empty(0), "e x p", [(0, 0)])


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


def test_variables():
    with pytest.raises(AssertionError, match="invalid QoI expression"):
        check_all_codecs(np.array([]), 'V["123"]', [(0, 0)])
    with pytest.raises(AssertionError, match="invalid QoI expression"):
        check_all_codecs(np.array([]), 'V["a 123"]', [(0, 0)])
    with pytest.raises(AssertionError, match="identifier"):
        check_all_codecs(np.array([]), 'V["$a"]', [(0, 0)])
    with pytest.raises(AssertionError, match=r'unresolved variable V\["a"\]'):
        check_all_codecs(np.array([]), 'V["a"]', [(0, 0)])
    with pytest.raises(AssertionError, match=r'unresolved variable V\["b"\]'):
        check_all_codecs(np.array([]), 'let(V["a"], 3)(x + V["b"])', [(0, 0)])
    with pytest.raises(AssertionError, match="let name"):
        check_all_codecs(np.array([]), "let(1, 2)(x)", [(0, 0)])
    with pytest.raises(AssertionError, match="let value"):
        check_all_codecs(np.array([]), 'let(V["a"], log)(x + V["a"])', [(0, 0)])
    with pytest.raises(AssertionError, match="let within"):
        check_all_codecs(np.array([]), 'let(V["a"], x + 1)(log)', [(0, 0)])
    with pytest.raises(AssertionError, match=r"fresh \(not overridden\)"):
        check_all_codecs(
            np.array([]), 'let(V["a"], x + 1)(let(V["a"], V["a"])(V["a"]))', [(0, 0)]
        )
    with pytest.raises(AssertionError, match="pairs of names and values"):
        check_all_codecs(
            np.array([]), 'let(V["a"], x + 1, V["b"])(V["a"] + V["b"])', [(0, 0)]
        )
    check_all_codecs(np.array([]), 'let(V["a"], 3)(x + V["a"])', [(0, 0)])
    check_all_codecs(
        np.array([]),
        'let(V["a"], 3)(x + let(V["b"], V["a"] - 1)(V["b"] * 2))',
        [(0, 0)],
    )
    check_all_codecs(
        np.array([]), 'let(V["a"], x + 1, V["b"], x - 1)(V["a"] + V["b"])', [(0, 0)]
    )
    check_all_codecs(np.array([]), 'c["$x"] * x', [(0, 0)])

    with pytest.raises(AssertionError, match="out of border"):
        check_all_codecs(np.array([]), 'let(V["a"], X + 1)(V["a"][1])', [(0, 0)])
    with pytest.raises(AssertionError, match="index greater"):
        check_all_codecs(np.array([]), 'let(V["a"], X + 1)(V["a"][0,1])', [(0, 0)])
    check_all_codecs(np.array([]), 'let(V["a"], 3)(X + V["a"])[0]', [(0, 0)])
    check_all_codecs(np.array([]), 'let(V["a"], X)(V["a"][0])', [(0, 0)])
    check_all_codecs(np.array([]), 'let(V["a"], X)(V["a"][0,0])', [(0, 0), (0, 0)])
    check_all_codecs(np.array([]), 'let(V["a"], X)(V["a"][I])', [(0, 0), (0, 0)])
    check_all_codecs(np.array([]), 'asum(C["$X"] * X)', [(1, 1)])


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


def test_finite_difference():
    data = np.arange(81, dtype=float).reshape(9, 9)
    valid_5x5_neighbourhood = [
        dict(axis=0, before=4, after=4, boundary="valid"),
        dict(axis=1, before=4, after=4, boundary="valid"),
    ]

    safeguard = StencilQuantityOfInterestErrorBoundSafeguard(
        "finite_difference(x,order=0,accuracy=2,type=0,axis=0,grid_spacing=1)",
        valid_5x5_neighbourhood,
        "abs",
        0,
    )
    assert f"{safeguard._qoi_expr}" == "X[4, 4]"
    check_all_codecs(
        data,
        "finite_difference(x,order=0,accuracy=2,type=0,axis=0,grid_spacing=1)",
        [(4, 4), (4, 4)],
    )

    safeguard = StencilQuantityOfInterestErrorBoundSafeguard(
        "finite_difference(x,order=1,accuracy=1,type=1,axis=0,grid_spacing=1)",
        valid_5x5_neighbourhood,
        "abs",
        0,
    )
    assert f"{safeguard._qoi_expr}" == "-X[4, 4] + X[5, 4]"
    check_all_codecs(
        data,
        "finite_difference(x,order=1,accuracy=1,type=1,axis=0,grid_spacing=1)",
        [(4, 4), (4, 4)],
    )

    safeguard = StencilQuantityOfInterestErrorBoundSafeguard(
        "finite_difference(x,order=1,accuracy=1,type=-1,axis=0,grid_spacing=1)",
        valid_5x5_neighbourhood,
        "abs",
        0,
    )
    assert f"{safeguard._qoi_expr}" == "-X[3, 4] + X[4, 4]"
    check_all_codecs(
        data,
        "finite_difference(x,order=1,accuracy=1,type=-1,axis=0,grid_spacing=1)",
        [(4, 4), (4, 4)],
    )

    safeguard = StencilQuantityOfInterestErrorBoundSafeguard(
        "finite_difference(x,order=1,accuracy=2,type=0,axis=0,grid_spacing=1)",
        valid_5x5_neighbourhood,
        "abs",
        0,
    )
    assert f"{safeguard._qoi_expr}" == "-X[3, 4]/2 + X[5, 4]/2"
    check_all_codecs(
        data,
        "finite_difference(x,order=1,accuracy=2,type=0,axis=0,grid_spacing=1)",
        [(4, 4), (4, 4)],
    )

    safeguard = StencilQuantityOfInterestErrorBoundSafeguard(
        "finite_difference(x,order=1,accuracy=2,type=0,axis=1,grid_spacing=1)",
        valid_5x5_neighbourhood,
        "abs",
        0,
    )
    assert f"{safeguard._qoi_expr}" == "-X[4, 3]/2 + X[4, 5]/2"
    check_all_codecs(
        data,
        "finite_difference(x,order=1,accuracy=2,type=0,axis=1,grid_spacing=1)",
        [(4, 4), (4, 4)],
    )

    safeguard = StencilQuantityOfInterestErrorBoundSafeguard(
        "finite_difference(x,order=2,accuracy=2,type=0,axis=0,grid_spacing=1)",
        valid_5x5_neighbourhood,
        "abs",
        0,
    )
    assert f"{safeguard._qoi_expr}" == "X[3, 4] - 2*X[4, 4] + X[5, 4]"
    check_all_codecs(
        data,
        "finite_difference(x,order=2,accuracy=2,type=0,axis=0,grid_spacing=1)",
        [(4, 4), (4, 4)],
    )

    safeguard = StencilQuantityOfInterestErrorBoundSafeguard(
        "finite_difference(finite_difference(x,order=1,accuracy=2,type=0,axis=0,grid_spacing=1),order=1,accuracy=2,type=0,axis=0,grid_spacing=1)",
        valid_5x5_neighbourhood,
        "abs",
        0,
    )
    assert f"{safeguard._qoi_expr}" == "X[2, 4]/4 - X[4, 4]/2 + X[6, 4]/4"
    check_all_codecs(
        data,
        "finite_difference(finite_difference(x,order=1,accuracy=2,type=0,axis=0,grid_spacing=1),order=1,accuracy=2,type=0,axis=0,grid_spacing=1)",
        [(4, 4), (4, 4)],
    )

    safeguard = StencilQuantityOfInterestErrorBoundSafeguard(
        "finite_difference(finite_difference(x,order=1,accuracy=2,type=0,axis=0,grid_spacing=1),order=1,accuracy=2,type=0,axis=1,grid_spacing=1)",
        valid_5x5_neighbourhood,
        "abs",
        0,
    )
    assert f"{safeguard._qoi_expr}" == "X[3, 3]/4 - X[3, 5]/4 - X[5, 3]/4 + X[5, 5]/4"
    check_all_codecs(
        data,
        "finite_difference(finite_difference(x,order=1,accuracy=2,type=0,axis=0,grid_spacing=1),order=1,accuracy=2,type=0,axis=1,grid_spacing=1)",
        [(4, 4), (4, 4)],
    )


def test_finite_difference_array():
    data = np.arange(5)
    decoded = np.zeros(5)

    with pytest.raises(AssertionError, match="with respect to an array expression"):
        encode_decode_mock(
            data,
            decoded,
            safeguards=[
                dict(
                    kind="qoi_eb_stencil",
                    qoi="finite_difference(X[1:-1], order=1, accuracy=2, type=0, axis=0, grid_spacing=1)",
                    neighbourhood=[
                        dict(
                            axis=0,
                            before=1,
                            after=1,
                            boundary="valid",
                        )
                    ],
                    type="abs",
                    eb=0.1,
                ),
            ],
        )


def test_finite_difference_constant_grid_spacing():
    data = np.arange(5, dtype=float)

    safeguard = StencilQuantityOfInterestErrorBoundSafeguard(
        qoi='finite_difference(x, order=1, accuracy=2, type=0, axis=0, grid_spacing=c["dx"])',
        neighbourhood=[
            dict(
                axis=0,
                before=1,
                after=1,
                boundary="valid",
            )
        ],
        type="abs",
        eb=0.1,
    )

    safeguard.compute_safe_intervals(data, late_bound=Bindings(dx=1.0))
    safeguard.compute_safe_intervals(
        data, late_bound=Bindings(dx=np.array([0.1, 0.2, 0.3, 0.2, 0.1]))
    )

    with pytest.raises(
        AssertionError,
        match="grid_spacing must be a non-zero finite number or a constant scalar expression",
    ):
        safeguard = StencilQuantityOfInterestErrorBoundSafeguard(
            qoi='finite_difference(x, order=1, accuracy=2, type=0, axis=0, grid_spacing=C["dx"])',
            neighbourhood=[
                dict(
                    axis=0,
                    before=1,
                    after=1,
                    boundary="valid",
                )
            ],
            type="abs",
            eb=0.1,
        )

    safeguard = StencilQuantityOfInterestErrorBoundSafeguard(
        qoi='finite_difference(x, order=1, accuracy=2, type=0, axis=0, grid_spacing=sin(c["dx"])**2)',
        neighbourhood=[
            dict(
                axis=0,
                before=1,
                after=1,
                boundary="valid",
            )
        ],
        type="abs",
        eb=0.1,
    )

    safeguard.compute_safe_intervals(data, late_bound=Bindings(dx=1.0))
    safeguard.compute_safe_intervals(
        data, late_bound=Bindings(dx=np.array([0.1, 0.2, 0.3, 0.2, 0.1]))
    )


def test_finite_difference_arbitrary_grid():
    data = np.arange(5, dtype=float)

    safeguard = StencilQuantityOfInterestErrorBoundSafeguard(
        qoi='finite_difference(x, order=1, accuracy=2, type=0, axis=0, grid_centre=c["i"])',
        neighbourhood=[
            dict(
                axis=0,
                before=1,
                after=1,
                boundary="valid",
            )
        ],
        type="abs",
        eb=0.1,
    )

    safeguard.compute_safe_intervals(
        data, late_bound=Bindings(i=np.array([0.1, 0.2, 0.4, 0.5, 0.8]))
    )

    with pytest.raises(
        AssertionError,
        match="grid_centre must be a constant scalar array element expression",
    ):
        safeguard = StencilQuantityOfInterestErrorBoundSafeguard(
            qoi='finite_difference(x, order=1, accuracy=2, type=0, axis=0, grid_centre=C["i"])',
            neighbourhood=[
                dict(
                    axis=0,
                    before=1,
                    after=1,
                    boundary="valid",
                )
            ],
            type="abs",
            eb=0.1,
        )

    safeguard = StencilQuantityOfInterestErrorBoundSafeguard(
        qoi='finite_difference(x, order=1, accuracy=2, type=0, axis=0, grid_centre=sin(c["i"])**2)',
        neighbourhood=[
            dict(
                axis=0,
                before=1,
                after=1,
                boundary="valid",
            )
        ],
        type="abs",
        eb=0.1,
    )

    safeguard.compute_safe_intervals(
        data, late_bound=Bindings(i=np.array([0.1, 0.2, 0.4, 0.5, 0.8]))
    )


def test_finite_difference_periodic_grid():
    with pytest.raises(
        AssertionError,
        match="grid_period must be a positive finite number",
    ):
        safeguard = StencilQuantityOfInterestErrorBoundSafeguard(
            qoi='finite_difference(x, order=1, accuracy=2, type=0, axis=0, grid_centre=c["i"], grid_period=c["p"])',
            neighbourhood=[
                dict(
                    axis=0,
                    before=1,
                    after=1,
                    boundary="valid",
                )
            ],
            type="abs",
            eb=0.1,
        )

    safeguard = StencilQuantityOfInterestErrorBoundSafeguard(
        qoi='finite_difference(x, order=1, accuracy=2, type=0, axis=0, grid_centre=c["i"], grid_period=1)',
        neighbourhood=[
            dict(
                axis=0,
                before=1,
                after=1,
                boundary="valid",
            )
        ],
        type="abs",
        eb=0.1,
    )

    safeguard.compute_safe_intervals(
        np.arange(5, dtype=float),
        late_bound=Bindings(i=np.array([0.1, 0.2, 0.4, 0.5, 0.8])),
    )

    safeguard = StencilQuantityOfInterestErrorBoundSafeguard(
        qoi="finite_difference(x, order=1, accuracy=2, type=0, axis=0, grid_spacing=0.75, grid_period=1)",
        neighbourhood=[
            dict(
                axis=0,
                before=1,
                after=1,
                boundary="valid",
            )
        ],
        type="abs",
        eb=0.1,
    )

    safeguard.compute_safe_intervals(
        np.arange(5, dtype=np.uint64), late_bound=Bindings.empty()
    )


@pytest.mark.parametrize("dtype", sorted(d.name for d in Safeguards.supported_dtypes()))
def test_periodic_delta_transform(dtype):
    def delta_transform(x, period):
        half_period = period / 2

        # map delta to [-period/2, +period/2]
        # ((... % period) + period) % period is required for numpy_quaddtype
        return ((((x + half_period) % period) + period) % period) - half_period

    float_dtype = to_float(np.array((), dtype=dtype)).dtype

    assert (
        delta_transform(
            np.array(-0.75, dtype=float_dtype), np.array(1, dtype=float_dtype)
        )
        > 0
    )
    assert (
        delta_transform(
            np.array(-0.25, dtype=float_dtype), np.array(1, dtype=float_dtype)
        )
        < 0
    )
    assert (
        delta_transform(
            np.array(0.0, dtype=float_dtype), np.array(1, dtype=float_dtype)
        )
        == 0
    )
    assert (
        delta_transform(
            np.array(0.25, dtype=float_dtype), np.array(1, dtype=float_dtype)
        )
        > 0
    )
    assert (
        delta_transform(
            np.array(0.75, dtype=float_dtype), np.array(1, dtype=float_dtype)
        )
        < 0
    )

    for x, period in [
        (np.linspace(-15, 15), 10),
        (np.linspace(-np.pi * 25, np.pi * 25), np.pi * 10),
        (np.linspace(-800, 800), 360),
    ]:
        periodic = delta_transform(x.astype(dtype), period)
        assert np.all(periodic >= (-period / 2))
        assert np.all(periodic <= (period / 2))

        periodic = delta_transform(to_float(x.astype(dtype)), period)
        assert np.all(periodic >= (-period / 2))
        assert np.all(periodic <= (period / 2))


def test_matmul():
    data = np.arange(16, dtype=float).reshape(4, 4)
    valid_3x3_neighbourhood = [
        dict(axis=0, before=1, after=1, boundary="valid"),
        dict(axis=1, before=1, after=1, boundary="valid"),
    ]

    safeguard = StencilQuantityOfInterestErrorBoundSafeguard(
        "matmul(A[[-1, -2, -3]], matmul(X, tr(A[[1, 2, 3]])))[0,0]",
        valid_3x3_neighbourhood,
        "abs",
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
    safeguard = StencilQuantityOfInterestErrorBoundSafeguard(
        "X[I-1] + X[I+2]",
        [dict(axis=0, before=1, after=4, boundary="valid")],
        "abs",
        0,
    )
    assert f"{safeguard._qoi_expr}" == "X[0] + X[3]"

    safeguard = StencilQuantityOfInterestErrorBoundSafeguard(
        "X[I[0]][I[1]]",
        [
            dict(axis=0, before=1, after=1, boundary="valid"),
            dict(axis=1, before=1, after=1, boundary="valid"),
        ],
        "abs",
        0,
    )
    assert f"{safeguard._qoi_expr}" == "X[1, 1]"

    safeguard = StencilQuantityOfInterestErrorBoundSafeguard(
        "X[0][0]",
        [
            dict(axis=0, before=1, after=1, boundary="valid"),
            dict(axis=1, before=1, after=1, boundary="valid"),
        ],
        "abs",
        0,
    )
    assert f"{safeguard._qoi_expr}" == "X[0, 0]"

    safeguard = StencilQuantityOfInterestErrorBoundSafeguard(
        "(X[I+A[-1,0]]+X[I+A[+1,0]]+X[I+A[0,-1]]+X[I+A[0,+1]])/4",
        [
            dict(axis=0, before=1, after=1, boundary="valid"),
            dict(axis=1, before=1, after=1, boundary="valid"),
        ],
        "abs",
        0,
    )
    assert f"{safeguard._qoi_expr}" == "X[0, 1]/4 + X[1, 0]/4 + X[1, 2]/4 + X[2, 1]/4"

    safeguard = StencilQuantityOfInterestErrorBoundSafeguard(
        "asum(X * A[[0.25, 0.5, 0.25], [0.5, 1.0, 0.5], [0.25, 0.5, 0.25]])",
        [
            dict(axis=0, before=1, after=1, boundary="valid"),
            dict(axis=1, before=1, after=1, boundary="valid"),
        ],
        "abs",
        0,
    )
    assert (
        f"{safeguard._qoi_expr}"
        == "0.25*X[0, 0] + 0.5*X[0, 1] + 0.25*X[0, 2] + 0.5*X[1, 0] + 1.0*X[1, 1] + 0.5*X[1, 2] + 0.25*X[2, 0] + 0.5*X[2, 1] + 0.25*X[2, 2]"
    )


def test_lambdify_indexing():
    import inspect

    from compression_safeguards.safeguards._qois.compile import sympy_expr_to_numpy

    safeguard = StencilQuantityOfInterestErrorBoundSafeguard(
        "asum(X * A[[0.25, 0.5, 0.25], [0.5, 1.0, 0.5], [0.25, 0.5, 0.25]])",
        [
            dict(axis=0, before=1, after=1, boundary="valid"),
            dict(axis=1, before=1, after=1, boundary="valid"),
        ],
        "abs",
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


def test_fuzzer_finite_difference_int_iter():
    data = np.array([65373], dtype=np.uint16)
    decoded = np.array([42246], dtype=np.uint16)

    for boundary in BoundaryCondition:
        encode_decode_mock(
            data,
            decoded,
            safeguards=[
                dict(
                    kind="qoi_eb_stencil",
                    qoi="finite_difference(x, order=0, accuracy=1, type=-1, axis=0, grid_spacing=2.2250738585072014e-308)",
                    neighbourhood=[
                        dict(
                            axis=0,
                            before=0,
                            after=0,
                            boundary=boundary,
                            constant_boundary=42
                            if boundary == BoundaryCondition.constant
                            else None,
                        )
                    ],
                    type="abs",
                    eb=2.2250738585072014e-308,
                ),
            ],
        )


def test_fuzzer_finite_difference_fraction_overflow():
    data = np.array([7], dtype=np.int8)
    decoded = np.array([0], dtype=np.int8)

    for boundary in BoundaryCondition:
        encode_decode_mock(
            data,
            decoded,
            safeguards=[
                dict(
                    kind="qoi_eb_stencil",
                    qoi="finite_difference(x, order=7, accuracy=6, type=-1, axis=0, grid_spacing=7.215110354450764e305)",
                    neighbourhood=[
                        dict(
                            axis=0,
                            before=12,
                            after=0,
                            boundary=boundary,
                            constant_boundary=42
                            if boundary == BoundaryCondition.constant
                            else None,
                        )
                    ],
                    type="abs",
                    eb=2.2250738585072014e-308,
                ),
            ],
        )


def test_fuzzer_finite_difference_fraction_compare():
    data = np.array([1978047305655558])

    for boundary in BoundaryCondition:
        encode_decode_zero(
            data,
            safeguards=[
                dict(kind="same", value=7),
                dict(
                    kind="qoi_eb_stencil",
                    qoi="finite_difference(x, order=7, accuracy=7, type=1, axis=0, grid_spacing=2.2250738585072014e-308)",
                    neighbourhood=[
                        dict(
                            axis=0,
                            before=0,
                            after=13,
                            boundary=boundary,
                            constant_boundary=42
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


def test_fuzzer_finite_difference_eb_abs():
    data = np.array([[-27, 8, 8], [8, 8, 8], [8, 8, 8]], dtype=np.int8)
    decoded = np.array([[8, 8, 8], [8, 8, 8], [8, 8, 8]], dtype=np.int8)

    for boundary in BoundaryCondition:
        encode_decode_mock(
            data,
            decoded,
            safeguards=[
                dict(
                    kind="qoi_eb_stencil",
                    qoi="finite_difference(x, order=4, accuracy=4, type=1, axis=0, grid_spacing=4)",
                    neighbourhood=[
                        dict(
                            axis=0,
                            before=0,
                            after=7,
                            boundary=boundary,
                            constant_boundary=42
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


def test_fuzzer_finite_difference_fraction_float_overflow():
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
                    qoi="finite_difference(x, order=1, accuracy=3, type=1, axis=0, grid_spacing=59)",
                    neighbourhood=[
                        dict(
                            axis=0,
                            before=0,
                            after=3,
                            boundary=boundary,
                            constant_boundary=42
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


def test_late_bound_broadcast():
    safeguard = StencilQuantityOfInterestErrorBoundSafeguard(
        qoi="x",
        neighbourhood=[
            dict(axis=0, before=1, after=1, boundary="valid"),
            dict(axis=1, before=1, after=1, boundary="valid"),
        ],
        type="abs",
        eb="eb",
    )
    assert safeguard.late_bound == {"eb"}

    data = np.arange(16).reshape(4, 4)

    with pytest.raises(ValueError, match="could not be broadcast together"):
        safeguard.compute_safe_intervals(data, late_bound=Bindings(eb=np.ones((4, 4))))

    safeguard.compute_safe_intervals(data, late_bound=Bindings(eb=np.ones(tuple())))
    safeguard.compute_safe_intervals(data, late_bound=Bindings(eb=np.ones((1,))))
    safeguard.compute_safe_intervals(data, late_bound=Bindings(eb=np.ones((1, 1))))
    safeguard.compute_safe_intervals(data, late_bound=Bindings(eb=np.ones((2, 2))))

    with pytest.raises(ValueError, match="more dimensions"):
        safeguard.compute_safe_intervals(
            data, late_bound=Bindings(eb=np.ones((2, 2, 1)))
        )


def test_late_bound_lossless_cast():
    safeguard = StencilQuantityOfInterestErrorBoundSafeguard(
        qoi='x + c["c"]',
        neighbourhood=[
            dict(axis=0, before=0, after=0, boundary="valid"),
            dict(axis=1, before=0, after=0, boundary="valid"),
        ],
        type="abs",
        eb=0.1,
    )
    assert safeguard.late_bound == {"c"}

    data = np.arange(16, dtype=np.float32).reshape(4, 4)

    safeguard.compute_safe_intervals(data, late_bound=Bindings(c=np.iinfo(np.int8).max))
    safeguard.compute_safe_intervals(
        data, late_bound=Bindings(c=np.iinfo(np.uint8).max)
    )
    safeguard.compute_safe_intervals(
        data, late_bound=Bindings(c=np.iinfo(np.int16).max)
    )
    safeguard.compute_safe_intervals(
        data, late_bound=Bindings(c=np.iinfo(np.uint16).max)
    )

    with pytest.raises(ValueError, match="cannot losslessly cast"):
        safeguard.compute_safe_intervals(
            data, late_bound=Bindings(c=np.iinfo(np.int32).max)
        )
    with pytest.raises(ValueError, match="cannot losslessly cast"):
        safeguard.compute_safe_intervals(
            data, late_bound=Bindings(c=np.iinfo(np.uint32).max)
        )
    with pytest.raises(ValueError, match="cannot losslessly cast"):
        safeguard.compute_safe_intervals(
            data, late_bound=Bindings(c=np.iinfo(np.int64).max)
        )
    with pytest.raises(ValueError, match="cannot losslessly cast"):
        safeguard.compute_safe_intervals(
            data, late_bound=Bindings(c=np.iinfo(np.uint64).max)
        )

    safeguard.compute_safe_intervals(
        data, late_bound=Bindings(c=np.finfo(np.float16).smallest_subnormal)
    )
    safeguard.compute_safe_intervals(
        data, late_bound=Bindings(c=np.finfo(np.float32).smallest_subnormal)
    )

    with pytest.raises(ValueError, match="cannot losslessly cast"):
        safeguard.compute_safe_intervals(
            data, late_bound=Bindings(c=np.finfo(np.float64).smallest_subnormal)
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
    assert safeguard.late_bound == {"eb"}

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
    assert safeguard.late_bound == {"eb"}

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
    assert safeguard.late_bound == {"eb"}

    data = np.arange(6).reshape(2, 3)

    late_bound = Bindings(eb=np.array([[2]]))

    valid = safeguard.compute_safe_intervals(data, late_bound=late_bound)
    # these bounds could be laxer but we're currently distributing across the
    #  full neighbourhood
    assert np.all(valid._lower == (data.flatten() - np.array([2, 2, 2, 2, 2, 2])))
    assert np.all(valid._upper == (data.flatten() + np.array([4, 4, 4, 4, 4, 4])))

    ok = safeguard.check_pointwise(data, -data, late_bound=late_bound)
    assert np.all(ok == np.array([True, True, True, True, False, True]).reshape(2, 3))


def test_finite_difference_dx():
    data = np.array([1, 2, 3], dtype=np.int8)
    decoded = np.array([0, 0, 0], dtype=np.int8)

    for boundary in BoundaryCondition:
        safeguards = Safeguards(
            safeguards=[
                dict(
                    kind="qoi_eb_stencil",
                    qoi="finite_difference(x, order=1, accuracy=2, type=0, axis=0, grid_spacing=0.1)",
                    neighbourhood=[
                        dict(
                            axis=0,
                            before=1,
                            after=1,
                            boundary=boundary,
                            constant_boundary=42
                            if boundary == BoundaryCondition.constant
                            else None,
                        )
                    ],
                    type="abs",
                    eb=1,
                ),
            ]
        )

        correction = safeguards.compute_correction(data, decoded)
        corrected = safeguards.apply_correction(decoded, correction)

        data_finite_difference = safeguards.safeguards[0].evaluate_qoi(
            data, Bindings.empty()
        )
        corrected_finite_difference = safeguards.safeguards[0].evaluate_qoi(
            corrected, Bindings.empty()
        )

        assert data_finite_difference[len(data_finite_difference) // 2] == 10
        assert (
            np.abs(10 - corrected_finite_difference[len(data_finite_difference) // 2])
            <= 1
        )


def test_late_bound_constant():
    safeguard = StencilQuantityOfInterestErrorBoundSafeguard(
        qoi='X[0,0] * c["zero"] + X[I] / C["f"][I]',
        neighbourhood=[
            dict(axis=0, before=1, after=0, boundary="valid"),
            dict(axis=1, before=0, after=0, boundary="valid"),
        ],
        type="abs",
        eb=1,
    )
    assert safeguard.late_bound == {"f", "zero"}

    data = np.arange(6).reshape(2, 3)

    late_bound = Bindings(
        f=np.array([16, 8, 4]),
        zero=0,
    )

    valid = safeguard.compute_safe_intervals(data, late_bound=late_bound)
    assert np.all(valid._lower == (data.flatten() - np.array([16, 8, 4, 16, 8, 4])))
    assert np.all(valid._upper == (data.flatten() + np.array([16, 8, 4, 16, 8, 4])))

    ok = safeguard.check_pointwise(data, -data, late_bound=late_bound)
    assert np.all(ok == np.array([True, True, True, True, True, False]).reshape(2, 3))


@pytest.mark.parametrize("check", CHECKS)
def test_pointwise_normalised_absolute_error(check):
    # pointwise normalised / range-relative absolute error bound
    check('(x - c["$x_min"]) / (c["$x_max"] - c["$x_min"])')


def test_late_bound_constant_boundary():
    safeguard = StencilQuantityOfInterestErrorBoundSafeguard(
        qoi="X[0,0] - X[2,0]",
        neighbourhood=[
            dict(
                axis=0,
                before=1,
                after=1,
                boundary="constant",
                constant_boundary="const",
            ),
            dict(
                axis=1,
                before=0,
                after=0,
                boundary="constant",
                constant_boundary="const2",
            ),
        ],
        type="abs",
        eb=1,
    )
    assert safeguard.late_bound == {"const", "const2"}

    data = np.arange(6, dtype="uint8").reshape(2, 3)

    with pytest.raises(
        ValueError, match="cannot broadcast a non-scalar to a scalar array"
    ):
        safeguard.compute_safe_intervals(
            data, late_bound=Bindings(const=data, const2=4)
        )

    with pytest.raises(
        ValueError,
        match=r"cannot losslessly cast \(some\) late-bound parameter const2 values from int64 to uint8",
    ):
        safeguard.compute_safe_intervals(data, late_bound=Bindings(const=1, const2=256))

    safeguard.compute_safe_intervals(data, late_bound=Bindings(const=1, const2=255))
