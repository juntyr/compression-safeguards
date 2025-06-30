from enum import Enum, auto

import sympy as sp
import sympy.tensor.array.expressions  # noqa: F401
from typing_extensions import assert_never  # MSPV 3.11

from ...utils.cast import _isfinite
from .array import NumPyLikeArray
from .vars import LateBoundConstant


def create_finite_difference_for_neighbourhood(
    X: sp.tensor.array.expressions.ArraySymbol,
    shape: tuple[int, ...],
    I: tuple[int, ...],  # noqa: E741
):
    def finite_difference(
        expr, /, *, order, accuracy, type, axis, grid_spacing=None, grid_centre=None
    ):
        assert isinstance(expr, sp.Basic), (
            "finite_difference expr must be a scalar array element expression"
        )
        assert expr.has(sp.tensor.array.expressions.ArrayElement) and (
            not expr.has(NumPyLikeArray)
        ), (
            "cannot compute the finite_difference with respect to an array expression, provide a scalar expression (e.g. the centre value) instead"
        )
        assert any(not isinstance(s, LateBoundConstant) for s in expr.free_symbols), (
            "finite_difference expr must not be constant"
        )

        assert isinstance(order, sp.Integer), (
            "finite_difference order must be an integer"
        )
        order = int(order)
        assert order >= 0, "finite_difference order must be non-negative"
        assert isinstance(accuracy, sp.Integer), (
            "finite_difference accuracy must be an integer"
        )
        accuracy = int(accuracy)
        assert accuracy > 0, "finite_difference accuracy must be positive"

        assert isinstance(type, sp.Integer), "finite_difference type must be an integer"
        assert type in (-1, 0, +1), (
            "finite_difference type must be 1 (forward), 0 (central), or -1 (backward)"
        )
        type = [
            _FiniteDifference.central,
            _FiniteDifference.forward,
            _FiniteDifference.backwards,
        ][type]

        if type == _FiniteDifference.central:
            assert accuracy % 2 == 0, (
                "finite_difference accuracy must be even for a central finite difference"
            )

        assert isinstance(axis, sp.Integer), "finite_difference axis must be an integer"
        axis = int(axis)
        assert axis >= -len(shape) and axis < len(shape), (
            "finite_difference axis must be in range of the dimension of the neighbourhood"
        )

        offsets = _finite_difference_offsets(type, order, accuracy)

        if (grid_spacing is not None) and (grid_centre is None):
            assert isinstance(grid_spacing, sp.Number) or (
                isinstance(grid_spacing, sp.Expr)
                and (not grid_spacing.has(NumPyLikeArray))
                and all(
                    isinstance(s, LateBoundConstant) for s in grid_spacing.free_symbols
                )
            ), (
                "finite_difference grid_spacing must be a non-zero finite number or a constant scalar expression"
            )
            if isinstance(grid_spacing, sp.Number):
                assert grid_spacing != 0, (
                    "finite_difference grid_spacing must not be zero"
                )
                assert isinstance(grid_spacing, (sp.Integer, sp.Rational)) or _isfinite(
                    float(grid_spacing)
                ), "finite_difference grid_spacing must be finite"

            coefficients = _finite_difference_coefficients(
                order,
                sp.Integer(0),
                tuple(sp.Integer(o) * grid_spacing for o in offsets),
            )
        elif (grid_centre is not None) and (grid_spacing is None):
            assert (
                isinstance(grid_centre, sp.Expr)
                and grid_centre.has(sp.tensor.array.expressions.ArrayElement)
                and (not grid_centre.has(NumPyLikeArray))
                and all(
                    isinstance(s, LateBoundConstant) for s in grid_centre.free_symbols
                )
            ), (
                "finite_difference grid_centre must be a constant scalar array element expression"
            )

            coefficients = _finite_difference_coefficients(
                order,
                grid_centre,
                tuple(
                    _apply_finite_difference_offset(grid_centre, axis, o, X, shape, I)
                    for o in offsets
                ),
            )
        else:
            assert False, (
                "finite_difference takes either the grid_spacing or the grid_centre parameter"
            )

        return sp.Add(
            *[
                _apply_finite_difference_offset(expr, axis, o, X, shape, I) * c
                for o, c in zip(offsets, coefficients)
            ]
        )

    return finite_difference


def _apply_finite_difference_offset(
    expr: sp.Basic,
    axis: int,
    offset: int,
    X: sp.tensor.array.expressions.ArraySymbol,
    shape: tuple[int, ...],
    I: tuple[int, ...],  # noqa: E741
):
    assert expr.func is not NumPyLikeArray

    if expr.is_Number:
        return expr

    if (
        expr.func is sp.tensor.array.expressions.ArrayElement
        and len(expr.args) == 2
        and (
            expr.args[0] == X or isinstance(expr.args[0].name, LateBoundConstant)  # type: ignore
        )
    ):
        name, idxs = expr.args
        indices = list(idxs)  # type: ignore
        indices[axis] += offset
        assert indices[axis] >= 0, (
            f"cannot compute the finite_difference on axis {axis} since the neighbourhood for {name} is insufficiently large: before should be at least {I[axis] - indices[axis]}"
        )
        assert indices[axis] < shape[axis], (
            f"cannot compute the finite_difference on axis {axis} since the neighbourhood for {name} is insufficiently large: after should be at least {indices[axis] - I[axis]}"
        )
        return sp.tensor.array.expressions.ArrayElement(name, indices)

    return expr.func(
        *[
            _apply_finite_difference_offset(a, axis, offset, X, shape, I)
            for a in expr.args
        ]
    )


class _FiniteDifference(Enum):
    """
    Different types of finite differences.
    """

    central = auto()
    r"""
    Central finite difference, computed over the indices
    $\{i-k; \ldots; i; \ldots; i+k\}$.
    """

    forward = auto()
    r"""
    Forward finite difference, computed over the indices $\{i; \ldots; i+k\}$.
    """

    backwards = auto()
    r"""
    Backward finite difference, computed over the indices $\{i-k; \ldots; i\}$.
    """


def _finite_difference_offsets(
    type: _FiniteDifference,
    order: int,
    accuracy: int,
) -> tuple[int, ...]:
    match type:
        case _FiniteDifference.central:
            noffsets = order + (order % 2) - 1 + accuracy
            p = (noffsets - 1) // 2
            return (0,) + tuple(j for i in range(1, p + 1) for j in (i, -i))
        case _FiniteDifference.forward:
            return tuple(i for i in range(order + accuracy))
        case _FiniteDifference.backwards:
            return tuple(-i for i in range(order + accuracy))
        case _:
            assert_never(type)


def _finite_difference_coefficients(
    order: int,
    centre: sp.Expr,
    offsets: tuple[sp.Expr, ...],
) -> tuple[sp.Expr, ...]:
    """
    Finite difference coefficient algorithm from:

    Fornberg, B. (1988). Generation of finite difference formulas on arbitrarily
    spaced grids. *Mathematics of Computation*, 51(184), 699-706. Available from:
    [doi:10.1090/s0025-5718-1988-0935077-0](https://doi.org/10.1090/s0025-5718-1988-0935077-0).
    """

    x0 = centre
    M = order
    a = offsets
    N = len(a) - 1

    coeffs = {
        (0, 0, 0): sp.Integer(1),
    }

    c1 = sp.Integer(1)

    for n in range(1, N + 1):
        c2 = sp.Integer(1)
        for v in range(0, n):
            c3 = a[n] - a[v]
            c2 *= c3
            if n <= M:
                coeffs[(n, n - 1, v)] = sp.Integer(0)
            for m in range(0, min(n, M) + 1):
                if m > 0:
                    coeffs[(m, n, v)] = (
                        ((a[n] - x0) * coeffs[(m, n - 1, v)])
                        - (m * coeffs[(m - 1, n - 1, v)])
                    ) / c3
                else:
                    coeffs[(m, n, v)] = ((a[n] - x0) * coeffs[(m, n - 1, v)]) / c3
        for m in range(0, min(n, M) + 1):
            if m > 0:
                coeffs[(m, n, n)] = (c1 / c2) * (
                    (m * coeffs[(m - 1, n - 1, n - 1)])
                    - ((a[n - 1] - x0) * coeffs[(m, n - 1, n - 1)])
                )
            else:
                coeffs[(m, n, n)] = -(c1 / c2) * (
                    (a[n - 1] - x0) * coeffs[(m, n - 1, n - 1)]
                )
        c1 = c2

    return tuple(coeffs[M, N, v] for v in range(0, N + 1))
