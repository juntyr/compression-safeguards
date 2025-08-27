from typing import Callable

import numpy as np

from ...utils._compat import (
    _broadcast_to,
    _isnan,
    _nan_to_zero_inf_to_finite,
    _nextafter,
    _where,
)
from ...utils.cast import as_bits
from .expr.typing import F, Ns, Ps


def guarantee_arg_within_expr_bounds(
    expr: Callable[[np.ndarray[Ps, np.dtype[F]]], np.ndarray[Ps, np.dtype[F]]],
    exprv: np.ndarray[Ps, np.dtype[F]],
    argv: np.ndarray[Ps, np.dtype[F]],
    argv_bound_guess: np.ndarray[Ps, np.dtype[F]],
    expr_lower: np.ndarray[Ps, np.dtype[F]],
    expr_upper: np.ndarray[Ps, np.dtype[F]],
) -> np.ndarray[Ps, np.dtype[F]]:
    """
    Ensure that `argv_bound_guess`, a guess for a lower or upper bound on
    argument `argv`, meets the lower and upper bounds `expr_lower` and
    `expr_upper` on the expression `expr`, where `exprv = expr(argv)`.

    Parameters
    ----------
    expr : Callable[[np.ndarray[Ps, np.dtype[F]]], np.ndarray[Ps, np.dtype[F]]]
        Evaluate the expression, given the argument `argv`.
    exprv : np.ndarray[Ps, np.dtype[F]]
        Evaluation of the expression on the argument `argv`.
    argv : np.ndarray[Ps, np.dtype[F]]
        Pointwise expression argument.
    argv_bound_guess : np.ndarray[Ps, np.dtype[F]]
        Provided guess for the bound on the argument `argv`.
    expr_lower : np.ndarray[Ps, np.dtype[F]]
        Pointwise lower bound on the expression, must be less than or equal to
        `exprv`.
    expr_upper : np.ndarray[Ps, np.dtype[F]]
        Pointwise upper bound on the expression, must be greater than or equal
        to `exprv`.

    Returns
    -------
    argv_bound_guess : np.ndarray[Ns, np.dtype[F]]
        Refined bound that guarantees that
        `expr_lower <= expr(argv_bound_guess) <= expr_upper` or
        `isnan(exprv) & isnan(expr(argv_bound_guess))`.
    """

    return guarantee_data_within_expr_bounds(
        expr, exprv, argv, argv_bound_guess, expr_lower, expr_upper
    )


def guarantee_data_within_expr_bounds(
    expr: Callable[[np.ndarray[Ns, np.dtype[F]]], np.ndarray[Ps, np.dtype[F]]],
    exprv: np.ndarray[Ps, np.dtype[F]],
    Xs: np.ndarray[Ns, np.dtype[F]],
    Xs_bound_guess: np.ndarray[Ns, np.dtype[F]],
    expr_lower: np.ndarray[Ps, np.dtype[F]],
    expr_upper: np.ndarray[Ps, np.dtype[F]],
) -> np.ndarray[Ns, np.dtype[F]]:
    """
    Ensure that `Xs_bound_guess`, a guess for a lower or upper bound on `Xs`,
    meets the lower and upper bounds `expr_lower` and `expr_upper` on the
    expression `expr`, where `exprv = expr(Xs)`.

    Parameters
    ----------
    expr : Callable[[np.ndarray[Ns, np.dtype[F]]], np.ndarray[Ps, np.dtype[F]]]
        Evaluate the expression, given the stencil-extended data `Xs`.
    exprv : np.ndarray[Ps, np.dtype[F]]
        Evaluation of the expression on the stencil-extended data `Xs`.
    Xs : np.ndarray[Ns, np.dtype[F]]
        Stencil-extended data.
    Xs_bound_guess : np.ndarray[Ns, np.dtype[F]]
        Provided guess for the bound on the stencil-extended data `Xs`.
    expr_lower : np.ndarray[Ps, np.dtype[F]]
        Pointwise lower bound on the expression, must be less than or equal to
        `exprv`.
    expr_upper : np.ndarray[Ps, np.dtype[F]]
        Pointwise upper bound on the expression, must be greater than or equal
        to `exprv`.

    Returns
    -------
    Xs_bound_guess : np.ndarray[Ns, np.dtype[F]]
        Refined bound that guarantees that
        `expr_lower <= expr(Xs_bound_guess) <= expr_upper` or
        `isnan(exprv) & isnan(expr(Xs_bound_guess))`.
    """

    exprv = np.array(exprv)
    Xs = np.array(Xs)
    Xs_bound_guess = np.array(Xs_bound_guess)
    expr_lower = np.array(expr_lower)
    expr_upper = np.array(expr_upper)

    assert exprv.dtype == Xs.dtype
    assert Xs_bound_guess.dtype == Xs.dtype
    assert expr_lower.dtype == Xs.dtype
    assert expr_upper.dtype == Xs.dtype

    # check if any derived expression exceeds the expression bounds
    def exceeds_expr_bounds(
        Xs_bound_guess: np.ndarray[Ns, np.dtype[F]],
    ) -> np.ndarray[Ns, np.dtype[np.bool]]:
        exprv_Xs_bound_guess = expr(Xs_bound_guess)

        in_bounds: np.ndarray[Ps, np.dtype[np.bool]] = _where(
            # NaN expressions must be preserved as NaN
            _isnan(exprv),
            _isnan(exprv_Xs_bound_guess),
            # otherwise check that the expression result is in bounds
            (
                (exprv_Xs_bound_guess >= expr_lower)
                & (exprv_Xs_bound_guess <= expr_upper)
            )
            # also allow bitwise equivalent inputs, which is needed for
            #  rewrites where the rewrite might evaluate outside the bounds
            #  even for the original input
            | np.all(
                as_bits(Xs_bound_guess, kind="V") == as_bits(Xs, kind="V"),
                axis=tuple(range(exprv.ndim, Xs.ndim)),
            ),
        )

        bounds_exceeded: np.ndarray[Ns, np.dtype[np.bool]] = _broadcast_to(
            (~in_bounds).reshape(exprv.shape + (1,) * (Xs.ndim - exprv.ndim)),
            Xs.shape,
        )
        return bounds_exceeded

    # FIXME: what to do about NaNs?

    for _ in range(3):
        bounds_exceeded = exceeds_expr_bounds(Xs_bound_guess)

        if not np.any(bounds_exceeded):
            return _where(Xs_bound_guess == Xs, Xs, Xs_bound_guess)

        # nudge the guess towards the data by 1 ULP
        Xs_bound_guess = _where(
            bounds_exceeded, _nextafter(Xs_bound_guess, Xs), Xs_bound_guess
        )

    Xs_diff = _nan_to_zero_inf_to_finite(Xs_bound_guess - Xs)

    # exponential backoff for the distance
    backoff = Xs.dtype.type(0.5)

    while True:
        bounds_exceeded = exceeds_expr_bounds(Xs_bound_guess)

        if not np.any(bounds_exceeded):
            return _where(Xs_bound_guess == Xs, Xs, Xs_bound_guess)

        # shove the guess towards the data by exponentially reducing the
        #  difference
        Xs_diff *= backoff
        backoff = np.divide(backoff, 2)
        Xs_bound_guess = _where(bounds_exceeded, Xs + Xs_diff, Xs_bound_guess)
