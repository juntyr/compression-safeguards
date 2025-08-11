from typing import Callable

import numpy as np

from ...utils.cast import (
    _isfinite,
    _isinf,
    _isnan,
    _nan_to_zero_inf_to_finite,
    _nextafter,
)
from ...utils.typing import F, S
from .expr.typing import Ns, Ps


def ensure_bounded_derived_error(
    expr: Callable[[np.ndarray[S, np.dtype[F]]], np.ndarray[S, np.dtype[F]]],
    exprv: np.ndarray[S, np.dtype[F]],
    xv: np.ndarray[S, np.dtype[F]],
    eb_x_guess: np.ndarray[S, np.dtype[F]],
    eb_expr_lower: np.ndarray[S, np.dtype[F]],
    eb_expr_upper: np.ndarray[S, np.dtype[F]],
) -> np.ndarray[S, np.dtype[F]]:
    """
    Ensure that an error bound on an expression is met by an error bound on
    the input data by nudging the provided guess.

    Parameters
    ----------
    expr : Callable[[np.ndarray[S, np.dtype[F]]], np.ndarray[S, np.dtype[F]]]
        Expression over which the error bound will be ensured.

        The expression takes in the error bound guess and returns the value of
        the expression for this error.
    exprv : np.ndarray[S, np.dtype[F]]
        Evaluation of the expression for the zero-error case.
    xv : np.ndarray[S, np.dtype[F]]
        Actual values of the input data, which are only used for better
        refinement of the error bound guess.
    eb_x_guess : np.ndarray[S, np.dtype[F]]
        Provided guess for the error bound on the initial data.
    eb_expr_lower : np.ndarray[S, np.dtype[F]]
        Finite pointwise lower bound on the expression error, must be negative
        or zero.
    eb_expr_upper : np.ndarray[S, np.dtype[F]]
        Finite pointwise upper bound on the expression error, must be positive
        or zero.

    Returns
    -------
    eb_x : np.ndarray[S, np.dtype[F]]
        Finite pointwise error bound on the input data.
    """

    # check if any derived expression exceeds the error bound
    # this check matches the QoI safeguard's validity check
    def is_eb_exceeded(eb_x_guess):
        exprv_x_guess = expr(eb_x_guess)
        return ~np.where(
            _isfinite(exprv),
            # check that eb_lower <= (guess - exprv) <= eb_upper
            ((exprv_x_guess - exprv) >= eb_expr_lower)
            & ((exprv_x_guess - exprv) <= eb_expr_upper)
            # check that (exprv + eb_lower) <= guess <= (exprv + eb_upper)
            & (exprv_x_guess >= (exprv + eb_expr_lower))
            & (exprv_x_guess <= (exprv + eb_expr_upper)),
            # why do we need to check both? because floaaaaaaaaaaaaaaaaaaaaaats
            np.where(
                _isinf(exprv),
                exprv_x_guess == exprv,
                _isnan(exprv_x_guess),
            ),
        ) & (eb_x_guess != 0)

    eb_exceeded = is_eb_exceeded(eb_x_guess)

    if not np.any(eb_exceeded):
        return eb_x_guess

    # first try to nudge the error bound itself
    # we can nudge with nextafter since the expression values are floating
    #  point
    eb_x_guess = np.where(eb_exceeded, _nextafter(eb_x_guess, 0), eb_x_guess)  # type: ignore

    # check again
    eb_exceeded = is_eb_exceeded(eb_x_guess)

    if not np.any(eb_exceeded):
        return eb_x_guess

    # second try to nudge it with respect to the data
    eb_x_guess = np.where(eb_exceeded, _nextafter(xv + eb_x_guess, xv) - xv, eb_x_guess)  # type: ignore

    # check again
    eb_exceeded = is_eb_exceeded(eb_x_guess)

    if not np.any(eb_exceeded):
        return eb_x_guess

    while True:
        # finally fall back to repeatedly cutting it in half
        eb_x_guess = np.where(eb_exceeded, eb_x_guess * 0.5, eb_x_guess)  # type: ignore

        eb_exceeded = is_eb_exceeded(eb_x_guess)

        if not np.any(eb_exceeded):
            return eb_x_guess


def ensure_bounded_expression(
    expr: Callable[[np.ndarray[Ps, np.dtype[F]]], np.ndarray[Ps, np.dtype[F]]],
    exprv: np.ndarray[Ps, np.dtype[F]],
    argv: np.ndarray[Ps, np.dtype[F]],
    argv_guess: np.ndarray[Ps, np.dtype[F]],
    expr_lower: np.ndarray[Ps, np.dtype[F]],
    expr_upper: np.ndarray[Ps, np.dtype[F]],
) -> np.ndarray[Ps, np.dtype[F]]:
    return ensure_bounded_expression_v2(
        expr, exprv, argv, argv_guess, expr_lower, expr_upper
    )


def ensure_bounded_expression_v2(
    expr: Callable[[np.ndarray[Ns, np.dtype[F]]], np.ndarray[Ps, np.dtype[F]]],
    exprv: np.ndarray[Ps, np.dtype[F]],
    Xs: np.ndarray[Ns, np.dtype[F]],
    Xs_guess: np.ndarray[Ns, np.dtype[F]],
    expr_lower: np.ndarray[Ps, np.dtype[F]],
    expr_upper: np.ndarray[Ps, np.dtype[F]],
) -> np.ndarray[Ns, np.dtype[F]]:
    # check if any derived expression exceeds the error bound
    # this check matches the QoI safeguard's validity check
    def are_bounds_exceeded(Xs_guess: np.ndarray[Ns, np.dtype[F]]):
        exprv_Xs_guess = expr(Xs_guess)
        return ~np.where(
            # NaN expressions must be preserved as NaN
            _isnan(exprv),
            _isnan(exprv_Xs_guess),
            # otherwise check that the expression result is in bounds
            ((exprv_Xs_guess >= expr_lower) & (exprv_Xs_guess <= expr_upper))
            # also allow equivalent inputs, which is needed for rewrites where
            #  the rewrite might evaluate outside the bounds even for the
            #  original input
            | (Xs_guess == Xs),
        )

    for _ in range(3):
        bounds_exceeded = are_bounds_exceeded(Xs_guess)

        if not np.any(bounds_exceeded):
            return Xs_guess

        # try to nudge the guess towards the data
        Xs_guess = np.where(bounds_exceeded, _nextafter(Xs_guess, Xs), Xs_guess)  # type: ignore

    Xs_diff = _nan_to_zero_inf_to_finite(Xs_guess - Xs)

    while True:
        bounds_exceeded = are_bounds_exceeded(Xs_guess)

        if not np.any(bounds_exceeded):
            return Xs_guess

        Xs_diff /= 2
        Xs_guess = np.where(bounds_exceeded, Xs + Xs_diff, Xs_guess)  # type: ignore

    bounds_exceeded = are_bounds_exceeded(Xs_guess)

    # TODO: improve with np.spacing
    return np.where(bounds_exceeded, Xs, Xs_guess)  # type: ignore

    below = Xs_guess < Xs

    last_backoff = 1
    backoff = 2

    while True:
        bounds_exceeded = are_bounds_exceeded(Xs_guess)

        if not np.any(bounds_exceeded):
            return Xs_guess

        # try to nudge the guess towards the data, using increasing steps
        nudge = _nextafter(Xs_guess, Xs) - Xs_guess

        if backoff == 13:
            print(bounds_exceeded)
            print(expr_lower)
            print(expr(Xs_guess))
            print(expr_upper)
            print(Xs)
            print(Xs_guess)
            print("\n=====\n")

        Xs_nudged = Xs_guess + nudge * backoff

        Xs_guess = np.where(  # type: ignore
            bounds_exceeded,
            np.where(
                below,
                np.minimum(Xs, Xs_nudged),
                np.maximum(Xs, Xs_nudged),
            ),
            Xs_guess,
        )

        # Fibonacci backoff
        last_backoff, backoff = backoff, backoff + last_backoff
