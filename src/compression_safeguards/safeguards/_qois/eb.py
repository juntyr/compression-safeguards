from typing import Callable

import numpy as np

from ...utils.cast import (
    _isfinite,
    _isinf,
    _isnan,
    _nextafter,
)
from ...utils.typing import F, S


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
        return ~np.where(
            _isfinite(exprv),
            ((expr(eb_x_guess) - exprv) >= eb_expr_lower)
            & ((expr(eb_x_guess) - exprv) <= eb_expr_upper),
            np.where(
                _isinf(exprv),
                expr(eb_x_guess) == exprv,
                _isnan(expr(eb_x_guess)),
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
