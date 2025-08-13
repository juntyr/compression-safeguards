from typing import Callable

import numpy as np

from ...utils.cast import (
    _isnan,
    _nan_to_zero_inf_to_finite,
    _nextafter,
)
from .expr.typing import F, Ns, Ps


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
    def are_bounds_exceeded(
        Xs_guess: np.ndarray[Ns, np.dtype[F]],
    ) -> np.ndarray[Ns, np.dtype[F]]:
        exprv_Xs_guess = expr(Xs_guess)
        return np.broadcast_to(  # type: ignore
            ~np.where(
                # NaN expressions must be preserved as NaN
                _isnan(exprv),
                _isnan(exprv_Xs_guess),
                # otherwise check that the expression result is in bounds
                ((exprv_Xs_guess >= expr_lower) & (exprv_Xs_guess <= expr_upper))
                # also allow equivalent inputs, which is needed for rewrites where
                #  the rewrite might evaluate outside the bounds even for the
                #  original input
                | np.all(Xs_guess == Xs, axis=tuple(range(exprv.ndim, Xs.ndim))),
            ).reshape(exprv.shape + (1,) * (Xs.ndim - exprv.ndim)),
            Xs.shape,
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
