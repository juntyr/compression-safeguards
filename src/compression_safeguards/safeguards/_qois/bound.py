from collections.abc import Callable
from enum import Enum, auto
from warnings import warn

import numpy as np

from ...utils._compat import (
    _broadcast_to,
    _ensure_array,
    _is_sign_negative_number,
    _is_sign_positive_number,
)
from ...utils.error import QuantityOfInterestRuntimeWarning
from .typing import Ci, F, J, Ns, Ps, Ps2, np_sndarray, np_sndarray2


def guarantee_arg_within_expr_bounds(
    expr: Callable[
        [np.ndarray[tuple[Ps], np.dtype[F]]], np.ndarray[tuple[Ps], np.dtype[F]]
    ],
    exprv: np.ndarray[tuple[Ps], np.dtype[F]],
    argv: np.ndarray[tuple[Ps], np.dtype[F]],
    argv_bound_guess: np.ndarray[tuple[Ps], np.dtype[F]],
    expr_lower: np.ndarray[tuple[Ps], np.dtype[F]],
    expr_upper: np.ndarray[tuple[Ps], np.dtype[F]],
) -> np.ndarray[tuple[Ps], np.dtype[F]]:
    """
    Ensure that `argv_bound_guess`, a guess for a lower or upper bound on
    argument `argv`, meets the lower and upper bounds `expr_lower` and
    `expr_upper` on the expression `expr`, where `exprv = expr(argv)`.

    Parameters
    ----------
    expr : Callable[[np.ndarray[tuple[Ps], np.dtype[F]]], np.ndarray[tuple[Ps], np.dtype[F]]]
        Evaluate the expression, given the argument `argv`.
    exprv : np.ndarray[tuple[Ps], np.dtype[F]]
        Evaluation of the expression on the argument `argv`.
    argv : np.ndarray[tuple[Ps], np.dtype[F]]
        Pointwise expression argument.
    argv_bound_guess : np.ndarray[tuple[Ps], np.dtype[F]]
        Provided guess for the bound on the argument `argv`.
    expr_lower : np.ndarray[tuple[Ps], np.dtype[F]]
        Pointwise lower bound on the expression, must be less than or equal to
        `exprv`.
    expr_upper : np.ndarray[tuple[Ps], np.dtype[F]]
        Pointwise upper bound on the expression, must be greater than or equal
        to `exprv`.

    Returns
    -------
    argv_bound_guess : np.ndarray[Ps, np.dtype[F]]
        Refined bound that guarantees that
        `expr_lower <= expr(argv_bound_guess) <= expr_upper` or
        `isnan(exprv) & isnan(expr(argv_bound_guess))`.
    """

    return _guarantee_data_within_expr_bounds_inner(
        expr,
        exprv,
        argv,
        argv_bound_guess,
        expr_lower,
        expr_upper,
        warn_on_bounds_exceeded=False,
    )


def guarantee_stacked_arg_within_expr_bounds(
    expr: Callable[
        [np.ndarray[tuple[J, Ps], np.dtype[F]]], np.ndarray[tuple[J, Ps], np.dtype[F]]
    ],
    exprv: np.ndarray[tuple[J, Ps], np.dtype[F]],
    argv: np.ndarray[tuple[J, Ps], np.dtype[F]],
    argv_bound_guess: np.ndarray[tuple[J, Ps], np.dtype[F]],
    expr_lower: np.ndarray[tuple[J, Ps], np.dtype[F]],
    expr_upper: np.ndarray[tuple[J, Ps], np.dtype[F]],
) -> np.ndarray[tuple[J, Ps], np.dtype[F]]:
    """
    Ensure that `argv_bound_guess`, a guess for a lower or upper bound on
    argument `argv`, meets the lower and upper bounds `expr_lower` and
    `expr_upper` on the expression `expr`, where `exprv = expr(argv)`.

    Parameters
    ----------
    expr : Callable[[np.ndarray[tuple[J, Ps], np.dtype[F]]], np.ndarray[tuple[J, Ps], np.dtype[F]]]
        Evaluate the expression, given the argument `argv`.
    exprv : np.ndarray[tuple[J, Ps], np.dtype[F]]
        Evaluation of the expression on the argument `argv`.
    argv : np.ndarray[tuple[J, Ps], np.dtype[F]]
        Pointwise expression argument.
    argv_bound_guess : np.ndarray[tuple[J, Ps], np.dtype[F]]
        Provided guess for the bound on the argument `argv`.
    expr_lower : np.ndarray[tuple[J, Ps], np.dtype[F]]
        Pointwise lower bound on the expression, must be less than or equal to
        `exprv`.
    expr_upper : np.ndarray[tuple[J, Ps], np.dtype[F]]
        Pointwise upper bound on the expression, must be greater than or equal
        to `exprv`.

    Returns
    -------
    argv_bound_guess : np.ndarray[tuple[J, Ps], np.dtype[F]]
        Refined bound that guarantees that
        `expr_lower <= expr(argv_bound_guess) <= expr_upper` or
        `isnan(exprv) & isnan(expr(argv_bound_guess))`.
    """

    return _guarantee_data_within_expr_bounds_inner(
        expr,
        exprv,
        argv,
        argv_bound_guess,
        expr_lower,
        expr_upper,
        warn_on_bounds_exceeded=False,
    )


def guarantee_data_within_expr_bounds(
    expr: Callable[
        [np_sndarray[Ps, Ns, np.dtype[F]]],
        np.ndarray[tuple[Ps], np.dtype[F]],
    ],
    exprv: np.ndarray[tuple[Ps], np.dtype[F]],
    Xs: np_sndarray[Ps, Ns, np.dtype[F]],
    Xs_bound_guess: np_sndarray[Ps, Ns, np.dtype[F]],
    expr_lower: np.ndarray[tuple[Ps], np.dtype[F]],
    expr_upper: np.ndarray[tuple[Ps], np.dtype[F]],
    *,
    warn_on_bounds_exceeded: bool,
) -> np_sndarray[Ps, Ns, np.dtype[F]]:
    """
    Ensure that `Xs_bound_guess`, a guess for a lower or upper bound on `Xs`,
    meets the lower and upper bounds `expr_lower` and `expr_upper` on the
    expression `expr`, where `exprv = expr(Xs)`.

    Parameters
    ----------
    expr : Callable[[np_sndarray[Ps, Ns, np.dtype[F]]], np.ndarray[tuple[Ps], np.dtype[F]]]
        Evaluate the expression, given the stencil-extended data `Xs`.
    exprv : np.ndarray[tuple[Ps], np.dtype[F]]
        Evaluation of the expression on the stencil-extended data `Xs`.
    Xs : np_sndarray[Ps, Ns, np.dtype[F]]
        Stencil-extended data.
    Xs_bound_guess : np_sndarray[Ps, Ns, np.dtype[F]]
        Provided guess for the bound on the stencil-extended data `Xs`.
    expr_lower : np.ndarray[tuple[Ps], np.dtype[F]]
        Pointwise lower bound on the expression, must be less than or equal to
        `exprv`.
    expr_upper : np.ndarray[tuple[Ps], np.dtype[F]]
        Pointwise upper bound on the expression, must be greater than or equal
        to `exprv`.
    warn_on_bounds_exceeded : bool
        If [`True`][True], a warning is emitted when the `Xs_bound_guess` does
        not already meet the `expr_lower` and `expr_upper` bounds.

    Returns
    -------
    Xs_bound_guess : np_sndarray[Ps, Ns, np.dtype[F]]
        Refined bound that guarantees that
        `expr_lower <= expr(Xs_bound_guess) <= expr_upper` or
        `isnan(exprv) & isnan(expr(Xs_bound_guess))`.
    """

    return _guarantee_data_within_expr_bounds_inner(
        expr,
        exprv,
        Xs,
        Xs_bound_guess,
        expr_lower,
        expr_upper,
        warn_on_bounds_exceeded=warn_on_bounds_exceeded,
    )


def _guarantee_data_within_expr_bounds_inner(
    expr: Callable[
        [np_sndarray2[Ps2, Ns, np.dtype[F]]],
        np.ndarray[Ps2, np.dtype[F]],
    ],
    exprv: np.ndarray[Ps2, np.dtype[F]],
    Xs: np_sndarray2[Ps2, Ns, np.dtype[F]],
    Xs_bound_guess: np_sndarray2[Ps2, Ns, np.dtype[F]],
    expr_lower: np.ndarray[Ps2, np.dtype[F]],
    expr_upper: np.ndarray[Ps2, np.dtype[F]],
    *,
    warn_on_bounds_exceeded: bool,
) -> np_sndarray2[Ps2, Ns, np.dtype[F]]:
    exprv = _ensure_array(exprv)
    Xs = _ensure_array(Xs)
    Xs_bound_guess = _ensure_array(Xs_bound_guess, copy=True)
    expr_lower = _ensure_array(expr_lower)
    expr_upper = _ensure_array(expr_upper)

    assert exprv.dtype == Xs.dtype
    assert Xs_bound_guess.dtype == Xs.dtype
    assert expr_lower.dtype == Xs.dtype
    assert expr_upper.dtype == Xs.dtype

    # check if any derived expression exceeds the expression bounds
    def exceeds_expr_bounds(
        Xs_bound_guess: np_sndarray2[Ps2, Ns, np.dtype[F]],
    ) -> np_sndarray2[Ps2, Ns, np.dtype[np.bool]]:
        exprv_Xs_bound_guess = expr(Xs_bound_guess)

        in_bounds: np.ndarray[Ps2, np.dtype[np.bool]] = (
            np.isnan(exprv) & np.isnan(exprv_Xs_bound_guess)
        ) | (
            np.greater_equal(exprv_Xs_bound_guess, expr_lower)
            # if lower >= +0.0, then guess must be >= +0.0
            & (
                ~_is_sign_positive_number(expr_lower)
                | _is_sign_positive_number(exprv_Xs_bound_guess)
            )
            & np.less_equal(exprv_Xs_bound_guess, expr_upper)
            # if upper <= -0.0, then guess must be <= -0.0
            & (
                ~_is_sign_negative_number(expr_upper)
                | _is_sign_negative_number(exprv_Xs_bound_guess)
            )
        )

        bounds_exceeded: np_sndarray2[Ps2, Ns, np.dtype[np.bool]] = _broadcast_to(
            (~in_bounds).reshape(exprv.shape + (1,) * (Xs.ndim - exprv.ndim)),
            Xs.shape,
        )
        return bounds_exceeded

    for _ in range(3):
        bounds_exceeded = exceeds_expr_bounds(Xs_bound_guess)

        if not np.any(bounds_exceeded):
            return Xs_bound_guess

        if warn_on_bounds_exceeded:
            warn_on_bounds_exceeded = False
            warn(
                "guaranteed data bounds do not meet the expression bounds",
                category=QuantityOfInterestRuntimeWarning,
            )

        # nudge the guess towards the data by 1 ULP
        np.nextafter(Xs_bound_guess, Xs, out=Xs_bound_guess, where=bounds_exceeded)

    Xs_diff = _ensure_array(np.nan_to_num(Xs_bound_guess - Xs))

    # exponential backoff for the distance
    backoff = Xs.dtype.type(0.5)

    for _ in range(6):
        bounds_exceeded = exceeds_expr_bounds(Xs_bound_guess)

        if not np.any(bounds_exceeded):
            return Xs_bound_guess

        # shove the guess towards the data by exponentially reducing the
        #  difference
        Xs_diff *= backoff
        backoff = np.divide(backoff, 2)
        np.add(Xs, Xs_diff, out=Xs_bound_guess, where=bounds_exceeded)

    warn(
        "data bounds required excessive nudging",
        category=QuantityOfInterestRuntimeWarning,
    )

    return _ensure_array(Xs, copy=True)


class DataBounds(Enum):
    # not yet checked, will likely need to be corrected
    unchecked = auto()
    # already checked, check should succeed
    checked = auto()
    # it is impossible for anything to go wrong, only use for wrappers that forward
    infallible = auto()


def checked_data_bounds(func: Ci) -> Ci:
    setattr(func, "_checked_data_bounds", DataBounds.checked)
    return func


def data_bounds(bounds: DataBounds) -> Callable[[Ci], Ci]:
    def data_bounds(func: Ci) -> Ci:
        setattr(func, "_checked_data_bounds", bounds)
        return func

    return data_bounds


def data_bounds_checks(func: Callable) -> DataBounds:
    return getattr(func, "_checked_data_bounds", DataBounds.unchecked)
