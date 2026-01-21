"""
Error bounds that can be guaranteed by various safeguards.
"""

__all__ = ["ErrorBound"]

from enum import Enum, auto
from typing import Never, assert_never

import numpy as np

from ..utils._compat import _ensure_array, _where
from ..utils.cast import from_float, from_total_order, to_float, to_total_order
from ..utils.error import ctx
from ..utils.typing import F, S, T


class ErrorBound(Enum):
    """
    Different types of error bounds that can be guaranteed by various
    safeguards, including:

    - [`ErrorBoundSafeguard`][...pointwise.eb.ErrorBoundSafeguard]
    - [`PointwiseQuantityOfInterestErrorBoundSafeguard`][...pointwise.qoi.eb.PointwiseQuantityOfInterestErrorBoundSafeguard]
    - [`StencilQuantityOfInterestErrorBoundSafeguard`][...stencil.qoi.eb.StencilQuantityOfInterestErrorBoundSafeguard]
    """

    abs = auto()
    r"""
    Absolute error bound, which guarantees that the pointwise absolute error is
    less than or equal to the provided bound $\epsilon_{abs}$:

    \[
    |x - \hat{x}| \leq \epsilon_{abs}
    \]

    or equivalently

    \[
    (x - \epsilon_{abs}) \leq \hat{x} \leq (x + \epsilon_{abs})
    \]

    for a finite $\epsilon_{abs} \geq 0$.
    """

    rel = auto()
    r"""
    Relative error bound, which guarantees that the pointwise relative error is
    less than or equal to the provided bound $\epsilon_{rel}$:

    \[
    |x - \hat{x}| \leq |x| \cdot \epsilon_{rel}
    \]

    or equivalently

    \[
    (x - |x| \cdot \epsilon_{rel}) \leq \hat{x} \leq (x + |x| \cdot \epsilon_{rel})
    \]

    for a finite $\epsilon_{rel} \geq 0$.

    The relative error bound preserves zero values with the same bit pattern.
    """

    ratio = auto()
    r"""
    Ratio error bound, which guarantees that the ratios between the original
    and the corrected values as well as their inverse ratios are less than or
    equal to the provided bound $\epsilon_{ratio}$:

    \[
        \left\{\begin{array}{lr}
            0 \quad &\text{if } x = \hat{x} = 0 \\
            \inf \quad &\text{if } \text{sign}(x) \neq \text{sign}(\hat{x}) \\
            |\log(|x|) - \log(|\hat{x}|)| \quad &\text{otherwise}
        \end{array}\right\} \leq \log(\epsilon_{ratio})
    \]

    or equivalently

    \[
    \begin{split}
        (x \mathbin{/} \epsilon_{ratio}) \leq \hat{x} \leq (x \cdot \epsilon_{ratio}) \quad &\text{if } x \geq 0 \\
        (x \cdot \epsilon_{ratio}) \leq \hat{x} \leq (x \mathbin{/} \epsilon_{ratio}) \quad &\text{otherwise}
    \end{split}
    \]

    for a finite $\epsilon_{ratio} \geq 1$.

    Since the $\epsilon_{ratio}$ bound is finite, ratio error bound also
    guarantees that the sign of each corrected value matches the sign of each
    original value and that a corrected value is zero if and only if it is zero
    in the original data.

    The ratio error bound is sometimes also known as a decimal error bound[^1]
    [^2] if the ratio is expressed as the difference in orders of magnitude. A
    decimal error bound of e.g. $2$ (two orders of magnitude difference / x100
    ratio) can be expressed using
    $\epsilon_{ratio} = {10}^{\epsilon_{decimal}}$.

    The ratio error bound can also be used to guarantee a relative-like error
    bound, e.g. $\epsilon_{ratio} = 1.02$ corresponds to a $2\%$ relative-like
    error bound.

    [^1]: Gustafson, J. L., & Yonemoto, I. T. (2017). Beating floating-point at
        its Own Game: Posit Arithmetic. *Supercomputing Frontiers and
        Innovations*, 4(2). Available from:
        [doi:10.14529/jsfi170206](https://doi.org/10.14529/jsfi170206).

    [^2]: Klöwer, M., Düben, P. D., & Palmer, T. N. (2019). Posits as an
        alternative to floats for weather and climate models. *CoNGA'19:
        Proceedings of the Conference for Next Generation Arithmetic 2019*, 1-8.
        Available from:
        [doi:10.1145/3316279.3316281](https://doi.org/10.1145/3316279.3316281).
    """


def _check_error_bound(
    type: ErrorBound,
    eb: int | float | np.ndarray[tuple[()] | S, np.dtype[F]],
) -> None | Never:
    """
    Check if the error bound value `eb` is valid for the error bound `type`.

    Parameters
    ----------
    type : ErrorBound
        The error bound type.
    eb : int | float | np.ndarray[tuple[()] | S, np.dtype[F]]
        The error bound value.

    Raises
    ------
    ValueError
        if the error bound `value` is invalid.
    """

    match type:
        case ErrorBound.abs:
            if not np.all(eb >= 0):
                raise (
                    ValueError("must be non-negative for an absolute error bound") | ctx
                )
        case ErrorBound.rel:
            if not np.all(eb >= 0):
                raise (
                    ValueError("must be non-negative for a relative error bound") | ctx
                )
        case ErrorBound.ratio:
            if not np.all(eb >= 1):
                raise ValueError("must be >= 1 for a ratio error bound") | ctx
        case _:
            assert_never(type)

    if (not isinstance(eb, int)) and (not np.all(np.isfinite(eb))):
        raise ValueError("must be finite") | ctx

    return None


def _compute_finite_absolute_error_bound(
    type: ErrorBound,
    eb: np.ndarray[tuple[()] | S, np.dtype[F]],
    data_float: np.ndarray[S, np.dtype[F]],
) -> np.ndarray[tuple[()] | S, np.dtype[F]]:
    """
    Compute the actual absolute error bound value, against which the error is
    compared against, from the error bound `eb` and the data.

    The computation is only defined for finite data elements, i.e. the produced
    error bound should not be used for non-finite data elements.

    Parameters
    ----------
    type : ErrorBound
        The error bound type.
    eb : np.ndarray[tuple[()] | S, np.dtype[F]]
        The error bound value. It must have already been checked using
        `_check_error_bound`.
    data_float : np.ndarray[S, np.dtype[F]]
        The original data array in floating-point representation.

    Returns
    -------
    eb_abs : np.ndarray[tuple[()] | S, np.dtype[F]]
        The absolute error bound value.

        The error bound is scalar if `eb` is scalar and the error bound does
        not depend on the data, otherwise it has the same shape as the data.
    """

    match type:
        case ErrorBound.abs:
            return eb
        case ErrorBound.rel:
            with np.errstate(
                divide="ignore", over="ignore", under="ignore", invalid="ignore"
            ):
                eb_rel_as_abs = np.nan_to_num(np.abs(data_float) * eb)
            assert np.all((eb_rel_as_abs >= 0) & np.isfinite(eb_rel_as_abs))
            return eb_rel_as_abs
        case ErrorBound.ratio:
            return eb
        case _:
            assert_never(type)


@np.errstate(divide="ignore", over="ignore", under="ignore", invalid="ignore")
def _compute_finite_absolute_error(
    type: ErrorBound,
    data_float: np.ndarray[S, np.dtype[F]],
    prediction_float: np.ndarray[S, np.dtype[F]],
) -> np.ndarray[S, np.dtype[F]]:
    """
    Compute the absolute error value, which is compared against the absolute
    error bound, from the data and prediction arrays.

    The computation is only defined for finite data elements, i.e. the produced
    error should not be used for non-finite data or prediction elements.

    Parameters
    ----------
    type : ErrorBound
        The error bound type.
    data_float : np.ndarray[S, np.dtype[F]]
        The original data array in floating-point representation.
    prediction_float : np.ndarray[S, np.dtype[F]]
        The prediction data array in floating-point representation.

    Returns
    -------
    err_abs : np.ndarray[S, np.dtype[F]]
        The absolute error value between the original and prediction data.
    """

    match type:
        # for the relative error bound, we also just return the absolute error
        #  here since the error bound will include the data value that together
        #  form a relative error bound
        case ErrorBound.abs | ErrorBound.rel:
            return np.abs(np.subtract(data_float, prediction_float))
        case ErrorBound.ratio:
            err_abs: np.ndarray[S, np.dtype[F]] = _ensure_array(
                np.divide(prediction_float, data_float)
            )
            np.divide(
                data_float,
                prediction_float,
                out=err_abs,
                where=(np.abs(data_float) > np.abs(prediction_float)),
            )
            err_abs[(data_float == 0) & (prediction_float == 0)] = 0
            err_abs[np.sign(data_float) != np.sign(prediction_float)] = np.inf
            return err_abs
        case _:
            assert_never(type)


def _apply_finite_error_bound(
    type: ErrorBound,
    eb: np.ndarray[tuple[()] | S, np.dtype[F]],
    data: np.ndarray[S, np.dtype[T]],
    data_float: np.ndarray[S, np.dtype[F]],
) -> tuple[np.ndarray[S, np.dtype[T]], np.ndarray[S, np.dtype[T]]]:
    """
    Apply the error bound `eb` to the `data` to produce the inclusive lower and
    upper bounds within which the error bound is satisfied.

    This function calls `_compute_finite_absolute_error_bound` internally.

    The computation is only defined for finite data elements, i.e. the produced
    bounds should not be used for non-finite data elements.

    Parameters
    ----------
    type : ErrorBound
        The error bound type.
    eb : np.ndarray[tuple[()] | S, np.dtype[F]]
        The error bound value. It must have already been checked using
        `_check_error_bound`.
    data : np.ndarray[S, np.dtype[T]]
        The original data array.
    data_float : np.ndarray[S, np.dtype[F]]
        The original data array in floating-point representation.

    Returns
    -------
    lower, upper : tuple[np.ndarray[S, np.dtype[T]], np.ndarray[S, np.dtype[T]]]
        The lower and upper bounds within which the error bound is satisfied.
    """

    eb_abs = _compute_finite_absolute_error_bound(type, eb, data_float)

    match type:
        case ErrorBound.abs | ErrorBound.rel:
            with np.errstate(
                divide="ignore", over="ignore", under="ignore", invalid="ignore"
            ):
                lower: np.ndarray[S, np.dtype[T]] = from_float(
                    data_float - eb_abs, data.dtype
                )
                upper: np.ndarray[S, np.dtype[T]] = from_float(
                    data_float + eb_abs, data.dtype
                )

            # correct rounding errors in the lower and upper bound
            with np.errstate(
                divide="ignore", over="ignore", under="ignore", invalid="ignore"
            ):
                lower_outside_eb: np.ndarray[S, np.dtype[np.bool]] = np.greater(
                    data_float - to_float(lower, ftype=data_float.dtype), eb_abs
                )
                upper_outside_eb: np.ndarray[S, np.dtype[np.bool]] = np.greater(
                    to_float(upper, ftype=data_float.dtype) - data_float, eb_abs
                )

            lower = _ensure_array(
                from_total_order(
                    np.add(to_total_order(lower), lower_outside_eb),  # type: ignore
                    data.dtype,
                )
            )
            upper = _ensure_array(
                from_total_order(
                    np.subtract(to_total_order(upper), upper_outside_eb),  # type: ignore
                    data.dtype,
                )
            )

            # a zero-error bound must preserve exactly, e.g. even for -0.0
            if np.any(eb == 0):
                np.copyto(lower, data, where=(eb == 0), casting="no")
                np.copyto(upper, data, where=(eb == 0), casting="no")
        case ErrorBound.ratio:
            with np.errstate(
                divide="ignore", over="ignore", under="ignore", invalid="ignore"
            ):
                data_mul, data_div = (
                    from_float(data_float * eb_abs, data.dtype),
                    from_float(data_float / eb_abs, data.dtype),
                )
            lower = _where(np.less(data, 0), data_mul, data_div)
            upper = _where(np.less(data, 0), data_div, data_mul)

            # correct rounding errors in the lower and upper bound
            with np.errstate(
                divide="ignore", over="ignore", under="ignore", invalid="ignore"
            ):
                lower_outside_eb = (
                    np.abs(
                        _where(
                            np.less(data, 0),
                            to_float(lower, ftype=data_float.dtype) / data_float,
                            data_float / to_float(lower, ftype=data_float.dtype),
                        )
                    )
                    > eb_abs
                )
                upper_outside_eb = (
                    np.abs(
                        _where(
                            np.less(data, 0),
                            data_float / to_float(upper, ftype=data_float.dtype),
                            to_float(upper, ftype=data_float.dtype) / data_float,
                        )
                    )
                    > eb_abs
                )

            lower = _ensure_array(
                from_total_order(
                    np.add(to_total_order(lower), lower_outside_eb),  # type: ignore
                    data.dtype,
                )
            )
            upper = _ensure_array(
                from_total_order(
                    np.subtract(to_total_order(upper), upper_outside_eb),  # type: ignore
                    data.dtype,
                )
            )

            # a ratio of 1 bound must preserve exactly, e.g. even for -0.0
            if np.any(eb == 1):
                np.copyto(lower, data, where=(eb == 1), casting="no")
                np.copyto(upper, data, where=(eb == 1), casting="no")
        case _:
            assert_never(type)

    if type in (ErrorBound.rel, ErrorBound.ratio):
        # special case zero to handle +0.0 and -0.0
        np.copyto(lower, data, where=(data == 0), casting="no")
        np.copyto(upper, data, where=(data == 0), casting="no")

    return (lower, upper)


def _apply_finite_qoi_error_bound(
    type: ErrorBound,
    eb: np.ndarray[tuple[()] | S, np.dtype[F]],
    qoi_float: np.ndarray[S, np.dtype[F]],
) -> tuple[np.ndarray[S, np.dtype[F]], np.ndarray[S, np.dtype[F]]]:
    """
    Apply the error bound `eb` to the QoI array to produce the inclusive lower
    and upper bounds within which the error bound is satisfied.

    This function calls `_compute_finite_absolute_error_bound` internally.

    Unlike `_apply_finite_error_bound`, which is applied in the original data
    space, this function is applied in floating-point QoI space and uses a
    different approach for correcting rounding errors with nudging.

    The computation is only defined for finite data elements, i.e. the produced
    bounds should not be used for non-finite QoI elements.

    Parameters
    ----------
    type : ErrorBound
        The error bound type.
    eb : np.ndarray[tuple[()] | S, np.dtype[F]]
        The error bound value. It must have already been checked using
        `_check_error_bound`.
    qoi_float : np.ndarray[S, np.dtype[F]]
        The floating-point QoI array.

    Returns
    -------
    lower, upper : tuple[np.ndarray[S, np.dtype[F]], np.ndarray[S, np.dtype[F]]]
        The lower and upper bounds within which the error bound is satisfied.
    """

    eb_abs = _compute_finite_absolute_error_bound(type, eb, qoi_float)

    match type:
        case ErrorBound.abs | ErrorBound.rel:
            with np.errstate(
                divide="ignore", over="ignore", under="ignore", invalid="ignore"
            ):
                lower: np.ndarray[S, np.dtype[F]] = _ensure_array(
                    np.subtract(qoi_float, eb_abs)
                )
                upper: np.ndarray[S, np.dtype[F]] = _ensure_array(
                    np.add(qoi_float, eb_abs)
                )

            # optimistically allow both -0.0 and +0.0
            lower[lower == 0] = -0.0
            upper[upper == 0] = +0.0

            # correct rounding errors in the lower and upper bound
            with np.errstate(
                divide="ignore", over="ignore", under="ignore", invalid="ignore"
            ):
                lower_outside_eb: np.ndarray[S, np.dtype[np.bool]] = np.greater(
                    qoi_float - lower, eb_abs
                )
                upper_outside_eb: np.ndarray[S, np.dtype[np.bool]] = np.greater(
                    upper - qoi_float, eb_abs
                )

            # we can nudge with nextafter since the QoIs are floating-point
            np.nextafter(
                lower,
                qoi_float,
                out=lower,
                where=(lower_outside_eb & np.isfinite(qoi_float)),
            )
            np.nextafter(
                upper,
                qoi_float,
                out=upper,
                where=(upper_outside_eb & np.isfinite(qoi_float)),
            )

            # a zero-error bound must preserve exactly, e.g. even for -0.0
            if np.any(eb == 0):
                np.copyto(lower, qoi_float, where=(eb == 0), casting="no")
                np.copyto(upper, qoi_float, where=(eb == 0), casting="no")
        case ErrorBound.ratio:
            with np.errstate(
                divide="ignore", over="ignore", under="ignore", invalid="ignore"
            ):
                data_mul, data_div = (
                    qoi_float * eb_abs,
                    qoi_float / eb_abs,
                )
            lower = _ensure_array(_where(np.less(qoi_float, 0), data_mul, data_div))
            upper = _ensure_array(_where(np.less(qoi_float, 0), data_div, data_mul))

            # correct rounding errors in the lower and upper bound
            with np.errstate(
                divide="ignore", over="ignore", under="ignore", invalid="ignore"
            ):
                lower_outside_eb = (
                    np.abs(
                        _where(
                            qoi_float < 0,
                            lower / qoi_float,
                            qoi_float / lower,
                        )
                    )
                    > eb_abs
                )
                upper_outside_eb = (
                    np.abs(
                        _where(
                            qoi_float < 0,
                            qoi_float / upper,
                            upper / qoi_float,
                        )
                    )
                    > eb_abs
                )

            # we can nudge with nextafter since the QoIs are floating-point
            np.nextafter(
                lower,
                qoi_float,
                out=lower,
                where=(lower_outside_eb & np.isfinite(qoi_float)),
            )
            np.nextafter(
                upper,
                qoi_float,
                out=upper,
                where=(upper_outside_eb & np.isfinite(qoi_float)),
            )

            # a ratio of 1 bound must preserve exactly, e.g. even for -0.0
            if np.any(eb == 1):
                np.copyto(lower, qoi_float, where=(eb == 1), casting="no")
                np.copyto(upper, qoi_float, where=(eb == 1), casting="no")
        case _:
            assert_never(type)

    if type in (ErrorBound.rel, ErrorBound.ratio):
        # special case zero to handle +0.0 and -0.0
        np.copyto(lower, qoi_float, where=(qoi_float == 0), casting="no")
        np.copyto(upper, qoi_float, where=(qoi_float == 0), casting="no")

    return (lower, upper)
