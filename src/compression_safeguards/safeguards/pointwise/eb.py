"""
Error bound safeguard.
"""

__all__ = ["ErrorBound", "ErrorBoundSafeguard"]

from enum import Enum, auto

import numpy as np
from typing_extensions import assert_never  # MSPV 3.11

from ...utils.bindings import Bindings
from ...utils.cast import (
    _isfinite,
    _isinf,
    _isnan,
    _nan_to_zero,
    _sign,
    as_bits,
    from_float,
    from_total_order,
    to_finite_float,
    to_float,
    to_total_order,
)
from ...utils.intervals import Interval, IntervalUnion, Lower, Upper
from ...utils.typing import F, S, T
from .abc import PointwiseSafeguard


class ErrorBound(Enum):
    """
    Different types of error bounds that can be guaranteed by the [`ErrorBoundSafeguard`][compression_safeguards.safeguards.pointwise.eb.ErrorBoundSafeguard].
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
    and the decoded values as well as their inverse ratios are less than or
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
    guarantees that the sign of each decoded value matches the sign of each
    original value and that a decoded value is zero if and only if it is zero
    in the original data.

    The ratio error bound is sometimes also known as a decimal error bound[^1]
    [^2] if the ratio is expressed as the difference in orders of magnitude. A
    decimal error bound of e.g. $2$ (two orders of magnitude difference / x100
    ratio) can be expressed using
    $\epsilon_{ratio} = {10}^{\epsilon_{decimal}}$.

    The ratio error bound can also be used to guarantee a relative-like error
    bound, e.g. $\epsilon_{ratio} = 1.02$ corresponds to a $2\%$ relative-like
    error bound.

    [^1]: Gustafson, J. L., & Yonemoto, I. T. (2017). Beating Floating Point at
        its Own Game: Posit Arithmetic. *Supercomputing Frontiers and
        Innovations*, 4(2). Available from:
        [doi:10.14529/jsfi170206](https://doi.org/10.14529/jsfi170206).

    [^2]: Klöwer, M., Düben, P. D., & Palmer, T. N. (2019). Posits as an
        alternative to floats for weather and climate models. *CoNGA'19:
        Proceedings of the Conference for Next Generation Arithmetic 2019*, 1-8.
        Available from:
        [doi:10.1145/3316279.3316281](https://doi.org/10.1145/3316279.3316281).
    """


class ErrorBoundSafeguard(PointwiseSafeguard):
    """
    The `ErrorBoundSafeguard` guarantees that the pointwise error `type` is less than or equal to the provided bound `eb`.

    Infinite values are preserved with the same bit pattern. If `equal_nan` is
    set to [`True`][True], decoding a NaN value to a NaN value with a different
    bit pattern also satisfies the error bound. If `equal_nan` is set to
    [`False`][False], NaN values are also preserved with the same bit pattern.

    Parameters
    ----------
    type : str | ErrorBound
        The type of error bound that is enforced by this safeguard.
    eb : int | float
        The value of the error bound that is enforced by this safeguard.
    equal_nan: bool
        Whether decoding a NaN value to a NaN value with a different bit
        pattern satisfies the error bound.
    """

    __slots__ = ("_type", "_eb", "_equal_nan")
    _type: ErrorBound
    _eb: int | float
    _equal_nan: bool

    kind = "eb"

    def __init__(
        self, type: str | ErrorBound, eb: int | float, *, equal_nan: bool = False
    ):
        self._type = type if isinstance(type, ErrorBound) else ErrorBound[type]

        match self._type:
            case ErrorBound.abs:
                assert eb >= 0, "eb must be non-negative for an absolute error bound"
            case ErrorBound.rel:
                assert eb >= 0, "eb must be non-negative for a relative error bound"
            case ErrorBound.ratio:
                assert eb >= 1, "eb must be >= 1 for a ratio error bound"
            case _:
                assert_never(self._type)

        assert isinstance(eb, int) or _isfinite(eb), "eb must be finite"

        self._eb = eb
        self._equal_nan = equal_nan

    @np.errstate(divide="ignore", over="ignore", under="ignore", invalid="ignore")
    def check_pointwise(
        self,
        data: np.ndarray[S, np.dtype[T]],
        decoded: np.ndarray[S, np.dtype[T]],
        *,
        late_bound: Bindings,
    ) -> np.ndarray[S, np.dtype[np.bool]]:
        """
        Check which elements in the `decoded` array satisfy the error bound.

        Parameters
        ----------
        data : np.ndarray
            Data to be encoded.
        decoded : np.ndarray
            Decoded data.
        late_bound : Bindings
            Bindings for late-bound parameters, including for this safeguard.

        Returns
        -------
        ok : np.ndarray
            Pointwise, `True` if the check succeeded for this element.
        """

        data_float: np.ndarray = to_float(data)
        decoded_float: np.ndarray = to_float(decoded)

        finite_ok = _compute_finite_absolute_error(
            self._type, data_float, decoded_float
        ) <= _compute_finite_absolute_error_bound(self._type, self._eb, data_float)

        # bitwise equality for inf and NaNs (unless equal_nan)
        same_bits = as_bits(data) == as_bits(decoded)
        both_nan = self._equal_nan and (_isnan(data) & _isnan(decoded))

        ok = np.where(
            _isfinite(data),
            finite_ok,
            np.where(
                _isinf(data),
                same_bits,
                both_nan if self._equal_nan else same_bits,
            ),
        )

        return ok  # type: ignore

    def compute_safe_intervals(
        self,
        data: np.ndarray[S, np.dtype[T]],
        *,
        late_bound: Bindings,
    ) -> IntervalUnion[T, int, int]:
        """
        Compute the intervals in which the error bound is upheld with respect
        to the `data`.

        Parameters
        ----------
        data : np.ndarray
            Data for which the safe intervals should be computed.
        late_bound : Bindings
            Bindings for late-bound parameters, including for this safeguard.

        Returns
        -------
        intervals : IntervalUnion
            Union of intervals in which the error bound is upheld.
        """

        dataf = data.flatten()

        valid = (
            Interval.empty_like(dataf)
            .preserve_inf(dataf)
            .preserve_nan(dataf, equal_nan=self._equal_nan)
        )

        lower, upper = _apply_finite_error_bound(
            self._type, self._eb, data, to_float(data)
        )

        Lower(lower) <= valid[_isfinite(dataf)] <= Upper(upper)

        return valid.into_union()

    def get_config(self) -> dict:
        """
        Returns the configuration of the safeguard.

        Returns
        -------
        config : dict
            Configuration of the safeguard.
        """

        return dict(
            kind=type(self).kind,
            type=self._type.name,
            eb=self._eb,
            equal_nan=self._equal_nan,
        )


def _compute_finite_absolute_error_bound(
    type: ErrorBound,
    eb: int | float,
    data_float: np.ndarray[S, np.dtype[F]],
) -> np.ndarray[tuple[()] | S, np.dtype[F]]:
    match type:
        case ErrorBound.abs:
            with np.errstate(
                divide="ignore", over="ignore", under="ignore", invalid="ignore"
            ):
                eb_abs: np.ndarray[tuple[()], np.dtype[F]] = to_finite_float(
                    eb, data_float.dtype
                )
            assert eb_abs >= 0

            return eb_abs
        case ErrorBound.rel:
            with np.errstate(
                divide="ignore", over="ignore", under="ignore", invalid="ignore"
            ):
                eb_rel_as_abs: np.ndarray[S, np.dtype[F]] = to_finite_float(
                    eb,
                    data_float.dtype,
                    map=lambda eb_rel: np.abs(data_float) * eb_rel,
                )
                eb_rel_as_abs = _nan_to_zero(eb_rel_as_abs)
            assert np.all((eb_rel_as_abs >= 0) & _isfinite(eb_rel_as_abs))

            return eb_rel_as_abs
        case ErrorBound.ratio:
            with np.errstate(
                divide="ignore", over="ignore", under="ignore", invalid="ignore"
            ):
                eb_ratio: np.ndarray[tuple[()], np.dtype[F]] = to_finite_float(
                    eb, data_float.dtype
                )
            assert eb_ratio >= 1.0

            return eb_ratio
        case _:
            assert_never(type)


def _compute_finite_absolute_error(
    type: ErrorBound,
    data_float: np.ndarray[S, np.dtype[F]],
    decoded_float: np.ndarray[S, np.dtype[F]],
) -> np.ndarray[S, np.dtype[F]]:
    match type:
        case ErrorBound.abs | ErrorBound.rel:
            return np.abs(data_float - decoded_float)  # type: ignore
        case ErrorBound.ratio:
            return np.where(  # type: ignore
                (data_float == 0) & (decoded_float == 0),
                np.array(0, dtype=data_float.dtype),
                np.where(
                    _sign(data_float) != _sign(decoded_float),
                    np.array(np.inf, dtype=data_float.dtype),
                    np.where(
                        np.abs(data_float) > np.abs(decoded_float),
                        data_float / decoded_float,
                        decoded_float / data_float,
                    ),
                ),
            )
        case _:
            assert_never(type)


def _apply_finite_error_bound(
    type: ErrorBound,
    eb: int | float,
    data: np.ndarray[S, np.dtype[T]],
    data_float: np.ndarray[S, np.dtype[F]],
) -> tuple[np.ndarray[S, np.dtype[T]], np.ndarray[S, np.dtype[T]]]:
    eb_float = _compute_finite_absolute_error_bound(type, eb, data_float)

    match type:
        case ErrorBound.abs | ErrorBound.rel:
            with np.errstate(
                divide="ignore", over="ignore", under="ignore", invalid="ignore"
            ):
                lower = from_float(data_float - eb_float, data.dtype)
                upper = from_float(data_float + eb_float, data.dtype)

            # correct rounding errors in the lower and upper bound
            with np.errstate(
                divide="ignore", over="ignore", under="ignore", invalid="ignore"
            ):
                lower_outside_eb = (data_float - to_float(lower)) > eb_float
                upper_outside_eb = (to_float(upper) - data_float) > eb_float

            lower = from_total_order(
                to_total_order(lower) + lower_outside_eb,
                data.dtype,
            )
            upper = from_total_order(
                to_total_order(upper) - upper_outside_eb,
                data.dtype,
            )

            # a zero-error bound must preserve exactly, e.g. even for -0.0
            if np.any(eb == 0):
                lower = np.where(eb == 0, data, lower)
                upper = np.where(eb == 0, data, upper)
        case ErrorBound.ratio:
            with np.errstate(
                divide="ignore", over="ignore", under="ignore", invalid="ignore"
            ):
                data_mul, data_div = (
                    from_float(data_float * eb_float, data.dtype),
                    from_float(data_float / eb_float, data.dtype),
                )
            lower = np.where(data < 0, data_mul, data_div)
            upper = np.where(data < 0, data_div, data_mul)

            # correct rounding errors in the lower and upper bound
            with np.errstate(
                divide="ignore", over="ignore", under="ignore", invalid="ignore"
            ):
                lower_outside_eb = (
                    np.abs(
                        np.where(
                            data < 0,
                            to_float(lower) / data_float,
                            data_float / to_float(lower),
                        )
                    )
                    > eb_float
                )
                upper_outside_eb = (
                    np.abs(
                        np.where(
                            data < 0,
                            data_float / to_float(upper),
                            to_float(upper) / data_float,
                        )
                    )
                    > eb_float
                )

            lower = from_total_order(
                to_total_order(lower) + lower_outside_eb,
                data.dtype,
            )
            upper = from_total_order(
                to_total_order(upper) - upper_outside_eb,
                data.dtype,
            )

            # a ratio of 1 bound must preserve exactly, e.g. even for -0.0
            if np.any(eb == 1):
                lower = np.where(eb == 1, data, lower)
                upper = np.where(eb == 1, data, upper)
        case _:
            assert_never(type)

    if type in (ErrorBound.rel, ErrorBound.ratio):
        # special case zero to handle +0.0 and -0.0
        lower = np.where(data == 0, data, lower)
        upper = np.where(data == 0, data, upper)

    return (lower, upper)


def _compute_safe_eb_diff_interval(
    data: np.ndarray[S, np.dtype[T]],
    data_float: np.ndarray[S, np.dtype[F]],
    eb_lower: np.ndarray[S | tuple[()], np.dtype[F]],
    eb_upper: np.ndarray[S | tuple[()], np.dtype[F]],
    equal_nan: bool,
) -> Interval[T, int]:
    dataf: np.ndarray[tuple[int], np.dtype[T]] = data.flatten()
    dataf_float: np.ndarray[tuple[int], np.dtype[F]] = data_float.flatten()
    eb_lowerf: np.ndarray[tuple[int], np.dtype[F]] = np.array(eb_lower).flatten()
    eb_upperf: np.ndarray[tuple[int], np.dtype[F]] = np.array(eb_upper).flatten()

    assert np.all(_isfinite(eb_lowerf) & (eb_lowerf <= 0))
    assert np.all(_isfinite(eb_upperf) & (eb_upperf >= 0))

    valid = (
        Interval.empty_like(dataf)
        .preserve_inf(dataf)
        .preserve_nan(dataf, equal_nan=equal_nan)
    )

    with np.errstate(divide="ignore", over="ignore", under="ignore", invalid="ignore"):
        Lower(from_float(dataf_float + eb_lowerf, dataf.dtype)) <= valid[
            _isfinite(dataf)
        ] <= Upper(from_float(dataf_float + eb_upperf, dataf.dtype))

    # correct rounding errors in the lower and upper bound
    with np.errstate(divide="ignore", over="ignore", under="ignore", invalid="ignore"):
        # we don't use abs(data - bound) here to accommodate unsigned ints
        lower_bound_outside_eb_abs = (to_float(valid._lower) - dataf_float) < eb_lowerf
        upper_bound_outside_eb_abs = (to_float(valid._upper) - dataf_float) > eb_upperf

    valid._lower[_isfinite(dataf)] = from_total_order(
        to_total_order(valid._lower) + lower_bound_outside_eb_abs,
        dataf.dtype,
    )[_isfinite(dataf)]
    valid._upper[_isfinite(dataf)] = from_total_order(
        to_total_order(valid._upper) - upper_bound_outside_eb_abs,
        dataf.dtype,
    )[_isfinite(dataf)]

    # a zero-error bound must preserve exactly, e.g. even for -0.0
    if np.any(eb_lowerf == 0):
        Lower(dataf) <= valid[_isfinite(dataf) & (eb_lowerf == 0)]
    if np.any(eb_upperf == 0):
        valid[_isfinite(dataf) & (eb_upperf == 0)] <= Upper(dataf)

    return valid
