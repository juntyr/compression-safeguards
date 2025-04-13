"""
Decimal error bound safeguard.
"""

__all__ = ["DecimalErrorBoundSafeguard"]

import numpy as np

from .abc import ElementwiseSafeguard, S, T
from ...cast import (
    to_float,
    from_float,
    as_bits,
    to_total_order,
    from_total_order,
    to_finite_float,
    F,
)
from ...intervals import IntervalUnion, Interval, Lower, Upper


class DecimalErrorBoundSafeguard(ElementwiseSafeguard):
    r"""
    The `DecimalErrorBoundSafeguard` guarantees that the elementwise decimal
    error is less than or equal to the provided bound `eb_decimal`.

    The decimal error quantifies the orders of magnitude that the lossy-decoded
    value $\hat{x}$ is away from the original value $x$. It is defined as
    follows[^1] [^2]:

    \[
        \text{decimal error} = \begin{cases}
            0 & \quad \text{if } x = \hat{x} = 0 \\
            \inf & \quad \text{if } \text{sign}(x) \neq \text{sign}(\hat{x}) \\
            \left| \log_{10}{\left( \frac{x}{\hat{x}} \right)} \right| & \quad \text{otherwise}
        \end{cases}
    \]

    The decimal error is defined to be infinite if the signs of the data and
    decoded data do not match. Since the `eb_decimal` error bound must be
    finite, the `DecimalErrorBoundSafeguard` also guarantees that the sign of
    each decode value matches the sign of each original value and that a
    decoded value is zero if and only if it is zero in the original data.

    Infinite values are preserved with the same bit pattern. If `equal_nan` is
    set to [`True`][True], decoding a NaN value to a NaN value with a different
    bitpattern also satisfies the error bound. If `equal_nan` is set to
    [`False`][False], NaN values are also preserved with the same bit pattern.

    [^1]: Gustafson, J. L., & Yonemoto, I. T. (2017). Beating Floating Point at
        its Own Game: Posit Arithmetic. *Supercomputing Frontiers and
        Innovations*, 4(2). Available from:
        [doi:10.14529/jsfi170206](https://doi.org/10.14529/jsfi170206).

    [^2]: Klöwer, M., Düben, P. D., & Palmer, T. N. (2019). Posits as an
        alternative to floats for weather and climate models. *CoNGA'19:
        Proceedings of the Conference for Next Generation Arithmetic 2019*, 1-8.
        Available from:
        [doi:10.1145/3316279.3316281](https://doi.org/10.1145/3316279.3316281).

    Parameters
    ----------
    eb_decimal : int | float
        The non-negative decimal error bound that is enforced by this safeguard.
        `eb_decimal=1.0` corresponds to a 10x relative error bound.
    equal_nan: bool
        Whether decoding a NaN value to a NaN value with a different bit
        pattern satisfies the error bound.
    """

    __slots__ = ("_eb_decimal", "_equal_nan")
    _eb_decimal: int | float
    _equal_nan: bool

    kind = "decimal"

    def __init__(self, eb_decimal: int | float, *, equal_nan: bool = False):
        assert eb_decimal >= 0, "eb_decimal must be non-negative"
        assert np.isfinite(eb_decimal), "eb_decimal must be finite"

        self._eb_decimal = eb_decimal
        self._equal_nan = equal_nan

    @np.errstate(divide="ignore", over="ignore", under="ignore", invalid="ignore")
    def check_elementwise(
        self, data: np.ndarray[S, T], decoded: np.ndarray[S, T]
    ) -> np.ndarray[S, np.dtype[np.bool]]:
        """
        Check which elements in the `decoded` array satisfy the decimal error
        bound.

        Parameters
        ----------
        data : np.ndarray
            Data to be encoded.
        decoded : np.ndarray
            Decoded data.

        Returns
        -------
        ok : np.ndarray
            Per-element, `True` if the check succeeded for this element.
        """

        decimal_bound = self._decimal_error(data, decoded) <= self._eb_decimal

        # bitwise equality for inf and NaNs (unless equal_nan)
        same_bits = as_bits(data) == as_bits(decoded)
        both_nan = self._equal_nan and (np.isnan(data) & np.isnan(decoded))

        ok = np.where(
            np.isfinite(data),
            decimal_bound,
            np.where(
                np.isinf(data),
                same_bits,
                both_nan if self._equal_nan else same_bits,
            ),
        )

        return ok  # type: ignore

    def compute_safe_intervals(
        self, data: np.ndarray[S, T]
    ) -> IntervalUnion[T, int, int]:
        """
        Compute the intervals in which the decimal error bound is upheld with
        respect to the `data`.

        Parameters
        ----------
        data : np.ndarray
            Data for which the safe intervals should be computed.

        Returns
        -------
        intervals : IntervalUnion
            Union of intervals in which the decimal error bound is upheld.
        """

        data_float: np.ndarray = to_float(data)

        with np.errstate(
            divide="ignore", over="ignore", under="ignore", invalid="ignore"
        ):
            eb_ratio = to_finite_float(
                10, data_float.dtype, map=lambda x: np.power(x, self._eb_decimal)
            )
        assert eb_ratio >= 1.0

        return _compute_safe_eb_ratio_interval(
            data, data_float, eb_ratio, equal_nan=self._equal_nan
        ).into_union()  # type: ignore

    def get_config(self) -> dict:
        """
        Returns the configuration of the safeguard.

        Returns
        -------
        config : dict
            Configuration of the safeguard.
        """

        return dict(
            kind=type(self).kind, eb_decimal=self._eb_decimal, equal_nan=self._equal_nan
        )

    @np.errstate(divide="ignore", over="ignore", under="ignore", invalid="ignore")
    def _decimal_error(
        self, x: np.ndarray[S, T], y: np.ndarray[S, T]
    ) -> np.ndarray[S, F]:
        sign_x, sign_y = np.sign(x), np.sign(y)

        # 0               : if x == 0 and y == 0
        # inf             : if sign(x) != sign(y)
        # abs(log10(x/y)) : otherwise
        return np.where(
            (sign_x == 0) & (sign_y == 0),
            to_float(np.array(0.0)),
            np.where(
                sign_x != sign_y,
                to_float(np.array(np.inf)),
                np.abs(np.log10(to_float(x) / to_float(y))),
            ),
        )  # type: ignore


def _compute_safe_eb_ratio_interval(
    data: np.ndarray[S, T],
    data_float: np.ndarray[S, F],
    eb_ratio: np.ndarray[tuple[()], F],
    equal_nan: bool,
) -> Interval[T, int]:
    dataf: np.ndarray[tuple[int], T] = data.flatten()
    dataf_float: np.ndarray[tuple[int], F] = data_float.flatten()

    valid = (
        Interval.empty_like(dataf)
        .preserve_inf(dataf)
        .preserve_nan(dataf, equal_nan=equal_nan)
    )

    with np.errstate(divide="ignore", over="ignore", under="ignore", invalid="ignore"):
        data_mul, data_div = (
            from_float(dataf_float * eb_ratio, dataf.dtype),
            from_float(dataf_float / eb_ratio, dataf.dtype),
        )
        Lower(np.where(dataf < 0, data_mul, data_div)) <= valid[
            np.isfinite(dataf)
        ] <= Upper(np.where(dataf < 0, data_div, data_mul))

    # correct rounding errors in the lower and upper bound
    with np.errstate(divide="ignore", over="ignore", under="ignore", invalid="ignore"):
        lower_bound_outside_eb_ratio = (
            np.abs(
                np.where(
                    dataf < 0,
                    to_float(valid._lower) / dataf_float,
                    dataf_float / to_float(valid._lower),
                )
            )
            > eb_ratio
        )
        upper_bound_outside_eb_ratio = (
            np.abs(
                np.where(
                    dataf < 0,
                    dataf_float / to_float(valid._upper),
                    to_float(valid._upper) / dataf_float,
                )
            )
            > eb_ratio
        )

    valid._lower[np.isfinite(dataf)] = from_total_order(
        to_total_order(valid._lower) + lower_bound_outside_eb_ratio,
        dataf.dtype,
    )[np.isfinite(dataf)]
    valid._upper[np.isfinite(dataf)] = from_total_order(
        to_total_order(valid._upper) - upper_bound_outside_eb_ratio,
        dataf.dtype,
    )[np.isfinite(dataf)]

    return valid
