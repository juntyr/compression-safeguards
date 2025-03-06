"""
Decimal error bound safeguard.
"""

__all__ = ["DecimalErrorBoundSafeguard"]

import numpy as np

from . import ElementwiseSafeguard, _as_bits
from ...intervals import (
    IntervalUnion,
    Interval,
    Lower,
    Upper,
    _to_total_order,
    _from_total_order,
)
from ...cast import to_float, from_float


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

    In cases where the arithmetic evaluation of the error bound not well-
    defined, e.g. for infinite or NaN values, producing the exact same
    bitpattern is defined to satisfy the error bound. If `equal_nan` is set to
    [`True`][True], decoding a NaN value to a NaN value with a different
    bitpattern also satisfies the error bound.

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
    def check(self, data: np.ndarray, decoded: np.ndarray) -> bool:
        """
        Check if the `decoded` array satisfies the decimal error bound.

        Parameters
        ----------
        data : np.ndarray
            Data to be encoded.
        decoded : np.ndarray
            Decoded data.

        Returns
        -------
        ok : bool
            `True` if the check succeeded.
        """

        decimal_bound = self._decimal_error(data, decoded) <= self._eb_decimal

        # bitwise equality for inf and NaNs (unless equal_nan)
        same_bits = _as_bits(data) == _as_bits(decoded)
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

        return bool(np.all(ok))

    def compute_safe_intervals(self, data: np.ndarray) -> IntervalUnion:
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

        data = data.flatten()
        data_float = to_float(data)

        valid = (
            Interval.empty_like(data)
            .preserve_inf(data)
            .preserve_nan(data, equal_nan=self._equal_nan)
        )

        with np.errstate(
            divide="ignore", over="ignore", under="ignore", invalid="ignore"
        ):
            eb_decimal_multipler = min(
                np.power(np.array(10, dtype=data_float.dtype), self._eb_decimal).astype(
                    data_float.dtype
                ),
                np.finfo(data_float.dtype).max,
            )
        assert eb_decimal_multipler >= 1.0 and np.isfinite(eb_decimal_multipler)

        with np.errstate(
            divide="ignore", over="ignore", under="ignore", invalid="ignore"
        ):
            data_mul, data_div = (
                from_float(data_float * eb_decimal_multipler, data.dtype),
                from_float(data_float / eb_decimal_multipler, data.dtype),
            )
            Lower(np.where(data < 0, data_mul, data_div)) <= valid[
                np.isfinite(data)
            ] <= Upper(np.where(data < 0, data_div, data_mul))

        # correct rounding errors in the lower and upper bound
        with np.errstate(
            divide="ignore", over="ignore", under="ignore", invalid="ignore"
        ):
            lower_bound_outside_eb_decimal = (
                np.abs(
                    np.where(
                        data < 0,
                        to_float(valid._lower) / data_float,
                        data_float / to_float(valid._lower),
                    )
                )
                > eb_decimal_multipler
            )
            upper_bound_outside_eb_decimal = (
                np.abs(
                    np.where(
                        data < 0,
                        data_float / to_float(valid._upper),
                        to_float(valid._upper) / data_float,
                    )
                )
                > eb_decimal_multipler
            )

        valid._lower[np.isfinite(data)] = _from_total_order(
            _to_total_order(valid._lower) + lower_bound_outside_eb_decimal,
            data.dtype,
        )[np.isfinite(data)]
        valid._upper[np.isfinite(data)] = _from_total_order(
            _to_total_order(valid._upper) - upper_bound_outside_eb_decimal,
            data.dtype,
        )[np.isfinite(data)]

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
            kind=type(self).kind, eb_decimal=self._eb_decimal, equal_nan=self._equal_nan
        )

    @np.errstate(divide="ignore", over="ignore", under="ignore", invalid="ignore")
    def _decimal_error(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        sign_x, sign_y = np.sign(x), np.sign(y)

        # 0               : if x == 0 and y == 0
        # inf             : if sign(x) != sign(y)
        # abs(log10(x/y)) : otherwise
        return np.where(
            (sign_x == 0) & (sign_y == 0),
            0.0,
            np.where(
                sign_x != sign_y, np.inf, (np.abs(np.log10(to_float(x) / to_float(y))))
            ),
        )
