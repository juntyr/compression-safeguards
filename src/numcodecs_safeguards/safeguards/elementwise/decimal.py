"""
Decimal error bound safeguard.
"""

__all__ = ["DecimalErrorBoundSafeguard"]

import numpy as np

from . import ElementwiseSafeguard
from ...intervals import (
    IntervalUnion,
    Interval,
    Lower,
    Upper,
    Minimum,
    Maximum,
    _to_total_order,
    _from_total_order,
)


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
        Proceedings of the Conference for Next Generation Arithmetic 2019*, 1–8.
        Available from:
        [doi:10.1145/3316279.3316281](https://doi.org/10.1145/3316279.3316281).

    Parameters
    ----------
    eb_decimal : int | float
        The positive decimal error bound that is enforced by this safeguard.
        `eb_decimal=1.0` corresponds to a 100% relative error bound.
    equal_nan: bool
        Whether decoding a NaN value to a NaN value with a different bit
        pattern satisfies the error bound.
    """

    __slots__ = ("_eb_decimal", "_equal_nan")
    _eb_decimal: int | float
    _equal_nan: bool

    kind = "decimal"
    _priority = 0

    def __init__(self, eb_decimal: int | float, *, equal_nan: bool = False):
        assert eb_decimal > 0, "eb_decimal must be positive"
        assert np.isfinite(eb_decimal), "eb_decimal must be finite"

        self._eb_decimal = eb_decimal
        self._equal_nan = equal_nan

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
        valid = Interval.empty_like(data)

        with np.errstate(
            divide="ignore", over="ignore", under="ignore", invalid="ignore"
        ):
            eb_decimal_multipler = np.minimum(
                np.power(10, self._eb_decimal),
                np.finfo(self._eb_decimal).max
                if np.issubdtype(np.array(self._eb_decimal).dtype, np.floating)
                else np.iinfo(self._eb_decimal).max,  # type: ignore
            )

        if np.issubdtype(data.dtype, np.floating):
            Lower(data) <= valid[np.isinf(data)] <= Upper(data)

        if np.issubdtype(data.dtype, np.floating):
            if self._equal_nan:
                nan_min = np.array(
                    np.array(np.inf, dtype=data.dtype).view(
                        data.dtype.str.replace("f", "u")
                    )
                    + 1
                ).view(data.dtype)
                nan_max = np.array(-1, dtype=data.dtype.str.replace("f", "i")).view(
                    data.dtype
                )

                # any NaN with the same sign is valid
                Lower(
                    np.where(
                        np.signbit(data),
                        np.copysign(nan_max, -1),
                        np.copysign(nan_min, +1),
                    )
                ) <= valid[np.isnan(data)] <= Upper(
                    np.where(
                        np.signbit(data),
                        np.copysign(nan_min, -1),
                        np.copysign(nan_max, +1),
                    )
                )
            else:
                Lower(data) <= valid[np.isnan(data)] <= Upper(data)

        with np.errstate(
            divide="ignore", over="ignore", under="ignore", invalid="ignore"
        ):
            data_mul, data_div = (
                data * eb_decimal_multipler,
                data / eb_decimal_multipler,
            )
            Lower(np.where(data < 0, data_mul, data_div)) <= valid[
                np.isfinite(data)
            ] <= Upper(np.where(data < 0, data_div, data_mul))

        if np.issubdtype(data.dtype, np.integer):
            # saturate the error bounds so that they don't wrap around
            Minimum <= valid[valid._lower > data]
            valid[valid._upper < data] <= Maximum

        # correct rounding errors in the lower and upper bound
        with np.errstate(
            divide="ignore", over="ignore", under="ignore", invalid="ignore"
        ):
            lower_bound_outside_eb_decimal = (
                np.where(data < 0, valid._lower / data, data / valid._lower)
                > eb_decimal_multipler
            )
            upper_bound_outside_eb_decimal = (
                np.where(data < 0, data / valid._upper, valid._upper / data)
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
