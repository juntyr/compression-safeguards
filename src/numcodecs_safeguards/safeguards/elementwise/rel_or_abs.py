"""
Relative (or absolute) error bound safeguard.
"""

__all__ = ["RelativeOrAbsoluteErrorBoundSafeguard"]

import numpy as np

from .abc import ElementwiseSafeguard
from ...cast import to_float, from_float, as_bits
from ...intervals import (
    IntervalUnion,
    Interval,
    Lower,
    Upper,
    _to_total_order,
    _from_total_order,
)


class RelativeOrAbsoluteErrorBoundSafeguard(ElementwiseSafeguard):
    r"""
    The `RelativeOrAbsoluteErrorBoundSafeguard` guarantees that either the
    absolute error between the logarithms of the values is less than or
    equal to $\log$(1 + `eb_rel`), and/or the absolute error between the values
    is less than or equal to the provided absolute error bound `eb_abs`.

    Infinite values are preserved with the same bit pattern. If `equal_nan` is
    set to [`True`][True], decoding a NaN value to a NaN value with a different
    bitpattern also satisfies the error bound. If `equal_nan` is set to
    [`False`][False], NaN values are also preserved with the same bit pattern.

    Parameters
    ----------
    eb_rel : int | float
        The non-negative relative error bound that is enforced by this
        safeguard. `eb_rel=0.02` corresponds to a 2% relative bound.
    eb_abs : int | float
        The non-negative absolute error bound that is enforced by this
        safeguard.
    equal_nan: bool
        Whether decoding a NaN value to a NaN value with a different bit
        pattern satisfies the error bound.
    """

    __slots__ = ("_eb_rel", "_eb_abs", "_equal_nan")
    _eb_rel: int | float
    _eb_abs: int | float
    _equal_nan: bool

    kind = "rel_or_abs"

    def __init__(
        self, eb_rel: int | float, eb_abs: int | float, *, equal_nan: bool = False
    ):
        assert eb_rel >= 0, "eb_rel must be non-negative"
        assert np.isfinite(eb_rel), "eb_rel must be finite"
        assert eb_abs >= 0, "eb_abs must be non-negative"
        assert np.isfinite(eb_abs), "eb_abs must be finite"

        self._eb_rel = eb_rel
        self._eb_abs = eb_abs
        self._equal_nan = equal_nan

    @np.errstate(divide="ignore", over="ignore", under="ignore", invalid="ignore")
    def check(self, data: np.ndarray, decoded: np.ndarray) -> bool:
        """
        Check if the `decoded` array satisfies the relative or the absolute
        error bound.

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

        # abs(data - decoded) <= self._eb_abs, but works for unsigned ints
        absolute_bound = (
            np.where(
                data > decoded,
                data - decoded,
                decoded - data,
            )
            <= self._eb_abs
        )
        relative_bound = (np.sign(data) == np.sign(decoded)) & (
            (
                np.where(
                    np.abs(data) > np.abs(decoded),
                    to_float(data) / to_float(decoded),
                    to_float(decoded) / to_float(data),
                )
                - 1
            )
            <= self._eb_rel
        )
        # bitwise equality for inf and NaNs (unless equal_nan)
        same_bits = as_bits(data) == as_bits(decoded)
        both_nan = self._equal_nan and (np.isnan(data) & np.isnan(decoded))

        ok = np.where(
            np.isfinite(data),
            relative_bound | absolute_bound,
            np.where(
                np.isinf(data),
                same_bits,
                both_nan if self._equal_nan else same_bits,
            ),
        )

        return bool(np.all(ok))

    def compute_safe_intervals(self, data: np.ndarray) -> IntervalUnion:
        """
        Compute the intervals in which the relative or absolute error bound is
        upheld with respect to the `data`.

        Parameters
        ----------
        data : np.ndarray
            Data for which the safe intervals should be computed.

        Returns
        -------
        intervals : IntervalUnion
            Union of intervals in which the relative or absolute error bound is
            upheld.
        """

        data = data.flatten()
        data_float: np.ndarray = to_float(data)

        valid = (
            Interval.empty_like(data)
            .preserve_inf(data)
            .preserve_nan(data, equal_nan=self._equal_nan)
        )

        with np.errstate(
            divide="ignore", over="ignore", under="ignore", invalid="ignore"
        ):
            eb_rel_multipler = min(
                np.array(self._eb_rel).astype(data_float.dtype) + 1,
                np.finfo(data_float.dtype).max,
            )
        assert eb_rel_multipler >= 1.0 and np.isfinite(eb_rel_multipler)

        with np.errstate(
            divide="ignore", over="ignore", under="ignore", invalid="ignore"
        ):
            data_mul, data_div = (
                from_float(data_float * eb_rel_multipler, data.dtype),
                from_float(data_float / eb_rel_multipler, data.dtype),
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
                > eb_rel_multipler
            )
            upper_bound_outside_eb_decimal = (
                np.abs(
                    np.where(
                        data < 0,
                        data_float / to_float(valid._upper),
                        to_float(valid._upper) / data_float,
                    )
                )
                > eb_rel_multipler
            )

        valid._lower[np.isfinite(data)] = _from_total_order(
            _to_total_order(valid._lower) + lower_bound_outside_eb_decimal,
            data.dtype,
        )[np.isfinite(data)]
        valid._upper[np.isfinite(data)] = _from_total_order(
            _to_total_order(valid._upper) - upper_bound_outside_eb_decimal,
            data.dtype,
        )[np.isfinite(data)]

        # create a separate interval for the absolute error bound
        valid_abs = Interval.empty_like(data)

        with np.errstate(
            divide="ignore", over="ignore", under="ignore", invalid="ignore"
        ):
            eb_abs = min(
                np.array(self._eb_abs).astype(data_float.dtype),
                np.finfo(data_float.dtype).max,
            )
        assert eb_abs >= 0.0 and np.isfinite(eb_abs)

        with np.errstate(
            divide="ignore", over="ignore", under="ignore", invalid="ignore"
        ):
            Lower(from_float(data_float - eb_abs, data.dtype)) <= valid_abs[
                np.isfinite(data)
            ] <= Upper(from_float(data_float + eb_abs, data.dtype))

        # correct rounding errors in the lower and upper bound
        with np.errstate(
            divide="ignore", over="ignore", under="ignore", invalid="ignore"
        ):
            # we don't use abs(data - bound) here to accommodate unsigned ints
            lower_bound_outside_eb_abs = (
                data_float - to_float(valid_abs._lower)
            ) > self._eb_abs
            upper_bound_outside_eb_abs = (
                to_float(valid_abs._upper) - data_float
            ) > self._eb_abs

        valid_abs._lower[np.isfinite(data)] = _from_total_order(
            _to_total_order(valid_abs._lower) + lower_bound_outside_eb_abs,
            data.dtype,
        )[np.isfinite(data)]
        valid_abs._upper[np.isfinite(data)] = _from_total_order(
            _to_total_order(valid_abs._upper) - upper_bound_outside_eb_abs,
            data.dtype,
        )[np.isfinite(data)]

        # combine the absolute and relative error bounds
        Lower(np.minimum(valid._lower, valid_abs._lower)) <= valid[
            np.isfinite(data)
        ] <= Upper(np.maximum(valid._upper, valid_abs._upper))

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
            eb_rel=self._eb_rel,
            eb_abs=self._eb_abs,
            equal_nan=self._equal_nan,
        )
