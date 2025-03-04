"""
Relative (or absolute) error bound safeguard.
"""

__all__ = ["RelativeOrAbsoluteErrorBoundSafeguard"]

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


class RelativeOrAbsoluteErrorBoundSafeguard(ElementwiseSafeguard):
    r"""
    The `RelativeOrAbsoluteErrorBoundSafeguard` guarantees that the elementwise
    absolute error between the *logarithms*\* of the values is less than or
    equal to $\log(1 + eb_{rel})$ where `eb_rel` is e.g. 2%.

    The logarithm* here is adapted to support positive, negative, and zero
    values. For values close to zero, where the relative error is not well-
    defined, the absolute elementwise error is guaranteed to be less than or
    equal to the absolute error bound.

    Put simply, each element satisfies the relative or the absolute error bound
    (or both). In cases where the arithmetic evaluation of the error bound is
    not well-defined, e.g. for infinite or NaN values, producing the exact same
    bitpattern is defined to satisfy the error bound. If `equal_nan` is set to
    [`True`][True], decoding a NaN value to a NaN value with a different
    bitpattern also satisfies the error bound.

    Parameters
    ----------
    eb_rel : int | float
        The positive relative error bound that is enforced by this safeguard.
        `eb_rel=0.02` corresponds to a 2% relative bound.
    eb_abs : int | float
        The positive absolute error bound that is enforced by this safeguard.
    equal_nan: bool
        Whether decoding a NaN value to a NaN value with a different bit
        pattern satisfies the error bound.
    """

    __slots__ = ("_eb_rel", "_eb_abs", "_equal_nan")
    _eb_rel: int | float
    _eb_abs: int | float
    _equal_nan: bool

    kind = "rel_or_abs"
    _priority = 0

    def __init__(
        self, eb_rel: int | float, eb_abs: int | float, *, equal_nan: bool = False
    ):
        assert eb_rel > 0, "eb_rel must be positive"
        assert np.isfinite(eb_rel), "eb_rel must be finite"
        assert eb_abs > 0, "eb_abs must be positive"
        assert np.isfinite(eb_abs), "eb_abs must be finite"

        self._eb_rel = eb_rel
        self._eb_abs = eb_abs
        self._equal_nan = equal_nan

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
        valid = Interval.empty_like(data)

        with np.errstate(
            divide="ignore", over="ignore", under="ignore", invalid="ignore"
        ):
            eb_rel_multipler = np.array(self._eb_rel + 1, dtype=data.dtype)
        if eb_rel_multipler < 1 or not np.isfinite(eb_rel_multipler):
            eb_rel_multipler = np.array(
                np.finfo(data.dtype).max
                if np.issubdtype(data.dtype, np.floating)
                else np.iinfo(data.dtype).max,
                dtype=data.dtype,
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
                data * eb_rel_multipler,
                data / eb_rel_multipler,
            )
            Lower(np.where(data < 0, data_mul, data_div)) <= valid[
                np.isfinite(data)
            ] <= Upper(np.where(data < 0, data_div, data_mul))

            # TODO: also add the abs error bound

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
                > eb_rel_multipler
            )
            upper_bound_outside_eb_decimal = (
                np.where(data < 0, data / valid._upper, valid._upper / data)
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
