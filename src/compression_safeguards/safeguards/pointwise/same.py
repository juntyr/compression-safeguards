"""
Same value safeguard.
"""

__all__ = ["SameValueSafeguard"]

import numpy as np

from ...utils.bindings import Bindings, Parameter
from ...utils.cast import as_bits, from_total_order, to_total_order
from ...utils.intervals import Interval, IntervalUnion, Lower, Maximum, Minimum, Upper
from ...utils.typing import S, T
from .abc import PointwiseSafeguard


class SameValueSafeguard(PointwiseSafeguard):
    """
    The `SameValueSafeguard` guarantees that if an element has a special
    `value` in the input, that element also has bitwise the same value in the
    decompressed output.

    This safeguard can be used for preserving e.g. zero values, missing values,
    pre-computed extreme values, or any other value of importance.

    By default, elements that do *not* have the special `value` in the input
    may still have the value in the output. Enabling the `exclusive` flag
    enforces that an element in the output only has the special `value` if and
    only if it also has the `value` in the input, e.g. to ensure that only
    missing values in the input have the missing value bitpattern in the
    output.

    Beware that +0.0 and -0.0 are semantically equivalent in floating point but
    have different bitwise patterns. To preserve both, two same value
    safeguards are needed, one for each bitpattern.

    Parameters
    ----------
    value : int | float | str | Parameter
        The value of or the late-bound parameter name for the certain `value`
        that is preserved by this safeguard.
    exclusive : bool
        If [`True`][True], non-`value` elements in the data stay non-`value`
        after decoding. If [`False`][False], non-`value` values may have the
        `value` after decoding.
    """

    __slots__ = ("_value", "_exclusive")
    _value: int | float | Parameter
    _exclusive: bool

    kind = "same"

    def __init__(
        self, value: int | float | str | Parameter, *, exclusive: bool = False
    ):
        if isinstance(value, Parameter):
            self._value = value
        elif isinstance(value, str):
            self._value = Parameter(value)
        else:
            self._value = value

        self._exclusive = exclusive

    def check_pointwise(
        self,
        data: np.ndarray[S, np.dtype[T]],
        decoded: np.ndarray[S, np.dtype[T]],
        *,
        late_bound: Bindings,
    ) -> np.ndarray[S, np.dtype[np.bool]]:
        """
        Check which elements preserve the special `value` from the `data` to
        the `decoded` array.

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

        value: np.ndarray[tuple[()] | S, np.dtype[T]] = (
            late_bound.resolve_ndarray(
                self._value,
                data.shape,
                data.dtype,
            )
            if isinstance(self._value, Parameter)
            else self._value_like(data.dtype)
        )
        value_bits = as_bits(value)

        data_bits = as_bits(data)
        decoded_bits = as_bits(decoded)

        if self._exclusive:
            # value if and only if where value
            return (data_bits == value_bits) == (decoded_bits == value_bits)

        # value must stay value, everything else can be arbitrary
        return (data_bits != value_bits) | (decoded_bits == value_bits)

    def compute_safe_intervals(
        self,
        data: np.ndarray[S, np.dtype[T]],
        *,
        late_bound: Bindings,
    ) -> IntervalUnion[T, int, int]:
        """
        Compute the intervals in which the same value guarantee is upheld with
        respect to the `data`.

        Parameters
        ----------
        data : np.ndarray
            Data for which the safe intervals should be computed.
        late_bound : Bindings
            Bindings for late-bound parameters, including for this safeguard.

        Returns
        -------
        intervals : IntervalUnion
            Union of intervals in which the same value guarantee is upheld.
        """

        valuef = (
            late_bound.resolve_ndarray(
                self._value,
                data.shape,
                data.dtype,
            ).flatten()
            if isinstance(self._value, Parameter)
            else self._value_like(data.dtype)
        )
        valuef_bits = as_bits(valuef)

        dataf = data.flatten()
        dataf_bits = as_bits(dataf)

        valid = Interval.empty_like(dataf)

        if not self._exclusive:
            # preserve value elements exactly, do not constrain other elements
            valid = Interval.full_like(dataf)
            Lower(valuef) <= valid[dataf_bits == valuef_bits] <= Upper(valuef)
            return valid.into_union()

        valuef_total: np.ndarray = to_total_order(valuef)

        total_min = np.iinfo(valuef_total.dtype).min
        total_max = np.iinfo(valuef_total.dtype).max

        valid_below = Interval.empty_like(dataf)
        valid_above = Interval.empty_like(dataf)

        Lower(valuef) <= valid_below[dataf_bits == valuef_bits] <= Upper(valuef)

        with np.errstate(over="ignore", under="ignore"):
            below_upper = np.array(from_total_order(valuef_total - 1, data.dtype))
            above_lower = np.array(from_total_order(valuef_total + 1, data.dtype))

        # non-value elements must exclude value from their interval,
        #  leading to a union of two intervals, below and above value
        Minimum <= valid_below[
            (dataf_bits != valuef_bits) & (valuef_total > total_min)
        ] <= Upper(below_upper)

        Lower(above_lower) <= valid_above[
            (dataf_bits != valuef_bits) & (valuef_total < total_max)
        ] <= Maximum

        return valid_below.union(valid_above)

    def get_config(self) -> dict:
        """
        Returns the configuration of the safeguard.

        Returns
        -------
        config : dict
            Configuration of the safeguard.
        """

        return dict(kind=type(self).kind, value=self._value, exclusive=self._exclusive)

    def _value_like(self, dtype: np.dtype[T]) -> np.ndarray[tuple[()], np.dtype[T]]:
        value = np.array(self._value)
        if value.dtype != dtype:
            with np.errstate(
                divide="ignore", over="ignore", under="ignore", invalid="ignore"
            ):
                # TODO: should we silently cast here
                value = value.astype(dtype, casting="unsafe")
        return value  # type: ignore
