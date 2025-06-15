"""
Zero-is-zero safeguard.
"""

__all__ = ["ZeroIsZeroSafeguard"]

import numpy as np

from ...utils.bindings import Bindings, Parameter
from ...utils.cast import as_bits, from_total_order, to_total_order
from ...utils.intervals import Interval, IntervalUnion, Lower, Maximum, Minimum, Upper
from ...utils.typing import S, T
from .abc import PointwiseSafeguard


class ZeroIsZeroSafeguard(PointwiseSafeguard):
    """
    The `ZeroIsZeroSafeguard` guarantees that values that are zero in the input
    are also *exactly* zero in the decompressed output. By default, non-zero
    values may be zero in the output, though the `exclusive` parameter can also
    enforce that only zero values are zero after decompression.

    This safeguard can also be used to enforce that another constant value is
    bitwise preserved, e.g. a missing value constant or a semantic "zero" value
    that is represented as a non-zero number.

    Beware that +0.0 and -0.0 are semantically equivalent in floating point but
    have different bitwise patterns. If you want to preserve both, you need to
    use two safeguards, one configured for each zero.

    Parameters
    ----------
    zero : int | float | str | Parameter, optional
        The value of or the late-bound parameter name for the constant "zero"
        value that is preserved by this safeguard.
    exclusive : bool
        If [`True`][True], non-`zero` values in the data stay non-`zero` after
        decoding. If [`False`][False], non-`zero` values may be `zero` after
        decoding.
    """

    __slots__ = ("_zero", "_exclusive")
    _zero: int | float | Parameter
    _exclusive: bool

    kind = "zero"

    def __init__(
        self, zero: int | float | str | Parameter = 0, exclusive: bool = False
    ):
        if isinstance(zero, Parameter):
            self._zero = zero
        elif isinstance(zero, str):
            self._zero = Parameter(zero)
        else:
            self._zero = zero

        self._exclusive = exclusive

    def check_pointwise(
        self,
        data: np.ndarray[S, np.dtype[T]],
        decoded: np.ndarray[S, np.dtype[T]],
        *,
        late_bound: Bindings,
    ) -> np.ndarray[S, np.dtype[np.bool]]:
        """
        Check which elements are either
        - non-zero in the `data` array,
        - or zero in the `data` *and* the `decoded` array.

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

        zero: np.ndarray[tuple[()] | S, np.dtype[T]] = (
            late_bound.resolve_ndarray(
                self._zero,
                data.shape,
                data.dtype,
            )
            if isinstance(self._zero, Parameter)
            else self._zero_like(data.dtype)
        )
        zero_bits = as_bits(zero)

        if self._exclusive:
            # must only be zero where zero
            return (as_bits(data) == zero_bits) == (as_bits(decoded) == zero_bits)

        # zeros must stay zeros, everything else can be arbitrary
        return (as_bits(data) != zero_bits) | (as_bits(decoded) == zero_bits)

    def compute_safe_intervals(
        self,
        data: np.ndarray[S, np.dtype[T]],
        *,
        late_bound: Bindings,
    ) -> IntervalUnion[T, int, int]:
        """
        Compute the intervals in which the zero-is-zero guarantee is upheld with
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
            Union of intervals in which the zero-is-zero guarantee is upheld.
        """

        zero = (
            late_bound.resolve_ndarray(
                self._zero,
                data.shape,
                data.dtype,
            )
            if isinstance(self._zero, Parameter)
            else self._zero_like(data.dtype)
        )
        zerof = zero.flatten()

        dataf = data.flatten()
        valid = Interval.empty_like(dataf)

        if not self._exclusive:
            # preserve zero values exactly, do not constrain other values
            valid = Interval.full_like(dataf)
            Lower(zerof) <= valid[as_bits(dataf) == as_bits(zerof)] <= Upper(zerof)
            return valid.into_union()

        zerof_total: np.ndarray = to_total_order(zerof)

        total_min = np.iinfo(zerof_total.dtype).min
        total_max = np.iinfo(zerof_total.dtype).max

        valid_below = Interval.empty_like(dataf)
        valid_above = Interval.empty_like(dataf)

        Lower(zerof) <= valid_below[as_bits(dataf) == as_bits(zerof)] <= Upper(zerof)

        upper = from_total_order(zerof_total - 1, data.dtype)
        lower = from_total_order(zerof_total + 1, data.dtype)

        # non-zero values must exclude zero from their interval,
        #  leading to a union of two intervals, below and above zero
        Minimum <= valid_below[
            (as_bits(dataf) != as_bits(zerof)) & (zerof_total > total_min)
        ] <= Upper(upper)

        Lower(lower) <= valid_above[
            (as_bits(dataf) != as_bits(zerof)) & (zerof_total < total_max)
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

        return dict(kind=type(self).kind, zero=self._zero, exclusive=self._exclusive)

    def _zero_like(self, dtype: np.dtype[T]) -> np.ndarray[tuple[()], np.dtype[T]]:
        zero = np.array(self._zero)
        if zero.dtype != dtype:
            with np.errstate(
                divide="ignore", over="ignore", under="ignore", invalid="ignore"
            ):
                zero = zero.astype(dtype)
        return zero  # type: ignore
