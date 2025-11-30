"""
Same value safeguard.
"""

__all__ = ["SameValueSafeguard"]

from collections.abc import Set
from typing import ClassVar, Literal

import numpy as np
from typing_extensions import override  # MSPV 3.12

from ...utils._compat import _ensure_array, _logical_and
from ...utils.bindings import Bindings, Parameter
from ...utils.cast import as_bits, from_total_order, lossless_cast, to_total_order
from ...utils.error import TypeCheckError, ctx
from ...utils.intervals import Interval, IntervalUnion, Lower, Maximum, Minimum, Upper
from ...utils.typing import JSON, S, T
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

    Beware that +0.0 and -0.0 are semantically equivalent in floating-point but
    have different bitwise patterns. To preserve both, two same value
    safeguards are needed, one for each bitpattern.

    Parameters
    ----------
    value : int | float | str | Parameter
        The value of or the late-bound parameter name for the certain `value`
        that is preserved by this safeguard. Literal values are (unsafely) cast
        to the data dtype before binary comparison.
    exclusive : bool
        If [`True`][True], non-`value` elements in the data stay non-`value`
        after applying corrections. If [`False`][False], non-`value` values may
        have the `value` after applying corrections.

    Raises
    ------
    TypeCheckError
        if any parameter has the wrong type.
    """

    __slots__: tuple[str, ...] = ("_value", "_exclusive")
    _value: int | float | Parameter
    _exclusive: bool

    kind: ClassVar[str] = "same"

    def __init__(
        self, value: int | float | str | Parameter, *, exclusive: bool = False
    ) -> None:
        with ctx.safeguard(self):
            with ctx.parameter("value"):
                TypeCheckError.check_instance_or_raise(
                    value, int | float | str | Parameter
                )
                if isinstance(value, Parameter):
                    self._value = value
                elif isinstance(value, str):
                    self._value = Parameter(value)
                else:
                    self._value = value

            with ctx.parameter("exclusive"):
                TypeCheckError.check_instance_or_raise(exclusive, bool)
                self._exclusive = exclusive

    @property
    @override
    def late_bound(self) -> Set[Parameter]:
        """
        The set of late-bound parameters that this safeguard has.

        Late-bound parameters are only bound when checking and applying the
        safeguard, in contrast to the normal early-bound parameters that are
        configured during safeguard initialisation.

        Late-bound parameters can be used for parameters that depend on the
        specific data that is to be safeguarded.
        """

        return (
            frozenset([self._value])
            if isinstance(self._value, Parameter)
            else frozenset()
        )

    @override
    def check_pointwise(
        self,
        data: np.ndarray[S, np.dtype[T]],
        prediction: np.ndarray[S, np.dtype[T]],
        *,
        late_bound: Bindings,
        where: Literal[True] | np.ndarray[S, np.dtype[np.bool]] = True,
    ) -> np.ndarray[S, np.dtype[np.bool]]:
        """
        Check which elements preserve the special `value` from the `data` to
        the `prediction` array.

        Parameters
        ----------
        data : np.ndarray[S, np.dtype[T]]
            Original data array, relative to which the `prediction` is checked.
        prediction : np.ndarray[S, np.dtype[T]]
            Prediction for the `data` array.
        late_bound : Bindings
            Bindings for late-bound parameters, including for this safeguard.
        where : Literal[True] | np.ndarray[S, np.dtype[np.bool]]
            Only check at data points where the condition is [`True`][True].

        Returns
        -------
        ok : np.ndarray[S, np.dtype[np.bool]]
            Pointwise, `True` if the check succeeded for this element.

        Raises
        ------
        LateBoundParameterResolutionError
            if the `value` is late-bound but its late-bound parameter is not in
            `late_bound`.
        ValueError
            if the late-bound `value` could not be broadcast to the `data`'s
            shape.
        TypeError
            if the `value` is floating-point but the `data` is integer.
        ValueError
            if not all `value`s could not be losslessly converted to the
            `data`'s type.
        """

        with ctx.safeguard(self), ctx.parameter("value"):
            value: np.ndarray[tuple[()] | S, np.dtype[T]] = (
                late_bound.resolve_ndarray_with_lossless_cast(
                    self._value,
                    data.shape,
                    data.dtype,
                )
                if isinstance(self._value, Parameter)
                else lossless_cast(self._value, data.dtype)
            )
        value_bits: np.ndarray[tuple[()] | S, np.dtype[np.unsignedinteger]] = as_bits(
            value
        )

        data_bits: np.ndarray[S, np.dtype[np.unsignedinteger]] = as_bits(data)
        prediction_bits: np.ndarray[S, np.dtype[np.unsignedinteger]] = as_bits(
            prediction
        )

        ok: np.ndarray[S, np.dtype[np.bool]]
        if self._exclusive:
            # value if and only if where value
            ok = (data_bits == value_bits) == (prediction_bits == value_bits)
        else:
            # value must stay value, everything else can be arbitrary
            ok = (data_bits != value_bits) | (prediction_bits == value_bits)
        ok = _ensure_array(ok)
        if where is not True:
            ok[~where] = True

        return ok

    @override
    def compute_safe_intervals(
        self,
        data: np.ndarray[S, np.dtype[T]],
        *,
        late_bound: Bindings,
        where: Literal[True] | np.ndarray[S, np.dtype[np.bool]] = True,
    ) -> IntervalUnion[T, int, int]:
        """
        Compute the intervals in which the same value guarantee is upheld with
        respect to the `data`.

        Parameters
        ----------
        data : np.ndarray[S, np.dtype[T]]
            Data for which the safe intervals should be computed.
        late_bound : Bindings
            Bindings for late-bound parameters, including for this safeguard.
        where : Literal[True] | np.ndarray[S, np.dtype[np.bool]]
            Only compute the safe intervals at pointwise checks where the
            condition is [`True`][True].

        Returns
        -------
        intervals : IntervalUnion[T, int, int]
            Union of intervals in which the same value guarantee is upheld.

        Raises
        ------
        LateBoundParameterResolutionError
            if the `value` is late-bound but its late-bound parameter is not in
            `late_bound`.
        ValueError
            if the late-bound `value` could not be broadcast to the `data`'s
            shape.
        TypeError
            if the `value` is floating-point but the `data` is integer.
        ValueError
            if not all `value`s could not be losslessly converted to the
            `data`'s type.
        """

        with ctx.safeguard(self), ctx.parameter("value"):
            valuef: np.ndarray[tuple[()] | tuple[int], np.dtype[T]] = (
                late_bound.resolve_ndarray_with_lossless_cast(
                    self._value,
                    data.shape,
                    data.dtype,
                ).flatten()
                if isinstance(self._value, Parameter)
                else lossless_cast(self._value, data.dtype)
            )
        valuef_bits: np.ndarray[
            tuple[()] | tuple[int], np.dtype[np.unsignedinteger]
        ] = as_bits(valuef)

        dataf: np.ndarray[tuple[int], np.dtype[T]] = data.flatten()
        dataf_bits: np.ndarray[tuple[int], np.dtype[np.unsignedinteger]] = as_bits(
            dataf
        )

        wheref: Literal[True] | np.ndarray[tuple[int], np.dtype[np.bool]] = (
            True if where is True else where.flatten()
        )

        valid = Interval.empty_like(dataf)

        if not self._exclusive:
            # preserve value elements exactly, do not constrain other elements
            valid = Interval.full_like(dataf)
            Lower(valuef) <= valid[dataf_bits == valuef_bits] <= Upper(valuef)
            return valid.preserve_only_where(wheref).into_union()

        valuef_total: np.ndarray[
            tuple[()] | tuple[int], np.dtype[np.unsignedinteger]
        ] = to_total_order(valuef)

        total_min = np.array(np.iinfo(valuef_total.dtype).min, dtype=valuef_total.dtype)
        total_max = np.array(np.iinfo(valuef_total.dtype).max, dtype=valuef_total.dtype)

        valid_below = Interval.empty_like(dataf)
        valid_above = Interval.empty_like(dataf)

        Lower(valuef) <= valid_below[dataf_bits == valuef_bits] <= Upper(valuef)

        with np.errstate(over="ignore", under="ignore"):
            below_upper = _ensure_array(from_total_order(valuef_total - 1, data.dtype))
            above_lower = _ensure_array(from_total_order(valuef_total + 1, data.dtype))

        # non-value elements must exclude value from their interval,
        #  leading to a union of two intervals, below and above value
        Minimum <= valid_below[
            (dataf_bits != valuef_bits) & (valuef_total > total_min)
        ] <= Upper(below_upper)

        Lower(above_lower) <= valid_above[
            (dataf_bits != valuef_bits) & (valuef_total < total_max)
        ] <= Maximum

        return valid_below.union(valid_above).preserve_only_where(wheref)

    @override
    def compute_footprint(
        self,
        foot: np.ndarray[S, np.dtype[np.bool]],
        *,
        late_bound: Bindings,
        where: Literal[True] | np.ndarray[S, np.dtype[np.bool]] = True,
    ) -> np.ndarray[S, np.dtype[np.bool]]:
        """
        Compute the footprint of the `foot` array, e.g. for expanding data
        points into the pointwise checks that they contribute to.

        The footprint is equivalent to `foot & where`.

        Parameters
        ----------
        foot : np.ndarray[S, np.dtype[np.bool]]
            Array for which the footprint is computed.
        late_bound : Bindings
            Bindings for late-bound parameters, including for this safeguard.
        where : Literal[True] | np.ndarray[S, np.dtype[np.bool]]
            Only compute the footprint at pointwise checks where the condition
            is [`True`][True].

        Returns
        -------
        print : np.ndarray[S, np.dtype[np.bool]]
            The footprint of the `foot` array.
        """

        return _logical_and(foot, where)

    @override
    def compute_inverse_footprint(
        self,
        foot: np.ndarray[S, np.dtype[np.bool]],
        *,
        late_bound: Bindings,
        where: Literal[True] | np.ndarray[S, np.dtype[np.bool]] = True,
    ) -> np.ndarray[S, np.dtype[np.bool]]:
        """
        Compute the inverse footprint of the `foot` array, e.g. for expanding
        pointwise check fails into the points that could have contributed to
        the failures.

        The inverse footprint is equivalent to `foot & where`.

        Parameters
        ----------
        foot : np.ndarray[S, np.dtype[np.bool]]
            Array for which the inverse footprint is computed.
        late_bound : Bindings
            Bindings for late-bound parameters, including for this safeguard.
        where : Literal[True] | np.ndarray[S, np.dtype[np.bool]]
            Only compute the inverse footprint at pointwise checks where the
            condition is [`True`][True].

        Returns
        -------
        print : np.ndarray[S, np.dtype[np.bool]]
            The inverse footprint of the `foot` array.
        """

        return _logical_and(foot, where)

    @override
    def get_config(self) -> dict[str, JSON]:
        """
        Returns the configuration of the safeguard.

        Returns
        -------
        config : dict[str, JSON]
            Configuration of the safeguard.
        """

        return dict(kind=type(self).kind, value=self._value, exclusive=self._exclusive)
