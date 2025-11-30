"""
Error bound safeguard.
"""

__all__ = ["ErrorBoundSafeguard"]

from collections.abc import Set
from typing import ClassVar, Literal

import numpy as np
from typing_extensions import override  # MSPV 3.12

from ...utils._compat import _ensure_array, _logical_and
from ...utils.bindings import Bindings, Parameter
from ...utils.cast import ToFloatMode, as_bits, saturating_finite_float_cast, to_float
from ...utils.error import TypeCheckError, ctx, lookup_enum_or_raise
from ...utils.intervals import Interval, IntervalUnion, Lower, Upper
from ...utils.typing import JSON, S, T
from ..eb import (
    ErrorBound,
    _apply_finite_error_bound,
    _check_error_bound,
    _compute_finite_absolute_error,
    _compute_finite_absolute_error_bound,
)
from .abc import PointwiseSafeguard


class ErrorBoundSafeguard(PointwiseSafeguard):
    """
    The `ErrorBoundSafeguard` guarantees that the pointwise error `type` is less than or equal to the provided bound `eb`.

    Infinite values are preserved with the same bit pattern. If `equal_nan` is
    set to [`True`][True], correcting a NaN value to a NaN value with a
    different bit pattern also satisfies the error bound. If `equal_nan` is set
    to [`False`][False], NaN values are also preserved with the same bit
    pattern.

    The error bound can be verified by casting the data and error bound to a
    sufficiently large floating-point type, selected by
    [`ToFloatMode.lossless.floating_point_dtype_for`][.....utils.cast.ToFloatMode.floating_point_dtype_for],
    using [`to_float`][.....utils.cast.to_float].

    Parameters
    ----------
    type : str | ErrorBound
        The type of error bound that is enforced by this safeguard.
    eb : int | float | str | Parameter
        The value of or late-bound parameter name for the error bound that is
        enforced by this safeguard.
    equal_nan: bool
        Whether correcting a NaN value to a NaN value with a different bit
        pattern satisfies the error bound.

    Raises
    ------
    TypeCheckError
        if any parameter has the wrong type.
    ValueError
        if `type` does not name a valid error bound.
    ValueError
        if `eb` is an invalid error bound value for the error bound `type`.
    """

    __slots__: tuple[str, ...] = ("_type", "_eb", "_equal_nan")
    _type: ErrorBound
    _eb: int | float | Parameter
    _equal_nan: bool

    kind: ClassVar[str] = "eb"

    def __init__(
        self,
        type: str | ErrorBound,
        eb: int | float | str | Parameter,
        *,
        equal_nan: bool = False,
    ) -> None:
        with ctx.safeguard(self):
            with ctx.parameter("type"):
                TypeCheckError.check_instance_or_raise(type, str | ErrorBound)
                self._type = (
                    type
                    if isinstance(type, ErrorBound)
                    else lookup_enum_or_raise(ErrorBound, type)
                )

            with ctx.parameter("eb"):
                TypeCheckError.check_instance_or_raise(
                    eb, int | float | str | Parameter
                )
                if isinstance(eb, Parameter):
                    self._eb = eb
                elif isinstance(eb, str):
                    self._eb = Parameter(eb)
                else:
                    _check_error_bound(self._type, eb)
                    self._eb = eb

            with ctx.parameter("equal_nan"):
                TypeCheckError.check_instance_or_raise(equal_nan, bool)
                self._equal_nan = equal_nan

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

        return frozenset([self._eb]) if isinstance(self._eb, Parameter) else frozenset()

    @np.errstate(divide="ignore", over="ignore", under="ignore", invalid="ignore")
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
        Check which elements in the `prediction` array satisfy the error bound.

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
            if the error bound `eb` is late-bound but its late-bound parameter
            is not in `late_bound`.
        ValueError
            if the late-bound `eb` could not be broadcast to the `data`'s shape.
        ValueError
            if the late-bound `eb` is non-finite, i.e. infinite or NaN, or an
            invalid error bound value for the error bound `type`.
        """

        ftype: np.dtype[np.floating] = ToFloatMode.lossless.floating_point_dtype_for(
            data.dtype
        )
        data_float: np.ndarray[S, np.dtype[np.floating]] = to_float(data, ftype=ftype)
        prediction_float: np.ndarray[S, np.dtype[np.floating]] = to_float(
            prediction, ftype=ftype
        )

        with ctx.safeguard(self), ctx.parameter("eb"):
            eb: np.ndarray[tuple[()] | S, np.dtype[np.floating]] = (
                late_bound.resolve_ndarray_with_saturating_finite_float_cast(
                    self._eb,
                    data.shape,
                    ftype,
                )
                if isinstance(self._eb, Parameter)
                else saturating_finite_float_cast(self._eb, ftype)
            )
            if isinstance(self._eb, Parameter):
                _eb: Parameter = self._eb
                with ctx.late_bound_parameter(_eb):
                    _check_error_bound(self._type, eb)

        finite_ok: np.ndarray[S, np.dtype[np.bool]] = np.less_equal(
            _compute_finite_absolute_error(self._type, data_float, prediction_float),
            _compute_finite_absolute_error_bound(self._type, eb, data_float),
        )

        # bitwise equality for inf and NaNs (unless equal_nan)
        same_bits = as_bits(data) == as_bits(prediction)
        both_nan = self._equal_nan and (np.isnan(data) & np.isnan(prediction))

        ok: np.ndarray[S, np.dtype[np.bool]] = _ensure_array(finite_ok)
        np.copyto(ok, same_bits, where=~np.isfinite(data), casting="no")
        if self._equal_nan:
            np.copyto(ok, both_nan, where=np.isnan(data), casting="no")
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
        Compute the intervals in which the error bound is upheld with respect
        to the `data`.

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
            Union of intervals in which the error bound is upheld.

        Raises
        ------
        LateBoundParameterResolutionError
            if the error bound `eb` is late-bound but its late-bound parameter
            is not in `late_bound`.
        ValueError
            if the late-bound `eb` could not be broadcast to the `data`'s shape.
        ValueError
            if the late-bound `eb` is non-finite, i.e. infinite or NaN, or an
            invalid error bound value for the error bound `type`.
        """

        ftype: np.dtype[np.floating] = ToFloatMode.lossless.floating_point_dtype_for(
            data.dtype
        )

        with ctx.safeguard(self), ctx.parameter("eb"):
            eb: np.ndarray[tuple[()] | S, np.dtype[np.floating]] = (
                late_bound.resolve_ndarray_with_saturating_finite_float_cast(
                    self._eb,
                    data.shape,
                    ftype,
                )
                if isinstance(self._eb, Parameter)
                else saturating_finite_float_cast(self._eb, ftype)
            )
            if isinstance(self._eb, Parameter):
                _eb: Parameter = self._eb
                with ctx.late_bound_parameter(_eb):
                    _check_error_bound(self._type, eb)

        lower, upper = _apply_finite_error_bound(
            self._type, eb, data, to_float(data, ftype=ftype)
        )

        dataf: np.ndarray[tuple[int], np.dtype[T]] = data.flatten()
        valid = Interval.empty_like(dataf).preserve_inf(dataf)

        Lower(lower.flatten()) <= valid[np.isfinite(dataf)] <= Upper(upper.flatten())

        wheref: Literal[True] | np.ndarray[tuple[int], np.dtype[np.bool]] = (
            True if where is True else where.flatten()
        )

        return valid.preserve_any_nan(
            dataf, equal_nan=self._equal_nan
        ).preserve_only_where(wheref)

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

        return dict(
            kind=type(self).kind,
            type=self._type.name,
            eb=str(self._eb) if isinstance(self._eb, Parameter) else self._eb,
            equal_nan=self._equal_nan,
        )
