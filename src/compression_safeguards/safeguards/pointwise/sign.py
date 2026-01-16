"""
Sign-preserving safeguard.
"""

__all__ = ["SignPreservingSafeguard"]

from collections.abc import Set
from typing import ClassVar, Literal

import numpy as np
from typing_extensions import override  # MSPV 3.12

from ...utils._compat import _ensure_array, _logical_and
from ...utils.bindings import Bindings, Parameter
from ...utils.cast import from_total_order, lossless_cast, to_total_order
from ...utils.error import TypeCheckError, ctx
from ...utils.intervals import Interval, IntervalUnion, Lower, Upper
from ...utils.typing import JSON, S, T
from .abc import PointwiseSafeguard


class SignPreservingSafeguard(PointwiseSafeguard):
    r"""
    The `SignPreservingSafeguard` guarantees that values have the same sign
    (-1, 0, +1) in the decompressed output as they have in the input data.

    NaN values are preserved as NaN values with the same sign bit.

    This safeguard can be configured to preserve the sign relative to a custom
    `offset`, e.g. to preserve global minima and maxima. The sign is then
    defined based on arithmetic comparison, i.e.

    \[
        \text{sign}(x, offset) = \begin{cases}
            -\text{NaN} \quad &\text{if } x \text{ is} -\text{NaN} \\
            -1 \quad &\text{if } x < offset \\
            0 \quad &\text{if } x = offset \\
            +1 \quad &\text{if } x > offset \\
            +\text{NaN} \quad &\text{if } x \text{ is} +\text{NaN}
        \end{cases}
    \]

    This safeguard should be combined with e.g. an error bound, as it by itself
    accepts *any* value with the same sign.

    Parameters
    ----------
    offset : int | float | str | Parameter, optional
        The non-NaN value of or the late-bound parameter name for the offset
        compared to which the sign is computed. By default, the offset is
        zero. Values that are above / below / equal to the offset are
        guaranteed to stay above / below / equal to the offset, respectively.
        Literal values are (unsafely) cast to the data dtype before comparison.

    Raises
    ------
    TypeCheckError
        if any parameter has the wrong type.
    ValueError
        if `offset` is NaN.
    """

    __slots__: tuple[str, ...] = ("_offset",)
    _offset: int | float | Parameter

    kind: ClassVar[str] = "sign"

    def __init__(self, *, offset: int | float | str | Parameter = 0) -> None:
        with ctx.safeguard(self):
            with ctx.parameter("offset"):
                TypeCheckError.check_instance_or_raise(
                    offset, int | float | str | Parameter
                )
                if isinstance(offset, Parameter):
                    self._offset = offset
                elif isinstance(offset, str):
                    self._offset = Parameter(offset)
                elif isinstance(offset, int) or (not np.isnan(offset)):
                    self._offset = offset
                else:
                    raise ValueError("must not be NaN") | ctx

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
            frozenset([self._offset])
            if isinstance(self._offset, Parameter)
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
        Check for which elements in the `prediction` array the signs match the
        signs of the `data` array elements'.

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
            if the `offset` is late-bound but its late-bound parameter is not in
            `late_bound`.
        ValueError
            if the late-bound `offset` could not be broadcast to the `data`'s
            shape.
        ValueError
            if the late-bound `offset` contains NaN values.
        TypeError
            if the `offset` is floating-point but the `data` is integer.
        ValueError
            if not all `offset`s could not be losslessly converted to the
            `data`'s type.
        """

        with ctx.safeguard(self), ctx.parameter("offset"):
            offset: np.ndarray[tuple[()] | S, np.dtype[T]] = (
                late_bound.resolve_ndarray_with_lossless_cast(
                    self._offset,
                    data.shape,
                    data.dtype,
                )
                if isinstance(self._offset, Parameter)
                else lossless_cast(self._offset, data.dtype)
            )
            if isinstance(self._offset, Parameter):
                if np.any(np.isnan(offset)):
                    with ctx.late_bound_parameter(self._offset):
                        raise ValueError("must not contain any NaN values") | ctx

        # values equal to the offset (sign=0) stay equal
        # values below (sign=-1) stay below,
        # values above (sign=+1) stay above
        # NaN values keep their sign bit
        ok: np.ndarray[S, np.dtype[np.bool]] = _ensure_array(prediction == offset)
        np.less(prediction, offset, out=ok, where=np.less(data, offset))
        np.greater(prediction, offset, out=ok, where=np.greater(data, offset))
        np.copyto(
            ok,
            np.isnan(prediction) & (np.signbit(data) == np.signbit(prediction)),
            where=np.isnan(data),
            casting="no",
        )
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
        Compute the intervals in which the `data`'s sign is preserved.

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
            Union of intervals in which the `data`'s sign is preserved.

        Raises
        ------
        LateBoundParameterResolutionError
            if the `offset` is late-bound but its late-bound parameter is not
            in `late_bound`.
        ValueError
            if the late-bound `offset` could not be broadcast to the `data`'s
            shape.
        ValueError
            if the late-bound `offset` contains NaN values.
        TypeError
            if the `offset` is floating-point but the `data` is integer.
        ValueError
            if not all `offset`s could not be losslessly converted to the
            `data`'s type.
        """

        with ctx.safeguard(self), ctx.parameter("offset"):
            offsetf: np.ndarray[tuple[()] | tuple[int], np.dtype[T]] = (
                late_bound.resolve_ndarray_with_lossless_cast(
                    self._offset,
                    data.shape,
                    data.dtype,
                ).flatten()
                if isinstance(self._offset, Parameter)
                else lossless_cast(self._offset, data.dtype)
            )
            if isinstance(self._offset, Parameter):
                if np.any(np.isnan(offsetf)):
                    with ctx.late_bound_parameter(self._offset):
                        raise ValueError("must not contain any NaN values") | ctx
        offsetf_total: np.ndarray[
            tuple[()] | tuple[int], np.dtype[np.unsignedinteger]
        ] = to_total_order(offsetf)

        with np.errstate(over="ignore", under="ignore"):
            if np.issubdtype(data.dtype, np.floating):
                # special case for floating-point -0.0 / +0.0 which both have
                #  zero sign and thus have weird below / above intervals
                smallest_subnormal = np.finfo(data.dtype).smallest_subnormal  # type: ignore
                below_upper = _ensure_array(
                    from_total_order(offsetf_total - 1, data.dtype)
                )
                below_upper[offsetf == 0] = -smallest_subnormal
                above_lower = _ensure_array(
                    from_total_order(offsetf_total + 1, data.dtype)
                )
                above_lower[offsetf == 0] = smallest_subnormal
            else:
                below_upper = _ensure_array(
                    from_total_order(offsetf_total - 1, data.dtype)
                )
                above_lower = _ensure_array(
                    from_total_order(offsetf_total + 1, data.dtype)
                )

        dataf = data.flatten()
        valid = (
            Interval.empty_like(dataf)
            .preserve_signed_nan(dataf, equal_nan=True)
            .preserve_non_nan(dataf)
        )

        # preserve zero-sign values exactly
        Lower(dataf) <= valid[dataf == offsetf] <= Upper(dataf)
        valid[dataf < offsetf] <= Upper(below_upper)
        Lower(above_lower) <= valid[dataf > offsetf]

        wheref: Literal[True] | np.ndarray[tuple[int], np.dtype[np.bool]] = (
            True if where is True else where.flatten()
        )

        return valid.preserve_only_where(wheref).into_union()

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

        return dict(kind=type(self).kind, offset=self._offset)
