"""
Error bound safeguard.
"""

__all__ = ["ErrorBoundSafeguard"]

from collections.abc import Set

import numpy as np

from ...utils.bindings import Bindings, Parameter
from ...utils.cast import (
    _isfinite,
    _isinf,
    _isnan,
    as_bits,
    to_float,
)
from ...utils.intervals import Interval, IntervalUnion, Lower, Upper
from ...utils.typing import S, T
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
    set to [`True`][True], decoding a NaN value to a NaN value with a different
    bit pattern also satisfies the error bound. If `equal_nan` is set to
    [`False`][False], NaN values are also preserved with the same bit pattern.

    Parameters
    ----------
    type : str | ErrorBound
        The type of error bound that is enforced by this safeguard.
    eb : int | float | str | Parameter
        The value of or late-bound parameter name for the error bound that is
        enforced by this safeguard.
    equal_nan: bool
        Whether decoding a NaN value to a NaN value with a different bit
        pattern satisfies the error bound.
    """

    __slots__ = ("_type", "_eb", "_equal_nan")
    _type: ErrorBound
    _eb: int | float | Parameter
    _equal_nan: bool

    kind = "eb"

    def __init__(
        self,
        type: str | ErrorBound,
        eb: int | float | str | Parameter,
        *,
        equal_nan: bool = False,
    ):
        self._type = type if isinstance(type, ErrorBound) else ErrorBound[type]

        if isinstance(eb, Parameter):
            self._eb = eb
        elif isinstance(eb, str):
            self._eb = Parameter(eb)
        else:
            _check_error_bound(self._type, eb)
            self._eb = eb

        self._equal_nan = equal_nan

    @property
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
    def check_pointwise(
        self,
        data: np.ndarray[S, np.dtype[T]],
        decoded: np.ndarray[S, np.dtype[T]],
        *,
        late_bound: Bindings,
    ) -> np.ndarray[S, np.dtype[np.bool]]:
        """
        Check which elements in the `decoded` array satisfy the error bound.

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

        data_float: np.ndarray = to_float(data)
        decoded_float: np.ndarray = to_float(decoded)

        eb = (
            late_bound.resolve_ndarray(
                self._eb,
                data_float.shape,
                data_float.dtype,
            )
            if isinstance(self._eb, Parameter)
            else self._eb
        )
        _check_error_bound(self._type, eb)

        finite_ok = _compute_finite_absolute_error(
            self._type, data_float, decoded_float
        ) <= _compute_finite_absolute_error_bound(self._type, eb, data_float)

        # bitwise equality for inf and NaNs (unless equal_nan)
        same_bits = as_bits(data) == as_bits(decoded)
        both_nan = self._equal_nan and (_isnan(data) & _isnan(decoded))

        ok = np.where(
            _isfinite(data),
            finite_ok,
            np.where(
                _isinf(data),
                same_bits,
                both_nan if self._equal_nan else same_bits,
            ),
        )

        return ok  # type: ignore

    def compute_safe_intervals(
        self,
        data: np.ndarray[S, np.dtype[T]],
        *,
        late_bound: Bindings,
    ) -> IntervalUnion[T, int, int]:
        """
        Compute the intervals in which the error bound is upheld with respect
        to the `data`.

        Parameters
        ----------
        data : np.ndarray
            Data for which the safe intervals should be computed.
        late_bound : Bindings
            Bindings for late-bound parameters, including for this safeguard.

        Returns
        -------
        intervals : IntervalUnion
            Union of intervals in which the error bound is upheld.
        """

        dataf = data.flatten()

        valid = (
            Interval.empty_like(dataf)
            .preserve_inf(dataf)
            .preserve_signed_nan(dataf, equal_nan=self._equal_nan)
        )

        data_float: np.ndarray = to_float(data)

        eb = (
            late_bound.resolve_ndarray(
                self._eb,
                data_float.shape,
                data_float.dtype,
            )
            if isinstance(self._eb, Parameter)
            else self._eb
        )
        _check_error_bound(self._type, eb)

        lower, upper = _apply_finite_error_bound(self._type, eb, data, to_float(data))

        Lower(lower.flatten()) <= valid[_isfinite(dataf)] <= Upper(upper.flatten())

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
            type=self._type.name,
            eb=self._eb,
            equal_nan=self._equal_nan,
        )
