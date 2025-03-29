"""
Relative (or absolute) error bound safeguard.
"""

__all__ = ["RelativeOrAbsoluteErrorBoundSafeguard"]

import numpy as np

from .abc import ElementwiseSafeguard
from .abs import _compute_safe_eb_abs_interval
from .decimal import _compute_safe_eb_rel_interval
from ...cast import to_float, as_bits, to_finite_float
from ...intervals import IntervalUnion, Lower, Upper


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

        data_float: np.ndarray = to_float(data)

        with np.errstate(
            divide="ignore", over="ignore", under="ignore", invalid="ignore"
        ):
            eb_abs: np.ndarray = to_finite_float(self._eb_abs, data_float.dtype)
        assert eb_abs >= 0.0

        with np.errstate(
            divide="ignore", over="ignore", under="ignore", invalid="ignore"
        ):
            eb_rel_multipler: np.ndarray = to_finite_float(
                self._eb_rel, data_float.dtype, map=lambda x: x + 1
            )
        assert eb_rel_multipler >= 1.0

        # compute the intervals for the absolute and relative error bounds
        valid_abs = _compute_safe_eb_abs_interval(
            data, data_float, eb_abs, equal_nan=self._equal_nan
        )
        valid_rel = _compute_safe_eb_rel_interval(
            data, data_float, eb_rel_multipler, equal_nan=self._equal_nan
        )

        # combine (union) the absolute and relative error bounds
        # we can union since the intervals overlap, at minimum at data
        valid = valid_abs
        Lower(np.minimum(valid_abs._lower, valid_rel._lower)) <= valid[
            np.isfinite(data.flatten())
        ] <= Upper(np.maximum(valid_abs._upper, valid_rel._upper))

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
