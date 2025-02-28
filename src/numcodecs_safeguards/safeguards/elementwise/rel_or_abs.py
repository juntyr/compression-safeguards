"""
Relative (or absolute) error bound safeguard.
"""

__all__ = ["RelativeOrAbsoluteErrorBoundSafeguard"]

import numpy as np

from . import ElementwiseSafeguard, _as_bits


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
    eb_rel : float
        The positive relative error bound that is enforced by this safeguard.
        `eb_rel=0.02` corresponds to a 2% relative bound.
    eb_abs : int | float
        The positive absolute error bound that is enforced by this safeguard.
    equal_nan: bool
        Whether decoding a NaN value to a NaN value with a different bit
        pattern satisfies the error bound.
    """

    __slots__ = ("_eb_rel", "_eb_abs", "_equal_nan")
    _eb_rel: float
    _eb_abs: int | float
    _equal_nan: bool

    kind = "rel_or_abs"
    _priority = 0

    def __init__(self, eb_rel: float, eb_abs: int | float, *, equal_nan: bool = False):
        assert eb_rel > 0.0, "eb_rel must be positive"
        assert np.isfinite(eb_rel), "eb_rel must be finite"
        assert eb_abs > 0.0, "eb_abs must be positive"
        assert np.isfinite(eb_abs), "eb_abs must be finite"

        self._eb_rel = eb_rel
        self._eb_abs = eb_abs
        self._equal_nan = equal_nan

    @np.errstate(divide="ignore", over="ignore", under="ignore", invalid="ignore")
    def _check_elementwise(self, data: np.ndarray, decoded: np.ndarray) -> np.ndarray:
        """
        Check which elements in the `decoded` array satisfy the relative or the
        absolute error bound.

        Parameters
        ----------
        data : np.ndarray
            Data to be encoded.
        decoded : np.ndarray
            Decoded data.

        Returns
        -------
        ok : np.ndarray
            Per-element, `True` if the check succeeded for this element.
        """

        return (
            (np.abs(self._log(data) - self._log(decoded)) <= np.log(1.0 + self._eb_rel))
            | (np.abs(data - decoded) <= self._eb_abs)
            | (_as_bits(data) == _as_bits(decoded))
            | (self._equal_nan and (np.isnan(data) & np.isnan(decoded)))
        )

    @np.errstate(divide="ignore", over="ignore", under="ignore", invalid="ignore")
    def _compute_correction(
        self,
        data: np.ndarray,
        decoded: np.ndarray,
    ) -> np.ndarray:
        # remember which elements already passed the check
        already_correct = self.check_elementwise(data, decoded)

        log_eb_rel = np.log(1.0 + self._eb_rel)

        data_log = self._log(data)
        decoded_log = self._log(decoded)

        error_log = decoded_log - data_log
        correction_log = np.round(error_log / (log_eb_rel * 2.0)) * (log_eb_rel * 2.0)

        corrected_log = decoded_log - correction_log
        corrected = self._exp(corrected_log).astype(data.dtype)

        # don't apply corrections to already-passing elements
        corrected = np.where(already_correct, decoded, corrected)

        return np.where(
            self.check_elementwise(data, corrected),
            corrected,
            data,
        )

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

    def _log(self, x: np.ndarray) -> np.ndarray:
        a = 0.5 * self._eb_abs / (1.0 + self._eb_rel)

        return np.sign(x) * np.log(np.maximum(np.abs(x), a) / a)

    def _exp(self, x: np.ndarray) -> np.ndarray:
        a = 0.5 * self._eb_abs / (1.0 + self._eb_rel)

        return np.sign(x) * np.exp(np.abs(x)) * a
