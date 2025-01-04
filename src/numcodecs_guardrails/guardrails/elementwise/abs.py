__all__ = ["AbsoluteErrorBoundGuardrail"]

import numpy as np

from . import ElementwiseGuardrail, _as_bits


class AbsoluteErrorBoundGuardrail(ElementwiseGuardrail):
    __slots__ = ("_eb_abs",)
    _eb_abs: float

    kind = "abs"
    _priority = 0

    def __init__(self, eb_abs: float):
        assert eb_abs > 0.0, "eb_abs must be positive"
        assert np.isfinite(eb_abs), "eb_abs must be finite"

        self._eb_abs = eb_abs

    @np.errstate(divide="ignore", over="ignore", under="ignore", invalid="ignore")
    def check_elementwise(self, data: np.ndarray, decoded: np.ndarray) -> np.ndarray:
        """
        Check which elements in the `decoded` array satisfy the absolute error
        bound.

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

        return (np.abs(data - decoded) <= self._eb_abs) | (
            _as_bits(data) == _as_bits(decoded)
        )

    @np.errstate(divide="ignore", over="ignore", under="ignore", invalid="ignore")
    def _compute_correction(
        self,
        data: np.ndarray,
        decoded: np.ndarray,
    ) -> np.ndarray:
        error = decoded - data
        correction = (
            np.round(error / (self._eb_abs * 2.0)) * (self._eb_abs * 2.0)
        ).astype(data.dtype)
        corrected = decoded - correction

        return np.where(
            self.check_elementwise(data, corrected),
            corrected,
            data,
        )

    def get_config(self) -> dict:
        """
        Returns the configuration of the guardrail.

        Returns
        -------
        config : dict
            Configuration of the guardrail.
        """

        return dict(kind=type(self).kind, eb_abs=self._eb_abs)
