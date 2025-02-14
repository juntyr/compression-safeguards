__all__ = ["DecimalErrorBoundSafeguard"]

import numpy as np

from . import ElementwiseSafeguard, _as_bits


class DecimalErrorBoundSafeguard(ElementwiseSafeguard):
    __slots__ = ("_eb_decimal",)
    _eb_decimal: float

    kind = "decimal"
    _priority = 0

    def __init__(self, eb_decimal: float):
        assert eb_decimal > 0.0, "eb_decimal must be positive"
        assert np.isfinite(eb_decimal), "eb_decimal must be finite"

        self._eb_decimal = eb_decimal

    def check_elementwise(self, data: np.ndarray, decoded: np.ndarray) -> np.ndarray:
        """
        Check which elements have matching signs in the `data` and the
        `decoded` array.

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

        return (self._decimal_error(data, decoded) <= self._eb_decimal) | (
            _as_bits(data) == _as_bits(decoded)
        )

    @np.errstate(divide="ignore", over="ignore", under="ignore", invalid="ignore")
    def _compute_correction(
        self,
        data: np.ndarray,
        decoded: np.ndarray,
    ) -> np.ndarray:
        # if sign(data) == -sign(decoded), flip the sign
        decoded = decoded * (1 - (np.sign(data) == -np.sign(decoded)) * 2)

        # round the decimal error to the desired precision
        decimal_error = self._decimal_error(data, decoded)
        decimal_correction = np.round(decimal_error / (self._eb_decimal * 2.0)) * (
            self._eb_decimal * 2.0
        )

        # apply the decimal error correction
        decimal_corrected = np.log10(np.abs(decoded)) + decimal_correction * np.sign(
            np.abs(data) - np.abs(decoded)
        )
        corrected = np.power(10.0, decimal_corrected, dtype=data.dtype) * np.sign(data)
        corrected = corrected.astype(data.dtype)

        # fall back to the original data if the arithmetic evaluation of the
        #  error correction fails, e.g. for 0 != 0 or infinite or NaN values
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

        return dict(kind=type(self).kind, eb_decimal=self._eb_decimal)

    @np.errstate(divide="ignore", over="ignore", under="ignore", invalid="ignore")
    def _decimal_error(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        # 0               : if x == 0 and y == 0
        # inf             : if sign(x) != sign(y)
        # abs(log10(x/y)) : otherwise
        return np.where(
            np.sign(x) == np.sign(y),
            np.abs(np.log10(np.abs(x)) - np.log10(np.abs(y))),
            np.inf,
        )
