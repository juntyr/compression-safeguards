"""
Decimal error bound safeguard.
"""

__all__ = ["DecimalErrorBoundSafeguard"]

import numpy as np

from . import ElementwiseSafeguard, _as_bits


class DecimalErrorBoundSafeguard(ElementwiseSafeguard):
    r"""
    The `DecimalErrorBoundSafeguard` guarantees that the elementwise decimal
    error is less than or equal to the provided bound `eb_decimal`.

    The decimal error quantifies the orders of magnitude that the lossy-decoded
    value $\hat{x}$ is away from the original value $x$. It is defined as
    follows[^1] [^2]:

    \[
        \text{decimal error} = \begin{cases}
            0 & \quad \text{if } x = \hat{x} = 0 \\
            \inf & \quad \text{if } \text{sign}(x) \neq \text{sign}(\hat{x}) \\
            \left| \log_{10}{\left( \frac{x}{\hat{x}} \right)} \right| & \quad \text{otherwise}
        \end{cases}
    \]

    The decimal error is defined to be infinite if the signs of the data and
    decoded data do not match. Since the `eb_decimal` error bound must be
    finite, the `DecimalErrorBoundSafeguard` also guarantees that the sign of
    each decode value matches the sign of each original value and that a
    decoded value is zero if and only if it is zero in the original data.

    In cases where the arithmetic evaluation of the error bound not well-
    defined, e.g. for infinite or NaN values, producing the exact same
    bitpattern is defined to satisfy the error bound.
    
    [^1]: Gustafson, J. L., & Yonemoto, I. T. (2017). Beating Floating Point at
        its Own Game: Posit Arithmetic. *Supercomputing Frontiers and
        Innovations*, 4(2). Available from:
        [doi:10.14529/jsfi170206](https://doi.org/10.14529/jsfi170206).
    
    [^2]: Klöwer, M., Düben, P. D., & Palmer, T. N. (2019). Posits as an
        alternative to floats for weather and climate models. *CoNGA'19:
        Proceedings of the Conference for Next Generation Arithmetic 2019*, 1–8.
        Available from:
        [doi:10.1145/3316279.3316281](https://doi.org/10.1145/3316279.3316281).

    Parameters
    ----------
    eb_decimal : float
        The positive decimal error bound that is enforced by this safeguard.
        `eb_decimal=1.0` corresponds to a 100% relative error bound.
    """

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
        corrected = np.power(10.0, decimal_corrected) * np.sign(data)
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
            np.abs(np.log10(x / y)),
            np.inf,
        )
