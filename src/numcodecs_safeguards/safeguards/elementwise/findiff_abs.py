"""
Finite difference absolute error bound safeguard.
"""

__all__ = ["FiniteDifferenceAbsoluteErrorBoundSafeguard"]

from enum import auto, Enum
from fractions import Fraction

import numpy as np

from . import ElementwiseSafeguard, _as_bits


class FiniteDifference(Enum):
    """
    Different types of finite difference that can be safeguarded by the
    [`FiniteDifferenceAbsoluteErrorBoundSafeguard`][numcodecs_safeguards.safeguards.elementwise.abs_findiff.FiniteDifferenceAbsoluteErrorBoundSafeguard].
    """

    central = auto()
    r"""
    Central finite difference, computed over the indices
    $\{i-k; \ldots; i; \ldots; i+k\}$.
    """

    forward = auto()
    r"""
    Forward finite difference, computed over the indices $\{i; \ldots; i+k\}$.
    """

    backwards = auto()
    r"""
    Backward finite difference, computed over the indices $\{i-k; \ldots; i\}$.
    """


class FiniteDifferenceAbsoluteErrorBoundSafeguard(ElementwiseSafeguard):
    """
    The `FiniteDifferenceAbsoluteErrorBoundSafeguard` guarantees that the
    elementwise absolute error of the finite-difference-approximated derivative
    is less than or equal to the provided bound `eb_abs`.

    If the finite difference for an element evaluates to an infinite value,
    this safeguard guarantees that the finite difference on the decoded value
    produces the exact same infinite value. For a NaN finite difference, this
    safeguard guarantees that the finite difference on the decoded value is
    also NaN, but does not guarantee that it has the same bitpattern.

    Parameters
    ----------
    order : int
        The non-negative order of the derivative that is approximayed by a
        finite difference.
    accuracy : int
        The positive order of accuracy of the finite difference approximation.

        The order of accuracy must be even for a central finite difference.
    type : FiniteDifference
        The type of finite difference.
    eb_abs : float
        The positive absolute error bound that is enforced by this safeguard.
    """

    __slots__ = ("_order", "_accuracy", "_type", "_eb_abs", "_eb_abs_impl")
    _order: int
    _accuracy: int
    _type: FiniteDifference
    _eb_abs: float
    _eb_abs_impl: float

    kind = "findiff_abs"
    _priority = 0

    def __init__(
        self, order: int, accuracy: int, type: FiniteDifference, eb_abs: float
    ):
        assert order >= 0, "order must be non-negative"
        assert accuracy > 0, "accuracy must be positive"

        if type == FiniteDifference.central:
            assert accuracy % 2 == 0, (
                "accuracy must be even for a central finite difference"
            )

        assert eb_abs > 0.0, "eb_abs must be positive"
        assert np.isfinite(eb_abs), "eb_abs must be finite"

        self._order = order
        self._accuracy = accuracy
        self._type = type
        self._eb_abs = eb_abs

        offsets = _finite_difference_offsets(order, accuracy, type)
        coefficients = _finite_difference_coefficients(order, offsets)

        self._eb_abs_impl = eb_abs / sum(abs(c) for c in coefficients)

    @np.errstate(divide="ignore", over="ignore", under="ignore", invalid="ignore")
    def check_elementwise(self, data: np.ndarray, decoded: np.ndarray) -> np.ndarray:
        """
        Check for which elements in the `decoded` array the finite differences
        satisfy the absolute error bound.

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
            (np.abs(data - decoded) <= self._eb_abs_impl)
            | (_as_bits(data) == _as_bits(decoded))
            | (np.isnan(data) == np.isnan(decoded))
        )

    @np.errstate(divide="ignore", over="ignore", under="ignore", invalid="ignore")
    def _compute_correction(
        self,
        data: np.ndarray,
        decoded: np.ndarray,
    ) -> np.ndarray:
        error = decoded - data
        correction = (
            np.round(error / (self._eb_abs_impl * 2.0)) * (self._eb_abs_impl * 2.0)
        ).astype(data.dtype)
        corrected = decoded - correction

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
            order=self._order,
            accuracy=self._accuracy,
            type=self._type,
            eb_abs=self._eb_abs,
        )


def _finite_difference_offsets(
    order: int,
    accuracy: int,
    type: FiniteDifference,
) -> list[int]:
    match type:
        case FiniteDifference.central:
            noffsets = order + (order % 2) - 1 + accuracy
            p = (noffsets - 1) // 2
            return [0] + [j for i in range(1, p + 1) for j in [i, -i]]
        case FiniteDifference.forward:
            return [i for i in range(order + accuracy)]
        case FiniteDifference.backwards:
            return [-i for i in range(order + accuracy)]


def _finite_difference_coefficients(
    order: int,
    offsets: list[int],
) -> list[Fraction]:
    """
    Finite difference coefficient algorithm from:

    Fornberg, B. (1988). Generation of finite difference formulas on arbitrarily
    spaced grids. Mathematics of Computation, 51(184), 699-706. Available from:
    https://doi.org/10.1090/s0025-5718-1988-0935077-0.
    """

    # x0 = 0
    M = order
    a = [Fraction(o) for o in offsets]
    N = len(a) - 1

    coeffs = {
        (0, 0, 0): Fraction(1),
    }

    c1 = 1

    for n in range(1, N + 1):
        c2 = 1
        for v in range(0, n):
            c3 = a[n] - a[v]
            c2 *= c3
            if n <= M:
                coeffs[(n, n - 1, v)] = Fraction(0)
            for m in range(0, min(n, M) + 1):
                if m > 0:
                    coeffs[(m, n, v)] = (
                        (a[n] * coeffs[(m, n - 1, v)]) - (m * coeffs[(m - 1, n - 1, v)])
                    ) / c3
                else:
                    coeffs[(m, n, v)] = (a[n] * coeffs[(m, n - 1, v)]) / c3
        for m in range(0, min(n, M) + 1):
            if m > 0:
                coeffs[(m, n, n)] = (c1 / c2) * (
                    (m * coeffs[(m - 1, n - 1, n - 1)])
                    - (a[n - 1] * coeffs[(m, n - 1, n - 1)])
                )
            else:
                coeffs[(m, n, n)] = -(c1 / c2) * (a[n - 1] * coeffs[(m, n - 1, n - 1)])
        c1 = c2

    return [coeffs[M, N, v] for v in range(0, N + 1)]
