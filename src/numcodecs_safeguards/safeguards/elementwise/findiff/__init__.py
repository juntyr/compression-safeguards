"""
Implementations for the provided elementwise safeguards on finite differences.
"""

__all__ = ["FiniteDifference"]

from enum import auto, Enum
from fractions import Fraction


class FiniteDifference(Enum):
    """
    Different types of finite difference that can be safeguarded by the
    [`FiniteDifferenceAbsoluteErrorBoundSafeguard`][numcodecs_safeguards.safeguards.elementwise.findiff.abs.FiniteDifferenceAbsoluteErrorBoundSafeguard].
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


def _finite_difference_offsets(
    type: FiniteDifference,
    order: int,
    accuracy: int,
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

    c1 = Fraction(1)

    for n in range(1, N + 1):
        c2 = Fraction(1)
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
