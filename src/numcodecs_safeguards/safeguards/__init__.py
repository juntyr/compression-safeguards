"""
Implementations for the provided safeguards.
"""

__all__ = ["Safeguards"]

from enum import Enum

from .elementwise.abs import AbsoluteErrorBoundSafeguard
from .elementwise.decimal import DecimalErrorBoundSafeguard
from .elementwise.ratio_or_abs import RatioOrAbsoluteErrorBoundSafeguard
from .elementwise.rel import RelativeErrorBoundSafeguard
from .elementwise.sign import SignPreservingSafeguard
from .elementwise.zero import ZeroIsZeroSafeguard
from .stencil.findiff.abs import (
    FiniteDifferenceAbsoluteErrorBoundSafeguard,
)
from .stencil.monotonicity import MonotonicityPreservingSafeguard


class Safeguards(Enum):
    """
    Enumeration of all supported safeguards:
    """

    # exact values
    zero = ZeroIsZeroSafeguard
    """Enforce that zero (or another constant) is exactly preserved."""

    # error bounds
    abs = AbsoluteErrorBoundSafeguard
    """Enforce an absolute error bound."""

    rel = RelativeErrorBoundSafeguard
    """Enforce a relative error bound."""

    ratio_or_abs = RatioOrAbsoluteErrorBoundSafeguard
    """Enforce a ratio error bound, fall back to an absolute error bound close to zero."""

    decimal = DecimalErrorBoundSafeguard
    """Enforce a decimal error bound."""

    # finite difference error bounds
    findiff_abs = FiniteDifferenceAbsoluteErrorBoundSafeguard
    """Enforce an absolute error bound for the finite differences."""

    # monotonicity
    monotonicity = MonotonicityPreservingSafeguard
    """Enforce that monotonic sequences remain monotonic."""

    # sign
    sign = SignPreservingSafeguard
    """Enforce that the sign (-1, 0, +1) of each element is preserved."""
