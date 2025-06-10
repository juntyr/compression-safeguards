"""
Implementations for the provided [`Safeguard`][compression_safeguards.safeguards.abc.Safeguard]s.
"""

__all__ = ["SafeguardKind"]

from enum import Enum

from .combinators.all import AllSafeguards
from .combinators.any import AnySafeguard
from .combinators.safe import AlwaysSafeguard
from .combinators.select import SelectSafeguard
from .pointwise.eb import ErrorBoundSafeguard
from .pointwise.qoi.abs import PointwiseQuantityOfInterestAbsoluteErrorBoundSafeguard
from .pointwise.sign import SignPreservingSafeguard
from .pointwise.zero import ZeroIsZeroSafeguard
from .stencil.monotonicity import MonotonicityPreservingSafeguard
from .stencil.qoi.abs import StencilQuantityOfInterestAbsoluteErrorBoundSafeguard


class SafeguardKind(Enum):
    """
    Enumeration of all supported safeguards:
    """

    # exact values
    zero = ZeroIsZeroSafeguard
    """Enforce that zero (or another constant) is exactly preserved."""

    # sign
    sign = SignPreservingSafeguard
    """Enforce that the sign (-1, 0, +1) of each element is preserved."""

    # pointwise error bounds
    eb = ErrorBoundSafeguard
    """Enforce an absolute / relative / ratio (decimal) error bound."""

    # quantity of interest error bounds
    qoi_abs_pw = PointwiseQuantityOfInterestAbsoluteErrorBoundSafeguard
    """Enforce an absolute error bound on a pointwise derived quantity of interest."""

    qoi_abs_stencil = StencilQuantityOfInterestAbsoluteErrorBoundSafeguard
    """Enforce an absolute error bound on a derived quantity of interest over a data neighbourhood."""

    # monotonicity
    monotonicity = MonotonicityPreservingSafeguard
    """Enforce that monotonic sequences remain monotonic."""

    # logical combinators
    all = AllSafeguards
    """Enforce that all of the inner safeguards' guarantees are met."""

    any = AnySafeguard
    """Enforce that any one of the inner safeguards' guarantees are met."""

    safe = AlwaysSafeguard
    """All elements are always safe."""

    select = SelectSafeguard
    """Select, pointwise, which safeguard's guarantees to enforce."""
