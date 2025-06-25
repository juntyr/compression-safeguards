"""
Implementations for the provided [`Safeguard`][compression_safeguards.safeguards.abc.Safeguard]s.
"""

__all__ = ["SafeguardKind"]

from enum import Enum

from .combinators.all import AllSafeguards
from .combinators.any import AnySafeguard
from .combinators.assume_safe import AssumeAlwaysSafeguard
from .combinators.select import SelectSafeguard
from .pointwise.eb import ErrorBoundSafeguard
from .pointwise.qoi.eb import PointwiseQuantityOfInterestErrorBoundSafeguard
from .pointwise.same import SameValueSafeguard
from .pointwise.sign import SignPreservingSafeguard
from .stencil.monotonicity import MonotonicityPreservingSafeguard
from .stencil.qoi.eb import StencilQuantityOfInterestErrorBoundSafeguard


class SafeguardKind(Enum):
    """
    Enumeration of all supported safeguards:
    """

    # same value
    same = SameValueSafeguard
    """Enforce that a special value is exactly preserved."""

    # sign
    sign = SignPreservingSafeguard
    """Enforce that the sign (-1, 0, +1) of each element is preserved."""

    # error bounds
    eb = ErrorBoundSafeguard
    """Enforce a pointwise error bound."""

    qoi_eb_pw = PointwiseQuantityOfInterestErrorBoundSafeguard
    """Enforce an error bound on a pointwise derived quantity of interest."""

    qoi_eb_stencil = StencilQuantityOfInterestErrorBoundSafeguard
    """Enforce an error bound on a derived quantity of interest over a data neighbourhood."""

    # monotonicity
    monotonicity = MonotonicityPreservingSafeguard
    """Enforce that monotonic sequences remain monotonic."""

    # logical combinators
    all = AllSafeguards
    """Enforce that all of the inner safeguards' guarantees are met."""

    any = AnySafeguard
    """Enforce that any one of the inner safeguards' guarantees are met."""

    assume_safe = AssumeAlwaysSafeguard
    """All elements are assumed to always be safe."""

    select = SelectSafeguard
    """Select, pointwise, which safeguard's guarantees to enforce."""
