"""
Implementations for the provided [`Safeguard`][compression_safeguards.safeguards.abc.Safeguard]s.
"""

__all__ = ["SafeguardKind"]

from enum import Enum

from ..utils.error import TypeCheckError, ctx, lookup_enum_or_raise
from ..utils.typing import JSON
from .abc import Safeguard
from .combinators.all import AllSafeguards
from .combinators.any import AnySafeguard
from .combinators.assume_safe import AssumeAlwaysSafeguard
from .combinators.select import SelectSafeguard
from .pointwise.eb import ErrorBoundSafeguard
from .pointwise.qoi.eb import PointwiseQuantityOfInterestErrorBoundSafeguard
from .pointwise.same import SameValueSafeguard
from .pointwise.sign import SignPreservingSafeguard
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

    # logical combinators
    all = AllSafeguards
    """Enforce that all of the inner safeguards' guarantees are met."""

    any = AnySafeguard
    """Enforce that any one of the inner safeguards' guarantees are met."""

    assume_safe = AssumeAlwaysSafeguard
    """All elements are assumed to always be safe."""

    select = SelectSafeguard
    """Select, pointwise, which safeguard's guarantees to enforce."""

    @staticmethod
    def from_config(config: dict[str, JSON]) -> Safeguard:
        """
        Instantiate a safeguard from a configuration [`dict`][dict].

        The `config` must contain the safeguard's `kind`.

        Parameters
        ----------
        config : dict[str, JSON]
            Configuration of the safeguard.

        Returns
        -------
        safeguard : Safeguard
            Instantiated safeguard.

        Raises
        ------
        ValueError
            if the `config` does not contain the safeguard `kind` or the
            `kind` is unknown.
        TypeCheckError
            if the safeguard `kind` is not a [`str`][str]ing.
        ...
            if instantiating the safeguard raises an exception.
        """

        if "kind" not in config:
            raise ValueError("missing safeguard `kind`") | ctx

        kind = config["kind"]

        with ctx.parameter("kind"):
            TypeCheckError.check_instance_or_raise(kind, str)
            safeguard: type[Safeguard] = lookup_enum_or_raise(
                SafeguardKind,
                kind,  # type: ignore
            ).value

        return safeguard.from_config({p: v for p, v in config.items() if p != "kind"})
