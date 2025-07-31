__all__ = ["F", "Ps", "PsI", "Ns"]

from typing import TypeVar

from .....utils.typing import F

Ps = TypeVar("Ps", bound=tuple[int, ...], covariant=True)
""" Any pointwise array shape [...X] (covariant). """

PsI = TypeVar("PsI", bound=tuple[int, ...])
""" Any pointwise array shape [...X] (invariant). """

Ns = TypeVar("Ns", bound=tuple[int, ...], covariant=True)
""" Any stencil neighbourhood array shape [...X, ...S] (covariant). """
