__all__ = ["F", "Fi", "Ps", "PsI", "Ns", "Ci"]

from typing import Callable, TypeVar

import numpy as np

F = TypeVar("F", bound=np.floating, covariant=True)
""" Any numpy [`floating`][numpy.floating] data type (covariant). """

Fi = TypeVar("Fi", bound=np.floating)
""" Any numpy [`floating`][numpy.floating] data type (invariant). """

Ps = TypeVar("Ps", bound=tuple[int, ...], covariant=True)
""" Any pointwise array shape [...X] (covariant). """

PsI = TypeVar("PsI", bound=tuple[int, ...])
""" Any pointwise array shape [...X] (invariant). """

Ns = TypeVar("Ns", bound=tuple[int, ...], covariant=True)
""" Any stencil neighbourhood array shape [...X, ...S] (covariant). """

Ci = TypeVar("Ci", bound=Callable)
""" Any callable type (invariant). """
