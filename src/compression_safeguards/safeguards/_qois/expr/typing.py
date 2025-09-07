__all__ = ["F", "Fi", "Ps", "PsI", "Ns", "Ci", "Es"]

from collections.abc import Callable
from typing import TypeVar

import numpy as np
from typing_extensions import TypeVarTuple  # MSPV 3.11

F = TypeVar("F", bound=np.floating, covariant=True)
""" Any numpy [`floating`][numpy.floating]-point data type (covariant). """

Fi = TypeVar("Fi", bound=np.floating)
""" Any numpy [`floating`][numpy.floating]-point data type (invariant). """

Ps = TypeVar("Ps", bound=tuple[int, ...], covariant=True)
""" Any pointwise array shape [...X] (covariant). """

PsI = TypeVar("PsI", bound=tuple[int, ...])
""" Any pointwise array shape [...X] (invariant). """

Ns = TypeVar("Ns", bound=tuple[int, ...], covariant=True)
""" Any stencil neighbourhood array shape [...X, ...S] (covariant). """

Ci = TypeVar("Ci", bound=Callable)
""" Any callable type (invariant). """

# FIXME: actually bound the types to be Expr
# https://discuss.python.org/t/how-to-use-typevartuple/67502
Es = TypeVarTuple("Es")
""" Tuple of [`Expr`][compression_safeguards.safeguards._qois.expr.abc.Expr]s. """
