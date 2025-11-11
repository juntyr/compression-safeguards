__all__ = ["T", "F", "Fi", "J", "Ps", "Ns", "np_sndarray", "Ci", "Es"]

from collections.abc import Callable
from typing import TYPE_CHECKING, TypeAlias, TypeVar

import numpy as np
from typing_extensions import (
    TypeVarTuple,  # MSPV 3.11
    Unpack,  # MSPV 3.11
)

T = TypeVar("T", bound=np.dtype[np.generic], covariant=True)
""" Any numpy [`dtype`][numpy.dtype] (covariant). """

F = TypeVar("F", bound=np.floating, covariant=True)
""" Any numpy [`floating`][numpy.floating]-point data type (covariant). """

Fi = TypeVar("Fi", bound=np.floating)
""" Any numpy [`floating`][numpy.floating]-point data type (invariant). """

J = TypeVar("J", bound=int, covariant=True)
""" Any [`int`][int] (covariant). """

Ps = TypeVar("Ps", bound=int, covariant=True)
""" Any flattened pointwise array shape [X.size] (covariant). """

Ns = TypeVar("Ns", bound=tuple[int, ...], covariant=True)
""" Any stencil neighbourhood array shape [*S.shape] (covariant). """

if TYPE_CHECKING:
    np_sndarray: TypeAlias = np.ndarray[tuple[Ps, Unpack[Ns]], T]  # type: ignore
    """ Any stencil-extended [`np.ndarray[tuple[Ps, Unpack[Ns]], T]`][numpy.ndarray]. """
else:
    # Unpack[TypeVar(bound=tuple)] is not yet supported
    np_sndarray: TypeAlias = np.ndarray[tuple[Ps, Ns], T]  # type: ignore

Ci = TypeVar("Ci", bound=Callable)
""" Any callable type (invariant). """

# FIXME: actually bound the types to be Expr
# https://discuss.python.org/t/how-to-use-typevartuple/67502
Es = TypeVarTuple("Es")
""" Tuple of [`Expr`][...abc.Expr]s. """
