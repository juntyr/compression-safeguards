__all__ = [
    "T",
    "F",
    "Fi",
    "J",
    "Ps",
    "Ps2",
    "Ns",
    "np_sndarray",
    "np_sndarray2",
    "Ci",
    "Es",
    "coerce_to_flat",
]

from collections.abc import Callable
from typing import TYPE_CHECKING, TypeAlias, TypeVar, TypeVarTuple

import numpy as np

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

Ps2 = TypeVar("Ps2", bound=tuple[int, ...], covariant=True)
""" Any pointwise array shape X.shape (covariant). """

Ns = TypeVar("Ns", bound=tuple[int, ...], covariant=True)
""" Any stencil neighbourhood array shape [*S.shape] (covariant). """

if TYPE_CHECKING:
    np_sndarray: TypeAlias = np.ndarray[tuple[Ps, *Ns], T]  # type: ignore
    """ Any stencil-extended [`np.ndarray[tuple[Ps, *Ns], T]`][numpy.ndarray]. """
else:
    # *TypeVar(bound=tuple) is not yet supported
    np_sndarray: TypeAlias = np.ndarray[tuple[Ps, Ns], T]  # type: ignore

if TYPE_CHECKING:
    np_sndarray2: TypeAlias = np.ndarray[tuple[*Ps2, *Ns], T]  # type: ignore
    """ Any stencil-extended [`np.ndarray[tuple[*Ps2, *Ns], T]`][numpy.ndarray]. """
else:
    # *TypeVar(bound=tuple), *TypeVar(bound=tuple) is not supported
    np_sndarray2: TypeAlias = np.ndarray[tuple[Ps2, Ns], T]  # type: ignore

Ci = TypeVar("Ci", bound=Callable)
""" Any callable type (invariant). """

# FIXME: actually bound the types to be Expr
# https://discuss.python.org/t/how-to-use-typevartuple/67502
Es = TypeVarTuple("Es")
""" Tuple of [`Expr`][...expr.abc.Expr]s. """


def coerce_to_flat(a: np_sndarray[Ps, tuple[()], T]) -> np.ndarray[tuple[Ps], T]:
    """
    Coerce `np.ndarray[tuple[Ps, *tuple[()]], T]` to `np.ndarray[tuple[Ps], T]`.
    """

    return a  # type: ignore
