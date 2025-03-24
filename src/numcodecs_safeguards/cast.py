"""
Utility functions to cast arrays to floating point and binary representation.
"""

from typing import Any, TypeVar

import numpy as np

T = TypeVar("T", bound=np.dtype)
""" Any numpy [`dtype`][numpy.dtype] type variable. """
F = TypeVar("F", bound=np.dtype)
""" Any numpy [`floating`][numpy.floating] dtype type variable. """
S = TypeVar("S", bound=tuple[int, ...])
""" Any array shape. """


def to_float(x: np.ndarray[S, T]) -> np.ndarray[S, F]:
    """
    Losslessly convert the array `x` to floating point.

    Floating point arrays are passed through, integer arrays are cast to a
    dtype that can represent all integer values without loss in precision.

    Parameters
    ----------
    x : np.ndarray[S, T]
        The array to cast.

    Returns
    -------
    cast : np.ndarray[S, F]
        The cast array with a floating dtype.
    """

    if np.issubdtype(x.dtype, np.floating):
        return x  # type: ignore

    ftype = {
        np.dtype(np.int8): np.float16,
        np.dtype(np.int16): np.float32,
        np.dtype(np.int32): np.float64,
        np.dtype(np.int64): np.float128,
        np.dtype(np.uint8): np.float16,
        np.dtype(np.uint16): np.float32,
        np.dtype(np.uint32): np.float64,
        np.dtype(np.uint64): np.float128,
    }[x.dtype]

    return x.astype(ftype)  # type: ignore


def from_float(x: np.ndarray[S, F], dtype: T) -> np.ndarray[S, T]:
    """
    Reverses the cast of the array `x`, using the
    [`to_float`][numcodecs_safeguards.cast.to_float], back to the original
    `dtype`.

    If the original `dtype` was integer, the rounding conversion is lossy.
    Infinite values are clamped to the minimum/maximum integer values.

    Parameters
    ----------
    x : np.ndarray[S, F]
        The array to un-cast.

    Returns
    -------
    cast : np.ndarray[S, T]
        The un-cast array with the original `dtype`.
    """

    if x.dtype == dtype:
        return x  # type: ignore

    info = np.iinfo(dtype)
    imin, imax = np.array(info.min, dtype=dtype), np.array(info.max, dtype=dtype)

    with np.errstate(divide="ignore", over="ignore", under="ignore", invalid="ignore"):
        return np.where(x < imin, imin, np.where(x <= imax, x.astype(dtype), imax))  # type: ignore


def as_bits(a: np.ndarray[S, T], *, kind: str = "u") -> np.ndarray[S, Any]:
    """
    Reinterprets the array `a` to its binary representation.

    Parameters
    ----------
    a : np.ndarray[S, T]
        The array to reinterpret as binary.
    kind : str, optional
        The kind of binary dtype, e.g. `"u"` or `"i"`.

    Returns
    -------
    binary : np.ndarray[S, Any]
        The binary representation of the array `a`.
    """

    return a.view(a.dtype.str.replace("f", kind).replace("i", kind).replace("u", kind))  # type: ignore
