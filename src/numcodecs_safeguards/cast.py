"""
Utility functions to cast arrays to floating point, binary, and total-order representations.
"""

__all__ = [
    "to_float",
    "to_finite_float",
    "from_float",
    "as_bits",
    "to_total_order",
    "from_total_order",
]

from typing import Any, Callable, TypeVar

import numpy as np

T = TypeVar("T", bound=np.dtype)
""" Any numpy [`dtype`][numpy.dtype] type variable. """
F = TypeVar("F", bound=np.dtype)
""" Any numpy [`floating`][numpy.floating] dtype type variable. """
U = TypeVar("U", bound=np.dtype)
""" Any numpy [`unsigned`][numpy.unsigned] dtype type variable. """
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
        The array to convert.

    Returns
    -------
    converted : np.ndarray[S, F]
        The converted array with a floating dtype.
    """

    if np.issubdtype(x.dtype, np.floating):
        return x  # type: ignore

    ftype = {
        np.dtype(np.int8): np.float16,
        np.dtype(np.int16): np.float32,
        np.dtype(np.int32): np.float64,
        np.dtype(np.int64): _float128,
        np.dtype(np.uint8): np.float16,
        np.dtype(np.uint16): np.float32,
        np.dtype(np.uint32): np.float64,
        np.dtype(np.uint64): _float128,
    }[x.dtype]

    return x.astype(ftype)  # type: ignore


def to_finite_float(
    x: int | float | np.ndarray[S, T],
    dtype: F,
    *,
    map: None | Callable[[np.ndarray[S, F]], np.ndarray[S, F]] = None,
) -> np.ndarray[S, F]:
    """
    Convert `x` to the floating-point dtype `F` and apply an optional `map`ping function.

    The result is clamped between the minimum and maximum floating point values
    to guarantee that it is finite.

    The dtype `F` should come from a prior use of the
    [`to_float`][numcodecs_safeguards.cast.to_float] helper function.

    Parameters
    ----------
    x : int | float | np.ndarray
        The value or array to convert.
    dtype : np.dtype
        The floating-point dtype to convert `x` to.
    map : None | Callable
        The mapping function to apply to `x`.

    Returns
    -------
    converted : np.ndarray[tuple[int, ...], F]
        The converted value or array with `dtype`.
    """

    xf: np.ndarray[S, F] = np.array(x).astype(dtype)  # type: ignore

    if map is not None:
        xf = np.array(map(xf)).astype(dtype)  # type: ignore

    if np.dtype(dtype) == _float128:
        minv, maxv = _float128_min, _float128_max
    else:
        minv, maxv = np.finfo(dtype).min, np.finfo(dtype).max

    return np.maximum(minv, np.minimum(xf, maxv))  # type: ignore


def from_float(x: np.ndarray[S, F], dtype: T) -> np.ndarray[S, T]:
    """
    Reverses the conversion of the array `x`, using the
    [`to_float`][numcodecs_safeguards.cast.to_float], back to the original
    `dtype`.

    If the original `dtype` was integer, the rounding conversion is lossy.
    Infinite values are clamped to the minimum/maximum integer values.

    Parameters
    ----------
    x : np.ndarray[S, F]
        The array to re-convert.
    dtype : np.dtype
        The original dtype.

    Returns
    -------
    converted : np.ndarray[S, T]
        The re-converted array with the original `dtype`.
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


def to_total_order(a: np.ndarray[S, T]) -> np.ndarray[S, U]:
    """
    Reinterprets the array `a` to its total-order unsigned binary
    representation.

    In their total-order representation, the smallest value is mapped to
    unsigned zero, and the largest value is mapped to the largest unsigned
    value.

    For floating point values, this implementation is based on Michael Herf's
    `FloatFlip` function, see <http://stereopsis.com/radix.html>.

    Parameters
    ----------
    a : np.ndarray[S, T]
        The array to reinterpret to its total-order.

    Returns
    -------
    ordered : np.ndarray[S, U]
        The total-order unsigned binary representation of the array `a`.
    """

    if np.issubdtype(a.dtype, np.unsignedinteger):
        return a  # type: ignore

    utype = a.dtype.str.replace("i", "u").replace("f", "u")

    if np.issubdtype(a.dtype, np.signedinteger):
        return a.view(utype) + np.array(np.iinfo(a.dtype).max, dtype=utype) + 1

    if not np.issubdtype(a.dtype, np.floating):
        raise TypeError(f"unsupported interval type {a.dtype}")

    itype = a.dtype.str.replace("f", "i")
    bits = np.iinfo(utype).bits

    mask = (-((a.view(dtype=utype) >> (bits - 1)).view(dtype=itype))).view(
        dtype=utype
    ) | (np.array(1, dtype=utype) << (bits - 1))

    return a.view(dtype=utype) ^ mask


def from_total_order(a: np.ndarray[S, U], dtype: T) -> np.ndarray[S, T]:
    """
    Reverses the reinterpretation of the array `a` back from total-order
    unsigned binary to the provided `dtype`.

    For floating point values, this implementation is based on Michael Herf's
    `IFloatFlip` function, see <http://stereopsis.com/radix.html>.

    Parameters
    ----------
    a : np.ndarray[S, U]
        The array to reverse-reinterpret back from its total-order.

    Returns
    -------
    array : np.ndarray[S, T]
        The array with its original `dtype`.
    """

    assert np.issubdtype(a.dtype, np.unsignedinteger)

    if np.issubdtype(dtype, np.unsignedinteger):
        return a  # type: ignore

    if np.issubdtype(dtype, np.signedinteger):
        return a.view(dtype) + np.iinfo(dtype).max + 1  # type: ignore

    if not np.issubdtype(dtype, np.floating):
        raise TypeError(f"unsupported interval type {dtype}")

    utype = dtype.str.replace("f", "u")
    itype = dtype.str.replace("f", "i")
    bits = np.iinfo(utype).bits

    mask = ((a >> (bits - 1)).view(dtype=itype) - 1).view(dtype=utype) | (
        np.array(1, dtype=utype) << (bits - 1)
    )

    return (a ^ mask).view(dtype=dtype)


try:
    _float128: np.dtype = np.dtype(np.float128)
    assert np.finfo(np.float128).bits == 128
    _float128_min = np.finfo(np.float128).min
    _float128_max = np.finfo(np.float128).max
except (AttributeError, AssertionError):
    try:
        import numpy_quaddtype

        _float128 = numpy_quaddtype.QuadPrecDType("sleef")
        _float128_min = numpy_quaddtype.min_value
        _float128_max = numpy_quaddtype.max_value
    except ImportError:
        raise TypeError("""
numcodecs_safeguards requires float128 support:
- numpy.float128 either does not exist is does not offer 128 bit precision
- numpy_quaddtype is not installed
""") from None
