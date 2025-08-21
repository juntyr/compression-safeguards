"""
Utility functions to cast arrays to floating point, binary, and total-order representations.
"""

__all__ = [
    "to_float",
    "from_float",
    "as_bits",
    "to_total_order",
    "from_total_order",
    "lossless_cast",
    "saturating_finite_float_cast",
]

from typing import Any

import numpy as np

from ._compat import _isfinite, _isnan, _nan_to_zero_inf_to_finite, _rint
from ._float128 import _float128_dtype
from .typing import F, S, T, U


def to_float(x: np.ndarray[S, np.dtype[T]]) -> np.ndarray[S, np.dtype[F]]:
    """
    Losslessly convert the array `x` to floating point.

    Floating point arrays are passed through, integer arrays are cast to a
    dtype that can represent all integer values without loss in precision.

    Parameters
    ----------
    x : np.ndarray[S, np.dtype[T]]
        The array to convert.

    Returns
    -------
    converted : np.ndarray[S, np.dtype[F]]
        The converted array with a floating dtype.
    """

    if np.issubdtype(x.dtype, np.floating):
        return x  # type: ignore

    ftype = {
        np.dtype(np.int8): np.float16,
        np.dtype(np.int16): np.float32,
        np.dtype(np.int32): np.float64,
        np.dtype(np.int64): _float128_dtype,
        np.dtype(np.uint8): np.float16,
        np.dtype(np.uint16): np.float32,
        np.dtype(np.uint32): np.float64,
        np.dtype(np.uint64): _float128_dtype,
    }[x.dtype]

    # lossless cast from integer to floating point with a sufficiently large
    #  mantissa
    return x.astype(ftype, casting="safe")  # type: ignore


def from_float(
    x: np.ndarray[S, np.dtype[F]], dtype: np.dtype[T]
) -> np.ndarray[S, np.dtype[T]]:
    """
    Reverses the conversion of the array `x`, using the
    [`to_float`][compression_safeguards.utils.cast.to_float], back to the
    original `dtype`.

    If the original `dtype` was integer, the rounding conversion is lossy.
    Infinite values are clamped to the minimum/maximum integer values.

    Parameters
    ----------
    x : np.ndarray[S, np.dtype[F]]
        The array to re-convert.
    dtype : np.dtype
        The original dtype.

    Returns
    -------
    converted : np.ndarray[S, np.dtype[T]]
        The re-converted array with the original `dtype`.
    """

    x = np.array(x)

    if x.dtype == dtype:
        return x  # type: ignore

    info = np.iinfo(dtype)  # type: ignore
    imin, imax = np.array(info.min, dtype=dtype), np.array(info.max, dtype=dtype)

    with np.errstate(divide="ignore", over="ignore", under="ignore", invalid="ignore"):
        # lossy cast from floating point to integer
        # round first with rint (round to nearest, ties to nearest even)
        return np.where(
            x < imin,
            imin,
            np.where(x <= imax, _rint(x).astype(dtype, casting="unsafe"), imax),
        )  # type: ignore


def as_bits(
    a: np.ndarray[S, np.dtype[T]], *, kind: str = "u"
) -> np.ndarray[S, np.dtype[Any]]:
    """
    Reinterprets the array `a` to its binary representation.

    Parameters
    ----------
    a : np.ndarray[S, np.dtype[T]]
        The array to reinterpret as binary.
    kind : str, optional
        The kind of binary dtype, e.g. `"u"` or `"i"`.

    Returns
    -------
    binary : np.ndarray[S, np.dtype[Any]]
        The binary representation of the array `a`.
    """

    return a.view(
        a.dtype.str.replace("f", kind)
        .replace("i", kind)
        .replace("u", kind)
        # numpy_quaddtype currently does not set its kind properly
        .replace(_float128_dtype.kind, kind)
    )  # type: ignore


def to_total_order(a: np.ndarray[S, np.dtype[T]]) -> np.ndarray[S, np.dtype[U]]:
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
    a : np.ndarray[S, np.dtype[T]]
        The array to reinterpret to its total-order.

    Returns
    -------
    ordered : np.ndarray[S, np.dtype[U]]
        The total-order unsigned binary representation of the array `a`.
    """

    if np.issubdtype(a.dtype, np.unsignedinteger):
        return a  # type: ignore

    utype = a.dtype.str.replace("i", "u").replace("f", "u")

    if np.issubdtype(a.dtype, np.signedinteger):
        shift = np.iinfo(a.dtype).max  # type: ignore
        with np.errstate(
            over="ignore",
            under="ignore",
        ):
            return (
                a.view(utype) + np.array(shift, dtype=utype) + np.array(1, dtype=utype)
            )

    if not np.issubdtype(a.dtype, np.floating):
        raise TypeError(f"unsupported interval type {a.dtype}")

    itype = a.dtype.str.replace("f", "i")
    bits = np.iinfo(utype).bits

    mask = (-((a.view(dtype=utype) >> (bits - 1)).view(dtype=itype))).view(
        dtype=utype
    ) | (np.array(1, dtype=utype) << (bits - 1))

    return a.view(dtype=utype) ^ mask


def from_total_order(
    a: np.ndarray[S, np.dtype[U]], dtype: np.dtype[T]
) -> np.ndarray[S, np.dtype[T]]:
    """
    Reverses the reinterpretation of the array `a` back from total-order
    unsigned binary to the provided `dtype`.

    For floating point values, this implementation is based on Michael Herf's
    `IFloatFlip` function, see <http://stereopsis.com/radix.html>.

    Parameters
    ----------
    a : np.ndarray[S, np.dtype[U]]
        The array to reverse-reinterpret back from its total-order.

    Returns
    -------
    array : np.ndarray[S, np.dtype[T]]
        The array with its original `dtype`.
    """

    assert np.issubdtype(a.dtype, np.unsignedinteger)

    if np.issubdtype(dtype, np.unsignedinteger):
        return a  # type: ignore

    if np.issubdtype(dtype, np.signedinteger):
        shift = np.array(np.iinfo(dtype).max, dtype=dtype)  # type: ignore
        with np.errstate(
            over="ignore",
            under="ignore",
        ):
            return a.view(dtype) + shift + dtype.type(1)

    if not np.issubdtype(dtype, np.floating):
        raise TypeError(f"unsupported interval type {dtype}")

    utype = dtype.str.replace("f", "u")
    itype = dtype.str.replace("f", "i")
    bits = np.iinfo(utype).bits

    mask = ((a >> (bits - 1)).view(dtype=itype) - 1).view(dtype=utype) | (
        np.array(1, dtype=utype) << (bits - 1)
    )

    return (a ^ mask).view(dtype=dtype)


def lossless_cast(
    x: int | float | np.number | np.ndarray[S, np.dtype[np.number]],
    dtype: np.dtype[T],
    context: str,
) -> np.ndarray[tuple[()] | S, np.dtype[T]]:
    """
    Try to losslessly convert `x` to the provided `dtype`.

    A lossless conversion is one that can be reversed while preserving the
    original value. Integer values can be losslessly converted to integer or
    floating point types with sufficient precision. Floating point values
    can only be converted to floating point types.

    Parameters
    ----------
    x : int | float | np.ndarray[S, np.dtype[np.number]]
        The value or array to convert.
    dtype : np.dtype[T]
        The dtype to which the value or array should be converted.

    Returns
    -------
    converted : np.narray[tuple[()] | S, np.dtype[T]]
        The losslessly converted value or array with the given `dtype`.

    Raises
    ------
    TypeError
        If floating point values are converted to integer values.

    Raises
    ------
    ValueError
        If not all values could be losslessly converted.
    """

    xa = np.array(x)
    dtype_from = xa.dtype

    if np.issubdtype(dtype_from, np.floating) and not np.issubdtype(dtype, np.floating):
        raise TypeError(
            f"cannot losslessly cast {context} from {dtype_from} to {dtype}"
        )

    # we use unsafe casts here since we later check them for safety
    with np.errstate(divide="ignore", over="ignore", under="ignore", invalid="ignore"):
        xa_to = np.array(xa).astype(dtype, casting="unsafe")
        xa_back = xa_to.astype(dtype_from, casting="unsafe")

    lossless_same = np.where(_isnan(xa), _isnan(xa_back), xa == xa_back)

    if not np.all(lossless_same):
        raise ValueError(
            f"cannot losslessly cast (some) {context} values from {dtype_from} to {dtype}"
        )

    return xa_to


def saturating_finite_float_cast(
    x: int | float | np.number | np.ndarray[S, np.dtype[np.number]],
    dtype: np.dtype[F],
    context: str,
) -> np.ndarray[tuple[()] | S, np.dtype[F]]:
    """
    Try to convert the finite `x` to the provided floating-point `dtype`.
    Under- and overflows are clamped to finite values.

    Parameters
    ----------
    x : int | float | np.ndarray[S, np.dtype[np.number]]
        The value or array to convert.
    dtype : np.dtype[F]
        The floating-point dtype to which the value or array should be
        converted.

    Returns
    -------
    converted : np.narray[tuple[()] | S, np.dtype[F]]
        The losslessly converted value or array with the given `dtype`.

    Raises
    ------
    ValueError
        If some values are non-finite, i.e. infinite or NaN.
    """

    assert np.issubdtype(dtype, np.floating) or (dtype == _float128_dtype)

    xa = np.array(x)

    if not isinstance(x, int) and not np.all(_isfinite(xa)):
        raise ValueError(
            f"cannot cast non-finite {context} values from {xa.dtype} to saturating finite {dtype}"
        )

    # we use unsafe casts here since but are safe since
    # - we know that inputs are all finite
    # - we cast to float, where under- and overflows saturate to np.inf
    # - we later clamp the values to finite
    with np.errstate(divide="ignore", over="ignore", under="ignore", invalid="ignore"):
        xa_to = np.array(xa).astype(dtype, casting="unsafe")

    return _nan_to_zero_inf_to_finite(xa_to)
