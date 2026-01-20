"""
Utility functions to cast arrays to floating-point, binary, and total-order representations.
"""

__all__ = [
    "ToFloatMode",
    "to_float",
    "from_float",
    "as_bits",
    "to_total_order",
    "from_total_order",
    "lossless_cast",
    "saturating_finite_float_cast",
]

from enum import Enum, auto
from typing import assert_never

import numpy as np

from ._compat import _ensure_array, _is_of_dtype
from ._float128 import _float128_dtype
from .error import TypeSetError, ctx
from .typing import F, S, T, U


class ToFloatMode(Enum):
    """
    Mode for losslessly converting numeric data to floating-point.
    """

    lossless = auto()
    """
    Automatically select the smallest floating-point data type that can
    losslessly represent the input data.
    """

    float16 = auto()
    """
    Use 16 bit floating-point precision, if lossless.
    """

    float32 = auto()
    """
    Use 32 bit floating-point precision, if lossless.
    """

    float64 = auto()
    """
    Use 64 bit floating-point precision, if lossless.
    """

    float128 = auto()
    """
    Use 128 bit floating-point precision, if lossless.
    """

    def floating_point_dtype_for(self, dtype: np.dtype[T]) -> np.dtype[np.floating]:
        """
        Select the floating-point dtype for the input `dtype`.

        Only data type supported by the safeguards (see
        [`Safeguards.supported_dtypes`][.....api.Safeguards.supported_dtypes])
        are supported by this method.

        The [`numpy_quaddtype`](https://pypi.org/project/numpy-quaddtype/)
        package is used to provide a true 128 bit floating-point data type.

        Parameters
        ----------
        dtype : np.dtype[T]
            Data type of the input data.

        Returns
        -------
        ftype : np.dtype[np.floating]
            Floating-point data type that can losslessly represent all values
            from the input `dtype`.

        Raises
        ------
        TypeError
            if `dtype` cannot be losslessly cast to `self`.
        """

        match self:
            case ToFloatMode.lossless:
                SMALLEST_LOSSLESS_FTYPE: dict[
                    np.dtype[np.number], np.dtype[np.floating]
                ] = {
                    np.dtype(np.int8): np.dtype(np.float16),
                    np.dtype(np.int16): np.dtype(np.float32),
                    np.dtype(np.int32): np.dtype(np.float64),
                    np.dtype(np.int64): _float128_dtype,
                    np.dtype(np.uint8): np.dtype(np.float16),
                    np.dtype(np.uint16): np.dtype(np.float32),
                    np.dtype(np.uint32): np.dtype(np.float64),
                    np.dtype(np.uint64): _float128_dtype,
                    np.dtype(np.float16): np.dtype(np.float16),
                    np.dtype(np.float32): np.dtype(np.float32),
                    np.dtype(np.float64): np.dtype(np.float64),
                }
                return SMALLEST_LOSSLESS_FTYPE[dtype]
            case ToFloatMode.float16:
                if dtype in (
                    np.dtype(np.int8),
                    np.dtype(np.uint8),
                    np.dtype(np.float16),
                ):
                    return np.dtype(np.float16)
                raise (
                    TypeError(f"cannot losslessly cast {dtype.name} to float16") | ctx
                )
            case ToFloatMode.float32:
                if dtype in (
                    np.dtype(np.int8),
                    np.dtype(np.uint8),
                    np.dtype(np.int16),
                    np.dtype(np.uint16),
                    np.dtype(np.float16),
                    np.dtype(np.float32),
                ):
                    return np.dtype(np.float32)
                raise (
                    TypeError(f"cannot losslessly cast {dtype.name} to float32") | ctx
                )
            case ToFloatMode.float64:
                if dtype in (
                    np.dtype(np.int8),
                    np.dtype(np.uint8),
                    np.dtype(np.int16),
                    np.dtype(np.uint16),
                    np.dtype(np.int32),
                    np.dtype(np.uint32),
                    np.dtype(np.float16),
                    np.dtype(np.float32),
                    np.dtype(np.float64),
                ):
                    return np.dtype(np.float64)
                raise (
                    TypeError(f"cannot losslessly cast {dtype.name} to float64") | ctx
                )
            case ToFloatMode.float128:
                if dtype in (
                    np.dtype(np.int8),
                    np.dtype(np.uint8),
                    np.dtype(np.int16),
                    np.dtype(np.uint16),
                    np.dtype(np.int32),
                    np.dtype(np.uint32),
                    np.dtype(np.int64),
                    np.dtype(np.uint64),
                    np.dtype(np.float16),
                    np.dtype(np.float32),
                    np.dtype(np.float64),
                ):
                    return _float128_dtype
                raise (
                    TypeError(f"cannot losslessly cast {dtype.name} to float128") | ctx
                )
            case _:
                assert_never(self)


def to_float(
    x: np.ndarray[S, np.dtype[T]], ftype: np.dtype[F]
) -> np.ndarray[S, np.dtype[F]]:
    """
    Losslessly convert the array `x` to the floating-point data type `ftype`.

    `ftype` must be a floating-point data type that can represent all values
    of the data type of `x` without loss in precision:

    * For floating-point data, it is at least the input data type.

    * For integer data, it is a floating-point type with sufficient precision
    to represent all integer values, i.e. a type whose mantissa has more bits
    than the integer type. For the supported floating-point types, this
    corresponds to choosing a floating-point data type with a larger bit width
    (e.g. at least [`np.float64`][numpy.float64] for [`np.int32`][numpy.int32]
    or [`np.uint32`][numpy.uint32] data).

    The
    [`ToFloatMode.floating_point_dtype_for`][..ToFloatMode.floating_point_dtype_for]
    method can be used to select a floating-point data type that fits the above
    criteria.

    Parameters
    ----------
    x : np.ndarray[S, np.dtype[T]]
        The array to convert.
    ftype : np.dtype[F]
        The floating-point data type to convert to, which must be able to
        losslessly represent all values of the data type of `x`.

    Returns
    -------
    converted : np.ndarray[S, np.dtype[F]]
        The converted array with the chosen floating-point data type.
    """

    assert np.issubdtype(ftype, np.floating)

    if np.issubdtype(x.dtype, np.floating):
        assert ftype.itemsize >= x.dtype.itemsize
    else:
        assert np.finfo(ftype).nmant >= x.dtype.itemsize

    if _is_of_dtype(x, ftype):
        return x

    # FIXME: https://github.com/numpy/numpy-user-dtypes/issues/163
    with np.errstate(divide="ignore", over="ignore", under="ignore", invalid="ignore"):
        # lossless cast to floating-point data type with a sufficiently large
        #  mantissa
        return x.astype(ftype, casting="safe")


def from_float(
    x: np.ndarray[S, np.dtype[F]], dtype: np.dtype[T]
) -> np.ndarray[S, np.dtype[T]]:
    """
    Reverses the conversion of the array `x`, converted using the
    [`to_float`][..to_float], back to the original `dtype`.

    If the original `dtype` was floating-point with lower precision, the
    conversion is lossy.

    If the original `dtype` was integer, the rounding conversion is lossy.
    Infinite values are clamped to the minimum/maximum integer values.

    Parameters
    ----------
    x : np.ndarray[S, np.dtype[F]]
        The floating-point array to re-convert.
    dtype : np.dtype[T]
        The original dtype.

    Returns
    -------
    converted : np.ndarray[S, np.dtype[T]]
        The re-converted array with the original `dtype`.
    """

    x = _ensure_array(x)

    assert np.issubdtype(x.dtype, np.floating)

    if _is_of_dtype(x, dtype):
        return x

    if np.issubdtype(dtype, np.floating):
        with np.errstate(
            divide="ignore", over="ignore", under="ignore", invalid="ignore"
        ):
            # lossy cast to lower-precision floating-point number
            return x.astype(dtype, casting="unsafe")

    info = np.iinfo(dtype)  # type: ignore
    imin, imax = np.array(info.min, dtype=dtype), np.array(info.max, dtype=dtype)

    with np.errstate(divide="ignore", over="ignore", under="ignore", invalid="ignore"):
        # lossy cast from floating-point to integer
        # round first with rint (round to nearest, ties to nearest even)
        converted: np.ndarray[S, np.dtype[T]] = _ensure_array(np.rint(x)).astype(
            dtype, casting="unsafe"
        )
        converted[np.greater(x, imax.astype(x.dtype, casting="safe"))] = imax
        converted[np.less(x, imin.astype(x.dtype, casting="safe"))] = imin

    return converted


def as_bits(
    a: np.ndarray[S, np.dtype[T]],
) -> np.ndarray[S, np.dtype[U]]:
    """
    Reinterprets the array `a` to its binary unsigned integer representation.

    Parameters
    ----------
    a : np.ndarray[S, np.dtype[T]]
        The array to reinterpret as binary.

    Returns
    -------
    binary : np.ndarray[S, np.dtype[U]]
        The binary unsigned integer representation of the array `a`.
    """

    return a.view(a.dtype.str.replace("f", "u").replace("i", "u"))


def to_total_order(a: np.ndarray[S, np.dtype[T]]) -> np.ndarray[S, np.dtype[U]]:
    """
    Reinterprets the array `a` to its total-order unsigned binary
    representation.

    In their total-order representation, the smallest value is mapped to
    unsigned zero, and the largest value is mapped to the largest unsigned
    value.

    For floating-point values, this implementation is based on Michael Herf's
    `FloatFlip` function, see <http://stereopsis.com/radix.html>.

    Parameters
    ----------
    a : np.ndarray[S, np.dtype[T]]
        The array to reinterpret to its total-order.

    Returns
    -------
    ordered : np.ndarray[S, np.dtype[U]]
        The total-order unsigned binary representation of the array `a`.

    Raises
    ------
    TypeSetError
        if `a` is not a signed/unsigned integer or floating array.
    """

    TypeSetError.check_or_raise(
        a.dtype.type, np.unsignedinteger | np.signedinteger | np.floating
    )

    if np.issubdtype(a.dtype, np.unsignedinteger):
        return a  # type: ignore

    utype = np.dtype(a.dtype.str.replace("i", "u").replace("f", "u"))

    if np.issubdtype(a.dtype, np.signedinteger):
        shift = np.iinfo(a.dtype).max  # type: ignore
        with np.errstate(
            over="ignore",
            under="ignore",
        ):
            return a.view(utype) + utype.type(shift) + utype.type(1)

    itype = a.dtype.str.replace("f", "i")
    bits = np.iinfo(utype).bits

    mask = (-((a.view(dtype=utype) >> (bits - 1)).view(dtype=itype))).view(
        dtype=utype
    ) | (utype.type(1) << (bits - 1))

    return a.view(dtype=utype) ^ mask


def from_total_order(
    a: np.ndarray[S, np.dtype[U]], dtype: np.dtype[T]
) -> np.ndarray[S, np.dtype[T]]:
    """
    Reverses the reinterpretation of the array `a` back from total-order
    unsigned binary to the provided `dtype`.

    For floating-point values, this implementation is based on Michael Herf's
    `IFloatFlip` function, see <http://stereopsis.com/radix.html>.

    Parameters
    ----------
    a : np.ndarray[S, np.dtype[U]]
        The array to reverse-reinterpret back from its total-order.

    Returns
    -------
    array : np.ndarray[S, np.dtype[T]]
        The array with its original `dtype`.

    Raises
    ------
    TypeSetError
        if `a` is not an unsigned integer array.
    TypeSetError
        if `dtype` is not a signed/unsigned integer or floating data type.
    """

    TypeSetError.check_or_raise(a.dtype.type, np.unsignedinteger)
    TypeSetError.check_or_raise(
        dtype.type, np.unsignedinteger | np.signedinteger | np.floating
    )

    if np.issubdtype(dtype, np.unsignedinteger):
        return a  # type: ignore

    if np.issubdtype(dtype, np.signedinteger):
        shift = np.array(np.iinfo(dtype).max, dtype=dtype)  # type: ignore
        with np.errstate(
            over="ignore",
            under="ignore",
        ):
            return a.view(dtype) + shift + dtype.type(1)

    utype = np.dtype(dtype.str.replace("f", "u"))
    itype = dtype.str.replace("f", "i")
    bits = np.iinfo(utype).bits

    mask = ((a >> (bits - 1)).view(dtype=itype) - 1).view(dtype=utype) | (
        utype.type(1) << (bits - 1)
    )

    return (a ^ mask).view(dtype=dtype)


def lossless_cast(
    x: int | float | np.number | np.ndarray[S, np.dtype[np.number]],
    dtype: np.dtype[T],
) -> np.ndarray[tuple[()] | S, np.dtype[T]]:
    """
    Try to losslessly convert `x` to the provided `dtype`.

    A lossless conversion is one that can be reversed while preserving the
    original value. Integer values can be losslessly converted to integer or
    floating-point types with sufficient precision. Floating-point values
    can only be converted to floating-point types.

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
        if floating-point values are converted to an integer `dtype`.
    ValueError
        if not all values could be losslessly converted to `dtype`.
    """

    xa = np.array(x, copy=None)
    dtype_from = xa.dtype

    if np.issubdtype(dtype_from, np.floating) and not np.issubdtype(dtype, np.floating):
        raise TypeError(f"cannot losslessly cast from {dtype_from} to {dtype}") | ctx

    # we use unsafe casts here since we later check them for safety
    with np.errstate(divide="ignore", over="ignore", under="ignore", invalid="ignore"):
        xa_to = _ensure_array(xa).astype(dtype, casting="unsafe")
        xa_back = xa_to.astype(dtype_from, casting="unsafe")

    lossless_same = xa == xa_back
    lossless_same |= np.isnan(xa) & np.isnan(xa_back)

    if isinstance(x, int):
        lossless_same &= (x < 0) == np.signbit(xa_back)
    else:
        lossless_same &= np.signbit(xa) == np.signbit(xa_back)

    if not np.all(lossless_same):
        raise (
            ValueError(
                f"cannot losslessly cast (some) values from {dtype_from} to "
                + f"{dtype}"
            )
            | ctx
        )

    return xa_to


def saturating_finite_float_cast(
    x: int | float | np.number | np.ndarray[S, np.dtype[np.number]],
    dtype: np.dtype[F],
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
        if any values are non-finite, i.e. infinite or NaN.
    """

    assert np.issubdtype(dtype, np.floating)

    xa = np.array(x, copy=None)

    if not isinstance(x, int) and not np.all(np.isfinite(xa)):
        raise (
            ValueError(
                f"cannot cast non-finite values from {xa.dtype.name} to "
                + f"saturating finite {dtype.name}"
            )
            | ctx
        )

    # we use unsafe casts here since but are safe since
    # - we know that inputs are all finite
    # - we cast to float, where under- and overflows saturate to np.inf
    # - we later clamp the values to finite
    with np.errstate(divide="ignore", over="ignore", under="ignore", invalid="ignore"):
        xa_to = _ensure_array(xa).astype(dtype, casting="unsafe")

    # the above checks guarantee that there are no NaNs in xa
    if isinstance(x, int):
        assert (x < 0) == np.signbit(xa_to)
    else:
        assert np.all(np.signbit(xa) == np.signbit(xa_to))

    return np.nan_to_num(xa_to)
