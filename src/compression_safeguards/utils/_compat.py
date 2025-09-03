"""
Private compatibility functions, mostly wrappers around numpy, that ensure
equivalent behaviour for all supported dtypes, e.g. numpy_quaddtype.
"""

__all__ = [
    "_nan_to_zero",
    "_nan_to_zero_inf_to_finite",
    "_nextafter",
    "_symmetric_modulo",
    "_minimum_zero_sign_sensitive",
    "_maximum_zero_sign_sensitive",
    "_where",
    "_broadcast_to",
    "_is_negative",
    "_is_negative_zero",
    "_is_positive",
    "_is_positive_zero",
    "_floating_max",
    "_floating_smallest_subnormal",
    "_pi",
    "_e",
]

from typing import overload

import numpy as np

from ._float128 import (
    _float128,
    _float128_dtype,
    _float128_e,
    _float128_max,
    _float128_min,
    _float128_pi,
    _float128_smallest_normal,
    _float128_smallest_subnormal,
    _float128_type,
)
from .typing import F, S, Si, T, Ti


# reimplementation of np.nan_to_num that also works for numpy_quaddtype and
#  keeps infinities intact
@np.errstate(invalid="ignore")
def _nan_to_zero(a: np.ndarray[S, np.dtype[T]]) -> np.ndarray[S, np.dtype[T]]:
    out = np.array(a, copy=True)
    out[np.isnan(out)] = 0
    return out


# reimplementation of np.nan_to_num that also works for numpy_quaddtype and
#  makes all values finite
@np.errstate(invalid="ignore")
def _nan_to_zero_inf_to_finite(
    a: np.ndarray[S, np.dtype[T]],
) -> np.ndarray[S, np.dtype[T]]:
    out: np.ndarray[S, np.dtype[T]] = np.array(a, copy=True)

    if out.dtype == _float128_dtype:
        fmin, fmax = _float128_min, _float128_max
    elif not np.issubdtype(out.dtype, np.floating):
        return out
    else:
        finfo = np.finfo(out.dtype)  # type: ignore
        fmin, fmax = finfo.min, finfo.max

    out[np.isnan(a)] = 0
    out[a == -np.inf] = fmin
    out[a == np.inf] = fmax
    return out


# wrapper around np.nextafter that also works for numpy_quaddtype
# Implementation is variant 2 from https://stackoverflow.com/a/70512834
@np.errstate(invalid="ignore")
def _nextafter(
    a: np.ndarray[S, np.dtype[F]], b: int | float | np.ndarray[S, np.dtype[F]]
) -> np.ndarray[S, np.dtype[F]]:
    if (type(a) is not _float128_type) and (
        not isinstance(a, np.ndarray) or a.dtype != _float128_dtype
    ):
        return np.nextafter(a, b)  # type: ignore

    a = np.array(a, copy=None)
    b = np.array(b, dtype=a.dtype, copy=None)

    _float128_incr_subnormal = np.full(a.shape, _float128_smallest_subnormal)
    np.negative(_float128_incr_subnormal, out=_float128_incr_subnormal, where=(a < 0))

    abs_a_zero_mantissa = (
        (
            np.atleast_1d(a).view(np.uint8)
            & np.atleast_1d(np.full_like(a, np.inf)).view(np.uint8)
        )
        .view(a.dtype)
        .reshape(a.shape)
    )
    _float128_incr_normal = abs_a_zero_mantissa / (_float128(2) ** 112)

    # zero, subnormal, or smallest normal
    out_subnormal = np.array(np.subtract(a, _float128_incr_subnormal), copy=None)
    out_subnormal[a == (-_float128_smallest_subnormal)] = -0.0
    np.copyto(
        out_subnormal,
        np.add(a, _float128_incr_subnormal),
        where=((a < b) == (a >= 0)),
        casting="no",
    )

    out: np.ndarray[S, np.dtype[F]] = np.array(a, copy=True)
    # normal
    # note: implementation deviates here since numpy_quaddtype
    #       divides/multiplies with error
    #       based on https://stackoverflow.com/a/70512041
    np.add(
        out,
        _float128_incr_normal / 2,
        out=out,
        where=((a < b) & (a == -abs_a_zero_mantissa)),
    )
    np.add(
        out,
        _float128_incr_normal,
        out=out,
        where=((a < b) & (a != -abs_a_zero_mantissa)),
    )
    np.subtract(
        out,
        _float128_incr_normal / 2,
        out=out,
        where=((a >= b) & (a == abs_a_zero_mantissa)),
    )
    np.subtract(
        out,
        _float128_incr_normal,
        out=out,
        where=((a >= b) & (a != abs_a_zero_mantissa)),
    )
    # zero, subnormal, or smallest normal
    np.copyto(
        out, out_subnormal, where=(np.abs(a) <= _float128_smallest_normal), casting="no"
    )
    # infinity
    out[np.isinf(a)] = _float128_max
    out[np.isinf(a) & (a < 0)] = _float128_min
    # equal
    np.copyto(out, b, where=(a == b), casting="no")
    # unordered, at least one is NaN
    np.add(out, b, out=out, where=(np.isnan(a) | np.isnan(b)))

    return out


# wrapper around np.mod(p, q) that guarantees that the result is in [-q/2, q/2]
def _symmetric_modulo(
    p: np.ndarray[S, np.dtype[F]], q: np.ndarray[S, np.dtype[F]]
) -> np.ndarray[S, np.dtype[F]]:
    q2: np.ndarray[S, np.dtype[F]] = np.divide(q, 2)
    out: np.ndarray[S, np.dtype[F]] = np.array(np.mod(p + q2, q), copy=None)
    np.subtract(out, q2, out=out)
    return out


# wrapper around np.minimum that also works for +0.0 and -0.0
@overload
def _minimum_zero_sign_sensitive(
    a: Ti, b: np.ndarray[S, np.dtype[Ti]]
) -> np.ndarray[S, np.dtype[Ti]]: ...


@overload
def _minimum_zero_sign_sensitive(
    a: np.ndarray[S, np.dtype[Ti]], b: Ti
) -> np.ndarray[S, np.dtype[Ti]]: ...


@overload
def _minimum_zero_sign_sensitive(
    a: np.ndarray[S, np.dtype[T]], b: np.ndarray[S, np.dtype[T]]
) -> np.ndarray[S, np.dtype[T]]: ...


def _minimum_zero_sign_sensitive(a, b):
    a = np.array(a, copy=None)
    b = np.array(b, copy=None)
    minimum = np.minimum(a, b)
    if np.issubdtype(a.dtype, np.integer) and np.issubdtype(b.dtype, np.integer):
        return minimum
    minimum_array = np.array(minimum, copy=None)
    a = _broadcast_to(a.astype(minimum_array.dtype), minimum_array.shape)
    b = _broadcast_to(b.astype(minimum_array.dtype), minimum_array.shape)
    np.copyto(
        minimum_array,
        a,
        where=((minimum == 0) & (np.signbit(a) > np.signbit(b))),
        casting="no",
    )
    np.copyto(
        minimum_array,
        b,
        where=((minimum == 0) & (np.signbit(a) < np.signbit(b))),
        casting="no",
    )
    return minimum_array


# wrapper around np.maximum that also works for +0.0 and -0.0
@overload
def _maximum_zero_sign_sensitive(
    a: Ti, b: np.ndarray[S, np.dtype[Ti]]
) -> np.ndarray[S, np.dtype[Ti]]: ...


@overload
def _maximum_zero_sign_sensitive(
    a: np.ndarray[S, np.dtype[Ti]], b: Ti
) -> np.ndarray[S, np.dtype[Ti]]: ...


@overload
def _maximum_zero_sign_sensitive(
    a: np.ndarray[S, np.dtype[T]], b: np.ndarray[S, np.dtype[T]]
) -> np.ndarray[S, np.dtype[T]]: ...


def _maximum_zero_sign_sensitive(a, b):
    a = np.array(a, copy=None)
    b = np.array(b, copy=None)
    maximum = np.maximum(a, b)
    if np.issubdtype(a.dtype, np.integer) and np.issubdtype(b.dtype, np.integer):
        return maximum
    maximum_array = np.array(maximum, copy=None)
    a = _broadcast_to(a.astype(maximum_array.dtype), maximum_array.shape)
    b = _broadcast_to(b.astype(maximum_array.dtype), maximum_array.shape)
    np.copyto(
        maximum_array,
        a,
        where=((maximum == 0) & (np.signbit(a) < np.signbit(b))),
        casting="no",
    )
    np.copyto(
        maximum_array,
        b,
        where=((maximum == 0) & (np.signbit(a) > np.signbit(b))),
        casting="no",
    )
    return maximum_array


# wrapper around np.where but with better type hints
@overload
def _where(
    cond: np.ndarray[S, np.dtype[np.bool]],
    a: np.ndarray[S, np.dtype[T]],
    b: np.ndarray[S, np.dtype[T]],
) -> np.ndarray[S, np.dtype[T]]: ...


@overload
def _where(
    cond: np.ndarray[S, np.dtype[np.bool]],
    a: np.ndarray[S, np.dtype[np.bool]],
    b: np.ndarray[S, np.dtype[np.bool]],
) -> np.ndarray[S, np.dtype[np.bool]]: ...


@overload
def _where(
    cond: np.ndarray[S, np.dtype[np.bool]], a: Ti, b: np.ndarray[S, np.dtype[Ti]]
) -> np.ndarray[S, np.dtype[Ti]]: ...


@overload
def _where(
    cond: np.ndarray[S, np.dtype[np.bool]], a: np.ndarray[S, np.dtype[Ti]], b: Ti
) -> np.ndarray[S, np.dtype[Ti]]: ...


@overload
def _where(
    cond: np.ndarray[S, np.dtype[np.bool]], a: Ti, b: Ti
) -> np.ndarray[S, np.dtype[Ti]]: ...


@overload
def _where(cond: bool, a: Ti, b: Ti) -> Ti: ...


def _where(cond, a, b):
    return np.where(cond, a, b)


# wrapper around np.broadcast_to but with better type hints
@overload
def _broadcast_to(
    a: np.ndarray[tuple[int, ...], np.dtype[T]], shape: Si
) -> np.ndarray[Si, np.dtype[T]]: ...


@overload
def _broadcast_to(
    a: np.ndarray[tuple[int, ...], np.dtype[np.bool]], shape: Si
) -> np.ndarray[Si, np.dtype[np.bool]]: ...


@overload
def _broadcast_to(a: Ti, shape: Si) -> np.ndarray[Si, np.dtype[Ti]]: ...


def _broadcast_to(a, shape):
    return np.broadcast_to(a, shape)


# wrapper around a < 0 that also works for -0.0 (is negative)
def _is_negative(
    a: np.ndarray[S, np.dtype[F]],
) -> np.ndarray[S, np.dtype[np.bool]]:
    # check not just for a < 0 but also for a == -0.0
    return np.less_equal(a, 0) & (np.copysign(a, 1) == -1)


# check for x == -0.0
def _is_negative_zero(
    a: np.ndarray[S, np.dtype[F]],
) -> np.ndarray[S, np.dtype[np.bool]]:
    return (a == 0) & (np.copysign(a, 1) == -1)


# wrapper around a > 0 that also works for -0.0 (is not positive)
def _is_positive(
    a: np.ndarray[S, np.dtype[F]],
) -> np.ndarray[S, np.dtype[np.bool]]:
    # check not just for a > 0 but also for a == +0.0
    return np.greater_equal(a, 0) & (np.copysign(a, 1) == +1)


# check for x == +0.0
def _is_positive_zero(
    a: np.ndarray[S, np.dtype[F]],
) -> np.ndarray[S, np.dtype[np.bool]]:
    return (a == 0) & (np.copysign(a, 1) == +1)


# wrapper around np.finfo(dtype).max that also works for numpy_quaddtype
def _floating_max(dtype: np.dtype[F]) -> F:
    if dtype == _float128_dtype:
        return _float128_max  # type: ignore
    return dtype.type(np.finfo(dtype).max)


# wrapper around np.finfo(dtype).smallest_subnormal that also works for
#  numpy_quaddtype
def _floating_smallest_subnormal(dtype: np.dtype[F]) -> F:
    if dtype == _float128_dtype:
        return _float128_smallest_subnormal  # type: ignore
    return dtype.type(np.finfo(dtype).smallest_subnormal)


# wrapper around np.pi, of the dtype, that also works for numpy_quaddtype
def _pi(dtype: np.dtype[F]) -> F:
    if dtype == _float128_dtype:
        return _float128_pi  # type: ignore
    return dtype.type(np.pi)


# wrapper around np.e, of the dtype, that also works for numpy_quaddtype
def _e(dtype: np.dtype[F]) -> F:
    if dtype == _float128_dtype:
        return _float128_e  # type: ignore
    return dtype.type(np.e)
