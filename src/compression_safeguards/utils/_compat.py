"""
Private compatibility functions, mostly wrappers around numpy, that ensure
equivalent behaviour for all supported dtypes, e.g. numpy_quaddtype.
"""

__all__ = [
    "_nan_to_zero_inf_to_finite",
    "_nextafter",
    "_place",
    "_logical_not",
    "_as_logical",
    "_symmetric_modulo",
    "_minimum_zero_sign_sensitive",
    "_maximum_zero_sign_sensitive",
    "_where",
    "_reshape",
    "_broadcast_to",
    "_stack",
    "_ensure_array",
    "_ones",
    "_zeros",
    "_logical_and",
    "_sliding_window_view",
    "_is_sign_negative_number",
    "_is_negative_zero",
    "_is_sign_positive_number",
    "_is_positive_zero",
    "_is_of_dtype",
    "_is_of_shape",
    "_floating_max",
    "_floating_smallest_subnormal",
    "_pi",
    "_e",
]

from collections.abc import Sequence
from typing import Literal, TypeGuard, TypeVar, overload

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

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
from .typing import TB, F, Fi, S, Si, T, Ti

N = TypeVar("N", bound=int, covariant=True)
""" Any [`int`][int] (covariant). """


# reimplementation of np.nan_to_num that also works for numpy_quaddtype and
#  makes all values finite
# FIXME: https://github.com/numpy/numpy-user-dtypes/issues/163
@np.errstate(invalid="ignore")
def _nan_to_zero_inf_to_finite(
    a: np.ndarray[S, np.dtype[T]],
) -> np.ndarray[S, np.dtype[T]]:
    out: np.ndarray[S, np.dtype[T]] = _ensure_array(a, copy=True)

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
# FIXME: https://github.com/numpy/numpy-user-dtypes/issues/163
@np.errstate(invalid="ignore")
def _nextafter(
    a: np.ndarray[S, np.dtype[F]], b: np.ndarray[S, np.dtype[F]]
) -> np.ndarray[S, np.dtype[F]]:
    if (type(a) is not _float128_type) and (
        not isinstance(a, np.ndarray) or a.dtype != _float128_dtype
    ):
        return np.nextafter(a, b)

    a = _ensure_array(a)
    b = _ensure_array(b)

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
    out_subnormal = _ensure_array(np.subtract(a, _float128_incr_subnormal))
    out_subnormal[a == (-_float128_smallest_subnormal)] = -0.0
    np.add(a, _float128_incr_subnormal, out=out_subnormal, where=((a < b) == (a >= 0)))

    out: np.ndarray[S, np.dtype[F]] = _ensure_array(a, copy=True)
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


# wrapper around np.place that also works for numpy_quaddtype
# FIXME: https://github.com/numpy/numpy-user-dtypes/issues/236
def _place(
    a: np.ndarray[S, np.dtype[TB]],
    mask: np.ndarray[S, np.dtype[np.bool]],
    vals: np.ndarray[tuple[int], np.dtype[TB]],
) -> None:
    if (type(a) is not _float128_type) and (
        not isinstance(a, np.ndarray) or a.dtype != _float128_dtype
    ):
        return np.place(a, mask, vals)

    return np.put(a, np.flatnonzero(mask), vals)


# wrapper around np.logical_not that also works for numpy_quaddtype
@overload
def _logical_not(a: Ti) -> bool: ...


@overload
def _logical_not(a: np.ndarray[S, np.dtype[T]]) -> np.ndarray[S, np.dtype[np.bool]]: ...


def _logical_not(a):
    if (type(a) is not _float128_type) and (
        not isinstance(a, np.ndarray) or a.dtype != _float128_dtype
    ):
        return np.logical_not(a)

    return a == 0


# helper for around np.any and np.all that also works for numpy_quaddtype
def _as_logical(a: np.ndarray[S, np.dtype[T]]) -> np.ndarray[S, np.dtype[T | np.bool]]:
    if (type(a) is not _float128_type) and (
        not isinstance(a, np.ndarray) or a.dtype != _float128_dtype
    ):
        return a

    return a != 0


# wrapper around np.mod(p, q) that guarantees that the result is in [-q/2, q/2]
@overload
def _symmetric_modulo(
    p: np.ndarray[S, np.dtype[F]], q: np.ndarray[S, np.dtype[F]]
) -> np.ndarray[S, np.dtype[F]]: ...


@overload
def _symmetric_modulo(p: Fi, q: Fi) -> Fi: ...


def _symmetric_modulo(p, q):
    q2 = np.divide(q, 2)
    out = _ensure_array(np.add(p, q2))
    np.mod(out, q, out=out)
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
    a = _ensure_array(a)
    b = _ensure_array(b)
    minimum = np.minimum(a, b)
    if np.issubdtype(a.dtype, np.integer) and np.issubdtype(b.dtype, np.integer):
        return minimum
    minimum_array = _ensure_array(minimum)
    a = _broadcast_to(
        a.astype(minimum_array.dtype, casting="safe"), minimum_array.shape
    )
    b = _broadcast_to(
        b.astype(minimum_array.dtype, casting="safe"), minimum_array.shape
    )
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
    a = _ensure_array(a)
    b = _ensure_array(b)
    maximum = np.maximum(a, b)
    if np.issubdtype(a.dtype, np.integer) and np.issubdtype(b.dtype, np.integer):
        return maximum
    maximum_array = _ensure_array(maximum)
    a = _broadcast_to(
        a.astype(maximum_array.dtype, casting="safe"), maximum_array.shape
    )
    b = _broadcast_to(
        b.astype(maximum_array.dtype, casting="safe"), maximum_array.shape
    )
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
    a: np.ndarray[S, np.dtype[TB]],
    b: np.ndarray[S, np.dtype[TB]],
) -> np.ndarray[S, np.dtype[TB]]: ...


@overload
def _where(cond: bool, a: Ti, b: Ti) -> Ti: ...


def _where(cond, a, b):
    return np.where(cond, a, b)


# wrapper around np.reshape but with better type hints
def _reshape(
    a: np.ndarray[tuple[int, ...], np.dtype[TB]], shape: Si
) -> np.ndarray[Si, np.dtype[TB]]:
    return np.reshape(a, shape)


# wrapper around np.broadcast_to but with better type hints
@overload
def _broadcast_to(
    a: np.ndarray[tuple[int, ...], np.dtype[TB]], shape: Si
) -> np.ndarray[Si, np.dtype[TB]]: ...


@overload
def _broadcast_to(a: Ti, shape: Si) -> np.ndarray[Si, np.dtype[Ti]]: ...


def _broadcast_to(a, shape):
    return np.broadcast_to(a, shape)


# wrapper around np.stack but with better type hints
@overload
def _stack(
    arrays: tuple[np.ndarray[tuple[N], np.dtype[T]]],
) -> np.ndarray[tuple[Literal[2], N], np.dtype[T]]: ...


@overload
def _stack(
    arrays: Sequence[np.ndarray[tuple[N], np.dtype[T]]],
) -> np.ndarray[tuple[int, N], np.dtype[T]]: ...


def _stack(arrays):
    return np.stack(arrays)


# wrapper around np.array(a, copy=(copy=None)) but with better type hints
def _ensure_array(
    a: np.ndarray[S, np.dtype[TB]], copy: None | bool = None
) -> np.ndarray[S, np.dtype[TB]]:
    return np.array(a, copy=copy)


# wrapper around np.ones but with better type hints
def _ones(shape: Si, dtype: np.dtype[TB]) -> np.ndarray[Si, np.dtype[TB]]:
    return np.ones(shape, dtype=dtype)


# wrapper around np.zeros but with better type hints
def _zeros(shape: Si, dtype: np.dtype[TB]) -> np.ndarray[Si, np.dtype[TB]]:
    return np.zeros(shape, dtype=dtype)


# wrapper around np.logical_and with better type hints
def _logical_and(
    a: np.ndarray[S, np.dtype[np.bool]],
    b: Literal[True] | np.ndarray[S, np.dtype[np.bool]],
) -> np.ndarray[S, np.dtype[np.bool]]:
    return a & b  # type: ignore


# wrapper around np.lib.stride_tricks.sliding_window_view with better type hints
def _sliding_window_view(
    a: np.ndarray[tuple[int, ...], np.dtype[TB]],
    window_shape: int | tuple[int, ...],
    *,
    axis: int | tuple[int, ...],
    writeable: Literal[False],
) -> np.ndarray[tuple[int, ...], np.dtype[TB]]:
    return sliding_window_view(
        a,
        window_shape,
        # the docs say that tuple[int, ...] is allowed here
        axis=axis,  # type: ignore
        writeable=writeable,
    )


# wrapper around a < 0 that also works for -0.0 (is negative) but excludes NaNs
def _is_sign_negative_number(
    a: np.ndarray[S, np.dtype[F]],
) -> np.ndarray[S, np.dtype[np.bool]]:
    # check not just for a < 0 but also for a == -0.0
    return np.less_equal(a, 0) & (np.copysign(1, a) == -1)


# check for x == -0.0
def _is_negative_zero(
    a: np.ndarray[S, np.dtype[F]],
) -> np.ndarray[S, np.dtype[np.bool]]:
    return (a == 0) & (np.copysign(1, a) == -1)


# wrapper around a > 0 that also works for -0.0 (is not positive) but excludes
#  NaNs
def _is_sign_positive_number(
    a: np.ndarray[S, np.dtype[F]],
) -> np.ndarray[S, np.dtype[np.bool]]:
    # check not just for a > 0 but also for a == +0.0
    return np.greater_equal(a, 0) & (np.copysign(1, a) == +1)


# check for x == +0.0
def _is_positive_zero(
    a: np.ndarray[S, np.dtype[F]],
) -> np.ndarray[S, np.dtype[np.bool]]:
    return (a == 0) & (np.copysign(1, a) == +1)


# type guard for x.dtype == dtype
def _is_of_dtype(
    x: np.ndarray[S, np.dtype[np.number]], dtype: np.dtype[T]
) -> TypeGuard[np.ndarray[S, np.dtype[T]]]:
    return x.dtype == dtype


# type guard for x.shape == shape
def _is_of_shape(
    x: np.ndarray[tuple[int, ...], np.dtype[T]], shape: Si
) -> TypeGuard[np.ndarray[Si, np.dtype[T]]]:
    return x.shape == shape


# wrapper around np.finfo(dtype).max that also works for numpy_quaddtype
def _floating_max(dtype: np.dtype[F]) -> F:
    if isinstance(_float128_max, dtype.type):
        return _float128_max
    return dtype.type(np.finfo(dtype).max)


# wrapper around np.finfo(dtype).smallest_subnormal that also works for
#  numpy_quaddtype
def _floating_smallest_subnormal(dtype: np.dtype[F]) -> F:
    if isinstance(_float128_smallest_subnormal, dtype.type):
        return _float128_smallest_subnormal
    return dtype.type(np.finfo(dtype).smallest_subnormal)


# wrapper around np.pi, of the dtype, that also works for numpy_quaddtype
def _pi(dtype: np.dtype[F]) -> F:
    if isinstance(_float128_pi, dtype.type):
        return _float128_pi
    return dtype.type(np.pi)


# wrapper around np.e, of the dtype, that also works for numpy_quaddtype
def _e(dtype: np.dtype[F]) -> F:
    if isinstance(_float128_e, dtype.type):
        return _float128_e
    return dtype.type(np.e)
