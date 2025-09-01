"""
Private compatibility functions, mostly wrappers around numpy, that ensure
equivalent behaviour for all supported dtypes, e.g. numpy_quaddtype.
"""

__all__ = [
    "_isnan",
    "_isinf",
    "_isfinite",
    "_nan_to_zero",
    "_nan_to_zero_inf_to_finite",
    "_sign",
    "_nextafter",
    "_reciprocal",
    "_symmetric_modulo",
    "_rint",
    "_sinh",
    "_cosh",
    "_tanh",
    "_asinh",
    "_acosh",
    "_atanh",
    "_signbit_non_nan",
    "_minimum",
    "_maximum",
    "_where",
    "_broadcast_to",
    "_is_negative",
    "_is_positive",
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


# wrapper around np.isnan that also works for numpy_quaddtype
@overload
def _isnan(
    a: int | float,
) -> bool:
    pass


@overload
def _isnan(
    a: np.ndarray[S, np.dtype[T]],
) -> np.ndarray[S, np.dtype[np.bool]]:
    pass


@np.errstate(invalid="ignore")
def _isnan(a):
    if (type(a) is not _float128_type) and (
        not isinstance(a, np.ndarray) or a.dtype != _float128_dtype
    ):
        return np.isnan(a)  # type: ignore
    return ~np.array(np.abs(a) <= np.inf, copy=None)  # type: ignore


# wrapper around np.isinf that also works for numpy_quaddtype
@overload
def _isinf(
    a: int | float,
) -> bool:
    pass


@overload
def _isinf(
    a: np.ndarray[S, np.dtype[T]],
) -> np.ndarray[S, np.dtype[np.bool]]:
    pass


@np.errstate(invalid="ignore")
def _isinf(a):
    if (type(a) is not _float128_type) and (
        not isinstance(a, np.ndarray) or a.dtype != _float128_dtype
    ):
        return np.isinf(a)  # type: ignore
    return np.abs(a) == np.inf  # type: ignore


# wrapper around np.isfinite that also works for numpy_quaddtype
@overload
def _isfinite(
    a: int | float,
) -> bool:
    pass


@overload
def _isfinite(
    a: np.ndarray[S, np.dtype[T]],
) -> np.ndarray[S, np.dtype[np.bool]]:
    pass


@np.errstate(invalid="ignore")
def _isfinite(a):
    if (type(a) is not _float128_type) and (
        not isinstance(a, np.ndarray) or a.dtype != _float128_dtype
    ):
        return np.isfinite(a)  # type: ignore
    return np.abs(a) < np.inf  # type: ignore


# wrapper around np.nan_to_num that also works for numpy_quaddtype
@np.errstate(invalid="ignore")
def _nan_to_zero(a: np.ndarray[S, np.dtype[T]]) -> np.ndarray[S, np.dtype[T]]:
    if (type(a) is not _float128_type) and (
        not isinstance(a, np.ndarray) or a.dtype != _float128_dtype
    ):
        return np.nan_to_num(a, nan=0, posinf=np.inf, neginf=-np.inf)  # type: ignore
    out: np.ndarray[S, np.dtype[T]] = np.array(a, copy=True)
    out[_isnan(a)] = 0
    return out


# wrapper around np.nan_to_num that also works for numpy_quaddtype
@np.errstate(invalid="ignore")
def _nan_to_zero_inf_to_finite(
    a: np.ndarray[S, np.dtype[T]],
) -> np.ndarray[S, np.dtype[T]]:
    if (type(a) is not _float128_type) and (
        not isinstance(a, np.ndarray) or a.dtype != _float128_dtype
    ):
        return np.nan_to_num(a, nan=0, posinf=None, neginf=None)  # type: ignore
    out: np.ndarray[S, np.dtype[T]] = np.array(a, copy=True)
    out[_isnan(a)] = 0
    out[a == -np.inf] = _float128_min
    out[a == np.inf] = _float128_max
    return out


# wrapper around np.sign that also works for numpy_quaddtype
@np.errstate(invalid="ignore")
def _sign(a: np.ndarray[S, np.dtype[T]]) -> np.ndarray[S, np.dtype[T]]:
    if (type(a) is not _float128_type) and (
        not isinstance(a, np.ndarray) or a.dtype != _float128_dtype
    ):
        return np.sign(a)  # type: ignore
    out: np.ndarray[S, np.dtype[T]] = np.array(a, copy=True)
    out[a == 0] = 0
    out[a < 0] = -1
    out[a > 0] = +1
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

    _float128_neg_zero = -_float128(0)

    _float128_incr_subnormal = np.where(
        a < 0, -_float128_smallest_subnormal, _float128_smallest_subnormal
    )

    abs_a_zero_mantissa = (
        (
            np.atleast_1d(a).view(np.uint8)
            & np.atleast_1d(np.full_like(a, np.inf)).view(np.uint8)
        )
        .view(a.dtype)
        .reshape(a.shape)
    )
    _float128_incr_normal = abs_a_zero_mantissa / (_float128(2) ** 112)

    r = np.where(
        _isnan(a) | _isnan(b),
        a + b,  # unordered, at least one is NaN
        np.where(
            a == b,
            b,  # equal
            np.where(
                _isinf(a),
                np.where(a < 0, _float128_min, _float128_max),  # infinity
                np.where(
                    np.abs(a) > _float128_smallest_normal,
                    # note: implementation deviates here since numpy_quaddtype
                    #       divides/multiplies with error
                    #       based on https://stackoverflow.com/a/70512041
                    np.where(
                        a < b,
                        np.where(
                            a == -abs_a_zero_mantissa,
                            a + _float128_incr_normal / 2,
                            a + _float128_incr_normal,
                        ),
                        np.where(
                            a == abs_a_zero_mantissa,
                            a - _float128_incr_normal / 2,
                            a - _float128_incr_normal,
                        ),
                    ),  # normal
                    np.where(  # zero, subnormal, or smallest normal
                        (a < b) == (a >= 0),
                        (a + _float128_incr_subnormal),
                        np.where(
                            a == (-_float128_smallest_subnormal),
                            _float128_neg_zero,
                            a - _float128_incr_subnormal,
                        ),
                    ),
                ),
            ),
        ),
    )

    return r  # type: ignore


# wrapper around np.reciprocal that also works for numpy_quaddtype
@np.errstate(invalid="ignore")
def _reciprocal(a: np.ndarray[S, np.dtype[F]]) -> np.ndarray[S, np.dtype[F]]:
    if (type(a) is not _float128_type) and (
        not isinstance(a, np.ndarray) or a.dtype != _float128_dtype
    ):
        return np.reciprocal(a)
    return np.divide(1, a)


# wrapper around np.mod(p, q) that guarantees that the result is in [-q/2, q/2]
def _symmetric_modulo(
    p: np.ndarray[S, np.dtype[F]], q: np.ndarray[S, np.dtype[F]]
) -> np.ndarray[S, np.dtype[F]]:
    q2: np.ndarray[S, np.dtype[F]] = np.divide(q, 2)
    res: np.ndarray[S, np.dtype[F]] = np.mod(p + q2, q)
    if (type(p) is _float128_type) or (p.dtype == _float128_dtype):
        res += q
        np.mod(res, q, out=res)
    return np.subtract(res, q2)


# wrapper around np.rint(a) that also works for numpy_quaddtype
def _rint(a: np.ndarray[S, np.dtype[F]]) -> np.ndarray[S, np.dtype[F]]:
    if (type(a) is not _float128_type) and (
        not isinstance(a, np.ndarray) or a.dtype != _float128_dtype
    ):
        return np.rint(a)  # type: ignore

    halfway = np.array(np.trunc(a), copy=None)
    halfway[_is_negative(a)] -= 0.5
    halfway[_is_positive(a)] += 0.5

    out: np.ndarray[S, np.dtype[F]] = np.array(np.rint(a), copy=None)
    np.copyto(out, np.floor(a), where=(a < halfway), casting="no")
    np.copyto(out, np.ceil(a), where=(a > halfway), casting="no")
    return out


# wrapper around np.sinh(a) that also works for numpy_quaddtype
def _sinh(a: np.ndarray[S, np.dtype[F]]) -> np.ndarray[S, np.dtype[F]]:
    if (type(a) is not _float128_type) and (
        not isinstance(a, np.ndarray) or a.dtype != _float128_dtype
    ):
        return np.sinh(a)  # type: ignore

    # propagate sinh(-0.0) = -0.0
    out: np.ndarray[S, np.dtype[F]] = np.array((np.exp(a) - np.exp(-a)) / 2, copy=None)
    np.copyto(out, a, where=(out == a), casting="no")
    return out


# wrapper around np.cosh(a) that also works for numpy_quaddtype
def _cosh(a: np.ndarray[S, np.dtype[F]]) -> np.ndarray[S, np.dtype[F]]:
    if (type(a) is not _float128_type) and (
        not isinstance(a, np.ndarray) or a.dtype != _float128_dtype
    ):
        return np.cosh(a)  # type: ignore

    return np.divide(np.add(np.exp(a), np.exp(-a)), 2)


# wrapper around np.tanh(a) that also works for numpy_quaddtype
def _tanh(a: np.ndarray[S, np.dtype[F]]) -> np.ndarray[S, np.dtype[F]]:
    if (type(a) is not _float128_type) and (
        not isinstance(a, np.ndarray) or a.dtype != _float128_dtype
    ):
        return np.tanh(a)  # type: ignore

    # explicitly set tanh(huge) = 1 where overflow to inf/inf = NaN might occur
    # propagate tanh(-0.0) = -0.0
    expa: np.ndarray[S, np.dtype[F]] = np.exp(a * 2)
    out: np.ndarray[S, np.dtype[F]] = np.array(
        np.divide(np.subtract(expa, 1), np.add(expa, 1)), copy=None
    )
    out[expa == np.inf] = 1
    np.copyto(out, a, where=(out == a), casting="no")
    return out


# wrapper around np.asinh(a) that also works for numpy_quaddtype
def _asinh(a: np.ndarray[S, np.dtype[F]]) -> np.ndarray[S, np.dtype[F]]:
    if (type(a) is not _float128_type) and (
        not isinstance(a, np.ndarray) or a.dtype != _float128_dtype
    ):
        return np.asinh(a)  # type: ignore

    # evaluate asinh(abs(a)) first since it overflows correctly in
    #  sqrt(square(a)), then copy the sign of the input a
    # propagate asinh(-0.0) = -0.0
    out: np.ndarray[S, np.dtype[F]] = np.array(
        np.log(np.abs(a) + np.sqrt(np.square(np.abs(a)) + 1)), copy=None
    )
    np.negative(out, out=out, where=(a < 0))
    np.copyto(out, a, where=(out == a), casting="no")
    return out


# wrapper around np.acosh(a) that also works for numpy_quaddtype
def _acosh(a: np.ndarray[S, np.dtype[F]]) -> np.ndarray[S, np.dtype[F]]:
    if (type(a) is not _float128_type) and (
        not isinstance(a, np.ndarray) or a.dtype != _float128_dtype
    ):
        return np.acosh(a)  # type: ignore

    # explicitly set acosh(<1) = NaN to avoid overflow in sqrt(square(a) - 1)
    out: np.ndarray[S, np.dtype[F]] = np.array(
        np.log(a + np.sqrt(np.square(a) - 1)), copy=None
    )
    out[a < 1] = np.nan
    return out


# wrapper around np.atanh(a) that also works for numpy_quaddtype
def _atanh(a: np.ndarray[S, np.dtype[F]]) -> np.ndarray[S, np.dtype[F]]:
    if (type(a) is not _float128_type) and (
        not isinstance(a, np.ndarray) or a.dtype != _float128_dtype
    ):
        return np.atanh(a)  # type: ignore

    # propagate atanh(-0.0) = -0.0
    out: np.ndarray[S, np.dtype[F]] = np.array(
        (np.log(1 + a) - np.log(1 - a)) / 2, copy=None
    )
    np.copyto(out, a, where=(out == a), casting="no")
    return out


# wrapper around np.signbit(a) that also works for numpy_quaddtype, but only
#  for non-NaN values
def _signbit_non_nan(a: np.ndarray[S, np.dtype[T]]) -> np.ndarray[S, np.dtype[np.bool]]:
    if (type(a) is not _float128_type) and (
        not isinstance(a, np.ndarray) or a.dtype != _float128_dtype
    ):
        return np.signbit(a)  # type: ignore
    return (a < 0) | (_reciprocal(a) < 0)  # type: ignore


# wrapper around np.minimum that also works for +0.0 and -0.0
@overload
def _minimum(a: Ti, b: np.ndarray[S, np.dtype[Ti]]) -> np.ndarray[S, np.dtype[Ti]]:
    pass


@overload
def _minimum(a: np.ndarray[S, np.dtype[Ti]], b: Ti) -> np.ndarray[S, np.dtype[Ti]]:
    pass


@overload
def _minimum(
    a: np.ndarray[S, np.dtype[T]], b: np.ndarray[S, np.dtype[T]]
) -> np.ndarray[S, np.dtype[T]]:
    pass


def _minimum(a, b):
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
        where=(_signbit_non_nan(a) > _signbit_non_nan(b)),
        casting="no",
    )
    np.copyto(
        minimum_array,
        b,
        where=(_signbit_non_nan(a) < _signbit_non_nan(b)),
        casting="no",
    )
    return minimum_array


# wrapper around np.maximum that also works for +0.0 and -0.0
@overload
def _maximum(a: Ti, b: np.ndarray[S, np.dtype[Ti]]) -> np.ndarray[S, np.dtype[Ti]]:
    pass


@overload
def _maximum(a: np.ndarray[S, np.dtype[Ti]], b: Ti) -> np.ndarray[S, np.dtype[Ti]]:
    pass


@overload
def _maximum(
    a: np.ndarray[S, np.dtype[T]], b: np.ndarray[S, np.dtype[T]]
) -> np.ndarray[S, np.dtype[T]]:
    pass


def _maximum(a, b):
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
        where=(_signbit_non_nan(a) < _signbit_non_nan(b)),
        casting="no",
    )
    np.copyto(
        maximum_array,
        b,
        where=(_signbit_non_nan(a) > _signbit_non_nan(b)),
        casting="no",
    )
    return maximum_array


# wrapper around np.where but with better type hints
@overload
def _where(
    cond: np.ndarray[S, np.dtype[np.bool]],
    a: np.ndarray[S, np.dtype[T]],
    b: np.ndarray[S, np.dtype[T]],
) -> np.ndarray[S, np.dtype[T]]:
    pass


@overload
def _where(
    cond: np.ndarray[S, np.dtype[np.bool]],
    a: np.ndarray[S, np.dtype[np.bool]],
    b: np.ndarray[S, np.dtype[np.bool]],
) -> np.ndarray[S, np.dtype[np.bool]]:
    pass


@overload
def _where(
    cond: np.ndarray[S, np.dtype[np.bool]], a: Ti, b: np.ndarray[S, np.dtype[Ti]]
) -> np.ndarray[S, np.dtype[Ti]]:
    pass


@overload
def _where(
    cond: np.ndarray[S, np.dtype[np.bool]], a: np.ndarray[S, np.dtype[Ti]], b: Ti
) -> np.ndarray[S, np.dtype[Ti]]:
    pass


@overload
def _where(
    cond: np.ndarray[S, np.dtype[np.bool]], a: Ti, b: Ti
) -> np.ndarray[S, np.dtype[Ti]]:
    pass


@overload
def _where(cond: bool, a: Ti, b: Ti) -> Ti:
    pass


def _where(cond, a, b):
    return np.where(cond, a, b)


# wrapper around np.broadcast_to but with better type hints
@overload
def _broadcast_to(
    a: np.ndarray[tuple[int, ...], np.dtype[T]], shape: Si
) -> np.ndarray[Si, np.dtype[T]]:
    pass


@overload
def _broadcast_to(
    a: np.ndarray[tuple[int, ...], np.dtype[np.bool]], shape: Si
) -> np.ndarray[Si, np.dtype[np.bool]]:
    pass


@overload
def _broadcast_to(a: Ti, shape: Si) -> np.ndarray[Si, np.dtype[Ti]]:
    pass


def _broadcast_to(a, shape):
    return np.broadcast_to(a, shape)


# wrapper around a < 0 that also works for -0.0 (is negative)
def _is_negative(
    a: np.ndarray[S, np.dtype[F]],
) -> np.ndarray[S, np.dtype[np.bool]]:
    # check not just for a < 0 but also for a == -0.0
    return np.less(a, 0) | np.less(_reciprocal(a), 0)


# wrapper around a > 0 that also works for -0.0 (is not positive)
def _is_positive(
    x: np.ndarray[S, np.dtype[F]],
) -> np.ndarray[S, np.dtype[np.bool]]:
    # check not just for x > 0 but also for x == +0.0
    return np.greater(x, 0) | np.greater(_reciprocal(x), 0)


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
