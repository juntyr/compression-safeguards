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
    "_asinh",
    "_floating_max",
    "_floating_smallest_subnormal",
    "_pi",
    "_e",
]

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
from .typing import F, S, T


# wrapper around np.isnan that also works for numpy_quaddtype
@np.errstate(invalid="ignore")
def _isnan(
    a: int | float | np.ndarray[S, np.dtype[T]],
) -> bool | np.ndarray[S, np.dtype[np.bool]]:
    if (type(a) is not _float128_type) and (
        not isinstance(a, np.ndarray) or a.dtype != _float128_dtype
    ):
        return np.isnan(a)  # type: ignore
    return ~np.array(np.abs(a) <= np.inf)  # type: ignore


# wrapper around np.isinf that also works for numpy_quaddtype
@np.errstate(invalid="ignore")
def _isinf(
    a: int | float | np.ndarray[S, np.dtype[T]],
) -> bool | np.ndarray[S, np.dtype[np.bool]]:
    if (type(a) is not _float128_type) and (
        not isinstance(a, np.ndarray) or a.dtype != _float128_dtype
    ):
        return np.isinf(a)  # type: ignore
    return np.abs(a) == np.inf  # type: ignore


# wrapper around np.isfinite that also works for numpy_quaddtype
@np.errstate(invalid="ignore")
def _isfinite(
    a: int | float | np.ndarray[S, np.dtype[T]],
) -> bool | np.ndarray[S, np.dtype[np.bool]]:
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
    return np.where(_isnan(a), _float128(0), a)  # type: ignore


# wrapper around np.nan_to_num that also works for numpy_quaddtype
@np.errstate(invalid="ignore")
def _nan_to_zero_inf_to_finite(
    a: np.ndarray[S, np.dtype[T]],
) -> np.ndarray[S, np.dtype[T]]:
    if (type(a) is not _float128_type) and (
        not isinstance(a, np.ndarray) or a.dtype != _float128_dtype
    ):
        return np.nan_to_num(a, nan=0, posinf=None, neginf=None)  # type: ignore
    return np.where(
        _isnan(a),
        _float128(0),
        np.where(a == -np.inf, _float128_min, np.where(a == np.inf, _float128_max, a)),
    )  # type: ignore


# wrapper around np.sign that also works for numpy_quaddtype
@np.errstate(invalid="ignore")
def _sign(a: np.ndarray[S, np.dtype[T]]) -> np.ndarray[S, np.dtype[T]]:
    if (type(a) is not _float128_type) and (
        not isinstance(a, np.ndarray) or a.dtype != _float128_dtype
    ):
        return np.sign(a)  # type: ignore
    return np.where(_isnan(a), a, np.where(a == 0, 0, np.where(a < 0, -1, +1)))  # type: ignore


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

    a = np.array(a)
    b = np.array(b, dtype=a.dtype)

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
        res = np.mod(res + q, q)
    return np.subtract(res, q2)


# wrapper around np.rint(a) that also works for numpy_quaddtype
def _rint(a: np.ndarray[S, np.dtype[F]]) -> np.ndarray[S, np.dtype[F]]:
    if (type(a) is not _float128_type) and (
        not isinstance(a, np.ndarray) or a.dtype != _float128_dtype
    ):
        return np.rint(a)  # type: ignore

    a = np.array(a)

    halfway = np.trunc(a) + np.where(a < 0, a.dtype.type(-0.5), a.dtype.type(0.5))

    return np.where(  # type: ignore
        a == halfway,
        # we trust numpy_quaddtype on actual halfway cases
        np.rint(a),
        np.where(a < halfway, np.floor(a), np.ceil(a)),
    )


# wrapper around np.sinh(a) that also works for numpy_quaddtype
def _sinh(a: np.ndarray[S, np.dtype[F]]) -> np.ndarray[S, np.dtype[F]]:
    if (type(a) is not _float128_type) and (
        not isinstance(a, np.ndarray) or a.dtype != _float128_dtype
    ):
        return np.sinh(a)  # type: ignore

    sinh = (np.exp(a) - np.exp(-a)) / 2

    # propagate sinh(-0.0) = -0.0
    return np.where(sinh == a, a, sinh)  # type: ignore


# wrapper around np.asinh(a) that also works for numpy_quaddtype
def _asinh(a: np.ndarray[S, np.dtype[F]]) -> np.ndarray[S, np.dtype[F]]:
    if (type(a) is not _float128_type) and (
        not isinstance(a, np.ndarray) or a.dtype != _float128_dtype
    ):
        return np.asinh(a)  # type: ignore

    asinh = np.log(a + np.sqrt(np.square(a) + 1))

    # propagate asinh(-0.0) = -0.0
    return np.where(asinh == a, a, asinh)  # type: ignore


# wrapper around np.finfo(dtype).max that also works for numpy_quaddtype
def _floating_max(dtype: np.dtype[F]) -> F:
    if dtype == _float128_dtype:
        return _float128_max  # type: ignore
    return np.finfo(dtype).max  # type: ignore


# wrapper around np.finfo(dtype).smallest_subnormal that also works for
#  numpy_quaddtype
def _floating_smallest_subnormal(dtype: np.dtype[F]) -> F:
    if dtype == _float128_dtype:
        return _float128_smallest_subnormal  # type: ignore
    return np.finfo(dtype).smallest_subnormal  # type: ignore


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
