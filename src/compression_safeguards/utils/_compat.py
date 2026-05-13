"""
Private compatibility functions, mostly wrappers around numpy, that ensure
equivalent behaviour for all supported dtypes and provide good type hints.
"""

__all__ = [
    "_place",
    "_round_ties_even",
    "_floor_modulo",
    "_ceil_modulo",
    "_trunc_modulo",
    "_round_ties_even_modulo",
    "_euclidean_modulo",
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
    "_floating_pi",
    "_floating_e",
]

from collections.abc import Sequence
from typing import Literal, TypeGuard, TypeVar, overload

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from ._float128 import _float128_dtype, _float128_e, _float128_pi, _float128_type
from .typing import TB, F, Fi, S, Si, T, Ti

N = TypeVar("N", bound=int, covariant=True)
""" Any [`int`][int] (covariant). """


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


# round to the nearest integer, with ties to the nearest even integer
_round_ties_even = np.rint


# modulo based on floor division, p - q*floor(p/q),
#  which guarantees that the result is in (-q, -0] for q<0 and [+0, +q) for q>0
@overload
def _floor_modulo(
    p: np.ndarray[S, np.dtype[F]], q: np.ndarray[S, np.dtype[F]]
) -> np.ndarray[S, np.dtype[F]]: ...


@overload
def _floor_modulo(p: Fi, q: Fi) -> Fi: ...


def _floor_modulo(p, q):
    out = _ensure_array(np.mod(p, q))
    np.copysign(out, q, out=out)
    return out


# modulo based on ceil division, p - q*ceil(p/q),
#  which guarantees that the result is in [+0, +q) for q<0 and (-q, -0] for q>0
@overload
def _ceil_modulo(
    p: np.ndarray[S, np.dtype[F]], q: np.ndarray[S, np.dtype[F]]
) -> np.ndarray[S, np.dtype[F]]: ...


@overload
def _ceil_modulo(p: Fi, q: Fi) -> Fi: ...


def _ceil_modulo(p, q):
    out = _ensure_array(_floor_modulo(p, q))
    out -= q
    np.fmod(out, q, out=out)  # ensure that |out| < |q|
    np.copyto(out, p, where=(np.isfinite(p) & np.isinf(q) & (np.sign(p) != np.sign(q))))
    np.copyto(
        out, -q, where=(np.isfinite(p) & np.isinf(q) & (np.sign(p) == np.sign(q)))
    )
    np.copysign(out, -q, out=out)
    return out


# modulo based on trunc division, p - q*trunc(p/q),
#  which guarantees that the result is in [-q, -0] for p<0 and [+0, +q] for p>0
@overload
def _trunc_modulo(
    p: np.ndarray[S, np.dtype[F]], q: np.ndarray[S, np.dtype[F]]
) -> np.ndarray[S, np.dtype[F]]: ...


@overload
def _trunc_modulo(p: Fi, q: Fi) -> Fi: ...


def _trunc_modulo(p, q):
    out = _ensure_array(np.fmod(p, q))
    np.copysign(out, p, out=out)
    return out


# modulo based on rounded division, p - q*round(p/q),
#  which guarantees that the result is in [-q/2, q/2]
@overload
def _round_ties_even_modulo(
    p: np.ndarray[S, np.dtype[F]], q: np.ndarray[S, np.dtype[F]]
) -> np.ndarray[S, np.dtype[F]]: ...


@overload
def _round_ties_even_modulo(p: Fi, q: Fi) -> Fi: ...


def _round_ties_even_modulo(p, q):
    q2 = np.divide(q, 2)
    out = _ensure_array(np.add(p, q2))
    np.mod(out, q, out=out)
    np.subtract(out, q2, out=out)
    np.copyto(out, p, where=(np.isfinite(p) & np.isinf(q)))
    np.copyto(out, p, where=((p == 0) & ~np.isnan(q) & (q != 0)))
    return out


# modulo based on Euclidean division, p - |q|*floor(p/|q|),
#  which guarantees that the result is in [+0, +q)
@overload
def _euclidean_modulo(
    p: np.ndarray[S, np.dtype[F]], q: np.ndarray[S, np.dtype[F]]
) -> np.ndarray[S, np.dtype[F]]: ...


@overload
def _euclidean_modulo(p: Fi, q: Fi) -> Fi: ...


def _euclidean_modulo(p, q):
    return _floor_modulo(p, np.abs(q))


# wrapper around np.minimum that also works for +0.0 and -0.0
@overload
def _minimum_zero_sign_sensitive(
    a: Ti,
    b: np.ndarray[S, np.dtype[Ti]],
    out: None | np.ndarray[S, np.dtype[Ti]] = None,
    where: None | np.ndarray[S, np.dtype[np.bool]] = None,
) -> np.ndarray[S, np.dtype[Ti]]: ...


@overload
def _minimum_zero_sign_sensitive(
    a: np.ndarray[S, np.dtype[Ti]],
    b: Ti,
    out: None | np.ndarray[S, np.dtype[Ti]] = None,
    where: None | np.ndarray[S, np.dtype[np.bool]] = None,
) -> np.ndarray[S, np.dtype[Ti]]: ...


@overload
def _minimum_zero_sign_sensitive(
    a: np.ndarray[S, np.dtype[T]],
    b: np.ndarray[S, np.dtype[T]],
    out: None | np.ndarray[S, np.dtype[T]] = None,
    where: None | np.ndarray[S, np.dtype[np.bool]] = None,
) -> np.ndarray[S, np.dtype[T]]: ...


def _minimum_zero_sign_sensitive(a, b, out=None, where=None):
    def maybe_where(w):
        return w if where is None else where & w

    a = _ensure_array(a)
    b = _ensure_array(b)

    both_zero = (a == 0) & (b == 0)
    minimum = _ensure_array(np.minimum(a, b, out=out, where=maybe_where(~both_zero)))

    if np.issubdtype(a.dtype, np.integer) and np.issubdtype(b.dtype, np.integer):
        minimum[maybe_where(both_zero)] = 0
        return minimum

    a = _broadcast_to(a.astype(minimum.dtype, casting="safe"), minimum.shape)
    b = _broadcast_to(b.astype(minimum.dtype, casting="safe"), minimum.shape)

    signbit_a = np.signbit(a)
    signbit_b = np.signbit(b)

    np.copyto(
        minimum,
        a,
        where=maybe_where(both_zero & (signbit_a >= signbit_b)),
        casting="no",
    )
    np.copyto(
        minimum, b, where=maybe_where(both_zero & (signbit_a < signbit_b)), casting="no"
    )

    return minimum


# wrapper around np.maximum that also works for +0.0 and -0.0
@overload
def _maximum_zero_sign_sensitive(
    a: Ti,
    b: np.ndarray[S, np.dtype[Ti]],
    out: None | np.ndarray[S, np.dtype[Ti]] = None,
    where: None | np.ndarray[S, np.dtype[np.bool]] = None,
) -> np.ndarray[S, np.dtype[Ti]]: ...


@overload
def _maximum_zero_sign_sensitive(
    a: np.ndarray[S, np.dtype[Ti]],
    b: Ti,
    out: None | np.ndarray[S, np.dtype[Ti]] = None,
    where: None | np.ndarray[S, np.dtype[np.bool]] = None,
) -> np.ndarray[S, np.dtype[Ti]]: ...


@overload
def _maximum_zero_sign_sensitive(
    a: np.ndarray[S, np.dtype[T]],
    b: np.ndarray[S, np.dtype[T]],
    out: None | np.ndarray[S, np.dtype[T]] = None,
    where: None | np.ndarray[S, np.dtype[np.bool]] = None,
) -> np.ndarray[S, np.dtype[T]]: ...


def _maximum_zero_sign_sensitive(a, b, out=None, where=None):
    def maybe_where(w):
        return w if where is None else where & w

    a = _ensure_array(a)
    b = _ensure_array(b)

    both_zero = (a == 0) & (b == 0)
    maximum = _ensure_array(np.maximum(a, b, out=out, where=maybe_where(~both_zero)))

    if np.issubdtype(a.dtype, np.integer) and np.issubdtype(b.dtype, np.integer):
        maximum[maybe_where(both_zero)] = 0
        return maximum

    a = _broadcast_to(a.astype(maximum.dtype, casting="safe"), maximum.shape)
    b = _broadcast_to(b.astype(maximum.dtype, casting="safe"), maximum.shape)

    signbit_a = np.signbit(a)
    signbit_b = np.signbit(b)

    np.copyto(
        maximum,
        a,
        where=maybe_where(both_zero & (signbit_a <= signbit_b)),
        casting="no",
    )
    np.copyto(
        maximum, b, where=maybe_where(both_zero & (signbit_a > signbit_b)), casting="no"
    )

    return maximum


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


# wrapper around np.pi, of the dtype, that also works for numpy_quaddtype
def _floating_pi(dtype: np.dtype[F]) -> F:
    if isinstance(_float128_pi, dtype.type):
        return _float128_pi
    return dtype.type(np.pi)


# wrapper around np.e, of the dtype, that also works for numpy_quaddtype
def _floating_e(dtype: np.dtype[F]) -> F:
    if isinstance(_float128_e, dtype.type):
        return _float128_e
    return dtype.type(np.e)
