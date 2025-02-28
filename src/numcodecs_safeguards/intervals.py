from typing import Any, Generic, Literal, TypeVar

import numpy as np

T = TypeVar("T", bound=np.dtype)
N = TypeVar("N", bound=Literal[1])
U = TypeVar("U", bound=Literal[1])
V = TypeVar("V", bound=Literal[1])


class Interval(Generic[T, N]):
    _lower: np.ndarray[tuple[N], T]
    _upper: np.ndarray[tuple[N], T]

    def __init__(
        self,
        *,
        _lower: np.ndarray[tuple[N], T],
        _upper: np.ndarray[tuple[N], T],
    ) -> None:
        self._lower = _lower
        self._upper = _upper

    @staticmethod
    def empty(dtype: T, n: N) -> "Interval[T, N]":
        single = IntervalUnion.empty(dtype, n, 1)
        return Interval(
            _lower=single._lower.reshape(-1),  # type: ignore
            _upper=single._upper.reshape(-1),  # type: ignore
        )

    @staticmethod
    def empty_like(a: np.ndarray[tuple[int, ...], T]) -> "Interval[T, Any]":
        return Interval.empty(a.dtype, a.size)

    def __getitem__(self, key) -> "IndexedInterval[T, Any]":
        return IndexedInterval(_lower=self._lower, _upper=self._upper, _index=key)

    def into_union(self) -> "IntervalUnion[T, N, Literal[1]]":
        return IntervalUnion(
            _lower=self._lower.reshape(1, -1),  # type: ignore
            _upper=self._upper.reshape(1, -1),  # type: ignore
        )

    def __repr__(self) -> str:
        return f"Interval(lower={self._lower!r}, upper={self._upper!r})"


class IndexedInterval(Generic[T, N]):
    _lower: np.ndarray[tuple[N], T]
    _upper: np.ndarray[tuple[N], T]
    _index: Any

    def __init__(
        self,
        *,
        _lower: np.ndarray[tuple[N], T],
        _upper: np.ndarray[tuple[N], T],
        _index: Any,
    ) -> None:
        self._lower = _lower
        self._upper = _upper
        self._index = _index


def _minimum(dtype: np.dtype):
    if np.issubdtype(dtype, np.integer):
        return np.iinfo(dtype).min

    if np.issubdtype(dtype, np.floating):
        btype = dtype.str.replace("f", "u")
        bmin = np.iinfo(btype).max  # produces -NaN (0xffff...)
        return np.array(bmin, dtype=btype).view(dtype)

    raise TypeError(f"unsupported interval type {dtype}")


class _Minimum:
    def __le__(self, interval) -> IndexedInterval:
        if not isinstance(interval, IndexedInterval):
            return NotImplemented

        interval._lower[interval._index] = _minimum(interval._lower.dtype)

        return interval


Minimum = _Minimum()


def _maximum(dtype: np.dtype):
    if np.issubdtype(dtype, np.integer):
        return np.iinfo(dtype).max

    if np.issubdtype(dtype, np.floating):
        btype = dtype.str.replace("f", "u")
        bmin = np.iinfo(btype).max  # produces -NaN (0xffff...)
        return np.copysign(np.array(bmin, dtype=btype).view(dtype), +1)

    raise TypeError(f"unsupported interval type {dtype}")


class _Maximum:
    def __ge__(self, interval) -> IndexedInterval:
        if not isinstance(interval, IndexedInterval):
            return NotImplemented

        interval._upper[interval._index] = _maximum(interval._upper.dtype)

        return interval


Maximum = _Maximum()


class Lower:
    _lower: np.ndarray

    def __init__(self, lower: np.ndarray) -> None:
        self._lower = lower

    def __le__(self, interval) -> IndexedInterval:
        if not isinstance(self._lower, np.ndarray):
            return NotImplemented

        if not isinstance(interval, IndexedInterval):
            return NotImplemented

        # if self._lower.dtype != interval._lower.dtype:
        #     return NotImplemented

        if self._lower.shape == ():
            lower = self._lower
        elif self._lower.shape == interval._lower.shape:
            lower = self._lower[interval._index]
        elif self._lower.shape == interval._lower[interval._index].shape:
            lower = self._lower
        else:
            return NotImplemented

        interval._lower[interval._index] = lower

        return interval


class Upper:
    _upper: np.ndarray

    def __init__(self, upper: np.ndarray) -> None:
        self._upper = upper

    def __ge__(self, interval) -> IndexedInterval:
        if not isinstance(self._upper, np.ndarray):
            return NotImplemented

        if not isinstance(interval, IndexedInterval):
            return NotImplemented

        # if self._upper.dtype != interval._upper.dtype:
        #     return NotImplemented

        if self._upper.shape == ():
            upper = self._upper
        elif self._upper.shape == interval._upper.shape:
            upper = self._upper[interval._index]
        elif self._upper.shape == interval._upper[interval._index].shape:
            upper = self._upper
        else:
            return NotImplemented

        interval._upper[interval._index] = upper

        return interval


class IntervalUnion(Generic[T, N, U]):
    _lower: np.ndarray[tuple[U, N], T]
    _upper: np.ndarray[tuple[U, N], T]

    def __init__(
        self,
        *,
        _lower: np.ndarray[tuple[U, N], T],
        _upper: np.ndarray[tuple[U, N], T],
    ) -> None:
        self._lower = _lower
        self._upper = _upper

    @staticmethod
    def empty(dtype: T, n: N, u: U) -> "IntervalUnion[T, N, U]":
        return IntervalUnion(
            _lower=np.full((u, n), _maximum(dtype), dtype=dtype),
            _upper=np.full((u, n), _minimum(dtype), dtype=dtype),
        )

    def intersect(self, other: "IntervalUnion[T, N, V]") -> "IntervalUnion[T, N, Any]":
        ((u, n), (v, _)) = self._lower.shape, other._lower.shape

        if n == 0:
            return IntervalUnion.empty(
                self._lower.dtype, n, min(u, v)
            )

        out: IntervalUnion[T, N, Any] = IntervalUnion.empty(
            self._lower.dtype, n, max(u, v)
        )
        n_intervals = np.zeros(n, dtype=int)

        for i in range(u):
            for j in range(v):
                intersection_lower = np.maximum(
                    _to_total_order(self._lower[i]), _to_total_order(other._lower[j])
                )
                intersection_upper = np.minimum(
                    _to_total_order(self._upper[i]), _to_total_order(other._upper[j])
                )

                has_intersection = intersection_lower <= intersection_upper

                out._lower[n_intervals[has_intersection], has_intersection] = (
                    _from_total_order(
                        intersection_lower[has_intersection], out._lower.dtype
                    )
                )
                out._upper[n_intervals[has_intersection], has_intersection] = (
                    _from_total_order(
                        intersection_upper[has_intersection], out._upper.dtype
                    )
                )

                n_intervals += has_intersection

        assert np.amin(n_intervals) > 0, "intersection must not be empty"

        uv = np.amax(n_intervals)

        return IntervalUnion(_lower=out._lower[:uv], _upper=out._upper[:uv])  # type: ignore

    def contains(
        self, other: np.ndarray[tuple[N], T]
    ) -> np.ndarray[tuple[N], np.dtype[np.bool]]:
        other = _to_total_order(other)

        (u, n) = self._lower.shape
        is_contained = np.zeros((n,), dtype=bool)

        for i in range(u):
            is_contained |= (other >= _to_total_order(self._lower[i])) & (
                other <= _to_total_order(self._upper[i])
            )

        return is_contained

    def encode(self, decoded: np.ndarray[tuple[N], T]) -> np.ndarray[tuple[N], T]:
        is_contained = self.contains(decoded)

        # simple encoding:
        #  1. if decoded is in the interval, use it
        #  2. otherwise pick the lower bound of the first interval
        encoding_pick = self._lower[0].copy()
        encoding_pick[is_contained] = decoded[is_contained]

        return encoding_pick

    def __repr__(self) -> str:
        return f"IntervalUnion(lower={self._lower!r}, upper={self._upper!r})"


def _to_total_order(x: np.ndarray) -> np.ndarray:
    """
    FloatFlip in http://stereopsis.com/radix.html
    """

    if np.issubdtype(x.dtype, np.integer):
        return x

    if not np.issubdtype(x.dtype, np.floating):
        raise TypeError(f"unsupported interval type {x.dtype}")

    utype = x.dtype.str.replace("f", "u")
    itype = x.dtype.str.replace("f", "i")
    bits = np.iinfo(utype).bits

    mask = (-((x.view(dtype=utype) >> (bits - 1)).view(dtype=itype))).view(
        dtype=utype
    ) | (np.array(1, dtype=utype) << (bits - 1))

    return x.view(dtype=utype) ^ mask


def _from_total_order(x: np.ndarray, dtype: np.dtype) -> np.ndarray:
    """
    IFloatFlip in http://stereopsis.com/radix.html
    """

    assert np.issubdtype(x.dtype, np.integer)

    if np.issubdtype(dtype, np.integer):
        return x

    if not np.issubdtype(dtype, np.floating):
        raise TypeError(f"unsupported interval type {dtype}")

    utype = dtype.str.replace("f", "u")
    itype = dtype.str.replace("f", "i")
    bits = np.iinfo(utype).bits

    mask = ((x >> (bits - 1)).view(dtype=itype) - 1).view(dtype=utype) | (
        np.array(1, dtype=utype) << (bits - 1)
    )

    return (x ^ mask).view(dtype=dtype)
