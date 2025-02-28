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


class _Minimum:
    def __le__(self, interval) -> IndexedInterval:
        if not isinstance(interval, IndexedInterval):
            return NotImplemented

        # FIXME
        assert np.issubdtype(interval._lower.dtype, np.integer), (
            "only integer intervals supported for now"
        )

        interval._lower[interval._index] = np.iinfo(interval._lower.dtype).min

        return interval


Minimum = _Minimum()


class _Maximum:
    def __ge__(self, interval) -> IndexedInterval:
        if not isinstance(interval, IndexedInterval):
            return NotImplemented

        # FIXME
        assert np.issubdtype(interval._upper.dtype, np.integer), (
            "only integer intervals supported for now"
        )

        interval._upper[interval._index] = np.iinfo(interval._upper.dtype).max

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

        if not (
            (self._lower.shape == ())
            or (self._lower.shape == interval._lower[interval._index].shape)
        ):
            return NotImplemented

        interval._lower[interval._index] = self._lower

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

        if not (
            (self._upper.shape == ())
            or (self._upper.shape == interval._upper[interval._index].shape)
        ):
            return NotImplemented

        interval._upper[interval._index] = self._upper

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
        # FIXME
        assert np.issubdtype(dtype, np.integer), (
            "only integer intervals supported for now"
        )

        info = np.iinfo(dtype)

        return IntervalUnion(
            _lower=np.full((u, n), info.max, dtype=dtype),
            _upper=np.full((u, n), info.min, dtype=dtype),
        )

    def intersect(self, other: "IntervalUnion[T, N, V]") -> "IntervalUnion[T, N, Any]":
        ((u, n), (v, _)) = self._lower.shape, other._lower.shape

        out: IntervalUnion[T, N, Any] = IntervalUnion.empty(
            self._lower.dtype, n, max(u, v)
        )
        n_intervals = np.zeros(n, dtype=int)

        for i in range(u):
            for j in range(v):
                intersection_lower = np.maximum(self._lower[i], other._lower[j])
                intersection_upper = np.minimum(self._upper[i], other._upper[j])

                has_intersection = intersection_lower <= intersection_upper

                out._lower[n_intervals, has_intersection] = intersection_lower[
                    has_intersection
                ]
                out._upper[n_intervals, has_intersection] = intersection_upper[
                    has_intersection
                ]

                n_intervals += has_intersection

        assert np.amin(n_intervals) > 0, "intersection must not be empty"

        uv = np.amax(n_intervals)

        return IntervalUnion(_lower=out._lower[:uv], _upper=out._upper[:uv])  # type: ignore

    def __repr__(self) -> str:
        return f"IntervalUnion(lower={self._lower!r}, upper={self._upper!r})"
