from dataclasses import dataclass
from typing import Any, Generic, Literal, TypeVar

import numpy as np

T = TypeVar("T", bound=np.dtype)
N = TypeVar("N", bound=Literal[1])
U = TypeVar("U", bound=Literal[1])
V = TypeVar("V", bound=Literal[1])


@dataclass
class IntervalUnion(Generic[T, N, U]):
    lower: np.ndarray[tuple[U, N], T]
    upper: np.ndarray[tuple[U, N], T]

    # @staticmethod
    # def from_single(lower: np.ndarray[tuple[N], T], upper: np.ndarray[tuple[N], T]) -> "IntervalUnion[T, N, Literal[1]]":
    #     return IntervalUnion(lower=lower.reshape((1, -1)), upper=upper.reshape((1, -1)))

    @staticmethod
    def empty(dtype: T, n: N, u: U) -> "IntervalUnion[T, N, U]":
        # FIXME
        assert np.issubdtype(dtype, np.integer), (
            "only integer intervals supported for now"
        )

        info = np.iinfo(dtype)

        return IntervalUnion(
            lower=np.full((u, n), info.max, dtype=dtype),
            upper=np.full((u, n), info.min, dtype=dtype),
        )

    @staticmethod
    def full(dtype: T, n: N, u: U) -> "IntervalUnion[T, N, U]":
        # FIXME
        assert np.issubdtype(dtype, np.integer), (
            "only integer intervals supported for now"
        )

        info = np.iinfo(dtype)

        return IntervalUnion(
            lower=np.full((u, n), info.min, dtype=dtype),
            upper=np.full((u, n), info.max, dtype=dtype),
        )

    @staticmethod
    def from_singular(a: np.ndarray[tuple[N], T]) -> "IntervalUnion[T, N, Literal[1]]":
        return IntervalUnion(
            lower=a.reshape((1, -1)).copy(),  # type: ignore
            upper=a.reshape((1, -1)).copy(),  # type: ignore
        )

    def intersect(self, other: "IntervalUnion[T, N, V]") -> "IntervalUnion[T, N, Any]":
        ((u, n), (v, _)) = self.lower.shape, other.lower.shape

        out: IntervalUnion[T, N, Any] = IntervalUnion.empty(
            self.lower.dtype, n, max(u, v)
        )
        n_intervals = np.zeros(n, dtype=int)

        for i in range(u):
            for j in range(v):
                intersection_lower = np.maximum(self.lower[i], other.lower[j])
                intersection_upper = np.minimum(self.upper[i], other.upper[j])

                has_intersection = intersection_lower <= intersection_upper

                out.lower[n_intervals, has_intersection] = intersection_lower[
                    has_intersection
                ]
                out.upper[n_intervals, has_intersection] = intersection_upper[
                    has_intersection
                ]

                n_intervals += has_intersection

        assert np.amin(n_intervals) > 0, "intersection must not be empty"

        uv = np.amax(n_intervals)

        return IntervalUnion(lower=out.lower[:uv], upper=out.upper[:uv])  # type: ignore
