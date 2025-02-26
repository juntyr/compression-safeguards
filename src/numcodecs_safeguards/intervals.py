from dataclasses import dataclass
from typing import Generic, TypeVar

import numpy as np

T = TypeVar("T")
N = TypeVar("N")
U = TypeVar("U")
V = TypeVar("V")


@dataclass
class IntervalUnion(Generic[T, N, U]):
    lower: np.ndarray[tuple[U, N], T]
    upper: np.ndarray[tuple[U, N], T]

    # @staticmethod
    # def from_single(lower: np.ndarray[tuple[N], T], upper: np.ndarray[tuple[N], T]) -> "IntervalUnion[T, N, Literal[1]]":
    #     return IntervalUnion(lower=lower.reshape((1, -1)), upper=upper.reshape((1, -1)))

    @staticmethod
    def empty(dtype: np.dtype, n: int, u: int) -> "IntervalUnion":
        # FIXME
        assert np.issubdtype(dtype, np.integer), (
            "only integer intervals supported for now"
        )

        info = np.iinfo(dtype)

        return IntervalUnion(
            lower=np.full((u, n), info.max, dtype=dtype),
            upper=np.full((u, n), info.min, dtype=dtype),
        )

    def intersect(self, other: "IntervalUnion[T, N, V]") -> "IntervalUnion[T, N]":
        (u, n), (v, _) = self.lower.shape, other.lower.shape

        out = IntervalUnion.empty(self.lower.dtype, n, max(u, v))
        n_intervals = np.zeros(n, dtype=int)

        for i in range(u):
            for j in range(v):
                intersection_lower = np.maximum(self.lower[i], other.lower[j])
                intersection_upper = np.minimum(self.upper[i], other.upper[j])

                has_intersection = intersection_lower > intersection_upper

                out.lower[n_intervals] = np.where(
                    has_intersection, intersection_lower, out.lower[n_intervals]
                )
                out.upper[n_intervals] = np.where(
                    has_intersection, intersection_upper, out.upper[n_intervals]
                )

                n_intervals += has_intersection

        assert np.amin(n_intervals) > 0, "intersection must not be empty"

        uv = np.amax(n_intervals)
        out.lower, out.upper = out.lower[:uv], out.upper[:uv]

        return out
