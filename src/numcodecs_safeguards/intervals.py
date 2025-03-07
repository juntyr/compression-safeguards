from typing import Any, Generic, Literal, TypeVar
from typing_extensions import Self  # MSPV 3.11

import numpy as np

T = TypeVar("T", bound=np.dtype)
N = TypeVar("N", bound=Literal[1])
U = TypeVar("U", bound=Literal[1])
V = TypeVar("V", bound=Literal[1])
S = TypeVar("S", bound=tuple[int, ...])


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

    def preserve_inf(self, a: np.ndarray[tuple[N], T]) -> Self:
        if not np.issubdtype(a.dtype, np.floating):
            return self

        # bitwise preserve infinite values
        Lower(a) <= self[np.isinf(a)] <= Upper(a)

        return self

    def preserve_nan(self, a: np.ndarray[tuple[N], T], *, equal_nan: bool) -> Self:
        if not np.issubdtype(a.dtype, np.floating):
            return self

        if not equal_nan:
            # bitwise preserve NaN values
            Lower(a) <= self[np.isnan(a)] <= Upper(a)
            return self

        # smallest (positive) NaN bit pattern: 0b s 1..1 0..0
        nan_min = np.array(_as_bits(np.array(np.inf, dtype=a.dtype)) + 1).view(a.dtype)
        # largest (negative) NaN bit pattern: 0b s 1..1 1..1
        nan_max = np.array(-1, dtype=a.dtype.str.replace("f", "i")).view(a.dtype)

        # any NaN with the same sign is valid
        # this is slightly stricter than what equal_nan requires
        Lower(
            np.where(
                # ensure the NaN has the correct sign
                np.signbit(a),
                np.copysign(nan_max, -1),
                np.copysign(nan_min, +1),
            )
        ) <= self[np.isnan(a)] <= Upper(
            np.where(
                np.signbit(a),
                np.copysign(nan_min, -1),
                np.copysign(nan_max, +1),
            )
        )

        return self

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
    def __le__(self, interval: IndexedInterval[T, N]) -> IndexedInterval[T, N]:
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
    def __ge__(self, interval: IndexedInterval[T, N]) -> IndexedInterval[T, N]:
        if not isinstance(interval, IndexedInterval):
            return NotImplemented

        interval._upper[interval._index] = _maximum(interval._upper.dtype)

        return interval


Maximum = _Maximum()


class Lower:
    _lower: np.ndarray

    def __init__(self, lower: np.ndarray) -> None:
        self._lower = lower

    def __le__(self, interval: IndexedInterval[T, N]) -> IndexedInterval[T, N]:
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

    def __ge__(self, interval: IndexedInterval[T, N]) -> IndexedInterval[T, N]:
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
            return IntervalUnion.empty(self._lower.dtype, n, min(u, v))

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

    def contains(self, other: np.ndarray[S, T]) -> np.ndarray[S, np.dtype[np.bool]]:
        other_flat = _to_total_order(other).flatten()

        (u, n) = self._lower.shape
        is_contained = np.zeros((n,), dtype=np.dtype(bool))

        for i in range(u):
            is_contained |= (other_flat >= _to_total_order(self._lower[i])) & (
                other_flat <= _to_total_order(self._upper[i])
            )

        return is_contained.reshape(other.shape)  # type: ignore

    def encode_simple(self, decoded: np.ndarray[S, T]) -> np.ndarray[S, T]:
        # simple encoding:
        #  1. if decoded is in the interval, use it
        #  2. otherwise pick the lower bound of the first interval
        pick = self._lower[0].reshape(decoded.shape).copy()
        pick = np.where(self.contains(decoded), decoded, pick)

        return pick

    def encode_more_zeros(self, decoded: np.ndarray[S, T]) -> np.ndarray[S, T]:
        if decoded.size == 0:
            return decoded

        (_, n) = self._lower.shape

        # (a) if decoded is in the interval, use it
        contains_decoded = self.contains(decoded).flatten()

        # 1. convert everything to bits in total order
        decoded_bits = _to_total_order(decoded).reshape(1, -1)
        todtype = decoded_bits.dtype
        decoded_bits = _as_bits(decoded_bits)

        lower, upper = _to_total_order(self._lower), _to_total_order(self._upper)
        interval_nonempty = lower <= upper
        lower, upper = _as_bits(lower), _as_bits(upper)

        # 2. look at the difference between the decoded value and the interval
        #    we assume that decoded is not inside the interval, since that's
        #    handled separately with the special case (a)
        lower = decoded_bits - lower
        upper = decoded_bits - upper

        # 3. ensure that lower <= upper also in binary
        flip = (lower > upper) & interval_nonempty
        lower, upper = np.where(flip, upper, lower), np.where(flip, lower, upper)

        # 4. 0b1111...1111
        allbits = np.array(-1, dtype=upper.dtype.str.replace("u", "i")).view(
            upper.dtype
        )

        # 5. if there are several intervals, pick the one with the smallest
        #    lower bound, ensuring that empty intervals are not picked
        least = np.where(interval_nonempty, lower, allbits).argmin(axis=0)
        lower, upper = lower[least, np.arange(n)], upper[least, np.arange(n)]
        assert np.all(lower <= upper)

        # 6. count the number of leading zero bits in lower and upper
        lower_lz = _count_leading_zeros(lower)
        upper_lz = _count_leading_zeros(upper)

        # 7. if upper_lz < lower_lz,
        #    i.e. ceil(log2(upper)) > ceil(log2(lower)),
        #    (2 ** ceil(log2(lower))) - 1 is a tighter upper bound
        #
        #    upper: 0b00..01xxxxxxxxxxx
        #    lower: 0b00..00..01yyyyyyy
        # -> upper: 0b00..00..011111111
        upper = np.where(upper_lz < lower_lz, allbits >> lower_lz, upper)

        # 8. count the number of leading zero bits in (lower ^ upper) to find
        #    the most significant bit where lower and upper differ.
        #    Since upper > lower, at this bit i, upper[i] = 1 and lower[i] = 0.
        lxu_lz = _count_leading_zeros(lower ^ upper)

        # 9. pick such that the binary difference starts and ends with a
        #    maximally long sequence of zeros
        #
        #    upper: 0b00..01xxx1zzzzzzz
        #    lower: 0b00..01yyy0wwwwwww
        #  -> pick: 0b00..01xxx10000000
        # assert False, f"{contains_decoded} {lower} {upper} {lxu_lz}"
        pick = upper & ~(allbits >> (lxu_lz + 1))

        # 10. undo the difference with decoded and enforce the special case from
        #     (a) that decoded values inside the interval are kept as-is
        pick = np.where(contains_decoded, decoded_bits, decoded_bits - pick)

        # 11. convert everything back from total-ordered bits to value space
        pick = _from_total_order(pick.view(todtype), decoded.dtype).reshape(
            decoded.shape
        )
        assert np.all(self.contains(pick))

        return pick.reshape(decoded.shape)

    def encode(self, decoded: np.ndarray[S, T]) -> np.ndarray[S, T]:
        return self.encode_more_zeros(decoded)

    def __repr__(self) -> str:
        return f"IntervalUnion(lower={self._lower!r}, upper={self._upper!r})"


def _to_total_order(x: np.ndarray) -> np.ndarray:
    """
    FloatFlip in http://stereopsis.com/radix.html
    """

    if np.issubdtype(x.dtype, np.unsignedinteger):
        return x

    utype = x.dtype.str.replace("i", "u").replace("f", "u")

    if np.issubdtype(x.dtype, np.signedinteger):
        return x.view(utype) + np.array(np.iinfo(x.dtype).max, dtype=utype) + 1

    if not np.issubdtype(x.dtype, np.floating):
        raise TypeError(f"unsupported interval type {x.dtype}")

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

    assert np.issubdtype(x.dtype, np.unsignedinteger)

    if np.issubdtype(dtype, np.unsignedinteger):
        return x

    if np.issubdtype(dtype, np.signedinteger):
        return x.view(dtype) + np.iinfo(dtype).max + 1

    if not np.issubdtype(dtype, np.floating):
        raise TypeError(f"unsupported interval type {dtype}")

    utype = dtype.str.replace("f", "u")
    itype = dtype.str.replace("f", "i")
    bits = np.iinfo(utype).bits

    mask = ((x >> (bits - 1)).view(dtype=itype) - 1).view(dtype=utype) | (
        np.array(1, dtype=utype) << (bits - 1)
    )

    return (x ^ mask).view(dtype=dtype)


def _count_leading_zeros(x: np.ndarray) -> np.ndarray:
    """
    https://stackoverflow.com/a/79189999
    """

    x_bits = _as_bits(x)
    nbits = np.iinfo(x_bits.dtype).bits

    assert nbits <= 64

    if nbits <= 16:
        return (nbits - np.frexp(x_bits.astype(np.uint32))[1]).astype(np.uint8)

    if nbits <= 32:
        return (nbits - np.frexp(x_bits.astype(np.uint64))[1]).astype(np.uint8)

    # nbits <= 64
    _, high_exp = np.frexp(x_bits.astype(np.uint64) >> 32)
    _, low_exp = np.frexp(x_bits.astype(np.uint64) & 0xFFFFFFFF)
    return (nbits - np.where(high_exp, high_exp + 32, low_exp)).astype(np.uint8)


def _as_bits(a: np.ndarray) -> np.ndarray:
    return a.view(a.dtype.str.replace("f", "u").replace("i", "u"))
