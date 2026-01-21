"""
Types and helpers to construct safe intervals for the safeguards.
"""

__all__ = [
    "Interval",
    "IntervalUnion",
    "Minimum",
    "Maximum",
    "Lower",
    "Upper",
    "N",
    "Ni",
    "U",
    "Ui",
]

from typing import Any, Generic, Literal, Self, TypeVar

import numpy as np
from typing_extensions import override  # MSPV 3.12

from ._compat import _ensure_array, _reshape, _where, _zeros
from ._compat import _maximum_zero_sign_sensitive as _np_maximum
from ._compat import _minimum_zero_sign_sensitive as _np_minimum
from .cast import as_bits, from_total_order, to_total_order
from .error import TypeSetError
from .typing import S, T

N = TypeVar("N", bound=int, covariant=True)
""" The number of elements in the interval (covariant). """

Ni = TypeVar("Ni", bound=int)
""" The number of elements in the interval (invariant). """

U = TypeVar("U", bound=int, covariant=True)
""" The maximum number of intervals in an interval union (covariant). """

Ui = TypeVar("Ui", bound=int)
""" The maximum number of intervals in an interval union (invariant). """


class Interval(Generic[T, N]):
    """
    Single interval over a `N`-sized [`ndarray`][numpy.ndarray] of data type `T`.

    ## Construction

    An interval should only be constructed using

    - [`Interval.empty`][.empty] or [`Interval.empty_like`][.empty_like] for an
      empty interval that contains no values
    - [`Interval.full`][.full] or [`Interval.full_like`][.full_like] for a
      full interval that contains all possible values

    ## Overriding lower and upper bounds

    The lower and upper bounds of the interval can be set as follows:

    ```python
    Lower(lower_data) <= interval <= Upper(upper_data)
    ```

    They can also be set to the smallest and largest possible values using:

    ```python
    Minimum <= interval <= Maximum
    ```

    If only the interval bounds for some array members should be updated, only
    the interval itself needs to be indexed:

    ```python
    # bound infinite values to be exactly themselves
    Lower(data) <= interval[np.isinf(data)] <= Upper(data)
    ```

    ## Common lower and upper bounds

    The following common lower and upper bounds are provided for ease of use:

    - [`Interval.preserve_inf`][.preserve_inf]
    - [`Interval.preserve_signed_nan`][.preserve_signed_nan]
    - [`Interval.preserve_any_nan`][.preserve_any_nan]
    - [`Interval.preserve_finite`][.preserve_finite]
    - [`Interval.preserve_non_nan`][.preserve_non_nan]

    ## Interval operations

    Two intervals can be

    - intersected using [`Interval.intersect`][.intersect]
    - unioned using [`Interval.union`][.union]

    or converted into a single-member union of intervals using
    [`Interval.into_union`][.into_union].
    """

    __slots__: tuple[str, ...] = ("_lower", "_upper")
    _lower: np.ndarray[tuple[N], np.dtype[T]]
    _upper: np.ndarray[tuple[N], np.dtype[T]]

    def __init__(
        self,
        *,
        _lower: np.ndarray[tuple[N], np.dtype[T]],
        _upper: np.ndarray[tuple[N], np.dtype[T]],
    ) -> None:
        self._lower = _lower
        self._upper = _upper

    @property
    def dtype(self) -> np.dtype[T]:
        """
        Dtype `T` of the interval.
        """
        return self._lower.dtype

    @property
    def n(self) -> N:
        """
        Size `N` of the interval.
        """
        (n,) = self._lower.shape
        return n

    def non_empty_width(self) -> np.ndarray[tuple[N], np.dtype[np.unsignedinteger]]:
        """
        Binary width of the interval.

        The interval must not have any empty entries.

        Single-element intervals have width zero, full intervals have maximum
        width.

        Returns
        -------
        widths : np.ndarray[tuple[N], np.dtype[np.unsignedinteger]]
            The binary widths of the intervals.
        """

        lt: np.ndarray[tuple[N], np.dtype[np.unsignedinteger]] = to_total_order(
            self._lower
        )
        ut: np.ndarray[tuple[N], np.dtype[np.unsignedinteger]] = to_total_order(
            self._upper
        )

        assert np.all(lt <= ut)

        return ut - lt

    @staticmethod
    def empty(dtype: np.dtype[T], n: Ni) -> "Interval[T, Ni]":
        """
        Create an empty interval that contains no values.

        Parameters
        ----------
        dtype : np.dtype[T]
            The dtype of the interval
        n : Ni
            The size of the interval

        Returns
        -------
        empty : Interval[T, Ni]
            The empty interval
        """

        single: IntervalUnion[T, Ni, Literal[1]] = IntervalUnion.empty(dtype, n, 1)
        return Interval(
            _lower=single._lower[0],
            _upper=single._upper[0],
        )

    @staticmethod
    def empty_like(a: np.ndarray[tuple[int, ...], np.dtype[T]]) -> "Interval[T, int]":
        """
        Create an empty interval that contains no values and has the same dtype and size as `a`.

        Parameters
        ----------
        a : np.ndarray[tuple[int, ...], np.dtype[T]]
            An array whose dtype and size the interval gets.

        Returns
        -------
        empty : Interval[T, int]
            The empty interval
        """

        return Interval.empty(a.dtype, a.size)

    @staticmethod
    def full(dtype: np.dtype[T], n: Ni) -> "Interval[T, Ni]":
        """
        Create a full interval that contains all possible values of `dtype`.

        Parameters
        ----------
        dtype : np.dtype[T]
            The dtype of the interval
        n : Ni
            The size of the interval

        Returns
        -------
        full : Interval[T, Ni]
            The full interval
        """

        return Interval(
            _lower=np.full(n, _minimum(dtype), dtype=dtype),
            _upper=np.full(n, _maximum(dtype), dtype=dtype),
        )

    @staticmethod
    def full_like(a: np.ndarray[tuple[int, ...], np.dtype[T]]) -> "Interval[T, int]":
        """
        Create a full interval that contains all possible values and has the same dtype and size as `a`.

        Parameters
        ----------
        a : np.ndarray[tuple[int, ...], np.dtype[T]]
            An array whose dtype and size the interval gets

        Returns
        -------
        full : Interval[T, int]
            The full interval
        """

        return Interval.full(a.dtype, a.size)

    def __getitem__(self, key) -> "IndexedInterval[T, N]":
        return IndexedInterval(_lower=self._lower, _upper=self._upper, _index=key)

    def preserve_inf(self, a: np.ndarray[tuple[N], np.dtype[T]]) -> Self:
        """
        Preserve all infinite values in `a` exactly.

        Specifically, set their lower and upper bounds in this interval to
        their values.

        Equivalent to

        ```python
        Lower(a) <= self[np.isinf(a)] <= Upper(a)
        ```

        Parameters
        ----------
        a : np.ndarray[tuple[N], np.dtype[T]]
            The arrays whose infinite values this interval should preserve

        Returns
        -------
        self : Self
            The modified `self`.
        """

        if not np.issubdtype(a.dtype, np.floating):
            return self

        # bitwise preserve infinite values
        Lower(a) <= self[np.isinf(a)] <= Upper(a)

        return self

    def preserve_signed_nan(
        self, a: np.ndarray[tuple[N], np.dtype[T]], *, equal_nan: bool
    ) -> Self:
        """
        Preserve all NaN values in `a`, preserving their sign bit.

        - If `equal_nan` is [`True`][True], the intervals corresponding to
          the NaN values will include all possible NaN values with the same
          sign bit.
        - If `equal_nan` is [`False`][False], all NaN values are preserved
          exactly.

        Parameters
        ----------
        a : np.ndarray[tuple[N], np.dtype[T]]
            The arrays whose NaN values this interval should preserve
        equal_nan : bool
            Whether any NaN values matches another NaN value or if NaN values
            should be preserved exactly

        Returns
        -------
        self : Self
            The modified `self`.
        """

        if not np.issubdtype(a.dtype, np.floating):
            return self

        if not equal_nan:
            # bitwise preserve NaN values
            Lower(a) <= self[np.isnan(a)] <= Upper(a)
            return self

        # smallest (positive) NaN bit pattern: 0b s 1..1 0..0
        nan_min = np.array(as_bits(np.array(np.inf, dtype=a.dtype)) + 1).view(a.dtype)
        # largest (negative) NaN bit pattern: 0b s 1..1 1..1
        nan_max = np.array(-1, dtype=a.dtype.str.replace("f", "i")).view(a.dtype)

        # any NaN with the same sign is valid
        # this is slightly stricter than what equal_nan requires
        Lower(
            _where(
                # ensure the NaN has the correct sign
                np.signbit(a),
                np.copysign(nan_max, -1),
                np.copysign(nan_min, +1),
            )
        ) <= self[np.isnan(a)] <= Upper(
            _where(
                np.signbit(a),
                np.copysign(nan_min, -1),
                np.copysign(nan_max, +1),
            )
        )

        return self

    def preserve_any_nan(
        self, a: np.ndarray[tuple[N], np.dtype[T]], *, equal_nan: bool
    ) -> "IntervalUnion[T, N, int]":
        """
        Preserve all NaN values in `a`, ignoring their sign bit.

        - If `equal_nan` is [`True`][True], the intervals corresponding to
          the NaN values will include all possible NaN values, irrespective of
          their sign bit.
        - If `equal_nan` is [`False`][False], all NaN values are preserved
          exactly.

        Since there are two disjoint value regions of NaNs (positive and
        negative), this method returns a union of intervals.

        Parameters
        ----------
        a : np.ndarray[tuple[N], np.dtype[T]]
            The arrays whose NaN values this interval should preserve
        equal_nan : bool
            Whether any NaN values matches another NaN value or if NaN values
            should be preserved exactly

        Returns
        -------
        union : IntervalUnion[T, N, int]
            The union of the existing intervals for non-NaN values and the
            NaN-preserving intervals for NaN values.
        """

        if not np.issubdtype(a.dtype, np.floating):
            return self.into_union()

        (n,) = a.shape

        lower: Interval[T, N] = Interval.empty(a.dtype, n)
        # copy over the intervals for non-NaN elements
        Lower(self._lower) <= lower[~np.isnan(a)] <= Upper(self._upper)

        if (not equal_nan) or (not np.any(np.isnan(a))):
            # bitwise preserve NaN values
            Lower(a) <= lower[np.isnan(a)] <= Upper(a)
            return lower.into_union()

        # smallest (positive) NaN bit pattern: 0b s 1..1 0..0
        nan_min = np.array(as_bits(np.array(np.inf, dtype=a.dtype)) + 1).view(a.dtype)
        # largest (negative) NaN bit pattern: 0b s 1..1 1..1
        nan_max = np.array(-1, dtype=a.dtype.str.replace("f", "i")).view(a.dtype)

        upper: Interval[T, N] = Interval.empty(a.dtype, n)

        # create lower interval of all negative NaNs
        Lower(np.array(np.copysign(nan_max, -1))) <= lower[np.isnan(a)] <= Upper(
            np.array(np.copysign(nan_min, -1))
        )
        # create upper interval of all positive NaNs
        Lower(np.array(np.copysign(nan_min, +1))) <= upper[np.isnan(a)] <= Upper(
            np.array(np.copysign(nan_max, +1))
        )

        return lower.union(upper)

    def preserve_finite(self, a: np.ndarray[tuple[N], np.dtype[T]]) -> Self:
        """
        Preserve all finite values in `a` as finite values.

        Specifically, set their lower and upper bounds to exclude non-finite
        values.

        Parameters
        ----------
        a : np.ndarray[tuple[N], np.dtype[T]]
            The arrays whose non-finite values this interval should preserve as
            non-finite.

        Returns
        -------
        self : Self
            The modified `self`.
        """

        if not np.issubdtype(a.dtype, np.floating):
            Minimum <= self[:] <= Maximum
            return self

        # nextafter produces the largest and smallest finite floating-point
        #  values
        Lower(
            np.array(
                np.nextafter(
                    a.dtype.type(-np.inf),  # type: ignore
                    a.dtype.type(0),
                )
            )
        ) <= self[np.isfinite(a)] <= Upper(
            np.array(
                np.nextafter(
                    a.dtype.type(np.inf),  # type: ignore
                    a.dtype.type(0),
                )
            )
        )

        return self

    def preserve_non_nan(self, a: np.ndarray[tuple[N], np.dtype[T]]) -> Self:
        """
        Preserve all non-NaN values in `a` as non-NaN values.

        Specifically, set their lower and upper bounds to exclude NaN values.

        Parameters
        ----------
        a : np.ndarray[tuple[N], np.dtype[T]]
            The arrays whose non-NaN values this interval should preserve as
            non-NaN.

        Returns
        -------
        self : Self
            The modified `self`.
        """

        if not np.issubdtype(a.dtype, np.floating):
            Minimum <= self[:] <= Maximum
            return self

        Lower(np.array(-np.inf, dtype=a.dtype)) <= self[~np.isnan(a)] <= Upper(
            np.array(np.inf, dtype=a.dtype)
        )

        return self

    def intersect(self, other: "Interval[T, N]") -> "Interval[T, N]":
        """
        Computes the intersection with the `other` interval.

        Parameters
        ----------
        other : Interval[T, N]
            The other interval to intersect with

        Returns
        -------
        intersection : Interval[T, N]
            The intersection of `self` and `other`.
        """

        (n,) = self._lower.shape

        if n == 0:
            return Interval.empty(self._lower.dtype, n)

        out: Interval[T, N] = Interval.empty(
            self._lower.dtype,
            n,
        )

        intersection_lower: np.ndarray[tuple[N], np.dtype[np.unsignedinteger]] = (
            _np_maximum(to_total_order(self._lower), to_total_order(other._lower))
        )
        intersection_upper: np.ndarray[tuple[N], np.dtype[np.unsignedinteger]] = (
            _np_minimum(to_total_order(self._upper), to_total_order(other._upper))
        )

        out._lower[:] = from_total_order(intersection_lower, out._lower.dtype)
        out._upper[:] = from_total_order(intersection_upper, out._upper.dtype)

        return out

    def union(
        self, other: "Interval[T, N] | IntervalUnion[T, N, int]"
    ) -> "IntervalUnion[T, N, int]":
        """
        Computes the union with the `other` interval (union).

        Parameters
        ----------
        other : Interval[T, N] | IntervalUnion[T, N, int]
            The other interval (union) to union with

        Returns
        -------
        intersection : IntervalUnion[T, N, int]
            The union of `self` and `other`.
        """

        return self.into_union().union(other)

    def into_union(self) -> "IntervalUnion[T, N, Literal[1]]":
        """
        Convert this interval into a single-member union of exactly this one interval.

        Returns
        -------
        union : IntervalUnion[T, N, Literal[1]]
            The union of only this interval
        """

        return IntervalUnion(
            _lower=_reshape(self._lower, (1, self.n)),
            _upper=_reshape(self._upper, (1, self.n)),
        )

    def preserve_only_where(
        self, where: Literal[True] | np.ndarray[tuple[N], np.dtype[np.bool]]
    ) -> Self:
        """
        Preserve safe intervals only at the data points where the `where`
        condition is [`True`][True].

        This method must only be used for pointwise safe intervals.

        Parameters
        ----------
        where : Literal[True] | np.ndarray[tuple[N], np.dtype[np.bool]]
            Only preserve safe intervals at data points where the condition is
            [`True`][True].

        Returns
        -------
        self : Self
            The modified `self`.
        """

        if where is True:
            return self

        # reset all elements where where is False to a full interval
        Minimum <= self[~where] <= Maximum
        return self

    @override
    def __repr__(self) -> str:
        return f"Interval(lower={self._lower!r}, upper={self._upper!r})"


class IndexedInterval(Generic[T, N]):
    __slots__: tuple[str, ...] = ("_lower", "_upper", "_index")
    _lower: np.ndarray[tuple[N], np.dtype[T]]
    _upper: np.ndarray[tuple[N], np.dtype[T]]
    _index: Any

    def __init__(
        self,
        *,
        _lower: np.ndarray[tuple[N], np.dtype[T]],
        _upper: np.ndarray[tuple[N], np.dtype[T]],
        _index: Any,
    ) -> None:
        self._lower = _lower
        self._upper = _upper
        self._index = _index


def _minimum(dtype: np.dtype[T]) -> np.ndarray[tuple[()], np.dtype[T]]:
    TypeSetError.check_or_raise(dtype.type, np.integer | np.floating)

    if np.issubdtype(dtype, np.integer):
        return np.array(np.iinfo(dtype).min, dtype=dtype)  # type: ignore

    btype = dtype.str.replace("f", "u")
    bmin = np.iinfo(btype).max  # produces -NaN (0xffff...)
    return np.array(bmin, dtype=btype).view(dtype)


class _Minimum:
    __slots__: tuple[str, ...] = ()

    def __le__(self, interval: IndexedInterval[T, N]) -> IndexedInterval[T, N]:
        if not isinstance(interval, IndexedInterval):
            return NotImplemented

        interval._lower[interval._index] = _minimum(interval._lower.dtype)

        return interval


Minimum = _Minimum()
""" The smallest representable value """


def _maximum(dtype: np.dtype[T]) -> np.ndarray[tuple[()], np.dtype[T]]:
    TypeSetError.check_or_raise(dtype.type, np.integer | np.floating)

    if np.issubdtype(dtype, np.integer):
        return np.array(np.iinfo(dtype).max, dtype=dtype)  # type: ignore

    btype = dtype.str.replace("f", "u")
    bmin = np.iinfo(btype).max  # produces -NaN (0xffff...)
    return np.copysign(np.array(bmin, dtype=btype).view(dtype), +1)  # type: ignore


class _Maximum:
    __slots__: tuple[str, ...] = ()

    def __ge__(self, interval: IndexedInterval[T, N]) -> IndexedInterval[T, N]:
        if not isinstance(interval, IndexedInterval):
            return NotImplemented

        interval._upper[interval._index] = _maximum(interval._upper.dtype)

        return interval


Maximum = _Maximum()
""" The largest representable value """


class Lower:
    """
    Array wrapper to override an [`Interval`][compression_safeguards.utils.intervals.Interval]'s lower bound using comparison syntax.

    ```python
    Lower(lower_bound) <= interval
    ```

    Parameters
    ----------
    lower : np.ndarray[tuple[int, ...], np.dtype[np.number]]
        The lower bound array
    """

    __slots__: tuple[str, ...] = ("_lower",)
    _lower: np.ndarray[tuple[int, ...], np.dtype[np.number]]

    def __init__(self, lower: np.ndarray[tuple[int, ...], np.dtype[np.number]]) -> None:
        self._lower = lower

    def __le__(self, interval: IndexedInterval[T, N]) -> IndexedInterval[T, N]:
        if not isinstance(self._lower, np.ndarray):
            return NotImplemented

        if not isinstance(interval, IndexedInterval):
            return NotImplemented

        if self._lower.dtype != interval._lower.dtype:
            return NotImplemented

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
    """
    Array wrapper to override an [`Interval`][compression_safeguards.utils.intervals.Interval]'s upper bound using comparison syntax.

    ```python
    interval <= Upper(upper_bound)
    ```

    Parameters
    ----------
    upper : np.ndarray[tuple[int, ...], np.dtype[np.number]]
        The upper bound array
    """

    __slots__: tuple[str, ...] = ("_upper",)
    _upper: np.ndarray[tuple[int, ...], np.dtype[np.number]]

    def __init__(self, upper: np.ndarray[tuple[int, ...], np.dtype[np.number]]) -> None:
        self._upper = upper

    def __ge__(self, interval: IndexedInterval[T, N]) -> IndexedInterval[T, N]:
        if not isinstance(self._upper, np.ndarray):
            return NotImplemented

        if not isinstance(interval, IndexedInterval):
            return NotImplemented

        if self._upper.dtype != interval._upper.dtype:
            return NotImplemented

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
    """
    Union of `U` intervals, each over a `N`-sized [`ndarray`][numpy.ndarray] of data type `T`.
    """

    __slots__: tuple[str, ...] = ("_lower", "_upper")
    # invariants:
    # - the lower/upper bounds are in sorted order
    # - no non-empty intervals come after empty intervals
    # - no intervals intersect within the union
    # - no intervals can be adjacent within the union, e.g. [1..3] and [4..5]
    _lower: np.ndarray[tuple[U, N], np.dtype[T]]
    _upper: np.ndarray[tuple[U, N], np.dtype[T]]

    def __init__(
        self,
        *,
        _lower: np.ndarray[tuple[U, N], np.dtype[T]],
        _upper: np.ndarray[tuple[U, N], np.dtype[T]],
    ) -> None:
        self._lower = _lower
        self._upper = _upper

    @property
    def dtype(self) -> np.dtype[T]:
        """
        Dtype `T` of the interval union.
        """
        return self._lower.dtype

    @property
    def n(self) -> N:
        """
        Size `N` of the interval union.
        """
        u, n = self._lower.shape
        return n

    @property
    def u(self) -> U:
        """
        Number of intervals `U` in the interval union.
        """
        u, n = self._lower.shape
        return u

    def non_empty_width(self) -> np.ndarray[tuple[N], np.dtype[np.unsignedinteger]]:
        """
        Binary width of the interval union.

        The interval union must not have any empty entries.

        Single-element intervals have width zero, full intervals have maximum
        width.

        Returns
        -------
        widths : np.ndarray[tuple[N], np.dtype[np.unsignedinteger]]
            The binary widths of the intervals.
        """

        (u, n) = self._lower.shape
        interval_width_acc: np.ndarray[tuple[N], np.dtype[np.unsignedinteger]] = _zeros(
            (n,), dtype=to_total_order(np.array((), dtype=self._lower.dtype)).dtype
        )

        for i in range(u):
            lt: np.ndarray[tuple[N], np.dtype[np.unsignedinteger]] = to_total_order(
                self._lower[i]
            )
            ut: np.ndarray[tuple[N], np.dtype[np.unsignedinteger]] = to_total_order(
                self._upper[i]
            )

            if i == 0:
                assert np.all(lt <= ut)

            np.add(
                interval_width_acc,
                ut - lt + 1,
                out=interval_width_acc,
                where=(ut >= lt),
            )

        interval_width_acc -= 1

        return interval_width_acc

    @staticmethod
    def empty(dtype: np.dtype[T], n: Ni, u: Ui) -> "IntervalUnion[T, Ni, Ui]":
        """
        Create an empty interval union that contains no values.

        Parameters
        ----------
        dtype : np.dtype[T]
            The dtype of the intervals
        n : Ni
            The size of the intervals
        u : Ui
            The number of intervals in the union

        Returns
        -------
        empty : IntervalUnion[T, Ni, Ui]
            The empty interval union
        """

        return IntervalUnion(
            _lower=np.full((u, n), _maximum(dtype), dtype=dtype),
            _upper=np.full((u, n), _minimum(dtype), dtype=dtype),
        )

    def intersect(
        self, other: "IntervalUnion[T, N, int]"
    ) -> "IntervalUnion[T, N, int]":
        """
        Computes the intersection with the `other` interval union.

        Parameters
        ----------
        other : IntervalUnion[T, N, int]
            The other interval union to intersect with

        Returns
        -------
        intersection : IntervalUnion[T, N, int]
            The intersection of `self` and `other`.
        """

        ((u, n), (v, _)) = self._lower.shape, other._lower.shape

        if n == 0:
            return IntervalUnion.empty(self._lower.dtype, n, 0)

        uv: int = u + v - 1
        out: IntervalUnion[T, N, int] = IntervalUnion.empty(
            self._lower.dtype,
            n,
            uv,
        )
        n_intervals = np.zeros(n, dtype=int)

        for i in range(u):
            for j in range(v):
                intersection_lower: np.ndarray[
                    tuple[U, N], np.dtype[np.unsignedinteger]
                ] = _np_maximum(
                    to_total_order(self._lower[i]), to_total_order(other._lower[j])
                )
                intersection_upper: np.ndarray[
                    tuple[U, N], np.dtype[np.unsignedinteger]
                ] = _np_minimum(
                    to_total_order(self._upper[i]), to_total_order(other._upper[j])
                )

                has_intersection = intersection_lower <= intersection_upper

                out._lower[n_intervals[has_intersection], has_intersection] = (
                    from_total_order(
                        intersection_lower[has_intersection], out._lower.dtype
                    )
                )
                out._upper[n_intervals[has_intersection], has_intersection] = (
                    from_total_order(
                        intersection_upper[has_intersection], out._upper.dtype
                    )
                )

                n_intervals += has_intersection

        uv = np.amax(n_intervals)

        return IntervalUnion(_lower=out._lower[:uv], _upper=out._upper[:uv])  # type: ignore

    def union(
        self, other: "Interval[T, N] | IntervalUnion[T, N, int]"
    ) -> "IntervalUnion[T, N, int]":
        """
        Computes the union with the `other` interval (union).

        Parameters
        ----------
        other : Interval[T, N] | IntervalUnion[T, N, int]
            The other interval (union) to union with

        Returns
        -------
        union : IntervalUnion[T, N, int]
            The union of `self` and `other`.
        """

        otheru = other if isinstance(other, IntervalUnion) else other.into_union()

        ((u, n), (v, _)) = self._lower.shape, otheru._lower.shape

        if n == 0:
            return IntervalUnion.empty(self._lower.dtype, n, 0)

        if u == 0:
            return otheru

        if v == 0:
            return self

        uv: int = u + v
        out: IntervalUnion[T, N, int] = IntervalUnion.empty(
            self._lower.dtype,
            n,
            uv,
        )
        n_intervals = np.zeros(n, dtype=int)

        i_s = np.zeros(n, dtype=int)
        j_s = np.zeros(n, dtype=int)

        while (np.amin(i_s) < u) or (np.amin(j_s) < v):
            lower_i: np.ndarray[tuple[int], np.dtype[np.unsignedinteger]] = (
                to_total_order(
                    np.take_along_axis(
                        self._lower, _np_minimum(i_s, u - 1).reshape(1, -1), axis=0
                    ).flatten()
                )
            )
            upper_i: np.ndarray[tuple[int], np.dtype[np.unsignedinteger]] = (
                to_total_order(
                    np.take_along_axis(
                        self._upper, _np_minimum(i_s, u - 1).reshape(1, -1), axis=0
                    ).flatten()
                )
            )

            lower_j: np.ndarray[tuple[int], np.dtype[np.unsignedinteger]] = (
                to_total_order(
                    np.take_along_axis(
                        otheru._lower, _np_minimum(j_s, v - 1).reshape(1, -1), axis=0
                    ).flatten()
                )
            )
            upper_j: np.ndarray[tuple[int], np.dtype[np.unsignedinteger]] = (
                to_total_order(
                    np.take_along_axis(
                        otheru._upper, _np_minimum(j_s, v - 1).reshape(1, -1), axis=0
                    ).flatten()
                )
            )

            lower_o: np.ndarray[tuple[int], np.dtype[np.unsignedinteger]] = (
                to_total_order(
                    np.take_along_axis(
                        out._lower,
                        _np_maximum(n_intervals - 1, 0).reshape(1, -1),
                        axis=0,
                    ).flatten()
                )
            )
            upper_o: np.ndarray[tuple[int], np.dtype[np.unsignedinteger]] = (
                to_total_order(
                    np.take_along_axis(
                        out._upper,
                        _np_maximum(n_intervals - 1, 0).reshape(1, -1),
                        axis=0,
                    ).flatten()
                )
            )

            # only valid if in-bounds and non-empty
            valid_i = (i_s < u) & (lower_i <= upper_i)
            valid_j = (j_s < v) & (lower_j <= upper_j)
            valid_o = (n_intervals > 0) & (lower_o <= upper_o)

            # choose the next valid interval with the lower lower bound
            choose_i = valid_i & ((lower_i < lower_j) | ~valid_j)
            choose_j = valid_j & ((lower_i >= lower_j) | ~valid_i)

            lower_ij = _ensure_array(lower_j)
            np.copyto(lower_ij, lower_i, where=choose_i, casting="no")
            upper_ij = _ensure_array(upper_j)
            np.copyto(upper_ij, upper_i, where=choose_i, casting="no")

            # check if the selected interval intersects with the previously
            #  output interval
            has_intersection_with_out = (
                (valid_i | valid_j)
                & valid_o
                & (
                    # check for normal intersection
                    (_np_maximum(lower_ij, lower_o) <= _np_minimum(upper_ij, upper_o))
                    |
                    # check for adjacent intervals, e.g [1..3] | [4..5] -> [1..5]
                    ((lower_ij > upper_o) & (lower_ij == (upper_o + 1)))
                )
            )

            # - intersection -> next is intersection
            # - no intersection -> next is next interval
            next_lower = _ensure_array(lower_ij, copy=True)
            np.copyto(
                next_lower,
                _np_minimum(lower_o, lower_ij),
                where=has_intersection_with_out,
                casting="no",
            )

            next_upper = _ensure_array(upper_ij, copy=True)
            np.copyto(
                next_upper,
                _np_maximum(upper_o, upper_ij),
                where=has_intersection_with_out,
                casting="no",
            )

            # update either the previous or the next output interval
            has_next = has_intersection_with_out | choose_i | choose_j
            out._lower[
                n_intervals[has_next] - has_intersection_with_out[has_next],
                has_next,
            ] = from_total_order(next_lower[has_next], out._lower.dtype)
            out._upper[
                n_intervals[has_next] - has_intersection_with_out[has_next],
                has_next,
            ] = from_total_order(next_upper[has_next], out._upper.dtype)

            # advance to the next interval if we wrote out a new one
            n_intervals += has_next & (~has_intersection_with_out)

            # advance the interval that was chosen earlier
            i_s += choose_i | (~valid_j)
            j_s += choose_j | (~valid_i)

        uv = np.amax(n_intervals)

        return IntervalUnion(_lower=out._lower[:uv], _upper=out._upper[:uv])  # type: ignore

    def preserve_only_where(
        self, where: Literal[True] | np.ndarray[tuple[N], np.dtype[np.bool]]
    ) -> "IntervalUnion[T, N, int]":
        """
        Preserve safe intervals only at the data points where the `where`
        condition is [`True`][True].

        This method must only be used for pointwise safe intervals.

        Parameters
        ----------
        where : Literal[True] | np.ndarray[tuple[N], np.dtype[np.bool]]
            Only preserve safe intervals at data points where the condition is
            [`True`][True].

        Returns
        -------
        self : IntervalUnion[T, N, int]
            The interval union where only safe intervals at data points where
            the condition is [`True`][True] are preserved.
        """

        if where is True:
            return self

        return self.union(
            Interval.empty(self._lower.dtype, self.n).preserve_only_where(where)
        )

    def contains(
        self, other: np.ndarray[S, np.dtype[T]]
    ) -> np.ndarray[S, np.dtype[np.bool]]:
        """
        Check if this interval union contains the elements of the `other` array.

        Parameters
        ----------
        other : np.ndarray[S, np.dtype[T]]
            The array whose elements' membership in interval union should be
            checked

        Returns
        -------
        contains : np.ndarray[S, np.dtype[np.bool]]
            The pointwise result of the contains check
        """

        other_flat: np.ndarray[tuple[int], np.dtype[np.unsignedinteger]] = (
            to_total_order(other).flatten()
        )

        (u, n) = self._lower.shape
        is_contained = np.zeros((n,), dtype=np.dtype(bool))

        for i in range(u):
            is_contained |= (other_flat >= to_total_order(self._lower[i])) & (
                other_flat <= to_total_order(self._upper[i])
            )

        return _reshape(is_contained, other.shape)

    def _pick_simple(
        self, prediction: np.ndarray[S, np.dtype[T]]
    ) -> np.ndarray[S, np.dtype[T]]:
        # simple pick:
        #  1. if prediction is in the interval, use it
        #  2. otherwise pick the lower bound of the first interval
        pick: np.ndarray[S, np.dtype[T]] = _ensure_array(
            self._lower[0].reshape(prediction.shape), copy=True
        )
        np.copyto(pick, prediction, where=self.contains(prediction), casting="no")
        return pick

    def _pick_random(
        self,
        prediction: np.ndarray[S, np.dtype[T]],
        rng: np.random.Generator,
    ) -> np.ndarray[S, np.dtype[T]]:
        # random pick: pick a random member of the interval union
        if prediction.size == 0:
            return prediction

        (u, n) = self._lower.shape
        interval_width_acc = self.non_empty_width()

        # use endpoint-inclusive sampling since the single-element width is
        #  zero and the full-interval width is maximal (so +1 would overflow)
        pick_indices: np.ndarray[tuple[N], np.dtype[np.unsignedinteger]] = rng.integers(
            interval_width_acc, dtype=interval_width_acc.dtype, endpoint=True
        )

        interval_width_acc[:] = 0

        pick_flat: np.ndarray[tuple[N], np.dtype[T]] = np.empty(
            (n,), dtype=self._lower.dtype
        )

        for i in range(u):
            lt: np.ndarray[tuple[N], np.dtype[np.unsignedinteger]] = to_total_order(
                self._lower[i]
            )
            ut: np.ndarray[tuple[N], np.dtype[np.unsignedinteger]] = to_total_order(
                self._upper[i]
            )
            np.copyto(
                pick_flat,
                from_total_order(
                    lt + (pick_indices - interval_width_acc), dtype=self._lower.dtype
                ),
                where=(
                    (ut >= lt)
                    & (pick_indices >= interval_width_acc)
                    & (pick_indices <= (interval_width_acc + (ut - lt)))
                ),
                casting="no",
            )
            np.add(
                interval_width_acc,
                ut - lt + 1,
                out=interval_width_acc,
                where=(ut >= lt),
            )

        return _reshape(pick_flat, prediction.shape)

    def _pick_more_zeros(
        self, prediction: np.ndarray[S, np.dtype[T]]
    ) -> np.ndarray[S, np.dtype[T]]:
        if prediction.size == 0:
            return prediction

        (_, n) = self._lower.shape

        # (a) if prediction is in the interval, use it
        contains_prediction = self.contains(prediction).flatten()

        # 1. convert everything to bits in total order
        prediction_bits: np.ndarray[
            tuple[Literal[1], int], np.dtype[np.unsignedinteger]
        ] = _reshape(to_total_order(prediction), (1, -1))

        lower: np.ndarray[tuple[U, N], np.dtype[np.unsignedinteger]] = to_total_order(
            self._lower
        )
        upper: np.ndarray[tuple[U, N], np.dtype[np.unsignedinteger]] = to_total_order(
            self._upper
        )
        interval_nonempty = lower <= upper

        # 2. look at the total order binary difference between the interval
        #    bounds and the prediction
        #    we assume that prediction is not inside the interval, since that
        #    is handled separately with the special case (a)
        lower = lower - prediction_bits
        upper = upper - prediction_bits

        # 3. only work with "positive" unsigned values
        negative: np.ndarray[tuple[U, N], np.dtype[np.bool]] = np.less(
            # lower has an unsigned integer dtype, make signed to compare
            lower.astype(lower.dtype.str.replace("u", "i")),
            0,
        )
        np.copyto(lower, ~lower + 1, where=negative, casting="no")
        np.copyto(upper, ~upper + 1, where=negative, casting="no")

        # 4. ensure that lower <= upper also in binary
        flip: np.ndarray[tuple[U, N], np.dtype[np.bool]] = (
            np.greater(lower, upper) & interval_nonempty
        )
        lower, upper = _where(flip, upper, lower), _where(flip, lower, upper)

        # 5. 0b1111...1111
        allbits = np.array(-1, dtype=upper.dtype.str.replace("u", "i")).view(
            upper.dtype
        )

        # 6. if there are several intervals, pick the one with the smallest
        #    lower bound, ensuring that empty intervals are not picked
        least = _where(interval_nonempty, lower, allbits).argmin(axis=0)
        lower, upper = lower[least, np.arange(n)], upper[least, np.arange(n)]
        negative = negative[least, np.arange(n)]
        assert np.all(lower <= upper)

        # 7. count the number of leading zero bits in lower and upper
        lower_lz: np.ndarray[tuple[U, N], np.dtype[np.uint8]] = _count_leading_zeros(
            lower
        )
        upper_lz: np.ndarray[tuple[U, N], np.dtype[np.uint8]] = _count_leading_zeros(
            upper
        )

        # 8. if upper_lz < lower_lz,
        #    i.e. ceil(log2(upper)) > ceil(log2(lower)),
        #    2 ** ceil(log2(lower)) is a tighter upper bound
        #
        #    upper: 0b00..01xxxxxxxxxxx
        #    lower: 0b00..00..01yyyyyyy
        # -> upper: 0b00..00..100000000
        #
        #    we actually end up choosing, if possible (larger than upper above)
        #    upper: 0b00..0010000000000
        #    since ~half the upper bound works well for symmetric intervals
        np.copyto(
            upper,
            (allbits >> _np_minimum(upper_lz + 2, lower_lz)) + 1,
            where=np.less(upper_lz, lower_lz),
            casting="no",
        )

        # 9. count the number of leading zero bits in (lower ^ upper) to find
        #    the most significant bit where lower and upper differ.
        #    Since upper > lower, at this bit i, upper[i] = 1 and lower[i] = 0.
        lxu_lz = _count_leading_zeros(lower ^ upper)

        # 10. pick such that the binary difference starts and ends with a
        #     maximally long sequence of zeros
        #
        #     upper: 0b00..01xxx1zzzzzzz
        #     lower: 0b00..01yyy0wwwwwww
        #   -> pick: 0b00..01xxx10000000
        pick_bits = _ensure_array(upper & ~(allbits >> (lxu_lz + 1))).reshape(1, -1)

        # 11. undo the negation step to allow "negative" unsigned values again
        np.copyto(pick_bits, ~pick_bits + 1, where=negative, casting="no")

        # 12. undo the difference with prediction and enforce the special case
        #     from (a) that prediction values inside the interval are kept as-is
        pick_bits += prediction_bits
        np.copyto(pick_bits, prediction_bits, where=contains_prediction, casting="no")

        # 13. convert everything back from total-ordered bits to value space
        pick: np.ndarray[S, np.dtype[T]] = _reshape(
            from_total_order(pick_bits, prediction.dtype), prediction.shape
        )
        assert np.all(self.contains(pick))

        return pick

    def pick(
        self, prediction: np.ndarray[S, np.dtype[T]]
    ) -> np.ndarray[S, np.dtype[T]]:
        """
        Pick a member of the interval union that minimises the cost of correcting the prediction to be a member of the interval union.

        The metric for minimising the correction cost is unspecified and may be
        approximate.

        Parameters
        ----------
        prediction : np.ndarray[S, np.dtype[T]]
            A prediction for a member of the interval union

        Returns
        -------
        member : np.ndarray[S, np.dtype[T]]
            A member of the interval union
        """

        # ensure that the prediction size and dtype match the interval union
        assert prediction.size == self._lower.shape[1]
        assert prediction.dtype == self._lower.dtype

        # ensure that every element has a non-empty interval
        assert np.all(to_total_order(self._lower[0]) <= to_total_order(self._upper[0]))

        return self._pick_more_zeros(prediction)

    @override
    def __repr__(self) -> str:
        return f"IntervalUnion(lower={self._lower!r}, upper={self._upper!r})"


def _count_leading_zeros(
    x: np.ndarray[S, np.dtype[np.unsignedinteger]],
) -> np.ndarray[S, np.dtype[np.uint8]]:
    """
    https://stackoverflow.com/a/79189999
    """

    nbits = np.iinfo(x.dtype).bits

    assert nbits <= 64

    if nbits <= 16:
        # safe cast from integer type to a larger integer type,
        # then lossless truncation of the number of leading zeros
        return np.subtract(  # type: ignore
            nbits, np.frexp(x.astype(np.uint32, casting="safe"))[1]
        ).astype(np.uint8, casting="unsafe")

    if nbits <= 32:
        # safe cast from integer type to a larger integer type,
        # then lossless truncation of the number of leading zeros
        return np.subtract(  # type: ignore
            nbits, np.frexp(x.astype(np.uint64, casting="safe"))[1]
        ).astype(np.uint8, casting="unsafe")

    # nbits <= 64
    # safe cast from integer type to a larger integer type,
    _, high_exp = np.frexp(x.astype(np.uint64, casting="safe") >> 32)
    _, low_exp = np.frexp(x.astype(np.uint64, casting="safe") & 0xFFFFFFFF)
    # then lossless truncation of the number of leading zeros
    high_exp += 32
    exp = _ensure_array(low_exp)
    np.copyto(exp, high_exp, where=(high_exp > 32), casting="no")
    return (nbits - exp).astype(np.uint8, casting="unsafe")
