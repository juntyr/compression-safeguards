from abc import abstractmethod
from typing import Callable

import numpy as np

from ....utils.typing import T


class Expr:
    __slots__ = ()

    @property
    @abstractmethod
    def has_data(self) -> bool:
        pass

    @abstractmethod
    def constant_fold(self, dtype: np.dtype[T]) -> T | "Expr":
        pass

    @abstractmethod
    def eval(
        self, dtype: np.dtype[T], X: np.ndarray[tuple[int, ...], np.dtype[T]]
    ) -> T | np.ndarray[tuple[int, ...], np.dtype[T]]:
        pass


class FoldedScalarConst(Expr):
    __slots__ = "_const"
    _const: np.number

    def __init__(self, const: np.number):
        self._const = const

    @property
    def has_data(self) -> bool:
        return False

    def constant_fold(self, dtype: np.dtype[T]) -> T | Expr:
        assert dtype == np.dtype(self._const)
        return self._const  # type: ignore

    def eval(
        self, dtype: np.dtype[T], X: np.ndarray[tuple[int, ...], np.dtype[T]]
    ) -> T | np.ndarray[tuple[int, ...], np.dtype[T]]:
        assert dtype == np.dtype(self._const)
        return self._const  # type: ignore

    @staticmethod
    def constant_fold_unary(
        expr: Expr, dtype: np.dtype[T], m: Callable[[T], T]
    ) -> T | Expr:
        fexpr = expr.constant_fold(dtype)
        if isinstance(fexpr, dtype.type):
            return m(fexpr)
        return fexpr

    @staticmethod
    def constant_fold_binary(
        left: Expr,
        right: Expr,
        dtype: np.dtype[T],
        m: Callable[[T, T], T],
        rm: Callable[[Expr, Expr], Expr],
    ) -> T | Expr:
        fleft = left.constant_fold(dtype)
        fright = right.constant_fold(dtype)
        if isinstance(fleft, dtype.type):
            if isinstance(fright, dtype.type):
                return m(fleft, fright)
            return rm(FoldedScalarConst(fleft), fright)  # type: ignore
        if isinstance(fright, dtype.type):
            return rm(fleft, FoldedScalarConst(fright))  # type: ignore
        return rm(fleft, fright)  # type: ignore


class DataScalar(Expr):
    __slots__ = ()

    @property
    def has_data(self) -> bool:
        return True

    def constant_fold(self, dtype: np.dtype[T]) -> T | Expr:
        return self

    @abstractmethod
    def eval(
        self, dtype: np.dtype[T], X: np.ndarray[tuple[int, ...], np.dtype[T]]
    ) -> T | np.ndarray[tuple[int, ...], np.dtype[T]]:
        return X


class DataArrayElement(Expr):
    __slots__ = "_index"
    _index: tuple[int, ...]

    def __init__(self, index: tuple[int, ...]):
        self._index = index

    @property
    def has_data(self) -> bool:
        return True

    def constant_fold(self, dtype: np.dtype[T]) -> T | Expr:
        return self

    @abstractmethod
    def eval(
        self, dtype: np.dtype[T], X: np.ndarray[tuple[int, ...], np.dtype[T]]
    ) -> T | np.ndarray[tuple[int, ...], np.dtype[T]]:
        return X[(...,) + self._index]


class Number(Expr):
    __slots__ = "_n"
    _n: str

    def __init__(self, n: str):
        self._n = n

    @property
    def has_data(self) -> bool:
        return False

    def constant_fold(self, dtype: np.dtype[T]) -> T | Expr:
        return dtype.type(self._n)  # type: ignore

    def eval(
        self, dtype: np.dtype[T], X: np.ndarray[tuple[int, ...], np.dtype[T]]
    ) -> T | np.ndarray[tuple[int, ...], np.dtype[T]]:
        return dtype.type(self._n)  # type: ignore


class Pi(Expr):
    __slots__ = ()

    @property
    def has_data(self) -> bool:
        return False

    def constant_fold(self, dtype: np.dtype[T]) -> T | Expr:
        return dtype.type(np.pi)

    def eval(
        self, dtype: np.dtype[T], X: np.ndarray[tuple[int, ...], np.dtype[T]]
    ) -> T | np.ndarray[tuple[int, ...], np.dtype[T]]:
        return dtype.type(np.pi)


class Euler(Expr):
    __slots__ = ()

    @property
    def has_data(self) -> bool:
        return False

    def constant_fold(self, dtype: np.dtype[T]) -> T | Expr:
        return dtype.type(np.e)

    def eval(
        self, dtype: np.dtype[T], X: np.ndarray[tuple[int, ...], np.dtype[T]]
    ) -> T | np.ndarray[tuple[int, ...], np.dtype[T]]:
        return dtype.type(np.e)


class ScalarNegate(Expr):
    __slots__ = ("_a",)
    _a: Expr

    def __init__(self, a: Expr):
        self._a = a

    @property
    def has_data(self) -> bool:
        return self._a.has_data

    def constant_fold(self, dtype: np.dtype[T]) -> T | Expr:
        return FoldedScalarConst.constant_fold_unary(self._a, dtype, np.negative)

    def eval(
        self, dtype: np.dtype[T], X: np.ndarray[tuple[int, ...], np.dtype[T]]
    ) -> T | np.ndarray[tuple[int, ...], np.dtype[T]]:
        return np.negative(self._a.eval(dtype, X))


class ScalarAdd(Expr):
    __slots__ = ("_a", "_b")
    _a: Expr
    _b: Expr

    def __init__(self, a: Expr, b: Expr):
        self._a = a
        self._b = b

    @property
    def has_data(self) -> bool:
        return self._a.has_data or self._b.has_data

    def constant_fold(self, dtype: np.dtype[T]) -> T | Expr:
        return FoldedScalarConst.constant_fold_binary(
            self._a, self._b, dtype, np.add, ScalarAdd
        )

    def eval(
        self, dtype: np.dtype[T], X: np.ndarray[tuple[int, ...], np.dtype[T]]
    ) -> T | np.ndarray[tuple[int, ...], np.dtype[T]]:
        return np.add(self._a.eval(dtype, X), self._b.eval(dtype, X))


class ScalarMultiply(Expr):
    __slots__ = ("_a", "_b")
    _a: Expr
    _b: Expr

    def __init__(self, a: Expr, b: Expr):
        self._a = a
        self._b = b

    @property
    def has_data(self) -> bool:
        return self._a.has_data or self._b.has_data

    def constant_fold(self, dtype: np.dtype[T]) -> T | Expr:
        return FoldedScalarConst.constant_fold_binary(
            self._a, self._b, dtype, np.multiply, ScalarMultiply
        )

    def eval(
        self, dtype: np.dtype[T], X: np.ndarray[tuple[int, ...], np.dtype[T]]
    ) -> T | np.ndarray[tuple[int, ...], np.dtype[T]]:
        return np.multiply(self._a.eval(dtype, X), self._b.eval(dtype, X))


class ScalarDivide(Expr):
    __slots__ = ("_a", "_b")
    _a: Expr
    _b: Expr

    def __init__(self, a: Expr, b: Expr):
        self._a = a
        self._b = b

    @property
    def has_data(self) -> bool:
        return self._a.has_data or self._b.has_data

    def constant_fold(self, dtype: np.dtype[T]) -> T | Expr:
        return FoldedScalarConst.constant_fold_binary(
            self._a, self._b, dtype, np.divide, ScalarDivide
        )

    def eval(
        self, dtype: np.dtype[T], X: np.ndarray[tuple[int, ...], np.dtype[T]]
    ) -> T | np.ndarray[tuple[int, ...], np.dtype[T]]:
        return np.divide(self._a.eval(dtype, X), self._b.eval(dtype, X))


class ScalarExponentiation(Expr):
    __slots__ = ("_a", "_b")
    _a: Expr
    _b: Expr

    def __init__(self, a: Expr, b: Expr):
        self._a = a
        self._b = b

    @property
    def has_data(self) -> bool:
        return self._a.has_data or self._b.has_data

    def constant_fold(self, dtype: np.dtype[T]) -> T | Expr:
        return FoldedScalarConst.constant_fold_binary(
            self._a, self._b, dtype, np.power, ScalarExponentiation
        )

    def eval(
        self, dtype: np.dtype[T], X: np.ndarray[tuple[int, ...], np.dtype[T]]
    ) -> T | np.ndarray[tuple[int, ...], np.dtype[T]]:
        return np.power(self._a.eval(dtype, X), self._b.eval(dtype, X))


class ScalarLn(Expr):
    __slots__ = ("_a",)
    _a: Expr

    def __init__(self, a: Expr):
        self._a = a

    @property
    def has_data(self) -> bool:
        return self._a.has_data

    def constant_fold(self, dtype: np.dtype[T]) -> T | Expr:
        return FoldedScalarConst.constant_fold_unary(self._a, dtype, np.log)

    def eval(
        self, dtype: np.dtype[T], X: np.ndarray[tuple[int, ...], np.dtype[T]]
    ) -> T | np.ndarray[tuple[int, ...], np.dtype[T]]:
        return np.log(self._a.eval(dtype, X))


class ScalarExp(Expr):
    __slots__ = ("_a",)
    _a: Expr

    def __init__(self, a: Expr):
        self._a = a

    @property
    def has_data(self) -> bool:
        return self._a.has_data

    def constant_fold(self, dtype: np.dtype[T]) -> T | Expr:
        return FoldedScalarConst.constant_fold_unary(self._a, dtype, np.exp)

    def eval(
        self, dtype: np.dtype[T], X: np.ndarray[tuple[int, ...], np.dtype[T]]
    ) -> T | np.ndarray[tuple[int, ...], np.dtype[T]]:
        return np.exp(self._a.eval(dtype, X))


class Array(Expr):
    __slots__ = "_array"
    _array: np.ndarray

    def __init__(self, el: Expr, *els: Expr):
        if isinstance(el, Array):
            aels = [el._array]
            for e in els:
                assert isinstance(e, Array) and e.shape == el.shape
                aels.append(e._array)
            self._array = np.array(aels)
        else:
            for e in els:
                assert not isinstance(e, Array)
            self._array = np.array((el,) + els)

    @property
    def has_data(self) -> bool:
        return any(e.has_data for e in self._array.flat)

    def constant_fold(self, dtype: np.dtype[T]) -> T | Expr:
        def fold_array_element(e: Expr) -> Expr:
            fe = e.constant_fold(dtype)
            if isinstance(fe, dtype.type):
                return FoldedScalarConst(fe)
            return fe  # type: ignore

        return Array.map_unary(self, fold_array_element)

    def eval(
        self, dtype: np.dtype[T], X: np.ndarray[tuple[int, ...], np.dtype[T]]
    ) -> T | np.ndarray[tuple[int, ...], np.dtype[T]]:
        return np.fromiter(
            (e.eval(dtype, X) for e in self._array.flat),
            dtype=dtype,
            count=self._array.size,
        ).reshape(self._array.shape)

    @property
    def shape(self) -> tuple[int, ...]:
        return self._array.shape

    @staticmethod
    def map_unary(expr: Expr, m: Callable[[Expr], Expr]) -> Expr:
        if isinstance(expr, Array):
            out = Array.__new__(Array)
            out._array = np.fromiter(
                (m(e) for e in expr._array.flat), dtype=object, count=expr._array.size
            ).reshape(expr._array.shape)
            return out
        return m(expr)

    @staticmethod
    def map_binary(left: Expr, right: Expr, m: Callable[[Expr, Expr], Expr]) -> Expr:
        if isinstance(left, Array):
            if isinstance(right, Array):
                assert left.shape == right.shape
                out = Array.__new__(Array)
                out._array = np.fromiter(
                    (m(l, r) for l, r in zip(left._array.flat, right._array.flat)),
                    dtype=object,
                    count=left._array.size,
                ).reshape(left._array.shape)
                return out

            out = Array.__new__(Array)
            out._array = np.fromiter(
                (m(l, right) for l in left._array.flat),
                dtype=object,
                count=left._array.size,
            ).reshape(left._array.shape)
            return out

        if isinstance(right, Array):
            out = Array.__new__(Array)
            out._array = np.fromiter(
                (m(left, r) for r in right._array.flat),
                dtype=object,
                count=right._array.size,
            ).reshape(right._array.shape)
            return out

        return m(left, right)

    def index(self, index: tuple[int | slice, ...]) -> Expr:
        a = self._array[index]
        if isinstance(a, np.ndarray):
            out = Array.__new__(Array)
            out._array = a
            return out
        return a
