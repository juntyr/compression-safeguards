from abc import abstractmethod
from typing import Callable

import numpy as np

from ....utils.typing import F, S
from ..eb import ensure_bounded_derived_error


class Expr:
    __slots__ = ()

    @property
    @abstractmethod
    def has_data(self) -> bool:
        pass

    @abstractmethod
    def constant_fold(self, dtype: np.dtype[F]) -> F | "Expr":
        pass

    @abstractmethod
    def eval(
        self, X: np.ndarray[tuple[int, ...], np.dtype[F]]
    ) -> F | np.ndarray[tuple[int, ...], np.dtype[F]]:
        pass

    @abstractmethod
    def compute_data_error_bound_unchecked(
        self,
        eb_expr_lower: np.ndarray[S, np.dtype[F]],
        eb_expr_upper: np.ndarray[S, np.dtype[F]],
        X: np.ndarray[tuple[int, ...], np.dtype[F]],
    ) -> tuple[np.ndarray[S, np.dtype[F]], np.ndarray[S, np.dtype[F]]]:
        pass

    def compute_data_error_bound(
        self,
        eb_expr_lower: np.ndarray[S, np.dtype[F]],
        eb_expr_upper: np.ndarray[S, np.dtype[F]],
        X: np.ndarray[tuple[int, ...], np.dtype[F]],
    ) -> tuple[np.ndarray[S, np.dtype[F]], np.ndarray[S, np.dtype[F]]]:
        tl, tu = self.compute_data_error_bound_unchecked(
            eb_expr_lower, eb_expr_upper, X
        )

        exprv_: F | np.ndarray[tuple[int, ...], np.dtype[F]] = self.eval(X)
        assert isinstance(exprv_, np.ndarray)
        assert exprv_.shape == X.shape
        exprv: np.ndarray[S, np.dtype[F]] = exprv_  # type: ignore

        # handle rounding errors in the lower error bound computation
        tl = ensure_bounded_derived_error(
            lambda tl: np.where(  # type: ignore
                tl == 0,
                exprv,
                self.eval(X + tl),
            ),
            exprv,
            X,  # type: ignore
            tl,
            eb_expr_lower,
            eb_expr_upper,
        )
        tu = ensure_bounded_derived_error(
            lambda tu: np.where(  # type: ignore
                tu == 0,
                exprv,
                self.eval(X + tu),
            ),
            exprv,
            X,  # type: ignore
            tu,
            eb_expr_lower,
            eb_expr_upper,
        )

        return tl, tu


class Group(Expr):
    __slots__ = ("_expr",)
    _expr: Expr

    def __init__(self, expr: Expr):
        self._expr = expr._expr if isinstance(expr, Group) else expr

    @property
    def has_data(self) -> bool:
        return self._expr.has_data

    def constant_fold(self, dtype: np.dtype[F]) -> F | "Expr":
        fexpr = self._expr.constant_fold(dtype)
        # fully constant folded -> allow further folding
        if isinstance(fexpr, dtype.type):
            return fexpr
        # partially / not constant folded -> stop further folding
        return Group(fexpr)  # type: ignore

    def eval(
        self, X: np.ndarray[tuple[int, ...], np.dtype[F]]
    ) -> F | np.ndarray[tuple[int, ...], np.dtype[F]]:
        return self._expr.eval(X)

    def compute_data_error_bound_unchecked(
        self,
        eb_expr_lower: np.ndarray[S, np.dtype[F]],
        eb_expr_upper: np.ndarray[S, np.dtype[F]],
        X: np.ndarray[tuple[int, ...], np.dtype[F]],
    ) -> tuple[np.ndarray[S, np.dtype[F]], np.ndarray[S, np.dtype[F]]]:
        return self._expr.compute_data_error_bound(eb_expr_lower, eb_expr_upper, X)

    def compute_data_error_bound(
        self,
        eb_expr_lower: np.ndarray[S, np.dtype[F]],
        eb_expr_upper: np.ndarray[S, np.dtype[F]],
        X: np.ndarray[tuple[int, ...], np.dtype[F]],
    ) -> tuple[np.ndarray[S, np.dtype[F]], np.ndarray[S, np.dtype[F]]]:
        return self.compute_data_error_bound_unchecked(eb_expr_lower, eb_expr_upper, X)


class FoldedScalarConst(Expr):
    __slots__ = ("_const",)
    _const: np.number

    def __init__(self, const: np.number):
        self._const = const

    @property
    def has_data(self) -> bool:
        return False

    def constant_fold(self, dtype: np.dtype[F]) -> F | Expr:
        assert dtype == np.dtype(self._const)
        return self._const  # type: ignore

    def eval(
        self, X: np.ndarray[tuple[int, ...], np.dtype[F]]
    ) -> F | np.ndarray[tuple[int, ...], np.dtype[F]]:
        assert X.dtype == np.dtype(self._const)
        return self._const  # type: ignore

    def compute_data_error_bound_unchecked(
        self,
        eb_expr_lower: np.ndarray[S, np.dtype[F]],
        eb_expr_upper: np.ndarray[S, np.dtype[F]],
        X: np.ndarray[tuple[int, ...], np.dtype[F]],
    ) -> tuple[np.ndarray[S, np.dtype[F]], np.ndarray[S, np.dtype[F]]]:
        assert False, "constants have no error bounds"

    @staticmethod
    def constant_fold_unary(
        expr: Expr, dtype: np.dtype[F], m: Callable[[F], F]
    ) -> F | Expr:
        fexpr = expr.constant_fold(dtype)
        if isinstance(fexpr, dtype.type):
            return m(fexpr)
        return fexpr

    @staticmethod
    def constant_fold_binary(
        left: Expr,
        right: Expr,
        dtype: np.dtype[F],
        m: Callable[[F, F], F],
        rm: Callable[[Expr, Expr], Expr],
    ) -> F | Expr:
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

    def constant_fold(self, dtype: np.dtype[F]) -> F | Expr:
        return self

    def eval(
        self, X: np.ndarray[tuple[int, ...], np.dtype[F]]
    ) -> F | np.ndarray[tuple[int, ...], np.dtype[F]]:
        return X

    def compute_data_error_bound_unchecked(
        self,
        eb_expr_lower: np.ndarray[S, np.dtype[F]],
        eb_expr_upper: np.ndarray[S, np.dtype[F]],
        X: np.ndarray[tuple[int, ...], np.dtype[F]],
    ) -> tuple[np.ndarray[S, np.dtype[F]], np.ndarray[S, np.dtype[F]]]:
        return (eb_expr_lower, eb_expr_upper)


class DataArrayElement(Expr):
    __slots__ = ("_index",)
    _index: tuple[int, ...]

    def __init__(self, index: tuple[int, ...]):
        self._index = index

    @property
    def has_data(self) -> bool:
        return True

    def constant_fold(self, dtype: np.dtype[F]) -> F | Expr:
        return self

    def eval(
        self, X: np.ndarray[tuple[int, ...], np.dtype[F]]
    ) -> F | np.ndarray[tuple[int, ...], np.dtype[F]]:
        return X[(...,) + self._index]

    def compute_data_error_bound_unchecked(
        self,
        eb_expr_lower: np.ndarray[S, np.dtype[F]],
        eb_expr_upper: np.ndarray[S, np.dtype[F]],
        X: np.ndarray[tuple[int, ...], np.dtype[F]],
    ) -> tuple[np.ndarray[S, np.dtype[F]], np.ndarray[S, np.dtype[F]]]:
        return (eb_expr_lower, eb_expr_upper)


class Number(Expr):
    __slots__ = ("_n",)
    _n: str

    def __init__(self, n: str):
        self._n = n

    @property
    def has_data(self) -> bool:
        return False

    def constant_fold(self, dtype: np.dtype[F]) -> F | Expr:
        return dtype.type(self._n)  # type: ignore

    def eval(
        self, X: np.ndarray[tuple[int, ...], np.dtype[F]]
    ) -> F | np.ndarray[tuple[int, ...], np.dtype[F]]:
        return X.dtype.type(self._n)  # type: ignore

    def compute_data_error_bound_unchecked(
        self,
        eb_expr_lower: np.ndarray[S, np.dtype[F]],
        eb_expr_upper: np.ndarray[S, np.dtype[F]],
        X: np.ndarray[tuple[int, ...], np.dtype[F]],
    ) -> tuple[np.ndarray[S, np.dtype[F]], np.ndarray[S, np.dtype[F]]]:
        assert False, "constants have no error bounds"


class Pi(Expr):
    __slots__ = ()

    @property
    def has_data(self) -> bool:
        return False

    def constant_fold(self, dtype: np.dtype[F]) -> F | Expr:
        return dtype.type(np.pi)

    def eval(
        self, X: np.ndarray[tuple[int, ...], np.dtype[F]]
    ) -> F | np.ndarray[tuple[int, ...], np.dtype[F]]:
        return X.dtype.type(np.pi)

    def compute_data_error_bound_unchecked(
        self,
        eb_expr_lower: np.ndarray[S, np.dtype[F]],
        eb_expr_upper: np.ndarray[S, np.dtype[F]],
        X: np.ndarray[tuple[int, ...], np.dtype[F]],
    ) -> tuple[np.ndarray[S, np.dtype[F]], np.ndarray[S, np.dtype[F]]]:
        assert False, "constants have no error bounds"


class Euler(Expr):
    __slots__ = ()

    @property
    def has_data(self) -> bool:
        return False

    def constant_fold(self, dtype: np.dtype[F]) -> F | Expr:
        return dtype.type(np.e)

    def eval(
        self, X: np.ndarray[tuple[int, ...], np.dtype[F]]
    ) -> F | np.ndarray[tuple[int, ...], np.dtype[F]]:
        return X.dtype.type(np.e)

    def compute_data_error_bound_unchecked(
        self,
        eb_expr_lower: np.ndarray[S, np.dtype[F]],
        eb_expr_upper: np.ndarray[S, np.dtype[F]],
        X: np.ndarray[tuple[int, ...], np.dtype[F]],
    ) -> tuple[np.ndarray[S, np.dtype[F]], np.ndarray[S, np.dtype[F]]]:
        assert False, "constants have no error bounds"


class ScalarNegate(Expr):
    __slots__ = ("_a",)
    _a: Expr

    def __init__(self, a: Expr):
        self._a = a

    @property
    def has_data(self) -> bool:
        return self._a.has_data

    def constant_fold(self, dtype: np.dtype[F]) -> F | Expr:
        return FoldedScalarConst.constant_fold_unary(self._a, dtype, np.negative)

    def eval(
        self, X: np.ndarray[tuple[int, ...], np.dtype[F]]
    ) -> F | np.ndarray[tuple[int, ...], np.dtype[F]]:
        return np.negative(self._a.eval(X))

    def compute_data_error_bound_unchecked(
        self,
        eb_expr_lower: np.ndarray[S, np.dtype[F]],
        eb_expr_upper: np.ndarray[S, np.dtype[F]],
        X: np.ndarray[tuple[int, ...], np.dtype[F]],
    ) -> tuple[np.ndarray[S, np.dtype[F]], np.ndarray[S, np.dtype[F]]]:
        return self._a.compute_data_error_bound(-eb_expr_upper, -eb_expr_lower, X)

    def compute_data_error_bound(
        self,
        eb_expr_lower: np.ndarray[S, np.dtype[F]],
        eb_expr_upper: np.ndarray[S, np.dtype[F]],
        X: np.ndarray[tuple[int, ...], np.dtype[F]],
    ) -> tuple[np.ndarray[S, np.dtype[F]], np.ndarray[S, np.dtype[F]]]:
        return self.compute_data_error_bound_unchecked(eb_expr_lower, eb_expr_upper, X)


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

    def constant_fold(self, dtype: np.dtype[F]) -> F | Expr:
        return FoldedScalarConst.constant_fold_binary(
            self._a, self._b, dtype, np.add, ScalarAdd
        )

    def eval(
        self, X: np.ndarray[tuple[int, ...], np.dtype[F]]
    ) -> F | np.ndarray[tuple[int, ...], np.dtype[F]]:
        return np.add(self._a.eval(X), self._b.eval(X))

    def compute_data_error_bound_unchecked(
        self,
        eb_expr_lower: np.ndarray[S, np.dtype[F]],
        eb_expr_upper: np.ndarray[S, np.dtype[F]],
        X: np.ndarray[tuple[int, ...], np.dtype[F]],
    ) -> tuple[np.ndarray[S, np.dtype[F]], np.ndarray[S, np.dtype[F]]]:
        raise NotImplementedError


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

    def constant_fold(self, dtype: np.dtype[F]) -> F | Expr:
        return FoldedScalarConst.constant_fold_binary(
            self._a, self._b, dtype, np.multiply, ScalarMultiply
        )

    def eval(
        self, X: np.ndarray[tuple[int, ...], np.dtype[F]]
    ) -> F | np.ndarray[tuple[int, ...], np.dtype[F]]:
        return np.multiply(self._a.eval(X), self._b.eval(X))

    def compute_data_error_bound_unchecked(
        self,
        eb_expr_lower: np.ndarray[S, np.dtype[F]],
        eb_expr_upper: np.ndarray[S, np.dtype[F]],
        X: np.ndarray[tuple[int, ...], np.dtype[F]],
    ) -> tuple[np.ndarray[S, np.dtype[F]], np.ndarray[S, np.dtype[F]]]:
        raise NotImplementedError


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

    def constant_fold(self, dtype: np.dtype[F]) -> F | Expr:
        return FoldedScalarConst.constant_fold_binary(
            self._a, self._b, dtype, np.divide, ScalarDivide
        )

    def eval(
        self, X: np.ndarray[tuple[int, ...], np.dtype[F]]
    ) -> F | np.ndarray[tuple[int, ...], np.dtype[F]]:
        return np.divide(self._a.eval(X), self._b.eval(X))

    def compute_data_error_bound_unchecked(
        self,
        eb_expr_lower: np.ndarray[S, np.dtype[F]],
        eb_expr_upper: np.ndarray[S, np.dtype[F]],
        X: np.ndarray[tuple[int, ...], np.dtype[F]],
    ) -> tuple[np.ndarray[S, np.dtype[F]], np.ndarray[S, np.dtype[F]]]:
        raise NotImplementedError


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

    def constant_fold(self, dtype: np.dtype[F]) -> F | Expr:
        return FoldedScalarConst.constant_fold_binary(
            self._a, self._b, dtype, np.power, ScalarExponentiation
        )

    def eval(
        self, X: np.ndarray[tuple[int, ...], np.dtype[F]]
    ) -> F | np.ndarray[tuple[int, ...], np.dtype[F]]:
        return np.power(self._a.eval(X), self._b.eval(X))

    def compute_data_error_bound_unchecked(
        self,
        eb_expr_lower: np.ndarray[S, np.dtype[F]],
        eb_expr_upper: np.ndarray[S, np.dtype[F]],
        X: np.ndarray[tuple[int, ...], np.dtype[F]],
    ) -> tuple[np.ndarray[S, np.dtype[F]], np.ndarray[S, np.dtype[F]]]:
        raise NotImplementedError


class ScalarLn(Expr):
    __slots__ = ("_a",)
    _a: Expr

    def __init__(self, a: Expr):
        self._a = a

    @property
    def has_data(self) -> bool:
        return self._a.has_data

    def constant_fold(self, dtype: np.dtype[F]) -> F | Expr:
        return FoldedScalarConst.constant_fold_unary(self._a, dtype, np.log)

    def eval(
        self, X: np.ndarray[tuple[int, ...], np.dtype[F]]
    ) -> F | np.ndarray[tuple[int, ...], np.dtype[F]]:
        return np.log(self._a.eval(X))

    def compute_data_error_bound_unchecked(
        self,
        eb_expr_lower: np.ndarray[S, np.dtype[F]],
        eb_expr_upper: np.ndarray[S, np.dtype[F]],
        X: np.ndarray[tuple[int, ...], np.dtype[F]],
    ) -> tuple[np.ndarray[S, np.dtype[F]], np.ndarray[S, np.dtype[F]]]:
        raise NotImplementedError


class ScalarExp(Expr):
    __slots__ = ("_a",)
    _a: Expr

    def __init__(self, a: Expr):
        self._a = a

    @property
    def has_data(self) -> bool:
        return self._a.has_data

    def constant_fold(self, dtype: np.dtype[F]) -> F | Expr:
        return FoldedScalarConst.constant_fold_unary(self._a, dtype, np.exp)

    def eval(
        self, X: np.ndarray[tuple[int, ...], np.dtype[F]]
    ) -> F | np.ndarray[tuple[int, ...], np.dtype[F]]:
        return np.exp(self._a.eval(X))

    def compute_data_error_bound_unchecked(
        self,
        eb_expr_lower: np.ndarray[S, np.dtype[F]],
        eb_expr_upper: np.ndarray[S, np.dtype[F]],
        X: np.ndarray[tuple[int, ...], np.dtype[F]],
    ) -> tuple[np.ndarray[S, np.dtype[F]], np.ndarray[S, np.dtype[F]]]:
        raise NotImplementedError


class Array(Expr):
    __slots__ = ("_array",)
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

    def constant_fold(self, dtype: np.dtype[F]) -> F | Expr:
        def fold_array_element(e: Expr) -> Expr:
            fe = e.constant_fold(dtype)
            if isinstance(fe, dtype.type):
                return FoldedScalarConst(fe)
            return fe  # type: ignore

        return Array.map_unary(self, fold_array_element)

    def eval(
        self, X: np.ndarray[tuple[int, ...], np.dtype[F]]
    ) -> F | np.ndarray[tuple[int, ...], np.dtype[F]]:
        return np.fromiter(
            (e.eval(X) for e in self._array.flat),
            dtype=X.dtype,
            count=self._array.size,
        ).reshape(self._array.shape)

    def compute_data_error_bound_unchecked(
        self,
        eb_expr_lower: np.ndarray[S, np.dtype[F]],
        eb_expr_upper: np.ndarray[S, np.dtype[F]],
        X: np.ndarray[tuple[int, ...], np.dtype[F]],
    ) -> tuple[np.ndarray[S, np.dtype[F]], np.ndarray[S, np.dtype[F]]]:
        assert False, "cannot derive error bounds over an array expression"

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
                    (m(le, ri) for le, ri in zip(left._array.flat, right._array.flat)),
                    dtype=object,
                    count=left._array.size,
                ).reshape(left._array.shape)
                return out

            out = Array.__new__(Array)
            out._array = np.fromiter(
                (m(le, right) for le in left._array.flat),
                dtype=object,
                count=left._array.size,
            ).reshape(left._array.shape)
            return out

        if isinstance(right, Array):
            out = Array.__new__(Array)
            out._array = np.fromiter(
                (m(left, ri) for ri in right._array.flat),
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

    def transpose(self) -> "Array":
        out = Array.__new__(Array)
        out._array = self._array.T
        return out
