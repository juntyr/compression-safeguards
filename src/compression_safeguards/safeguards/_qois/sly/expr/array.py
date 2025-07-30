from collections.abc import Mapping
from typing import Callable

import numpy as np

from .....utils.bindings import Parameter
from .....utils.typing import F, S
from .abc import Expr
from .constfold import FoldedScalarConst


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

    @property
    def late_bound_constants(self) -> frozenset[Parameter]:
        late_bound = set()
        for e in self._array.flat:
            late_bound.update(e.late_bound_constants)
        return frozenset(late_bound)

    def constant_fold(self, dtype: np.dtype[F]) -> F | Expr:
        return Array.map_unary(
            self, lambda e: FoldedScalarConst.constant_fold_expr(e, dtype)
        )

    def eval(
        self,
        X: np.ndarray[tuple[int, ...], np.dtype[F]],
        late_bound: Mapping[Parameter, np.ndarray[tuple[int, ...], np.dtype[F]]],
    ) -> F | np.ndarray[tuple[int, ...], np.dtype[F]]:
        return np.fromiter(
            (e.eval(X, late_bound) for e in self._array.flat),
            dtype=X.dtype,
            count=self._array.size,
        ).reshape(self._array.shape)

    def compute_data_error_bound_unchecked(
        self,
        eb_expr_lower: np.ndarray[S, np.dtype[F]],
        eb_expr_upper: np.ndarray[S, np.dtype[F]],
        X: np.ndarray[tuple[int, ...], np.dtype[F]],
        late_bound: Mapping[Parameter, np.ndarray[tuple[int, ...], np.dtype[F]]],
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

    def __repr__(self) -> str:
        return f"{self._array!r}"
