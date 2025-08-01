import itertools
from collections.abc import Mapping
from typing import Callable

import numpy as np

from .....utils.bindings import Parameter
from .abc import Expr
from .addsub import ScalarAdd
from .constfold import FoldedScalarConst
from .data import Data
from .divmul import ScalarMultiply
from .group import Group
from .typing import F, Ns, Ps, PsI


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

    @staticmethod
    def from_data_shape(shape: tuple[int, ...]) -> "Array":
        out = Array.__new__(Array)
        out._array = np.empty(shape, dtype=object)

        for i in itertools.product(*[range(a) for a in shape]):
            out._array[i] = Data(index=i)

        return out

    @property
    def has_data(self) -> bool:
        return any(e.has_data for e in self._array.flat)

    @property
    def data_indices(self) -> frozenset[tuple[int, ...]]:
        indices = set()
        for e in self._array.flat:
            indices.update(e.data_indices)
        return frozenset(indices)

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
        x: PsI,
        Xs: np.ndarray[Ns, np.dtype[F]],
        late_bound: Mapping[Parameter, np.ndarray[Ns, np.dtype[F]]],
    ) -> np.ndarray[PsI, np.dtype[F]]:
        assert False, "cannot evaluate an array expression"

    def compute_data_error_bound_unchecked(
        self,
        eb_expr_lower: np.ndarray[Ps, np.dtype[F]],
        eb_expr_upper: np.ndarray[Ps, np.dtype[F]],
        X: np.ndarray[Ps, np.dtype[F]],
        Xs: np.ndarray[Ns, np.dtype[F]],
        late_bound: Mapping[Parameter, np.ndarray[Ns, np.dtype[F]]],
    ) -> tuple[np.ndarray[Ps, np.dtype[F]], np.ndarray[Ps, np.dtype[F]]]:
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

    def sum(self) -> Expr:
        acc = None
        for e in self._array.flat:
            acc = e if acc is None else ScalarAdd(acc, e)  # type: ignore
        assert acc is not None
        # we can return a group here since acc is not an array
        assert not isinstance(acc, Array)
        return Group(acc)

    @staticmethod
    def matmul(left: "Array", right: "Array") -> "Array":
        assert len(left.shape) == 2, "can only matmul(a, b) a 2D array a"
        assert len(right.shape) == 2, "can only matmul(a, b) a 2D array b"
        assert left.shape[1] == right.shape[0], (
            "can only matmul(a, b) with shapes (n, k) x (k, m) -> (n, m)"
        )
        out = Array.__new__(Array)
        out._array = np.empty((left.shape[0], right.shape[1]), dtype=object)
        for n in range(left.shape[0]):
            for m in range(right.shape[1]):
                acc = None
                for k in range(left.shape[1]):
                    kk = ScalarMultiply(left._array[n, k], right._array[k, m])
                    acc = kk if acc is None else ScalarAdd(acc, kk)  # type: ignore
                assert acc is not None
                # we can return a group here since acc is not an array
                assert not isinstance(acc, Array)
                out._array[n, m] = Group(acc)
        return out

    def __repr__(self) -> str:
        return f"{self._array!r}".removeprefix("array(").removesuffix(", dtype=object)")
