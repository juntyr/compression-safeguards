import itertools
from collections.abc import Iterator, Mapping
from typing import Callable

import numpy as np

from ....utils.bindings import Parameter
from .abc import Expr
from .addsub import ScalarAdd
from .constfold import ScalarFoldedConstant
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
                if not (isinstance(e, Array) and e.shape == el.shape):
                    raise ValueError(
                        f"elements must all have the consistent shape {el.shape}"
                    )
                aels.append(e._array)
            self._array = np.array(aels)
        else:
            for e in els:
                if isinstance(e, Array):
                    raise ValueError("elements must all be scalar")
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

    def apply_array_element_offset(
        self,
        axis: int,
        offset: int,
    ) -> Expr:
        return Array.map_unary(
            self, lambda e: e.apply_array_element_offset(axis, offset)
        )

    @property
    def late_bound_constants(self) -> frozenset[Parameter]:
        late_bound = set()
        for e in self._array.flat:
            late_bound.update(e.late_bound_constants)
        return frozenset(late_bound)

    def constant_fold(self, dtype: np.dtype[F]) -> F | Expr:
        return Array.map_unary(
            self, lambda e: ScalarFoldedConstant.constant_fold_expr(e, dtype)
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

    def compute_data_bounds_unchecked(
        self,
        expr_lower: np.ndarray[Ps, np.dtype[F]],
        expr_upper: np.ndarray[Ps, np.dtype[F]],
        X: np.ndarray[Ps, np.dtype[F]],
        Xs: np.ndarray[Ns, np.dtype[F]],
        late_bound: Mapping[Parameter, np.ndarray[Ns, np.dtype[F]]],
    ) -> tuple[np.ndarray[Ns, np.dtype[F]], np.ndarray[Ns, np.dtype[F]]]:
        assert False, "cannot derive data bounds over an array expression"

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
        if not (isinstance(left, Array) or isinstance(right, Array)):
            return m(left, right)

        shape = None
        size = 0
        iters: list[Iterator[Expr]] = []

        for i, expr in enumerate([left, right]):
            if isinstance(expr, Array):
                if shape is not None and expr.shape != shape:
                    raise ValueError(
                        f"shape mismatch between operands, expected {shape} but found {expr.shape} for operand {i + 1}"
                    )
                shape = expr.shape
                size = expr._array.size
                iters.append(expr._array.flat)
            else:
                iters.append(itertools.repeat(expr))

        out = Array.__new__(Array)
        out._array = np.fromiter(
            (m(li, ri) for li, ri in zip(*iters)),
            dtype=object,
            count=size,
        ).reshape(shape)
        return out

    @staticmethod
    def map_ternary(
        left: Expr, middle: Expr, right: Expr, m: Callable[[Expr, Expr, Expr], Expr]
    ) -> Expr:
        if not (
            isinstance(left, Array)
            or isinstance(middle, Array)
            or isinstance(right, Array)
        ):
            return m(left, middle, right)

        shape = None
        size = 0
        iters: list[Iterator[Expr]] = []

        for i, expr in enumerate([left, middle, right]):
            if isinstance(expr, Array):
                if shape is not None and expr.shape != shape:
                    raise ValueError(
                        f"shape mismatch between operands, expected {shape} but found {expr.shape} for operand {i + 1}"
                    )
                shape = expr.shape
                size = expr._array.size
                iters.append(expr._array.flat)
            else:
                iters.append(itertools.repeat(expr))

        out = Array.__new__(Array)
        out._array = np.fromiter(
            (m(li, mi, ri) for li, mi, ri in zip(*iters)),
            dtype=object,
            count=size,
        ).reshape(shape)
        return out

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
        if len(left.shape) != 2:
            raise ValueError("can only matmul(a, b) a 2D array a")
        if len(right.shape) != 2:
            raise ValueError("can only matmul(a, b) a 2D array b")
        if left.shape[1] != right.shape[0]:
            raise ValueError(
                f"can only matmul(a, b) with shapes (n, k) x (k, m) -> (n, m) but got {left.shape} x {right.shape}"
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
                # we can apply a group here since acc is not an array
                assert not isinstance(acc, Array)
                out._array[n, m] = Group(acc)
        return out

    def __repr__(self) -> str:
        return f"{self._array!r}".removeprefix("array(").removesuffix(", dtype=object)")
