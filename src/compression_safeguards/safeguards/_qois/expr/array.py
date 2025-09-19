import itertools
from collections.abc import Callable, Iterator, Mapping

import numpy as np
from typing_extensions import (
    TypeVarTuple,  # MSPV 3.11
    Unpack,  # MSPV 3.11
    override,  # MSPV 3.12
)

from ....utils.bindings import Parameter
from .abc import AnyExpr, Expr
from .addsub import ScalarAdd
from .constfold import ScalarFoldedConstant
from .data import Data
from .divmul import ScalarMultiply
from .group import Group
from .typing import Es, F, Ns, Ps, PsI

# FIXME: actually bound the types to be Expr
# https://discuss.python.org/t/how-to-use-typevartuple/67502
Es2 = TypeVarTuple("Es2")
""" Tuple of [`Expr`][compression_safeguards.safeguards._qois.expr.abc.Expr]s. """


class Array(Expr[AnyExpr, Unpack[Es]]):
    __slots__: tuple[str, ...] = ("_array",)
    _array: np.ndarray

    def __init__(self, el: AnyExpr, *els: AnyExpr) -> None:
        if isinstance(el, Array):
            aels = [el._array]
            for e in els:
                if not (isinstance(e, Array) and e.shape == el.shape):
                    raise ValueError(
                        f"elements must all have the consistent shape {el.shape}"
                    )
                aels.append(e._array)
            self._array = np.array(aels, copy=None)
        else:
            for e in els:
                if isinstance(e, Array):
                    raise ValueError("elements must all be scalar")
            self._array = np.array((el,) + els, copy=None)

    @staticmethod
    def from_data_shape(shape: tuple[int, ...]) -> "Array[Unpack[tuple[AnyExpr, ...]]]":
        out = Array.__new__(Array)
        out._array = np.empty(shape, dtype=object)

        for i in itertools.product(*[range(a) for a in shape]):
            out._array[i] = Data(index=i)

        return out

    @property
    @override
    def args(self) -> tuple[AnyExpr, Unpack[Es]]:
        if self._array.ndim == 1:
            return tuple(self._array)
        return tuple(a for a in self._array)

    @override
    def with_args(self, el: AnyExpr, *els: Unpack[Es]) -> "Array[Unpack[Es]]":
        return Array(el, *els)  # type: ignore

    @override
    def constant_fold(self, dtype: np.dtype[F]) -> F | AnyExpr:
        return Array.map(
            lambda e: ScalarFoldedConstant.constant_fold_expr(e, dtype),  # type: ignore
            self,
        )

    @override
    def eval(
        self,
        x: PsI,
        Xs: np.ndarray[Ns, np.dtype[F]],
        late_bound: Mapping[Parameter, np.ndarray[Ns, np.dtype[F]]],
    ) -> np.ndarray[PsI, np.dtype[F]]:
        assert False, "cannot evaluate an array expression"

    @override
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
    def map(map: Callable[[Unpack[Es2]], AnyExpr], *exprs: Unpack[Es2]) -> AnyExpr:
        if not any(isinstance(e, Array) for e in exprs):
            return map(*exprs)

        shape = None
        size = 0
        iters: list[Iterator[AnyExpr]] = []

        for i, expr in enumerate(exprs):  # type: ignore
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
            (map(*its) for its in zip(*iters)),
            dtype=object,
            count=size,
        ).reshape(shape)
        return out

    def index(self, index: tuple[int | slice, ...]) -> AnyExpr:
        a = self._array[index]
        if isinstance(a, np.ndarray):
            out = Array.__new__(Array)
            out._array = a
            return out
        return a

    def transpose(self) -> "Array[Unpack[tuple[AnyExpr, ...]]]":
        out = Array.__new__(Array)
        out._array = self._array.T
        return out

    def sum(self) -> AnyExpr:
        acc = None
        for e in self._array.flat:
            if acc is None:
                acc = e
            else:
                acc = ScalarAdd(acc, e)
        assert acc is not None
        # we can return a group here since acc is not an array
        assert not isinstance(acc, Array)
        return Group(acc)

    @staticmethod
    def matmul(
        left: "Array[Unpack[tuple[AnyExpr, ...]]]",
        right: "Array[Unpack[tuple[AnyExpr, ...]]]",
    ) -> "Array[Unpack[tuple[AnyExpr, ...]]]":
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
                acc: None | AnyExpr = None
                for k in range(left.shape[1]):
                    kk = ScalarMultiply(left._array[n, k], right._array[k, m])
                    if acc is None:
                        acc = kk
                    else:
                        acc = ScalarAdd(acc, kk)
                assert acc is not None
                # we can apply a group here since acc is not an array
                assert not isinstance(acc, Array)
                out._array[n, m] = Group(acc)
        return out

    @override
    def __repr__(self) -> str:
        return f"{self._array!r}".removeprefix("array(").removesuffix(", dtype=object)")
