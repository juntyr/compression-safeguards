import itertools
from collections.abc import Callable, Iterator, Mapping
from typing import Self

import numpy as np
from typing_extensions import override  # MSPV 3.12

from ....utils.bindings import Parameter
from ....utils.error import ctx
from ..typing import Es, F, Ns, Ps, np_sndarray
from .abc import AnyExpr, Expr
from .addsub import ScalarLeftAssociativeSum
from .constfold import ScalarFoldedConstant
from .data import Data
from .divmul import ScalarMultiply
from .group import Group


class Array(Expr[AnyExpr, *tuple[AnyExpr, ...]]):
    __slots__: tuple[str, ...] = ("_array",)
    _array: np.ndarray

    def __init__(self, el: AnyExpr, *els: AnyExpr) -> None:
        if isinstance(el, Array):
            aels = [el._array]
            for e in els:
                if not (isinstance(e, Array) and e.shape == el.shape):
                    raise (
                        ValueError(
                            "elements must all have the consistent shape "
                            + f"{el.shape}"
                        )
                        | ctx
                    )
                aels.append(e._array)
            self._array = np.array(aels, copy=None)
        else:
            for e in els:
                if isinstance(e, Array):
                    raise ValueError("elements must all be scalar") | ctx
            self._array = np.array((el, *els), copy=None)

    @staticmethod
    def from_data_shape(shape: tuple[int, ...]) -> "Array":
        out = Array.__new__(Array)
        out._array = np.empty(shape, dtype=object)

        for i in itertools.product(*[range(a) for a in shape]):
            out._array[i] = Data(index=i)

        return out

    @property
    @override
    def args(self) -> tuple[AnyExpr, *tuple[AnyExpr, ...]]:
        if self._array.ndim == 1:
            return tuple(self._array)
        return tuple(Array(*a) for a in self._array)  # type: ignore

    @override
    def with_args(self, el: AnyExpr, *els: AnyExpr) -> Self:
        return type(self)(el, *els)

    @override
    def constant_fold(self, dtype: np.dtype[F]) -> F | AnyExpr:
        return Array.map(
            lambda e: ScalarFoldedConstant.constant_fold_expr(e, dtype),
            self,
        )

    @override
    def eval(
        self,
        Xs: np_sndarray[Ps, Ns, np.dtype[F]],
        late_bound: Mapping[Parameter, np_sndarray[Ps, Ns, np.dtype[F]]],
    ) -> np.ndarray[tuple[Ps], np.dtype[F]]:
        assert False, "cannot evaluate an array expression"

    @override
    def compute_data_bounds_unchecked(
        self,
        expr_lower: np.ndarray[tuple[Ps], np.dtype[F]],
        expr_upper: np.ndarray[tuple[Ps], np.dtype[F]],
        Xs: np_sndarray[Ps, Ns, np.dtype[F]],
        late_bound: Mapping[Parameter, np_sndarray[Ps, Ns, np.dtype[F]]],
    ) -> tuple[
        np_sndarray[Ps, Ns, np.dtype[F]],
        np_sndarray[Ps, Ns, np.dtype[F]],
    ]:
        assert False, "cannot derive data bounds over an array expression"

    @property
    def shape(self) -> tuple[int, ...]:
        return self._array.shape

    @property
    def size(self) -> int:
        return self._array.size

    @staticmethod
    def map(map: Callable[[*Es], AnyExpr], *exprs: *Es) -> AnyExpr:
        if not any(isinstance(e, Array) for e in exprs):
            return map(*exprs)

        shape = None
        size = 0
        iters: list[Iterator[AnyExpr]] = []

        for i, expr in enumerate(exprs):  # type: ignore
            if isinstance(expr, Array):
                if shape is not None and expr.shape != shape:
                    raise (
                        ValueError(
                            f"shape mismatch between operands, expected {shape}"
                            + f" but found {expr.shape} for operand {i + 1}"
                        )
                        | ctx
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

    def transpose(self) -> Self:
        out = Array.__new__(type(self))
        out._array = self._array.T
        return out

    def sum(self) -> AnyExpr:
        sum_: AnyExpr = ScalarLeftAssociativeSum(*list(self._array.flat))
        # we can return a group here since sum_ is not an array
        assert not isinstance(sum_, Array)
        return Group(sum_)

    def flatlist(self) -> list[AnyExpr]:
        return list(self._array.flat)

    @staticmethod
    def matmul(
        left: "Array",
        right: "Array",
    ) -> "Array":
        if len(left.shape) != 2:
            raise ValueError("can only matmul(a, b) a 2D array a") | ctx
        if len(right.shape) != 2:
            raise ValueError("can only matmul(a, b) a 2D array b") | ctx
        if left.shape[1] != right.shape[0]:
            raise (
                ValueError(
                    "can only matmul(a, b) with shapes (n, k) x (k, m) -> "
                    + f"(n, m) but got {left.shape} x {right.shape}"
                )
                | ctx
            )
        out = Array.__new__(Array)
        out._array = np.empty((left.shape[0], right.shape[1]), dtype=object)
        for n in range(left.shape[0]):
            for m in range(right.shape[1]):
                sum_: AnyExpr = ScalarLeftAssociativeSum(
                    *[
                        ScalarMultiply(left._array[n, k], right._array[k, m])
                        for k in range(left.shape[1])
                    ]
                )
                # we can apply a group here since sum_ is not an array
                assert not isinstance(sum_, Array)
                out._array[n, m] = Group(sum_)
        return out

    @override
    def __repr__(self) -> str:
        return f"{self._array!r}".removeprefix("array(").removesuffix(", dtype=object)")
