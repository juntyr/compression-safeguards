from collections.abc import Mapping

import numpy as np

from ....utils.bindings import Parameter
from ..bound import DataBounds, data_bounds
from .abc import Expr
from .typing import F, Ns, Ps, PsI


class Data(Expr):
    __slots__ = ("_index",)
    _index: tuple[int, ...]

    SCALAR: "Data"

    def __init__(self, index: tuple[int, ...]):
        self._index = index

    @property
    def args(self) -> tuple[()]:
        return ()

    def with_args(self) -> "Data":
        return Data(self._index)

    @property  # type: ignore
    def has_data(self) -> bool:
        return True

    @property  # type: ignore
    def data_indices(self) -> frozenset[tuple[int, ...]]:
        return frozenset([self._index])

    def apply_array_element_offset(  # type: ignore
        self,
        axis: int,
        offset: int,
    ) -> "Data":
        index = list(self._index)
        index[axis] += offset
        return Data(index=tuple(index))

    def constant_fold(self, dtype: np.dtype[F]) -> F | Expr:
        return self

    def eval(
        self,
        x: PsI,
        Xs: np.ndarray[Ns, np.dtype[F]],
        late_bound: Mapping[Parameter, np.ndarray[Ns, np.dtype[F]]],
    ) -> np.ndarray[PsI, np.dtype[F]]:
        out: np.ndarray[tuple[int, ...], np.dtype[F]] = Xs[(...,) + self._index]
        assert out.shape == x
        return out  # type: ignore

    @data_bounds(DataBounds.infallible)
    def compute_data_bounds_unchecked(
        self,
        expr_lower: np.ndarray[Ps, np.dtype[F]],
        expr_upper: np.ndarray[Ps, np.dtype[F]],
        X: np.ndarray[Ps, np.dtype[F]],
        Xs: np.ndarray[Ns, np.dtype[F]],
        late_bound: Mapping[Parameter, np.ndarray[Ns, np.dtype[F]]],
    ) -> tuple[np.ndarray[Ns, np.dtype[F]], np.ndarray[Ns, np.dtype[F]]]:
        X_lower: np.ndarray[Ns, np.dtype[F]] = np.copy(Xs)
        X_lower[np.isfinite(Xs)] = -np.inf
        X_lower[(...,) + self._index] = expr_lower

        X_upper: np.ndarray[Ns, np.dtype[F]] = np.copy(Xs)
        X_upper[np.isfinite(Xs)] = np.inf
        X_upper[(...,) + self._index] = expr_upper

        return X_lower, X_upper

    def __repr__(self) -> str:
        if self._index == ():
            return "x"
        return f"X[{','.join(str(i) for i in self._index)}]"


Data.SCALAR = Data(index=())


class LateBoundConstant(Expr):
    __slots__ = ("_name", "_index")
    _name: Parameter
    _index: tuple[int, ...]

    def __init__(self, name: Parameter, index: tuple[int, ...]):
        self._name = name
        self._index = index

    @staticmethod
    def like(name: Parameter, data: Data) -> "LateBoundConstant":
        return LateBoundConstant(name, data._index)

    @property
    def name(self) -> Parameter:
        return self._name

    @property
    def args(self) -> tuple[()]:
        return ()

    def with_args(self) -> "LateBoundConstant":  # type: ignore
        return LateBoundConstant(self._name, self._index)

    def apply_array_element_offset(  # type: ignore
        self,
        axis: int,
        offset: int,
    ) -> "LateBoundConstant":
        index = list(self._index)
        index[axis] += offset
        return LateBoundConstant(self._name, index=tuple(index))

    @property  # type: ignore
    def late_bound_constants(self) -> frozenset[Parameter]:
        return frozenset([self.name])

    def constant_fold(self, dtype: np.dtype[F]) -> F | Expr:
        return self

    def eval(
        self,
        x: PsI,
        Xs: np.ndarray[Ns, np.dtype[F]],
        late_bound: Mapping[Parameter, np.ndarray[Ns, np.dtype[F]]],
    ) -> np.ndarray[PsI, np.dtype[F]]:
        out: np.ndarray[tuple[int, ...], np.dtype[F]] = late_bound[self.name][
            (...,) + self._index
        ]
        assert out.shape == x
        return out  # type: ignore

    def compute_data_bounds_unchecked(
        self,
        expr_lower: np.ndarray[Ps, np.dtype[F]],
        expr_upper: np.ndarray[Ps, np.dtype[F]],
        X: np.ndarray[Ps, np.dtype[F]],
        Xs: np.ndarray[Ns, np.dtype[F]],
        late_bound: Mapping[Parameter, np.ndarray[Ns, np.dtype[F]]],
    ) -> tuple[np.ndarray[Ns, np.dtype[F]], np.ndarray[Ns, np.dtype[F]]]:
        assert False, "late-bound constants have no data bounds"

    def __repr__(self) -> str:
        if self._index == ():
            return f'c["{self._name}"]'
        return f'C["{self._name}"][{",".join(str(i) for i in self._index)}]'
