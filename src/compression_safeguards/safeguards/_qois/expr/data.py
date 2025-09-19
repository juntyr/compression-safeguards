from collections.abc import Mapping
from warnings import warn

import numpy as np
from typing_extensions import override  # MSPV 3.12

from ....utils._compat import _is_of_shape
from ....utils.bindings import Parameter
from ..bound import DataBounds, data_bounds
from .abc import AnyExpr, Expr
from .typing import F, Ns, Ps, PsI


class Data(Expr[()]):
    __slots__: tuple[str, ...] = ("_index",)
    _index: tuple[int, ...]

    SCALAR: "Data"

    def __init__(self, index: tuple[int, ...]) -> None:
        self._index = index

    @property
    def index(self) -> tuple[int, ...]:
        return self._index

    @property
    @override
    def args(self) -> tuple[()]:
        return ()

    @override
    def with_args(self) -> "Data":
        return Data(self._index)

    @property  # type: ignore
    @override
    def has_data(self) -> bool:
        return True

    @property  # type: ignore
    @override
    def data_indices(self) -> frozenset[tuple[int, ...]]:
        return frozenset([self._index])

    @override  # type: ignore
    def apply_array_element_offset(
        self,
        axis: int,
        offset: int,
    ) -> "Data":
        index = list(self._index)
        index[axis] += offset
        return Data(index=tuple(index))

    @override
    def constant_fold(self, dtype: np.dtype[F]) -> F | AnyExpr:
        return self

    @override
    def eval(
        self,
        x: PsI,
        Xs: np.ndarray[Ns, np.dtype[F]],
        late_bound: Mapping[Parameter, np.ndarray[Ns, np.dtype[F]]],
    ) -> np.ndarray[PsI, np.dtype[F]]:
        out: np.ndarray[tuple[int, ...], np.dtype[F]] = Xs[(...,) + self._index]
        assert _is_of_shape(out, x)
        return out

    @data_bounds(DataBounds.infallible)
    @override
    def compute_data_bounds_unchecked(
        self,
        expr_lower: np.ndarray[Ps, np.dtype[F]],
        expr_upper: np.ndarray[Ps, np.dtype[F]],
        X: np.ndarray[Ps, np.dtype[F]],
        Xs: np.ndarray[Ns, np.dtype[F]],
        late_bound: Mapping[Parameter, np.ndarray[Ns, np.dtype[F]]],
    ) -> tuple[np.ndarray[Ns, np.dtype[F]], np.ndarray[Ns, np.dtype[F]]]:
        exprv = Xs[(...,) + self._index]

        if not np.all((expr_lower <= exprv) | np.isnan(exprv)):
            warn("data lower bounds are above the data values")
        if not np.all((expr_upper >= exprv) | np.isnan(exprv)):
            warn("data upper bounds are below the data values")

        Xs_lower: np.ndarray[Ns, np.dtype[F]] = np.full(Xs.shape, X.dtype.type(-np.inf))
        Xs_lower[np.isnan(Xs)] = np.nan
        Xs_lower[(...,) + self._index] = expr_lower

        Xs_upper: np.ndarray[Ns, np.dtype[F]] = np.full(Xs.shape, X.dtype.type(np.inf))
        Xs_upper[np.isnan(Xs)] = np.nan
        Xs_upper[(...,) + self._index] = expr_upper

        return Xs_lower, Xs_upper

    @override
    def __repr__(self) -> str:
        if self._index == ():
            return "x"
        return f"X[{','.join(str(i) for i in self._index)}]"


Data.SCALAR = Data(index=())


class LateBoundConstant(Expr[()]):
    __slots__: tuple[str, ...] = ("_name", "_index")
    _name: Parameter
    _index: tuple[int, ...]

    def __init__(self, name: Parameter, index: tuple[int, ...]) -> None:
        self._name = name
        self._index = index

    @staticmethod
    def like(name: Parameter, data: Data) -> "LateBoundConstant":
        return LateBoundConstant(name, data.index)

    @property
    def name(self) -> Parameter:
        return self._name

    @property
    def index(self) -> tuple[int, ...]:
        return self._index

    @property
    @override
    def args(self) -> tuple[()]:
        return ()

    @override
    def with_args(self) -> "LateBoundConstant":  # type: ignore
        return LateBoundConstant(self._name, self._index)

    @override  # type: ignore
    def apply_array_element_offset(
        self,
        axis: int,
        offset: int,
    ) -> "LateBoundConstant":
        index = list(self._index)
        index[axis] += offset
        return LateBoundConstant(self._name, index=tuple(index))

    @property  # type: ignore
    @override
    def late_bound_constants(self) -> frozenset[Parameter]:
        return frozenset([self.name])

    @override
    def constant_fold(self, dtype: np.dtype[F]) -> F | AnyExpr:
        return self

    @override
    def eval(
        self,
        x: PsI,
        Xs: np.ndarray[Ns, np.dtype[F]],
        late_bound: Mapping[Parameter, np.ndarray[Ns, np.dtype[F]]],
    ) -> np.ndarray[PsI, np.dtype[F]]:
        out: np.ndarray[tuple[int, ...], np.dtype[F]] = late_bound[self.name][
            (...,) + self._index
        ]
        assert _is_of_shape(out, x)
        return out

    @override
    def compute_data_bounds_unchecked(
        self,
        expr_lower: np.ndarray[Ps, np.dtype[F]],
        expr_upper: np.ndarray[Ps, np.dtype[F]],
        X: np.ndarray[Ps, np.dtype[F]],
        Xs: np.ndarray[Ns, np.dtype[F]],
        late_bound: Mapping[Parameter, np.ndarray[Ns, np.dtype[F]]],
    ) -> tuple[np.ndarray[Ns, np.dtype[F]], np.ndarray[Ns, np.dtype[F]]]:
        assert False, "late-bound constants have no data bounds"

    @override
    def __repr__(self) -> str:
        if self._index == ():
            return f'c["{self._name}"]'
        return f'C["{self._name}"][{",".join(str(i) for i in self._index)}]'
