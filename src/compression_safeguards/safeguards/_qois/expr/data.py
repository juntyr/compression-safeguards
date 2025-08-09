from collections.abc import Mapping

import numpy as np

from ....utils.bindings import Parameter
from ....utils.cast import _isfinite
from .abc import Expr
from .typing import F, Ns, Ps, PsI


class Data(Expr):
    __slots__ = ("_index",)
    _index: tuple[int, ...]

    def __init__(self, index: tuple[int, ...]):
        self._index = index

    @property
    def has_data(self) -> bool:
        return True

    @property
    def data_indices(self) -> frozenset[tuple[int, ...]]:
        return frozenset([self._index])

    def apply_array_element_offset(
        self,
        axis: int,
        offset: int,
    ) -> Expr:
        index = list(self._index)
        index[axis] += offset
        return Data(index=tuple(index))

    @property
    def late_bound_constants(self) -> frozenset[Parameter]:
        return frozenset()

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

    def compute_data_error_bound_unchecked(
        self,
        eb_expr_lower: np.ndarray[Ps, np.dtype[F]],
        eb_expr_upper: np.ndarray[Ps, np.dtype[F]],
        X: np.ndarray[Ps, np.dtype[F]],
        Xs: np.ndarray[Ns, np.dtype[F]],
        late_bound: Mapping[Parameter, np.ndarray[Ns, np.dtype[F]]],
    ) -> tuple[np.ndarray[Ps, np.dtype[F]], np.ndarray[Ps, np.dtype[F]]]:
        return (eb_expr_lower, eb_expr_upper)

    def compute_data_error_bound(
        self,
        eb_expr_lower: np.ndarray[Ps, np.dtype[F]],
        eb_expr_upper: np.ndarray[Ps, np.dtype[F]],
        X: np.ndarray[Ps, np.dtype[F]],
        Xs: np.ndarray[Ns, np.dtype[F]],
        late_bound: Mapping[Parameter, np.ndarray[Ns, np.dtype[F]]],
    ) -> tuple[np.ndarray[Ps, np.dtype[F]], np.ndarray[Ps, np.dtype[F]]]:
        # data just returns the computed error bounds
        return self.compute_data_error_bound_unchecked(
            eb_expr_lower, eb_expr_upper, X, Xs, late_bound
        )

    def compute_data_bounds_unchecked(
        self,
        expr_lower: np.ndarray[Ps, np.dtype[F]],
        expr_upper: np.ndarray[Ps, np.dtype[F]],
        X: np.ndarray[Ps, np.dtype[F]],
        Xs: np.ndarray[Ns, np.dtype[F]],
        late_bound: Mapping[Parameter, np.ndarray[Ns, np.dtype[F]]],
    ) -> tuple[np.ndarray[Ns, np.dtype[F]], np.ndarray[Ns, np.dtype[F]]]:
        X_lower: np.ndarray[Ns, np.dtype[F]] = np.where(
            _isfinite(Xs), X.dtype.type(np.inf), Xs
        )  # type: ignore
        X_upper: np.ndarray[Ns, np.dtype[F]] = np.where(
            _isfinite(Xs), X.dtype.type(-np.inf), Xs
        )  # type: ignore

        X_lower[(...,) + self._index] = expr_lower
        X_upper[(...,) + self._index] = expr_upper

        return X_lower, X_upper

    def compute_data_bounds(
        self,
        expr_lower: np.ndarray[Ps, np.dtype[F]],
        expr_upper: np.ndarray[Ps, np.dtype[F]],
        X: np.ndarray[Ps, np.dtype[F]],
        Xs: np.ndarray[Ns, np.dtype[F]],
        late_bound: Mapping[Parameter, np.ndarray[Ns, np.dtype[F]]],
    ) -> tuple[np.ndarray[Ns, np.dtype[F]], np.ndarray[Ns, np.dtype[F]]]:
        # data just returns the computed error bounds
        return self.compute_data_bounds_unchecked(
            expr_lower, expr_upper, X, Xs, late_bound
        )

    def __repr__(self) -> str:
        if self._index == ():
            return "x"
        return f"X[{','.join(str(i) for i in self._index)}]"


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
    def has_data(self) -> bool:
        return False

    @property
    def data_indices(self) -> frozenset[tuple[int, ...]]:
        return frozenset()

    def apply_array_element_offset(
        self,
        axis: int,
        offset: int,
    ) -> Expr:
        index = list(self._index)
        index[axis] += offset
        return LateBoundConstant(self._name, index=tuple(index))

    @property
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

    def compute_data_error_bound_unchecked(
        self,
        eb_expr_lower: np.ndarray[Ps, np.dtype[F]],
        eb_expr_upper: np.ndarray[Ps, np.dtype[F]],
        X: np.ndarray[Ps, np.dtype[F]],
        Xs: np.ndarray[Ns, np.dtype[F]],
        late_bound: Mapping[Parameter, np.ndarray[Ns, np.dtype[F]]],
    ) -> tuple[np.ndarray[Ps, np.dtype[F]], np.ndarray[Ps, np.dtype[F]]]:
        assert False, "late-bound constants have no error bounds"

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
