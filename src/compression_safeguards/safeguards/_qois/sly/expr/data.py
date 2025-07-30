from collections.abc import Mapping

import numpy as np

from .....utils.bindings import Parameter
from .....utils.typing import F, S
from .abc import Expr


class Data(Expr):
    __slots__ = ("_index",)
    _index: tuple[int, ...]

    def __init__(self, index: tuple[int, ...]):
        self._index = index

    @property
    def has_data(self) -> bool:
        return True

    @property
    def late_bound_constants(self) -> frozenset[Parameter]:
        return frozenset()

    def constant_fold(self, dtype: np.dtype[F]) -> F | Expr:
        return self

    def eval(
        self,
        X: np.ndarray[tuple[int, ...], np.dtype[F]],
        late_bound: Mapping[Parameter, np.ndarray[tuple[int, ...], np.dtype[F]]],
    ) -> F | np.ndarray[tuple[int, ...], np.dtype[F]]:
        return X[(...,) + self._index]

    def compute_data_error_bound_unchecked(
        self,
        eb_expr_lower: np.ndarray[S, np.dtype[F]],
        eb_expr_upper: np.ndarray[S, np.dtype[F]],
        X: np.ndarray[tuple[int, ...], np.dtype[F]],
        late_bound: Mapping[Parameter, np.ndarray[tuple[int, ...], np.dtype[F]]],
    ) -> tuple[np.ndarray[S, np.dtype[F]], np.ndarray[S, np.dtype[F]]]:
        return (eb_expr_lower, eb_expr_upper)

    def compute_data_error_bound(
        self,
        eb_expr_lower: np.ndarray[S, np.dtype[F]],
        eb_expr_upper: np.ndarray[S, np.dtype[F]],
        X: np.ndarray[tuple[int, ...], np.dtype[F]],
        late_bound: Mapping[Parameter, np.ndarray[tuple[int, ...], np.dtype[F]]],
    ) -> tuple[np.ndarray[S, np.dtype[F]], np.ndarray[S, np.dtype[F]]]:
        # data just returns the computed error bounds
        return self.compute_data_error_bound_unchecked(
            eb_expr_lower, eb_expr_upper, X, late_bound
        )

    def __repr__(self) -> str:
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
    def late_bound_constants(self) -> frozenset[Parameter]:
        return frozenset([self.name])

    def constant_fold(self, dtype: np.dtype[F]) -> F | Expr:
        return self

    def eval(
        self,
        X: np.ndarray[tuple[int, ...], np.dtype[F]],
        late_bound: Mapping[Parameter, np.ndarray[tuple[int, ...], np.dtype[F]]],
    ) -> F | np.ndarray[tuple[int, ...], np.dtype[F]]:
        return late_bound[self.name][(...,) + self._index]

    def compute_data_error_bound_unchecked(
        self,
        eb_expr_lower: np.ndarray[S, np.dtype[F]],
        eb_expr_upper: np.ndarray[S, np.dtype[F]],
        X: np.ndarray[tuple[int, ...], np.dtype[F]],
        late_bound: Mapping[Parameter, np.ndarray[tuple[int, ...], np.dtype[F]]],
    ) -> tuple[np.ndarray[S, np.dtype[F]], np.ndarray[S, np.dtype[F]]]:
        assert False, "late-bound constants have no error bounds"

    def __repr__(self) -> str:
        return f'C["{self._name}"][{",".join(str(i) for i in self._index)}]'
