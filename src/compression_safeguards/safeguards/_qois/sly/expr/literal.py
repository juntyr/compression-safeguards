from collections.abc import Mapping

import numpy as np

from .....utils.bindings import Parameter
from .....utils.cast import (
    _float128_dtype,
    _float128_e,
    _float128_pi,
)
from .....utils.typing import F, S
from .abc import Expr


class Number(Expr):
    __slots__ = ("_n",)
    _n: str

    def __init__(self, n: str):
        self._n = n

    @property
    def has_data(self) -> bool:
        return False

    @property
    def late_bound_constants(self) -> frozenset[Parameter]:
        return frozenset()

    def constant_fold(self, dtype: np.dtype[F]) -> F | Expr:
        # FIXME: what about folding integers?
        return dtype.type(self._n)  # type: ignore

    def eval(
        self,
        X: np.ndarray[tuple[int, ...], np.dtype[F]],
        late_bound: Mapping[Parameter, np.ndarray[tuple[int, ...], np.dtype[F]]],
    ) -> F | np.ndarray[tuple[int, ...], np.dtype[F]]:
        return X.dtype.type(self._n)  # type: ignore

    def compute_data_error_bound_unchecked(
        self,
        eb_expr_lower: np.ndarray[S, np.dtype[F]],
        eb_expr_upper: np.ndarray[S, np.dtype[F]],
        X: np.ndarray[tuple[int, ...], np.dtype[F]],
        late_bound: Mapping[Parameter, np.ndarray[tuple[int, ...], np.dtype[F]]],
    ) -> tuple[np.ndarray[S, np.dtype[F]], np.ndarray[S, np.dtype[F]]]:
        assert False, "number literals have no error bounds"

    def __repr__(self) -> str:
        return self._n


class Pi(Expr):
    __slots__ = ()

    @property
    def has_data(self) -> bool:
        return False

    @property
    def late_bound_constants(self) -> frozenset[Parameter]:
        return frozenset()

    def constant_fold(self, dtype: np.dtype[F]) -> F | Expr:
        if dtype == _float128_dtype:
            return _float128_pi  # type: ignore
        return dtype.type(np.pi)

    def eval(
        self,
        X: np.ndarray[tuple[int, ...], np.dtype[F]],
        late_bound: Mapping[Parameter, np.ndarray[tuple[int, ...], np.dtype[F]]],
    ) -> F | np.ndarray[tuple[int, ...], np.dtype[F]]:
        if X.dtype == _float128_dtype:
            return _float128_pi  # type: ignore
        return X.dtype.type(np.pi)

    def compute_data_error_bound_unchecked(
        self,
        eb_expr_lower: np.ndarray[S, np.dtype[F]],
        eb_expr_upper: np.ndarray[S, np.dtype[F]],
        X: np.ndarray[tuple[int, ...], np.dtype[F]],
        late_bound: Mapping[Parameter, np.ndarray[tuple[int, ...], np.dtype[F]]],
    ) -> tuple[np.ndarray[S, np.dtype[F]], np.ndarray[S, np.dtype[F]]]:
        assert False, "pi has no error bounds"

    def __repr__(self) -> str:
        return "pi"


class Euler(Expr):
    __slots__ = ()

    @property
    def has_data(self) -> bool:
        return False

    @property
    def late_bound_constants(self) -> frozenset[Parameter]:
        return frozenset()

    def constant_fold(self, dtype: np.dtype[F]) -> F | Expr:
        if dtype == _float128_dtype:
            return _float128_e  # type: ignore
        return dtype.type(np.e)

    def eval(
        self,
        X: np.ndarray[tuple[int, ...], np.dtype[F]],
        late_bound: Mapping[Parameter, np.ndarray[tuple[int, ...], np.dtype[F]]],
    ) -> F | np.ndarray[tuple[int, ...], np.dtype[F]]:
        if X.dtype == _float128_dtype:
            return _float128_pi  # type: ignore
        return X.dtype.type(np.e)

    def compute_data_error_bound_unchecked(
        self,
        eb_expr_lower: np.ndarray[S, np.dtype[F]],
        eb_expr_upper: np.ndarray[S, np.dtype[F]],
        X: np.ndarray[tuple[int, ...], np.dtype[F]],
        late_bound: Mapping[Parameter, np.ndarray[tuple[int, ...], np.dtype[F]]],
    ) -> tuple[np.ndarray[S, np.dtype[F]], np.ndarray[S, np.dtype[F]]]:
        assert False, "Euler's e has no error bounds"

    def __repr__(self) -> str:
        return "e"
