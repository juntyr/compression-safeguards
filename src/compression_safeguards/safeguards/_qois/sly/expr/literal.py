from collections.abc import Mapping

import numpy as np

from .....utils.bindings import Parameter
from .....utils.cast import (
    _float128_dtype,
    _float128_e,
    _float128_pi,
)
from .abc import Expr
from .typing import F, Ns, Ps, PsI


class Number(Expr):
    __slots__ = ("_n",)
    _n: str

    def __init__(self, n: str):
        self._n = n

    @property
    def has_data(self) -> bool:
        return False

    @property
    def data_indices(self) -> frozenset[tuple[int, ...]]:
        return frozenset()

    @property
    def late_bound_constants(self) -> frozenset[Parameter]:
        return frozenset()

    def constant_fold(self, dtype: np.dtype[F]) -> F | Expr:
        # FIXME: what about folding integers?
        return dtype.type(self._n)  # type: ignore

    def eval(
        self,
        x: PsI,
        Xs: np.ndarray[Ns, np.dtype[F]],
        late_bound: Mapping[Parameter, np.ndarray[Ns, np.dtype[F]]],
    ) -> np.ndarray[PsI, np.dtype[F]]:
        return Xs.dtype.type(self._n)  # type: ignore

    def compute_data_error_bound_unchecked(
        self,
        eb_expr_lower: np.ndarray[Ps, np.dtype[F]],
        eb_expr_upper: np.ndarray[Ps, np.dtype[F]],
        X: np.ndarray[Ps, np.dtype[F]],
        Xs: np.ndarray[Ns, np.dtype[F]],
        late_bound: Mapping[Parameter, np.ndarray[Ns, np.dtype[F]]],
    ) -> tuple[np.ndarray[Ps, np.dtype[F]], np.ndarray[Ps, np.dtype[F]]]:
        assert False, "number literals have no error bounds"

    def __repr__(self) -> str:
        return self._n


class Pi(Expr):
    __slots__ = ()

    @property
    def has_data(self) -> bool:
        return False

    @property
    def data_indices(self) -> frozenset[tuple[int, ...]]:
        return frozenset()

    @property
    def late_bound_constants(self) -> frozenset[Parameter]:
        return frozenset()

    def constant_fold(self, dtype: np.dtype[F]) -> F | Expr:
        if dtype == _float128_dtype:
            return _float128_pi  # type: ignore
        return dtype.type(np.pi)

    def eval(
        self,
        x: PsI,
        Xs: np.ndarray[Ns, np.dtype[F]],
        late_bound: Mapping[Parameter, np.ndarray[Ns, np.dtype[F]]],
    ) -> np.ndarray[PsI, np.dtype[F]]:
        if Xs.dtype == _float128_dtype:
            return _float128_pi  # type: ignore
        return Xs.dtype.type(np.pi)  # type: ignore

    def compute_data_error_bound_unchecked(
        self,
        eb_expr_lower: np.ndarray[Ps, np.dtype[F]],
        eb_expr_upper: np.ndarray[Ps, np.dtype[F]],
        X: np.ndarray[Ps, np.dtype[F]],
        Xs: np.ndarray[Ns, np.dtype[F]],
        late_bound: Mapping[Parameter, np.ndarray[Ns, np.dtype[F]]],
    ) -> tuple[np.ndarray[Ps, np.dtype[F]], np.ndarray[Ps, np.dtype[F]]]:
        assert False, "pi has no error bounds"

    def __repr__(self) -> str:
        return "pi"


class Euler(Expr):
    __slots__ = ()

    @property
    def has_data(self) -> bool:
        return False

    @property
    def data_indices(self) -> frozenset[tuple[int, ...]]:
        return frozenset()

    @property
    def late_bound_constants(self) -> frozenset[Parameter]:
        return frozenset()

    def constant_fold(self, dtype: np.dtype[F]) -> F | Expr:
        if dtype == _float128_dtype:
            return _float128_e  # type: ignore
        return dtype.type(np.e)

    def eval(
        self,
        x: PsI,
        Xs: np.ndarray[Ns, np.dtype[F]],
        late_bound: Mapping[Parameter, np.ndarray[Ns, np.dtype[F]]],
    ) -> np.ndarray[PsI, np.dtype[F]]:
        if Xs.dtype == _float128_dtype:
            return _float128_pi  # type: ignore
        return Xs.dtype.type(np.e)  # type: ignore

    def compute_data_error_bound_unchecked(
        self,
        eb_expr_lower: np.ndarray[Ps, np.dtype[F]],
        eb_expr_upper: np.ndarray[Ps, np.dtype[F]],
        X: np.ndarray[Ps, np.dtype[F]],
        Xs: np.ndarray[Ns, np.dtype[F]],
        late_bound: Mapping[Parameter, np.ndarray[Ns, np.dtype[F]]],
    ) -> tuple[np.ndarray[Ps, np.dtype[F]], np.ndarray[Ps, np.dtype[F]]]:
        assert False, "Euler's e has no error bounds"

    def __repr__(self) -> str:
        return "e"
