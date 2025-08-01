from abc import abstractmethod
from collections.abc import Mapping

import numpy as np

from ....utils.bindings import Parameter
from ..eb import ensure_bounded_derived_error
from .typing import F, Ns, Ps, PsI


class Expr:
    __slots__ = ()

    @property
    @abstractmethod
    def has_data(self) -> bool:
        pass

    @property
    @abstractmethod
    def data_indices(self) -> frozenset[tuple[int, ...]]:
        pass

    @abstractmethod
    def apply_array_element_offset(
        self,
        axis: int,
        offset: int,
    ) -> "Expr":
        pass

    @property
    @abstractmethod
    def late_bound_constants(self) -> frozenset[Parameter]:
        pass

    @abstractmethod
    def constant_fold(self, dtype: np.dtype[F]) -> F | "Expr":
        pass

    @abstractmethod
    def eval(
        self,
        x: PsI,
        Xs: np.ndarray[Ns, np.dtype[F]],
        late_bound: Mapping[Parameter, np.ndarray[Ns, np.dtype[F]]],
    ) -> np.ndarray[PsI, np.dtype[F]]:
        pass

    @abstractmethod
    def compute_data_error_bound_unchecked(
        self,
        eb_expr_lower: np.ndarray[Ps, np.dtype[F]],
        eb_expr_upper: np.ndarray[Ps, np.dtype[F]],
        X: np.ndarray[Ps, np.dtype[F]],
        Xs: np.ndarray[Ns, np.dtype[F]],
        late_bound: Mapping[Parameter, np.ndarray[Ns, np.dtype[F]]],
    ) -> tuple[np.ndarray[Ps, np.dtype[F]], np.ndarray[Ps, np.dtype[F]]]:
        pass

    def compute_data_error_bound(
        self,
        eb_expr_lower: np.ndarray[Ps, np.dtype[F]],
        eb_expr_upper: np.ndarray[Ps, np.dtype[F]],
        X: np.ndarray[Ps, np.dtype[F]],
        Xs: np.ndarray[Ns, np.dtype[F]],
        late_bound: Mapping[Parameter, np.ndarray[Ns, np.dtype[F]]],
    ) -> tuple[np.ndarray[Ps, np.dtype[F]], np.ndarray[Ps, np.dtype[F]]]:
        tl: np.ndarray[Ps, np.dtype[F]]
        tu: np.ndarray[Ps, np.dtype[F]]
        tl, tu = self.compute_data_error_bound_unchecked(
            eb_expr_lower, eb_expr_upper, X, Xs, late_bound
        )

        exprv = self.eval(X.shape, Xs, late_bound)

        # handle rounding errors in the lower error bound computation
        tl = ensure_bounded_derived_error(
            lambda tl: np.where(  # type: ignore
                tl == 0,
                exprv,
                self.eval(
                    X.shape,
                    Xs + tl.reshape(list(X.shape) + [1] * (Xs.ndim - X.ndim)),
                    late_bound,
                ),
            ),
            exprv,
            X,
            tl,
            eb_expr_lower,
            eb_expr_upper,
        )
        tu = ensure_bounded_derived_error(
            lambda tu: np.where(  # type: ignore
                tu == 0,
                exprv,
                self.eval(
                    X.shape,
                    Xs + tu.reshape(list(X.shape) + [1] * (Xs.ndim - X.ndim)),
                    late_bound,
                ),
            ),
            exprv,
            X,
            tu,
            eb_expr_lower,
            eb_expr_upper,
        )

        return tl, tu
