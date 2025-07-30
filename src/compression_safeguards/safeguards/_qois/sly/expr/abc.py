from abc import abstractmethod
from collections.abc import Mapping

import numpy as np

from .....utils.bindings import Parameter
from .....utils.typing import F, S
from ...eb import ensure_bounded_derived_error


class Expr:
    __slots__ = ()

    @property
    @abstractmethod
    def has_data(self) -> bool:
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
        X: np.ndarray[tuple[int, ...], np.dtype[F]],
        late_bound: Mapping[Parameter, np.ndarray[tuple[int, ...], np.dtype[F]]],
    ) -> F | np.ndarray[tuple[int, ...], np.dtype[F]]:
        pass

    @abstractmethod
    def compute_data_error_bound_unchecked(
        self,
        eb_expr_lower: np.ndarray[S, np.dtype[F]],
        eb_expr_upper: np.ndarray[S, np.dtype[F]],
        X: np.ndarray[tuple[int, ...], np.dtype[F]],
        late_bound: Mapping[Parameter, np.ndarray[tuple[int, ...], np.dtype[F]]],
    ) -> tuple[np.ndarray[S, np.dtype[F]], np.ndarray[S, np.dtype[F]]]:
        pass

    def compute_data_error_bound(
        self,
        eb_expr_lower: np.ndarray[S, np.dtype[F]],
        eb_expr_upper: np.ndarray[S, np.dtype[F]],
        X: np.ndarray[tuple[int, ...], np.dtype[F]],
        late_bound: Mapping[Parameter, np.ndarray[tuple[int, ...], np.dtype[F]]],
    ) -> tuple[np.ndarray[S, np.dtype[F]], np.ndarray[S, np.dtype[F]]]:
        tl, tu = self.compute_data_error_bound_unchecked(
            eb_expr_lower, eb_expr_upper, X, late_bound
        )

        exprv_: F | np.ndarray[tuple[int, ...], np.dtype[F]] = self.eval(X, late_bound)
        assert isinstance(exprv_, np.ndarray)
        assert exprv_.shape == X.shape
        exprv: np.ndarray[S, np.dtype[F]] = exprv_  # type: ignore

        # handle rounding errors in the lower error bound computation
        tl = ensure_bounded_derived_error(
            lambda tl: np.where(  # type: ignore
                tl == 0,
                exprv,
                self.eval(X + tl, late_bound),
            ),
            exprv,
            X,  # type: ignore
            tl,
            eb_expr_lower,
            eb_expr_upper,
        )
        tu = ensure_bounded_derived_error(
            lambda tu: np.where(  # type: ignore
                tu == 0,
                exprv,
                self.eval(X + tu, late_bound),
            ),
            exprv,
            X,  # type: ignore
            tu,
            eb_expr_lower,
            eb_expr_upper,
        )

        return tl, tu
