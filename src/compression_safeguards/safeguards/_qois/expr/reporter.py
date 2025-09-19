from abc import ABC, abstractmethod
from collections.abc import Mapping

import numpy as np
from typing_extensions import override  # MSPV 3.12

from ....utils.bindings import Parameter
from ..bound import DataBounds, data_bounds
from .abc import AnyExpr, Expr
from .literal import Number
from .typing import F, Ns, Ps, PsI


class Reporter(ABC):
    """
    Abstract base class for reporters that are informed during quantity of
    interest expression computations.
    """

    __slots__: tuple[str, ...] = ()

    @abstractmethod
    def enter(self, expr: AnyExpr) -> None:
        """
        Report that `expr` has been entered.

        Parameters
        ----------
        expr : Expr
            The expression that has been entered.
        """

    @abstractmethod
    def exit(self, expr: AnyExpr) -> None:
        """
        Report that `expr` has been exited.

        Parameters
        ----------
        expr : Expr
            The expression that has been exited.
        """


class ReportingExpr(Expr[AnyExpr]):
    """
    A reporting expression wraps around an existing `expr`, forwarding all
    functionality to it while also informing the `reporter`.

    Parameters
    ----------
    expr : Expr
        The expression that will be wrapped and reported on.
    reporter : Reporter
        The reporter that will report on the expression.
    """

    __slots__: tuple[str, ...] = ("_expr", "_reporter")
    _expr: AnyExpr
    _reporter: Reporter

    def __init__(self, expr: AnyExpr, reporter: Reporter):
        self._expr = expr
        self._reporter = reporter

    def __new__(cls, expr: AnyExpr, reporter: Reporter) -> "ReportingExpr | Number":  # type: ignore[misc]
        if isinstance(expr, ReportingExpr | Number):
            return expr
        return super().__new__(cls)

    @property
    @override
    def args(self) -> tuple[AnyExpr]:
        return (self._expr,)

    @override
    def with_args(self, expr: AnyExpr) -> "ReportingExpr | Number":
        return ReportingExpr(expr, self._reporter)

    @property  # type: ignore[misc]
    @override
    def expr_size(self) -> int:
        return self._expr.expr_size

    @property  # type: ignore[misc]
    @override
    def data_expr_size(self) -> int:
        return self._expr.data_expr_size

    @property  # type: ignore[misc]
    @override
    def has_data(self) -> bool:
        return self._expr.has_data

    @override
    def constant_fold(self, dtype: np.dtype[F]) -> F | AnyExpr:
        fexpr = self._expr.constant_fold(dtype)
        # partially / not constant folded -> stop further folding
        if isinstance(fexpr, Expr):
            return ReportingExpr(fexpr, self._reporter)
        # fully constant folded -> allow further folding
        return fexpr

    @override
    def eval(
        self,
        x: PsI,
        Xs: np.ndarray[Ns, np.dtype[F]],
        late_bound: Mapping[Parameter, np.ndarray[Ns, np.dtype[F]]],
    ) -> np.ndarray[PsI, np.dtype[F]]:
        return self._expr.eval(x, Xs, late_bound)

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
        self._reporter.enter(self._expr)
        try:
            return self._expr.compute_data_bounds(
                expr_lower, expr_upper, X, Xs, late_bound
            )
        finally:
            self._reporter.exit(self._expr)

    @override
    def __repr__(self) -> str:
        return repr(self._expr)
