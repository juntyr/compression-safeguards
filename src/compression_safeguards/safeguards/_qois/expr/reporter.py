from abc import ABC, abstractmethod
from collections.abc import Mapping

import numpy as np

from ....utils.bindings import Parameter
from ..bound import DataBounds, data_bounds
from .abc import Expr
from .literal import Number
from .typing import F, Ns, Ps, PsI


class Reporter(ABC):
    """
    Abstract base class for reporters that are informed during quantity of
    interest expression computations.
    """

    __slots__ = ()

    @abstractmethod
    def enter(self, expr: Expr) -> None:
        """
        Report that `expr` has been entered.

        Parameters
        ----------
        expr : Expr
            The expression that has been entered.
        """

    @abstractmethod
    def exit(self, expr: Expr) -> None:
        """
        Report that `expr` has been exited.

        Parameters
        ----------
        expr : Expr
            The expression that has been exited.
        """


class ReportingExpr(Expr[Expr]):
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

    __slots__ = ("_expr", "_reporter")
    _expr: Expr
    _reporter: Reporter

    def __new__(cls, expr: Expr, reporter: Reporter):
        if isinstance(expr, (ReportingExpr, Number)):
            return expr
        this = super(ReportingExpr, cls).__new__(cls)
        this._expr = expr
        this._reporter = reporter
        return this

    @property
    def args(self) -> tuple[Expr]:
        return (self._expr,)

    def with_args(self, expr: Expr) -> "ReportingExpr":
        return ReportingExpr(expr, self._reporter)

    @property  # type: ignore
    def expr_size(self) -> int:
        return self._expr.expr_size

    @property  # type: ignore
    def data_expr_size(self) -> int:
        return self._expr.data_expr_size

    @property  # type: ignore
    def has_data(self) -> bool:
        return self._expr.has_data

    def constant_fold(self, dtype: np.dtype[F]) -> F | Expr:
        fexpr = self._expr.constant_fold(dtype)
        # partially / not constant folded -> stop further folding
        if isinstance(fexpr, Expr):
            return ReportingExpr(fexpr, self._reporter)
        # fully constant folded -> allow further folding
        return fexpr

    def eval(
        self,
        x: PsI,
        Xs: np.ndarray[Ns, np.dtype[F]],
        late_bound: Mapping[Parameter, np.ndarray[Ns, np.dtype[F]]],
    ) -> np.ndarray[PsI, np.dtype[F]]:
        return self._expr.eval(x, Xs, late_bound)

    @data_bounds(DataBounds.infallible)
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

    def __repr__(self) -> str:
        return repr(self._expr)
