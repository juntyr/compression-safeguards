from collections.abc import Mapping

import numpy as np

from ....utils._compat import (
    _is_negative,
    _is_positive,
    _maximum_zero_sign_sensitive,
    _minimum_zero_sign_sensitive,
)
from ....utils.bindings import Parameter
from ..bound import checked_data_bounds, guarantee_arg_within_expr_bounds
from .abc import Expr
from .constfold import ScalarFoldedConstant
from .typing import F, Ns, Ps, PsI


class ScalarReciprocal(Expr[Expr]):
    __slots__ = ("_a",)
    _a: Expr

    def __init__(self, a: Expr):
        self._a = a

    @property
    def args(self) -> tuple[Expr]:
        return (self._a,)

    def with_args(self, a: Expr) -> "ScalarReciprocal":
        return ScalarReciprocal(a)

    def constant_fold(self, dtype: np.dtype[F]) -> F | Expr:
        return ScalarFoldedConstant.constant_fold_unary(
            self._a,
            dtype,
            np.reciprocal,
            ScalarReciprocal,
        )

    def eval(
        self,
        x: PsI,
        Xs: np.ndarray[Ns, np.dtype[F]],
        late_bound: Mapping[Parameter, np.ndarray[Ns, np.dtype[F]]],
    ) -> np.ndarray[PsI, np.dtype[F]]:
        return np.reciprocal(self._a.eval(x, Xs, late_bound))

    @checked_data_bounds
    def compute_data_bounds_unchecked(
        self,
        expr_lower: np.ndarray[Ps, np.dtype[F]],
        expr_upper: np.ndarray[Ps, np.dtype[F]],
        X: np.ndarray[Ps, np.dtype[F]],
        Xs: np.ndarray[Ns, np.dtype[F]],
        late_bound: Mapping[Parameter, np.ndarray[Ns, np.dtype[F]]],
    ) -> tuple[np.ndarray[Ns, np.dtype[F]], np.ndarray[Ns, np.dtype[F]]]:
        # evaluate arg and reciprocal(arg)
        arg = self._a
        argv = arg.eval(X.shape, Xs, late_bound)
        exprv = np.reciprocal(argv)

        # compute the argument bounds
        # ensure that reciprocal(...) keeps the same sign as arg
        # TODO: an interval union could represent that the two disjoint
        #       intervals in the future
        arg_lower: np.ndarray[Ps, np.dtype[F]] = _minimum_zero_sign_sensitive(
            expr_upper, X.dtype.type(-0.0)
        )
        np.copyto(arg_lower, expr_upper, where=_is_positive(exprv), casting="no")
        arg_lower = np.reciprocal(arg_lower)
        arg_lower = _minimum_zero_sign_sensitive(argv, arg_lower)

        arg_upper: np.ndarray[Ps, np.dtype[F]] = _maximum_zero_sign_sensitive(
            X.dtype.type(+0.0), expr_lower
        )
        np.copyto(arg_upper, expr_lower, where=_is_negative(exprv), casting="no")
        arg_upper = np.reciprocal(arg_upper)
        arg_upper = _maximum_zero_sign_sensitive(argv, arg_upper)

        # we need to force argv if expr_lower == expr_upper
        np.copyto(arg_lower, argv, where=(expr_lower == expr_upper), casting="no")
        np.copyto(arg_upper, argv, where=(expr_lower == expr_upper), casting="no")

        # handle rounding errors in reciprocal(reciprocal(...)) early
        arg_lower = guarantee_arg_within_expr_bounds(
            lambda arg_lower: np.reciprocal(arg_lower),
            exprv,
            argv,
            arg_lower,
            expr_lower,
            expr_upper,
        )
        arg_upper = guarantee_arg_within_expr_bounds(
            lambda arg_upper: np.reciprocal(arg_upper),
            exprv,
            argv,
            arg_upper,
            expr_lower,
            expr_upper,
        )

        return arg.compute_data_bounds(
            arg_lower,
            arg_upper,
            X,
            Xs,
            late_bound,
        )

    def __repr__(self) -> str:
        return f"reciprocal({self._a!r})"
