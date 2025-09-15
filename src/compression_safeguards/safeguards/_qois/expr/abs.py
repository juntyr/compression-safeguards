from collections.abc import Mapping

import numpy as np

from ....utils._compat import _is_sign_negative_number
from ....utils.bindings import Parameter
from ..bound import checked_data_bounds
from .abc import Expr
from .constfold import ScalarFoldedConstant
from .typing import F, Ns, Ps, PsI


class ScalarAbs(Expr[Expr]):
    __slots__ = ("_a",)
    _a: Expr

    def __init__(self, a: Expr):
        self._a = a

    @property
    def args(self) -> tuple[Expr]:
        return (self._a,)

    def with_args(self, a: Expr) -> "ScalarAbs":
        return ScalarAbs(a)

    def constant_fold(self, dtype: np.dtype[F]) -> F | Expr:
        return ScalarFoldedConstant.constant_fold_unary(
            self._a, dtype, np.abs, ScalarAbs
        )

    def eval(
        self,
        x: PsI,
        Xs: np.ndarray[Ns, np.dtype[F]],
        late_bound: Mapping[Parameter, np.ndarray[Ns, np.dtype[F]]],
    ) -> np.ndarray[PsI, np.dtype[F]]:
        return np.abs(self._a.eval(x, Xs, late_bound))

    @checked_data_bounds
    def compute_data_bounds_unchecked(
        self,
        expr_lower: np.ndarray[Ps, np.dtype[F]],
        expr_upper: np.ndarray[Ps, np.dtype[F]],
        X: np.ndarray[Ps, np.dtype[F]],
        Xs: np.ndarray[Ns, np.dtype[F]],
        late_bound: Mapping[Parameter, np.ndarray[Ns, np.dtype[F]]],
    ) -> tuple[np.ndarray[Ns, np.dtype[F]], np.ndarray[Ns, np.dtype[F]]]:
        # evaluate arg
        arg = self._a
        argv = arg.eval(X.shape, Xs, late_bound)

        # flip and swap the expr bounds to get the bounds on arg
        # abs(...) cannot be negative, but
        #  - a > 0 and 0 < el <= eu -> al = el, au = eu
        #  - a < 0 and 0 < el <= eu -> al = -eu, au = -el
        #  - el <= 0 -> al = -eu, au = eu
        # TODO: an interval union could represent that the two sometimes-
        #       disjoint intervals in the future
        arg_lower: np.ndarray[Ps, np.dtype[F]] = np.copy(expr_lower)
        np.negative(
            expr_upper,
            out=arg_lower,
            where=(np.less_equal(expr_lower, 0) | _is_sign_negative_number(argv)),
        )
        arg_upper: np.ndarray[Ps, np.dtype[F]] = np.copy(expr_upper)
        np.negative(
            expr_lower,
            out=arg_upper,
            where=(np.greater(expr_lower, 0) & _is_sign_negative_number(argv)),
        )

        return arg.compute_data_bounds(
            arg_lower,
            arg_upper,
            X,
            Xs,
            late_bound,
        )

    def __repr__(self) -> str:
        return f"abs({self._a!r})"
