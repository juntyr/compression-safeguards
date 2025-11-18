from collections.abc import Mapping

import numpy as np
from typing_extensions import override  # MSPV 3.12

from ....utils._compat import _ensure_array, _is_sign_negative_number
from ....utils.bindings import Parameter
from ..bound import checked_data_bounds
from ..typing import F, Ns, Ps, np_sndarray
from .abc import AnyExpr, Expr
from .constfold import ScalarFoldedConstant


class ScalarAbs(Expr[AnyExpr]):
    __slots__: tuple[str, ...] = ("_a",)
    _a: AnyExpr

    def __init__(self, a: AnyExpr) -> None:
        self._a = a

    @property
    @override
    def args(self) -> tuple[AnyExpr]:
        return (self._a,)

    @override
    def with_args(self, a: AnyExpr) -> "ScalarAbs":
        return ScalarAbs(a)

    @override
    def constant_fold(self, dtype: np.dtype[F]) -> F | AnyExpr:
        return ScalarFoldedConstant.constant_fold_unary(
            self._a, dtype, np.abs, ScalarAbs
        )

    @override
    def eval(
        self,
        Xs: np_sndarray[Ps, Ns, np.dtype[F]],
        late_bound: Mapping[Parameter, np_sndarray[Ps, Ns, np.dtype[F]]],
    ) -> np.ndarray[tuple[Ps], np.dtype[F]]:
        return np.abs(self._a.eval(Xs, late_bound))

    @checked_data_bounds
    @override
    def compute_data_bounds_unchecked(
        self,
        expr_lower: np.ndarray[tuple[Ps], np.dtype[F]],
        expr_upper: np.ndarray[tuple[Ps], np.dtype[F]],
        Xs: np_sndarray[Ps, Ns, np.dtype[F]],
        late_bound: Mapping[Parameter, np_sndarray[Ps, Ns, np.dtype[F]]],
    ) -> tuple[
        np_sndarray[Ps, Ns, np.dtype[F]],
        np_sndarray[Ps, Ns, np.dtype[F]],
    ]:
        # evaluate arg
        arg = self._a
        argv = arg.eval(Xs, late_bound)

        # flip and swap the expr bounds to get the bounds on arg
        # abs(...) cannot be negative, but
        #  - a > 0 and 0 < el <= eu -> al = el, au = eu
        #  - a < 0 and 0 < el <= eu -> al = -eu, au = -el
        #  - el <= 0 -> al = -eu, au = eu
        # TODO: an interval union could represent that the two sometimes-
        #       disjoint intervals in the future
        arg_lower: np.ndarray[tuple[Ps], np.dtype[F]] = _ensure_array(
            expr_lower, copy=True
        )
        np.negative(
            expr_upper,
            out=arg_lower,
            where=(np.less_equal(expr_lower, 0) | _is_sign_negative_number(argv)),
        )
        arg_upper: np.ndarray[tuple[Ps], np.dtype[F]] = _ensure_array(
            expr_upper, copy=True
        )
        np.negative(
            expr_lower,
            out=arg_upper,
            where=(np.greater(expr_lower, 0) & _is_sign_negative_number(argv)),
        )

        return arg.compute_data_bounds(
            arg_lower,
            arg_upper,
            Xs,
            late_bound,
        )

    @override
    def __repr__(self) -> str:
        return f"abs({self._a!r})"
