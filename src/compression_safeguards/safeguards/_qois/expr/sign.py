from collections.abc import Mapping

import numpy as np
from typing_extensions import override  # MSPV 3.12

from ....utils._compat import _is_positive_zero
from ....utils.bindings import Parameter
from ..bound import checked_data_bounds
from ..typing import F, Ns, Ps, np_sndarray
from .abc import AnyExpr, Expr
from .constfold import ScalarFoldedConstant


class ScalarSign(Expr[AnyExpr]):
    __slots__: tuple[str, ...] = ("_a",)
    _a: AnyExpr

    def __init__(self, a: AnyExpr):
        self._a = a

    @property
    @override
    def args(self) -> tuple[AnyExpr]:
        return (self._a,)

    @override
    def with_args(self, a: AnyExpr) -> "ScalarSign":
        return ScalarSign(a)

    @override
    def constant_fold(self, dtype: np.dtype[F]) -> F | AnyExpr:
        return ScalarFoldedConstant.constant_fold_unary(
            self._a,
            dtype,
            np.sign,
            ScalarSign,
        )

    @override
    def eval(
        self,
        Xs: np_sndarray[Ps, Ns, np.dtype[F]],
        late_bound: Mapping[Parameter, np_sndarray[Ps, Ns, np.dtype[F]]],
    ) -> np.ndarray[tuple[Ps], np.dtype[F]]:
        return np.sign(self._a.eval(Xs, late_bound))

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
        # evaluate arg and sign(arg)
        arg = self._a
        argv = arg.eval(Xs, late_bound)
        exprv: np.ndarray[tuple[Ps], np.dtype[F]] = np.sign(argv)

        # evaluate the lower and upper sign bounds that satisfy the expression bound
        expr_lower = np.ceil(expr_lower)
        expr_upper = np.floor(expr_upper)

        smallest_subnormal = np.finfo(Xs.dtype).smallest_subnormal

        # compute the lower and upper arg bounds that produce the sign bounds
        # sign(-0.0) = +0.0 and sign(+0.0) = +0.0
        # - expr_lower = -0.0 (exprv >= +0.0) -> arg_lower = -0.0, exprv = +0.0
        # - expr_lower = +0.0 (exprv >= +0.0) -> arg_lower = -0.0, exprv = +0.0
        # - expr_upper = -0.0 (exprv < 0) -> arg_upper = -subnormal, exprv = -1
        # - expr_upper = +0.0 (exprv <= +0.0) -> arg_upper = +0.0, exprv = +0.0
        arg_lower: np.ndarray[tuple[Ps], np.dtype[F]] = np.full(
            Xs.shape[:1], smallest_subnormal
        )
        arg_lower[np.less_equal(expr_lower, -1)] = -np.inf
        arg_lower[expr_lower == 0] = -0.0
        np.copyto(arg_lower, exprv, where=np.isnan(exprv), casting="no")

        arg_upper: np.ndarray[tuple[Ps], np.dtype[F]] = np.full(
            Xs.shape[:1], -smallest_subnormal
        )
        arg_upper[np.greater_equal(expr_upper, +1)] = np.inf
        arg_upper[_is_positive_zero(expr_upper)] = +0.0
        np.copyto(arg_upper, exprv, where=np.isnan(exprv), casting="no")

        return arg.compute_data_bounds(
            arg_lower,
            arg_upper,
            Xs,
            late_bound,
        )

    @override
    def __repr__(self) -> str:
        return f"sign({self._a!r})"
