from collections.abc import Mapping

import numpy as np

from ....utils._compat import _floating_smallest_subnormal, _isnan, _sign
from ....utils.bindings import Parameter
from .abc import Expr
from .constfold import ScalarFoldedConstant
from .typing import F, Ns, Ps, PsI


class ScalarSign(Expr):
    __slots__ = ("_a",)
    _a: Expr

    def __init__(self, a: Expr):
        self._a = a

    @property
    def has_data(self) -> bool:
        return self._a.has_data

    @property
    def data_indices(self) -> frozenset[tuple[int, ...]]:
        return self._a.data_indices

    def apply_array_element_offset(
        self,
        axis: int,
        offset: int,
    ) -> Expr:
        return ScalarSign(
            self._a.apply_array_element_offset(axis, offset),
        )

    @property
    def late_bound_constants(self) -> frozenset[Parameter]:
        return self._a.late_bound_constants

    def constant_fold(self, dtype: np.dtype[F]) -> F | Expr:
        return ScalarFoldedConstant.constant_fold_unary(
            self._a,
            dtype,
            _sign,  # type: ignore
            ScalarSign,
        )

    def eval(
        self,
        x: PsI,
        Xs: np.ndarray[Ns, np.dtype[F]],
        late_bound: Mapping[Parameter, np.ndarray[Ns, np.dtype[F]]],
    ) -> np.ndarray[PsI, np.dtype[F]]:
        return _sign(self._a.eval(x, Xs, late_bound))

    def compute_data_bounds_unchecked(
        self,
        expr_lower: np.ndarray[Ps, np.dtype[F]],
        expr_upper: np.ndarray[Ps, np.dtype[F]],
        X: np.ndarray[Ps, np.dtype[F]],
        Xs: np.ndarray[Ns, np.dtype[F]],
        late_bound: Mapping[Parameter, np.ndarray[Ns, np.dtype[F]]],
    ) -> tuple[np.ndarray[Ns, np.dtype[F]], np.ndarray[Ns, np.dtype[F]]]:
        # evaluate arg and sign(arg)
        arg = self._a
        argv = arg.eval(X.shape, Xs, late_bound)
        exprv: np.ndarray[Ps, np.dtype[F]] = _sign(argv)

        # evaluate the lower and upper sign bounds that satisfy the expression bound
        expr_lower = np.maximum(-1, np.ceil(expr_lower))
        expr_upper = np.minimum(np.floor(expr_upper), +1)

        smallest_subnormal = _floating_smallest_subnormal(X.dtype)

        # compute the lower and upper arg bounds that produce the sign bounds
        # sign(-0.0) = +0.0 and sign(+0.0) = +0.0
        arg_lower: np.ndarray[Ps, np.dtype[F]] = np.where(  # type: ignore
            _isnan(exprv),
            exprv,
            np.where(
                expr_lower == 0,
                X.dtype.type(-0.0),
                np.where(
                    expr_lower < 0,
                    np.array(-np.inf, dtype=X.dtype),
                    smallest_subnormal,
                ),
            ),
        )
        arg_upper: np.ndarray[Ps, np.dtype[F]] = np.where(  # type: ignore
            _isnan(exprv),
            exprv,
            np.where(
                expr_upper == 0,
                X.dtype.type(+0.0),
                np.where(
                    expr_upper < 0,
                    -smallest_subnormal,
                    np.array(np.inf, dtype=X.dtype),
                ),
            ),
        )

        return arg.compute_data_bounds(
            arg_lower,
            arg_upper,
            X,
            Xs,
            late_bound,
        )

    def compute_data_bounds(
        self,
        expr_lower: np.ndarray[Ps, np.dtype[F]],
        expr_upper: np.ndarray[Ps, np.dtype[F]],
        X: np.ndarray[Ps, np.dtype[F]],
        Xs: np.ndarray[Ns, np.dtype[F]],
        late_bound: Mapping[Parameter, np.ndarray[Ns, np.dtype[F]]],
    ) -> tuple[np.ndarray[Ns, np.dtype[F]], np.ndarray[Ns, np.dtype[F]]]:
        # the unchecked method already handles rounding errors for sign,
        #  which is weakly monotonic
        return self.compute_data_bounds_unchecked(
            expr_lower, expr_upper, X, Xs, late_bound
        )

    def __repr__(self) -> str:
        return f"sign({self._a!r})"
