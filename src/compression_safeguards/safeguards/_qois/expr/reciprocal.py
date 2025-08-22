from collections.abc import Mapping

import numpy as np

from ....utils._compat import _maximum, _minimum, _reciprocal, _where
from ....utils.bindings import Parameter
from ..bound import guarantee_arg_within_expr_bounds
from .abc import Expr
from .constfold import ScalarFoldedConstant
from .typing import F, Ns, Ps, PsI


class ScalarReciprocal(Expr):
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
        return ScalarReciprocal(
            self._a.apply_array_element_offset(axis, offset),
        )

    @property
    def late_bound_constants(self) -> frozenset[Parameter]:
        return self._a.late_bound_constants

    def constant_fold(self, dtype: np.dtype[F]) -> F | Expr:
        return ScalarFoldedConstant.constant_fold_unary(
            self._a,
            dtype,
            _reciprocal,  # type: ignore
            ScalarReciprocal,
        )

    def eval(
        self,
        x: PsI,
        Xs: np.ndarray[Ns, np.dtype[F]],
        late_bound: Mapping[Parameter, np.ndarray[Ns, np.dtype[F]]],
    ) -> np.ndarray[PsI, np.dtype[F]]:
        return _reciprocal(self._a.eval(x, Xs, late_bound))

    def compute_data_bounds_unchecked(
        self,
        expr_lower: np.ndarray[Ps, np.dtype[F]],
        expr_upper: np.ndarray[Ps, np.dtype[F]],
        X: np.ndarray[Ps, np.dtype[F]],
        Xs: np.ndarray[Ns, np.dtype[F]],
        late_bound: Mapping[Parameter, np.ndarray[Ns, np.dtype[F]]],
    ) -> tuple[np.ndarray[Ns, np.dtype[F]], np.ndarray[Ns, np.dtype[F]]]:
        def _is_negative(
            x: np.ndarray[Ps, np.dtype[F]],
        ) -> np.ndarray[Ps, np.dtype[np.bool]]:
            # check not just for x < 0 but also for x == -0.0
            return (x < 0) | (_reciprocal(x) < 0)  # type: ignore

        # evaluate arg and reciprocal(arg)
        arg = self._a
        argv = arg.eval(X.shape, Xs, late_bound)
        exprv = _reciprocal(argv)

        # compute the argument bounds
        # ensure that reciprocal(...) keeps the same sign as arg
        arg_lower: np.ndarray[Ps, np.dtype[F]] = _minimum(
            argv,
            _reciprocal(
                _where(
                    _is_negative(exprv),
                    _minimum(expr_upper, X.dtype.type(-0.0)),
                    expr_upper,
                )
            ),
        )
        arg_upper: np.ndarray[Ps, np.dtype[F]] = _maximum(
            argv,
            _reciprocal(
                _where(
                    _is_negative(exprv),
                    expr_lower,
                    _maximum(X.dtype.type(+0.0), expr_lower),
                )
            ),
        )

        # handle rounding errors in reciprocal(reciprocal(...)) early
        arg_lower = guarantee_arg_within_expr_bounds(
            lambda arg_lower: _reciprocal(arg_lower),
            exprv,
            argv,
            arg_lower,
            expr_lower,
            expr_upper,
        )
        arg_upper = guarantee_arg_within_expr_bounds(
            lambda arg_upper: _reciprocal(arg_upper),
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

    def compute_data_bounds(
        self,
        expr_lower: np.ndarray[Ps, np.dtype[F]],
        expr_upper: np.ndarray[Ps, np.dtype[F]],
        X: np.ndarray[Ps, np.dtype[F]],
        Xs: np.ndarray[Ns, np.dtype[F]],
        late_bound: Mapping[Parameter, np.ndarray[Ns, np.dtype[F]]],
    ) -> tuple[np.ndarray[Ns, np.dtype[F]], np.ndarray[Ns, np.dtype[F]]]:
        # the unchecked method already handles rounding errors for reciprocal,
        #  which is strictly monotonic in two disjoint segments
        return self.compute_data_bounds_unchecked(
            expr_lower, expr_upper, X, Xs, late_bound
        )

    def __repr__(self) -> str:
        return f"reciprocal({self._a!r})"
