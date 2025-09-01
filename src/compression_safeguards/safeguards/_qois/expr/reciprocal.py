from collections.abc import Mapping

import numpy as np

from ....utils._compat import (
    _is_negative,
    _maximum,
    _minimum,
    _reciprocal,
    _where,
)
from ....utils.bindings import Parameter
from ..bound import checked_data_bounds, guarantee_arg_within_expr_bounds
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
        exprv = _reciprocal(argv)

        # compute the argument bounds
        # ensure that reciprocal(...) keeps the same sign as arg
        # TODO: an interval union could represent that the two disjoint
        #       intervals in the future
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

        # we need to force argv if expr_lower == expr_upper
        arg_lower = _where(expr_lower == expr_upper, argv, arg_lower)
        arg_upper = _where(expr_lower == expr_upper, argv, arg_upper)

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

    def __repr__(self) -> str:
        return f"reciprocal({self._a!r})"
