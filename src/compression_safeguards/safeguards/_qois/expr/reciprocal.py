from collections.abc import Mapping

import numpy as np
from typing_extensions import override  # MSPV 3.12

from ....utils._compat import (
    _ensure_array,
    _is_sign_negative_number,
    _is_sign_positive_number,
    _maximum_zero_sign_sensitive,
    _minimum_zero_sign_sensitive,
)
from ....utils.bindings import Parameter
from ..bound import checked_data_bounds, guarantee_arg_within_expr_bounds
from ..typing import F, Ns, Ps, np_sndarray
from .abc import AnyExpr, Expr
from .constfold import ScalarFoldedConstant


class ScalarReciprocal(Expr[AnyExpr]):
    __slots__: tuple[str, ...] = ("_a",)
    _a: AnyExpr

    def __init__(self, a: AnyExpr):
        self._a = a

    @property
    @override
    def args(self) -> tuple[AnyExpr]:
        return (self._a,)

    @override
    def with_args(self, a: AnyExpr) -> "ScalarReciprocal":
        return ScalarReciprocal(a)

    @override
    def constant_fold(self, dtype: np.dtype[F]) -> F | AnyExpr:
        return ScalarFoldedConstant.constant_fold_unary(
            self._a,
            dtype,
            np.reciprocal,
            ScalarReciprocal,
        )

    @override
    def eval(
        self,
        Xs: np_sndarray[Ps, Ns, np.dtype[F]],
        late_bound: Mapping[Parameter, np_sndarray[Ps, Ns, np.dtype[F]]],
    ) -> np.ndarray[tuple[Ps], np.dtype[F]]:
        return np.reciprocal(self._a.eval(Xs, late_bound))

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
        # evaluate arg and reciprocal(arg)
        arg = self._a
        argv = arg.eval(Xs, late_bound)
        exprv = np.reciprocal(argv)

        # compute the argument bounds
        # ensure that reciprocal(...) keeps the same sign as arg
        # TODO: an interval union could represent that the two disjoint
        #       intervals in the future
        arg_lower: np.ndarray[tuple[Ps], np.dtype[F]] = _ensure_array(
            _minimum_zero_sign_sensitive(expr_upper, Xs.dtype.type(-0.0))
        )
        np.copyto(
            arg_lower, expr_upper, where=_is_sign_positive_number(exprv), casting="no"
        )
        np.reciprocal(arg_lower, out=arg_lower)
        arg_lower = _ensure_array(_minimum_zero_sign_sensitive(argv, arg_lower))

        arg_upper: np.ndarray[tuple[Ps], np.dtype[F]] = _ensure_array(
            _maximum_zero_sign_sensitive(Xs.dtype.type(+0.0), expr_lower)
        )
        np.copyto(
            arg_upper, expr_lower, where=_is_sign_negative_number(exprv), casting="no"
        )
        np.reciprocal(arg_upper, out=arg_upper)
        arg_upper = _ensure_array(_maximum_zero_sign_sensitive(argv, arg_upper))

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
            Xs,
            late_bound,
        )

    @override
    def __repr__(self) -> str:
        return f"reciprocal({self._a!r})"
