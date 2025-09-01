from collections.abc import Mapping

import numpy as np

from ....utils._compat import (
    _floating_smallest_subnormal,
    _is_negative,
    _maximum,
    _minimum,
)
from ....utils.bindings import Parameter
from ..bound import checked_data_bounds, guarantee_arg_within_expr_bounds
from .abc import Expr
from .constfold import ScalarFoldedConstant
from .typing import F, Ns, Ps, PsI


class ScalarSqrt(Expr):
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
        return ScalarSqrt(
            self._a.apply_array_element_offset(axis, offset),
        )

    @property
    def late_bound_constants(self) -> frozenset[Parameter]:
        return self._a.late_bound_constants

    def constant_fold(self, dtype: np.dtype[F]) -> F | Expr:
        return ScalarFoldedConstant.constant_fold_unary(
            self._a, dtype, np.sqrt, ScalarSqrt
        )

    def eval(
        self,
        x: PsI,
        Xs: np.ndarray[Ns, np.dtype[F]],
        late_bound: Mapping[Parameter, np.ndarray[Ns, np.dtype[F]]],
    ) -> np.ndarray[PsI, np.dtype[F]]:
        return np.sqrt(self._a.eval(x, Xs, late_bound))

    @checked_data_bounds
    def compute_data_bounds_unchecked(
        self,
        expr_lower: np.ndarray[Ps, np.dtype[F]],
        expr_upper: np.ndarray[Ps, np.dtype[F]],
        X: np.ndarray[Ps, np.dtype[F]],
        Xs: np.ndarray[Ns, np.dtype[F]],
        late_bound: Mapping[Parameter, np.ndarray[Ns, np.dtype[F]]],
    ) -> tuple[np.ndarray[Ns, np.dtype[F]], np.ndarray[Ns, np.dtype[F]]]:
        # for sqrt(-0.0), we should return -0.0 as the inverse
        # this ensures that 1/sqrt(-0.0) doesn't become 1/sqrt(0.0)
        def _sqrt_inv(x: np.ndarray[Ps, np.dtype[F]]) -> np.ndarray[Ps, np.dtype[F]]:
            out: np.ndarray[Ps, np.dtype[F]] = np.array(
                np.square(_maximum(X.dtype.type(0), x)), copy=None
            )
            np.copyto(out, x, where=(x == 0), casting="no")
            return out

        # evaluate arg and sqrt(arg)
        arg = self._a
        argv = arg.eval(X.shape, Xs, late_bound)
        exprv = np.sqrt(argv)

        smallest_subnormal = _floating_smallest_subnormal(X.dtype)

        # apply the inverse function to get the bounds on arg
        # sqrt(-0.0) = -0.0 and sqrt(+0.0) = +0.0
        # sqrt(...) is NaN for negative values and can then take any negative
        #  value
        # otherwise ensure that the bounds on sqrt(...) are non-negative
        # if arg_lower == argv and argv == -0.0, we need to guarantee that
        #  arg_lower is also -0.0, same for arg_upper
        arg_lower: np.ndarray[Ps, np.dtype[F]] = np.array(
            _minimum(argv, _sqrt_inv(expr_lower)), copy=None
        )
        arg_lower[np.less(argv, 0)] = -np.inf

        arg_upper: np.ndarray[Ps, np.dtype[F]] = np.array(
            _maximum(argv, _sqrt_inv(expr_upper)), copy=None
        )
        arg_upper[np.less(argv, 0)] = -smallest_subnormal

        # we need to force argv if expr_lower == expr_upper
        np.copyto(arg_lower, argv, where=(expr_lower == expr_upper), casting="no")
        np.copyto(arg_upper, argv, where=(expr_lower == expr_upper), casting="no")

        # handle rounding errors in sqrt(square(...)) early
        arg_lower = guarantee_arg_within_expr_bounds(
            lambda arg_lower: np.sqrt(arg_lower),
            exprv,
            argv,
            arg_lower,
            expr_lower,
            expr_upper,
        )
        arg_upper = guarantee_arg_within_expr_bounds(
            lambda arg_upper: np.sqrt(arg_upper),
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
        return f"sqrt({self._a!r})"


class ScalarSquare(Expr):
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
        return ScalarSquare(
            self._a.apply_array_element_offset(axis, offset),
        )

    @property
    def late_bound_constants(self) -> frozenset[Parameter]:
        return self._a.late_bound_constants

    def constant_fold(self, dtype: np.dtype[F]) -> F | Expr:
        return ScalarFoldedConstant.constant_fold_unary(
            self._a, dtype, np.square, ScalarSquare
        )

    def eval(
        self,
        x: PsI,
        Xs: np.ndarray[Ns, np.dtype[F]],
        late_bound: Mapping[Parameter, np.ndarray[Ns, np.dtype[F]]],
    ) -> np.ndarray[PsI, np.dtype[F]]:
        return np.square(self._a.eval(x, Xs, late_bound))

    @checked_data_bounds
    def compute_data_bounds_unchecked(
        self,
        expr_lower: np.ndarray[Ps, np.dtype[F]],
        expr_upper: np.ndarray[Ps, np.dtype[F]],
        X: np.ndarray[Ps, np.dtype[F]],
        Xs: np.ndarray[Ns, np.dtype[F]],
        late_bound: Mapping[Parameter, np.ndarray[Ns, np.dtype[F]]],
    ) -> tuple[np.ndarray[Ns, np.dtype[F]], np.ndarray[Ns, np.dtype[F]]]:
        # evaluate arg and square(arg)
        arg = self._a
        argv = arg.eval(X.shape, Xs, late_bound)
        exprv = np.square(argv)

        # apply the inverse function to get the bounds on arg
        al = np.sqrt(_maximum(expr_lower, X.dtype.type(0)))
        au = np.sqrt(expr_upper)

        # flip and swap the expr bounds to get the bounds on arg
        # square(...) cannot be negative, but
        #  - a > 0 and 0 < el <= eu -> al = el, au = eu
        #  - a < 0 and 0 < el <= eu -> al = -eu, au = -el
        #  - el <= 0 -> al = -eu, au = eu
        # TODO: an interval union could represent that the two sometimes-
        #       disjoint intervals in the future
        arg_lower: np.ndarray[Ps, np.dtype[F]] = np.array(al, copy=True)
        np.copyto(
            arg_lower,
            -au,
            where=(np.less_equal(expr_lower, 0) | _is_negative(argv)),
            casting="no",
        )
        arg_lower = _minimum(argv, arg_lower)

        arg_upper: np.ndarray[Ps, np.dtype[F]] = np.array(au, copy=True)
        np.copyto(
            arg_upper,
            -al,
            where=(np.greater(expr_lower, 0) & _is_negative(argv)),
            casting="no",
        )
        arg_upper = _maximum(argv, arg_upper)

        # we need to force argv if expr_lower == expr_upper
        np.copyto(arg_lower, argv, where=(expr_lower == expr_upper), casting="no")
        np.copyto(arg_upper, argv, where=(expr_lower == expr_upper), casting="no")

        # handle rounding errors in square(sqrt(...)) early
        arg_lower = guarantee_arg_within_expr_bounds(
            lambda arg_lower: np.square(arg_lower),
            exprv,
            argv,
            arg_lower,
            expr_lower,
            expr_upper,
        )
        arg_upper = guarantee_arg_within_expr_bounds(
            lambda arg_upper: np.square(arg_upper),
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
        return f"square({self._a!r})"
