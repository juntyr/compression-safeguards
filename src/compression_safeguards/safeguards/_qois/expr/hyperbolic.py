from collections.abc import Mapping

import numpy as np
from typing_extensions import override  # MSPV 3.12

from ....utils._compat import (
    _ensure_array,
    _is_sign_negative_number,
    _maximum_zero_sign_sensitive,
    _minimum_zero_sign_sensitive,
)
from ....utils.bindings import Parameter
from ..bound import checked_data_bounds, guarantee_arg_within_expr_bounds
from ..typing import F, Ns, Ps, np_sndarray
from .abc import AnyExpr, Expr
from .constfold import ScalarFoldedConstant


class ScalarSinh(Expr[AnyExpr]):
    __slots__: tuple[str, ...] = ("_a",)
    _a: AnyExpr

    def __init__(self, a: AnyExpr) -> None:
        self._a = a

    @property
    @override
    def args(self) -> tuple[AnyExpr]:
        return (self._a,)

    @override
    def with_args(self, a: AnyExpr) -> "ScalarSinh":
        return ScalarSinh(a)

    @override
    def constant_fold(self, dtype: np.dtype[F]) -> F | AnyExpr:
        return ScalarFoldedConstant.constant_fold_unary(
            self._a,
            dtype,
            np.sinh,
            ScalarSinh,
        )

    @override
    def eval(
        self,
        Xs: np_sndarray[Ps, Ns, np.dtype[F]],
        late_bound: Mapping[Parameter, np_sndarray[Ps, Ns, np.dtype[F]]],
    ) -> np.ndarray[tuple[Ps], np.dtype[F]]:
        return np.sinh(self._a.eval(Xs, late_bound))

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
        # evaluate arg and sinh(arg)
        arg = self._a
        argv = arg.eval(Xs, late_bound)
        exprv = np.sinh(argv)

        # apply the inverse function to get the bounds on arg
        # if arg_lower == argv and argv == -0.0, we need to guarantee that
        #  arg_lower is also -0.0, same for arg_upper
        arg_lower: np.ndarray[tuple[Ps], np.dtype[F]] = _ensure_array(
            _minimum_zero_sign_sensitive(argv, np.asinh(expr_lower))
        )
        arg_upper: np.ndarray[tuple[Ps], np.dtype[F]] = _ensure_array(
            _maximum_zero_sign_sensitive(argv, np.asinh(expr_upper))
        )

        # we need to force argv if expr_lower == expr_upper
        np.copyto(arg_lower, argv, where=(expr_lower == expr_upper), casting="no")
        np.copyto(arg_upper, argv, where=(expr_lower == expr_upper), casting="no")

        # handle rounding errors in sinh(asinh(...)) early
        arg_lower = guarantee_arg_within_expr_bounds(
            lambda arg_lower: np.sinh(arg_lower),
            exprv,
            argv,
            arg_lower,
            expr_lower,
            expr_upper,
        )
        arg_upper = guarantee_arg_within_expr_bounds(
            lambda arg_upper: np.sinh(arg_upper),
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
        return f"sinh({self._a!r})"


class ScalarCosh(Expr[AnyExpr]):
    __slots__: tuple[str, ...] = ("_a",)
    _a: AnyExpr

    def __init__(self, a: AnyExpr) -> None:
        self._a = a

    @property
    @override
    def args(self) -> tuple[AnyExpr]:
        return (self._a,)

    @override
    def with_args(self, a: AnyExpr) -> "ScalarCosh":
        return ScalarCosh(a)

    @override
    def constant_fold(self, dtype: np.dtype[F]) -> F | AnyExpr:
        return ScalarFoldedConstant.constant_fold_unary(
            self._a,
            dtype,
            np.cosh,
            ScalarCosh,
        )

    @override
    def eval(
        self,
        Xs: np_sndarray[Ps, Ns, np.dtype[F]],
        late_bound: Mapping[Parameter, np_sndarray[Ps, Ns, np.dtype[F]]],
    ) -> np.ndarray[tuple[Ps], np.dtype[F]]:
        return np.cosh(self._a.eval(Xs, late_bound))

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
        # evaluate arg and cosh(arg)
        arg = self._a
        argv = arg.eval(Xs, late_bound)
        exprv = np.cosh(argv)

        # apply the inverse function to get the bounds on arg
        al = np.acosh(_maximum_zero_sign_sensitive(expr_lower, Xs.dtype.type(1)))
        au = np.acosh(expr_upper)

        # flip and swap the expr bounds to get the bounds on arg
        # cosh(...) cannot be less than 1, but
        #  - a > 0 and 1 < el <= eu -> al = el, au = eu
        #  - a < 0 and 1 < el <= eu -> al = -eu, au = -el
        #  - el <= 1 -> al = -eu, au = eu
        # TODO: an interval union could represent that the two sometimes-
        #       disjoint intervals in the future
        arg_lower: np.ndarray[tuple[Ps], np.dtype[F]] = _ensure_array(al, copy=True)
        np.negative(
            au,
            out=arg_lower,
            where=(np.less_equal(expr_lower, 1) | _is_sign_negative_number(argv)),
        )
        arg_lower = _ensure_array(_minimum_zero_sign_sensitive(argv, arg_lower))

        arg_upper: np.ndarray[tuple[Ps], np.dtype[F]] = _ensure_array(au, copy=True)
        np.negative(
            al,
            out=arg_upper,
            where=(np.greater(expr_lower, 1) & _is_sign_negative_number(argv)),
        )
        arg_upper = _ensure_array(_maximum_zero_sign_sensitive(argv, arg_upper))

        # we need to force argv if expr_lower == expr_upper
        np.copyto(arg_lower, argv, where=(expr_lower == expr_upper), casting="no")
        np.copyto(arg_upper, argv, where=(expr_lower == expr_upper), casting="no")

        # handle rounding errors in cosh(acosh(...)) early
        arg_lower = guarantee_arg_within_expr_bounds(
            lambda arg_lower: np.cosh(arg_lower),
            exprv,
            argv,
            arg_lower,
            expr_lower,
            expr_upper,
        )
        arg_upper = guarantee_arg_within_expr_bounds(
            lambda arg_upper: np.cosh(arg_upper),
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
        return f"cosh({self._a!r})"


class ScalarTanh(Expr[AnyExpr]):
    __slots__: tuple[str, ...] = ("_a",)
    _a: AnyExpr

    def __init__(self, a: AnyExpr) -> None:
        self._a = a

    @property
    @override
    def args(self) -> tuple[AnyExpr]:
        return (self._a,)

    @override
    def with_args(self, a: AnyExpr) -> "ScalarTanh":
        return ScalarTanh(a)

    @override
    def constant_fold(self, dtype: np.dtype[F]) -> F | AnyExpr:
        return ScalarFoldedConstant.constant_fold_unary(
            self._a,
            dtype,
            np.tanh,
            ScalarTanh,
        )

    @override
    def eval(
        self,
        Xs: np_sndarray[Ps, Ns, np.dtype[F]],
        late_bound: Mapping[Parameter, np_sndarray[Ps, Ns, np.dtype[F]]],
    ) -> np.ndarray[tuple[Ps], np.dtype[F]]:
        return np.tanh(self._a.eval(Xs, late_bound))

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
        # evaluate arg and tanh(arg)
        arg = self._a
        argv = arg.eval(Xs, late_bound)
        exprv = np.tanh(argv)

        # apply the inverse function to get the bounds on arg
        # ensure that the bounds on tanh(...) are in [-1, +1]
        # if arg_lower == argv and argv == -0.0, we need to guarantee that
        #  arg_lower is also -0.0, same for arg_upper
        arg_lower: np.ndarray[tuple[Ps], np.dtype[F]] = _ensure_array(
            _minimum_zero_sign_sensitive(
                argv,
                np.atanh(_maximum_zero_sign_sensitive(Xs.dtype.type(-1), expr_lower)),
            )
        )
        arg_upper: np.ndarray[tuple[Ps], np.dtype[F]] = _ensure_array(
            _maximum_zero_sign_sensitive(
                argv,
                np.atanh(_minimum_zero_sign_sensitive(expr_upper, Xs.dtype.type(1))),
            )
        )

        # we need to force argv if expr_lower == expr_upper
        np.copyto(arg_lower, argv, where=(expr_lower == expr_upper), casting="no")
        np.copyto(arg_upper, argv, where=(expr_lower == expr_upper), casting="no")

        # handle rounding errors in tanh(atanh(...)) early
        arg_lower = guarantee_arg_within_expr_bounds(
            lambda arg_lower: np.tanh(arg_lower),
            exprv,
            argv,
            arg_lower,
            expr_lower,
            expr_upper,
        )
        arg_upper = guarantee_arg_within_expr_bounds(
            lambda arg_upper: np.tanh(arg_upper),
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
        return f"tanh({self._a!r})"


class ScalarAsinh(Expr[AnyExpr]):
    __slots__: tuple[str, ...] = ("_a",)
    _a: AnyExpr

    def __init__(self, a: AnyExpr) -> None:
        self._a = a

    @property
    @override
    def args(self) -> tuple[AnyExpr]:
        return (self._a,)

    @override
    def with_args(self, a: AnyExpr) -> "ScalarAsinh":
        return ScalarAsinh(a)

    @override
    def constant_fold(self, dtype: np.dtype[F]) -> F | AnyExpr:
        return ScalarFoldedConstant.constant_fold_unary(
            self._a,
            dtype,
            np.asinh,
            ScalarAsinh,
        )

    @override
    def eval(
        self,
        Xs: np_sndarray[Ps, Ns, np.dtype[F]],
        late_bound: Mapping[Parameter, np_sndarray[Ps, Ns, np.dtype[F]]],
    ) -> np.ndarray[tuple[Ps], np.dtype[F]]:
        return np.asinh(self._a.eval(Xs, late_bound))

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
        # evaluate arg and asinh(arg)
        arg = self._a
        argv = arg.eval(Xs, late_bound)
        exprv = np.asinh(argv)

        # apply the inverse function to get the bounds on arg
        # if arg_lower == argv and argv == -0.0, we need to guarantee that
        #  arg_lower is also -0.0, same for arg_upper
        arg_lower: np.ndarray[tuple[Ps], np.dtype[F]] = _ensure_array(
            _minimum_zero_sign_sensitive(argv, np.sinh(expr_lower))
        )
        arg_upper: np.ndarray[tuple[Ps], np.dtype[F]] = _ensure_array(
            _maximum_zero_sign_sensitive(argv, np.sinh(expr_upper))
        )

        # we need to force argv if expr_lower == expr_upper
        np.copyto(arg_lower, argv, where=(expr_lower == expr_upper), casting="no")
        np.copyto(arg_upper, argv, where=(expr_lower == expr_upper), casting="no")

        # handle rounding errors in asinh(sinh(...)) early
        arg_lower = guarantee_arg_within_expr_bounds(
            lambda arg_lower: np.asinh(arg_lower),
            exprv,
            argv,
            arg_lower,
            expr_lower,
            expr_upper,
        )
        arg_upper = guarantee_arg_within_expr_bounds(
            lambda arg_upper: np.asinh(arg_upper),
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
        return f"asinh({self._a!r})"


class ScalarAcosh(Expr[AnyExpr]):
    __slots__: tuple[str, ...] = ("_a",)
    _a: AnyExpr

    def __init__(self, a: AnyExpr) -> None:
        self._a = a

    @property
    @override
    def args(self) -> tuple[AnyExpr]:
        return (self._a,)

    @override
    def with_args(self, a: AnyExpr) -> "ScalarAcosh":
        return ScalarAcosh(a)

    @override
    def constant_fold(self, dtype: np.dtype[F]) -> F | AnyExpr:
        return ScalarFoldedConstant.constant_fold_unary(
            self._a,
            dtype,
            np.acosh,
            ScalarAcosh,
        )

    @override
    def eval(
        self,
        Xs: np_sndarray[Ps, Ns, np.dtype[F]],
        late_bound: Mapping[Parameter, np_sndarray[Ps, Ns, np.dtype[F]]],
    ) -> np.ndarray[tuple[Ps], np.dtype[F]]:
        return np.acosh(self._a.eval(Xs, late_bound))

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
        # evaluate arg and acosh(arg)
        arg = self._a
        argv = arg.eval(Xs, late_bound)
        exprv = np.acosh(argv)

        eps_one = np.nextafter(Xs.dtype.type(1), Xs.dtype.type(0))

        # apply the inverse function to get the bounds on arg
        # acosh(...) is NaN for values smaller than 1 and can then take any
        #  value smaller than one
        # otherwise ensure that the bounds on acosh(...) are non-negative
        arg_lower: np.ndarray[tuple[Ps], np.dtype[F]] = _ensure_array(
            np.cosh(_maximum_zero_sign_sensitive(Xs.dtype.type(0), expr_lower))
        )
        arg_lower[np.less(argv, 1)] = -np.inf
        arg_lower = _ensure_array(_minimum_zero_sign_sensitive(argv, arg_lower))

        arg_upper: np.ndarray[tuple[Ps], np.dtype[F]] = _ensure_array(
            np.cosh(expr_upper)
        )
        arg_upper[np.less(argv, 1)] = eps_one
        arg_upper = _ensure_array(_maximum_zero_sign_sensitive(argv, arg_upper))

        # we need to force argv if expr_lower == expr_upper
        np.copyto(arg_lower, argv, where=(expr_lower == expr_upper), casting="no")
        np.copyto(arg_upper, argv, where=(expr_lower == expr_upper), casting="no")

        # handle rounding errors in acosh(cosh(...)) early
        arg_lower = guarantee_arg_within_expr_bounds(
            lambda arg_lower: np.acosh(arg_lower),
            exprv,
            argv,
            arg_lower,
            expr_lower,
            expr_upper,
        )
        arg_upper = guarantee_arg_within_expr_bounds(
            lambda arg_upper: np.acosh(arg_upper),
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
        return f"acosh({self._a!r})"


class ScalarAtanh(Expr[AnyExpr]):
    __slots__: tuple[str, ...] = ("_a",)
    _a: AnyExpr

    def __init__(self, a: AnyExpr) -> None:
        self._a = a

    @property
    @override
    def args(self) -> tuple[AnyExpr]:
        return (self._a,)

    @override
    def with_args(self, a: AnyExpr) -> "ScalarAtanh":
        return ScalarAtanh(a)

    @override
    def constant_fold(self, dtype: np.dtype[F]) -> F | AnyExpr:
        return ScalarFoldedConstant.constant_fold_unary(
            self._a,
            dtype,
            np.atanh,
            ScalarAtanh,
        )

    @override
    def eval(
        self,
        Xs: np_sndarray[Ps, Ns, np.dtype[F]],
        late_bound: Mapping[Parameter, np_sndarray[Ps, Ns, np.dtype[F]]],
    ) -> np.ndarray[tuple[Ps], np.dtype[F]]:
        return np.atanh(self._a.eval(Xs, late_bound))

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
        # evaluate arg and atanh(arg)
        arg = self._a
        argv = arg.eval(Xs, late_bound)
        exprv = np.atanh(argv)

        one_eps = np.nextafter(Xs.dtype.type(1), Xs.dtype.type(2))

        # apply the inverse function to get the bounds on arg
        # atanh(...) is NaN when abs(...) > 1 and can then take any value > 1
        # if arg_lower == argv and argv == -0.0, we need to guarantee that
        #  arg_lower is also -0.0, same for arg_upper
        arg_lower: np.ndarray[tuple[Ps], np.dtype[F]] = _ensure_array(
            np.tanh(expr_lower)
        )
        arg_lower[np.greater(argv, 1)] = one_eps
        arg_lower[np.less(argv, -1)] = -np.inf
        arg_lower = _ensure_array(_minimum_zero_sign_sensitive(argv, arg_lower))

        arg_upper: np.ndarray[tuple[Ps], np.dtype[F]] = _ensure_array(
            np.tanh(expr_upper)
        )
        arg_upper[np.greater(argv, 1)] = np.inf
        arg_upper[np.less(argv, -1)] = -one_eps
        arg_upper = _ensure_array(_maximum_zero_sign_sensitive(argv, arg_upper))

        # we need to force argv if expr_lower == expr_upper
        np.copyto(arg_lower, argv, where=(expr_lower == expr_upper), casting="no")
        np.copyto(arg_upper, argv, where=(expr_lower == expr_upper), casting="no")

        # handle rounding errors in atanh(tanh(...)) early
        arg_lower = guarantee_arg_within_expr_bounds(
            lambda arg_lower: np.atanh(arg_lower),
            exprv,
            argv,
            arg_lower,
            expr_lower,
            expr_upper,
        )
        arg_upper = guarantee_arg_within_expr_bounds(
            lambda arg_upper: np.atanh(arg_upper),
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
        return f"atanh({self._a!r})"
