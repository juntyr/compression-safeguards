from collections.abc import Callable, Mapping
from enum import Enum, auto

import numpy as np
from typing_extensions import override  # MSPV 3.12

from ....utils._compat import (
    _ensure_array,
    _is_negative_zero,
    _maximum_zero_sign_sensitive,
    _minimum_zero_sign_sensitive,
)
from ....utils.bindings import Parameter
from ..bound import checked_data_bounds, guarantee_arg_within_expr_bounds
from ..typing import F, Ns, Ps, np_sndarray
from .abc import AnyExpr, Expr
from .constfold import ScalarFoldedConstant


class Logarithm(Enum):
    ln = auto()
    log2 = auto()
    log10 = auto()


class ScalarLog(Expr[AnyExpr]):
    __slots__: tuple[str, ...] = ("_log", "_a")
    _log: Logarithm
    _a: AnyExpr

    def __init__(self, log: Logarithm, a: AnyExpr) -> None:
        self._log = log
        self._a = a

    @property
    @override
    def args(self) -> tuple[AnyExpr]:
        return (self._a,)

    @override
    def with_args(self, a: AnyExpr) -> "ScalarLog":
        return ScalarLog(self._log, a)

    @override
    def constant_fold(self, dtype: np.dtype[F]) -> F | AnyExpr:
        return ScalarFoldedConstant.constant_fold_unary(
            self._a,
            dtype,
            LOGARITHM_UFUNC[self._log],  # type: ignore
            lambda e: ScalarLog(self._log, e),
        )

    @override
    def eval(
        self,
        Xs: np_sndarray[Ps, Ns, np.dtype[F]],
        late_bound: Mapping[Parameter, np_sndarray[Ps, Ns, np.dtype[F]]],
    ) -> np.ndarray[tuple[Ps], np.dtype[F]]:
        return (LOGARITHM_UFUNC[self._log])(self._a.eval(Xs, late_bound))

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
        # evaluate arg and log(arg)
        arg = self._a
        argv = arg.eval(Xs, late_bound)
        exprv = (LOGARITHM_UFUNC[self._log])(argv)

        smallest_subnormal = np.finfo(Xs.dtype).smallest_subnormal

        # apply the inverse function to get the bounds on arg
        # log(...) is NaN for negative values and can then take any negative
        #  value
        # if arg_lower == argv and argv == -0.0, we need to guarantee that
        #  arg_lower is also -0.0, same for arg_upper
        arg_lower: np.ndarray[tuple[Ps], np.dtype[F]] = _ensure_array(
            _minimum_zero_sign_sensitive(
                argv, (LOGARITHM_EXPONENTIAL_UFUNC[self._log])(expr_lower)
            )
        )
        arg_lower[np.less(argv, 0)] = -np.inf

        arg_upper: np.ndarray[tuple[Ps], np.dtype[F]] = _ensure_array(
            _maximum_zero_sign_sensitive(
                argv, (LOGARITHM_EXPONENTIAL_UFUNC[self._log])(expr_upper)
            )
        )
        arg_upper[np.less(argv, 0)] = -smallest_subnormal

        # an upper bound of -0.0 must be nudged downwards from 1 to 1-eps
        arg_upper[_is_negative_zero(expr_upper)] = np.nextafter(
            np.array(Xs.dtype.type(1)), np.array(Xs.dtype.type(0))
        )

        # we need to force argv if expr_lower == expr_upper
        np.copyto(arg_lower, argv, where=(expr_lower == expr_upper), casting="no")
        np.copyto(arg_upper, argv, where=(expr_lower == expr_upper), casting="no")

        # handle rounding errors in log(exp(...)) early
        arg_lower = guarantee_arg_within_expr_bounds(
            lambda arg_lower: (LOGARITHM_UFUNC[self._log])(arg_lower),
            exprv,
            argv,
            arg_lower,
            expr_lower,
            expr_upper,
        )
        arg_upper = guarantee_arg_within_expr_bounds(
            lambda arg_upper: (LOGARITHM_UFUNC[self._log])(arg_upper),
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
        return f"{self._log.name}({self._a!r})"


LOGARITHM_UFUNC: dict[
    Logarithm,
    Callable[[np.ndarray[tuple[Ps], np.dtype[F]]], np.ndarray[tuple[Ps], np.dtype[F]]],
] = {
    Logarithm.ln: np.log,
    Logarithm.log2: np.log2,
    Logarithm.log10: np.log10,
}


class Exponential(Enum):
    exp = auto()
    exp2 = auto()
    exp10 = auto()


class ScalarExp(Expr[AnyExpr]):
    __slots__: tuple[str, ...] = ("_exp", "_a")
    _exp: Exponential
    _a: AnyExpr

    def __init__(self, exp: Exponential, a: AnyExpr) -> None:
        self._exp = exp
        self._a = a

    @property
    @override
    def args(self) -> tuple[AnyExpr]:
        return (self._a,)

    @override
    def with_args(self, a: AnyExpr) -> "ScalarExp":
        return ScalarExp(self._exp, a)

    @override
    def constant_fold(self, dtype: np.dtype[F]) -> F | AnyExpr:
        return ScalarFoldedConstant.constant_fold_unary(
            self._a,
            dtype,
            EXPONENTIAL_UFUNC[self._exp],  # type: ignore
            lambda e: ScalarExp(self._exp, e),
        )

    @override
    def eval(
        self,
        Xs: np_sndarray[Ps, Ns, np.dtype[F]],
        late_bound: Mapping[Parameter, np_sndarray[Ps, Ns, np.dtype[F]]],
    ) -> np.ndarray[tuple[Ps], np.dtype[F]]:
        return (EXPONENTIAL_UFUNC[self._exp])(self._a.eval(Xs, late_bound))

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
        # evaluate arg and exp(arg)
        arg = self._a
        argv = arg.eval(Xs, late_bound)
        exprv = (EXPONENTIAL_UFUNC[self._exp])(argv)

        # apply the inverse function to get the bounds on arg
        # exp(...) cannot be negative, so ensure the bounds on expr also cannot
        # if arg_lower == argv and argv == -0.0, we need to guarantee that
        #  arg_lower is also -0.0, same for arg_upper
        arg_lower: np.ndarray[tuple[Ps], np.dtype[F]] = _ensure_array(
            _minimum_zero_sign_sensitive(
                argv,
                (EXPONENTIAL_LOGARITHM_UFUNC[self._exp])(
                    _maximum_zero_sign_sensitive(Xs.dtype.type(0), expr_lower)
                ),
            )
        )
        arg_upper: np.ndarray[tuple[Ps], np.dtype[F]] = _ensure_array(
            _maximum_zero_sign_sensitive(
                argv, (EXPONENTIAL_LOGARITHM_UFUNC[self._exp])(expr_upper)
            )
        )

        # we need to force argv if expr_lower == expr_upper
        np.copyto(arg_lower, argv, where=(expr_lower == expr_upper), casting="no")
        np.copyto(arg_upper, argv, where=(expr_lower == expr_upper), casting="no")

        # handle rounding errors in exp(log(...)) early
        arg_lower = guarantee_arg_within_expr_bounds(
            lambda arg_lower: (EXPONENTIAL_UFUNC[self._exp])(arg_lower),
            exprv,
            argv,
            arg_lower,
            expr_lower,
            expr_upper,
        )
        arg_upper = guarantee_arg_within_expr_bounds(
            lambda arg_upper: (EXPONENTIAL_UFUNC[self._exp])(arg_upper),
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
        return f"{self._exp.name}({self._a!r})"


EXPONENTIAL_UFUNC: dict[
    Exponential,
    Callable[[np.ndarray[tuple[Ps], np.dtype[F]]], np.ndarray[tuple[Ps], np.dtype[F]]],
] = {
    Exponential.exp: np.exp,
    Exponential.exp2: np.exp2,
    Exponential.exp10: lambda x: np.power(10, x),
}

LOGARITHM_EXPONENTIAL_UFUNC: dict[
    Logarithm,
    Callable[[np.ndarray[tuple[Ps], np.dtype[F]]], np.ndarray[tuple[Ps], np.dtype[F]]],
] = {
    Logarithm.ln: EXPONENTIAL_UFUNC[Exponential.exp],
    Logarithm.log2: EXPONENTIAL_UFUNC[Exponential.exp2],
    Logarithm.log10: EXPONENTIAL_UFUNC[Exponential.exp10],
}

EXPONENTIAL_LOGARITHM_UFUNC: dict[
    Exponential,
    Callable[[np.ndarray[tuple[Ps], np.dtype[F]]], np.ndarray[tuple[Ps], np.dtype[F]]],
] = {
    Exponential.exp: LOGARITHM_UFUNC[Logarithm.ln],
    Exponential.exp2: LOGARITHM_UFUNC[Logarithm.log2],
    Exponential.exp10: LOGARITHM_UFUNC[Logarithm.log10],
}


class ScalarLogWithBase(Expr[AnyExpr, AnyExpr]):
    __slots__: tuple[str, ...] = ("_a", "_b")
    _a: AnyExpr
    _b: AnyExpr

    def __init__(self, a: AnyExpr, b: AnyExpr) -> None:
        self._a = a
        self._b = b

    @property
    @override
    def args(self) -> tuple[AnyExpr, AnyExpr]:
        return (self._a, self._b)

    @override
    def with_args(self, a: AnyExpr, b: AnyExpr) -> "ScalarLogWithBase":
        return ScalarLogWithBase(a, b)

    @override
    def constant_fold(self, dtype: np.dtype[F]) -> F | AnyExpr:
        from .divmul import ScalarDivide  # noqa: PLC0415

        return ScalarDivide(
            ScalarLog(
                Logarithm.ln,
                self._a,
            ),
            ScalarLog(
                Logarithm.ln,
                self._b,
            ),
        ).constant_fold(dtype)

    @override
    def eval(
        self,
        Xs: np_sndarray[Ps, Ns, np.dtype[F]],
        late_bound: Mapping[Parameter, np_sndarray[Ps, Ns, np.dtype[F]]],
    ) -> np.ndarray[tuple[Ps], np.dtype[F]]:
        from .divmul import ScalarDivide  # noqa: PLC0415

        return ScalarDivide(
            ScalarLog(
                Logarithm.ln,
                self._a,
            ),
            ScalarLog(
                Logarithm.ln,
                self._b,
            ),
        ).eval(Xs, late_bound)

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
        from .divmul import ScalarDivide  # noqa: PLC0415

        return ScalarDivide(
            ScalarLog(
                Logarithm.ln,
                self._a,
            ),
            ScalarLog(
                Logarithm.ln,
                self._b,
            ),
        ).compute_data_bounds(expr_lower, expr_upper, Xs, late_bound)

    @override
    def __repr__(self) -> str:
        return f"log({self._a!r}, base={self._b!r})"
