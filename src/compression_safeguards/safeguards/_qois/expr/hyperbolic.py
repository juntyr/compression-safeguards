from collections.abc import Mapping

import numpy as np

from ....utils._compat import (
    _acosh,
    _asinh,
    _atanh,
    _cosh,
    _is_negative,
    _maximum,
    _minimum,
    _nextafter,
    _sinh,
    _tanh,
    _where,
)
from ....utils.bindings import Parameter
from ..bound import guarantee_arg_within_expr_bounds, guaranteed_data_bounds
from .abc import Expr
from .constfold import ScalarFoldedConstant
from .typing import F, Ns, Ps, PsI


class ScalarSinh(Expr):
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
        return ScalarSinh(
            self._a.apply_array_element_offset(axis, offset),
        )

    @property
    def late_bound_constants(self) -> frozenset[Parameter]:
        return self._a.late_bound_constants

    def constant_fold(self, dtype: np.dtype[F]) -> F | Expr:
        return ScalarFoldedConstant.constant_fold_unary(
            self._a,
            dtype,
            _sinh,  # type: ignore
            ScalarSinh,
        )

    def eval(
        self,
        x: PsI,
        Xs: np.ndarray[Ns, np.dtype[F]],
        late_bound: Mapping[Parameter, np.ndarray[Ns, np.dtype[F]]],
    ) -> np.ndarray[PsI, np.dtype[F]]:
        return _sinh(self._a.eval(x, Xs, late_bound))

    @guaranteed_data_bounds
    def compute_data_bounds_unchecked(
        self,
        expr_lower: np.ndarray[Ps, np.dtype[F]],
        expr_upper: np.ndarray[Ps, np.dtype[F]],
        X: np.ndarray[Ps, np.dtype[F]],
        Xs: np.ndarray[Ns, np.dtype[F]],
        late_bound: Mapping[Parameter, np.ndarray[Ns, np.dtype[F]]],
    ) -> tuple[np.ndarray[Ns, np.dtype[F]], np.ndarray[Ns, np.dtype[F]]]:
        # evaluate arg and sinh(arg)
        arg = self._a
        argv = arg.eval(X.shape, Xs, late_bound)
        exprv = _sinh(argv)

        # apply the inverse function to get the bounds on arg
        # if arg_lower == argv and argv == -0.0, we need to guarantee that
        #  arg_lower is also -0.0, same for arg_upper
        arg_lower: np.ndarray[Ps, np.dtype[F]] = _minimum(argv, _asinh(expr_lower))
        arg_upper: np.ndarray[Ps, np.dtype[F]] = _maximum(argv, _asinh(expr_upper))

        # we need to force argv if expr_lower == expr_upper
        arg_lower = _where(expr_lower == expr_upper, argv, arg_lower)
        arg_upper = _where(expr_lower == expr_upper, argv, arg_upper)

        # handle rounding errors in sinh(asinh(...)) early
        arg_lower = guarantee_arg_within_expr_bounds(
            lambda arg_lower: _sinh(arg_lower),
            exprv,
            argv,
            arg_lower,
            expr_lower,
            expr_upper,
        )
        arg_upper = guarantee_arg_within_expr_bounds(
            lambda arg_upper: _sinh(arg_upper),
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
        return f"sinh({self._a!r})"


class ScalarCosh(Expr):
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
        return ScalarCosh(
            self._a.apply_array_element_offset(axis, offset),
        )

    @property
    def late_bound_constants(self) -> frozenset[Parameter]:
        return self._a.late_bound_constants

    def constant_fold(self, dtype: np.dtype[F]) -> F | Expr:
        return ScalarFoldedConstant.constant_fold_unary(
            self._a,
            dtype,
            _cosh,  # type: ignore
            ScalarCosh,
        )

    def eval(
        self,
        x: PsI,
        Xs: np.ndarray[Ns, np.dtype[F]],
        late_bound: Mapping[Parameter, np.ndarray[Ns, np.dtype[F]]],
    ) -> np.ndarray[PsI, np.dtype[F]]:
        return _cosh(self._a.eval(x, Xs, late_bound))

    @guaranteed_data_bounds
    def compute_data_bounds_unchecked(
        self,
        expr_lower: np.ndarray[Ps, np.dtype[F]],
        expr_upper: np.ndarray[Ps, np.dtype[F]],
        X: np.ndarray[Ps, np.dtype[F]],
        Xs: np.ndarray[Ns, np.dtype[F]],
        late_bound: Mapping[Parameter, np.ndarray[Ns, np.dtype[F]]],
    ) -> tuple[np.ndarray[Ns, np.dtype[F]], np.ndarray[Ns, np.dtype[F]]]:
        # evaluate arg and cosh(arg)
        arg = self._a
        argv = arg.eval(X.shape, Xs, late_bound)
        exprv = _cosh(argv)

        # apply the inverse function to get the bounds on arg
        al = _acosh(_maximum(expr_lower, X.dtype.type(1)))
        au = _acosh(expr_upper)

        # flip and swap the expr bounds to get the bounds on arg
        # cosh(...) cannot be less than 1, but
        #  - a > 0 and 1 < el <= eu -> al = el, au = eu
        #  - a < 0 and 1 < el <= eu -> al = -eu, au = -el
        #  - el <= 1 -> al = -eu, au = eu
        # TODO: an interval union could represent that the two sometimes-
        #       disjoint intervals in the future
        arg_lower: np.ndarray[Ps, np.dtype[F]] = _minimum(
            argv, _where(np.less_equal(expr_lower, 1) | _is_negative(argv), -au, al)
        )
        arg_upper: np.ndarray[Ps, np.dtype[F]] = _maximum(
            argv, _where(np.greater(expr_lower, 1) & _is_negative(argv), -al, au)
        )

        # we need to force argv if expr_lower == expr_upper
        arg_lower = _where(expr_lower == expr_upper, argv, arg_lower)
        arg_upper = _where(expr_lower == expr_upper, argv, arg_upper)

        # handle rounding errors in cosh(acosh(...)) early
        arg_lower = guarantee_arg_within_expr_bounds(
            lambda arg_lower: _cosh(arg_lower),
            exprv,
            argv,
            arg_lower,
            expr_lower,
            expr_upper,
        )
        arg_upper = guarantee_arg_within_expr_bounds(
            lambda arg_upper: _cosh(arg_upper),
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
        return f"cosh({self._a!r})"


class ScalarTanh(Expr):
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
        return ScalarTanh(
            self._a.apply_array_element_offset(axis, offset),
        )

    @property
    def late_bound_constants(self) -> frozenset[Parameter]:
        return self._a.late_bound_constants

    def constant_fold(self, dtype: np.dtype[F]) -> F | Expr:
        return ScalarFoldedConstant.constant_fold_unary(
            self._a,
            dtype,
            _tanh,  # type: ignore
            ScalarTanh,
        )

    def eval(
        self,
        x: PsI,
        Xs: np.ndarray[Ns, np.dtype[F]],
        late_bound: Mapping[Parameter, np.ndarray[Ns, np.dtype[F]]],
    ) -> np.ndarray[PsI, np.dtype[F]]:
        return _tanh(self._a.eval(x, Xs, late_bound))

    @guaranteed_data_bounds
    def compute_data_bounds_unchecked(
        self,
        expr_lower: np.ndarray[Ps, np.dtype[F]],
        expr_upper: np.ndarray[Ps, np.dtype[F]],
        X: np.ndarray[Ps, np.dtype[F]],
        Xs: np.ndarray[Ns, np.dtype[F]],
        late_bound: Mapping[Parameter, np.ndarray[Ns, np.dtype[F]]],
    ) -> tuple[np.ndarray[Ns, np.dtype[F]], np.ndarray[Ns, np.dtype[F]]]:
        # evaluate arg and tanh(arg)
        arg = self._a
        argv = arg.eval(X.shape, Xs, late_bound)
        exprv = _tanh(argv)

        # apply the inverse function to get the bounds on arg
        # ensure that the bounds on tanh(...) are in [-1, +1]
        # if arg_lower == argv and argv == -0.0, we need to guarantee that
        #  arg_lower is also -0.0, same for arg_upper
        arg_lower: np.ndarray[Ps, np.dtype[F]] = _minimum(
            argv, _atanh(_maximum(X.dtype.type(-1), expr_lower))
        )
        arg_upper: np.ndarray[Ps, np.dtype[F]] = _maximum(
            argv, _atanh(_minimum(expr_upper, X.dtype.type(1)))
        )

        # we need to force argv if expr_lower == expr_upper
        arg_lower = _where(expr_lower == expr_upper, argv, arg_lower)
        arg_upper = _where(expr_lower == expr_upper, argv, arg_upper)

        # handle rounding errors in tanh(atanh(...)) early
        arg_lower = guarantee_arg_within_expr_bounds(
            lambda arg_lower: _tanh(arg_lower),
            exprv,
            argv,
            arg_lower,
            expr_lower,
            expr_upper,
        )
        arg_upper = guarantee_arg_within_expr_bounds(
            lambda arg_upper: _tanh(arg_upper),
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
        return f"tanh({self._a!r})"


class ScalarAsinh(Expr):
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
        return ScalarAsinh(
            self._a.apply_array_element_offset(axis, offset),
        )

    @property
    def late_bound_constants(self) -> frozenset[Parameter]:
        return self._a.late_bound_constants

    def constant_fold(self, dtype: np.dtype[F]) -> F | Expr:
        return ScalarFoldedConstant.constant_fold_unary(
            self._a,
            dtype,
            _asinh,  # type: ignore
            ScalarAsinh,
        )

    def eval(
        self,
        x: PsI,
        Xs: np.ndarray[Ns, np.dtype[F]],
        late_bound: Mapping[Parameter, np.ndarray[Ns, np.dtype[F]]],
    ) -> np.ndarray[PsI, np.dtype[F]]:
        return _asinh(self._a.eval(x, Xs, late_bound))

    @guaranteed_data_bounds
    def compute_data_bounds_unchecked(
        self,
        expr_lower: np.ndarray[Ps, np.dtype[F]],
        expr_upper: np.ndarray[Ps, np.dtype[F]],
        X: np.ndarray[Ps, np.dtype[F]],
        Xs: np.ndarray[Ns, np.dtype[F]],
        late_bound: Mapping[Parameter, np.ndarray[Ns, np.dtype[F]]],
    ) -> tuple[np.ndarray[Ns, np.dtype[F]], np.ndarray[Ns, np.dtype[F]]]:
        # evaluate arg and asinh(arg)
        arg = self._a
        argv = arg.eval(X.shape, Xs, late_bound)
        exprv = _asinh(argv)

        # apply the inverse function to get the bounds on arg
        # if arg_lower == argv and argv == -0.0, we need to guarantee that
        #  arg_lower is also -0.0, same for arg_upper
        arg_lower: np.ndarray[Ps, np.dtype[F]] = _minimum(argv, _sinh(expr_lower))
        arg_upper: np.ndarray[Ps, np.dtype[F]] = _maximum(argv, _sinh(expr_upper))

        # we need to force argv if expr_lower == expr_upper
        arg_lower = _where(expr_lower == expr_upper, argv, arg_lower)
        arg_upper = _where(expr_lower == expr_upper, argv, arg_upper)

        # handle rounding errors in asinh(sinh(...)) early
        arg_lower = guarantee_arg_within_expr_bounds(
            lambda arg_lower: _asinh(arg_lower),
            exprv,
            argv,
            arg_lower,
            expr_lower,
            expr_upper,
        )
        arg_upper = guarantee_arg_within_expr_bounds(
            lambda arg_upper: _asinh(arg_upper),
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
        return f"asinh({self._a!r})"


class ScalarAcosh(Expr):
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
        return ScalarAcosh(
            self._a.apply_array_element_offset(axis, offset),
        )

    @property
    def late_bound_constants(self) -> frozenset[Parameter]:
        return self._a.late_bound_constants

    def constant_fold(self, dtype: np.dtype[F]) -> F | Expr:
        return ScalarFoldedConstant.constant_fold_unary(
            self._a,
            dtype,
            _acosh,  # type: ignore
            ScalarAcosh,
        )

    def eval(
        self,
        x: PsI,
        Xs: np.ndarray[Ns, np.dtype[F]],
        late_bound: Mapping[Parameter, np.ndarray[Ns, np.dtype[F]]],
    ) -> np.ndarray[PsI, np.dtype[F]]:
        return _acosh(self._a.eval(x, Xs, late_bound))

    @guaranteed_data_bounds
    def compute_data_bounds_unchecked(
        self,
        expr_lower: np.ndarray[Ps, np.dtype[F]],
        expr_upper: np.ndarray[Ps, np.dtype[F]],
        X: np.ndarray[Ps, np.dtype[F]],
        Xs: np.ndarray[Ns, np.dtype[F]],
        late_bound: Mapping[Parameter, np.ndarray[Ns, np.dtype[F]]],
    ) -> tuple[np.ndarray[Ns, np.dtype[F]], np.ndarray[Ns, np.dtype[F]]]:
        # evaluate arg and acosh(arg)
        arg = self._a
        argv = arg.eval(X.shape, Xs, late_bound)
        exprv = _acosh(argv)

        eps_one = _nextafter(np.array(1, dtype=X.dtype), np.array(0, dtype=X.dtype))

        # apply the inverse function to get the bounds on arg
        # acosh(...) is NaN for values smaller than 1 and can then take any
        #  value smaller than one
        # otherwise ensure that the bounds on acosh(...) are non-negative
        arg_lower: np.ndarray[Ps, np.dtype[F]] = _where(
            np.less(argv, 1),
            X.dtype.type(-np.inf),
            _minimum(argv, _cosh(_maximum(X.dtype.type(0), expr_lower))),
        )
        arg_upper: np.ndarray[Ps, np.dtype[F]] = _where(
            np.less(argv, 1),
            eps_one,
            _maximum(argv, _cosh(expr_upper)),
        )

        # we need to force argv if expr_lower == expr_upper
        arg_lower = _where(expr_lower == expr_upper, argv, arg_lower)
        arg_upper = _where(expr_lower == expr_upper, argv, arg_upper)

        # handle rounding errors in acosh(cosh(...)) early
        arg_lower = guarantee_arg_within_expr_bounds(
            lambda arg_lower: _acosh(arg_lower),
            exprv,
            argv,
            arg_lower,
            expr_lower,
            expr_upper,
        )
        arg_upper = guarantee_arg_within_expr_bounds(
            lambda arg_upper: _acosh(arg_upper),
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
        return f"acosh({self._a!r})"


class ScalarAtanh(Expr):
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
        return ScalarAtanh(
            self._a.apply_array_element_offset(axis, offset),
        )

    @property
    def late_bound_constants(self) -> frozenset[Parameter]:
        return self._a.late_bound_constants

    def constant_fold(self, dtype: np.dtype[F]) -> F | Expr:
        return ScalarFoldedConstant.constant_fold_unary(
            self._a,
            dtype,
            _atanh,  # type: ignore
            ScalarAtanh,
        )

    def eval(
        self,
        x: PsI,
        Xs: np.ndarray[Ns, np.dtype[F]],
        late_bound: Mapping[Parameter, np.ndarray[Ns, np.dtype[F]]],
    ) -> np.ndarray[PsI, np.dtype[F]]:
        return _atanh(self._a.eval(x, Xs, late_bound))

    @guaranteed_data_bounds
    def compute_data_bounds_unchecked(
        self,
        expr_lower: np.ndarray[Ps, np.dtype[F]],
        expr_upper: np.ndarray[Ps, np.dtype[F]],
        X: np.ndarray[Ps, np.dtype[F]],
        Xs: np.ndarray[Ns, np.dtype[F]],
        late_bound: Mapping[Parameter, np.ndarray[Ns, np.dtype[F]]],
    ) -> tuple[np.ndarray[Ns, np.dtype[F]], np.ndarray[Ns, np.dtype[F]]]:
        # evaluate arg and atanh(arg)
        arg = self._a
        argv = arg.eval(X.shape, Xs, late_bound)
        exprv = _atanh(argv)

        one_eps = _nextafter(np.array(1, dtype=X.dtype), np.array(2, dtype=X.dtype))

        # apply the inverse function to get the bounds on arg
        # atanh(...) is NaN when abs(...) > 1 and can then take any value > 1
        # if arg_lower == argv and argv == -0.0, we need to guarantee that
        #  arg_lower is also -0.0, same for arg_upper
        arg_lower: np.ndarray[Ps, np.dtype[F]] = _where(
            np.less(argv, -1),
            X.dtype.type(-np.inf),
            _where(
                np.greater(argv, 1),
                one_eps,
                _minimum(argv, _tanh(expr_lower)),
            ),
        )
        arg_upper: np.ndarray[Ps, np.dtype[F]] = _where(
            np.less(argv, -1),
            -one_eps,
            _where(
                np.greater(argv, 1),
                X.dtype.type(np.inf),
                _maximum(argv, _tanh(expr_upper)),
            ),
        )

        # we need to force argv if expr_lower == expr_upper
        arg_lower = _where(expr_lower == expr_upper, argv, arg_lower)
        arg_upper = _where(expr_lower == expr_upper, argv, arg_upper)

        # handle rounding errors in atanh(tanh(...)) early
        arg_lower = guarantee_arg_within_expr_bounds(
            lambda arg_lower: _atanh(arg_lower),
            exprv,
            argv,
            arg_lower,
            expr_lower,
            expr_upper,
        )
        arg_upper = guarantee_arg_within_expr_bounds(
            lambda arg_upper: _atanh(arg_upper),
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
        return f"atanh({self._a!r})"
