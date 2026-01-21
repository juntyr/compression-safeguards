import operator
from collections.abc import Mapping

import numpy as np
from typing_extensions import override  # MSPV 3.12

from ....utils._compat import (
    _broadcast_to,
    _ensure_array,
    _is_negative_zero,
    _is_positive_zero,
    _maximum_zero_sign_sensitive,
    _minimum_zero_sign_sensitive,
    _stack,
)
from ....utils.bindings import Parameter
from ..bound import checked_data_bounds, guarantee_stacked_arg_within_expr_bounds
from ..typing import F, Ns, Ps, np_sndarray
from .abc import AnyExpr, Expr
from .abs import ScalarAbs
from .constfold import ScalarFoldedConstant
from .literal import Number
from .neg import ScalarNegate


class ScalarAdd(Expr[AnyExpr, AnyExpr]):
    __slots__: tuple[str, ...] = ("_a", "_b")
    _a: AnyExpr
    _b: AnyExpr

    def __init__(self, a: AnyExpr, b: AnyExpr) -> None:
        self._a = a
        self._b = b

    def __new__(cls, a: AnyExpr, b: AnyExpr) -> "ScalarAdd | Number":  # type: ignore[misc]
        ab = Number.symbolic_fold_binary(a, b, operator.add)
        if ab is not None:
            return ab
        return super().__new__(cls)

    @property
    @override
    def args(self) -> tuple[AnyExpr, AnyExpr]:
        return (self._a, self._b)

    @override
    def with_args(self, a: AnyExpr, b: AnyExpr) -> "ScalarAdd | Number":
        return ScalarAdd(a, b)

    @override
    def constant_fold(self, dtype: np.dtype[F]) -> F | AnyExpr:
        return ScalarFoldedConstant.constant_fold_binary(
            self._a, self._b, dtype, np.add, ScalarAdd
        )

    @override
    def eval(
        self,
        Xs: np_sndarray[Ps, Ns, np.dtype[F]],
        late_bound: Mapping[Parameter, np_sndarray[Ps, Ns, np.dtype[F]]],
    ) -> np.ndarray[tuple[Ps], np.dtype[F]]:
        return np.add(self._a.eval(Xs, late_bound), self._b.eval(Xs, late_bound))

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
        return compute_left_associate_sum_data_bounds(
            self, expr_lower, expr_upper, Xs, late_bound
        )

    @override
    def __repr__(self) -> str:
        return f"{self._a!r} + {self._b!r}"


class ScalarSubtract(Expr[AnyExpr, AnyExpr]):
    __slots__: tuple[str, ...] = ("_a", "_b")
    _a: AnyExpr
    _b: AnyExpr

    def __init__(self, a: AnyExpr, b: AnyExpr) -> None:
        self._a = a
        self._b = b

    def __new__(cls, a: AnyExpr, b: AnyExpr) -> "ScalarSubtract | Number":  # type: ignore[misc]
        ab = Number.symbolic_fold_binary(a, b, operator.sub)
        if ab is not None:
            return ab
        return super().__new__(cls)

    @property
    @override
    def args(self) -> tuple[AnyExpr, AnyExpr]:
        return (self._a, self._b)

    @override
    def with_args(self, a: AnyExpr, b: AnyExpr) -> "ScalarSubtract | Number":
        return ScalarSubtract(a, b)

    @override
    def constant_fold(self, dtype: np.dtype[F]) -> F | AnyExpr:
        return ScalarFoldedConstant.constant_fold_binary(
            self._a, self._b, dtype, np.subtract, ScalarSubtract
        )

    @override
    def eval(
        self,
        Xs: np_sndarray[Ps, Ns, np.dtype[F]],
        late_bound: Mapping[Parameter, np_sndarray[Ps, Ns, np.dtype[F]]],
    ) -> np.ndarray[tuple[Ps], np.dtype[F]]:
        return np.subtract(self._a.eval(Xs, late_bound), self._b.eval(Xs, late_bound))

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
        return compute_left_associate_sum_data_bounds(
            self, expr_lower, expr_upper, Xs, late_bound
        )

    @override
    def __repr__(self) -> str:
        return f"{self._a!r} - {self._b!r}"


# left-associative sum over at least three terms, i.e. a + b + c = (a + b) + c
#  - zero terms: sum identity = 0
#  - one term: term
#  - two terms: binary sum
# this class avoids the deep nesting that's required to represent a
#  left-associative sum with ScalarAdd's
class ScalarLeftAssociativeSum(Expr[AnyExpr, AnyExpr, AnyExpr, *tuple[AnyExpr, ...]]):
    __slots__: tuple[str, ...] = ("_a", "_b", "_c", "_ds")
    _a: AnyExpr
    _b: AnyExpr
    _c: AnyExpr
    _ds: tuple[AnyExpr, ...]

    def __new__(cls, *ts: AnyExpr) -> AnyExpr | Number:  # type: ignore[misc]
        # base case: sum identity
        if len(ts) == 0:
            return Number.ZERO

        # we can only fold consecutive symbolic integers from the left
        a, *tsr = ts
        i = 0
        broken = False
        for i, b in enumerate(tsr):
            ab = Number.symbolic_fold_binary(a, b, operator.add)
            if ab is None:
                broken = True
                break
            a = ab
        i += not broken

        match tsr[i:]:
            # addition over one term
            case ():
                return a
            # binary addition
            case (b,):
                return ScalarAdd(a, b)
            # sum over at least three terms
            case tsri:
                sum_ = super().__new__(cls)
                sum_._a = a
                sum_._b, sum_._c, *ds = tsri
                sum_._ds = tuple(ds)
                return sum_

    @property
    @override
    def args(self) -> tuple[AnyExpr, AnyExpr, AnyExpr, *tuple[AnyExpr, ...]]:
        return (self._a, self._b, self._c, *self._ds)

    @override
    def with_args(  # type: ignore[override]
        self, a: AnyExpr, b: AnyExpr, c: AnyExpr, *ds: AnyExpr
    ) -> AnyExpr | Number:
        return ScalarLeftAssociativeSum(a, b, c, *ds)

    @override
    def constant_fold(self, dtype: np.dtype[F]) -> F | AnyExpr:
        # we first individually fold each term
        facc = self._a.constant_fold(dtype)
        fts = [self._b.constant_fold(dtype), self._c.constant_fold(dtype)] + [
            d.constant_fold(dtype) for d in self._ds
        ]

        # next, we can only combine consecutive folded terms from the left
        i = 0
        broken = False
        for i, ft in enumerate(fts):
            if isinstance(facc, Expr) or isinstance(ft, Expr):
                broken = True
                break
            facc = np.add(facc, ft)
        i += not broken

        # if all were folded, return the folded constant
        if not broken:
            return facc

        # otherwise turn all folded constant terms back into expressions
        faccexpr = ScalarFoldedConstant.from_folded(facc)
        ftexprs = [ScalarFoldedConstant.from_folded(ft) for ft in fts[i:]]

        # and create a sum of the folded terms
        return ScalarLeftAssociativeSum(faccexpr, *ftexprs)

    @override
    def eval(
        self,
        Xs: np_sndarray[Ps, Ns, np.dtype[F]],
        late_bound: Mapping[Parameter, np_sndarray[Ps, Ns, np.dtype[F]]],
    ) -> np.ndarray[tuple[Ps], np.dtype[F]]:
        # evaluate the sum left-associative, i.e. a + b + c = (a + b) + c
        acc: np.ndarray[tuple[Ps], np.dtype[F]] = np.add(
            self._a.eval(Xs, late_bound), self._b.eval(Xs, late_bound)
        )

        acc += self._c.eval(Xs, late_bound)

        for d in self._ds:
            acc += d.eval(Xs, late_bound)

        return acc

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
        return compute_left_associate_sum_data_bounds(
            self, expr_lower, expr_upper, Xs, late_bound
        )

    @override
    def __repr__(self) -> str:
        abc = f"{self._a!r} + {self._b!r} + {self._c!r}"

        if len(self._ds) <= 0:
            return abc

        return f"{abc} + {' + '.join(repr(d) for d in self._ds)}"


def compute_left_associate_sum_data_bounds(
    expr: ScalarAdd | ScalarSubtract | ScalarLeftAssociativeSum,
    expr_lower: np.ndarray[tuple[Ps], np.dtype[F]],
    expr_upper: np.ndarray[tuple[Ps], np.dtype[F]],
    Xs: np_sndarray[Ps, Ns, np.dtype[F]],
    late_bound: Mapping[Parameter, np_sndarray[Ps, Ns, np.dtype[F]]],
) -> tuple[
    np_sndarray[Ps, Ns, np.dtype[F]],
    np_sndarray[Ps, Ns, np.dtype[F]],
]:
    def _zero_add(
        a: np.ndarray[tuple[Ps], np.dtype[F]], b: np.ndarray[tuple[Ps], np.dtype[F]]
    ) -> np.ndarray[tuple[Ps], np.dtype[F]]:
        sum_: np.ndarray[tuple[Ps], np.dtype[F]] = _ensure_array(np.add(a, b))
        np.copyto(sum_, a, where=(b == 0), casting="no")
        return sum_

    left_associative_sum = as_left_associative_sum(expr)

    termvs: list[np.ndarray[tuple[Ps], np.dtype[F]]] = []
    abs_factorvs: list[None | np.ndarray[tuple[Ps], np.dtype[F]]] = []

    for term in left_associative_sum:
        abs_factor = get_expr_left_associative_abs_factor_approximate(term)
        termvs.append(term.eval(Xs, late_bound))
        abs_factorvs.append(
            None
            if abs_factor is None
            else np.multiply(
                abs_factor.eval(Xs, late_bound),
                _ensure_array(term.eval_has_data(Xs, late_bound)).astype(Xs.dtype),
            )
        )

    # evaluate the total expression sum
    exprv: np.ndarray[tuple[Ps], np.dtype[F]] = _ensure_array(
        sum(termvs[1:], start=termvs[0])
    )
    expr_lower = _ensure_array(expr_lower)
    expr_upper = _ensure_array(expr_upper)

    # compute the sum of absolute factors
    total_abs_factor_: None | np.ndarray[tuple[Ps], np.dtype[F]] = None
    for abs_factorv in abs_factorvs:
        if abs_factorv is None:
            continue
        if total_abs_factor_ is None:
            total_abs_factor_ = _ensure_array(abs_factorv, copy=True)
        else:
            total_abs_factor_ += abs_factorv
    assert total_abs_factor_ is not None
    total_abs_factor: np.ndarray[tuple[Ps], np.dtype[F]] = total_abs_factor_

    # drop into expression difference bounds to divide up the bound
    # for NaN sums, we use a zero difference to ensure NaNs don't
    #  accidentally propagate into the term difference bounds
    expr_lower_diff: np.ndarray[tuple[Ps], np.dtype[F]] = _ensure_array(
        np.subtract(expr_lower, exprv)
    )
    expr_lower_diff[np.isnan(expr_lower_diff)] = 0
    expr_upper_diff: np.ndarray[tuple[Ps], np.dtype[F]] = _ensure_array(
        np.subtract(expr_upper, exprv)
    )
    expr_upper_diff[np.isnan(expr_upper_diff)] = 0

    # if exprv is NaN, expr_[lower|upper]_diff are zero
    # if expr_[lower|upper] is infinite but exprv is finite,
    #  expr_[lower|upper]_diff is the same infinity as expr_[lower|upper]
    # if expr_[lower|upper] is infinite and exprv is infinite,
    #  (a) -inf - -inf and inf - inf are NaN, so expr_[lower|upper]_diff is
    #      zero
    #  (b) -inf - inf and inf - -inf are +-inf, so expr_[lower|upper]_diff is
    #      the same infinity as expr_[lower|upper]
    tfl: np.ndarray[tuple[Ps], np.dtype[F]] = _ensure_array(
        np.divide(expr_lower_diff, total_abs_factor)
    )
    tfu: np.ndarray[tuple[Ps], np.dtype[F]] = _ensure_array(
        np.divide(expr_upper_diff, total_abs_factor)
    )

    fmax = np.finfo(Xs.dtype).max

    # ensure that the bounds never contain both -inf and +inf since that would
    #  allow NaN to sneak in
    inf_clash = (tfl == -np.inf) & (tfu == np.inf)
    tfl[inf_clash] = -fmax
    tfu[inf_clash] = fmax

    any_nan: np.ndarray[tuple[Ps], np.dtype[np.bool]] = np.isnan(termvs[0])
    for termv in termvs[1:]:
        any_nan |= np.isnan(termv)

    # stack the lower and upper bounds for each term factor
    # if total_abs_factor or exprv is non-finite:
    #  - non-finite values must be preserved exactly
    #  - if there is any NaN term, finite values can have any finite value
    #  - otherwise finite values are also preserved exactly, for simplicity
    # if total_abs_factor is zero, all abs_factorv are also zero and the term
    #  should keep the same value (unless the term is finite and there is any
    #  NaN term, see above)
    # otherwise we split up the difference bound for the terms by their factors
    tl_stack_: list[np.ndarray[tuple[Ps], np.dtype[F]]] = []
    for termv, abs_factorv in zip(termvs, abs_factorvs):
        if abs_factorv is None:
            continue
        tl: np.ndarray[tuple[Ps], np.dtype[F]] = _ensure_array(termv, copy=True)
        tl[np.isfinite(termv) & any_nan] = -fmax
        np.copyto(
            tl,
            _zero_add(termv, tfl * abs_factorv),
            where=(
                (total_abs_factor != 0)
                & np.isfinite(total_abs_factor)
                & np.isfinite(exprv)
            ),
            casting="no",
        )
        # ensure that finite terms don't overflow into infinity unless allowed
        #  by the difference bounds
        tl[
            (tl == -np.inf)
            & np.isfinite(termv)
            & np.isfinite(tfl)
            & np.isfinite(abs_factorv)
        ] = -fmax
        # optimistically try to include -0.0
        tl[(tl == 0) & ~_is_positive_zero(expr_lower)] = -0.0
        tl_stack_.append(tl)
    tl_stack = _stack(tl_stack_)
    tu_stack_: list[np.ndarray[tuple[Ps], np.dtype[F]]] = []
    for termv, abs_factorv in zip(termvs, abs_factorvs):
        if abs_factorv is None:
            continue
        tu: np.ndarray[tuple[Ps], np.dtype[F]] = _ensure_array(termv, copy=True)
        tu[np.isfinite(termv) & any_nan] = fmax
        np.copyto(
            tu,
            _zero_add(termv, tfu * abs_factorv),
            where=(
                (total_abs_factor != 0)
                & np.isfinite(total_abs_factor)
                & np.isfinite(exprv)
            ),
            casting="no",
        )
        # ensure that finite terms don't overflow into infinity unless allowed
        #  by the difference bounds
        tu[
            (tu == np.inf)
            & np.isfinite(termv)
            & np.isfinite(tfl)
            & np.isfinite(abs_factorv)
        ] = fmax
        # optimistically try to include +0.0
        tu[(tu == 0) & ~_is_negative_zero(expr_upper)] = +0.0
        tu_stack_.append(tu)
    tu_stack = _stack(tu_stack_)

    def compute_term_sum(
        t_stack: np.ndarray[tuple[int, Ps], np.dtype[F]],
    ) -> np.ndarray[tuple[int, Ps], np.dtype[F]]:
        total_sum: None | np.ndarray[tuple[Ps], np.dtype[F]] = None
        i = 0

        for termv, abs_factorv in zip(termvs, abs_factorvs):
            if total_sum is None:
                if abs_factorv is None:
                    total_sum = _ensure_array(termv, copy=True)
                else:
                    total_sum = _ensure_array(t_stack[i], copy=True)
            elif abs_factorv is None:
                total_sum += termv
            else:
                total_sum += t_stack[i]
            i += abs_factorv is not None

        assert total_sum is not None

        return _broadcast_to(
            _ensure_array(total_sum).reshape((1, *exprv.shape)),
            (t_stack.shape[0], *exprv.shape),
        )

    # handle rounding errors in the total absolute factor early
    tl_stack = guarantee_stacked_arg_within_expr_bounds(
        compute_term_sum,
        _broadcast_to(
            exprv.reshape((1, *exprv.shape)), (tl_stack.shape[0], *exprv.shape)
        ),
        _stack(
            [
                termv
                for termv, abs_factorv in zip(termvs, abs_factorvs)
                if abs_factorv is not None
            ]
        ),
        tl_stack,
        _broadcast_to(
            expr_lower.reshape((1, *exprv.shape)),
            (tl_stack.shape[0], *exprv.shape),
        ),
        _broadcast_to(
            expr_upper.reshape((1, *exprv.shape)),
            (tl_stack.shape[0], *exprv.shape),
        ),
    )
    tu_stack = guarantee_stacked_arg_within_expr_bounds(
        compute_term_sum,
        _broadcast_to(
            exprv.reshape((1, *exprv.shape)), (tu_stack.shape[0], *exprv.shape)
        ),
        _stack(
            [
                termv
                for termv, abs_factorv in zip(termvs, abs_factorvs)
                if abs_factorv is not None
            ]
        ),
        tu_stack,
        _broadcast_to(
            expr_lower.reshape((1, *exprv.shape)),
            (tu_stack.shape[0], *exprv.shape),
        ),
        _broadcast_to(
            expr_upper.reshape((1, *exprv.shape)),
            (tu_stack.shape[0], *exprv.shape),
        ),
    )

    xl: np_sndarray[Ps, Ns, np.dtype[F]]
    xu: np_sndarray[Ps, Ns, np.dtype[F]]
    Xs_lower_: None | np_sndarray[Ps, Ns, np.dtype[F]] = None
    Xs_upper_: None | np_sndarray[Ps, Ns, np.dtype[F]] = None
    i = 0
    for term, abs_factorv in zip(left_associative_sum, abs_factorvs):
        if abs_factorv is None:
            continue

        # recurse into the terms with a weighted bound
        xl, xu = term.compute_data_bounds(
            tl_stack[i],
            tu_stack[i],
            Xs,
            late_bound,
        )

        # combine the inner data bounds
        if Xs_lower_ is None:
            Xs_lower_ = xl
        else:
            Xs_lower_ = _maximum_zero_sign_sensitive(Xs_lower_, xl)
        if Xs_upper_ is None:
            Xs_upper_ = xu
        else:
            Xs_upper_ = _minimum_zero_sign_sensitive(Xs_upper_, xu)

        i += 1

    assert Xs_lower_ is not None
    assert Xs_upper_ is not None
    Xs_lower: np_sndarray[Ps, Ns, np.dtype[F]] = Xs_lower_
    Xs_upper: np_sndarray[Ps, Ns, np.dtype[F]] = Xs_upper_

    Xs_lower = _minimum_zero_sign_sensitive(Xs_lower, Xs)
    Xs_upper = _maximum_zero_sign_sensitive(Xs_upper, Xs)

    return Xs_lower, Xs_upper


def as_left_associative_sum(
    expr: ScalarAdd | ScalarSubtract | ScalarLeftAssociativeSum,
) -> tuple[AnyExpr, ...]:
    terms_rev: list[AnyExpr] = []

    while True:
        # a left associative sum ((a + b) + c) + d
        # has the reverse stack d, c, b, a
        if isinstance(expr, ScalarLeftAssociativeSum):
            terms_rev.extend(expr._ds[::-1])
            terms_rev.append(expr._c)

        # rewrite ( a - b ) as ( a + (-b) ), which is bitwsie equivalent for
        #  floating-point numbers
        terms_rev.append(
            ScalarNegate(expr._b) if isinstance(expr, ScalarSubtract) else expr._b
        )

        if isinstance(expr._a, ScalarAdd | ScalarSubtract | ScalarLeftAssociativeSum):
            expr = expr._a
        else:
            terms_rev.append(expr._a)
            break

    return tuple(terms_rev[::-1])


def get_expr_left_associative_abs_factor_approximate(expr: AnyExpr) -> None | AnyExpr:
    from .divmul import ScalarDivide, ScalarMultiply  # noqa: PLC0415

    if not expr.has_data:
        return None

    if not isinstance(expr, ScalarMultiply | ScalarDivide):
        return Number.ONE

    factor_stack: list[tuple[AnyExpr, type[ScalarMultiply] | type[ScalarDivide]]] = []

    while True:
        factor_stack.append((expr._b, type(expr)))

        if isinstance(expr._a, ScalarMultiply | ScalarDivide):
            expr = expr._a
        else:
            factor_stack.append(
                (Number.ONE if expr._a.has_data else expr._a, ScalarMultiply)
            )
            break

    while len(factor_stack) > 1:
        (a, _), (b, ty) = factor_stack.pop(), factor_stack.pop()
        factor_stack.append(
            (
                ty(a, Number.ONE if b.has_data else b),
                ScalarMultiply,
            )
        )

    [(factor, _)] = factor_stack

    return ScalarAbs(factor)
