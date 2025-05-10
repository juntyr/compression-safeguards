"""
Quantity of interest (QOI) absolute error bound safeguard.
"""

__all__ = ["QuantityOfInterestAbsoluteErrorBoundSafeguard"]

import functools

import numpy as np
import sympy as sp

from . import Expr
from ..abc import PointwiseSafeguard, S, T
from ..abs import _compute_safe_eb_diff_interval
from ....cast import (
    as_bits,
    to_float,
    to_finite_float,
    from_total_order,
    to_total_order,
)
from ....intervals import IntervalUnion


class QuantityOfInterestAbsoluteErrorBoundSafeguard(PointwiseSafeguard):
    """
    The `QuantityOfInterestAbsoluteErrorBoundSafeguard` guarantees that the
    pointwise absolute error on a derived quantity of interest (QOI) is less
    than or equal to the provided bound `eb_abs`.

    The quantity of interest is specified as a non-constant expression, in
    string form, on the pointwise value `x`. For example, to bound the error on
    the square of `x`, set `qoi=Expr("x**2")`.

    If the derived quantity of interest for an element evaluates to an infinite
    value, this safeguard guarantees that the quantity of interest on the
    decoded value produces the exact same infinite value. For a NaN quantity of
    interest, this safeguard guarantees that the quantity of interest on the
    decoded value is also NaN, but does not guarantee that it has the same
    bit pattern.

    The qoi expression is written using the following EBNF grammar for `expr`:

    ```ebnf
    expr    =
        literal
      | const
      | var
      | unary
      | binary
    ;

    literal =
        int
      | float
    ;

    int     = ? integer literal ?;
    float   = ? floating point literal ?;

    const   =
        "e"                               (* Euler's number *)
      | "pi"                              (* pi *)
    ;

    var     = "x";                        (* pointwise data value *)

    unary   =
        "(", expr, ")"                    (* parenthesis *)
      | "-", expr                         (* negation *)
      | "sqrt", "(", expr, ")"            (* square root *)
      | "ln", "(", expr, ")"              (* natural logarithm *)
      | "exp", "(", expr, ")"             (* exponential e^x *)
    ;

    binary  =
        expr, "+", expr                   (* addition *)
      | expr, "-", expr                   (* subtraction *)
      | expr, "*", expr                   (* multiplication *)
      | expr, "/", expr                   (* division *)
      | expr, "**", expr                  (* exponentiation *)
      | "log", "(", expr, ",", expr, ")"  (* logarithm log(a, base) *)
    ;
    ```

    The implementation of the absolute error bound on pointwise quantities of
    interest is based on:

    > Pu Jiao, Sheng Di, Hanqi Guo, Kai Zhao, Jiannan Tian, Dingwen Tao, Xin
    Liang, and Franck Cappello. (2022). Toward Quantity-of-Interest Preserving
    Lossy Compression for Scientific Data. *Proceedings of the VLDB Endowment*.
    16, 4 (December 2022), 697-710. Available from:
    <https://doi.org/10.14778/3574245.3574255>.

    Parameters
    ----------
    qoi : Expr
        The non-constant expression for computing the derived quantity of
        interest for a pointwise value `x`.
    eb_abs : int | float
        The non-negative absolute error bound on the quantity of interest that
        is enforced by this safeguard.
    """

    __slots__ = (
        "_qoi",
        "_eb_abs",
        "_qoi_expr",
        "_x",
    )
    _qoi: Expr
    _eb_abs: int | float
    _qoi_expr: sp.Basic
    _x: sp.Symbol

    kind = "qoi_abs"

    def __init__(self, qoi: Expr, eb_abs: int | float):
        assert eb_abs >= 0, "eb_abs must be non-negative"
        assert isinstance(eb_abs, int) or np.isfinite(eb_abs), "eb_abs must be finite"

        self._qoi = qoi
        self._eb_abs = eb_abs

        self._x = sp.Symbol("x", real=True)

        assert len(qoi.strip()) > 0, "qoi expression must not be empty"
        try:
            qoi_expr = sp.parse_expr(
                self._qoi,
                local_dict=dict(x=self._x),
                global_dict=dict(
                    # literals
                    Integer=sp.Integer,
                    Float=sp.Float,
                    Rational=sp.Rational,
                    # constants
                    pi=sp.pi,
                    e=sp.E,
                    # operators
                    sqrt=sp.sqrt,
                    exp=sp.exp,
                    ln=sp.ln,
                    log=sp.log,
                ),
                transformations=(sp.parsing.sympy_parser.auto_number,),
            )
            _canary = str(qoi_expr)
        except Exception as err:
            raise AssertionError(
                f"failed to parse qoi expression {qoi!r}: {err}"
            ) from err
        assert len(qoi_expr.free_symbols) > 0, "qoi expression must not be constant"

        self._qoi_expr = qoi_expr

    @np.errstate(divide="ignore", over="ignore", under="ignore", invalid="ignore")
    def check_pointwise(
        self, data: np.ndarray[S, T], decoded: np.ndarray[S, T]
    ) -> np.ndarray[S, np.dtype[np.bool]]:
        """
        Check which elements in the `decoded` array satisfy the absolute error
        bound for the quantity of interest on the `data`.

        Parameters
        ----------
        data : np.ndarray
            Data to be encoded.
        decoded : np.ndarray
            Decoded data.

        Returns
        -------
        ok : np.ndarray
            Per-element, `True` if the check succeeded for this element.
        """

        data_float: np.ndarray = to_float(data)

        qoi_lambda = sp.lambdify(
            self._x,
            self._qoi_expr,
            modules="numpy",
            printer=_create_sympy_numpy_printer(data_float.dtype),
            docstring_limit=0,
        )

        qoi_data = (qoi_lambda)(data_float)
        qoi_decoded = (qoi_lambda)(to_float(decoded))

        absolute_bound = (
            np.where(
                qoi_data > qoi_decoded,
                qoi_data - qoi_decoded,
                qoi_decoded - qoi_data,
            )
            <= self._eb_abs
        )
        same_bits = as_bits(qoi_data, kind="V") == as_bits(qoi_decoded, kind="V")
        both_nan = np.isnan(qoi_data) & np.isnan(qoi_decoded)

        ok = np.where(
            np.isfinite(qoi_data),
            absolute_bound,
            np.where(
                np.isinf(qoi_data),
                same_bits,
                both_nan,
            ),
        )

        return ok  # type: ignore

    def compute_safe_intervals(
        self, data: np.ndarray[S, T]
    ) -> IntervalUnion[T, int, int]:
        """
        Compute the intervals in which the absolute error bound is upheld with
        respect to the quantity of interest on the `data`.

        Parameters
        ----------
        data : np.ndarray
            Data for which the safe intervals should be computed.

        Returns
        -------
        intervals : IntervalUnion
            Union of intervals in which the absolute error bound is upheld.
        """

        data_float: np.ndarray = to_float(data)

        with np.errstate(
            divide="ignore", over="ignore", under="ignore", invalid="ignore"
        ):
            eb_abs: np.ndarray = to_finite_float(self._eb_abs, data_float.dtype)
        assert eb_abs >= 0

        qoi_lambda = sp.lambdify(
            self._x,
            self._qoi_expr,
            modules="numpy",
            printer=_create_sympy_numpy_printer(data_float.dtype),
            docstring_limit=0,
        )

        # ensure the error bounds are representable in QOI space
        with np.errstate(
            divide="ignore", over="ignore", under="ignore", invalid="ignore"
        ):
            # compute the error-bound adjusted QOIs
            data_qoi = (qoi_lambda)(data_float)
            qoi_lower = data_qoi - eb_abs
            qoi_upper = data_qoi + eb_abs

            # check if they're representable within the error bound
            qoi_lower_outside_eb_abs = (data_qoi - qoi_lower) > eb_abs
            qoi_upper_outside_eb_abs = (qoi_upper - data_qoi) > eb_abs

            # otherwise nudge the error-bound adjusted QOIs
            qoi_lower = from_total_order(
                to_total_order(qoi_lower) + qoi_lower_outside_eb_abs,
                data_float.dtype,
            )
            qoi_upper = from_total_order(
                to_total_order(qoi_upper) - qoi_upper_outside_eb_abs,
                data_float.dtype,
            )

            # compute the adjusted error bound
            eb_abs_lower: np.ndarray = to_finite_float(
                qoi_lower - data_qoi, data_float.dtype
            )
            eb_abs_upper: np.ndarray = to_finite_float(
                qoi_upper - data_qoi, data_float.dtype
            )
            eb_abs_lower = np.nan_to_num(eb_abs_lower, nan=0)
            eb_abs_upper = np.nan_to_num(eb_abs_upper, nan=0)
        assert np.all((eb_abs_lower <= 0) & np.isfinite(eb_abs_lower))
        assert np.all((eb_abs_upper >= 0) & np.isfinite(eb_abs_upper))

        with np.errstate(
            divide="ignore", over="ignore", under="ignore", invalid="ignore"
        ):
            eb_abs_qoi_lower, eb_abs_qoi_upper = _compute_eb_abs_qoi(
                self._qoi_expr,
                sp.Symbol("x", real=True),
                data_float,
                eb_abs_lower,
                eb_abs_upper,
            )
        assert np.all((eb_abs_qoi_lower <= 0) & np.isfinite(eb_abs_qoi_lower))
        assert np.all((eb_abs_qoi_upper >= 0) & np.isfinite(eb_abs_qoi_upper))

        # with np.errstate(
        #     divide="ignore", over="ignore", under="ignore", invalid="ignore"
        # ):
        #     print(
        #         self._eb_abs,
        #         eb_abs_qoi_lower,
        #         eb_abs_qoi_upper,
        #         (qoi_lambda)(data_float + eb_abs_qoi_lower) - (qoi_lambda)(data_float),
        #         (qoi_lambda)(data_float + eb_abs_qoi_upper) - (qoi_lambda)(data_float),
        #         flush=True,
        #     )

        return _compute_safe_eb_diff_interval(
            data,
            data_float,
            eb_abs_qoi_lower,
            eb_abs_qoi_upper,
            equal_nan=True,
        ).into_union()  # type: ignore

    def get_config(self) -> dict:
        """
        Returns the configuration of the safeguard.

        Returns
        -------
        config : dict
            Configuration of the safeguard.
        """

        return dict(kind=type(self).kind, qoi=self._qoi, eb_abs=self._eb_abs)


@np.errstate(divide="ignore", over="ignore", under="ignore", invalid="ignore")
def _compute_eb_abs_qoi(
    expr: sp.Basic,
    x: sp.Symbol,
    xv: np.ndarray,
    tauv_lower: np.ndarray,
    tauv_upper: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    # print(
    #     f"Compute for {expr} with x={xv} for {tauv_lower} <= e <= {tauv_upper}",
    #     flush=True,
    # )

    """
    Inspired by:

    Pu Jiao, Sheng Di, Hanqi Guo, Kai Zhao, Jiannan Tian, Dingwen Tao, Xin
    Liang, and Franck Cappello. (2022). Toward Quantity-of-Interest Preserving
    Lossy Compression for Scientific Data. Proc. VLDB Endow. 16, 4 (December
    2022), 697-710. Available from: https://doi.org/10.14778/3574245.3574255.
    """

    assert len(expr.free_symbols) > 0, "constants have no error bounds"

    # x
    if expr == x:
        return (tauv_lower, tauv_upper)

    # abs(...) is only used internally in exp(ln(abs(...)))
    if expr.func is sp.Abs and len(expr.args) == 1:
        # evaluate arg
        (arg,) = expr.args
        argv = sp.lambdify(
            [x], arg, modules="numpy", printer=_create_sympy_numpy_printer(xv.dtype)
        )(xv)
        # flip the lower/upper error bound if the arg is negative
        tl = np.where(argv < 0, -tauv_upper, tauv_lower)
        tu = np.where(argv < 0, -tauv_lower, tauv_upper)
        return _compute_eb_abs_qoi(arg, x, xv, tl, tu)

    # ln(...)
    # sympy automatically transforms log(..., base) into ln(...)/ln(base)
    if expr.func is sp.log and len(expr.args) == 1:
        # evaluate arg and ln(arg)
        (arg,) = expr.args
        argv = sp.lambdify(
            [x], arg, modules="numpy", printer=_create_sympy_numpy_printer(xv.dtype)
        )(xv)
        exprv = np.log(argv)

        # update the error bounds
        tl = np.where(
            (tauv_lower == 0) | (argv <= 0) | ~np.isfinite(argv),
            0,
            np.exp(exprv + tauv_lower) - argv,
        )
        tl = to_finite_float(tl, xv.dtype)
        tl = np.nan_to_num(tl, nan=0)

        tu = np.where(
            (tauv_upper == 0) | (argv <= 0) | ~np.isfinite(argv),
            0,
            np.exp(exprv + tauv_upper) - argv,
        )
        tu = to_finite_float(tu, xv.dtype)
        tu = np.nan_to_num(tu, nan=0)

        # FIXME: handle rounding better
        while True:
            tauv_lower_outside = (np.log(argv + tl) - exprv) < tauv_lower
            if not np.any(tauv_lower_outside & np.isfinite(exprv)):
                break
            tl *= 0.5
        while True:
            tauv_upper_outside = (np.log(argv + tu) - exprv) > tauv_upper
            if not np.any(tauv_upper_outside & np.isfinite(exprv)):
                break
            tu *= 0.5
        tauv_lower, tauv_upper = tl, tu

        # base case for ln(x)
        if arg == x:
            return tauv_lower, tauv_upper
        # composition using Lemma 3 from Jiao et al.
        return _compute_eb_abs_qoi(arg, x, xv, tauv_lower, tauv_upper)

    # e^(...)
    if expr.func is sp.exp and len(expr.args) == 1:
        # evaluate arg and e^arg
        (arg,) = expr.args
        argv = sp.lambdify(
            [x], arg, modules="numpy", printer=_create_sympy_numpy_printer(xv.dtype)
        )(xv)
        exprv = np.exp(argv)

        # update the error bounds
        # ensure that ln is not passed a negative argument
        tl = np.where(
            (tauv_lower == 0) | ~np.isfinite(argv),
            0,
            np.log(np.maximum(0, exprv + tauv_lower)) - argv,
        )
        tl = to_finite_float(tl, xv.dtype)
        tl = np.nan_to_num(tl, nan=0)

        tu = np.where(
            (tauv_upper == 0) | ~np.isfinite(argv),
            0,
            np.log(np.maximum(0, exprv + tauv_upper)) - argv,
        )
        tu = to_finite_float(tu, xv.dtype)
        tu = np.nan_to_num(tu, nan=0)

        # FIXME: handle rounding better
        while True:
            tauv_lower_outside = (np.exp(argv + tl) - exprv) < tauv_lower
            if not np.any(tauv_lower_outside & np.isfinite(exprv)):
                break
            tl *= 0.5
        while True:
            tauv_upper_outside = (np.exp(argv + tu) - exprv) > tauv_upper
            if not np.any(tauv_upper_outside & np.isfinite(exprv)):
                break
            tu *= 0.5
        tauv_lower, tauv_upper = tl, tu

        # base case for e^x
        if arg == x:
            return tauv_lower, tauv_upper
        # composition using Lemma 3 from Jiao et al.
        return _compute_eb_abs_qoi(arg, x, xv, tauv_lower, tauv_upper)

    # rewrite a ** b as e^(b*ln(abs(a)))
    # this is mathematically incorrect for a < 0 but works for deriving error bounds
    if expr.is_Pow and len(expr.args) == 2:
        a, b = expr.args
        tl, tu = _compute_eb_abs_qoi(
            sp.exp(b * sp.ln(sp.Abs(a)), evaluate=False), x, xv, tauv_lower, tauv_upper
        )

        exprl = sp.lambdify(
            [x], expr, modules="numpy", printer=_create_sympy_numpy_printer(xv.dtype)
        )
        exprv = (exprl)(xv)

        # FIXME: handle rounding better
        while True:
            tauv_lower_outside = ((exprl)(xv + tl) - exprv) < tauv_lower
            if not np.any(tauv_lower_outside & np.isfinite(exprv)):
                break
            tl *= 0.5
        while True:
            tauv_upper_outside = ((exprl)(xv + tu) - exprv) > tauv_upper
            if not np.any(tauv_upper_outside & np.isfinite(exprv)):
                break
            tu *= 0.5

        return tl, tu

    # a_1 * e_1 + ... + a_n * e_n + c (weighted sum)
    # using Corollary 2 and Lemma 4 from Jiao et al.
    if expr.is_Add:
        # find all non-constant terms
        terms = [arg for arg in expr.args if len(arg.free_symbols) > 0]

        factors = []
        for i, term in enumerate(terms):
            # extract the weighting factor of the term
            if term.is_Mul:
                factors.append(
                    sp.lambdify(
                        [],
                        sp.Mul(
                            *[arg for arg in term.args if len(arg.free_symbols) == 0]  # type: ignore
                        ),
                        modules="numpy",
                        printer=_create_sympy_numpy_printer(xv.dtype),
                    )()
                )
                terms[i] = sp.Mul(
                    *[arg for arg in term.args if len(arg.free_symbols) > 0]  # type: ignore
                )
            else:
                factors.append(1)
        total_abs_factor = np.sum(np.abs(factors))

        ebs_lower, ebs_upper = None, None
        for term, factor in zip(terms, factors):
            # FIXME: ensure we round towards zero here
            tl = to_finite_float(
                (-tauv_upper if factor < 0 else tauv_lower) / total_abs_factor, xv.dtype
            )
            tu = to_finite_float(
                (-tauv_lower if factor < 0 else tauv_upper) / total_abs_factor, xv.dtype
            )
            tl = np.nan_to_num(tl, nan=0)
            tu = np.nan_to_num(tu, nan=0)

            # recurse into the terms with a weighted error bound
            eb_lower, eb_upper = _compute_eb_abs_qoi(
                term,
                x,
                xv,
                tl,
                tu,
            )
            # combine the inner error bounds
            if ebs_lower is None:
                ebs_lower = eb_lower
            else:
                ebs_lower = np.maximum(ebs_lower, eb_lower)
            if ebs_upper is None:
                ebs_upper = eb_upper
            else:
                ebs_upper = np.minimum(ebs_upper, eb_upper)

        return ebs_lower, ebs_upper  # type: ignore

    # rewrite f * e_1 * ... * e_n (product) as f * e^(ln(abs(e_1) + ... + ln(abs(e_n)))
    # this is mathematically incorrect if the product is negative,
    #  but works for deriving error bounds
    if expr.is_Mul:
        # extract the constant factor and reduce tauv
        factor = sp.lambdify(
            [],
            sp.Mul(*[arg for arg in expr.args if len(arg.free_symbols) == 0]),  # type: ignore
            modules="numpy",
            printer=_create_sympy_numpy_printer(xv.dtype),
        )()  # type: ignore

        # flip the lower/upper error bound if the factor is negative
        # FIXME: ensure we round towards zero here
        tl = to_finite_float(
            (tauv_upper if factor < 0 else tauv_lower) / factor, xv.dtype
        )
        tu = to_finite_float(
            (tauv_lower if factor < 0 else tauv_upper) / factor, xv.dtype
        )
        tl = np.nan_to_num(tl, nan=0)
        tu = np.nan_to_num(tu, nan=0)

        # find all non-constant terms
        terms = [arg for arg in expr.args if len(arg.free_symbols) > 0]

        if len(terms) == 1:
            return _compute_eb_abs_qoi(terms[0], x, xv, tl, tu)

        return _compute_eb_abs_qoi(
            sp.exp(sp.Add(*[sp.log(sp.Abs(term)) for term in terms]), evaluate=False),
            x,
            xv,
            tl,
            tu,
        )

    raise ValueError(f"unsupported expression kind {expr} (= {sp.srepr(expr)} =)")


@functools.cache
def _create_sympy_numpy_printer(dtype: np.dtype):
    class NumPyDtypePrinter(sp.printing.numpy.NumPyPrinter):
        def __init__(self, settings=None):
            self._dtype = dtype.name
            if settings is None:
                settings = dict()
            settings["precision"] = np.finfo(dtype).precision + 1
            super().__init__(settings)

        def _print_Integer(self, expr):
            return str(expr.p)

        def _print_Rational(self, expr):
            if expr.q == 1:
                return str(expr.p)
            else:
                return f"{self._dtype}({expr.p}) / {self._dtype}({expr.q})"

        def _print_Float(self, expr):
            s = super()._print_Float(expr)
            return f"{self._dtype}({s!r})"

    return NumPyDtypePrinter
