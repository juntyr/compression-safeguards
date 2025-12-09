import itertools
from contextlib import contextmanager

from sly import Parser

from ...utils.bindings import Parameter
from ...utils.error import ctx
from .expr.abc import AnyExpr
from .expr.abs import ScalarAbs
from .expr.addsub import ScalarAdd, ScalarLeftAssociativeSum, ScalarSubtract
from .expr.array import Array
from .expr.classification import ScalarIsFinite, ScalarIsInf, ScalarIsNaN
from .expr.combinators import ScalarAll, ScalarAny, ScalarNot
from .expr.comparison import (
    ScalarEqual,
    ScalarGreater,
    ScalarGreaterEqual,
    ScalarLess,
    ScalarLessEqual,
    ScalarNotEqual,
)
from .expr.data import Data, LateBoundConstant
from .expr.divmul import ScalarDivide, ScalarMultiply
from .expr.finite_difference import (
    FiniteDifference,
    ScalarSymmetricModulo,
    finite_difference_coefficients,
    finite_difference_offsets,
)
from .expr.group import Group
from .expr.hyperbolic import (
    ScalarAcosh,
    ScalarAsinh,
    ScalarAtanh,
    ScalarCosh,
    ScalarSinh,
    ScalarTanh,
)
from .expr.literal import Euler, Number, Pi
from .expr.logexp import Exponential, Logarithm, ScalarExp, ScalarLog, ScalarLogWithBase
from .expr.neg import ScalarNegate
from .expr.power import ScalarPower
from .expr.reciprocal import ScalarReciprocal
from .expr.round import ScalarCeil, ScalarFloor, ScalarRoundTiesEven, ScalarTrunc
from .expr.sign import ScalarSign
from .expr.square import ScalarSqrt, ScalarSquare
from .expr.trigonometric import (
    ScalarAcos,
    ScalarAsin,
    ScalarAtan,
    ScalarCos,
    ScalarSin,
    ScalarTan,
)
from .expr.where import ScalarWhere
from .lexer import QoILexer


class QoIParser(Parser):
    __slots__: tuple[str, ...] = ("_x", "_X", "_I", "_vars", "_text")
    _x: Data
    _X: None | Array
    _I: None | tuple[int, ...]
    _vars: dict[Parameter, AnyExpr]
    _text: str

    def __init__(
        self,
        *,
        x: Data,
        X: None | Array,
        I: None | tuple[int, ...],  # noqa: E741
    ):
        self._x = x
        self._X = X
        self._I = I
        self._vars = dict()

    def parse2(self, text: str, tokens):
        self._text = text
        tokens, tokens2 = itertools.tee(tokens)

        if next(tokens2, None) is None:
            raise (
                SyntaxError("expression must not be empty", ("<qoi>", None, None, None))
                | ctx
            )

        return super().parse(tokens)

    tokens = QoILexer.tokens

    # === operator precedence and associativity ===
    precedence = (
        # lowest precedence: comparisons
        ("nonassoc", LESS, LESS_EQUAL, EQUAL, NOT_EQUAL, GREATER_EQUAL, GREATER),  # type: ignore[name-defined]  # noqa: F821
        ("left", PLUS, MINUS),  # type: ignore[name-defined]  # noqa: F821
        ("left", TIMES, DIVIDE),  # type: ignore[name-defined]  # noqa: F821
        ("right", UPLUS, UMINUS),  # type: ignore[name-defined]  # noqa: F821
        ("right", POWER),  # type: ignore[name-defined]  # noqa: F821
        ("left", INDEX, TRANSPOSE),  # type: ignore[name-defined]  # noqa: F821
        # highest precedence: array indexing and transpose
        # help array indexing by giving the opening `[` an even higher precedence
        ("left", LBRACK),  # type: ignore[name-defined]  # noqa: F821
    )

    # === grammar rules ===

    # top-level: qoi := expr | { assign } return expr;
    @_("expr")  # type: ignore[name-defined, no-redef]  # noqa: F821
    def qoi(self, p):  # noqa: F811
        self.assert_or_error(
            not isinstance(p.expr, Array),
            p,
            lambda: f"expression must be a scalar but is an array expression of shape {p.expr.shape}",
        )
        self.assert_or_error(p.expr.has_data, p, "expression must not be constant")
        return p.expr

    @_("many_assign return_expr")  # type: ignore[name-defined, no-redef]  # noqa: F821
    def qoi(self, p):  # noqa: F811
        return p.return_expr

    @_("RETURN expr SEMI")  # type: ignore[name-defined]  # noqa: F821
    def return_expr(self, p):
        self.assert_or_error(
            not isinstance(p.expr, Array),
            p,
            lambda: f"return expression must be a scalar but is an array expression of shape {p.expr.shape}",
        )
        self.assert_or_error(
            p.expr.has_data, p, "return expression must not be constant"
        )
        return p.expr

    # variable assignment: assign := V["id"] = expr;
    @_("VS LBRACK quotedparameter RBRACK ASSIGN expr SEMI")  # type: ignore[name-defined, no-redef]  # noqa: F821
    def assign(self, p):  # noqa: F811
        self.assert_or_error(
            self._X is None, p, "stencil QoI variables use upper-case `V`"
        )
        self.assert_or_error(
            not p.quotedparameter.is_builtin,
            p,
            "variable name must be built-in (start with `$`)",
        )
        self.assert_or_error(
            p.quotedparameter not in self._vars,
            p,
            f'cannot override already-defined variable v["{p.quotedparameter}"]',
        )
        self._vars[p.quotedparameter] = Array.map(Group, p.expr)

    @_("VA LBRACK quotedparameter RBRACK ASSIGN expr SEMI")  # type: ignore[name-defined, no-redef]  # noqa: F821
    def assign(self, p):  # noqa: F811
        self.assert_or_error(
            self._X is not None, p, "pointwise QoI variables use lower-case `v`"
        )
        self.assert_or_error(
            not p.quotedparameter.is_builtin,
            p,
            "variable name must be built-in (start with `$`)",
        )
        self.assert_or_error(
            p.quotedparameter not in self._vars,
            p,
            f'cannot override already-defined variable V["{p.quotedparameter}"]',
        )
        self._vars[p.quotedparameter] = Array.map(Group, p.expr)

    @_("ID ASSIGN expr SEMI")  # type: ignore[name-defined, no-redef]  # noqa: F821
    def assign(self, p):  # noqa: F811
        self.raise_error(
            p,
            f'cannot assign to identifier `{p.ID}`, assign to a variable {"v" if self._X is None else "V"}["{p.ID}"] instead',
        )

    @_("assign many_assign")  # type: ignore[name-defined, no-redef]  # noqa: F821
    def many_assign(self, p):  # noqa: F811
        pass

    @_("empty")  # type: ignore[name-defined, no-redef]  # noqa: F821
    def many_assign(self, p):  # noqa: F811
        pass

    # integer literal (non-expression)
    @_("INTEGER")  # type: ignore[name-defined, no-redef]  # noqa: F821
    def integer(self, p):  # noqa: F811
        return p.INTEGER

    @_("PLUS INTEGER")  # type: ignore[name-defined, no-redef]  # noqa: F821
    def integer(self, p):  # noqa: F811
        return p.INTEGER

    @_("MINUS INTEGER")  # type: ignore[name-defined, no-redef]  # noqa: F821
    def integer(self, p):  # noqa: F811
        return -p.INTEGER

    # expressions

    # integer and floating-point literals
    @_("INTEGER")  # type: ignore[name-defined, no-redef]  # noqa: F821
    def expr(self, p):  # noqa: F811
        return Number.from_symbolic_int(p.INTEGER)

    @_("FLOAT")  # type: ignore[name-defined, no-redef]  # noqa: F821
    def expr(self, p):  # noqa: F811
        return Number(p.FLOAT)

    @_("INF")  # type: ignore[name-defined, no-redef]  # noqa: F821
    def expr(self, p):  # noqa: F811
        return Number("inf")

    @_("NAN")  # type: ignore[name-defined, no-redef]  # noqa: F821
    def expr(self, p):  # noqa: F811
        return Number("nan")

    # array literal
    @_("LBRACK expr many_comma_expr RBRACK")  # type: ignore[name-defined, no-redef]  # noqa: F821
    def expr(self, p):  # noqa: F811
        with self.with_error_context(
            p, lambda err: f"invalid array literal: {err}", exception=ValueError
        ):
            return Array(*([p.expr] + p.many_comma_expr))

    @_("LBRACK RBRACK")  # type: ignore[name-defined, no-redef]  # noqa: F821
    def expr(self, p):  # noqa: F811
        self.raise_error(p, "invalid empty array literal")

    @_("comma_expr many_comma_expr")  # type: ignore[name-defined]  # noqa: F821
    def many_comma_expr(self, p):
        return [p.comma_expr] + p.many_comma_expr

    @_("COMMA")  # type: ignore[name-defined, no-redef]  # noqa: F821
    def many_comma_expr(self, p):  # noqa: F811
        return []

    @_("empty")  # type: ignore[name-defined, no-redef]  # noqa: F821
    def many_comma_expr(self, p):  # noqa: F811
        return []

    @_("COMMA expr")  # type: ignore[name-defined]  # noqa: F821
    def comma_expr(self, p):
        return p.expr

    # unary operators (positive, negative):
    #  expr := OP expr
    @_("PLUS expr %prec UPLUS")  # type: ignore[name-defined, no-redef]  # noqa: F821
    def expr(self, p):  # noqa: F811
        return p.expr

    @_("MINUS expr %prec UMINUS")  # type: ignore[name-defined, no-redef]  # noqa: F821
    def expr(self, p):  # noqa: F811
        return Array.map(ScalarNegate, p.expr)

    # binary operators (add, subtract, multiply, divide, power):
    #  expr := expr OP expr
    @_("expr PLUS expr")  # type: ignore[name-defined, no-redef]  # noqa: F821
    def expr(self, p):  # noqa: F811
        with self.with_error_context(p, lambda err: f"{err}", exception=ValueError):
            return Array.map(ScalarAdd, p.expr0, p.expr1)

    @_("expr MINUS expr")  # type: ignore[name-defined, no-redef]  # noqa: F821
    def expr(self, p):  # noqa: F811
        with self.with_error_context(p, lambda err: f"{err}", exception=ValueError):
            return Array.map(ScalarSubtract, p.expr0, p.expr1)

    @_("expr TIMES expr")  # type: ignore[name-defined, no-redef]  # noqa: F821
    def expr(self, p):  # noqa: F811
        with self.with_error_context(p, lambda err: f"{err}", exception=ValueError):
            return Array.map(ScalarMultiply, p.expr0, p.expr1)

    @_("expr DIVIDE expr")  # type: ignore[name-defined, no-redef]  # noqa: F821
    def expr(self, p):  # noqa: F811
        with self.with_error_context(p, lambda err: f"{err}", exception=ValueError):
            return Array.map(ScalarDivide, p.expr0, p.expr1)

    @_("expr POWER expr")  # type: ignore[name-defined, no-redef]  # noqa: F821
    def expr(self, p):  # noqa: F811
        with self.with_error_context(p, lambda err: f"{err}", exception=ValueError):
            return Array.map(ScalarPower, p.expr0, p.expr1)

    # binary comparison operators
    #  expr := expr OP expr
    @_("expr LESS_EQUAL expr")  # type: ignore[name-defined, no-redef]  # noqa: F821
    def expr(self, p):  # noqa: F811
        with self.with_error_context(p, lambda err: f"{err}", exception=ValueError):
            return Array.map(ScalarLessEqual, p.expr0, p.expr1)

    @_("expr LESS expr")  # type: ignore[name-defined, no-redef]  # noqa: F821
    def expr(self, p):  # noqa: F811
        with self.with_error_context(p, lambda err: f"{err}", exception=ValueError):
            return Array.map(ScalarLess, p.expr0, p.expr1)

    @_("expr EQUAL expr")  # type: ignore[name-defined, no-redef]  # noqa: F821
    def expr(self, p):  # noqa: F811
        with self.with_error_context(p, lambda err: f"{err}", exception=ValueError):
            return Array.map(ScalarEqual, p.expr0, p.expr1)

    @_("expr NOT_EQUAL expr")  # type: ignore[name-defined, no-redef]  # noqa: F821
    def expr(self, p):  # noqa: F811
        with self.with_error_context(p, lambda err: f"{err}", exception=ValueError):
            return Array.map(ScalarNotEqual, p.expr0, p.expr1)

    @_("expr GREATER_EQUAL expr")  # type: ignore[name-defined, no-redef]  # noqa: F821
    def expr(self, p):  # noqa: F811
        with self.with_error_context(p, lambda err: f"{err}", exception=ValueError):
            return Array.map(ScalarGreaterEqual, p.expr0, p.expr1)

    @_("expr GREATER expr")  # type: ignore[name-defined, no-redef]  # noqa: F821
    def expr(self, p):  # noqa: F811
        with self.with_error_context(p, lambda err: f"{err}", exception=ValueError):
            return Array.map(ScalarGreater, p.expr0, p.expr1)

    # array transpose: expr := expr.T
    @_("expr TRANSPOSE")  # type: ignore[name-defined, no-redef]  # noqa: F821
    def expr(self, p):  # noqa: F811
        self.assert_or_error(
            isinstance(p.expr, Array), p, "cannot transpose scalar non-array expression"
        )
        return p.expr.transpose()

    # group in parentheses
    @_("LPAREN expr RPAREN")  # type: ignore[name-defined, no-redef]  # noqa: F821
    def expr(self, p):  # noqa: F811
        return Array.map(Group, p.expr)

    # optional trailing comma separator
    @_("COMMA")  # type: ignore[name-defined, no-redef]  # noqa: F821
    def maybe_comma(self, p):  # noqa: F811
        pass

    @_("empty")  # type: ignore[name-defined, no-redef]  # noqa: F821
    def maybe_comma(self, p):  # noqa: F811
        pass

    # constants: expr := e | pi
    @_("EULER")  # type: ignore[name-defined, no-redef]  # noqa: F821
    def expr(self, p):  # noqa: F811
        return Euler()

    @_("PI")  # type: ignore[name-defined, no-redef]  # noqa: F821
    def expr(self, p):  # noqa: F811
        return Pi()

    # data, late-bound constants, variables
    @_("XS")  # type: ignore[name-defined, no-redef]  # noqa: F821
    def expr(self, p):  # noqa: F811
        return self._x

    @_("XA")  # type: ignore[name-defined, no-redef]  # noqa: F821
    def expr(self, p):  # noqa: F811
        self.assert_or_error(
            self._X is not None,
            p,
            "data neighbourhood `X` is not available in pointwise QoIs, use pointwise `x` instead",
        )
        return self._X

    @_("CS LBRACK quotedparameter RBRACK")  # type: ignore[name-defined, no-redef]  # noqa: F821
    def expr(self, p):  # noqa: F811
        return LateBoundConstant.like(p.quotedparameter, self._x)

    @_("CA LBRACK quotedparameter RBRACK")  # type: ignore[name-defined, no-redef]  # noqa: F821
    def expr(self, p):  # noqa: F811
        self.assert_or_error(
            self._X is not None,
            p,
            "late-bound constant neighbourhood `C` is not available in pointwise QoIs, use pointwise `c` instead",
        )
        return Array.map(
            lambda e: LateBoundConstant.like(p.quotedparameter, e),
            self._X,
        )

    @_("VS LBRACK quotedparameter RBRACK")  # type: ignore[name-defined, no-redef]  # noqa: F821
    def expr(self, p):  # noqa: F811
        self.assert_or_error(
            self._X is None, p, "stencil QoI variables use upper-case `V`"
        )
        self.assert_or_error(
            not p.quotedparameter.is_builtin,
            p,
            "variable name must not be built-in (start with `$`)",
        )
        self.assert_or_error(
            p.quotedparameter in self._vars,
            p,
            f'undefined variable v["{p.quotedparameter}"]',
        )
        return self._vars[p.quotedparameter]

    @_("VA LBRACK quotedparameter RBRACK")  # type: ignore[name-defined, no-redef]  # noqa: F821
    def expr(self, p):  # noqa: F811
        self.assert_or_error(
            self._X is not None, p, "pointwise QoI variables use lower-case `v`"
        )
        self.assert_or_error(
            not p.quotedparameter.is_builtin,
            p,
            "variable name must not be built-in (start with `$`)",
        )
        self.assert_or_error(
            p.quotedparameter in self._vars,
            p,
            f'undefined variable V["{p.quotedparameter}"]',
        )
        return self._vars[p.quotedparameter]

    @_("STRING")  # type: ignore[name-defined, no-redef]  # noqa: F821
    def quotedparameter(self, p):  # noqa: F811
        with self.with_error_context(
            p, f'invalid quoted parameter "{p.STRING}": must be a valid identifier'
        ):
            return Parameter(p.STRING)

    # array indexing
    @_("expr LBRACK IDX RBRACK %prec INDEX")  # type: ignore[name-defined, no-redef]  # noqa: F821
    def expr(self, p):  # noqa: F811
        self.assert_or_error(
            self._I is not None, p, "index `I` is not available in pointwise QoIs"
        )
        self.assert_or_error(
            isinstance(p.expr, Array), p, "cannot index scalar non-array expression"
        )
        with self.with_error_context(p, lambda err: f"{err}", exception=IndexError):
            return p.expr.index(self._I)

    @_("expr LBRACK index_ many_comma_index RBRACK %prec INDEX")  # type: ignore[name-defined, no-redef]  # noqa: F821
    def expr(self, p):  # noqa: F811
        self.assert_or_error(
            isinstance(p.expr, Array), p, "cannot index scalar non-array expression"
        )
        with self.with_error_context(p, lambda err: f"{err}", exception=IndexError):
            return p.expr.index(tuple([p.index_] + p.many_comma_index))

    @_("comma_index many_comma_index")  # type: ignore[name-defined]  # noqa: F821
    def many_comma_index(self, p):
        return [p.comma_index] + p.many_comma_index

    @_("COMMA")  # type: ignore[name-defined, no-redef]  # noqa: F821
    def many_comma_index(self, p):  # noqa: F811
        return []

    @_("empty")  # type: ignore[name-defined, no-redef]  # noqa: F821
    def many_comma_index(self, p):  # noqa: F811
        return []

    @_("COMMA index_")  # type: ignore[name-defined]  # noqa: F821
    def comma_index(self, p):
        return p.index_

    @_("integer_expr")  # type: ignore[name-defined]  # noqa: F821
    def index_(self, p):
        return p.integer_expr

    @_("maybe_integer_expr COLON maybe_integer_expr")  # type: ignore[name-defined, no-redef]  # noqa: F821
    def index_(self, p):  # noqa: F811
        return slice(p.maybe_integer_expr0, p.maybe_integer_expr1, None)

    @_("maybe_integer_expr COLON maybe_integer_expr COLON maybe_integer_expr")  # type: ignore[name-defined, no-redef]  # noqa: F821
    def index_(self, p):  # noqa: F811
        return slice(
            p.maybe_integer_expr0, p.maybe_integer_expr1, p.maybe_integer_expr2
        )

    @_("integer_expr")  # type: ignore[name-defined]  # noqa: F821
    def maybe_integer_expr(self, p):
        return p.integer_expr

    @_("empty")  # type: ignore[name-defined, no-redef]  # noqa: F821
    def maybe_integer_expr(self, p):  # noqa: F811
        return None

    @_("expr")  # type: ignore[name-defined]  # noqa: F821
    def integer_expr(self, p):
        self.assert_or_error(
            isinstance(p.expr, Number) and p.expr.as_int() is not None,
            p,
            "cannot index by non-integer expression",
        )
        return p.expr.as_int()

    @_("IDX LBRACK index_ many_comma_index RBRACK")  # type: ignore[name-defined, no-redef]  # noqa: F821
    def expr(self, p):  # noqa: F811
        self.assert_or_error(
            self._I is not None, p, "index `I` is not available in pointwise QoIs"
        )
        idx = Array(*tuple(Number.from_symbolic_int(i) for i in self._I))
        with self.with_error_context(
            p,
            lambda err: f"{err} for `I`, the 1D per-axis stencil centre index array",
            exception=IndexError,
        ):
            return idx.index(tuple([p.index_] + p.many_comma_index))

    # functions

    # logarithms and exponentials
    @_("LN LPAREN expr maybe_comma RPAREN")  # type: ignore[name-defined, no-redef]  # noqa: F821
    def expr(self, p):  # noqa: F811
        return Array.map(lambda e: ScalarLog(Logarithm.ln, e), p.expr)

    @_("LOG2 LPAREN expr maybe_comma RPAREN")  # type: ignore[name-defined, no-redef]  # noqa: F821
    def expr(self, p):  # noqa: F811
        return Array.map(lambda e: ScalarLog(Logarithm.log2, e), p.expr)

    @_("LOG10 LPAREN expr maybe_comma RPAREN")  # type: ignore[name-defined, no-redef]  # noqa: F821
    def expr(self, p):  # noqa: F811
        return Array.map(lambda e: ScalarLog(Logarithm.log10, e), p.expr)

    @_("LOG LPAREN expr COMMA BASE ASSIGN expr maybe_comma RPAREN")  # type: ignore[name-defined, no-redef]  # noqa: F821
    def expr(self, p):  # noqa: F811
        with self.with_error_context(p, lambda err: f"{err}", exception=ValueError):
            return Array.map(ScalarLogWithBase, p.expr0, p.expr1)

    @_("EXP LPAREN expr maybe_comma RPAREN")  # type: ignore[name-defined, no-redef]  # noqa: F821
    def expr(self, p):  # noqa: F811
        return Array.map(lambda e: ScalarExp(Exponential.exp, e), p.expr)

    @_("EXP2 LPAREN expr maybe_comma RPAREN")  # type: ignore[name-defined, no-redef]  # noqa: F821
    def expr(self, p):  # noqa: F811
        return Array.map(lambda e: ScalarExp(Exponential.exp2, e), p.expr)

    @_("EXP10 LPAREN expr maybe_comma RPAREN")  # type: ignore[name-defined, no-redef]  # noqa: F821
    def expr(self, p):  # noqa: F811
        return Array.map(lambda e: ScalarExp(Exponential.exp10, e), p.expr)

    # exponentiation
    @_("SQRT LPAREN expr maybe_comma RPAREN")  # type: ignore[name-defined, no-redef]  # noqa: F821
    def expr(self, p):  # noqa: F811
        return Array.map(ScalarSqrt, p.expr)

    @_("SQUARE LPAREN expr maybe_comma RPAREN")  # type: ignore[name-defined, no-redef]  # noqa: F821
    def expr(self, p):  # noqa: F811
        return Array.map(ScalarSquare, p.expr)

    @_("RECIPROCAL LPAREN expr maybe_comma RPAREN")  # type: ignore[name-defined, no-redef]  # noqa: F821
    def expr(self, p):  # noqa: F811
        return Array.map(ScalarReciprocal, p.expr)

    # absolute value
    @_("ABS LPAREN expr maybe_comma RPAREN")  # type: ignore[name-defined, no-redef]  # noqa: F821
    def expr(self, p):  # noqa: F811
        return Array.map(ScalarAbs, p.expr)

    # sign and rounding
    @_("SIGN LPAREN expr maybe_comma RPAREN")  # type: ignore[name-defined, no-redef]  # noqa: F821
    def expr(self, p):  # noqa: F811
        return Array.map(ScalarSign, p.expr)

    @_("FLOOR LPAREN expr maybe_comma RPAREN")  # type: ignore[name-defined, no-redef]  # noqa: F821
    def expr(self, p):  # noqa: F811
        return Array.map(ScalarFloor, p.expr)

    @_("CEIL LPAREN expr maybe_comma RPAREN")  # type: ignore[name-defined, no-redef]  # noqa: F821
    def expr(self, p):  # noqa: F811
        return Array.map(ScalarCeil, p.expr)

    @_("TRUNC LPAREN expr maybe_comma RPAREN")  # type: ignore[name-defined, no-redef]  # noqa: F821
    def expr(self, p):  # noqa: F811
        return Array.map(ScalarTrunc, p.expr)

    @_("ROUND_TIES_EVEN LPAREN expr maybe_comma RPAREN")  # type: ignore[name-defined, no-redef]  # noqa: F821
    def expr(self, p):  # noqa: F811
        return Array.map(ScalarRoundTiesEven, p.expr)

    # trigonometric
    @_("SIN LPAREN expr maybe_comma RPAREN")  # type: ignore[name-defined, no-redef]  # noqa: F821
    def expr(self, p):  # noqa: F811
        return Array.map(ScalarSin, p.expr)

    @_("COS LPAREN expr maybe_comma RPAREN")  # type: ignore[name-defined, no-redef]  # noqa: F821
    def expr(self, p):  # noqa: F811
        return Array.map(ScalarCos, p.expr)

    @_("TAN LPAREN expr maybe_comma RPAREN")  # type: ignore[name-defined, no-redef]  # noqa: F821
    def expr(self, p):  # noqa: F811
        return Array.map(ScalarTan, p.expr)

    @_("ASIN LPAREN expr maybe_comma RPAREN")  # type: ignore[name-defined, no-redef]  # noqa: F821
    def expr(self, p):  # noqa: F811
        return Array.map(ScalarAsin, p.expr)

    @_("ACOS LPAREN expr maybe_comma RPAREN")  # type: ignore[name-defined, no-redef]  # noqa: F821
    def expr(self, p):  # noqa: F811
        return Array.map(ScalarAcos, p.expr)

    @_("ATAN LPAREN expr maybe_comma RPAREN")  # type: ignore[name-defined, no-redef]  # noqa: F821
    def expr(self, p):  # noqa: F811
        return Array.map(ScalarAtan, p.expr)

    # hyperbolic
    @_("SINH LPAREN expr maybe_comma RPAREN")  # type: ignore[name-defined, no-redef]  # noqa: F821
    def expr(self, p):  # noqa: F811
        return Array.map(ScalarSinh, p.expr)

    @_("COSH LPAREN expr maybe_comma RPAREN")  # type: ignore[name-defined, no-redef]  # noqa: F821
    def expr(self, p):  # noqa: F811
        return Array.map(ScalarCosh, p.expr)

    @_("TANH LPAREN expr maybe_comma RPAREN")  # type: ignore[name-defined, no-redef]  # noqa: F821
    def expr(self, p):  # noqa: F811
        return Array.map(ScalarTanh, p.expr)

    @_("ASINH LPAREN expr maybe_comma RPAREN")  # type: ignore[name-defined, no-redef]  # noqa: F821
    def expr(self, p):  # noqa: F811
        return Array.map(ScalarAsinh, p.expr)

    @_("ACOSH LPAREN expr maybe_comma RPAREN")  # type: ignore[name-defined, no-redef]  # noqa: F821
    def expr(self, p):  # noqa: F811
        return Array.map(ScalarAcosh, p.expr)

    @_("ATANH LPAREN expr maybe_comma RPAREN")  # type: ignore[name-defined, no-redef]  # noqa: F821
    def expr(self, p):  # noqa: F811
        return Array.map(ScalarAtanh, p.expr)

    # classification
    @_("ISFINITE LPAREN expr maybe_comma RPAREN")  # type: ignore[name-defined, no-redef]  # noqa: F821
    def expr(self, p):  # noqa: F811
        return Array.map(ScalarIsFinite, p.expr)

    @_("ISINF LPAREN expr maybe_comma RPAREN")  # type: ignore[name-defined, no-redef]  # noqa: F821
    def expr(self, p):  # noqa: F811
        return Array.map(ScalarIsInf, p.expr)

    @_("ISNAN LPAREN expr maybe_comma RPAREN")  # type: ignore[name-defined, no-redef]  # noqa: F821
    def expr(self, p):  # noqa: F811
        return Array.map(ScalarIsNaN, p.expr)

    # combinators
    @_("NOT LPAREN expr maybe_comma RPAREN")  # type: ignore[name-defined, no-redef]  # noqa: F821
    def expr(self, p):  # noqa: F811
        return Array.map(ScalarNot, p.expr)

    @_(  # type: ignore[name-defined, no-redef]  # noqa: F821
        "ALL LPAREN expr maybe_comma RPAREN"
    )
    def expr(self, p):  # noqa: F811
        expr = p.expr
        self.assert_or_error(
            isinstance(expr, Array) and expr.size >= 2,
            p,
            "`all` expr must be an array expression with at least two elements",
        )
        a, b, *cs = expr.flatlist()

        # must not fail since flat Array elements cannot be arrays themselves
        assert not isinstance(a, Array)
        assert not isinstance(b, Array)
        assert not any(isinstance(c, Array) for c in cs)

        return ScalarAll(a, b, *cs)

    @_(  # type: ignore[name-defined, no-redef]  # noqa: F821
        "ANY LPAREN expr maybe_comma RPAREN"
    )
    def expr(self, p):  # noqa: F811
        expr = p.expr
        self.assert_or_error(
            isinstance(expr, Array) and expr.size >= 2,
            p,
            "`any` expr must be an array expression with at least two elements",
        )
        a, b, *cs = expr.flatlist()

        # must not fail since flat Array elements cannot be arrays themselves
        assert not isinstance(a, Array)
        assert not isinstance(b, Array)
        assert not any(isinstance(c, Array) for c in cs)

        return ScalarAny(a, b, *cs)

    @_("WHERE LPAREN expr COMMA expr COMMA expr maybe_comma RPAREN")  # type: ignore[name-defined, no-redef]  # noqa: F821
    def expr(self, p):  # noqa: F811
        with self.with_error_context(p, lambda err: f"{err}", exception=ValueError):
            return Array.map(ScalarWhere, p.expr0, p.expr1, p.expr2)

    # array operations
    @_("SIZE LPAREN expr maybe_comma RPAREN")  # type: ignore[name-defined, no-redef]  # noqa: F821
    def expr(self, p):  # noqa: F811
        self.assert_or_error(
            isinstance(p.expr, Array), p, "scalar non-array expression has no size"
        )
        return Number.from_symbolic_int(p.expr.size)

    @_("SHAPE LPAREN expr maybe_comma RPAREN")  # type: ignore[name-defined, no-redef]  # noqa: F821
    def expr(self, p):  # noqa: F811
        self.assert_or_error(
            isinstance(p.expr, Array), p, "scalar non-array expression has no shape"
        )
        return Array(*tuple(Number.from_symbolic_int(s) for s in p.expr.shape))

    @_("SUM LPAREN expr maybe_comma RPAREN")  # type: ignore[name-defined, no-redef]  # noqa: F821
    def expr(self, p):  # noqa: F811
        self.assert_or_error(
            isinstance(p.expr, Array), p, "cannot sum over scalar non-array expression"
        )
        return p.expr.sum()

    @_("MATMUL LPAREN expr COMMA expr maybe_comma RPAREN")  # type: ignore[name-defined, no-redef]  # noqa: F821
    def expr(self, p):  # noqa: F811
        self.assert_or_error(
            isinstance(p.expr0, Array),
            p,
            "cannot matmul(a, b) with scalar non-array parameter a",
        )
        self.assert_or_error(
            isinstance(p.expr1, Array),
            p,
            "cannot matmul(a, b) with scalar non-array parameter b",
        )
        with self.with_error_context(p, lambda err: f"{err}", exception=ValueError):
            return Array.matmul(p.expr0, p.expr1)

    # finite difference
    @_(  # type: ignore[name-defined, no-redef]  # noqa: F821
        "FINITE_DIFFERENCE LPAREN expr COMMA ORDER ASSIGN integer COMMA ACCURACY ASSIGN integer COMMA TYPE ASSIGN integer COMMA AXIS ASSIGN integer finite_difference_grid_spacing finite_difference_grid_period RPAREN"
    )
    def expr(self, p):  # noqa: F811
        self.assert_or_error(
            self._X is not None,
            p,
            "`finite_difference` is not available in pointwise QoIs",
        )

        expr = p.expr
        self.assert_or_error(
            p.expr.has_data,
            p,
            f"`finite_difference` expr must reference the data `{'x' if self._X is None else 'X'}`",
        )
        self.assert_or_error(
            not isinstance(p.expr, Array),
            p,
            "`finite_difference` expr must be a scalar array element expression, e.g. the centre value, not an array",
        )

        order = p.integer0
        self.assert_or_error(
            order >= 0, p, "`finite_difference` order must be non-negative"
        )

        accuracy = p.integer1
        self.assert_or_error(
            accuracy > 0, p, "`finite_difference` accuracy must be positive"
        )

        TYPES: dict[int, FiniteDifference] = {
            -1: FiniteDifference.backwards,
            0: FiniteDifference.central,
            1: FiniteDifference.forward,
        }
        type = p.integer2
        self.assert_or_error(
            type in TYPES,
            p,
            "`finite_difference` type must be 1 (forward), 0 (central), or -1 (backward)",
        )
        type = TYPES[type]
        if type == FiniteDifference.central:
            self.assert_or_error(
                accuracy % 2 == 0,
                p,
                "`finite_difference` accuracy must be even for a central finite difference",
            )

        axis = p.integer3
        self.assert_or_error(
            axis >= -len(self._X.shape) and axis < len(self._X.shape),
            p,
            f"`finite_difference` axis must be in range of the dimension {len(self._X.shape)} of the neighbourhood",
        )

        offsets = finite_difference_offsets(type, order, accuracy)

        grid_period = p.finite_difference_grid_period
        delta_transform = (
            (lambda e: e)
            if grid_period is None
            else (
                lambda e: Array.map(lambda f: ScalarSymmetricModulo(f, grid_period), e)
            )
        )

        if "spacing" in p.finite_difference_grid_spacing:
            grid_spacing = Group(p.finite_difference_grid_spacing["spacing"])

            coefficients = finite_difference_coefficients(
                order,
                tuple(
                    ScalarMultiply(Number.from_symbolic_int(o), grid_spacing)
                    for o in offsets
                ),
                lambda a: a,
                delta_transform=delta_transform,
            )
        else:
            grid_centre = Group(p.finite_difference_grid_spacing["centre"])

            coefficients = finite_difference_coefficients(
                order,
                tuple(grid_centre.apply_array_element_offset(axis, o) for o in offsets),
                lambda a: Group(ScalarSubtract(a, grid_centre)),
                delta_transform=delta_transform,
            )

        terms = [
            ScalarMultiply(expr.apply_array_element_offset(axis, o), c)
            for o, c in zip(offsets, coefficients)
        ]
        # even order=0 produces at least one term
        assert len(terms) > 0
        sum_ = ScalarLeftAssociativeSum(*terms)

        required_axis_before = 0
        required_axis_after = 0

        for idx in sum_.data_indices:
            required_axis_before = max(required_axis_before, self._I[axis] - idx[axis])
            required_axis_after = max(required_axis_after, idx[axis] - self._I[axis])

        self.assert_or_error(
            (required_axis_before <= self._I[axis])
            and (required_axis_after <= (self._X.shape[axis] - self._I[axis] - 1)),
            p,
            f"cannot compute the `finite_difference` on axis {axis} since the "
            "neighbourhood is insufficiently large: before should be at least "
            f"{required_axis_before} and after should be at least "
            f"{required_axis_after}",
        )

        assert not isinstance(sum_, Array)
        return Group(sum_)

    @_("COMMA GRID_SPACING ASSIGN expr")  # type: ignore[name-defined, no-redef]  # noqa: F821
    def finite_difference_grid_spacing(self, p):  # noqa: F811
        self.assert_or_error(
            not p.expr.has_data,
            p,
            f"`finite_difference` grid_spacing must not reference the data `{'x' if self._X is None else 'X'}`",
        )
        self.assert_or_error(
            not isinstance(p.expr, Array),
            p,
            "`finite_difference` grid_spacing must be a constant scalar expression, not an array",
        )
        return dict(spacing=p.expr)

    @_("COMMA GRID_CENTRE ASSIGN expr")  # type: ignore[name-defined, no-redef]  # noqa: F821
    def finite_difference_grid_spacing(self, p):  # noqa: F811
        self.assert_or_error(
            not p.expr.has_data,
            p,
            f"`finite_difference` grid_centre must not reference the data `{'x' if self._X is None else 'X'}`",
        )
        self.assert_or_error(
            not isinstance(p.expr, Array),
            p,
            "`finite_difference` grid_centre must be a constant scalar array element expression, not an array",
        )
        self.assert_or_error(
            len(p.expr.late_bound_constants) > 0,
            p,
            "`finite_difference` grid_centre must reference a late-bound constant",
        )
        return dict(centre=p.expr)

    @_("COMMA GRID_PERIOD ASSIGN expr maybe_comma")  # type: ignore[name-defined, no-redef]  # noqa: F821
    def finite_difference_grid_period(self, p):  # noqa: F811
        self.assert_or_error(
            not p.expr.has_data,
            p,
            f"`finite_difference` grid_period must not reference the data `{'x' if self._X is None else 'X'}`",
        )
        self.assert_or_error(
            len(p.expr.late_bound_constants) == 0,
            p,
            "finite_difference grid_period must not reference late-bound constants",
        )
        self.assert_or_error(
            not isinstance(p.expr, Array),
            p,
            "`finite_difference` grid_period must be a constant scalar number, not an array",
        )
        return p.expr

    @_("maybe_comma")  # type: ignore[name-defined, no-redef]  # noqa: F821
    def finite_difference_grid_period(self, p):  # noqa: F811
        pass

    # empty rule
    @_("")  # type: ignore[name-defined]  # noqa: F821
    def empty(self, p):
        pass

    # === parser error handling ===
    def error(self, t):
        actions = self._lrtable.lr_action[self.state]
        options = ", ".join(QoILexer.token_to_name(a) for a in actions)
        oneof = " one of" if len(actions) > 1 else ""

        if t is None:
            raise (
                SyntaxError(
                    f"expected more input but found EOF\nexpected{oneof} {options}",
                    (
                        "<qoi>",
                        1 + self._text.count("\n"),
                        len(self._text) - self._text.rfind("\n"),
                        self._text[self._text.rfind("\n") + 1 :],
                    ),
                )
                | ctx
            )

        t_value = f'"{t.value}"' if t.type == "STRING" else t.value

        raise (
            SyntaxError(
                f"unexpected token `{t_value}`\nexpected{oneof} {options}",
                (
                    "<qoi>",
                    t.lineno,
                    self.find_column(t),
                    self._text.splitlines()[t.lineno - 1],
                ),
            )
            | ctx
        )

    def raise_error(self, t, message):
        raise (
            SyntaxError(
                message,
                (
                    "<qoi>",
                    t.lineno,
                    self.find_column(t),
                    self._text.splitlines()[t.lineno - 1],
                ),
            )
            | ctx
        )

    def assert_or_error(self, check, t, message):
        if not check:
            self.raise_error(t, message() if callable(message) else message)

    @contextmanager
    def with_error_context(self, t, message, exception=Exception):
        try:
            yield
        except exception as err:
            if callable(message):
                self.raise_error(t, message(err))
            else:
                self.raise_error(t, message)

    def find_column(self, token) -> int:
        last_cr = self._text.rfind("\n", 0, token.index)
        if last_cr < 0:
            last_cr = 0
        column = (token.index - last_cr) + 1
        return column
