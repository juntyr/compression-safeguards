import itertools

from sly import Parser  # from sly.yacc import SlyLogger

from ...utils.bindings import Parameter
from .expr.abc import Expr
from .expr.abs import ScalarAbs
from .expr.addsub import ScalarAdd, ScalarSubtract
from .expr.array import Array
from .expr.data import Data, LateBoundConstant
from .expr.divmul import ScalarDivide, ScalarMultiply
from .expr.group import Group
from .expr.hyperbolic import Hyperbolic, ScalarHyperbolic
from .expr.literal import Euler, Number, Pi
from .expr.logexp import Exponential, Logarithm, ScalarExp, ScalarLog, ScalarLogWithBase
from .expr.neg import ScalarNegate
from .expr.power import ScalarPower
from .expr.round import ScalarCeil, ScalarFloor, ScalarRoundTiesEven, ScalarTrunc
from .expr.sign import ScalarSign
from .expr.square import ScalarSqrt, ScalarSquare
from .expr.trigonometric import (
    ScalarAsin,
    ScalarSin,
    ScalarTrigonometric,
    Trigonometric,
)
from .lexer import QoILexer


class NullWriter:
    def write(self, s):
        pass


class QoIParser(Parser):
    __slots__ = ("_x", "_X", "_I", "_vars", "_text")
    _x: Data
    _X: None | Array
    _I: None | tuple[int, ...]
    _vars: dict[Parameter, Expr]
    _text: str

    # log = SlyLogger(NullWriter())

    def __init__(self, *, x: Data, X: None | Array, I: None | tuple[int, ...]):  # noqa: E741
        self._x = x
        self._X = X
        self._I = I
        self._vars = dict()

    def parse(self, text: str, tokens):  # type: ignore
        self._text = text
        tokens, tokens2 = itertools.tee(tokens)

        if next(tokens2, None) is None:
            raise SyntaxError("expression must not be empty")

        return super().parse(tokens)

    tokens = QoILexer.tokens

    # === operator precedence and associativity ===
    precedence = (
        # lowest precedence: add and subtract
        ("left", PLUS, MINUS),  # type: ignore[name-defined]  # noqa: F821
        ("left", TIMES, DIVIDE),  # type: ignore[name-defined]  # noqa: F821
        ("right", UPLUS, UMINUS),  # type: ignore[name-defined]  # noqa: F821
        ("right", POWER),  # type: ignore[name-defined]  # noqa: F821
        ("left", INDEX, TRANSPOSE),  # type: ignore[name-defined]  # noqa: F821
        # highest precedence: array indexing and transpose
    )

    # === grammar rules ===

    # top-level: qoi := expr | { assign } return expr;
    @_("expr")  # type: ignore[name-defined, no-redef]  # noqa: F821
    def qoi(self, p):  # noqa: F811
        self.assert_or_error(
            not isinstance(p.expr, Array),
            p,
            lambda: f"QoI expression must be a scalar but is an array expression of shape {p.expr.shape}",
        )
        self.assert_or_error(p.expr.has_data, p, "QoI expression must not be constant")
        return p.expr

    @_("many_assign return_expr")  # type: ignore[name-defined, no-redef]  # noqa: F821
    def qoi(self, p):  # noqa: F811
        return p.return_expr

    @_("RETURN expr SEMI")  # type: ignore[name-defined]  # noqa: F821
    def return_expr(self, p):
        self.assert_or_error(
            not isinstance(p.expr, Array),
            p,
            lambda: f"QoI return expression must be a scalar but is an array expression of shape {p.expr.shape}",
        )
        self.assert_or_error(
            p.expr.has_data, p, "QoI return expression must not be constant"
        )
        return p.expr

    # variable assignment: assign := V["id"] = expr;
    @_("VS LBRACK quotedparameter RBRACK EQUAL expr SEMI")  # type: ignore[name-defined, no-redef]  # noqa: F821
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
        self._vars[p.quotedparameter] = Array.map_unary(p.expr, Group)

    @_("VA LBRACK quotedparameter RBRACK EQUAL expr SEMI")  # type: ignore[name-defined, no-redef]  # noqa: F821
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
        self._vars[p.quotedparameter] = Array.map_unary(p.expr, Group)

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

    # integer and floating point literals
    @_("INTEGER")  # type: ignore[name-defined, no-redef]  # noqa: F821
    def expr(self, p):  # noqa: F811
        return Number.from_symbolic_int(p.INTEGER)

    @_("FLOAT")  # type: ignore[name-defined, no-redef]  # noqa: F821
    def expr(self, p):  # noqa: F811
        return Number(p.FLOAT)

    # array literal
    @_("LBRACK expr many_comma_expr RBRACK")  # type: ignore[name-defined, no-redef]  # noqa: F821
    def expr(self, p):  # noqa: F811
        try:
            return Array(*([p.expr] + p.many_comma_expr))
        except ValueError as err:
            self.raise_error(p, f"invalid array literal: {err}")

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
        return Array.map_unary(p.expr, ScalarNegate)

    # binary operators (add, subtract, multiply, divide, power):
    #  expr := expr OP expr
    @_("expr PLUS expr")  # type: ignore[name-defined, no-redef]  # noqa: F821
    def expr(self, p):  # noqa: F811
        try:
            return Array.map_binary(p.expr0, p.expr1, ScalarAdd)
        except ValueError as err:
            self.raise_error(p, f"{err}")

    @_("expr MINUS expr")  # type: ignore[name-defined, no-redef]  # noqa: F821
    def expr(self, p):  # noqa: F811
        try:
            return Array.map_binary(p.expr0, p.expr1, ScalarSubtract)
        except ValueError as err:
            self.raise_error(p, f"{err}")

    @_("expr TIMES expr")  # type: ignore[name-defined, no-redef]  # noqa: F821
    def expr(self, p):  # noqa: F811
        try:
            return Array.map_binary(p.expr0, p.expr1, ScalarMultiply)
        except ValueError as err:
            self.raise_error(p, f"{err}")

    @_("expr POWER expr")  # type: ignore[name-defined, no-redef]  # noqa: F821
    def expr(self, p):  # noqa: F811
        try:
            return Array.map_binary(p.expr0, p.expr1, ScalarPower)
        except ValueError as err:
            self.raise_error(p, f"{err}")

    @_("expr DIVIDE expr")  # type: ignore[name-defined, no-redef]  # noqa: F821
    def expr(self, p):  # noqa: F811
        try:
            return Array.map_binary(p.expr0, p.expr1, ScalarDivide)
        except ValueError as err:
            self.raise_error(p, f"{err}")

    # array transpose: expr := expr.T
    @_("expr TRANSPOSE %prec TRANSPOSE")  # type: ignore[name-defined, no-redef]  # noqa: F821
    def expr(self, p):  # noqa: F811
        self.assert_or_error(
            isinstance(p.expr, Array), p, "cannot transpose scalar non-array expression"
        )
        return p.expr.transpose()

    # group in parentheses
    @_("LPAREN expr RPAREN")  # type: ignore[name-defined, no-redef]  # noqa: F821
    def expr(self, p):  # noqa: F811
        return Array.map_unary(p.expr, Group)

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
        return Array.map_unary(
            self._X,
            lambda e: LateBoundConstant.like(p.quotedparameter, e),
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
        try:
            return Parameter(p.STRING)
        except Exception:
            self.raise_error(
                p, f'invalid quoted parameter "{p.STRING}": must be a valid identifier'
            )

    # array indexing
    @_("expr LBRACK IDX RBRACK %prec INDEX")  # type: ignore[name-defined, no-redef]  # noqa: F821
    def expr(self, p):  # noqa: F811
        self.assert_or_error(
            self._I is not None, p, "index `I` is not available in pointwise QoIs"
        )
        self.assert_or_error(
            isinstance(p.expr, Array), p, "cannot index scalar non-array expression"
        )
        try:
            return p.expr.index(self._I)
        except IndexError as err:
            self.raise_error(p, f"{err}")

    @_("expr LBRACK index_ many_comma_index RBRACK %prec INDEX")  # type: ignore[name-defined, no-redef]  # noqa: F821
    def expr(self, p):  # noqa: F811
        self.assert_or_error(
            isinstance(p.expr, Array), p, "cannot index scalar non-array expression"
        )
        try:
            return p.expr.index(tuple([p.index_] + p.many_comma_index))
        except IndexError as err:
            self.raise_error(p, f"{err}")

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

    @_("expr")  # type: ignore[name-defined]  # noqa: F821
    def index_(self, p):
        self.assert_or_error(
            isinstance(p.expr, Number) and p.expr.as_int() is not None,
            p,
            "cannot index by non-integer expression",
        )
        return p.expr.as_int()

    @_("IDX LBRACK integer RBRACK")  # type: ignore[name-defined, no-redef]  # noqa: F821
    def expr(self, p):  # noqa: F811
        self.assert_or_error(
            self._I is not None, p, "index `I` is not available in pointwise QoIs"
        )
        return Number.from_symbolic_int(self._I[p.integer])

    # functions

    # unknown function
    # @_("ID LPAREN expr many_comma_expr RPAREN")  # type: ignore[name-defined, no-redef]  # noqa: F821
    # def expr(self, p):  # noqa: F811
    #     self.raise_error(p, f"unknown function `{p.ID}`")

    # logarithms and exponentials
    @_("LN LPAREN expr maybe_comma RPAREN")  # type: ignore[name-defined, no-redef]  # noqa: F821
    def expr(self, p):  # noqa: F811
        return Array.map_unary(p.expr, lambda e: ScalarLog(Logarithm.ln, e))

    @_("LOG2 LPAREN expr maybe_comma RPAREN")  # type: ignore[name-defined, no-redef]  # noqa: F821
    def expr(self, p):  # noqa: F811
        return Array.map_unary(p.expr, lambda e: ScalarLog(Logarithm.log2, e))

    @_("LOG LPAREN expr COMMA BASE EQUAL expr maybe_comma RPAREN")  # type: ignore[name-defined, no-redef]  # noqa: F821
    def expr(self, p):  # noqa: F811
        try:
            return Array.map_binary(
                p.expr0, p.expr1, lambda a, b: ScalarLogWithBase(a, b)
            )
        except ValueError as err:
            self.raise_error(p, f"{err}")

    @_("EXP LPAREN expr maybe_comma RPAREN")  # type: ignore[name-defined, no-redef]  # noqa: F821
    def expr(self, p):  # noqa: F811
        return Array.map_unary(p.expr, lambda e: ScalarExp(Exponential.exp, e))

    @_("EXP2 LPAREN expr maybe_comma RPAREN")  # type: ignore[name-defined, no-redef]  # noqa: F821
    def expr(self, p):  # noqa: F811
        return Array.map_unary(p.expr, lambda e: ScalarExp(Exponential.exp2, e))

    # exponentiation
    @_("SQRT LPAREN expr maybe_comma RPAREN")  # type: ignore[name-defined, no-redef]  # noqa: F821
    def expr(self, p):  # noqa: F811
        return Array.map_unary(p.expr, ScalarSqrt)

    @_("SQUARE LPAREN expr maybe_comma RPAREN")  # type: ignore[name-defined, no-redef]  # noqa: F821
    def expr(self, p):  # noqa: F811
        return Array.map_unary(p.expr, ScalarSquare)

    # absolute value
    @_("ABS LPAREN expr maybe_comma RPAREN")  # type: ignore[name-defined, no-redef]  # noqa: F821
    def expr(self, p):  # noqa: F811
        return Array.map_unary(p.expr, ScalarAbs)

    # sign and rounding
    @_("SIGN LPAREN expr maybe_comma RPAREN")  # type: ignore[name-defined, no-redef]  # noqa: F821
    def expr(self, p):  # noqa: F811
        return Array.map_unary(p.expr, ScalarSign)

    @_("FLOOR LPAREN expr maybe_comma RPAREN")  # type: ignore[name-defined, no-redef]  # noqa: F821
    def expr(self, p):  # noqa: F811
        return Array.map_unary(p.expr, ScalarFloor)

    @_("CEIL LPAREN expr maybe_comma RPAREN")  # type: ignore[name-defined, no-redef]  # noqa: F821
    def expr(self, p):  # noqa: F811
        return Array.map_unary(p.expr, ScalarCeil)

    @_("TRUNC LPAREN expr maybe_comma RPAREN")  # type: ignore[name-defined, no-redef]  # noqa: F821
    def expr(self, p):  # noqa: F811
        return Array.map_unary(p.expr, ScalarTrunc)

    @_("ROUND_TIES_EVEN LPAREN expr maybe_comma RPAREN")  # type: ignore[name-defined, no-redef]  # noqa: F821
    def expr(self, p):  # noqa: F811
        return Array.map_unary(p.expr, ScalarRoundTiesEven)

    # trigonometric
    @_("SIN LPAREN expr maybe_comma RPAREN")  # type: ignore[name-defined, no-redef]  # noqa: F821
    def expr(self, p):  # noqa: F811
        return Array.map_unary(p.expr, ScalarSin)

    @_("COS LPAREN expr maybe_comma RPAREN")  # type: ignore[name-defined, no-redef]  # noqa: F821
    def expr(self, p):  # noqa: F811
        return Array.map_unary(
            p.expr, lambda e: ScalarTrigonometric(Trigonometric.cos, e)
        )

    @_("TAN LPAREN expr maybe_comma RPAREN")  # type: ignore[name-defined, no-redef]  # noqa: F821
    def expr(self, p):  # noqa: F811
        return Array.map_unary(
            p.expr, lambda e: ScalarTrigonometric(Trigonometric.tan, e)
        )

    @_("COT LPAREN expr maybe_comma RPAREN")  # type: ignore[name-defined, no-redef]  # noqa: F821
    def expr(self, p):  # noqa: F811
        return Array.map_unary(
            p.expr, lambda e: ScalarTrigonometric(Trigonometric.cot, e)
        )

    @_("SEC LPAREN expr maybe_comma RPAREN")  # type: ignore[name-defined, no-redef]  # noqa: F821
    def expr(self, p):  # noqa: F811
        return Array.map_unary(
            p.expr, lambda e: ScalarTrigonometric(Trigonometric.sec, e)
        )

    @_("CSC LPAREN expr maybe_comma RPAREN")  # type: ignore[name-defined, no-redef]  # noqa: F821
    def expr(self, p):  # noqa: F811
        return Array.map_unary(
            p.expr, lambda e: ScalarTrigonometric(Trigonometric.csc, e)
        )

    @_("ASIN LPAREN expr maybe_comma RPAREN")  # type: ignore[name-defined, no-redef]  # noqa: F821
    def expr(self, p):  # noqa: F811
        return Array.map_unary(p.expr, ScalarAsin)

    @_("ACOS LPAREN expr maybe_comma RPAREN")  # type: ignore[name-defined, no-redef]  # noqa: F821
    def expr(self, p):  # noqa: F811
        return Array.map_unary(
            p.expr, lambda e: ScalarTrigonometric(Trigonometric.acos, e)
        )

    @_("ATAN LPAREN expr maybe_comma RPAREN")  # type: ignore[name-defined, no-redef]  # noqa: F821
    def expr(self, p):  # noqa: F811
        return Array.map_unary(
            p.expr, lambda e: ScalarTrigonometric(Trigonometric.atan, e)
        )

    @_("ACOT LPAREN expr maybe_comma RPAREN")  # type: ignore[name-defined, no-redef]  # noqa: F821
    def expr(self, p):  # noqa: F811
        return Array.map_unary(
            p.expr, lambda e: ScalarTrigonometric(Trigonometric.acot, e)
        )

    @_("ASEC LPAREN expr maybe_comma RPAREN")  # type: ignore[name-defined, no-redef]  # noqa: F821
    def expr(self, p):  # noqa: F811
        return Array.map_unary(
            p.expr, lambda e: ScalarTrigonometric(Trigonometric.asec, e)
        )

    @_("ACSC LPAREN expr maybe_comma RPAREN")  # type: ignore[name-defined, no-redef]  # noqa: F821
    def expr(self, p):  # noqa: F811
        return Array.map_unary(
            p.expr, lambda e: ScalarTrigonometric(Trigonometric.acsc, e)
        )

    # hypergeometric
    @_("SINH LPAREN expr maybe_comma RPAREN")  # type: ignore[name-defined, no-redef]  # noqa: F821
    def expr(self, p):  # noqa: F811
        return Array.map_unary(p.expr, lambda e: ScalarHyperbolic(Hyperbolic.sinh, e))

    @_("COSH LPAREN expr maybe_comma RPAREN")  # type: ignore[name-defined, no-redef]  # noqa: F821
    def expr(self, p):  # noqa: F811
        return Array.map_unary(p.expr, lambda e: ScalarHyperbolic(Hyperbolic.cosh, e))

    @_("TANH LPAREN expr maybe_comma RPAREN")  # type: ignore[name-defined, no-redef]  # noqa: F821
    def expr(self, p):  # noqa: F811
        return Array.map_unary(p.expr, lambda e: ScalarHyperbolic(Hyperbolic.tanh, e))

    @_("COTH LPAREN expr maybe_comma RPAREN")  # type: ignore[name-defined, no-redef]  # noqa: F821
    def expr(self, p):  # noqa: F811
        return Array.map_unary(p.expr, lambda e: ScalarHyperbolic(Hyperbolic.coth, e))

    @_("SECH LPAREN expr maybe_comma RPAREN")  # type: ignore[name-defined, no-redef]  # noqa: F821
    def expr(self, p):  # noqa: F811
        return Array.map_unary(p.expr, lambda e: ScalarHyperbolic(Hyperbolic.sech, e))

    @_("CSCH LPAREN expr maybe_comma RPAREN")  # type: ignore[name-defined, no-redef]  # noqa: F821
    def expr(self, p):  # noqa: F811
        return Array.map_unary(p.expr, lambda e: ScalarHyperbolic(Hyperbolic.csch, e))

    @_("ASINH LPAREN expr maybe_comma RPAREN")  # type: ignore[name-defined, no-redef]  # noqa: F821
    def expr(self, p):  # noqa: F811
        return Array.map_unary(p.expr, lambda e: ScalarHyperbolic(Hyperbolic.asinh, e))

    @_("ACOSH LPAREN expr maybe_comma RPAREN")  # type: ignore[name-defined, no-redef]  # noqa: F821
    def expr(self, p):  # noqa: F811
        return Array.map_unary(p.expr, lambda e: ScalarHyperbolic(Hyperbolic.acosh, e))

    @_("ATANH LPAREN expr maybe_comma RPAREN")  # type: ignore[name-defined, no-redef]  # noqa: F821
    def expr(self, p):  # noqa: F811
        return Array.map_unary(p.expr, lambda e: ScalarHyperbolic(Hyperbolic.atanh, e))

    @_("ACOTH LPAREN expr maybe_comma RPAREN")  # type: ignore[name-defined, no-redef]  # noqa: F821
    def expr(self, p):  # noqa: F811
        return Array.map_unary(p.expr, lambda e: ScalarHyperbolic(Hyperbolic.acoth, e))

    @_("ASECH LPAREN expr maybe_comma RPAREN")  # type: ignore[name-defined, no-redef]  # noqa: F821
    def expr(self, p):  # noqa: F811
        return Array.map_unary(p.expr, lambda e: ScalarHyperbolic(Hyperbolic.asech, e))

    @_("ACSCH LPAREN expr maybe_comma RPAREN")  # type: ignore[name-defined, no-redef]  # noqa: F821
    def expr(self, p):  # noqa: F811
        return Array.map_unary(p.expr, lambda e: ScalarHyperbolic(Hyperbolic.acsch, e))

    # array operations
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
        try:
            return Array.matmul(p.expr0, p.expr1)
        except ValueError as err:
            self.raise_error(p, f"{err}")

    # finite difference
    @_(  # type: ignore[name-defined, no-redef]  # noqa: F821
        "FINITE_DIFFERENCE LPAREN expr COMMA ORDER EQUAL integer COMMA ACCURACY EQUAL integer COMMA TYPE EQUAL integer COMMA AXIS EQUAL integer finite_difference_grid_spacing finite_difference_grid_period RPAREN"
    )
    def expr(self, p):  # noqa: F811
        from .expr.finite_difference import (
            FiniteDifference,
            ScalarSymmetricModulo,
            finite_difference_coefficients,
            finite_difference_offsets,
        )

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

        type = p.integer2
        self.assert_or_error(
            type in (-1, 0, +1),
            p,
            "`finite_difference` type must be 1 (forward), 0 (central), or -1 (backward)",
        )
        type = [
            FiniteDifference.central,
            FiniteDifference.forward,
            FiniteDifference.backwards,
        ][type]
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
                lambda e: Array.map_unary(
                    e, lambda f: ScalarSymmetricModulo(f, grid_period)
                )
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
        assert len(terms) > 0
        acc = terms[0]
        for t in terms[1:]:
            acc = ScalarAdd(acc, t)

        for idx in acc.data_indices:
            for i, a, c in zip(idx, self._X.shape, self._I):
                self.assert_or_error(
                    i >= 0,
                    p,
                    f"cannot compute the `finite_difference` on axis {axis} since the neighbourhood is insufficiently large: before should be at least {c - i}",
                )
                self.assert_or_error(
                    i < a,
                    p,
                    f"cannot compute the `finite_difference` on axis {axis} since the neighbourhood is insufficiently large: after should be at least {i - c}",
                )

        assert not isinstance(acc, Array)
        return Group(acc)

    @_("COMMA GRID_SPACING EQUAL expr")  # type: ignore[name-defined, no-redef]  # noqa: F821
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

    @_("COMMA GRID_CENTRE EQUAL expr")  # type: ignore[name-defined, no-redef]  # noqa: F821
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

    @_("COMMA GRID_PERIOD EQUAL expr maybe_comma")  # type: ignore[name-defined, no-redef]  # noqa: F821
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
        options = ", ".join(QoILexer.token_to_name(t) for t in actions)
        oneof = " one of" if len(actions) > 1 else ""

        if t is None:
            raise SyntaxError(
                f"expected more input but found EOF\nexpected{oneof} {options}"
            )

        raise SyntaxError(
            f"unexpected token `{t.value}` at line {t.lineno}, column {self.find_column(t)}\nexpected{oneof} {options}"
        )

    def raise_error(self, t, message):
        raise SyntaxError(f"{message} at line {t.lineno}, column {self.find_column(t)}")

    def assert_or_error(self, check, t, message):
        if not check:
            self.raise_error(t, message() if callable(message) else message)

    def find_column(self, token):
        last_cr = self._text.rfind("\n", 0, token.index)
        if last_cr < 0:
            last_cr = 0
        column = (token.index - last_cr) + 1
        return column
