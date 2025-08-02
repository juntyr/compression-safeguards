import itertools

from sly import Parser  # from sly.yacc import SlyLogger

from ...utils.bindings import Parameter
from .expr.abc import Expr
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

    precedence = (
        ("left", PLUS, MINUS),  # type: ignore[name-defined]  # noqa: F821
        ("left", TIMES, DIVIDE),  # type: ignore[name-defined]  # noqa: F821
        ("right", UPLUS, UMINUS),  # type: ignore[name-defined]  # noqa: F821
        ("right", POWER),  # type: ignore[name-defined]  # noqa: F821
        ("left", INDEX, TRANSPOSE),  # type: ignore[name-defined]  # noqa: F821
    )

    @_("expr")  # type: ignore[name-defined, no-redef]  # noqa: F821
    def qoi(self, p):  # noqa: F811
        assert not isinstance(p.expr, Array), (
            f"QoI expr must be scalar but is an array expression of shape {p.expr.shape}"
        )
        assert p.expr.has_data, "QoI expr must not be constant"
        return p.expr

    @_("many_assign RETURN expr SEMI")  # type: ignore[name-defined, no-redef]  # noqa: F821
    def qoi(self, p):  # noqa: F811
        assert not isinstance(p.expr, Array), (
            f"QoI expr must be scalar but is an array expression of shape {p.expr.shape}"
        )
        assert p.expr.has_data, "QoI expr must not be constant"
        return p.expr

    @_("assign many_assign")  # type: ignore[name-defined, no-redef]  # noqa: F821
    def many_assign(self, p):  # noqa: F811
        pass

    @_("empty")  # type: ignore[name-defined, no-redef]  # noqa: F821
    def many_assign(self, p):  # noqa: F811
        pass

    @_("QUOTEDID")  # type: ignore[name-defined, no-redef]  # noqa: F821
    def quotedid(self, p):  # noqa: F811
        return p.QUOTEDID

    @_("QUOTE ID QUOTE")  # type: ignore[name-defined, no-redef]  # noqa: F821
    def quotedid(self, p):  # noqa: F811
        assert False, "quoted identifier cannot contain whitespace or comments"

    @_("QUOTE ID")  # type: ignore[name-defined, no-redef]  # noqa: F821
    def quotedid(self, p):  # noqa: F811
        assert False, "missing closing quote for quoted identifier"

    @_("VS LBRACK quotedid RBRACK EQUAL expr SEMI")  # type: ignore[name-defined, no-redef]  # noqa: F821
    def assign(self, p):  # noqa: F811
        assert self._X is None
        assert not p.quotedid.startswith("$")
        assert p.quotedid not in self._vars
        self._vars[Parameter(p.quotedid)] = Array.map_unary(p.expr, Group)

    @_("VA LBRACK quotedid RBRACK EQUAL expr SEMI")  # type: ignore[name-defined, no-redef]  # noqa: F821
    def assign(self, p):  # noqa: F811
        assert self._X is not None
        assert not p.quotedid.startswith("$")
        assert p.quotedid not in self._vars
        self._vars[Parameter(p.quotedid)] = Array.map_unary(p.expr, Group)

    @_("VS LBRACK quotedid RBRACK")  # type: ignore[name-defined, no-redef]  # noqa: F821
    def expr(self, p):  # noqa: F811
        assert self._X is None
        assert not p.quotedid.startswith("$")
        return self._vars[Parameter(p.quotedid)]

    @_("VA LBRACK quotedid RBRACK")  # type: ignore[name-defined, no-redef]  # noqa: F821
    def expr(self, p):  # noqa: F811
        assert self._X is not None
        assert not p.quotedid.startswith("$")
        return self._vars[Parameter(p.quotedid)]

    @_("expr PLUS expr")  # type: ignore[name-defined, no-redef]  # noqa: F821
    def expr(self, p):  # noqa: F811
        return Array.map_binary(p.expr0, p.expr1, ScalarAdd)

    @_("expr MINUS expr")  # type: ignore[name-defined, no-redef]  # noqa: F821
    def expr(self, p):  # noqa: F811
        return Array.map_binary(p.expr0, p.expr1, ScalarSubtract)

    @_("expr TIMES expr")  # type: ignore[name-defined, no-redef]  # noqa: F821
    def expr(self, p):  # noqa: F811
        return Array.map_binary(p.expr0, p.expr1, ScalarMultiply)

    @_("expr POWER expr")  # type: ignore[name-defined, no-redef]  # noqa: F821
    def expr(self, p):  # noqa: F811
        return Array.map_binary(p.expr0, p.expr1, ScalarPower)

    @_("expr DIVIDE expr")  # type: ignore[name-defined, no-redef]  # noqa: F821
    def expr(self, p):  # noqa: F811
        return Array.map_binary(p.expr0, p.expr1, ScalarDivide)

    @_("PLUS expr %prec UPLUS")  # type: ignore[name-defined, no-redef]  # noqa: F821
    def expr(self, p):  # noqa: F811
        return p.expr

    @_("MINUS expr %prec UMINUS")  # type: ignore[name-defined, no-redef]  # noqa: F821
    def expr(self, p):  # noqa: F811
        return Array.map_unary(p.expr, ScalarNegate)

    @_("LPAREN expr RPAREN")  # type: ignore[name-defined, no-redef]  # noqa: F821
    def expr(self, p):  # noqa: F811
        return Array.map_unary(p.expr, Group)

    @_("")  # type: ignore[name-defined]  # noqa: F821
    def empty(self, p):
        pass

    @_("LBRACK expr many_comma_expr RBRACK")  # type: ignore[name-defined, no-redef]  # noqa: F821
    def expr(self, p):  # noqa: F811
        return Array(*([p.expr] + p.many_comma_expr))

    @_("LBRACK RBRACK")  # type: ignore[name-defined, no-redef]  # noqa: F821
    def expr(self, p):  # noqa: F811
        assert False, "illegal empty array literal"

    @_("expr LBRACK index_ many_comma_index RBRACK %prec INDEX")  # type: ignore[name-defined, no-redef]  # noqa: F821
    def expr(self, p):  # noqa: F811
        assert isinstance(p.expr, Array), "only array expressions can be indexed"
        return p.expr.index(tuple([p.index_] + p.many_comma_index))

    @_("expr LBRACK IDX RBRACK %prec INDEX")  # type: ignore[name-defined, no-redef]  # noqa: F821
    def expr(self, p):  # noqa: F811
        assert self._I is not None
        assert isinstance(p.expr, Array), "only array expressions can be indexed"
        return p.expr.index(self._I)

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
        import numpy as np

        expr = p.expr.constant_fold(np.dtype(int))
        assert isinstance(expr, np.dtype(int).type), (
            "cannot index by non-integer expression"
        )
        return int(expr)

    @_("IDX LBRACK integer RBRACK")  # type: ignore[name-defined, no-redef]  # noqa: F821
    def expr(self, p):  # noqa: F811
        assert self._I is not None
        return Number(f"{self._I[p.integer]}")

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

    @_("COMMA")  # type: ignore[name-defined, no-redef]  # noqa: F821
    def maybe_comma(self, p):  # noqa: F811
        pass

    @_("empty")  # type: ignore[name-defined, no-redef]  # noqa: F821
    def maybe_comma(self, p):  # noqa: F811
        pass

    @_("FLOAT")  # type: ignore[name-defined, no-redef]  # noqa: F821
    def expr(self, p):  # noqa: F811
        return Number(p.FLOAT)

    @_("INTEGER")  # type: ignore[name-defined, no-redef]  # noqa: F821
    def expr(self, p):  # noqa: F811
        return Number(f"{p.INTEGER}")

    @_("EULER")  # type: ignore[name-defined, no-redef]  # noqa: F821
    def expr(self, p):  # noqa: F811
        return Euler()

    @_("PI")  # type: ignore[name-defined, no-redef]  # noqa: F821
    def expr(self, p):  # noqa: F811
        return Pi()

    @_("XS")  # type: ignore[name-defined, no-redef]  # noqa: F821
    def expr(self, p):  # noqa: F811
        return self._x

    @_("XA")  # type: ignore[name-defined, no-redef]  # noqa: F821
    def expr(self, p):  # noqa: F811
        assert self._X is not None
        return self._X

    @_("LN LPAREN expr maybe_comma RPAREN")  # type: ignore[name-defined, no-redef]  # noqa: F821
    def expr(self, p):  # noqa: F811
        return Array.map_unary(p.expr, lambda e: ScalarLog(Logarithm.ln, e))

    @_("LOG2 LPAREN expr maybe_comma RPAREN")  # type: ignore[name-defined, no-redef]  # noqa: F821
    def expr(self, p):  # noqa: F811
        return Array.map_unary(p.expr, lambda e: ScalarLog(Logarithm.log2, e))

    @_("LOG LPAREN expr COMMA BASE EQUAL expr maybe_comma RPAREN")  # type: ignore[name-defined, no-redef]  # noqa: F821
    def expr(self, p):  # noqa: F811
        return Array.map_binary(p.expr0, p.expr1, lambda a, b: ScalarLogWithBase(a, b))

    @_("EXP LPAREN expr maybe_comma RPAREN")  # type: ignore[name-defined, no-redef]  # noqa: F821
    def expr(self, p):  # noqa: F811
        return Array.map_unary(p.expr, lambda e: ScalarExp(Exponential.exp, e))

    @_("EXP2 LPAREN expr maybe_comma RPAREN")  # type: ignore[name-defined, no-redef]  # noqa: F821
    def expr(self, p):  # noqa: F811
        return Array.map_unary(p.expr, lambda e: ScalarExp(Exponential.exp2, e))

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

    @_("SQRT LPAREN expr maybe_comma RPAREN")  # type: ignore[name-defined, no-redef]  # noqa: F821
    def expr(self, p):  # noqa: F811
        return Array.map_unary(p.expr, ScalarSqrt)

    @_("SQUARE LPAREN expr maybe_comma RPAREN")  # type: ignore[name-defined, no-redef]  # noqa: F821
    def expr(self, p):  # noqa: F811
        return Array.map_unary(p.expr, ScalarSquare)

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

    # @_("ID LPAREN expr many_comma_expr RPAREN")  # type: ignore[name-defined, no-redef]  # noqa: F821
    # def expr(self, p):  # noqa: F811
    #     assert False, f"unknown function `{p.ID}`"

    @_("SUM LPAREN expr maybe_comma RPAREN")  # type: ignore[name-defined, no-redef]  # noqa: F821
    def expr(self, p):  # noqa: F811
        assert isinstance(p.expr, Array), "can only sum over an array expression"
        return p.expr.sum()

    @_("MATMUL LPAREN expr COMMA expr maybe_comma RPAREN")  # type: ignore[name-defined, no-redef]  # noqa: F821
    def expr(self, p):  # noqa: F811
        assert isinstance(p.expr0, Array) and isinstance(p.expr1, Array), (
            "can only matmul(a, b) with arrays a and b"
        )
        return Array.matmul(p.expr0, p.expr1)

    @_("expr TRANSPOSE %prec TRANSPOSE")  # type: ignore[name-defined, no-redef]  # noqa: F821
    def expr(self, p):  # noqa: F811
        assert isinstance(p.expr, Array), "can only transpose an array expression"
        return p.expr.transpose()

    @_("CS LBRACK quotedid RBRACK")  # type: ignore[name-defined, no-redef]  # noqa: F821
    def expr(self, p):  # noqa: F811
        return LateBoundConstant.like(Parameter(p.quotedid), self._x)

    @_("CA LBRACK quotedid RBRACK")  # type: ignore[name-defined, no-redef]  # noqa: F821
    def expr(self, p):  # noqa: F811
        assert self._X is not None
        return Array.map_unary(
            self._X,
            lambda e: LateBoundConstant.like(Parameter(p.quotedid), e),
        )

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

        assert self._X is not None
        assert self._I is not None

        expr = p.expr
        assert p.expr.has_data, (
            f"finite_difference expr must reference the data {'x' if self._X is None else 'X'}"
        )
        assert not isinstance(p.expr, Array), (
            "finite_difference expr must be a scalar array element expression, e.g. the centre value, not an array"
        )

        order = p.integer0
        assert order >= 0, "finite_difference order must be non-negative"

        accuracy = p.integer1
        assert accuracy > 0, "finite_difference accuracy must be positive"

        type = p.integer2
        assert type in (-1, 0, +1), (
            "finite_difference type must be 1 (forward), 0 (central), or -1 (backward)"
        )
        type = [
            FiniteDifference.central,
            FiniteDifference.forward,
            FiniteDifference.backwards,
        ][type]
        if type == FiniteDifference.central:
            assert accuracy % 2 == 0, (
                "finite_difference accuracy must be even for a central finite difference"
            )

        axis = p.integer3
        assert axis >= -len(self._X.shape) and axis < len(self._X.shape), (
            "finite_difference axis must be in range of the dimension of the neighbourhood"
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
                tuple(ScalarMultiply(Number(f"{o}"), grid_spacing) for o in offsets),
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
                assert i >= 0, (
                    f"cannot compute the finite_difference on axis {axis} since the neighbourhood is insufficiently large: before should be at least {c - i}"
                )
                assert i < a, (
                    f"cannot compute the finite_difference on axis {axis} since the neighbourhood is insufficiently large: after should be at least {i - c}"
                )

        assert not isinstance(acc, Array)
        return Group(acc)

    @_("INTEGER")  # type: ignore[name-defined, no-redef]  # noqa: F821
    def integer(self, p):  # noqa: F811
        return p.INTEGER

    @_("PLUS INTEGER")  # type: ignore[name-defined, no-redef]  # noqa: F821
    def integer(self, p):  # noqa: F811
        return p.INTEGER

    @_("MINUS INTEGER")  # type: ignore[name-defined, no-redef]  # noqa: F821
    def integer(self, p):  # noqa: F811
        return -p.INTEGER

    @_("COMMA GRID_SPACING EQUAL expr")  # type: ignore[name-defined, no-redef]  # noqa: F821
    def finite_difference_grid_spacing(self, p):  # noqa: F811
        assert not p.expr.has_data, (
            f"finite_difference grid_spacing must not reference the data {'x' if self._X is None else 'X'}"
        )
        assert not isinstance(p.expr, Array), (
            "finite_difference grid_spacing must be a constant scalar expression, not an array"
        )
        return dict(spacing=p.expr)

    @_("COMMA GRID_CENTRE EQUAL expr")  # type: ignore[name-defined, no-redef]  # noqa: F821
    def finite_difference_grid_spacing(self, p):  # noqa: F811
        assert not p.expr.has_data, (
            f"finite_difference grid_centre must not reference the data {'x' if self._X is None else 'X'}"
        )
        assert not isinstance(p.expr, Array), (
            "finite_difference grid_centre must be a constant scalar array element expression, not an array"
        )
        assert len(p.expr.late_bound_constants) > 0, (
            "finite_difference grid_centre must reference a late-bound constant"
        )
        return dict(centre=p.expr)

    @_("COMMA GRID_PERIOD EQUAL expr maybe_comma")  # type: ignore[name-defined, no-redef]  # noqa: F821
    def finite_difference_grid_period(self, p):  # noqa: F811
        import numpy as np

        assert not p.expr.has_data, (
            f"finite_difference grid_period must not reference the data {'x' if self._X is None else 'X'}"
        )
        assert len(p.expr.late_bound_constants) == 0, (
            "finite_difference grid_period must not reference late-bound constants"
        )
        period = p.expr.constant_fold(np.dtype(np.float64))
        assert isinstance(period, np.float64), (
            "finite_difference grid_period must be a constant scalar number"
        )
        return p.expr

    @_("maybe_comma")  # type: ignore[name-defined, no-redef]  # noqa: F821
    def finite_difference_grid_period(self, p):  # noqa: F811
        pass

    def error(self, t):
        actions = self._lrtable.lr_action[self.state]
        options = ", ".join(token_to_name(t) for t in actions)
        oneof = " one of" if len(actions) > 1 else ""

        if t is None:
            raise SyntaxError(
                f"expected more input but found EOF\nexpected{oneof} {options}"
            )

        raise SyntaxError(
            f"illegal token `{t.value}` at line {t.lineno}, column {find_column(self._text, t)}\nexpected{oneof} {options}"
        )


def find_column(text, token):
    last_cr = text.rfind("\n", 0, token.index)
    if last_cr < 0:
        last_cr = 0
    column = (token.index - last_cr) + 1
    return column


def token_to_name(token: str) -> str:
    return {
        "INTEGER": "integer",
        "FLOAT": "floating-point number",
        "PLUS": "`+`",
        "TIMES": "`*`",
        "MINUS": "`-`",
        "DIVIDE": "`/`",
        "LPAREN": "`(`",
        "RPAREN": "`)`",
        "POWER": "`**`",
        "LBRACK": "`[`",
        "RBRACK": "`]`",
        "COMMA": "`,`",
        "EQUAL": "`=`",
        "SEMI": "`;`",
        "EULER": "`e`",
        "PI": "`pi`",
        "XS": "`x`",
        "XA": "`X`",
        "IDX": "`I`",
        "LN": "`ln`",
        "LOG2": "`log2`",
        "LOG": "`log`",
        "BASE": "`base`",
        "EXP": "`exp`",
        "EXP2": "`exp2`",
        "ID": "identifier",
        "SUM": "`sum`",
        "MATMUL": "`matmul`",
        "TRANSPOSE": "`.T`",
        "CS": "`c`",
        "CA": "`C`",
        "QUOTEDID": "quoted identifier",
        "QUOTE": '`"`',
        "VS": "`v`",
        "VA": "`V`",
        "RETURN": "`return`",
        "SIGN": "`sign`",
        "FLOOR": "`floor`",
        "CEIL": "`ceil`",
        "TRUNC": "`trunc`",
        "ROUND_TIES_EVEN": "`round_ties_even`",
        "SQRT": "`sqrt`",
        "SQUARE": "`square`",
        "SIN": "`sin`",
        "COS": "`cos`",
        "TAN": "`tan`",
        "COT": "`cot`",
        "SEC": "`sec`",
        "CSC": "`csc`",
        "ASIN": "`asin`",
        "ACOS": "`acos`",
        "ATAN": "`atan`",
        "ACOT": "`acot`",
        "ASEC": "`asec`",
        "ACSC": "`acsc`",
        "SINH": "`sinh`",
        "COSH": "`cosh`",
        "TANH": "`tanh`",
        "COTH": "`coth`",
        "SECH": "`sech`",
        "CSCH": "`csch`",
        "ASINH": "`asinh`",
        "ACOSH": "`acosh`",
        "ATANH": "`atanh`",
        "ACOTH": "`acoth`",
        "ASECH": "`asech`",
        "ACSCH": "`acsch`",
        "FINITE_DIFFERENCE": "`finite_difference`",
        "ORDER": "`order`",
        "ACCURACY": "`accuracy`",
        "TYPE": "`type`",
        "AXIS": "`axis`",
        "GRID_SPACING": "`grid_spacing`",
        "GRID_CENTRE": "`grid_centre`",
        "GRID_PERIOD": "`grid_period`",
    }.get(token, f"<{token}>")
