import itertools

from sly import Parser  # from sly.yacc import SlyLogger

from ....utils.bindings import Parameter
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
from .expr.power import ScalarExponentiation, ScalarSqrt, ScalarSquare
from .expr.round import ScalarCeil, ScalarFloor, ScalarRoundTiesEven, ScalarTrunc
from .expr.sign import ScalarSign
from .lexer import QoILexer


class NullWriter:
    def write(self, s):
        pass


class QoIParser(Parser):
    __slots__ = ("_x", "_X", "_vars", "_text")
    _x: Data
    _X: None | Array
    _vars: dict[Parameter, Expr]
    _text: str

    # log = SlyLogger(NullWriter())

    def __init__(self, *, x: Data, X: None | Array):
        self._x = x
        self._X = X
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
        self._vars[Parameter(p.quotedid)] = Group(p.expr)

    @_("VA LBRACK quotedid RBRACK EQUAL expr SEMI")  # type: ignore[name-defined, no-redef]  # noqa: F821
    def assign(self, p):  # noqa: F811
        assert self._X is not None
        assert not p.quotedid.startswith("$")
        assert p.quotedid not in self._vars
        self._vars[Parameter(p.quotedid)] = Group(p.expr)

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
        return Array.map_binary(p.expr0, p.expr1, ScalarExponentiation)

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

    @_("expr LBRACK index_ many_comma_index RBRACK %prec INDEX")  # type: ignore[name-defined, no-redef]  # noqa: F821
    def expr(self, p):  # noqa: F811
        assert isinstance(p.expr, Array), "only array expressions can be indexed"
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

    @_("NUMBER")  # type: ignore[name-defined]  # noqa: F821
    def index_(self, p):
        return int(p.NUMBER)

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

    @_("NUMBER")  # type: ignore[name-defined, no-redef]  # noqa: F821
    def expr(self, p):  # noqa: F811
        return Number(p.NUMBER)

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
        acc = None
        for e in p.expr._array.flat:
            acc = e if acc is None else ScalarAdd(acc, e)
        assert acc is not None
        return Group(acc)

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
        "NUMBER": "number",
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
        "LN": "`ln`",
        "LOG2": "`log2`",
        "LOG": "`log`",
        "BASE": "`base`",
        "EXP": "`exp`",
        "EXP2": "`exp2`",
        "ID": "identifier",
        "SUM": "`sum`",
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
    }.get(token, f"<{token}>")
