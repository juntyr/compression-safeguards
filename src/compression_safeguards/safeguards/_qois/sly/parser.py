import itertools

from sly import Parser

from ....utils.bindings import Parameter
from .expr.abc import Expr
from .expr.add import ScalarAdd
from .expr.array import Array
from .expr.data import Data, LateBoundConstant
from .expr.divmul import ScalarDivide, ScalarMultiply
from .expr.group import Group
from .expr.literal import Euler, Number, Pi
from .expr.logexp import ScalarExp, ScalarLn
from .expr.neg import ScalarNegate
from .expr.power import ScalarExponentiation
from .expr.round import ScalarCeil, ScalarFloor, ScalarRoundTiesEven, ScalarTrunc
from .expr.sign import ScalarSign
from .lexer import QoILexer


class QoIParser(Parser):
    __slots__ = ("_x", "_X", "_vars", "_text")
    _x: Data
    _X: None | Array
    _vars: dict[Parameter, Expr]
    _text: str

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

    @_("many_assign RETURN expr")  # type: ignore[name-defined, no-redef]  # noqa: F821
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

    @_("VS LBRACK QUOTE ID QUOTE RBRACK EQUAL expr SEMI")  # type: ignore[name-defined, no-redef]  # noqa: F821
    def assign(self, p):  # noqa: F811
        assert self._X is None
        assert p.ID not in self._vars
        self._vars[Parameter(p.ID)] = Group(p.expr)

    @_("VA LBRACK QUOTE ID QUOTE RBRACK EQUAL expr SEMI")  # type: ignore[name-defined, no-redef]  # noqa: F821
    def assign(self, p):  # noqa: F811
        assert self._X is not None
        assert p.ID not in self._vars
        self._vars[Parameter(p.ID)] = Group(p.expr)

    @_("VS LBRACK QUOTE ID QUOTE RBRACK")  # type: ignore[name-defined, no-redef]  # noqa: F821
    def expr(self, p):  # noqa: F811
        assert self._X is None
        return self._vars[Parameter(p.ID)]

    @_("VA LBRACK QUOTE ID QUOTE RBRACK")  # type: ignore[name-defined, no-redef]  # noqa: F821
    def expr(self, p):  # noqa: F811
        assert self._X is not None
        return self._vars[Parameter(p.ID)]

    @_("expr PLUS expr")  # type: ignore[name-defined, no-redef]  # noqa: F821
    def expr(self, p):  # noqa: F811
        return Array.map_binary(p.expr0, p.expr1, ScalarAdd)

    @_("expr MINUS expr")  # type: ignore[name-defined, no-redef]  # noqa: F821
    def expr(self, p):  # noqa: F811
        return Array.map_binary(
            p.expr0, Array.map_unary(p.expr1, ScalarNegate), ScalarAdd
        )

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

    @_("LN LPAREN expr RPAREN")  # type: ignore[name-defined, no-redef]  # noqa: F821
    def expr(self, p):  # noqa: F811
        return Array.map_unary(p.expr, ScalarLn)

    @_("EXP LPAREN expr RPAREN")  # type: ignore[name-defined, no-redef]  # noqa: F821
    def expr(self, p):  # noqa: F811
        return Array.map_unary(p.expr, ScalarExp)

    @_("SIGN LPAREN expr RPAREN")  # type: ignore[name-defined, no-redef]  # noqa: F821
    def expr(self, p):  # noqa: F811
        return Array.map_unary(p.expr, ScalarSign)

    @_("FLOOR LPAREN expr RPAREN")  # type: ignore[name-defined, no-redef]  # noqa: F821
    def expr(self, p):  # noqa: F811
        return Array.map_unary(p.expr, ScalarFloor)

    @_("CEIL LPAREN expr RPAREN")  # type: ignore[name-defined, no-redef]  # noqa: F821
    def expr(self, p):  # noqa: F811
        return Array.map_unary(p.expr, ScalarCeil)

    @_("TRUNC LPAREN expr RPAREN")  # type: ignore[name-defined, no-redef]  # noqa: F821
    def expr(self, p):  # noqa: F811
        return Array.map_unary(p.expr, ScalarTrunc)

    @_("ROUND_TIES_EVEN LPAREN expr RPAREN")  # type: ignore[name-defined, no-redef]  # noqa: F821
    def expr(self, p):  # noqa: F811
        return Array.map_unary(p.expr, ScalarRoundTiesEven)

    # @_("ID LPAREN expr many_comma_expr RPAREN")  # type: ignore[name-defined, no-redef]  # noqa: F821
    # def expr(self, p):  # noqa: F811
    #     assert False, f"unknown function `{p.ID}`"

    @_("SUM LPAREN expr RPAREN")  # type: ignore[name-defined, no-redef]  # noqa: F821
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

    @_("CS LBRACK QUOTE maybe_dollar ID QUOTE RBRACK")  # type: ignore[name-defined, no-redef]  # noqa: F821
    def expr(self, p):  # noqa: F811
        return LateBoundConstant.like(Parameter(p.maybe_dollar + p.ID), self._x)

    @_("CA LBRACK QUOTE maybe_dollar ID QUOTE RBRACK")  # type: ignore[name-defined, no-redef]  # noqa: F821
    def expr(self, p):  # noqa: F811
        assert self._X is not None
        return Array.map_unary(
            self._X,
            lambda e: LateBoundConstant.like(Parameter(p.maybe_dollar + p.ID), e),
        )

    @_("DOLLAR")  # type: ignore[name-defined, no-redef]  # noqa: F821
    def maybe_dollar(self, p):  # noqa: F811
        return "$"

    @_("empty")  # type: ignore[name-defined, no-redef]  # noqa: F821
    def maybe_dollar(self, p):  # noqa: F811
        return ""

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
        "EXP": "`exp`",
        "ID": "identifier",
        "SUM": "`sum`",
        "TRANSPOSE": "`.T`",
        "CS": "`c`",
        "CA": "`C`",
        "QUOTE": '`"`',
        "DOLLAR": "`$`",
        "VS": "`v`",
        "VA": "`V`",
        "RETURN": "`return`",
        "SIGN": "`sign`",
        "FLOOR": "`floor`",
        "CEIL": "`ceil`",
        "TRUNC": "`trunc`",
        "ROUND_TIES_EVEN": "`round_ties_even`",
    }.get(token, f"<{token}>")
