from sly import Parser

from .expr import (
    Array,
    DataArrayElement,
    DataScalar,
    Euler,
    Group,
    Number,
    Pi,
    ScalarAdd,
    ScalarDivide,
    ScalarExp,
    ScalarExponentiation,
    ScalarLn,
    ScalarMultiply,
    ScalarNegate,
)
from .lexer import QoILexer


class QoIParser(Parser):
    def __init__(self, *, x: DataScalar | DataArrayElement, X: None | Array):
        self.x = x
        self.X = X

    tokens = QoILexer.tokens

    precedence = (
        ("left", PLUS, MINUS),  # type: ignore[name-defined]  # noqa: F821
        ("left", TIMES, DIVIDE),  # type: ignore[name-defined]  # noqa: F821
        ("right", UPLUS, UMINUS),  # type: ignore[name-defined]  # noqa: F821
        ("right", POWER),  # type: ignore[name-defined]  # noqa: F821
        # FIXME: index has wrong precedence
        ("left", INDEX, TRANSPOSE),  # type: ignore[name-defined]  # noqa: F821
    )

    @_("expr")  # type: ignore[name-defined]  # noqa: F821
    def qoi(self, p):
        assert not isinstance(p.expr, Array), (
            f"QoI expr must be scalar but is an array expression of shape {p.expr.shape}"
        )
        assert p.expr.has_data, "QoI expr must not be constant"
        return p.expr

    @_("expr PLUS expr")  # type: ignore[name-defined, no-redef]  # noqa: F821
    def expr(self, p):
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
        return self.x

    @_("XA")  # type: ignore[name-defined, no-redef]  # noqa: F821
    def expr(self, p):  # noqa: F811
        assert self.X is not None
        return self.X

    @_("LN LPAREN expr RPAREN")  # type: ignore[name-defined, no-redef]  # noqa: F821
    def expr(self, p):  # noqa: F811
        return Array.map_unary(p.expr, ScalarLn)

    @_("EXP LPAREN expr RPAREN")  # type: ignore[name-defined, no-redef]  # noqa: F821
    def expr(self, p):  # noqa: F811
        return Array.map_unary(p.expr, ScalarExp)

    @_("ID")  # type: ignore[name-defined, no-redef]  # noqa: F821
    def expr(self, p):  # noqa: F811
        assert False, f"unknown id {p.ID}"

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
