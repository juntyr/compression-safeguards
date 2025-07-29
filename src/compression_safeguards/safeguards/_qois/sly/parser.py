import numpy as np
from sly import Parser

from .lexer import QoILexer
from .expr import (
    Expr,
    Array,
    ScalarAdd,
    ScalarMultiply,
    ScalarDivide,
    ScalarExp,
    ScalarLn,
    ScalarExponentiation,
    ScalarNegate,
    FoldedScalarConst,
    Number,
    Pi,
    Euler,
)


class QoIParser(Parser):
    def __init__(self, *, x: Expr, X: None | Expr):
        self.x = x
        self.X = X

    tokens = QoILexer.tokens

    precedence = (
        ("left", PLUS, MINUS),
        ("left", TIMES, DIVIDE),
        ("right", UPLUS, UMINUS),
        ("right", POWER),
        ("left", INDEX),
    )

    @_("expr")
    def qoi(self, p):
        assert not isinstance(p.expr, Array), (
            f"QoI expr must be scalar but is an array expression of shape {p.expr.shape}"
        )
        return p.expr

    @_("expr PLUS expr")
    def expr(self, p):
        return Array.map_binary(p.expr0, p.expr1, ScalarAdd)

    @_("expr MINUS expr")
    def expr(self, p):
        return Array.map_binary(
            p.expr0, Array.map_unary(p.expr1, ScalarNegate), ScalarAdd
        )

    @_("expr TIMES expr")
    def expr(self, p):
        return Array.map_binary(p.expr0, p.expr1, ScalarMultiply)

    @_("expr POWER expr")
    def expr(self, p):
        return Array.map_binary(p.expr0, p.expr1, ScalarExponentiation)

    @_("expr DIVIDE expr")
    def expr(self, p):
        return Array.map_binary(p.expr0, p.expr1, ScalarDivide)

    @_("PLUS expr %prec UPLUS")
    def expr(self, p):
        return p.expr

    @_("MINUS expr %prec UMINUS")
    def expr(self, p):
        return Array.map_unary(p.expr, ScalarNegate)

    @_("LPAREN expr RPAREN")
    def expr(self, p):
        return p.expr

    @_("")
    def empty(self, p):
        pass

    @_("LBRACK expr many_comma_expr RBRACK")
    def expr(self, p):
        return Array(*([p.expr] + p.many_comma_expr))

    @_("expr LBRACK index_ many_comma_index RBRACK %prec INDEX")
    def expr(self, p):
        assert isinstance(p.expr, Array), "only array expressions can be indexed"
        return p.expr.index(tuple([p.index_] + p.many_comma_index))

    @_("comma_index many_comma_index")
    def many_comma_index(self, p):
        return [p.comma_index] + p.many_comma_index

    @_("COMMA")
    def many_comma_index(self, p):
        return []

    @_("empty")
    def many_comma_index(self, p):
        return []

    @_("COMMA index_")
    def comma_index(self, p):
        return p.index_

    @_("NUMBER")
    def index_(self, p):
        return int(p.NUMBER)

    @_("comma_expr many_comma_expr")
    def many_comma_expr(self, p):
        return [p.comma_expr] + p.many_comma_expr

    @_("COMMA")
    def many_comma_expr(self, p):
        return []

    @_("empty")
    def many_comma_expr(self, p):
        return []

    @_("COMMA expr")
    def comma_expr(self, p):
        return p.expr

    @_("NUMBER")
    def expr(self, p):
        return Number(p.NUMBER)

    @_("EULER")
    def expr(self, p):
        return Euler()

    @_("PI")
    def expr(self, p):
        return Pi()

    @_("XS")
    def expr(self, p):
        return self.x

    @_("XA")
    def expr(self, p):
        assert self.X is not None
        return self.X

    @_("LN LPAREN expr RPAREN")
    def expr(self, p):
        return Array.map_unary(p.expr, ScalarLn)

    @_("EXP LPAREN expr RPAREN")
    def expr(self, p):
        return Array.map_unary(p.expr, ScalarExp)

    @_("ID")
    def expr(self, p):
        assert False, f"unknown id {p.ID}"

    @_("SUM LPAREN expr RPAREN")
    def expr(self, p):
        assert isinstance(p.expr, Array), "can only sum over an array expression"
        acc = None
        for e in p.expr._array.flat:
            acc = e if acc is None else ScalarAdd(acc, e)
        assert acc is not None
        return acc
