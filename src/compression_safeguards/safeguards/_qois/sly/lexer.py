__all__ = ["QoILexer"]

from sly import Lexer


class QoILexer(Lexer):
    tokens = {
        NUMBER,  # type: ignore[name-defined]  # noqa: F821
        PLUS,  # type: ignore[name-defined]  # noqa: F821
        TIMES,  # type: ignore[name-defined]  # noqa: F821
        MINUS,  # type: ignore[name-defined]  # noqa: F821
        DIVIDE,  # type: ignore[name-defined]  # noqa: F821
        LPAREN,  # type: ignore[name-defined]  # noqa: F821
        RPAREN,  # type: ignore[name-defined]  # noqa: F821
        POWER,  # type: ignore[name-defined]  # noqa: F821
        LBRACK,  # type: ignore[name-defined]  # noqa: F821
        RBRACK,  # type: ignore[name-defined]  # noqa: F821
        COMMA,  # type: ignore[name-defined]  # noqa: F821
        EQUAL,  # type: ignore[name-defined]  # noqa: F821
        SEMI,  # type: ignore[name-defined]  # noqa: F821
        EULER,  # type: ignore[name-defined]  # noqa: F821
        PI,  # type: ignore[name-defined]  # noqa: F821
        XS,  # type: ignore[name-defined]  # noqa: F821
        XA,  # type: ignore[name-defined]  # noqa: F821
        LN,  # type: ignore[name-defined]  # noqa: F821
        LOG2,  # type: ignore[name-defined]  # noqa: F821
        LOG,  # type: ignore[name-defined]  # noqa: F821
        BASE,  # type: ignore[name-defined]  # noqa: F821
        EXP,  # type: ignore[name-defined]  # noqa: F821
        EXP2,  # type: ignore[name-defined]  # noqa: F821
        ID,  # type: ignore[name-defined]  # noqa: F821
        SUM,  # type: ignore[name-defined]  # noqa: F821
        TRANSPOSE,  # type: ignore[name-defined]  # noqa: F821
        CS,  # type: ignore[name-defined]  # noqa: F821
        CA,  # type: ignore[name-defined]  # noqa: F821
        QUOTEDID,  # type: ignore[name-defined]  # noqa: F821
        QUOTE,  # type: ignore[name-defined]  # noqa: F821
        VS,  # type: ignore[name-defined]  # noqa: F821
        VA,  # type: ignore[name-defined]  # noqa: F821
        RETURN,  # type: ignore[name-defined]  # noqa: F821
        SIGN,  # type: ignore[name-defined]  # noqa: F821
        FLOOR,  # type: ignore[name-defined]  # noqa: F821
        CEIL,  # type: ignore[name-defined]  # noqa: F821
        TRUNC,  # type: ignore[name-defined]  # noqa: F821
        ROUND_TIES_EVEN,  # type: ignore[name-defined]  # noqa: F821
        SQRT,  # type: ignore[name-defined]  # noqa: F821
        SQUARE,  # type: ignore[name-defined]  # noqa: F821
        SIN,  # type: ignore[name-defined]  # noqa: F821
        COS,  # type: ignore[name-defined]  # noqa: F821
        TAN,  # type: ignore[name-defined]  # noqa: F821
        COT,  # type: ignore[name-defined]  # noqa: F821
        SEC,  # type: ignore[name-defined]  # noqa: F821
        CSC,  # type: ignore[name-defined]  # noqa: F821
        ASIN,  # type: ignore[name-defined]  # noqa: F821
        ACOS,  # type: ignore[name-defined]  # noqa: F821
        ATAN,  # type: ignore[name-defined]  # noqa: F821
        ACOT,  # type: ignore[name-defined]  # noqa: F821
        ASEC,  # type: ignore[name-defined]  # noqa: F821
        ACSC,  # type: ignore[name-defined]  # noqa: F821
        SINH,  # type: ignore[name-defined]  # noqa: F821
        COSH,  # type: ignore[name-defined]  # noqa: F821
        TANH,  # type: ignore[name-defined]  # noqa: F821
        COTH,  # type: ignore[name-defined]  # noqa: F821
        SECH,  # type: ignore[name-defined]  # noqa: F821
        CSCH,  # type: ignore[name-defined]  # noqa: F821
        ASINH,  # type: ignore[name-defined]  # noqa: F821
        ACOSH,  # type: ignore[name-defined]  # noqa: F821
        ATANH,  # type: ignore[name-defined]  # noqa: F821
        ACOTH,  # type: ignore[name-defined]  # noqa: F821
        ASECH,  # type: ignore[name-defined]  # noqa: F821
        ACSCH,  # type: ignore[name-defined]  # noqa: F821
    }
    ignore = " \t"

    # Tokens
    NUMBER = r"[0-9]+(\.[0-9]+)?(e(\+|-)?[0-9]+)?"

    # Special symbols
    PLUS = r"\+"
    MINUS = r"-"
    POWER = r"\*\*"
    TIMES = r"\*"
    DIVIDE = r"/"
    LPAREN = r"\("
    RPAREN = r"\)"
    LBRACK = r"\["
    RBRACK = r"\]"
    COMMA = r","
    TRANSPOSE = r"\.T"
    EQUAL = r"="
    SEMI = r";"

    @_(r'"(\$)?[a-zA-Z_][a-zA-Z0-9_]*"')  # type: ignore[name-defined]  # noqa: F821
    def QUOTEDID(self, t):
        t.value = t.value[1:-1]
        return t

    QUOTE = r'"'

    # Identifiers
    ID = r"[a-zA-Z_][a-zA-Z0-9_]*"
    ID["e"] = EULER  # type: ignore[index, name-defined]  # noqa: F821
    ID["pi"] = PI  # type: ignore[index, name-defined]  # noqa: F821
    ID["x"] = XS  # type: ignore[index, name-defined]  # noqa: F821
    ID["X"] = XA  # type: ignore[index, name-defined]  # noqa: F821
    ID["ln"] = LN  # type: ignore[index, name-defined]  # noqa: F821
    ID["log2"] = LOG2  # type: ignore[index, name-defined]  # noqa: F821
    ID["log"] = LOG  # type: ignore[index, name-defined]  # noqa: F821
    ID["base"] = BASE  # type: ignore[index, name-defined]  # noqa: F821
    ID["exp"] = EXP  # type: ignore[index, name-defined]  # noqa: F821
    ID["exp2"] = EXP2  # type: ignore[index, name-defined]  # noqa: F821
    ID["sum"] = SUM  # type: ignore[index, name-defined]  # noqa: F821
    ID["c"] = CS  # type: ignore[index, name-defined]  # noqa: F821
    ID["C"] = CA  # type: ignore[index, name-defined]  # noqa: F821
    ID["v"] = VS  # type: ignore[index, name-defined]  # noqa: F821
    ID["V"] = VA  # type: ignore[index, name-defined]  # noqa: F821
    ID["return"] = RETURN  # type: ignore[index, name-defined]  # noqa: F821
    ID["sign"] = SIGN  # type: ignore[index, name-defined]  # noqa: F821
    ID["floor"] = FLOOR  # type: ignore[index, name-defined]  # noqa: F821
    ID["ceil"] = CEIL  # type: ignore[index, name-defined]  # noqa: F821
    ID["trunc"] = TRUNC  # type: ignore[index, name-defined]  # noqa: F821
    ID["round_ties_even"] = ROUND_TIES_EVEN  # type: ignore[index, name-defined]  # noqa: F821
    ID["sqrt"] = SQRT  # type: ignore[index, name-defined]  # noqa: F821
    ID["square"] = SQRT  # type: ignore[index, name-defined]  # noqa: F821
    ID["sin"] = SIN  # type: ignore[index, name-defined]  # noqa: F821
    ID["cos"] = COS  # type: ignore[index, name-defined]  # noqa: F821
    ID["tan"] = TAN  # type: ignore[index, name-defined]  # noqa: F821
    ID["cot"] = COT  # type: ignore[index, name-defined]  # noqa: F821
    ID["sec"] = SEC  # type: ignore[index, name-defined]  # noqa: F821
    ID["csc"] = CSC  # type: ignore[index, name-defined]  # noqa: F821
    ID["asin"] = ASIN  # type: ignore[index, name-defined]  # noqa: F821
    ID["acos"] = ACOS  # type: ignore[index, name-defined]  # noqa: F821
    ID["atan"] = ATAN  # type: ignore[index, name-defined]  # noqa: F821
    ID["acot"] = ACOT  # type: ignore[index, name-defined]  # noqa: F821
    ID["asec"] = ASEC  # type: ignore[index, name-defined]  # noqa: F821
    ID["acsc"] = ACSC  # type: ignore[index, name-defined]  # noqa: F821
    ID["sinh"] = SINH  # type: ignore[index, name-defined]  # noqa: F821
    ID["cosh"] = COSH  # type: ignore[index, name-defined]  # noqa: F821
    ID["tanh"] = TANH  # type: ignore[index, name-defined]  # noqa: F821
    ID["coth"] = COTH  # type: ignore[index, name-defined]  # noqa: F821
    ID["sech"] = SECH  # type: ignore[index, name-defined]  # noqa: F821
    ID["csch"] = CSCH  # type: ignore[index, name-defined]  # noqa: F821
    ID["asinh"] = ASINH  # type: ignore[index, name-defined]  # noqa: F821
    ID["acosh"] = ACOSH  # type: ignore[index, name-defined]  # noqa: F821
    ID["atanh"] = ATANH  # type: ignore[index, name-defined]  # noqa: F821
    ID["acoth"] = ACOTH  # type: ignore[index, name-defined]  # noqa: F821
    ID["asech"] = ASECH  # type: ignore[index, name-defined]  # noqa: F821
    ID["acsch"] = ACSCH  # type: ignore[index, name-defined]  # noqa: F821

    # Ignored pattern
    ignore_newline = r"\n+"
    ignore_comment = r"#[^\n]*"

    # Extra action for newlines
    def ignore_newline(self, t):  # type: ignore[no-redef]  # noqa: F811
        self.lineno += t.value.count("\n")

    def error(self, t):
        raise SyntaxError(
            f"illegal character `{t.value[0]}` at line {self.lineno}, column {find_column(self.text, t)}"
        )


def find_column(text, token):
    last_cr = text.rfind("\n", 0, token.index)
    if last_cr < 0:
        last_cr = 0
    column = (token.index - last_cr) + 1
    return column
