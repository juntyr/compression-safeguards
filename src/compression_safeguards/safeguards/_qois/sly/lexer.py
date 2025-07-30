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
        EXP,  # type: ignore[name-defined]  # noqa: F821
        ID,  # type: ignore[name-defined]  # noqa: F821
        SUM,  # type: ignore[name-defined]  # noqa: F821
        TRANSPOSE,  # type: ignore[name-defined]  # noqa: F821
        CS,  # type: ignore[name-defined]  # noqa: F821
        CA,  # type: ignore[name-defined]  # noqa: F821
        QUOTE,  # type: ignore[name-defined]  # noqa: F821
        DOLLAR,  # type: ignore[name-defined]  # noqa: F821
        VS,  # type: ignore[name-defined]  # noqa: F821
        VA,  # type: ignore[name-defined]  # noqa: F821
        RETURN,  # type: ignore[name-defined]  # noqa: F821
        SIGN,  # type: ignore[name-defined]  # noqa: F821
        FLOOR,  # type: ignore[name-defined]  # noqa: F821
        CEIL,  # type: ignore[name-defined]  # noqa: F821
        TRUNC,  # type: ignore[name-defined]  # noqa: F821
        ROUND_TIES_EVEN,  # type: ignore[name-defined]  # noqa: F821
        SINH,  # type: ignore[name-defined]  # noqa: F821
        COSH,  # type: ignore[name-defined]  # noqa: F821
        TANH,  # type: ignore[name-defined]  # noqa: F821
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
    QUOTE = r'"'
    DOLLAR = r"\$"
    EQUAL = r"="
    SEMI = r";"

    # Identifiers
    ID = r"[a-zA-Z_][a-zA-Z0-9_]*"
    ID["e"] = EULER  # type: ignore[index, name-defined]  # noqa: F821
    ID["pi"] = PI  # type: ignore[index, name-defined]  # noqa: F821
    ID["x"] = XS  # type: ignore[index, name-defined]  # noqa: F821
    ID["X"] = XA  # type: ignore[index, name-defined]  # noqa: F821
    ID["ln"] = LN  # type: ignore[index, name-defined]  # noqa: F821
    ID["exp"] = EXP  # type: ignore[index, name-defined]  # noqa: F821
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
    ID["sinh"] = SINH  # type: ignore[index, name-defined]  # noqa: F821
    ID["cosh"] = COSH  # type: ignore[index, name-defined]  # noqa: F821
    ID["tanh"] = TANH  # type: ignore[index, name-defined]  # noqa: F821

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
