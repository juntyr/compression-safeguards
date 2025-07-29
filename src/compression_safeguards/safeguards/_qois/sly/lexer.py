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

    # Ignored pattern
    ignore_newline = r"\n+"
    ignore_comment = r"#[^\n]*"

    # Extra action for newlines
    def ignore_newline(self, t):  # type: ignore[no-redef]  # noqa: F811
        self.lineno += t.value.count("\n")

    def error(self, t):
        print("Illegal character '%s'" % t.value[0])
        self.index += 1
