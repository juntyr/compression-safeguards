__all__ = ["QoILexer"]

from sly import Lexer


class QoILexer(Lexer):
    tokens = {
        NUMBER,
        PLUS,
        TIMES,
        MINUS,
        DIVIDE,
        LPAREN,
        RPAREN,
        POWER,
        LBRACK,
        RBRACK,
        COMMA,
        EULER,
        PI,
        XS,
        XA,
        LN,
        EXP,
        ID,
        SUM,
        TRANSPOSE,
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

    # Identifiers
    ID = r"[a-zA-Z_][a-zA-Z0-9_]*"
    ID["e"] = EULER
    ID["pi"] = PI
    ID["x"] = XS
    ID["X"] = XA
    ID["ln"] = LN
    ID["exp"] = EXP
    ID["sum"] = SUM

    # Ignored pattern
    ignore_newline = r"\n+"
    ignore_comment = r"#[^\n]*"

    # Extra action for newlines
    def ignore_newline(self, t):
        self.lineno += t.value.count("\n")

    def error(self, t):
        print("Illegal character '%s'" % t.value[0])
        self.index += 1
