__all__ = ["QoILexer"]

from contextlib import contextmanager

from sly import Lexer


class QoILexer(Lexer):
    # === token declarations ===
    tokens = {
        # literals
        INTEGER,  # type: ignore[name-defined]  # noqa: F821
        FLOAT,  # type: ignore[name-defined]  # noqa: F821
        STRING,  # type: ignore[name-defined]  # noqa: F821
        # operators
        PLUS,  # type: ignore[name-defined]  # noqa: F821
        MINUS,  # type: ignore[name-defined]  # noqa: F821
        POWER,  # type: ignore[name-defined]  # noqa: F821
        TIMES,  # type: ignore[name-defined]  # noqa: F821
        DIVIDE,  # type: ignore[name-defined]  # noqa: F821
        EQUAL,  # type: ignore[name-defined]  # noqa: F821
        # array transpose
        TRANSPOSE,  # type: ignore[name-defined]  # noqa: F821
        # groups
        LPAREN,  # type: ignore[name-defined]  # noqa: F821
        RPAREN,  # type: ignore[name-defined]  # noqa: F821
        LBRACK,  # type: ignore[name-defined]  # noqa: F821
        RBRACK,  # type: ignore[name-defined]  # noqa: F821
        # separators
        COMMA,  # type: ignore[name-defined]  # noqa: F821
        SEMI,  # type: ignore[name-defined]  # noqa: F821
        # identifiers
        ID,  # type: ignore[name-defined]  # noqa: F821
        # statements
        RETURN,  # type: ignore[name-defined]  # noqa: F821
        # constants
        EULER,  # type: ignore[name-defined]  # noqa: F821
        PI,  # type: ignore[name-defined]  # noqa: F821
        # data, late-bound constants, variables
        XS,  # type: ignore[name-defined]  # noqa: F821
        XA,  # type: ignore[name-defined]  # noqa: F821
        CS,  # type: ignore[name-defined]  # noqa: F821
        CA,  # type: ignore[name-defined]  # noqa: F821
        VS,  # type: ignore[name-defined]  # noqa: F821
        VA,  # type: ignore[name-defined]  # noqa: F821
        # array indexing
        IDX,  # type: ignore[name-defined]  # noqa: F821
        # functions
        # logarithms and exponentials
        LN,  # type: ignore[name-defined]  # noqa: F821
        LOG2,  # type: ignore[name-defined]  # noqa: F821
        LOG,  # type: ignore[name-defined]  # noqa: F821
        EXP,  # type: ignore[name-defined]  # noqa: F821
        EXP2,  # type: ignore[name-defined]  # noqa: F821
        # exponentiation
        SQRT,  # type: ignore[name-defined]  # noqa: F821
        SQUARE,  # type: ignore[name-defined]  # noqa: F821
        # absolute value
        ABS,  # type: ignore[name-defined]  # noqa: F821
        # sign and rounding
        SIGN,  # type: ignore[name-defined]  # noqa: F821
        FLOOR,  # type: ignore[name-defined]  # noqa: F821
        CEIL,  # type: ignore[name-defined]  # noqa: F821
        TRUNC,  # type: ignore[name-defined]  # noqa: F821
        ROUND_TIES_EVEN,  # type: ignore[name-defined]  # noqa: F821
        # trigonometric
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
        # hyperbolic
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
        # array operations
        SUM,  # type: ignore[name-defined]  # noqa: F821
        MATMUL,  # type: ignore[name-defined]  # noqa: F821
        # finite difference
        FINITE_DIFFERENCE,  # type: ignore[name-defined]  # noqa: F821
        # keyword arguments
        BASE,  # type: ignore[name-defined]  # noqa: F821
        ORDER,  # type: ignore[name-defined]  # noqa: F821
        ACCURACY,  # type: ignore[name-defined]  # noqa: F821
        TYPE,  # type: ignore[name-defined]  # noqa: F821
        AXIS,  # type: ignore[name-defined]  # noqa: F821
        GRID_SPACING,  # type: ignore[name-defined]  # noqa: F821
        GRID_CENTRE,  # type: ignore[name-defined]  # noqa: F821
        GRID_PERIOD,  # type: ignore[name-defined]  # noqa: F821
    }

    # === ignored whitespace patterns ===
    ignore = " \t"
    ignore_comment = r"#[^\n]*"

    @_(r"\n+")  # type: ignore[name-defined]  # noqa: F821
    def ignore_newline(self, t):  # type: ignore[no-redef]  # noqa: F811
        self.lineno += t.value.count("\n")

    # === token definitions ===

    # literals
    @_(r"[0-9]+(\.[0-9]*)?(e(\+|-)?[0-9]*)?")  # type: ignore[name-defined]  # noqa: F821
    def FLOAT(self, t):
        if ("." in t.value) or ("e" in t.value):
            # floating-point literal
            self.assert_or_error(
                t.value[-1] in "0123456789",
                t,
                f"invalid floating point literal `{t.value}` requires at least one digit after '.' and 'e'",
            )
            return t
        else:
            # integer literal, pre-parsed
            with self.with_error_context(t, "excessive integer literal"):
                t.value = int(t.value)
            t.type = "INTEGER"
            return t

    @_(r'"[^"]*["]?')  # type: ignore[name-defined]  # noqa: F821
    def STRING(self, t):
        self.assert_or_error(
            t.value[-1] == '"', t, 'invalid string literal with missing closing `"`'
        )
        t.value = t.value[1:-1]
        return t

    # operators
    PLUS = r"\+"
    MINUS = r"-"
    POWER = r"\*\*"
    TIMES = r"\*"
    DIVIDE = r"/"
    EQUAL = r"="

    # array transpose
    TRANSPOSE = r"\.T"

    # groups
    LPAREN = r"\("
    RPAREN = r"\)"
    LBRACK = r"\["
    RBRACK = r"\]"

    # separators
    COMMA = r","
    SEMI = r";"

    # identifiers
    ID = r"[a-zA-Z_][a-zA-Z0-9_]*"

    # statements
    ID["return"] = RETURN  # type: ignore[index, name-defined]  # noqa: F821

    # constants
    ID["e"] = EULER  # type: ignore[index, name-defined]  # noqa: F821
    ID["pi"] = PI  # type: ignore[index, name-defined]  # noqa: F821

    # data, late-bound constants, variables
    ID["x"] = XS  # type: ignore[index, name-defined]  # noqa: F821
    ID["X"] = XA  # type: ignore[index, name-defined]  # noqa: F821
    ID["c"] = CS  # type: ignore[index, name-defined]  # noqa: F821
    ID["C"] = CA  # type: ignore[index, name-defined]  # noqa: F821
    ID["v"] = VS  # type: ignore[index, name-defined]  # noqa: F821
    ID["V"] = VA  # type: ignore[index, name-defined]  # noqa: F821

    # array indexing
    ID["I"] = IDX  # type: ignore[index, name-defined]  # noqa: F821

    # functions
    # logarithms and exponentials
    ID["ln"] = LN  # type: ignore[index, name-defined]  # noqa: F821
    ID["log2"] = LOG2  # type: ignore[index, name-defined]  # noqa: F821
    ID["log"] = LOG  # type: ignore[index, name-defined]  # noqa: F821
    ID["exp"] = EXP  # type: ignore[index, name-defined]  # noqa: F821
    ID["exp2"] = EXP2  # type: ignore[index, name-defined]  # noqa: F821
    # exponentiation
    ID["sqrt"] = SQRT  # type: ignore[index, name-defined]  # noqa: F821
    ID["square"] = SQUARE  # type: ignore[index, name-defined]  # noqa: F821
    # absolute value
    ID["abs"] = ABS  # type: ignore[index, name-defined]  # noqa: F821
    # sign and rounding
    ID["sign"] = SIGN  # type: ignore[index, name-defined]  # noqa: F821
    ID["floor"] = FLOOR  # type: ignore[index, name-defined]  # noqa: F821
    ID["ceil"] = CEIL  # type: ignore[index, name-defined]  # noqa: F821
    ID["trunc"] = TRUNC  # type: ignore[index, name-defined]  # noqa: F821
    ID["round_ties_even"] = ROUND_TIES_EVEN  # type: ignore[index, name-defined]  # noqa: F821
    # trigonometric
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
    # hypergeometric
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
    # array operations
    ID["sum"] = SUM  # type: ignore[index, name-defined]  # noqa: F821
    ID["matmul"] = MATMUL  # type: ignore[index, name-defined]  # noqa: F821
    # finite difference
    ID["finite_difference"] = FINITE_DIFFERENCE  # type: ignore[index, name-defined]  # noqa: F821

    # keyword arguments
    ID["base"] = BASE  # type: ignore[index, name-defined]  # noqa: F821
    ID["order"] = ORDER  # type: ignore[index, name-defined]  # noqa: F821
    ID["accuracy"] = ACCURACY  # type: ignore[index, name-defined]  # noqa: F821
    ID["type"] = TYPE  # type: ignore[index, name-defined]  # noqa: F821
    ID["axis"] = AXIS  # type: ignore[index, name-defined]  # noqa: F821
    ID["grid_spacing"] = GRID_SPACING  # type: ignore[index, name-defined]  # noqa: F821
    ID["grid_centre"] = GRID_CENTRE  # type: ignore[index, name-defined]  # noqa: F821
    ID["grid_period"] = GRID_PERIOD  # type: ignore[index, name-defined]  # noqa: F821

    # === lexer error handling ===
    def error(self, t):
        self.raise_error(t, f"unexpected character `{t.value[0]}`")

    def raise_error(self, t, message):
        raise SyntaxError(f"{message} at line {t.lineno}, column {self.find_column(t)}")

    def assert_or_error(self, check, t, message):
        if not check:
            self.raise_error(t, message)

    @contextmanager
    def with_error_context(self, t, message, exception=Exception):
        try:
            yield
        except exception as err:
            if callable(message):
                self.raise_error(t, message(err))
            else:
                self.raise_error(t, message)

    def find_column(self, token):
        last_cr = self.text.rfind("\n", 0, token.index)
        if last_cr < 0:
            last_cr = 0
        column = (token.index - last_cr) + 1
        return column

    @staticmethod
    def token_to_name(token: str) -> str:
        return {
            # literals
            "INTEGER": "integer",
            "FLOAT": "floating-point number",
            "STRING": "string",
            # operators
            "PLUS": "`+`",
            "MINUS": "`-`",
            "POWER": "`**`",
            "TIMES": "`*`",
            "DIVIDE": "`/`",
            "EQUAL": "`=`",
            # array transpose
            "TRANSPOSE": "`.T`",
            # groups
            "LPAREN": "`(`",
            "RPAREN": "`)`",
            "LBRACK": "`[`",
            "RBRACK": "`]`",
            # separators
            "COMMA": "`,`",
            "SEMI": "`;`",
            # identifiers
            "ID": "identifier",
            # statements
            "RETURN": "`return`",
            # constants
            "EULER": "`e`",
            "PI": "`pi`",
            # data, late-bound constants, variables
            "XS": "`x`",
            "XA": "`X`",
            "CS": "`c`",
            "CA": "`C`",
            "VS": "`v`",
            "VA": "`V`",
            # array indexing
            "IDX": "`I`",
            # functions
            # logarithms and exponentials
            "LN": "`ln`",
            "LOG2": "`log2`",
            "LOG": "`log`",
            "EXP": "`exp`",
            "EXP2": "`exp2`",
            # exponentiation
            "SQRT": "`sqrt`",
            "SQUARE": "`square`",
            # absolute value
            "ABS": "`abs`",
            # sign and rounding
            "SIGN": "`sign`",
            "FLOOR": "`floor`",
            "CEIL": "`ceil`",
            "TRUNC": "`trunc`",
            "ROUND_TIES_EVEN": "`round_ties_even`",
            # trigonometric
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
            # hyperbolic
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
            # array operations
            "SUM": "`sum`",
            "MATMUL": "`matmul`",
            # finite difference
            "FINITE_DIFFERENCE": "`finite_difference`",
            # keyword arguments
            "BASE": "`base`",
            "ORDER": "`order`",
            "ACCURACY": "`accuracy`",
            "TYPE": "`type`",
            "AXIS": "`axis`",
            "GRID_SPACING": "`grid_spacing`",
            "GRID_CENTRE": "`grid_centre`",
            "GRID_PERIOD": "`grid_period`",
        }.get(token, f"<{token}>")
