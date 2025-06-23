import re

QOI_INT_LITERAL_PATTERN = r"[+-]?[0-9]+"
QOI_FLOAT_LITERAL_PATTERN = r"[+-]?[0-9]+\.[0-9]+(?:e[+-]?[0-9]+)?"

QOI_COMMENT_PATTERN = re.compile(r"#[^\n]*")
QOI_WHITESPACE_PATTERN = re.compile(r"[ \t\n]+")
