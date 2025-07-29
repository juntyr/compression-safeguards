import numpy as np

from .lexer import QoILexer
from .parser import QoIParser
from .expr import DataArrayElement, Array, FoldedScalarConst

if __name__ == "__main__":
    lexer = QoILexer()
    parser = QoIParser(
        x=DataArrayElement((1,)),
        X=Array(DataArrayElement((0,)), DataArrayElement((1,)), DataArrayElement((2,))),
    )
    while True:
        try:
            text = input("qoi > ")
        except EOFError:
            break
        if len(text) == 0:
            break
        try:
            expr = parser.parse(lexer.tokenize(text))
            if expr is None:
                continue
            dtype = np.dtype(np.float64)
            expr = expr.constant_fold(dtype)
            expr = FoldedScalarConst(expr) if isinstance(expr, dtype.type) else expr
            print(expr.eval(dtype, np.array([1.0, 2.0, 3.0], dtype=dtype)))
        except AssertionError as err:
            print(err)
