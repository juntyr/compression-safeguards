import numpy as np

from .expr import Array, Data, FoldedScalarConst
from .lexer import QoILexer
from .parser import QoIParser

if __name__ == "__main__":
    lexer = QoILexer()
    parser = QoIParser(
        x=Data(index=(1,)),
        X=Array(Data(index=(0,)), Data(index=(1,)), Data(index=(2,))),
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
            X = np.array([1.0, 2.0, 3.0])
            expr = expr.constant_fold(X.dtype)
            expr = FoldedScalarConst(expr) if isinstance(expr, X.dtype.type) else expr
            print(expr.eval(X))
        except AssertionError as err:
            print(err)
