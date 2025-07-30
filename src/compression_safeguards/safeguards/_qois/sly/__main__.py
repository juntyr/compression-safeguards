import numpy as np

from .expr.array import Array
from .expr.constfold import FoldedScalarConst
from .expr.data import Data
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
            expr = parser.parse(text, lexer.tokenize(text))
            if expr is None:
                continue
            print(f"parsed: {expr!r}")
            X = np.array([1.0, 2.0, 3.0])
            expr = FoldedScalarConst.constant_fold_expr(expr, X.dtype)
            print(f"folded: {expr!r}")
            print(f"eval: {expr.eval(X, dict())}")
        except Exception as err:
            print(f"{type(err).__name__}: {err}")
