import numpy as np

from ..qois import StencilQuantityOfInterestExpression
from . import StencilQuantityOfInterest
from .expr.constfold import ScalarFoldedConstant

if __name__ == "__main__":
    while True:
        try:
            qoi = input("qoi > ")
        except EOFError:
            break
        if len(qoi) == 0:
            break
        try:
            qoi_expr = StencilQuantityOfInterest(
                StencilQuantityOfInterestExpression(qoi),
                stencil_shape=(3,),
                stencil_I=(1,),
            )
            if qoi_expr is None:
                continue
            print(f"parsed: {qoi_expr!r}")  # noqa: T201
            Xs = np.array([[1.0, 2.0, 3.0]])
            print(  # noqa: T201
                f"folded: {ScalarFoldedConstant.constant_fold_expr(qoi_expr._expr, Xs.dtype)!r}"
            )
            print(f"eval: {qoi_expr.eval(Xs, dict())}")  # noqa: T201
        except Exception as err:
            print(f"{type(err).__name__}: {err}")  # noqa: T201
