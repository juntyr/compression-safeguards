from traceback import print_exception

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
            print(f"parsed: {qoi_expr!r}")  # noqa: T201
            Xs: np.ndarray[tuple[int, int], np.dtype[np.float64]] = (
                np.arange(3).astype(np.float64).reshape(1, 3)
            )
            print(  # noqa: T201
                f"folded: {ScalarFoldedConstant.constant_fold_expr(qoi_expr._expr, Xs.dtype)!r}"
            )
            print(f"eval: {qoi_expr.eval(Xs, dict())}")  # noqa: T201
        except Exception as err:
            print_exception(err, limit=0)
