import sympy as sp
from typing_extensions import Never  # MSPV 3.11

from .array import NumPyLikeArray


def asum(x, /):
    assert isinstance(x, NumPyLikeArray), "can only compute the sum over an array"
    return sum(sp.tensor.array.arrayop.Flatten(x), sp.Integer(0))


def tr(x, /):
    assert isinstance(x, NumPyLikeArray) and x.rank() == 2, (
        "can only compute the transpose over a matrix (2d array)"
    )
    return x.transpose()


def matmul(a, b, /):
    assert isinstance(a, NumPyLikeArray) and a.rank() == 2, (
        "matmul can only multiply matrices (2d arrays), a is not"
    )
    assert isinstance(b, NumPyLikeArray) and b.rank() == 2, (
        "matmul can only multiply matrices (2d arrays), b is not"
    )
    assert a.shape[1] == b.shape[0], (
        "matmul shapes do not match (n x k) x (k x m) -> (n x m)"
    )
    result = sp.tensorcontraction(sp.tensorproduct(a, b), (1, 2))
    result.__class__ = NumPyLikeArray
    return result


FUNCTIONS = dict(asum=asum, tr=tr, matmul=matmul)


class ArrayLiteral:
    __slots__ = ()

    def __new__(cls, *args, **kwargs) -> Never:
        raise TypeError("cannot call array constructor")

    def __class_getitem__(cls, index) -> NumPyLikeArray:
        return NumPyLikeArray(index if isinstance(index, tuple) else (index,))


CONSTRUCTORS = dict(A=ArrayLiteral)
