import sympy as sp

from .array import NumPyLikeArray


def asum(x, /):
    assert isinstance(x, NumPyLikeArray), "can only compute the sum over an array"
    return sum(sp.tensor.array.arrayop.Flatten(x), sp.Integer(0))


FUNCTIONS = dict(asum=asum)


class ArrayLiteral:
    __slots__ = ()

    def __new__(cls, *args, **kwargs):
        raise TypeError("cannot call array constructor")

    def __class_getitem__(cls, index):
        return NumPyLikeArray(index)


CONSTRUCTORS = dict(A=ArrayLiteral)
