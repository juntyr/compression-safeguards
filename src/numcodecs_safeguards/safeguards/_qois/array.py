import sympy as sp


class NumPyLikeArray(sp.Array):
    __slots__ = ()

    def __add__(self, other):
        if isinstance(other, NumPyLikeArray):
            if self.shape != other.shape:
                raise ValueError("array shape mismatch")
            result_list = [
                i + j
                for i, j in zip(
                    sp.tensor.array.arrayop.Flatten(self),
                    sp.tensor.array.arrayop.Flatten(other),
                )
            ]
            return type(self)(result_list, self.shape)
        other = sp.sympify(other)
        result_list = [i + other for i in sp.tensor.array.arrayop.Flatten(self)]
        return type(self)(result_list, self.shape)

    def __radd__(self, other):
        other = sp.sympify(other)
        result_list = [other + i for i in sp.tensor.array.arrayop.Flatten(self)]
        return type(self)(result_list, self.shape)

    def __sub__(self, other):
        if isinstance(other, NumPyLikeArray):
            if self.shape != other.shape:
                raise ValueError("array shape mismatch")
            result_list = [
                i - j
                for i, j in zip(
                    sp.tensor.array.arrayop.Flatten(self),
                    sp.tensor.array.arrayop.Flatten(other),
                )
            ]
            return type(self)(result_list, self.shape)
        other = sp.sympify(other)
        result_list = [i - other for i in sp.tensor.array.arrayop.Flatten(self)]
        return type(self)(result_list, self.shape)

    def __rsub__(self, other):
        other = sp.sympify(other)
        result_list = [other - i for i in sp.tensor.array.arrayop.Flatten(self)]
        return type(self)(result_list, self.shape)

    def __mul__(self, other):
        if isinstance(other, NumPyLikeArray):
            if self.shape != other.shape:
                raise ValueError("array shape mismatch")
            result_list = [
                i * j
                for i, j in zip(
                    sp.tensor.array.arrayop.Flatten(self),
                    sp.tensor.array.arrayop.Flatten(other),
                )
            ]
            return type(self)(result_list, self.shape)
        other = sp.sympify(other)
        result_list = [i * other for i in sp.tensor.array.arrayop.Flatten(self)]
        return type(self)(result_list, self.shape)

    def __rmul__(self, other):
        other = sp.sympify(other)
        result_list = [other * i for i in sp.tensor.array.arrayop.Flatten(self)]
        return type(self)(result_list, self.shape)

    def __truediv__(self, other):
        if isinstance(other, NumPyLikeArray):
            if self.shape != other.shape:
                raise ValueError("array shape mismatch")
            result_list = [
                i / j
                for i, j in zip(
                    sp.tensor.array.arrayop.Flatten(self),
                    sp.tensor.array.arrayop.Flatten(other),
                )
            ]
            return type(self)(result_list, self.shape)
        other = sp.sympify(other)
        result_list = [i / other for i in sp.tensor.array.arrayop.Flatten(self)]
        return type(self)(result_list, self.shape)

    def __rtruediv__(self, other):
        other = sp.sympify(other)
        result_list = [other / i for i in sp.tensor.array.arrayop.Flatten(self)]
        return type(self)(result_list, self.shape)

    def __pow__(self, other):
        if isinstance(other, NumPyLikeArray):
            if self.shape != other.shape:
                raise ValueError("array shape mismatch")
            result_list = [
                i**j
                for i, j in zip(
                    sp.tensor.array.arrayop.Flatten(self),
                    sp.tensor.array.arrayop.Flatten(other),
                )
            ]
            return type(self)(result_list, self.shape)
        other = sp.sympify(other)
        result_list = [i**other for i in sp.tensor.array.arrayop.Flatten(self)]
        return type(self)(result_list, self.shape)

    def __rpow__(self, other):
        other = sp.sympify(other)
        result_list = [other**i for i in sp.tensor.array.arrayop.Flatten(self)]
        return type(self)(result_list, self.shape)

    # TODO: also support "matrix" multiplication
