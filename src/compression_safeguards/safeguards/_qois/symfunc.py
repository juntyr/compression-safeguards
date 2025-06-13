import sympy as sp


class trunc(sp.Function):
    """
    trunc(x)

    The trunc functions rounds x towards zero.

    trunc can be evaluated by using the doit() method.
    """

    is_integer = True

    @classmethod
    def eval(cls, x):
        if isinstance(x, sp.Number):
            return sp.Integer(x)

    def _eval_evalf(self, prec):
        return self.doit(deep=False)._eval_evalf(prec)

    def doit(self, deep=True, **hints):
        (x,) = self.args

        if deep:
            x = x.doit(deep=deep, **hints)

        return sp.Integer(x)


# FIXME: https://github.com/sympy/sympy/issues/28141
class sign(sp.Function):
    """
    sign(x)

    The sign functions computes the sign of x.

    sign can be evaluated by using the doit() method.
    """

    @classmethod
    def eval(cls, x):
        if isinstance(x, sp.Number):
            return sp.sign(x)

    def _eval_evalf(self, prec):
        return self.doit(deep=False)._eval_evalf(prec)

    def doit(self, deep=True, **hints):
        (x,) = self.args

        if deep:
            x = x.doit(deep=deep, **hints)

        return sp.sign(x)
