import sympy as sp


class trunc(sp.Function):
    """
    trunc(x)

    The trunc functions rounds x towards zero.

    trunc can be evaluated by using the doit() method.
    """

    is_integer = True

    @classmethod
    def eval(cls, x) -> None | sp.Integer:
        if isinstance(x, sp.Number):
            return sp.Integer(x)
        return None

    def _eval_evalf(self, prec) -> sp.Float:
        return self.doit(deep=False)._eval_evalf(prec)

    def doit(self, deep=True, **hints) -> sp.Integer:
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
    def eval(cls, x) -> None | sp.Integer:
        if isinstance(x, sp.Number):
            return sp.sign(x)
        return None

    def _eval_evalf(self, prec) -> sp.Float:
        return self.doit(deep=False)._eval_evalf(prec)

    def doit(self, deep=True, **hints) -> sp.Integer:
        (x,) = self.args

        if deep:
            x = x.doit(deep=deep, **hints)

        return sp.sign(x)


class round_ties_even(sp.Function):
    """
    round_ties_even(x)

    The round_ties_even functions rounds x to the nearest integer,
    rounding ties to the nearest even integer.

    round_ties_even can be evaluated by using the doit() method.
    """

    is_integer = True

    @classmethod
    def eval(cls, x) -> None | sp.Integer:
        if isinstance(x, sp.Number):
            return x.round()
        return None

    def _eval_evalf(self, prec) -> sp.Float:
        return self.doit(deep=False)._eval_evalf(prec)

    def doit(self, deep=True, **hints) -> sp.Integer:
        (x,) = self.args

        if deep:
            x = x.doit(deep=deep, **hints)

        return x.round()  # type: ignore
