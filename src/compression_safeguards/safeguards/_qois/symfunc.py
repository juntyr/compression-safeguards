import sympy as sp


class trunc(sp.Function):
    """
    trunc(x)

    The trunc function rounds x towards zero.

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

    The sign function computes the sign of x.

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

    The round_ties_even function rounds x to the nearest integer,
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


class symmetric_modulo(sp.Function):
    """
    symmetric_modulo(p, q)

    The symmetric_modulo function computes r = p % q and guarantees that
    -q/2 <= r < q/2.

    symmetric_modulo can be evaluated by using the doit() method.
    """

    @classmethod
    def eval(cls, p, q) -> None | sp.Number:
        m = sp.Mod(p + (q / 2), q) - (q / 2)
        if isinstance(m, sp.Number):
            return m
        return None

    def _eval_evalf(self, prec) -> sp.Float:
        return self.doit(deep=False)._eval_evalf(prec)

    def doit(self, deep=True, **hints) -> sp.Float:
        (p, q) = self.args

        if deep:
            p = p.doit(deep=deep, **hints)
            q = q.doit(deep=deep, **hints)

        return sp.Mod(p + (q / 2), q) - (q / 2)  # type: ignore


class identity(sp.Function):
    pass


class ordered_sum(sp.Function):
    @classmethod
    def eval(cls, *xs):
        if all(len(x.free_symbols) == 0 for x in xs):
            return sp.Add(*xs)
        return None

    def _eval_evalf(self, prec) -> sp.Float:
        return self.doit(deep=False)._eval_evalf(prec)

    def doit(self, deep=True, **hints):
        xs = self.args

        if deep:
            xs = (x.doit(deep=deep, **hints) for x in xs)

        return sp.Add(*xs)
