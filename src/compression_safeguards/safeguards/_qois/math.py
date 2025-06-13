import sympy as sp

from .array import NumPyLikeArray
from .symfunc import sign as sp_sign
from .symfunc import trunc as sp_trunc


def sqrt(x, /):
    return x ** sp.Rational(1, 2)


def exp(x, /):
    return sp.E**x


def ln(x, /):
    if isinstance(x, NumPyLikeArray):
        return x.applyfunc(sp.ln)
    return sp.ln(x)


def log(x, /, *, base):
    if isinstance(x, NumPyLikeArray):
        ln_x = x.applyfunc(sp.ln)
    else:
        ln_x = sp.ln(x)
    if isinstance(base, NumPyLikeArray):
        ln_base = base.applyfunc(sp.ln)
    else:
        ln_base = sp.ln(base)
    return ln_x / ln_base


def sign(x, /):
    if isinstance(x, NumPyLikeArray):
        return x.applyfunc(sp_sign)
    return sp_sign(x)


def floor(x, /):
    if isinstance(x, NumPyLikeArray):
        return x.applyfunc(sp.floor)
    return sp.floor(x)


def ceil(x, /):
    if isinstance(x, NumPyLikeArray):
        return x.applyfunc(sp.ceiling)
    return sp.ceiling(x)


def trunc(x, /):
    if isinstance(x, NumPyLikeArray):
        return x.applyfunc(sp_trunc)
    return sp_trunc(x)


def sin(x, /):
    if isinstance(x, NumPyLikeArray):
        return x.applyfunc(sp.sin)
    return sp.sin(x)


def cos(x, /):
    if isinstance(x, NumPyLikeArray):
        return x.applyfunc(sp.cos)
    return sp.cos(x)


def tan(x, /):
    if isinstance(x, NumPyLikeArray):
        return x.applyfunc(sp.tan)
    return sp.tan(x)


def cot(x, /):
    if isinstance(x, NumPyLikeArray):
        return x.applyfunc(sp.cot)
    return sp.cot(x)


def sec(x, /):
    if isinstance(x, NumPyLikeArray):
        return x.applyfunc(sp.sec)
    return sp.sec(x)


def csc(x, /):
    if isinstance(x, NumPyLikeArray):
        return x.applyfunc(sp.csc)
    return sp.csc(x)


def asin(x, /):
    if isinstance(x, NumPyLikeArray):
        return x.applyfunc(sp.asin)
    return sp.asin(x)


def acos(x, /):
    if isinstance(x, NumPyLikeArray):
        return x.applyfunc(sp.acos)
    return sp.acos(x)


def atan(x, /):
    if isinstance(x, NumPyLikeArray):
        return x.applyfunc(sp.atan)
    return sp.atan(x)


def acot(x, /):
    if isinstance(x, NumPyLikeArray):
        return x.applyfunc(sp.acot)
    return sp.acot(x)


def asec(x, /):
    if isinstance(x, NumPyLikeArray):
        return x.applyfunc(sp.asec)
    return sp.asec(x)


def acsc(x, /):
    if isinstance(x, NumPyLikeArray):
        return x.applyfunc(sp.acsc)
    return sp.acsc(x)


def sinh(x, /):
    if isinstance(x, NumPyLikeArray):
        return x.applyfunc(sp.sinh)
    return sp.sinh(x)


def cosh(x, /):
    if isinstance(x, NumPyLikeArray):
        return x.applyfunc(sp.cosh)
    return sp.cosh(x)


def tanh(x, /):
    if isinstance(x, NumPyLikeArray):
        return x.applyfunc(sp.tanh)
    return sp.tanh(x)


def coth(x, /):
    if isinstance(x, NumPyLikeArray):
        return x.applyfunc(sp.coth)
    return sp.coth(x)


def sech(x, /):
    if isinstance(x, NumPyLikeArray):
        return x.applyfunc(sp.sech)
    return sp.sech(x)


def csch(x, /):
    if isinstance(x, NumPyLikeArray):
        return x.applyfunc(sp.csch)
    return sp.csch(x)


def asinh(x, /):
    if isinstance(x, NumPyLikeArray):
        return x.applyfunc(sp.asinh)
    return sp.asinh(x)


def acosh(x, /):
    if isinstance(x, NumPyLikeArray):
        return x.applyfunc(sp.acosh)
    return sp.acosh(x)


def atanh(x, /):
    if isinstance(x, NumPyLikeArray):
        return x.applyfunc(sp.atanh)
    return sp.atanh(x)


def acoth(x, /):
    if isinstance(x, NumPyLikeArray):
        return x.applyfunc(sp.acoth)
    return sp.acoth(x)


def asech(x, /):
    if isinstance(x, NumPyLikeArray):
        return x.applyfunc(sp.asech)
    return sp.asech(x)


def acsch(x, /):
    if isinstance(x, NumPyLikeArray):
        return x.applyfunc(sp.acsch)
    return sp.acsch(x)


CONSTANTS = dict(
    pi=sp.pi,
    e=sp.E,
)

FUNCTIONS = dict(
    # elementary functions
    sqrt=sqrt,
    exp=exp,
    ln=ln,
    log=log,
    # special functions
    sign=sign,
    # rounding functions
    floor=floor,
    ceil=ceil,
    trunc=trunc,
    # trigonometric functions
    sin=sin,
    cos=cos,
    tan=tan,
    cot=cot,
    sec=sec,
    csc=csc,
    asin=asin,
    acos=acos,
    atan=atan,
    acot=acot,
    asec=asec,
    acsc=acsc,
    # hyperbolic functions
    sinh=sinh,
    cosh=cosh,
    tanh=tanh,
    coth=coth,
    sech=sech,
    csch=csch,
    asinh=asinh,
    acosh=acosh,
    atanh=atanh,
    acoth=acoth,
    asech=asech,
    acsch=acsch,
)
