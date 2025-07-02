import sympy as sp

from .vars import LateBoundConstant


def rewrite_qoi_expr(expr: sp.Basic) -> sp.Basic:
    if expr.is_Number or expr.is_symbol:
        return expr

    if expr.is_Add or expr.is_Mul:
        args = expr.args

        non_const_args = [
            a
            for a in args
            if not all(isinstance(s, LateBoundConstant) for s in a.free_symbols)
        ]
        const_args = [
            a
            for a in args
            if all(isinstance(s, LateBoundConstant) for s in a.free_symbols)
        ]

        if len(const_args) > 0 and len(non_const_args) > 0:
            return ({sp.Add: NonAssociativeAdd, sp.Mul: NonAssociativeMul}[expr.func])(
                (expr.func)(*[rewrite_qoi_expr(a) for a in non_const_args]),
                (expr.func)(*[rewrite_qoi_expr(a) for a in const_args]),
            )

    return (expr.func)(*[rewrite_qoi_expr(a) for a in expr.args])


class NonAssociativeAdd(sp.Function):
    """
    Non-associative addition that is used to ensure arguments are not mixed.
    """


class NonAssociativeMul(sp.Function):
    """
    Non-associative multiplication that is used to ensure arguments are not
    mixed.
    """
