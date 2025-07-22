import sympy as sp

from .vars import LateBoundConstant


def rewrite_qoi_expr(expr: sp.Basic) -> sp.Basic:
    """
    Rewrite the QoI expression `expr` such that sums and products over several
    terms, where some are constant and some are not, have a fixed evaluation
    order.

    In particular, the fixed order guarantees that the constant terms can be
    evaluated separately from the non-constant terms, and the sum / product of
    the results is exactly the same as if they had been evaluated together.

    This property is needed to ensure that sums and products can be split up
    (when deriving the data error bounds from a QoI expression) such that
    providing error bound guarantees on the parts also provide the same
    guarantees when evaluating the whole QoI expression (after compilation) in
    one go.

    Parameters
    ----------
    expr : sp.Basic
        The QoI expression.

    Returns
    -------
    rewritten : sp.Basic
        The rewritten QoI expression.
    """

    # early return for atoms
    if expr.is_Number or expr.is_symbol:
        return expr

    # early return for constant expressions
    if all(isinstance(s, LateBoundConstant) for s in expr.free_symbols):
        return expr

    if expr.is_Add or expr.is_Mul:
        args = expr.args

        # separate constant and non-constant terms
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

        # only rewrite if both constant and non-constant terms exist
        if len(const_args) > 0 and len(non_const_args) > 0:
            return ({sp.Add: NonAssociativeAdd, sp.Mul: NonAssociativeMul}[expr.func])(
                (expr.func)(*[rewrite_qoi_expr(a) for a in non_const_args]),
                (expr.func)(*[rewrite_qoi_expr(a) for a in const_args]),
            )

    # recursive application
    return (expr.func)(*[rewrite_qoi_expr(a) for a in expr.args])


class NonAssociativeAdd(sp.Function):
    """
    Non-associative addition that is used to ensure arguments are not
    reordered.
    """


class NonAssociativeMul(sp.Function):
    """
    Non-associative multiplication that is used to ensure arguments are not
    reordered.
    """
