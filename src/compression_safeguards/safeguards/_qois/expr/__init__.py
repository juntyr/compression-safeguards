from collections.abc import Callable, Mapping, Set
from typing import Any

import numpy as np

from ....utils.bindings import Parameter
from ..context import Context
from ..typing import F, Ns, Ps, np_sndarray
from .abc import AnyExpr


def compute_expr_data_bounds(
    expr: AnyExpr,
    expr_lower: np.ndarray[tuple[Ps], np.dtype[F]],
    expr_upper: np.ndarray[tuple[Ps], np.dtype[F]],
    Xs: np_sndarray[Ps, Ns, np.dtype[F]],
    late_bound: Mapping[Parameter, np_sndarray[Ps, Ns, np.dtype[F]]],
) -> tuple[np_sndarray[Ps, Ns, np.dtype[F]], np_sndarray[Ps, Ns, np.dtype[F]]]:
    expr = deduplicate_expr(expr)

    Xs_lower_out: list[None | np_sndarray[Ps, Ns, np.dtype[F]]] = [None]
    Xs_upper_out: list[None | np_sndarray[Ps, Ns, np.dtype[F]]] = [None]

    def callback(
        Xs_lower: np_sndarray[Ps, Ns, np.dtype[F]],
        Xs_upper: np_sndarray[Ps, Ns, np.dtype[F]],
    ) -> None:
        Xs_lower_out[0] = Xs_lower
        Xs_upper_out[0] = Xs_upper

    expr.deferred_compute_data_bounds(
        expr_lower, expr_upper, Xs, late_bound, Context(expr), callback
    )

    assert Xs_lower_out[0] is not None
    assert Xs_upper_out[0] is not None

    return Xs_lower_out[0], Xs_upper_out[0]


def map_expr(expr: AnyExpr, *, mapper: Callable[[AnyExpr], AnyExpr]) -> AnyExpr:
    """
    Recursively maps the expression `mapper` function over the
    `expr`ession and its sub-expression arguments.

    Parameters
    ----------
    expr : AnyExpr
        The expression to map over.
    mapper : Callable[[AnyExpr], AnyExpr]
        The expression mapper, which is applied to an expression whose
        sub-expression arguments have already been mapped, i.e. the mapper
        is *not* responsible for recursion.

    Returns
    -------
    mapped : AnyExpr
        The mapped expression.
    """

    return mapper(expr.with_args(*(map_expr(a, mapper=mapper) for a in expr.args)))


def deduplicate_expr(expr: AnyExpr) -> AnyExpr:
    """
    Recursively deduplicate the `expr`ession and its sub-expression
    arguments using common sub-expression elimination.

    Where previously different expression objects described the same
    symbolic expression, without applying any simplification rules,
    only one expression object will be used for all of them in the
    returned expression.

    Parameters
    ----------
    expr : AnyExpr
        The expression to deduplicate.

    Returns
    -------
    deduplicated : AnyExpr
        The deduplicated expression.
    """

    cache: dict[
        tuple[type[AnyExpr], tuple[AnyExpr, ...], tuple[Any, ...]], AnyExpr
    ] = {}

    def deduplication_mapper(e: AnyExpr) -> AnyExpr:
        # e.args have already been deduplicated since the mapper is applied in
        #  post-order
        key = (type(e), e.args, e.extra)

        cached = cache.get(key, None)
        if cached is not None:
            return cached

        cache[key] = e
        return e

    return map_expr(expr, mapper=deduplication_mapper)


def pre_visit_expr(expr: AnyExpr, *, visitor: Callable[[AnyExpr], None]) -> None:
    """
    Recursively visit the expression tree in pre-order, calling the `visitor`
    function first on the `expr`ession and then its sub-expression arguments.

    Parameters
    ----------
    expr : AnyExpr
        The expression to visit in pre-order.
    visitor : Callable[[AnyExpr], None]
        The expression visitor, which is applied to an expression whose
        sub-expression arguments have not yet been visited. The visitor is
        *not* responsible for recursion.
    """

    visitor(expr)
    for a in expr.args:
        pre_visit_expr(a, visitor=visitor)


def post_visit_expr(expr: AnyExpr, *, visitor: Callable[[AnyExpr], None]) -> None:
    """
    Recursively visit the expression tree in post-order, calling the `visitor`
    function first on the sub-expression arguments and then the `expr`ession.

    Parameters
    ----------
    expr : AnyExpr
        The expression to visit in post-order.
    visitor : Callable[[AnyExpr], None]
        The expression visitor, which is applied to an expression whose
        sub-expression arguments have already been visited, i.e. the
        visitor is *not* responsible for recursion.
    """

    for a in expr.args:
        post_visit_expr(a, visitor=visitor)
    visitor(expr)


def expr_data_indices(expr: AnyExpr) -> frozenset[tuple[int, ...]]:
    """
    Compute the full set of data indices `X[is]` that the `expr`ession and
    its subexpression arguments use.

    Parameters
    ----------
    expr : AnyExpr
        The expression for which the set of data indices are computed.

    Returns
    -------
    data_indices : frozenset[tuple[int, ...]]
        The set of data indices used by the `expr`ession.
    """

    data_indices: set[tuple[int, ...]] = set()

    def visit_data_indices(e: AnyExpr) -> None:
        if not hasattr(e, "data_indices"):
            indices: Set[tuple[int, ...]] = e.data_indices  # type: ignore
            data_indices.update(indices)

    pre_visit_expr(expr, visitor=visit_data_indices)

    return frozenset(data_indices)


def expr_late_bound_constants(expr: AnyExpr) -> frozenset[Parameter]:
    """
    Compute the full set of late-bound constant parameters that the
    `expr`ession and its subexpression arguments use.
    """

    late_bound_constants: set[Parameter] = set()

    def visit_late_bound_constants(e: AnyExpr) -> None:
        if hasattr(e, "late_bound_constants"):
            constants: Set[Parameter] = e.late_bound_constants  # type: ignore
            late_bound_constants.update(constants)

    pre_visit_expr(expr, visitor=visit_late_bound_constants)

    return frozenset(late_bound_constants)
