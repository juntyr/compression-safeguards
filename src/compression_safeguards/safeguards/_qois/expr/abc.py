from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from typing import (
    TYPE_CHECKING,
    Any,
    Generic,
    Protocol,
    Self,
    TypeAlias,
    assert_never,
    final,
)
from warnings import warn

import numpy as np
from typing_extensions import override  # MSPV 3.12

from ....utils._compat import (
    _maximum_zero_sign_sensitive,
    _minimum_zero_sign_sensitive,
    _zeros,
)
from ....utils.bindings import Parameter
from ....utils.error import QuantityOfInterestRuntimeWarning
from ..bound import DataBounds, data_bounds_checks, guarantee_data_within_expr_bounds
from ..typing import Es, F, Fc, Ns, Ps, Psc, np_sndarray

if TYPE_CHECKING:
    from .literal import Number


class Callback(Protocol, Generic[Psc, Ns, Fc]):
    def __call__(
        self,
        Xs_lower: np_sndarray[Psc, Ns, np.dtype[Fc]],
        Xs_upper: np_sndarray[Psc, Ns, np.dtype[Fc]],
    ) -> None: ...


class Expr(ABC, Generic[*Es]):
    """
    Abstract base class for the quantity of interest expression abstract syntax
    tree.
    """

    __slots__: tuple[str, ...] = ()

    @property
    @abstractmethod
    def args(self) -> tuple[*Es]:
        """
        The sub-expression arguments of this expression.
        """

    @property
    @abstractmethod
    def extra(self) -> tuple[Any, ...]:
        pass

    @abstractmethod
    def with_args(self, *args: *Es) -> "Self | Number":
        """
        Reconstruct this expression with different sub-expression arguments.

        Parameters
        ----------
        *args : *Es
            The modified sub-expression arguments, derived from
            [`self.args`][..args].

        Returns
        -------
        expr : Self
            The modified expression.
        """

    @final
    def visit_expr(self, v: Callable[["AnyExpr"], None]) -> None:
        args: tuple[AnyExpr, ...] = self.args  # type: ignore
        for a in args:
            a.visit_expr(v)
        this: AnyExpr = self  # type: ignore
        return v(this)

    @final
    def map_expr(self, m: "Callable[[AnyExpr], AnyExpr]") -> "AnyExpr":
        """
        Recursively maps the expression mapping function `m` over this
        expression and its sub-expression arguments.

        Parameters
        ----------
        m : Callable[[AnyExpr], AnyExpr]
            The expression mapper, which is applied to an expression whose
            sub-expression arguments have already been mapped, i.e. the mapper
            is *not* responsible for recursion.

        Returns
        -------
        expr : AnyExpr
            The mapped expression.
        """

        args: tuple[AnyExpr, ...] = self.args  # type: ignore
        mapped_args: tuple[AnyExpr, ...] = tuple(a.map_expr(m) for a in args)
        mapped_self: AnyExpr = self.with_args(*mapped_args)  # type: ignore
        return m(mapped_self)

    @final
    def deduplicate_expr(
        self,
        cache: dict[
            tuple[type["AnyExpr"], tuple["AnyExpr", ...], tuple[Any, ...]], "AnyExpr"
        ],
    ) -> "AnyExpr":
        ty: type[AnyExpr] = type(self)  # type: ignore

        args: tuple[AnyExpr, ...] = self.args  # type: ignore
        deduplicated_args: tuple[AnyExpr, ...] = tuple(
            a.deduplicate_expr(cache) for a in args
        )

        key = (ty, deduplicated_args, self.extra)

        cached = cache.get(key, None)

        if cached is not None:
            return cached

        deduplicated_self: AnyExpr = self.with_args(*deduplicated_args)  # type: ignore

        cache[key] = deduplicated_self

        return deduplicated_self

    @final
    @property
    def expr_size(self) -> int:
        """
        The size of the expression tree, counting the number of nodes.
        """

        args: tuple[AnyExpr, ...] = self.args  # type: ignore

        return sum(a.expr_size for a in args) + 1

    @final
    @property
    def data_expr_size(self) -> int:
        """
        The size of the expression tree, counting the number of data-dependent
        nodes.
        """

        args: tuple[AnyExpr, ...] = self.args  # type: ignore

        if len(args) == 0:
            return int(self.has_data)

        return sum(a.expr_size for a in args if a.has_data) + 1

    @final
    @property
    def has_data(self) -> bool:
        """
        Does this expression reference the data `x` or `X[i]`?
        """

        args: tuple[AnyExpr, ...] = self.args  # type: ignore

        return any(a.has_data for a in args)

    @final
    def eval_has_data(
        self,
        Xs: np_sndarray[Ps, Ns, np.dtype[F]],
        late_bound: Mapping[Parameter, np_sndarray[Ps, Ns, np.dtype[F]]],
    ) -> np.ndarray[tuple[Ps], np.dtype[np.bool]]:
        """
        Evaluate if the pointwise expression values references the data `x` or
        `X[i]`.

        Parameters
        ----------
        Xs : np_sndarray[Ps, Ns, np.dtype[F]]
            The stencil-extended data, in floating-point format, which must be
            of shape [Ps, ...stencil_shape].
        late_bound : Mapping[Parameter, np_sndarray[Ps, Ns, np.dtype[F]]]
            The late-bound constants parameters for this expression, with the
            same shape and floating-point dtype as the stencil-extended data.

        Returns
        -------
        vals : np.ndarray[tuple[Ps], np.dtype[np.bool]]
            Pointwise [`True`][True] if the point references the data.
        """

        if not self.has_data:
            return _zeros((Xs.shape[0],), dtype=np.dtype(np.bool))

        args: tuple[AnyExpr, ...] = self.args  # type: ignore

        match args:
            case ():
                return _zeros((Xs.shape[0],), dtype=np.dtype(np.bool))
            case _:
                a, *as_ = args
                has_data = a.eval_has_data(Xs, late_bound)
                for a in as_:
                    has_data |= a.eval_has_data(Xs, late_bound)
                return has_data

    @final
    @property
    def data_indices(self) -> frozenset[tuple[int, ...]]:
        """
        The set of data indices `X[is]` that this expression uses.
        """

        args: tuple[AnyExpr, ...] = self.args  # type: ignore

        match args:
            case ():
                return frozenset()
            case (a,):
                return a.data_indices
            case _:
                indices: set[tuple[int, ...]] = set()
                for a in args:
                    indices.update(a.data_indices)
                return frozenset(indices)

    @final
    def apply_array_element_offset(
        self,
        axis: int,
        offset: int,
    ) -> "AnyExpr":
        """
        Apply an `offset` to the array element indices along the given `axis`.

        This method applies to data and late-bound constants.

        Parameters
        ----------
        axis : int
            The axis along which the array element indices are offset.
        offset : int
            The offset that is applied to the array element indices.

        Returns
        -------
        expr : AnyExpr
            The modified expression.
        """

        return self.with_args(
            *(a.apply_array_element_offset(axis, offset) for a in self.args)  # type: ignore
        )

    @final
    @property
    def late_bound_constants(self) -> frozenset[Parameter]:
        """
        The set of late-bound constant parameters that this expression uses.
        """

        args: tuple[AnyExpr, ...] = self.args  # type: ignore

        match args:
            case ():
                return frozenset()
            case (a,):
                return a.late_bound_constants
            case _:
                late_bound: set[Parameter] = set()
                for a in args:
                    late_bound.update(a.late_bound_constants)
                return frozenset(late_bound)

    # FIXME: constant_fold based on self.args and self.with_args is blocked on
    #        not being able to relate on TypeVarTuple to another, here *Expr to
    #        *F, see e.g. https://github.com/python/typing/issues/1216
    @abstractmethod
    def constant_fold(self, dtype: np.dtype[F]) -> "F | AnyExpr":
        """
        Apply scalar constant folding for the given `dtype` to this expression.

        Parameters
        ----------
        dtype : np.dtype[F]
            The dtype for which constant expressions are evaluated and folded.

        Returns
        -------
        folded : F | Expr
            The constant-folded scalar number or expression, which may have
            been simplified.
        """

    @abstractmethod
    def eval(
        self,
        Xs: np_sndarray[Ps, Ns, np.dtype[F]],
        late_bound: Mapping[Parameter, np_sndarray[Ps, Ns, np.dtype[F]]],
    ) -> np.ndarray[tuple[Ps], np.dtype[F]]:
        """
        Evaluate this expression on the stencil-extended data `Xs`.

        Parameters
        ----------
        Xs : np_sndarray[Ps, Ns, np.dtype[F]]
            The stencil-extended data, in floating-point format, which must be
            of shape [Ps, ...stencil_shape].
        late_bound : Mapping[Parameter, np_sndarray[Ps, Ns, np.dtype[F]]]
            The late-bound constants parameters for this expression, with the
            same shape and floating-point dtype as the stencil-extended data.

        Returns
        -------
        vals : np.ndarray[tuple[Ps], np.dtype[F]]
            The pointwise expression values.
        """

    @abstractmethod
    def deferred_compute_data_bounds_unchecked(
        self,
        expr_lower: np.ndarray[tuple[Ps], np.dtype[F]],
        expr_upper: np.ndarray[tuple[Ps], np.dtype[F]],
        Xs: np_sndarray[Ps, Ns, np.dtype[F]],
        late_bound: Mapping[Parameter, np_sndarray[Ps, Ns, np.dtype[F]]],
        ctx: "Context",
        callback: Callback[Ps, Ns, F],
    ) -> None:
        pass

    @final
    def deferred_compute_data_bounds(
        self,
        expr_lower: np.ndarray[tuple[Ps], np.dtype[F]],
        expr_upper: np.ndarray[tuple[Ps], np.dtype[F]],
        Xs: np_sndarray[Ps, Ns, np.dtype[F]],
        late_bound: Mapping[Parameter, np_sndarray[Ps, Ns, np.dtype[F]]],
        ctx: "Context",
        callback: Callback[Ps, Ns, F],
    ) -> None:
        ready = ctx.push_expr_bounds(
            self,  # type: ignore
            expr_lower,
            expr_upper,
            callback,
        )

        if ready is None:
            return None

        expr_lower = ready.expr_lower
        expr_upper = ready.expr_upper

        if (
            data_bounds_checks(self.deferred_compute_data_bounds_unchecked)
            != DataBounds.infallible
        ):
            exprv: np.ndarray[tuple[Ps], np.dtype[F]] = self.eval(Xs, late_bound)
            if not np.all((expr_lower <= exprv) | np.isnan(exprv)):
                warn(
                    "expression lower bounds are above the expression values",
                    category=QuantityOfInterestRuntimeWarning,
                )
            if not np.all((expr_upper >= exprv) | np.isnan(exprv)):
                warn(
                    "expression upper bounds are below the expression values",
                    category=QuantityOfInterestRuntimeWarning,
                )
        else:
            exprv = None  # type: ignore

        def wrapped_callback(
            Xs_lower: np_sndarray[Ps, Ns, np.dtype[F]],
            Xs_upper: np_sndarray[Ps, Ns, np.dtype[F]],
        ) -> None:
            warn_on_bounds_exceeded: bool

            match data_bounds_checked := data_bounds_checks(
                self.deferred_compute_data_bounds_unchecked
            ):
                case DataBounds.infallible:
                    return ready.apply_callbacks(Xs_lower, Xs_upper)
                case DataBounds.unchecked:
                    warn_on_bounds_exceeded = False
                case DataBounds.checked:
                    warn_on_bounds_exceeded = True
                case _:
                    assert_never(data_bounds_checked)

            # ensure that the original data values are within the data bounds
            Xs_lower = _minimum_zero_sign_sensitive(Xs, Xs_lower)
            Xs_upper = _maximum_zero_sign_sensitive(Xs, Xs_upper)

            # handle rounding errors in the lower bound computation
            Xs_lower = guarantee_data_within_expr_bounds(
                lambda Xs_lower: self.eval(
                    Xs_lower,
                    late_bound,
                ),
                exprv,
                Xs,
                Xs_lower,
                expr_lower,
                expr_upper,
                warn_on_bounds_exceeded=warn_on_bounds_exceeded,
            )
            Xs_upper = guarantee_data_within_expr_bounds(
                lambda Xs_upper: self.eval(
                    Xs_upper,
                    late_bound,
                ),
                exprv,
                Xs,
                Xs_upper,
                expr_lower,
                expr_upper,
                warn_on_bounds_exceeded=warn_on_bounds_exceeded,
            )

            return ready.apply_callbacks(Xs_lower, Xs_upper)

        return self.deferred_compute_data_bounds_unchecked(
            expr_lower,
            expr_upper,
            Xs,
            late_bound,
            ctx,
            wrapped_callback,
        )

    @abstractmethod
    @override
    def __repr__(self) -> str:
        pass


AnyExpr: TypeAlias = Expr[*tuple["AnyExpr", ...]]
""" Expression with sub-expression arguments """

EmptyExpr: TypeAlias = Expr[()]
""" Expression with zero arguments """


def compute_expr_data_bounds(
    expr: AnyExpr,
    expr_lower: np.ndarray[tuple[Ps], np.dtype[F]],
    expr_upper: np.ndarray[tuple[Ps], np.dtype[F]],
    Xs: np_sndarray[Ps, Ns, np.dtype[F]],
    late_bound: Mapping[Parameter, np_sndarray[Ps, Ns, np.dtype[F]]],
) -> tuple[np_sndarray[Ps, Ns, np.dtype[F]], np_sndarray[Ps, Ns, np.dtype[F]]]:
    expr = expr.deduplicate_expr({})

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


class Context(Generic[Ps, Ns, F]):
    __slots__: tuple[str, ...] = ("_context",)
    _context: dict[AnyExpr, "ExprContext[Ps, Ns, F]"]

    def __init__(self, expr: AnyExpr) -> None:
        self._context = dict()

        def visit_dependencies_once(e: AnyExpr) -> None:
            if e in self._context:
                return

            self._context[e] = ExprContext(0)

            for a in e.args:
                visit_dependencies_once(a)
                self._context[a]._dependents += 1

        visit_dependencies_once(expr)
        self._context[expr]._dependents += 1

    def push_expr_bounds(
        self,
        expr: "AnyExpr",
        expr_lower: np.ndarray[tuple[Ps], np.dtype[F]],
        expr_upper: np.ndarray[tuple[Ps], np.dtype[F]],
        callback: Callback[Ps, Ns, F],
    ) -> "None | ReadyExprContext[Ps, Ns, F]":
        ctx = self._context[expr]

        if ctx._expr_bounds is None:
            ctx._expr_bounds = (np.copy(expr_lower), np.copy(expr_upper))
        else:
            ctx._expr_bounds = (
                _maximum_zero_sign_sensitive(ctx._expr_bounds[0], expr_lower),
                _minimum_zero_sign_sensitive(ctx._expr_bounds[1], expr_upper),
            )
        ctx._callbacks.append(callback)
        assert len(ctx._callbacks) <= ctx._dependents

        if len(ctx._callbacks) < ctx._dependents:
            return None
        self._context.pop(expr)

        return ReadyExprContext(
            expr_lower=ctx._expr_bounds[0],
            expr_upper=ctx._expr_bounds[1],
            callbacks=tuple(ctx._callbacks),
        )


class ExprContext(Generic[Ps, Ns, F]):
    __slots__: tuple[str, ...] = ("_dependents", "_callbacks", "_expr_bounds")
    _dependents: int
    _callbacks: list[Callback[Ps, Ns, F]]
    _expr_bounds: (
        None
        | tuple[np.ndarray[tuple[Ps], np.dtype[F]], np.ndarray[tuple[Ps], np.dtype[F]]]
    )

    def __init__(self, dependents: int) -> None:
        self._dependents = dependents
        self._callbacks = []
        self._expr_bounds = None


class ReadyExprContext(Generic[Ps, Ns, F]):
    __slots__: tuple[str, ...] = ("_expr_lower", "_expr_upper", "_callbacks")
    _expr_lower: np.ndarray[tuple[Ps], np.dtype[F]]
    _expr_upper: np.ndarray[tuple[Ps], np.dtype[F]]
    _callbacks: tuple[Callback[Ps, Ns, F], ...]

    def __init__(
        self,
        expr_lower: np.ndarray[tuple[Ps], np.dtype[F]],
        expr_upper: np.ndarray[tuple[Ps], np.dtype[F]],
        callbacks: tuple[Callback[Ps, Ns, F], ...],
    ) -> None:
        self._expr_lower = expr_lower
        self._expr_upper = expr_upper
        self._callbacks = callbacks

    @property
    def expr_lower(self) -> np.ndarray[tuple[Ps], np.dtype[F]]:
        return self._expr_lower

    @property
    def expr_upper(self) -> np.ndarray[tuple[Ps], np.dtype[F]]:
        return self._expr_upper

    def apply_callbacks(
        self,
        Xs_lower: np_sndarray[Ps, Ns, np.dtype[F]],
        Xs_upper: np_sndarray[Ps, Ns, np.dtype[F]],
    ) -> None:
        for callback in self._callbacks:
            callback(np.copy(Xs_lower), np.copy(Xs_upper))
