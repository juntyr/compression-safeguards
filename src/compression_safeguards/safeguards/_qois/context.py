from typing import Generic, Protocol

import numpy as np

from ...utils._compat import (
    _ensure_array,
    _maximum_zero_sign_sensitive,
    _minimum_zero_sign_sensitive,
)
from .typing import TYPE_CHECKING, F, Fc, Ns, Ps, Psc, np_sndarray

if TYPE_CHECKING:
    from .expr.abc import AnyExpr


class Callback(Protocol, Generic[Psc, Ns, Fc]):
    """
    Callback for reporting stencil-extended lower and upper bounds on the data.
    """

    def __call__(
        self,
        Xs_lower: np_sndarray[Psc, Ns, np.dtype[Fc]],
        Xs_upper: np_sndarray[Psc, Ns, np.dtype[Fc]],
    ) -> None:
        """
        Callback that is called once the stencil-extended lower and upper bounds
        `Xs_lower` and `Xs_upper` on the stencil-extended data `Xs` have been
        computed.

        These bounds have not yet been combined across neighbouring points
        that contribute to the same QoI points.

        Parameters
        ----------
        Xs_lower : np_sndarray[Ps, Ns, np.dtype[F]]
            The stencil-extended lower bounds on the stencil-extended data `Xs`.
        Xs_upper : np_sndarray[Ps, Ns, np.dtype[F]]
            The stencil-extended upper bounds on the stencil-extended data `Xs`.
        """


class Context(Generic[Ps, Ns, F]):
    """
    Context for the deferred computation of bounds on the data `Xs`.

    Parameters
    ----------
    expr : AnyExpr
        The root expression for which the new context should be created.
    """

    __slots__: tuple[str, ...] = ("_context",)
    _context: dict["AnyExpr", "_DeferredExprContext[Ps, Ns, F]"]

    def __init__(self, expr: "AnyExpr") -> None:
        self._context = dict()

        def visit_dependencies_once(e: "AnyExpr") -> None:
            if e in self._context:
                return

            self._context[e] = _DeferredExprContext(0)

            for a in e.args:
                visit_dependencies_once(a)
                self._context[a]._dependents += 1

        visit_dependencies_once(expr)
        self._context[expr]._dependents += 1

    def register_expr_bounds(
        self,
        expr: "AnyExpr",
        expr_lower: np.ndarray[tuple[Ps], np.dtype[F]],
        expr_upper: np.ndarray[tuple[Ps], np.dtype[F]],
        callback: Callback[Ps, Ns, F],
    ) -> "None | ReadyExprContext[Ps, Ns, F]":
        """
        Declare the lower and upper bounds `expr_lower` and `expr_upper` on the
        `expr`ession and register a `callback` that will be called once the
        bounds on the data `Xs` have been computed for `expr`.

        Parameters
        ----------
        expr : AnyExpr
            The expression for which the bounds are declared.
        expr_lower : np.ndarray[tuple[Ps], np.dtype[F]]
            The pointwise lower bound on the expression.
        expr_upper : np.ndarray[tuple[Ps], np.dtype[F]]
            The pointwise upper bound on the expression.
        callback : Callback[Ps, Ns, F]
            A callback that will be called once the stencil-extended lower and
            upper bounds `Xs_lower` and `Xs_upper` on the stencil-extended data
            `Xs` have been computed.

            These bounds have not yet been combined across neighbouring points
            that contribute to the same QoI points.

        Returns
        -------
        None
            if the bounds on the data for the `expr` cannot yet be computed.

        Returns
        -------
        ReadyExprContext[Ps, Ns, F]
            if the bounds on the data can and must now be computed.
        """

        from .expr.abc import DataExpr  # noqa: PLC0415

        if isinstance(expr, DataExpr):
            return ReadyExprContext(
                expr_lower=expr_lower, expr_upper=expr_upper, callbacks=(callback,)
            )

        ctx = self._context[expr]

        # short circuit in case there is only one dependent
        if ctx._dependents == 1 and len(ctx._callbacks) == 0:
            self._context.pop(expr)
            return ReadyExprContext(
                expr_lower=expr_lower, expr_upper=expr_upper, callbacks=(callback,)
            )

        if ctx._expr_bounds is None:
            ctx._expr_bounds = (
                _ensure_array(expr_lower, copy=True),
                _ensure_array(expr_upper, copy=True),
            )
        else:
            _maximum_zero_sign_sensitive(
                ctx._expr_bounds[0], expr_lower, out=ctx._expr_bounds[0]
            )
            _minimum_zero_sign_sensitive(
                ctx._expr_bounds[1], expr_upper, out=ctx._expr_bounds[1]
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


class _DeferredExprContext(Generic[Ps, Ns, F]):
    """
    Container for the deferred computation context for one expression.

    Parameters
    ----------
    dependents : int
        The number of expressions that depend on this expression.
    """

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
    """
    Permission to eagerly compute the bounds on the data `Xs` for an expression
    with the bounds `expr_lower` and `expr_upper`, after which the `callbacks`
    must be called with the computed bounds `Xs_lower` and `Xs_upper`.

    Parameters
    ----------
    expr_lower : np.ndarray[tuple[Ps], np.dtype[F]]
        The pointwise lower bound on the expression.
    expr_upper : np.ndarray[tuple[Ps], np.dtype[F]]
        The pointwise upper bound on the expression.
    callbacks : tuple[Callback[Ps, Ns, F], ...]
        The callbacks that will be called once the stencil-extended lower and
        upper bounds `Xs_lower` and `Xs_upper` on the stencil-extended data
        `Xs` have been computed.

        These bounds have not yet been combined across neighbouring points
        that contribute to the same QoI points.
    """

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
        """
        The pointwise lower bound on the expression.
        """

        return self._expr_lower

    @property
    def expr_upper(self) -> np.ndarray[tuple[Ps], np.dtype[F]]:
        """
        The pointwise upper bound on the expression.
        """

        return self._expr_upper

    def apply_callbacks(
        self,
        Xs_lower: np_sndarray[Ps, Ns, np.dtype[F]],
        Xs_upper: np_sndarray[Ps, Ns, np.dtype[F]],
    ) -> None:
        """
        Apply the callbacks with the stencil-extended lower and upper bounds
        `Xs_lower` and `Xs_upper` on the stencil-extended data `Xs`.

        These bounds have not yet been combined across neighbouring points
        that contribute to the same QoI points.

        Parameters
        ----------
        Xs_lower : np_sndarray[Ps, Ns, np.dtype[F]]
            The stencil-extended lower bounds on the stencil-extended data `Xs`.
        Xs_upper : np_sndarray[Ps, Ns, np.dtype[F]]
            The stencil-extended upper bounds on the stencil-extended data `Xs`.
        """

        # short circuit in case there is only one callback
        match self._callbacks:
            case (callback,):
                return callback(Xs_lower, Xs_upper)

        for callback in self._callbacks:
            callback(
                _ensure_array(Xs_lower, copy=True),
                _ensure_array(Xs_upper, copy=True),
            )


class DataBoundsAccumulator(Generic[Ps, Ns, F]):
    """
    Accumulator for the bounds over the stencil-extended data `Xs` that are
    derived across several expression `terms`.

    Once all per-term bounds have been computed, the provided `callback` will
    be called with the stencil-extended lower and upper bounds `Xs_lower` and
    `Xs_upper` on the stencil-extended data `Xs`.

    These bounds have not yet been combined across neighbouring points
    that contribute to the same QoI points.

    Parameters
    ----------
    Xs : np_sndarray[Ps, Ns, np.dtype[F]]
        The stencil-extended data, in floating-point format, which must be
        of shape [Ps, ...stencil_shape].
    terms : int
        The number of terms in the expression.
    callback : Callback[Ps, Ns, F]
        A callback that will be called once the stencil-extended lower and
        upper bounds `Xs_lower` and `Xs_upper` on the stencil-extended data
        `Xs` have been computed.

        These bounds have not yet been combined across neighbouring points
        that contribute to the same QoI points.
    """

    __slots__: tuple[str, ...] = (
        "_Xs",
        "_Xs_lower_out",
        "_Xs_upper_out",
        "_can_override_out",
        "_terms_completed",
        "_callback",
    )
    _Xs: np_sndarray[Ps, Ns, np.dtype[F]]
    _Xs_lower_out: np_sndarray[Ps, Ns, np.dtype[F]]
    _Xs_upper_out: np_sndarray[Ps, Ns, np.dtype[F]]
    _can_override_out: bool
    _terms_completed: list[bool]
    _callback: None | Callback

    def __init__(
        self,
        *,
        Xs: np_sndarray[Ps, Ns, np.dtype[F]],
        terms: int,
        callback: Callback[Ps, Ns, F],
    ) -> None:
        self._Xs = Xs

        self._Xs_lower_out = np.full(Xs.shape, Xs.dtype.type(-np.inf))
        self._Xs_upper_out = np.full(Xs.shape, Xs.dtype.type(np.inf))

        self._can_override_out = True
        self._terms_completed = [False for _ in range(terms)]

        self._callback = callback

        self.check_for_completion()

    def on_complete_term(
        self,
        Xs_lower: np_sndarray[Ps, Ns, np.dtype[F]],
        Xs_upper: np_sndarray[Ps, Ns, np.dtype[F]],
        *,
        term: int,
        where: None | np_sndarray[Ps, Ns, np.dtype[np.bool]] = None,
    ) -> None:
        """
        Callback that can be passed as the `callback` parameter in
        [`Expr.compute_data_bounds`][...expr.abc.Expr.compute_data_bounds]
        to accumulate the stencil-extended lower and upper bounds `Xs_lower`
        and `Xs_upper` for a specific `term` into the combined lower and
        upper bounds.

        This callback automatically calls [`complete_term`][..complete_term].

        Parameters
        ----------
        Xs_lower : np_sndarray[Ps, Ns, np.dtype[F]]
            The stencil-extended lower bounds on the stencil-extended data
            `Xs`, for the `term`.
        Xs_upper : np_sndarray[Ps, Ns, np.dtype[F]]
            The stencil-extended upper bounds on the stencil-extended data
            `Xs`, for the `term`.
        term : int
            The index of the term for which the data bounds have been computed.
        where : None | np_sndarray[Ps, Ns, np.dtype[np.bool]]
            Optional mask to only integrate the data bounds for the `term` for
            some elements.
        """

        # combine the inner data bounds
        if self._can_override_out:
            np.copyto(self._Xs_lower_out, Xs_lower)
            np.copyto(self._Xs_upper_out, Xs_upper)
        else:
            _maximum_zero_sign_sensitive(
                self._Xs_lower_out, Xs_lower, out=self._Xs_lower_out, where=where
            )
            _minimum_zero_sign_sensitive(
                self._Xs_upper_out, Xs_upper, out=self._Xs_upper_out, where=where
            )
        self._can_override_out = False

        self.complete_term(term)

    def complete_term(self, term: int) -> None:
        """
        Eagerly mark the `term` as completed.

        This method automatically calls
        [`check_for_completion`][..check_for_completion].

        Parameters
        ----------
        term : int
            The index of the term that has been completed.
        """

        self._terms_completed[term] = True

        self.check_for_completion()

    def check_for_completion(self) -> None:
        """
        Check if all terms have been completed.

        If all terms have been completed, call the callback on the accumulated
        stencil-extended lower and upper bounds `Xs_lower` and `Xs_upper` on
        the data `Xs`.

        The callback is only called once, the first time that the completion
        check succeeds.
        """

        if self._callback is None:
            return

        if not all(self._terms_completed):
            return

        callback = self._callback
        self._callback = None

        # ensure that the bounds on Xs include Xs
        _minimum_zero_sign_sensitive(
            self._Xs_lower_out, self._Xs, out=self._Xs_lower_out
        )
        _maximum_zero_sign_sensitive(
            self._Xs_upper_out, self._Xs, out=self._Xs_upper_out
        )

        return callback(self._Xs_lower_out, self._Xs_upper_out)
