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
    def __call__(
        self,
        Xs_lower: np_sndarray[Psc, Ns, np.dtype[Fc]],
        Xs_upper: np_sndarray[Psc, Ns, np.dtype[Fc]],
    ) -> None: ...


class Context(Generic[Ps, Ns, F]):
    __slots__: tuple[str, ...] = ("_context",)
    _context: dict["AnyExpr", "ExprContext[Ps, Ns, F]"]

    def __init__(self, expr: "AnyExpr") -> None:
        self._context = dict()

        def visit_dependencies_once(e: "AnyExpr") -> None:
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
        from .expr.data import Data, ScalarAnyDataConstant  # noqa: PLC0415

        if isinstance(expr, Data | ScalarAnyDataConstant):
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
        # short circuit in case there is only one callback
        match self._callbacks:
            case (callback,):
                return callback(Xs_lower, Xs_upper)

        for callback in self._callbacks:
            callback(
                _ensure_array(Xs_lower, copy=True),
                _ensure_array(Xs_upper, copy=True),
            )


class AccumulateXsBoundsCallback(Generic[Ps, Ns, F]):
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
        self, *, Xs: np_sndarray[Ps, Ns, np.dtype[F]], terms: int, callback: Callback
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
        self._terms_completed[term] = True

        self.check_for_completion()

    def check_for_completion(self) -> None:
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
