"""
Test-only Monotonicity-preserving safeguard.
"""

__all__ = ["Monotonicity", "MonotonicityPreservingSafeguard"]

from collections.abc import Set
from enum import Enum
from operator import ge, gt, le, lt
from typing import ClassVar, Literal

import numpy as np
from typing_extensions import (
    assert_never,  # MSPV 3.11
    override,  # MSPV 3.12
)

from compression_safeguards.safeguards.stencil import (
    BoundaryCondition,
    NeighbourhoodAxis,
    _pad_with_boundary,
)
from compression_safeguards.safeguards.stencil.abc import StencilSafeguard
from compression_safeguards.utils._compat import _ones, _place, _sliding_window_view
from compression_safeguards.utils.bindings import Bindings, Parameter
from compression_safeguards.utils.cast import lossless_cast
from compression_safeguards.utils.error import TypeCheckError, ctx, lookup_enum_or_raise
from compression_safeguards.utils.intervals import IntervalUnion
from compression_safeguards.utils.typing import JSON, S, T

_STRICT = ((lt, gt, False, False),) * 2
_STRICT_WITH_CONSTS = ((lt, gt, True, False),) * 2
_STRICT_TO_WEAK = ((lt, gt, False, False), (le, ge, True, True))
_WEAK = ((le, ge, False, True), (le, ge, True, True))


class Monotonicity(Enum):
    strict = _STRICT
    """
    Strictly increasing/decreasing sequences in the input array are guaranteed
    to be strictly increasing/decreasing in the corrected array.
    """

    strict_with_consts = _STRICT_WITH_CONSTS
    """
    Strictly increasing/decreasing/constant sequences in the input array are
    guaranteed to be strictly increasing/decreasing/constant in the corrected
    array.
    """

    strict_to_weak = _STRICT_TO_WEAK
    """
    Strictly increasing/decreasing sequences in the input array are guaranteed
    to be *weakly* increasing/decreasing (or constant) in the corrected array.
    """

    weak = _WEAK
    """
    Weakly increasing/decreasing (but not constant) sequences in the input
    array are guaranteed to be weakly increasing/decreasing (or constant) in
    the corrected array.
    """


class MonotonicityPreservingSafeguard(StencilSafeguard):
    __slots__: tuple[str, ...] = (
        "_monotonicity",
        "_window",
        "_boundary",
        "_constant_boundary",
        "_axis",
    )
    _monotonicity: Monotonicity
    _window: int
    _boundary: BoundaryCondition
    _constant_boundary: None | int | float | Parameter
    _axis: None | int

    kind: ClassVar[str] = "monotonicity"

    def __init__(
        self,
        monotonicity: str | Monotonicity,
        window: int,
        boundary: str | BoundaryCondition,
        constant_boundary: None | int | float | str | Parameter = None,
        axis: None | int = None,
    ) -> None:
        with ctx.safeguard(self):
            with ctx.parameter("monotonicity"):
                TypeCheckError.check_instance_or_raise(monotonicity, str | Monotonicity)
                self._monotonicity = (
                    monotonicity
                    if isinstance(monotonicity, Monotonicity)
                    else lookup_enum_or_raise(Monotonicity, monotonicity)
                )

            with ctx.parameter("window"):
                TypeCheckError.check_instance_or_raise(window, int)
                if window <= 0:
                    raise ValueError("must be positive") | ctx
                self._window = window

            with ctx.parameter("boundary"):
                TypeCheckError.check_instance_or_raise(
                    boundary, str | BoundaryCondition
                )
                self._boundary = (
                    boundary
                    if isinstance(boundary, BoundaryCondition)
                    else lookup_enum_or_raise(BoundaryCondition, boundary)
                )

            with ctx.parameter("constant_boundary"):
                TypeCheckError.check_instance_or_raise(
                    constant_boundary, None | int | float | str | Parameter
                )

                if (self._boundary != BoundaryCondition.constant) != (
                    constant_boundary is None
                ):
                    raise (
                        ValueError(
                            "must be provided if and only if the constant boundary condition is used"
                        )
                        | ctx
                    )

                if isinstance(constant_boundary, Parameter):
                    self._constant_boundary = constant_boundary
                elif isinstance(constant_boundary, str):
                    self._constant_boundary = Parameter(constant_boundary)
                else:
                    self._constant_boundary = constant_boundary

                if isinstance(
                    self._constant_boundary, Parameter
                ) and self._constant_boundary in ["$x", "$X"]:
                    raise (
                        ValueError(
                            f"must be scalar but late-bound constant data {self._constant_boundary} may not be"
                        )
                        | ctx
                    )

            with ctx.parameter("axis"):
                TypeCheckError.check_instance_or_raise(axis, None | int)
                self._axis = axis

    @property
    @override
    def late_bound(self) -> Set[Parameter]:
        return (
            frozenset([self._constant_boundary])
            if isinstance(self._constant_boundary, Parameter)
            else frozenset()
        )

    @override
    def compute_check_neighbourhood_for_data_shape(
        self, data_shape: tuple[int, ...]
    ) -> tuple[dict[BoundaryCondition, NeighbourhoodAxis], ...]:
        raise NotImplementedError()

    @override
    def check_pointwise(
        self,
        data: np.ndarray[S, np.dtype[T]],
        prediction: np.ndarray[S, np.dtype[T]],
        *,
        late_bound: Bindings,
        where: Literal[True] | np.ndarray[S, np.dtype[np.bool]] = True,
    ) -> np.ndarray[S, np.dtype[np.bool]]:
        with ctx.safeguard(self), ctx.parameter("constant_boundary"):
            constant_boundary = (
                None
                if self._constant_boundary is None
                else late_bound.resolve_ndarray_with_lossless_cast(
                    self._constant_boundary, (), data.dtype
                )
                if isinstance(self._constant_boundary, Parameter)
                else lossless_cast(self._constant_boundary, data.dtype)
            )

        ok: np.ndarray[S, np.dtype[np.bool]] = _ones(
            data.shape, dtype=np.dtype(np.bool)
        )

        window = 1 + self._window * 2

        for axis, alen in enumerate(data.shape):
            if (
                self._axis is not None
                and axis != self._axis
                and axis != (data.ndim + self._axis)
            ):
                continue

            if self._boundary == BoundaryCondition.valid and alen < window:
                continue

            if alen == 0:
                continue

            data_boundary = _pad_with_boundary(
                data,
                self._boundary,
                self._window,
                self._window,
                constant_boundary,
                axis,
            )
            prediction_boundary = _pad_with_boundary(
                prediction,
                self._boundary,
                self._window,
                self._window,
                constant_boundary,
                axis,
            )

            data_windows: np.ndarray[tuple[int, int], np.dtype[T]] = (
                _sliding_window_view(
                    data_boundary, window, axis=axis, writeable=False
                ).reshape((-1, window))
            )
            prediction_windows: np.ndarray[tuple[int, int], np.dtype[T]] = (
                _sliding_window_view(
                    prediction_boundary, window, axis=axis, writeable=False
                ).reshape((-1, window))
            )

            valid_slice = [slice(None)] * data.ndim
            if self._boundary == BoundaryCondition.valid:
                valid_slice[axis] = slice(self._window, -self._window)

            # optimization: only evaluate the monotonicity where necessary
            if where is not True:
                where_flat = where[tuple(valid_slice)].flatten()
                data_windows = np.compress(where_flat, data_windows, axis=0)
                prediction_windows = np.compress(where_flat, prediction_windows, axis=0)

            data_monotonic: np.ndarray[tuple[int], np.dtype[np.float64]] = (
                self._monotonic_sign(data_windows, is_prediction=False)  # type: ignore
            )
            prediction_monotonic: np.ndarray[tuple[int], np.dtype[np.float64]] = (
                self._monotonic_sign(prediction_windows, is_prediction=True)  # type: ignore
            )

            # for monotonic windows, check that the monotonicity matches
            axis_ok_ = self._monotonic_sign_not_equal(
                data_monotonic, prediction_monotonic
            )

            # the check succeeds where `where` is False
            axis_ok: np.ndarray[tuple[int], np.dtype[np.bool]]
            if where is True:
                axis_ok = axis_ok_
            else:
                axis_ok = _ones(where_flat.shape, np.dtype(np.bool))
                _place(axis_ok, where_flat, axis_ok_)

            # the check succeeds for boundary points that were excluded by a
            #  valid boundary
            ok[tuple(valid_slice)] &= ~axis_ok.reshape(ok[tuple(valid_slice)].shape)

        return ok

    @override
    def compute_safe_intervals(
        self,
        data: np.ndarray[S, np.dtype[T]],
        *,
        late_bound: Bindings,
        where: Literal[True] | np.ndarray[S, np.dtype[np.bool]] = True,
    ) -> IntervalUnion[T, int, int]:
        raise NotImplementedError()

    def compute_footprint(
        self,
        foot: np.ndarray[S, np.dtype[np.bool]],
        *,
        late_bound: Bindings,
        where: Literal[True] | np.ndarray[S, np.dtype[np.bool]] = True,
    ) -> np.ndarray[S, np.dtype[np.bool]]:
        raise NotImplementedError()

    def compute_inverse_footprint(
        self,
        foot: np.ndarray[S, np.dtype[np.bool]],
        *,
        late_bound: Bindings,
        where: Literal[True] | np.ndarray[S, np.dtype[np.bool]] = True,
    ) -> np.ndarray[S, np.dtype[np.bool]]:
        raise NotImplementedError()

    @override
    def get_config(self) -> dict[str, JSON]:
        config: dict[str, JSON] = dict(
            kind=type(self).kind,
            monotonicity=self._monotonicity.name,
            window=self._window,
            boundary=self._boundary.name,
            constant_boundary=str(self._constant_boundary)
            if isinstance(self._constant_boundary, Parameter)
            else self._constant_boundary,
            axis=self._axis,
        )

        if self._constant_boundary is None:
            del config["constant_boundary"]

        return config

    def _monotonic_sign(
        self,
        x: np.ndarray[S, np.dtype[T]],
        *,
        is_prediction: bool,
    ) -> np.ndarray[tuple[int, ...], np.dtype[np.float64]]:
        (lt, gt, eq, _is_weak) = self._monotonicity.value[int(is_prediction)]

        # default to NaN
        monotonic: np.ndarray[tuple[int, ...], np.dtype[np.float64]] = np.full(
            x.shape[:-1], np.nan
        )

        # use comparison instead of diff to account for uints

        # +1: all(x[i+1] > x[i])
        monotonic[np.all(gt(x[..., 1:], x[..., :-1]), axis=-1)] = +1
        # -1: all(x[i+1] < x[i])
        monotonic[np.all(lt(x[..., 1:], x[..., :-1]), axis=-1)] = -1

        # 0/NaN: all(x[i+1] == x[i])
        monotonic[np.all(x[..., 1:] == x[..., :-1], axis=-1)] = 0 if eq else np.nan

        # NaN values cannot participate in monotonic sequences
        # NaN: any(isnan(x[i]))
        monotonic[np.any(np.isnan(x), axis=-1)] = np.nan

        return monotonic

    def _monotonic_sign_not_equal(
        self,
        data_monotonic: np.ndarray[S, np.dtype[T]],
        prediction_monotonic: np.ndarray[S, np.dtype[T]],
    ) -> np.ndarray[S, np.dtype[np.bool]]:
        match self._monotonicity:
            case Monotonicity.strict | Monotonicity.strict_with_consts:
                return ~np.isnan(data_monotonic) & (
                    prediction_monotonic != data_monotonic
                )
            case Monotonicity.strict_to_weak | Monotonicity.weak:
                # having the opposite sign or no sign are both not equal
                return ~np.isnan(data_monotonic) & (
                    (prediction_monotonic == -data_monotonic)
                    | np.isnan(prediction_monotonic)
                )
            case _:
                assert_never(self._monotonicity)
