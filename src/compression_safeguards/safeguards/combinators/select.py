"""
Logical selector (switch case) combinator safeguard.
"""

__all__ = ["SelectSafeguard"]

from abc import ABC
from collections.abc import Collection, Set
from typing import ClassVar

import numpy as np
from typing_extensions import override  # MSPV 3.12

from ...utils.bindings import Bindings, Parameter
from ...utils.error import (
    ErrorContext,
    IndexErrorWithContext,
    TypeCheckError,
    ValueErrorWithContext,
)
from ...utils.intervals import IntervalUnion
from ...utils.typing import JSON, S, T
from ..abc import Safeguard
from ..pointwise.abc import PointwiseSafeguard
from ..stencil import BoundaryCondition, NeighbourhoodAxis
from ..stencil.abc import StencilSafeguard


class SelectSafeguard(Safeguard):
    """
    The `SelectSafeguard` guarantees that, for each element, the guarantees of
    the pointwise selected safeguard are upheld. This combinator allows
    selecting between several safeguards with per-element granularity.

    This combinator can be used to describe simple regions of interest where
    different safeguards, e.g. with different error bounds, are applied to
    different parts of the data.

    At the moment, only pointwise and stencil safeguards and combinations
    thereof can be combined by this select-combinator. The combinator is a
    pointwise or a stencil safeguard, depending on the safeguards it combines.

    Parameters
    ----------
    selector : str | Parameter
        Late-bound parameter name that is used to select between the
        `safeguards`.
    safeguards : Collection[dict[str, JSON] | PointwiseSafeguard | StencilSafeguard]
        At least one safeguard configuration [`dict`][dict]s or already
        initialized
        [`PointwiseSafeguard`][compression_safeguards.safeguards.pointwise.abc.PointwiseSafeguard]
        or
        [`StencilSafeguard`][compression_safeguards.safeguards.stencil.abc.StencilSafeguard].
    """

    __slots__: tuple[str, ...] = ()

    kind: ClassVar[str] = "select"

    def __init__(
        self,
        *,
        selector: str | Parameter,
        safeguards: Collection[dict[str, JSON] | PointwiseSafeguard | StencilSafeguard],
    ) -> None:
        pass

    def __new__(  # type: ignore
        cls,
        *,
        selector: str | Parameter,
        safeguards: Collection[dict[str, JSON] | PointwiseSafeguard | StencilSafeguard],
    ) -> "_SelectPointwiseSafeguard | _SelectStencilSafeguard":
        from ... import SafeguardKind  # noqa: PLC0415

        with ErrorContext().enter() as ctx, ctx.safeguardty(cls):
            with ctx.parameter("selector"):
                TypeCheckError.check_instance_or_raise(selector, str | Parameter)
                selector = (
                    selector if isinstance(selector, Parameter) else Parameter(selector)
                )

            with ctx.parameter("safeguards"):
                TypeCheckError.check_instance_or_raise(safeguards, Collection)

                if len(safeguards) <= 0:
                    raise ValueErrorWithContext(
                        "can only select over at least one safeguard"
                    )

                safeguards_: list[PointwiseSafeguard | StencilSafeguard] = []
                safeguard: dict[str, JSON] | Safeguard
                for i, safeguard in enumerate(safeguards):
                    with ctx.index(i):
                        TypeCheckError.check_instance_or_raise(
                            safeguard, dict | PointwiseSafeguard | StencilSafeguard
                        )
                        if isinstance(safeguard, dict):
                            safeguard = SafeguardKind.from_config(safeguard)
                        if not isinstance(
                            safeguard, PointwiseSafeguard | StencilSafeguard
                        ):
                            raise TypeCheckError(
                                PointwiseSafeguard | StencilSafeguard, safeguard
                            )
                        safeguards_.append(safeguard)

        if all(isinstance(safeguard, PointwiseSafeguard) for safeguard in safeguards_):
            return _SelectPointwiseSafeguard(selector, *safeguards_)  # type: ignore
        else:
            return _SelectStencilSafeguard(selector, *safeguards_)

    @property
    def selector(self) -> Parameter:  # type: ignore
        """
        The late-bound selector parameter.
        """
        ...

    @property
    def safeguards(self) -> tuple[PointwiseSafeguard | StencilSafeguard, ...]:  # type: ignore
        """
        The set of safeguards between which this combinator selects.
        """

        ...

    @property
    @override
    def late_bound(self) -> Set[Parameter]:  # type: ignore
        """
        The set of late-bound parameters that this safeguard has.

        Late-bound parameters are only bound when checking and applying the
        safeguard, in contrast to the normal early-bound parameters that are
        configured during safeguard initialisation.

        Late-bound parameters can be used for parameters that depend on the
        specific data that is to be safeguarded.
        """

        ...

    @override
    def check(  # type: ignore
        self,
        data: np.ndarray[S, np.dtype[T]],
        prediction: np.ndarray[S, np.dtype[T]],
        *,
        late_bound: Bindings,
    ) -> bool:
        """
        Check if, for all elements, the selected safeguard succeed the check.

        Parameters
        ----------
        data : np.ndarray[S, np.dtype[T]]
            Original data array, relative to which the `prediction` is checked.
        prediction : np.ndarray[S, np.dtype[T]]
            Prediction for the `data` array.
        late_bound : Bindings
            Bindings for late-bound parameters, including for this safeguard.

        Returns
        -------
        ok : bool
            `True` if the check succeeded.
        """

        ...

    def check_pointwise(  # type: ignore
        self,
        data: np.ndarray[S, np.dtype[T]],
        prediction: np.ndarray[S, np.dtype[T]],
        *,
        late_bound: Bindings,
    ) -> np.ndarray[S, np.dtype[np.bool]]:
        """
        Check for which elements the selected safeguard succeed the check.

        Parameters
        ----------
        data : np.ndarray[S, np.dtype[T]]
            Original data array, relative to which the `prediction` is checked.
        prediction : np.ndarray[S, np.dtype[T]]
            Prediction for the `data` array.
        late_bound : Bindings
            Bindings for late-bound parameters, including for this safeguard.

        Returns
        -------
        ok : np.ndarray[S, np.dtype[np.bool]]
            Pointwise, `True` if the check succeeded for this element.
        """

        ...

    def compute_safe_intervals(  # type: ignore
        self,
        data: np.ndarray[S, np.dtype[T]],
        *,
        late_bound: Bindings,
    ) -> IntervalUnion[T, int, int]:
        """
        Compute the safe intervals for the selected safeguard.

        Parameters
        ----------
        data : np.ndarray[S, np.dtype[T]]
            Data for which the safe intervals should be computed.
        late_bound : Bindings
            Bindings for late-bound parameters, including for this safeguard.

        Returns
        -------
        intervals : IntervalUnion[T, int, int]
            The safe intervals.
        """

        ...

    @override
    def get_config(self) -> dict[str, JSON]:  # type: ignore
        """
        Returns the configuration of the safeguard.

        Returns
        -------
        config : dict
            Configuration of the safeguard.
        """

        ...


class _SelectSafeguardBase(ABC):
    __slots__: tuple[str, ...] = ()

    kind: ClassVar[str] = "select"

    @property
    def selector(self) -> Parameter:
        return self._selector  # type: ignore

    @property
    def safeguards(self) -> tuple[PointwiseSafeguard | StencilSafeguard, ...]:
        return self._safeguards  # type: ignore

    @property
    def late_bound(self) -> Set[Parameter]:
        return frozenset(
            [self.selector] + [b for s in self.safeguards for b in s.late_bound]
        )

    def check_pointwise(
        self,
        data: np.ndarray[S, np.dtype[T]],
        prediction: np.ndarray[S, np.dtype[T]],
        *,
        late_bound: Bindings,
    ) -> np.ndarray[S, np.dtype[np.bool]]:
        with (
            ErrorContext().enter() as ctx,
            ctx.safeguardty(SelectSafeguard),
            ctx.parameter("selector"),
        ):
            selector = late_bound.resolve_ndarray_with_lossless_cast(
                self.selector, data.shape, np.dtype(np.int_)
            )

        oks = [
            safeguard.check_pointwise(data, prediction, late_bound=late_bound)
            for safeguard in self.safeguards
        ]

        try:
            return np.choose(selector, oks)  # type: ignore
        except IndexError as err:
            with (
                ErrorContext().enter() as ctx,
                ctx.parameter("selector"),
                ctx.late_bound_parameter(self.selector),
            ):
                raise IndexErrorWithContext(str(err)) from None

    def compute_safe_intervals(
        self,
        data: np.ndarray[S, np.dtype[T]],
        *,
        late_bound: Bindings,
    ) -> IntervalUnion[T, int, int]:
        with (
            ErrorContext().enter() as ctx,
            ctx.safeguardty(SelectSafeguard),
            ctx.parameter("selector"),
        ):
            selector = late_bound.resolve_ndarray_with_lossless_cast(
                self.selector, data.shape, np.dtype(np.int_)
            )

        valids = [
            safeguard.compute_safe_intervals(data, late_bound=late_bound)
            for safeguard in self.safeguards
        ]

        umax = max(valid.u for valid in valids)

        for i, v in enumerate(valids):
            vnew = IntervalUnion.empty(data.dtype, data.size, umax)
            vnew._lower[: v.u] = v._lower
            vnew._upper[: v.u] = v._upper
            valids[i] = vnew

        valid: IntervalUnion[T, int, int] = IntervalUnion.empty(
            data.dtype, data.size, umax
        )
        try:
            valid._lower = (
                np.take_along_axis(
                    np.stack([v._lower.T for v in valids], axis=0),
                    selector.flatten().reshape((1, data.size, 1)),
                    axis=0,
                )
                .reshape(data.size, umax)
                .T
            )
            valid._upper = (
                np.take_along_axis(
                    np.stack([v._upper.T for v in valids], axis=0),
                    selector.flatten().reshape((1, data.size, 1)),
                    axis=0,
                )
                .reshape(data.size, umax)
                .T
            )
        except IndexError as err:
            with (
                ErrorContext().enter() as ctx,
                ctx.parameter("selector"),
                ctx.late_bound_parameter(self.selector),
            ):
                raise IndexErrorWithContext(str(err)) from None

        return valid

    def get_config(self) -> dict[str, JSON]:
        return dict(
            kind=type(self).kind,
            selector=self.selector,
            safeguards=[safeguard.get_config() for safeguard in self.safeguards],
        )

    @override
    def __repr__(self) -> str:
        return f"{SelectSafeguard.__name__}(selector={self.selector!r}, safeguards={list(self.safeguards)!r})"


class _SelectPointwiseSafeguard(_SelectSafeguardBase, PointwiseSafeguard):
    __slots__: tuple[str, ...] = ("_selector", "_safeguards")
    _selector: Parameter
    _safeguards: tuple[PointwiseSafeguard, ...]

    def __init__(self, selector: Parameter, *safeguards: PointwiseSafeguard) -> None:
        self._selector = selector

        for safeguard in safeguards:
            assert isinstance(safeguard, PointwiseSafeguard), (
                f"{safeguard!r} is not a pointwise safeguard"
            )
        self._safeguards = safeguards


class _SelectStencilSafeguard(_SelectSafeguardBase, StencilSafeguard):
    __slots__: tuple[str, ...] = ("_selector", "_safeguards")
    _selector: Parameter
    _safeguards: tuple[PointwiseSafeguard | StencilSafeguard, ...]

    def __init__(
        self, selector: Parameter, *safeguards: PointwiseSafeguard | StencilSafeguard
    ) -> None:
        self._selector = selector

        for safeguard in safeguards:
            assert isinstance(safeguard, PointwiseSafeguard | StencilSafeguard), (
                f"{safeguard!r} is not a pointwise or stencil safeguard"
            )
        self._safeguards = safeguards

    @override
    def compute_check_neighbourhood_for_data_shape(
        self, data_shape: tuple[int, ...]
    ) -> tuple[dict[BoundaryCondition, NeighbourhoodAxis], ...]:
        neighbourhood: list[dict[BoundaryCondition, NeighbourhoodAxis]] = [
            dict() for _ in data_shape
        ]

        for safeguard in self._safeguards:
            if not isinstance(safeguard, StencilSafeguard):
                continue

            safeguard_neighbourhood = (
                safeguard.compute_check_neighbourhood_for_data_shape(data_shape)
            )

            for i, sn in enumerate(safeguard_neighbourhood):
                ni = neighbourhood[i]

                for b, s in sn.items():
                    if b in ni:
                        neighbourhood[i][b] = NeighbourhoodAxis(
                            max(ni[b].before, s.before), max(ni[b].after, s.after)
                        )
                    else:
                        neighbourhood[i][b] = s

        return tuple(neighbourhood)
