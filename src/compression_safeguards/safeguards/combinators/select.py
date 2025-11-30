"""
Logical selector (switch case) combinator safeguard.
"""

__all__ = ["SelectSafeguard"]

from abc import ABC
from collections.abc import Collection, Set
from typing import ClassVar, Literal

import numpy as np
from typing_extensions import override  # MSPV 3.12

from ...utils.bindings import Bindings, Parameter
from ...utils.error import TypeCheckError, ctx
from ...utils.intervals import Interval, IntervalUnion
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
    selector : int | str | Parameter
        Literal index or late-bound parameter name that is used to select
        between the `safeguards`.
    safeguards : Collection[dict[str, JSON] | PointwiseSafeguard | StencilSafeguard]
        At least one safeguard configuration [`dict`][dict]s or already
        initialized [`PointwiseSafeguard`][....pointwise.abc.PointwiseSafeguard]
        or [`StencilSafeguard`][....stencil.abc.StencilSafeguard].

    Raises
    ------
    TypeCheckError
        if any parameter has the wrong type.
    ValueError
        if the `safeguards` collection is empty.
    IndexError
        if the `selector` is a literal [`int`][int] index that is not within
        the `0 <= selector < len(safeguards)` bounds.
    ...
        if instantiating a safeguard raises an exception.
    """

    __slots__: tuple[str, ...] = ()

    kind: ClassVar[str] = "select"

    def __init__(
        self,
        *,
        selector: int | str | Parameter,
        safeguards: Collection[dict[str, JSON] | PointwiseSafeguard | StencilSafeguard],
    ) -> None:
        pass

    def __new__(  # type: ignore
        cls,
        *,
        selector: int | str | Parameter,
        safeguards: Collection[dict[str, JSON] | PointwiseSafeguard | StencilSafeguard],
    ) -> "_SelectPointwiseSafeguard | _SelectStencilSafeguard":
        from ... import SafeguardKind  # noqa: PLC0415

        with ctx.safeguardty(cls):
            with ctx.parameter("selector"):
                TypeCheckError.check_instance_or_raise(selector, int | str | Parameter)
                selector = (
                    Parameter(selector) if isinstance(selector, str) else selector
                )

            with ctx.parameter("safeguards"):
                TypeCheckError.check_instance_or_raise(safeguards, Collection)

                if len(safeguards) <= 0:
                    raise (
                        ValueError("can only select over at least one safeguard") | ctx
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
                        TypeCheckError.check_instance_or_raise(
                            safeguard, PointwiseSafeguard | StencilSafeguard
                        )
                        safeguards_.append(safeguard)  # type: ignore

            with ctx.parameter("selector"):
                if isinstance(selector, int) and (
                    (selector < 0) or (selector >= len(safeguards_))
                ):
                    raise IndexError("invalid index") | ctx

        pointwise_safeguards_: list[PointwiseSafeguard] = [
            safeguard
            for safeguard in safeguards_
            if isinstance(safeguard, PointwiseSafeguard)
        ]

        if len(pointwise_safeguards_) == len(safeguards_):
            return _SelectPointwiseSafeguard(selector, *pointwise_safeguards_)
        else:
            return _SelectStencilSafeguard(selector, *safeguards_)

    @property
    def selector(self) -> int | Parameter:  # type: ignore
        """
        The selector index or late-bound selector parameter.
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
        where: Literal[True] | np.ndarray[S, np.dtype[np.bool]] = True,
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
        where : Literal[True] | np.ndarray[S, np.dtype[np.bool]]
            Only check at data points where the condition is [`True`][True].

        Returns
        -------
        ok : bool
            `True` if the check succeeded.

        Raises
        ------
        LateBoundParameterResolutionError
            if the `selector`'s late-bound parameter is not in `late_bound`.
        ValueError
            if the late-bound `selector` could not be broadcast to the `data`'s
            shape.
        TypeError
            if the late-bound `selector` is floating-point.
        ValueError
            if not all late-bound `selector` values could be losslessly
            converted to integer indices.
        IndexError
            if the late-bound `selector` indices are invalid for the
            selected-over `safeguards`.
        ...
            if checking a safeguard raises an exception.
        """

        ...

    def check_pointwise(  # type: ignore
        self,
        data: np.ndarray[S, np.dtype[T]],
        prediction: np.ndarray[S, np.dtype[T]],
        *,
        late_bound: Bindings,
        where: Literal[True] | np.ndarray[S, np.dtype[np.bool]] = True,
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
        where : Literal[True] | np.ndarray[S, np.dtype[np.bool]]
            Only check at data points where the condition is [`True`][True].

        Returns
        -------
        ok : np.ndarray[S, np.dtype[np.bool]]
            Pointwise, `True` if the check succeeded for this element.

        Raises
        ------
        LateBoundParameterResolutionError
            if the `selector`'s late-bound parameter is not in `late_bound`.
        ValueError
            if the late-bound `selector` could not be broadcast to the `data`'s
            shape.
        TypeError
            if the late-bound `selector` is floating-point.
        ValueError
            if not all late-bound `selector` values could be losslessly
            converted to integer indices.
        IndexError
            if the late-bound `selector` indices are invalid for the
            selected-over `safeguards`.
        ...
            if checking a safeguard raises an exception.
        """

        ...

    def compute_safe_intervals(  # type: ignore
        self,
        data: np.ndarray[S, np.dtype[T]],
        *,
        late_bound: Bindings,
        where: Literal[True] | np.ndarray[S, np.dtype[np.bool]] = True,
    ) -> IntervalUnion[T, int, int]:
        """
        Compute the safe intervals for the selected safeguard.

        Parameters
        ----------
        data : np.ndarray[S, np.dtype[T]]
            Data for which the safe intervals should be computed.
        late_bound : Bindings
            Bindings for late-bound parameters, including for this safeguard.
        where : Literal[True] | np.ndarray[S, np.dtype[np.bool]]
            Only compute the safe intervals at pointwise checks where the
            condition is [`True`][True].

        Returns
        -------
        intervals : IntervalUnion[T, int, int]
            The safe intervals.

        Raises
        ------
        LateBoundParameterResolutionError
            if the `selector`'s late-bound parameter is not in `late_bound`.
        ValueError
            if the late-bound `selector` could not be broadcast to the `data`'s
            shape.
        TypeError
            if the late-bound `selector` is floating-point.
        ValueError
            if not all late-bound `selector` values could be losslessly
            converted to integer indices.
        IndexError
            if the late-bound `selector` indices are invalid for the
            selected-over `safeguards`.
        ...
            if computing the safe intervals for a safeguard raises an exception.
        """

        ...

    def compute_footprint(  # type: ignore
        self,
        foot: np.ndarray[S, np.dtype[np.bool]],
        *,
        late_bound: Bindings,
        where: Literal[True] | np.ndarray[S, np.dtype[np.bool]] = True,
    ) -> np.ndarray[S, np.dtype[np.bool]]:
        """
        Compute the footprint of the `foot` array, e.g. for expanding data
        points into the pointwise checks that they contribute to.

        Parameters
        ----------
        foot : np.ndarray[S, np.dtype[np.bool]]
            Array for which the footprint is computed.
        late_bound : Bindings
            Bindings for late-bound parameters, including for this safeguard.
        where : Literal[True] | np.ndarray[S, np.dtype[np.bool]]
            Only compute the inverse footprint at pointwise checks where the
            condition is [`True`][True].

        Returns
        -------
        print : np.ndarray[S, np.dtype[np.bool]]
            The footprint of the `foot` array.
        """

        ...

    def compute_inverse_footprint(  # type: ignore
        self,
        foot: np.ndarray[S, np.dtype[np.bool]],
        *,
        late_bound: Bindings,
        where: Literal[True] | np.ndarray[S, np.dtype[np.bool]] = True,
    ) -> np.ndarray[S, np.dtype[np.bool]]:
        """
        Compute the inverse footprint of the `foot` array, e.g. for expanding
        pointwise check fails into the points that could have contributed to
        the failures.

        Parameters
        ----------
        foot : np.ndarray[S, np.dtype[np.bool]]
            Array for which the inverse footprint is computed.
        late_bound : Bindings
            Bindings for late-bound parameters, including for this safeguard.
        where : Literal[True] | np.ndarray[S, np.dtype[np.bool]]
            Only compute the inverse footprint at pointwise checks where the
            condition is [`True`][True].

        Returns
        -------
        print : np.ndarray[S, np.dtype[np.bool]]
            The inverse footprint of the `foot` array.
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
    def selector(self) -> int | Parameter:
        return self._selector  # type: ignore

    @property
    def safeguards(self) -> tuple[PointwiseSafeguard | StencilSafeguard, ...]:
        return self._safeguards  # type: ignore

    @property
    def late_bound(self) -> Set[Parameter]:
        parameters = frozenset(b for s in self.safeguards for b in s.late_bound)

        if isinstance(self.selector, Parameter):
            parameters = parameters.union([self.selector])

        return parameters

    def check_pointwise(
        self,
        data: np.ndarray[S, np.dtype[T]],
        prediction: np.ndarray[S, np.dtype[T]],
        *,
        late_bound: Bindings,
        where: Literal[True] | np.ndarray[S, np.dtype[np.bool]] = True,
    ) -> np.ndarray[S, np.dtype[np.bool]]:
        # if a single safeguard is selected, just forward to it
        if isinstance(self.selector, int):
            return self.safeguards[self.selector].check_pointwise(
                data, prediction, late_bound=late_bound, where=where
            )

        with ctx.safeguardty(SelectSafeguard), ctx.parameter("selector"):
            selector = late_bound.resolve_ndarray_with_lossless_cast(
                self.selector, data.shape, np.dtype(np.intp)
            )

            with ctx.late_bound_parameter(self.selector):
                if np.any(selector < 0) or np.any(selector >= len(self.safeguards)):
                    raise IndexError("invalid indices") | ctx

        # only perform the checks where selected
        oks = [
            safeguard.check_pointwise(
                data, prediction, late_bound=late_bound, where=(selector == i) & where
            )
            for i, safeguard in enumerate(self.safeguards)
        ]

        # choose the check result from the selected safeguard
        with (
            ctx.safeguardty(SelectSafeguard),
            ctx.parameter("selector"),
            ctx.late_bound_parameter(self.selector),
        ):
            return np.choose(selector, oks)  # type: ignore

    def compute_safe_intervals(
        self,
        data: np.ndarray[S, np.dtype[T]],
        *,
        late_bound: Bindings,
        where: Literal[True] | np.ndarray[S, np.dtype[np.bool]] = True,
    ) -> IntervalUnion[T, int, int]:
        # if a single safeguard is selected, just forward to it
        if isinstance(self.selector, int):
            return self.safeguards[self.selector].compute_safe_intervals(
                data, late_bound=late_bound, where=where
            )

        with ctx.safeguardty(SelectSafeguard), ctx.parameter("selector"):
            selector = late_bound.resolve_ndarray_with_lossless_cast(
                self.selector, data.shape, np.dtype(np.intp)
            )

            with ctx.late_bound_parameter(self.selector):
                if np.any(selector < 0) or np.any(selector >= len(self.safeguards)):
                    raise IndexError("invalid indices") | ctx

        valid: IntervalUnion[T, int, int] = Interval.full_like(data).into_union()

        # compute the safe intervals only where needed and then intersect
        # the `where` condition ensures that we only compute restrictions for
        #  the data points for which a safeguard was selected (and that other
        #  points do not impose such requirements)
        # but that the computed requirements can extend beyond the pointwise
        for i, safeguard in enumerate(self.safeguards):
            where_i = (selector == i) & where
            if np.any(where_i):
                valid = valid.intersect(
                    safeguard.compute_safe_intervals(
                        data, late_bound=late_bound, where=where_i
                    )
                )

        return valid

    def compute_footprint(
        self,
        foot: np.ndarray[S, np.dtype[np.bool]],
        *,
        late_bound: Bindings,
        where: Literal[True] | np.ndarray[S, np.dtype[np.bool]] = True,
    ) -> np.ndarray[S, np.dtype[np.bool]]:
        # if a single safeguard is selected, just forward to it
        if isinstance(self.selector, int):
            return self.safeguards[self.selector].compute_footprint(
                foot, late_bound=late_bound, where=where
            )

        with ctx.safeguardty(SelectSafeguard), ctx.parameter("selector"):
            selector = late_bound.resolve_ndarray_with_lossless_cast(
                self.selector, foot.shape, np.dtype(np.intp)
            )

            with ctx.late_bound_parameter(self.selector):
                if np.any(selector < 0) or np.any(selector >= len(self.safeguards)):
                    raise IndexError("invalid indices") | ctx

        footprint = np.zeros_like(foot)

        # per point, only the selected safeguard contributes to the combined
        #  footprint
        # the union of the footprints is taken since the per-point footprint
        #  from a stencil safeguard may extend to neighbouring points
        for i, safeguard in enumerate(self.safeguards):
            where_i = (selector == i) & where
            if np.any(where_i):
                footprint |= safeguard.compute_footprint(
                    foot, late_bound=late_bound, where=where_i
                )

        return footprint

    def compute_inverse_footprint(
        self,
        foot: np.ndarray[S, np.dtype[np.bool]],
        *,
        late_bound: Bindings,
        where: Literal[True] | np.ndarray[S, np.dtype[np.bool]] = True,
    ) -> np.ndarray[S, np.dtype[np.bool]]:
        # if a single safeguard is selected, just forward to it
        if isinstance(self.selector, int):
            return self.safeguards[self.selector].compute_inverse_footprint(
                foot, late_bound=late_bound, where=where
            )

        with ctx.safeguardty(SelectSafeguard), ctx.parameter("selector"):
            selector = late_bound.resolve_ndarray_with_lossless_cast(
                self.selector, foot.shape, np.dtype(np.intp)
            )

            with ctx.late_bound_parameter(self.selector):
                if np.any(selector < 0) or np.any(selector >= len(self.safeguards)):
                    raise IndexError("invalid indices") | ctx

        footprint = np.zeros_like(foot)

        # per point, only the selected safeguard contributes to the combined
        #  footprint
        # the union of the footprints is taken since the per-point footprint
        #  from a stencil safeguard may extend to neighbouring points
        for i, safeguard in enumerate(self.safeguards):
            where_i = (selector == i) & where
            if np.any(where_i):
                footprint |= safeguard.compute_inverse_footprint(
                    foot, late_bound=late_bound, where=where_i
                )

        return footprint

    def get_config(self) -> dict[str, JSON]:
        return dict(
            kind=type(self).kind,
            selector=str(self.selector)
            if isinstance(self.selector, Parameter)
            else self.selector,
            safeguards=[safeguard.get_config() for safeguard in self.safeguards],
        )

    @override
    def __repr__(self) -> str:
        return f"{SelectSafeguard.__name__}(selector={self.selector!r}, safeguards={list(self.safeguards)!r})"


class _SelectPointwiseSafeguard(_SelectSafeguardBase, PointwiseSafeguard):
    __slots__: tuple[str, ...] = ("_selector", "_safeguards")
    _selector: int | Parameter
    _safeguards: tuple[PointwiseSafeguard, ...]

    def __init__(
        self, selector: int | Parameter, *safeguards: PointwiseSafeguard
    ) -> None:
        self._selector = selector

        for safeguard in safeguards:
            assert isinstance(safeguard, PointwiseSafeguard), (
                f"{safeguard!r} is not a pointwise safeguard"
            )
        self._safeguards = safeguards


class _SelectStencilSafeguard(_SelectSafeguardBase, StencilSafeguard):
    __slots__: tuple[str, ...] = ("_selector", "_safeguards")
    _selector: int | Parameter
    _safeguards: tuple[PointwiseSafeguard | StencilSafeguard, ...]

    def __init__(
        self,
        selector: int | Parameter,
        *safeguards: PointwiseSafeguard | StencilSafeguard,
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
