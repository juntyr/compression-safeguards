"""
Implementations for the provided [`StencilSafeguard`][compression_safeguards.safeguards.stencil.abc.StencilSafeguard]s.
"""

__all__ = ["BoundaryCondition", "NeighbourhoodAxis", "NeighbourhoodBoundaryAxis"]

from enum import Enum, auto
from functools import reduce
from typing import Literal, Self, assert_never

import numpy as np
from typing_extensions import override  # MSPV 3.12

from ...utils._compat import _sliding_window_view
from ...utils.bindings import Parameter
from ...utils.error import TypeCheckError, ctx, lookup_enum_or_raise
from ...utils.typing import JSON, TB, S


class BoundaryCondition(Enum):
    """
    Different types of boundary conditions that can be applied to the data
    array domain boundaries for [`StencilSafeguard`][..abc.StencilSafeguard]s.

    Since stencil safeguards operate over small neighbourhoods of data points,
    points at the boundary, where part of the neighbourhood may not exist, need
    to be treated specially.
    """

    valid = auto()
    """
    The boundary is not extended, instead the safeguard is only applied to
    and checked for points where the entire neighbourhood is valid.
    """

    constant = auto()
    """
    The boundary is extended by a constant value.
    """

    edge = auto()
    """
    The boundary is extended by the edge value.
    """

    reflect = auto()
    """
    The boundary is extended by reflecting along the edge value. The edge value
    itself is not repeated.
    """

    symmetric = auto()
    """
    The boundary is extended by reflecting after the edge value. The edge value
    itself is repeated as well.
    """

    wrap = auto()
    """
    The boundary is extended by wrapping the domain around, as if the domain was
    on a torus (Pac-Man style).
    """


class NeighbourhoodAxis:
    """
    Specification of the shape of the data neighbourhood along a single axis.

    Parameters
    ----------
    before : int
        The non-negative number of values to include before the centre of a
        data neighbourhood.

        e.g. setting `before=1` means that the neighbourhood contains the
        previous value.
    after : int
        The non-negative number of values to include after the centre of a
        data neighbourhood.

        e.g. setting `after=2` means that the neighbourhood contains the
        two next values.

    Raises
    ------
    TypeCheckError
        if any parameter has the wrong type.
    ValueError
        if `before` or `after` is negative.
    """

    __slots__: tuple[str, ...] = ("_before", "_after")
    _before: int
    _after: int

    def __init__(
        self,
        before: int,
        after: int,
    ) -> None:
        with ctx.parameter("before"):
            TypeCheckError.check_instance_or_raise(before, int)
            if before < 0:
                raise ValueError("must be non-negative") | ctx
            self._before = before

        with ctx.parameter("after"):
            TypeCheckError.check_instance_or_raise(after, int)
            if after < 0:
                raise ValueError("must be non-negative") | ctx
            self._after = after

    @property
    def before(self) -> int:
        """
        The non-negative number of values to include before the centre of a
        data neighbourhood.
        """
        return self._before

    @property
    def after(self) -> int:
        """
        The non-negative number of values to include after the centre of a
        data neighbourhood.
        """
        return self._after

    @override
    def __repr__(self) -> str:
        return f"{type(self).__name__}(before={self.before}, after={self.after})"


class NeighbourhoodBoundaryAxis:
    """
    Specification of the shape of the data neighbourhood and its boundary
    condition along a single axis.

    Parameters
    ----------
    axis : int
        The axis along which the boundary condition is applied.
    before : int
        The non-negative number of values to include before the centre of a
        data neighbourhood.

        e.g. setting `before=1` means that the neighbourhood contains the
        previous value.
    after : int
        The non-negative number of values to include after the centre of a
        data neighbourhood.

        e.g. setting `after=2` means that the neighbourhood contains the
        two next values.
    boundary : str | BoundaryCondition
        The boundary condition that is applied to this axis near the data
        array domain boundary to fill the data neighbourhood, e.g. by extending
        values.
    constant_boundary : None | int | float | str | Parameter
        The optional value of or the late-bound parameter name for the constant
        value with which the data array domain is extended for a constant
        boundary. The value must be losslessly convertible to the data dtype.

    Raises
    ------
    TypeCheckError
        if any parameter has the wrong type.
    ValueError
        if `before` or `after` is negative.
    ValueError
        if `boundary` does not name a valid boundary condition variant.
    ValueError
        if `constant_boundary` is, not, provided if and only if the `boundary`
        is constant.
    ValueError
        if `constant_boundary` uses the non-scalar `$x` or `$X` late-bound
        parameters.
    """

    __slots__: tuple[str, ...] = ("_axis", "_shape", "_boundary", "_constant_boundary")
    _axis: int
    _shape: NeighbourhoodAxis
    _boundary: BoundaryCondition
    _constant_boundary: None | int | float | Parameter

    def __init__(
        self,
        axis: int,
        before: int,
        after: int,
        boundary: str | BoundaryCondition,
        constant_boundary: None | int | float | str | Parameter = None,
    ) -> None:
        with ctx.parameter("axis"):
            TypeCheckError.check_instance_or_raise(axis, int)
            self._axis = axis

        self._shape = NeighbourhoodAxis(before, after)

        with ctx.parameter("boundary"):
            TypeCheckError.check_instance_or_raise(boundary, str | BoundaryCondition)
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
                        "must be provided if and only if the constant "
                        + "boundary condition is used"
                    )
                    | ctx
                )

            if isinstance(constant_boundary, Parameter):
                self._constant_boundary = constant_boundary
            elif isinstance(constant_boundary, str):
                self._constant_boundary = Parameter(constant_boundary)
            else:
                self._constant_boundary = constant_boundary

            if isinstance(self._constant_boundary, Parameter):
                if self._constant_boundary in ["$x", "$X"]:
                    raise (
                        ValueError(
                            "must be a scalar but late-bound constant data "
                            + f"{self._constant_boundary} may not be"
                        )
                        | ctx
                    )

    @property
    def axis(self) -> int:
        """
        The axis along which the boundary condition is applied.
        """
        return self._axis

    @property
    def before(self) -> int:
        """
        The non-negative number of values to include before the centre of a
        data neighbourhood.
        """
        return self._shape.before

    @property
    def after(self) -> int:
        """
        The non-negative number of values to include after the centre of a
        data neighbourhood.
        """
        return self._shape.after

    @property
    def shape(self) -> NeighbourhoodAxis:
        """
        The shape of the data neighbourhood.
        """
        return self._shape

    @property
    def boundary(self) -> BoundaryCondition:
        """
        The boundary condition that is applied to this axis near the data
        array domain boundary to fill the data neighbourhood, e.g. by extending
        values.
        """
        return self._boundary

    @property
    def constant_boundary(self) -> None | int | float | Parameter:
        """
        The optional value of or the late-bound parameter name for the constant
        value with which the data array domain is extended for a constant
        boundary.
        """
        return self._constant_boundary

    def get_config(self) -> dict[str, JSON]:
        """
        Returns the configuration of the data neighbourhood.

        Returns
        -------
        config : dict[str, JSON]
            Configuration of the data neighbourhood.
        """

        config: dict[str, JSON] = dict(
            axis=self.axis,
            before=self.before,
            after=self.after,
            boundary=self.boundary.name,
            constant_boundary=str(self.constant_boundary)
            if isinstance(self.constant_boundary, Parameter)
            else self.constant_boundary,
        )

        if self.constant_boundary is None:
            del config["constant_boundary"]

        return config

    @classmethod
    def from_config(cls, config: dict[str, JSON]) -> Self:
        """
        Instantiate the data neighbourhood from a configuration [`dict`][dict].

        Parameters
        ----------
        config : dict[str, JSON]
            Configuration of the data neighbourhood.

        Returns
        -------
        neighbourhood : Self
            Instantiated data neighbourhood.
        """

        return cls(**config)  # type: ignore

    @override
    def __repr__(self) -> str:
        return f"{type(self).__name__}({', '.join(f'{k}={v!r}' for k, v in self.get_config().items())})"


def _pad_with_boundary(
    a: np.ndarray[S, np.dtype[TB]],
    boundary: BoundaryCondition,
    pad_before: int,
    pad_after: int,
    constant: None | np.ndarray[tuple[()], np.dtype[TB]],
    axis: int,
) -> np.ndarray[tuple[int, ...], np.dtype[TB]]:
    if (axis >= a.ndim) or (axis < -a.ndim):
        return a

    pad_width = [(0, 0)] * a.ndim
    pad_width[axis] = (pad_before, pad_after)

    kwargs: dict[str, None | str | np.ndarray[tuple[()], np.dtype[TB]]] = dict()
    match boundary:
        case BoundaryCondition.valid:
            return a
        case BoundaryCondition.constant:
            mode = "constant"
            kwargs["constant_values"] = constant
        case BoundaryCondition.edge:
            mode = "edge"
        case BoundaryCondition.reflect:
            mode = "reflect"
            kwargs["reflect_type"] = "even"
        case BoundaryCondition.symmetric:
            mode = "symmetric"
            kwargs["reflect_type"] = "even"
        case BoundaryCondition.wrap:
            mode = "wrap"
        case _:
            assert_never(boundary)

    return np.pad(a, pad_width, mode, **kwargs)  # type: ignore


def _reverse_neighbourhood_indices(
    data_shape: tuple[int, ...],
    neighbourhood: tuple[NeighbourhoodBoundaryAxis, ...],
    window_used: np.ndarray[tuple[int, ...], np.dtype[np.bool]],
    where_flat: Literal[True] | np.ndarray[tuple[int], np.dtype[np.bool]],
) -> np.ndarray[tuple[int, int], np.dtype[np.intp]]:
    data_size = reduce(lambda x, y: x * y, data_shape, 1)

    window = tuple(axis.before + 1 + axis.after for axis in neighbourhood)
    window_size = reduce(lambda x, y: x * y, window, 1)

    # compute how the data indices are distributed into windows
    # i.e. for each derived element, which data does it depend on
    indices_boundary = np.arange(data_size).reshape(data_shape)
    for axis in neighbourhood:
        indices_boundary = _pad_with_boundary(
            indices_boundary,
            axis.boundary,
            axis.before,
            axis.after,
            None if axis.constant_boundary is None else np.full((), data_size),
            axis.axis,
        )
    indices_windows = _sliding_window_view(
        indices_boundary,
        window,
        axis=tuple(axis.axis for axis in neighbourhood),
        writeable=False,
    ).reshape((-1, window_size))

    # compute the reverse: for each data element, which windows is it in
    # i.e. for each data element, which derived elements does it contribute to
    #      and thus which data bounds affect it
    reverse_indices_windows = np.full(
        (data_size, np.sum(window_used.astype(int))), indices_windows.size
    )
    reverse_indices_counter = np.zeros(data_size, dtype=np.intp)
    for i, u in enumerate(window_used.flat):
        # skip window indices that are not used
        if not u:
            continue
        # manual loop to account for potential aliasing:
        # with a wrapping boundary, more than one j for the same window
        #  position j could refer back to the same data element
        for j in range(indices_windows.shape[0]):
            # skip back-contributions from data elements where the safety
            #  requirements are disabled
            if (where_flat is not True) and (not where_flat[j]):
                continue
            idx = indices_windows[j, i]
            if idx != data_size:
                # lazily allocate more to account for all possible edge cases
                if reverse_indices_counter[idx] >= reverse_indices_windows.shape[1]:
                    new_reverse_indices_windows = np.full(
                        (data_size, reverse_indices_windows.shape[1] * 2),
                        indices_windows.size,
                    )
                    new_reverse_indices_windows[
                        :, : reverse_indices_windows.shape[1]
                    ] = reverse_indices_windows
                    reverse_indices_windows = new_reverse_indices_windows
                # update the reverse mapping
                reverse_indices_windows[idx][reverse_indices_counter[idx]] = (
                    j * window_used.size
                ) + i
                reverse_indices_counter[idx] += 1

    return reverse_indices_windows
