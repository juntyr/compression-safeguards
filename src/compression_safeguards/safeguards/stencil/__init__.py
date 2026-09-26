"""
Implementations for the provided [`StencilSafeguard`][compression_safeguards.safeguards.stencil.abc.StencilSafeguard]s.
"""

__all__ = ["BoundaryCondition", "NeighbourhoodAxis", "NeighbourhoodBoundaryAxis"]

from enum import Enum, auto
from functools import reduce
from typing import Literal, Self, assert_never

import numpy as np
from typing_extensions import override  # MSPV 3.12

from ...utils._compat import _reshape, _sliding_window_view
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

    indices_windows: np.ndarray[tuple[int, int] | tuple[int], np.dtype[np.int_]]
    indices_windows = _reshape(
        _sliding_window_view(
            indices_boundary,
            window,
            axis=tuple(axis.axis for axis in neighbourhood),
            writeable=False,
        ),
        (-1, window_size),
    )

    # track the indices of the window indices
    indices_windows_indices = np.arange(indices_windows.size).reshape(
        indices_windows.shape
    )

    fill_value = indices_windows.size

    # skip back-contributions from data elements where the safety requirements
    #  are disabled
    if where_flat is not True:
        indices_windows = indices_windows[where_flat]
        indices_windows_indices = indices_windows_indices[where_flat]

    # skip window indices that are not used
    indices_windows = indices_windows[:, window_used.flatten()]
    indices_windows_indices = indices_windows_indices[:, window_used.flatten()]

    indices_windows = indices_windows.flatten()
    indices_windows_indices = indices_windows_indices.flatten()

    # sort the indices, such that windows that read the same data are together
    # use a stable sort to ensure consistent results, independent of chunking
    argindices = np.argsort(indices_windows, stable=True)
    indices_windows_sorted = indices_windows[argindices]

    # indices_windows might include fill values, of value data_size, which
    #  represent constant values that come from no data index
    # exclude those, conveniently largest values, from indices_windows_sorted
    #  to ensure that we only track valid data indices
    only_fill_index = np.searchsorted(indices_windows_sorted, data_size)
    argindices = argindices[:only_fill_index]
    indices_windows_sorted = indices_windows_sorted[:only_fill_index]

    # find the starts of the runs of common indices
    indices_run_starts = np.r_[
        0, np.flatnonzero(indices_windows_sorted[1:] != indices_windows_sorted[:-1]) + 1
    ]

    # find the inverse mapping from sorted indices to their unique indices
    _, indices_windows_sorted_inverse = np.unique(
        indices_windows_sorted, return_inverse=True, sorted=True
    )

    # find the offsets inside each index run, e.g. for a sequence
    #  [a, b, b, b, c, d, d],
    # the offsets will be
    #  [0, 0, 1, 2, 0, 0, 1]
    indices_run_offsets = (
        np.arange(indices_windows_sorted.size)
        - indices_run_starts[indices_windows_sorted_inverse]
    )

    indices_max_run_length = np.amax(indices_run_offsets, initial=-1) + 1

    # compute the reverse: for each data element, which windows is it in
    # i.e. for each data element, which derived elements does it contribute to
    #      and thus which data bounds affect it
    reverse_indices_windows = np.full(data_size * indices_max_run_length, fill_value)
    # store the reverse mapping
    #  - this is complicated since each data element may be referenced by
    #    multiple windows, and we need to ensure that they don't override
    #    each other's contributions when run with vectorisation
    #  - so we precompute a unique run-slot for each back-reference
    #  - since we sorted the indices earlier to find the runs, we also need
    #    to apply the same reordering to the back-references
    reverse_indices_windows[
        indices_windows_sorted * indices_max_run_length + indices_run_offsets
    ] = indices_windows_indices[argindices]
    reverse_indices_windows = reverse_indices_windows.reshape(
        data_size, indices_max_run_length
    )

    return reverse_indices_windows
