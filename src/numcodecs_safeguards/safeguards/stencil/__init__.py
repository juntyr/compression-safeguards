"""
Implementations for the provided stencil safeguards.
"""

__all__ = ["BoundaryCondition", "NeighbourhoodBoundaryAxis"]

from enum import Enum, auto

import numpy as np
from typing_extensions import Self  # MSPV 3.11


class BoundaryCondition(Enum):
    """
    Different types of boundary conditions that can be applied to the data
    array domain boundaries for
    [`StencilSafeguard`][numcodecs_safeguards.safeguards.stencil.abc.StencilSafeguard]s.

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
    """

    __slots__ = ("_before", "_after")
    _before: int
    _after: int

    def __init__(
        self,
        before: int,
        after: int,
    ):
        assert type(before) is int, "before must be an integer"
        assert before >= 0, "before must be non-negative"
        self._before = before

        assert type(after) is int, "after must be an integer"
        assert after >= 0, "after must be non-negative"
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
    constant_boundary : None | int | float
        Optional constant value with which the data array domain is extended
        for a constant boundary. The value must be safely convertable (without
        over- or underflow or invalid values) to the data type.
    """

    __slots__ = ("_axis", "_shape", "_boundary", "_constant_boundary")
    _axis: int
    _shape: NeighbourhoodAxis
    _boundary: BoundaryCondition
    _constant_boundary: None | int | float

    def __init__(
        self,
        axis: int,
        before: int,
        after: int,
        boundary: str | BoundaryCondition,
        constant_boundary: None | int | float = None,
    ):
        self._axis = axis
        self._shape = NeighbourhoodAxis(before, after)

        self._boundary = (
            boundary
            if isinstance(boundary, BoundaryCondition)
            else BoundaryCondition[boundary]
        )
        assert (self._boundary != BoundaryCondition.constant) == (
            constant_boundary is None
        ), (
            "constant_boundary must be provided if and only if the constant boundary condition is used"
        )
        self._constant_boundary = constant_boundary

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
    def constant_boundary(self) -> None | int | float:
        """
        Optional constant value with which the data array domain is extended
        for a constant boundary.
        """
        return self._constant_boundary

    def get_config(self) -> dict:
        """
        Returns the configuration of the data neighbourhood.

        Returns
        -------
        config : dict
            Configuration of the data neighbourhood.
        """

        config = dict(
            axis=self.axis,
            before=self.before,
            after=self.after,
            boundary=self.boundary.name,
            constant_boundary=self.constant_boundary,
        )

        if self.constant_boundary is None:
            del config["constant_boundary"]

        return config

    @classmethod
    def from_config(cls, config: dict) -> Self:
        """
        Instantiate the data neighbourhood from a configuration [`dict`][dict].

        Parameters
        ----------
        config : dict
            Configuration of the data neighbourhood.

        Returns
        -------
        neighbourhood : Self
            Instantiated data neighbourhood.
        """

        return cls(**config)

    def __repr__(self) -> str:
        return f"{type(self).__name__}({', '.join(f'{k}={v!r}' for k, v in self.get_config().items())})"


def _pad_with_boundary(
    a: np.ndarray,
    boundary: BoundaryCondition,
    pad_before: int,
    pad_after: int,
    constant: None | int | float,
    axis: int,
) -> np.ndarray:
    if (axis >= a.ndim) or (axis < -a.ndim):
        return a

    pad_width = [(0, 0)] * a.ndim
    pad_width[axis] = (pad_before, pad_after)

    kwargs = dict()
    match boundary:
        case BoundaryCondition.valid:
            return a
        case BoundaryCondition.constant:
            mode = "constant"
            try:
                with np.errstate(over="raise", under="raise", invalid="raise"):
                    kwargs["constant_values"] = np.array(constant, dtype=a.dtype)
            except Exception as err:
                raise ValueError(
                    f"constant boundary has invalid value {constant} for data array of dtype {a.dtype.name}"
                ) from err
        case BoundaryCondition.edge:
            mode = "edge"
        case BoundaryCondition.reflect:
            mode = "reflect"
            kwargs["reflect_type"] = "even"  # type: ignore
        case BoundaryCondition.symmetric:
            mode = "symmetric"
            kwargs["reflect_type"] = "even"  # type: ignore
        case BoundaryCondition.wrap:
            mode = "wrap"

    return np.pad(a, pad_width, mode, **kwargs)  # type: ignore
