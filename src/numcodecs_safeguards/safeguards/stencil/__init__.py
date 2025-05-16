"""
Implementations for the provided stencil safeguards.
"""

__all__ = ["BoundaryCondition"]

from enum import Enum, auto

import numpy as np


class BoundaryCondition(Enum):
    """
    Different types of boundary conditions that can be applied to the data
    domain boundaries for
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


def _pad_with_boundary_single(
    a: np.ndarray,
    boundary: BoundaryCondition,
    pad_before: int,
    pad_after: int,
    constant: None | int | float,
    axis: None | int,
) -> np.ndarray:
    if axis is not None and ((axis >= a.ndim) or (axis < -a.ndim)):
        return a

    return _pad_with_boundary(
        a,
        boundary,
        ((pad_before,) * (a.ndim if axis is None else 1)),
        ((pad_after,) * (a.ndim if axis is None else 1)),
        constant,
        axes=(tuple(range(a.ndim)) if axis is None else (axis,)),
    )


def _pad_with_boundary(
    a: np.ndarray,
    boundary: BoundaryCondition,
    pad_before: tuple[int, ...],
    pad_after: tuple[int, ...],
    constant: None | int | float,
    axes: tuple[int, ...],
) -> np.ndarray:
    pad_width = [(0, 0)] * a.ndim
    for axis, pb, pa in zip(axes, pad_before, pad_after):
        pad_width[axis] = (pb, pa)

    kwargs = dict()
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
            kwargs["reflect_type"] = "even"  # type: ignore
        case BoundaryCondition.symmetric:
            mode = "symmetric"
            kwargs["reflect_type"] = "even"  # type: ignore
        case BoundaryCondition.wrap:
            mode = "wrap"

    return np.pad(a, pad_width, mode, **kwargs)  # type: ignore
