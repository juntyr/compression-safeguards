"""
Implementations for the provided stencil quantity of interest (QoI) safeguards.
"""

__all__ = ["NeighbourhoodAxis", "StencilExpr"]

from typing import NewType

from typing_extensions import Self  # MSPV 3.11

from .. import BoundaryCondition

StencilExpr = NewType("StencilExpr", str)


class NeighbourhoodAxis:
    """
    Specification of the shape of the data neighbourhood along a single axis.

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
        domain boundary to fill the data neighbourhood, e.g. by extending
        values.
    constant_boundary : None | int | float
        Optional constant value with which the data domain is extended for a
        constant boundary.
    """

    __slots__ = ("_axis", "_before", "_after", "_boundary", "_constant_boundary")
    _axis: int
    _before: int
    _after: int
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

        assert type(before) is int, "before must be an integer"
        assert before >= 0, "before must be non-negative"
        self._before = before

        assert type(after) is int, "after must be an integer"
        assert after >= 0, "after must be non-negative"
        self._after = after

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

    def get_config(self) -> dict:
        """
        Returns the configuration of the data neighbourhood.

        Returns
        -------
        config : dict
            Configuration of the data neighbourhood.
        """

        config = dict(
            axis=self._axis,
            before=self._before,
            after=self._after,
            boundary=self._boundary.name,
            constant_boundary=self._constant_boundary,
        )

        if self._constant_boundary is None:
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
