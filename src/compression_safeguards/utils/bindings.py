"""
Types and helpers for late-bound safeguard parameters.
"""

__all__ = ["Parameter", "Bindings"]

from collections.abc import Set
from types import MappingProxyType
from typing import Any

import numpy as np

from .typing import Si, T


class Parameter(str):
    """
    Parameter name / identifier type.

    Parameters
    ----------
    param : str
        Name of the parameter, which must be a valid identifier.
    """

    def __init__(self, param: str):
        pass

    def __new__(cls, param: str):
        if isinstance(param, Parameter):
            return param
        assert param.isidentifier(), f"parameter `{param}` must be a valid identifier"
        return super(Parameter, cls).__new__(cls, param)

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self})"


class Bindings:
    """
    Bindings from parameter names to values.

    Parameters
    ----------
    kwargs : Any
        Mapping from parameters to values as keyword arguments.
    """

    __slots__ = ("_bindings",)
    _bindings: MappingProxyType[Parameter, Any]

    def __init__(self, **kwargs) -> None:
        self._bindings = MappingProxyType(
            {Parameter(name): value for name, value in kwargs.items()}
        )

    @staticmethod
    def empty() -> "Bindings":
        """
        Create empty bindings.

        Returns
        -------
        empty : Bindings
            Empty bindings.
        """

        return Bindings()

    def __contains__(self, param: Parameter) -> bool:
        """
        Checks if the `param`eter is contained in the bindings.

        Parameters
        ----------
        param : Parameter
            The parameter for which to check.

        Returns
        -------
        found : bool
            [`True`][True] if the bindings contain the `param`eter,
            [`False`][False] otherwise.
        """
        return param in self._bindings

    def parameters(self) -> Set[Parameter]:
        """
        Access the set of parameter names for which bindings exist.

        Returns
        -------
        params : Set[Parameter]
            The set of parameters in these bindings.
        """

        return self._bindings.keys()

    def resolve_ndarray(
        self, param: Parameter, shape: Si, dtype: np.dtype[T]
    ) -> np.ndarray[Si, np.dtype[T]]:
        """
        Resolve the `param`eter to a numpy array with the given `shape` and
        `dtype`.

        The `param`eter must be contained in the bindings and refer to a value
        that can be broadcast to the `shape` and safely cast to the `dtype`.

        Parameters
        ----------
        param : Parameter
            The parameter that will be resolved.
        shape : Si
            The shape of the array to resolve to.
        dtype : np.dtype[T]
            The dtype of the array to resolve to.

        Returns
        -------
        array : np.ndarray[Si, np.dtype[T]]
            A read-only view to the resolved array of the given `shape` and
            `dtype`.
        """

        assert param in self._bindings, f"LateBound is missing binding for {param}"

        view = (
            np.broadcast_to(self._bindings[param], shape)
            .astype(dtype, casting="safe")
            .view()
        )
        view.flags.writeable = False

        return view
