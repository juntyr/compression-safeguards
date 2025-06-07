"""
Types and helpers for late-bound safeguard parameters.
"""

from collections.abc import KeysView
from types import MappingProxyType
from typing import Any

import numpy as np

from .typing import Si, T


class Parameter(str):
    def __new__(cls, param: str):
        assert param.isidentifier(), f"parameter `{param}` must be a valid identifier"
        return super(Parameter, cls).__new__(cls, param)

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self})"


class LateBound:
    __slots__ = ("_bindings",)
    _bindings: MappingProxyType[Parameter, Any]

    def __init__(self, **kwargs) -> None:
        self._bindings = MappingProxyType(
            {Parameter(name): value for name, value in kwargs.items()}
        )

    def __contains__(self, param: Parameter) -> bool:
        return param in self._bindings

    def keys(self) -> KeysView[Parameter]:
        return self._bindings.keys()

    def resolve_ndarray(
        self, param: Parameter, shape: Si, dtype: np.dtype[T]
    ) -> np.ndarray[Si, np.dtype[T]]:
        assert param in self._bindings, f"LateBound is missing binding for {param}"

        return np.broadcast_to(self._bindings[param], shape).astype(
            dtype, casting="safe"
        )
