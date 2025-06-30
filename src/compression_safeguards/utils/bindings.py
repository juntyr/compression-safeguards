"""
Types and helpers for late-bound safeguard parameters.
"""

__all__ = ["Parameter", "Value", "Bindings"]

from base64 import standard_b64decode, standard_b64encode
from collections.abc import Set
from io import BytesIO
from types import MappingProxyType
from typing import TypeAlias

import numpy as np
from typing_extensions import Self  # MSPV 3.11

from .cast import lossless_cast, saturating_finite_float_cast
from .typing import F, Si, T


class Parameter(str):
    """
    Parameter name / identifier type.

    Parameters
    ----------
    param : str
        Name of the parameter, which must be a valid identifier.
    """

    __slots__ = ()

    def __init__(self, param: str):
        pass

    def __new__(cls, param: str):
        if isinstance(param, Parameter):
            return param
        assert (param[1:] if param.startswith("$") else param).isidentifier(), (
            f"parameter `{param}` must be a valid identifier"
        )
        return super(Parameter, cls).__new__(cls, param)

    @property
    def is_builtin(self) -> bool:
        """
        Is the parameter a built-in parameter (starts with an `$`)?
        """

        return self.startswith("$")

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self})"


Value: TypeAlias = (
    int | float | np.number | np.ndarray[tuple[int, ...], np.dtype[np.number]]
)
"""
Parameter value type that includes scalar numbers and arrays thereof.
"""


class Bindings:
    """
    Bindings from parameter names to values.

    Parameters
    ----------
    **kwargs : Value
        Mapping from parameters to values as keyword arguments.
    """

    __slots__ = ("_bindings",)
    _bindings: MappingProxyType[Parameter, Value]

    def __init__(self, **kwargs: Value) -> None:
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

    def update(self, **kwargs: Value) -> Self:
        """
        Create new bindings that contain the old and the new parameters, where
        new parameters may override old ones.

        Parameters
        ----------
        **kwargs : Value
            Mapping from new parameters to values as keyword arguments.

        Returns
        -------
        bindings : Bindings
            The updated bindings.
        """

        return Bindings(**self._bindings, **kwargs)  # type: ignore

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

    def resolve_ndarray_with_lossless_cast(
        self, param: Parameter, shape: Si, dtype: np.dtype[T]
    ) -> np.ndarray[Si, np.dtype[T]]:
        """
        Resolve the `param`eter to a numpy array with the given `shape` and
        `dtype`.

        The `param`eter must be contained in the bindings and refer to a value
        that can be broadcast to the `shape` and losslessly converted to the
        `dtype`.

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

        # cast first then broadcast to allow zero-copy broadcasts of scalars
        #  to arrays of any shape
        view = np.broadcast_to(
            lossless_cast(
                self._bindings[param], dtype, f"late-bound parameter {param}"
            ),
            shape,
        ).view()
        view.flags.writeable = False

        return view  # type: ignore

    def resolve_ndarray_with_saturating_finite_float_cast(
        self, param: Parameter, shape: Si, dtype: np.dtype[F]
    ) -> np.ndarray[Si, np.dtype[T]]:
        """
        Resolve the `param`eter to a numpy array with the given `shape` and
        floating-point `dtype`.

        The `param`eter must be contained in the bindings and refer to a finite
        value that can be broadcast to the `shape`. It will be converted to the
        floating-point `dtype`, with under- and overflows being clamped to
        finite values.

        Parameters
        ----------
        param : Parameter
            The parameter that will be resolved.
        shape : Si
            The shape of the array to resolve to.
        dtype : np.dtype[F]
            The floating-point dtype of the array to resolve to.

        Returns
        -------
        array : np.ndarray[Si, np.dtype[F]]
            A read-only view to the resolved array of the given `shape` and
            `dtype`.
        """

        assert param in self._bindings, f"LateBound is missing binding for {param}"

        # cast first then broadcast to allow zero-copy broadcasts of scalars
        #  to arrays of any shape
        view = np.broadcast_to(
            saturating_finite_float_cast(
                self._bindings[param], dtype, f"late-bound parameter {param}"
            ),
            shape,
        ).view()
        view.flags.writeable = False

        return view  # type: ignore

    def get_config(self) -> dict:
        """
        Returns the configuration of the bindings in a JSON compatible format.

        Returns
        -------
        config : dict
            Configuration of the bindings.
        """

        return {str(p): _encode_value(p, v) for p, v in self._bindings.items()}

    @classmethod
    def from_config(cls, config: dict) -> Self:
        """
        Instantiate the bindings from a configuration [`dict`][dict].

        Parameters
        ----------
        config : dict
            Configuration of the bindings.

        Returns
        -------
        bindings : Self
            Instantiated bindings.
        """

        return cls(**{p: _decode_value(p, v) for p, v in config.items()})


_NPZ_DATA_URI_BASE64: str = "data:application/x-npz;base64,"


def _encode_value(p: Parameter, v: Value) -> int | float | str:
    if isinstance(v, (int, float)):
        return v

    io = BytesIO()
    np.savez_compressed(io, allow_pickle=False, **{str(p): v})

    return f"{_NPZ_DATA_URI_BASE64}{standard_b64encode(io.getvalue()).decode(encoding='ascii')}"


def _decode_value(p: Parameter, v: int | float | str) -> Value:
    if isinstance(v, (int, float)):
        return v

    assert v.startswith(_NPZ_DATA_URI_BASE64), (
        "value must be encoded as a .npz data URI in base64 format"
    )

    io = BytesIO(
        standard_b64decode(v[len(_NPZ_DATA_URI_BASE64) :].encode(encoding="ascii"))
    )

    with np.load(io, allow_pickle=False) as f:
        return f[str(p)]
