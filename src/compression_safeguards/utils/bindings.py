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

from ._compat import _broadcast_to
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

    def __init__(self, param: str) -> None:
        pass

    def __new__(cls, param: str) -> "Parameter":
        if isinstance(param, Parameter):
            return param
        assert param.removeprefix("$").isidentifier(), (
            f"parameter `{param}` must be a valid identifier"
        )
        return super().__new__(cls, param)

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

        return type(self)(**self._bindings, **kwargs)  # type: ignore

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
        value: np.ndarray[tuple[int, ...], np.dtype[T]] = lossless_cast(
            self._bindings[param], dtype, f"late-bound parameter {param}"
        )

        try:
            value_view: np.ndarray[Si, np.dtype[T]] = _broadcast_to(value, shape).view()
        except ValueError:
            raise ValueError(
                f"cannot broadcast late-bound parameter {param} with shape {value.shape} to shape {shape}"
            )

        value_view.flags.writeable = False

        return value_view

    def resolve_ndarray_with_saturating_finite_float_cast(
        self, param: Parameter, shape: Si, dtype: np.dtype[F]
    ) -> np.ndarray[Si, np.dtype[F]]:
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
        value: np.ndarray[tuple[int, ...], np.dtype[F]] = saturating_finite_float_cast(
            self._bindings[param], dtype, f"late-bound parameter {param}"
        )

        try:
            value_view: np.ndarray[Si, np.dtype[F]] = _broadcast_to(value, shape).view()
        except ValueError:
            raise ValueError(
                f"cannot broadcast late-bound parameter {param} with shape {value.shape} to shape {shape}"
            )

        value_view.flags.writeable = False

        return value_view

    def expect_broadcastable_to(self, shape: tuple[int, ...]):
        """
        Check that all late-bound array parameters can be broadcast to the
        given `shape`.

        Parameters
        ----------
        shape : tuple[int, ...]
            The shape that all array parameters should be broadcastable to.

        Raises
        ------
        AssertionError
            if an array parameter cannot be broadcast to the `shape`
        """

        for param, value in self._bindings.items():
            if isinstance(value, int | float | np.number):
                # scalar
                continue

            if not isinstance(value, np.ndarray):
                # bail out
                continue

            if value.ndim == 0:
                # scalar array
                continue

            assert value.ndim == len(shape), (
                f"param {param} has dimension {value.ndim}, expected {len(shape)}"
            )

            assert all((v == 1) or (v == s) for v, s in zip(value.shape, shape)), (
                f"param {param} has shape {value.shape}, expected {shape}"
            )

    def apply_slice_index(self, index: tuple[slice, ...]) -> Self:
        """
        Apply the slice `index` to the late-bound array values and return the sliced bindings.

        The `index` is only applied to an array value if
        - the value is not scalar (no effect)
        - the value has the same number of dimensions as the `index` (bail out)

        Furthermore, the index is only applied along dimensions with a size
        greater than 1, since smaller dimensions can be broadcast.

        Parameters
        ----------
        index : tuple[slice, ...]
            The slice index that is applied to the late-bound array values.

        Returns
        -------
        bindings : Bindings
            The sliced bindings.

        Raises
        ------
        IndexError
            if an array value has the right number of dimensions but indexing
            with `index` fails
        """

        def apply_index_to_value(value: Value) -> Value:
            if isinstance(value, int | float | np.number):
                # scalar
                return value

            if not isinstance(value, np.ndarray):
                # bail out
                return value

            if value.ndim == 0:
                # scalar array
                return value

            if value.ndim != len(index):
                # bail out
                return value

            # only apply the index along axes that cannot be broadcast
            value_index = tuple(
                i if a > 1 else slice(None, None) for i, a in zip(index, value.shape)
            )
            return value[value_index]

        return type(self)(  # type: ignore
            **{
                param: apply_index_to_value(value)
                for param, value in self._bindings.items()
            }
        )

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
    # we cannot use isinstance here since np.float64 is a subclass of float
    if type(v) in (int, float):
        return v  # type: ignore

    io = BytesIO()
    np.savez_compressed(io, allow_pickle=False, **{str(p): v})

    return f"{_NPZ_DATA_URI_BASE64}{standard_b64encode(io.getvalue()).decode(encoding='ascii')}"


def _decode_value(p: Parameter, v: int | float | str) -> Value:
    if isinstance(v, int | float):
        return v

    assert v.startswith(_NPZ_DATA_URI_BASE64), (
        "value must be encoded as a .npz data URI in base64 format"
    )

    io = BytesIO(
        standard_b64decode(v[len(_NPZ_DATA_URI_BASE64) :].encode(encoding="ascii"))
    )

    with np.load(io, allow_pickle=False) as f:
        return f[str(p)]
