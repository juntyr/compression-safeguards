__all__ = [
    "ParameterValueError",
    "LateBoundParameterValueError",
    "ParameterTypeError",
    "LateBoundParameterTypeError",
    "QuantityOfInterestSyntaxError",
]

from contextlib import AbstractContextManager, contextmanager
from enum import Enum
from types import UnionType
from typing import Generic, TypeVar

from typing_extensions import (
    Never,  # MSPV 3.11
    override,  # MSPV 3.12
)

from ..safeguards.abc import Safeguard
from .bindings import Parameter

T = TypeVar("T", covariant=True)
""" Any type (covariant). """

Ei = TypeVar("Ei", bound=Enum)
""" Any enum type (invariant). """


class ParameterValueError(ValueError):
    safeguard: type[Safeguard]
    parameter: Parameter
    message: str

    def __init__(
        self, safeguard: type[Safeguard], parameter: Parameter, message: str
    ) -> None:
        self.safeguard = safeguard
        self.parameter = parameter
        self.message = message

        super().__init__(safeguard, parameter, message)

    @override
    def __str__(self) -> str:
        return f"{self.safeguard.kind}.{self.parameter}: {self.message}"


class LateBoundParameterValueError(ParameterValueError):
    pass


class ParameterTypeError(Generic[T], TypeError):
    safeguard: type[Safeguard]
    parameter: Parameter
    expected: type | UnionType
    found: object

    def __init__(
        self,
        safeguard: type[Safeguard],
        parameter: Parameter,
        expected: type | UnionType,
        found: object,
    ) -> None:
        self.safeguard = safeguard
        self.parameter = parameter
        self.expected = expected
        self.found = found

        super().__init__(safeguard, parameter, expected, found)

    @override
    def __str__(self) -> str:
        return f"{self.safeguard.kind}.{self.parameter}: expected {self.expected} but found {self.found} of type {type(self.found)}"


class LateBoundParameterTypeError(ParameterTypeError[T]):
    pass


# class LateBoundSelectorIndexError(IndexError):
#     pass


class QuantityOfInterestSyntaxError(SyntaxError):
    def __init__(self, message: str, lineno: int, column: int) -> None:
        super().__init__(message, ("<qoi>", lineno, column, None))

    @staticmethod
    def root(message: str) -> "QuantityOfInterestSyntaxError":
        return QuantityOfInterestSyntaxError(message, None, None)  # type: ignore


# class ParameterComplexityWarning(UserWarning):
#     pass


# class QuantityOfInterestRuntimeWarning(RuntimeWarning):
#     pass


def _check_instance(obj: object, tyx: type | UnionType) -> None | Never:
    if isinstance(obj, tyx):
        return None
    raise _TypeCheckError(tyx, obj)


class _TypeCheckError(TypeError):
    __slots__: tuple[str, ...] = ("expected", "found")
    expected: type | UnionType
    found: object

    def __init__(self, expected: type | UnionType, found: object) -> None:
        self.expected = expected
        self.found = found
        super().__init__(expected, found)

    @override
    def __str__(self) -> str:
        return f"expected {self.expected} but found {self.found} of type {type(self.found)}"


def _validate_safeguard(safeguard: Safeguard) -> "_SafeguardValidationContext":
    return _SafeguardValidationContext(type(safeguard))


class _SafeguardValidationContext(
    AbstractContextManager["_SafeguardValidationContext", None]
):
    __slots__: tuple[str, ...] = ("_safeguard",)
    _safeguard: type[Safeguard]

    def __init__(self, safeguard: type[Safeguard]):
        self._safeguard = safeguard

    @override
    def __exit__(self, exc_type, exc_value, traceback) -> None:
        return None

    @contextmanager
    def parameter(self, parameter: str | Parameter):
        try:
            yield
        except _TypeCheckError as err:
            raise ParameterTypeError(
                self._safeguard, Parameter(parameter), err.expected, err.found
            )
        except ValueError as err:
            raise ParameterValueError(self._safeguard, Parameter(parameter), str(err))

    @contextmanager
    def enum_parameter(self, parameter: str | Parameter, enum: type[Ei]):
        try:
            yield
        except _TypeCheckError as err:
            raise ParameterTypeError(
                self._safeguard, Parameter(parameter), err.expected, err.found
            )
        except KeyError as err:
            raise ParameterValueError(
                self._safeguard,
                Parameter(parameter),
                f"unknown {enum.__name__} {err.args[0]!r}, use one of "
                + f"{', '.join(repr(m) for m in enum.__members__)}",
            )
        except ValueError as err:
            raise ParameterValueError(self._safeguard, Parameter(parameter), str(err))

    @contextmanager
    def late_bound_parameter(self, parameter: str | Parameter):
        try:
            yield
        except _TypeCheckError as err:
            raise LateBoundParameterTypeError(
                self._safeguard, Parameter(parameter), err.expected, err.found
            )
        except ValueError as err:
            raise LateBoundParameterValueError(
                self._safeguard, Parameter(parameter), str(err)
            )
