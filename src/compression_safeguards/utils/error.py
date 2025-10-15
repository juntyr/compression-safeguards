__all__ = [
    "ErrorContext",
    "IncompatibleSafeguardsVersion",
    "UnsupportedSafeguardError",
    "ParameterValueError",
    "IncompatibleChunkStencilError",
    "LateBoundParameterValueError",
    "ParameterTypeError",
    "LateBoundParameterTypeError",
    "QuantityOfInterestSyntaxError",
    "LateBoundSelectorIndexError",
    "LateBoundParameterResolutionError",
]

from contextlib import AbstractContextManager, contextmanager
from enum import Enum
from types import UnionType
from typing import TypeVar

import numpy as np
from semver import Version
from typing_extensions import (
    Never,  # MSPV 3.11
    override,  # MSPV 3.12
)

from compression_safeguards.safeguards.abc import Safeguard

from .bindings import Parameter

Ei = TypeVar("Ei", bound=Enum)
""" Any enum type (invariant). """


class ErrorContext:
    __slots__: tuple[str, ...] = ("_path",)
    _path: tuple[str, ...]

    def __init__(self, *path: str):
        self._path = path

    def enter(self) -> "ErrorContextManager":
        return ErrorContextManager(self)

    def push(self, p: str) -> "ErrorContext":
        return ErrorContext(*self._path, p)

    def extend(self, other: "ErrorContext") -> "ErrorContext":
        return ErrorContext(*self._path, *other._path)

    @override
    def __str__(self) -> str:
        return f"{'.'.join(self._path)}"


class ErrorContextManager(AbstractContextManager["ErrorContextManager", None]):
    __slots__: tuple[str, ...] = ("_context",)
    _context: ErrorContext

    def __init__(self, context: ErrorContext):
        self._context = context

    @override
    def __exit__(self, exc_type, exc_value, traceback) -> None:
        return None

    @contextmanager
    def repr(self, value: object):
        old_context = self._context
        self._context = old_context.push(repr(value))
        yield self

    @contextmanager
    def parameter(self, name: str | Parameter):
        old_context = self._context
        self._context = old_context.push(str(name))
        yield self

    @contextmanager
    def index(self, index: int):
        old_context = self._context
        self._context = old_context.push(f"[{index}]")
        yield self

    @property
    def ctx(self) -> ErrorContext:
        return self._context


class IncompatibleSafeguardsVersion(ValueError):
    safeguards: Version
    incompatible: Version

    def __init__(self, safeguards: Version, incompatible: Version) -> None:
        assert not incompatible.is_compatible(safeguards)
        self.safeguards = safeguards
        self.incompatible = incompatible
        super().__init__(safeguards, incompatible)

    @staticmethod
    def check_or_raise(safeguards: Version, version: Version) -> None | Never:
        if version.is_compatible(safeguards):
            return None
        raise IncompatibleSafeguardsVersion(safeguards, version)

    @override
    def __str__(self) -> str:
        return (
            f"{self.incompatible} is not semantic-versioning-compatible with "
            + f"the safeguards version {self.safeguards}"
        )


class UnsupportedSafeguardError(ValueError):
    safeguards: tuple[Safeguard, ...]

    def __init__(self, safeguards: tuple[Safeguard, ...]) -> None:
        self.safeguards = safeguards
        super().__init__(safeguards)

    @override
    def __str__(self) -> str:
        return repr(list(self.safeguards))


class SafeguardsSafetyBug(RuntimeError):
    message: str

    def __init__(self, message: str) -> None:
        self.message = message
        super().__init__(message)

        message = (
            "This is a bug in the implementation of the "
            + "`compression-safeguards`. Please report it at "
            + "<https://github.com/juntyr/compression-safeguards/issues>."
        )

        # MSPV 3.11
        if getattr(self, "add_note", None) is not None:
            self.add_note(message)  # type: ignore
        else:
            self.message = f"{self.message}\n\n{message}"

    @override
    def __str__(self) -> str:
        return self.message


class IncompatibleChunkStencilError(ValueError):
    message: str
    axis: int

    def __init__(self, message: str, axis: int) -> None:
        self.message = message
        self.axis = axis
        super().__init__(message, axis)

    @override
    def __str__(self) -> str:
        return f"{self.message} on axis {self.axis}"


class ParameterValueError(ValueError):
    message: str
    context: ErrorContext

    def __init__(self, message: str, context: ErrorContext) -> None:
        self.message = message
        self.context = context
        super().__init__(message, context)

    @classmethod
    def lookup_enum_or_raise(
        cls, enum: type[Ei], name: str, context: ErrorContext
    ) -> Ei | Never:
        if name in enum.__members__:
            return enum.__members__[name]

        raise cls(
            f"unknown {enum.__name__} {name!r}, use one of "
            + f"{', '.join(repr(m) for m in enum.__members__)}",
            context,
        )

    @override
    def __str__(self) -> str:
        return f"{self.context}: {self.message}"


class LateBoundParameterValueError(ParameterValueError):
    pass


class ParameterTypeError(TypeError):
    expected: type | UnionType
    found: object
    context: ErrorContext

    def __init__(
        self,
        expected: type | UnionType,
        found: object,
        context: ErrorContext,
    ) -> None:
        self.expected = expected
        self.found = found
        self.context = context
        super().__init__(expected, found, context)

    @classmethod
    def check_instance_or_raise(
        cls, obj: object, expected: type | UnionType, context: ErrorContext
    ) -> None | Never:
        if isinstance(obj, expected):
            return None
        raise cls(expected, obj, context)

    @override
    def __str__(self) -> str:
        return f"{self.context}: expected {self.expected} but found {self.found} of type {type(self.found)}"


class LateBoundParameterTypeError(ParameterTypeError):
    pass


class LateBoundSelectorIndexError(IndexError):
    message: str
    context: ErrorContext

    def __init__(self, message: str, context: ErrorContext) -> None:
        self.message = message
        self.context = context
        super().__init__(message, context)

    @override
    def __str__(self) -> str:
        return f"{self.context}: {self.message}"


class QuantityOfInterestSyntaxError(SyntaxError):
    def __init__(self, message: str, lineno: int, column: int) -> None:
        super().__init__(message, ("<qoi>", lineno, column, None))

    @staticmethod
    def root(message: str) -> "QuantityOfInterestSyntaxError":
        return QuantityOfInterestSyntaxError(message, None, None)  # type: ignore


class LateBoundParameterResolutionError(KeyError):
    expected: frozenset[Parameter]
    provided: frozenset[Parameter]

    def __init__(self, expected: frozenset[Parameter], provided: frozenset[Parameter]):
        assert expected != provided
        self.expected = expected
        self.provided = provided
        super().__init__(expected, provided)

    @staticmethod
    def check_or_raise(
        expected: frozenset[Parameter], provided: frozenset[Parameter]
    ) -> None | Never:
        if expected == provided:
            return None
        raise LateBoundParameterResolutionError(expected, provided)

    @override
    def __str__(self) -> str:
        missing = self.expected - self.provided
        missing_str = (
            "missing late-bound parameter"
            + ("s " if len(missing) > 1 else " ")
            + ", ".join(f"`{p}`" for p in missing)
        )

        extraneous = self.provided - self.expected
        extraneous_str = (
            "extraneous late-bound parameter"
            + ("s " if len(extraneous) > 1 else " ")
            + ", ".join(f"`{p}`" for p in extraneous)
        )

        if len(missing) <= 0:
            return extraneous_str

        if len(extraneous) <= 0:
            return missing_str

        return f"{missing} and {extraneous}"


class UnsupportedDateTypeError(TypeError):
    dtype: np.dtype
    supported: frozenset[np.dtype]

    def __init__(self, dtype: np.dtype, supported: frozenset[np.dtype]):
        assert dtype not in supported
        self.dtype = dtype
        self.supported = supported
        super().__init__(dtype, supported)

    @staticmethod
    def check_or_raise(dtype: np.dtype, supported: frozenset[np.dtype]) -> None | Never:
        if dtype in supported:
            return None
        raise UnsupportedDateTypeError(dtype, supported)

    @override
    def __str__(self) -> str:
        return f"unsupported data type {self.dtype.name}, only {', '.join(d.name for d in sorted(self.supported))} are supported"


# class ParameterComplexityWarning(UserWarning):
#     pass


# class QuantityOfInterestRuntimeWarning(RuntimeWarning):
#     pass
