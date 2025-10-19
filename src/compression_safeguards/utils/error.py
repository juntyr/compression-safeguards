__all__ = [
    "ErrorContext",
    "IncompatibleSafeguardsVersion",
    "UnsupportedSafeguardError",
    "IncompatibleChunkStencilError",
    "TypeCheckError",
    "LateBoundParameterResolutionError",
]

from abc import ABC, abstractmethod
from contextlib import AbstractContextManager, contextmanager
from enum import Enum
from types import UnionType
from typing import TYPE_CHECKING, TypeVar

import numpy as np
from semver import Version
from typing_extensions import (
    Never,  # MSPV 3.11
    override,  # MSPV 3.12
)

if TYPE_CHECKING:
    from ..safeguards.abc import Safeguard
    from .bindings import Parameter

Ei = TypeVar("Ei", bound=Enum)
""" Any enum type (invariant). """


class ContextFragment(ABC):
    __slots__: tuple[str, ...] = ()

    @override
    @abstractmethod
    def __str__(self) -> str:
        pass

    @property
    def separator(self) -> str:
        return "."


class IndexContextFragment(ContextFragment):
    __slots__: tuple[str, ...] = ("_index",)
    _index: int

    def __init__(self, index: int):
        self._index = index

    @property
    def index(self) -> int:
        return self._index

    @override
    def __str__(self) -> str:
        return f"[{self.index}]"

    @property
    @override
    def separator(self) -> str:
        return ""


class SafeguardTypeContextFragment(ContextFragment):
    __slots__: tuple[str, ...] = ("_safeguard",)
    _safeguard: type["Safeguard"]

    def __init__(self, safeguard: type["Safeguard"]):
        self._safeguard = safeguard

    @property
    def parameter(self) -> type["Safeguard"]:
        return self._safeguard

    @override
    def __str__(self) -> str:
        return self._safeguard.kind


class ParameterContextFragment(ContextFragment):
    __slots__: tuple[str, ...] = ("_parameter",)
    _parameter: str

    def __init__(self, parameter: str):
        self._parameter = parameter

    @property
    def parameter(self) -> str:
        return self._parameter

    @override
    def __str__(self) -> str:
        return self._parameter


class LateBoundParameterContextFragment(ContextFragment):
    __slots__: tuple[str, ...] = ("_parameter",)
    _parameter: "Parameter"

    def __init__(self, parameter: "Parameter"):
        self._parameter = parameter

    @property
    def parameter(self) -> str:
        return str(self._parameter)

    @override
    def __str__(self) -> str:
        return str(self._parameter)

    @property
    @override
    def separator(self) -> str:
        return "="


_EXCEPTIONS_WITH_CONTEXT: dict[type[BaseException], type[BaseException]] = dict()


class ErrorContext:
    __slots__: tuple[str, ...] = ("_context",)
    _context: tuple[ContextFragment, ...]

    def __init__(self, *context: ContextFragment):
        self._context = context

    def enter(self) -> "ErrorContextManager":
        return ErrorContextManager(self)

    def push(self, c: ContextFragment) -> "ErrorContext":
        return ErrorContext(*self._context, c)

    def extend(self, other: "ErrorContext") -> "ErrorContext":
        return ErrorContext(*self._context, *other._context)

    def to_str_followed_by(self, follow: str) -> str:
        if self._context == ():
            return ""
        return f"{self}{follow}"

    def __ror__(self, other: BaseException) -> BaseException:
        if isinstance(other, ErrorContextMixin):
            other._context = self.extend(other.context)
            return other

        ty = type(other)
        ty_with_context = _EXCEPTIONS_WITH_CONTEXT.get(ty, None)

        if ty_with_context is None:

            def __str__with_context(self) -> str:
                return f"{self.context.to_str_followed_by(': ')}{super(type(self), self).__str__()}"

            ty_with_context = type(
                ty.__name__,
                (ty, ErrorContextMixin),
                dict(__str__=__str__with_context, __module__=ty.__module__),
            )
            _EXCEPTIONS_WITH_CONTEXT[ty] = ty_with_context

        other_with_context = ty_with_context(*other.args)
        other_with_context._context = self  # type: ignore
        return other_with_context

    @override
    def __str__(self) -> str:
        match self._context:
            case ():
                return ""
            case (c,):
                return str(c)
            case _:
                c, *cs = self._context
                acc = [str(c)]
                for c in cs:
                    acc.append(c.separator)
                    acc.append(str(c))
                return "".join(acc)


class ErrorContextManager(AbstractContextManager["ErrorContextManager", None]):
    __slots__: tuple[str, ...] = ("_context",)
    _context: ErrorContext

    def __init__(self, context: ErrorContext):
        self._context = context

    @override
    def __exit__(self, exc_type, exc_value, traceback) -> None:
        return None

    @contextmanager
    def fragment(self, fragment: ContextFragment):
        try:
            yield
        except Exception as err:
            err2 = err | ErrorContext(fragment)
            if isinstance(err, ErrorContextMixin):
                raise
            raise err2

    def parameter(self, name: str):
        return self.fragment(ParameterContextFragment(name))

    def late_bound_parameter(self, name: "Parameter"):
        return self.fragment(LateBoundParameterContextFragment(name))

    def index(self, index: int):
        return self.fragment(IndexContextFragment(index))

    def safeguard(self, safeguard: "Safeguard"):
        return self.fragment(SafeguardTypeContextFragment(type(safeguard)))

    def safeguardty(self, safeguard: type["Safeguard"]):
        return self.fragment(SafeguardTypeContextFragment(safeguard))

    def __ror__(self, other: BaseException) -> BaseException:
        return other | ErrorContext()


class ErrorContextMixin:
    # cannot use slots since they are incompatible with multiple inheritance
    # __slots__: tuple[str, ...] = ("_context",)

    _context: ErrorContext

    @property
    def context(self) -> ErrorContext:
        try:
            return self._context
        except AttributeError:
            context = ErrorContext()
            self._context = context
            return context


class IncompatibleSafeguardsVersion(ValueError):
    __slots__: tuple[str, ...] = ()

    def __init__(self, safeguards: Version, incompatible: Version) -> None:
        assert not incompatible.is_compatible(safeguards)
        super().__init__(safeguards, incompatible)

    @staticmethod
    def check_or_raise(safeguards: Version, version: Version) -> None | Never:
        if version.is_compatible(safeguards):
            return None
        raise IncompatibleSafeguardsVersion(safeguards, version) | ErrorContext()

    @property
    def safeguards(self) -> Version:
        (safeguards, _incompatible) = self.args
        return safeguards

    @property
    def incompatible(self) -> Version:
        (_safeguards, incompatible) = self.args
        return incompatible

    @override
    def __str__(self) -> str:
        return (
            f"{self.incompatible} is not semantic-versioning-compatible with "
            + f"the safeguards version {self.safeguards}"
        )


class UnsupportedSafeguardError(NotImplementedError):
    __slots__: tuple[str, ...] = ()

    def __init__(self, safeguards: tuple["Safeguard", ...]) -> None:
        super().__init__(safeguards)

    @property
    def safeguards(self) -> tuple["Safeguard", ...]:
        (safeguards,) = self.args
        return safeguards

    @override
    def __str__(self) -> str:
        return repr(list(self.safeguards))


class SafeguardsSafetyBug(RuntimeError):
    __slots__: tuple[str, ...] = ()

    def __init__(self, message: str) -> None:
        note = (
            "This is a bug in the implementation of the "
            + "`compression-safeguards`. Please report it at "
            + "<https://github.com/juntyr/compression-safeguards/issues>."
        )

        if not hasattr(self, "add_note"):
            message = f"{message}\n\n{note}"

        super().__init__(message)

        # MSPV 3.11
        if hasattr(self, "add_note"):
            self.add_note(message)  # type: ignore


class IncompatibleChunkStencilError(ValueError):
    __slots__: tuple[str, ...] = ()

    def __init__(self, message: str, axis: int) -> None:
        super().__init__(message, axis)

    @property
    def message(self) -> str:
        (message, _axis) = self.args
        return message

    @property
    def axis(self) -> int:
        (_message, axis) = self.args
        return axis

    @override
    def __str__(self) -> str:
        return f"{self.message} on axis {self.axis}"


def lookup_enum_or_raise(
    enum: type[Ei], name: str, error: type[Exception] = ValueError
) -> Ei | Never:
    if name in enum.__members__:
        return enum.__members__[name]

    raise (
        error(
            f"unknown {enum.__name__} {name!r}, use one of "
            + f"{', '.join(repr(m) for m in enum.__members__)}"
        )
        | ErrorContext()
    )


class TypeCheckError(TypeError):
    __slots__: tuple[str, ...] = ()

    def __init__(
        self,
        expected: type | UnionType,
        found: object,
    ) -> None:
        super().__init__(expected, found)

    @classmethod
    def check_instance_or_raise(
        cls, obj: object, expected: type | UnionType
    ) -> None | Never:
        if isinstance(obj, expected):
            return None
        raise cls(expected, obj) | ErrorContext()

    @property
    def expected(self) -> type | UnionType:
        (expected, _found) = self.args
        return expected

    @property
    def found(self) -> object:
        (_expected, found) = self.args
        return found

    @override
    def __str__(self) -> str:
        return f"expected {self.expected} but found {self.found} of type {type(self.found)}"


class LateBoundParameterResolutionError(KeyError):
    __slots__: tuple[str, ...] = ()

    def __init__(
        self, expected: frozenset["Parameter"], provided: frozenset["Parameter"]
    ):
        assert expected != provided
        super().__init__(expected, provided)

    @staticmethod
    def check_or_raise(
        expected: frozenset["Parameter"], provided: frozenset["Parameter"]
    ) -> None | Never:
        if expected == provided:
            return None
        raise LateBoundParameterResolutionError(expected, provided) | ErrorContext()

    @property
    def expected(self) -> frozenset["Parameter"]:
        (expected, _provided) = self.args
        return expected

    @property
    def provided(self) -> frozenset["Parameter"]:
        (_expected, provided) = self.args
        return provided

    @override
    def __str__(self) -> str:
        missing = self.expected - self.provided
        missing_str = (
            "missing late-bound parameter"
            + ("s " if len(missing) > 1 else " ")
            + ", ".join(f"`{p}`" for p in sorted(missing))
        )

        extraneous = self.provided - self.expected
        extraneous_str = (
            "extraneous late-bound parameter"
            + ("s " if len(extraneous) > 1 else " ")
            + ", ".join(f"`{p}`" for p in sorted(extraneous))
        )

        if len(missing) <= 0:
            return extraneous_str

        if len(extraneous) <= 0:
            return missing_str

        return f"{missing} and {extraneous}"


class UnsupportedDateTypeError(TypeError):
    __slots__: tuple[str, ...] = ()

    def __init__(self, dtype: np.dtype, supported: frozenset[np.dtype]):
        assert dtype not in supported
        super().__init__(dtype, supported)

    @staticmethod
    def check_or_raise(dtype: np.dtype, supported: frozenset[np.dtype]) -> None | Never:
        if dtype in supported:
            return None
        raise UnsupportedDateTypeError(dtype, supported) | ErrorContext()

    @property
    def dtype(self) -> np.dtype:
        (dtype, _supported) = self.args
        return dtype

    @property
    def supported(self) -> frozenset[np.dtype]:
        (_dtype, supported) = self.args
        return supported

    @override
    def __str__(self) -> str:
        msg = f"unsupported data type {self.dtype.name}"

        if len(self.supported) <= 0:
            return msg

        return (
            f"{msg}, only {', '.join(d.name for d in sorted(self.supported))} "
            + "are supported"
        )


# class ParameterComplexityWarning(UserWarning):
#     pass


# class QuantityOfInterestRuntimeWarning(RuntimeWarning):
#     pass
