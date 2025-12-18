"""
Exception and warning types and helpers to raise exceptions with context.

Exceptions raised in this package carry [`ErrorContext`][.ErrorContext], which
can be accessed through the [`ErrorContextMixin`][.ErrorContextMixin].
"""

__all__ = [
    "ContextLayer",
    "ErrorContext",
    "ctx",
    "ErrorContextMixin",
    "SafeguardTypeContextLayer",
    "ParameterContextLayer",
    "LateBoundParameterContextLayer",
    "IndexContextLayer",
    "TypeCheckError",
    "TypeSetError",
    "LateBoundParameterResolutionError",
    "SafeguardsSafetyBug",
    "QuantityOfInterestRuntimeWarning",
    "Ei",
    "lookup_enum_or_raise",
]

from abc import ABC, abstractmethod
from collections.abc import Set
from contextlib import AbstractContextManager, contextmanager
from enum import Enum
from types import UnionType
from typing import TYPE_CHECKING, Never, Self, TypeVar, final

import numpy as np
from typing_extensions import override  # MSPV 3.12

if TYPE_CHECKING:
    from ..safeguards.abc import Safeguard
    from .bindings import Parameter

Ei = TypeVar("Ei", bound=Enum)
""" Any enum type (invariant). """


class ContextLayer(ABC):
    """
    Context layer abstract base class.
    """

    __slots__: tuple[str, ...] = ()

    @override
    @abstractmethod
    def __str__(self) -> str:
        pass

    @property
    def separator(self) -> str:
        """
        Separator to print before this layer, `'.'` by default.
        """

        return "."


class ErrorContext:
    """
    Context in which an error was raised.

    The context can be added to an exception using the
    `err | ErrorContext(...)` syntax.

    Parameters
    ----------
    *context : ContextLayer
        The layers of the context.
    """

    __slots__: tuple[str, ...] = ("_layers",)
    _layers: tuple[ContextLayer, ...]

    def __init__(self, *context: ContextLayer):
        self._layers = context

    @property
    def layers(self) -> tuple[ContextLayer, ...]:
        """
        The layers of the context, from outermost to innermost.
        """

        return self._layers

    def __ror__(self, other: BaseException) -> BaseException:
        if isinstance(other, ErrorContextMixin):
            other._context = ErrorContext(*self.layers, *other.context.layers)
            return other

        ty = type(other)
        ty_with_context = _EXCEPTIONS_WITH_CONTEXT.get(ty, None)

        if ty_with_context is None:

            def __str__with_context(self) -> str:
                context_str = str(self.context)
                err_str = super(type(self), self).__str__()
                if context_str == "":
                    return err_str
                return f"{context_str}: {err_str}"

            ty_with_context = type(
                ty.__name__,
                (ty, ErrorContextMixin),
                dict(__str__=__str__with_context, __module__=ty.__module__),
            )
            _EXCEPTIONS_WITH_CONTEXT[ty] = ty_with_context

        other_with_context = ty_with_context(*other.args)
        other_with_context._context = self  # type: ignore
        other_with_context.__cause__ = other.__cause__
        other_with_context.__context__ = other.__context__
        other_with_context.__suppress_context__ = other.__suppress_context__
        other_with_context.__traceback__ = other.__traceback__
        if hasattr(other, "__notes__"):
            # only present after add_note() is called
            other_with_context.__notes__ = other.__notes__
        return other_with_context

    @override
    def __str__(self) -> str:
        match self.layers:
            case ():
                return ""
            case (c,):
                return str(c)
            case _:
                c, *cs = self.layers
                acc = [str(c)]
                for c in cs:
                    acc.append(c.separator)
                    acc.append(str(c))
                return "".join(acc)


_EXCEPTIONS_WITH_CONTEXT: dict[type[BaseException], type[BaseException]] = dict()


@final
class _ctxmeta(type):
    @override
    def __ror__(self, other: BaseException) -> BaseException:  # type: ignore
        return other | ErrorContext()


@final
class ctx(metaclass=_ctxmeta):
    """
    Singleton error context type with which additional context scopes can be entered.

    An exception can be prepared for receiving context using the `err | ctx`
    syntax.
    """

    __slots__: tuple[str, ...] = ()

    def __new__(cls) -> Self:
        raise TypeError(f"{cls} is a singleton") | ctx

    @contextmanager
    @staticmethod
    def layer(layer: ContextLayer):
        """
        Context manager that adds one `layer` of context to any exception that
        is raised within.

        Parameters
        ----------
        layer : ContextLayer
            The layer of context to add.
        """

        try:
            yield
        except Exception as err:
            err2 = err | ErrorContext(layer)
            if isinstance(err, ErrorContextMixin):
                raise
            raise err2

    @staticmethod
    def safeguard(safeguard: "Safeguard") -> AbstractContextManager[None]:
        """
        Context manager that adds one layer of context, the type of a
        `safeguard`, to any exception that is raised within.

        The added context layer will be a
        [`SafeguardTypeContextLayer`][...SafeguardTypeContextLayer].

        Parameters
        ----------
        safeguard : Safeguard
            A safeguard whose type to add as a layer of context.
        """

        return ctx.layer(SafeguardTypeContextLayer(type(safeguard)))

    @staticmethod
    def safeguardty(safeguard: type["Safeguard"]) -> AbstractContextManager[None]:
        """
        Context manager that adds one layer of context, the `safeguard` type,
        to any exception that is raised within.

        The added context layer will be a
        [`SafeguardTypeContextLayer`][...SafeguardTypeContextLayer].

        Parameters
        ----------
        safeguard : type[Safeguard]
            The type of a safeguard to add as a layer of context.
        """

        return ctx.layer(SafeguardTypeContextLayer(safeguard))

    @staticmethod
    def parameter(name: str) -> AbstractContextManager[None]:
        """
        Context manager that adds one layer of context, the `name` of a
        parameter, to any exception that is raised within.

        The added context layer will be a
        [`ParameterContextLayer`][...ParameterContextLayer].

        Parameters
        ----------
        name : str
            The name of a parameter to add as a layer of context.
        """

        return ctx.layer(ParameterContextLayer(name))

    @staticmethod
    def late_bound_parameter(name: "Parameter") -> AbstractContextManager[None]:
        """
        Context manager that adds one layer of context, the `name` of a
        late-bound parameter, to any exception that is raised within.

        The added context layer will be a
        [`LateBoundParameterContextLayer`][...LateBoundParameterContextLayer].

        Parameters
        ----------
        name : Parameter
            The name of a late-bound parameter to add as a layer of context.
        """

        return ctx.layer(LateBoundParameterContextLayer(name))

    @staticmethod
    def index(index: int) -> AbstractContextManager[None]:
        """
        Context manager that adds one layer of context, an `index`, to any
        exception that is raised within.

        The added context layer will be an
        [`IndexContextLayer`][...IndexContextLayer].

        Parameters
        ----------
        index : int
            An index to add as a layer of context.
        """

        return ctx.layer(IndexContextLayer(index))

    @staticmethod
    def __ror__(other: BaseException) -> BaseException:  # type: ignore
        return other | ctx


class ErrorContextMixin:
    """
    Mixin for exceptions that have [`ErrorContext`][compression_safeguards.utils.error.ErrorContext].

    The `context` can be accessed after checking
    `isinstance(err, ErrorContextMixin)`.
    """

    # cannot use slots since they are incompatible with multiple inheritance
    # __slots__: tuple[str, ...] = ("_context",)

    _context: ErrorContext

    @property
    def context(self) -> ErrorContext:
        """
        The context in which this exception was raised.
        """

        try:
            return self._context
        except AttributeError:
            context = ErrorContext()
            self._context = context
            return context


class SafeguardTypeContextLayer(ContextLayer):
    """
    Safeguard type context layer.

    Parameters
    ----------
    safeguard : type[Safeguard]
        The safeguard type.
    """

    __slots__: tuple[str, ...] = ("_safeguard",)
    __match_args__: tuple[str, ...] = ("safeguard",)
    _safeguard: type["Safeguard"]

    def __init__(self, safeguard: type["Safeguard"]):
        self._safeguard = safeguard

    @property
    def safeguard(self) -> type["Safeguard"]:
        """
        The safeguard type.
        """

        return self._safeguard

    @override
    def __str__(self) -> str:
        return self._safeguard.kind

    @override
    def __eq__(self, value: object, /) -> bool:
        return (
            isinstance(value, SafeguardTypeContextLayer)
            and value.safeguard == self.safeguard
        )


class ParameterContextLayer(ContextLayer):
    """
    Parameter context layer.

    Parameters
    ----------
    parameter : str
        The name of the parameter.
    """

    __slots__: tuple[str, ...] = ("_parameter",)
    __match_args__: tuple[str, ...] = ("parameter",)
    _parameter: str

    def __init__(self, parameter: str):
        self._parameter = parameter

    @property
    def parameter(self) -> str:
        """
        The name of the parameter.
        """

        return self._parameter

    @override
    def __str__(self) -> str:
        return self._parameter

    @override
    def __eq__(self, value: object, /) -> bool:
        return (
            isinstance(value, ParameterContextLayer)
            and value.parameter == self.parameter
        )


class LateBoundParameterContextLayer(ContextLayer):
    """
    Late-bound parameter context layer.

    Parameters
    ----------
    parameter : Parameter
        The late-bound parameter.
    """

    __slots__: tuple[str, ...] = ("_parameter",)
    __match_args__: tuple[str, ...] = ("parameter",)
    _parameter: "Parameter"

    def __init__(self, parameter: "Parameter"):
        self._parameter = parameter

    @property
    def parameter(self) -> "Parameter":
        """
        The late-bound parameter.
        """

        return self._parameter

    @override
    def __str__(self) -> str:
        return str(self._parameter)

    @property
    @override
    def separator(self) -> str:
        """
        `'='` separator to print before this layer.
        """

        return "="

    @override
    def __eq__(self, value: object, /) -> bool:
        return (
            isinstance(value, LateBoundParameterContextLayer)
            and value.parameter == self.parameter
        )


class IndexContextLayer(ContextLayer):
    """
    Index context layer.

    Parameters
    ----------
    index : int
        The index.
    """

    __slots__: tuple[str, ...] = ("_index",)
    __match_args__: tuple[str, ...] = ("index",)
    _index: int

    def __init__(self, index: int):
        self._index = index

    @property
    def index(self) -> int:
        """
        The index.
        """

        return self._index

    @override
    def __str__(self) -> str:
        return f"[{self.index}]"

    @property
    @override
    def separator(self) -> str:
        """
        Empty `""` separator to print before this layer.
        """

        return ""

    @override
    def __eq__(self, value: object, /) -> bool:
        return isinstance(value, IndexContextLayer) and value.index == self.index


class TypeCheckError(TypeError):
    """
    [`TypeError`][TypeError] that is raised when a value fails a type check.

    Parameters
    ----------
    expected : type | UnionType
        The expected type or type union.
    found : object
        The value that failed the type check.
    """

    __slots__: tuple[str, ...] = ()

    def __init__(
        self,
        expected: type | UnionType,
        found: object,
    ) -> None:
        super().__init__(expected, found)

    # TODO: once a TypeAssert exists for Python, return it
    @classmethod
    def check_instance_or_raise(
        cls, obj: object, expected: type | UnionType
    ) -> None | Never:
        """
        Check `isinstance(obj, expected)` or raise a type check error, with
        context.

        Parameters
        ----------
        obj : object
            The value to type check.
        expected : type | UnionType
            The expected type or type union.

        Raises
        ------
        TypeCheckError
            if `obj` is not an instance of `expected`.
        """

        if isinstance(obj, expected):
            return None
        raise cls(expected, obj) | ctx

    @property
    def expected(self) -> type | UnionType:
        """
        The expected type or type union.
        """

        (expected, _found) = self.args
        return expected

    @property
    def found(self) -> object:
        """
        The value that failed the type check.
        """

        (_expected, found) = self.args
        return found

    @override
    def __str__(self) -> str:
        return f"expected {self.expected} but found {self.found} of type {type(self.found)}"


class TypeSetError(TypeError):
    """
    [`TypeError`][TypeError] that is raised when a type is not in a type set.

    Parameters
    ----------
    expected : type | UnionType
        The expected type or type union.
    found : type
        The type that failed the type set check.
    """

    __slots__: tuple[str, ...] = ()

    def __init__(self, expected: type | UnionType, found: type):
        super().__init__(expected, found)

    # TODO: once a TypeAssert exists for Python, return it
    @classmethod
    def check_or_raise(cls, ty: type, expected: type | UnionType) -> None | Never:
        """
        Check `issubclass(ty, expected)` or raise a type set error, with
        context.

        Parameters
        ----------
        ty : type
            The type to check.
        expected : type | UnionType
            The expected type or type union.

        Raises
        ------
        TypeSetError
            if `ty` is not a subclass of `expected`.
        """

        if issubclass(ty, expected):
            return None
        raise cls(expected, ty) | ctx

    @classmethod
    def check_dtype_or_raise(
        cls, dtype: np.dtype, supported: Set[np.dtype]
    ) -> None | Never:
        """
        Check if `dtype` is in the set of `supported` data types or raise a
        type set error, with context.

        Parameters
        ----------
        dtype : np.dtype
            The data type to check.
        supported : Set[np.dtype]
            The set of supported data types.

        Raises
        ------
        TypeSetError
            if `dtype` is not in `supported`.
        """

        if dtype in supported:
            return None

        expected: type | UnionType = Never  # type: ignore
        for s in sorted(supported, key=lambda d: d.name):
            expected |= s.type

        raise cls(expected, dtype.type) | ctx

    @property
    def expected(self) -> type | UnionType:
        """
        The expected type or type union.
        """

        (expected, _found) = self.args
        return expected

    @property
    def found(self) -> type:
        """
        The type that failed the type set check.
        """

        (_expected, found) = self.args
        return found

    @override
    def __str__(self) -> str:
        return f"expected {self.expected} but found {self.found}"


class LateBoundParameterResolutionError(KeyError):
    """
    [`KeyError`][KeyError] that is raised when late-bound parameter resolution fails because one or more parameters are missing or extraneous.

    Parameters
    ----------
    expected : Set[Parameter]
        The set of expected late-bound parameters.
    provided : Set[Parameter]
        The type that failed the type set check.
    """

    __slots__: tuple[str, ...] = ()

    def __init__(self, expected: Set["Parameter"], provided: Set["Parameter"]):
        assert expected != provided
        super().__init__(expected, provided)

    @staticmethod
    def check_or_raise(
        expected: Set["Parameter"], provided: Set["Parameter"]
    ) -> None | Never:
        """
        Check if the `expected` set of late-bound parameters matches the
        `provided` set or raise a late-bound parameter resolution error, with
        context.

        Parameters
        ----------
        expected : Set[Parameter]
            The expected set of late-bound parameters.
        provided : Set[Parameter]
            The provided set of late-bound parameters.

        Raises
        ------
        LateBoundParameterResolutionError
            if `provided` is not equal to `expected`.
        """

        if expected == provided:
            return None
        raise LateBoundParameterResolutionError(expected, provided) | ctx

    @property
    def expected(self) -> Set["Parameter"]:
        """
        The expected set of late-bound parameters.
        """

        (expected, _provided) = self.args
        return expected

    @property
    def provided(self) -> Set["Parameter"]:
        """
        The provided set of late-bound parameters.
        """

        (_expected, provided) = self.args
        return provided

    @property
    def missing(self) -> Set["Parameter"]:
        """
        The missing (expected but not provided) set of late-bound parameters.
        """

        return self.expected - self.provided

    @property
    def extraneous(self) -> Set["Parameter"]:
        """
        The extraneous (provided but not expected) set of late-bound parameters.
        """

        return self.provided - self.expected

    @override
    def __str__(self) -> str:
        missing = self.missing
        missing_str = (
            "missing late-bound parameter"
            + ("s " if len(missing) > 1 else " ")
            + ", ".join(f"`{p}`" for p in sorted(missing))
        )

        extraneous = self.extraneous
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


class SafeguardsSafetyBug(RuntimeError):
    """
    [`RuntimeError`][RuntimeError] that is raised when a fatal safety bug occurs.

    A fatal safety bug occurs when the safeguards are unable to provide the
    requested safety requirement.

    By raising this error, the `compression-safeguards` avoid violating the
    safety requirements.

    When this error is raised, it is a fatal bug in the implementation of the
    `compression-safeguards`, which should be reported at
    <https://github.com/juntyr/compression-safeguards/issues>.

    Parameters
    ----------
    message : str
        The error message.
    """

    __slots__: tuple[str, ...] = ()

    def __init__(self, message: str) -> None:
        super().__init__(message)

        self.add_note(
            "This is a fatal bug in the implementation of the "
            + "`compression-safeguards`. Please report it at "
            + "<https://github.com/juntyr/compression-safeguards/issues>."
        )


class QuantityOfInterestRuntimeWarning(RuntimeWarning):
    """
    [`RuntimeWarning`][RuntimeWarning] that is raised when a recoverable quantity of interest sanity check fails.

    A quantity of interest safeguard raises this warning if it fails at
    providing the requested safety requirement using regular means but is still
    able to recover and uphold the requirement in the end.

    When this warning is raised, it is a non-fatal bug in the implementation of
    the `compression-safeguards`, which should be reported at
    <https://github.com/juntyr/compression-safeguards/issues>.

    Parameters
    ----------
    message : str
        The warning message.
    """

    __slots__: tuple[str, ...] = ()

    def __init__(self, message: str) -> None:
        super().__init__(message)

        self.add_note(
            "This is a non-fatal bug in the implementation of the "
            + "`compression-safeguards`. Please report it at "
            + "<https://github.com/juntyr/compression-safeguards/issues>."
        )


def lookup_enum_or_raise(
    enum: type[Ei], name: str, error: type[Exception] = ValueError
) -> Ei | Never:
    """
    Look up and return the `enum` member with the given `name` or raise an `error`, with context.

    Parameters
    ----------
    enum : type[Ei]
        The enum in which the `name` is looked up.
    name : str
        The enum member name to look up.
    error : type[Exception]
        The type of error that is raised if the `name` is not a member of the
        `enum`.

    Returns
    -------
    member : Ei
        The `enum` member with the given `name`.

    Raises
    ------
    error
        if the `name` is not a member of the `enum`.
    """

    if name in enum.__members__:
        return enum.__members__[name]

    raise (
        error(
            f"unknown {enum.__name__} {name!r}, use one of "
            + f"{', '.join(repr(m) for m in enum.__members__)}"
        )
        | ctx
    )
