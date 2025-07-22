from typing import Callable

import sympy as sp
from typing_extensions import Never  # MSPV 3.11

from ...utils.bindings import Parameter


class VariableEnvironment:
    __slots__ = ("_symbol", "_variables")
    _variables: dict[str, sp.Basic]

    def __init__(self, symbol: str) -> None:
        self._symbol = symbol
        self._variables = dict()

    def __call__(self, *args, **kwargs) -> Never:
        raise TypeError(f"cannot call variable environment `{self._symbol}`")

    def __getitem__(self, name: str) -> "UnresolvedVariable | sp.Basic":
        assert isinstance(name, str) and name.isidentifier(), (
            f"variable environment `{self._symbol}` name must be a valid identifier string"
        )

        if name in self._variables:
            return self._variables[name]

        return UnresolvedVariable(name, self)


class UnresolvedVariable:
    __slots__ = ("_name", "_env")
    _name: str
    _env: VariableEnvironment

    def __init__(self, name: str, env: VariableEnvironment) -> None:
        self._name = name
        self._env = env

    def _sympy_(self) -> Never:
        raise TypeError(
            f'unresolved variable {self._env._symbol}["{self._name}"], perhaps you forgot to define it within a let expression'
        )


class LateBoundConstant(sp.Symbol):
    __slots__ = ()

    @property
    def parameter(self) -> Parameter:
        return Parameter(self.name[self.name.index("[") + 2 : -2])


class LateBoundConstantEnvironment:
    __slots__ = ("_symbol", "_create_symbol")
    _symbol: str
    _create_symbol: Callable[[str], LateBoundConstant]

    def __init__(
        self, symbol: str, create_symbol: Callable[[str], LateBoundConstant]
    ) -> None:
        self._symbol = symbol
        self._create_symbol = create_symbol

    def __call__(self, *args, **kwargs) -> Never:
        raise TypeError(f"cannot call constant environment `{self._symbol}`")

    def __getitem__(self, name: str) -> LateBoundConstant:
        assert (
            isinstance(name, str)
            and (name[1:] if name.startswith("$") else name).isidentifier()
        ), (
            f"constant environment `{self._symbol}` name must be a valid (built-in) identifier string"
        )
        return self._create_symbol(f'{self._symbol}["{name}"]')


def let(
    name: UnresolvedVariable, value: sp.Basic, /, *args: UnresolvedVariable | sp.Basic
) -> Callable[[sp.Basic], sp.Basic]:
    assert len(args) % 2 == 0, "let must be called with pairs of names and values"

    names = (name,) + args[0::2]
    values = (value,) + args[1::2]

    env: dict[str, sp.Basic] = dict()

    for n, v in zip(names, values):
        assert isinstance(n, UnresolvedVariable), (
            "let name must be a fresh (not overridden) variable"
            + (
                f' `{name._env._symbol}["var"]`'
                if isinstance(name, UnresolvedVariable)
                else ""
            )
        )
        assert isinstance(v, sp.Basic), "let value must be an expression"

        env[n._name] = v

    name._env._variables.update(env)

    def let_(within: sp.Basic, /) -> sp.Basic:
        assert isinstance(within, sp.Basic), "let within must be an expression"

        for n in env:
            del name._env._variables[n]

        return within

    return let_


FUNCTIONS = dict(let=let)
