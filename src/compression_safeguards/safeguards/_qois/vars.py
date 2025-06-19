from typing import Callable

import sympy as sp

from ...utils.bindings import Parameter


class VariableEnvironment:
    __slots__ = "_variables"
    _variables: dict[str, sp.Basic]

    def __init__(self):
        self._variables = dict()

    def __call__(self, *args, **kwargs):
        raise TypeError("cannot call variable environment `V`")

    def __getitem__(self, name: str) -> "UnresolvedVariable | sp.Basic":
        assert isinstance(name, str) and name.isidentifier(), (
            "variable environment `V` name must be a valid identifier string"
        )

        if name in self._variables:
            return self._variables[name]

        return UnresolvedVariable(name, self)


class UnresolvedVariable:
    __slots__ = ("_name", "_env")
    _name: str
    _env: VariableEnvironment

    def __init__(self, name: str, env: VariableEnvironment):
        self._name = name
        self._env = env

    def __str__(self) -> str:
        return f'V["{self._name}"]'


class LateBoundConstant(sp.Symbol):
    __slots__ = ()

    @property
    def parameter(self) -> Parameter:
        return Parameter(self.name[3:-2])


class LateBoundConstantEnvironment:
    __slots__ = ("_create_symbol",)
    _create_symbol: Callable[[str], LateBoundConstant]

    def __init__(self, create_symbol: Callable[[str], LateBoundConstant]):
        self._create_symbol = create_symbol

    def __call__(self, *args, **kwargs):
        raise TypeError("cannot call constant environment `C`")

    def __getitem__(self, name: str) -> LateBoundConstant:
        assert isinstance(name, str) and name.isidentifier(), (
            "constant environment `C` name must be a valid identifier string"
        )
        return self._create_symbol(f'C["{name}"]')


def let(name: UnresolvedVariable, value: sp.Basic, /) -> Callable[[sp.Basic], sp.Basic]:
    assert isinstance(name, UnresolvedVariable), (
        'let name must be a fresh (not overridden) `V["var"]` expression'
    )
    assert isinstance(value, sp.Basic), "let value must be an expression"

    name._env._variables[name._name] = value

    def let_(within: sp.Basic, /) -> sp.Basic:
        assert isinstance(within, sp.Basic), "let within must be an expression"
        del name._env._variables[name._name]
        return within

    return let_


FUNCTIONS = dict(let=let)
