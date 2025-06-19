from typing import Callable

import sympy as sp

from ...utils.bindings import Parameter


def let(name: sp.Symbol, value: sp.Basic, within: sp.Basic, /) -> sp.Basic:
    assert (
        isinstance(name, sp.Symbol)
        and name.name.startswith("V[")
        and name.name.endswith("]")
    ), 'let name must be a `V["var"]` expression'
    assert isinstance(value, sp.Basic), "let value must be an expression"
    assert isinstance(within, sp.Basic), "let within must be an expression"
    return within.subs(name, value)


FUNCTIONS = dict(let=let)


class VariableSymbol(sp.Symbol):
    __slots__ = ()

    def __getitem__(self, index):
        return sp.Indexed(self, index)


class VariableEnvironment:
    __slots__ = ()

    def __new__(cls, *args, **kwargs):
        raise TypeError("cannot call variable environment `V`")

    def __class_getitem__(cls, name: str) -> VariableSymbol:
        assert isinstance(name, str) and name.isidentifier(), (
            "variable environment `V` name must be a valid identifier string"
        )
        return VariableSymbol(f'V["{name}"]')


CONSTRUCTORS = dict(V=VariableEnvironment)


class LateBoundConstant(sp.Symbol):
    __slots__ = ()

    @property
    def parameter(self) -> Parameter:
        return Parameter(self.name[3:-2])


def create_late_bound_constant_environment(
    create_symbol: Callable[[str], LateBoundConstant],
):
    class LateBoundConstantEnvironment:
        __slots__ = ()

        def __new__(cls, *args, **kwargs):
            raise TypeError("cannot call constant environment `C`")

        def __class_getitem__(cls, name: str):
            assert isinstance(name, str) and name.isidentifier(), (
                "constant environment `C` name must be a valid identifier string"
            )
            return create_symbol(f'C["{name}"]')

    return LateBoundConstantEnvironment
