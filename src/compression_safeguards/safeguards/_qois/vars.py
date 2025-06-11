import sympy as sp


def let(name: sp.Symbol, value: sp.Basic, within: sp.Basic) -> sp.Basic:
    assert (
        isinstance(name, sp.Symbol)
        and name.name.startswith("V[")
        and name.name.endswith("]")
    ), 'let name must be a V["var"] expression'
    assert isinstance(value, sp.Basic), "let value must be an expression"
    assert isinstance(within, sp.Basic), "let within must be an expression"
    return within.subs(name, value)


FUNCTIONS = dict(let=let)


class VariableSymbol(sp.Symbol):
    def __getitem__(self, index):
        return sp.Indexed(self, index)


class VariableEnvironment:
    __slots__ = ()

    def __new__(cls, *args, **kwargs):
        raise TypeError("cannot call variable environment `V`")

    def __class_getitem__(cls, name: str):
        assert isinstance(name, str) and name.isidentifier(), (
            "variable environment `V` name must be a valid identifier string"
        )
        return VariableSymbol(f'V["{name}"]')


CONSTRUCTORS = dict(V=VariableEnvironment)
