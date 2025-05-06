import atheris

with atheris.instrument_imports():
    import numcodecs
    import numpy as np

import sympy as sympy

with atheris.instrument_imports():
    import sys
    import types
    import typing
    import warnings
    from collections.abc import Sequence
    from enum import Enum
    from inspect import signature

    import numcodecs.registry
    import numpy as np
    from numcodecs.abc import Codec
    from numcodecs_safeguards import (
        SafeguardsCodec,
        Safeguards,
    )
    from numcodecs_safeguards.safeguards.abc import Safeguard
    from numcodecs_safeguards.quantizer import _SUPPORTED_DTYPES
    from numcodecs_safeguards.safeguards.pointwise.qoi import Expr


warnings.filterwarnings("error")


class FuzzCodec(Codec):
    codec_id = "fuzz"

    def __init__(self, data, decoded):
        self.data = data
        self.decoded = decoded

    def encode(self, buf):
        return b""

    def decode(self, buf, out=None):
        assert len(buf) == 0
        assert out is not None
        out[:] = self.decoded
        return out


numcodecs.registry.register_codec(FuzzCodec)


def generate_parameter(data: atheris.FuzzedDataProvider, ty: type, depth: int):
    if ty is types.NoneType:
        return None
    if ty is float:
        return data.ConsumeFloat()
    if ty is int:
        return data.ConsumeInt(1)
    if ty is bool:
        return data.ConsumeBool()

    if typing.get_origin(ty) is Sequence:
        if len(typing.get_args(ty)) == 1:
            return [
                generate_parameter(data, typing.get_args(ty)[0], depth)
                for _ in range(data.ConsumeIntInRange(0, 3 - depth))
            ]

    if typing.get_origin(ty) in (typing.Union, types.UnionType):
        tys = typing.get_args(ty)

        if len(tys) == 2 and tys[0] is str and issubclass(tys[1], Enum):
            return list(tys[1])[data.ConsumeIntInRange(0, len(tys[1]) - 1)]

        if len(tys) == 2 and tys[0] is dict and issubclass(tys[1], Safeguard):
            return generate_safeguard_config(data, depth + 1)

        ty = tys[data.ConsumeIntInRange(0, len(tys) - 1)]

        return generate_parameter(data, ty, depth)

    if ty is Expr:
        ATOMS = ["x", int, float, "e", "pi"]
        OPS = ["neg", "+", "-", "*", "/", "**", "log", "sqrt", "ln", "exp"]

        atoms = []
        for _ in range(data.ConsumeIntInRange(2, 4)):
            atom = ATOMS[data.ConsumeIntInRange(0, len(ATOMS) - 1)]
            if atom is int:
                atom = str(data.ConsumeInt(2))
            elif atom is float:
                atom = str(data.ConsumeRegularFloat())
            atoms.append(atom)

        done = False
        while not done:
            done = len(atoms) == 1
            atom1 = atoms.pop(data.ConsumeIntInRange(0, len(atoms) - 1))
            atom2 = (
                atoms.pop(data.ConsumeIntInRange(0, len(atoms) - 1))
                if len(atoms) > 0
                else "1"
            )
            op = OPS[data.ConsumeIntInRange(0, len(OPS) - 1)]
            if op == "neg":
                atoms.append(f"(-{atom1})")
            elif op in ("sqrt", "ln", "exp"):
                atoms.append(f"{op}({atom1})")
            elif op == "log":
                atoms.append(f"log({atom1},{atom2})")
            else:
                atoms.append(f"({atom1}{op}{atom2})")
        [atom] = atoms
        return atom

    assert False, f"unknown parameter type {ty!r}"


def generate_safeguard_config(data: atheris.FuzzedDataProvider, depth: int):
    kind = list(Safeguards)[data.ConsumeIntInRange(0, len(Safeguards) - 1)]

    return {
        "kind": kind.name,
        **{
            p: generate_parameter(data, v.annotation, depth)
            for p, v in signature(kind.value).parameters.items()
        },
    }


def check_one_input(data):
    data = atheris.FuzzedDataProvider(data)

    safeguards = [
        generate_safeguard_config(data, 0) for _ in range(data.ConsumeIntInRange(0, 8))
    ]

    dtype: np.ndtype = list(_SUPPORTED_DTYPES)[
        data.ConsumeIntInRange(0, len(_SUPPORTED_DTYPES) - 1)
    ]
    sizea: int = data.ConsumeIntInRange(0, 20)
    sizeb: int = data.ConsumeIntInRange(0, 20 // max(1, sizea))
    size = sizea * sizeb

    # input data and the decoded data
    raw = data.ConsumeBytes(size * dtype.itemsize)
    decoded = data.ConsumeBytes(size * dtype.itemsize)

    if len(raw) != size * dtype.itemsize:
        return

    if len(decoded) != size * dtype.itemsize:
        return

    raw = np.frombuffer(raw, dtype=dtype)
    decoded = np.frombuffer(decoded, dtype=dtype)

    if sizeb != 0:
        raw = raw.reshape((sizea, sizeb))
        decoded = decoded.reshape((sizea, sizeb))

    try:
        safeguard = SafeguardsCodec(
            codec=FuzzCodec(raw, decoded),
            safeguards=safeguards,
        )
    except (AssertionError, Warning):
        return

    grepr = repr(safeguard)
    gconfig = safeguard.get_config()

    safeguard = numcodecs.registry.get_codec(gconfig)
    assert safeguard.get_config() == gconfig, (
        f"{safeguard.get_config()!r} vs {gconfig!r}"
    )
    assert repr(safeguard) == grepr, f"{safeguard!r} vs {grepr}"

    try:
        encoded = safeguard.encode(raw)
        safeguard.decode(encoded, out=np.empty_like(raw))
    except Exception as err:
        print(f"\n===\n\ncodec = {grepr}\n\n===\n")
        raise err

    # test using the safeguards without a codec
    safeguard = SafeguardsCodec(
        codec=None,
        safeguards=safeguards,
    )

    grepr = repr(safeguard)
    gconfig = safeguard.get_config()

    safeguard = numcodecs.registry.get_codec(gconfig)
    assert safeguard.get_config() == gconfig

    try:
        encoded = safeguard.encode(raw)
        safeguard.decode(encoded, out=np.empty_like(raw))
    except Exception as err:
        print(f"\n===\n\ncodec = {grepr}\n\ndata = {raw!r}\n\n===\n")
        raise err


atheris.Setup(sys.argv, check_one_input)
atheris.Fuzz()
