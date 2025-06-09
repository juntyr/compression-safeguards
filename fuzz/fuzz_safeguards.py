import atheris

with atheris.instrument_imports():
    import numcodecs
    import numcodecs.compat
    import numpy as np

import sympy as sympy
from timeoutcontext import timeout

with atheris.instrument_imports():
    import sys
    import types
    import typing
    import warnings
    from collections.abc import Collection, Sequence
    from enum import Enum
    from inspect import signature

    import numcodecs.registry
    import numpy as np
    from numcodecs.abc import Codec
    from numcodecs_safeguards import SafeguardsCodec

    from compression_safeguards import SafeguardKind, Safeguards
    from compression_safeguards.safeguards._qois.amath import (
        FUNCTIONS as AMATH_FUNCTIONS,
    )
    from compression_safeguards.safeguards._qois.math import CONSTANTS as MATH_CONSTANTS
    from compression_safeguards.safeguards._qois.math import FUNCTIONS as MATH_FUNCTIONS
    from compression_safeguards.safeguards.abc import Safeguard
    from compression_safeguards.safeguards.pointwise.qoi import PointwiseExpr
    from compression_safeguards.safeguards.stencil import NeighbourhoodBoundaryAxis
    from compression_safeguards.safeguards.stencil.qoi import StencilExpr
    from compression_safeguards.utils.bindings import Parameter


warnings.filterwarnings("error")


class FuzzCodec(Codec):
    __slots__ = ("data", "decoded")

    codec_id = "fuzz"  # type: ignore

    def __init__(self, data, decoded):
        self.data = data
        self.decoded = decoded

    def encode(self, buf):
        return b""

    def decode(self, buf, out=None):
        assert len(buf) == 0
        return numcodecs.compat.ndarray_copy(np.copy(self.decoded), out)

    def get_config(self):
        return dict(id=type(self).codec_id, data=self.data, decoded=self.decoded)

    def __repr__(self):
        config = {k: v for k, v in self.get_config().items() if k != "id"}
        return f"{type(self).__name__}({', '.join(f'{k}={v!r}' for k, v in config.items())})"


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

    if typing.get_origin(ty) in (Collection, Sequence):
        if len(typing.get_args(ty)) == 1:
            return [
                generate_parameter(data, typing.get_args(ty)[0], depth)
                for _ in range(data.ConsumeIntInRange(0, 3 - depth))
            ]

    if typing.get_origin(ty) in (typing.Union, types.UnionType):
        tys = typing.get_args(ty)

        if len(tys) == 2 and tys[0] is str and issubclass(tys[1], Enum):
            return list(tys[1])[data.ConsumeIntInRange(0, len(tys[1]) - 1)]

        if len(tys) == 2 and tys[0] is dict and tys[1] is NeighbourhoodBoundaryAxis:
            return {
                p: generate_parameter(data, v.annotation, depth)
                for p, v in signature(NeighbourhoodBoundaryAxis).parameters.items()
            }

        if len(tys) == 2 and tys[0] is str and tys[1] is Parameter:
            # since numcodecs_safeguards doesn't yet support late-bound
            #  parameters, we can just generate a constant name here
            return "param"

        if (
            len(tys) > 1
            and tys[0] is dict
            and all(issubclass(t, Safeguard) for t in tys[1:])
        ):
            return generate_safeguard_config(data, depth + 1)

        ty = tys[data.ConsumeIntInRange(0, len(tys) - 1)]

        return generate_parameter(data, ty, depth)

    if ty in (PointwiseExpr, StencilExpr):
        ATOMS = ["x", int, float] + list(MATH_CONSTANTS)
        OPS = [
            "neg",
            "+",
            "-",
            "*",
            "/",
            "**",
        ] + list(MATH_FUNCTIONS)

        if ty is StencilExpr:
            ATOMS += ["X", "I"]
            OPS += ["index", "findiff"] + list(AMATH_FUNCTIONS)

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
            elif op in ("log", "matmul"):
                atoms.append(f"log({atom1},{atom2})")
            elif op in tuple(MATH_FUNCTIONS) + tuple(AMATH_FUNCTIONS):
                atoms.append(f"{op}({atom1})")
            elif op == "index":
                atoms.append(
                    f"{atom1}[{data.ConsumeIntInRange(0, 20)}, {data.ConsumeIntInRange(0, 20)}]"
                )
            elif op == "findiff":
                atoms.append(
                    f"findiff({atom1}, order={data.ConsumeIntInRange(0, 3)}, accuracy={data.ConsumeIntInRange(1, 4)}, type={data.ConsumeIntInRange(-1, 1)}, dx={data.ConsumeRegularFloat()}, axis={data.ConsumeIntInRange(0, 1)})"
                )
            else:
                atoms.append(f"({atom1}{op}{atom2})")
        [atom] = atoms
        return atom

    assert False, f"unknown parameter type {ty!r}"


def generate_safeguard_config(data: atheris.FuzzedDataProvider, depth: int):
    kind = list(SafeguardKind)[data.ConsumeIntInRange(0, len(SafeguardKind) - 1)]

    return {
        "kind": kind.name,
        **{
            p: generate_parameter(data, v.annotation, depth)
            for p, v in signature(kind.value).parameters.items()
        },
    }


def check_one_input(data) -> None:
    data = atheris.FuzzedDataProvider(data)

    safeguards = [
        generate_safeguard_config(data, 0) for _ in range(data.ConsumeIntInRange(0, 8))
    ]

    dtype: np.dtype = list(Safeguards.supported_dtypes())[
        data.ConsumeIntInRange(0, len(Safeguards.supported_dtypes()) - 1)
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
        with timeout(1):
            safeguard = SafeguardsCodec(
                codec=FuzzCodec(raw, decoded),
                safeguards=safeguards,
            )
    except (AssertionError, Warning, TimeoutError):
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
        if (
            (
                isinstance(err, IndexError)
                and ("axis index" in str(err))
                and ("out of bounds for array of shape" in str(err))
            )
            or (
                isinstance(err, IndexError)
                and ("duplicate axis index" in str(err))
                and ("normalised to" in str(err))
                and ("for array of shape" in str(err))
            )
            or (
                isinstance(err, ValueError)
                and ("constant boundary has invalid value" in str(err))
            )
        ):
            return
        print(f"\n===\n\ncodec = {grepr}\n\n===\n")
        raise err

    # test using the safeguards with the zero codec
    safeguard = SafeguardsCodec(
        codec=dict(id="zero"),
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
