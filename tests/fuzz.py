import atheris

with atheris.instrument_imports():
    import sys
    import types
    import typing
    import warnings
    from enum import Enum
    from inspect import signature, Parameter

    import numcodecs.registry
    import numpy as np
    from numcodecs.abc import Codec
    from numcodecs_safeguards import (
        SafeguardsCodec,
        Safeguards,
        _SUPPORTED_DTYPES,
    )


class FuzzCodec(Codec):
    codec_id = "fuzz"

    def __init__(self, data, decoded):
        self.data = data
        self.decoded = decoded

    def encode(self, buf):
        return b""

    def decode(self, buf, out=None):
        assert buf == b""
        assert out is not None
        out[:] = self.decoded
        return out


numcodecs.registry.register_codec(FuzzCodec)


def generate_parameter(data: atheris.FuzzedDataProvider, p: Parameter):
    if p.annotation is float:
        return data.ConsumeFloat()
    if p.annotation is int:
        return data.ConsumeInt(1)
    if p.annotation is bool:
        return data.ConsumeBool()

    if typing.get_origin(p.annotation) in (typing.Union, types.UnionType):
        tys = typing.get_args(p.annotation)

        if len(tys) == 2 and tys[0] is str and issubclass(tys[1], Enum):
            return list(tys[1])[data.ConsumeIntInRange(0, len(tys[1]) - 1)]

        ty = tys[data.ConsumeIntInRange(0, len(tys) - 1)]

        if ty is types.NoneType:
            return None
        if ty is float:
            return data.ConsumeFloat()
        if ty is int:
            return data.ConsumeInt(1)
        if ty is bool:
            return data.ConsumeBool()

    assert False, f"unknown parameter type {p.annotation!r}"


def check_one_input(data):
    data = atheris.FuzzedDataProvider(data)

    # top-level metadata: which safeguards and what type of data
    kinds: list[Safeguards] = [kind for kind in Safeguards if data.ConsumeBool()]
    dtype: np.ndtype = list(_SUPPORTED_DTYPES)[
        data.ConsumeIntInRange(0, len(Safeguards) - 1)
    ]
    size: int = data.ConsumeIntInRange(0, 10)

    # input data and the decoded data
    raw = data.ConsumeBytes(size * dtype.itemsize)
    decoded = data.ConsumeBytes(size * dtype.itemsize)

    if len(raw) != size * dtype.itemsize:
        return

    if len(decoded) != size * dtype.itemsize:
        return

    raw = np.frombuffer(raw, dtype=dtype)
    decoded = np.frombuffer(decoded, dtype=dtype)

    # safeguard parameters
    safeguards = [
        {
            "kind": kind.name,
            **{
                p: generate_parameter(data, v)
                for p, v in signature(kind.value).parameters.items()
            },
        }
        for kind in kinds
    ]

    warnings.filterwarnings("error")

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
    assert safeguard.get_config() == gconfig

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
        print(f"\n===\n\ncodec = {grepr}\n\n===\n")
        raise err


atheris.Setup(sys.argv, check_one_input)
atheris.Fuzz()
