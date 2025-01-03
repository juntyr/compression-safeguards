import atheris

with atheris.instrument_imports():
    import sys

    from functools import partial
    from inspect import signature

    import numcodecs.registry
    import numpy as np

    from numcodecs.abc import Codec
    from numcodecs_guardrails import (
        GuardrailsCodec,
        GuardrailKind,
        SUPPORTED_DTYPES,
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


def check_one_input(data):
    data = atheris.FuzzedDataProvider(data)

    # top-level metadata: which guardrails and what type of data
    kinds: list[GuardrailKind] = [kind for kind in GuardrailKind if data.ConsumeBool()]
    dtype: np.ndtype = list(SUPPORTED_DTYPES)[
        data.ConsumeIntInRange(0, len(GuardrailKind) - 1)
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

    # guardrail parameters
    guardrails = [
        {
            "kind": kind.name,
            **{
                p: (
                    {
                        float: data.ConsumeFloat,
                        int: partial(data.ConsumeInt, 1),
                        bool: data.ConsumeBool,
                    }[v.annotation]
                )()
                for p, v in signature(kind.value).parameters.items()
            },
        }
        for kind in kinds
    ]

    try:
        guardrail = GuardrailsCodec(
            codec=FuzzCodec(raw, decoded),
            guardrails=guardrails,
        )
    except AssertionError:
        return

    grepr = repr(guardrail)
    gconfig = guardrail.get_config()

    guardrail = numcodecs.registry.get_codec(gconfig)
    assert guardrail.get_config() == gconfig

    try:
        encoded = guardrail.encode(raw)
        guardrail.decode(encoded, out=np.empty_like(raw))
    except Exception as err:
        print(f"\n===\n\ncodec = {grepr}\n\n===\n")
        raise err


atheris.Setup(sys.argv, check_one_input)
atheris.Fuzz()
