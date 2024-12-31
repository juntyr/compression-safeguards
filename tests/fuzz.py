import atheris

with atheris.instrument_imports():
    import sys

    from inspect import signature

    import numpy as np

    from numcodecs.abc import Codec
    from numcodecs_guardrail import (
        GuardrailCodec,
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


def check_one_input(data):
    data = atheris.FuzzedDataProvider(data)

    kind: GuardrailKind = list(GuardrailKind)[  # type: ignore
        data.ConsumeIntInRange(0, len(GuardrailKind) - 1)
    ]
    dtype: np.ndtype = list(SUPPORTED_DTYPES)[
        data.ConsumeIntInRange(0, len(GuardrailKind) - 1)
    ]
    size: int = data.ConsumeIntInRange(0, 10)

    parameters = {
        p: data.ConsumeFloat() for p, v in signature(kind.value).parameters.items()
    }

    raw = data.ConsumeBytes(size * dtype.itemsize)
    decoded = data.ConsumeBytes(size * dtype.itemsize)

    if len(raw) != size * dtype.itemsize:
        return

    if len(decoded) != size * dtype.itemsize:
        return

    raw = np.frombuffer(raw, dtype=dtype)
    decoded = np.frombuffer(decoded, dtype=dtype)

    try:
        guardrail = GuardrailCodec(
            FuzzCodec(raw, decoded), guardrail=kind, **parameters
        )
    except AssertionError:
        return

    try:
        encoded = guardrail.encode(raw)
        guardrail.decode(encoded, out=np.empty_like(raw))
    except Exception as err:
        print(f"\n===\n\ncodec = {guardrail!r}\n\n===\n")
        raise err


atheris.Setup(sys.argv, check_one_input)
atheris.Fuzz()
