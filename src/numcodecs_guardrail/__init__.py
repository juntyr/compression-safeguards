__all__ = ["GuardrailCodec", "GuardrailKind"]

from enum import Enum
from io import BytesIO

import numcodecs
import numcodecs.compat
import numcodecs.registry
import numcodecs.zlib
import numpy as np
import varint

from numcodecs.abc import Codec

from .guardrail import Guardrail
from .guardrail.abs import AbsoluteErrorBoundGuardrail
from .guardrail.rel_or_abs import RelativeOrAbsoluteErrorBoundGuardrail


GuardrailKind: type = Enum(
    "GuardrailKind",
    {
        kind.kind: kind
        for kind in [
            AbsoluteErrorBoundGuardrail,
            RelativeOrAbsoluteErrorBoundGuardrail,
        ]
    },
)


class GuardrailCodec(Codec):
    __slots__ = ("_codec", "_lossless", "_guardrail")
    _codec: Codec
    _lossless: Codec
    _guardrail: Guardrail

    codec_id = "guardrail"

    def __init__(
        self,
        codec: dict | Codec,
        guardrail: str | GuardrailKind,  # type: ignore
        **kwargs,
    ):
        self._codec = (
            codec if isinstance(codec, Codec) else numcodecs.registry.get_codec(codec)
        )
        self._lossless = numcodecs.zlib.Zlib(level=9)

        guardrail = (
            guardrail
            if isinstance(guardrail, GuardrailKind)
            else GuardrailKind[guardrail]
        )

        self._guardrail = (guardrail.value)(**kwargs)

    def encode(self, buf):
        data = numcodecs.compat.ensure_ndarray(buf)

        assert data.dtype in (
            np.dtype("float32"),
            np.dtype("float64"),
        ), "can only encode f32 and f64 arrays"

        encoded = self._codec.encode(np.copy(data))
        encoded = numcodecs.compat.ensure_ndarray(encoded)

        assert encoded.dtype == np.dtype("uint8"), "codec must encode to bytes"
        assert len(encoded.shape) <= 1, "codec must encode to 1D bytes"
        encoded = numcodecs.compat.ensure_bytes(encoded)

        decoded = np.empty_like(data)
        decoded = self._codec.decode(np.copy(encoded), out=decoded)
        decoded = numcodecs.compat.ensure_ndarray(decoded)

        assert decoded.dtype == data.dtype, "codec must roundtrip dtype"
        assert decoded.shape == data.shape, "codec must roundtrip shape"

        if self._guardrail.check(data, decoded):
            correction = bytes()
        else:
            correction = self._guardrail.encode_correction(
                data, decoded, lossless=self._lossless
            )

            corrected = self._guardrail.apply_correction(
                decoded, correction, lossless=self._lossless
            )
            assert self._guardrail.check(
                data, corrected
            ), "guardrail correction must pass the check"

        correction_len = varint.encode(len(correction))

        return correction_len + encoded + correction

    def decode(self, buf, out=None):
        assert out is not None, "can only decode into known dtype and shape"
        out = numcodecs.compat.ensure_ndarray(out)

        buf = numcodecs.compat.ensure_ndarray(buf)
        assert buf.dtype == np.dtype("uint8"), "codec must decode from bytes"
        assert len(buf.shape) <= 1, "codec must decode from 1D bytes"
        buf = numcodecs.compat.ensure_bytes(buf)

        buf_io = BytesIO(buf)
        correction_len = varint.decode_stream(buf_io)

        if correction_len > 0:
            encoded = buf[buf_io.tell() : -correction_len]
            correction = buf[-correction_len:]
        else:
            encoded = buf[buf_io.tell() :]
            correction = bytes()

        decoded = self._codec.decode(encoded, out=out)

        if correction_len > 0:
            corrected = self._guardrail.apply_correction(
                decoded, correction, lossless=self._lossless
            )
        else:
            corrected = decoded

        return numcodecs.compat.ndarray_copy(corrected, out)

    def get_config(self):
        return dict(
            id=type(self).codec_id,
            codec=self._codec.get_config(),
            guardrail=self._guardrail.kind,
            **self._guardrail.get_config(),
        )

    def __repr__(self):
        config = dict(
            codec=self._codec,
            guardrail=self._guardrail.kind,
            **self._guardrail.get_config(),
        )

        repr = ", ".join(f"{p}={v!r}" for p, v in config.items())

        return f"{type(self).__name__}({repr})"


numcodecs.registry.register_codec(GuardrailCodec)
