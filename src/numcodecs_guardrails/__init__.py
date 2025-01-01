__all__ = ["GuardrailsCodec", "GuardrailKind"]

from collections.abc import Buffer
from enum import Enum
from io import BytesIO
from typing import Optional

import numcodecs
import numcodecs.compat
import numcodecs.registry
import numcodecs.zlib
import numpy as np
import varint

from numcodecs.abc import Codec

from .guardrails import Guardrail
from .guardrails.abs import AbsoluteErrorBoundGuardrail
from .guardrails.rel_or_abs import RelativeOrAbsoluteErrorBoundGuardrail


class GuardrailKind(Enum):
    abs = AbsoluteErrorBoundGuardrail
    abs_or_rel = RelativeOrAbsoluteErrorBoundGuardrail


SUPPORTED_DTYPES: set[np.dtype] = {
    np.dtype("int8"),
    np.dtype("int16"),
    np.dtype("int32"),
    np.dtype("int64"),
    np.dtype("uint8"),
    np.dtype("uint16"),
    np.dtype("uint32"),
    np.dtype("uint64"),
    np.dtype("float32"),
    np.dtype("float64"),
}


class GuardrailsCodec(Codec):
    __slots__ = ("_codec", "_lossless", "_guardrail")
    _codec: Codec
    _lossless: Codec
    _guardrail: Guardrail

    codec_id = "guardrails"

    def __init__(
        self,
        codec: dict | Codec,
        guardrail: str | GuardrailKind,
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

    def encode(self, buf: Buffer) -> Buffer:
        """Encode the data in `buf`.

        Parameters
        ----------
        buf : Buffer
            Data to be encoded. May be any object supporting the new-style
            buffer protocol.

        Returns
        -------
        enc : Buffer
            Encoded data. May be any object supporting the new-style buffer
            protocol.
        """

        data = numcodecs.compat.ensure_ndarray(buf)

        assert (
            data.dtype in SUPPORTED_DTYPES
        ), f"can only encode arrays of dtype {', '.join(d.str for d in SUPPORTED_DTYPES)}"

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

    def decode(self, buf: Buffer, out: Optional[Buffer] = None) -> Buffer:
        """Decode the data in `buf`.

        Parameters
        ----------
        buf : Buffer
            Encoded data. May be any object supporting the new-style buffer
            protocol.
        out : Buffer, optional
            Writeable buffer to store decoded data. N.B. if provided, this buffer must
            be exactly the right size to store the decoded data.

        Returns
        -------
        dec : Buffer
            Decoded data. May be any object supporting the new-style
            buffer protocol.
        """

        assert out is not None, "can only decode into known dtype and shape"
        out = numcodecs.compat.ensure_ndarray(out)

        buf = numcodecs.compat.ensure_ndarray(buf)
        assert buf.dtype == np.dtype("uint8"), "codec must decode from bytes"
        assert len(buf.shape) <= 1, "codec must decode from 1D bytes"
        buf = numcodecs.compat.ensure_bytes(buf)

        buf_io = BytesIO(buf)
        correction_len = varint.decode_stream(buf_io)
        assert correction_len >= 0

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

    def get_config(self) -> dict:
        """
        Returns the configuration of the codec with guardrails.

        [`numcodecs.registry.get_codec(config)`][numcodecs.registry.get_codec]
        can be used to reconstruct this stack from the returned config.

        Returns
        -------
        config : dict
            Configuration of the codec with guardrails.
        """

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


numcodecs.registry.register_codec(GuardrailsCodec)
