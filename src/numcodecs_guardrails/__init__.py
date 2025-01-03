__all__ = ["GuardrailsCodec", "GuardrailKind"]

from collections.abc import Buffer, Sequence
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
from .guardrails.elementwise import ElementwiseGuardrail
from .guardrails.elementwise.abs import AbsoluteErrorBoundGuardrail
from .guardrails.elementwise.monotonic import MonotonicGuardrail
from .guardrails.elementwise.rel_or_abs import RelativeOrAbsoluteErrorBoundGuardrail


class GuardrailKind(Enum):
    # error bounds
    abs = AbsoluteErrorBoundGuardrail
    rel_or_abs = RelativeOrAbsoluteErrorBoundGuardrail
    # monotonic
    monotonic = MonotonicGuardrail


FORMAT_VERSION: str = "0.1.x"


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
    __slots__ = ("_version", "_codec", "_lossless", "_guardrails")
    _version: str
    _codec: Codec
    _lossless: Codec
    _elementwise_guardrails: tuple[ElementwiseGuardrail]

    codec_id = "guardrails"

    def __init__(
        self,
        *,
        codec: dict | Codec,
        guardrails: Sequence[dict | Guardrail],
        version: Optional[str] = None,
    ):
        if version is not None:
            assert version == FORMAT_VERSION

        self._codec = (
            codec if isinstance(codec, Codec) else numcodecs.registry.get_codec(codec)
        )
        self._lossless = numcodecs.zlib.Zlib(level=9)

        guardrails = [
            guardrail
            if isinstance(guardrail, Guardrail)
            else GuardrailKind[guardrail["kind"]].value(
                **{p: v for p, v in guardrail.items() if p != "kind"}
            )
            for guardrail in guardrails
        ]

        self._elementwise_guardrails = tuple(
            sorted(
                (
                    guardrail
                    for guardrail in guardrails
                    if isinstance(guardrail, ElementwiseGuardrail)
                ),
                key=lambda guardrail: guardrail._priority,
            )
        )
        guardrails = [
            guardrail
            for guardrail in guardrails
            if not isinstance(guardrail, ElementwiseGuardrail)
        ]

        assert len(guardrails) == 0, f"unsupported guardrails {guardrails:!r}"

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

        correction = None
        for guardrail in self._elementwise_guardrails:
            if not guardrail.check(data, decoded if correction is None else correction):
                correction = guardrail.compute_correction(data, decoded)
                assert guardrail.check(
                    data,
                    correction,
                ), "guardrail correction must pass the check"

        if correction is None:
            correction = bytes()
        else:
            correction = ElementwiseGuardrail.encode_correction(
                decoded,
                correction,
                self._lossless,
            )

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
            corrected = ElementwiseGuardrail.apply_correction(
                decoded,
                correction,
                self._lossless,
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
            version=FORMAT_VERSION,
            codec=self._codec.get_config(),
            guardrails=self._elementwise_guardrails,
        )

    def __repr__(self):
        return f"{type(self).__name__}(codec={self._codec!r}, guardrails={self._elementwise_guardrails!r})"


numcodecs.registry.register_codec(GuardrailsCodec)
