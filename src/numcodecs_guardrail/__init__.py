from enum import StrEnum
from typing import Optional

import numcodecs
import numcodecs.abc
import numcodecs.compat
import numcodecs.registry
import numcodecs.zlib
import numpy as np


class GuardrailErrorBound(StrEnum):
    ABSOLUTE = "abs"


class Guardrail(numcodecs.abc.Codec):
    __slots__ = ("_codec", "_lossless", "_error_bound", "_eb_abs")
    _codec: numcodecs.abc.Codec
    _lossless: numcodecs.abc.Codec
    _error_bound: GuardrailErrorBound
    _eb_abs: Optional[float]

    codec_id = "guardrail"

    def __init__(
        self,
        codec: dict,
        error_bound: str | GuardrailErrorBound,
        eb_abs: Optional[float],
    ):
        self._codec = numcodecs.registry.get_codec(codec)
        self._lossless = numcodecs.zlib.Zlib(level=9)
        self._error_bound = (
            error_bound
            if isinstance(error_bound, GuardrailErrorBound)
            else GuardrailErrorBound[error_bound]
        )
        self._eb_abs = eb_abs

        if self._error_bound is GuardrailErrorBound.ABSOLUTE:
            assert self._eb_abs is not None, "eb_abs must not be None"
            assert self._eb_abs > 0.0, "eb_abs must be positive"

    def encode(self, buf):
        data = numcodecs.compat.ensure_ndarray(buf)
        data_dtype = data.dtype
        data_shape = data.shape

        assert data.dtype in (
            np.dtype("float32"),
            np.dtype("float64"),
        ), "can only encode f32 and f64 arrays"

        encoded = self._codec.encode(np.copy(data))
        decoded = self._codec.decode(np.copy(encoded))

        encoded = numcodecs.compat.ensure_ndarray(encoded)
        encoded_dtype = numcodecs.compat.ensure_ndarray(encoded).dtype
        encoded_shape = encoded.shape

        encoded = numcodecs.compat.ensure_bytes(encoded)
        decoded = numcodecs.compat.ensure_ndarray(decoded)

        error = decoded - data
        correction = np.round(error / (self._eb_abs * 2.0)).astype(np.int64)

        correction = self._lossless.encode(correction)
        correction = numcodecs.compat.ensure_bytes(correction)

        # TODO: encode a header that details the data shape
        return data_dtype, data_shape, encoded, encoded_dtype, encoded_shape, correction

    def decode(self, buf, out=None):
        # TODO: decode a header that details the data shape
        data_dtype, data_shape, encoded, encoded_dtype, encoded_shape, correction = buf

        encoded = np.frombuffer(encoded, encoded_dtype).reshape(encoded_shape)
        decoded = self._codec.decode(encoded)

        correction = self._lossless.decode(correction)
        correction = numcodecs.compat.ensure_bytes(correction)
        correction = np.frombuffer(correction, np.int64).reshape(data_shape)
        correction = correction.astype(data_dtype) * (self._eb_abs * 2.0)

        corrected = decoded + correction

        return numcodecs.compat.ndarray_copy(corrected, out)

    def get_config(self):
        config = dict(
            id=type(self).codec_id,
            codec=self._codec.get_config(),
            error_bound=str(self._error_bound),
        )

        if self._eb_abs is not None:
            config["eb_abs"] = self._eb_abs

        return config

    def __repr__(self):
        params = dict(codec=self._codec, error_bound=self._error_bound)

        if self._eb_abs is not None:
            params["eb_abs"] = self._eb_abs

        repr = ", ".join(f"{p}={v!r}" for p, v in params.items())

        return f"{type(self).__name__}({repr})"
