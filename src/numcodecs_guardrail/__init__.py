from abc import ABC, abstractmethod
from enum import Enum

import numcodecs
import numcodecs.abc
import numcodecs.compat
import numcodecs.registry
import numcodecs.zlib
import numpy as np


class Guardrail(ABC):
    kind: str

    @abstractmethod
    def check(self, data: np.ndarray, decoded: np.ndarray) -> bool:
        pass

    @abstractmethod
    def encode_correction(self, data: np.ndarray, decoded: np.ndarray) -> bytes:
        pass

    @abstractmethod
    def apply_correction(self, decoded: np.ndarray, correction: bytes) -> np.ndarray:
        pass

    @abstractmethod
    def get_config(self) -> dict:
        pass

    @classmethod
    def from_config(cls, config: dict):
        return cls(**config)


class AbsoluteErrorBoundGuardrail(Guardrail):
    __slots__ = "_eb_abs"
    _eb_abs: float

    kind = "abs"

    def __init__(self, eb_abs: float):
        assert eb_abs is not None, "eb_abs must not be None"
        assert eb_abs > 0.0, "eb_abs must be positive"

        self._eb_abs = eb_abs

    def check(self, data: np.ndarray, decoded: np.ndarray) -> bool:
        return np.amax(np.abs(data - decoded)) <= self._eb_abs

    def encode_correction(self, data: np.ndarray, decoded: np.ndarray) -> bytes:
        error = decoded - data
        correction = np.round(error / (self._eb_abs * 2.0)).astype(np.int64)

        correction = self._lossless.encode(correction)

        return numcodecs.compat.ensure_bytes(correction)

    def apply_correction(self, decoded: np.ndarray, correction: bytes) -> np.ndarray:
        correction = self._lossless.decode(correction)
        correction = numcodecs.compat.ensure_bytes(correction)
        correction = np.frombuffer(correction, np.int64).reshape(decoded.shape)
        correction = correction.astype(decoded.dtype) * (self._eb_abs * 2.0)

        return decoded + correction

    def get_config(self):
        return dict(eb_abs=self._eb_abs)


class GuardrailKind(Enum):
    ABSOLUTE = AbsoluteErrorBoundGuardrail.kind


class GuardrailCodec(numcodecs.abc.Codec):
    __slots__ = ("_codec", "_lossless", "_guardrail")
    _codec: numcodecs.abc.Codec
    _lossless: numcodecs.abc.Codec
    _guardrail: Guardrail

    codec_id = "guardrail"

    def __init__(
        self,
        codec: dict,
        guardrail: str | GuardrailKind,
        **kwargs,
    ):
        self._codec = numcodecs.registry.get_codec(codec)
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
            # TODO: less metadata
            correction = bytes()
        else:
            correction = self._guardrail.encode_correction(data, decoded)

        # TODO: return metadata, encoded, optional correction
        return encoded + correction

    def decode(self, buf, out=None):
        assert out is not None, "can only decode into known dtype and shape"
        out = numcodecs.compat.ensure_ndarray(out)

        # TODO: decode the metadata into encoded and optional correction
        encoded, correction = buf

        decoded = self._codec.decode(encoded)

        corrected = self._guardrail.apply_correction(decoded, correction)

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
