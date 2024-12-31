from abc import ABC, abstractmethod
from io import BytesIO
from typing import Optional, final

import numpy as np
import varint

from numcodecs.abc import Codec


class Guardrail(ABC):
    kind: str

    @final
    def check(self, data: np.ndarray, decoded: np.ndarray) -> bool:
        return np.all(self.check_elementwise(data, decoded))

    @abstractmethod
    def check_elementwise(self, data: np.ndarray, decoded: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def encode_correction(
        self, data: np.ndarray, decoded: np.ndarray, *, lossless: Codec
    ) -> bytes:
        pass

    @abstractmethod
    def apply_correction(
        self, decoded: np.ndarray, correction: bytes, *, lossless: Codec
    ) -> np.ndarray:
        pass

    @abstractmethod
    def get_config(self) -> dict:
        pass

    @classmethod
    def from_config(cls, config: dict):
        return cls(**config)


def as_bits(a: np.ndarray, *, like: Optional[np.ndarray] = None) -> np.ndarray:
    return np.frombuffer(
        a,
        dtype=np.dtype(
            (a if like is None else like).dtype.str.replace("f", "u").replace("i", "u")
        ),
    )


def runlength_encode(a: np.ndarray) -> bytes:
    """
    Encode the array `a` using run-length encoding.

    Currently, only zero-runs are RL-encoded and non-zero values are stored
    verbatim in non-zero runs.
    """

    a = a.flatten()
    zeros = a == 0

    # run-length encoding of the "is-a-zero" mask
    starts = np.r_[0, np.flatnonzero(zeros[1:] != zeros[:-1]) + 1]
    lengths = np.diff(np.r_[starts, len(a)])

    # store all non-zero values and the first zero of each zero-run
    indices = np.r_[0, np.flatnonzero((~zeros[1:]) | (zeros[1:] != zeros[:-1])) + 1]
    values = a[indices]

    encoded = [varint.encode(length) for length in lengths]
    encoded.append(values.tobytes())

    return b"".join(encoded)


def runlength_decode(b: bytes, *, like: np.ndarray) -> np.ndarray:
    """
    Decode the bytes `b` using run-length encoding.

    Currently, only zero-runs are RL-encoded and non-zero values are stored
    verbatim in non-zero runs.
    """

    lengths = []
    total_length = 0

    b_io = BytesIO(b)

    while total_length < like.size:
        length = varint.decode_stream(b_io)
        assert length > 0
        total_length += length
        lengths.append(length)

    assert total_length >= 0

    decoded = np.zeros(like.size, dtype=like.dtype)

    if total_length == 0:
        return decoded.reshape(like.shape)

    values = np.frombuffer(b, dtype=like.dtype, offset=b_io.tell())

    id, iv = 0, 0
    for length in lengths:
        if values[iv] == 0:
            iv += 1
        else:
            decoded[id : id + length] = values[iv : iv + length]
            iv += length
        id += length

    return decoded.reshape(like.shape)
