from typing import TypeVar

import numpy as np

T = TypeVar("T", bound=np.dtype)
T2 = TypeVar("T", bound=np.dtype)
S = TypeVar("S", bound=tuple[int, ...])


def to_float(x: np.ndarray[S, T]) -> np.ndarray[S, T2]:
    if np.issubdtype(x.dtype, np.floating):
        return x

    ftype = {
        np.dtype(np.int8): np.float16,
        np.dtype(np.int16): np.float32,
        np.dtype(np.int32): np.float64,
        np.dtype(np.int64): np.float128,
        np.dtype(np.uint8): np.float16,
        np.dtype(np.uint16): np.float32,
        np.dtype(np.uint32): np.float64,
        np.dtype(np.uint64): np.float128,
    }[x.dtype]

    return x.astype(ftype)
