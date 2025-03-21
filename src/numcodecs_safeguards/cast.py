from typing import Any, TypeVar

import numpy as np

T = TypeVar("T", bound=np.dtype)
S = TypeVar("S", bound=tuple[int, ...])


def to_float(x: np.ndarray[S, T]) -> np.ndarray[S, Any]:
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

    return x.astype(ftype)  # type: ignore


def from_float(x: np.ndarray[S, Any], dtype: T) -> np.ndarray[S, T]:
    if x.dtype == dtype:
        return x

    info = np.iinfo(dtype)
    imin, imax = np.array(info.min, dtype=dtype), np.array(info.max, dtype=dtype)

    with np.errstate(divide="ignore", over="ignore", under="ignore", invalid="ignore"):
        return np.where(x < imin, imin, np.where(x <= imax, x.astype(dtype), imax))  # type: ignore


def as_bits(a: np.ndarray, *, kind: str = "u") -> np.ndarray:
    return a.view(a.dtype.str.replace("f", kind).replace("i", kind).replace("u", kind))
