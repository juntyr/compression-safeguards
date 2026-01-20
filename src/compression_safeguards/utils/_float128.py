"""
float128, a floating-point dtype for numpy with true 128bit precision.
"""

__all__ = [
    "_float128",
    "_float128_type",
    "_float128_dtype",
    "_float128_pi",
    "_float128_e",
]

from collections.abc import Callable
from typing import TYPE_CHECKING, TypeAlias

import numpy as np
import numpy_quaddtype
from numpy._typing import _128Bit

if TYPE_CHECKING:
    _float128_type: TypeAlias = np.floating[_128Bit]
else:
    _float128_type: type[np.floating[_128Bit]] = numpy_quaddtype.QuadPrecision

_float128: Callable[[int | float | str], _float128_type] = (
    numpy_quaddtype.SleefQuadPrecision
)
_float128_dtype: np.dtype[_float128_type] = numpy_quaddtype.SleefQuadPrecDType()
_float128_pi: _float128_type = numpy_quaddtype.pi
_float128_e: _float128_type = numpy_quaddtype.e
