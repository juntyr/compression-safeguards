"""
float128, a floating-point dtype for numpy with true 128bit precision.
"""

__all__ = [
    "_float128",
    "_float128_type",
    "_float128_dtype",
    "_float128_min",
    "_float128_max",
    "_float128_smallest_normal",
    "_float128_smallest_subnormal",
    "_float128_pi",
    "_float128_e",
]

from typing import TYPE_CHECKING, TypeAlias

import numpy as np
import numpy_quaddtype
from numpy._typing import _128Bit

if TYPE_CHECKING:
    _float128_type: TypeAlias = np.floating[_128Bit]
else:
    _float128_type: type[np.floating[_128Bit]] = numpy_quaddtype.QuadPrecision  # type: ignore

_float128: type[_float128_type] = numpy_quaddtype.SleefQuadPrecision  # type: ignore
_float128_dtype: np.dtype[_float128_type] = numpy_quaddtype.SleefQuadPrecDType()
_float128_min: _float128_type = -numpy_quaddtype.max_value
_float128_max: _float128_type = numpy_quaddtype.max_value
_float128_smallest_normal: _float128_type = numpy_quaddtype.smallest_normal
_float128_smallest_subnormal: _float128_type = numpy_quaddtype.smallest_subnormal
_float128_pi: _float128_type = numpy_quaddtype.pi
_float128_e: _float128_type = numpy_quaddtype.e
