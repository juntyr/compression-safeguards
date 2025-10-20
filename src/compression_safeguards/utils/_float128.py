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

from typing import TypeAlias

import numpy as np

try:
    _float128: TypeAlias = np.float128
    _float128_type: TypeAlias = np.float128
    _float128_dtype: np.dtype[_float128] = np.dtype(np.float128)
    if (np.finfo(np.float128).nmant + np.finfo(np.float128).nexp + 1) != 128:
        raise TypeError("numpy.float128 does not offer true 128 bit precision")
    _float128_min: _float128 = np.finfo(np.float128).min
    _float128_max: _float128 = np.finfo(np.float128).max
    _float128_smallest_normal: _float128 = np.finfo(np.float128).smallest_normal
    _float128_smallest_subnormal: _float128 = np.finfo(np.float128).smallest_subnormal
    _float128_pi: _float128 = np.float128("3.14159265358979323846264338327950288")
    _float128_e: _float128 = np.float128("2.71828182845904523536028747135266249")
except (AttributeError, TypeError):
    try:
        import numpy_quaddtype

        _float128 = numpy_quaddtype.SleefQuadPrecision  # type: ignore
        _float128_type = numpy_quaddtype.QuadPrecision  # type: ignore
        _float128_dtype = numpy_quaddtype.SleefQuadPrecDType()
        _float128_min = -numpy_quaddtype.max_value
        _float128_max = numpy_quaddtype.max_value
        _float128_smallest_normal = numpy_quaddtype.smallest_normal
        _float128_smallest_subnormal = numpy_quaddtype.smallest_subnormal
        _float128_pi = numpy_quaddtype.pi
        _float128_e = numpy_quaddtype.e
    except ImportError:
        raise TypeError("""
compression_safeguards requires float128 support:
- numpy.float128 either does not exist is does not offer true 128 bit precision
- numpy_quaddtype is not installed
""") from None
